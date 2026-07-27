"""LLM-guided importance sampling for the quality-evaluation subset.

The ``CostModelAgent`` scores each candidate query plan on a small shared subset
of rows (``agent_cost_model/datasubset/{use_case}/sf_{scale_factor}/Q{id}_subset.csv``).  Drawing
that subset uniformly at random often yields rows that are irrelevant to the
query's semantic predicates, so plan-quality differences wash out.

``LLM_Sampler`` replaces the random draw with an LLM-guided, embedding-based
importance sample:

  1. One LLM call identifies the *non-semantic* filters and/or textual
     descriptions worth sampling around (``identify_sampling_targets``).
  2. For each target we importance-sample rows using embedding cosine
     similarity to the description (summing image + text similarities when the
     use case has both).

Experimental cost model: every row's image and concatenated-text embeddings are
computed **once per (use case, embedding model)** and cached under
``{cache_dir}/cached_results/``, shared across every query of that use case.
Re-running loads the cache and embeds only the ids not already present (per-id
incremental); switching ``embedding_model`` uses a separate cache file so vectors
from different models never mix.  The reported
``cost``/``latency`` are each split into two sources: ``embedding`` (the *sum* of
the precomputed per-row costs for the rows whose embeddings were actually used,
each counted once, plus the description embeddings) and ``target`` (the single
identifier-LLM call).

Run standalone via ``python -m agent_cost_model.llm_sampler --help`` or wired
into ``demo.py`` before ``agent.run(...)``.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import re
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# Keep sample size / seed consistent with PhysicalPipeline.run_subset so the
# subset we write matches the size the agent expects.
try:
    from agent.physical_pipeline import NUM_SAMPLES, SUBSET_SEED
except ImportError:  # pragma: no cover - fallback when run from repo root layouts differ
    NUM_SAMPLES, SUBSET_SEED = 10, 42


_SYSTEM_PROMPT = """You help build a small, *informative* evaluation subset for a data query.

You are given a natural-language query over a table of rows and the table's columns.
Your job is to identify the distinct kinds of rows we should over-sample so that a
quality evaluation on the sampled subset is actually meaningful (i.e. the subset
contains the rows the query searches for, not just random rows).

Return ONLY a JSON array. Each element is an object with two keys:
  - "filter":      a Python boolean expression over a variable named `row`
                   (e.g. "row[\\"size\\"] >= 20"), or null if no non-semantic
                   filter applies. Use ONLY simple, deterministic column
                   predicates here (comparisons, membership) -- never anything
                   that requires understanding an image or free text.
  - "description": a short textual description of the semantic/visual content we
                   want to find (e.g. "bag of items that are red, blue, or white"), or null if the item is purely a filter.

This "description" is fed straight into an embedding model and matched against the
rows by cosine similarity. So output ONLY the phrase(s) that carry semantic weight
-- the actual content to look for -- with no query framing, instructions, or
meta-language. For example:
  - output "white"  NOT  "product whose primary color is white"
  - output "brown coffee table"   NOT  "the image shows a brown coffee table"

A description must contain ONLY the items/topics of interest -- never negations or
exclusions. Words like "not", "no", "exclude", "excluding", "except", "without" do
NOT work in embedding similarity (the embedder still keys on the excluded thing and
would match it). Drop the negated part entirely and keep only what you DO want:
  - output "table"       NOT  "table excluding desks"
  - output "blue bag"    NOT  "bag that is blue but not green"
If a constraint is a negation over a structured column, express it as a `filter`
instead; if it cannot be, just omit it from the description.

Guidelines:
  - Break the query into its independent semantic components; emit one array
    element per component.
  - Within a description, separate DISJOINT concepts with a period ".". A period
    means AND -- every concept should apply to a matching row. Within a concept,
    list synonyms / subtypes separated by commas ","; a comma means OR -- any one
    of them counts as that concept being present. Split independent attributes
    (e.g. a color) from the item type into separate concepts.
    e.g. "white. furniture, table, couch, desk"
         -> concept 1 = the color white;  concept 2 = the furniture type
  - Keep each concept short; add only a few (about 0-4) common synonyms or subtypes
    to broaden the match. Stay brief and general -- do NOT enumerate exhaustively.
    e.g. "white. sofa, couch, settee"  (good: color AND item, a few synonyms)
         "white sofa"                   (split the color out; add a couple synonyms)
         "white. sofa, couch, settee, loveseat, divan, chesterfield, ..."  (too much)
  - Descriptions must be purely positive -- only the items/topics of interest, no
    negation or exclusion language.
  - If the query has no useful sampling target, return an empty array [].

Examples:

Query: "Based on the image representation of the product, find the prod_id of
products where the image shows a bag of items that are red, blue, or white."
Answer:
[{"filter": null, "description": "bag, tote, backpack. red, blue, white"}]

Query: "Based on product images and descriptions, find all-white furniture sets for a living room.
Furniture sets consist of a sofa, a coffee table (less than $500), and a rug (made from nylon)"
Answer:
[{"filter": null, "description": "white. sofa, couch, settee"},
 {"filter": "row[\\"length\\"] < 40", "description": "white. coffee table"},
 {"filter": "row[\\"materials\\"] == 'nylon'", "description": "white. rug, carpet, mat"}]
"""


# Used when semantic descriptions cannot contribute to scoring (no usable text/image
# embedding and no keyword matching) -- e.g. an image-only query run with
# no_image_emb. Then we only ask for non-semantic filters.
_SYSTEM_PROMPT_FILTER_ONLY = """You help build a small, *informative* evaluation subset for a data query.

You are given a natural-language query over a table of rows and the table's columns.
Identify the non-semantic FILTERS that select the rows relevant to the query, so we
can over-sample from them.

Return ONLY a JSON array. Each element is an object with a single key:
  - "filter": a Python boolean expression over a variable named `row`
              (e.g. "row[\\"price\\"] <= 500", "row[\\"gender\\"] == 'Men'").
              Use ONLY simple, deterministic column predicates (comparisons,
              membership) over the columns shown -- never anything that requires
              understanding an image or free text.

Guidelines:
  - Emit one array element per independent filter condition expressible from the
    structured columns.
  - If the query has no useful filter, return an empty array [].

Example:
Query: "Find men's products priced at $500 or less."
Answer:
[{"filter": "row[\\"gender\\"] == 'Men'"},
 {"filter": "row[\\"price\\"] <= 500"}]
"""


class LLM_Sampler:
    """LLM-guided, embedding-based importance sampler for the quality subset."""

    def __init__(
        self,
        query_id: int | str,
        use_case: str,
        query_text: str,
        df: pd.DataFrame,
        *,
        scale_factor: int,
        llm_client: Any,
        embedding_model: str = "qwen/qwen3-embedding-8b",
        id_col: str = "prod_id",
        image_dir: str | None = None,
        text_cols: list[str] | None = None,
        sample_size: int = NUM_SAMPLES,
        sample_method: str = "importance-sampling",
        keyword: bool = False,
        no_image_emb: bool = False,
        seed: int = SUBSET_SEED,
        cache_dir: str | None = None,
        subset_out_path: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str | None = None,
    ) -> None:
        if sample_method not in ("importance-sampling", "topk"):
            raise ValueError(
                f"sample_method must be 'importance-sampling' or 'topk', got {sample_method!r}"
            )
        self.query_id = query_id
        self.use_case = use_case
        self.scale_factor = scale_factor
        self.query_text = query_text
        self.df = df.reset_index(drop=True)
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.id_col = id_col
        self.image_dir = image_dir
        self.no_image_emb = no_image_emb
        # Single gate for all image work (embedding, similarity, cost). When
        # no_image_emb is set we ignore images entirely even if the dataset has them.
        self._use_image = bool(image_dir) and not no_image_emb
        self.sample_size = sample_size
        self.sample_method = sample_method
        self.keyword = keyword
        self.seed = seed

        # Text columns default: every object/string column except the id column.
        all_text_cols = [
            c
            for c in self.df.columns
            if c != id_col and not pd.api.types.is_numeric_dtype(self.df[c])
        ]
        self.text_cols = all_text_cols if text_cols is None else text_cols
        # Keyword matching scans ALL text columns of the row, independent of the
        # embedding modality (so it works even for image-only queries with text_cols=[]).
        self._keyword_cols = all_text_cols

        self.cache_dir = pathlib.Path(cache_dir or f"agent_cost_model/sampling/{use_case}")
        self.cached_results_dir = self.cache_dir / "cached_results"
        print(f"Using cached results from: {self.cached_results_dir}")
        self.results_path = self.cache_dir / f"sf_{scale_factor}" / "sampling_results.json"
        self.subset_out_path = pathlib.Path(
            subset_out_path
            or f"agent_cost_model/datasubset/{use_case}/sf_{scale_factor}/Q{query_id}_subset.csv"
        )

        # Embeddings are called over raw HTTP (not the OpenAI SDK helper): the
        # SDK defaults to base64 encoding + a post-parser that hides provider
        # errors behind a cryptic "No embedding data received". A direct call
        # with encoding_format="float" is broadly compatible and lets us surface
        # the real response body when something goes wrong.
        import httpx

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ["OPENROUTER_API_KEY"]
        self._http = httpx.Client(timeout=120.0)

        # Per-run accumulators (identifier LLM call + description embeddings).
        self._id_llm_cost = 0.0
        self._id_llm_latency = 0.0
        self._desc_embed_cost = 0.0
        self._desc_embed_latency = 0.0
        self._desc_cache: dict[str, np.ndarray] = {}
        self._raw_targets: list[dict] = []

        # Precomputed row embeddings + per-row costs (populated by _load_or_precompute).
        self._text_emb: dict[int, np.ndarray] = {}
        self._image_emb: dict[int, np.ndarray] = {}
        self._row_costs: dict[int, dict] = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _softmax(sims: list[float]) -> np.ndarray:
        arr = np.asarray(sims, dtype=np.float64)
        arr = arr - arr.max()
        exp = np.exp(arr)
        total = exp.sum()
        if total == 0.0:
            return np.full(len(sims), 1.0 / len(sims))
        return exp / total

    def _row_text(self, row: pd.Series) -> str:
        return " ".join(str(row[c]) for c in self.text_cols if pd.notna(row[c]))

    @staticmethod
    def _norm(text: str) -> str:
        """Space-joined lowercase full-word tokens (punctuation stripped).

        Normalizing both a phrase and the row text this way gives full-word AND
        phrase matching: 'dressing' -> 'dressing' (so a 'dress' target does not
        match it), and 'crew-neck'/'crew neck' both -> 'crew neck', so the phrase
        matches only when both words appear contiguously.
        """
        return " ".join(re.findall(r"[a-z0-9]+", text.lower()))

    def _row_keyword_text(self, row: pd.Series) -> str:
        """Normalized, space-padded token string of a row's text data.

        Padded with a leading/trailing space so `f" {phrase} " in text` is an
        exact full-word / phrase match at any position.
        """
        text = " ".join(str(row[c]) for c in self._keyword_cols if pd.notna(row[c]))
        return f" {self._norm(text)} "

    @classmethod
    def _target_groups(cls, desc: str) -> list[list[str]]:
        """Split a target description into concept groups of (possibly multi-word) phrases.

        A period "." separates concepts (AND); commas within a concept separate
        synonym phrases (OR). Each phrase is normalized and matched as a
        contiguous unit, e.g.
        "men's. round neck, crew neck. short sleeve" ->
        [["men s"], ["round neck", "crew neck"], ["short sleeve"]].
        """
        groups = []
        for concept in desc.split("."):
            phrases = [p for p in (cls._norm(part) for part in concept.split(",")) if p]
            if phrases:
                groups.append(phrases)
        return groups

    # ------------------------------------------------------------- embeddings
    def _embed_call(self, input_value: Any) -> tuple[np.ndarray, float, float]:
        """POST one input to the embeddings endpoint; return (vec, cost, latency).

        Uses ``encoding_format="float"`` and reads the raw JSON so provider
        errors (bad model id, unsupported endpoint, ...) surface clearly instead
        of the SDK's opaque "No embedding data received".
        """
        body = {
            "model": self.embedding_model,
            "input": input_value,
            "encoding_format": "float",
            "usage": {"include": True},  # ask OpenRouter to report dollar cost
        }
        t0 = time.time()
        resp = self._http.post(
            f"{self._base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        latency = time.time() - t0
        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding request failed ({resp.status_code}) for model "
                f"{self.embedding_model!r}: {resp.text[:800]}"
            )
        payload = resp.json()
        data = payload.get("data")
        if not data:
            raise RuntimeError(
                f"No embedding data from {self.embedding_model!r}. "
                f"Response: {json.dumps(payload)[:800]}"
            )
        vec = np.asarray(data[0]["embedding"], dtype=np.float32)
        usage = payload.get("usage") or {}
        cost = 0.0
        for key in ("cost", "total_cost", "estimated_cost"):
            value = usage.get(key)
            if value is not None:
                try:
                    cost = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        return vec, cost, latency

    def _embed_text(self, text: str) -> tuple[np.ndarray, float, float]:
        return self._embed_call(text or " ")

    def _embed_image(self, path: pathlib.Path) -> tuple[np.ndarray, float, float]:
        """Embed an image. Requires ``embedding_model`` to be multimodal.

        The image is sent as a base64 data URL. Some providers expect a
        different multimodal ``input`` shape; adjust here if your embedding
        endpoint rejects the data URL.
        """
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        return self._embed_call(f"data:image/jpeg;base64,{b64}")

    def _embed_description(self, desc: str) -> np.ndarray:
        """Embed a query description once; accumulate its cost/latency."""
        if desc in self._desc_cache:
            return self._desc_cache[desc]
        vec, cost, latency = self._embed_text(desc)
        self._desc_embed_cost += cost
        self._desc_embed_latency += latency
        self._desc_cache[desc] = vec
        return vec

    @staticmethod
    def _model_slug(model: str) -> str:
        """Filesystem-safe slug for an embedding model id (keeps caches separate)."""
        return re.sub(r"[^A-Za-z0-9._-]", "_", model)

    @staticmethod
    def _empty_cost() -> dict:
        return {"text_cost": 0.0, "text_latency": 0.0, "image_cost": 0.0, "image_latency": 0.0}

    def _cache_paths(self) -> tuple[pathlib.Path, pathlib.Path]:
        # Row embeddings are shared across ALL queries of a use case (the same
        # underlying rows), so the cache is keyed only by embedding model. The
        # use case is already encoded in cached_results_dir.
        slug = self._model_slug(self.embedding_model)
        emb = self.cached_results_dir / f"{slug}_embeddings.npz"
        cost = self.cached_results_dir / f"{slug}_embed_costs.json"
        return emb, cost

    def _load_or_precompute(self) -> None:
        """Load cached per-row embeddings; embed only the ids not yet cached.

        The cache is keyed by (use case, embedding model) and shared across every
        query of that use case. Each row's concatenated text and (when
        ``image_dir`` is set) its image are embedded at most once, ever, for a
        given model. New ids in ``self.df`` are embedded and merged into the
        existing cache; the cache only ever grows, so previously embedded ids are
        never recomputed — even across different queries.
        """
        emb_path, cost_path = self._cache_paths()

        # 1. Load whatever is already cached for this (query, model).
        if emb_path.exists() and cost_path.exists():
            data = np.load(emb_path, allow_pickle=True)
            # Text arrays may be absent for image-only caches (no text columns).
            if "ids" in data.files and "text" in data.files:
                ids = [int(x) for x in data["ids"].tolist()]
                text = data["text"]
                self._text_emb = {ids[i]: text[i] for i in range(len(ids))}
            if "image_ids" in data.files:
                iids = [int(x) for x in data["image_ids"].tolist()]
                image = data["image"]
                self._image_emb = {iids[i]: image[i] for i in range(len(iids))}
            with open(cost_path) as fh:
                raw = json.load(fh)
            rows = raw.get("rows", raw)  # tolerate a legacy flat {id: {...}} file
            self._row_costs = {int(k): v for k, v in rows.items()}

        # 2. Embed only ids missing from the cache; merge into the in-memory maps.
        dirty = False
        for _, row in self.df.iterrows():
            rid = int(row[self.id_col])
            # Skip text embeddings entirely when the table has no text columns
            # (e.g. an image-only query like Q2 whose table is just prod_id).
            if self.text_cols and rid not in self._text_emb:
                tvec, tcost, tlat = self._embed_text(self._row_text(row))
                self._text_emb[rid] = tvec
                entry = self._row_costs.setdefault(rid, self._empty_cost())
                entry["text_cost"], entry["text_latency"] = tcost, tlat
                dirty = True
            if self._use_image and rid not in self._image_emb:
                img_path = pathlib.Path(self.image_dir) / f"{rid}.jpg"
                if img_path.exists():
                    ivec, icost, ilat = self._embed_image(img_path)
                    self._image_emb[rid] = ivec
                    entry = self._row_costs.setdefault(rid, self._empty_cost())
                    entry["image_cost"], entry["image_latency"] = icost, ilat
                    dirty = True

        # 3. Persist the (grown) cache only when something new was embedded.
        if dirty or not emb_path.exists():
            self._save_row_cache(emb_path, cost_path)

    def _save_row_cache(self, emb_path: pathlib.Path, cost_path: pathlib.Path) -> None:
        """Write the union of all embedded ids + which embedding model produced them."""
        self.cached_results_dir.mkdir(parents=True, exist_ok=True)
        text_ids = list(self._text_emb.keys())
        save_kwargs: dict[str, Any] = {
            "ids": np.asarray(text_ids),
            "text": np.asarray([self._text_emb[i] for i in text_ids]),
        }
        if self._image_emb:
            image_ids = list(self._image_emb.keys())
            save_kwargs["image_ids"] = np.asarray(image_ids)
            save_kwargs["image"] = np.asarray([self._image_emb[i] for i in image_ids])
        np.savez(emb_path, **save_kwargs)
        with open(cost_path, "w") as fh:
            json.dump(
                {
                    "embedding_model": self.embedding_model,
                    "rows": {str(k): v for k, v in self._row_costs.items()},
                },
                fh,
                indent=2,
            )

    # --------------------------------------------------------------- LLM step
    def _compile_filter(self, expr: Any) -> Optional[Callable[[pd.Series], bool]]:
        """Compile a filter string into a safe predicate over a row Series."""
        if expr is None:
            return None
        if not isinstance(expr, str):
            return None
        text = expr.strip()
        if text == "" or text.lower() in ("null", "none"):
            return None
        safe_globals = {"__builtins__": {}}
        try:
            if text.startswith("lambda"):
                raw = eval(text, safe_globals, {})  # noqa: S307 - constrained globals

                def fn(row, _raw=raw):
                    try:
                        return bool(_raw(row))
                    except Exception:
                        return False

                fn.expr = text
                return fn
            code = compile(text, "<filter>", "eval")

            def fn(row, _code=code):
                try:
                    return bool(eval(_code, safe_globals, {"row": row}))  # noqa: S307
                except Exception:
                    return False

            fn.expr = text
            return fn
        except Exception:
            return None

    def _parse_targets(self, content: str) -> list[dict]:
        text = (content or "").strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            data = json.loads(text)
        except Exception:
            return []
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        out: list[dict] = []
        for item in data:
            if isinstance(item, dict):
                out.append(
                    {"filter": item.get("filter"), "description": item.get("description")}
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out.append({"filter": item[0], "description": item[1]})
        return out

    def _descriptions_usable(self) -> bool:
        """A description only helps if it can be scored -- embedded (text/image) or
        keyword-matched against row text. If none of those apply (e.g. an image-only
        query run with no_image_emb), asking for descriptions is pointless.
        """
        return bool(
            self._use_image or self.text_cols or (self.keyword and self._keyword_cols)
        )

    def identify_sampling_targets(self) -> list[tuple[Optional[Callable], Optional[str]]]:
        """Single LLM call → list of (filter_fn | None, description | None)."""
        system = _SYSTEM_PROMPT if self._descriptions_usable() else _SYSTEM_PROMPT_FILTER_ONLY
        schema = ", ".join(f"{c} ({self.df[c].dtype})" for c in self.df.columns)
        # Show the first 15 rows so the model can see the data format/vocabulary;
        # cap each row at 3000 chars (some columns hold long text/JSON blobs).
        sample_rows = [
            str(row.to_dict())[:3000] for _, row in self.df.head(15).iterrows()
        ]
        sample_block = "\n".join(sample_rows)
        user = (
            f"Query:\n{self.query_text}\n\n"
            f"Columns: {schema}\n\n"
            f"First {len(sample_rows)} rows of the dataset (each truncated to 3000 chars):\n"
            f"{sample_block}\n\n"
            "Return the JSON array of sampling targets."
        )
        t0 = time.time()
        content, _reasoning, meta = self.llm_client.generate(
            system, [{"role": "user", "content": user}]
        )
        self._id_llm_latency += time.time() - t0
        self._id_llm_cost += float((meta or {}).get("cost_usd", 0.0) or 0.0)

        self._raw_targets = self._parse_targets(content)
        return [
            (self._compile_filter(t.get("filter")), t.get("description"))
            for t in self._raw_targets
        ]

    # ----------------------------------------------------------------- sample
    def _row_similarity(self, desc_vec: np.ndarray, rid: int) -> float:
        # Gate on the current query's modality, not on what's cached: the shared
        # cache may hold text embeddings from a text query that an image-only
        # query must ignore (and vice versa).
        sim = 0.0
        if self.text_cols:
            tvec = self._text_emb.get(rid)
            if tvec is not None:
                sim += self._cosine(desc_vec, tvec)
        if self._use_image:
            ivec = self._image_emb.get(rid)
            if ivec is not None:
                sim += self._cosine(desc_vec, ivec)
        return sim

    def sample(self) -> list[int]:
        """Run the full sampling loop; write the subset CSV + results.json."""
        targets = self.identify_sampling_targets()
        if len(targets)==0:
            print("No sampling targets identified; drawing a uniform random subset.")
        else:
            for idx, (fn, desc) in enumerate(targets):
                print(f"{idx}) filter={fn.expr if fn else None}, description={desc!r}")
        rng = np.random.default_rng(self.seed)
        all_ids = [int(x) for x in self.df[self.id_col].tolist()]

        used_rows: set[int] = set()

        if not targets:
            k = min(self.sample_size, len(all_ids))
            sampled = [int(x) for x in rng.choice(all_ids, size=k, replace=False)]
        else:
            self._load_or_precompute()
            # Precompute each row's normalized (space-padded) text once for keyword matching.
            kw_index: dict[int, str] = {}
            if self.keyword:
                kw_index = {
                    int(row[self.id_col]): self._row_keyword_text(row)
                    for _, row in self.df.iterrows()
                }
            n = len(targets)
            per_item = max(1, round(self.sample_size / n))
            picks: list[int] = []
            selected: set[int] = set()  # ids already chosen by earlier items
            for i, (fn, desc) in enumerate(targets):
                k = per_item if i < n - 1 else max(1, self.sample_size - per_item * (n - 1))
                if fn is not None:
                    mask = self.df.apply(fn, axis=1)
                    pool_ids = [int(x) for x in self.df.loc[mask, self.id_col].tolist()]
                else:
                    pool_ids = list(all_ids)
                if not pool_ids:
                    continue
                # Score is meaningful only when at least one embedding modality
                # is available; without it, similarity is 0 for every row and
                # topk / importance-sampling degenerates to arbitrary selection.
                can_score = bool(self._use_image or self.text_cols)
                if desc is None or not can_score:
                    # Uniform draw from the filtered pool (after excluding already-picked ids).
                    avail_ids = [rid for rid in pool_ids if rid not in selected]
                    take = min(k, len(avail_ids))
                    if take <= 0:
                        continue
                    chosen = rng.choice(avail_ids, size=take, replace=False)
                else:
                    desc_vec = self._embed_description(desc)
                    # Concept groups of the LLM target, for the optional keyword boost.
                    # Period "." = separate concepts (AND); comma "," = synonyms (OR).
                    target_groups = self._target_groups(desc) if self.keyword else []
                    # Embed all rows passing the filter; score by cosine similarity.
                    sims_by_id = {}
                    for rid in pool_ids:
                        used_rows.add(rid)
                        s = self._row_similarity(desc_vec, rid)
                        # +1 for EACH concept group with >=1 phrase present in the
                        # row text. Phrases match as contiguous units, e.g. target
                        # "round neck, crew neck. short sleeve": "crew neck" counts
                        # only if both words appear together, not just "crew".
                        if target_groups:
                            row_text = kw_index.get(rid, " ")
                            s += sum(
                                1
                                for phrases in target_groups
                                if any(f" {p} " in row_text for p in phrases)
                            )
                        sims_by_id[rid] = s
                    # Exclude ids already picked so a collision selects another unique
                    # id from the remaining candidates instead of shrinking the subset.
                    avail_ids = [rid for rid in pool_ids if rid not in selected]
                    take = min(k, len(avail_ids))
                    if take <= 0:
                        continue
                    if self.sample_method == "topk":
                        # Deterministic: the `take` highest-similarity rows.
                        chosen = sorted(avail_ids, key=lambda rid: sims_by_id[rid], reverse=True)[:take]
                    else:
                        # Importance sampling: draw weighted by similarity (softmax).
                        weights = self._softmax([sims_by_id[rid] for rid in avail_ids])
                        chosen = rng.choice(avail_ids, size=take, replace=False, p=weights)
                for rid in chosen:
                    rid = int(rid)
                    picks.append(rid)
                    selected.add(rid)

            # picks are unique by construction; top-up to ~sample_size if pools were small.
            sampled = list(dict.fromkeys(picks))
            if len(sampled) < self.sample_size:
                remaining = [i for i in all_ids if i not in sampled]
                if remaining:
                    extra = rng.choice(
                        remaining,
                        size=min(self.sample_size - len(sampled), len(remaining)),
                        replace=False,
                    )
                    sampled.extend(int(x) for x in extra)
            sampled = sampled[: self.sample_size]

        self._write_subset(sampled)
        self._write_results(sampled, used_rows)
        return sampled

    # ------------------------------------------------------------------ output
    def _write_subset(self, sampled_ids: list[int]) -> None:
        order = {rid: pos for pos, rid in enumerate(sampled_ids)}
        subset = self.df[self.df[self.id_col].isin(sampled_ids)].copy()
        subset = (
            subset.assign(_ord=subset[self.id_col].map(order))
            .sort_values("_ord")
            .drop(columns="_ord")
        )
        self.subset_out_path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(self.subset_out_path, index=False)

    def _write_results(self, sampled_ids: list[int], used_rows: set[int]) -> None:
        # Only count the modalities this query actually used (the shared cache may
        # store both text and image costs for a row, but an image-only query did
        # not incur/use the text embedding, and vice versa).
        row_cost = 0.0
        row_latency = 0.0
        for rid in used_rows:
            c = self._row_costs.get(rid, {})
            if self.text_cols:
                row_cost += c.get("text_cost", 0.0)
                row_latency += c.get("text_latency", 0.0)
            if self._use_image:
                row_cost += c.get("image_cost", 0.0)
                row_latency += c.get("image_latency", 0.0)
        # Separate the two cost/latency sources: the embedding work (description +
        # per-row embeddings) vs. the single target-making LLM call.
        embedding_cost = self._desc_embed_cost + row_cost
        embedding_latency = self._desc_embed_latency + row_latency
        target_cost = self._id_llm_cost
        target_latency = self._id_llm_latency

        record = {
            "query_id": str(self.query_id),
            "target_latency": round(target_latency, 6),
            "embedding_latency": round(embedding_latency, 6),
            "target_cost": round(target_cost, 8),
            "embedding_cost": round(embedding_cost, 8),
            "embedding_model": self.embedding_model,
            "sample_method": self.sample_method,
            "keyword": self.keyword,
            "no_image_emb": self.no_image_emb,
            "sampled_ids": [int(x) for x in sampled_ids],
            "targets": self._raw_targets,
        }

        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results: dict = {}
        if self.results_path.exists():
            try:
                with open(self.results_path) as fh:
                    results = json.load(fh)
            except Exception:
                results = {}
        results[str(self.query_id)] = record
        with open(self.results_path, "w") as fh:
            json.dump(results, fh, indent=2)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run LLM_Sampler for one query.")
    ap.add_argument("--use-case", default="ecomm")
    ap.add_argument("--scale-factor", type=int, default=500)
    ap.add_argument("--query-id", type=int, required=True)
    ap.add_argument("--data-dir", default=None, help="defaults to agent_cost_model/dataset/{use_case}/sf_{scale_factor}")
    ap.add_argument("--csv", default=None, help="source CSV filename; defaults to styles_details.csv")
    ap.add_argument("--query", required=True, help="natural-language query text")
    ap.add_argument("--id-col", default="prod_id")
    ap.add_argument("--model", default="openai/gpt-5", help="identifier LLM (OpenRouter id)")
    ap.add_argument("--embedding-model", default="qwen/qwen3-embedding-8b")
    ap.add_argument("--images", action="store_true", help="use per-row image embeddings")
    ap.add_argument("--sample-method", choices=["importance-sampling", "topk"],
                    default="importance-sampling",
                    help="how to pick rows from cosine-similarity scores per described target")
    ap.add_argument("--keyword", action="store_true",
                    help="add +1 to a row's similarity if any full word of the LLM target appears in its text")
    ap.add_argument("--no_image_emb", action="store_true",
                    help="ignore image embeddings when scoring, even if the dataset has images (avoids costly image embedding)")
    args = ap.parse_args()

    from cost_model_agent import OpenRouterClient

    data_dir = args.data_dir or f"agent_cost_model/dataset/{args.use_case}/sf_{args.scale_factor}"
    csv = args.csv or f"styles_details_Q{args.query_id}.csv"
    df = pd.read_csv(os.path.join(data_dir, csv))
    image_dir = os.path.join(data_dir, "images") if args.images else None

    sampler = LLM_Sampler(
        query_id=args.query_id,
        use_case=args.use_case,
        scale_factor=args.scale_factor,
        query_text=args.query,
        df=df,
        llm_client=OpenRouterClient(args.model, reasoning_effort="medium"),
        embedding_model=args.embedding_model,
        id_col=args.id_col,
        image_dir=image_dir,
        sample_method=args.sample_method,
        keyword=args.keyword,
        no_image_emb=args.no_image_emb,
    )
    targets = sampler.identify_sampling_targets()
    print("LLM identified sampling targets:") #debug print
    for i, (fn, desc) in enumerate(targets):
        print(f"  {i + 1}. {desc}")
    sampled = sampler.sample()
    print(f"sampled ids ({len(sampled)}): {sampled}")
    print(f"wrote subset -> {sampler.subset_out_path}")
    print(f"wrote results -> {sampler.results_path}")


if __name__ == "__main__":
    _main()
