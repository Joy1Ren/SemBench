"""Oracle-based quality evaluator for the cost model agent.

Per-operator quality:
  sem_filter / sem_join: agreement rate between plan decisions and oracle decisions.
  sem_map: oracle judges whether each output field is correct (batched LLM call).

Plan quality: plan output vs. oracle plan output using the original quality metrics
  (f1, relative_error, spearman_correlation, accuracy).

Oracle plan output is cached per plan_name in llm_judge_dir. In addition, oracle LLM
calls are memoized per-operator across plans (see _MemoizingGenerator): a semantic
operator shared by two plans is judged by the oracle only once.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

def _id_str(v) -> str:
    """Stringify an ID value, converting whole-number floats to int first."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def normalize_eval_df(df: pd.DataFrame, use_case: str, query_id: int) -> pd.DataFrame:
    """Apply use-case/query-specific column normalization for evaluator compatibility.

    Shared by the plan-output and oracle/ground-truth paths so their id columns
    always line up. Mirrors the per-query dialect output shape expected by the
    scenario evaluators.
    """
    # A plan that filters out every record yields an empty, column-less DataFrame.
    # Without this guard the per-query branches below would raise an opaque
    # KeyError ("None of [Index(['id'])] are in the [columns]") from df[["id"]],
    # masking the real "plan produced no output" condition. Return a canonical
    # empty id-frame so a zero-row plan scores as empty instead of crashing.
    if use_case == "ecomm":
        if query_id in (1, 2, 13, 14):
            if len(df) == 0:
                return pd.DataFrame({"id": []})
            df = df.rename(columns={"prod_id": "id"})
            df = df[["id"]]
        elif query_id in (3,4,5,6):
            if len(df) == 0:
                return pd.DataFrame({"id": [], "category": []})
            df = df.rename(columns={"prod_id": "id"})
            df = df.iloc[:, :2] #keeps prod_id, category
        elif query_id in (7, 8, 9):
            if len(df) == 0:
                return pd.DataFrame({"id": []})
            df["id"] = df.iloc[:, :2].apply(lambda col: col.map(_id_str)).agg("-".join, axis=1)
            df = df[["id"]]
        elif query_id == 10:
            if len(df) == 0:
                return pd.DataFrame({"id": []})
            assert df.shape[1] == 3, f"ecomm Q10 expects 3 columns, got {df.shape[1]}"
            df["id"] = df.iloc[:, :3].apply(lambda col: col.map(_id_str)).agg("-".join, axis=1)
            df = df[["id"]]
        elif query_id == 11:
            if len(df) == 0:
                return pd.DataFrame({"id": []})
            assert df.shape[1] == 4, f"ecomm Q11 expects 4 columns, got {df.shape[1]}"
            df["id"] = df.iloc[:, :4].apply(lambda col: col.map(_id_str)).agg("-".join, axis=1)
            df = df[["id"]]
        elif query_id == 12:
            if len(df) == 0:
                return pd.DataFrame({"id": []})
            assert df.shape[1] == 3, f"ecomm Q12 expects 3 columns, got {df.shape[1]}"
            df["id"] = df.iloc[:, :3].apply(
                lambda r: json.dumps(
                    {"id": int(r.iloc[0]), "brand": r.iloc[1], "category": r.iloc[2]},
                    separators=(",", ":"),
                ),
                axis=1,
            )
            df = df[["id"]]
    return df


def _load_evaluator(use_case: str, scale_factor: int, agent_dir: str):
    """Import and instantiate the concrete evaluator for the given use_case."""
    if use_case == "movie":
        from scenario.movie.evaluation.evaluate import MovieEvaluator
        return MovieEvaluator(use_case, scale_factor, agent_dir)
    if use_case == "ecomm":
        from scenario.ecomm.evaluation.evaluate import EcommEvaluator  # type: ignore
        return EcommEvaluator(use_case, scale_factor, agent_dir)
    raise ValueError(f"No evaluator registered for use_case={use_case!r}")


@dataclass
class QualityResult:
    quality: float                          # plan quality vs oracle output (0-1, NaN if failed/N/A)
    per_sem_op_quality: dict[str, float] = field(default_factory=dict)
    quality_note: str | None = None         # human-readable reason when quality is not a real score
                                            # (e.g. oracle produced no rows on this subset)
    # {op_name: 0-1} for each semantic op where quality could be computed


class _MemoizingGenerator:
    """Wraps a Palimpzest ``Generator`` so oracle LLM calls are memoized by
    (operator identity, input content) and reused across plans.

    Two plans that share a semantic operator feed that operator identical inputs (the
    subset is fixed and the shared upstream produces identical records), so its per-record
    oracle calls build identical prompts. We key on the operator's condition / generated
    fields plus the projected input field values, so the second plan reuses the first
    plan's oracle verdicts instead of re-calling the LLM. A cache hit returns a zero-cost
    ``GenerationStats``, so reported oracle cost reflects only real (cache-miss) calls.

    The key is **content-based**: if the upstream differs and the input content changes,
    the key changes and the oracle is re-run — reuse happens only when inputs are truly
    identical. This is the operator-level analogue of PZ's validator ``join_cache`` /
    ``filter_cache``. The operator still builds fresh output DataRecords (correct
    source_indices) from the cached verdict, so downstream scoring is unaffected.
    """

    def __init__(self, inner, cache: dict, model_value, reasoning_effort, stats: dict):
        self._inner = inner
        self._cache = cache
        self._model_value = model_value
        self._reff = reasoning_effort
        self._stats = stats

    def __getattr__(self, name):
        # delegate any other attribute access (e.g. .model) to the wrapped generator
        return getattr(self.__dict__["_inner"], name)

    @staticmethod
    def _content(record, cols) -> tuple | None:
        if record is None:
            return None
        keys = list(cols) if cols else []
        if not keys:
            try:
                schema_cls = record.schema if isinstance(record.schema, type) else type(record.schema)
                keys = list(schema_cls.model_fields)
            except Exception:
                keys = []
        return tuple(sorted((c, repr(getattr(record, c, None))) for c in keys))

    def __call__(self, candidate, fields=None, right_candidate=None, **kwargs):
        from palimpzest.core.models import GenerationStats
        import copy as _copy
        cols = kwargs.get("project_cols")
        cond = kwargs.get("filter_condition") or kwargs.get("join_condition")
        field_sig = tuple(sorted(
            (n, getattr(fi, "description", "") or "") for n, fi in (fields or {}).items()
        ))
        key = (
            self._model_value, self._reff, cond, field_sig,
            self._content(candidate, cols), self._content(right_candidate, cols),
        )
        try:
            hkey = hash(key)
        except TypeError:
            hkey = hash(repr(key))
        if hkey in self._cache:
            self._stats["hits"] += 1
            return _copy.deepcopy(self._cache[hkey]), None, GenerationStats(), None
        self._stats["misses"] += 1
        field_answers, reasoning, gen_stats, messages = self._inner(
            candidate, fields, right_candidate=right_candidate, **kwargs
        )
        self._cache[hkey] = _copy.deepcopy(field_answers)
        return field_answers, reasoning, gen_stats, messages


class QualityEvaluator:
    """Evaluates plan quality against an oracle (strong LLM model)."""

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        use_case: str,
        scale_factor: int,
        agent_dir: str,
        llm_judge_dir: str | Path,
        oracle_reasoning_effort: str | None = None,
    ) -> None:
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model          # string, used for OpenRouterClient judge calls
        self._oracle_reasoning_effort = oracle_reasoning_effort
        self._query_id = query_id
        self._use_case = use_case
        self._scale_factor = scale_factor
        self._agent_dir = agent_dir
        self._llm_judge_dir = Path(llm_judge_dir)
        self.total_oracle_cost_usd = 0.0
        self._canonical_oracle_df: "pd.DataFrame | None" = None

        # Cross-plan per-operator oracle cache: (operator identity, input content) ->
        # field_answers. Shared by every oracle pipeline this evaluator runs, so a semantic
        # operator that appears in multiple plans is judged by the oracle only once.
        self._oracle_call_cache: dict = {}
        self._oracle_cache_stats: dict = {"hits": 0, "misses": 0}

        # Resolve oracle model string → pz.Model enum once at init so make_oracle_copy
        # receives the enum directly (avoids re-resolving on every execute_plan call and
        # surfaces bad model names early).
        self._oracle_pz_model = None
        try:
            from agent.physical_pipeline import _str_to_pz_model
            self._oracle_pz_model = _str_to_pz_model(oracle_model)
        except Exception as e:
            print(f"[QualityEvaluator] could not resolve oracle pz model '{oracle_model}': {e}")

        self._evaluator = None
        try:
            self._evaluator = _load_evaluator(use_case, scale_factor, agent_dir)
        except Exception as e:
            print(f"[QualityEvaluator] evaluator init failed: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        plan,                         # PhysicalPipeline
        plan_name: str,
        plan_context,                 # SubsetExecutionContext
        plan_output_df: pd.DataFrame, # already normalized for evaluator
    ) -> QualityResult:
        """Run oracle evaluation and return QualityResult."""
        oracle_df, oracle_context = self._get_oracle_context(plan, plan_name)
        if oracle_df is not None and not oracle_df.empty and (
            self._canonical_oracle_df is None or self._canonical_oracle_df.empty
        ):
            self._canonical_oracle_df = oracle_df
        gt_df = self._canonical_oracle_df

        # Per-op quality
        per_sem_op_quality: dict[str, float] = {}
        if oracle_context is not None:
            for stage_idx, info in plan_context.per_sem_op_info.items():
                op_name = info["op_name"]
                op_type = info["op_type"]
                oracle_info = oracle_context.per_sem_op_info.get(stage_idx)
                try:
                    if op_type in ("sem_filter", "sem_join") and oracle_info is not None:
                        max_pairs = len(plan_context.sampled_records) if op_type == "sem_join" else None
                        q = self._score_filter_join_op(info["samples"], oracle_info["samples"], max_pairs=max_pairs)
                        if q is not None:
                            per_sem_op_quality[op_name] = q
                    elif op_type == "sem_map":
                        q = self._score_map_op(info)
                        if q is not None:
                            per_sem_op_quality[op_name] = q
                except Exception as e:
                    print(f"[QualityEvaluator] per-op quality failed for {op_name}: {e}")

        # Plan quality
        quality = float("nan")
        quality_note = None
        if gt_df is None or gt_df.empty:
            # The oracle produced no rows on this subset, so there is nothing to score the plan
            # against — quality is undefined. Keep it NaN but attach a note so the agent's
            # observation can explain why (rather than showing a bare NaN).
            quality_note = "subset produces empty result, so plan quality is N/A"
        elif self._evaluator is not None:
            try:
                import dataclasses
                # Serialize any list-valued columns so the evaluator can hash them
                def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
                    df = df.copy()
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
                    return df
                plan_output_df = _sanitize(plan_output_df)
                gt_df = _sanitize(gt_df)
                qm = self._evaluator._evaluate_single_query(
                    self._query_id, plan_output_df, gt_df
                )
                qm_dict = dataclasses.asdict(qm)
                qm_type = type(qm).__name__
                if "Retrieval" in qm_type:
                    quality = float(qm_dict.get("f1_score", float("nan")))
                elif "Aggregation" in qm_type:
                    quality = 1.0 / (1.0 + float(qm_dict.get("relative_error", 1.0)))
                elif "Rank" in qm_type:
                    quality = float(qm_dict.get("spearman_correlation", float("nan")))
                elif "SingleAccuracy" in qm_type:
                    quality = float(qm_dict.get("accuracy", float("nan")))
            except Exception as e:
                print(f"[QualityEvaluator] plan quality evaluation failed: {e}")

        return QualityResult(quality=quality, per_sem_op_quality=per_sem_op_quality, quality_note=quality_note)

    # ------------------------------------------------------------------
    # Oracle plan execution and caching
    # ------------------------------------------------------------------

    def _install_oracle_cache(self, oracle_pipeline) -> None:
        """Wrap every semantic operator's Generator in the oracle pipeline (including a
        join's right sub-pipeline) with the shared cross-plan memoizing cache, so shared
        operators reuse oracle verdicts instead of re-calling the LLM."""
        for op in oracle_pipeline:
            pz_op = getattr(op, "_pz_op", None)
            gen = getattr(pz_op, "generator", None)
            if gen is None or isinstance(gen, _MemoizingGenerator):
                continue
            model_value = getattr(getattr(op, "model", None), "value", None)
            reff = getattr(pz_op, "reasoning_effort", None)
            pz_op.generator = _MemoizingGenerator(
                gen, self._oracle_call_cache, model_value, reff, self._oracle_cache_stats
            )

    def _get_oracle_context(
        self, plan, plan_name: str
    ) -> tuple[pd.DataFrame | None, object | None]:
        self._llm_judge_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._llm_judge_dir / f"Q{self._query_id}_{plan_name}_gt.csv"
        op_cache_path = self._llm_judge_dir / f"Q{self._query_id}_{plan_name}_op_decisions.json"

        if csv_path.exists():
            oracle_df = pd.read_csv(csv_path)
            oracle_context = self._load_op_decisions_cache(op_cache_path)
            return oracle_df, oracle_context

        try:
            # Prefer the resolved pz.Model enum; fall back to string (triggers _str_to_pz_model)
            oracle_model_arg = self._oracle_pz_model if self._oracle_pz_model is not None else self._oracle_model
            oracle_pipeline = plan.make_oracle_copy(oracle_model_arg, self._oracle_reasoning_effort)
            # Point every semantic operator's Generator at the shared cross-plan cache, so
            # operators this plan shares with an already-evaluated plan reuse the oracle's
            # verdicts instead of re-calling the LLM.
            self._install_oracle_cache(oracle_pipeline)
            subset_cache_path = (
                Path(__file__).parent / "datasubset" / self._use_case / f"sf_{self._scale_factor}" / f"Q{self._query_id}_subset.csv"
            )
            if not subset_cache_path.exists():
                raise FileNotFoundError(
                    f"Subset CSV not found at {subset_cache_path}. "
                    "A plan must be executed before the oracle can run."
                )
            # Oracle wall-clock latency is irrelevant (the oracle exists only for quality
            # scoring and its result is cached), so plan_dict is ignored here.
            hits0, misses0 = self._oracle_cache_stats["hits"], self._oracle_cache_stats["misses"]
            oracle_per_op_list, oracle_context, _ = oracle_pipeline.run_subset(
                subset_cache_path=str(subset_cache_path)
            )
            # Count the oracle's ACTUAL cost on every run (cache hits contribute ~0), so the
            # per-operator cache's savings show up in total_oracle_cost_usd.
            self.total_oracle_cost_usd += sum(
                float(row.get("cost_usd", 0.0) or 0.0) for row in oracle_per_op_list
            )
            hits = self._oracle_cache_stats["hits"] - hits0
            misses = self._oracle_cache_stats["misses"] - misses0
            # print(f"[QualityEvaluator] oracle op-cache for {plan_name}: {hits} hits, {misses} misses "
            #       f"({hits + misses} LLM calls, avoided so far: {self._oracle_cache_stats['hits']})")
        except Exception as e:
            print(f"[QualityEvaluator] oracle pipeline run failed: {e}")
            return None, None

        oracle_df = pd.DataFrame(oracle_context.output_records)
        oracle_df = self._normalize_df(oracle_df)
        oracle_df.to_csv(csv_path, index=False)
        self._save_op_decisions_cache(op_cache_path, oracle_context)

        oracle_result_path = (
            Path(__file__).parent / "datasubset" / self._use_case / f"sf_{self._scale_factor}" / f"Q{self._query_id}_oracle_result.csv"
        )
        if not oracle_result_path.exists():
            oracle_result_path.parent.mkdir(parents=True, exist_ok=True)
            oracle_df.to_csv(oracle_result_path, index=False)

        return oracle_df, oracle_context

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply use-case/query-specific column normalization (mirrors ExecutePlanTool)."""
        return normalize_eval_df(df, self._use_case, self._query_id)

    # ------------------------------------------------------------------
    # Op-level decision caching
    # ------------------------------------------------------------------

    def _save_op_decisions_cache(self, path: Path, oracle_context) -> None:
        cache: dict[str, dict] = {}
        for stage_idx, info in oracle_context.per_sem_op_info.items():
            decisions: dict[str, bool] = {}
            for inp, out in info["samples"]:
                if hasattr(inp, "source_indices"):
                    decisions[str(inp.source_indices)] = out is not None
            if decisions:
                cache[str(stage_idx)] = {"op_type": info["op_type"], "decisions": decisions}
        path.write_text(json.dumps(cache))

    def _load_op_decisions_cache(self, path: Path):
        """Return a minimal object with per_sem_op_info populated from JSON cache."""
        if not path.exists():
            return None
        try:
            from agent.physical_pipeline import SubsetExecutionContext
            raw = json.loads(path.read_text())
            per_sem_op_info: dict[int, dict] = {}
            for stage_str, v in raw.items():
                stage_idx = int(stage_str)
                # Represent cached decisions as (source_idx_str, passed_bool) tuples
                samples = [(src, passed) for src, passed in v["decisions"].items()]
                per_sem_op_info[stage_idx] = {
                    "op_name": f"_oracle_op{stage_idx}",
                    "op_type": v.get("op_type", "sem_filter"),
                    "attributes": {"model": self._oracle_model},
                    "samples": samples,
                }
            return SubsetExecutionContext(
                sampled_records=[],
                output_records=[],
                per_sem_op_info=per_sem_op_info,
                has_join=False,
                right_sampled_records=None,
            )
        except Exception as e:
            print(f"[QualityEvaluator] op decisions cache load failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Per-operator quality scoring
    # ------------------------------------------------------------------

    def _score_filter_join_op(self, plan_samples: list, oracle_samples: list, max_pairs: int | None = None) -> float | None:
        """Agreement rate between plan filter and oracle filter on the same records.

        For sem_join, pass max_pairs=n to randomly sample n pairs from the common
        set (instead of scoring all n^2 pairs).
        """
        def _to_decisions(samples) -> dict[str, bool]:
            d: dict[str, bool] = {}
            for item in samples:
                if not (isinstance(item, tuple) and len(item) == 2):
                    continue
                inp, out = item
                if hasattr(inp, "source_indices"):
                    # DataRecord sample: out is DataRecord or None
                    # source_indices may be a list (joined records) — stringify for hashing
                    d[str(inp.source_indices)] = out is not None
                elif isinstance(inp, str):
                    # Cached format: (source_idx_str, passed_bool)
                    d[inp] = bool(out)
            return d

        plan_dec = _to_decisions(plan_samples)
        oracle_dec = _to_decisions(oracle_samples)
        common = set(plan_dec) & set(oracle_dec)
        if not common:
            return None
        keys = list(common)
        if max_pairs is not None and len(keys) > max_pairs:
            keys = random.sample(keys, max_pairs)
        return sum(1 for k in keys if plan_dec[k] == oracle_dec[k]) / len(keys)

    def _oracle_generate(self, content: str | list) -> str:
        """Call oracle client; handle both str and (str, reasoning) return types.

        `content` may be a plain string or an OpenAI-style multimodal content
        list (mix of {"type": "text", ...} and {"type": "image_url", ...} parts).
        """
        result = self._oracle_client.generate(
            system="You are a rigorous evaluator. Return only valid JSON.",
            messages=[{"role": "user", "content": content}],
        )
        if isinstance(result, tuple):
            if len(result) >= 3 and isinstance(result[2], dict):
                self.total_oracle_cost_usd += float(result[2].get("cost_usd", 0.0) or 0.0)
            return result[0]
        return result

    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})

    @classmethod
    def _maybe_image_url(cls, value) -> str | None:
        """If `value` is a path to an image file, return a base64 data URL; else None.

        sem_map inputs of type ImageFilepath hold the on-disk path to the image.
        The oracle judge call is otherwise text-only, so without this the oracle
        never sees the pixels and scores vision maps ~0.
        """
        if not isinstance(value, str):
            return None
        if os.path.splitext(value)[1].lower() not in cls._IMAGE_EXTS:
            return None
        try:
            if not os.path.isfile(value):
                return None
            with open(value, "rb") as f:
                data = f.read()
            mime = mimetypes.guess_type(value)[0] or "image/jpeg"
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    def _score_map_op(self, info: dict) -> float | None:
        """Oracle judges whether plan's sem_map output fields are correct (batched call).

        Image-valued input fields are attached to the judge call as vision
        inputs so the oracle can actually verify vision-derived output fields.
        """
        samples = info["samples"]
        cols_names: list[str] = info["attributes"].get("cols", [])
        if not cols_names:
            return None

        valid: list[tuple] = [
            (inp, out) for inp, out in samples
            if out is not None and hasattr(inp, "source_indices")
        ]
        if not valid:
            return None

        def _dr_to_dict(dr) -> dict:
            schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            return {k: getattr(dr, k, None) for k in schema_cls.model_fields}

        n_fields = len(cols_names)
        # Build an OpenAI-style multimodal content list: per-record text, with any
        # image-valued input fields attached as image_url parts right after it.
        content: list[dict] = [{
            "type": "text",
            "text": (
                "You are evaluating whether a semantic map operator produced correct outputs.\n"
                f"Output fields to evaluate: {cols_names}"
            ),
        }]
        for i, (inp, out) in enumerate(valid):
            inp_d = _dr_to_dict(inp)
            out_d = _dr_to_dict(out)
            mapped = {
                k: ("" if out_d.get(k) is None or (
                    isinstance(out_d.get(k), float) and __import__("math").isnan(out_d.get(k))
                ) else out_d.get(k))
                for k in cols_names if k in out_d
            }
            image_urls: list[str] = []
            text_inp: dict = {}
            for k, v in inp_d.items():
                url = self._maybe_image_url(v)
                if url is not None:
                    text_inp[k] = "<image attached below>"
                    image_urls.append(url)
                else:
                    text_inp[k] = v
            content.append({
                "type": "text",
                "text": (
                    f"\nRecord {i}:\n"
                    f"  INPUT: {json.dumps(text_inp, default=str)}\n"
                    f"  OPERATOR output fields {cols_names}: {json.dumps(mapped, default=str)}"
                ),
            })
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})

        content.append({
            "type": "text",
            "text": (
                f"\n\nFor each record and each output field, score 1 if correct, 0 if incorrect.\n"
                f"Return ONLY valid JSON: "
                f'{{\"scores\": [[field0_rec0, field1_rec0, ...], [field0_rec1, ...], ...]}} '
                f"with {len(valid)} inner lists each of length {n_fields}."
            ),
        })

        try:
            response = self._oracle_generate(content)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            scores_2d: list[list] = data.get("scores", [])
            if not scores_2d:
                return None
            n_records = len(valid)
            # Average per field across records, then average across fields
            field_avgs = []
            for j in range(n_fields):
                field_vals = [
                    float(scores_2d[r][j])
                    for r in range(min(len(scores_2d), n_records))
                    if j < len(scores_2d[r])
                ]
                if field_vals:
                    field_avgs.append(sum(field_vals) / len(field_vals))
            return sum(field_avgs) / len(field_avgs) if field_avgs else None
        except Exception as e:
            print(f"[QualityEvaluator] sem_map scoring failed: {e}")
            return None
