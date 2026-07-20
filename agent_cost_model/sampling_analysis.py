#!/usr/bin/env python
"""Analyze how the LLM sampler's cosine-similarity scores treat a query's ground
truth, and how importance sampling would draw from them.

This mirrors ``agent_cost_model/llm_sampler.py``: a row's score is the SUM of
cosine similarities between the (text-embedded) query and each ACTIVE modality's
cached row embedding, and importance sampling draws ``SAMPLE_SIZE`` rows without
replacement with probability softmax(scores). We reuse the same shared per-use-
case embedding cache, so nothing is recomputed except the one query embedding.

Run from the repo root with a gemini-embedding-2-capable key in the environment:

    export OPENROUTER_API_KEY=sk-or-...
    .venv/bin/python agent_cost_model/sampling_analysis.py

Edit the CONSTANTS block below for a different query / use case / modalities.
"""
from __future__ import annotations
import json, os, re, sys, time
import pandas as pd
import numpy as np
import httpx
import matplotlib
matplotlib.use("Agg")  # file output only; no display needed
import matplotlib.pyplot as plt

# Reuse the production sampler's filter/keyword helpers (pure, no network calls).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_sampler import LLM_Sampler

# ============================ EDIT PER QUERY ============================
USE_CASE    = "ecomm"
QUERY_ID    = "4"                                # results.json key for this query
QUERY       = "blue"              # description the sampler embeds
MODALITIES  = ["image"]                         # subset of {"text", "image"} to score on
# GT = [10037, 10102, 3312, 3462, 41825]  # 2
GT          = pd.read_csv("agent_cost_model/styles_details_blue.csv")['prod_id'].tolist()  # ground-truth ids for this query
# GT = [1623, 5299, 5300, 5303, 5314, 1624, 5301] #1
FILTER      = ""                          # optional row predicate, e.g. 'row["price"] <= 500'; empty = no filter
KEYWORD     = False                           # +1.0 per QUERY concept group present in a row ('.'=AND groups, ','=OR synonyms)

EMBED_MODEL = "google/gemini-embedding-2"      # must match the cached embedding space
SAMPLE_SIZE = 10                               # llm_sampler.NUM_SAMPLES
SEED        = 42                               # llm_sampler.SUBSET_SEED
N_TRIALS    = 20000                            # Monte-Carlo trials for inclusion prob
# =======================================================================

# ---- derived paths (match llm_sampler's cache layout; usually no need to edit)
SLUG      = re.sub(r"[^A-Za-z0-9._-]", "_", EMBED_MODEL)
CACHE_DIR = f"agent_cost_model/sampling/{USE_CASE}/cached_results"
SHARED    = f"{CACHE_DIR}/{SLUG}_embeddings.npz"     # shared cache: ids/text + image_ids/image
RESULTS   = f"agent_cost_model/sampling/{USE_CASE}/results.json"
PLOT_OUT  = f"agent_cost_model/sampling/{USE_CASE}/Q{QUERY_ID}_blue_{'keyword+cos' if KEYWORD else 'cos'}_cdf.png"
SOURCE_CSV = f"agent_cost_model/dataset/{USE_CASE}/styles_details_Q4.csv"  # row attributes for FILTER
ID_COL     = "prod_id"                              # key linking source rows to embedding ids
BASE_URL  = "https://openrouter.ai/api/v1"
API_KEY   = os.environ.get("OPENROUTER_API_KEY")

# npz keys per modality: (id_array, vec_array)
_KEYS = {"text": ("ids", "text"), "image": ("image_ids", "image")}


# ----------------------------------------------------------- load embeddings
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if (np.isfinite(v).all() and n > 0) else None


def load_modality(cache, modality):
    id_key, vec_key = _KEYS[modality]
    ids = [int(x) for x in cache[id_key].tolist()]
    vecs = cache[vec_key]
    out = {}
    for i, rid in enumerate(ids):
        u = _unit(vecs[i].astype(np.float64))
        if u is not None:
            out[rid] = u
    return out


if not os.path.exists(SHARED):
    sys.exit(f"cache not found: {SHARED} (run from repo root)")
for m in MODALITIES:
    if m not in _KEYS:
        sys.exit(f"unknown modality {m!r}; choose from {list(_KEYS)}")

_cache = np.load(SHARED, allow_pickle=True)
EMB = {m: load_modality(_cache, m) for m in MODALITIES}          # modality -> {id: unit vec}
POOL = sorted(set().union(*[set(EMB[m]) for m in MODALITIES]))   # ids with >=1 active vector


# ------------------------------------------ working set = SOURCE_CSV (+ FILTER)
# The rows we score are those in SOURCE_CSV (a per-query table that may be a
# subset of the full dataset), intersected with what's embedded in the shared
# cache -- we never score against the whole cache directly. The optional FILTER
# further restricts these rows. Mirrors llm_sampler: the filter runs FIRST, so
# cosine similarity (and every stat/plot downstream) sees only survivors.
#
# Reuse the production sampler's own filter/keyword logic (_compile_filter,
# _words, _row_keyword_words, _keyword_cols) so this analysis stays in lock-step
# with it. We pass llm_client=None and never call sample()/identify_sampling_
# targets(), so no LLM or network request is made here -- only its pure helpers.
if not os.path.exists(SOURCE_CSV):
    sys.exit(f"source table not found: {SOURCE_CSV} (run from repo root)")
_src = pd.read_csv(SOURCE_CSV)
_rows = {int(r[ID_COL]): r for _, r in _src.iterrows()}             # id -> row Series
_sampler = LLM_Sampler(
    query_id=QUERY_ID, use_case=USE_CASE, query_text=QUERY, df=_src,
    llm_client=None, embedding_model=EMBED_MODEL, id_col=ID_COL,
    sample_size=SAMPLE_SIZE, seed=SEED,
    api_key=API_KEY or "unused",  # placeholder: we only call its pure helpers, never its embed/HTTP methods
)

n_cache = len(POOL)
POOL = [rid for rid in POOL if rid in _rows]                        # restrict to SOURCE_CSV rows
print(f"source {SOURCE_CSV}: {len(POOL)} of {n_cache} cached rows are in the source table")

_filter_fn = _sampler._compile_filter(FILTER)
if _filter_fn is not None:
    n_before = len(POOL)
    POOL = [rid for rid in POOL if _filter_fn(_rows[rid])]
    print(f"filter {FILTER!r}: {len(POOL)} of {n_before} source rows survive"
          f" ({n_before - len(POOL)} filtered out)")

POOL = sorted(POOL)
N = len(POOL)
_survivor = set(POOL)                                               # fast membership for plots
missing = [g for g in GT if g not in _survivor]
print(f"use case={USE_CASE}  query_id={QUERY_ID}  modalities={MODALITIES}")
print(f"scored rows: {N}   GT covered: {[g for g in GT if g in _survivor]}"
      + (f"   MISSING: {missing}" if missing else ""))


# ------------------------------------------------------------- embed query
def embed_text(text):
    if not API_KEY:
        sys.exit("Set OPENROUTER_API_KEY (a key that can call EMBED_MODEL).")
    t0 = time.time()
    r = httpx.post(f"{BASE_URL}/embeddings",
                   headers={"Authorization": f"Bearer {API_KEY}"},
                   json={"model": EMBED_MODEL, "input": text, "encoding_format": "float",
                         "usage": {"include": True}}, timeout=120.0)
    if r.status_code != 200:
        sys.exit(f"embed failed {r.status_code}: {r.text[:400]}")
    j = r.json()
    cost = (j.get("usage") or {}).get("cost", 0.0)
    return np.asarray(j["data"][0]["embedding"], dtype=np.float64), cost, time.time() - t0


q, qcost, qlat = embed_text(QUERY)
q = q / np.linalg.norm(q)
print(f"embedded query {QUERY!r}  (${qcost:.8f}, {qlat:.2f}s)\n")


# ------------------------------------------- score = sum of per-modality cos
# One matmul per modality (vectors are already unit rows, so dot == cosine).
per_mod = {}
for m in MODALITIES:
    mids = list(EMB[m])
    Mmat = np.stack([EMB[m][r] for r in mids])          # (n_m, D) unit rows
    with np.errstate(all="ignore"):                     # silence spurious matmul FPE flags
        dots = Mmat @ q
    per_mod[m] = {mids[i]: float(dots[i]) for i in range(len(mids))}
sims = {rid: sum(per_mod[m].get(rid, 0.0) for m in MODALITIES) for rid in POOL}

# --------------------- optional keyword boost (+1.0 per matching concept group)
# Mirrors llm_sampler's keyword flag: split QUERY into concept groups (period "."
# separates concepts / AND, comma "," lists synonyms / OR) and add +1.0 for EACH
# group that has >=1 word present in the row's text. e.g. "black. watch, bag":
# a row containing "black" and "bag" hits both groups -> +2.0.
kw_boosted = set()
if KEYWORD:
    target_groups = _sampler._target_groups(QUERY)
    for rid in POOL:
        row_words = _sampler._row_keyword_words(_rows[rid])
        n_hits = sum(1 for g in target_groups if g & row_words)
        if n_hits:
            sims[rid] += n_hits
            kw_boosted.add(rid)
    print(f"keyword boost {[sorted(g) for g in target_groups]}: +1.0/group to "
          f"{len(kw_boosted)} of {N} rows ({len(kw_boosted & set(GT))} of them GT)")

vals = np.array(sorted(sims.values(), reverse=True))


def rank(s):  return int((vals > s).sum()) + 1
def pct(s):   return 100 * (vals < s).mean()


label = " + ".join(f"cos(query, {m})" for m in MODALITIES) + (" + keyword_group_hits" if KEYWORD else "")
print(f"=== score = {label}   distribution over {N} rows ===")
print(f"max={vals[0]:.4f}  p95={np.percentile(vals,95):.4f}  p90={np.percentile(vals,90):.4f}"
      f"  median={np.median(vals):.4f}  min={vals[-1]:.4f}\n")

print("=== GROUND-TRUTH scores ===")
per_hdr = "  ".join(f"{m[:3]}" for m in MODALITIES) if len(MODALITIES) > 1 else ""
print(f"{'id':>7} {'score':>8} {'rank':>10} {'pctile':>7}  {per_hdr:<{max(1,len(per_hdr))}}")
for g in sorted(GT, key=lambda g: -sims.get(g, -1e9)):
    if g not in sims:
        print(f"{g:>7}  (not scored)"); continue
    s = sims[g]
    breakdown = ("  ".join(f"{per_mod[m].get(g, float('nan')):.2f}" for m in MODALITIES)
                 if len(MODALITIES) > 1 else "")
    print(f"{g:>7} {s:>8.4f} {str(rank(s))+'/'+str(N):>10} {pct(s):>6.0f}%  "
          f"{breakdown:<{max(1,len(per_hdr))}}")

# --------------------------------------------- what the sampler actually kept
SAMPLED = []
if os.path.exists(RESULTS):
    with open(RESULTS) as fh:
        entry = json.load(fh).get(str(QUERY_ID), {})
    SAMPLED = [int(x) for x in entry.get("sampled_ids", [])]
    if SAMPLED:
        smean = np.mean([sims[i] for i in SAMPLED if i in sims])
        print(f"\nresults.json sampled_ids ({entry.get('sample_method','?')}): "
              f"mean score={smean:.4f}, ranks={sorted(rank(sims[i]) for i in SAMPLED if i in sims)}")

print("\n=== top-15 rows by score ===")
for i in sorted(sims, key=lambda i: -sims[i])[:15]:
    tag = " <-- GT" if i in GT else (" <-- sampled" if i in SAMPLED else "")
    print(f"  {i:>7} {sims[i]:.4f}{tag}")


# ================= IMPORTANCE SAMPLING (softmax over scores) =================
def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


order = sorted(POOL, key=lambda i: -sims[i])          # high score first
weights = softmax([sims[i] for i in order])           # p(pick) per row, softmax(score)
w_by_id = dict(zip(order, weights))

# One realized draw with the sampler's seed (size-SAMPLE_SIZE, no replacement).
draw = np.random.default_rng(SEED).choice(order, size=min(SAMPLE_SIZE, N),
                                          replace=False, p=weights)
draw = [int(x) for x in draw]

# Monte-Carlo inclusion probability: P(id in a size-k importance sample).
rng = np.random.default_rng(SEED)
incl = np.zeros(N, dtype=np.float64)
idx = {rid: j for j, rid in enumerate(order)}
for _ in range(N_TRIALS):
    for rid in rng.choice(order, size=min(SAMPLE_SIZE, N), replace=False, p=weights):
        incl[idx[int(rid)]] += 1
incl /= N_TRIALS
incl_by_id = {rid: incl[idx[rid]] for rid in order}

print(f"\n=== IMPORTANCE SAMPLING: softmax(score), size {SAMPLE_SIZE}, seed {SEED} ===")
print(f"one realized draw ({len(draw)} ids): {draw}")
print(f"  -> GT drawn: {[g for g in GT if g in draw]}")
print(f"\n{'id':>7} {'score':>8} {'weight':>10} {'incl@%d' % SAMPLE_SIZE:>9}   name")
print("  (weight = softmax share of one pick; incl = P(id in the size-k subset), "
      f"{N_TRIALS} trials)")
for g in sorted(GT, key=lambda g: -sims.get(g, -1e9)):
    if g not in sims:
        print(f"{g:>7}  (not scored)"); continue
    print(f"{g:>7} {sims[g]:>8.4f} {w_by_id[g]:>10.5f} {100*incl_by_id[g]:>8.1f}%")
gt_incl = np.mean([incl_by_id[g] for g in GT if g in incl_by_id])
print(f"\nmean GT inclusion probability: {100*gt_incl:.1f}%   "
      f"(uniform baseline would be {100*min(SAMPLE_SIZE, N)/N:.1f}%)")


# ================================ CDF PLOT ================================
def ecdf(values):
    xs = np.sort(np.asarray(values, dtype=np.float64))
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


fig, ax = plt.subplots(figsize=(7.5, 5))
xs, ys = ecdf(list(sims.values()))
ax.plot(xs, ys, color="0.35", ls="none", marker=".", ms=3, label=f"all rows (n={N})")

# per-modality CDFs when the score is a sum of more than one modality
# (restricted to survivors so it lines up with the filtered "all rows" curve)
if len(MODALITIES) > 1:
    for m in MODALITIES:
        mxs, mys = ecdf([per_mod[m][i] for i in POOL if i in per_mod[m]])
        ax.plot(mxs, mys, ls="none", marker=".", ms=2, alpha=0.8, label=f"cos(query, {m})")

# mark each GT id on the combined-score CDF (markers only, no per-point labels)
def cdf_at(x):  return float((xs <= x).mean())
gt_scored = sorted((g for g in GT if g in sims), key=lambda g: sims[g])
for g in gt_scored:
    s = sims[g]
    ax.scatter([s], [cdf_at(s)], color="crimson", zorder=5, s=30)
ax.scatter([], [], color="crimson", s=30, label="ground truth")

# right-side snapshot: the top min(10, len(GT)) GT rows by score (best rank first)
top_gt = sorted(gt_scored, key=lambda g: sims[g], reverse=True)[:min(10, len(GT))]
if top_gt:
    body = "\n".join(f"{g:>7}   r{rank(sims[g])}/{N}" for g in top_gt)
    ax.text(1.02, 1.0, f"top {len(top_gt)} GT by score\n{body}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            family="monospace", color="crimson",
            bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))

ax.set_xlabel(f"score  ({label})")
ax.set_ylabel("empirical CDF")
ax.set_title(f"{USE_CASE} Q{QUERY_ID}: cosine-score CDF — {QUERY!r}")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")  # bbox_inches keeps the right-side box in frame
print(f"\nwrote CDF plot -> {PLOT_OUT}")
