#!/usr/bin/env python3
"""check_cost_model.py

Validate SampleBasedCostModel by estimating a plan BEFORE execution (naive
fallback path) and AFTER execution (sample-fitted path), then comparing both
estimates to the observed actuals.

Usage (run from repo root or agent_cost_model/):
    python3 agent_cost_model/check_cost_model.py \
        agent_cost_model/final_answer/sampleCost_agent_retry/Q5.json

    # Pick a specific plan from the JSON (default: first plan in plan_codes):
    python3 agent_cost_model/check_cost_model.py \
        agent_cost_model/final_answer/sampleCost_agent_retry/Q5.json --plan p2

    # Pass raw plan-code as a string directly:
    python3 agent_cost_model/check_cost_model.py --code "..."

What it checks
--------------
Before execution
  SampleBasedCostModel has no sampled data, so every operator falls back to
  _naive_op_stats.  For LLM operators (sem_filter, sem_join, …) this uses the
  MODEL_CARDS price table; for non-LLM operators it uses fixed constants.

After execution
  pipeline.run() produces per-operator stats (cost_usd, latency_s, num_records,
  num_passed, op_id, op_type).  These are fed into a fresh ResultsStore so
  _compute_op_stats can fit per-op averages.  estimate_plan should now use the
  sampled values (source="sampled" in the details dict).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- make agent/ and agent_cost_model/ importable from any working dir -------
_here = Path(__file__).resolve().parent          # agent_cost_model/
_root = _here.parent                             # repo root
for _p in [str(_here), str(_root), str(_root / "agent")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd  # noqa: E402

try:
    import palimpzest as pz  # noqa: E402
except ImportError as _e:
    sys.exit(f"palimpzest not installed: {_e}")

try:
    from agent.physical_pipeline import PhysicalPipeline  # noqa: E402
except ImportError:
    from physical_pipeline import PhysicalPipeline  # type: ignore  # noqa: E402

from cost_model_agent import (  # noqa: E402
    ResultsStore,
    iter_operators,
    get_op_type,
    get_op_id,
    get_op_model,
)
from sample_based_cost_model import SampleBasedCostModel  # noqa: E402

# Make src/ importable for MovieEvaluator
_src = _root / "src"
for _p in [str(_src)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from scenario.movie.evaluation.evaluate import MovieEvaluator  # noqa: E402
    _EVALUATOR_AVAILABLE = True
except ImportError:
    _EVALUATOR_AVAILABLE = False

DATA_DIR = _here / "dataset" / "movie"

import litellm as _litellm
_litellm.suppress_debug_info = True
_orig_completion = _litellm.completion
def _openrouter_completion(model, **kwargs):
    if model.startswith("vertex_ai/"):
        model = "google/" + model[len("vertex_ai/"):]
    elif model.startswith("together_ai/"):
        model = model[len("together_ai/"):]
    if not model.startswith("openrouter/"):
        model = "openrouter/" + model
    return _orig_completion(model=model, **kwargs)
_litellm.completion = _openrouter_completion

# ---------------------------------------------------------------------------
# Plan builder
# ---------------------------------------------------------------------------

def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def build_plan(code: str, plan_name: str = "plan") -> PhysicalPipeline:
    """Execute plan-building code and return the resulting PhysicalPipeline.

    The code string may use: PhysicalPipeline, pz, pd, load_data, plan_name.
    The last expression (e.g. bare `pipeline`) is ignored by exec(); we find
    the pipeline by looking for the variable named 'pipeline', then falling
    back to any PhysicalPipeline in the local namespace.
    """
    ns: dict = {
        "PhysicalPipeline": PhysicalPipeline,
        "pz": pz,
        "pd": pd,
        "load_data": load_data,
        "plan_name": plan_name,
    }
    exec(compile(code, "<plan_code>", "exec"), ns)  # noqa: S102

    # Prefer the variable explicitly named 'pipeline', then scan the namespace.
    if "pipeline" in ns and isinstance(ns["pipeline"], PhysicalPipeline):
        return ns["pipeline"]
    for val in ns.values():
        if isinstance(val, PhysicalPipeline):
            return val
    raise RuntimeError(
        "Plan code did not assign a PhysicalPipeline to 'pipeline'. "
        "Make sure the code ends with `pipeline = ...` or similar."
    )


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_cost_model(code: str, plan_name: str = "plan", query_id: int | None = None) -> None:
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Plan: {plan_name!r}")
    print(sep)

    # ------------------------------------------------------------------
    # Step 1: build plan (no execution yet)
    # ------------------------------------------------------------------
    print("\n[1] Building plan (no execution)…")
    pipeline = build_plan(code, plan_name)
    ops = list(iter_operators(pipeline))
    print(f"    Operators  : {[get_op_type(op) for op in ops]}")
    print(f"    Models     : {[get_op_model(op) for op in ops]}")
    print(f"    Source rows: {len(pipeline._df):,}  (pipeline._df)")

    # ------------------------------------------------------------------
    # Step 2: estimate BEFORE execution (naive fallback)
    # ------------------------------------------------------------------
    print("\n[2] Estimating cost BEFORE execution (naive fallback)…")
    empty_store = ResultsStore(rows=[])
    model_before = SampleBasedCostModel(empty_store)
    est_before = model_before.estimate_plan(pipeline)
    print(f"    {est_before}")
    print("    Per-operator breakdown:")
    for op_id, detail in est_before.details.items():
        src = detail.get("source", "naive")
        card = detail.get("cardinality_in", "?")
        cost = detail.get("cost", 0.0)
        print(f"      {op_id:12s}  cardinality_in={card:6}  cost=${cost:.6f}  [{src}]")

    # ------------------------------------------------------------------
    # Step 3: execute the plan (costs real money / time for LLM calls)
    # ------------------------------------------------------------------
    print("\n[3] Executing plan — this may call LLMs and incur real cost…")
    result_collection, per_op_list, plan_dict = pipeline.run()
    actual_cost = plan_dict["cost_usd"]
    actual_time = plan_dict["latency_s"]
    n_out = len(result_collection.data_records)
    print(f"    Actual cost   : ${actual_cost:.4f}")
    print(f"    Actual latency: {actual_time:.2f}s")
    print(f"    Output records: {n_out}")
    print("    Per-operator stats from run():")
    for row in per_op_list:
        print(f"      op_id={row.get('op_id','?')!r:14s}  op_type={row.get('op_type','?')!r:14s}"
              f"  num_records={row.get('num_records',0):4}  cost=${row.get('cost_usd',0):.6f}")

    # Check that op_id / op_type are present (these are required for the cost
    # model to key stats per operator; missing keys was a bug).
    missing = [i for i, r in enumerate(per_op_list) if not r.get("op_id") or not r.get("op_type")]
    if missing:
        print(f"  WARNING: rows {missing} in per_op_list are missing 'op_id' or 'op_type'.")
        print("           _compute_op_stats will group them all under key=None — fix run().")

    # ------------------------------------------------------------------
    # Step 4: feed execution results into a new ResultsStore
    # ------------------------------------------------------------------
    print("\n[4] Fitting cost model on observed operator stats…")
    op_store = ResultsStore(rows=per_op_list)
    model_after = SampleBasedCostModel(op_store)
    print("    Fitted op_id → stats:")
    for op_id, stats in model_after.op_id_to_stats.items():
        print(f"      {op_id!r:14s}  cost/rec=${stats['cost']:.6f}  "
              f"time/rec={stats['time']:.4f}s  selectivity={stats['selectivity']:.3f}")
    if not model_after.op_id_to_stats:
        print("    (empty — per_op_list lacked op_id/op_type keys; see warning above)")

    # ------------------------------------------------------------------
    # Step 5: estimate AFTER execution (sample-based)
    # ------------------------------------------------------------------
    print("\n[5] Estimating cost AFTER execution (sample-based)…")
    est_after = model_after.estimate_plan(pipeline)
    print(f"    {est_after}")
    print("    Per-operator breakdown:")
    for op_id, detail in est_after.details.items():
        src = detail.get("source", "naive")
        card = detail.get("cardinality_in", "?")
        cost = detail.get("cost", 0.0)
        print(f"      {op_id:12s}  cardinality_in={card:6}  cost=${cost:.6f}  [{src}]")

    # ------------------------------------------------------------------
    # Step 6: evaluate quality against ground truth
    # ------------------------------------------------------------------
    if query_id is not None and _EVALUATOR_AVAILABLE:
        print(f"\n[6] Evaluating quality for Q{query_id}…")
        gt_path = _root / "files" / "movie" / "raw_results" / "ground_truth" / f"Q{query_id}.csv"
        if not gt_path.exists():
            print(f"    Ground truth not found: {gt_path}")
        else:
            try:
                gt_df = pd.read_csv(gt_path)
                result_df = result_collection.to_df()
                evaluator = MovieEvaluator(use_case="movie", scale_factor=2000)
                metric = evaluator._evaluate_single_query(query_id, result_df, gt_df)
                import dataclasses
                print(f"    Quality metrics: {dataclasses.asdict(metric)}")
            except Exception as exc:
                print(f"    Evaluation failed: {exc}")
    elif query_id is None:
        print("\n[6] Skipping quality evaluation (no query_id).")
    else:
        print("\n[6] Skipping quality evaluation (MovieEvaluator not importable).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test SampleBasedCostModel before and after plan execution."
    )
    ap.add_argument(
        "query_json",
        nargs="?",
        default = "final_answer/sampleCost_agent/Q3.json",
        help="Path to a Q<N>.json file with a 'plan_codes' dict (agent output).",
    )
    ap.add_argument(
        "--plan",
        default=None,
        help="Plan name to test (default: first key in plan_codes).",
    )
    ap.add_argument(
        "--code",
        default=None,
        help="Raw plan-code string (alternative to --query-json).",
    )
    ap.add_argument(
        "--query-id",
        type=int,
        default=None,
        help="Query number for quality evaluation (default: inferred from filename).",
    )
    args = ap.parse_args()

    if args.query_json:
        path = Path(args.query_json)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        with path.open() as f:
            data = json.load(f)
        plan_codes: dict = data.get("plan_codes", {})
        if not plan_codes:
            sys.exit("No 'plan_codes' key found in the JSON file.")
        name = args.plan or next(iter(plan_codes))
        if name not in plan_codes:
            sys.exit(f"Plan {name!r} not in plan_codes. Available: {list(plan_codes)}")
        # Always infer query_id from filename like "Q6.json" → 6
        stem = path.stem  # e.g. "Q6"
        if not (stem.startswith("Q") and stem[1:].isdigit()):
            sys.exit(f"Cannot infer query_id from filename {path.name!r}; expected Q<N>.json")
        qid = int(stem[1:])
        check_cost_model(plan_codes[name], plan_name=name, query_id=qid)

    elif args.code:
        check_cost_model(args.code, plan_name=args.plan or "plan", query_id=args.query_id)

    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
