"""
Execute one or more query-plan templates, evaluate them, and persist raw results
and performance metrics.

This mirrors the CustomCostModel agent's plan execution + final evaluation
(`WritePlanTool` / `ExecutePlanTool` / `CostModelAgent._run_final_evaluation` in
agent_cost_model/cost_model_agent.py):
  * the plan is a bare-expression script whose LAST expression yields the plan
    object — a PhysicalPipeline (physical-pipeline wrapped) or a pz.Dataset
    (PZ wrapped). Do NOT call .run() in the plan; this script runs it.
  * plan code is run with the same injected env the agent gives it: `load_data`,
    `add_image_data`, `plan_name`, `data_dir`, `pd`, `os`, `pz`, `PhysicalPipeline`.
    `load_data` reads CSVs from ../agent_cost_model/experiments/dataset/{use_case}/sf_{sf}/.
  * the plan file is treated as a Jinja template. The only supported template
    variable is `model`, which should be used for every operator in that plan.
  * every query/model/run combination is executed with a fresh environment so the
    plan is independently instantiated each time.

Plan file: files/{use_case}/queries/dialects/palimpzest/Q{id}_palimpzest.txt

Outputs:
  raw results -> files/{use_case}/raw_results/palimpzest/{plan name}/sf_{scale_factor}/Q{id}_{model}_{runcount}.csv
  metrics     -> files/{use_case}/metrics/{plan name}.json

Usage:
  1. Edit `QUERY_IDS` and `MODELS` below.
  2. Run: python3 single_run.py --use-case ecomm --sf 500 --num-runs 2
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import math
import os
import re
import sys
import time
from pathlib import Path

from jinja2 import Template

PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))           # agent_cost_model submodule
sys.path.insert(0, str(PROJECT_ROOT))            # SemBench source
sys.path.insert(0, str(PROJECT_ROOT / "src"))    # runner/, scenario/, evaluator/

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import pandas as pd

# Importing the runner module patches litellm.completion to route every LLM call
# through OpenRouter, which the PZ model strings in the plans rely on.
from runner.generic_palimpzest_runner.generic_palimpzest_runner import (  # noqa: F401
    GenericPalimpzestRunner,
)

import palimpzest as pz
from agent.physical_pipeline import PhysicalPipeline
from agent_cost_model.experiments.SemBench.quality_evaluator import load_evaluator, normalize_eval_df

# Default model used to run a PZ-wrapped (pz.Dataset) plan, whose operators do not
# pin a model themselves. PhysicalPipeline plans pin models per operator and ignore this.
DEFAULT_MODEL = "pz.Model.GOOGLE_GEMINI_2_5_FLASH"
_MODEL_MAP = {
    "gemini-2.0-flash": "GEMINI_2_0_FLASH",
    "gemini-2.5-flash": "GEMINI_2_5_FLASH",
    "gpt-4o-mini": "GPT_4o_MINI",
    "gpt-5-mini": "GPT_5_MINI",
}

# Edit these lists directly to choose which queries and models to run.
# QUERY_IDS = [id for id in range(1,14) if id not in [7,10,11]]
QUERY_IDS =[10]
MODELS = ["pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE", "pz.Model.GOOGLE_GEMINI_2_5_FLASH"]
# "pz.Model.GOOGLE_GEMINI_2_5_PRO"

# Plan filename: Q{id}_{plan name}(.txt)
_NAME_RE = re.compile(r"^Q(?P<qid>\d+)_(?P<plan>.+)$")


def parse_plan_filename(txt_path: Path) -> tuple[int, str]:
    """Return (query_id, plan_name) parsed from `Q{id}_{plan}`."""
    m = _NAME_RE.match(txt_path.stem)
    if not m:
        raise ValueError(
            f"Filename {txt_path.name!r} does not match 'Q{{id}}_{{plan name}}'"
        )
    return int(m.group("qid")), m.group("plan")


def extract_plan_code(text: str) -> str:
    """Return the executable plan code from a raw script or an agent trace.

    Agent traces end with a `Final Plan` section (a `====` banner) followed by
    the code; a raw script is returned as-is (minus any ```python fences).
    """
    if "Final Plan" in text:
        after = text.rsplit("Final Plan", 1)[1]
        lines = after.splitlines()
        # Drop the leading banner/blank lines (`====...` and empties).
        while lines and set(lines[0].strip()) <= {"="}:
            lines.pop(0)
        text = "\n".join(lines)
    text = text.strip()
    # Strip a surrounding ```python ... ``` fence if present.
    fence = re.match(r"^```(?:python|py)?\s*\n(.*)\n```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return text.strip()


def resolve_data_dir(use_case: str, scale_factor: int, mode: str) -> Path:
    """Dir the plan reads source tables from, per run mode.

    physical: PhysicalPipeline plans read CSVs from the CostModel agent's dataset.
    pz:       direct PZ dialect plans read .parquet from the benchmark data dir.
    """
    if mode == "pz":
        return PROJECT_ROOT / "files" / use_case / "data" / f"sf_{scale_factor}"
    ds_root = WORKSPACE_ROOT / "agent_cost_model" / "experiments" / "dataset" / use_case
    for cand in (ds_root / f"sf_{scale_factor}", ds_root):
        if cand.exists():
            return cand
    return PROJECT_ROOT / "files" / use_case / "data" / f"sf_{scale_factor}"


def sanitize_model_name(model_name: str) -> str:
    """Filesystem-safe model identifier used in output filenames."""
    model_tag = model_name.removeprefix("pz.Model.")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model_tag)


def normalize_model_for_pz(model_name: str) -> str:
    """Normalize a CLI model value into a `pz.Model` attribute name."""
    if model_name.startswith("pz.Model."):
        return model_name.split(".", 2)[-1]
    if model_name in _MODEL_MAP:
        return _MODEL_MAP[model_name]
    if hasattr(pz.Model, model_name):
        return model_name
    return "GEMINI_2_5_FLASH"


def resolve_plan_template(use_case: str, query_id: int) -> Path:
    """Resolve the fixed template path for a given query id."""
    if use_case == "ecomm":
        txt_path = (
            PROJECT_ROOT
            / "files"
            / use_case
            / "queries"
            / "dialects"
            / "palimpzest"
            / f"Q{query_id}_palimpzest.txt"
        )
    elif use_case == "movie":
        txt_path = (
            PROJECT_ROOT
            / "files"
            / use_case
            / "query"
            / "palimpzest"
            / f"Q{query_id}_palimpzest.txt"
        )
    if not txt_path.exists():
        raise SystemExit(f"Plan file not found for Q{query_id}: {txt_path}")
    return txt_path


def render_plan_code(template_text: str, model_name: str, sf: int) -> str:
    """Render the plan template with the selected model configuration."""
    return extract_plan_code(Template(template_text).render(model=model_name, sf=sf))


def load_metrics(metrics_path: Path) -> dict[str, dict]:
    """Load the metrics JSON as a dict keyed by `{query_id}_{runcount}`."""
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text())
    except Exception:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _metric_sort_key(key: str) -> tuple[int, str, int]:
    parts = key.split("_")
    try:
        qid = int(parts[0])
    except (ValueError, IndexError):
        qid = 0
    try:
        run = int(parts[-1])
    except (ValueError, IndexError):
        run = 0
    model = "_".join(parts[1:-1])
    return (qid, model, run)


def append_metric(metrics_path: Path, record_key: str, record: dict) -> None:
    """Upsert a compact metric record into the metrics JSON."""
    store = load_metrics(metrics_path)
    store[record_key] = record
    store = dict(sorted(store.items(), key=lambda kv: _metric_sort_key(kv[0])))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(store, indent=2))


def build_plan_env(plan_name: str, data_dir: str) -> dict:
    """Injected globals for the plan script — mirrors the CostModel agent's
    WritePlanTool `plan_variables` plus the pz-dialect helpers (pd/os/data_dir)."""

    def load_data(filename: str) -> pd.DataFrame:
        """Read a source table, tolerating a .csv/.parquet extension mismatch.

        PhysicalPipeline plans ask for `foo.csv`; the PZ dialect reads `foo.parquet`.
        Whichever exists under data_dir is used, so a plan works regardless of which
        extension the on-disk file happens to have.
        """
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            stem, _ = os.path.splitext(filename)
            for alt in (stem + ".csv", stem + ".parquet"):
                if os.path.exists(os.path.join(data_dir, alt)):
                    path = os.path.join(data_dir, alt)
                    break
        return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
        pipeline.map(
            udf=lambda row: {col_name: os.path.join(data_dir, "images", str(int(float(row["prod_id"]))) + ".jpg")},
            cols=[{"name": col_name, "type": pz.ImageFilepath, "description": ""}],
        )
        return pipeline

    return {
        "PhysicalPipeline": PhysicalPipeline,
        "pz": pz,
        "pd": pd,
        "os": os,
        "load_data": load_data,
        "add_image_data": add_image_data,
        "plan_name": plan_name,
        "data_dir": data_dir,
    }


def make_pz_config(model_name: str) -> "pz.QueryProcessorConfig":
    """Default config for running a PZ-wrapped (pz.Dataset) plan."""
    model_attr = normalize_model_for_pz(model_name)
    model = getattr(pz.Model, model_attr)
    return pz.QueryProcessorConfig(
        policy=pz.MaxQuality(),
        execution_strategy="parallel",
        available_models=[model],
        max_workers=20,
        join_parallelism=20,
        verbose=False,
        progress=True,
    )


def _eval_last_expression(plan_code: str, env: dict):
    """Exec the plan script and return the value of its final bare expression."""
    tree = ast.parse(plan_code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        exec(compile(tree, "<plan>", "exec"), env)
        expr = ast.Expression(last.value)
        ast.fix_missing_locations(expr)
        return eval(compile(expr, "<plan>", "eval"), env)
    # No trailing expression: fall back to a conventionally-named plan variable.
    exec(compile(tree, "<plan>", "exec"), env)
    for name in ("pipeline", "output", "ds", "dataset"):
        if name in env:
            return env[name]
    raise ValueError(
        "Plan must end with a bare expression yielding the plan object "
        "(PhysicalPipeline or pz.Dataset)."
    )


def execute_plan(plan_code, env, model_name):
    """Run the plan and return (results_df, latency_s, cost_usd)."""
    plan_obj = _eval_last_expression(plan_code, env)

    start = time.time()
    # PhysicalPipeline.run() -> (DataRecordCollection, per_op_list, plan_dict)
    if isinstance(plan_obj, PhysicalPipeline):
        collection, op_results, plan_dict = plan_obj.run()
        wall = time.time() - start
        results_df = collection.to_df()
        latency = plan_dict.get("latency_s", wall)
        cost = sum(row.get("cost_usd", 0.0) for row in op_results)
        return results_df, latency, cost

    # Already-executed DataRecordCollection (plan called .run() itself).
    if hasattr(plan_obj, "execution_stats") and hasattr(plan_obj, "to_df"):
        collection = plan_obj
    else:
        # pz.Dataset -> run with a default config.
        collection = plan_obj.run(make_pz_config(model_name))

    wall = time.time() - start
    results_df = collection.to_df() if hasattr(collection, "to_df") else collection
    stats = getattr(collection, "execution_stats", None)
    latency = (
        getattr(stats, "plan_execution_time", None)
        or getattr(stats, "total_execution_time", None)
        or wall
    )
    cost = getattr(stats, "total_execution_cost", None) or getattr(stats, "plan_execution_cost", 0.0)
    return results_df, latency, float(cost or 0.0)


def evaluate(use_case, scale_factor, plan_name, query_id, results_df, gt_path):
    """Normalize plan output and score it against ground truth. Returns (metric_type, quality)."""
    evaluator = load_evaluator(use_case, scale_factor)
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
    else:
        print(f"Ground truth CSV missing ({gt_path}); generating via evaluator.")
        gt_df = evaluator._get_ground_truth(query_id)

    norm_df = results_df.copy()
    # pz-dialect plans emit `product_id` (renamed from PZ's internal `id`); the
    # PhysicalPipeline convention is `prod_id`, which normalize_eval_df maps to `id`.
    if "product_id" in norm_df.columns and "prod_id" not in norm_df.columns:
        norm_df = norm_df.rename(columns={"product_id": "prod_id"})
    norm_df = normalize_eval_df(norm_df, use_case, query_id)

    qm = evaluator._evaluate_single_query(query_id, norm_df, gt_df)
    qm_dict = dataclasses.asdict(qm)
    qm_type = type(qm).__name__

    if "Retrieval" in qm_type:
        return "f1_score", float(qm_dict.get("f1_score", math.nan))
    if "Aggregation" in qm_type:
        return "relative_error", 1.0 / (1.0 + float(qm_dict.get("relative_error", 1.0)))
    if "Rank" in qm_type:
        return "spearman_correlation", float(qm_dict.get("spearman_correlation", math.nan))
    if "SingleAccuracy" in qm_type:
        return "accuracy", float(qm_dict.get("accuracy", math.nan))
    return "unknown", math.nan


def run_single_execution(
    *,
    use_case: str,
    scale_factor: int,
    mode: str,
    data_dir: str,
    plan_name: str,
    query_id: int,
    model_name: str,
    run_count: int,
    template_text: str,
) -> None:
    """Render, execute, evaluate, and persist one independent plan run."""
    plan_code = render_plan_code(template_text, model_name, scale_factor)
    env = build_plan_env(plan_name, data_dir)

    print(
        f"Use case: {use_case} | Q{query_id} | plan: {plan_name} | "
        f"sf: {scale_factor} | mode: {mode} | model: {model_name} | run: {run_count}"
    )
    print("Executing plan on full dataset...")
    results_df, latency, cost = execute_plan(plan_code, env, model_name)
    print(f"  rows: {len(results_df)} | latency: {latency:.2f}s | cost: ${cost:.6f}")

    plan_dir = PROJECT_ROOT / "files" / use_case / "raw_results" / "palimpzest" / plan_name / f"sf_{scale_factor}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    model_tag = sanitize_model_name(model_name)
    raw_path = plan_dir / f"Q{query_id}_{model_tag}_{run_count}.csv"
    results_df.to_csv(raw_path, index=False)
    print(f"Raw results -> {raw_path}")

    gt_path = PROJECT_ROOT / "files" / use_case / "raw_results" / "ground_truth" / f"sf_{scale_factor}" / f"Q{query_id}.csv"
    metric_type = "unknown"
    quality = math.nan
    try:
        metric_type, quality = evaluate(use_case, scale_factor, plan_name, query_id, results_df, gt_path)
        print(f"Evaluation: {metric_type} = {quality}")
    except Exception as exc:  # noqa: BLE001
        import traceback

        print(f"Evaluation failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    metrics_path = (
        PROJECT_ROOT
        / "files"
        / use_case
        / "metrics"
        / f"sf_{scale_factor}"
        / f"{plan_name}.json"
    )
    append_metric(
        metrics_path,
        f"{query_id}_{sanitize_model_name(model_name)}_{run_count}",
        {
            "query_id": query_id,
            "scale_factor": scale_factor,
            "latency": round(latency, 4),
            "cost": round(cost, 6),
            "metric_type": metric_type,
            "quality": quality,
            "model": model_name,
        },
    )
    print(f"Metrics -> {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--use-case", required=True, help="Use case name, e.g. ecomm")
    parser.add_argument("--sf", type=int, required=True, help="Scale factor")
    parser.add_argument("--mode", choices=["physical", "pz"], default="physical",
                        help="physical: PhysicalPipeline plan (CSV cost-model dir); "
                             "pz: direct PZ dialect plan (parquet benchmark dir). Default: physical")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent runs per query/model")
    parser.add_argument("--data-dir", help="Override the dir the plan reads source tables from")
    args = parser.parse_args()

    use_case = args.use_case
    scale_factor = args.sf
    data_dir = args.data_dir or str(resolve_data_dir(use_case, scale_factor, args.mode))
    print(f"Reading source tables from: {data_dir}")
    if args.num_runs < 1:
        raise SystemExit("--num-runs must be at least 1")

    query_ids = QUERY_IDS
    models = MODELS
    if not query_ids:
        raise SystemExit("QUERY_IDS must contain at least one query id")
    if not models:
        raise SystemExit("MODELS must contain at least one model")

    for query_id in query_ids:
        txt_path = resolve_plan_template(use_case, query_id)
        parsed_qid, plan_name = parse_plan_filename(txt_path)
        if parsed_qid != query_id:
            raise SystemExit(
                f"Resolved plan file {txt_path.name!r} does not match requested query id Q{query_id}"
            )
        template_text = txt_path.read_text()
        for model_name in models:
            for run_count in range(1, args.num_runs + 1):
                run_single_execution(
                    use_case=use_case,
                    scale_factor=scale_factor,
                    mode=args.mode,
                    data_dir=data_dir,
                    plan_name=plan_name,
                    query_id=query_id,
                    model_name=model_name,
                    run_count=run_count,
                    template_text=template_text,
                )


if __name__ == "__main__":
    main()
