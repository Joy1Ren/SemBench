"""Runnable end-to-end demo for `cost_model_agent.py`.

Run it two ways:

    # Offline: a scripted "LLM" drives the whole loop (no API key, no network).
    # Use this to confirm the harness + sandbox work on your machine.
    python3 demo.py --offline

    # Real: drive with an OpenRouter model.
    export OPENROUTER_API_KEY=sk-or-...
    python3 demo.py --model openai/gpt-5

Requirements: `local_python_executor.py` next to this file. For --offline you
need nothing else. For the real path: `pip install openai`.

The plans here are tiny DUCK-TYPED stand-ins for palimpzest `PhysicalPlan`s, so
the demo runs without a full palimpzest dataset. The agent code never imports
them — it only relies on iterating operators + reading attributes — so swapping
in real `PhysicalPlan`s (from palimpzest's optimizer) requires no agent changes:
just put real plans in the `plans` dict.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
import json

from cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore


# ---------------------------------------------------------------------------
# Duck-typed stand-ins for palimpzest PhysicalPlan / PhysicalOperator.
# Real palimpzest plans are iterable over their operators (post-order over the
# operator tree); these mimic just enough of that surface.
# ---------------------------------------------------------------------------
@dataclass
class DemoOp:
    op_id: str
    op_type: str
    model: str | None = None
    cardinality: int = 100  # estimated number of input records


class DemoPlan:
    def __init__(self, name: str, ops: list[DemoOp]):
        self.name = name
        self.ops = ops

    def __iter__(self):
        return iter(self.ops)

    def __len__(self):
        return len(self.ops)


def build_plans() -> dict[str, DemoPlan]:
    return {
        "plan_a": DemoPlan("plan_a", [
            DemoOp("a.scan", "MarshalAndScanDataOp", cardinality=1000),
            DemoOp("a.filter", "SemFilterOp", model="openai/gpt-5-mini", cardinality=1000),
            DemoOp("a.map", "SemMapOp", model="openai/gpt-5", cardinality=200),
        ]),
        "plan_b": DemoPlan("plan_b", [
            DemoOp("b.scan", "MarshalAndScanDataOp", cardinality=1000),
            DemoOp("b.filter", "SemFilterOp", model="anthropic/claude-haiku-4.5", cardinality=1000),
            DemoOp("b.map", "SemMapOp", model="openai/gpt-5-mini", cardinality=400),
            DemoOp("b.agg", "SemAggOp", model="openai/gpt-5", cardinality=1),
        ]),
    }


def seed_results() -> ResultsStore:
    """A few observed semantic-filter / map invocations to fit a v1 model from."""
    rows: list[dict] = []
    # SemFilterOp observations on gpt-5-mini
    for i in range(6):
        rows.append({
            "op_id": "a.filter", "op_type": "SemFilterOp", "model": "openai/gpt-5-mini",
            "input_id": f"doc_{i}", "output": i % 2 == 0,
            "cost_usd": 0.00091 + 0.00002 * i, "latency_s": 0.82 + 0.05 * i,
            "input_tokens": 360 + 5 * i, "output_tokens": 6,
        })
    # SemMapOp observations on gpt-5
    for i in range(5):
        rows.append({
            "op_id": "a.map", "op_type": "SemMapOp", "model": "openai/gpt-5",
            "input_id": f"doc_{i}", "output": f"extracted_value_{i}",
            "cost_usd": 0.0041 + 0.0003 * i, "latency_s": 1.7 + 0.1 * i,
            "input_tokens": 540 + 10 * i, "output_tokens": 130 + 4 * i,
        })
    return ResultsStore(rows=rows)


# ---------------------------------------------------------------------------
# Offline scripted "LLM": replays a canned, correct sequence of steps so you can
# verify the harness, sandbox, class-authoring, and tools without any API key.
# Each call returns the next canned assistant turn, ignoring the prompt.
# ---------------------------------------------------------------------------
class ScriptedLLM:
    def __init__(self) -> None:
        self._turns = iter(_SCRIPT)

    def generate(self, system: str, messages: list[dict]) -> str:
        return next(self._turns)


_SCRIPT = [
    # 1. orient: inspect a plan
    "Let me inspect plan_a.\n```python\ninspect_plan(plans[\"plan_a\"])\n```",
    # 2. look at the observed data
    "Now the observed stats.\n```python\nprint(results.summary())\n```",
    # 3. author + install a cost model fitted from the observations
    '''Fit per-op-type mean cost/latency and install it.
```python
from collections import defaultdict
import statistics

class FittedCostModel:
    def __init__(self, results):
        cost_by, lat_by = defaultdict(list), defaultdict(list)
        for r in results.rows:
            cost_by[r["op_type"]].append(r["cost_usd"])
            lat_by[r["op_type"]].append(r["latency_s"])
        self.mean_cost = {k: statistics.mean(v) for k, v in cost_by.items()}
        self.mean_lat = {k: statistics.mean(v) for k, v in lat_by.items()}

    def estimate_plan(self, plan):
        cost = 0.0
        time = 0.0
        per_op = {}
        for op in iter_operators(plan):
            t = get_op_type(op)
            card = getattr(op, "cardinality", 1) or 1
            c = self.mean_cost.get(t, 0.0) * card
            l = self.mean_lat.get(t, 0.0) * card
            cost += c
            time += l
            per_op[get_op_id(op)] = {"cost": round(c, 6), "time": round(l, 4)}
        return PlanCostEstimate(cost=cost, time=time, quality=1.0, details=per_op)

update_cost_model(FittedCostModel, notes="v1: per-op-type mean cost/latency * cardinality")
```''',
    # 4. apply it
    "Estimate both plans.\n```python\nprint(estimate_plan_cost(plans[\"plan_a\"]))\nprint(estimate_plan_cost(plans[\"plan_b\"]))\n```",
    # 5. gather more observations, then re-estimate to compare
    "Gather more data on plan_b, then re-estimate.\n```python\nexecute_subplan(plans[\"plan_b\"], n=8, seed=1)\n```",
    # 6. final answer
    '''```json
{
  "cost_model_version": 1,
  "approach": "Per-op-type mean cost and latency fitted from observed invocations, scaled by each operator's estimated input cardinality and summed over the plan.",
  "estimates": {
    "plan_a": {"cost": 0.0, "time": 0.0, "quality": 1.0},
    "plan_b": {"cost": 0.0, "time": 0.0, "quality": 1.0}
  },
  "notes": "Demo run. cost/time numbers above are placeholders; see the printed estimate_plan_cost observations for actual fitted values."
}
```''',
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use the scripted LLM (no API key)")
    ap.add_argument("--model", default="openai/gpt-5.4", help="OpenRouter model id")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--runcount", type=int, default=None, help="run index; saves result under key '<query_id>_<runcount>' instead of overwriting '<query_id>'")
    args = ap.parse_args()

    # plans = build_plans()
    # results = seed_results()
    plans = {}
    op_results = ResultsStore([])
    plan_results = ResultsStore([])

    agent_type = "execute_oracle"
    USE_CASE = "ecomm"
    llm = ScriptedLLM() if args.offline else OpenRouterClient(args.model, reasoning_effort="medium")
    agent = CostModelAgent(llm, max_steps=args.max_steps, verbose=True, agent_dir=f"{agent_type}_agent", use_case = USE_CASE)


    query_id = 12
    tasks = {
        4: """For each product, use the image to extract the primary color of the depicted product.
            Return `prod_id` and the primary color in a column titled 'category'.""",
        11: """Based on product images and descriptions, find matching all-black outfits
            consisting of shoes, bottomwear (excluding swimwear), topwear (excluding swimwear),
            and an accessory (watch, jewellery, or bag priced at $500 or less),
            where all four items are from the same brand. Return the four prod_id columns
            in this order: shoes prod_id, bottomwear prod_id, topwear prod_id, accessory prod_id""",
        12: """For each Adidas or Puma product, use the product image and description
            to generate the following columns in this order: prod_id, brand name (lowercase),
            and master category classified as 'accessories', 'apparel', or 'footwear'.""",
        13: """Based on product images and descriptions, find men's running shirts
            with round neck and short sleeves, in blue or black (not bright colors
            like white, and definitely not green), with a striped design,
            suitable for outdoor running in warm weather. Return prod_id."""
    }
    task = tasks[query_id]
    # with open(f"files/movie/query/natural_language/Q{query_id}.txt") as f:
    #     task = f.read().strip()
    answer = agent.run(task, plans,
                       plan_results = plan_results,
                       op_results = op_results,
                       mode = agent_type,
                       query_info={
                            "use_case": USE_CASE,
                            "scale_factor": 500,
                            "query_id": query_id,
                            "data_dir": f"agent_cost_model/dataset/{USE_CASE}",
                            "gt_dir": f"files/{USE_CASE}/raw_results/ground_truth",
                        "runcount": args.runcount,
                        })

    print("\n=== FINAL ANSWER ===")
    print(answer)
    qkey = f"Q{query_id}_{args.runcount}" if args.runcount is not None else f"Q{query_id}"
    output_path = f"agent_cost_model/final_answer/{USE_CASE}/{agent_type}_agent/{qkey}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(answer, f, indent=2)
    print("\n=== observed-results store after run ===")
    print(op_results.summary())


if __name__ == "__main__":
    main()
