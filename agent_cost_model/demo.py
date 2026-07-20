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
import pandas as pd

from cost_model_agent import CostModelAgent, OpenRouterClient, ResultsStore
from llm_sampler import LLM_Sampler


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
    ap.add_argument("--helper-model", default="openai/gpt-5.4", help="OpenRouter model id for the cost helper agent (customCost mode); defaults to --model")
    ap.add_argument("--max-steps", type=int, default=40, help="max steps for the main agent (cost helper has its own max)")
    ap.add_argument("--runcount", type=int, default=None, help="run index; saves result under key '<query_id>_<runcount>' instead of overwriting '<query_id>'")
    ap.add_argument("--embedding-model", default=None, help="embedding model id for LLM_Sampler; when set, the quality-eval subset is drawn by LLM-guided sampling instead of at random. Which modalities are embedded is read from the `modalities` map below; use a multimodal model for queries whose modality includes 'image'.")
    ap.add_argument("--sample-method", choices=["importance-sampling", "topk"], default="importance-sampling", help="how LLM_Sampler picks rows from cosine-similarity scores: 'importance-sampling' (weighted random draw) or 'topk' (highest-scoring rows)")
    ap.add_argument("--keyword", action="store_true", help="add +1 to a row's similarity if any full word of the LLM target text appears in the row's text data (full-word match)")
    args = ap.parse_args()

    # plans = build_plans()
    # results = seed_results()
    plans = {}
    op_results = ResultsStore([])
    plan_results = ResultsStore([])

    # agent_type = "execute_oracle"
    agent_type = "customCost_oracle_helper"
    USE_CASE = "ecomm"
    llm = ScriptedLLM() if args.offline else OpenRouterClient(args.model, reasoning_effort="medium")
    agent = CostModelAgent(llm, max_steps=args.max_steps, verbose=True, agent_dir=f"{agent_type}_agent", use_case = USE_CASE, helper_model=args.helper_model)


    query_id = 6
    source_data = f"styles_details_image.csv"
    # Per-query modality: drives whether the sampler embeds/compares text, images, or both.
    modalities = {
        **{q: "image and text" for q in [8, 11, 12, 13, 14]},
        **{q: "text" for q in [1, 3, 5, 7]},
        **{q: "image" for q in [2, 4, 6, 9, 10]},
    }
    tasks = {
        1: """Based on the textual description and the title of the product,
            find the `prod_id` of products that are backpacks from Reebok.""",
        2: """Based on the image representation of the product,
            find the `prod_id` of products where the image shows
            a pair of sports shoes that are predominantly yellow and silver.""",
        3: """For each product, extract the brand name from the product description and title.
            Return the product id and the brand in a column titled 'category'.""",
        4: """For each product, use the image to extract the primary color of the depicted product.
            Return `prod_id` and the primary color in a column titled 'category'.""",
        5: """Based solely on the title and description of the product, classify each product into one of the following categories:
            Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
            Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
            Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
            Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
            Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.
            Each product can only have one category.
            Return `prod_id` and the category.""",
        6: """Based solely on the image of the product, classify each product into one of the following categories:
            Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
            Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
            Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
            Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
            Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.
            Each product can only have one category.
            Return `prod_id` and the category.""",
        7: """Find all pairs of products priced at $500 or less where both products
            are of the same category and from the same brand based on their descriptions.
            Return the two `prod_id` columns in this order: first product's `prod_id`, second product's `prod_id`.""",
        8: """Perform a self-join of the dataset.
            For each product with a product description having at least 3000 characters,
            find the matching product images based on the title and the description
            of the product.""",
        9: """Based on product images, find pairs of distinct products under $800
            in a single base color (Black, Blue, Red, White, Orange, or Green)
            that depict objects of the same category and the same dominant surface color.
            Return the two `prod_id` columns in this order: first product's `prod_id`, second product's `prod_id`.""",
        10: """Based on product images, find matching outfits consisting of shoes,
            bottomwear, and topwear in Black, Blue, Red, or White,
            where all three items are from the same brand, same color,
            and each is priced at $1000 or less. Return the three `prod_id` columns in
            this order: shoes `prod_id`, bottomwear `prod_id`, topwear `prod_id`.""",
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
            suitable for outdoor running in warm weather. Return prod_id.""",
        14: """For each fashion product that costs less than 130,
            find the single image that best matches the product's textual description
            (including the product name and full description),
            but only if that image also depicts white socks."""
    }
    task = tasks[query_id]
    # with open(f"files/movie/query/natural_language/Q{query_id}.txt") as f:
    #     task = f.read().strip()

    data_dir = f"agent_cost_model/dataset/{USE_CASE}"

    # LLM-guided importance sampling of the quality-eval subset. Runs BEFORE the
    # agent and simply overwrites Q{id}_subset.csv, which PhysicalPipeline.run_subset
    # reads on its cache-hit path. When --embedding-model is omitted, the agent
    # falls back to its default random subset. Independent of the agent (no shared
    # state); all cost/latency/sample results are logged to sampling/{use_case}/results.json.
    if args.embedding_model:
        source_df = pd.read_csv(os.path.join(data_dir, source_data))
        modality = modalities[query_id]
        # Modality is the single source of truth for which embeddings are used:
        #   - image_dir set  => per-row image embeddings + image cosine similarity
        #   - text_cols=[]   => no text embeddings and text excluded from cosine similarity
        #     (overrides any text columns present in the source CSV)
        image_dir = os.path.join(data_dir, "images") if "image" in modality else None
        text_cols = None if "text" in modality else []
        sampler = LLM_Sampler(
            query_id=query_id,
            use_case=USE_CASE,
            query_text=task,
            df=source_df,
            llm_client=llm,
            embedding_model=args.embedding_model,
            image_dir=image_dir,
            text_cols=text_cols,
            sample_method=args.sample_method,
            keyword=args.keyword,
        )
        sampled_ids = sampler.sample()
        print(f"\n=== LLM_Sampler subset ({len(sampled_ids)} rows) -> {sampler.subset_out_path} ===")
        print(sampled_ids)

    # answer = agent.run(task, plans,
    #                    plan_results = plan_results,
    #                    op_results = op_results,
    #                    mode = agent_type,
    #                    query_info={
    #                         "use_case": USE_CASE,
    #                         "scale_factor": 500,
    #                         "query_id": query_id,
    #                         "data_dir": data_dir,
    #                         "gt_dir": f"files/{USE_CASE}/raw_results/ground_truth",
    #                     "runcount": args.runcount,
    #                     })

    # print("\n=== FINAL ANSWER ===")
    # print(answer)
    # qkey = f"Q{query_id}_{args.runcount}" if args.runcount is not None else f"Q{query_id}"
    # output_path = f"agent_cost_model/final_answer/{USE_CASE}/{agent_type}_agent/{qkey}.json"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # with open(output_path, "w") as f:
    #     json.dump(answer, f, indent=2)
    # print("\n=== observed-results store after run ===")
    # print(op_results.summary())


if __name__ == "__main__":
    main()
