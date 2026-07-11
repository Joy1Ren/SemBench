import json
import pandas as pd
qid = 5
agent_dir = "sampleCost_agent_retry"
final_answer_path = f"final_answer/{agent_dir}/Q{qid}.json"
results_path = f"metrics/Q{qid}_{agent_dir}_results.csv"

results_df = pd.read_csv(results_path)
quality_metrics = ["f1_score","precision","recall"]
print(results_df[["plan_name", "latency_s","cost_usd","input_tokens","output_tokens"] + quality_metrics])

with open(final_answer_path, "r") as f:
    file = json.load(f)

for plan_name, plan_code in file["plan_codes"].items():
    # if plan_name not in ['p5', 'p8']: continue
    print(f"=============={plan_name}==============")
    print(plan_code)