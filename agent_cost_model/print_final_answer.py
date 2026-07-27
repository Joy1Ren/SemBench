import json
import pandas as pd
qid = 5
run = "1"                      # which run's final answer to print (key inside Q{qid}.json)
use_case = "ecomm"
scale_factor = 500
agent_dir = "customCost_oracle_helper_agent"
final_answer_path = f"final_answer/{use_case}/{agent_dir}/sf_{scale_factor}/Q{qid}.json"
results_path = f"metrics/{use_case}/sf_{scale_factor}/Q{qid}_{run}_{agent_dir}_results.csv"

results_df = pd.read_csv(results_path)
quality_metrics = ["f1_score","precision","recall"]
print(results_df[["plan_name", "latency_s","cost_usd","input_tokens","output_tokens"] + quality_metrics])

with open(final_answer_path, "r") as f:
    file = json.load(f)

# Final answers are nested by run number; fall back to the first run present.
answer = file.get(run) or next(iter(file.values()))
for plan_name, plan_code in answer["plan_codes"].items():
    # if plan_name not in ['p5', 'p8']: continue
    print(f"=============={plan_name}==============")
    print(plan_code)