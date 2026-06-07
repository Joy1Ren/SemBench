import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


from agent.planner_agent import PlannerAgent, format_trace


MOVIE_DESCRIPTION = (
    "The `Movies` table contains information about movies including their titles, scores, "
    "ratings, release dates, runtime, and other metadata.\n"
    "The `Reviews` table contains movie reviews with associated metadata including the review text."
)


def main():
    agent_dir_name = "physical_planner_agent_test"
    agent = PlannerAgent(model_id="openai/gpt-5.4",
                         dataset_dir="agent/dataset/movie",
                         use_physical=True,
                         query_info = {
                                "use_case": "movie",
                                "query_id": 2,
                                "scale-factor": 2000,
                            })

    # code_dir = f"src/scenario/movie/runner/palimpzest_runner/{agent_dir_name}"
    # trace_dir = f"agent/traces/movie/{agent_dir_name}"
    # os.makedirs(code_dir, exist_ok=True)
    # os.makedirs(trace_dir, exist_ok=True)

    # for qid in [8,]:
    #     with open(f"files/movie/query/natural_language/Q{qid}.txt") as f:
    #         query = f.read().strip()

    #     result = agent.plan(query, MOVIE_DESCRIPTION)

    #     with open(f"{code_dir}/Q{qid}.py", "w") as f:
    #         f.write(result.code or "")

    #     with open(f"{trace_dir}/Q{qid}.txt", "w") as f:
    #         f.write(format_trace(result))
    qid = 2
    plan_code = pathlib.Path(f"src/scenario/movie/runner/palimpzest_runner/physical_planner_agent/Q{qid}.py").read_text()
    agent.execute_query_plan(plan_code)


main()
