from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass, field

import pandas as pd
from openrouter import OpenRouter
import yaml
from jinja2 import Template

from agent.local_python_executor import LocalPythonExecutor
from runner.generic_palimpzest_runner.generic_palimpzest_runner import GenericPalimpzestRunner
from scenario.movie.evaluation.evaluate import MovieEvaluator

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


@dataclass
class PlanResult:
    code: str | None
    messages: list[dict] = field(default_factory=list)


def format_trace(result: PlanResult) -> str:
    """Format a PlanResult into a human-readable trace of thoughts, actions, and observations."""
    lines = []
    step = 0
    # messages[0] is the system prompt (omitted); messages[1] is the initial user query
    for msg in result.messages[1:]:
        role, content = msg["role"], msg["content"]
        if role == "assistant":
            step += 1
            lines.append(f"{'=' * 60}")
            lines.append(f"Step {step}")
            lines.append(f"{'=' * 60}")
            lines.append(content)
            lines.append("")
        elif role == "user":
            lines.append(content)
            lines.append("")
    lines.append(f"{'=' * 60}")
    lines.append("Final Plan")
    lines.append(f"{'=' * 60}")
    lines.append(result.code if result.code is not None else "(no plan returned)")
    return "\n".join(lines)

_DEFAULT_PZ_OPERATORS = {
    "sem_filter": "ds.sem_filter(filter: str) — Semantically filter rows where `filter` is true.",
    "sem_map": "ds.sem_map(cols: list[dict]) — Add new LLM-derived columns. Each dict has 'name', 'type', 'description'.",
    "sem_join": "ds.sem_join(other, condition: str) — Semantically join two datasets where `condition` holds. Produces schema names 'name' and 'name_right' for sem_map columns from left and right datasets.",
    "filter": "ds.filter(fn: Callable) — Exact (non-semantic) row filter using a lambda.",
    "project": "ds.project(columns: list[str]) — Select a subset of columns.",
    "limit": "ds.limit(n: int) — Keep at most n rows.",
    "groupby": "ds.groupby(GroupBySig(group_by_fields, agg_funcs, agg_fields)) — Group and aggregate. Produces schema name 'agg_func(agg_field)', e.g. 'count(reviewId)' or 'average(score)'.",
}

_PHYSICAL_SEMANTIC_OPERATORS = {
    "sem_filter": "pipeline.sem_filter(condition: str, model: pz.Model) — LLM filter; keeps rows where condition is true.",
    "sem_map": "pipeline.sem_map(cols: list[dict], model: pz.Model) — Add LLM-derived columns. Each dict has 'name', 'type', 'description'.",
    "sem_join": "pipeline.sem_join(other: PhysicalPipeline, condition: str, model: pz.Model, join_parallelism: int = 20) — LLM join; keeps pairs where condition holds.",
}

_PHYSICAL_NONSEMANTIC_OPERATORS = {
    "filter": "pipeline.filter(fn: Callable[[dict], bool]) — Exact row filter using a Python callable.",
    "project": "pipeline.project(cols: list[str]) — Select a subset of columns.",
    "limit": "pipeline.limit(n: int) — Keep at most n rows.",
    "groupby": "pipeline.groupby(group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]) — Group and aggregate. agg_funcs are 'count' or 'average'.",
}

_AVAILABLE_MODELS = {
    # --- Small / ultra-fast open-weight models ---
    "pz.Model.DEEPSEEK_R1_DISTILL_QWEN_1_5B":
        "Very cheap reasoning-focused small model. Good for lightweight filters, binary classification, simple extraction, and high-throughput pipelines.",

    "pz.Model.VLLM_QWEN_1_5_0_5B_CHAT":
        "Extremely fast and cheap local/VLLM model. Best for trivial transformations, routing, tagging, heuristics, and latency-sensitive operators.",

    "pz.Model.LLAMA3_2_3B":
        "Small open-weight instruct model. Good for inexpensive summarization, filtering, and basic extraction tasks with moderate quality requirements.",

    # --- Medium open-weight models ---
    "pz.Model.LLAMA3_1_8B":
        "Strong small-to-medium open model with good cost/performance tradeoff. Suitable for semantic filters, joins, lightweight reasoning, and structured extraction.",

    "pz.Model.GEMINI_2_0_FLASH":
        "Fast multimodal model optimized for low latency. Good for general semantic querying, extraction, and simple reasoning workloads.",

    "pz.Model.GEMINI_2_5_FLASH":
        "Fast, cheap, strong general-purpose model. Excellent default choice for semantic maps, filters, joins, and summarization pipelines.",

    "pz.Model.GOOGLE_GEMINI_2_5_FLASH":
        "Fast, inexpensive Gemini endpoint with strong quality/latency balance. Good for large-scale semantic query execution.",

    "pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE":
        "Ultra-cheap Gemini variant optimized for throughput and latency. Best for simple filtering, tagging, deduplication, and routing operators.",

    "pz.Model.CLAUDE_3_5_HAIKU":
        "Fast Anthropic model. Good for concise summarization, extraction, lightweight reasoning, and latency-sensitive workflows.",

    "pz.Model.GPT_4o_MINI":
        "Low-cost OpenAI multimodal model with strong speed/quality tradeoff. Good for extraction, semantic filtering, and conversational transformations.",

    "pz.Model.GPT_4_1_MINI":
        "Fast and capable OpenAI reasoning model. Strong for structured extraction, semantic joins, code-aware transformations, and medium-complexity reasoning.",

    "pz.Model.GPT_4_1_NANO":
        "Ultra-fast, ultra-cheap OpenAI model. Best for simple filters, classification, routing, and lightweight semantic operators.",

    "pz.Model.GPT_5_NANO":
        "Very fast GPT-5 variant optimized for inexpensive high-throughput semantic operations and lightweight reasoning.",

    "pz.Model.GPT_5_MINI":
        "Balanced GPT-5 model with strong reasoning, speed, and cost efficiency. Excellent default for semantic query plans and multi-step operators.",

    # --- Large / high-quality models ---
    "pz.Model.DEEPSEEK_V3":
        "High-quality open reasoning model with excellent coding and analytical performance. Good for complex extraction, synthesis, and reasoning-heavy operators.",

    "pz.Model.LLAMA3_3_70B":
        "Large open-weight instruct model with strong reasoning and generation quality. Suitable for difficult semantic joins, nuanced extraction, and summarization.",

    "pz.Model.LLAMA_4_MAVERICK":
        "Large multimodal Llama model optimized for long-context reasoning and complex semantic tasks.",

    "pz.Model.CLAUDE_3_5_SONNET":
        "High-quality Anthropic model with strong reasoning and writing quality. Good for nuanced summarization, extraction, and synthesis tasks.",

    "pz.Model.CLAUDE_3_7_SONNET":
        "Advanced Anthropic reasoning model. Excellent for difficult semantic joins, long-context analysis, planning, and complex transformations.",

    "pz.Model.GPT_4o":
        "Strong multimodal OpenAI model balancing reasoning, speed, and cost. Good for general-purpose semantic querying and multimodal pipelines.",

    "pz.Model.GPT_4_1":
        "High-quality OpenAI reasoning model. Strong for complex extraction, multi-step reasoning, semantic joins, and code-aware operations.",

    "pz.Model.GPT_5":
        "Highest-quality GPT-5 model for difficult reasoning, planning, synthesis, and complex semantic query execution where quality matters most.",

    "pz.Model.o4_MINI":
        "Reasoning-focused OpenAI model optimized for analytical and multi-step tasks. Good for difficult semantic filtering, planning, and operator orchestration.",

    "pz.Model.GEMINI_2_5_PRO":
        "High-end Gemini reasoning model. Strong for long-context understanding, complex extraction, and analytical semantic queries.",

    "pz.Model.GOOGLE_GEMINI_2_5_PRO":
        "Advanced Gemini model with strong reasoning and multimodal capabilities. Best for difficult semantic operators and long-context workflows.",

    # --- Vision / multimodal specialized models ---
    "pz.Model.LLAMA3_2_90B_V":
        "Large multimodal vision-language model. Good for image-document extraction, OCR-style reasoning, and multimodal semantic querying.",

    "pz.Model.GPT_4o_AUDIO_PREVIEW":
        "Experimental audio-capable GPT-4o variant for speech/audio understanding workflows and multimodal pipelines.",

    "pz.Model.GPT_4o_MINI_AUDIO_PREVIEW":
        "Low-cost audio-enabled GPT-4o variant for lightweight speech and audio processing tasks.",

    # --- Embedding / representation models ---
    "pz.Model.TEXT_EMBEDDING_3_SMALL":
        "Efficient embedding model for retrieval, semantic similarity, clustering, vector search, and approximate semantic joins.",

    "pz.Model.CLIP_VIT_B_32":
        "Vision-language embedding model for image-text similarity, multimodal retrieval, and semantic image search.",
}

_PROMPTS_FILE = pathlib.Path(__file__).parent / "prompts.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)
PLANNER_AGENT_SYSTEM_PROMPT: str = _PROMPTS["PLANNER_INSTRUCTION"]
PHYSICAL_PLANNER_SYSTEM_PROMPT: str = _PROMPTS["PHYSICAL_PLANNER_INSTRUCTION"]



def _extract_code(text: str) -> str | None:
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


class PlannerAgent:
    def __init__(
        self,
        model_id: str,
        dataset_dir: str,
        pz_operators: dict[str, str] | None = None,
        max_steps: int = 5,
        use_physical: bool = False,
        query_info: dict = {
            "use_case": "movie",
            "query_id": 5,
            "scale-factor": 2000,
        }
    ):
        self.model_id = model_id
        self.dataset_dir = pathlib.Path(dataset_dir)
        self.max_steps = max_steps
        self.query_info = query_info
        self.client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        
        import litellm
        # use openrouter for api calling
        _original_litellm_completion = litellm.completion
        def _openrouter_completion(model, **kwargs):
            if model.startswith("vertex_ai/"):
                model = "google/" + model[len("vertex_ai/"):]
            if not model.startswith("openrouter/"):
                model = "openrouter/" + model
            return _original_litellm_completion(model=model, **kwargs)
        litellm.completion = _openrouter_completion

        if use_physical:
            self.system_prompt = Template(PHYSICAL_PLANNER_SYSTEM_PROMPT).render(
                code_opening_tag="```python",
                code_closing_tag="```",
                physical_semantic_operators=_PHYSICAL_SEMANTIC_OPERATORS,
                physical_nonsemantic_operators=_PHYSICAL_NONSEMANTIC_OPERATORS,
                available_models=_AVAILABLE_MODELS,
            )
        else:
            self.system_prompt = Template(PLANNER_AGENT_SYSTEM_PROMPT).render(
                code_opening_tag="```python",
                code_closing_tag="```",
                operators=pz_operators or _DEFAULT_PZ_OPERATORS,
            )

        self._static_tools = self._build_tools()

    def _build_tools(self) -> dict:
        dataset_dir = self.dataset_dir

        def list_files() -> str:
            files = sorted(f.name for f in dataset_dir.iterdir() if f.suffix == ".csv")
            return "Available files: " + ", ".join(files)

        def explore_schema(filename: str) -> str:
            df = pd.read_csv(dataset_dir / filename, nrows=0)
            lines = [f"  {col}: {dtype}" for col, dtype in df.dtypes.items()]
            return f"{filename} schema:\n" + "\n".join(lines)

        def explore_sample(filename: str, n: int = 5) -> str:
            df = pd.read_csv(dataset_dir / filename, nrows=n)
            return f"{filename} sample ({n} rows):\n{df.to_string(index=False)}"

        def typename(obj) -> str:
            """Get the type name of an object without dunder access (which the executor blocks)."""
            t = type(obj)
            type_str = str(t)
            if "'" in type_str:
                return type_str.split("'")[1].split(".")[-1]
            return type_str

        def final_answer(plan_code: str) -> str:
            return plan_code

        return {
            "list_files": list_files,
            "explore_schema": explore_schema,
            "explore_sample": explore_sample,
            "typename": typename,
            "final_answer": final_answer,
        }

    def _make_executor(self) -> LocalPythonExecutor:
        # Fresh executor per plan() call — prevents state leakage across runs
        executor = LocalPythonExecutor(
            additional_authorized_imports=["pandas", "json", "re", "csv", "textwrap", "palimpzest"],
        )
        executor.send_tools(self._static_tools)
        # Inject dataset_dir as a plain string so the LLM can use it with pd.read_csv directly
        executor.send_variables({"dataset_dir": str(self.dataset_dir)})
        return executor

    def plan(self, query: str, dataset_description: str) -> PlanResult:
        """Run the agent loop and return a PlanResult with the plan code and full message trace."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Query: {query}\n\nDataset(s):\n{dataset_description}"},
        ]
        executor = self._make_executor()

        for step in range(1, self.max_steps + 1):
            response = self._call_llm(messages)
            messages.append({"role": "assistant", "content": response})

            code = _extract_code(response)
            if code is None:
                messages.append({
                    "role": "user",
                    "content": "Error: No code block found. Wrap your code in ```python ... ```.",
                })
                continue

            try:
                exec_result = executor(code)
            except Exception as e:
                messages.append({"role": "user", "content": f"Observation:\nError: {e}"})
                continue

            if exec_result.is_final_answer:
                return PlanResult(code=str(exec_result.output), messages=messages)

            parts = []
            if exec_result.logs:
                parts.append(exec_result.logs.strip())
            if exec_result.output is not None:
                parts.append(f"Output: {exec_result.output}")
            observation = "\n".join(parts) or "(no output)"

            if step == self.max_steps - 1:
                observation += "\nNext step is your last. You must call final_answer(plan_code) in your next response."

            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        return PlanResult(code=None, messages=messages)
    
    def execute_query_plan(self, plan_code: str):
        "run generated query plan with sembench run call"
        from agent.physical_pipeline import PhysicalPipeline
        import palimpzest as pz

        # plan_code is a self-contained script: def agent_execute(): ... / agent_execute()
        # PhysicalPipeline, pz, and pd are injected since the plan uses them without importing.
        exec_executor = LocalPythonExecutor(
            additional_authorized_imports=["pandas", "palimpzest"],
        )
        exec_executor.send_tools({})  # no tools needed for execution, just the variables below
        exec_executor.send_variables({
            "PhysicalPipeline": PhysicalPipeline,
            "pz": pz,
            "pd": pd,
        })

        runner = GenericPalimpzestRunner(
            use_case=self.query_info["use_case"],
            scale_factor=self.query_info["scale-factor"],
            model_name="mixed",
            concurrent_llm_worker=20,
            skip_setup=True,
            agent_dir_name="physical_planner_agent_test",
        )

        metric = runner.execute_query(
            self.query_info["query_id"],
            query_fn=lambda: exec_executor(plan_code).output,
        )
        runner.metrics[self.query_info["query_id"]] = metric
        runner.save_results(self.query_info["query_id"], metric.results)
        runner.save_metrics()

        evaluator = MovieEvaluator(
            use_case=self.query_info["use_case"],
            scale_factor=self.query_info["scale-factor"],
            agent_dir_name="physical_planner_agent_test",
        )
        evaluator.evaluate_system("palimpzest", [self.query_info["query_id"]])

        

    def _call_llm(self, messages: list[dict]) -> str:
        response = self.client.chat.send(model=self.model_id, messages=messages)
        return response.choices[0].message.content
