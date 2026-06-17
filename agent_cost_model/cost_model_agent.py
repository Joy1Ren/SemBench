"""A minimal, self-contained tool-loop agent for *cost-model* reasoning over
physical query plans.

This is a stripped-down cousin of the OfficeQA `MultiTurnAgent` / `SearchAgent`.
It keeps the parts that matter for the cost-model research project and drops
everything that doesn't:

  KEPT
    * a bounded multi-turn loop (<= max_steps)
    * one fenced block per step: a ```python``` tool call OR a ```json``` final answer
    * python execution in a sandboxed `LocalPythonExecutor`
    * a flat message trajectory (list of {"role", "content"} dicts)

  DROPPED (relative to SearchAgent)
    * the `TextBlock` / `ChunkBlock` trajectory abstraction (no retrieval ⇒ no
      pruning / redaction), so messages are plain strings
    * vector search / grep / read-document tools
    * the skunk `ExecutionContext`, `PromptedCall`, `LLMClient`, prompt overrides

  ADDED (the point of this agent)
    * `inspect_plan`      -- introspect a physical (sub)plan's operators
    * `estimate_plan_cost`-- apply the *current* cost model to a (sub)plan
    * `update_cost_model` -- install a cost-model class the agent just authored
    * `execute_subplan`   -- (stub) partially execute a (sub)plan, appending
                             observed per-operator stats to the results store

The agent's job, each run, is to look at a plan + the observed-execution data,
*write a `CostModel` class in python*, install it, apply it, and (optionally)
gather more observations by partially executing subplans and then refine the
model — closing the estimate → observe → update loop.

Dependencies: only `local_python_executor.py` (you already have it) and an LLM
client. `palimpzest` is imported lazily/guarded — the agent operates on real
`PhysicalPlan` / `PhysicalOperator` objects when available, but the harness only
relies on a tiny duck-typed surface (iterate operators; read a few attributes)
so you can also drive it with the stand-ins in `demo.py`.
"""

from __future__ import annotations

import inspect
import json
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Guarded palimpzest import.
#
# We import the real types when palimpzest is installed (for nicer reprs and so
# the cost model the agent writes can `isinstance`-check if it wants), but the
# harness never *requires* them: every code path degrades to duck typing. See
# `get_op_type` / `get_op_id` / `iter_operators` below.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    from palimpzest.query.optimizer.plan import PhysicalPlan  # type: ignore
    from palimpzest.query.operators.physical import PhysicalOperator  # type: ignore
    from palimpzest.core.models import OperatorCostEstimates, PlanCost  # type: ignore

    HAVE_PALIMPZEST = True
except Exception:  # ImportError, or a partial install
    PhysicalPlan = PhysicalOperator = OperatorCostEstimates = PlanCost = None  # type: ignore
    HAVE_PALIMPZEST = False


# ---------------------------------------------------------------------------
# Guarded SemBench evaluator import.
# Add src/ to sys.path relative to this file so the import works regardless
# of which directory the script is run from.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_src_dir = str(_Path(__file__).resolve().parents[1] / "src")
if _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)

try:
    from scenario.movie.evaluation.evaluate import MovieEvaluator as _MovieEvaluator
    _HAVE_EVALUATOR = True
    print("[cost_model_agent] MovieEvaluator imported OK")
except Exception as _e:
    _MovieEvaluator = None  # type: ignore
    _HAVE_EVALUATOR = False
    print(f"[cost_model_agent] MovieEvaluator import failed: {type(_e).__name__}: {_e}")


# ===========================================================================
# Errors (local, tiny — we don't pull in skunk.errors)
# ===========================================================================
class ParseError(Exception):
    """The model's reply was not a single valid fenced block."""

    def __init__(self, raw: str, detail: str):
        super().__init__(detail)
        self.raw = raw
        self.detail = detail


class StepFailed(Exception):
    """The agent ran out of steps without an accepted final answer."""

    def __init__(self, reason: str, diagnostic: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic


# ===========================================================================
# LLM client seam
#
# The agent only needs a single synchronous call: given a system prompt and a
# list of {role, content} messages, return the assistant's text. Implement this
# Protocol however you like; `OpenRouterClient` below is the default.
# ===========================================================================
class LLMClient(Protocol):
    def generate(self, system: str, messages: list[dict]) -> tuple[str, str | None]: ...


class OpenRouterClient:
    """OpenRouter-backed `LLMClient`, via the OpenAI-compatible endpoint.

    Uses the widely-installed `openai` SDK pointed at OpenRouter (the same
    provider the SearchAgent supports). `pip install openai`, then set
    `OPENROUTER_API_KEY`. `model` is a full OpenRouter id, e.g.
    "openai/gpt-5", "anthropic/claude-sonnet-4.6", "google/gemini-2.5-flash".

    (If you prefer the native `openrouter` SDK used in skunk's llm_client.py,
    swap `generate` for a `client.chat.send(...)` call — same message shape.)
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,  # "minimal" | "low" | "medium" | "high"
    ) -> None:
        import os

        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    def generate(self, system: str, messages: list[dict]) -> tuple[str, str | None]:
        msgs = [{"role": "system", "content": system}, *messages]
        extra_body: dict = {}
        if self.reasoning_effort:
            extra_body["reasoning"] = {"effort": self.reasoning_effort}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            extra_body=extra_body or None,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning")
        return content, reasoning


# ===========================================================================
# Cost-model data types + the contract the agent's CostModel must satisfy
# ===========================================================================
@dataclass
class PlanCostEstimate:
    """The estimate a cost model produces for a whole (sub)plan.

    `cost` is dollars, `time` is wall-clock seconds (latency), `quality` is an
    optional [0, 1] score. `details` is free-form (e.g. per-operator breakdown)
    so the agent can show its work."""

    cost: float
    time: float
    quality: float | None = None
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        q = "None" if self.quality is None else f"{self.quality:.3f}"
        return f"PlanCostEstimate(cost=${self.cost:.4f}, time={self.time:.3f}s, quality={q})"


class CostModel:
    """Optional base class for the cost model the *agent* writes.

    The only hard requirement enforced by `update_cost_model` is a callable
    `estimate_plan(self, plan) -> PlanCostEstimate`. Subclassing this is not
    required (a plain class with that method works), but it documents the
    contract and gives a default __repr__.
    """

    def estimate_plan(self, plan: Any) -> PlanCostEstimate:  # pragma: no cover
        raise NotImplementedError(
            "Write estimate_plan(self, plan) -> PlanCostEstimate in your subclass."
        )


# ---------------------------------------------------------------------------
# Duck-typed plan/operator helpers — work for real palimpzest objects AND for
# the stand-ins in demo.py. These are injected into the sandbox so the agent's
# CostModel can introspect operators uniformly.
# ---------------------------------------------------------------------------
def iter_operators(plan: Any) -> list[Any]:
    """All operators in a plan. Real `PhysicalPipeline` is iterable (topological order over
    the operator tree). Falls back to a `.operators` / `.ops` attribute, else treats `plan`
    as already-a-list."""
    try:
        return list(plan)
    except TypeError:
        # for attr in ("operators", "ops"):
        #     if hasattr(plan, attr):
        #         return list(getattr(plan, attr))
        raise TypeError(f"don't know how to iterate operators of {plan!r}")


def get_op_type(op: Any) -> str:
    """Operator type label, robust across palimpzest ops and demo ops."""
    return getattr(op, "op_type", None) or type(op).__name__


def get_op_id(op: Any) -> str:
    """Return PhysicalPipeline's `params_id`"""
    try:
        return op.params_id
    except Exception as e:
        raise Exception(f"failed to get op_id from {op!r}") from e

def get_op_model(op: Any) -> str | None:
    """The LLM model an operator uses, if any (None for non-LLM ops)."""
    m = getattr(op, "model", None)
    if m is None:
        return None
    # palimpzest models are often enums with a `.value`; fall back to str.
    return getattr(m, "value", None) or str(m)


def describe_operator(op: Any) -> dict:
    """A compact, JSON-friendly view of one operator: id, type, model, and a
    handful of likely-relevant numeric/string attributes (best effort)."""
    return {
        "op_id": get_op_id(op),
        "op_type": get_op_type(op),
        "attrs": op.attributes if hasattr(op, "attributes") else {},
    }


# ===========================================================================
# Observed-execution results store
# ===========================================================================
@dataclass
class ResultsStore:
    """Append-only log of observed per-operator-invocation stats.

    One row per (operator, single input) execution. The canonical columns:

        op_id          str    -- which operator instance produced this row
        op_type        str    -- e.g. "SemFilterOp", "SemMapOp"
        model          str   -- LLM model used (None for non-LLM ops)
        input_id       str    -- id of / pointer to the input record
        output         any    -- the operator's decision/output for this input
                                  (e.g. True/False for a filter)
        cost_usd       float  -- dollar cost of this invocation
        latency_s      float  -- wall-clock latency of this invocation
        input_tokens   int
        output_tokens  int

    Extra columns are fine; the cost model can use whatever it finds. Access the
    data as `results.rows` (list[dict]) or `results.df` (pandas, if installed).
    """

    rows: list[dict] = field(default_factory=list)

    def append(self, new_rows: list[dict]) -> int:
        self.rows.extend(new_rows)
        return len(new_rows)

    @property
    def df(self):
        """pandas view (requires pandas). Use `.rows` if you don't have pandas."""
        import pandas as pd

        return pd.DataFrame(self.rows)

    def summary(self) -> dict:
        """Counts + mean cost/latency per op_type — a quick orientation aid."""
        by_type: dict[str, dict[str, float]] = {}
        for r in self.rows:
            t = r.get("op_type", "?")
            agg = by_type.setdefault(t, {"n": 0, "cost_usd": 0.0, "latency_s": 0.0})
            agg["n"] += 1
            agg["cost_usd"] += float(r.get("cost_usd", 0.0) or 0.0)
            agg["latency_s"] += float(r.get("latency_s", 0.0) or 0.0)
        for t, agg in by_type.items():
            n = max(agg["n"], 1)
            agg["mean_cost_usd"] = round(agg["cost_usd"] / n, 6)
            agg["mean_latency_s"] = round(agg["latency_s"] / n, 4)
        return {"total_rows": len(self.rows), "by_op_type": by_type}


# ===========================================================================
# Cost-model registry (what `estimate_plan_cost` applies; what
# `update_cost_model` swaps in). Versioned so the agent can see its history.
# ===========================================================================
@dataclass
class _Version:
    version: int
    model: Any
    notes: str


class CostModelRegistry:
    def __init__(self) -> None:
        self._history: list[_Version] = []

    @property
    def version(self) -> int:
        return len(self._history)

    def current(self) -> Any:
        if not self._history:
            raise RuntimeError(
                "No cost model installed yet. Author a class with "
                "estimate_plan(self, plan) -> PlanCostEstimate and call "
                "update_cost_model(YourClass)."
            )
        return self._history[-1].model

    def install(self, model: Any, notes: str = "") -> int:
        self._history.append(_Version(self.version + 1, model, notes))
        return self.version

    def describe(self) -> str:
        if not self._history:
            return "(no cost model installed)"
        lines = []
        for v in self._history:
            lines.append(f"  v{v.version}: {type(v.model).__name__} — {v.notes or '(no notes)'}")
        return "\n".join(lines)


# ===========================================================================
# Tools
# ===========================================================================
class Tool(ABC):
    name: str
    doc: str

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


# class InspectPlanTool(Tool):
#     name = "inspect_plan"
#     doc = """\
# ### inspect_plan(plan)
# Return a structured view of a physical (sub)plan: the list of its operators in
# execution order, each with `op_id`, `op_type`, `attributes`.
# Use this first to learn the shape of a plan before modeling it.

# ```python
# inspect_plan(plans["p1"])
# ```"""

#     def __call__(self, plan: Any) -> dict:
#         ops = [describe_operator(op) for op in iter_operators(plan)]
#         return {"n_operators": len(ops), "operators": ops}


class EstimatePlanCostTool(Tool):
    name = "estimate_plan_cost"
    doc = """\
### estimate_plan_cost(plan)
Apply the CURRENTLY INSTALLED cost model to a physical (sub)plan and return its
`PlanCostEstimate` (dollar cost, latency seconds, optional quality). Errors if
you have not installed a cost model yet (do that with `update_cost_model`).

```python
estimate_plan_cost(plans["p1"])
```"""

    def __init__(self, registry: CostModelRegistry):
        self._registry = registry

    def __call__(self, plan: Any) -> PlanCostEstimate:
        model = self._registry.current()
        est = model.estimate_plan(plan)
        if not isinstance(est, PlanCostEstimate):
            # Be lenient: accept a dict the agent returned and coerce it.
            if isinstance(est, dict):
                est = PlanCostEstimate(**est)
            else:
                raise TypeError(
                    "estimate_plan must return a PlanCostEstimate (or a dict with "
                    f"cost/time/quality), got {type(est).__name__}"
                )
        return est


class UpdateCostModelTool(Tool):
    name = "update_cost_model"
    doc = """\
### update_cost_model(cost_model, notes="")
Install a cost model. Pass EITHER a class you just defined (it will be
instantiated for you) OR an instance you constructed. The model must define
`estimate_plan(self, plan) -> PlanCostEstimate`. If your class's __init__ takes
a `op_results` or `plan_results` argument, the observed-results store is passed to it automatically,
so you can fit coefficients from the data. use `iter_operators(plan)` to access
the operators in the plan in topological order.
Returns the new version number.

```python
class MyCostModel:
    def __init__(self, *, op_results, plan_results):
        # fit per-op coefficients from op_results.df and plan_results.df
        self.op_rows = op_results.rows
        self.plan_rows = plan_results.rows
    def estimate_plan(self, plan: PhysicalPipeline):
        cost = time = 0.0
        for op in iter_operators(plan):
            # ... look up coefficients by get_op_type(op) / get_op_model(op) / describe_operator(op)...
            cost += 0.0
            time += 0.0
        return PlanCostEstimate(cost=cost, time=time, quality=None)

update_cost_model(MyCostModel, notes="v1: per-op-type mean cost from results")
```"""

    def __init__(self, registry: CostModelRegistry, *, op_results: ResultsStore, plan_results: ResultsStore):
        self._registry = registry
        self._op_results = op_results
        self._plan_results = plan_results

    def __call__(self, cost_model: Any, notes: str = "") -> dict:
        # A class → instantiate it. Inspect __init__ to decide whether to pass
        # the results store (so the agent can fit from observations).
        if isinstance(cost_model, type):
            model = self._instantiate(cost_model)
        else:
            model = cost_model
        if not callable(getattr(model, "estimate_plan", None)):
            raise TypeError(
                "cost model must define estimate_plan(self, plan) -> PlanCostEstimate"
            )
        version = self._registry.install(model, notes=notes)
        return {"installed_version": version, "type": type(model).__name__, "notes": notes}

    def _instantiate(self, cls: type) -> Any:
        try:
            params = inspect.signature(cls).parameters
            kwargs = {}
            if "op_results" in params: kwargs["op_results"] = self._op_results
            if "plan_results" in params: kwargs["plan_results"] = self._plan_results
            return cls(**kwargs)
        except (ValueError, TypeError):
            pass  # builtins / sandbox funcs may not introspect cleanly
        return cls()


# class ExecuteSubplanTool(Tool):
#     name = "execute_subplan"
#     doc = """\
# ### execute_subplan(plan, n=5, seed=0)
# STUB partial execution: simulate running each operator in a (sub)plan on `n`
# sample inputs, appending one observed-stats row per (operator, input) to the
# results store. Use this to gather MORE observations, then refit your cost model
# and compare its estimate to the freshly observed totals — this closes the
# estimate → observe → update loop. Deterministic given `seed`.

# NOTE: this does not call any real LLM or palimpzest execution; it produces
# plausible synthetic stats so you can develop the update logic. Swap in real
# execution later. Returns a summary of what was appended.

# ```python
# execute_subplan(plans["p1"], n=10, seed=1)
# ```"""

#     # Rough per-op-type generators for synthetic stats. Keyed by a substring of
#     # the op type so it matches "SemFilterOp", "SemFilter", etc.
#     _PROFILES = {
#         "Filter": dict(cost=0.0012, lat=0.9, in_tok=380, out_tok=8, boolean=True),
#         "Map":    dict(cost=0.0035, lat=1.6, in_tok=520, out_tok=140, boolean=False),
#         "Join":   dict(cost=0.0061, lat=2.4, in_tok=900, out_tok=60, boolean=False),
#         "Agg":    dict(cost=0.0040, lat=1.4, in_tok=700, out_tok=120, boolean=False),
#         "Scan":   dict(cost=0.0,    lat=0.05, in_tok=0, out_tok=0, boolean=False),
#         "Retrieve": dict(cost=0.0002, lat=0.3, in_tok=0, out_tok=0, boolean=False),
#     }
#     _DEFAULT = dict(cost=0.0020, lat=1.0, in_tok=400, out_tok=40, boolean=False)

#     def __init__(self, results: ResultsStore):
#         self._results = results

#     def _profile(self, op_type: str) -> dict:
#         for key, prof in self._PROFILES.items():
#             if key.lower() in op_type.lower():
#                 return prof
#         return self._DEFAULT

#     def __call__(self, plan: Any, n: int = 5, seed: int = 0) -> dict:
#         import random

#         rng = random.Random(seed)
#         new_rows: list[dict] = []
#         for op in iter_operators(plan):
#             op_type = get_op_type(op)
#             op_id = get_op_id(op)
#             model = get_op_model(op)
#             prof = self._profile(op_type)
#             for i in range(n):
#                 jitter = lambda x: x * (1.0 + rng.uniform(-0.25, 0.25))  # noqa: E731
#                 in_tok = int(jitter(prof["in_tok"]))
#                 out_tok = int(jitter(prof["out_tok"]))
#                 output: Any = rng.random() < 0.5 if prof["boolean"] else f"out_{i}"
#                 new_rows.append({
#                     "op_id": op_id,
#                     "op_type": op_type,
#                     "model": model,
#                     "input_id": f"{op_id}:in_{i}",
#                     "output": output,
#                     "cost_usd": round(jitter(prof["cost"]), 8),
#                     "latency_s": round(jitter(prof["lat"]), 4),
#                     "input_tokens": in_tok,
#                     "output_tokens": out_tok,
#                 })
#         appended = self._results.append(new_rows)
#         return {
#             "appended_rows": appended,
#             "total_rows": len(self._results.rows),
#             "summary": self._results.summary(),
#         }


class ListFilesTool(Tool):
    name = "list_files"
    doc = """\
### list_files()
List all CSV files available in the data directory.

```python
list_files()
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self) -> str:
        files = sorted(f.name for f in self._data_dir.iterdir() if f.suffix == ".csv")
        return "Available files: " + ", ".join(files)


class ExploreSchemaT(Tool):
    name = "explore_schema"
    doc = """\
### explore_schema(filename)
Show the column names and dtypes of a CSV file in the data directory.

```python
explore_schema("Reviews.csv")
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self, filename: str) -> str:
        import pandas as pd
        df = pd.read_csv(self._data_dir / filename, nrows=0)
        lines = [f"  {col}: {dtype}" for col, dtype in df.dtypes.items()]
        return f"{filename} schema:\n" + "\n".join(lines)


class ExploreSampleTool(Tool):
    name = "explore_sample"
    doc = """\
### explore_sample(filename, n=5)
Show the first `n` rows of a CSV file in the data directory.

```python
explore_sample("Reviews.csv", n=3)
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self, filename: str, n: int = 5) -> str:
        import pandas as pd
        df = pd.read_csv(self._data_dir / filename, nrows=n)
        return f"{filename} sample ({n} rows):\n{df.to_string(index=False)}"


class WritePlanTool(Tool):
    name = "write_plan"
    doc = """\
### write_plan(code, name)
Build and store a physical query plan WITHOUT executing it. `code` is a Python
string that constructs a PhysicalPipeline and returns it as its last expression —
do NOT call `.run()` in the plan code; `execute_plan` handles execution.
`name` is the plan identifier you choose (e.g. "p1").

After this call, `plans[name]` holds the built pipeline so you can immediately call
`estimate_plan_cost(plans[name])` for a pre-execution cost estimate.
Use the load_data(filename) function to read CSVs from the data directory in your plan code.

```python
write_plan(\"\"\"
pipeline = PhysicalPipeline(plan_name, "Emails.csv", load_data("Emails.csv"))
pipeline.sem_filter("this email quotes someone outside of the the sender's company", model=pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE)
pipeline.project(["emailId"])
pipeline.limit(5)
pipeline
\"\"\", "p1")
# `plan_name` is automatically set to the name you pass (here "p1")
# plans["p1"] now holds the built pipeline — call estimate_plan_cost(plans["p1"])
```"""

    def __init__(self, plan_codes: dict, plans: dict, data_dir: str) -> None:
        import pathlib
        self._plan_codes = plan_codes
        self._plans = plans
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self, code: str, plan_name: str) -> dict:
        import pandas as pd

        try:
            from agent.physical_pipeline import PhysicalPipeline
        except ImportError:
            from physical_pipeline import PhysicalPipeline  # type: ignore
        try:
            import palimpzest as pz
        except ImportError as exc:
            raise ImportError("palimpzest is required to build plans") from exc

        from local_python_executor import LocalPythonExecutor

        data_dir = self._data_dir

        def load_data(filename: str) -> pd.DataFrame:
            return pd.read_csv(data_dir / filename)

        build_executor = LocalPythonExecutor(
            additional_authorized_imports=["pandas", "palimpzest"],
        )
        build_executor.send_tools({})
        build_executor.send_variables({
            "PhysicalPipeline": PhysicalPipeline,
            "pz": pz,
            "pd": pd,
            "load_data": load_data,
            "plan_name": plan_name,
        })

        exec_result = build_executor(code)
        pipeline = exec_result.output
        if not isinstance(pipeline, PhysicalPipeline):
            raise TypeError(
                f"Plan code must return a PhysicalPipeline as its last expression "
                f"(got {type(pipeline).__name__}). Do not call .run() in the plan code."
            )

        self._plan_codes[plan_name] = code
        self._plans[plan_name] = pipeline
        return {"plan_name": plan_name, "total_plans": len(self._plan_codes)} #TODO: total plan count is kind of useless


class ExecutePlanTool(Tool):
    name = "execute_plan"
    doc = """\
### execute_plan(name)
Execute the stored plan `name` in a sandboxed environment. Plan-level and operator-level
quality, latency, cost, and token usage are printed and appended to `plan_results` and `op_results`, respectively.
After execution, `plans[name]` holds the PhysicalPipeline.

Returns `plan_results` and `op_results` respectively
The stats are: cost_usd, latency_s, input_tokens, output_tokens, quality, and any other quality metrics the evaluator produces
```python
p1_plan_results, p1_op_results = execute_plan("p1")
```"""

    def __init__(
        self,
        plan_codes: dict,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        use_case: str,
        scale_factor: int,
        query_id: int,
        data_dir: str,
        agent_dir: str,
        gt_dir: str,
        raw_results_dir: str,
    ) -> None:
        import pathlib

        self._plan_codes = plan_codes
        self._plans = plans
        self._op_results = op_results
        self._plan_results = plan_results
        self._use_case = use_case
        self._scale_factor = scale_factor
        self._query_id = query_id
        self._data_dir = pathlib.Path(data_dir)
        self._agent_dir = agent_dir
        self._gt_dir = pathlib.Path(gt_dir)
        self._raw_results_dir = pathlib.Path(raw_results_dir) / agent_dir
        # Build evaluator once (loads domain CSVs for the use_case)
        self._evaluator = None
        if _HAVE_EVALUATOR:
            try:
                self._evaluator = _MovieEvaluator(use_case, scale_factor, agent_dir)
            except Exception as e:
                print(f"[ExecutePlanTool] evaluator init failed: {type(e).__name__}: {e}")

    def __call__(self, plan_name: str) -> dict:
        import dataclasses

        import pandas as pd

        pipeline = self._plans.get(plan_name)
        if pipeline is None:
            raise KeyError(
                f"No plan named {plan_name!r}. Call write_plan first. "
                f"Available: {list(self._plans)}"
            )

        result_collection, op_results, plan_results = pipeline.run() #TO DO: put this in local python executor
        self._plans[plan_name] = pipeline

        # -- save raw query output CSV (mirrors GenericRunner.save_results) ---
        results_df = result_collection.to_df()
        self._raw_results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._raw_results_dir / f"Q{self._query_id}_{plan_name}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"[execute_plan] raw results → {csv_path}")

        # get plan quality
        quality_metric = "unknown"
        if self._evaluator is None:
            print(f"[execute_plan] evaluator not available — skipping evaluation for {plan_name}")
        else:
            try:
                gt_path = self._gt_dir / f"Q{self._query_id}.csv"
                gt_df = pd.read_csv(gt_path)
                qm = self._evaluator._evaluate_single_query(
                    self._query_id, results_df, gt_df
                )
                # Flatten quality fields directly into the row (no nested dict)
                # so results.df has clean flat columns.
                for k, v in dataclasses.asdict(qm).items():
                    plan_results[k] = v
                qm_type = type(qm).__name__
                if "Retrieval" in qm_type:
                    quality_metric = "f1_score"
                elif "Aggregation" in qm_type:
                    quality_metric = "relative_error"
                elif "Rank" in qm_type:
                    quality_metric = "spearman_correlation"
                print(f"[execute_plan] evaluation succeeded for {plan_name}: quality_metric={quality_metric}")
            except Exception as e:
                print(f"[execute_plan] evaluation failed for {plan_name}: {type(e).__name__}: {e}")
                plan_results["eval_error"] = str(e)
    
        if quality_metric in plan_results:
            if quality_metric == "relative_error": plan_results["quality"] = 1- plan_results[quality_metric]
            else: plan_results['quality'] = plan_results[quality_metric]
        plan_results['plan_name'] = plan_name
        self._plan_results.append([plan_results])
        self._op_results.append(op_results)
        print(plan_results, op_results)
        return self._plan_results, self._op_results


# ===========================================================================
# The agent
# ===========================================================================
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)


@dataclass
class _Step:
    code: str | None = None
    result: Any = None
    raw: str = ""


def _parse_step(text: str) -> _Step:
    """First fenced block → python tool call or json final answer."""
    m = _FENCE_RE.search(text)
    if m is None:
        raise ParseError(
            raw=text,
            detail="no fenced block — emit ONE ```python``` block (a tool call) or "
            "ONE ```json``` block (your final answer).",
        )
    lang, body = m.group(1).lower(), m.group(2).strip()
    if lang != "json":
        return _Step(code=body, raw=text)
    try:
        return _Step(result=json.loads(body), raw=text)
    except json.JSONDecodeError as e:
        raise ParseError(raw=text, detail=f"final-answer JSON was malformed — {e}") from e


# _DEFAULT_PZ_OPERATORS = {
#     "sem_filter": "ds.sem_filter(filter: str) — Semantically filter rows where `filter` is true.",
#     "sem_map": "ds.sem_map(cols: list[dict]) — Add new LLM-derived columns. Each dict has 'name', 'type', 'description'.",
#     "sem_join": "ds.sem_join(other, condition: str) — Semantically join two datasets where `condition` holds. Produces schema names 'name' and 'name_right' for sem_map columns from left and right datasets.",
#     "filter": "ds.filter(fn: Callable) — Exact (non-semantic) row filter using a lambda.",
#     "project": "ds.project(columns: list[str]) — Select a subset of columns.",
#     "limit": "ds.limit(n: int) — Keep at most n rows.",
#     "groupby": "ds.groupby(GroupBySig(group_by_fields, agg_funcs, agg_fields)) — Group and aggregate. Produces schema name 'agg_func(agg_field)', e.g. 'count(reviewId)' or 'average(score)'.",
# }

_PHYSICAL_SEMANTIC_OPERATORS = {
    "sem_filter": "pipeline.sem_filter(condition: str, model: pz.Mode) — LLM filter; keeps rows where condition is true.",
    "sem_map": "pipeline.sem_map(cols: list[dict], model: pz.Model) — Add LLM-derived columns. col is list of dict {'name': str, 'type': type, 'description': str}",
    "sem_join": "pipeline.sem_join(other: PhysicalPipeline, condition: str, model: pz.Model) — LLM join; keeps pairs where condition holds.",
}

_PHYSICAL_NONSEMANTIC_OPERATORS = {
    "filter": "pipeline.filter(fn: Callable[[dict], bool]) — Exact row filter using a Python callable.",
    "project": "pipeline.project(cols: list[str]) — Select a subset of columns.",
    "limit": "pipeline.limit(n: int) — Keep at most n rows.",
    "groupby": "pipeline.groupby(group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]) — Group and aggregate. Produces schema name 'agg_func(agg_field)', e.g. 'count(reviewId)' or 'average(score)'.",
}
_AVAILABLE_MODELS = [f"pz.Model.{m.name}" for m in __import__("palimpzest").Model]

# - `PlanCostEstimate`: dataclass(cost, time, quality=None, details={{}}) — what your cost model returns
# - `CostModel`       : optional base class for your cost model
# - `iter_operators`, `get_op_type`, `get_op_id`, `get_op_model`, `describe_operator` : plan/op helpers
_SYSTEM_TEMPLATE = """\
{briefing}

## HARD RULE — No data snooping or hardcoded indexes/phrases
`explore_sample` and `explore_schema` exist to help you understand **schema and format only**.
You MUST NOT use sample rows to identify specific records and then hardcode their IDs, row
indexes, or literal field values into a plan.  Every filter predicate in your plan must be a
**general condition** that could correctly classify records it has never seen — for example
`sem_filter("the text is clearly positive")` or `filter(lambda row: row["name"] == "John")`.
Writing plans like `filter(lambda row: row["id"] in [3, 17, 42])` or
`filter(lambda row: "good" in row["text"])` to cherry-pick rows you already know about is **cheating**
and will produce meaningless results.
Furthermore, physical plans should only involve trees of the given semantic and non-semantic operators.
Do not construct your own methods, rely on these operators only.
If your plan contains any hardcoded record IDs or values you copied from `explore_sample`
output, rewrite it before calling `execute_plan`.

## Tools (already imported into your python sandbox)

{tools_doc}

## Plan Writing (using PhysicalPipeline API)
You have access to the following operators to build physical plans.
Make sure to output the desired fields in the correct order, as specified in the task.
PhysicalPipeline already has a execution speedup involving `limit` operators:
once the limit number of records is passed, all previous operator execution calls are cancelled.
Thus, it is unnecessary to include intermediate `limit` operators to reduce cost/latency. The `limit`
operator should only be used at the end of a plan.

Semantic operators — require a model= argument:
All semantic operators have an optional "depends_on: list[str] | None" argument,
which is a list of field names to pass in for the LLM call (instead of the full input schema).
{physical_sem_ops}

Non-semantic operators — no model argument:
{physical_nonsem_ops}

## Available Models:
{available_models}

## Also available in your sandbox (no import needed)
- `plans`           : dict[str, plan] — physical plans; populated by write_plan (updated by execute_plan)
- `plan_codes`      : dict[str, str] — code strings stored by write_plan (keys = plan names)
- `plan_results`    : the observed-execution store (`plan_results.rows`, `plan_results.df`)
- `op_results`      : the observed-execution store (`op_results.rows`, `op_results.df`, `op_results.summary()`)
- stdlib: `math`, `statistics`, `random`, `collections`, `itertools`, `json`

You have <= {max_steps} steps. On each step output EXACTLY ONE fenced block:
  - a ```python``` block — runs in the sandbox; its stdout/return value comes back as your next observation; or
  - a ```json``` block — your final answer (parsed as data, not executed). Emit this once, when done.
DO NOT emit multiple fenced blocks in each step output. Keep each python block small and focused (one logical action).
Requirements for the final answer:
{final_answer_doc}"""


class CostModelAgent:
    """A bounded tool-loop agent that authors, applies, and refines cost models."""
    execute_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds.

        Your goal is to write physical plans, observe cost/latency/quality,
        and converge on the best cost-quality trade-off. Do not hard code the plan

        Suggested workflow:
        1. Explore the data: call `list_files()` to see available CSVs,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1".
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
           - `plans[name]` is populated immediately with the newly written PhysicalPipeline instance
        3. Execute with `execute_plan(name)`:
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality vs. ground truth).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate execution performance with `plan_results.df` and `op_results.df`. `quality` column depends on query type:
           retrieval → f1_score
           aggregation → relative_error
           ranking → spearman_correlation
        6. Based on `plan_results.df` and `op_results.df`, write a new plan that has higher quality or lower cost/latency
           and repeat from step 2.

        Note: performance results may have variance — gather several runs before concluding.
        Do not conclude the best plan too early -- use the allocated steps to better understand the
        cost, latency, and quality tradeoff on a plan and per-operator level. Consider various model choices for each operator,
        using cheaper/faster models for easier operators, and improving plan design.""")

    sampleCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds. A sample-based cost model (SampleBasedCostModel) is
        pre-installed and re-fits automatically on every `estimate_plan_cost` call —
        use it to compare plan variants before deciding which ones to execute.

        Your goal is to write physical plans, observe cost/latency/quality,
        and converge on the best cost-quality trade-off. Do not hard code the plan

        Suggested workflow:
        To get initial data and performance traces:
        1. Explore the data: call `list_files()` to see available CSVs,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1".
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
           - `plans[name]` is populated immediately — call `estimate_plan_cost(plans[name])`
             right after `write_plan` to get a pre-execution cost estimate.
        3. Execute with `execute_plan(name)`:
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality vs. ground truth).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate execution performance with `plan_results.df` and `op_results.df`. `quality` column depends on query type:
           retrieval → f1_score
           aggregation → relative_error
           ranking → spearman_correlation

        Then, iteratively write new plans, estimate their cost, and execute the most promising ones:
        - Use `plan_results.df` and `op_results.df` to identify which quality and cost/latency trade-off for different operators and attribute choices
            -- `plan_results.df` includes operator descriptions and input-output record pairs
            -- `op_results.df` includes per-operator performances, operators are named by `name_op1`, `name_op2`, ...
        - Always use `estimate_plan_cost(plans[name])` to get cost/latency estimates BEFORE executing the promising plans.
            -- the cost model averages past execution to get per-operator cost/latency estimates
            -- thus, only changing a few operators for each plan writing will produce more comparable estimations

        Note: performance results may have variance — gather several runs before concluding.
        Do not conclude the best plan too early -- use the allocated steps to better understand the
        cost, latency, and quality tradeoff on a plan and per-operator level. Consider various model choices for each operator,
        using cheaper/faster models for easier operators, and improving plan design.
        """)

    customCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds. 

        Your goal is to design a cost model (cost and latency) to help determine better physical plans,
        observe cost/latency/quality, and converge on the best cost/latency-quality trade-off.
        Do not hard code the plan.

        Suggested workflow:
        To get initial data and performance traces:
        1. Explore the data: call `list_files()` to see available CSVs,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1".
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
           - `plans[name]` is populated immediately — call `estimate_plan_cost(plans[name])`
             right after `write_plan` to get a pre-execution cost estimate.
        3. Execute with `execute_plan(name)`:
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality vs. ground truth).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate execution performance with `plan_results.df` and `op_results.df`. `quality` column depends on query type:
           retrieval → f1_score
           aggregation → relative_error
           ranking → spearman_correlation

        Then, design a cost model to help iteratively write new plans, estimate their cost, and execute the most promising ones:
        - Use `plan_results.df` and `op_results.df` to identify which quality and cost/latency trade-off for different operators and attribute choices
            -- `plan_results.df` includes operator descriptions and input-output record pairs
            -- `op_results.df` includes per-operator performances, operators are named by `name_op1`, `name_op2`, ...
        - Always use `estimate_plan_cost(plans[name])` to get cost/latency estimates BEFORE executing the promising plans.
            -- the cost model averages past execution to get per-operator cost/latency estimates
            -- thus, only changing a few operators for each plan writing will produce more comparable estimations
        - When building your cost model, consider taking sample averages, grouping by operator types,
            and utilizing information across plans and operators.
        - Install your cost model by calling `update_cost_model(YourClass, notes="v1: ...")`.
            If your class __init__ takes `op_results` or `plan_results`, they are passed automatically
            so you can fit coefficients from the accumulated data inside __init__.
            You may call `update_cost_model` multiple times to refine the model across versions. When results data is sparse, consider using
            prior estimation of operator/plan cost and latency.

        Note: performance results may have variance — gather several runs before concluding.
        Do not conclude the best plan too early -- use the allocated steps to better understand the
        cost, latency, and quality tradeoff on a plan and per-operator level. Consider various model choices for each operator,
        using cheaper/faster models for easier operators, and improving plan design.
    """)
 
    execute_final_answer_doc = textwrap.dedent("""\
        Emit a JSON object summarizing the best plan:
          {
            "best_plan": {
              "name": "...",
              "rationale": "why this plan wins (cost/quality trade-off)"
            }
          }
        Use plain JSON literals only (no python expressions, no trailing commas).""")
    
    sampleCost_final_answer_doc = textwrap.dedent("""\
        Emit a JSON object with the best plan and a cost-model explanation:
          {
            "best_plan": {
              "name": "p1",
              "rationale": "why this plan wins (cost/quality trade-off)"
            },
            "cost_model_explanation": "1-4 sentences on how the cost model guided your plan design. Which operators or models were cheapest/fastest?"
          }
        Use plain JSON literals only (no python expressions, no trailing commas).""")
    
    customCost_final_answer_doc = sampleCost_final_answer_doc

    def __init__(
        self,
        llm: LLMClient,
        *,
        data_dir: str = "dataset/movie",
        agent_dir: str = "no_name",
        max_steps: int = 12,
        max_recover_retries: int = 1,
        context_budget_chars: int = 200_000,
        authorized_imports: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        self.llm = llm
        self.agent_dir = agent_dir
        self.data_dir = data_dir
        self.max_steps = max_steps
        self.max_recover_retries = max_recover_retries
        self.context_budget_chars = context_budget_chars
        self.authorized_imports = authorized_imports or [
            "math", "statistics", "json", "collections", "itertools",
            "pathlib", "pandas", "palimpzest"
        ]
        self.verbose = verbose
        # Rebuilt per run(); kept on the instance so callers can read it after.
        self.messages: list[dict] = []
        self.reasoning_steps: list[str | None] = []
        self.trajectory_steps: list[dict] = []

    # -- logging -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _save_trajectory_df(self, query_info: dict) -> None:
        """Dump per-step reasoning and assistant responses to CSV."""
        if not self.trajectory_steps:
            return
        import pandas as pd
        import pathlib
        data_path = pathlib.Path(query_info["data_dir"])
        metrics_dir = data_path.parents[1] / "trajectory"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / f"Q{query_info['query_id']}_{self.agent_dir}_trajectory.csv"
        pd.DataFrame(self.trajectory_steps).to_csv(out, index=False)
        self._log(f"[run] trajectory → {out}")

    def _save_cost_model_codes(self, codes: dict, query_info: dict) -> None:
        """Save all versioned cost model code strings to agent_cost_model/costModel/Q{id}.json."""
        if not codes:
            return
        import json
        import pathlib
        out_dir = pathlib.Path("agent_cost_model/costModel")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"Q{query_info['query_id']}.json"
        with open(out, "w") as f:
            json.dump(codes, f, indent=2)
        self._log(f"[run] cost model codes → {out}")

    def _save_results_df(
        self, results: ResultsStore, query_info: dict, final_answer: dict | None = None
    ) -> None:
        """Dump the accumulated plan-level results table to CSV."""
        if not results.rows:
            return
        import pathlib
        data_path = pathlib.Path(query_info["data_dir"])
        metrics_dir = data_path.parents[1] / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out = metrics_dir / f"Q{query_info['query_id']}_{self.agent_dir}_results.csv"
        df = results.df
        df["use_case"] = query_info["use_case"]
        df["query_id"] = query_info["query_id"]
        best_name = (final_answer or {}).get("best_plan", {}).get("name")
        df["final_selected"] = df["plan_name"] == best_name if best_name else False
        df = df[[col for col in df.columns if col not in ["plan_str", "op_samples"]]+["plan_str", "op_samples"]] #display long text at end
        df.to_csv(out, index=False)
        self._log(f"[run] results table → {out}")

    # -- prompt assembly ---------------------------------------------------
    def _system_prompt(
        self,
        tools: list[Tool],
        briefing: str | None = None,
        final_answer_doc: str | None = None,
    ) -> str:
        return _SYSTEM_TEMPLATE.format(
            briefing=briefing if briefing is not None else self.briefing,
            tools_doc="\n\n".join(t.doc for t in tools),
            physical_sem_ops="\n".join(f"- {d}" for d in _PHYSICAL_SEMANTIC_OPERATORS.values()),
            physical_nonsem_ops="\n".join(f"- {d}" for d in _PHYSICAL_NONSEMANTIC_OPERATORS.values()),
            available_models="\n".join(f"- {m}" for m in _AVAILABLE_MODELS),
            max_steps=self.max_steps,
            final_answer_doc=final_answer_doc if final_answer_doc is not None else self.final_answer_doc,
        )

    def _opening_message(self, task: str, plans: dict, results: ResultsStore) -> str:
        plan_lines = []
        for name, plan in plans.items():
            ops = iter_operators(plan)
            op_str = "("+ ",".join(get_op_type(o) for o in ops) +")"
            plan_lines.append(f"  - {name}: [{op_str}]")
        if plan_lines:
            plans_section = "Available plans (`plans` dict):\n" + "\n".join(plan_lines)
        else:
            plans_section = "No plans yet — use write_plan to create the first one."
        return (
            f"{task}\n\n"
            f"{plans_section}\n\n"
            f"Observed-results store summary:\n"
            f"{json.dumps(results.summary(), indent=2)}\n\n"
            f"Begin."
        )

    def _trim(self, messages: list[dict]) -> list[dict]:
        budget = self.context_budget_chars
        if sum(len(m["content"]) for m in messages) <= budget:
            return messages
        head = [messages[0], {"role": "user", "content": "...(earlier steps truncated)..."}]
        remaining = budget - sum(len(m["content"]) for m in head)
        tail: list[dict] = []
        for m in reversed(messages[1:]):
            if remaining - len(m["content"]) < 0:
                break
            tail.append(m)
            remaining -= len(m["content"])
        return head + tail[::-1]

    # -- main loop ---------------------------------------------------------
    def run(
        self,
        task: str,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        *,
        mode: str,
        query_info: dict = {
            "use_case": "movie",
            "scale_factor": 2000,
            "query_id": 1,
            "data_dir": "agent_cost_model/dataset/movie",
            "gt_dir": "files/movie/raw_results/ground_truth",
            "raw_results_dir": "files/movie/raw_results/palimpzest",
        },
    ) -> Any:
        """Run the loop over `plans` and `plan_results`, `op_results`, returning the JSON final answer.

        `plans` is a dict {name: physical_plan}; `plan_results` and `op_results` are ResultsStore objects.
        A fresh CostModelRegistry is created per run.
        """
        import litellm as _litellm
        _litellm.suppress_debug_info = True
        _orig_completion = _litellm.completion
        _TOGETHER_TO_OPENROUTER: dict[str, str] = {
            "meta-llama/Llama-3.2-3B-Instruct-Turbo":         "meta-llama/llama-3.2-3b-instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":    "meta-llama/llama-3.1-8b-instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo":        "meta-llama/llama-3.3-70b-instruct",
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": "meta-llama/llama-3.2-90b-vision-instruct",
            "deepseek-ai/DeepSeek-V3":                         "deepseek/deepseek-chat",
            # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": removed — no active OpenRouter endpoints
        }
        def _openrouter_completion(model, **kwargs):
            if model.startswith("vertex_ai/"):
                model = "google/" + model[len("vertex_ai/"):]
            elif model.startswith("together_ai/"):
                together_name = model[len("together_ai/"):]
                model = _TOGETHER_TO_OPENROUTER.get(together_name, together_name)
            if not model.startswith("openrouter/"):
                model = "openrouter/" + model
            return _orig_completion(model=model, **kwargs)
        _litellm.completion = _openrouter_completion

        from local_python_executor import LocalPythonExecutor

        registry = CostModelRegistry()
        plan_codes: dict = {}  # populated by WritePlanTool; shared with ExecutePlanTool

        data_dir = query_info["data_dir"]
        
        tools = [
            ListFilesTool(data_dir),
            ExploreSchemaT(data_dir),
            ExploreSampleTool(data_dir),
            WritePlanTool(plan_codes, plans=plans, data_dir=data_dir),
            ExecutePlanTool(
                    plan_codes=plan_codes,
                    plans=plans,
                    plan_results=plan_results,
                    op_results=op_results,
                    use_case=query_info["use_case"],
                    scale_factor=query_info["scale_factor"],
                    query_id=query_info["query_id"],
                    data_dir=data_dir,
                    agent_dir=self.agent_dir,
                    gt_dir=query_info["gt_dir"],
                    raw_results_dir=query_info["raw_results_dir"],
                ),
            # InspectPlanTool()
        ]

        assert query_info is not None
        extra_sandbox_vars: dict = {}
        if mode == "customCost":
            tools += [
                EstimatePlanCostTool(registry),
                UpdateCostModelTool(registry, plan_results= plan_results, op_results = op_results),
            ]
            briefing = self.customCost_briefing
            final_answer_doc = self.customCost_final_answer_doc
        elif mode == "execute":
            briefing = self.execute_briefing
            final_answer_doc = self.execute_final_answer_doc
        elif mode == "sampleCost":
            try:
                from sample_based_cost_model import SampleBasedCostModel
            except ImportError:
                from agent_cost_model.sample_based_cost_model import SampleBasedCostModel  # type: ignore
            registry.install(SampleBasedCostModel(op_results), notes="v0: SampleBasedCostModel pre-installed")
            tools += [EstimatePlanCostTool(registry)]
            extra_sandbox_vars["SampleBasedCostModel"] = SampleBasedCostModel
            briefing = self.sampleCost_briefing
            final_answer_doc = self.sampleCost_final_answer_doc

        executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        executor.send_tools({t.name: t for t in tools})

        import pandas as pd
        def load_data(filename: str) -> pd.DataFrame:
            return pd.read_csv(data_dir / filename) #TODO: temporary implementation to allow agent to directly run load_data
        executor.send_variables({
            "load_data": load_data,
            "plans": plans,
            "plan_codes": plan_codes,
            "plan_results": plan_results,
            "op_results": op_results,   
            "PlanCostEstimate": PlanCostEstimate,
            "CostModel": CostModel,
            "iter_operators": iter_operators,
            "get_op_type": get_op_type,
            "get_op_id": get_op_id,
            "get_op_model": get_op_model,
            "describe_operator": describe_operator,
            "data_dir": str(query_info["data_dir"]),
            **extra_sandbox_vars,
        })

        system = self._system_prompt(tools, briefing, final_answer_doc)
        opening = self._opening_message(task, plans, op_results)
        self.messages = [{"role": "user", "content": opening}]
        self.reasoning_steps = []
        self.trajectory_steps = []
        self._log(f"\n=== system prompt ({len(system)} chars) ===\n{system}\n")
        self._log(f"=== task ===\n{opening}\n")

        cost_model_codes: dict[str, str] = {}
        step = 0
        while step < self.max_steps:
            step += 1
            try:
                raw = self._llm_step(system)
            except Exception as e:  # LLM transport failure
                self._log(f"[step {step}] LLM error: {e}")
                raise
            self.messages.append({"role": "assistant", "content": raw})
            reasoning = self.reasoning_steps[-1]
            self.trajectory_steps.append({"step": step, "reasoning": reasoning, "assistant": raw})
            if reasoning:
                self._log(f"\n--- reasoning (step {step}) ---\n{reasoning}\n")
            self._log(f"\n--- assistant (step {step}) ---\n{raw}\n")

            # parse (with a couple of format-error retries)
            try:
                parsed = self._parse_with_retries(system, raw)
            except ParseError as e:
                obs = f"Observation (step {step}): {e.detail}"
                self.messages.append({"role": "user", "content": obs})
                self._log(f"[parse error] {obs}")
                continue

            if parsed.code is None:  # final answer
                self._log(f"[final answer] {parsed.result}")
                if isinstance(parsed.result, dict):
                    parsed.result["plan_codes"] = plan_codes
                self._save_results_df(plan_results, query_info, final_answer=parsed.result)
                self._save_trajectory_df(query_info)
                if mode == "customCost":
                    self._save_cost_model_codes(cost_model_codes, query_info)
                return parsed.result

            # execute the python tool-call block
            prev_registry_version = registry.version
            try:
                out = executor(parsed.code)
            except Exception as e:
                obs = f"Observation (step {step}): exec failed — {type(e).__name__}: {e}"
                self.messages.append({"role": "user", "content": obs})
                self._log(obs)
                continue

            if mode == "customCost" and registry.version > prev_registry_version:
                cost_model_codes[f"v{registry.version}"] = parsed.code

            obs = self._format_observation(step, out)
            self.messages.append({"role": "user", "content": obs})
            self._log(obs)

        # out of steps — one forced terminal turn
        self._save_results_df(plan_results, query_info)
        self._save_trajectory_df(query_info)
        if mode == "customCost":
            self._save_cost_model_codes(cost_model_codes, query_info)
        result = self._terminal_turn(system)
        if isinstance(result, dict):
            result["plan_codes"] = plan_codes
        return result

    # -- helpers -----------------------------------------------------------
    def _llm_step(self, system: str, extra: list[dict] | None = None) -> str:
        msgs = self._trim(self.messages)
        if extra:
            msgs = msgs + extra
        content, reasoning = self.llm.generate(system, msgs)
        self.reasoning_steps.append(reasoning)
        return content

    def _parse_with_retries(self, system: str, raw: str) -> _Step:
        attempt = 0
        text = raw
        while True:
            try:
                return _parse_step(text)
            except ParseError as e:
                if attempt >= self.max_recover_retries:
                    raise
                attempt += 1
                fix = (
                    f"Your previous reply could not be parsed: {e.detail}\n"
                    "Re-send exactly ONE ```python``` or ```json``` fenced block."
                )
                # transient repair exchange (kept in history so the model sees it)
                self.messages.append({"role": "user", "content": fix})
                text = self._llm_step(system)
                self.messages.append({"role": "assistant", "content": text})

    @staticmethod
    def _format_observation(step: int, out: Any) -> str:
        parts = [f"Observation (step {step}):"]
        logs = (getattr(out, "logs", "") or "").strip()
        if logs:
            parts.append(f"[stdout]\n{logs}")
        result = out.output if hasattr(out, "output") else out
        result_s = "" if result is None else str(result).strip()
        if result_s and result_s != logs and result_s not in logs:
            parts.append(f"[result]\n{result_s}")
        if len(parts) == 1:
            parts.append("[no output]")
        return "\n\n".join(parts)

    _TERMINAL_PROMPT = (
        "You are out of steps. Do NOT call any tool — emit exactly ONE ```json``` block: "
        "either your best final answer in the required format, or "
        '{"error": "<2-4 sentence note on what you tried and what blocked you>"}.'
    )

    def _terminal_turn(self, system: str) -> Any:
        try:
            text = self._llm_step(system, extra=[{"role": "user", "content": self._TERMINAL_PROMPT}])
            parsed = _parse_step(text)
            if parsed.code is None:
                return parsed.result
        except Exception as e:
            raise StepFailed("max steps without accepted final answer", diagnostic=str(e)) from e
        raise StepFailed("max steps without accepted final answer", diagnostic=text)
