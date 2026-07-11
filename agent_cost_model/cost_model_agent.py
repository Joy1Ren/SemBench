from __future__ import annotations

import inspect
import json
import os
import pathlib
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
    from agent_cost_model.quality_evaluator import QualityEvaluator as _QualityEvaluator
except ImportError:
    try:
        from quality_evaluator import QualityEvaluator as _QualityEvaluator  # type: ignore
    except ImportError:
        _QualityEvaluator = None  # type: ignore


# ===========================================================================
# Errors
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
    def generate(self, system: str, messages: list[dict]) -> Any: ...


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
        self.total_cost_usd = 0.0
        self._client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.environ["OPENROUTER_API_KEY"],
        )

    @staticmethod
    def _extract_cost_usd(resp: Any) -> float:
        """Best-effort extraction of provider-reported dollar cost from a chat response."""
        candidates = []
        usage = getattr(resp, "usage", None)
        if usage is not None:
            candidates.extend([
                getattr(usage, "cost", None),
                getattr(usage, "total_cost", None),
                getattr(usage, "estimated_cost", None),
            ])
            usage_extra = getattr(usage, "model_extra", None) or {}
            if isinstance(usage_extra, dict):
                candidates.extend([
                    usage_extra.get("cost"),
                    usage_extra.get("total_cost"),
                    usage_extra.get("estimated_cost"),
                ])
        resp_extra = getattr(resp, "model_extra", None) or {}
        if isinstance(resp_extra, dict):
            candidates.extend([
                resp_extra.get("cost"),
                resp_extra.get("total_cost"),
                resp_extra.get("estimated_cost"),
            ])
            usage_extra = resp_extra.get("usage")
            if isinstance(usage_extra, dict):
                candidates.extend([
                    usage_extra.get("cost"),
                    usage_extra.get("total_cost"),
                    usage_extra.get("estimated_cost"),
                ])

        for value in candidates:
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0

    def generate(self, system: str, messages: list[dict]) -> tuple[str, str | None, dict[str, Any]]:
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
        cost_usd = self._extract_cost_usd(resp)
        self.total_cost_usd += cost_usd
        return content, reasoning, {"cost_usd": cost_usd}


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


def _normalize_plan_df(
    df: "pd.DataFrame",
    use_case: str,
    query_id: int,
) -> "pd.DataFrame":
    """Apply use-case/query-specific column normalization for evaluator compatibility."""
    if use_case == "ecomm":
        if query_id == 4:
            df = df.rename(columns={"prod_id": "id"})
        elif query_id == 11:
            if df.shape[1] >= 4:
                df["id"] = df.iloc[:, :4].astype(str).agg("-".join, axis=1)
                df = df[["id"]]
        elif query_id == 12:
            if df.shape[1] >= 3:
                import json as _json
                df["id"] = df.iloc[:, :3].apply(
                    lambda r: _json.dumps(
                        {"id": int(r.iloc[0]), "brand": r.iloc[1], "category": r.iloc[2]},
                        separators=(",", ":"),
                    ),
                    axis=1,
                )
                df = df[["id"]]
        elif query_id == 13:
            df = df.rename(columns={"prod_id": "id"})
            df = df[["id"]] if "id" in df.columns else df
    return df


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

    _LARGE_COLS = frozenset({"op_samples", "plan_str"})

    @property
    def df(self):
        """pandas view (requires pandas). Use `.rows` if you don't have pandas.
        Large blob columns (op_samples, plan_str) are excluded — use get_op_samples() instead."""
        import pandas as pd

        rows = [{k: v for k, v in r.items() if k not in self._LARGE_COLS} for r in self.rows]
        return pd.DataFrame(rows)

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


class EstimatePlanCostTool(Tool):
    name = "estimate_plan_cost"
    doc = """\
### estimate_plan_cost(plan)
Apply the CURRENTLY INSTALLED cost model to a physical (sub)plan and return its
`PlanCostEstimate` (dollar cost, latency seconds, optional quality). Errors if
you have not installed a cost model yet (do that with `update_cost_model`).
Note: estimates may not accurately reflect absolute execution costs, but provide
a good relative signal for comparing candidate plans. Use this before executing
to prioritize which plans are worth running.

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


class ComparePlanCostsTool(Tool):
    name = "compare_plan_costs"
    doc = """\
### compare_plan_costs(plan_names)
Estimate cost/latency for a list of NEW candidate plans using the CURRENTLY INSTALLED
cost model. ALL previously written plans (executed or not) are automatically included
for comparison. For executed plans, the actual observed cost/latency/quality are shown
alongside the estimate. Returns a table sorted by estimated cost.
Use this to rank candidate plans before deciding which to execute.
Errors if no cost model is installed yet.

```python
compare_plan_costs(["p3", "p4", "p5"])
# All previously written plans (e.g., p1, p2) are automatically included in the output.
```"""

    def __init__(self, registry: "CostModelRegistry", plans: dict, plan_results: "ResultsStore") -> None:
        self._registry = registry
        self._plans = plans
        self._plan_results = plan_results

    def __call__(self, plan_names: list) -> str:
        model = self._registry.current()

        actuals: dict[str, dict] = {}
        for row in self._plan_results.rows:
            name = row.get("plan_name")
            if name:
                actuals[name] = row

        # Combine: new candidates + all written plans (executed or not), deduplicated
        all_names = list(plan_names)
        for written_name in self._plans:
            if written_name not in all_names:
                all_names.append(written_name)

        rows = []
        errors = []
        for name in all_names:
            plan = self._plans.get(name)
            if plan is None:
                errors.append(f"  {name}: not found in plans dict (skipped)")
                continue
            try:
                est = model.estimate_plan(plan)
                if isinstance(est, dict):
                    est = PlanCostEstimate(**est)
            except Exception as e:
                errors.append(f"  {name}: estimation failed — {e}")
                continue
            actual = actuals.get(name, {})
            is_new = name in plan_names
            rows.append({
                "plan_name": name,
                "is_new": is_new,
                "est_cost": est.cost,
                "est_latency": est.time,
                "actual_cost": actual.get("cost_usd"),
                "actual_latency": actual.get("latency_s"),
                "actual_quality": actual.get("quality"),
            })

        rows.sort(key=lambda r: (r["est_cost"] is None, r["est_cost"] or 0.0))

        def fmt(v, fmt_str=".6f"):
            return "—" if v is None else format(float(v), fmt_str)

        header = f"{'plan':<12} {'new?':<6} {'est_cost':>12} {'est_latency':>12} {'actual_cost':>12} {'actual_latency':>14} {'actual_quality':>14}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            new_marker = "yes" if r["is_new"] else "no"
            lines.append(
                f"{r['plan_name']:<12} {new_marker:<6} {fmt(r['est_cost']):>12} {fmt(r['est_latency']):>12} "
                f"{fmt(r['actual_cost']):>12} {fmt(r['actual_latency']):>14} {fmt(r['actual_quality'], '.3f'):>14}"
            )
        if errors:
            lines.append("\nErrors:")
            lines.extend(errors)
        lines.append("\n(Sorted by estimated cost. All written plans included automatically. Absolute values may be inaccurate; use for relative comparison.)")
        return "\n".join(lines)


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
Optionally, use get_op_type(op) (e.g. 'sem_filter'), get_op_model(op) (e.g. 'google/gemini-2.5-flash-lite'), and describe_operator(op) to inspect each operator.
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
List all items in the data directory, showing whether each is a file or folder.

```python
list_files()
```"""

    def __init__(self, data_dir: str) -> None:
        import pathlib
        self._data_dir = pathlib.Path(data_dir)

    def __call__(self) -> str:
        lines = []
        for p in sorted(self._data_dir.iterdir()):
            kind = "dir" if p.is_dir() else "file"
            lines.append(f"  [{kind}] {p.name}")
        return "Data directory contents:\n" + "\n".join(lines)


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


class GetOpSamplesTool(Tool):
    name = "get_op_samples"
    doc = """\
### get_op_samples(plan_name, op_name=None, n=3)
Retrieve sample (input, output) pairs for an executed plan from `plan_results`.
Optionally scope to a single operator with `op_name` (e.g. "p1_op1").
Returns at most `n` samples per operator.

```python
get_op_samples("p1")                  # all operators, 3 samples each
get_op_samples("p1", "p1_op2", n=5)  # just op2, up to 5 samples
```"""

    def __init__(self, plan_results: ResultsStore) -> None:
        self._plan_results = plan_results

    def __call__(self, plan_name: str, op_name: str | None = None, n: int = 3) -> str:
        row = next((r for r in self._plan_results.rows if r.get("plan_name") == plan_name), None)
        if row is None:
            available = [r.get("plan_name") for r in self._plan_results.rows]
            return f"No executed plan named {plan_name!r}. Available: {available}"
        samples = row.get("op_samples", {})
        if not samples:
            return f"No op_samples recorded for plan {plan_name!r}."
        if op_name is not None:
            if op_name not in samples:
                return f"No operator {op_name!r} in plan {plan_name!r}. Available: {list(samples)}"
            samples = {op_name: samples[op_name]}
        lines = []
        for op, pairs in samples.items():
            lines.append(f"{op} ({min(len(pairs), n)}/{len(pairs)} samples shown):")
            for i, pair in enumerate(pairs[:n]):
                lines.append(f"  [{i}] input:  {pair['input']}")
                lines.append(f"       output: {pair['output']}")
        return "\n".join(lines)


class WritePlanTool(Tool):
    name = "write_plan"
    doc = """\
### write_plan(code, name, description="")
Build and store a physical query plan WITHOUT executing it. `code` is a Python
string that constructs a PhysicalPipeline and returns it as its last expression —
do NOT call `.run()` in the plan code; `execute_plan` handles execution.
`name` is the plan identifier you choose (e.g. "p1"). `description` is a short
high-level label of the plan and the optimizations it embodies
(e.g. "cheap sem_filter on truncated text, then sem_filter on image") — it is
shown back to you in cost/estimate tables and helps you compare optimization ideas.

After this call, `plans[name]["plan"]` holds the built pipeline and
`plans[name]["description"]` holds your label.
Use the load_data(filename) function to read CSVs from the data directory in your plan code.

```python
write_plan(\"\"\"
pipeline = PhysicalPipeline(plan_name, "emails", load_data("Emails.csv"))
pipeline.sem_filter("this email quotes someone outside of the the sender's company", model=pz.Model.GOOGLE_GEMINI_2_5_FLASH_LITE)
pipeline.project(["emailId"])
pipeline.limit(5)
pipeline
\"\"\", "p1", description="baseline: single cheap sem_filter on full text")
# `plan_name` is automatically set to the name you pass (here "p1")
# plans["p1"]["plan"] now holds the built pipeline
```"""

    def __init__(self, plan_codes: dict, plans: dict, executor: Any) -> None:
        self._plan_codes = plan_codes
        self._plans = plans
        self._executor = executor

    def __call__(self, code: str, plan_name: str, description: str = "") -> dict:
        try:
            from agent.physical_pipeline import PhysicalPipeline
        except ImportError:
            from physical_pipeline import PhysicalPipeline  # type: ignore

        self._executor.send_variables({"plan_name": plan_name})
        exec_result = self._executor(code)
        pipeline = exec_result.output
        if not isinstance(pipeline, PhysicalPipeline):
            raise TypeError(
                f"Plan code must return a PhysicalPipeline as its last expression "
                f"(got {type(pipeline).__name__}). Do not call .run() in the plan code."
            )

        self._plan_codes[plan_name] = code
        self._plans[plan_name] = {"plan": pipeline, "description": description}
        return {"plan_name": plan_name, "description": description, "total_plans": len(self._plan_codes)}


class ExecutePlanTool(Tool):
    name = "execute_plan"
    doc = """\
### execute_plan(name)
Execute the stored plan `name` on a reproducible sample of records. Plan-level and
operator-level quality, latency, cost, and token usage are appended to `plan_results`
and `op_results`, respectively. After execution, `plans[name]` holds the PhysicalPipeline.

Returns a compact summary dict with plan stats and per-operator stats. Key fields:
- `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
  Treat oracle quality scores as ground truth.
- `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
  which operator is the bottleneck.
- `cost_usd`, `latency_s`, `input_tokens`, `output_tokens`: aggregated over all ops.
To inspect accumulated results use `plan_results.df` and `op_results.df`.
To view sample input/output pairs use `get_op_samples(plan_name)`.
```python
execute_plan("p1")
```"""

    def __init__(
        self,
        plan_codes: dict,
        plans: dict,
        plan_results: ResultsStore,
        op_results: ResultsStore,
        use_case: str,
        query_id: int,
        data_dir: str,
        agent_dir: str,
        quality_evaluator: Any,
    ) -> None:
        import pathlib

        self._plan_codes = plan_codes
        self._plans = plans
        self._op_results = op_results
        self._plan_results = plan_results
        self._use_case = use_case
        self._query_id = query_id
        self._data_dir = pathlib.Path(data_dir)
        self._agent_dir = agent_dir
        self._quality_evaluator = quality_evaluator

    def __call__(self, plan_name: str) -> dict:
        import pandas as pd

        pipeline = self._plans.get(plan_name)
        if pipeline is None:
            raise KeyError(
                f"No plan named {plan_name!r}. Call write_plan first. "
                f"Available: {list(self._plans)}"
            )

        subset_path = pathlib.Path(f"agent_cost_model/datasubset/{self._use_case}/Q{self._query_id}_subset.csv")
        plan_exec_error: Exception | None = None
        per_op_list, plan_context = [], None
        try:
            per_op_list, plan_context = pipeline.run_subset(subset_cache_path=str(subset_path))
        except Exception as e:
            plan_exec_error = e
            print(f"[execute_plan] plan execution failed for {plan_name}: {type(e).__name__}: {e}")
        self._plans[plan_name] = pipeline

        plan_output_df = pd.DataFrame(
            plan_context.output_records if plan_context is not None else []
        )
        plan_output_df = _normalize_plan_df(plan_output_df, self._use_case, self._query_id)

        quality_result = None
        if self._quality_evaluator is not None:
            try:
                # Pass an empty SubsetExecutionContext when the plan failed so the oracle
                # still runs (populating _canonical_oracle_df for later plans).
                if plan_context is None:
                    from agent.physical_pipeline import SubsetExecutionContext
                    plan_context = SubsetExecutionContext(
                        sampled_records=[], output_records=[],
                        per_sem_op_info={}, has_join=False, right_sampled_records=None,
                    )
                quality_result = self._quality_evaluator.evaluate(
                    pipeline, plan_name, plan_context, plan_output_df
                )
            except Exception as e:
                print(f"[execute_plan] quality evaluation failed for {plan_name}: {type(e).__name__}: {e}")

        if plan_exec_error is not None:
            raise RuntimeError(
                f"Plan {plan_name!r} execution failed: {type(plan_exec_error).__name__}: {plan_exec_error}"
            )

        # Build op_samples for GetOpSamplesTool compatibility
        op_samples: dict[str, list] = {}
        for _stage_idx, info in plan_context.per_sem_op_info.items():
            op_n = info["op_name"]
            raw_samples = info.get("samples", [])
            op_samples[op_n] = [
                {"input": str(inp), "output": str(out) if out is not None else None}
                for inp, out in raw_samples[:5]
            ]

        total_cost = sum(e.get("cost_usd", 0) for e in per_op_list)
        total_latency = sum(e.get("latency_s", 0) for e in per_op_list)
        total_in_tok = sum(e.get("input_tokens", 0) for e in per_op_list)
        total_out_tok = sum(e.get("output_tokens", 0) for e in per_op_list)

        plan_row = {
            "plan_name": plan_name,
            "plan_str": str(pipeline),
            "cost_usd": total_cost,
            "latency_s": total_latency,
            "input_tokens": total_in_tok,
            "output_tokens": total_out_tok,
            "quality": quality_result.quality if quality_result is not None else float("nan"),
            "per_sem_op_quality": (
                quality_result.per_sem_op_quality if quality_result is not None else {}
            ),
            "op_samples": op_samples,
        }
        self._plan_results.append([plan_row])
        self._op_results.append(per_op_list)

        plan_summary = {k: v for k, v in plan_row.items() if k not in ("op_samples", "plan_str")}
        return {"plan_summary": plan_summary, "op_summary": per_op_list}


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

_PHYSICAL_SEMANTIC_OPERATORS = {
    "sem_filter": "pipeline.sem_filter(condition: str, model: pz.Mode) — LLM filter; keeps rows where condition is true.",
    "sem_map": "pipeline.sem_map(cols: list[dict], model: pz.Model) — Add LLM-derived columns. col is list of dict {'name': str, 'type': type, 'description': str}",
    "sem_join": "pipeline.sem_join(other: PhysicalPipeline, condition: str, model: pz.Model) — LLM join; keeps pairs where condition holds.",
}

_PHYSICAL_NONSEMANTIC_OPERATORS = {
    "filter": "pipeline.filter(fn: Callable[[dict], bool]) — Exact row filter using a Python callable.",
    "map": "pipeline.map(fn: Callable[[dict], dict], cols: list[dict]) - Exact row map using a Python callable to add new columns. col is list of dict {'name': str, 'type': type, 'description': str}",
    "project": "pipeline.project(cols: list[str]) — Select a subset of columns.",
    "limit": "pipeline.limit(n: int) — Keep at most n rows.",
    "groupby": "pipeline.groupby(group_by_fields: list[str], agg_funcs: list[str], agg_fields: list[str]) — Group and aggregate. Produces schema name 'agg_func(agg_field)', e.g. 'count(reviewId)' or 'average(score)'.",
}
_AVAILABLE_MODELS_TEXT = (pathlib.Path(__file__).parent / "available_models.txt").read_text()

_SYSTEM_TEMPLATE = """\
{briefing}

## HARD RULES

### No data snooping or hardcoded indexes/phrases
`explore_sample` and `explore_schema` exist to help you understand **schema and format only**.
You MUST NOT use sample rows to identify specific records and then hardcode their IDs, row
indexes, or literal field values into a plan.  Every filter predicate in your plan must be a
**general condition** that could correctly classify records it has never seen — for example
`sem_filter("the text is clearly positive")` or `filter(lambda row: row["name"] == "John")`.
Writing plans like `filter(lambda row: row["id"] in [3, 17, 42])`,
`filter(lambda row: "good" in row["text"])`, or using excessive keyword search to cherry-pick rows
is **cheating** and will produce meaningless results. Rely on semantic operators instead of constructing
complex keyword/regex/specific row selection.
Furthermore, physical plans should only involve trees of the given semantic and non-semantic operators.
Do not construct your own methods, rely on these operators only.
If your plan contains any hardcoded record IDs or values you copied from `explore_sample`
output, rewrite it before calling `execute_plan`.

{estimate_rule}

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
Using `depends_on` will reduce the LLM input and help reduce cost.
{physical_sem_ops}

Non-semantic operators — no model argument:
{physical_nonsem_ops}

## Available Models:
All models listed support both text and image inputs. Use `add_image_data` with a `depends_on`
that includes the image column to pass images to any model.
Within each tier, cheaper options are listed first. Prefer the cheaper models unless improving
quality requires a more expensive model.
{available_models}

## Also available in your sandbox (no import needed)
- `load_data(filename)` : read a CSV from the data directory and return a DataFrame.
    df = load_data("items.csv")
- `add_image_data(pipeline: PhysicalPipeline, col_name: str)` : returns a `PhysicalPipeline` with
    an added `col_name` column of type `pz.ImageFilepath` to `pipeline` that contains the path to each row's image file.
   Whenever a semantic operator depends on a column with type `pz.ImageFilepath`, it will
   encode it as a base64 image to send to the LLM as a vision input.

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
        and converge on the best cost-quality trade-off. Do not hard code the plan.

        Suggested workflow:
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
           - `plans[name]` is populated immediately with the newly written PhysicalPipeline instance.
        3. Execute with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck.
        5. Based on `plan_results.df` and `op_results.df`, write a new plan that has higher
           quality or lower cost/latency and repeat from step 2.
                                       
        Consider the following impacts on plan cost and latency:
        - Cardinality: costs scale non-linearly — a selective filter reduces cardinality into downstream
          operators.
        - Token count: LLM cost scales with input/output token count, not just record count.
          `op_results.df` includes `input_tokens` and `output_tokens`.
        - Column selection: the `depends_on=[...]` parameter on `sem_filter`/`sem_map` controls
          which columns enter the LLM context. Narrowing `depends_on` reduces input tokens
          and is a key cost lever to model and exploit.
        - Images: image inputs cost orders of magnitude more than text tokens. Avoiding `add_image_data`
          or excluding the image column from `depends_on` when not needed drastically cuts cost.
        - Text truncation: long text fields can be truncated with a `map` operator before semantic
          ops (e.g., `map(lambda row: {"text": row["text"][:500]}, cols=[...])`), reducing input tokens.
        """)

    sampleCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds. A sample-based cost model (SampleBasedCostModel) is
        pre-installed and re-fits automatically on every `estimate_plan_cost` call —
        use it to compare plan variants before deciding which ones to execute.

        Your goal is to write physical plans, observe cost/latency/quality,
        and converge on the best cost-quality trade-off. Do not hard code the plan.

        Suggested workflow:
        To get initial data and performance traces:
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write a plan with `write_plan(code, name)`.
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
           - `plans[name]` is populated immediately — call `estimate_plan_cost(plans[name])`
             right after `write_plan` to get a pre-execution cost estimate.
        3. Execute with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck.

        Then, iteratively write new plans, estimate their cost, and execute the most promising ones:
        - Use `plan_results.df` and `op_results.df` to identify quality and cost/latency trade-offs
            -- `plan_results.df` includes operator descriptions; use `get_op_samples(plan_name)` to view input/output pairs
            -- `op_results.df` includes per-operator performances, operators are named by `name_op1`, `name_op2`, ...
        - Always use `estimate_plan_cost(plans[name])` to get cost/latency estimates BEFORE executing promising plans.
            -- the cost model averages past execution to get per-operator cost/latency estimates
            -- thus, only changing a few operators for each plan writing will produce more comparable estimations
        """)

    customCost_briefing = textwrap.dedent("""\
        You are a query plan engineer optimizing physical plans for an optimized deep-research query system.
        Physical query plans are trees of operators (semantic filters, maps, joins,
        aggregations, scans, projects). Some operators call LLMs and cost real
        dollars and seconds.

        Your goal is to design a custom cost model (cost and latency) to guide iterative plan search,
        observe cost/latency/quality, and converge on the best cost/latency-quality trade-off.
        Do not hard code the plan.

        === Phase 1 — Bootstrap ===
        1. Explore the data: call `list_files()` to see available CSVs and folders,
           `explore_schema(filename)` and `explore_sample(filename)` to understand each table.
        2. Write 1–2 baseline plans with `write_plan(code, name)` that capture meaningfully different
           design approaches (e.g., one with a strong early filter, one without; one using a
           capable model, one using a cheaper model).
           - `code` builds a PhysicalPipeline instance and returns it as the last expression.
           - `name` is a string identifier you choose, e.g. "p1", "p2", "p3", ...
             Use a NEW unique name for each new plan — never reuse a name for a different plan.
           - NEVER hardcode row IDs, indexes, or specific field values you found by browsing
             data. Use `sem_filter` / `sem_map` with natural-language conditions or schema-level
             predicates (e.g. `filter(lambda row: row["score"] >= 4)`). Plans containing
             hardcoded record IDs or values copied from `explore_sample` output are invalid.
        3. Execute baseline plans with `execute_plan(name)`:
           - Runs on a reproducible sample of records (same records across all plans).
           - Appends plan-level stats to `plan_results` (cost_usd, latency_s, tokens, quality).
           - Appends per-operator stats to `op_results` (cost_usd, latency_s, tokens, num_records per op).
           - After execution, `plans[name]` is updated with the executed PhysicalPipeline.
        4. Evaluate with `plan_results.df` and `op_results.df`:
           - `quality`: 0–1 overall plan quality evaluated by an oracle. Higher is better.
             Treat oracle quality scores as ground truth — do not try to replicate or
             reverse-engineer the oracle; simply observe and optimize.
           - `per_sem_op_quality`: per-semantic-operator quality (0–1). Use to diagnose
             which operator is the bottleneck.
           - Use `get_op_samples(plan_name)` to inspect input/output pairs for an operator.
        5. Design and install a cost model with `update_cost_model(YourClass, notes="v1: ...")`.
           - If your class __init__ takes `op_results` or `plan_results`, they are passed automatically
             so you can fit coefficients from the accumulated data inside __init__.
           - You may call `update_cost_model` multiple times to refine the model across versions.

        === Designing your cost model ===
        Your cost model is a tool for RELATIVE comparison between candidate plans — the absolute
        values may not accurately reflect actual execution costs, but the relative estimates should
        guide which plans to prioritize for execution.

        Cost scales with both record count AND record size. Consider the following:
        - Cardinality: costs scale non-linearly — a selective filter reduces cardinality into downstream
          operators. Use `num_passed/num_records` to estimate downstream cardinality.
        - Token count: LLM cost scales with input/output token count, not just record count.
          `op_results.df` includes `input_tokens` and `output_tokens` — use these (divided by
          `num_records`) to get a per-record token cost proxy that captures actual record size.
        - Column selection: the `depends_on=[...]` parameter on `sem_filter`/`sem_map` controls
          which columns enter the LLM context. Narrowing `depends_on` reduces input tokens
          and is a key cost lever to model and exploit.
        - Images: image inputs cost orders of magnitude more than text tokens. Avoiding `add_image_data`
          or excluding the image column from `depends_on` when not needed drastically cuts cost.
        - Text truncation: long text fields can be truncated with a `map` operator before semantic
          ops (e.g., `map(lambda row: {"text": row["text"][:500]}, cols=[...])`), reducing input tokens.
        - When designing your cost model, use `input_tokens / num_records` as a per-record token
          size signal alongside selectivity. You have full freedom to model cost scaling any way
          you judge best — linear, step-function, cardinality-aware, or otherwise.
        - When results data is sparse, consider using naive prior estimates for unseen operators.

        === Phase 2 — Iteration cycles ===
        Each cycle:
        1. Decide what to improve: explicitly reason about ONE dimension to target — e.g., swap to
           a cheaper/faster model for an operator, consolidate or split prompts, reorder operators
           to push selective filters earlier, narrow `depends_on` to fewer columns, truncate long
           text fields, remove images, change logical structure.
        2. Write 2–4 candidate plans that each embody a specific targeted change. Use new unique names.
        3. Call `compare_plan_costs([name1, name2, ...])` on ALL new candidates.
           This tool automatically includes all previously executed plans in the output (with their
           actual cost/latency/quality shown alongside estimates), so you can directly compare new
           candidates against the plans you have already run.
        4. Execute only the 1–2 most promising plans based on the cost/quality tradeoff from the
           comparison — using what you already know about quality from previously executed plans
           to judge whether a cheaper plan is likely to maintain acceptable quality.
        5. Update the cost model if new observations reveal the model was systematically wrong.

        Caveat: if a change is not well-captured by the cost model (e.g., a new operator type with
        no prior observations, or a change that significantly alters output token counts), flag this
        uncertainty and lean toward executing to gather data rather than purely trusting the estimate.
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
        data_dir: str = "dataset/use_case",
        agent_dir: str = "no_name",
        use_case: str = "use_case",
        max_steps: int = 12,
        max_recover_retries: int = 1,
        context_budget_chars: int = 200_000,
        authorized_imports: list[str] | None = None,
        verbose: bool = True,
        oracle_model: str = "openai/o4-mini",
        oracle_reasoning_effort: str | None = "high",
    ) -> None:
        self.llm = llm
        self.agent_dir = agent_dir
        self.use_case = use_case
        self.data_dir = data_dir
        self.oracle_model = oracle_model
        self.oracle_reasoning_effort = oracle_reasoning_effort
        self.max_steps = max_steps
        self.max_recover_retries = max_recover_retries
        self.context_budget_chars = context_budget_chars
        self.authorized_imports = authorized_imports or [
            "math", "statistics", "json", "collections", "itertools",
            "palimpzest"
        ]
        self.verbose = verbose
        # Rebuilt per run(); kept on the instance so callers can read it after.
        self.messages: list[dict] = []
        self.reasoning_steps: list[str | None] = []
        self.trajectory_steps: list[dict] = []
        self.agent_cost_usd: float = 0.0
        self.execution_cost_usd: float = 0.0
        self.oracle_cost_usd: float = 0.0

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
        metrics_dir = data_path.parents[1] / "trajectory" / self.use_case
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = metrics_dir / f"{qkey}_{self.agent_dir}_trajectory.csv"
        pd.DataFrame(self.trajectory_steps).to_csv(out, index=False)
        self._log(f"[run] trajectory → {out}")

    def _save_cost_model_codes(self, codes: dict, query_info: dict) -> None:
        """Save all versioned cost model code strings to
        agent_cost_model/costModel/{use_case}/{agent_dir}/Q{id}.json."""
        if not codes:
            return
        import json
        import pathlib
        out_dir = pathlib.Path(f"agent_cost_model/costModel/{self.use_case}/{self.agent_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = out_dir / f"{qkey}.json"
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
        metrics_dir = data_path.parents[1] / "metrics" / self.use_case
        metrics_dir.mkdir(parents=True, exist_ok=True)
        rc = query_info.get("runcount")
        qkey = f"Q{query_info['query_id']}_{rc}" if rc is not None else f"Q{query_info['query_id']}"
        out = metrics_dir / f"{qkey}_{self.agent_dir}_results.csv"
        df = results.df
        df["use_case"] = query_info["use_case"]
        df["query_id"] = query_info["query_id"]
        best_name = (final_answer or {}).get("best_plan", {}).get("name")
        df["final_selected"] = df["plan_name"] == best_name if best_name else False
        # re-attach large blob columns (stripped by results.df) and move them to the end
        for col in ["plan_str", "op_samples"]:
            if col not in df.columns:
                df[col] = [r.get(col) for r in results.rows]
        df = df[[col for col in df.columns if col not in ["plan_str", "op_samples"]] + ["plan_str", "op_samples"]]
        df.to_csv(out, index=False)
        self._log(f"[run] results table → {out}")

    # -- prompt assembly ---------------------------------------------------
    def _system_prompt(
        self,
        tools: list[Tool],
        briefing: str | None = None,
        final_answer_doc: str | None = None,
        mode: str = "",
    ) -> str:
        if "customCost" in mode or "sampleCost" in mode:
            estimate_rule = (
                "### Estimate before executing\n"
                "Once a cost model is installed, you MUST call `compare_plan_costs` on all new candidate plans\n"
                "before executing any of them. Executing a plan without first consulting cost estimates wastes\n"
                "budget steps and defeats the purpose of the cost model."
            )
        else:
            estimate_rule = ""
        return _SYSTEM_TEMPLATE.format(
            briefing=briefing if briefing is not None else self.briefing,
            tools_doc="\n\n".join(t.doc for t in tools),
            physical_sem_ops="\n".join(f"- {d}" for d in _PHYSICAL_SEMANTIC_OPERATORS.values()),
            physical_nonsem_ops="\n".join(f"- {d}" for d in _PHYSICAL_NONSEMANTIC_OPERATORS.values()),
            available_models=_AVAILABLE_MODELS_TEXT,
            max_steps=self.max_steps,
            final_answer_doc=final_answer_doc if final_answer_doc is not None else self.final_answer_doc,
            estimate_rule=estimate_rule,
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
            f"Begin. Output exactly ONE ```python``` block for your first step — "
            f"a single tool call (e.g. list_files()). Do not write multiple blocks, "
            f"plan ahead in prose, or produce a final answer yet."
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
            "use_case": "use_case",
            "scale_factor": 0,
            "query_id": 0,
            "data_dir": "agent_cost_model/dataset/use_case",
            "gt_dir": "files/use_case/raw_results/ground_truth"
        },
    ) -> Any:
        """Run the loop over `plans` and `plan_results`, `op_results`, returning the JSON final answer.

        `plans` is a dict {name: physical_plan}; `plan_results` and `op_results` are ResultsStore objects.
        A fresh CostModelRegistry is created per run.
        """
        import litellm as _litellm
        _litellm.suppress_debug_info = True
        _litellm.drop_params = True  # OpenRouter rejects reasoning_effort for Google models
        _orig_completion = _litellm.completion
        def _openrouter_completion(model, **kwargs):
            if not model.startswith("openrouter/"):
                model = "openrouter/" + model
            return _orig_completion(model=model, **kwargs)
        _litellm.completion = _openrouter_completion

        from local_python_executor import LocalPythonExecutor

        registry = CostModelRegistry()
        plan_codes: dict = {}  # populated by WritePlanTool; shared with ExecutePlanTool
        data_dir = query_info["data_dir"]

        # Build oracle client and quality evaluator (oracle runs inside QualityEvaluator)
        oracle_client = OpenRouterClient(self.oracle_model, reasoning_effort=self.oracle_reasoning_effort)
        llm_judge_dir = f"files/{query_info['use_case']}/llm_judge"
        import shutil
        _llm_judge_path = pathlib.Path(llm_judge_dir)
        if _llm_judge_path.exists():
            shutil.rmtree(_llm_judge_path)
        _llm_judge_path.mkdir(parents=True, exist_ok=True)

        _oracle_result_path = (
            pathlib.Path(__file__).parent
            / "datasubset"
            / query_info["use_case"]
            / f"Q{query_info['query_id']}_oracle_result.csv"
        )
        _oracle_result_path.unlink(missing_ok=True)

        quality_evaluator = None
        if _QualityEvaluator is not None:
            try:
                quality_evaluator = _QualityEvaluator(
                    oracle_client=oracle_client,
                    oracle_model=self.oracle_model,
                    query_id=query_info["query_id"],
                    use_case=query_info["use_case"],
                    scale_factor=query_info["scale_factor"],
                    agent_dir=self.agent_dir,
                    llm_judge_dir=llm_judge_dir,
                    oracle_reasoning_effort=self.oracle_reasoning_effort,
                )
            except Exception as e:
                print(f"[run] QualityEvaluator init failed: {e}")

        import pandas as pd
        try:
            from agent.physical_pipeline import PhysicalPipeline
        except ImportError:
            from physical_pipeline import PhysicalPipeline  # type: ignore
        try:
            import palimpzest as pz
        except ImportError as exc:
            raise ImportError("palimpzest is required to build plans") from exc

        def load_data(filename: str) -> pd.DataFrame:
            return pd.read_csv(os.path.join(data_dir, filename))

        def add_image_data(pipeline: PhysicalPipeline, col_name: str = "image_file_path"):
            pipeline.map(
                udf=lambda row: {col_name: os.path.join(data_dir, "images", str(row["prod_id"]) + ".jpg")},
                cols=[{"name": col_name, "type": pz.ImageFilepath, "description": ""}],
            )
            return pipeline

        variables = {
            "load_data": load_data,
            "add_image_data": add_image_data,
            "PhysicalPipeline": PhysicalPipeline,
            "pz": pz,
            "plans": plans,
            "plan_codes": plan_codes,
            "plan_results": plan_results,
            "op_results": op_results,
            "iter_operators": iter_operators,
            "get_op_type": get_op_type,
            "get_op_id": get_op_id,
            "get_op_model": get_op_model,
            "describe_operator": describe_operator,
        }

        base_tools = [
            ListFilesTool(data_dir),
            ExploreSchemaT(data_dir),
            ExploreSampleTool(data_dir),
            GetOpSamplesTool(plan_results),
            ExecutePlanTool(
                plan_codes=plan_codes,
                plans=plans,
                plan_results=plan_results,
                op_results=op_results,
                use_case=query_info["use_case"],
                query_id=query_info["query_id"],
                data_dir=data_dir,
                agent_dir=self.agent_dir,
                quality_evaluator=quality_evaluator,
            ),
        ]

        if "customCost" in mode:
            base_tools += [
                EstimatePlanCostTool(registry),
                ComparePlanCostsTool(registry, plans=plans, plan_results=plan_results),
                UpdateCostModelTool(registry, plan_results=plan_results, op_results=op_results),
            ]
            variables.update({
                "PlanCostEstimate": PlanCostEstimate,
                "CostModel": CostModel,
            })
            briefing = self.customCost_briefing
            final_answer_doc = self.customCost_final_answer_doc
        elif "execute" in mode:
            briefing = self.execute_briefing
            final_answer_doc = self.execute_final_answer_doc
        elif "sampleCost" in mode:
            try:
                from sample_based_cost_model import SampleBasedCostModel
            except ImportError:
                from agent_cost_model.sample_based_cost_model import SampleBasedCostModel  # type: ignore
            registry.install(SampleBasedCostModel(op_results), notes="v0: SampleBasedCostModel pre-installed")
            base_tools += [EstimatePlanCostTool(registry)]
            variables.update({
                "PlanCostEstimate": PlanCostEstimate,
                "SampleBasedCostModel": SampleBasedCostModel,
            })
            briefing = self.sampleCost_briefing
            final_answer_doc = self.sampleCost_final_answer_doc

        # Create executor first so WritePlanTool can share the same sandbox state.
        executor = LocalPythonExecutor(additional_authorized_imports=self.authorized_imports)
        executor.send_variables(variables)
        write_plan_tool = WritePlanTool(plan_codes, plans=plans, executor=executor)
        tools = base_tools + [write_plan_tool]
        executor.send_tools({t.name: t for t in tools})

        system = self._system_prompt(tools, briefing, final_answer_doc, mode=mode)
        opening = self._opening_message(task, plans, op_results)
        self.messages = [{"role": "user", "content": opening}]
        self.reasoning_steps = []
        self.trajectory_steps = []
        self.agent_cost_usd = 0.0
        self.execution_cost_usd = 0.0
        self.oracle_cost_usd = 0.0
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
            self.trajectory_steps.append(
                {"step": step, "reasoning": reasoning, "assistant": raw, "observation": None}
            )
            if reasoning:
                self._log(f"\n--- reasoning (step {step}) ---\n{reasoning}\n")
            self._log(f"\n--- assistant (step {step}) ---\n{raw}\n")

            # parse (with a couple of format-error retries)
            try:
                parsed = self._parse_with_retries(system, raw)
            except ParseError as e:
                obs = f"Observation (step {step}): {e.detail}"
                self.messages.append({"role": "user", "content": obs})
                self.trajectory_steps[-1]["observation"] = obs
                self._log(f"[parse error] {obs}")
                continue

            if parsed.code is None:  # final answer
                self._log(f"[final answer] {parsed.result}")
                self.trajectory_steps[-1]["observation"] = f"[final answer] {parsed.result}"
                if isinstance(parsed.result, dict):
                    parsed.result["plan_codes"] = plan_codes
                self._save_results_df(plan_results, query_info, final_answer=parsed.result)
                self._save_trajectory_df(query_info)
                if "customCost" in mode:
                    self._save_cost_model_codes(cost_model_codes, query_info)
                self.execution_cost_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in plan_results.rows)
                self.oracle_cost_usd = (
                    float(getattr(quality_evaluator, "total_oracle_cost_usd", 0.0) or 0.0)
                    if quality_evaluator is not None else 0.0
                )
                self._run_final_evaluation(parsed.result, plans, query_info, plan_codes, plan_results)
                return parsed.result

            # execute the python tool-call block
            prev_registry_version = registry.version
            try:
                out = executor(parsed.code)
            except Exception as e:
                obs = f"Observation (step {step}): exec failed — {type(e).__name__}: {e}"
                self.messages.append({"role": "user", "content": obs})
                self.trajectory_steps[-1]["observation"] = obs
                self._log(obs)
                continue

            if "customCost" in mode and registry.version > prev_registry_version:
                cost_model_codes[f"v{registry.version}"] = parsed.code

            obs = self._format_observation(step, out)
            self.messages.append({"role": "user", "content": obs})
            self.trajectory_steps[-1]["observation"] = obs
            self._log(obs)

        # out of steps — one forced terminal turn
        self._save_results_df(plan_results, query_info)
        self._save_trajectory_df(query_info)
        if "customCost" in mode:
            self._save_cost_model_codes(cost_model_codes, query_info)
        result = self._terminal_turn(system)
        if isinstance(result, dict):
            result["plan_codes"] = plan_codes
        self.execution_cost_usd = sum(float(r.get("cost_usd", 0.0) or 0.0) for r in plan_results.rows)
        self.oracle_cost_usd = (
            float(getattr(quality_evaluator, "total_oracle_cost_usd", 0.0) or 0.0)
            if quality_evaluator is not None else 0.0
        )
        self._run_final_evaluation(result, plans, query_info, plan_codes, plan_results)
        return result

    # -- helpers -----------------------------------------------------------
    def _llm_step(self, system: str, extra: list[dict] | None = None) -> str:
        msgs = self._trim(self.messages)
        if extra:
            msgs = msgs + extra
        result = self.llm.generate(system, msgs)
        content = result
        reasoning = None
        meta: dict[str, Any] = {}
        if isinstance(result, tuple):
            if len(result) >= 1:
                content = result[0]
            if len(result) >= 2:
                reasoning = result[1]
            if len(result) >= 3 and isinstance(result[2], dict):
                meta = result[2]
        self.agent_cost_usd += float(meta.get("cost_usd", 0.0) or 0.0)
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

    _OBS_CHAR_LIMIT = 10_000

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
        obs = "\n\n".join(parts)
        limit = CostModelAgent._OBS_CHAR_LIMIT
        if len(obs) > limit:
            obs = obs[:limit] + (
                f"\n\n[output truncated — {len(obs) - limit} chars omitted. "
                "Use get_op_samples(plan_name, op_name) to inspect specific input/output pairs.]"
            )
        return obs

    def _run_final_evaluation(
        self,
        final_answer: Any,
        plans: dict,
        query_info: dict,
        plan_codes: dict | None = None,
        plan_results: "ResultsStore | None" = None,
    ) -> None:
        """Run the agent-selected plan on the full dataset vs. real ground truth; append to metrics JSON."""
        if not isinstance(final_answer, dict):
            return
        best_name = final_answer.get("best_plan", {}).get("name")
        if not best_name or best_name not in plans:
            self._log(f"[final_eval] plan {best_name!r} not in plans — skipping final evaluation")
            return

        pipeline = plans[best_name]
        query_id = query_info["query_id"]
        use_case = query_info["use_case"]
        scale_factor = query_info["scale_factor"]
        gt_dir = query_info.get("gt_dir", f"files/{use_case}/raw_results/ground_truth")

        import dataclasses
        import pathlib

        import pandas as pd

        gt_path = pathlib.Path(gt_dir) / f"Q{query_id}.csv"
        if not gt_path.exists():
            self._log(f"[final_eval] ground truth not found at {gt_path} — skipping")
            return

        self._log(f"[final_eval] running {best_name!r} on full dataset...")
        try:
            result_collection, op_results_full, _ = pipeline.run()
        except Exception as e:
            self._log(f"[final_eval] pipeline.run() failed: {type(e).__name__}: {e}")
            return

        results_df = result_collection.to_df()
        raw_results_dir = pathlib.Path(f"files/{use_case}/raw_results/palimpzest/{self.agent_dir}")
        raw_results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(raw_results_dir / f"Q{query_id}.csv", index=False)
        self._log(f"[final_eval] raw results → {raw_results_dir / f'Q{query_id}.csv'}")
        results_df = _normalize_plan_df(results_df, use_case, query_id)

        total_latency = sum(r.get("latency_s", 0) for r in op_results_full)
        total_cost = sum(r.get("cost_usd", 0) for r in op_results_full)

        quality = float("nan")
        metric_type = "unknown"
        try:
            from agent_cost_model.quality_evaluator import _load_evaluator
            evaluator = _load_evaluator(use_case, scale_factor, self.agent_dir)
            gt_df = pd.read_csv(gt_path)
            qm = evaluator._evaluate_single_query(query_id, results_df, gt_df)
            qm_dict = dataclasses.asdict(qm)
            qm_type = type(qm).__name__
            if "Retrieval" in qm_type:
                metric_type = "f1_score"
                quality = float(qm_dict.get("f1_score", float("nan")))
            elif "Aggregation" in qm_type:
                metric_type = "relative_error"
                quality = 1.0 - float(qm_dict.get("relative_error", 1.0))
            elif "Rank" in qm_type:
                metric_type = "spearman_correlation"
                quality = float(qm_dict.get("spearman_correlation", float("nan")))
            elif "SingleAccuracy" in qm_type:
                metric_type = "accuracy"
                quality = float(qm_dict.get("accuracy", float("nan")))
        except Exception as e:
            self._log(f"[final_eval] evaluation failed: {type(e).__name__}: {e}")

        metrics_path = pathlib.Path(f"files/{use_case}/metrics/{self.agent_dir}.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        entry: dict = {}
        if metrics_path.exists():
            try:
                entry = json.loads(metrics_path.read_text())
            except Exception:
                entry = {}
        plans_written = len(plan_codes) if plan_codes is not None else 0
        _executed_rows = [r.get("plan_name") for r in (plan_results.rows if plan_results is not None else []) if r.get("plan_name")]
        plans_executed = len(_executed_rows)
        unique_plans_executed = len(set(_executed_rows))
        runcount = query_info.get("runcount")
        metrics_key = f"{query_id}_{runcount}" if runcount is not None else str(query_id)
        entry[metrics_key] = {
            "query_id": str(query_id),
            "latency": round(total_latency, 4),
            "cost": round(total_cost, 6),
            "agent_cost": round(self.agent_cost_usd, 6),
            "subset_execution_cost": round(self.execution_cost_usd, 6),
            "oracle_cost": round(self.oracle_cost_usd, 6),
            "metric_type": metric_type,
            "quality": quality,
            "plans_written": plans_written,
            "plans_executed": plans_executed,
            "unique_plans_executed": unique_plans_executed,
        }
        metrics_path.write_text(json.dumps(entry, indent=2))
        self._log(f"[final_eval] metrics → {metrics_path}")

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
