"""Physical pipeline: chain PZ physical operators directly with per-operator model selection."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.lib.schemas import ImageFilepath, _create_pickleable_model, create_schema_from_df
from palimpzest.core.models import ExecutionStats
from palimpzest.query.operators.aggregate import ApplyGroupByOp
from palimpzest.query.operators.convert import LLMConvertBonded, NonLLMConvert
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.join import NestedLoopsJoin
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.project import ProjectOp

NUM_SAMPLES = 10
SUBSET_SEED = 42


def _str_to_pz_model(model_str: str) -> Model:
    """Map an OpenRouter/PZ model string to pz.Model enum value.

    Accepts exact matches or unambiguous prefix matches so callers can pass
    short names like "openai/o4-mini" for "openai/o4-mini-2025-04-16".
    """
    for m in Model:
        if m.value == model_str:
            return m
    # Prefix fallback: "openai/o4-mini" → "openai/o4-mini-2025-04-16"
    matches = [m for m in Model if m.value.startswith(model_str)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous oracle model {model_str!r}: matches "
            f"{[m.value for m in matches]}. Use a more specific string."
        )
    raise ValueError(
        f"Unknown oracle model: {model_str!r}. "
        f"Available: {[m.value for m in Model]}"
    )


@dataclass
class SubsetExecutionContext:
    """Execution context returned by run_subset(), used by QualityEvaluator."""
    sampled_records: list[dict]               # sampled left-side input rows as plain dicts
    output_records: list[dict]                # final pipeline output rows as plain dicts
    per_sem_op_info: dict[int, dict]          # stage_idx → {op_name, op_type, attributes, samples}
    has_join: bool
    right_sampled_records: list[dict] | None  # right-side sampled rows if join, else None


def _resolve_reasoning_effort(model: Model) -> str | None:
    """Mirror PZ optimizer logic: disable thinking tokens for reasoning models by default."""
    if model is None or not model.is_reasoning_model():
        return None
    if model.is_vertex_model() or model.is_google_model():
        if model in (getattr(Model, 'GEMINI_2_5_PRO', None), getattr(Model, 'GOOGLE_GEMINI_2_5_PRO', None)):
            return "low"
        return "disable"
    if model.is_openai_model():
        if model == getattr(Model, 'o4_MINI', None):
            return "low"
        return "minimal"
    return None


def _compute_op_id(op_type: str, params: dict) -> str:
    """Stable 10-char hex id derived from op_type and id params."""
    payload = json.dumps({"op_type": op_type, **{k: str(v) for k, v in params.items()}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _make_schema(field_defs: dict):
    # Use palimpzest's pickleable/cached schema so all schemas live in the same
    # registry and work correctly with union_schemas, from_parent, etc.
    safe_defs = {
        k: (ann, fi if isinstance(fi, FieldInfo) else FieldInfo(default=None))
        for k, (ann, fi) in field_defs.items()
    }
    return _create_pickleable_model(safe_defs)


# ------------------------------------------------------------------
# Operator base class and subclasses
# ------------------------------------------------------------------

class Operator:
    """Base class for PhysicalPipeline operators.

    Each subclass sets class-level `stage_type` (execution dispatch key) and
    `op_type` (human-readable name used in stats), and populates instance
    attributes `attributes`, `params_id`, and `_pz_op` in its __init__.
    """

    stage_type: str  # filter | convert | project | limit | groupby | join
    op_type: str     # sem_filter | sem_map | sem_join | filter | project | limit | groupby
    attributes: dict
    params_id: str #10-char hex from op_type and attributes

    def __init__(self):
        self.logical_op_id: str | None = None

    def __call__(self, *args, **kwargs):
        return self._pz_op(*args, **kwargs)

    def __str__(self) -> str:
        if not self.attributes:
            return self.op_type
        attr_str = "\n".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.op_type}({attr_str})"


class SemFilter(Operator):
    """LLM-based row filter."""
    stage_type = "filter"
    op_type = "sem_filter"

    def __init__(self, condition: str, model: Model, schema, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        self._pz_op = LLMFilter(
            model=model,
            filter=Filter(filter_condition=condition),
            output_schema=schema,
            input_schema=schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.attributes = {"condition": condition, "model": model.value, "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, {"model": model.value})
        # self.params_id = _compute_op_id(self.op_type, self.attributes)


def _has_image_field(schema, field_names: list[str] | None) -> bool:
    """Return True if any of the given fields (or all fields if None) in schema have ImageFilepath type."""
    fields_to_check = field_names if field_names is not None else list(schema.model_fields)
    for name in fields_to_check:
        fi = schema.model_fields.get(name)
        if fi is not None and fi.annotation in (ImageFilepath, ImageFilepath | None):
            return True
    return False


class SemMap(Operator):
    """LLM-based column derivation."""
    stage_type = "convert"
    op_type = "sem_map"

    def __init__(self, cols: list[dict], model: Model, input_schema, output_schema, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        is_image = _has_image_field(input_schema, depends_on)
        if is_image:
            prompt_strategy = PromptStrategy.COT_QA_IMAGE_NO_REASONING if (model.is_reasoning_model() and eff in (None, "minimal", "low", "disable")) else PromptStrategy.COT_QA_IMAGE
        else:
            prompt_strategy = PromptStrategy.COT_QA_NO_REASONING if (model.is_reasoning_model() and eff in (None, "minimal", "low", "disable")) else PromptStrategy.COT_QA
        self._pz_op = LLMConvertBonded(
            model=model,
            prompt_strategy=prompt_strategy,
            output_schema=output_schema,
            input_schema=input_schema,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self._cols_full = cols  # preserved for make_oracle_copy
        self.attributes = {"model": model.value, "cols": sorted(col["name"] for col in cols), "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, self.attributes)


class SemJoin(Operator):
    """LLM-based nested-loops join."""
    stage_type = "join"
    op_type = "sem_join"

    def __init__(
        self,
        other: "PhysicalPipeline",
        condition: str,
        model: Model,
        join_parallelism: int,
        depends_on: list[str] | None,
        schema,
        reasoning_effort_override: str | None = None,
    ):
        super().__init__()
        self.model = model
        eff = reasoning_effort_override if reasoning_effort_override is not None else _resolve_reasoning_effort(model)
        self._pz_op = NestedLoopsJoin(
            model=model,
            condition=condition,
            output_schema=schema,
            input_schema=schema,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            reasoning_effort=eff,
        )
        self._pz_op.model = model
        self.depends_on = depends_on
        self.other = other
        self.attributes = {"condition": condition, "model": model.value, "join_parallelism": join_parallelism, "depends_on": depends_on}
        self.params_id = _compute_op_id(self.op_type, {"model": model.value})
        # self.params_id = _compute_op_id(self.op_type, self.attributes)


class ExactFilter(Operator):
    """Exact (non-LLM) row filter."""
    stage_type = "filter"
    op_type = "filter"

    def __init__(self, fn: Callable, schema):
        super().__init__()
        self._pz_op = NonLLMFilter(
            filter=Filter(filter_fn=fn),
            output_schema=schema,
            input_schema=schema,
        )
        self._fn = fn  # preserved for make_oracle_copy
        try:
            fn_src = inspect.getsource(fn).strip()
        except (OSError, TypeError):
            fn_src = repr(fn)
        self.attributes = {"condition": fn_src}
        self.params_id = _compute_op_id(self.op_type, self.attributes)


class Map(Operator):
    """Deterministic (non-LLM) column derivation via UDF."""
    stage_type = "convert"
    op_type = "map"

    def __init__(self, udf: Callable, cols: list[dict], input_schema, output_schema):
        super().__init__()
        self._pz_op = NonLLMConvert(
            udf=udf,
            output_schema=output_schema,
            input_schema=input_schema,
        )
        self._udf = udf          # preserved for make_oracle_copy
        self._cols_full = cols   # preserved for make_oracle_copy
        try:
            fn_src = inspect.getsource(udf).strip()
        except (OSError, TypeError):
            fn_src = repr(udf)
        col_names = sorted(col["name"] for col in cols)
        self.attributes = {"cols": col_names, "udf": fn_src}
        self.params_id = _compute_op_id(self.op_type, {"cols": col_names, "udf": fn_src})


class Project(Operator):
    """Column projection."""
    stage_type = "project"
    op_type = "project"

    def __init__(self, cols: list[str], input_schema, output_schema):
        super().__init__()
        self._pz_op = ProjectOp(
            project_cols=cols,
            output_schema=output_schema,
            input_schema=input_schema,
        )
        self.attributes = {"project_cols": sorted(cols)}
        self.params_id = _compute_op_id(self.op_type, self.attributes)


class Limit(Operator):
    """Row limit."""
    stage_type = "limit"
    op_type = "limit"

    def __init__(self, n: int, schema):
        super().__init__()
        self._pz_op = LimitScanOp(
            limit=n,
            output_schema=schema,
            input_schema=schema,
        )
        self.n = n
        self.attributes = {"limit": n}
        self.params_id = _compute_op_id(self.op_type, self.attributes)


class GroupBy(Operator):
    """Group-by aggregation (barrier operator)."""
    stage_type = "groupby"
    op_type = "groupby"

    def __init__(
        self,
        group_by_fields: list[str],
        agg_funcs: list[str],
        agg_fields: list[str],
        input_schema,
    ):
        super().__init__()
        sig = GroupBySig(
            group_by_fields=group_by_fields,
            agg_funcs=agg_funcs,
            agg_fields=agg_fields,
        )
        self.output_schema = sig.output_schema()
        self._pz_op = ApplyGroupByOp(
            group_by_sig=sig,
            output_schema=self.output_schema,
            input_schema=input_schema,
        )
        self.attributes = {
            "group_by_fields": sorted(group_by_fields),
            "agg_pairs": sorted(zip(agg_funcs, agg_fields)),
        }
        self.params_id = _compute_op_id(self.op_type, self.attributes)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class PhysicalPipeline:
    """
    Fluent interface for chaining PZ physical operators with per-operator model selection.

    Semantic (require model):     sem_filter, sem_map, sem_join
    Non-semantic (no model):      filter, project, limit, groupby

    Usage:
        pipeline = PhysicalPipeline("plan1", "Reviews.csv", self.load_data("Reviews.csv"))
        pipeline.sem_filter("the review is clearly positive", model=pz.Model.CLAUDE_3_5_HAIKU)
        pipeline.project(["reviewId"])
        pipeline.limit(5)
        return pipeline.run()
    """

    def __init__(self, plan_name, source: str, data: pd.DataFrame, max_workers: int = 20):
        self.plan_name = plan_name
        self._source = source
        self._df = data
        self._max_workers = max_workers
        self._initial_schema = create_schema_from_df(data)
        # extract (annotation, FieldInfo) tuples for schema evolution
        self._defs = {
            name: (field.annotation, field)
            for name, field in self._initial_schema.model_fields.items()
        }
        self._schema = self._initial_schema
        self._ops: list[Operator] = []
        self._last_exec_stats = None      # populated by run(); used by to_results_row()

    def _flat_ops(self) -> list[tuple[str, "Operator"]]:
        """All operators in topological order with display names.
        Right-branch operators of a SemJoin appear before the join itself."""
        result = []
        main_idx = 1
        for op in self._ops:
            if isinstance(op, SemJoin):
                for name, right_op in op.other._flat_ops():
                    result.append((name, right_op))
            result.append((f"{self.plan_name}_op{main_idx}", op))
            main_idx += 1
        return result

    def __str__(self) -> str:
        flat = self._flat_ops()
        if not flat:
            return "EmptyPipeline"
        op_names = [name for name, _ in flat]
        pretty_text = f"Plan: {self.plan_name}: ({', '.join(op_names)})\nOperators in topological order:"
        for name, op in flat:
            pretty_text += f"\n ====={name}=====\n   {str(op)}"
        return pretty_text


    # ------------------------------------------------------------------
    # Semantic operators
    # ------------------------------------------------------------------

    def sem_filter(self, condition: str, model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None) -> "PhysicalPipeline":
        """LLM-based row filter. Keeps records where condition is true."""
        self._ops.append(SemFilter(condition=condition, model=model, schema=self._schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override))
        return self

    def sem_map(self, cols: list[dict], model: Model, depends_on: list[str] | None = None, reasoning_effort_override: str | None = None) -> "PhysicalPipeline":
        """
        LLM-based column derivation. Adds new fields to each record.

        cols: list of {"name": str, "type": type, "description": str}
            description is passed as FieldInfo and used by the LLM generator.
        """
        new_defs = {}
        for col in cols:
            name = col["name"]
            if name in self._defs:
                # Preserve the existing annotation to avoid union_schemas type mismatch
                ann = self._defs[name][0]
            else:
                ann = Optional[col.get("type", Any)]
            new_defs[name] = (ann, FieldInfo(default=None, description=col["description"]))
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(SemMap(cols=cols, model=model, input_schema=self._schema, output_schema=output_schema, depends_on=depends_on, reasoning_effort_override=reasoning_effort_override))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    def sem_join(
        self,
        other: "PhysicalPipeline",
        condition: str,
        model: Model,
        join_parallelism: int = 20,
        depends_on: list[str] | None = None,
        reasoning_effort_override: str | None = None,
    ) -> "PhysicalPipeline":
        """
        LLM-based join. Keeps pairs of (self record, other record) where condition holds.
        Self fields win on name collision with other fields.
        """
        merged_defs = {**other._defs, **self._defs}
        joined_schema = _make_schema(merged_defs)
        self._ops.append(SemJoin(
            other=other,
            condition=condition,
            model=model,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            schema=joined_schema,
            reasoning_effort_override=reasoning_effort_override,
        ))
        self._defs = merged_defs
        self._schema = joined_schema
        return self

    # ------------------------------------------------------------------
    # Non-semantic operators
    # ------------------------------------------------------------------

    def filter(self, fn: Callable[[dict], bool]) -> "PhysicalPipeline":
        """Exact (non-LLM) row filter. fn receives a record dict and returns bool."""
        self._ops.append(ExactFilter(fn=fn, schema=self._schema))
        return self

    def map(self, udf: Callable[[dict], dict], cols: list[dict]) -> "PhysicalPipeline":
        """
        Deterministic (non-LLM) column derivation via UDF.

        udf: callable receiving a record dict, returning a dict with the new field values.
        cols: list of {"name": str, "type": type, "description": str}
            The type annotation is preserved in the schema — use pz.ImageFilepath to
            signal image columns so PZ's prompt factory encodes them for LLM calls.
        """
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col.get("description", "")),
            )
            for col in cols
        }
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(Map(udf=udf, cols=cols, input_schema=self._schema, output_schema=output_schema))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    def project(self, cols: list[str]) -> "PhysicalPipeline":
        """Select a subset of columns."""
        projected_defs = {k: self._defs[k] for k in cols if k in self._defs}
        projected_schema = _make_schema(projected_defs)
        self._ops.append(Project(cols=cols, input_schema=self._schema, output_schema=projected_schema))
        self._defs = projected_defs
        self._schema = projected_schema
        return self

    def limit(self, n: int) -> "PhysicalPipeline":
        """Keep at most n records."""
        self._ops.append(Limit(n=n, schema=self._schema))
        return self

    def groupby(
        self,
        group_by_fields: list[str],
        agg_funcs: list[str],
        agg_fields: list[str],
    ) -> "PhysicalPipeline":
        """
        Group records and aggregate.

        group_by_fields: fields to group on
        agg_funcs:       per-agg-field function — "count" or "average"
        agg_fields:      fields to aggregate (parallel to agg_funcs)

        Output fields are the group_by_fields plus "<func>(<field>)" columns.
        """
        op = GroupBy(
            group_by_fields=group_by_fields,
            agg_funcs=agg_funcs,
            agg_fields=agg_fields,
            input_schema=self._schema,
        )
        self._ops.append(op)
        self._defs = {k: (f.annotation, f) for k, f in op.output_schema.model_fields.items()}
        self._schema = op.output_schema
        return self

    # ------------------------------------------------------------------
    # Plan introspection (compatible with cost_model_agent inspect_plan)
    # ------------------------------------------------------------------

    def __iter__(self):
        """Yield all operators in topological order (right-branch ops before their join)."""
        for _, op in self._flat_ops():
            yield op

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _build_initial_records(self, df: pd.DataFrame) -> list[DataRecord]:
        """Build DataRecord objects from a DataFrame using this pipeline's initial schema."""
        records: list[DataRecord] = []
        for i in range(len(df)):
            row = df.iloc[i].to_dict()
            dr = DataRecord(schema=self._initial_schema, source_indices=f"{self._source}-{i}")
            for k, v in row.items():
                setattr(dr, k, v)
            records.append(dr)
        return records

    def _execute(self) -> tuple[list[DataRecord], list, dict, dict]:
        """Execute using all rows in self._df. Delegates to _execute_core."""
        initial_records = self._build_initial_records(self._df)
        return self._execute_core(initial_records, max_samples=5)

    def _execute_core(self, initial_records: list[DataRecord], max_samples: int, skip_limit: bool = False) -> tuple[list[DataRecord], list, dict, dict]:
        from concurrent.futures import wait as fut_wait
        _POLL_INTERVAL = 0.3

        all_record_op_stats = []

        # Assign logical_op_ids and build stage_map for per-operator stat attribution.
        # _flat_ops() includes right-branch operators so all ops appear in per_op_list.
        stage_map: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            op.logical_op_id = op.params_id
            op._pz_op.logical_op_id = op.params_id
            stage_map[flat_idx] = {
                "logical_op_id": op.params_id,
                "op_name": op_name,
                "op_type": op.op_type,
                "attributes": op.attributes,
                "params_id": op.params_id,
            }

        n_ops = len(self._ops)
        if n_ops == 0:
            return initial_records, all_record_op_stats, stage_map, {}

        _MAX_SAMPLES = max_samples
        # (input_record, Future) for per-record ops; keyed by id(op) so right-branch ops
        # are captured alongside main-pipeline ops.
        op_sample_pairs: dict[int, list] = {}  # id(op) -> [(input_dr, future)]

        # Main pipeline queues
        input_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        future_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        input_queues[0] = initial_records[:]
        output_records: list[DataRecord] = []
        limit_val = None if skip_limit else next((op.n for op in self._ops if op.stage_type == "limit"), None)
        # batch_size for filter/convert/project: limit value when present, else None (submit all)
        batch_size = limit_val

        # Precompute which joins have a downstream limit op (enables incremental join, matching PZ).
        join_has_downstream_limit = {
            i: any(op.stage_type == "limit" for op in self._ops[i + 1:])
            for i, op in enumerate(self._ops) if op.stage_type == "join"
        }
        _join_call_counts: dict[int, int] = {}  # diagnostic: how many times each join fires

        # Initialize right-pipeline state for each join stage.
        # Each tick the right pipeline advances by batch_size records, matching PZ's scan batching.
        right_state: dict[int, dict] = {}
        for i, op in enumerate(self._ops):
            if op.stage_type != "join":
                continue
            other = op.other
            r_initial: list[DataRecord] = other._build_initial_records(other._df)
            for r_op in other._ops:
                if r_op.logical_op_id is None:
                    r_op.logical_op_id = r_op.params_id
                r_op._pz_op.logical_op_id = r_op.logical_op_id
            n_r = len(other._ops)
            right_state[i] = {
                "other": other,
                "n": n_r,
                "initial": r_initial,
                "n_fed": 0,                # how many right initial records fed into r_iq[0] so far
                "iq": {j: [] for j in range(n_r)},
                "fq": {j: [] for j in range(n_r)},
                "pending": [],             # right records ready for this tick's join call (incremental)
                "all_right": [],           # all right records produced so far (barrier join)
                "done": n_r == 0 and len(r_initial) == 0,
            }

        def any_pending() -> bool:
            return any(input_queues[i] or future_queues[i] for i in range(n_ops))

        def upstream_done(stage_idx: int) -> bool:
            return all(not input_queues[i] and not future_queues[i] for i in range(stage_idx))

        def drain(fq_dict: dict, key: int, timeout: float = _POLL_INTERVAL) -> list[DataRecord]:
            if not fq_dict.get(key):
                return []
            done, not_done = fut_wait(fq_dict[key], timeout=timeout)
            fq_dict[key] = list(not_done)
            passing = []
            for future in done:
                try:
                    result = future.result()
                    all_record_op_stats.extend(result.record_op_stats)
                    passing.extend(dr for dr in result.data_records if dr.passed_operator)
                except Exception as _exc:
                    print(f"[pipeline] record-level operator error (skipping record): {type(_exc).__name__}: {_exc}")
            return passing

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            while any_pending():

                # Advance main pipeline
                for stage_idx, op in enumerate(self._ops):

                    # Step 1: harvest upstream futures into this operator's input queue
                    if stage_idx > 0:
                        input_queues[stage_idx].extend(drain(future_queues, stage_idx - 1))

                    # Step 2: final operator — drain own future queue BEFORE submission
                    # to collect previous-tick results (matches PZ's _process_future_results ordering).
                    if stage_idx == n_ops - 1:
                        output_records.extend(drain(future_queues, stage_idx))

                    # Step 3: submit work — operator type determines the path
                    if op.stage_type == "limit" and input_queues[stage_idx]:
                        if skip_limit:
                            # Subset execution: pass all records through without truncation
                            to_pass = input_queues[stage_idx][:]
                            input_queues[stage_idx] = []
                        else:
                            space = limit_val - len(output_records)
                            to_pass = input_queues[stage_idx][:max(space, 0)]
                            input_queues[stage_idx] = input_queues[stage_idx][len(to_pass):]
                        if stage_idx == n_ops - 1:
                            output_records.extend(to_pass)
                            # Drain own future queue after limit to mirror PZ's second harvest,
                            # ensuring output_records is up to date before the early-stop check.
                            output_records.extend(drain(future_queues, stage_idx))
                        else:
                            input_queues[stage_idx + 1].extend(to_pass)

                    elif op.stage_type == "groupby" and upstream_done(stage_idx) and input_queues[stage_idx]:
                        # Aggregate barrier: wait for all upstream, then submit as single batch future
                        batch = input_queues[stage_idx][:]
                        input_queues[stage_idx].clear()
                        future_queues[stage_idx].append(executor.submit(op, batch))

                    elif op.stage_type == "join":
                        join_pz_op = op._pz_op
                        other = op.other
                        rs = right_state[stage_idx]
                        n_r = rs["n"]
                        r_iq = rs["iq"]
                        r_fq = rs["fq"]
                        has_dl = join_has_downstream_limit[stage_idx]

                        # Advance right pipeline by one tick (mirrors PZ's scan batching).
                        # Feed the next batch_size right records through the right ops each tick,
                        # so the join sees at most batch_size left × batch_size right per call.
                        if not rs["done"]:
                            n_remaining = len(rs["initial"]) - rs["n_fed"]
                            n_feed = min(batch_size if batch_size is not None else n_remaining, n_remaining)
                            new_right = rs["initial"][rs["n_fed"]:rs["n_fed"] + n_feed]
                            rs["n_fed"] += n_feed
                            if n_r == 0:
                                # No right ops: records are immediately ready
                                rs["pending"].extend(new_right)
                                rs["all_right"].extend(new_right)
                            elif new_right:
                                r_iq[0].extend(new_right)

                            # Advance right ops: harvest upstream → drain final → submit
                            for r_stage, r_op in enumerate(other._ops):
                                if r_stage > 0:
                                    r_iq[r_stage].extend(drain(r_fq, r_stage - 1, timeout=0))
                                if r_stage == n_r - 1:
                                    new_ready = drain(r_fq, r_stage)
                                    rs["pending"].extend(new_ready)
                                    rs["all_right"].extend(new_ready)
                                if r_op.stage_type in ("filter", "convert", "project") and r_iq.get(r_stage):
                                    r_bs = batch_size if batch_size is not None else len(r_iq[r_stage])
                                    r_batch = r_iq[r_stage][:r_bs]
                                    r_iq[r_stage] = r_iq[r_stage][r_bs:]
                                    for rec in r_batch:
                                        f = executor.submit(r_op, rec)
                                        r_fq[r_stage].append(f)
                                        r_op_key = id(r_op)
                                        if len(op_sample_pairs.get(r_op_key, [])) < _MAX_SAMPLES:
                                            op_sample_pairs.setdefault(r_op_key, []).append((rec, f))
                                elif r_op.stage_type == "groupby":
                                    r_upstream_done = all(not r_iq.get(j) and not r_fq.get(j) for j in range(r_stage))
                                    if r_upstream_done and r_iq.get(r_stage):
                                        r_gb = r_iq[r_stage][:]
                                        r_iq[r_stage].clear()
                                        r_fq[r_stage].append(executor.submit(r_op, r_gb))

                            rs["done"] = (
                                rs["n_fed"] >= len(rs["initial"])
                                and not any(r_iq.get(j) or r_fq.get(j) for j in range(n_r))
                            )

                        # Fire join
                        if has_dl:
                            # Incremental: pair this tick's batch_size left with this tick's right output.
                            # Matches PZ's join_has_downstream_limit_op path: fires as soon as both
                            # sides have records without waiting for upstream to finish.
                            left_batch = input_queues[stage_idx][:batch_size]
                            input_queues[stage_idx] = input_queues[stage_idx][len(left_batch):]
                            right_batch = rs["pending"][:]
                            rs["pending"] = []
                            if left_batch and right_batch:
                                _join_call_counts[stage_idx] = _join_call_counts.get(stage_idx, 0) + 1
                                prev_l = len(join_pz_op._left_input_records)
                                prev_r = len(join_pz_op._right_input_records)
                                pairs = (len(left_batch) * len(right_batch)
                                         + len(left_batch) * prev_r
                                         + prev_l * len(right_batch))
                                # print(f"[join call #{_join_call_counts[stage_idx]} stage={stage_idx}] "
                                #       f"new_l={len(left_batch)} new_r={len(right_batch)} "
                                #       f"prev_l={prev_l} prev_r={prev_r} => {pairs} pairs")
                                result_set, _ = join_pz_op(left_batch, right_batch)
                                if result_set is not None:
                                    future_queues[stage_idx].append(
                                        executor.submit(lambda rset=result_set: rset)
                                    )
                        elif upstream_done(stage_idx) and rs["done"] and input_queues[stage_idx]:
                            # Barrier: wait for all left upstream + right pipeline done,
                            # then join all left × all right in one call.
                            left_batch = input_queues[stage_idx][:]
                            input_queues[stage_idx].clear()
                            _join_call_counts[stage_idx] = _join_call_counts.get(stage_idx, 0) + 1
                            prev_l = len(join_pz_op._left_input_records)
                            prev_r = len(join_pz_op._right_input_records)
                            pairs = (len(left_batch) * len(rs["all_right"])
                                     + len(left_batch) * prev_r
                                     + prev_l * len(rs["all_right"]))
                            # print(f"[join call #{_join_call_counts[stage_idx]} stage={stage_idx} BARRIER] "
                            #       f"left={len(left_batch)} right={len(rs['all_right'])} "
                            #       f"prev_l={prev_l} prev_r={prev_r} => {pairs} pairs")
                            result_set, _ = join_pz_op(left_batch, rs["all_right"])
                            if result_set is not None:
                                future_queues[stage_idx].append(
                                    executor.submit(lambda rset=result_set: rset)
                                )

                    elif input_queues[stage_idx] and op.stage_type != "groupby":
                        # filter / convert / project: submit up to batch_size records per tick
                        # (batch_size=None means submit all ready records)
                        # groupby is a barrier and must never be dispatched per-record
                        batch = input_queues[stage_idx][:batch_size]
                        input_queues[stage_idx] = [] if batch_size is None else input_queues[stage_idx][batch_size:]
                        op_key = id(op)
                        for r in batch:
                            f = executor.submit(op, r)
                            future_queues[stage_idx].append(f)
                            if len(op_sample_pairs.get(op_key, [])) < _MAX_SAMPLES:
                                op_sample_pairs.setdefault(op_key, []).append((r, f))

                # Early stop once limit is satisfied
                if limit_val is not None and len(output_records) >= limit_val:
                    break

        # if _join_call_counts:
        #     print(f"[join summary] calls per stage: {_join_call_counts}")

        # Resolve per-record sample futures → (input_dr, output_dr | None) pairs.
        # Join and groupby stages have no samples (can't link inputs to outputs).
        # Keyed by id(op) so both main-pipeline and right-branch ops are included.
        op_samples: dict[int, list[tuple]] = {}
        for op_key, pairs in op_sample_pairs.items():
            op_samples[op_key] = []
            for inp, f in pairs:
                try:
                    result = f.result()
                    out_recs = [dr for dr in result.data_records if dr.passed_operator]
                    op_samples[op_key].append((inp, out_recs[0] if out_recs else None))
                except Exception:
                    op_samples[op_key].append((inp, None))

        return output_records[:limit_val] if limit_val is not None else output_records, all_record_op_stats, stage_map, op_samples

    def run(self) -> tuple[DataRecordCollection, list[dict], dict]:
        """Execute the pipeline and return (records, per_op_list, plan_dict).

        per_op_list: one dict per operator in pipeline order, each with keys:
            op_id, cost_usd, latency_s, input_tokens, output_tokens,
            num_records, num_passed, op_type, attributes, params_id

        num_records: number of input records the operator processed.
        num_passed:  number that passed (equal to num_records for non-filter ops).
        op_id:       alias for params_id; used by SampleBasedCostModel for lookup.

        plan_dict keys (each value is a single scalar):
            cost_usd, latency_s, input_tokens, output_tokens
        """
        def _fmt_record(dr: DataRecord | None) -> str:
            if dr is None:
                return "∅"
            schema_cls: type[BaseModel] = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            return repr({k: getattr(dr, k, None) for k in schema_cls.model_fields})

        start = time.time()
        records, all_record_op_stats, stage_map, op_samples = self._execute()
        elapsed = time.time() - start

        # Aggregate per-operator stats keyed by logical_op_id (= params_id).
        # num_records counts input invocations; num_passed counts records that
        # passed the operator (for selectivity estimation in filter ops).
        op_agg: dict[str, dict] = {}
        for r in all_record_op_stats:
            lid = r.logical_op_id
            if lid not in op_agg:
                op_agg[lid] = {
                    "cost_usd": 0.0, "latency_s": 0.0,
                    "input_tokens": 0, "output_tokens": 0,
                    "num_records": 0, "num_passed": 0,
                }
            op_agg[lid]["cost_usd"] += r.cost_per_record
            op_agg[lid]["latency_s"] += r.time_per_record
            op_agg[lid]["input_tokens"] += int(r.total_input_tokens)
            op_agg[lid]["output_tokens"] += int(r.total_output_tokens)
            op_agg[lid]["num_records"] += 1
            op_agg[lid]["num_passed"] += int(bool(getattr(r, "passed_operator", True)))

        _zero = {
            "cost_usd": 0.0, "latency_s": 0.0,
            "input_tokens": 0, "output_tokens": 0,
            "num_records": 0, "num_passed": 0,
        }
        per_op_list: list[dict] = []
        for stage_idx in sorted(stage_map):
            meta = stage_map[stage_idx]
            s = op_agg.get(meta["logical_op_id"], _zero)
            per_op_list.append({
                "op_name": meta["op_name"],
                "op_type": meta["op_type"],
                "op_id": meta["params_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        # for entry in per_op_list:
        #     print(
        #         f"[pipeline] {entry['op_name']} ({entry['op_type']}): "
        #         f"{entry['num_records']} in, {entry['num_passed']} passed",
        #         flush=True,
        #     )

        total_cost = sum(r.cost_per_record for r in all_record_op_stats)
        total_input_tokens = int(sum(r.total_input_tokens for r in all_record_op_stats))
        total_output_tokens = int(sum(r.total_output_tokens for r in all_record_op_stats))

        op_id_to_name = {id(op): name for name, op in self._flat_ops()}
        op_samples_dict: dict[str, list[dict]] = {}
        for op_key, pairs in op_samples.items():
            if not pairs:
                continue
            op_name = op_id_to_name.get(op_key)
            if op_name is None:
                continue
            op_samples_dict[op_name] = [
                {"input": _fmt_record(inp), "output": _fmt_record(out)}
                for inp, out in pairs
            ]

        plan_dict: dict = {
            "plan_name": self.plan_name,
            "latency_s": elapsed,
            "cost_usd": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "plan_str": str(self),
            "op_samples": op_samples_dict,
        }

        exec_stats = ExecutionStats(
            plan_execution_time=elapsed,
            total_execution_time=elapsed,
            plan_execution_cost=total_cost,
            total_execution_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )
        self._last_exec_stats = exec_stats
        return DataRecordCollection(records, execution_stats=exec_stats), per_op_list, plan_dict

    # ------------------------------------------------------------------
    # Subset execution
    # ------------------------------------------------------------------

    def run_subset(
        self,
        num_samples: int = NUM_SAMPLES,
        seed: int = SUBSET_SEED,
        subset_cache_path: str | None = None,
    ) -> tuple[list[dict], SubsetExecutionContext]:
        """Execute on a random sample of num_samples records.

        If subset_cache_path is given and the file already exists, the cached
        rows are loaded instead of resampling.  If the file does not exist yet
        the sample is drawn as usual and written to that path.

        Returns (per_op_list, SubsetExecutionContext).
        per_op_list: same schema as run() but stats are from the subset.
        SubsetExecutionContext: sampled records, output records, and per-LLM-op samples
          for use by QualityEvaluator.
        """
        import os as _os
        if subset_cache_path is not None and _os.path.exists(subset_cache_path):
            sampled_df = pd.read_csv(subset_cache_path).reset_index(drop=True)
            for col in sampled_df.columns:
                try:
                    sampled_df[col] = sampled_df[col].astype(self._df[col].dtype)
                except (ValueError, TypeError):
                    pass
        else:
            n = min(num_samples, len(self._df))
            sampled_df = self._df.sample(n=n, random_state=seed).reset_index(drop=True)
            if subset_cache_path is not None:
                _os.makedirs(_os.path.dirname(_os.path.abspath(subset_cache_path)), exist_ok=True)
                sampled_df.to_csv(subset_cache_path, index=False)
        sampled_records_list = sampled_df.to_dict(orient="records")

        has_join = any(op.stage_type == "join" for op in self._ops)
        right_df_backup: dict[int, pd.DataFrame] = {}
        right_sampled_records: list[dict] | None = None

        if has_join:
            right_sampled_records = []
            for op in self._ops:
                if op.stage_type == "join":
                    other = op.other
                    oid = id(other)
                    if oid not in right_df_backup:
                        right_df_backup[oid] = other._df
                        n_r = min(num_samples, len(other._df))
                        other._df = other._df.sample(n=n_r, random_state=seed).reset_index(drop=True)
                    right_sampled_records.extend(other._df.to_dict(orient="records"))

        try:
            initial_records = self._build_initial_records(sampled_df)
            records, all_record_op_stats, stage_map, op_samples = self._execute_core(
                initial_records, max_samples=num_samples, skip_limit=True
            )
        finally:
            for op in self._ops:
                if op.stage_type == "join":
                    oid = id(op.other)
                    if oid in right_df_backup:
                        op.other._df = right_df_backup[oid]

        # Build per_op_list (same aggregation logic as run())
        op_agg: dict[str, dict] = {}
        _zero: dict = {"cost_usd": 0.0, "latency_s": 0.0, "input_tokens": 0,
                       "output_tokens": 0, "num_records": 0, "num_passed": 0}
        for r in all_record_op_stats:
            lid = r.logical_op_id
            if lid not in op_agg:
                op_agg[lid] = dict(_zero)
            op_agg[lid]["cost_usd"] += r.cost_per_record
            op_agg[lid]["latency_s"] += r.time_per_record
            op_agg[lid]["input_tokens"] += int(r.total_input_tokens)
            op_agg[lid]["output_tokens"] += int(r.total_output_tokens)
            op_agg[lid]["num_records"] += 1
            op_agg[lid]["num_passed"] += int(bool(getattr(r, "passed_operator", True)))

        per_op_list: list[dict] = []
        for stage_idx in sorted(stage_map):
            meta = stage_map[stage_idx]
            s = op_agg.get(meta["logical_op_id"], _zero)
            per_op_list.append({
                "op_name": meta["op_name"],
                "op_type": meta["op_type"],
                "op_id": meta["params_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        # Convert output DataRecords to plain dicts
        def _dr_to_dict(dr: DataRecord) -> dict:
            schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            return {k: getattr(dr, k, None) for k in schema_cls.model_fields}

        output_records = [_dr_to_dict(dr) for dr in records]

        # Collect per-LLM-op info: keyed by flat_idx from _flat_ops() so that
        # op names are consistent with per_op_list (which also uses _flat_ops() names).
        # op_samples is keyed by id(op), so right-branch ops are included automatically.
        per_sem_op_info: dict[int, dict] = {}
        for flat_idx, (op_name, op) in enumerate(self._flat_ops()):
            if "model" in op.attributes:
                per_sem_op_info[flat_idx] = {
                    "op_name": op_name,
                    "op_type": op.op_type,
                    "attributes": op.attributes,
                    "samples": op_samples.get(id(op), []),
                }

        context = SubsetExecutionContext(
            sampled_records=sampled_records_list,
            output_records=output_records,
            per_sem_op_info=per_sem_op_info,
            has_join=has_join,
            right_sampled_records=right_sampled_records,
        )
        return per_op_list, context

    # ------------------------------------------------------------------
    # Oracle copy (for QualityEvaluator)
    # ------------------------------------------------------------------

    def make_oracle_copy(self, oracle_model: "Model | str", oracle_reasoning_effort: str | None = None) -> "PhysicalPipeline":
        """Return a copy of this pipeline with all LLM operators using oracle_model.

        oracle_model may be a pz.Model enum (preferred) or a model string
        that will be resolved via _str_to_pz_model.
        """
        if isinstance(oracle_model, str):
            oracle_model = _str_to_pz_model(oracle_model)
        oracle = PhysicalPipeline(
            plan_name=self.plan_name + "_oracle",
            source=self._source,
            data=self._df,
            max_workers=self._max_workers,
        )
        for op in self._ops:
            if isinstance(op, SemFilter):
                oracle.sem_filter(
                    condition=op.attributes["condition"],
                    model=oracle_model,
                    depends_on=getattr(op, "depends_on", None),
                    reasoning_effort_override=oracle_reasoning_effort,
                )
            elif isinstance(op, SemMap):
                cols = getattr(op, "_cols_full", None)
                if cols:
                    oracle.sem_map(
                        cols=cols,
                        model=oracle_model,
                        depends_on=getattr(op, "depends_on", None),
                        reasoning_effort_override=oracle_reasoning_effort,
                    )
            elif isinstance(op, SemJoin):
                oracle_other = op.other.make_oracle_copy(oracle_model, oracle_reasoning_effort)
                oracle.sem_join(
                    other=oracle_other,
                    condition=op.attributes["condition"],
                    model=oracle_model,
                    join_parallelism=op.attributes.get("join_parallelism", 20),
                    depends_on=getattr(op, "depends_on", None),
                    reasoning_effort_override=oracle_reasoning_effort,
                )
            elif isinstance(op, ExactFilter):
                fn = getattr(op, "_fn", None)
                if fn is not None:
                    oracle.filter(fn)
            elif isinstance(op, Map):
                udf = getattr(op, "_udf", None)
                cols = getattr(op, "_cols_full", None)
                if udf is not None and cols:
                    oracle.map(udf, cols)
            elif isinstance(op, Project):
                oracle.project(op.attributes["project_cols"])
            elif isinstance(op, Limit):
                oracle.limit(op.attributes["limit"])
            elif isinstance(op, GroupBy):
                group_by_fields = op.attributes["group_by_fields"]
                agg_pairs = list(op.attributes["agg_pairs"])
                agg_funcs = [p[0] for p in agg_pairs]
                agg_fields = [p[1] for p in agg_pairs]
                oracle.groupby(group_by_fields, agg_funcs, agg_fields)
        return oracle

    # def to_results_row(self, plan_id: str) -> dict:
    #     """Return a ResultsStore-compatible dict from the most recent run().

    #     Captures plan-level totals only (not per-operator breakdowns).
    #     Canonical columns match the ResultsStore schema in cost_model_agent:
    #         plan_id, cost_usd, latency_s, input_tokens, output_tokens
    #     """
    #     if self._last_exec_stats is None:
    #         raise RuntimeError("Call run() before to_results_row()")
    #     s = self._last_exec_stats
    #     return {
    #         "plan_id": plan_id,
    #         "cost_usd": s.total_execution_cost,
    #         "latency_s": s.plan_execution_time,
    #         "input_tokens": int(s.total_input_tokens),
    #         "output_tokens": int(s.total_output_tokens),
    #     }
