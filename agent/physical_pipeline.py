"""Physical pipeline: chain PZ physical operators directly with per-operator model selection."""
from __future__ import annotations

import hashlib
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import pandas as pd

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.lib.schemas import _create_pickleable_model, create_schema_from_df
from palimpzest.core.models import ExecutionStats
from palimpzest.query.operators.aggregate import ApplyGroupByOp
from palimpzest.query.operators.logical import GroupByAggregate
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.join import NestedLoopsJoin
from palimpzest.query.operators.limit import LimitScanOp
from palimpzest.query.operators.project import ProjectOp


def _resolve_reasoning_effort(model: Model) -> str | None:
    """Mirror PZ optimizer logic: disable thinking tokens for reasoning models by default."""
    if model is None or not model.is_reasoning_model():
        return None
    if model.is_vertex_model() or model.is_google_model():
        if model in (Model.GEMINI_2_5_PRO, getattr(Model, 'GOOGLE_GEMINI_2_5_PRO', None)):
            return "low"
        return "disable"
    if model.is_openai_model():
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

    def __init__(self, condition: str, model: Model, schema, depends_on: list[str] | None = None):
        super().__init__()
        self.model = model
        self._pz_op = LLMFilter(
            model=model,
            filter=Filter(filter_condition=condition),
            output_schema=schema,
            input_schema=schema,
            depends_on=depends_on,
            reasoning_effort=_resolve_reasoning_effort(model),
        )
        self._pz_op.model = model
        self.attributes = {"condition": condition, "model": model.value}
        self.params_id = _compute_op_id(self.op_type, {"model": model.value})
        # self.params_id = _compute_op_id(self.op_type, self.attributes)


class SemMap(Operator):
    """LLM-based column derivation."""
    stage_type = "convert"
    op_type = "sem_map"

    def __init__(self, cols: list[dict], model: Model, input_schema, output_schema, depends_on: list[str] | None = None):
        super().__init__()
        self.model = model
        self._pz_op = LLMConvertBonded(
            model=model,
            output_schema=output_schema,
            input_schema=input_schema,
            depends_on=depends_on,
            reasoning_effort=_resolve_reasoning_effort(model),
        )
        self._pz_op.model = model
        self.attributes = {"model": model.value, "cols": sorted(col["name"] for col in cols)}
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
    ):
        super().__init__()
        self.model = model
        self._pz_op = NestedLoopsJoin(
            model=model,
            condition=condition,
            output_schema=schema,
            input_schema=schema,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            reasoning_effort=_resolve_reasoning_effort(model),
        )
        self._pz_op.model = model
        self.other = other
        self.attributes = {"condition": condition, "model": model.value, "join_parallelism": join_parallelism}
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
        try:
            fn_src = inspect.getsource(fn).strip()
        except (OSError, TypeError):
            fn_src = repr(fn)
        self.attributes = {"condition": fn_src}
        self.params_id = _compute_op_id(self.op_type, self.attributes)


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
        # operator = GroupByAggregate(input_schema=self.schema, output_schema=output_schema, group_by_sig=groupby)
        # return Dataset(sources=[self], operator=operator, schema=output_schema)
        # self._pz_op = GroupByAggregate(
        #     input_schema=input_schema,
        #     output_schema = self.output_schema,
        #     group_by_sig=sig
        # )
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

    def __init__(self, plan_name, source_name: str, data: pd.DataFrame, max_workers: int = 20):
        self.plan_name = plan_name
        self._source = source_name
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

    def __str__(self) -> str:
        if not self._ops:
            return "EmptyPipeline"
        op_names = [self.plan_name + f"-op{idx}" for idx in range (1, len(self._ops)+1)]
        pretty_text = f"Plan: {self.plan_name}: ({",".join(f"{name}" for name in op_names)})" 
        pretty_text += "Operators in topological order:"
        for op_name, op in zip(op_names, self._ops):
            pretty_text += f"""\n ====={op_name}===== \n   {str(op)}"""
        return pretty_text


    # ------------------------------------------------------------------
    # Semantic operators
    # ------------------------------------------------------------------

    def sem_filter(self, condition: str, model: Model, depends_on: list[str] | None = None) -> "PhysicalPipeline":
        """LLM-based row filter. Keeps records where condition is true."""
        self._ops.append(SemFilter(condition=condition, model=model, schema=self._schema, depends_on=depends_on))
        return self

    def sem_map(self, cols: list[dict], model: Model, depends_on: list[str] | None = None) -> "PhysicalPipeline":
        """
        LLM-based column derivation. Adds new fields to each record.

        cols: list of {"name": str, "type": type, "description": str}
            description is passed as FieldInfo and used by the LLM generator.
        """
        new_defs = {
            col["name"]: (
                Optional[col.get("type", Any)],
                FieldInfo(default=None, description=col["description"]),
            )
            for col in cols
        }
        output_schema = _make_schema({**self._defs, **new_defs})
        self._ops.append(SemMap(cols=cols, model=model, input_schema=self._schema, output_schema=output_schema, depends_on=depends_on))
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
        """Yield one Operator per stage in pipeline order."""
        yield from self._ops

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self) -> tuple[list[DataRecord], list, dict]:
        from concurrent.futures import wait as fut_wait
        _POLL_INTERVAL = 0.3

        all_record_op_stats = []

        # Build initial DataRecords for the left (main) pipeline
        initial_records: list[DataRecord] = []
        for i in range(len(self._df)):
            row = self._df.iloc[i].to_dict()
            dr = DataRecord(schema=self._initial_schema, source_indices=f"{self._source}-{i}")
            for k, v in row.items():
                setattr(dr, k, v)
            initial_records.append(dr)

        # Assign logical_op_ids and build stage_map for per-operator stat attribution
        stage_map: dict[int, dict] = {}
        for i, op in enumerate(self._ops):
            op.logical_op_id = op.params_id
            op._pz_op.logical_op_id = op.params_id
            stage_map[i] = {
                "logical_op_id": op.params_id,
                "op_type": op.op_type,
                "attributes": op.attributes,
                "params_id": op.params_id,
            }

        n_ops = len(self._ops)
        if n_ops == 0:
            return initial_records, all_record_op_stats, stage_map, {}

        _MAX_SAMPLES = 5
        # (input_record, Future) for per-record ops; (input_record, input_record) for limit
        op_sample_pairs: dict[int, list] = {i: [] for i in range(n_ops)}

        # Main pipeline queues
        input_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        future_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        input_queues[0] = initial_records[:]
        output_records: list[DataRecord] = []
        limit_val = next((op.n for op in self._ops if op.stage_type == "limit"), None)
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
            r_initial: list[DataRecord] = []
            for j in range(len(other._df)):
                row = other._df.iloc[j].to_dict()
                dr = DataRecord(schema=other._initial_schema, source_indices=f"{other._source}-{j}")
                for k, v in row.items():
                    setattr(dr, k, v)
                r_initial.append(dr)
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
                result = future.result()
                all_record_op_stats.extend(result.record_op_stats)
                passing.extend(dr for dr in result.data_records if dr.passed_operator)
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
                        # Limit: pass records through without submitting to executor
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
                                        r_fq[r_stage].append(executor.submit(r_op, rec))
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
                        for r in batch:
                            f = executor.submit(op, r)
                            future_queues[stage_idx].append(f)
                            if len(op_sample_pairs[stage_idx]) < _MAX_SAMPLES:
                                op_sample_pairs[stage_idx].append((r, f))

                # Early stop once limit is satisfied
                if limit_val is not None and len(output_records) >= limit_val:
                    break

        if _join_call_counts:
            print(f"[join summary] calls per stage: {_join_call_counts}")

        # Resolve per-record sample futures → (input_dr, output_dr | None) pairs.
        # Join and groupby stages have no samples (can't link inputs to outputs).
        op_samples: dict[int, list[tuple]] = {}
        for stage_idx, pairs in op_sample_pairs.items():
            op_samples[stage_idx] = []
            for inp, f in pairs:
                result = f.result()
                out_recs = [dr for dr in result.data_records if dr.passed_operator]
                op_samples[stage_idx].append((inp, out_recs[0] if out_recs else None))

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
                "op_name": f"{self.plan_name}_op{stage_idx + 1}",
                "op_type": meta["op_type"],
                "op_id": meta["params_id"],
                "latency_s": s["latency_s"],
                "cost_usd": s["cost_usd"],
                "input_tokens": s["input_tokens"],
                "output_tokens": s["output_tokens"],
                "num_records": s["num_records"],
                "num_passed": s["num_passed"],
            })

        total_cost = sum(r.cost_per_record for r in all_record_op_stats)
        total_input_tokens = int(sum(r.total_input_tokens for r in all_record_op_stats))
        total_output_tokens = int(sum(r.total_output_tokens for r in all_record_op_stats))

        sample_lines = []
        for stage_idx in sorted(op_samples):
            pairs = op_samples[stage_idx]
            if not pairs:
                continue
            op_name = f"{self.plan_name}_op{stage_idx + 1}"
            sample_lines.append(f"{op_name}:")
            for inp, out in pairs:
                sample_lines.append(f"  ({_fmt_record(inp)}, {_fmt_record(out)})")

        plan_dict: dict = {
            "plan_name": self.plan_name,
            "latency_s": elapsed,
            "cost_usd": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "plan_str": str(self),
            "op_samples": "\n".join(sample_lines),
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

    def to_results_row(self, plan_id: str) -> dict:
        """Return a ResultsStore-compatible dict from the most recent run().

        Captures plan-level totals only (not per-operator breakdowns).
        Canonical columns match the ResultsStore schema in cost_model_agent:
            plan_id, cost_usd, latency_s, input_tokens, output_tokens
        """
        if self._last_exec_stats is None:
            raise RuntimeError("Call run() before to_results_row()")
        s = self._last_exec_stats
        return {
            "plan_id": plan_id,
            "cost_usd": s.total_execution_cost,
            "latency_s": s.plan_execution_time,
            "input_tokens": int(s.total_input_tokens),
            "output_tokens": int(s.total_output_tokens),
        }
