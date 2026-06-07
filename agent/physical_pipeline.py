"""Physical pipeline: chain PZ physical operators directly with per-operator model selection."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

import pandas as pd

from pydantic import create_model
from pydantic.fields import FieldInfo

from palimpzest.constants import Model
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.groupbysig import GroupBySig
from palimpzest.core.elements.records import DataRecord, DataRecordCollection
from palimpzest.core.lib.schemas import create_schema_from_df
from palimpzest.core.models import ExecutionStats
from palimpzest.query.operators.aggregate import ApplyGroupByOp
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter, NonLLMFilter
from palimpzest.query.operators.join import NestedLoopsJoin
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


def _make_schema(field_defs: dict):
    return create_model("Schema", **field_defs)


class PhysicalPipeline:
    """
    Fluent interface for chaining PZ physical operators with per-operator model selection.

    Semantic (require model):     sem_filter, sem_map, sem_join
    Non-semantic (no model):      filter, project, limit, groupby

    Usage:
        pipeline = PhysicalPipeline("Reviews.csv", self.load_data("Reviews.csv"))
        pipeline.sem_filter("the review is clearly positive", model=pz.Model.CLAUDE_3_5_HAIKU)
        pipeline.project(["reviewId"])
        pipeline.limit(5)
        return pipeline.run()
    """

    def __init__(self, source_name: str, data: pd.DataFrame, max_workers: int = 20):
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
        self._ops: list[tuple] = []

    # ------------------------------------------------------------------
    # Semantic operators
    # ------------------------------------------------------------------

    def sem_filter(self, condition: str, model: Model) -> "PhysicalPipeline":
        """LLM-based row filter. Keeps records where condition is true."""
        reasoning_effort = _resolve_reasoning_effort(model)
        op = LLMFilter(
            model=model,
            filter=Filter(filter_condition=condition),
            output_schema=self._schema,
            input_schema=self._schema,
            reasoning_effort=reasoning_effort,
        )
        self._ops.append(("filter", op))
        return self

    def sem_map(self, cols: list[dict], model: Model) -> "PhysicalPipeline":
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
        reasoning_effort = _resolve_reasoning_effort(model)
        op = LLMConvertBonded(
            model=model,
            output_schema=output_schema,
            input_schema=self._schema,
            reasoning_effort=reasoning_effort,
        )
        self._ops.append(("convert", op))
        self._defs = {**self._defs, **new_defs}
        self._schema = output_schema
        return self

    def sem_join(
        self,
        other: "PhysicalPipeline",
        condition: str,
        model: Model,
        join_parallelism: int = 64,
        depends_on: list[str] | None = None,
    ) -> "PhysicalPipeline":
        """
        LLM-based join. Keeps pairs of (self record, other record) where condition holds.
        Self fields win on name collision with other fields.
        """
        merged_defs = {**other._defs, **self._defs}
        joined_schema = _make_schema(merged_defs)
        reasoning_effort = _resolve_reasoning_effort(model)
        op = NestedLoopsJoin(
            model=model,
            condition=condition,
            output_schema=joined_schema,
            input_schema=joined_schema,
            join_parallelism=join_parallelism,
            depends_on=depends_on,
            reasoning_effort=reasoning_effort,
        )
        self._ops.append(("join", (op, other)))
        self._defs = merged_defs
        self._schema = joined_schema
        return self

    # ------------------------------------------------------------------
    # Non-semantic operators
    # ------------------------------------------------------------------

    def filter(self, fn: Callable[[dict], bool]) -> "PhysicalPipeline":
        """Exact (non-LLM) row filter. fn receives a record dict and returns bool."""
        op = NonLLMFilter(
            filter=Filter(filter_fn=fn),
            output_schema=self._schema,
            input_schema=self._schema,
        )
        self._ops.append(("filter", op))
        return self

    def project(self, cols: list[str]) -> "PhysicalPipeline":
        """Select a subset of columns."""
        projected_defs = {k: self._defs[k] for k in cols if k in self._defs}
        projected_schema = _make_schema(projected_defs)
        op = ProjectOp(
            project_cols=cols,
            output_schema=projected_schema,
            input_schema=self._schema,
        )
        self._ops.append(("project", op))
        self._defs = projected_defs
        self._schema = projected_schema
        return self

    def limit(self, n: int) -> "PhysicalPipeline":
        """Keep at most n records."""
        self._ops.append(("limit", n))
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
        sig = GroupBySig(
            group_by_fields=group_by_fields,
            agg_funcs=agg_funcs,
            agg_fields=agg_fields,
        )
        output_schema = sig.output_schema()
        op = ApplyGroupByOp(
            group_by_sig=sig,
            output_schema=output_schema,
            input_schema=self._schema,
        )
        self._ops.append(("groupby", op))
        self._defs = {k: (Optional[Any], None) for k in output_schema.model_fields}
        self._schema = output_schema
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self) -> tuple[list[DataRecord], list]:
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

        # Set logical_op_ids (required by PZ operators before execution)
        for step_idx, (op_type, op_or_val) in enumerate(self._ops):
            if op_type != "limit":
                op = op_or_val[0] if op_type == "join" else op_or_val
                if op.logical_op_id is None:
                    op.logical_op_id = f"pp-{step_idx}-{op_type}"

        n_ops = len(self._ops)
        if n_ops == 0:
            return initial_records, all_record_op_stats

        # Main pipeline queues
        input_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        future_queues: dict[int, list] = {i: [] for i in range(n_ops)}
        input_queues[0] = initial_records[:]
        output_records: list[DataRecord] = []
        limit_val = next((n for t, n in self._ops if t == "limit"), None)
        # batch_size for filter/convert/project: limit value when present, else None (submit all)
        batch_size = limit_val

        # Precompute which joins have a downstream limit op (enables incremental join, matching PZ).
        join_has_downstream_limit = {
            i: any(t == "limit" for t, _ in self._ops[i + 1:])
            for i, (t, _) in enumerate(self._ops) if t == "join"
        }
        _join_call_counts: dict[int, int] = {}  # diagnostic: how many times each join fires

        # Initialize right-pipeline state for each join stage.
        # Each tick the right pipeline advances by batch_size records, matching PZ's scan batching.
        right_state: dict[int, dict] = {}
        for i, (op_type, op_or_val) in enumerate(self._ops):
            if op_type != "join":
                continue
            _, other = op_or_val
            r_initial: list[DataRecord] = []
            for j in range(len(other._df)):
                row = other._df.iloc[j].to_dict()
                dr = DataRecord(schema=other._initial_schema, source_indices=f"{other._source}-{j}")
                for k, v in row.items():
                    setattr(dr, k, v)
                r_initial.append(dr)
            for step_idx, (rt, rv) in enumerate(other._ops):
                if rt != "limit":
                    rop = rv[0] if rt == "join" else rv
                    if rop.logical_op_id is None:
                        rop.logical_op_id = f"pp-r{i}-{step_idx}-{rt}"
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
                for stage_idx, (op_type, op_or_val) in enumerate(self._ops):

                    # Step 1: harvest upstream futures into this operator's input queue
                    if stage_idx > 0:
                        input_queues[stage_idx].extend(drain(future_queues, stage_idx - 1))

                    # Step 2: final operator — drain own future queue BEFORE submission
                    # to collect previous-tick results (matches PZ's _process_future_results ordering).
                    if stage_idx == n_ops - 1:
                        output_records.extend(drain(future_queues, stage_idx))

                    # Step 3: submit work — operator type determines the path
                    if op_type == "limit" and input_queues[stage_idx]:
                        # Limit: pass records through without submitting to executor
                        space = op_or_val - len(output_records)
                        to_pass = input_queues[stage_idx][:max(space, 0)]
                        input_queues[stage_idx] = input_queues[stage_idx][len(to_pass):]
                        if stage_idx == n_ops - 1:
                            output_records.extend(to_pass)
                            # Drain own future queue after limit to mirror PZ's second harvest,
                            # ensuring output_records is up to date before the early-stop check.
                            output_records.extend(drain(future_queues, stage_idx))
                        else:
                            input_queues[stage_idx + 1].extend(to_pass)

                    elif op_type == "groupby" and upstream_done(stage_idx) and input_queues[stage_idx]:
                        # Aggregate barrier: wait for all upstream, then submit as single batch future
                        batch = input_queues[stage_idx][:]
                        input_queues[stage_idx].clear()
                        future_queues[stage_idx].append(executor.submit(op_or_val, batch))

                    elif op_type == "join":
                        join_op, other = op_or_val
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
                            for r_stage, (r_type, r_op_or_val) in enumerate(other._ops):
                                if r_stage > 0:
                                    r_iq[r_stage].extend(drain(r_fq, r_stage - 1, timeout=0))
                                if r_stage == n_r - 1:
                                    new_ready = drain(r_fq, r_stage)
                                    rs["pending"].extend(new_ready)
                                    rs["all_right"].extend(new_ready)
                                if r_type in ("filter", "convert", "project") and r_iq.get(r_stage):
                                    r_bs = batch_size if batch_size is not None else len(r_iq[r_stage])
                                    r_batch = r_iq[r_stage][:r_bs]
                                    r_iq[r_stage] = r_iq[r_stage][r_bs:]
                                    for rec in r_batch:
                                        r_fq[r_stage].append(executor.submit(r_op_or_val, rec))
                                elif r_type == "groupby":
                                    r_upstream_done = all(not r_iq.get(j) and not r_fq.get(j) for j in range(r_stage))
                                    if r_upstream_done and r_iq.get(r_stage):
                                        r_gb = r_iq[r_stage][:]
                                        r_iq[r_stage].clear()
                                        r_fq[r_stage].append(executor.submit(r_op_or_val, r_gb))

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
                                prev_l = len(join_op._left_input_records)
                                prev_r = len(join_op._right_input_records)
                                pairs = (len(left_batch) * len(right_batch)
                                         + len(left_batch) * prev_r
                                         + prev_l * len(right_batch))
                                print(f"[join call #{_join_call_counts[stage_idx]} stage={stage_idx}] "
                                      f"new_l={len(left_batch)} new_r={len(right_batch)} "
                                      f"prev_l={prev_l} prev_r={prev_r} => {pairs} pairs")
                                result_set, _ = join_op(left_batch, right_batch)
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
                            prev_l = len(join_op._left_input_records)
                            prev_r = len(join_op._right_input_records)
                            pairs = (len(left_batch) * len(rs["all_right"])
                                     + len(left_batch) * prev_r
                                     + prev_l * len(rs["all_right"]))
                            print(f"[join call #{_join_call_counts[stage_idx]} stage={stage_idx} BARRIER] "
                                  f"left={len(left_batch)} right={len(rs['all_right'])} "
                                  f"prev_l={prev_l} prev_r={prev_r} => {pairs} pairs")
                            result_set, _ = join_op(left_batch, rs["all_right"])
                            if result_set is not None:
                                future_queues[stage_idx].append(
                                    executor.submit(lambda rset=result_set: rset)
                                )

                    elif input_queues[stage_idx]:
                        # filter / convert / project: submit up to batch_size records per tick
                        # (batch_size=None means submit all ready records)
                        batch = input_queues[stage_idx][:batch_size]
                        input_queues[stage_idx] = [] if batch_size is None else input_queues[stage_idx][batch_size:]
                        for r in batch:
                            future_queues[stage_idx].append(executor.submit(op_or_val, r))

                # Early stop once limit is satisfied
                if limit_val is not None and len(output_records) >= limit_val:
                    break

        if _join_call_counts:
            print(f"[join summary] calls per stage: {_join_call_counts}")
        return output_records[:limit_val] if limit_val is not None else output_records, all_record_op_stats

    def run(self) -> DataRecordCollection:
        """Execute the pipeline and return a DataRecordCollection with execution stats."""
        start = time.time()
        records, all_record_op_stats = self._execute()
        elapsed = time.time() - start

        total_cost = sum(r.cost_per_record for r in all_record_op_stats)
        total_input_tokens = int(sum(r.total_input_tokens for r in all_record_op_stats))
        total_output_tokens = int(sum(r.total_output_tokens for r in all_record_op_stats))

        exec_stats = ExecutionStats(
            plan_execution_time=elapsed,
            total_execution_time=elapsed,
            plan_execution_cost=total_cost,
            total_execution_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_tokens=total_input_tokens + total_output_tokens,
        )

        return DataRecordCollection(records, execution_stats=exec_stats)
