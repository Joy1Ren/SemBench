from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cost_model_agent import PlanCostEstimate, ResultsStore

# Mirrors palimpzest constants used in org_SampleBasedCostModel._compute_naive_plan_cost.
# Loaded lazily so this module works even without palimpzest installed.
try:
    from palimpzest.constants import (
        MODEL_CARDS,
        NAIVE_EST_FILTER_SELECTIVITY,
        NAIVE_EST_NUM_INPUT_TOKENS,
        NAIVE_EST_NUM_OUTPUT_TOKENS,
        LOCAL_SCAN_TIME_PER_KB,
        NAIVE_BYTES_PER_RECORD,
    )
except ImportError:
    MODEL_CARDS = {}
    NAIVE_EST_FILTER_SELECTIVITY = 0.5
    NAIVE_EST_NUM_INPUT_TOKENS = 1000
    NAIVE_EST_NUM_OUTPUT_TOKENS = 100
    LOCAL_SCAN_TIME_PER_KB = 1 / (500 * 1024)
    NAIVE_BYTES_PER_RECORD = 1024

# LLM filter outputs only TRUE/FALSE (~1.25 tokens); mirrors LLMFilter.naive_cost_estimates.
_NAIVE_OUTPUT_TOKENS_FILTER = 1.25


def _is_filter(op_type: str) -> bool:
    return "filter" in op_type.lower()


def _is_scan(op_type: str) -> bool:
    return "scan" in op_type.lower()


class SampleBasedCostModel:
    """Mirrors org_SampleBasedCostModel but works with CostModelAgent's ResultsStore
    and duck-typed plan operators instead of PZ SentinelPlanStats / PhysicalOperators.

    Fitting (_compute_op_stats):
        Groups ResultsStore rows by op_id (≡ full_op_id in the original).
        Computes mean cost_usd and latency_s per record, and selectivity for filter ops.
        This directly mirrors the original's groupby(full_op_id) inner loop.

    Naive fallback (_naive_op_stats):
        Mirrors _compute_naive_plan_cost: uses MODEL_CARDS for LLM operators (same
        token-count assumptions as LLMFilter/LLMConvert.naive_cost_estimates), and
        fixed constants for non-LLM operators.

    Inference (estimate_plan):
        Walks operators in execution order, looks up per-op stats (sampled or naive),
        accumulates total_cost and total_time by multiplying per-record values by
        running input cardinality, and propagates output cardinality via selectivity.
        This is the whole-plan version of the original's per-operator __call__.
    """

    def __init__(self, results: ResultsStore):
        self._results = results  # live reference — re-fitted on each estimate_plan call
        self.op_id_to_stats: dict[str, dict] = self._compute_op_stats(results)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _compute_op_stats(self, results: ResultsStore) -> dict[str, dict]:
        """Mirrors org_SampleBasedCostModel._compute_operator_stats.

        Groups rows by op_id (= params_id) and computes per-record cost, latency,
        and selectivity.  Two row formats are supported:

          Aggregated  — row has a "num_records" key (produced by pipeline.run() /
                        ExecutePlanTool).  cost_usd and latency_s are execution
                        totals; we divide by the summed num_records to recover
                        per-record values.  Filter selectivity uses "num_passed".

          Per-record  — no "num_records" key (produced by ExecuteSubplanTool or
                        hand-crafted tests).  Each row is one (operator, record)
                        invocation; the mean is taken directly.  Filter selectivity
                        uses the truthy "output" field.
        """
        if not results.rows:
            return {}

        groups: dict[str, list[dict]] = defaultdict(list)
        for row in results.rows:
            key = row.get("op_id") or row.get("params_id")
            groups[key].append(row)

        op_id_to_stats: dict[str, dict] = {}
        for op_id, rows in groups.items():
            op_type = rows[0].get("op_type", "")

            if "num_records" in rows[0]:
                # Aggregated format: divide totals by record count to get per-record stats.
                total_records = sum(r.get("num_records", 1) or 1 for r in rows)
                mean_cost = sum(r.get("cost_usd", 0.0) or 0.0 for r in rows) / total_records
                mean_time = sum(r.get("latency_s", 0.0) or 0.0 for r in rows) / total_records
                if _is_filter(op_type):
                    total_passed = sum(r.get("num_passed", 0) for r in rows)
                    selectivity = total_passed / total_records if total_records > 0 else NAIVE_EST_FILTER_SELECTIVITY
                    if selectivity == 1.0:
                        selectivity -= 1e-3
                else:
                    selectivity = 1.0
            else:
                # Per-record format: each row is one (operator, input-record) invocation.
                mean_cost = sum(r.get("cost_usd", 0.0) or 0.0 for r in rows) / len(rows)
                mean_time = sum(r.get("latency_s", 0.0) or 0.0 for r in rows) / len(rows)
                if _is_filter(op_type):
                    passed = sum(1 for r in rows if r.get("output"))
                    selectivity = passed / len(rows)
                    if selectivity == 1.0:
                        selectivity -= 1e-3
                else:
                    selectivity = 1.0

            op_id_to_stats[op_id] = {
                "cost": mean_cost,
                "time": mean_time,
                "selectivity": selectivity,
            }

        return op_id_to_stats

    # ------------------------------------------------------------------
    # Naive fallback (mirrors _compute_naive_plan_cost)
    # ------------------------------------------------------------------

    def _naive_op_stats(self, op: Any) -> dict:
        """Mirrors _compute_naive_plan_cost without calling op.naive_cost_estimates().

        Uses MODEL_CARDS for LLM operators (same formulas as LLMFilter/LLMConvert
        .naive_cost_estimates in palimpzest). Uses fixed constants for non-LLM ops.
        """
        from cost_model_agent import get_op_model, get_op_type
        op_type = get_op_type(op)
        model = get_op_model(op)

        if _is_scan(op_type):
            # Mirrors MarshalAndScanDataOp.naive_cost_estimates:
            #   time_per_record = LOCAL_SCAN_TIME_PER_KB * (NAIVE_BYTES_PER_RECORD / 1024)
            return {
                "cost": 0.0,
                "time": LOCAL_SCAN_TIME_PER_KB * (NAIVE_BYTES_PER_RECORD / 1024),
                "selectivity": 1.0,
            }

        if model is None:
            # Non-LLM operator (UDF filter or UDF convert).
            # Mirrors NonLLMFilter/NonLLMConvert.naive_cost_estimates: cost=0, time=1ms.
            return {
                "cost": 0.0,
                "time": 0.001,
                "selectivity": NAIVE_EST_FILTER_SELECTIVITY if _is_filter(op_type) else 1.0,
            }

        # LLM operator — use model card exactly as LLMFilter/LLMConvert.naive_cost_estimates.
        card = MODEL_CARDS.get(model)
        if card is not None:
            est_out_tokens = _NAIVE_OUTPUT_TOKENS_FILTER if _is_filter(op_type) else NAIVE_EST_NUM_OUTPUT_TOKENS
            time_per_record = card["seconds_per_output_token"] * est_out_tokens
            cost_per_record = (
                card["usd_per_input_token"] * NAIVE_EST_NUM_INPUT_TOKENS
                + card["usd_per_output_token"] * est_out_tokens
            )
            return {
                "cost": cost_per_record,
                "time": time_per_record,
                "selectivity": NAIVE_EST_FILTER_SELECTIVITY if _is_filter(op_type) else 1.0,
            }

        # Model not in MODEL_CARDS — use coarse defaults keyed by operator kind.
        return {
            "cost": 0.001 if _is_filter(op_type) else 0.003,
            "time": 0.9 if _is_filter(op_type) else 1.5,
            "selectivity": NAIVE_EST_FILTER_SELECTIVITY if _is_filter(op_type) else 1.0,
        }

    def _lookup(self, op: Any) -> dict:
        """Level 1: sampled op_id stats. Level 2 (fallback): naive estimates."""
        from cost_model_agent import get_op_id
        op_id = get_op_id(op)
        if op_id in self.op_id_to_stats:
            return self.op_id_to_stats[op_id]
        return self._naive_op_stats(op)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def estimate_plan(self, plan: Any) -> PlanCostEstimate:
        """Whole-plan version of the original's per-operator __call__.

        Walks operators in execution order. For each operator, looks up
        cost_per_record and time_per_record (sampled or naive), multiplies by
        the current input cardinality, then updates cardinality via selectivity.

        Initial cardinality comes from the first operator's cardinality attribute.
        """
        from cost_model_agent import get_op_id, get_op_type, iter_operators, PlanCostEstimate

        self.op_id_to_stats = self._compute_op_stats(self._results)
        cardinality = self._source_cardinality(plan)
        total_cost = 0.0
        total_time = 0.0
        details: dict[str, dict] = {}

        for op in iter_operators(plan):
            op_id = get_op_id(op)
            op_type = get_op_type(op)

            if _is_scan(op_type):
                # Scan sets the initial cardinality; its cost is 0 and time is negligible.
                # Cardinality does not change (selectivity = 1.0).
                stats = self._lookup(op)
                scan_time = stats["time"] * cardinality
                total_time += scan_time
                details[op_id] = {"cost": 0.0, "time": round(scan_time, 6), "cardinality_in": cardinality}
                continue

            stats = self._lookup(op)
            op_cost = stats["cost"] * cardinality
            op_time = stats["time"] * cardinality
            total_cost += op_cost
            total_time += op_time
            details[op_id] = {
                "cost": round(op_cost, 6),
                "time": round(op_time, 4),
                "cardinality_in": cardinality,
                "selectivity": stats["selectivity"],
                "source": "sampled" if get_op_id(op) in self.op_id_to_stats else "naive",
            }
            cardinality *= stats["selectivity"]

        return PlanCostEstimate(cost=total_cost, time=total_time, details=details)

    def _source_cardinality(self, plan: Any) -> float:
        """Returns the input cardinality of the plan's source dataset.

        For PhysicalPipeline the dataframe is stored as `_df`, so we use its
        length directly — this works for both pre-built (write_plan) and
        post-execution pipelines and avoids the 100-record fallback that would
        make estimates ~20x too low on a 2000-row dataset.

        Falls back to a `cardinality` attribute on the first operator (for
        demo/duck-typed plans), then to 100 if nothing else is available.
        """
        from cost_model_agent import iter_operators
        if hasattr(plan, "_df"):
            return float(len(plan._df))
        ops = iter_operators(plan)
        return float(getattr(ops[0], "cardinality", 100)) if ops else 100.0


# class org_SampleBasedCostModel:
#     """
#     """
#     def __init__(
#         self,
#         sentinel_plan_stats: SentinelPlanStats | None = None,
#         verbose: bool = False,
#         exp_name: str | None = None,
#     ):
#         # store verbose argument
#         self.verbose = verbose

#         # store experiment name if one is provided
#         self.exp_name = exp_name

#         # construct cost, time, quality, and selectivity matrices for each operator set;
#         self.operator_to_stats = self._compute_operator_stats(sentinel_plan_stats)
#         self.costed_full_op_ids = None if self.operator_to_stats is None else set([
#             full_op_id
#             for _, full_op_id_to_stats in self.operator_to_stats.items()
#             for full_op_id in full_op_id_to_stats
#         ])

#         # if there is a logical operator with no samples; add all of its op ids to costed_full_op_ids;
#         # this will lead to the cost model applying the naive cost estimates for all physical op ids
#         # in this logical operator (I think?)
#         # TODO

#         logger.info(f"Initialized SampleBasedCostModel with verbose={self.verbose}")
#         logger.debug(f"Initialized SampleBasedCostModel with params: {self.__dict__}")

#     def get_costed_full_op_ids(self):
#         return self.costed_full_op_ids

#     def _compute_operator_stats(self, sentinel_plan_stats: SentinelPlanStats | None) -> dict:
#         logger.debug("Computing operator statistics")
#         # if no stats are provided, simply return None
#         if sentinel_plan_stats is None:
#             return None

#         # flatten the nested dictionary of execution data and pull out fields relevant to cost estimation
#         execution_record_op_stats = []
#         for unique_logical_op_id, full_op_id_to_op_stats in sentinel_plan_stats.operator_stats.items():
#             logger.debug(f"Computing operator statistics for logical_op_id: {unique_logical_op_id}")
#             # flatten the execution data into a list of RecordOpStats
#             op_set_execution_data = [
#                 record_op_stats
#                 for _, op_stats in full_op_id_to_op_stats.items()
#                 for record_op_stats in op_stats.record_op_stats_lst
#             ]

#             # add entries from execution data into matrices
#             for record_op_stats in op_set_execution_data:
#                 record_op_stats_dict = {
#                     "unique_logical_op_id": unique_logical_op_id,
#                     "full_op_id": record_op_stats.full_op_id,
#                     "record_id": record_op_stats.record_id,
#                     "record_parent_ids": record_op_stats.record_parent_ids,
#                     "cost_per_record": record_op_stats.cost_per_record,
#                     "time_per_record": record_op_stats.time_per_record,
#                     "quality": record_op_stats.quality,
#                     "passed_operator": record_op_stats.passed_operator,
#                     "source_indices": record_op_stats.record_source_indices,
#                     "op_details": record_op_stats.op_details,
#                     "answer": record_op_stats.answer,
#                     "op_name": record_op_stats.op_name,
#                 }
#                 execution_record_op_stats.append(record_op_stats_dict)

#         # convert flattened execution data into dataframe
#         operator_stats_df = pd.DataFrame(execution_record_op_stats)

#         # for each full_op_id, compute its average cost_per_record, time_per_record, selectivity, and quality
#         operator_to_stats = {}
#         for unique_logical_op_id, logical_op_df in operator_stats_df.groupby("unique_logical_op_id"):
#             logger.debug(f"Computing operator statistics for unique_logical_op_id: {unique_logical_op_id}")
#             operator_to_stats[unique_logical_op_id] = {}

#             for full_op_id, physical_op_df in logical_op_df.groupby("full_op_id"):
#                 # compute the number of input records processed by this operator; use source_indices for scan operator(s)
#                 num_source_records = (
#                     physical_op_df.record_parent_ids.apply(tuple).nunique()
#                     if not physical_op_df.record_parent_ids.isna().all()
#                     else physical_op_df.source_indices.apply(tuple).nunique()
#                 )

#                 # compute selectivity; for filters this may be 1.0 on smalle samples;
#                 # always put something slightly less than 1.0 to ensure that filters are pushed down when possible
#                 selectivity = physical_op_df.passed_operator.sum() / num_source_records
#                 op_name = physical_op_df.op_name.iloc[0].lower()
#                 if selectivity == 1.0 and "filter" in op_name:
#                     selectivity -= 1e-3

#                 # compute quality; if all qualities are None then this will be NaN
#                 quality = physical_op_df.quality.mean()

#                 # set operator stats for this physical operator
#                 operator_to_stats[unique_logical_op_id][full_op_id] = {
#                     "cost": physical_op_df.cost_per_record.mean(),
#                     "time": physical_op_df.time_per_record.mean(),
#                     "quality": 1.0 if pd.isna(quality) else quality,
#                     "selectivity": selectivity,
#                 }

#         logger.debug(f"Done computing operator statistics for {len(operator_to_stats)} operators!")
#         return operator_to_stats

#     def _compute_naive_plan_cost(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None, right_source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
#         # get identifier for operator which is unique within sentinel plan but consistent across sentinels
#         full_op_id = operator.get_full_op_id()
#         logger.debug(f"Calling __call__ for {str(operator)} with full_op_id: {full_op_id}")

#         # initialize estimates of operator metrics based on naive (but sometimes precise) logic
#         if isinstance(operator, MarshalAndScanDataOp):
#             # get handle to scan operator and pre-compute its size (number of records)
#             datasource_len = len(operator.datasource)

#             source_op_estimates = OperatorCostEstimates(
#                 cardinality=datasource_len,
#                 time_per_record=0.0,
#                 cost_per_record=0.0,
#                 quality=1.0,
#             )

#             op_estimates = operator.naive_cost_estimates(source_op_estimates, input_record_size_in_bytes=NAIVE_BYTES_PER_RECORD)

#         elif isinstance(operator, ContextScanOp):
#             source_op_estimates = OperatorCostEstimates(
#                 cardinality=1.0,
#                 time_per_record=0.0,
#                 cost_per_record=0.0,
#                 quality=1.0,
#             )

#             op_estimates = operator.naive_cost_estimates(source_op_estimates)

#         elif isinstance(operator, JoinOp):
#             op_estimates = operator.naive_cost_estimates(source_op_estimates, right_source_op_estimates)

#         else:
#             op_estimates = operator.naive_cost_estimates(source_op_estimates)

#         # compute estimates for this operator
#         est_input_cardinality = (
#             source_op_estimates.cardinality * right_source_op_estimates.cardinality
#             if isinstance(operator, JoinOp)
#             else source_op_estimates.cardinality
#         )
#         op_time = op_estimates.time_per_record * est_input_cardinality
#         op_cost = op_estimates.cost_per_record * est_input_cardinality
#         op_quality = op_estimates.quality

#         # create and return PlanCost object for this op's statistics
#         op_plan_cost = PlanCost(
#             cost=op_cost,
#             time=op_time,
#             quality=op_quality,
#             op_estimates=op_estimates,
#         )
#         logger.debug(f"Done calling __call__ for {str(operator)} with full_op_id: {full_op_id}")
#         logger.debug(f"Plan cost: {op_plan_cost}")

#         return op_plan_cost

#     def __call__(self, operator: PhysicalOperator, source_op_estimates: OperatorCostEstimates | None = None, right_source_op_estimates: OperatorCostEstimates | None = None) -> PlanCost:
#         # for non-sentinel execution, we use naive estimates
#         full_op_id = operator.get_full_op_id()
#         unique_logical_op_id = operator.unique_logical_op_id
#         if self.operator_to_stats is None or unique_logical_op_id not in self.operator_to_stats:
#             return self._compute_naive_plan_cost(operator, source_op_estimates, right_source_op_estimates)

#         # NOTE: some physical operators may not have any sample execution data in this cost model;
#         #       these physical operators are filtered out of the Optimizer, thus we can assume that
#         #       we will have execution data for each operator passed into __call__; nevertheless, we
#         #       still perform a sanity check
#         # look up physical and logical op ids associated with this physical operator
#         physical_op_to_stats = self.operator_to_stats.get(unique_logical_op_id)
#         assert physical_op_to_stats is not None, f"No execution data for logical operator: {str(operator)}"
#         assert physical_op_to_stats.get(full_op_id) is not None, f"No execution data for physical operator: {str(operator)}"
#         logger.debug(f"Calling __call__ for {str(operator)}")

#         # look up stats for this operation
#         est_cost_per_record = self.operator_to_stats[unique_logical_op_id][full_op_id]["cost"]
#         est_time_per_record = self.operator_to_stats[unique_logical_op_id][full_op_id]["time"]
#         est_quality = self.operator_to_stats[unique_logical_op_id][full_op_id]["quality"]
#         est_selectivity = self.operator_to_stats[unique_logical_op_id][full_op_id]["selectivity"]

#         # create source_op_estimates for scan operators if they are not provided
#         if isinstance(operator, ScanPhysicalOp):
#             # get handle to scan operator and pre-compute its size (number of records)
#             datasource_len = len(operator.datasource)

#             source_op_estimates = OperatorCostEstimates(
#                 cardinality=datasource_len,
#                 time_per_record=0.0,
#                 cost_per_record=0.0,
#                 quality=1.0,
#             )

#         # generate new set of OperatorCostEstimates
#         est_input_cardinality = (
#             source_op_estimates.cardinality * right_source_op_estimates.cardinality
#             if isinstance(operator, JoinOp)
#             else source_op_estimates.cardinality
#         )
#         op_estimates = OperatorCostEstimates(
#             cardinality=est_selectivity * est_input_cardinality,
#             time_per_record=est_time_per_record,
#             cost_per_record=est_cost_per_record,
#             quality=est_quality,
#         )

#         # compute estimates for this operator
#         op_time = op_estimates.time_per_record * est_input_cardinality
#         op_cost = op_estimates.cost_per_record * est_input_cardinality
#         op_quality = op_estimates.quality

#         # construct and return op estimates
#         plan_cost = PlanCost(cost=op_cost, time=op_time, quality=op_quality, op_estimates=op_estimates)
#         logger.debug(f"Done calling __call__ for {str(operator)}")
#         logger.debug(f"Plan cost: {plan_cost}")
#         return plan_cost