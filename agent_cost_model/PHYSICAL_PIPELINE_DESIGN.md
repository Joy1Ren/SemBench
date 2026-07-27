# PhysicalPipeline — how it wraps and runs Palimpzest (PZ)

A design/architecture reference for [`agent/physical_pipeline.py`](../agent/physical_pipeline.py).
This document describes *how the pipeline is built and executed*, with special attention to how
`sem_join` runs. (For the separate latency write-up, see
[SEM_JOIN_LATENCY_FINDINGS.md](SEM_JOIN_LATENCY_FINDINGS.md).)

---

## 1. What PhysicalPipeline is

`PhysicalPipeline` is a thin **fluent wrapper around PZ's physical operators**. PZ normally builds
physical operators through its optimizer and hides per-operator construction options (which LLM
model, reasoning effort, prompt strategy, join parallelism, …). PhysicalPipeline lets you construct
those operators directly and chain them, so you have full control over each operator's design while
**reusing PZ's real operator implementations and execution semantics**.

Key idea: PhysicalPipeline **does not reimplement operators**. Every PhysicalPipeline operator holds a
real PZ physical operator in `self._pz_op` and simply *calls it*. What PhysicalPipeline *does*
reimplement is the **execution engine** (the scheduler loop), which is a faithful port of PZ's
`ParallelExecutionStrategy`.

```
PhysicalPipeline op   ─wraps→   PZ physical operator (self._pz_op)   ─calls→   Generator / LLM
        │
        └─ scheduled by PhysicalPipeline._execute_core  (a port of PZ ParallelExecutionStrategy)
```

---

## 2. Operator wrapping

Each `Operator` subclass sets two class fields and builds a PZ operator in `__init__`:

- `stage_type` — the scheduler dispatch key (`filter | convert | project | limit | groupby | join`).
- `op_type` — a human-readable name used in stats (`sem_filter | sem_map | sem_join | …`).
- `self._pz_op` — the real PZ `PhysicalOperator`.
- `self.attributes` / `self.params_id` — metadata + a stable 10-char id used for stat attribution.

`Operator.__call__` just forwards to the PZ op: `return self._pz_op(*args, **kwargs)`.

| PhysicalPipeline op | `stage_type` | Wrapped PZ operator | LLM? |
|---|---|---|---|
| `SemFilter` | `filter` | `LLMFilter` | yes |
| `SemMap` | `convert` | `LLMConvertBonded` | yes |
| `SemJoin` | `join` | `NestedLoopsJoin` | yes |
| `ExactFilter` | `filter` | `NonLLMFilter` | no |
| `Map` | `convert` | `NonLLMConvert` | no |
| `AddColSuffix` | `convert` | `NonLLMColSuffix` ‡ | no |
| `Join` | `join` | `NonLLMJoin` † | no |
| `Project` | `project` | `ProjectOp` | no |
| `Limit` | `limit` | `LimitScanOp` | no |
| `GroupBy` | `groupby` | `ApplyGroupByOp` | no |

† PZ ships no non-LLM join, so `NonLLMJoin` is defined in `physical_pipeline.py`: it subclasses PZ's
`JoinOp` and mirrors `NestedLoopsJoin`'s structure (nested loops over new/stored left×right, input
accumulation, `DataRecordSet` output) but decides each pair with a Python predicate
`fn(left_dict, right_dict) -> bool` instead of an LLM call — the join analogue of `NonLLMFilter` /
`NonLLMConvert`. It runs the predicate sequentially (no thread pool; there is no I/O to overlap).

‡ `NonLLMColSuffix` (also defined in `physical_pipeline.py`, subclassing PZ's `PhysicalOperator`)
renames every column `name -> f"{name}{suffix}"` — a deterministic 1:1 rename. `pipeline.add_col_suffix
(suffix)` is handy *before a join* to disambiguate columns so the two sides have no colliding names,
e.g. `left.add_col_suffix("_dish"); right.add_col_suffix("_table")` — the join then keeps both sets of
names as-is (no `_right` auto-suffix needed).

**Model & reasoning effort.** For LLM ops the constructor resolves a default reasoning effort with
`_resolve_reasoning_effort(model)` (mirrors PZ's optimizer: disable/minimize thinking tokens for
reasoning models unless overridden) and picks a prompt strategy (e.g. `COT_QA` vs `COT_QA_NO_REASONING`,
and the `_IMAGE` variants when an input field is an `ImageFilepath`).

**Schemas.** Schemas are pydantic models created through PZ's cached/pickleable factory
(`_create_pickleable_model` via `_make_schema`) so they live in PZ's schema registry and interoperate
with `union_schemas`, `from_parent`, `from_join_parents`, etc. Each builder method evolves
`self._defs` (the running `{name: (annotation, FieldInfo)}` map) and `self._schema`.

---

## 3. Building a plan (the fluent API)

```python
pipeline = PhysicalPipeline("plan1", "Reviews.csv", load_data("Reviews.csv"))
pipeline.sem_filter("the review is clearly positive", model=pz.Model.CLAUDE_3_5_HAIKU)
pipeline.project(["reviewId"])
pipeline.limit(5)
records, per_op_list, plan_dict = pipeline.run()
```

Semantic (need a `model`): `sem_filter`, `sem_map`, `sem_join`.
Non-semantic: `filter`, `map`, `add_col_suffix`, `join`, `project`, `limit`, `groupby`.

Each call appends an `Operator` to `self._ops` and updates the schema. `sem_map`/`map` add columns;
`project` restricts them; `sem_join` merges the two input schemas (see §6).

`_flat_ops()` returns operators in topological order **with the right branch of a join expanded before
the join itself** — this is what makes a join's right sub-pipeline appear in stats and iteration.

**Operator display names.** Each op gets a self-describing name `{plan_name}_op{N}_{op_type}` — e.g.
`p1_shoe_op3_sem_filter`, or `p2_right_op1_sem_map` for the right side of a join. It encodes which
pipeline the op belongs to (`plan_name`, which the plan author should set meaningfully — a join's right
branch uses *its* pipeline's name, e.g. `p2_right`), the op's position `N` in that pipeline, and the op
type. This name is what appears in `plan_str`, in each row of `per_op_list` (`op_name`), in
`op_samples`, and in `per_sem_op_quality`, so the agent can tell at a glance what each result refers to.
It is purely a display/dict-key string — plan↔oracle ops are matched by stage index and the cost model
keys by `params_id` (`op_id`), so the name never affects behavior.

---

## 4. The execution engine (`_execute_core`)

`run()` → `_execute()` → `_execute_core(initial_records, …)`. This is a port of PZ's
`ParallelExecutionStrategy._execute_plan`. Shared shape with PZ:

- **Per-stage queues.** `input_queues[i]` holds records waiting for operator *i*;
  `future_queues[i]` holds in-flight `Future`s produced by operator *i*.
- **One shared thread pool.** `ThreadPoolExecutor(max_workers=self._max_workers)` runs all per-record
  operators. (`sem_join` additionally uses its *own* pool — see §6.)
- **Initial records.** `_build_initial_records(df)` turns each DataFrame row into a PZ `DataRecord`
  (schema = the pipeline's initial schema, `source_indices = "<source>-<i>"`). These seed
  `input_queues[0]`.

The scheduler ticks until nothing is pending:

```
while any_pending():
    for stage_idx, op in enumerate(self._ops):
        1. harvest: input_queues[stage_idx] += drain(future_queues[stage_idx-1])   # completed upstream
        2. if final op: output_records += drain(future_queues[stage_idx])          # collect results
        3. submit work, dispatched on op.stage_type:
             limit    → pass through up to (limit - len(output)) records, enabling early exit
             groupby  → barrier: wait for all upstream, submit one batch future
             join     → advance right sub-pipeline, then fire the join   (see §6)
             else     → submit up to batch_size records as per-record futures
    if limit reached: break        # early stop
```

- **`drain(fq, key, timeout)`** waits up to `timeout` (default `_POLL_INTERVAL = 0.3s`, matching PZ's
  `PARALLEL_EXECUTION_SLEEP_INTERVAL_SECS`) for futures, moves finished results forward, and keeps only
  records with `passed_operator == True`.
- **`batch_size`** = the query's `limit` value when a `limit` exists, else `None` (submit everything).
  This mirrors PZ's `query_processor_factory` (`batch_size = limit`) and lets selective limits stop early.
- **Barrier operators.** `groupby` and a **limit-less join** wait for *all* upstream input to arrive
  (`upstream_done`) before firing.
- **Early stop.** When the plan ends in a `limit`, the loop breaks as soon as
  `len(output_records) >= limit`, so upstream work can stop before materializing everything.

`run()` wraps this, timing the whole call with `time.time()` and returning:

- `DataRecordCollection(records, execution_stats)`,
- `per_op_list` — one dict per operator (`op_id`, `latency_s`, `cost_usd`, tokens, `num_records`,
  `num_passed`, …), aggregated from `RecordOpStats` keyed by `logical_op_id` (= `params_id`),
- `plan_dict` — plan-level totals; **`plan_dict["latency_s"]` is the wall-clock elapsed time**.

---

## 5. DataRecords and stats

- LLM operators return a PZ `DataRecordSet` (records + `RecordOpStats`). Filters/joins set
  `passed_operator` on each output record; the scheduler propagates only passing records.
- `RecordOpStats` carries per-record `time_per_record`, `cost_per_record`, token counts, the model
  name, and (for joins/filters) the decision — these feed the cost model and quality evaluator.
- Per-operator sampling: up to `_MAX_SAMPLES` `(input_record, output_record)` pairs are captured for
  filter/convert/project ops; joins are captured separately as `(pair_record, pair_record|None)` over
  **every** pair (see §6), because the oracle quality judge scores a join across all its pairs.

---

## 6. How `sem_join` runs  ⭐

`sem_join(other, condition, model, join_parallelism=20, depends_on=…)` wraps PZ's `NestedLoopsJoin`.
Three things make the join special: (a) it has a **second input pipeline** (`other`), (b) the PZ
operator **parallelizes pairs internally and accumulates state across calls**, and (c) the scheduler
fires it in one of two modes.

### 6.1 The merged (join) schema

On `sem_join`, the left fields keep their names; each right field that collides is suffixed with
`_right` (repeatedly if needed):

```python
merged_defs = dict(self._defs)              # left names win
for name, field_def in other._defs.items():
    new_name = name
    while new_name in merged_defs:
        new_name = f"{new_name}_right"       # e.g. prod_id → prod_id_right
    merged_defs[new_name] = field_def
```

This mirrors PZ's `union_schemas(join=True)` / `DataRecord.from_join_parents`, so downstream operators
reference right-side columns as e.g. `brand_norm_right`.

### 6.2 The PZ operator: `NestedLoopsJoin`

`NestedLoopsJoin(left_candidates, right_candidates)`:

- Evaluates the cross product **one LLM call per `(left, right)` pair** via
  `_process_join_candidate_pair` → `Generator` (prompt strategy `COT_JOIN`), which returns a boolean
  `passed_operator`. The output record is `DataRecord.from_join_parents(schema, left, right)`.
- Runs those pair-calls on its **own** `ThreadPoolExecutor(max_workers=join_parallelism)` — this is the
  join's parallelism, separate from the scheduler's shared pool.
- **Accumulates inputs across calls.** It keeps `self._left_input_records` and
  `self._right_input_records`, and each call only computes the *new* cross-terms:

  ```
  new_left × new_right   +   new_left × stored_right   +   stored_left × new_right
  ```

  then appends the new inputs to the stored lists. So repeated calls never recompute a pair, and the
  total number of pairs across all calls is exactly |L| × |R|. It returns `None` when a call produced
  no output (e.g. one side empty so far).

### 6.3 The right side is a nested pipeline

`other` is a full `PhysicalPipeline`. At execution, `_execute_core` builds a **`right_state`** entry per
join with the right side's own initial records and its own per-stage queues, kept separate from the
main queues:

```python
right_state[i] = {
    "other": other, "n": n_r, "initial": r_initial,
    "n_fed": 0,                 # how many right inputs fed into the right pipeline so far
    "iq": {...}, "fq": {...},   # the right sub-pipeline's OWN input/future queues
    "pending": [],              # right outputs produced this tick, ready to join
    "all_right": [],            # all right outputs produced so far (used by the barrier path)
    "done": …,                  # True once the right pipeline is fully drained
    "has_right_join": bool,     # True if other._ops contains a join stage
    "right_future": Future|None # set when has_right_join=True (see below)
}
```

**Simple right pipeline** (`has_right_join = False`): every tick the scheduler advances the right
pipeline by one batch — feeds up to `batch_size` right initial records into the right pipeline,
harvests each right stage's finished futures into the next stage, and moves finished right *outputs*
into `pending` (and `all_right`). Only `filter`, `convert`, `project`, and `groupby` stage types are
handled by this inline advancement loop.

**Right pipeline that itself contains a join** (`has_right_join = True`): the inline advancement loop
cannot dispatch join operators (it has no join-firing path). Instead, once the executor is created,
the right pipeline is submitted as a **background future** using the **shared executor**:

```python
rs["right_future"] = executor.submit(
    other._execute_core, r_snap, max_samples, skip_limit, executor
)
```

`_execute_core` accepts an optional `_executor` parameter. When provided it skips creating a new
`ThreadPoolExecutor` and uses the passed-in one, so the right pipeline's operators compete for the
**same 20 worker slots** as the left pipeline — no extra workers are created. One slot is consumed
by the right pipeline's scheduling loop while it runs; the remaining slots service operators from
both branches.

Each tick the scheduler polls the future; when it completes all output records land in `rs["pending"]`
and `rs["all_right"]` at once and `rs["done"]` is set to `True`. The outer join then fires via the
barrier or incremental path depending solely on `has_dl` (§6.4).

### 6.4 Two firing modes (barrier and incremental)

The firing mode depends only on `has_dl` — whether a `limit` appears downstream. The `has_right_join`
flag controls *how* the right pipeline runs, not *when* the join fires. This matches PZ exactly.

**Barrier join** (no downstream limit) — wait for both sides to finish, then one call:

```python
if upstream_done(join) and right_state[join]["done"] and input_queues[join]:
    left_all  = input_queues[join][:]
    right_all = right_state[join]["all_right"]
    result_set, _ = join_pz_op(left_all, right_all)
```

For a `has_right_join` pipeline `rs["done"]` flips when the background future completes, so the
barrier condition is met as soon as both the left pipeline and the right future are done — same
semantics as PZ, which achieves the same by interleaving both branches in one loop.

**Incremental join** (downstream limit exists) — fire as records arrive, enabling early stopping:

```python
left_batch  = input_queues[join][:]
right_batch = right_state[join]["pending"][:]
if left_batch or right_batch:
    input_queues[join].clear()
    right_state[join]["pending"] = []
    result_set, _ = join_pz_op(left_batch, right_batch)
```

`batch_size = limit_val` throttles upstream operators to submit at most `limit_val` records per tick,
so the join sees records in small batches and the limit can stop the loop early — matching PZ's
`join_has_downstream_limit_op` branch exactly.

For a `has_right_join` pipeline the right records all arrive in `rs["pending"]` at once when the
future completes. The incremental path handles this naturally: it joins whatever left records are
available against the newly-arrived right batch that tick, then continues on subsequent ticks as more
left records come in.

Because `NestedLoopsJoin` accumulates `_left/_right_input_records` across calls, each call only
computes the *new* cross-terms — nothing is lost and nothing is recomputed.

**Comparison to PZ.** PZ represents plans as trees and yields all operators (both branches) in
post-order, so they advance in one shared loop with one thread pool. For a barrier join PZ checks
`_upstream_ops_finished` across all upstream topo-indices; for an incremental join it fires whenever
either side has inputs. Our implementation matches both behaviors: barrier fires when
`upstream_done AND rs["done"]`; incremental fires whenever `left_batch or right_batch`. The only
structural difference is that for `has_right_join` pipelines we run the right branch as a background
future in the shared pool rather than inlining it in the same loop — the join semantics and the 20-
worker budget are identical to PZ.

### 6.5 Synchronous firing + pass-through future

Like PZ, the join call runs **synchronously on the scheduler thread** — it is *not* submitted to the
shared executor. (PZ notes the same at `parallel_execution_strategy.py:149-151`: submitting the join
would race on its accumulation state.) The already-computed result set is then wrapped in a trivial
pass-through future so downstream stages consume join output through the same `future_queues`
mechanism as everything else:

```python
if result_set is not None:
    future_queues[join].append(executor.submit(lambda rset=result_set: rset))
```

Because the main loop can empty (all left consumed into the operator) before the right sub-pipeline
finishes, `any_pending()` also stays alive while any join's right sub-pipeline is still producing.

### 6.6 Concurrency summary for a join

```
scheduler shared pool (max_workers)
  ├── left pipeline per-record ops (sem_filter, sem_map, …)
  ├── right pipeline per-record ops (simple right: inline tick loop; nested-join right: background future)
  └── join's own pool (join_parallelism) ── pairwise LLM calls for one join call
                                             (synchronous from the loop's perspective)
```

For a simple right pipeline both branches interleave in the same tick loop and share the pool.
For a `has_right_join` right pipeline, `other._execute_core` runs as a submitted future using the
**same shared pool** (the executor is passed down via the `_executor` parameter). One slot is
consumed by the right pipeline's scheduling loop; the remaining slots service operators from both
branches. The right pipeline's inner join uses its own `join_parallelism` pool as usual.

In SemBench both the scheduler pool and join parallelism are configured to **20**
(`config/system/palimpzest/*.json` for PZ; the agent's prompts/defaults for PhysicalPipeline).
PZ's library defaults are 64/64.

### 6.7 Join stats & samples

Each pair yields a `RecordOpStats` (with the join condition, decision, tokens, cost, time). The
pipeline also collects **every** join pair as a `(pair_record, pair_record|None)` sample
(`join_op_samples`, keyed by `id(op)`), so the quality evaluator can compare plan-join decisions
against oracle-join decisions pair-by-pair.

### 6.8 Non-LLM `join` (exact predicate)

`pipeline.join(other, condition_fn, depends_on=…)` is the deterministic counterpart of `sem_join`,
wrapping `NonLLMJoin` (`stage_type="join"`, `op_type="join"`). Everything above about the join —
the merged `_right` schema (§6.1), the nested right sub-pipeline and `right_state` (§6.3), the
**barrier vs incremental** firing (§6.4), synchronous execution + pass-through future (§6.5) —
applies unchanged, because the scheduler dispatches purely on `stage_type == "join"` and uses only
the generic `op._pz_op(left, right)` / `_left_input_records` / `_right_input_records` interface.

What differs from `sem_join`:
- Each pair is decided by `condition_fn(left_dict, right_dict) -> bool` (the two raw records, each
  with its own pre-join field names — e.g. `lambda l, r: l["brand"] == r["brand"]`), not an LLM.
- No `model` / `join_parallelism` / prompt / reasoning effort; pairs run **sequentially** (the
  predicate is fast and I/O-free). Cost is `0` and `RecordOpStats` records `fn_call_duration_secs`.
- Because its `attributes` carry no `"model"`, it is **not** a semantic op: it is excluded from
  `per_sem_op_info`, so quality eval treats it as perfect-quality (like `ExactFilter` / `Map`), and
  `make_oracle_copy` copies it verbatim (only its right sub-pipeline's LLM ops are swapped to the
  oracle model).

Prefer `join` over `sem_join` whenever the match is an exact/computable predicate (e.g. equal brand
and category) — it's free and deterministic.

### 6.9 Self-joins — the upstream runs **once** (a deliberate optimization *beyond* PZ)

`pipeline.sem_join(pipeline, …)` / `pipeline.join(pipeline, …)` (same object on both sides) is
supported, and we run the shared upstream **once**, feeding its output to both join inputs.

**How PZ handles it (for contrast — PZ runs the upstream twice).** PZ's Cascades optimizer builds a
*memo* of **groups**, where a group is an equivalence class of logically-equivalent sub-plans, keyed by
a structural id (`hash(sorted fields + properties)`, `Group._compute_group_id`). For `a.sem_join(a)`
both sides are structurally identical, so they collapse into **one group `G`** and the join expression
is `Join(input_group_ids=[G, G])`. But a group is a **search-time** construct: it lets the optimizer
decide *how to implement* `G` once. When the final physical plan is extracted
(`_get_greedy_physical_plan`) it recurses into each input group **without memoization**, unfolding `G`
into **two** physical instances of A. PZ's own plan code says as much — it builds
`unique_full_op_id = "{topo_idx}-{full_op_id}"` "to differentiate between multiple instances of the
same physical operator **e.g. in self-joins**." So PZ **executes A twice** and counts its cost twice
(2N), attributed correctly by that per-position (`topo_idx`) keying.

**What we do instead.** Re-running a (possibly nondeterministic, and expensive) LLM `sem_map` on both
sides is wasteful and can even make the two sides differ. So PhysicalPipeline applies the
common-subexpression optimization PZ skips: when `other is self`, the join is flagged
`self_join=True` and stores `other=None` (no separate right pipeline — which also removes the
self-reference that would otherwise make `_flat_ops()`/`make_oracle_copy()` recurse). The upstream runs
once (the left chain); when the join fires it is called as `join_pz_op(left_batch, left_batch)` — the
**same records as both sides** — and `NestedLoopsJoin`'s accumulation builds the full L×L cross product
from the streamed batches. Consequences:
- `A` appears **once** in `_flat_ops`, runs once, cost counted once (N, vs PZ's 2N). Because there is
  no duplicate stage, no per-position stat reconciliation is needed here.
- Both sides are the identical output (a stronger self-join guarantee than PZ gives); the join still
  evaluates all N² ordered pairs.
- `make_oracle_copy` rebuilds the self-join by joining the oracle copy with itself.
- A genuine **two-pipeline** join (`left.sem_join(right)` with separately-built pipelines) is the
  distinct case where the upstream legitimately runs twice; PhysicalPipeline runs the right-side ops via
  `right_state`, matching PZ's two-instance behavior. (There, per-position stat keying — §5 / PZ's
  `topo_idx` — is what avoids double-counting; PhysicalPipeline currently keys op stats by `params_id`,
  so a two-pipeline join that shares an operator would over-count — a known gap, separate from
  self-joins.)

---

## 7. `run()`, `run_subset()`, and oracle copies

- **`run()`** — execute on all of `self._df`, apply the `limit`, return
  `(DataRecordCollection, per_op_list, plan_dict)`. Used for the final full execution.
- **`run_subset(num_samples, seed, subset_cache_path)`** — execute on a random `num_samples`-row
  sample (cached to `subset_cache_path` so runs are reproducible), with `skip_limit=True` so the
  limit does not truncate the exploration sample. For a **self-join**, both sides are drawn from the
  *same* cached subset so the pair space is consistent. Returns
  `(per_op_list, SubsetExecutionContext, plan_dict)` with sampled/outputs/per-op samples for the cost
  model and quality evaluator.
- **`make_oracle_copy(oracle_model, …)`** — clone the plan with every LLM operator (including a join's
  right sub-pipeline, recursively) rebound to a strong "oracle" model. The quality evaluator runs this
  copy via `run_subset` and scores the plan's decisions against the oracle's.

### 7.1 Two latencies: wall-clock vs summed-per-record

Both `run()` and `run_subset()` expose **two** latency numbers, for two different consumers:

| Number | Where | Meaning | Used for |
|---|---|---|---|
| **wall-clock** | `plan_dict["latency_s"]` = `time.time()` around `_execute_core` | real elapsed time, reflects the 20-way join/convert parallelism | reporting how long the plan actually took |
| **summed per-record** | each op's `latency_s` in `per_op_list` = `Σ time_per_record` | serial-equivalent time (ignores parallelism) | the **cost model** (per-op unit latency; its estimate sums per-op latencies too) |

For a 20-way-parallel operator the summed number is ~20× the wall-clock. The two must not be conflated:
report `plan_dict["latency_s"]` when you mean "how long did it take"; keep the per-op sums for cost-model
fitting/comparison (the estimator predicts a per-op sum, so its est-vs-act comparison stays consistent).

### 7.2 Caching in the agent optimization loop

The agent explores several candidate plans per query. Each plan is scored on the **same cached subset**
(`agent_cost_model/datasubset/<use_case>/sf_<sf>/Q<id>_subset.csv`). Two different caching policies apply:

**The plan itself is never cached — always re-executed.** `execute_subplan` calls
`pipeline.run_subset(...)` for every plan, even ones that share upstream operators with a
previously-run plan. This is deliberate: re-running is the only way to obtain a real **wall-clock**
latency measurement (`plan_dict["latency_s"]`) for that plan. If the plan's results were reused from a
cache, its measured execution time would be meaningless. (The plan row stores both `latency_s` — the
per-op sum, for the cost model — and `wall_latency_s` — the wall-clock, from `plan_dict`.)

**The oracle is cached — reused to save cost.** The oracle is a strong, expensive model used **only**
for quality scoring (it never contributes a latency number), so its results are cached and reused
instead of recomputed. `QualityEvaluator._get_oracle_context(plan, plan_name)` (in
[quality_evaluator.py](quality_evaluator.py)):

1. **Per-plan oracle cache.** Looks for `Q<id>_<plan_name>_gt.csv` (the oracle's output rows) and
   `Q<id>_<plan_name>_op_decisions.json` (per-op accept/reject decisions keyed by `source_indices`). If
   present, it loads them and **skips running the oracle entirely**. Otherwise it builds
   `make_oracle_copy(...)`, runs it once via `run_subset`, and writes both cache files.
2. **Shared canonical ground-truth.** The **first** plan's oracle output is stored as
   `self._canonical_oracle_df` and then reused as the ground-truth (`gt_df`) for **every** later plan's
   plan-level quality score — so the full oracle output is materialized once and shared across all plans.
   Oracle cost is likewise counted only on that first run.

So a plan's own LLM ops are always recomputed (for wall-clock latency), while the oracle side is cache-
backed. The guiding principle: **when another plan shares an operator, reuse the cached results for the
*oracle only* (to save the strong-model cost); still recompute the plan to measure its latency.**

**Per-operator oracle cache (cross-plan).** Beyond the per-plan cache above, oracle LLM calls are memoized
**per operator across plans** (`_MemoizingGenerator` in [quality_evaluator.py](quality_evaluator.py)):
- After `make_oracle_copy(...)`, `_install_oracle_cache` wraps every semantic operator's PZ `Generator`
  (including a join's right sub-pipeline) with a memoizer backed by one cache on the `QualityEvaluator`
  (so it persists across all plans for the query).
- The cache key is **content-based**: `(oracle model, reasoning effort, condition/generated-fields,
  input field values [+ right record for a join])`. The "input field values" are exactly the operator's
  `project_cols` = `get_input_fields()`, i.e. its **`depends_on` fields** when set (else all input
  fields) — the same fields PZ actually feeds to the LLM. So reuse depends **only on the fields the
  operator reads**, not on the rest of the record or the upstream that produced it: two plans reuse a
  shared operator's verdict whenever its `depends_on` field values match, **even if their upstreams
  differ** (e.g. an upstream `sem_map` that adds a field the operator doesn't depend on does not block
  reuse). If a depended-on field's value differs, the key differs and the oracle is correctly re-run —
  so there is never false reuse. (With no `depends_on`, PZ sends every input field to the LLM, so the key
  includes them all and reuse conservatively stops when the field set changes.) This is the
  operator-level analogue of PZ's validator `join_cache` / `filter_cache`.
- A cache hit returns a zero-cost `GenerationStats`, so `total_oracle_cost_usd` (now counted on every
  oracle run) reflects only real calls. The operator still builds fresh output records from the cached
  verdict, so scoring is unaffected.
- **Joins keep the full N² pairs** — every pair is still judged, but each unique `(condition, left, right)`
  pair is judged once across the whole optimization run; a second plan with the same join reuses all N²
  verdicts (0 new LLM calls).

---

## 8. Configuration knobs

| Knob | Where | Effect |
|---|---|---|
| `max_workers` | `PhysicalPipeline(...)` | size of the shared per-record thread pool |
| `join_parallelism` | `sem_join(...)` | size of the join's internal pair-call pool |
| `batch_size` | derived = query `limit` | records emitted per tick by per-record ops / fed to the right pipeline per tick |
| `_POLL_INTERVAL` | constant `0.3s` | future-wait timeout per drain (matches PZ) |
| `reasoning_effort_override` | per semantic op | overrides the default reasoning effort |

---

## 9. End-to-end example (a join plan)

```python
# left and right are separate PhysicalPipelines over the same source
left  = PhysicalPipeline("q7_left",  "styles", load_data("styles_details.csv"))
left.map(prep_row, cols=[...]); left.filter(lambda r: r["price_num"] <= 500.0)
left.sem_map([...], model=pz.Model.GPT_5_NANO, depends_on=["item_text"])
left.map(normalize_row, cols=[...]); left.project(["prod_id", "brand_norm", "category_norm"])

right = PhysicalPipeline("q7_right", "styles", load_data("styles_details.csv"))
# … identical shape …

left.sem_join(right,
              "normalized brand and category match",
              model=pz.Model.GPT_5_NANO,
              depends_on=["brand_norm", "category_norm", "brand_norm_right", "category_norm_right"])
left.filter(lambda r: r["prod_id_num"] < r["prod_id_num_right"])   # post-join filter
left.project(["prod_id", "prod_id_right"])
records, per_op, plan = left.run()
```

Execution: both sides run their `map → filter → sem_map → map → project` stages on the shared pool;
the join (no downstream limit here → **barrier**) fires once over all surviving left × all surviving
right using its own `join_parallelism` pool; the post-join `filter`/`project` then run on the results.
