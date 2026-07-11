"""Oracle-based quality evaluator for the cost model agent.

Per-operator quality:
  sem_filter / sem_join: agreement rate between plan decisions and oracle decisions.
  sem_map: oracle judges whether each output field is correct (batched LLM call).

Plan quality: plan output vs. oracle plan output using the original quality metrics
  (f1, relative_error, spearman_correlation, accuracy).

Oracle plan output is cached per plan_name in llm_judge_dir.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

def _load_evaluator(use_case: str, scale_factor: int, agent_dir: str):
    """Import and instantiate the concrete evaluator for the given use_case."""
    if use_case == "movie":
        from scenario.movie.evaluation.evaluate import MovieEvaluator
        return MovieEvaluator(use_case, scale_factor, agent_dir)
    if use_case == "ecomm":
        from scenario.ecomm.evaluation.evaluate import EcommEvaluator  # type: ignore
        return EcommEvaluator(use_case, scale_factor, agent_dir)
    raise ValueError(f"No evaluator registered for use_case={use_case!r}")


@dataclass
class QualityResult:
    quality: float                          # plan quality vs oracle output (0-1, NaN if failed)
    per_sem_op_quality: dict[str, float] = field(default_factory=dict)
    # {op_name: 0-1} for each semantic op where quality could be computed


class QualityEvaluator:
    """Evaluates plan quality against an oracle (strong LLM model)."""

    def __init__(
        self,
        oracle_client,
        oracle_model: str,
        query_id: int,
        use_case: str,
        scale_factor: int,
        agent_dir: str,
        llm_judge_dir: str | Path,
        oracle_reasoning_effort: str | None = None,
    ) -> None:
        self._oracle_client = oracle_client
        self._oracle_model = oracle_model          # string, used for OpenRouterClient judge calls
        self._oracle_reasoning_effort = oracle_reasoning_effort
        self._query_id = query_id
        self._use_case = use_case
        self._scale_factor = scale_factor
        self._agent_dir = agent_dir
        self._llm_judge_dir = Path(llm_judge_dir)
        self.total_oracle_cost_usd = 0.0
        self._canonical_oracle_df: "pd.DataFrame | None" = None

        # Resolve oracle model string → pz.Model enum once at init so make_oracle_copy
        # receives the enum directly (avoids re-resolving on every execute_plan call and
        # surfaces bad model names early).
        self._oracle_pz_model = None
        try:
            from agent.physical_pipeline import _str_to_pz_model
            self._oracle_pz_model = _str_to_pz_model(oracle_model)
        except Exception as e:
            print(f"[QualityEvaluator] could not resolve oracle pz model '{oracle_model}': {e}")

        self._evaluator = None
        try:
            self._evaluator = _load_evaluator(use_case, scale_factor, agent_dir)
        except Exception as e:
            print(f"[QualityEvaluator] evaluator init failed: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(
        self,
        plan,                         # PhysicalPipeline
        plan_name: str,
        plan_context,                 # SubsetExecutionContext
        plan_output_df: pd.DataFrame, # already normalized for evaluator
    ) -> QualityResult:
        """Run oracle evaluation and return QualityResult."""
        oracle_df, oracle_context = self._get_oracle_context(plan, plan_name)
        if oracle_df is not None and not oracle_df.empty and (
            self._canonical_oracle_df is None or self._canonical_oracle_df.empty
        ):
            self._canonical_oracle_df = oracle_df
        gt_df = self._canonical_oracle_df

        # Per-op quality
        per_sem_op_quality: dict[str, float] = {}
        if oracle_context is not None:
            for stage_idx, info in plan_context.per_sem_op_info.items():
                op_name = info["op_name"]
                op_type = info["op_type"]
                oracle_info = oracle_context.per_sem_op_info.get(stage_idx)
                try:
                    if op_type in ("sem_filter", "sem_join") and oracle_info is not None:
                        q = self._score_filter_join_op(info["samples"], oracle_info["samples"])
                        if q is not None:
                            per_sem_op_quality[op_name] = q
                    elif op_type == "sem_map":
                        q = self._score_map_op(info)
                        if q is not None:
                            per_sem_op_quality[op_name] = q
                except Exception as e:
                    print(f"[QualityEvaluator] per-op quality failed for {op_name}: {e}")

        # Plan quality
        quality = float("nan")
        if (
            self._evaluator is not None
            and gt_df is not None
            and not gt_df.empty
        ):
            try:
                import dataclasses
                # Serialize any list-valued columns so the evaluator can hash them
                def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
                    df = df.copy()
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
                    return df
                plan_output_df = _sanitize(plan_output_df)
                gt_df = _sanitize(gt_df)
                qm = self._evaluator._evaluate_single_query(
                    self._query_id, plan_output_df, gt_df
                )
                qm_dict = dataclasses.asdict(qm)
                qm_type = type(qm).__name__
                if "Retrieval" in qm_type:
                    quality = float(qm_dict.get("f1_score", float("nan")))
                elif "Aggregation" in qm_type:
                    quality = 1.0 - float(qm_dict.get("relative_error", 1.0))
                elif "Rank" in qm_type:
                    quality = float(qm_dict.get("spearman_correlation", float("nan")))
                elif "SingleAccuracy" in qm_type:
                    quality = float(qm_dict.get("accuracy", float("nan")))
            except Exception as e:
                print(f"[QualityEvaluator] plan quality evaluation failed: {e}")

        return QualityResult(quality=quality, per_sem_op_quality=per_sem_op_quality)

    # ------------------------------------------------------------------
    # Oracle plan execution and caching
    # ------------------------------------------------------------------

    def _get_oracle_context(
        self, plan, plan_name: str
    ) -> tuple[pd.DataFrame | None, object | None]:
        self._llm_judge_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._llm_judge_dir / f"Q{self._query_id}_{plan_name}_gt.csv"
        op_cache_path = self._llm_judge_dir / f"Q{self._query_id}_{plan_name}_op_decisions.json"

        if csv_path.exists():
            oracle_df = pd.read_csv(csv_path)
            oracle_context = self._load_op_decisions_cache(op_cache_path)
            return oracle_df, oracle_context

        try:
            # Prefer the resolved pz.Model enum; fall back to string (triggers _str_to_pz_model)
            oracle_model_arg = self._oracle_pz_model if self._oracle_pz_model is not None else self._oracle_model
            oracle_pipeline = plan.make_oracle_copy(oracle_model_arg, self._oracle_reasoning_effort)
            subset_cache_path = (
                Path(__file__).parent / "datasubset" / self._use_case / f"Q{self._query_id}_subset.csv"
            )
            if not subset_cache_path.exists():
                raise FileNotFoundError(
                    f"Subset CSV not found at {subset_cache_path}. "
                    "A plan must be executed before the oracle can run."
                )
            oracle_per_op_list, oracle_context = oracle_pipeline.run_subset(
                subset_cache_path=str(subset_cache_path)
            )
            if self._canonical_oracle_df is None:
                self.total_oracle_cost_usd += sum(
                    float(row.get("cost_usd", 0.0) or 0.0) for row in oracle_per_op_list
                )
        except Exception as e:
            print(f"[QualityEvaluator] oracle pipeline run failed: {e}")
            return None, None

        oracle_df = pd.DataFrame(oracle_context.output_records)
        oracle_df = self._normalize_df(oracle_df)
        oracle_df.to_csv(csv_path, index=False)
        self._save_op_decisions_cache(op_cache_path, oracle_context)

        oracle_result_path = (
            Path(__file__).parent / "datasubset" / self._use_case / f"Q{self._query_id}_oracle_result.csv"
        )
        if not oracle_result_path.exists():
            oracle_result_path.parent.mkdir(parents=True, exist_ok=True)
            oracle_df.to_csv(oracle_result_path, index=False)

        return oracle_df, oracle_context

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply use-case/query-specific column normalization (mirrors ExecutePlanTool)."""
        if self._use_case == "ecomm":
            if self._query_id == 4:
                df = df.rename(columns={"prod_id": "id"})
            elif self._query_id == 11:
                if df.shape[1] >= 4:
                    df["id"] = df.iloc[:, :4].astype(str).agg("-".join, axis=1)
                    df = df[["id"]]
            elif self._query_id == 12:
                if df.shape[1] >= 3:
                    df["id"] = df.iloc[:, :3].apply(
                        lambda r: json.dumps(
                            {"id": int(r.iloc[0]), "brand": r.iloc[1], "category": r.iloc[2]},
                            separators=(",", ":"),
                        ),
                        axis=1,
                    )
                    df = df[["id"]]
            elif self._query_id == 13:
                df = df.rename(columns={"prod_id": "id"})
                df = df[["id"]] if "id" in df.columns else df
        return df

    # ------------------------------------------------------------------
    # Op-level decision caching
    # ------------------------------------------------------------------

    def _save_op_decisions_cache(self, path: Path, oracle_context) -> None:
        cache: dict[str, dict] = {}
        for stage_idx, info in oracle_context.per_sem_op_info.items():
            decisions: dict[str, bool] = {}
            for inp, out in info["samples"]:
                if hasattr(inp, "source_indices"):
                    decisions[str(inp.source_indices)] = out is not None
            if decisions:
                cache[str(stage_idx)] = {"op_type": info["op_type"], "decisions": decisions}
        path.write_text(json.dumps(cache))

    def _load_op_decisions_cache(self, path: Path):
        """Return a minimal object with per_sem_op_info populated from JSON cache."""
        if not path.exists():
            return None
        try:
            from agent.physical_pipeline import SubsetExecutionContext
            raw = json.loads(path.read_text())
            per_sem_op_info: dict[int, dict] = {}
            for stage_str, v in raw.items():
                stage_idx = int(stage_str)
                # Represent cached decisions as (source_idx_str, passed_bool) tuples
                samples = [(src, passed) for src, passed in v["decisions"].items()]
                per_sem_op_info[stage_idx] = {
                    "op_name": f"_oracle_op{stage_idx}",
                    "op_type": v.get("op_type", "sem_filter"),
                    "attributes": {"model": self._oracle_model},
                    "samples": samples,
                }
            return SubsetExecutionContext(
                sampled_records=[],
                output_records=[],
                per_sem_op_info=per_sem_op_info,
                has_join=False,
                right_sampled_records=None,
            )
        except Exception as e:
            print(f"[QualityEvaluator] op decisions cache load failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Per-operator quality scoring
    # ------------------------------------------------------------------

    def _score_filter_join_op(self, plan_samples: list, oracle_samples: list) -> float | None:
        """Agreement rate between plan filter and oracle filter on the same records."""
        def _to_decisions(samples) -> dict[str, bool]:
            d: dict[str, bool] = {}
            for item in samples:
                if not (isinstance(item, tuple) and len(item) == 2):
                    continue
                inp, out = item
                if hasattr(inp, "source_indices"):
                    # DataRecord sample: out is DataRecord or None
                    # source_indices may be a list (joined records) — stringify for hashing
                    d[str(inp.source_indices)] = out is not None
                elif isinstance(inp, str):
                    # Cached format: (source_idx_str, passed_bool)
                    d[inp] = bool(out)
            return d

        plan_dec = _to_decisions(plan_samples)
        oracle_dec = _to_decisions(oracle_samples)
        common = set(plan_dec) & set(oracle_dec)
        if not common:
            return None
        return sum(1 for k in common if plan_dec[k] == oracle_dec[k]) / len(common)

    def _oracle_generate(self, prompt: str) -> str:
        """Call oracle client; handle both str and (str, reasoning) return types."""
        result = self._oracle_client.generate(
            system="You are a rigorous evaluator. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )
        if isinstance(result, tuple):
            if len(result) >= 3 and isinstance(result[2], dict):
                self.total_oracle_cost_usd += float(result[2].get("cost_usd", 0.0) or 0.0)
            return result[0]
        return result

    def _score_map_op(self, info: dict) -> float | None:
        """Oracle judges whether plan's sem_map output fields are correct (batched call)."""
        samples = info["samples"]
        cols_names: list[str] = info["attributes"].get("cols", [])
        if not cols_names:
            return None

        valid: list[tuple] = [
            (inp, out) for inp, out in samples
            if out is not None and hasattr(inp, "source_indices")
        ]
        if not valid:
            return None

        def _dr_to_dict(dr) -> dict:
            schema_cls = dr.schema if isinstance(dr.schema, type) else type(dr.schema)
            return {k: getattr(dr, k, None) for k in schema_cls.model_fields}

        record_texts: list[str] = []
        for i, (inp, out) in enumerate(valid):
            inp_d = _dr_to_dict(inp)
            out_d = _dr_to_dict(out)
            mapped = {k: out_d.get(k) for k in cols_names if k in out_d}
            record_texts.append(
                f"Record {i}:\n"
                f"  INPUT: {json.dumps(inp_d, default=str)}\n"
                f"  OPERATOR output fields {cols_names}: {json.dumps(mapped, default=str)}"
            )

        n_fields = len(cols_names)
        prompt = (
            "You are evaluating whether a semantic map operator produced correct outputs.\n\n"
            f"Output fields to evaluate: {cols_names}\n\n"
            + "\n\n".join(record_texts)
            + f"\n\nFor each record and each output field, score 1 if correct, 0 if incorrect.\n"
            f"Return ONLY valid JSON: "
            f'{{\"scores\": [[field0_rec0, field1_rec0, ...], [field0_rec1, ...], ...]}} '
            f"with {len(valid)} inner lists each of length {n_fields}."
        )

        try:
            response = self._oracle_generate(prompt)
            m = re.search(r"\{.*\}", response, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            scores_2d: list[list] = data.get("scores", [])
            if not scores_2d:
                return None
            n_records = len(valid)
            # Average per field across records, then average across fields
            field_avgs = []
            for j in range(n_fields):
                field_vals = [
                    float(scores_2d[r][j])
                    for r in range(min(len(scores_2d), n_records))
                    if j < len(scores_2d[r])
                ]
                if field_vals:
                    field_avgs.append(sum(field_vals) / len(field_vals))
            return sum(field_avgs) / len(field_avgs) if field_avgs else None
        except Exception as e:
            print(f"[QualityEvaluator] sem_map scoring failed: {e}")
            return None
