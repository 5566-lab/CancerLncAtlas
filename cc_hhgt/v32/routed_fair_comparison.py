"""Fair comparison of external routing and end-to-end hierarchical gating."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .multimodal_fusion import FUSION_FOLD_COLUMN, TARGET_KEYS, artifact_sha256


MODALITIES = ("mutation", "cnv", "atac")


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_ROUTED_FAIR_COMPARISON"
MODULE_ID = "external_vs_hierarchical_routing_comparison"


class RoutedComparisonError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComparisonContract:
    candidate_universe_sha256: str
    patient_modality_oof_sha256: str
    outer_pair_fold_sha256: str
    seed: int
    training_budget_id: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComparisonContract":
        required = {
            "candidate_universe_sha256",
            "patient_modality_oof_sha256",
            "outer_pair_fold_sha256",
            "seed",
            "training_budget_id",
        }
        if missing := sorted(required - set(value)):
            raise RoutedComparisonError(f"Comparison contract lacks: {missing}")
        contract = cls(**{key: value[key] for key in required})
        for field in (
            contract.candidate_universe_sha256,
            contract.patient_modality_oof_sha256,
            contract.outer_pair_fold_sha256,
        ):
            if len(str(field)) != 64:
                raise RoutedComparisonError("Comparison hashes must be SHA256 values")
        return contract


def exact_candidate_and_fold_hashes(frame: pd.DataFrame) -> tuple[str, str]:
    """Hash the ordered exact universe and its pair-fold assignment."""

    value = _canonical(frame, "comparison hash frame")
    if FUSION_FOLD_COLUMN not in value:
        raise RoutedComparisonError("Comparison hash frame lacks outer pair fold")
    candidate_digest = hashlib.sha256()
    fold_digest = hashlib.sha256()
    for row in value[list(TARGET_KEYS) + [FUSION_FOLD_COLUMN]].itertuples(index=False):
        key = "\t".join(map(str, row[:3])).encode("utf-8") + b"\n"
        candidate_digest.update(key)
        fold_digest.update(key[:-1] + b"\t" + str(int(row[3])).encode("ascii") + b"\n")
    return candidate_digest.hexdigest(), fold_digest.hexdigest()


def modality_source_composite_sha256(source_hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(source_hashes.items())), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_equal_contracts(
    external: Mapping[str, Any], hierarchical: Mapping[str, Any]
) -> ComparisonContract:
    first = ComparisonContract.from_mapping(external)
    second = ComparisonContract.from_mapping(hierarchical)
    if first != second:
        drift = {
            key: {"external": getattr(first, key), "hierarchical": getattr(second, key)}
            for key in first.__dataclass_fields__
            if getattr(first, key) != getattr(second, key)
        }
        raise RoutedComparisonError(f"Routing comparison is not fair: {drift}")
    return first


def _canonical(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if missing := sorted(set(TARGET_KEYS) - set(frame.columns)):
        raise RoutedComparisonError(f"{label} lacks exact keys: {missing}")
    value = frame.copy()
    value["cancer_id"] = value.cancer_id.astype(str).str.upper()
    for column in ("lncrna_id", "pathway_id"):
        value[column] = value[column].astype(str)
    if value[list(TARGET_KEYS)].duplicated().any():
        raise RoutedComparisonError(f"{label} has duplicate exact keys")
    return value.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def _metric(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    binary = (target >= 0.5).astype(int)
    result = {
        "logloss": float(-np.mean(target * np.log(clipped) + (1 - target) * np.log1p(-clipped))),
        "brier": float(np.mean(np.square(target - probability))),
        "auprc": float(average_precision_score(binary, probability)) if len(np.unique(binary)) > 1 else float("nan"),
        "auroc": float(roc_auc_score(binary, probability)) if len(np.unique(binary)) > 1 else float("nan"),
    }
    return result


def compare_routed_candidates(
    primary: pd.DataFrame,
    external: pd.DataFrame,
    hierarchical: pd.DataFrame,
    *,
    hierarchical_probability_column: str = "hierarchical_probability",
) -> pd.DataFrame:
    base = _canonical(primary, "primary OOF")
    outer = _canonical(external, "external routed OOF")
    inner = _canonical(hierarchical, "hierarchical OOF")
    required_base = {"fusion_target", "primary_probability", FUSION_FOLD_COLUMN}
    if missing := sorted(required_base - set(base.columns)):
        raise RoutedComparisonError(f"Primary OOF lacks: {missing}")
    if "discovery_adjusted_probability" not in outer:
        raise RoutedComparisonError("External routed OOF lacks discovery_adjusted_probability")
    if hierarchical_probability_column not in inner:
        raise RoutedComparisonError(f"Hierarchical OOF lacks {hierarchical_probability_column}")
    base_target = pd.to_numeric(base.fusion_target, errors="raise").to_numpy(float)
    base_primary = pd.to_numeric(base.primary_probability, errors="raise").to_numpy(float)
    if (
        not np.isfinite(base_target).all()
        or ((base_target < 0) | (base_target > 1)).any()
        or not np.isfinite(base_primary).all()
        or ((base_primary < 0) | (base_primary > 1)).any()
    ):
        raise RoutedComparisonError("Primary labels/probabilities are not finite within [0, 1]")
    for label, frame in (("external", outer), ("hierarchical", inner)):
        if len(frame) != len(base) or not frame[list(TARGET_KEYS)].equals(base[list(TARGET_KEYS)]):
            raise RoutedComparisonError(f"{label} candidate universe differs")
        required = {FUSION_FOLD_COLUMN, "fusion_target", "primary_probability"}
        required |= {
            f"{modality}_{suffix}"
            for modality in MODALITIES
            for suffix in ("probability", "available")
        }
        if missing := sorted(required - set(frame.columns)):
            raise RoutedComparisonError(f"{label} lacks formal parity columns: {missing}")
        if not np.array_equal(
            pd.to_numeric(frame[FUSION_FOLD_COLUMN]).to_numpy(int),
            pd.to_numeric(base[FUSION_FOLD_COLUMN]).to_numpy(int),
        ):
            raise RoutedComparisonError(f"{label} outer pair folds differ")
        candidate_target = pd.to_numeric(frame.fusion_target, errors="raise").to_numpy(float)
        candidate_primary = pd.to_numeric(frame.primary_probability, errors="raise").to_numpy(float)
        if not np.allclose(candidate_target, base_target, rtol=0.0, atol=1e-8, equal_nan=False):
            raise RoutedComparisonError(f"{label} labels differ from the primary OOF")
        if not np.allclose(candidate_primary, base_primary, rtol=0.0, atol=1e-7, equal_nan=False):
            raise RoutedComparisonError(f"{label} primary probabilities differ from the primary OOF")
    for modality in MODALITIES:
        available_column = f"{modality}_available"
        probability_column = f"{modality}_probability"
        external_available = outer[available_column]
        hierarchical_available = inner[available_column]
        if external_available.isna().any() or hierarchical_available.isna().any():
            raise RoutedComparisonError(f"{modality} availability contains null")
        external_available = external_available.astype(bool).to_numpy()
        hierarchical_available = hierarchical_available.astype(bool).to_numpy()
        if not np.array_equal(external_available, hierarchical_available):
            raise RoutedComparisonError(f"{modality} availability/callability differs")
        external_probability = pd.to_numeric(outer[probability_column], errors="coerce").to_numpy(float)
        hierarchical_probability = pd.to_numeric(inner[probability_column], errors="coerce").to_numpy(float)
        for label, probability, available in (
            ("external", external_probability, external_available),
            ("hierarchical", hierarchical_probability, hierarchical_available),
        ):
            if np.isnan(probability[available]).any() or np.isfinite(probability[~available]).any():
                raise RoutedComparisonError(f"{label} {modality} violates typed-null availability")
        if not np.allclose(
            external_probability[external_available],
            hierarchical_probability[hierarchical_available],
            rtol=0.0,
            atol=1e-7,
            equal_nan=False,
        ):
            raise RoutedComparisonError(f"{modality} OOF probabilities differ")
    target = base_target
    scores = {
        "primary": pd.to_numeric(base.primary_probability, errors="raise").to_numpy(float),
        "external_router": pd.to_numeric(outer.discovery_adjusted_probability, errors="raise").to_numpy(float),
        "hierarchical_end_to_end": pd.to_numeric(inner[hierarchical_probability_column], errors="raise").to_numpy(float),
    }
    if any(not np.isfinite(value).all() or ((value < 0) | (value > 1)).any() for value in scores.values()):
        raise RoutedComparisonError("Comparison probabilities are not finite within [0, 1]")
    rows: list[dict[str, Any]] = []
    groups = [("ALL_CANCERS", np.ones(len(base), bool))] + [
        (str(cancer), base.cancer_id.eq(cancer).to_numpy()) for cancer in sorted(base.cancer_id.unique())
    ]
    for cancer, mask in groups:
        primary_metric = _metric(target[mask], scores["primary"][mask])
        for method, probability in scores.items():
            metric = _metric(target[mask], probability[mask])
            rows.append(
                {
                    "cancer_id": cancer,
                    "method": method,
                    "rows": int(mask.sum()),
                    **metric,
                    "delta_logloss_vs_primary": primary_metric["logloss"] - metric["logloss"],
                    "delta_brier_vs_primary": primary_metric["brier"] - metric["brier"],
                    "regressed_from_primary": metric["logloss"] > primary_metric["logloss"],
                }
            )
    return pd.DataFrame(rows)


def materialize_routed_comparison(
    metrics: pd.DataFrame,
    *,
    output_root: str | Path,
    run_id: str,
    contract: ComparisonContract,
    sources: Mapping[str, str | Path],
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists():
        raise RoutedComparisonError(f"Comparison refuses output reuse: {output}")
    output.mkdir(parents=True)
    metrics_path = output / "fair_comparison_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False)
    per_cancer = metrics.loc[metrics.cancer_id.ne("ALL_CANCERS")]
    regression_counts = {
        method: int(group.regressed_from_primary.sum())
        for method, group in per_cancer.groupby("method", observed=True)
    }
    payload = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "contract": contract.__dict__,
        "same_outer_pair_folds": True,
        "same_patient_level_modality_oof": True,
        "same_labels_and_primary_baseline": True,
        "same_modality_availability_and_callability": True,
        "typed_null_gate_verified": True,
        "same_seed_and_training_budget": True,
        "heldout_results_not_used_for_model_selection": True,
        "per_cancer_regressions_from_primary": regression_counts,
        "metrics_path": str(metrics_path),
        "metrics_sha256": artifact_sha256(metrics_path),
        "sources": {
            key: {"path": str(Path(path).resolve()), "sha256": artifact_sha256(path)}
            for key, path in sources.items()
        },
        "candidate_only": True,
        "formal_v32_primary_unchanged": True,
    }
    success_path = output / "SUCCESS.json"
    success_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ComparisonContract",
    "RoutedComparisonError",
    "compare_routed_candidates",
    "exact_candidate_and_fold_hashes",
    "materialize_routed_comparison",
    "modality_source_composite_sha256",
    "validate_equal_contracts",
]
