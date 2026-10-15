"""Native drug-response analysis across cancer and annotated molecular subtypes.

This descriptive scan uses all assayed drugs, including those without curated
targets. It is separate from the target-derived model's prediction universe.
Signed measured association is not a causal drug effect or a predicted patient
response. Within-fold statistics are replication summaries, not fitted OOF
predictions or independent external validation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import drug_training as base
from .drug_training_streaming import _build_context, _association_statistics_vectorized, StreamingDrugTrainingConfig


@dataclass(frozen=True)
class DrugResponseScanConfig:
    seed: int = 20260726
    min_cell_lines_per_dataset: int = 5
    min_total_cell_lines: int = 8
    candidate_chunk_rows: int = 25000

    def __post_init__(self):
        if min(self.min_cell_lines_per_dataset, self.min_total_cell_lines) < 4 or self.candidate_chunk_rows < 1:
            raise ValueError("Drug response scan requires adequate unique model support and a positive chunk size")


def annotated_contexts(mapping: pd.DataFrame, subtypes: pd.DataFrame):
    """Return cancer-level and explicitly annotated subtype model sets."""
    contexts = []
    for cancer, group in mapping.groupby("cancer_id", sort=True):
        contexts.append((str(cancer), "ALL", "ALL", set(group.canonical_model_id), "CANCER_MODEL_MAP"))
    if subtypes.empty:
        return contexts
    required = {"canonical_model_id", "cancer_id", "subtype_system", "subtype_id", "subtype_available", "source"}
    if not required.issubset(subtypes):
        raise ValueError("Subtype table lacks source-bound model annotations")
    if subtypes.subtype_available.isna().any() or not subtypes.subtype_available.isin([True, False]).all():
        raise ValueError("Subtype availability must be explicit boolean")
    valid = subtypes.loc[subtypes.subtype_available].copy()
    for column in required - {"subtype_available"}:
        if valid[column].isna().any() or valid[column].astype(str).str.strip().eq("").any():
            raise ValueError("Available subtypes require complete identities and provenance")
    authority = mapping[["canonical_model_id", "cancer_id"]].drop_duplicates()
    joined = valid.merge(authority, on="canonical_model_id", how="left", suffixes=("", "_mapped"), validate="many_to_one")
    if not joined.cancer_id.eq(joined.cancer_id_mapped).all():
        raise ValueError("Subtype cancer disagrees with the unambiguous model map")
    if joined.groupby(["canonical_model_id", "subtype_system"]).subtype_id.nunique().gt(1).any():
        raise ValueError("Model has conflicting subtypes within one classification system")
    for (cancer, system, subtype), group in joined.groupby(["cancer_id", "subtype_system", "subtype_id"], sort=True):
        if system == "ALL":
            raise ValueError("ALL is reserved for the complete cancer cohort")
        contexts.append((str(cancer), str(system), str(subtype), set(group.canonical_model_id),
                         "|".join(sorted(set(group.source.astype(str))))))
    return contexts


def iter_response_scan(exact, response, expression, mapping, subtypes, targets, *, config=None):
    """Yield (context receipt, bounded candidate result frame) without fitting."""
    settings = config or DrugResponseScanConfig()
    exact = base.normalise_exact_candidates(exact)
    response = base.normalise_raw_response(response)
    expression = base.normalise_raw_expression(expression)
    mapping = base.assign_cell_line_folds(base.normalise_cell_line_map(mapping), settings.seed)
    target_ids = set(targets.drug_id.astype(str))
    config_stats = StreamingDrugTrainingConfig(seed=settings.seed,
        min_cell_lines_per_dataset=settings.min_cell_lines_per_dataset,
        min_total_cell_lines=settings.min_total_cell_lines)
    linked = response.merge(mapping[["dataset_id", "cell_line_id", "cancer_id"]],
                            on=["dataset_id", "cell_line_id"], how="inner", validate="many_to_one")
    drugs_by_cancer = linked.groupby("cancer_id").drug_id.unique().to_dict()
    contexts = annotated_contexts(mapping, subtypes)
    mapped_cancers = set(mapping.cancer_id)
    contexts += [(str(c), "ALL", "ALL", set(), "NO_MAPPED_MODEL") for c in sorted(set(exact.cancer_id)-mapped_cancers)]
    for cancer, system, subtype, models, source in contexts:
        genes = sorted(exact.loc[exact.cancer_id.eq(cancer), "lncrna_id"].unique())
        drugs = sorted(drugs_by_cancer.get(cancer, ()))
        identity = {"cancer_id": cancer, "subtype_system": system, "subtype_id": subtype}
        context_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        local_map = mapping.loc[mapping.canonical_model_id.isin(models)]
        context = _build_context(expression, response, mapping, sorted(models), cancer)
        fold_contexts = [_build_context(expression, response, mapping,
            sorted(local_map.loc[local_map.cell_line_fold_id.eq(fold), "canonical_model_id"].unique()), cancer) for fold in range(5)]
        receipt = {**identity, "context_id": context_id, "annotation_source": source, "canonical_models": len(models),
            "candidate_lncrnas": len(genes), "assayed_drugs_in_cancer": len(drugs),
            "target_annotated_drugs": len(set(drugs) & target_ids),
            "targetless_assayed_drugs": len(set(drugs)-target_ids), "candidate_rows": len(genes)*len(drugs),
            "context_available": bool(models and drugs and genes),
            "context_unavailable_reason": None if models and drugs and genes else "NO_MAPPED_ASSAY_OR_EXACT_LNCRNA_UNIVERSE"}
        if not receipt["candidate_rows"]:
            yield receipt, pd.DataFrame()
            continue
        # Enumerate the full cancer-assayed drug x exact-lncRNA relation. A
        # subtype lacking a measured drug retains its candidate with nulls.
        for start in range(0, receipt["candidate_rows"], settings.candidate_chunk_rows):
            positions = np.arange(start, min(start+settings.candidate_chunk_rows, receipt["candidate_rows"]))
            frame = pd.DataFrame({"cancer_id": cancer, "lncrna_id": np.asarray(genes)[positions % len(genes)],
                                  "drug_id": np.asarray(drugs)[positions // len(genes)]})
            stats = _association_statistics_vectorized(frame, context, {}, {}, config_stats)
            frame["subtype_system"] = system; frame["subtype_id"] = subtype
            frame["signed_expression_sensitivity_rho"] = stats.rho
            frame["association_available"] = stats.assay_available
            frame["unavailable_reason"] = pd.Series(stats.reasons, dtype="string").where(~stats.assay_available, pd.NA)
            frame["unique_cell_models"] = stats.n_cell_lines
            frame["contributing_datasets"] = stats.dataset_count
            frame["target_annotation_available"] = frame.drug_id.isin(target_ids)
            frame["identity_tier"] = np.where(frame.drug_id.str.startswith("ASSAY:"), "SOURCE_SCOPED_ASSAY", "RESOLVED_OR_DECLARED_IDENTIFIER")
            frame["observed_direction"] = pd.Series(np.where(stats.rho > 0, "HIGHER_EXPRESSION_HIGHER_SENSITIVITY",
                np.where(stats.rho < 0, "HIGHER_EXPRESSION_LOWER_SENSITIVITY", "ZERO_MEASURED_ASSOCIATION")), dtype="string").where(stats.assay_available, pd.NA)
            same = np.zeros(len(frame), dtype=int); opposite = same.copy(); available_folds = same.copy()
            for fold, fold_context in enumerate(fold_contexts):
                held = _association_statistics_vectorized(frame, fold_context, {}, {}, config_stats)
                frame[f"cell_fold_{fold}_rho"] = held.rho
                available_folds += held.assay_available.astype(int)
                comparable = held.assay_available & stats.assay_available & (held.rho != 0) & (stats.rho != 0)
                same += (comparable & (np.sign(held.rho) == np.sign(stats.rho))).astype(int)
                opposite += (comparable & (np.sign(held.rho) != np.sign(stats.rho))).astype(int)
            frame["folds_with_measured_association"] = available_folds
            frame["folds_with_same_direction"] = same; frame["folds_with_opposite_direction"] = opposite
            frame["causal_drug_effect_claimed"] = False
            yield receipt, frame
