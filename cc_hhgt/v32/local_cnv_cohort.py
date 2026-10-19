"""Candidate-defined local-CNV coverage, independent of observed output size."""
from pathlib import Path

import pandas as pd

from .formal_expression_sources import record
from .fold_local_scope import validate_candidates
from .cnv_only_training import TCGA_CANCERS

FORMAT = "LOCAL_CNV_ANNOTATION_COHORT_V1"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def from_definition(item, root, candidates, patient_map):
    path, definition = validate_candidates(item, root, candidates, replay=True)
    cancers = definition["cancers"]
    if not set(cancers) <= set(TCGA_CANCERS):
        raise ValueError("Local-CNV cohort contains an unknown cancer")
    frame = pd.read_parquet(candidates, columns=KEYS)
    if frame.isna().any().any() or frame.duplicated(KEYS).any() or set(frame.cancer_id) != set(cancers):
        raise ValueError("Local-CNV candidate key/cancer closure differs")
    counts = {c: int(frame.cancer_id.eq(c).sum()) for c in cancers}
    if any(n <= 0 or n > definition["budget_per_cancer"] for n in counts.values()):
        raise ValueError("Local-CNV candidate counts exceed the declared budget")
    patients = pd.read_csv(patient_map, sep="\t", dtype={"cancer_id": str, "patient_id": str})
    if not {"cancer_id", "patient_id", "patient_fold_id"} <= set(patients):
        raise ValueError("Local-CNV patient map lacks patient-first fold identities")
    selected = patients.loc[patients.cancer_id.isin(cancers)]
    if (selected[["cancer_id", "patient_id", "patient_fold_id"]].isna().any().any()
            or selected.patient_id.duplicated().any()
            or any(set(selected.loc[selected.cancer_id.eq(c), "patient_fold_id"]) != set(range(5)) for c in cancers)):
        raise ValueError("Local-CNV cohort requires distinct patients and all five folds per cancer")
    return {"format": FORMAT, "candidate_definition": record(path),
        "candidates": record(candidates), "patient_map": record(patient_map),
        "cancers": cancers, "requested_budget_per_cancer": definition["budget_per_cancer"],
        "candidate_rows_by_cancer": counts, "folds": 5, "candidate_rows": sum(counts.values()),
        "expected_pair_rows_all_folds": 5 * sum(counts.values())}


def from_config(config, root, candidates, patient_map):
    """No declaration retains the historical fixed 33 x 100,000 contract."""
    item = config.get("candidate_definition")
    return None if item is None else from_definition(item, root, candidates, patient_map)


def replay(value, root, *, candidates=None, patient_map=None):
    if value.get("format") != FORMAT:
        raise ValueError("Invalid local-CNV cohort contract")
    from .current_core_export import verified_path
    candidate_path = verified_path(value["candidates"], root)
    patient_path = verified_path(value["patient_map"], root)
    if ((candidates is not None and candidate_path != Path(candidates).resolve())
            or (patient_map is not None and patient_path != Path(patient_map).resolve())):
        raise ValueError("Local-CNV cohort candidate/patient input differs")
    expected = from_definition(value["candidate_definition"], root, candidate_path, patient_path)
    if expected != value:
        raise ValueError("Local-CNV cohort declaration differs from annotation-defined candidates")
    return expected


def require_receipt(receipt, cohort):
    if receipt.get("cohort") != cohort:
        raise ValueError("Local-CNV worker cohort differs from the release authority")


def require_partitions(root, cancers, layout):
    """A release must contain exactly its audited cancer/fold parquet files."""
    root = Path(root).resolve()
    expected = {root / layout.format(cancer=c, fold=f) for c in cancers for f in range(5)}
    actual = set(root.rglob("*.parquet"))
    if actual != expected:
        raise ValueError("Local-CNV partition coverage differs: missing or extra files")
