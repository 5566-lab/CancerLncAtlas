"""Annotation-authorized ATAC scope; raw availability is independent of output."""
from pathlib import Path

import pandas as pd

from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .fold_local_scope import validate_candidates

FORMAT = "ATAC_ANNOTATION_COHORT_V1"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def from_definition(item, root, candidates, patient_map):
    from .atac_training import FORMAL_CANCERS, RAW_COVERED_CANCERS, normalise_patient_folds
    path, definition = validate_candidates(item, root, candidates, replay=True)
    cancers = definition["cancers"]
    if not set(cancers) <= set(FORMAL_CANCERS):
        raise ValueError("ATAC cohort contains an unknown cancer")
    frame = pd.read_parquet(candidates, columns=KEYS)
    if frame.isna().any().any() or frame.duplicated(KEYS).any() or set(frame.cancer_id) != set(cancers):
        raise ValueError("ATAC candidate key/cancer closure differs")
    counts = {c: int(frame.cancer_id.eq(c).sum()) for c in cancers}
    if any(n <= 0 or n > definition["budget_per_cancer"] for n in counts.values()):
        raise ValueError("ATAC candidate counts exceed declared budget")
    patients = normalise_patient_folds(pd.read_csv(patient_map, sep="\t"))
    selected = patients.loc[patients.cancer_id.isin(cancers)]
    if set(selected.cancer_id) != set(cancers) or selected.patient_id.duplicated().any():
        raise ValueError("ATAC cohort requires distinct patients in every declared cancer")
    return {"format": FORMAT, "candidate_definition": record(path),
        "candidates": record(candidates), "patient_map": record(patient_map),
        "cancers": cancers, "requested_budget_per_cancer": definition["budget_per_cancer"],
        "candidate_rows_by_cancer": counts, "candidate_rows": sum(counts.values()), "folds": 5,
        "covered_cancers": [c for c in cancers if c in RAW_COVERED_CANCERS],
        "raw_gap_cancers": [c for c in cancers if c not in RAW_COVERED_CANCERS]}


def replay(value, root, *, candidates=None, patient_map=None):
    if value.get("format") != FORMAT:
        raise ValueError("Invalid ATAC cohort contract")
    candidate_path = verified_path(value["candidates"], root)
    patient_path = verified_path(value["patient_map"], root)
    if ((candidates is not None and candidate_path != Path(candidates).resolve())
            or (patient_map is not None and patient_path != Path(patient_map).resolve())):
        raise ValueError("ATAC cohort candidate/patient input differs")
    expected = from_definition(value["candidate_definition"], root, candidate_path, patient_path)
    if expected != value:
        raise ValueError("ATAC cohort differs from annotation-defined candidates")
    return expected


def materialization_cohort(payload, root, *, candidates=None, patient_map=None, membership=None):
    """Replay native input contract and immutable materialization, including downgrade checks."""
    from .atac_materialization import _verify_input_contract
    item = payload.get("cohort")
    contract_path = payload.get("input_contract_path")
    if contract_path is None:
        if item is not None: raise ValueError("Native ATAC cohort lacks input contract")
        return None
    path = verified_path({"path": contract_path, "sha256": payload["input_contract_sha256"]}, root)
    contract = read(path)
    if contract.get("candidate_definition") is None and item is None: return None
    if item is None or contract.get("candidate_definition") is None:
        raise ValueError("ATAC native cohort was removed from materialization/contract")
    cohort = replay(item, root, candidates=candidates, patient_map=patient_map)
    artifacts = payload["input_artifacts"]
    paths = {r["role"]: verified_path(r, root) for r in artifacts}
    if len(paths) != len(artifacts): raise ValueError("Duplicate ATAC input roles")
    expected_roles = {"candidates", "patient_folds", "pathway_membership", "normalized_atac_matrix",
                      "identifier_mapping", "peak_set", "promoters"}
    if set(paths) != expected_roles: raise ValueError("Incomplete ATAC materialization inputs")
    _verify_input_contract(path, paths, {r["role"]: r["sha256"] for r in artifacts})
    expected = from_definition(contract["candidate_definition"], path.parent, paths["candidates"], paths["patient_folds"])
    if expected != cohort or (membership is not None and paths["pathway_membership"] != Path(membership).resolve()):
        raise ValueError("ATAC materialization inputs differ from cohort/training inputs")
    manifest_path = verified_path({"path": payload["manifest_path"], "sha256": payload["manifest_sha256"]}, root)
    entries = read(manifest_path)["entries"]
    observed = {}
    for entry in entries:
        p = (manifest_path.parent/entry["relative_path"]).resolve()
        if not p.is_relative_to(manifest_path.parent) or str(p) in observed:
            raise ValueError("Invalid ATAC materialization manifest path")
        verified_path({**entry, "path": str(p)}, manifest_path.parent); observed[str(p)] = entry["sha256"]
    for r in [*payload["records"], {"path": payload["r_audit_path"], "sha256": payload["r_audit_sha256"]}]:
        p = verified_path(r, root)
        if observed.get(str(p)) != r["sha256"]: raise ValueError("ATAC manifest omits/differs from consumed materialization")
    audit = read(verified_path({"path": payload["r_audit_path"], "sha256": payload["r_audit_sha256"]}, root))
    if (audit.get("status") != "PASS" or audit.get("covered_cancers") != cohort["covered_cancers"]
            or audit.get("raw_gap_cancers") != cohort["raw_gap_cancers"]):
        raise ValueError("ATAC R materialization cohort differs")
    return cohort


def training_cohort(success, lineage, input_audit, root, *, candidates, patient_map=None, sources=None):
    """Validate native completed scope against the materializer actually consumed."""
    records = [r for r in input_audit["input_artifacts"] if r.get("role") == "materialization_success"]
    claimed = success.get("cohort")
    if len(records) > 1 or (claimed is not None and len(records) != 1):
        raise ValueError("ATAC completion lacks unique consumed materialization")
    actual = None
    if records:
        path = verified_path(records[0], root)
        payload = read(path)
        actual = materialization_cohort(payload, path.parent, candidates=candidates, patient_map=patient_map)
        if sources is not None:
            sources.bind(records[0], root)
            if actual:
                for r in [*payload["input_artifacts"], *payload["records"], actual["candidate_definition"],
                          {"path": payload["input_contract_path"], "sha256": payload["input_contract_sha256"]},
                          {"path": payload["manifest_path"], "sha256": payload["manifest_sha256"]}]:
                    sources.bind(r, path.parent)
    if actual != claimed or lineage.get("cohort") != claimed:
        raise ValueError("ATAC completion cohort differs from consumed materialization")
    return actual
