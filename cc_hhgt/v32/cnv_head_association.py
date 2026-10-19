"""Resolve CNV-head targets from a completed native association authority.

Native scope is replayed from annotation-defined candidates, never inferred
from observed output rows. Legacy inputs retain the original run identity.
"""
from pathlib import Path

from .current_core_export import verified_path
from .formal_expression_sources import read, record

LEGACY_RUN = "V32_LOCAL_CNV_CONFOUNDING_20260902_R3"


def association_contract(root, *, candidates=None, patient_map=None, capture=None):
    root = Path(root).resolve()
    success_path = root/"SUCCESS.json"
    success = read(success_path)
    cohort = success.get("cohort")
    if cohort is None:
        # A stripped native SUCCESS must not silently become the legacy route.
        for name in ("LOCAL_CNV_INDEPENDENT_AUDIT.json", "LOCAL_CNV_DECISION.json", "LINEAGE.json"):
            if (root/name).is_file() and read(root/name).get("cohort") is not None:
                raise ValueError("Native CNV association cohort was removed")
        return {"cohort": None, "run_id": LEGACY_RUN, "control_sources": [], "label_sources": {}}
    from .local_cnv_core_preparation import validate_local_cnv_audit
    from .local_cnv_cohort import replay
    validate_local_cnv_audit(root, require_correction=False)
    replay(cohort, root, candidates=candidates, patient_map=patient_map)
    run_id = success.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Native CNV association lacks an explicit run identity")
    controls = []

    def control(path, digest):
        path = verified_path({"path": str(path), "sha256": digest}, root)
        controls.append(record(path))
        return read(path) if capture is None else capture.json(path, role="association_control")

    control(root/"LOCAL_CNV_INDEPENDENT_AUDIT.json", success["independent_audit_sha256"])
    control(root/"LOCAL_CNV_DECISION.json", success["decision_sha256"])
    lineage = control(root/"LINEAGE.json", success["lineage_sha256"])
    if (lineage.get("run_id") != run_id or lineage.get("cohort") != cohort
            or set(lineage.get("receipt_sha256", {})) != set(cohort["cancers"])):
        raise ValueError("Native CNV association lineage run/cohort differs")
    labels = {}
    for cancer in cohort["cancers"]:
        receipt = control(root/"receipts"/(cancer+".json"), lineage["receipt_sha256"][cancer])
        outputs = receipt.get("pair_outputs", [])
        if (receipt.get("status") != "PASS" or receipt.get("run_id") != run_id
                or receipt.get("cancer_id") != cancer or receipt.get("cohort") != cohort
                or len(outputs) != 5 or {r["fold"] for r in outputs} != set(range(5))):
            raise ValueError("Native CNV worker identity or five-fold scope differs")
        for item in outputs:
            fold = item["fold"]
            expected = root/"pair_level"/f"cancer={cancer}"/f"fold={fold}"/"part.parquet"
            path = verified_path(item, root/"receipts")
            if (type(fold) is not int or path != expected
                    or item.get("rows") != cohort["candidate_rows_by_cancer"][cancer]):
                raise ValueError("Native CNV label partition identity differs")
            labels[(cancer, fold)] = record(path)
    return {"cohort": cohort, "run_id": run_id, "control_sources": controls, "label_sources": labels}
