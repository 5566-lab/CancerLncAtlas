"""Synchronize dataset declarations with hash-verified single-cell evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

LOW_FEATURE_UNIVERSE_THRESHOLD = 1000
_STALE_FLAGS = {
    "DONOR_CELLTYPE_METADATA_UNAVAILABLE", "LOW_LNCRNA_FEATURE_UNIVERSE",
    "KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION",
    "FRESH_DIRECT_ID_FEATURE_COUNT_DIFFERS_FROM_READ_ONLY_AUDIT", "NONE",
}


def adapt_verified_head_tables(association, celltype, candidates, manifest):
    """Adapt verified source IDs, source tiers and donor/compartment features."""
    ids = candidates.lncrna_id.astype(str).drop_duplicates()
    bare = ids.str.replace(r"^(?:LNC:|LNCRNA:)", "", regex=True)
    if bare.duplicated().any():
        raise ValueError("Ambiguous candidate lncRNA namespace")
    lookup = dict(zip(bare, ids))
    association, celltype = association.copy(), celltype.copy()
    for frame in (association, celltype):
        original = frame.lncrna_id.astype(str)
        key = original.str.replace(r"^(?:LNC:|LNCRNA:)", "", regex=True)
        frame["lncrna_id"] = key.map(lookup).fillna(original)
    if "source_tier" not in association:
        tiers = manifest.set_index(["cancer_id", "dataset_id"]).source_tier
        keys = pd.MultiIndex.from_frame(association[["cancer_id", "dataset_id"]])
        association["source_tier"] = tiers.reindex(keys).to_numpy()
        if association.source_tier.isna().any():
            raise ValueError("Association has no verified source tier")
    if "n_observations" not in association:
        association["n_observations"] = association.n_donors
    if "association_effect" not in association:
        association["association_effect"] = association.spearman_rho
    if "q_value" not in association:
        association["q_value"] = association.bh_q_global_tests
    # Association targets are compartment-level. Pool major types within each
    # donor using cell counts before the consumer's across-donor average.
    if not set(celltype.expression_summary_scale) <= {"MEAN_LOG1P_CPM10000", "MEAN_SOURCE_LOG_NORMALIZED"}:
        raise ValueError("Unsupported source expression scale")
    if (celltype.groupby(["cancer_id", "dataset_id"]).expression_summary_scale.nunique() != 1).any():
        raise ValueError("Mixed source expression scales within a dataset")
    keys = ["cancer_id", "dataset_id", "patient_id", "compartment", "lncrna_id"]
    if celltype[keys].isna().any().any() or (celltype.cell_count <= 0).any():
        raise ValueError("Invalid donor/compartment summary")
    celltype["expression_sum"] = celltype.mean_expression * celltype.cell_count
    grouped = celltype.groupby(keys, observed=True, as_index=False).agg(
        n_cells=("cell_count", "sum"),
        detected_cell_count=("detected_cell_count", "sum"),
        expression_sum=("expression_sum", "sum"),
        expression_summary_scale=("expression_summary_scale", "first"),
    )
    grouped["detection_rate"] = grouped.detected_cell_count / grouped.n_cells
    grouped["mean_log_expression"] = grouped.expression_sum / grouped.n_cells
    grouped["cell_type"] = grouped.compartment
    grouped = grouped.drop(columns="expression_sum")
    return association, grouped


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path, expected):
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"Evidence SHA256 mismatch: {path}")


def sync_verified_manifest(manifest, evidence):
    """Caller must verify evidence; untouched datasets retain their declarations."""
    result = manifest.copy()
    changes = []
    for cancer, record in sorted(evidence.items()):
        mask = result.cancer_id.astype(str).eq(cancer)
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected one manifest row for {cancer}")
        index = result.index[mask][0]
        if str(result.at[index, "dataset_id"]) != str(record["dataset_id"]):
            raise ValueError(f"Evidence dataset mismatch: {cancer}")
        count = int(record["lncrnas"])
        if count < LOW_FEATURE_UNIVERSE_THRESHOLD or record["donor_metadata_available"] is not True:
            raise ValueError(f"Evidence does not meet feature/metadata requirements: {cancer}")
        flags = str(result.at[index, "quality_flags"]).split(";") if "quality_flags" in result else []
        updates = {
            "formal_eligible": True,
            "donor_metadata_available": True,
            "lncrna_feature_universe_count": count,
            "fresh_direct_id_lncrna_feature_count": count,
            "lncrna_feature_universe_count_semantics": "VERIFIED_CURRENT_SOURCE_FEATURE_MAP",
            "feature_universe_status": "PASS",
            "quality_status": "PASS",
            "quality_flags": ";".join(f for f in flags if f and f not in _STALE_FLAGS) or "NONE",
            "blocking_reason": None,
        }
        for field in ("metadata_path", "h5_path", "measurement_scale"):
            if field in record:
                updates[field] = record[field]
        for field, value in updates.items():
            old = result.at[index, field] if field in result else None
            old = None if pd.isna(old) else old
            if hasattr(old, "item"):
                old = old.item()
            if old != value:
                changes.append({"cancer_id": cancer, "field": field, "before": old, "after": value})
            result.loc[mask, field] = value
    return result, changes


def reconcile_bound_manifest(manifest, run_status_path, binding_path):
    run_status_path, binding_path = Path(run_status_path), Path(binding_path)
    run = json.loads(run_status_path.read_text())
    binding = json.loads(binding_path.read_text())
    success = json.loads(binding_path.with_name("SUCCESS.json").read_text())
    if success.get("status") != "SUCCESS" or success.get("target_host") != "149":
        raise ValueError("Invalid formal cohort receipt")
    verify_file(binding_path, success.get("binding_sha256"))
    audit_path = Path(success["validation_audit_path"])
    verify_file(audit_path, success["validation_audit_sha256"])
    if json.loads(audit_path.read_text()).get("status") != "PASS":
        raise ValueError("Cohort independent audit did not pass")
    rows = binding["formal_bindings"]
    bound = {r["cancer_id"] for r in rows}
    if len(bound) != len(rows) or bound != set(run["formal_eligible_cancers"]):
        raise ValueError("Binding/run-status coverage mismatch")
    evidence = {}
    for row in rows:
        cancer = row["cancer_id"]
        record = run["per_cancer"][cancer]
        root = Path(row["output_root"])
        verify_file(root / "SUCCESS.json", row["success_sha256"])
        partition = json.loads((root / "SUCCESS.json").read_text())
        if partition.get("status") != "SUCCESS" or partition.get("cancer_id") != cancer:
            raise ValueError(f"Partition identity/status mismatch: {cancer}")
        if partition["dataset_id"] != record["dataset_id"]:
            raise ValueError(f"Partition dataset mismatch: {cancer}")
        for field in ("lncrnas", "donors", "association_evidence_rows"):
            if int(partition[field]) != int(row[field]):
                raise ValueError(f"Binding {field} mismatch: {cancer}")
        verify_file(root / "FILE_MANIFEST.parquet", row["file_manifest_sha256"])
        verify_file(record["metadata_path"], record["metadata_sha256"])
        md = pd.read_parquet(record["metadata_path"], columns=["cancer_id", "dataset_id", "patient_id", "cell_type_major"])
        for field in ("patient_id", "cell_type_major"):
            if md[field].isna().any() or md[field].astype(str).str.strip().eq("").any():
                raise ValueError(f"Missing {field} in verified metadata: {cancer}")
        if set(md.cancer_id) != {cancer} or set(md.dataset_id) != {record["dataset_id"]}:
            raise ValueError(f"Metadata identity mismatch: {cancer}")
        if record.get("source_success_path"):
            verify_file(record["source_success_path"], record["source_success_sha256"])
        evidence[cancer] = {
            **{field: record[field] for field in ("dataset_id", "metadata_path", "h5_path", "measurement_scale") if field in record},
            "lncrnas": int(partition["lncrnas"]),
            "donor_metadata_available": True,
            "donors": int(partition["donors"]),
            "association_rows": int(partition["association_evidence_rows"]),
            "output_root": str(root),
            "metadata_sha256": record["metadata_sha256"],
            "success_sha256": row["success_sha256"],
        }
    result, changes = sync_verified_manifest(manifest, evidence)
    missing = result.dataset_id.isna() | result.dataset_id.astype(str).str.strip().eq("")
    if result.loc[missing, "formal_eligible"].any():
        raise ValueError("Formal dataset ID missing")
    result.loc[missing, "dataset_id"] = "UNAVAILABLE_" + result.loc[missing, "cancer_id"]
    return result, {
        "target_host": "149", "binding_sha256": file_sha256(binding_path),
        "run_status_sha256": file_sha256(run_status_path),
        "changes": changes, "evidence": evidence,
        "bound_cancers": sorted(bound),
    }
