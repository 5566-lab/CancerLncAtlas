#!/usr/bin/env python3
"""Materialize fresh, patient-first V3.2 G0/G1/G2 graph authorities.

The workflow is intentionally split into three restart-safe phases:

``initialize``
    Hash every selected source file, build the six static authorities, and
    write five outer-train lncRNA-expression trees.
``coexpression-cancer``
    Recompute all five folds for one cancer from the frozen patient authority.
    Thirty-three invocations can be scheduled with bounded parallelism.
``finalize``
    Validate all 33 x 5 partitions and issue the immutable graph receipt.

Historical graph rows, checkpoints, predictions, rankings, and web tables are
never accepted.  Reads are restricted to CancerLncAtlas on /public8 or ${PRIVATE_WORK_ROOT};
writes are restricted to ./data/CancerLncAtlas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FORMAT = "CANCERLNCATLAS_V32_G012_AUTHORITY_MATERIALIZER_V1"
RECEIPT_FORMAT = "CANCERLNCATLAS_V32_FRESH_G012_INPUT_AUTHORITY_V1"
RECEIPT_STATUS = "PASS_FRESH_HASH_BOUND_G012_AUTHORITIES"
PATIENT_MAP_SHA256 = "e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253"
PATIENT_RECEIPT_SHA256 = "1ef32bda9d14d87997f3b81c34b8aacce172de390db5c29aee011ca73ba317e0"
SEED = 20260726
CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)
STATIC_IDS = (
    "detection", "signed_membership", "pathway_hierarchy",
    "lnc_protein_binding", "protein_gene_encoding", "ppi",
)
ALLOWED_READ_ROOTS = (
    Path("./data/CancerLncAtlas"),
    Path("./data/CancerLncAtlas"),
)
ALLOWED_WRITE_ROOT = Path("./data/CancerLncAtlas")
DEFAULT_COVARIATES = (
    "purity", "age_years", "sex", "stage", "molecular_subtype",
    "clinical_subtype", "technical_batch", "leukocyte_fraction",
    "cell_fraction_Macrophages.M0", "cell_fraction_Macrophages.M1",
    "cell_fraction_Macrophages.M2",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"Empty authority tree: {path}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise RuntimeError(f"Symlink forbidden in authority tree: {item}")
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_path(value: str | Path, label: str, *, directory: bool = False) -> Path:
    path = Path(value).resolve(strict=True)
    if path.is_symlink() or not any(_inside(path, root) for root in ALLOWED_READ_ROOTS):
        raise RuntimeError(f"Unsafe {label}: {path}")
    if directory and not path.is_dir():
        raise RuntimeError(f"{label} is not a directory: {path}")
    if not directory and not path.is_file():
        raise RuntimeError(f"{label} is not a file: {path}")
    return path


def _write_root(value: str | Path) -> Path:
    path = Path(value).resolve()
    if not _inside(path, ALLOWED_WRITE_ROOT) or path == ALLOWED_WRITE_ROOT:
        raise RuntimeError(f"Unsafe output root: {path}")
    return path


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    resolved = _read_path(path, "configuration")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("format") != "CANCERLNCATLAS_V32_G012_SOURCE_CONFIG_V1":
        raise RuntimeError("Unexpected G012 source configuration format")
    return config, _sha256(resolved)


def _verified_source(record: dict[str, Any], label: str) -> Path:
    path = _read_path(record["path"], label)
    expected = str(record.get("sha256", "")).lower()
    observed = _sha256(path)
    if expected != observed:
        raise RuntimeError(f"{label} SHA256 drift: {observed} != {expected}")
    return path


def _partition(root: Path, cancer: str) -> Path:
    path = (root / f"cancer_id={cancer}" / "part-0.parquet").resolve(strict=True)
    if path.is_symlink() or not path.is_file() or not _inside(path, root):
        raise RuntimeError(f"Missing/unsafe canonical {cancer} partition: {path}")
    return path


def _inventory_partition_root(root: Path, role: str) -> list[dict[str, Any]]:
    records = []
    for cancer in CANCERS:
        path = _partition(root, cancer)
        records.append(
            {
                "role": role,
                "cancer_id": cancer,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )
    return records


def _identity_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame.loc[:, columns].drop_duplicates().copy()
    for column in columns:
        values[column] = values[column].astype(str).str.strip()
    values = values.sort_values(columns, kind="stable")
    payload = "".join(
        "\t".join(map(str, row)) + "\n"
        for row in values.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_map(patient_map: pd.DataFrame, fold: int) -> pd.DataFrame:
    result = patient_map.copy()
    result["split"] = "train"
    result.loc[result.patient_fold_id.astype(int).eq(fold), "split"] = "test"
    result.loc[
        result.patient_fold_id.astype(int).eq((fold + 1) % 5), "split"
    ] = "validation"
    return result


def _load_patient_map(config: dict[str, Any]) -> pd.DataFrame:
    patient = _verified_source(config["patient_map"], "frozen patient map")
    receipt = _verified_source(config["patient_receipt"], "frozen patient receipt")
    if _sha256(patient) != PATIENT_MAP_SHA256 or _sha256(receipt) != PATIENT_RECEIPT_SHA256:
        raise RuntimeError("Configuration is not bound to the frozen patient-first authority")
    frame = pd.read_csv(patient, sep="\t", dtype=str)
    required = {"cancer_id", "sample_id", "patient_id", "patient_fold_id", "fold_seed"}
    if missing := sorted(required - set(frame)):
        raise RuntimeError(f"Patient map lacks fields: {missing}")
    if (
        len(frame) == 0
        or frame.sample_id.duplicated().any()
        or set(frame.cancer_id) != set(CANCERS)
        or set(pd.to_numeric(frame.fold_seed)) != {SEED}
    ):
        raise RuntimeError("Frozen patient map content gate failed")
    return frame


def _candidate_pathways(activity_records: list[dict[str, Any]]) -> list[str]:
    # The formal preparation uses the union and explicitly reindexes a cancer
    # to that fixed universe.  Rare pathways can be absent from one cancer's
    # physical partition and become zero-information columns for that cancer.
    # Requiring each partition to contain the complete union would incorrectly
    # reject valid sparse activity storage (BLCA is the first counterexample).
    universe: set[str] = set()
    for record in activity_records:
        values = pd.read_parquet(record["path"], columns=["pathway_id"])
        universe.update(values.pathway_id.astype(str))
    result = sorted(universe)
    if len(result) != 2_135:
        raise RuntimeError(f"Expected 2,135 exact pathways, observed {len(result)}")
    return result


def _static_authorities(
    config: dict[str, Any], temporary: Path, pathways: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    source = config["sources"]
    static = temporary / "static"
    static.mkdir(parents=True)

    rates = pd.read_csv(
        _verified_source(source["detection"], "detection source"), sep="\t"
    )
    detection_column = "detection_rate_logcpm_gt0"
    required = {"cancer_id", "lncrna_id", detection_column}
    if missing := sorted(required - set(rates)):
        raise RuntimeError(f"Detection source lacks fields: {missing}")
    detection = rates[["cancer_id", "lncrna_id", detection_column]].rename(
        columns={detection_column: "detection_rate"}
    )
    detection["cancer_id"] = detection.cancer_id.astype(str)
    detection["lncrna_id"] = detection.lncrna_id.astype(str)
    detection["detection_rate"] = pd.to_numeric(detection.detection_rate, errors="raise")
    if (
        len(detection) != 557_337
        or detection.duplicated(["cancer_id", "lncrna_id"]).any()
        or detection.lncrna_id.nunique() != 16_889
        or set(detection.cancer_id) != set(CANCERS)
        or not detection.detection_rate.between(0, 1).all()
    ):
        raise RuntimeError("Detection grid invariant failed")
    eligible = detection.loc[detection.detection_rate.ge(0.10)].copy()
    candidate_lncs = set(eligible.lncrna_id)
    shared = eligible.groupby("lncrna_id", observed=True).cancer_id.nunique()
    if len(eligible) != 76_734 or len(candidate_lncs) != 8_541 or int(shared.ge(3).sum()) != 4_712:
        raise RuntimeError("Detection eligibility counts differ from the frozen design")
    detection = detection.sort_values(["cancer_id", "lncrna_id"], kind="stable")
    _atomic_parquet(static / "detection.parquet", detection)

    exact = pd.read_parquet(
        _verified_source(source["exact_membership"], "exact membership source")
    )[["pathway_id", "gene_id"]]
    exact = exact.loc[exact.pathway_id.astype(str).isin(set(pathways))].copy()
    exact["pathway_id"] = exact.pathway_id.astype(str)
    exact["gene_core"] = exact.gene_id.astype(str).str.removeprefix("GENE:")
    if len(exact) != 352_205 or exact.duplicated(["gene_core", "pathway_id"]).any():
        raise RuntimeError(f"Exact candidate membership count/key drift: {len(exact)}")
    weighted = pd.read_parquet(
        _verified_source(source["weighted_membership"], "signed membership source")
    )
    weighted = weighted.loc[weighted.mapping_status.astype(str).eq("mapped")].copy()
    weighted["gene_core"] = weighted.gene_id.astype(str).str.removeprefix("GENE:")
    weighted["pathway_id"] = weighted.pathway_id.astype(str)
    weighted["weight"] = pd.to_numeric(weighted.weight, errors="coerce")
    weighted = weighted.loc[weighted.pathway_id.isin(set(pathways))]
    weighted = weighted[["gene_core", "pathway_id", "weight"]]
    conflict = weighted.groupby(["gene_core", "pathway_id"], observed=True).weight.nunique()
    if conflict.gt(1).any():
        raise RuntimeError("Signed membership has conflicting duplicate weights")
    weighted = weighted.drop_duplicates(["gene_core", "pathway_id"])
    exact_keys = set(exact[["gene_core", "pathway_id"]].itertuples(index=False, name=None))
    signed_keys = set(weighted[["gene_core", "pathway_id"]].itertuples(index=False, name=None))
    exact_only_keys = len(exact_keys - signed_keys)
    signed_only_keys = len(signed_keys - exact_keys)
    membership = weighted.copy()
    null_static_weights = int(membership.weight.isna().sum())
    membership["weight"] = membership.weight.fillna(1.0)
    membership["gene_id"] = "GENE:" + membership.gene_core
    membership["source_database"] = "CancerLncAtlas_static_signed_pathway_membership"
    membership = membership[["gene_id", "pathway_id", "weight", "source_database"]]
    membership = membership.sort_values(["gene_id", "pathway_id"], kind="stable")
    negative_memberships = int(membership.weight.lt(0).sum())
    if negative_memberships != 125_194 or membership.weight.eq(0).any():
        raise RuntimeError(
            f"Signed membership polarity drift: negative={negative_memberships}"
        )
    _atomic_parquet(static / "signed_membership.parquet", membership)

    hierarchy = pd.read_parquet(
        _verified_source(source["hierarchy"], "pathway hierarchy source")
    )
    hierarchy = hierarchy.loc[hierarchy.pathway_id.astype(str).isin(set(pathways))].copy()
    hierarchy["pathway_id"] = hierarchy.pathway_id.astype(str)
    hierarchy["pathway_family_id"] = hierarchy.pathway_family_id.astype(str)
    hierarchy["weight"] = pd.to_numeric(hierarchy.membership_weight, errors="raise")
    hierarchy["source_database"] = "CancerLncAtlas_static_pathway_family"
    hierarchy = hierarchy[["pathway_id", "pathway_family_id", "weight", "source_database"]]
    if (
        len(hierarchy) != 2_135
        or hierarchy.duplicated("pathway_id").any()
        or hierarchy.weight.lt(0).any()
        or int(hierarchy.weight.eq(0).sum()) != 21
        or int(hierarchy.weight.gt(0).sum()) != 2_114
    ):
        raise RuntimeError("Pathway hierarchy count/weight invariant failed")
    hierarchy = hierarchy.sort_values("pathway_id", kind="stable")
    _atomic_parquet(static / "pathway_hierarchy.parquet", hierarchy)

    binding = pd.read_parquet(
        _verified_source(source["binding"], "lncRNA-protein binding source")
    )
    binding["lncrna_id"] = binding.lncrna_id.astype(str).str.strip()
    binding["protein_id"] = binding.protein_id.astype(str).str.strip()
    if "cancer_id" not in binding or "is_context_specific" not in binding:
        raise RuntimeError("Binding source lacks strict cancer-context fields")
    context = binding.is_context_specific
    if context.isna().any() or not pd.api.types.is_bool_dtype(context.dtype):
        raise RuntimeError("Binding context flag is not a complete boolean")
    binding = binding.loc[
        binding.cancer_id.isna()
        & ~context
        & binding.lncrna_id.ne("")
        & binding.lncrna_id.isin(candidate_lncs)
        & binding.protein_id.str.startswith("UNIPROT:")
    ].copy()
    binding["weight"] = pd.to_numeric(binding.weight, errors="raise")
    if binding.weight.le(0).any() or not np.isfinite(binding.weight.to_numpy(float)).all():
        raise RuntimeError("Global binding contains invalid weights")
    binding["cancer_id"] = pd.Series(pd.NA, index=binding.index, dtype="string")
    binding["is_context_specific"] = False
    binding = binding[
        ["lncrna_id", "protein_id", "weight", "cancer_id", "is_context_specific", "source_database"]
    ].sort_values(["lncrna_id", "protein_id"], kind="stable")
    binding = binding.drop_duplicates(["lncrna_id", "protein_id"], keep="first")
    if len(binding) != 623_207 or binding.lncrna_id.nunique() != 2_567:
        raise RuntimeError(
            f"Global binding invariant failed: rows={len(binding)}, lnc={binding.lncrna_id.nunique()}"
        )
    _atomic_parquet(static / "lnc_protein_binding.parquet", binding)

    protein = pd.read_parquet(
        _verified_source(source["protein_gene"], "protein-gene source")
    )
    protein["weight"] = pd.to_numeric(protein.mapping_weight, errors="raise")
    protein = protein[["protein_id", "gene_id", "weight", "source_database"]]
    protein = protein.sort_values(["protein_id", "gene_id"], kind="stable").drop_duplicates(
        ["protein_id", "gene_id"], keep="first"
    )
    if len(protein) != 20_008 or protein.weight.le(0).any():
        raise RuntimeError("Protein-gene encoding invariant failed")
    _atomic_parquet(static / "protein_gene_encoding.parquet", protein)

    ppi = pd.read_parquet(_verified_source(source["ppi"], "STRING PPI source"))
    ppi["protein_id_a"] = ppi.protein_id_a.astype(str)
    ppi["protein_id_b"] = ppi.protein_id_b.astype(str)
    if ppi.protein_id_a.eq(ppi.protein_id_b).any():
        raise RuntimeError("PPI source contains a self-loop")
    forward = ppi.protein_id_a.lt(ppi.protein_id_b).to_numpy(bool)
    left = np.where(forward, ppi.protein_id_a, ppi.protein_id_b)
    right = np.where(forward, ppi.protein_id_b, ppi.protein_id_a)
    ppi["protein_id_a"] = left
    ppi["protein_id_b"] = right
    ppi["weight"] = pd.to_numeric(ppi.weight, errors="raise")
    ppi = ppi.sort_values(
        ["protein_id_a", "protein_id_b", "weight"],
        ascending=[True, True, False], kind="stable",
    ).drop_duplicates(["protein_id_a", "protein_id_b"], keep="first")
    ppi = ppi[["protein_id_a", "protein_id_b", "weight", "source_database"]]
    if len(ppi) != 77_426 or ppi.weight.le(0).any():
        raise RuntimeError(f"Canonical PPI invariant failed: rows={len(ppi)}")
    _atomic_parquet(static / "ppi.parquet", ppi)

    artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id in STATIC_IDS:
        path = static / f"{artifact_id}.parquet"
        artifacts[artifact_id] = {
            "path": str(path), "bytes": int(path.stat().st_size), "sha256": _sha256(path),
            "rows": int(pq.ParquetFile(path).metadata.num_rows),
        }
    counts = {
        "eligible_detection_pairs": len(eligible),
        "candidate_lncrnas": len(candidate_lncs),
        "signed_membership": len(membership),
        "negative_membership": negative_memberships,
        "membership_null_static_weight_defaulted_to_one": null_static_weights,
        "membership_exact_only_keys_not_promoted_to_positive": exact_only_keys,
        "membership_signed_only_keys_retained": signed_only_keys,
        "hierarchy_authority_rows": len(hierarchy),
        "hierarchy_nonzero_graph_rows": int(hierarchy.weight.gt(0).sum()),
        "global_binding": len(binding),
        "protein_gene": len(protein),
        "canonical_ppi_pairs": len(ppi),
    }
    return artifacts, counts


def _materialize_train_expression(
    temporary: Path,
    patient_map: pd.DataFrame,
    lnc_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audits = [
        {"outer_fold": fold, "rows": 0, "samples": 0, "patients": 0}
        for fold in range(5)
    ]
    identity_sets: list[list[pd.DataFrame]] = [[] for _ in range(5)]
    for record in lnc_records:
        cancer = str(record["cancer_id"])
        frame = pd.read_parquet(record["path"])
        required = {"cancer_id", "sample_id", "patient_id", "lncrna_id", "logcpm"}
        if missing := sorted(required - set(frame)):
            raise RuntimeError(f"{cancer} lncRNA expression lacks: {missing}")
        if set(frame.cancer_id.astype(str)) != {cancer}:
            raise RuntimeError(f"{cancer} lncRNA expression contains another cancer")
        observed = frame[["cancer_id", "sample_id", "patient_id"]].drop_duplicates()
        expected = patient_map.loc[
            patient_map.cancer_id.astype(str).eq(cancer),
            ["cancer_id", "sample_id", "patient_id"],
        ].drop_duplicates()
        joined = expected.merge(
            observed,
            on=["cancer_id", "sample_id", "patient_id"],
            how="left",
            indicator=True,
        )
        if joined._merge.ne("both").any():
            raise RuntimeError(f"{cancer} expression misses frozen patient-authority samples")
        # Source partitions may contain extra tumour samples that are absent
        # from the frozen activity-derived patient authority.  They are never
        # assigned a fold and are explicitly excluded from every output.
        authority_sample_ids = set(expected.sample_id.astype(str))
        frame = frame.loc[frame.sample_id.astype(str).isin(authority_sample_ids)].copy()
        for fold in range(5):
            split = _split_map(patient_map, fold)
            train_ids = set(
                split.loc[
                    split.cancer_id.astype(str).eq(cancer)
                    & split.split.astype(str).eq("train"), "sample_id"
                ].astype(str)
            )
            selected = frame.loc[frame.sample_id.astype(str).isin(train_ids)].copy()
            if set(selected.sample_id.astype(str)) != train_ids:
                raise RuntimeError(f"{cancer}/fold {fold} expression misses train samples")
            path = temporary / "folds" / f"fold_{fold}" / "train_expression" / f"part_{cancer}.parquet"
            _atomic_parquet(path, selected)
            identity_sets[fold].append(
                selected[["cancer_id", "sample_id", "patient_id"]].drop_duplicates()
            )
            audits[fold]["rows"] += int(len(selected))
    for fold in range(5):
        identity = pd.concat(identity_sets[fold], ignore_index=True).drop_duplicates()
        split = _split_map(patient_map, fold)
        expected = split.loc[split.split.eq("train"), ["cancer_id", "sample_id", "patient_id"]]
        if _identity_hash(identity, list(identity)) != _identity_hash(expected, list(expected)):
            raise RuntimeError(f"Fold {fold} expression identity hash failed")
        audits[fold].update(
            {
                "samples": int(identity[["cancer_id", "sample_id"]].drop_duplicates().shape[0]),
                "patients": int(identity[["cancer_id", "patient_id"]].drop_duplicates().shape[0]),
                "train_sample_patient_sha256": _identity_hash(
                    identity, ["cancer_id", "sample_id", "patient_id"]
                ),
                "train_patient_sha256": _identity_hash(identity, ["cancer_id", "patient_id"]),
            }
        )
    return audits


def initialize(config_path: Path) -> int:
    config, config_sha = _load_config(config_path)
    output = _write_root(config["output_root"])
    temporary = output.with_name(f".{output.name}.initialize.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"G012 authority output reuse is forbidden: {output}")
    patient_map = _load_patient_map(config)
    roots = {
        role: _read_path(config["sources"][role]["path"], role, directory=True)
        for role in ("lnc_expression_root", "gene_expression_root", "pathway_activity_root")
    }
    inventory = []
    for role, root in roots.items():
        inventory.extend(_inventory_partition_root(root, role))
    lnc_records = [row for row in inventory if row["role"] == "lnc_expression_root"]
    activity_records = [row for row in inventory if row["role"] == "pathway_activity_root"]
    pathways = _candidate_pathways(activity_records)
    temporary.mkdir(parents=True)
    try:
        static_artifacts, static_counts = _static_authorities(config, temporary, pathways)
        for artifact_id, record in static_artifacts.items():
            record["path"] = str(output / "static" / f"{artifact_id}.parquet")
        expression_audits = _materialize_train_expression(
            temporary, patient_map, lnc_records
        )
        for fold in range(5):
            (temporary / "folds" / f"fold_{fold}" / "train_coexpression").mkdir(
                parents=True, exist_ok=True
            )
        (temporary / "audits" / "coexpression").mkdir(parents=True)
        source_inventory = {
            "format": "CANCERLNCATLAS_V32_G012_SELECTED_SOURCE_INVENTORY_V1",
            "generated_at_utc": _now(),
            "configuration_sha256": config_sha,
            "patient_authority": {
                "manifest_sha256": PATIENT_MAP_SHA256,
                "receipt_sha256": PATIENT_RECEIPT_SHA256,
            },
            "partition_records": inventory,
            "static_sources": config["sources"],
            "gates": {
                "canonical_part_zero_only": True,
                "duplicate_partial_files_selected": False,
                "historical_graph_rows_used": False,
                "historical_model_outputs_used": False,
            },
        }
        _atomic_json(temporary / "SOURCE_INVENTORY.json", source_inventory)
        initialized = {
            "format": FORMAT,
            "status": "PASS_STATIC_AND_FOLD_EXPRESSION_INITIALIZED",
            "generated_at_utc": _now(),
            "configuration_sha256": config_sha,
            "cancer_scope": list(CANCERS),
            "pathways": len(pathways),
            "static_artifacts": static_artifacts,
            "static_counts": static_counts,
            "fold_expression": expression_audits,
            "coexpression_complete": False,
            "receipt_issued": False,
        }
        _atomic_json(temporary / "INITIALIZED.json", initialized)
        os.replace(temporary, output)
    except BaseException:
        raise
    print(json.dumps({"status": "PASS_INITIALIZED", "output": str(output)}))
    return 0


def _inventory_record(inventory: dict[str, Any], role: str, cancer: str) -> dict[str, Any]:
    rows = [
        row for row in inventory["partition_records"]
        if row["role"] == role and row["cancer_id"] == cancer
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Inventory lacks unique {role}/{cancer}")
    path = _read_path(rows[0]["path"], f"{role}/{cancer}")
    if _sha256(path) != rows[0]["sha256"]:
        raise RuntimeError(f"Inventory source drift: {role}/{cancer}")
    return rows[0]


def _empty_coexpression(fold: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cancer_id": pd.Series(dtype="string"),
            "lncrna_id": pd.Series(dtype="string"),
            "gene_id": pd.Series(dtype="string"),
            "rho": pd.Series(dtype="float64"),
            "source_split": pd.Series(dtype="string"),
            "edge_outer_fold": pd.Series(dtype="int64"),
            "p_value": pd.Series(dtype="float64"),
            "fdr": pd.Series(dtype="float64"),
            "n_patients": pd.Series(dtype="int64"),
            "residual_design_rank": pd.Series(dtype="int64"),
            "correlation_df": pd.Series(dtype="int64"),
            "method": pd.Series(dtype="string"),
            "computation_run_id": pd.Series(dtype="string"),
        }
    )


def coexpression_cancer(config_path: Path, cancer: str) -> int:
    if cancer not in CANCERS:
        raise RuntimeError(f"Unknown cancer: {cancer}")
    config, config_sha = _load_config(config_path)
    output = _write_root(config["output_root"])
    initialized = json.loads((output / "INITIALIZED.json").read_text(encoding="utf-8"))
    if initialized.get("configuration_sha256") != config_sha:
        raise RuntimeError("Configuration differs from initialized authority")
    inventory = json.loads((output / "SOURCE_INVENTORY.json").read_text(encoding="utf-8"))
    lnc_record = _inventory_record(inventory, "lnc_expression_root", cancer)
    gene_record = _inventory_record(inventory, "gene_expression_root", cancer)
    patient_map = _load_patient_map(config)
    final_paths = [
        output / "folds" / f"fold_{fold}" / "train_coexpression" / f"part_{cancer}.parquet"
        for fold in range(5)
    ]
    audit_path = output / "audits" / "coexpression" / f"{cancer}.json"
    if any(path.exists() for path in final_paths) or audit_path.exists():
        raise FileExistsError(f"Coexpression output reuse forbidden for {cancer}")

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cc_hhgt.v32 import bulk_coexpression_release as bulk

    lnc, lnc_audit = bulk._patient_matrix(
        Path(lnc_record["path"]), cancer=cancer, entity_column="lncrna_id"
    )
    gene, gene_audit = bulk._patient_matrix(
        Path(gene_record["path"]), cancer=cancer, entity_column="gene_id"
    )
    detection = pd.read_parquet(output / "static" / "detection.parquet")
    candidate_lncs = sorted(
        detection.loc[
            detection.cancer_id.astype(str).eq(cancer)
            & detection.detection_rate.ge(0.10), "lncrna_id"
        ].astype(str).unique()
    )
    if set(lnc.columns.astype(str)) != set(candidate_lncs):
        raise RuntimeError(f"{cancer} lncRNA expression/candidate authority mismatch")
    membership_genes = set(
        pd.read_parquet(
            output / "static" / "signed_membership.parquet", columns=["gene_id"]
        ).gene_id.astype(str)
    )
    encoding_genes = set(
        pd.read_parquet(
            output / "static" / "protein_gene_encoding.parquet", columns=["gene_id"]
        ).gene_id.astype(str)
    )
    gene = gene.loc[:, sorted(set(gene.columns.astype(str)) & (membership_genes | encoding_genes))]
    covariates_path = _verified_source(config["sources"]["covariates"], "covariates")
    covariates = pd.read_parquet(covariates_path)
    cancer_patients = set(
        patient_map.loc[patient_map.cancer_id.astype(str).eq(cancer), "patient_id"].astype(str)
    )
    if not cancer_patients.issubset(set(lnc.index.astype(str))) or not cancer_patients.issubset(
        set(gene.index.astype(str))
    ):
        raise RuntimeError(f"{cancer} expression misses frozen patient-authority patients")
    # Patient-authority-external source rows are never assigned a fold.
    lnc = lnc.reindex(sorted(cancer_patients))
    gene = gene.reindex(sorted(cancer_patients))

    run_id = "V32_G012_OUTER_TRAIN_COEXPRESSION_20260830_R1"
    fold_frames: list[pd.DataFrame] = []
    fold_audits: list[dict[str, Any]] = []
    for fold in range(5):
        split = _split_map(patient_map, fold)
        train_patients = sorted(
            set(
                split.loc[
                    split.cancer_id.astype(str).eq(cancer) & split.split.eq("train"),
                    "patient_id",
                ].astype(str)
            )
        )
        lnc_fold = lnc.reindex(train_patients)
        gene_fold = gene.reindex(train_patients)
        summary: dict[str, Any] = {
            "outer_fold": fold,
            "declared_train_patients": len(train_patients),
            "candidate_lncrnas": len(candidate_lncs),
            "source_split": "train",
        }
        if len(train_patients) < 40:
            edges = _empty_coexpression(fold)
            summary.update(
                {"status": "TYPED_UNAVAILABLE_INSUFFICIENT_TRAIN_PATIENTS", "edges": 0}
            )
        else:
            lnc_fold, dropped_lnc = bulk._variance_filter(lnc_fold, 0.01)
            gene_fold, dropped_gene = bulk._variance_filter(gene_fold, 0.01)
            if lnc_fold.shape[1] > 12_000 or gene_fold.shape[1] > 25_000:
                raise RuntimeError(f"{cancer}/fold {fold} feature ceiling exceeded")
            design, covariates_used, missing_covariate_patients = bulk._covariate_design(
                covariates,
                cancer=cancer,
                patients=train_patients,
                columns=DEFAULT_COVARIATES,
            )
            residual_rank = bulk.design_rank(design)
            correlation_df = len(train_patients) - residual_rank - 1
            if correlation_df <= 0:
                raise RuntimeError(f"{cancer}/fold {fold} has no correlation df")
            lnc_values = bulk._residual_rank_matrix(lnc_fold, design)
            gene_values = bulk._residual_rank_matrix(gene_fold, design)
            sorted_p, adjusted_p, cutoff, total_tests = bulk._conservative_bh_candidates(
                lnc_values,
                gene_values,
                n_patients=len(train_patients),
                residual_rank=residual_rank,
                block_size=128,
                min_abs_rho=0.20,
                max_fdr=0.05,
            )
            edges = bulk._select_edges(
                lnc_values,
                gene_values,
                lnc_ids=lnc_fold.columns.to_numpy(str),
                gene_ids=gene_fold.columns.to_numpy(str),
                cancer=cancer,
                n_patients=len(train_patients),
                residual_rank=residual_rank,
                block_size=128,
                min_abs_rho=0.20,
                max_fdr=0.05,
                max_edges_per_direction=75,
                sorted_candidate_p=sorted_p,
                adjusted_candidate_p=adjusted_p,
                p_cutoff=cutoff,
                run_id=run_id,
            )
            edges["source_split"] = "train"
            edges["edge_outer_fold"] = int(fold)
            edges = edges[
                [
                    "cancer_id", "lncrna_id", "gene_id", "rho", "source_split",
                    "edge_outer_fold", "p_value", "fdr", "n_patients",
                    "residual_design_rank", "correlation_df", "method",
                    "computation_run_id",
                ]
            ].sort_values(["cancer_id", "lncrna_id", "gene_id"], kind="stable")
            if (
                edges.duplicated(["cancer_id", "lncrna_id", "gene_id"]).any()
                or edges.rho.eq(0).any()
                or not np.isfinite(edges.rho.to_numpy(float)).all()
            ):
                raise RuntimeError(f"{cancer}/fold {fold} coexpression edge invariant failed")
            summary.update(
                {
                    "status": "PASS_FRESH_OUTER_TRAIN_COEXPRESSION",
                    "edges": int(len(edges)),
                    "positive_edges": int(edges.rho.gt(0).sum()),
                    "negative_edges": int(edges.rho.lt(0).sum()),
                    "variable_lncrnas": int(lnc_fold.shape[1]),
                    "variable_genes": int(gene_fold.shape[1]),
                    "dropped_lncrnas_variance": len(dropped_lnc),
                    "dropped_genes_variance": len(dropped_gene),
                    "residual_design_rank": int(residual_rank),
                    "correlation_df": int(correlation_df),
                    "total_tests": int(total_tests),
                    "bh_candidate_p_values": int(len(sorted_p)),
                    "bh_p_cutoff": cutoff,
                    "covariates_used": covariates_used,
                    "missing_covariate_patients": int(missing_covariate_patients),
                }
            )
        fold_frames.append(edges)
        fold_audits.append(summary)

    temporaries: list[Path] = []
    for path, frame in zip(final_paths, fold_frames):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporaries.append(temporary)
    for temporary, final in zip(temporaries, final_paths):
        os.replace(temporary, final)
    audit = {
        "format": FORMAT,
        "status": "PASS_FIVE_FOLD_CANCER_COEXPRESSION",
        "generated_at_utc": _now(),
        "cancer_id": cancer,
        "configuration_sha256": config_sha,
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            )
        },
        "source_files": {"lncrna": lnc_record, "gene": gene_record},
        "source_audits": {"lncrna": lnc_audit, "gene": gene_audit},
        "folds": fold_audits,
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in final_paths
        ],
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "gates": {
            "patient_first": True,
            "outer_train_only": True,
            "historical_coexpression_edges_used": False,
            "old_absolute_rho_edges_used": False,
        },
    }
    _atomic_json(audit_path, audit)
    print(json.dumps({"status": audit["status"], "cancer_id": cancer}))
    return 0


def _artifact_declaration(path: Path, role: str, **extra: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256(path) if path.is_file() else _tree_sha256(path),
        "source_role": role,
        "outcome_derived": False,
        "historical_model_output": False,
        "toy_or_synthetic": False,
        **extra,
    }


def finalize(config_path: Path) -> int:
    config, config_sha = _load_config(config_path)
    output = _write_root(config["output_root"])
    initialized = json.loads((output / "INITIALIZED.json").read_text(encoding="utf-8"))
    if initialized.get("configuration_sha256") != config_sha:
        raise RuntimeError("Configuration differs from initialized authority")
    receipt_path = output / "GRAPH_INPUT_AUTHORITY_RECEIPT.json"
    if receipt_path.exists() or (output / "COMPLETION.json").exists():
        raise FileExistsError("Final G012 receipt reuse is forbidden")
    patient_map = _load_patient_map(config)
    static_paths = {
        artifact_id: output / "static" / f"{artifact_id}.parquet"
        for artifact_id in STATIC_IDS
    }
    if not all(path.is_file() for path in static_paths.values()):
        raise RuntimeError("A static G012 authority is missing")
    static_roles = {
        "detection": "outcome_free_expression_detection",
        "signed_membership": "static_annotation",
        "pathway_hierarchy": "static_annotation",
        "lnc_protein_binding": "static_annotation",
        "protein_gene_encoding": "static_annotation",
        "ppi": "static_annotation",
    }
    artifacts = {
        artifact_id: _artifact_declaration(path, static_roles[artifact_id])
        for artifact_id, path in static_paths.items()
    }
    folds: dict[str, Any] = {}
    total_coexpression = 0
    for fold in range(5):
        expression = output / "folds" / f"fold_{fold}" / "train_expression"
        coexpression = output / "folds" / f"fold_{fold}" / "train_coexpression"
        expression_parts = sorted(expression.glob("part_*.parquet"))
        coexpression_parts = sorted(coexpression.glob("part_*.parquet"))
        if len(expression_parts) != 33 or len(coexpression_parts) != 33:
            raise RuntimeError(
                f"Fold {fold} lacks 33 expression/coexpression partitions: "
                f"{len(expression_parts)}/{len(coexpression_parts)}"
            )
        identities = pd.concat(
            [
                pd.read_parquet(
                    path, columns=["cancer_id", "sample_id", "patient_id"]
                ).drop_duplicates()
                for path in expression_parts
            ],
            ignore_index=True,
        ).drop_duplicates()
        split = _split_map(patient_map, fold)
        expected = split.loc[
            split.split.eq("train"), ["cancer_id", "sample_id", "patient_id"]
        ]
        sample_hash = _identity_hash(
            identities, ["cancer_id", "sample_id", "patient_id"]
        )
        patient_hash = _identity_hash(identities, ["cancer_id", "patient_id"])
        if sample_hash != _identity_hash(expected, list(expected)):
            raise RuntimeError(f"Fold {fold} expression identity drift")
        fold_coexpression_rows = 0
        for path in coexpression_parts:
            frame = pd.read_parquet(
                path,
                columns=[
                    "cancer_id", "lncrna_id", "gene_id", "rho",
                    "source_split", "edge_outer_fold",
                ],
            )
            fold_coexpression_rows += len(frame)
            if len(frame) and (
                not frame.source_split.astype(str).eq("train").all()
                or not pd.to_numeric(frame.edge_outer_fold).eq(fold).all()
                or frame.rho.eq(0).any()
                or frame.duplicated(["cancer_id", "lncrna_id", "gene_id"]).any()
            ):
                raise RuntimeError(f"Fold {fold} coexpression provenance/key drift: {path}")
        total_coexpression += fold_coexpression_rows
        common = {
            "outer_fold": fold,
            "patient_first": True,
            "source_split": "train",
            "train_sample_patient_sha256": sample_hash,
            "train_patient_sha256": patient_hash,
        }
        folds[str(fold)] = {
            "outer_fold": fold,
            "train_expression": _artifact_declaration(
                expression, "outer_train_expression", **common
            ),
            "train_coexpression": _artifact_declaration(
                coexpression,
                "outer_train_coexpression",
                rows=fold_coexpression_rows,
                **common,
            ),
        }
    receipt = {
        "format": RECEIPT_FORMAT,
        "status": RECEIPT_STATUS,
        "generated_at_utc": _now(),
        "analysis_version": "CancerLncAtlas_V3.2_G012_FRESH_PATIENT_FIRST",
        "cancer_scope": list(CANCERS),
        "configuration_sha256": config_sha,
        "patient_fold_authority": {
            "manifest_sha256": PATIENT_MAP_SHA256,
            "receipt_sha256": PATIENT_RECEIPT_SHA256,
        },
        "artifacts": artifacts,
        "fold_artifacts": folds,
        "counts": {
            **initialized["static_counts"],
            "five_fold_coexpression_rows": total_coexpression,
        },
        "source_inventory": _artifact_declaration(
            output / "SOURCE_INVENTORY.json", "static_annotation"
        ),
        "gates": {
            "historical_graph_rows_used": False,
            "legacy_graph_root_fallback_allowed": False,
            "toy_or_synthetic_fallback_allowed": False,
            "outer_train_expression_only": True,
            "outer_train_coexpression_only": True,
            "static_evidence_outcome_free": True,
            "same_node_and_relation_schema_all_variants": True,
            "sealed_test_labels_read": False,
        },
    }
    _atomic_json(receipt_path, receipt)
    receipt_sha = _sha256(receipt_path)
    _atomic_json(
        output / "INPUT_HASHES.json",
        {
            "format": "CANCERLNCATLAS_V32_G012_INPUT_HASHES_V1",
            "formal_graph_authority_receipt_sha256": receipt_sha,
            "patient_map_sha256": PATIENT_MAP_SHA256,
            "patient_receipt_sha256": PATIENT_RECEIPT_SHA256,
            "configuration_sha256": config_sha,
        },
    )
    completion = {
        "format": FORMAT,
        "status": "PASS_G012_GRAPH_INPUT_AUTHORITY_COMPLETE",
        "generated_at_utc": _now(),
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha,
        "coexpression_rows": total_coexpression,
        "training_started": False,
    }
    _atomic_json(output / "COMPLETION.json", completion)
    print(json.dumps(completion, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("initialize", "coexpression-cancer", "finalize"), required=True
    )
    parser.add_argument("--cancer", choices=CANCERS)
    args = parser.parse_args()
    thread_names = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    )
    if any(os.environ.get(name) != "1" for name in thread_names):
        raise RuntimeError("Every numerical thread environment must be pinned to 1")
    if args.phase == "initialize":
        return initialize(args.config)
    if args.phase == "coexpression-cancer":
        if not args.cancer:
            parser.error("--cancer is required for coexpression-cancer")
        return coexpression_cancer(args.config, args.cancer)
    return finalize(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
