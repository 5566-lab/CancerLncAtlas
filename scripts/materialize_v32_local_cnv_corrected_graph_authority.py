#!/usr/bin/env python3
"""Build a fresh G0 graph authority from audited E1 local-CNV edges.

Static graph inputs and outer-train expression identities remain bound to the
existing fresh patient-first authority.  Only the fold-local coexpression
artifact is replaced, using ``present_E1/rho_E1`` with no E0 fallback.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cc_hhgt.v32.formal_graph_authority import (
    GRAPH_INPUT_RECEIPT_FORMAT,
    GRAPH_INPUT_RECEIPT_STATUS,
)
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.local_cnv_core_preparation import (
    e1_coexpression_frame,
    sha256_file,
    validate_local_cnv_audit,
)


CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
    "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM", "STAD",
    "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def load_source_receipt(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256.lower():
        raise RuntimeError("Source graph receipt SHA256 drift")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("format") != GRAPH_INPUT_RECEIPT_FORMAT
        or receipt.get("status") != GRAPH_INPUT_RECEIPT_STATUS
        or set(receipt.get("fold_artifacts", {})) != {str(i) for i in range(5)}
    ):
        raise RuntimeError("Source graph receipt contract failed")
    return receipt


def initialize(args: argparse.Namespace) -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Corrected graph output reuse is forbidden: {output}")
    source = load_source_receipt(args.source_graph_receipt, args.source_graph_receipt_sha256)
    local = validate_local_cnv_audit(
        args.local_cnv_audit_root,
        expected_success_sha256=args.local_cnv_audit_success_sha256,
    )
    cancers = local.get("cohort", {}).get("cancers", list(CANCERS))
    if local.get("cohort") is not None and source.get("cancer_scope") != cancers:
        raise ValueError("Source graph cancer scope differs from the local-CNV cohort")
    output.mkdir(parents=True)
    from cc_hhgt.v32.local_cnv_sources import preflight
    preflight(output,"INITIALIZE","corrected_graph")
    for fold in range(5):
        (output / "folds" / f"fold_{fold}" / "train_coexpression").mkdir(
            parents=True
        )
    (output / "receipts").mkdir()
    atomic_json(output / "INITIALIZED.json", {
        "format": "CANCERLNCATLAS_V32_LOCAL_CNV_CORRECTED_GRAPH_INITIALIZED_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_graph_receipt": {
            "path": str(args.source_graph_receipt.resolve()),
            "sha256": args.source_graph_receipt_sha256.lower(),
        },
        "local_cnv_audit": local,
        "cancer_scope": cancers,
        "gates": {
            "e1_only": True,
            "e0_fallback_allowed": False,
            "sealed_test_accessed": False,
            "mutation_used": False,
            "training_started": False,
        },
    })
    print(json.dumps({"status": "PASS_INITIALIZED", "output": str(output)}))
    return 0


def materialize_cancer(args: argparse.Namespace) -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    cancer = str(args.cancer).upper()
    if cancer not in CANCERS:
        raise RuntimeError(f"Unknown cancer: {cancer}")
    output = args.output_root.resolve()
    from cc_hhgt.v32.local_cnv_sources import preflight
    preflight(output,cancer,"corrected_graph")
    initialized = json.loads((output / "INITIALIZED.json").read_text(encoding="utf-8"))
    if cancer not in initialized["cancer_scope"]:
        raise ValueError("Cancer is outside the initialized graph cohort")
    local = validate_local_cnv_audit(
        args.local_cnv_audit_root,
        expected_success_sha256=args.local_cnv_audit_success_sha256,
    )
    if initialized.get("local_cnv_audit", {}).get("success_sha256") != local["success_sha256"]:
        raise RuntimeError("Initialized local-CNV authority drift")
    receipt_path = output / "receipts" / f"{cancer}.json"
    final_paths = [
        output / "folds" / f"fold_{fold}" / "train_coexpression"
        / f"part_{cancer}.parquet"
        for fold in range(5)
    ]
    if receipt_path.exists() or any(path.exists() for path in final_paths):
        raise FileExistsError(f"Corrected graph cancer output reuse forbidden: {cancer}")
    records = []
    for fold, path in enumerate(final_paths):
        source_path = (
            args.local_cnv_audit_root.resolve() / "g0_edge_stability"
            / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
        )
        from cc_hhgt.v32.local_cnv_sources import partition_binding
        source_binding=partition_binding(source_path,args.local_cnv_audit_root.resolve()/"receipts"/(cancer+".json"))
        source_sha=sha256_file(source_path)
        source = pd.read_parquet(source_path)
        corrected = e1_coexpression_frame(source, expected_fold=fold)
        if sha256_file(source_path)!=source_sha:raise ValueError("Local-CNV E1 source changed during read")
        if len(corrected) and set(corrected.cancer_id.astype(str)) != {cancer}:
            raise RuntimeError(f"E1 cancer identity drift: {cancer}/{fold}")
        atomic_parquet(path, corrected)
        records.append({
            "fold": fold,
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "upstream_transformation":source_binding,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(corrected),
        })
    atomic_json(receipt_path, {
        "format": "CANCERLNCATLAS_V32_LOCAL_CNV_E1_GRAPH_CANCER_V1",
        "status": "PASS",
        "cancer_id": cancer,
        "local_cnv_audit_success_sha256": local["success_sha256"],
        "outputs": records,
        "gates": {
            "present_e1_only": True,
            "rho_e1_only": True,
            "e0_fallback_allowed": False,
            "typed_empty_partition_retained": True,
        },
    })
    print(json.dumps({"status": "PASS", "cancer_id": cancer, "rows": sum(x["rows"] for x in records)}))
    return 0


def finalize(args: argparse.Namespace) -> int:
    from cc_hhgt.v32.cnv_input_sources import require_host
    require_host()
    output = args.output_root.resolve()
    from cc_hhgt.v32.local_cnv_sources import preflight
    preflight(output,"FINALIZE","corrected_graph")
    if (output / "GRAPH_INPUT_AUTHORITY_RECEIPT.json").exists():
        raise FileExistsError("Corrected graph terminal receipt reuse is forbidden")
    initialized = json.loads((output / "INITIALIZED.json").read_text(encoding="utf-8"))
    source_path = Path(initialized["source_graph_receipt"]["path"])
    source = load_source_receipt(
        source_path, initialized["source_graph_receipt"]["sha256"]
    )
    local = validate_local_cnv_audit(
        args.local_cnv_audit_root,
        expected_success_sha256=args.local_cnv_audit_success_sha256,
    )
    fold_rows = {fold: 0 for fold in range(5)}
    cancer_receipt_hashes = {}
    source_receipts=[]
    cancers = local.get("cohort", {}).get("cancers", list(CANCERS))
    if initialized.get("cancer_scope") != cancers:
        raise ValueError("Initialized corrected graph cohort differs")
    from cc_hhgt.v32.local_cnv_cohort import require_partitions
    require_partitions(output / "folds", cancers, "fold_{fold}/train_coexpression/part_{cancer}.parquet")
    for cancer in cancers:
        receipt_path = output / "receipts" / f"{cancer}.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "PASS"
            or receipt.get("cancer_id") != cancer
            or receipt.get("local_cnv_audit_success_sha256") != local["success_sha256"]
            or receipt.get("gates", {}).get("e0_fallback_allowed") is not False
        ):
            raise RuntimeError(f"Invalid corrected graph cancer receipt: {cancer}")
        cancer_receipt_hashes[cancer] = sha256_file(receipt_path)
        source_receipts.append({"path":str(receipt_path),"sha256":sha256_file(receipt_path)})
        if len(receipt.get("outputs", [])) != 5 or {int(item["fold"]) for item in receipt.get("outputs", [])} != set(range(5)):
            raise RuntimeError(f"Corrected graph cancer lacks five folds: {cancer}")
        for record in receipt["outputs"]:
            fold = int(record["fold"])
            path = Path(record["path"])
            expected_path = output / "folds" / f"fold_{fold}" / "train_coexpression" / f"part_{cancer}.parquet"
            if path.resolve() != expected_path.resolve():
                raise ValueError("Corrected graph receipt path differs from declared partition")
            frame = pd.read_parquet(path)
            required = {
                "cancer_id", "lncrna_id", "gene_id", "rho",
                "source_split", "edge_outer_fold",
            }
            if missing := sorted(required - set(frame.columns)):
                raise RuntimeError(f"Corrected graph schema drift {cancer}/{fold}: {missing}")
            if (
                sha256_file(path) != record["sha256"]
                or len(frame) != int(record["rows"])
                or (len(frame) and not frame.cancer_id.astype(str).eq(cancer).all())
                or not frame.source_split.astype(str).eq("train").all()
                or not pd.to_numeric(frame.edge_outer_fold).eq(fold).all()
                or frame.duplicated(["cancer_id", "lncrna_id", "gene_id"]).any()
                or frame.rho.eq(0).any()
            ):
                raise RuntimeError(f"Corrected graph content drift: {cancer}/{fold}")
            fold_rows[fold] += len(frame)

    receipt = copy.deepcopy(source)
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["analysis_version"] = "CancerLncAtlas_V3.2_G012_LOCAL_CNV_ADJUSTED_E1"
    receipt["local_cnv_adjustment"] = {
        **local,
        "target": "E1_LOCAL",
        "source_columns": ["present_E1", "rho_E1"],
        "e0_fallback_allowed": False,
        "cancer_receipt_sha256": cancer_receipt_hashes,
        "source_receipts":source_receipts,
    }
    receipt["gates"]["local_cnv_adjusted_coexpression"] = True
    receipt["gates"]["e0_fallback_allowed"] = False
    total_rows = 0
    for fold in range(5):
        coexpression = output / "folds" / f"fold_{fold}" / "train_coexpression"
        parts = sorted(coexpression.glob("part_*.parquet"))
        if {p.name for p in parts} != {f"part_{c}.parquet" for c in cancers}:
            raise RuntimeError(f"Corrected graph fold {fold} differs from declared cancer partitions")
        declaration = receipt["fold_artifacts"][str(fold)]["train_coexpression"]
        declaration.update({
            "path": str(coexpression),
            "sha256": artifact_sha256(coexpression),
            "rows": fold_rows[fold],
            "local_cnv_adjusted": True,
            "local_cnv_target": "E1_LOCAL",
            "e0_fallback_allowed": False,
        })
        total_rows += fold_rows[fold]
    receipt["counts"]["five_fold_coexpression_rows"] = total_rows
    receipt["counts"]["five_fold_e1_coexpression_rows"] = total_rows
    receipt_path = output / "GRAPH_INPUT_AUTHORITY_RECEIPT.json"
    atomic_json(receipt_path, receipt)
    success = {
        "format": "CANCERLNCATLAS_V32_LOCAL_CNV_CORRECTED_GRAPH_SUCCESS_V1",
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_receipt_path": str(receipt_path),
        "graph_receipt_sha256": sha256_file(receipt_path),
        "local_cnv_audit_success_sha256": local["success_sha256"],
        "fold_rows": {str(k): v for k, v in fold_rows.items()},
        "gates": {
            "cancers": len(cancer_receipt_hashes) == len(cancers),
            "folds": len(fold_rows) == 5,
            "e1_only": True,
            "e0_fallback_allowed": False,
            "sealed_test_accessed": False,
            "mutation_used": False,
            "training_started": False,
        },
    }
    atomic_json(output / "SUCCESS.json", success)
    print(json.dumps({"status": "PASS", "graph_receipt_sha256": success["graph_receipt_sha256"], "rows": total_rows}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("initialize", "materialize-cancer", "finalize"), required=True
    )
    parser.add_argument("--source-graph-receipt", required=True, type=Path)
    parser.add_argument("--source-graph-receipt-sha256", required=True)
    parser.add_argument("--local-cnv-audit-root", required=True, type=Path)
    parser.add_argument("--local-cnv-audit-success-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cancer")
    args = parser.parse_args()
    if args.phase == "initialize":
        return initialize(args)
    if args.phase == "materialize-cancer":
        if not args.cancer:
            parser.error("--cancer is required for materialize-cancer")
        return materialize_cancer(args)
    return finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
