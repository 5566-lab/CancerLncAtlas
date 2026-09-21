#!/usr/bin/env python3
"""Build outcome-free RNA/OCLR context for every registered LOCO cancer."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pandas as pd

from cc_hhgt.common import file_sha256, write_table
from cc_hhgt.v30_integrity import atomic_write_json, merkle_sha256


FORBIDDEN_TOKENS = (
    "rnass",
    "dnass",
    "stemness_rna",
    "stemness_dna",
    "p_value",
    "fdr",
    "proxy_label",
    "label_class",
    "bulk_effect",
)


def _load_external_module(path: Path) -> ModuleType:
    # Give the dynamically loaded file the real package parent so its audited
    # relative imports (for example ``from .stats``) resolve against this
    # frozen worktree's cc_hhgt package.
    spec = importlib.util.spec_from_file_location(
        "cc_hhgt._v31_frozen_compact_context", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen compact context module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registered_loco_cancers(fold_manifest: Path) -> tuple[str, ...]:
    folds = pd.read_csv(fold_manifest, sep="\t")
    if "test_cancer" not in folds:
        raise RuntimeError("Fold manifest lacks test_cancer")
    cancers = tuple(sorted(folds.test_cancer.astype(str).unique()))
    if len(cancers) != 33:
        raise RuntimeError(f"Expected 33 registered LOCO cancers, observed {len(cancers)}")
    missing_required = {"HNSC", "LGG"}.difference(cancers)
    if missing_required:
        raise RuntimeError(f"Required formal cancers are missing from RNA context: {sorted(missing_required)}")
    return cancers


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        write_table(frame, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--frozen-context-module", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    asset_root = Path(args.asset_root).resolve()
    fold_manifest = Path(args.fold_manifest).resolve()
    module_path = Path(args.frozen_context_module).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Full RNA context builder refuses reuse: {output_root}")
    output_root.mkdir(parents=True)

    cancers = registered_loco_cancers(fold_manifest)
    context = _load_external_module(module_path)
    # The frozen implementation was originally scoped to the three-cancer
    # pilot.  Its algorithms consult this immutable tuple at runtime; replacing
    # only the scope preserves the already-audited feature mathematics.
    context.CANCERS = cancers
    canonical_path = (
        data_root / "input_snapshot" / "adapter" / "cancer_specific_patient_fold_manifest.tsv"
    )
    covariate_path = data_root / "processed" / "tcga_association_covariates.parquet"
    weights_path = asset_root / "malta" / "mRNAsi_OCLR_signed_gene_weights.tsv.gz"
    for path in (canonical_path, covariate_path, weights_path, fold_manifest, module_path):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)

    canonical, canonical_audit = context.load_canonical_samples(canonical_path)
    frame, build_audit = context.build_expression_oclr_context(
        data_root=data_root,
        canonical=canonical,
        weights_path=weights_path,
        covariate_path=covariate_path,
    )
    if tuple(sorted(frame.cancer_id.astype(str).unique())) != cancers:
        raise RuntimeError("Full RNA context output cancer scope drift")
    if frame.duplicated(["cancer_id", "lncrna_id"]).any():
        raise RuntimeError("Full RNA context contains duplicate cancer/lncRNA keys")
    forbidden = sorted(
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise RuntimeError(f"Outcome-derived fields entered RNA context: {forbidden}")
    features = [
        column
        for column in frame.columns
        if column not in {"cancer_id", "lncrna_id"}
        and not column.endswith("__available")
    ]
    for feature in features:
        availability = f"{feature}__available"
        if availability not in frame:
            raise RuntimeError(f"RNA feature lacks availability mask: {feature}")
        mask = frame[availability].fillna(False).astype(bool)
        if frame.loc[~mask, feature].notna().any():
            raise RuntimeError(f"Unavailable RNA feature is not NaN: {feature}")

    table_path = output_root / "FULL_RNA_CONTEXT.parquet"
    _atomic_table(frame, table_path)
    lineage_paths = [
        (module_path, "frozen audited context implementation"),
        (Path(__file__).resolve(), "33-cancer scope orchestrator"),
        (fold_manifest, "registered 33-cancer LOCO scope"),
        (canonical_path, "canonical tumor-only sample universe"),
        (covariate_path, "outcome-free covariate residualization"),
        (weights_path, "frozen signed Malta OCLR gene weights"),
    ]
    lineage_rows = [
        {
            "role": role,
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path, role in lineage_paths
    ]
    lineage_path = output_root / "FULL_RNA_CONTEXT_INPUT_LINEAGE.tsv"
    _atomic_table(pd.DataFrame(lineage_rows), lineage_path)
    audit = {
        "status": "PASS",
        "stage": "PHASE_B3_TARGET_RNA_CONTEXT_BUILD",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": "CPU_HEAVY_MEMORY",
        "gpu_training_started": False,
        "registered_cancers": list(cancers),
        "reference_only_cancers_excluded": [],
        "required_formal_cancers_included": ["HNSC", "LGG"],
        "rows": int(len(frame)),
        "features": features,
        "outcome_derived_columns_used": 0,
        "forbidden_columns_observed": forbidden,
        "official_malta_sample_score_table_read": False,
        "allowed_malta_asset": "signed OCLR gene weights only",
        "canonical_sample_audit": canonical_audit,
        "feature_build_audit": build_audit,
        "table_sha256": file_sha256(table_path),
        "input_lineage_sha256": file_sha256(lineage_path),
        "full_cancer_model_training_started": False,
        "failures": [],
    }
    audit_path = output_root / "FULL_RNA_CONTEXT_AUDIT.json"
    atomic_write_json(audit_path, audit)
    manifest_rows = [
        {
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in (table_path, lineage_path, audit_path)
    ]
    manifest_path = output_root / "FULL_RNA_CONTEXT_SHA256.tsv"
    _atomic_table(pd.DataFrame(manifest_rows), manifest_path)
    success = {
        "status": "PASS",
        "table_sha256": file_sha256(table_path),
        "audit_sha256": file_sha256(audit_path),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_merkle_sha256": merkle_sha256(manifest_rows),
        "full_cancer_model_training_started": False,
        "success_written_last": True,
    }
    atomic_write_json(output_root / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
