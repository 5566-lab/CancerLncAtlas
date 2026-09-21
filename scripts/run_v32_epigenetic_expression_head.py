#!/usr/bin/env python3
"""Run and publish the patient-OOF epigenetic lncRNA-expression head."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--regulatory-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--min-train-patients", type=int, default=20)
    parser.add_argument(
        "--signal-features",
        default="atac_distal_accessibility,distal_mutation_burden,promoter_methylation_beta,distal_methylation_beta",
        help="Comma-separated regulatory signal columns",
    )
    parser.add_argument(
        "--nuisance-features",
        default="local_cnv_log2,tumor_purity",
        help="Comma-separated adjustment columns",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve(strict=True)
    sys.path.insert(0, str(root))
    from cc_hhgt.v32.epigenetic_expression_head import (
        EpigeneticExpressionConfig,
        crossfit_epigenetic_expression,
    )

    source = args.regulatory_features.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Epigenetic expression output reuse refused: {output}")
    frame = pd.read_parquet(source)
    signal_features = tuple(
        item.strip() for item in args.signal_features.split(",") if item.strip()
    )
    nuisance_features = tuple(
        item.strip() for item in args.nuisance_features.split(",") if item.strip()
    )
    predictions, audit = crossfit_epigenetic_expression(
        frame,
        config=EpigeneticExpressionConfig(
            signal_features=signal_features,
            nuisance_features=nuisance_features,
            ridge_alpha=args.ridge_alpha,
            min_train_patients=args.min_train_patients,
        ),
    )
    output.mkdir(parents=True)
    prediction_path = output / "epigenetic_expression_patient_oof.parquet"
    temporary_prediction = output / ".epigenetic_expression_patient_oof.tmp.parquet"
    predictions.to_parquet(temporary_prediction, index=False, compression="zstd")
    os.replace(temporary_prediction, prediction_path)
    fit_records = audit.pop("fit_records")
    fit_path = output / "FIT_RECORDS.jsonl.gz"
    temporary_fit = output / ".FIT_RECORDS.jsonl.gz.tmp"
    with gzip.open(temporary_fit, "wt", encoding="utf-8", newline="\n") as handle:
        for record in fit_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary_fit, fit_path)
    audit.update(
        {
            "run_id": args.run_id,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "regulatory_features": {"path": str(source), "sha256": sha256(source)},
            "prediction": {
                "path": str(prediction_path),
                "sha256": sha256(prediction_path),
            },
            "fit_records": {
                "path": str(fit_path),
                "sha256": sha256(fit_path),
                "records": len(fit_records),
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "methylation_download_required_for_code_execution": False,
            "methylation_columns_may_be_typed_unavailable": True,
        }
    )
    atomic_json(output / "LINEAGE.json", audit)
    success = {
        "format": "CANCERLNCATLAS_V32_EPIGENETIC_EXPRESSION_HEAD_SUCCESS_V1",
        "status": audit["status"],
        "run_id": args.run_id,
        "prediction_sha256": audit["prediction"]["sha256"],
        "lineage_sha256": sha256(output / "LINEAGE.json"),
        "success_written_last": True,
    }
    atomic_json(output / "SUCCESS.json", success)
    print(json.dumps(success, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
