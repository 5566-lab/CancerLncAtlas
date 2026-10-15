#!/usr/bin/env python3
"""Analyze all native-assayed drugs by cancer and annotated subtype, without training."""
from __future__ import annotations

import argparse
import json
import socket
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cc_hhgt.v32.drug_training import _read_table
from cc_hhgt.v32.drug_response_analysis import DrugResponseScanConfig, iter_response_scan
from cc_hhgt.v32.input_lineage import artifact_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["exact", "response", "expression", "mapping", "subtypes", "targets", "output"]:
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--min-models", type=int, default=8)
    parser.add_argument("--min-models-per-dataset", type=int, default=5)
    parser.add_argument("--chunk-rows", type=int, default=25000)
    parser.add_argument("--dose-manifest", type=Path,
        help="Source-bound observed point-dose measurements tied to the actual native staging")
    args = parser.parse_args(argv)
    if socket.gethostname() != "149":
        raise RuntimeError("Native drug response analysis requires target host 149")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = DrugResponseScanConfig(min_total_cell_lines=args.min_models,
        min_cell_lines_per_dataset=args.min_models_per_dataset, candidate_chunk_rows=args.chunk_rows)
    (args.output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "statistical_analysis": "DESCRIPTIVE_SIGNED_NATIVE_RESPONSE", "config": asdict(settings)}, indent=2))
    inputs = {key: {"path": str(getattr(args, key).resolve()), "sha256": artifact_sha256(getattr(args, key))}
              for key in ["exact", "response", "expression", "mapping", "subtypes", "targets"]}
    tables = {key: _read_table(Path(value["path"])) for key, value in inputs.items()}
    dose_data = None
    if args.dose_manifest is not None:
        from cc_hhgt.v32.drug_dose_response import load_native_point_response
        dose_data = load_native_point_response(args.dose_manifest,
            {key:Path(record["path"]) for key,record in inputs.items()},tables)
    coverage = {}; outputs = {}; writer = None; active = None; rows = 0
    result_root = args.output / "associations"; result_root.mkdir()
    try:
        for receipt, frame in iter_response_scan(**tables, config=settings):
            context_id = receipt["context_id"]
            coverage[context_id] = receipt
            if frame.empty: continue
            if active != context_id:
                if writer is not None: writer.close()
                active = context_id
                path = result_root / (context_id + ".parquet")
                writer = pq.ParquetWriter(path, pa.Table.from_pandas(frame, preserve_index=False).schema, compression="zstd")
            writer.write_table(pa.Table.from_pandas(frame, preserve_index=False).cast(writer.schema))
            rows += len(frame)
    finally:
        if writer is not None: writer.close()
    pd.DataFrame(coverage.values()).to_parquet(args.output / "CANCER_SUBTYPE_COVERAGE.parquet", index=False)
    dose_audit = None
    if dose_data is not None:
        from cc_hhgt.v32.drug_dose_response import materialize_dose_response
        points, identity_disposition, measurement_disposition, origin = dose_data
        dose_audit = {**origin, **materialize_dose_response(args.output,points,identity_disposition,
            measurement_disposition,tables,settings)}
        from cc_hhgt.v32.drug_pharmacology_native import verify_artifact
        for record in origin["sources"].values(): verify_artifact(record, args.dose_manifest.parent)
        for path in [*sorted((args.output/"dose_associations").glob("*.parquet")),
                     *sorted((args.output/"dose_contrasts").glob("*.parquet")),
                     *sorted(args.output.glob("*DOSE*.parquet"))]:
            outputs[path.relative_to(args.output).as_posix()] = {"path":str(path.resolve()),"sha256":artifact_sha256(path),
                "public_download_eligible":not path.name.startswith("PRIVATE_")}
    for path in [*sorted(result_root.glob("*.parquet")), args.output/"CANCER_SUBTYPE_COVERAGE.parquet"]:
        outputs[path.relative_to(args.output).as_posix()] = {"path": str(path.resolve()), "sha256": artifact_sha256(path)}
    for record in inputs.values():
        if artifact_sha256(record["path"]) != record["sha256"]: raise RuntimeError("Input changed during native response scan")
    audit = {"format": "NATIVE_DRUG_RESPONSE_CONTEXT_SCAN_V1", "status": "PASS", "target_host": "149",
        "training_started": False, "rows": rows, "contexts": len(coverage), "inputs": inputs, "outputs": outputs,
        "config": asdict(settings), "estimand": "MEASURED_CELL_LINE_EXPRESSION_SENSITIVITY_ASSOCIATION",
        "candidate_scope": "ALL_CANCER_ASSAYED_DRUGS_CROSSED_WITH_EXACT_LNCRNA_UNIVERSE",
        "requires_target_annotation": False, "core_model_prediction": False, "p_values_computed": False,
        "within_fold_statistics_are_fitted_predictions": False, "causal_claimed": False,
        "external_validation_claimed": False, "tcga_patient_treatment_response_claimed": False}
    audit["dose_response_analysis"] = dose_audit
    audit["code_sha256"] = {name:artifact_sha256(path) for name,path in {
        "entry":Path(__file__),"signed_scan":Path(__file__).resolve().parents[1]/"cc_hhgt/v32/drug_response_analysis.py",
        "dose_scan":Path(__file__).resolve().parents[1]/"cc_hhgt/v32/drug_dose_response.py"}.items()}
    (args.output / "AUDIT.json").write_text(json.dumps(audit, indent=2)+"\n")
    (args.output / "SUCCESS.json").write_text(json.dumps({"status": "PASS", "audit_sha256": artifact_sha256(args.output/"AUDIT.json")}))
    print(json.dumps({"status": "PASS", "rows": rows, "contexts": len(coverage), "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
