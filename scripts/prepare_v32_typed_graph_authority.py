#!/usr/bin/env python3
"""Publish a new typed, global physical-binding authority on host 149.

The JSON manifest pins source_receipt, patient_map, patient_receipt and a
binding_declaration (the same static-annotation declaration as graph receipts),
plus explicit fold_expression_pattern and fold_coexpression_pattern. All five
folds and static inputs are checked by the existing authority loader. No old
prepared payload, historical directory default or fitting is needed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import socket

import pandas as pd

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.formal_graph import materialize_global_typed_lnc_protein_binding
from cc_hhgt.v32.formal_graph_authority import (
    STATIC_ARTIFACT_IDS, RELATION_SCHEMA_GENERIC_GLOBAL, RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1,
    load_formal_graph_input_authority, _artifact_record,
)
from cc_hhgt.v32.patient_fold_authority import validate_frozen_v32_patient_fold_binding

INPUT = "TYPED_GLOBAL_GRAPH_AUTHORITY_INPUT_V1"
POLICY = "GLOBAL_PHYSICAL_BINDING"


def prepare(manifest: Path, output: Path) -> Path:
    if socket.gethostname() != "149":
        raise RuntimeError("Typed graph preparation requires host 149")
    manifest = manifest.resolve(strict=True)
    spec = read(manifest)
    if spec.get("format") != INPUT:
        raise ValueError("Invalid typed graph authority input format")
    if output.exists():
        raise FileExistsError("Refusing to overwrite a typed graph authority generation")
    output.mkdir(parents=True)
    write(output / "LAUNCH_PREFLIGHT.json", {
        "target_host": "149", "training_started": False, "baseline_fitting_started": False,
        "source_manifest": record(manifest),
    })
    source_path = verified_path(spec["source_receipt"], manifest.parent)
    source = read(source_path)
    audit = validate_frozen_v32_patient_fold_binding(
        verified_path(spec["patient_map"], manifest.parent),
        verified_path(spec["patient_receipt"], manifest.parent),
    )
    settings = dict(patient_authority_audit=audit,
        fold_expression_pattern=spec["fold_expression_pattern"],
        fold_coexpression_pattern=spec["fold_coexpression_pattern"])

    def load(path, document, generation):
        return load_formal_graph_input_authority(
            receipt_path=path, receipt_sha256=record(path)["sha256"],
            static_inputs={key: (document["artifacts"][key]["path"], document["artifacts"][key]["sha256"])
                           for key in STATIC_ARTIFACT_IDS},
            relation_schema_generation=generation, **settings)

    load(source_path, source, source.get("relation_schema_generation", RELATION_SCHEMA_GENERIC_GLOBAL))
    declaration = deepcopy(spec["binding_declaration"])
    binding_path = verified_path(declaration, manifest.parent)
    declaration["path"] = str(binding_path)
    _artifact_record(artifact_id="lnc_protein_binding", explicit_path=binding_path,
        explicit_sha256=declaration["sha256"], declaration=declaration,
        expected_source_role="static_annotation")
    table = pd.read_parquet(binding_path)
    edges = materialize_global_typed_lnc_protein_binding(table)
    document = deepcopy(source)
    document["relation_schema_generation"] = RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1
    document["binding_context_policy"] = POLICY
    document["artifacts"]["lnc_protein_binding"] = declaration
    # The source receipt remains the authority for its old counts; do not
    # relabel those totals as describing a replaced binding table.
    if "counts" in document:
        document["source_receipt_counts"] = document.pop("counts")
    document["gates"].update(same_node_and_relation_schema_all_variants=False,
                             typed_assay_class_relations=True)
    document["typed_binding_preparation"] = {
        "format": "TYPED_GLOBAL_GRAPH_AUTHORITY_PREPARATION_V1", "target_host": "149",
        "source_receipt": record(source_path), "source_manifest": record(manifest),
        "binding_source": record(binding_path), "binding_context_policy": POLICY,
        "source_rows": len(table), "experimental_edge_rows_before_candidate_filter": len(edges),
        "excluded_predicted_source_rows": int(table.graph_assay_class.astype(str).str.strip().eq("predicted").sum()),
        "predicted_binding_included": False, "training_started": False,
    }
    path = output / "GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
    write(path, document)
    load(path, document, RELATION_SCHEMA_TYPED_ASSAY_CLASS_V1)
    write(output / "SUCCESS.json", {
        "status": "PASS", "target_host": "149", "receipt": record(path),
        "preparation": document["typed_binding_preparation"],
    })
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.manifest, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
