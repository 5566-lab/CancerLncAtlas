#!/usr/bin/env python3
"""Smoke-test run_v32_evidence_streaming_training.py on a tiny fixture stage + core root."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.evidence_streaming_training import (  # noqa: E402
    EMPTY_EVENT_COLUMNS,
    materialize_streaming_evidence_stage,
)

TMP = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/model/_runner_smoke")
if TMP.exists():
    import shutil
    shutil.rmtree(TMP)
TMP.mkdir(parents=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def build_fixture(root: Path) -> dict[str, object]:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    pathways = [f"PATH-{i:02d}" for i in range(12)]
    lncs = [f"LNC-{i:02d}" for i in range(12)]
    genes = [f"GENE-{i:02d}" for i in range(12)]
    members = pd.DataFrame({
        "pathway_id": pathways + ["PATH-MULTI-A", "PATH-MULTI-B"],
        "gene_id": genes + ["GENE-MULTI", "GENE-MULTI"],
        "member_type": ["gene"] * 14,
        "mapping_status": ["mapped"] * 14,
    })
    candidates = pd.DataFrame({
        "cancer_id": ["BRCA"] * 12 + ["LUAD", "BRCA", "BRCA"],
        "lncrna_id": lncs + [lncs[0], "LNC-MULTI", "LNC-MULTI"],
        "pathway_id": pathways + [pathways[0], "PATH-MULTI-A", "PATH-MULTI-B"],
    })
    rows = []
    for index, (lnc, gene) in enumerate(zip(lncs, genes)):
        rows.append({
            "interaction_id": f"INT-{index}", "source_row_id": f"SRC-{index}",
            "source_row_index": str(index + 1),
            "source_row_sha256": hashlib.sha256(f"raw-{index}".encode()).hexdigest(),
            "source_sha256": "a" * 64, "source_record_id": f"REC-{index}",
            "source_database": "DB-SHARED" if index < 2 else f"DB-{index % 3}",
            "source_dataset": "DS-SHARED" if index < 2 else "DS-1",
            "cancer_id": "PAN_CANCER", "lncrna_id": lnc, "partner_id": gene,
            "lncrna_mapping_route": "unique_symbol", "lncrna_mapping_candidate_count": "1",
            "partner_mapping_route": "unique_symbol", "partner_mapping_candidate_count": "1",
            "mapping_status": "MAPPED_BOTH",
            "relation_type": "physical_binding" if index == 0 else "regulatory",
            "direction": "positive" if index % 2 == 0 else "negative",
            "experiment_family": "RNA pull-down" if index == 0 else "knockdown",
            "is_experimental": "true", "is_predicted": "false",
            "pmid": "PMID-SHARED" if index < 2 else f"PMID-{index}",
            "species": "human", "cell_line": None if index == 3 else "CELL", "tissue": "breast",
        })
    rows.append({**rows[0], "interaction_id": "INT-0-DUP", "source_row_id": "SRC-0-DUP"})
    rows.append({
        **rows[5], "interaction_id": "INT-MULTI", "source_row_id": "SRC-MULTI",
        "source_row_sha256": hashlib.sha256(b"raw-multi").hexdigest(),
        "source_record_id": "REC-MULTI", "lncrna_id": "LNC-MULTI", "partner_id": "GENE-MULTI",
        "pmid": "PMID-MULTI|PMID-SECOND",
    })
    rows.append({
        **rows[6], "interaction_id": "INT-DIRECT", "source_row_id": "SRC-DIRECT",
        "source_row_sha256": hashlib.sha256(b"raw-direct").hexdigest(),
        "source_record_id": "REC-DIRECT", "pathway_id": pathways[6],
    })
    rows[7]["lncrna_mapping_route"] = "ambiguous_alias"
    rows[7]["lncrna_mapping_candidate_count"] = "2"
    rows[7]["mapping_status"] = "AMBIGUOUS_LNC_PARTNER_MAPPED"
    rows.extend([
        {**rows[8], "interaction_id": "INT-UNMAPPED", "source_row_id": "SRC-UNMAPPED",
         "source_row_sha256": hashlib.sha256(b"raw-unmapped").hexdigest(),
         "source_record_id": "REC-UNMAPPED", "lncrna_id": "",
         "lncrna_mapping_route": "unresolved_alias", "lncrna_mapping_candidate_count": "0",
         "mapping_status": "LNC_UNMAPPED_PARTNER_MAPPED"},
        {**rows[9], "interaction_id": "INT-FAMILY", "source_row_id": "SRC-FAMILY",
         "source_row_sha256": hashlib.sha256(b"raw-family").hexdigest(),
         "source_record_id": "REC-FAMILY", "partner_id": "", "pathway_family_id": "FAMILY:IMMUNE"},
        {**rows[10], "interaction_id": "INT-OUTSIDE", "source_row_id": "SRC-OUTSIDE",
         "source_row_sha256": hashlib.sha256(b"raw-outside").hexdigest(),
         "source_record_id": "REC-OUTSIDE", "pathway_id": "PATH-NOT-A-CANDIDATE", "partner_id": ""},
    ])
    relation = pd.DataFrame(rows)
    relation_path = inputs / "interaction_relation_recovered.parquet"
    event_path = inputs / "evidence_event_empty_by_contract.parquet"
    member_path = inputs / "exact_members.parquet"
    candidate_path = inputs / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    _write_parquet(relation, relation_path)
    schema = pa.schema([pa.field(c, pa.string()) for c in EMPTY_EVENT_COLUMNS])
    pq.write_table(pa.Table.from_pylist([], schema=schema), event_path)
    _write_parquet(members, member_path)
    _write_parquet(candidates, candidate_path)
    patient = pd.DataFrame({
        "cancer_id": ["BRCA"] * 5 + ["LUAD"] * 5,
        "patient_id": [f"PAT-{i}" for i in range(10)],
        "patient_fold_id": list(range(5)) * 2,
        "fold_seed": [20260726] * 10,
    })
    patient_path = inputs / "PATIENT_FOLD_AUTHORITY.tsv"
    patient.to_csv(patient_path, sep="\t", index=False)
    patient_receipt = {
        "format": "CANCERLNCATLAS_V32_PATIENT_FIRST_FOLD_RECEIPT_V1",
        "n_folds": 5,
        "artifacts": {"patient_fold_authority": {
            "filename": patient_path.name, "sha256": _sha(patient_path)}},
        "gates": {"all_five_folds_per_cancer": True, "patient_cross_cancer_count_zero": True,
                  "patient_cross_fold_count_zero": True, "output_reuse_forbidden": True},
        "observed": {"cancers": 2, "patients": 10},
    }
    patient_receipt_path = inputs / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    patient_receipt_path.write_text(json.dumps(patient_receipt), encoding="utf-8")
    conversion = {
        "format": "CANCERLNCATLAS_V32_EVIDENCE_RECOVERED_PARQUET_CONVERSION_V1",
        "status": "PASS_LOSSLESS_PARQUET_CONVERSION",
        "contract": {"independent_validation_required_and_bound": True},
        "counts": {"rows": len(relation)},
        "outputs": {
            "interaction_relation": {"path": str(relation_path.resolve()), "sha256": _sha(relation_path)},
            "evidence_event": {"path": str(event_path.resolve()), "sha256": _sha(event_path)},
        },
    }
    conversion_path = inputs / "CONVERSION_MANIFEST.json"
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    paths = {
        "interaction_relation": relation_path, "empty_evidence_event": event_path,
        "exact_pathway_members": member_path, "formal_candidates": candidate_path,
        "conversion_manifest": conversion_path, "patient_fold_authority": patient_path,
        "patient_fold_receipt": patient_receipt_path,
    }
    return {"paths": paths, "expected_hashes": {role: _sha(p) for role, p in paths.items()}}


def build_core_root(root: Path) -> None:
    rng = np.random.default_rng(7)
    node_sets = {"cancer": ["BRCA", "LUAD"], "lncRNA": [f"LNC-{i:02d}" for i in range(12)],
                 "pathway": [f"PATH-{i:02d}" for i in range(12)]}
    folds_payload = {}
    for fold in range(5):
        fold_dir = root / f"patient_fold={fold}"
        fold_dir.mkdir(parents=True)
        exports = {}
        for node_type, ids in node_sets.items():
            frame = pd.DataFrame({
                "node_id": ids,
                **{f"core_feature_{j}": rng.normal(size=len(ids)).round(4) for j in range(4)},
            })
            path = fold_dir / f"{node_type}.parquet"
            _write_parquet(frame, path)
            exports[node_type] = {"path": str(path.relative_to(root)), "sha256": _sha(path)}
        checkpoint = hashlib.sha256(f"ckpt-{fold}".encode()).hexdigest()
        params = hashlib.sha256(f"params-{fold}".encode()).hexdigest()
        folds_payload[str(fold)] = {
            "patient_fold": fold, "trained_from_scratch": True,
            "old_checkpoint_loaded": False,
            "checkpoint_format": "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1",
            "checkpoint_sha256": checkpoint, "core_parameter_sha256": params,
            "exports": exports,
        }
    manifest = {
        "export_format": "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": folds_payload,
    }
    (root / "CORE_EMBEDDING_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8")


def main() -> int:
    fixture = build_fixture(TMP)
    stage = TMP / "stage"
    materialize_streaming_evidence_stage(
        interaction_relation_path=fixture["paths"]["interaction_relation"],
        empty_evidence_event_path=fixture["paths"]["empty_evidence_event"],
        exact_pathway_members_path=fixture["paths"]["exact_pathway_members"],
        formal_candidates_path=fixture["paths"]["formal_candidates"],
        conversion_manifest_path=fixture["paths"]["conversion_manifest"],
        patient_fold_authority_path=fixture["paths"]["patient_fold_authority"],
        patient_fold_receipt_path=fixture["paths"]["patient_fold_receipt"],
        output_root=stage, expected_hashes=fixture["expected_hashes"],
        strict_formal=False, batch_size=3, row_group_size=4,
        memory_limit="512MB", threads=1,
    )
    print("stage materialized:", stage)
    core = TMP / "core"
    build_core_root(core)
    out = TMP / "training_out"
    runner = ROOT / "scripts/run_v32_evidence_streaming_training.py"
    py = sys.executable
    result = subprocess.run(
        [py, str(runner), "--stage-root", str(stage), "--core-embedding-root", str(core),
         "--output-root", str(out), "--device", "cpu", "--epochs", "2", "--patience", "2",
         "--max-bags-per-batch", "8"],
        capture_output=True, text=True, check=False,
    )
    print(result.stdout[-4000:])
    print(result.stderr[-2000:])
    manifest_path = out / "TRAINING_MANIFEST.json"
    if result.returncode != 0 or not manifest_path.is_file():
        print("SMOKE_FAILED", result.returncode)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print("SMOKE_OK status=", manifest["status"], "folds=", len(manifest["folds"]))
    for fold, rec in manifest["folds"].items():
        print(f"  fold {fold}: steps={rec['optimizer_steps']} train={rec['train_examples']} "
              f"eval={rec['evaluation_examples']} missing={rec['train_missing_core'] + rec['evaluation_missing_core']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
