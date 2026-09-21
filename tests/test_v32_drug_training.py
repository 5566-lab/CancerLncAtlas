from __future__ import annotations

import copy
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.drug_training import (
    CORE_EXPORT_FORMAT,
    CROSS_DATASET_ASSOCIATION_POLICY,
    EXPRESSION_MOMENT_POLICY,
    DatasetAssociationArrays,
    DrugTrainingConfig,
    DrugTrainingError,
    association_statistics,
    assign_cell_line_folds,
    combine_cross_dataset_associations,
    normalise_cell_line_map,
    normalise_raw_response,
    run_drug_training,
)
from cc_hhgt.v32.full_model_contract import validate_module_lineage, validate_public_module_frame
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.drug_training_streaming import (
    FACTORED_CANDIDATE_FORMAT,
    StreamingDrugTrainingConfig,
    _audit_factored_relation,
    _association_cache_audit,
    _association_statistics_vectorized,
    _build_context,
    _build_context_from_prepared,
    _disk_preflight,
    _expression_aware_candidate_count,
    _fold_bitmask_relation,
    _load_factored_definition,
    _prepare_association_cache,
    _preeligible_drugs,
    _response_fold_eligible_pairs,
    run_streaming_drug_training,
)
import cc_hhgt.v32.drug_training_streaming as drug_training_streaming
import cc_hhgt.v32.drug_staging as drug_staging
from cc_hhgt.v32.drug_staging import STAGING_FORMAT, TCGA_CANCERS


def _write_inputs(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(431)
    lncrnas = [f"LNC:ENSG{i:011d}" for i in range(6)]
    drugs = ["DRUG:D0", "DRUG:D1"]
    genes = ["GENE:ENSG00000000001", "GENE:ENSG00000000002"]
    exact = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "lncrna_id": lncrna, "pathway_id": pathway}
            for lncrna in lncrnas
            for pathway in ("P0", "P1")
        ]
    )
    paths: dict[str, Path] = {}
    paths["exact"] = root / "v32_exact_candidates.parquet"
    exact.to_parquet(paths["exact"], index=False)
    paths["candidates"] = root / "drug_candidates.parquet"
    pd.DataFrame(
        [
            {"cancer_id": "BRCA", "lncrna_id": lncrna, "drug_id": drug}
            for lncrna in lncrnas
            for drug in drugs
        ]
    ).to_parquet(paths["candidates"], index=False)

    mapping_rows: list[dict[str, object]] = []
    expression_rows: list[dict[str, object]] = []
    response_rows: list[dict[str, object]] = []
    noise = rng.normal(size=(60, 4))
    for dataset in ("GDSC_RAW", "PRISM_RAW"):
        for index in range(60):
            cell = f"{dataset}_CELL_{index:03d}"
            model = f"MODEL_{index:03d}"
            z = float(index) + (0.01 if dataset.startswith("GDSC") else -0.01)
            mapping_rows.append(
                {
                    "dataset_id": dataset,
                    "cell_line_id": cell,
                    "canonical_model_id": model,
                    "cancer_id": "BRCA",
                }
            )
            values = [z, -z, *noise[index]]
            for lncrna, value in zip(lncrnas, values, strict=True):
                expression_rows.append(
                    {
                        "dataset_id": dataset,
                        "cell_line_id": cell,
                        "lncrna_id": lncrna,
                        "expression_value": float(value),
                    }
                )
            response_rows.extend(
                [
                    {
                        "dataset_id": dataset,
                        "cell_line_id": cell,
                        "drug_id": "DRUG:D0",
                        "response_value": z,
                        "higher_is_sensitive": True,
                    },
                    {
                        "dataset_id": dataset,
                        "cell_line_id": cell,
                        "drug_id": "DRUG:D1",
                        "response_value": -z,
                        "higher_is_sensitive": True,
                    },
                ]
            )
    paths["mapping"] = root / "cell_line_map.parquet"
    paths["expression"] = root / "raw_lncrna_expression.parquet"
    paths["response"] = root / "raw_drug_response.parquet"
    pd.DataFrame(mapping_rows).to_parquet(paths["mapping"], index=False)
    pd.DataFrame(expression_rows).to_parquet(paths["expression"], index=False)
    pd.DataFrame(response_rows).to_parquet(paths["response"], index=False)
    paths["targets"] = root / "curated_drug_gene_target.parquet"
    pd.DataFrame(
        [
            {"drug_id": "DRUG:D0", "gene_id": genes[0]},
            {"drug_id": "DRUG:D1", "gene_id": genes[1]},
        ]
    ).to_parquet(paths["targets"], index=False)
    paths["curated"] = root / "curated_response_observations.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": lncrnas[0],
                "drug_id": "DRUG:D0",
                "pmid": "12345",
                "source_database": "ncRNADrug",
                "resistance_or_sensitivity": "sensitivity",
            }
        ]
    ).to_parquet(paths["curated"], index=False)

    core_root = root / "core"
    core_root.mkdir()
    manifest: dict[str, object] = {
        "export_format": CORE_EXPORT_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "module_id": "exact_pathway",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": {},
    }
    for fold in range(5):
        fold_root = core_root / f"patient_fold={fold}"
        fold_root.mkdir()
        fold_rng = np.random.default_rng(900 + fold)
        exports: dict[str, dict[str, object]] = {}
        for node_type, ids in (("lncRNA", lncrnas), ("gene", genes), ("cancer", ["BRCA"])):
            frame = pd.DataFrame(
                fold_rng.normal(size=(len(ids), 4)).astype(np.float32),
                columns=[f"core_feature_{index:03d}" for index in range(4)],
            )
            frame.insert(0, "node_id", ids)
            frame.insert(0, "node_index", range(len(ids)))
            path = fold_root / f"{node_type}.parquet"
            frame.to_parquet(path, index=False)
            exports[node_type] = {
                "path": str(path),
                "sha256": artifact_sha256(path),
                "rows": len(frame),
                "features": 4,
            }
        manifest["folds"][str(fold)] = {  # type: ignore[index]
            "patient_fold": fold,
            "checkpoint_sha256": f"{fold + 1:x}" * 64,
            "core_parameter_sha256": f"{fold + 6:x}" * 64,
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "exports": exports,
        }
    paths["core"] = root / "CORE_MANIFEST.json"
    paths["core"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths


def _canonical_json_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _snapshot_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if resolved.is_file():
        size = int(resolved.stat().st_size)
        artifact_type = "file"
    else:
        size = int(
            sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
        )
        artifact_type = "directory"
    return {
        "path": str(resolved),
        "artifact_type": artifact_type,
        "bytes": size,
        "sha256": artifact_sha256(resolved),
    }


def _unchanged_snapshot(paths: Mapping[str, Path]) -> dict[str, object]:
    start = {
        role: _snapshot_record(path)
        for role, path in sorted(paths.items())
    }
    return {
        "start": start,
        "end": copy.deepcopy(start),
        "unchanged": True,
    }


def _write_valid_exact_release_lineage(
    *, candidate: Path, prediction: Path, lineage: Path
) -> None:
    prediction_sha = artifact_sha256(prediction)
    prediction_rows = len(pd.read_parquet(prediction))
    source_records = []
    for fold in range(5):
        token = str(fold + 1)
        source_records.append(
            {
                "patient_fold": fold,
                "seed": 101 + fold,
                "prediction_path": f"fold_{fold}/prediction.parquet",
                "prediction_sha256": token * 64,
                "prediction_rows": prediction_rows,
                "metrics_path": f"fold_{fold}/metrics.json",
                "metrics_sha256": str(fold + 2) * 64,
                "checkpoint_path": f"fold_{fold}/checkpoint.pt",
                "checkpoint_sha256": str(fold + 3) * 64,
                "success_path": f"fold_{fold}/SUCCESS.json",
                "success_sha256": str(fold + 4) * 64,
                "candidate_alignment": "FULL_ROW_EXACT",
                "checkpoint_cycle": 1,
                "completed_cycles": 2,
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "trained_from_random_initialization": True,
                "old_checkpoint_loaded": False,
                "old_predictions_used_as_features": False,
                "old_rankings_used_as_outputs": False,
            }
        )
    payload = {
        "module_id": "exact_pathway",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "training_run_id": "v32-exact-pathway-five-fold-test",
        "training_status": "SUCCESS",
        "initialization_policy": "TRAIN_FROM_RANDOM_INITIALIZATION",
        "folds": 5,
        "seeds": [101, 102, 103, 104, 105],
        "code_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "input_manifest_sha256": "c" * 64,
        "checkpoint_manifest_sha256": "d" * 64,
        "input_artifacts": [
            {
                "path": str(candidate.resolve()),
                "sha256": artifact_sha256(candidate),
                "artifact_kind": "standardized_input",
            }
        ],
        "trained_from_scratch": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "prediction_sha256": prediction_sha,
        "prediction_rows": prediction_rows,
        "ensemble_source_manifest_path": "ENSEMBLE_SOURCE_MANIFEST.json",
        "ensemble_source_manifest_sha256": "e" * 64,
        "ensemble_source_records": source_records,
        "ensemble_columns": [
            "association_membership_probability",
            "association_direction_probability",
            "l1_probability",
            "ridge_probability",
            "graph_residual",
            "graph_gate",
        ],
        "ensemble_formula": "arithmetic_mean_of_five_patient_fold_predictions",
    }
    validate_module_lineage("exact_pathway", payload)
    lineage.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_formal_staging_manifest(
    *,
    root: Path,
    paths: Mapping[str, Path],
    formal_exact: Path,
    factor_root: Path,
    edge: Path,
    silent_path: Path,
    mapping_coverage: Mapping[str, object],
) -> tuple[Path, Path, Path]:
    prediction = root / "exact_pathway_five_fold_ensemble.parquet"
    pd.read_parquet(formal_exact)[
        ["cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"]
    ].to_parquet(prediction, index=False)
    exact_lineage = root / "EXACT_MODULE_LINEAGE.json"
    _write_valid_exact_release_lineage(
        candidate=formal_exact, prediction=prediction, lineage=exact_lineage
    )

    # These are the twelve roles snapshotted by a formal native staging run.
    # Tiny tests intentionally reuse immutable files across native roles.
    input_roles = {
        "gdsc1_native": paths["response"],
        "gdsc2_native": paths["response"],
        "prism_matrix_native": paths["response"],
        "prism_compound_native": paths["targets"],
        "cmp_expression_native": paths["expression"],
        "cmp_models_native": paths["mapping"],
        "drugcentral_target_native": paths["targets"],
        "hgnc_native": paths["targets"],
        "v32_exact_candidates": formal_exact,
        "exact_pathway_membership": edge,
        "exact_release_prediction_proof": prediction,
        "exact_release_lineage_proof": exact_lineage,
    }
    input_snapshot = _unchanged_snapshot(input_roles)
    project_root = Path(__file__).resolve().parents[1]
    code_snapshot = _unchanged_snapshot(
        {
            "drug_staging_module": project_root / "cc_hhgt" / "v32" / "drug_staging.py",
            "execution_entrypoint_0": project_root / "scripts" / "build_v32_drug_raw_inputs.py",
        }
    )
    runtime_fingerprint = drug_staging._runtime_fingerprint()
    runtime_sha = _canonical_json_sha(runtime_fingerprint)
    row_count = len(pd.read_parquet(formal_exact))
    exact_binding = {
        "status": "PASS",
        "binding_policy": "LITERAL_FOUR_KEY_BIDIRECTIONAL_EXCEPT_AND_ROW_UNIQUENESS",
        "key_columns": [
            "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"
        ],
        "candidate_path": str(formal_exact.resolve()),
        "candidate_sha256": artifact_sha256(formal_exact),
        "candidate_rows": row_count,
        "candidate_unique_keys": row_count,
        "release_prediction_path": str(prediction.resolve()),
        "release_prediction_sha256": artifact_sha256(prediction),
        "release_prediction_rows": row_count,
        "release_prediction_unique_keys": row_count,
        "release_lineage_path": str(exact_lineage.resolve()),
        "release_lineage_sha256": artifact_sha256(exact_lineage),
        "lineage_training_run_id": "v32-exact-pathway-five-fold-test",
        "lineage_newly_trained_v32": True,
        "candidate_null_keys": 0,
        "prediction_null_keys": 0,
        "candidate_minus_prediction": 0,
        "prediction_minus_candidate": 0,
    }
    provenance = {
        "status": "PASS",
        "input_artifacts": input_snapshot,
        "execution_code": code_snapshot,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_sha,
        "exact_candidate_source_binding": exact_binding,
    }
    provenance_path = root / "STAGING_PROVENANCE_SNAPSHOT.json"
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    staging_artifacts = {
        "raw_cell_line_drug_response": paths["response"],
        "raw_cell_line_lncrna_expression": paths["expression"],
        "cell_line_map": paths["mapping"],
        "drug_gene_target": paths["targets"],
        "drug_candidate_universe": factor_root,
    }
    manifest = {
        "staging_format": STAGING_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "staging_run_id": "v32-drug-native-staging-test",
        "status": "SUCCESS",
        "formal": True,
        "training_ready": True,
        "candidate_cancers": list(TCGA_CANCERS),
        "native_assayed_to_target_mapping_coverage": dict(mapping_coverage),
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "artifacts": {
            role: {"path": str(path.resolve()), "sha256": artifact_sha256(path)}
            for role, path in staging_artifacts.items()
        },
        "input_artifact_snapshot": input_snapshot,
        "input_artifacts_unchanged": True,
        "execution_code_snapshot": code_snapshot,
        "execution_code_unchanged": True,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_sha,
        "exact_candidate_source_binding": exact_binding,
        "staging_provenance_snapshot_path": str(provenance_path.resolve()),
        "staging_provenance_snapshot_sha256": artifact_sha256(provenance_path),
    }
    manifest_path = root / "STAGING_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path, prediction, exact_lineage


def _write_streaming_formal_fixture(root: Path) -> dict[str, Any]:
    paths = _write_inputs(root)
    formal_exact = root / "v32_exact_candidates_33c.parquet"
    local_exact = pd.read_parquet(paths["exact"])
    local_exact["pathway_family_id"] = local_exact.pathway_id.map(
        {"P0": "PF0", "P1": "PF1"}
    )
    pd.concat(
        [local_exact.assign(cancer_id=cancer) for cancer in TCGA_CANCERS],
        ignore_index=True,
    ).to_parquet(formal_exact, index=False)

    extra_drugs = [f"DRUG:EXTRA_{index:04d}" for index in range(2, 1000)]
    response = pd.read_parquet(paths["response"])
    response = pd.concat(
        [
            response,
            pd.DataFrame(
                {
                    "dataset_id": "GDSC_RAW",
                    "cell_line_id": "GDSC_RAW_CELL_000",
                    "drug_id": extra_drugs,
                    "response_value": 0.5,
                    "higher_is_sensitive": True,
                }
            ),
        ],
        ignore_index=True,
    )
    response.to_parquet(paths["response"], index=False)
    targets = pd.read_parquet(paths["targets"])
    targets = pd.concat(
        [
            targets,
            pd.DataFrame(
                {
                    "drug_id": extra_drugs,
                    "gene_id": "GENE:ENSG00000000001",
                }
            ),
        ],
        ignore_index=True,
    )
    targets.to_parquet(paths["targets"], index=False)

    factor_root = root / "factored_candidates"
    edge_root = factor_root / "_factors"
    edge_root.mkdir(parents=True)
    edge = edge_root / "pathway_drug_native_assayed.parquet"
    pd.DataFrame(
        [
            {"pathway_id": "P0", "drug_id": "DRUG:D0"},
            {"pathway_id": "P1", "drug_id": "DRUG:D1"},
        ]
    ).to_parquet(edge, index=False)
    definition = {
        "candidate_format": FACTORED_CANDIDATE_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "storage_mode": "factored",
        "target_keys": ["cancer_id", "lncrna_id", "drug_id"],
        "conceptual_candidate_rows": 396,
        "candidate_cancers": 33,
        "pathway_drug_edge_path": str(edge),
        "pathway_drug_edge_sha256": artifact_sha256(edge),
        "pathway_drug_edge_rows": 2,
        "exact_candidate_path": str(formal_exact),
        "exact_candidate_sha256": artifact_sha256(formal_exact),
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
    }
    (factor_root / "FACTORED_UNIVERSE.json").write_text(
        json.dumps(definition), encoding="utf-8"
    )
    silent_path = root / "native_assayed_without_target_mapping.parquet"
    pd.DataFrame(
        columns=["dataset_id", "source_drug_id", "drug_id", "drug_name"]
    ).to_parquet(silent_path, index=False)
    mapping_coverage = {
        "identity_scope": (
            "EXACT_DRUG_NAME_AFTER_UPPERCASE_ALPHANUMERIC_NORMALIZATION_"
            "NO_SYNONYM_CROSSWALK"
        ),
        "candidate_scope": "TARGET_ANNOTATED_NATIVE_ASSAYED_DRUGS",
        "synonym_crosswalk_applied": False,
        "native_assayed_unique_drugs": 1000,
        "native_assayed_with_target_mapping": 1000,
        "native_assayed_without_target_mapping_silent_nonmatch": 0,
        "mapping_fraction": 1.0,
        "minimum_mapped_drugs": 1000,
        "minimum_mapping_fraction": 0.15,
        "threshold_pass": True,
        "silent_nonmatch_sidecar_path": str(silent_path),
        "silent_nonmatch_sidecar_sha256": artifact_sha256(silent_path),
        "silent_nonmatch_sidecar_rows": 0,
    }
    manifest, prediction, exact_lineage = _write_formal_staging_manifest(
        root=root,
        paths=paths,
        formal_exact=formal_exact,
        factor_root=factor_root,
        edge=edge,
        silent_path=silent_path,
        mapping_coverage=mapping_coverage,
    )
    return {
        "paths": paths,
        "formal_exact": formal_exact,
        "factor_root": factor_root,
        "edge": edge,
        "staging_manifest": manifest,
        "exact_prediction": prediction,
        "exact_lineage": exact_lineage,
    }


def _expected_provenance_hashes(fixture: Mapping[str, Any]) -> dict[str, str]:
    return {
        "expected_staging_manifest_sha256": artifact_sha256(
            fixture["staging_manifest"]
        ),
        "expected_exact_release_prediction_sha256": artifact_sha256(
            fixture["exact_prediction"]
        ),
        "expected_exact_release_lineage_sha256": artifact_sha256(
            fixture["exact_lineage"]
        ),
    }


def _run_streaming_fixture(
    fixture: Mapping[str, Any],
    output: Path,
    *,
    expected_hashes: Mapping[str, str | None] | None = None,
    preflight_only: bool = True,
) -> dict[str, Any]:
    paths = fixture["paths"]
    hashes = dict(expected_hashes or {})
    return run_streaming_drug_training(
        exact_candidates_path=fixture["formal_exact"],
        factored_drug_candidates_path=fixture["factor_root"],
        raw_drug_response_path=paths["response"],
        raw_lncrna_expression_path=paths["expression"],
        cell_line_map_path=paths["mapping"],
        drug_gene_target_path=paths["targets"],
        curated_drug_response_path=paths["curated"],
        core_embedding_manifest_path=paths["core"],
        output_root=output,
        training_run_id=(
            "v32-drug-streaming-preflight-test"
            if preflight_only else "v32-drug-streaming-test"
        ),
        config=StreamingDrugTrainingConfig(
            seed=19,
            epochs=2,
            batch_size=32,
            hidden_features=8,
            patience=2,
            min_cell_lines_per_dataset=3,
            min_total_cell_lines=6,
            positive_abs_rho=0.85,
            negative_abs_rho=0.60,
            max_train_rows=100,
            max_validation_rows=100,
            prediction_batch_size=64,
            candidate_chunk_rows=3,
            private_evaluation_rows_per_fold=10,
            preflight_only=preflight_only,
        ),
        **hashes,
    )


def _rewrite_manifest_provenance(
    fixture: Mapping[str, Any], manifest: dict[str, Any]
) -> None:
    provenance = {
        "status": "PASS",
        "input_artifacts": manifest["input_artifact_snapshot"],
        "execution_code": manifest["execution_code_snapshot"],
        "runtime_fingerprint": manifest["runtime_fingerprint"],
        "runtime_fingerprint_sha256": manifest["runtime_fingerprint_sha256"],
        "exact_candidate_source_binding": manifest[
            "exact_candidate_source_binding"
        ],
    }
    provenance_path = Path(manifest["staging_provenance_snapshot_path"])
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    manifest["staging_provenance_snapshot_sha256"] = artifact_sha256(
        provenance_path
    )
    Path(fixture["staging_manifest"]).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def _synchronise_mutated_exact_proof(fixture: Mapping[str, Any]) -> None:
    """Refresh legitimate proof hashes while preserving spoofed PASS counts."""

    prediction = Path(fixture["exact_prediction"])
    lineage_path = Path(fixture["exact_lineage"])
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["prediction_sha256"] = artifact_sha256(prediction)
    lineage["prediction_rows"] = len(pd.read_parquet(prediction))
    lineage_path.write_text(json.dumps(lineage, sort_keys=True), encoding="utf-8")

    manifest_path = Path(fixture["staging_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for role, path in (
        ("exact_release_prediction_proof", prediction),
        ("exact_release_lineage_proof", lineage_path),
    ):
        record = _snapshot_record(path)
        manifest["input_artifact_snapshot"]["start"][role] = record
        manifest["input_artifact_snapshot"]["end"][role] = copy.deepcopy(record)
    binding = manifest["exact_candidate_source_binding"]
    binding["release_prediction_sha256"] = artifact_sha256(prediction)
    binding["release_lineage_sha256"] = artifact_sha256(lineage_path)
    _rewrite_manifest_provenance(fixture, manifest)


def test_rejects_old_aggregated_gdsc_prism_association() -> None:
    old = pd.DataFrame(
        {
            "dataset_id": ["PRISM"],
            "cell_line_id": ["not_really_a_line"],
            "drug_id": ["D0"],
            "response_value": [0.2],
            "higher_is_sensitive": [True],
            "fdr": [0.01],
            "rho": [0.8],
        }
    )
    with pytest.raises(DrugTrainingError, match="old/result-like"):
        normalise_raw_response(old)


def test_canonical_models_never_cross_folds() -> None:
    mapping = normalise_cell_line_map(
        pd.DataFrame(
            [
                {
                    "dataset_id": dataset,
                    "cell_line_id": f"{dataset}_{model}",
                    "canonical_model_id": model,
                    "cancer_id": "BRCA",
                }
                for dataset in ("GDSC", "PRISM")
                for model in [f"M{i}" for i in range(10)]
            ]
        )
    )
    split = assign_cell_line_folds(mapping, seed=7)
    assert split.groupby("canonical_model_id").cell_line_fold_id.nunique().max() == 1
    assert set(split.cell_line_fold_id) == set(range(5))


def test_association_statistics_keep_labels_private_and_missing_explicit(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    mapping = assign_cell_line_folds(normalise_cell_line_map(pd.read_parquet(paths["mapping"])), 11)
    response = normalise_raw_response(pd.read_parquet(paths["response"]))
    expression = pd.read_parquet(paths["expression"])
    from cc_hhgt.v32.drug_training import normalise_raw_expression

    expression = normalise_raw_expression(expression)
    candidates = pd.read_parquet(paths["candidates"])
    candidates = pd.concat(
        [candidates, pd.DataFrame([{"cancer_id": "BRCA", "lncrna_id": "LNC:MISSING", "drug_id": "DRUG:D0"}])],
        ignore_index=True,
    )
    stats = association_statistics(
        candidates,
        expression,
        response,
        mapping,
        mapping.canonical_model_id.unique(),
        {"DRUG:D0": 1, "DRUG:D1": 1},
        {"DRUG:D0": True, "DRUG:D1": True},
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=6,
        positive_abs_rho=0.85,
        negative_abs_rho=0.60,
    )
    assert stats.assay_available[:-1].all()
    assert stats.labels[0] == 1.0
    assert not stats.assay_available[-1]
    assert stats.reasons[-1] == "NO_DATASET_WITH_SUFFICIENT_MATCHED_CELL_LINES"


def test_five_fold_fresh_training_contract_and_public_scope(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "fresh_drug_output"
    result = run_drug_training(
        exact_candidates_path=paths["exact"],
        drug_candidates_path=paths["candidates"],
        raw_drug_response_path=paths["response"],
        raw_lncrna_expression_path=paths["expression"],
        cell_line_map_path=paths["mapping"],
        drug_gene_target_path=paths["targets"],
        curated_drug_response_path=paths["curated"],
        core_embedding_manifest_path=paths["core"],
        output_root=output,
        training_run_id="v32-drug-test",
        config=DrugTrainingConfig(
            seed=19,
            epochs=2,
            batch_size=32,
            hidden_features=8,
            patience=2,
            min_cell_lines_per_dataset=3,
            min_total_cell_lines=6,
            positive_abs_rho=0.85,
            negative_abs_rho=0.60,
            max_train_rows=100,
            max_validation_rows=100,
            prediction_batch_size=64,
        ),
    )
    assert result["status"] == "DEVELOPMENT_SUCCESS_NOT_RELEASEABLE"
    assert result["formal_release_eligible"] is False
    assert result["release_ready"] is False
    assert result["partial_not_publishable"] is True
    assert not (output / "SUCCESS.json").exists()
    assert (output / "DEVELOPMENT_SUCCESS.json").is_file()
    public = pd.read_parquet(result["prediction_path"])
    assert len(public) == 12
    assert public[list(("cancer_id", "lncrna_id", "drug_id"))].duplicated().sum() == 0
    assert public.changes_primary_ranking.eq(False).all()
    assert public.tcga_patient_response_claimed.eq(False).all()
    assert public.evidence_scope.eq("CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE").all()
    assert public.scientific_status.eq("diagnostic_only").all()
    assert "association_proxy_label" not in public
    assert public.curated_evidence_used_as_model_feature.eq(False).all()
    validate_public_module_frame("drug", public)
    lineage = json.loads((output / "MODULE_LINEAGE.json").read_text(encoding="utf-8"))
    validate_module_lineage("drug", lineage)
    assert lineage["old_gdsc_prism_association_tables_used"] is False
    assert lineage["private_head_trained_from_scratch"] is True
    assert len(lineage["fold_status"]) == 5
    assert {row["status"] for row in lineage["fold_status"]} == {"SUCCESS"}
    assert lineage["cell_line_response_never_claimed_as_patient_response"] is True
    assert lineage["formal_release_eligible"] is False
    assert lineage["dense_development_only"] is True


def test_training_run_id_must_be_lowercase_v32(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(DrugTrainingError, match="lowercase"):
        run_drug_training(
            exact_candidates_path=paths["exact"],
            drug_candidates_path=paths["candidates"],
            raw_drug_response_path=paths["response"],
            raw_lncrna_expression_path=paths["expression"],
            cell_line_map_path=paths["mapping"],
            drug_gene_target_path=paths["targets"],
            core_embedding_manifest_path=paths["core"],
            output_root=tmp_path / "bad_run_id",
            training_run_id="V32_BAD",
        )


def test_cli_help_names_raw_inputs() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_v32_drug_training.py"
    text = script.read_text(encoding="utf-8")
    assert "--raw-drug-response" in text
    assert "--raw-lncrna-expression" in text
    assert "--core-embedding-manifest" in text


def test_streaming_factored_training_is_sparse_atomic_and_fresh(tmp_path: Path) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    expected_hashes = _expected_provenance_hashes(fixture)
    output = tmp_path / "streaming_output"
    preflight_output = tmp_path / "streaming_preflight"
    preflight = _run_streaming_fixture(
        fixture,
        preflight_output,
        expected_hashes=expected_hashes,
    )
    assert preflight["status"] == "PREFLIGHT_PASS"
    assert preflight["cross_dataset_association_policy"] == (
        CROSS_DATASET_ASSOCIATION_POLICY
    )
    assert preflight["expression_moment_policy"] == EXPRESSION_MOMENT_POLICY
    assert preflight["heads_trained"] == 0
    assert preflight["predictions_written"] == 0
    assert preflight["core_target_preflight_audit"][
        "all_five_core_fold_bundles_loaded"
    ] is True
    assert len(preflight["core_target_preflight_audit"]["fold_records"]) == 5
    assert all(
        record["required_drugs_with_core_target"] == 1000
        for record in preflight["core_target_preflight_audit"]["fold_records"]
    )
    assert preflight["context_cache_audit"]["full_table_prepare_calls"] == 1
    assert preflight["context_cache_audit"]["context_slice_calls"] == 0
    result = _run_streaming_fixture(
        fixture,
        output,
        expected_hashes=expected_hashes,
        preflight_only=False,
    )
    assert result["status"] == "SUCCESS"
    assert result["cross_dataset_association_policy"] == (
        CROSS_DATASET_ASSOCIATION_POLICY
    )
    assert result["expression_moment_policy"] == EXPRESSION_MOMENT_POLICY
    assert result["conceptual_candidate_rows"] == 396
    parts = sorted(Path(result["prediction_path"]).rglob("*.parquet"))
    assert parts
    public = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True)
    assert public.availability.eq(True).all()
    assert public.drug_response_association_probability.between(0, 1).all()
    assert not list(Path(result["prediction_path"]).rglob("*.tmp*"))
    lineage = json.loads((output / "MODULE_LINEAGE.json").read_text(encoding="utf-8"))
    validate_module_lineage("drug", lineage)
    assert lineage["cross_dataset_association_policy"] == (
        CROSS_DATASET_ASSOCIATION_POLICY
    )
    assert lineage["expression_moment_policy"] == EXPRESSION_MOMENT_POLICY
    assert lineage["legacy_dense_core_materialized"] is False
    assert lineage["canonical_model_never_crosses_folds"] is True
    assert lineage["absent_key_means_unavailable_not_zero"] is True
    assert lineage["old_gdsc_prism_association_tables_used"] is False
    assert lineage["max_candidate_core_rows_resident"] == 3
    assert lineage["all_five_folds_have_optimizer_updates"] is True
    assert lineage["optimizer_steps_total"] > 0
    assert lineage["runtime_fingerprint"]["python"]
    assert lineage["factored_relation_audit"]["declarations_match"] is True
    assert lineage["native_assayed_to_target_mapping_coverage"][
        "native_assayed_with_target_mapping"
    ] == 1000
    assert lineage["core_target_preflight_audit"][
        "all_five_core_fold_bundles_loaded"
    ] is True
    assert lineage["context_cache_audit"]["full_table_prepare_calls"] == 1
    assert lineage["context_cache_audit"]["context_slice_calls"] > 0
    assert lineage["context_cache_audit"][
        "legacy_full_table_context_build_calls"
    ] == 0


@pytest.mark.parametrize(
    ("hash_name", "replacement"),
    [
        ("expected_staging_manifest_sha256", None),
        ("expected_exact_release_prediction_sha256", None),
        ("expected_exact_release_lineage_sha256", None),
        ("expected_staging_manifest_sha256", "0" * 64),
        ("expected_exact_release_prediction_sha256", "0" * 64),
        ("expected_exact_release_lineage_sha256", "0" * 64),
    ],
)
def test_formal_streaming_requires_exact_expected_provenance_hashes(
    tmp_path: Path, hash_name: str, replacement: str | None
) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    expected = _expected_provenance_hashes(fixture)
    expected[hash_name] = replacement  # type: ignore[assignment]
    with pytest.raises(DrugTrainingError, match="SHA|sha|hash"):
        _run_streaming_fixture(
            fixture,
            tmp_path / "expected_hash_attack",
            expected_hashes=expected,
        )


def test_formal_streaming_rejects_spoofed_snapshot_start_end_drift(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    manifest_path = Path(fixture["staging_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_artifact_snapshot"]["end"]["gdsc1_native"]["sha256"] = (
        "0" * 64
    )
    # Keep the separate provenance payload byte-for-byte consistent with the
    # forged manifest; the validator must still compare start and end itself.
    _rewrite_manifest_provenance(fixture, manifest)
    with pytest.raises(DrugTrainingError, match="start/end snapshots differ"):
        _run_streaming_fixture(
            fixture,
            tmp_path / "snapshot_drift_attack",
            expected_hashes=_expected_provenance_hashes(fixture),
        )


def test_formal_streaming_rejects_provenance_payload_disagreement(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    manifest_path = Path(fixture["staging_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance_path = Path(manifest["staging_provenance_snapshot_path"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime_fingerprint"] = {"forged": True}
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True), encoding="utf-8"
    )
    manifest["staging_provenance_snapshot_sha256"] = artifact_sha256(
        provenance_path
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(DrugTrainingError, match="does not exactly match"):
        _run_streaming_fixture(
            fixture,
            tmp_path / "provenance_payload_attack",
            expected_hashes=_expected_provenance_hashes(fixture),
        )


def test_formal_streaming_rejects_simplified_exact_lineage_attestation(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    prediction = Path(fixture["exact_prediction"])
    fake_lineage = {
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "module_id": "exact_pathway",
        "training_status": "SUCCESS",
        "trained_from_scratch": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "prediction_sha256": artifact_sha256(prediction),
        "prediction_rows": len(pd.read_parquet(prediction)),
        "training_run_id": "v32-exact-pathway-five-fold-test",
    }
    Path(fixture["exact_lineage"]).write_text(
        json.dumps(fake_lineage, sort_keys=True), encoding="utf-8"
    )
    _synchronise_mutated_exact_proof(fixture)
    with pytest.raises(DrugTrainingError, match="full V3.2 contract"):
        _run_streaming_fixture(
            fixture,
            tmp_path / "simplified_lineage_attack",
            expected_hashes=_expected_provenance_hashes(fixture),
        )


@pytest.mark.parametrize("attack", ["four_key_mismatch", "duplicate_key"])
def test_formal_streaming_recomputes_exact_four_key_binding(
    tmp_path: Path, attack: str
) -> None:
    fixture = _write_streaming_formal_fixture(tmp_path)
    prediction_path = Path(fixture["exact_prediction"])
    prediction = pd.read_parquet(prediction_path)
    if attack == "four_key_mismatch":
        prediction.loc[0, "pathway_family_id"] = "PF_FORGED"
    else:
        prediction.iloc[1] = prediction.iloc[0]
    prediction.to_parquet(prediction_path, index=False)
    # All three externally pinned hashes and the staging PASS declaration are
    # refreshed after tampering. Only an independent four-key recomputation can
    # detect the mismatch/duplicate.
    _synchronise_mutated_exact_proof(fixture)
    manifest = json.loads(
        Path(fixture["staging_manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["exact_candidate_source_binding"]["status"] == "PASS"
    with pytest.raises(
        DrugTrainingError,
        match="binding declaration drifted|four-key binding failed",
    ):
        _run_streaming_fixture(
            fixture,
            tmp_path / f"{attack}_output",
            expected_hashes=_expected_provenance_hashes(fixture),
        )


def test_vectorized_statistics_match_reference_without_fold_leakage(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    mapping = assign_cell_line_folds(
        normalise_cell_line_map(pd.read_parquet(paths["mapping"])), 19
    )
    response = normalise_raw_response(pd.read_parquet(paths["response"]))
    from cc_hhgt.v32.drug_training import normalise_raw_expression

    expression = normalise_raw_expression(pd.read_parquet(paths["expression"]))
    candidates = pd.read_parquet(paths["candidates"])
    models = mapping.loc[mapping.cell_line_fold_id.eq(0), "canonical_model_id"].unique()
    context = _build_context(expression, response, mapping, models, "BRCA")
    config = StreamingDrugTrainingConfig(
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=6,
        positive_abs_rho=0.85,
        negative_abs_rho=0.60,
    )
    observed = _association_statistics_vectorized(
        candidates, context,
        {"DRUG:D0": 1, "DRUG:D1": 1},
        {"DRUG:D0": True, "DRUG:D1": True},
        config,
    )
    expected = association_statistics(
        candidates, expression, response, mapping, models,
        {"DRUG:D0": 1, "DRUG:D1": 1},
        {"DRUG:D0": True, "DRUG:D1": True},
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=6,
        positive_abs_rho=0.85,
        negative_abs_rho=0.60,
    )
    np.testing.assert_allclose(observed.rho, expected.rho, equal_nan=True, atol=1e-6)
    np.testing.assert_allclose(observed.domain[:, 1:], expected.domain[:, 1:], atol=1e-6)
    np.testing.assert_array_equal(observed.labels, expected.labels)
    assert set(observed.n_cell_lines) == {len(models)}
    assert set(expected.n_cell_lines) == {len(models)}
    held_out = set(map(str, models))
    assert not held_out.intersection(
        mapping.loc[mapping.cell_line_fold_id.ne(0), "canonical_model_id"].astype(str)
    )


def _association_record(
    dataset: str,
    models: list[str],
    expression: list[float],
    sensitivity: list[float],
) -> DatasetAssociationArrays:
    return DatasetAssociationArrays(
        dataset_id=dataset,
        canonical_model_ids=np.asarray(models, dtype=object),
        expression=np.asarray(expression, dtype=float)[:, None],
        sensitivity=np.asarray(sensitivity, dtype=float),
    )


def test_complete_overlap_uses_unique_model_consensus() -> None:
    models = [f"M{i}" for i in range(6)]
    values = [1, 2, 3, 4, 5, 6]
    summary = combine_cross_dataset_associations(
        [
            _association_record("GDSC", models, values, values),
            _association_record("PRISM", models, values, values),
        ],
        candidate_count=1,
        min_cell_lines_per_dataset=3,
    )
    assert summary.has_cross_dataset_overlap[0]
    assert summary.n_cell_lines[0] == 6
    assert summary.dataset_count[0] == 2
    assert summary.rho[0] == pytest.approx(1.0)
    assert summary.expression_mean[0] == pytest.approx(3.5)
    assert summary.expression_sd[0] == pytest.approx(np.std(values))


def test_complete_overlap_opposite_screens_has_undefined_consensus() -> None:
    models = [f"M{i}" for i in range(6)]
    values = [1, 2, 3, 4, 5, 6]
    summary = combine_cross_dataset_associations(
        [
            _association_record("GDSC", models, values, values),
            _association_record("PRISM", models, values, list(reversed(values))),
        ],
        candidate_count=1,
        min_cell_lines_per_dataset=3,
    )
    assert summary.n_cell_lines[0] == 6
    assert summary.dataset_count[0] == 2
    assert np.isnan(summary.rho[0])


def test_partial_overlap_matches_rank_consensus_and_unique_expression_moments() -> None:
    summary = combine_cross_dataset_associations(
        [
            _association_record(
                "GDSC", ["A", "B", "C", "D"], [10, 20, 30, 40], [1, 2, 3, 4]
            ),
            _association_record(
                "PRISM", ["C", "D", "E", "F"], [50, 70, 80, 90], [4, 1, 3, 2]
            ),
        ],
        candidate_count=1,
        min_cell_lines_per_dataset=3,
    )
    expression_consensus = np.asarray([10, 20, 40, 55, 80, 90], dtype=float)
    sensitivity_consensus = np.asarray(
        [0.125, 0.375, 0.75, 0.50, 0.625, 0.375], dtype=float
    )
    expected_rho = np.corrcoef(
        pd.Series(expression_consensus).rank(method="average"),
        pd.Series(sensitivity_consensus).rank(method="average"),
    )[0, 1]
    assert summary.n_cell_lines[0] == 6
    assert summary.rho[0] == pytest.approx(expected_rho)
    assert summary.expression_mean[0] == pytest.approx(expression_consensus.mean())
    assert summary.expression_sd[0] == pytest.approx(expression_consensus.std())


def test_overlap_consensus_prevents_old_fisher_threshold_crossing() -> None:
    from cc_hhgt.v32.drug_training_streaming import _AssociationContext

    context = _AssociationContext(
        "BRCA",
        ("GDSC", "PRISM"),
        {
            "GDSC": pd.DataFrame(
                {"LNC:X": [10, 20, 30, 40]}, index=["A", "B", "C", "D"]
            ),
            "PRISM": pd.DataFrame(
                {"LNC:X": [50, 70, 80, 90]}, index=["C", "D", "E", "F"]
            ),
        },
        {
            ("GDSC", "DRUG:X"): pd.Series([1, 2, 3, 4], index=["A", "B", "C", "D"]),
            ("PRISM", "DRUG:X"): pd.Series([4, 1, 3, 2], index=["C", "D", "E", "F"]),
        },
    )
    stats = _association_statistics_vectorized(
        pd.DataFrame(
            [{"cancer_id": "BRCA", "lncrna_id": "LNC:X", "drug_id": "DRUG:X"}]
        ),
        context,
        {"DRUG:X": 1},
        {"DRUG:X": True},
        StreamingDrugTrainingConfig(
            min_cell_lines_per_dataset=3,
            min_total_cell_lines=6,
            positive_abs_rho=0.80,
            negative_abs_rho=0.10,
        ),
    )
    old_fisher = np.tanh(
        (np.arctanh(0.999999) + np.arctanh(-0.4)) / 2.0
    )
    assert old_fisher > 0.80
    assert stats.rho[0] == pytest.approx(0.376842, abs=1e-5)
    assert np.isnan(stats.labels[0])


def test_disjoint_datasets_preserve_old_fisher_and_pooled_moments() -> None:
    first_x = [1, 2, 3, 4]
    second_x = [10, 20, 30, 40, 50]
    summary = combine_cross_dataset_associations(
        [
            _association_record("GDSC", ["A", "B", "C", "D"], first_x, first_x),
            _association_record(
                "PRISM", ["E", "F", "G", "H", "I"], second_x, list(reversed(second_x))
            ),
        ],
        candidate_count=1,
        min_cell_lines_per_dataset=3,
    )
    expected = np.tanh(
        (
            np.arctanh(0.999999) * 1.0
            + np.arctanh(-0.999999) * 2.0
        )
        / 3.0
    )
    pooled = np.asarray(first_x + second_x, dtype=float)
    assert not summary.has_cross_dataset_overlap[0]
    assert summary.n_cell_lines[0] == 9
    assert summary.rho[0] == pytest.approx(expected)
    assert summary.expression_mean[0] == pytest.approx(pooled.mean())
    assert summary.expression_sd[0] == pytest.approx(pooled.std())


def test_prepared_context_cache_is_one_pass_and_reference_equivalent(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    mapping = assign_cell_line_folds(
        normalise_cell_line_map(pd.read_parquet(paths["mapping"])), 19
    )
    response = normalise_raw_response(pd.read_parquet(paths["response"]))
    from cc_hhgt.v32.drug_training import normalise_raw_expression
    from cc_hhgt.v32.drug_sparse_query import factor_expression_fold_coverage

    expression = normalise_raw_expression(pd.read_parquet(paths["expression"]))
    candidates = pd.read_parquet(paths["candidates"])
    config = StreamingDrugTrainingConfig(
        seed=19,
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=6,
        positive_abs_rho=0.85,
        negative_abs_rho=0.60,
    )
    prepared = _prepare_association_cache(
        expression, response, mapping, config
    )
    reference_eligibility = _response_fold_eligible_pairs(
        response, mapping, config
    ).sort_values(
        ["cancer_id", "drug_id", "cell_line_fold_id"], kind="stable"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        prepared.fold_eligible.sort_values(
            ["cancer_id", "drug_id", "cell_line_fold_id"], kind="stable"
        ).reset_index(drop=True),
        reference_eligibility,
        check_dtype=False,
    )
    reference_coverage = factor_expression_fold_coverage(
        expression,
        mapping,
        min_cell_lines_per_dataset=3,
        min_total_cell_lines=6,
    )
    pd.testing.assert_frame_equal(
        prepared.expression_fold_coverage,
        reference_coverage,
        check_dtype=False,
    )

    fold = 0
    validation_fold = (fold + 1) % 5
    model_sets = {
        "test": mapping.loc[
            mapping.cell_line_fold_id.eq(fold), "canonical_model_id"
        ].astype(str).drop_duplicates().tolist(),
        "validation": mapping.loc[
            mapping.cell_line_fold_id.eq(validation_fold), "canonical_model_id"
        ].astype(str).drop_duplicates().tolist(),
        "train": mapping.loc[
            ~mapping.cell_line_fold_id.isin([fold, validation_fold]),
            "canonical_model_id",
        ].astype(str).drop_duplicates().tolist(),
    }
    for models in model_sets.values():
        expected_context = _build_context(
            expression, response, mapping, models, "BRCA"
        )
        observed_context = _build_context_from_prepared(
            prepared, models, "BRCA"
        )
        assert observed_context.datasets == expected_context.datasets
        assert set(observed_context.response_by_dataset_drug) == set(
            expected_context.response_by_dataset_drug
        )
        for dataset in observed_context.datasets:
            pd.testing.assert_frame_equal(
                observed_context.expression_by_dataset[dataset],
                expected_context.expression_by_dataset[dataset],
                check_like=True,
                check_dtype=False,
            )
        for key in observed_context.response_by_dataset_drug:
            pd.testing.assert_series_equal(
                observed_context.response_by_dataset_drug[key].sort_index(),
                expected_context.response_by_dataset_drug[key].sort_index(),
                check_dtype=False,
            )
        observed = _association_statistics_vectorized(
            candidates,
            observed_context,
            {"DRUG:D0": 1, "DRUG:D1": 1},
            {"DRUG:D0": True, "DRUG:D1": True},
            config,
        )
        expected = _association_statistics_vectorized(
            candidates,
            expected_context,
            {"DRUG:D0": 1, "DRUG:D1": 1},
            {"DRUG:D0": True, "DRUG:D1": True},
            config,
        )
        np.testing.assert_allclose(
            observed.domain, expected.domain, equal_nan=True, atol=1e-7
        )
        np.testing.assert_allclose(
            observed.rho, expected.rho, equal_nan=True, atol=1e-7
        )
        np.testing.assert_array_equal(observed.labels, expected.labels)
        np.testing.assert_array_equal(
            observed.assay_available, expected.assay_available
        )
        np.testing.assert_array_equal(observed.reasons, expected.reasons)
        np.testing.assert_array_equal(
            observed.n_cell_lines, expected.n_cell_lines
        )

    audit = _association_cache_audit(prepared)
    assert audit["full_table_prepare_calls"] == 1
    assert audit["response_full_table_mapping_links"] == 1
    assert audit["expression_full_table_mapping_links"] == 1
    assert audit["context_slice_calls"] == 3
    assert audit["legacy_full_table_context_build_calls"] == 0
    assert audit[
        "response_or_expression_full_table_reprocessed_per_context"
    ] is False

    canonical_expression = expression.merge(
        mapping[[
            "dataset_id", "cell_line_id", "canonical_model_id"
        ]].drop_duplicates(),
        on=["dataset_id", "cell_line_id"],
        how="inner",
        validate="many_to_one",
    ).groupby(
        ["canonical_model_id", "lncrna_id"], observed=True, sort=False
    ).expression_value.mean().reset_index()
    canonical_prepared = _prepare_association_cache(
        canonical_expression, response, mapping, config
    )
    canonical_models = model_sets["test"]
    canonical_expected = _build_context(
        canonical_expression, response, mapping, canonical_models, "BRCA"
    )
    canonical_observed = _build_context_from_prepared(
        canonical_prepared, canonical_models, "BRCA"
    )
    assert canonical_observed.datasets == canonical_expected.datasets
    for dataset in canonical_observed.datasets:
        pd.testing.assert_frame_equal(
            canonical_observed.expression_by_dataset[dataset],
            canonical_expected.expression_by_dataset[dataset],
            check_like=True,
            check_dtype=False,
        )
    assert canonical_prepared.instrumentation["expression_mode"] == (
        "CANONICAL_MODEL"
    )


@pytest.fixture
def fold_mask_candidate_fixture(tmp_path: Path) -> dict[str, Any]:
    exact = pd.DataFrame(
        [
            ("BRCA", "LNC:L1", "P1"),
            ("BRCA", "LNC:L1", "P2"),
            ("BRCA", "LNC:L2", "P1"),
            ("BRCA", "LNC:L3", "P2"),
            ("LUAD", "LNC:L1", "P1"),
            ("LUAD", "LNC:L4", "P3"),
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )
    edge = pd.DataFrame(
        [
            ("P1", "DRUG:D1"),
            ("P2", "DRUG:D1"),
            ("P2", "DRUG:D2"),
            ("P2", "DRUG:D2"),
            ("P3", "DRUG:D3"),
        ],
        columns=["pathway_id", "drug_id"],
    )
    assay = pd.DataFrame(
        [
            ("BRCA", "DRUG:D1", 0),
            ("BRCA", "DRUG:D1", 1),
            ("BRCA", "DRUG:D1", 1),
            ("BRCA", "DRUG:D2", 2),
            ("LUAD", "DRUG:D1", 4),
            ("LUAD", "DRUG:D3", 3),
        ],
        columns=["cancer_id", "drug_id", "cell_line_fold_id"],
    )
    expression = pd.DataFrame(
        [
            ("BRCA", "LNC:L1", 1),
            ("BRCA", "LNC:L1", 4),
            ("BRCA", "LNC:L2", 0),
            ("BRCA", "LNC:L3", 2),
            ("BRCA", "LNC:L3", 2),
            ("LUAD", "LNC:L1", 4),
            ("LUAD", "LNC:L4", 3),
        ],
        columns=["cancer_id", "lncrna_id", "fold_id"],
    )
    exact_path = tmp_path / "fold_mask_exact.parquet"
    edge_path = tmp_path / "fold_mask_edge.parquet"
    exact.to_parquet(exact_path, index=False)
    edge.to_parquet(edge_path, index=False)
    return {
        "exact": exact,
        "edge": edge,
        "exact_path": exact_path,
        "edge_path": edge_path,
        "assay": assay,
        "expression": expression,
        "resource_audit": {
            "status": "PASS",
            "duckdb_max_temp_directory_size_bytes": 256 * 1024**2,
            "duckdb_memory_limit_bytes": 256 * 1024**2,
            "duckdb_threads": 1,
        },
    }


def test_fold_mask_candidate_count_is_itemwise_legacy_join_equivalent(
    tmp_path: Path,
    fold_mask_candidate_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = fold_mask_candidate_fixture
    assay_by_fold = fixture["assay"].rename(
        columns={"cell_line_fold_id": "fold_id"}
    )
    legacy_keys = (
        fixture["exact"]
        .merge(fixture["edge"], on="pathway_id", how="inner")
        .merge(
            assay_by_fold,
            on=["cancer_id", "drug_id"],
            how="inner",
        )
        .merge(
            fixture["expression"],
            on=["cancer_id", "lncrna_id", "fold_id"],
            how="inner",
        )[["cancer_id", "lncrna_id", "drug_id"]]
        .drop_duplicates()
        .sort_values(["cancer_id", "lncrna_id", "drug_id"], kind="stable")
        .reset_index(drop=True)
    )

    assay_masks = _fold_bitmask_relation(
        fixture["assay"],
        key_columns=("cancer_id", "drug_id"),
        fold_column="cell_line_fold_id",
        mask_column="assay_fold_mask",
        relation_label="test assay",
    )
    expression_masks = _fold_bitmask_relation(
        fixture["expression"],
        key_columns=("cancer_id", "lncrna_id"),
        fold_column="fold_id",
        mask_column="expression_fold_mask",
        relation_label="test expression",
    )
    assert int(
        assay_masks.loc[
            assay_masks.cancer_id.eq("BRCA")
            & assay_masks.drug_id.eq("DRUG:D1"),
            "assay_fold_mask",
        ].iloc[0]
    ) == 0b00011
    assert int(
        expression_masks.loc[
            expression_masks.cancer_id.eq("BRCA")
            & expression_masks.lncrna_id.eq("LNC:L1"),
            "expression_fold_mask",
        ].iloc[0]
    ) == 0b10010
    masked = (
        fixture["exact"]
        .merge(fixture["edge"], on="pathway_id", how="inner")
        .merge(assay_masks, on=["cancer_id", "drug_id"], how="inner")
        .merge(
            expression_masks,
            on=["cancer_id", "lncrna_id"],
            how="inner",
        )
    )
    masked_keys = (
        masked.loc[
            np.bitwise_and(
                masked.assay_fold_mask.astype(np.uint8),
                masked.expression_fold_mask.astype(np.uint8),
            ).ne(0),
            ["cancer_id", "lncrna_id", "drug_id"],
        ]
        .drop_duplicates()
        .sort_values(["cancer_id", "lncrna_id", "drug_id"], kind="stable")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(masked_keys, legacy_keys)

    bounded_context_calls: list[Path] = []
    original_bounded_duckdb = drug_training_streaming._bounded_duckdb

    @contextmanager
    def observed_bounded_duckdb(
        output: Path, resource_audit: Mapping[str, Any]
    ):
        bounded_context_calls.append(output)
        with original_bounded_duckdb(output, resource_audit) as connection:
            yield connection

    monkeypatch.setattr(
        drug_training_streaming, "_bounded_duckdb", observed_bounded_duckdb
    )
    observed_count = _expression_aware_candidate_count(
        fixture["exact_path"],
        fixture["edge_path"],
        fixture["assay"],
        fixture["expression"],
        tmp_path / "fold_mask_count",
        fixture["resource_audit"],
    )
    assert observed_count == len(legacy_keys) == 5
    assert len(bounded_context_calls) == 2
    assert not (tmp_path / "fold_mask_count" / ".duckdb_tmp").exists()


def test_sparse_query_validation_uses_dedicated_two_thread_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = (
        "CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT",
        "CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS",
        "CC_HHGT_DRUG_SPARSE_DUCKDB_TEMP_DIRECTORY",
        "CC_HHGT_DRUG_SPARSE_DUCKDB_MAX_TEMP_DIRECTORY_SIZE",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    resource_audit = {
        "status": "PASS",
        "duckdb_memory_limit_bytes": 16 * 1024**3,
        "duckdb_threads": 8,
        "sparse_query_duckdb_threads": 2,
        "duckdb_max_temp_directory_size_bytes": 128 * 1024**3,
    }
    output = tmp_path / "sparse-query-thread-budget"
    with drug_training_streaming._sparse_query_resource_environment(
        output, resource_audit
    ):
        assert os.environ["CC_HHGT_DRUG_SPARSE_DUCKDB_THREADS"] == "2"
        assert os.environ["CC_HHGT_DRUG_SPARSE_DUCKDB_MEMORY_LIMIT"] == str(
            16 * 1024**3
        ) + "B"
    assert all(key not in os.environ for key in keys)
    assert not (output / ".sparse_query_duckdb_tmp").exists()


def test_fold_mask_candidate_count_parameterizes_cancer_and_skips_orphans(
    tmp_path: Path, fold_mask_candidate_fixture: Mapping[str, Any]
) -> None:
    exact = pd.DataFrame(
        [
            ("BR'CA", "LNC:Q1", "P1"),
            ("BR'CA", "LNC:Q1", "P2"),
            ("ASSAY_ONLY", "LNC:A1", "P1"),
            ("EXPRESSION_ONLY", "LNC:E1", "P1"),
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    )
    edge = pd.DataFrame(
        [("P1", "DRUG:D1"), ("P2", "DRUG:D1"), ("P2", "DRUG:D1")],
        columns=["pathway_id", "drug_id"],
    )
    assay = pd.DataFrame(
        [("BR'CA", "DRUG:D1", 2), ("ASSAY_ONLY", "DRUG:D1", 2)],
        columns=["cancer_id", "drug_id", "cell_line_fold_id"],
    )
    expression = pd.DataFrame(
        [("BR'CA", "LNC:Q1", 2), ("EXPRESSION_ONLY", "LNC:E1", 2)],
        columns=["cancer_id", "lncrna_id", "fold_id"],
    )
    exact_path = tmp_path / "quoted_cancer_exact.parquet"
    edge_path = tmp_path / "quoted_cancer_edge.parquet"
    exact.to_parquet(exact_path, index=False)
    edge.to_parquet(edge_path, index=False)

    assert _expression_aware_candidate_count(
        exact_path,
        edge_path,
        assay,
        expression,
        tmp_path / "quoted_cancer_count",
        fold_mask_candidate_fixture["resource_audit"],
    ) == 1


def test_fold_mask_candidate_count_second_partition_failure_is_atomic(
    tmp_path: Path,
    fold_mask_candidate_fixture: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = fold_mask_candidate_fixture
    original_bounded_duckdb = drug_training_streaming._bounded_duckdb
    context_calls = 0

    @contextmanager
    def fail_second_bounded_duckdb(
        output: Path, resource_audit: Mapping[str, Any]
    ):
        nonlocal context_calls
        context_calls += 1
        with original_bounded_duckdb(output, resource_audit) as connection:
            if context_calls == 2:
                raise RuntimeError("injected second cancer failure")
            yield connection

    monkeypatch.setattr(
        drug_training_streaming, "_bounded_duckdb", fail_second_bounded_duckdb
    )
    output = tmp_path / "second_partition_failure"
    with pytest.raises(RuntimeError, match="injected second cancer failure"):
        _expression_aware_candidate_count(
            fixture["exact_path"],
            fixture["edge_path"],
            fixture["assay"],
            fixture["expression"],
            output,
            fixture["resource_audit"],
        )
    assert context_calls == 2
    assert not (output / ".duckdb_tmp").exists()


@pytest.mark.parametrize(
    ("relation", "invalid_fold"),
    [
        ("assay", None),
        ("assay", -1),
        ("assay", 5),
        ("assay", 1.5),
        ("expression", None),
        ("expression", -1),
        ("expression", 5),
        ("expression", np.inf),
    ],
)
def test_fold_mask_candidate_count_rejects_invalid_fold_ids(
    tmp_path: Path,
    fold_mask_candidate_fixture: Mapping[str, Any],
    relation: str,
    invalid_fold: object,
) -> None:
    fixture = fold_mask_candidate_fixture
    assay = fixture["assay"].copy()
    expression = fixture["expression"].copy()
    fold_column = "cell_line_fold_id" if relation == "assay" else "fold_id"
    target = assay if relation == "assay" else expression
    target[fold_column] = target[fold_column].astype(object)
    target.loc[target.index[0], fold_column] = invalid_fold
    with pytest.raises(DrugTrainingError, match="fold ids"):
        _expression_aware_candidate_count(
            fixture["exact_path"],
            fixture["edge_path"],
            assay,
            expression,
            tmp_path / f"invalid_{relation}",
            fixture["resource_audit"],
        )


def test_streaming_disk_guard_is_fail_closed(tmp_path: Path) -> None:
    config = StreamingDrugTrainingConfig(
        estimated_public_bytes_per_row=10**12,
    )
    audit = _disk_preflight(tmp_path, sparse_candidate_upper_bound=1, config=config)
    assert audit["status"] == "FAIL"
    assert audit["estimated_peak_incremental_bytes"] > audit["available_after_reserve_bytes"]


def test_formal_streaming_resource_floors_cannot_be_disabled() -> None:
    with pytest.raises(ValueError, match="bytes_per_row"):
        StreamingDrugTrainingConfig(estimated_public_bytes_per_row=159).validate()
    with pytest.raises(ValueError, match="atomic_write_multiplier"):
        StreamingDrugTrainingConfig(atomic_write_multiplier=2.24).validate()
    with pytest.raises(ValueError, match="disk_reserve_bytes"):
        StreamingDrugTrainingConfig(disk_reserve_bytes=2 * 1024**3 - 1).validate()
    with pytest.raises(ValueError, match="disk_reserve_fraction"):
        StreamingDrugTrainingConfig(disk_reserve_fraction=0.149).validate()


def test_multi_dataset_eligibility_uses_canonical_model_union() -> None:
    expression_by_dataset = {
        dataset: pd.DataFrame(
            {"LNC:X": [1.0, 2.0, 3.0, 4.0]},
            index=["M0", "M1", "M2", "M3"],
        )
        for dataset in ("GDSC", "PRISM")
    }
    responses = {
        (dataset, "DRUG:X"): pd.Series(
            [1.0, 2.0, 3.0, 4.0], index=["M0", "M1", "M2", "M3"]
        )
        for dataset in ("GDSC", "PRISM")
    }
    from cc_hhgt.v32.drug_training_streaming import _AssociationContext

    context = _AssociationContext(
        "BRCA", ("GDSC", "PRISM"), expression_by_dataset, responses
    )
    config = StreamingDrugTrainingConfig(
        min_cell_lines_per_dataset=3, min_total_cell_lines=6
    )
    assert _preeligible_drugs(context, config) == set()


def test_fold_eligibility_does_not_double_count_same_canonical_model() -> None:
    response = pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "cell_line_id": f"{dataset}_{model}",
                "drug_id": "DRUG:X",
                "sensitivity_value": float(index),
            }
            for dataset in ("GDSC", "PRISM")
            for index, model in enumerate(("M0", "M1", "M2", "M3"))
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "dataset_id": dataset,
                "cell_line_id": f"{dataset}_{model}",
                "canonical_model_id": model,
                "cancer_id": "BRCA",
                "cell_line_fold_id": 0,
            }
            for dataset in ("GDSC", "PRISM")
            for model in ("M0", "M1", "M2", "M3")
        ]
    )
    config = StreamingDrugTrainingConfig(
        min_cell_lines_per_dataset=3, min_total_cell_lines=6
    )
    assert _response_fold_eligible_pairs(response, mapping, config).empty


def test_factored_relation_is_independently_recomputed(tmp_path: Path) -> None:
    exact = tmp_path / "exact.parquet"
    edge = tmp_path / "edge.parquet"
    pd.DataFrame(
        [
            {"cancer_id": cancer, "lncrna_id": "LNC:X", "pathway_id": "P0"}
            for cancer in TCGA_CANCERS
        ]
    ).to_parquet(exact, index=False)
    pd.DataFrame([{"pathway_id": "P0", "drug_id": "DRUG:X"}]).to_parquet(
        edge, index=False
    )
    output = tmp_path / "output"
    output.mkdir()
    resource = {
        "status": "PASS",
        "duckdb_max_temp_directory_size_bytes": 2 * 1024**3,
        "duckdb_memory_limit_bytes": 512 * 1024**2,
        "duckdb_threads": 1,
    }
    definition = {
        "pathway_drug_edge_rows": 1,
        "conceptual_candidate_rows": 34,
    }
    with pytest.raises(DrugTrainingError, match="DISTINCT row count"):
        _audit_factored_relation(definition, exact, edge, output, resource)


def test_failed_formal_run_writes_only_fail_closed_marker(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    factor_root = tmp_path / "bad_factor"
    factors = factor_root / "_factors"
    factors.mkdir(parents=True)
    edge = factors / "edge.parquet"
    pd.DataFrame([{"pathway_id": "P0", "drug_id": "DRUG:D0"}]).to_parquet(
        edge, index=False
    )
    (factor_root / "FACTORED_UNIVERSE.json").write_text(
        json.dumps(
            {
                "candidate_format": FACTORED_CANDIDATE_FORMAT,
                "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "storage_mode": "factored",
                "target_keys": ["cancer_id", "lncrna_id", "drug_id"],
                "candidate_cancers": 1,
                "conceptual_candidate_rows": 6,
                "pathway_drug_edge_path": str(edge),
                "pathway_drug_edge_sha256": artifact_sha256(edge),
                "pathway_drug_edge_rows": 1,
                "exact_candidate_path": str(paths["exact"]),
                "exact_candidate_sha256": artifact_sha256(paths["exact"]),
                "old_association_tables_read": False,
                "old_predictions_used": False,
                "old_checkpoints_used": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "failed_output"
    with pytest.raises(DrugTrainingError, match="33 cancers"):
        run_streaming_drug_training(
            exact_candidates_path=paths["exact"],
            factored_drug_candidates_path=factor_root,
            raw_drug_response_path=paths["response"],
            raw_lncrna_expression_path=paths["expression"],
            cell_line_map_path=paths["mapping"],
            drug_gene_target_path=paths["targets"],
            core_embedding_manifest_path=paths["core"],
            output_root=output,
            training_run_id="v32-failure-marker-test",
        )
    failed = json.loads((output / "RUN_FAILED.json").read_text(encoding="utf-8"))
    assert failed["release_ready"] is False
    assert failed["partial_not_publishable"] is True
    assert not (output / "SUCCESS.json").exists()
    assert not (output / "PREFLIGHT_SUCCESS.json").exists()
    assert not (output / "RUN_IN_PROGRESS.json").exists()
