from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.full_model_contract import (
    validate_module_lineage,
    validate_public_module_frame,
)
from cc_hhgt.v32.genomic_training import (
    ASSAY_CALLABLE_SENTINEL,
    CORE_EXPORT_FORMAT,
    GenomicTrainingConfig,
    GenomicTrainingError,
    build_exact_pathway_calls,
    candidate_statistics,
    normalise_cnv_calls,
    normalise_candidates,
    run_genomic_training,
    segment_calls_from_raw,
)
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.genomic_partition_training import (
    collect_examples_partitioned,
    run_partitioned_genomic_training,
)
from cc_hhgt.v32.genomic_resource_staging import stage_genomic_cancer_memmaps


def _write_inputs(root: Path) -> dict[str, Path]:
    cancers = ("BRCA", "COAD")
    lncrnas = tuple(f"L{index}" for index in range(8))
    pathways = tuple(f"P{index}" for index in range(4))
    genes = {pathway: (f"G{2 * index}", f"G{2 * index + 1}") for index, pathway in enumerate(pathways)}

    candidates = pd.DataFrame(
        [
            {"cancer_id": cancer, "lncrna_id": lncrna, "pathway_id": pathway}
            for cancer in cancers
            for lncrna in lncrnas
            for pathway in pathways
        ]
    )
    candidate_path = root / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False)

    fold_rows: list[dict[str, object]] = []
    gene_rows: list[dict[str, object]] = []
    lnc_rows: list[dict[str, object]] = []
    cnv_gene_rows: list[dict[str, object]] = []
    cnv_lnc_rows: list[dict[str, object]] = []
    for cancer in cancers:
        for patient_index in range(20):
            patient = f"{cancer}_S{patient_index:02d}"
            fold_rows.append(
                {
                    "cancer_id": cancer,
                    "sample_id": patient,
                    "patient_id": patient,
                    "patient_fold_id": patient_index % 5,
                }
            )
            parity = patient_index % 2
            for pathway_index, pathway in enumerate(pathways):
                event = bool(parity if pathway_index % 2 == 0 else 1 - parity)
                for gene in genes[pathway]:
                    gene_rows.append(
                        {
                            "cancer_id": cancer,
                            "patient_id": patient,
                            "gene_id": gene,
                            "is_mutated": event,
                            "mutation_count": int(event),
                            "absence_is_wildtype": not event,
                        }
                    )
                    if cancer == "BRCA":
                        cnv_gene_rows.append(
                            {
                                "cancer_id": cancer,
                                "patient_id": patient,
                                "gene_id": gene,
                                "cnv_value": 0.8 if event else 0.0,
                                "cnv_callable": True,
                            }
                        )
            for lnc_index, lncrna in enumerate(lncrnas):
                event = bool(parity if lnc_index % 2 == 0 else 1 - parity)
                lnc_rows.append(
                    {
                        "cancer_id": cancer,
                        "patient_id": patient,
                        "lncrna_id": lncrna,
                        "lncrna_any_mutation": event,
                        "lncrna_mutation_available": True,
                        "exonic_variant_count": int(event),
                        "absence_is_wildtype": not event,
                    }
                )
                if cancer == "BRCA":
                    cnv_lnc_rows.append(
                        {
                            "cancer_id": cancer,
                            "patient_id": patient,
                            "lncrna_id": lncrna,
                            "cnv_value": 0.8 if event else 0.0,
                            "cnv_callable": True,
                        }
                    )

    fold_path = root / "patient_folds.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    membership_path = root / "exact_membership.parquet"
    pd.DataFrame(
        [
            {"pathway_id": pathway, "gene_id": gene}
            for pathway, members in genes.items()
            for gene in members
        ]
    ).to_parquet(membership_path, index=False)
    gene_path = root / "sample_gene_mutation.parquet"
    lnc_path = root / "sample_lncrna_mutation.parquet"
    pd.DataFrame(gene_rows).to_parquet(gene_path, index=False)
    pd.DataFrame(lnc_rows).to_parquet(lnc_path, index=False)
    cnv_gene_path = root / "cnv_gene_calls.parquet"
    cnv_lnc_path = root / "cnv_lncrna_calls.parquet"
    pd.DataFrame(cnv_gene_rows).to_parquet(cnv_gene_path, index=False)
    pd.DataFrame(cnv_lnc_rows).to_parquet(cnv_lnc_path, index=False)
    mc3_path = root / "mc3.maf"
    mc3_path.write_text("cancer_id\tpatient_id\tgene_id\nBRCA\tBRCA_S00\tG0\n", encoding="utf-8")

    core_root = root / "core"
    core_root.mkdir()
    manifest: dict[str, object] = {
        "export_format": CORE_EXPORT_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "module_id": "exact_pathway",
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
        "folds": {},
    }
    for fold in range(5):
        fold_root = core_root / f"patient_fold={fold}"
        fold_root.mkdir()
        rng = np.random.default_rng(100 + fold)
        lnc_embedding = pd.DataFrame(
            rng.normal(size=(len(lncrnas), 4)).astype(np.float32),
            columns=[f"core_feature_{index:03d}" for index in range(4)],
        )
        lnc_embedding.insert(0, "node_id", lncrnas)
        lnc_embedding.insert(0, "node_index", range(len(lncrnas)))
        pathway_embedding = pd.DataFrame(
            rng.normal(size=(len(pathways), 4)).astype(np.float32),
            columns=[f"core_feature_{index:03d}" for index in range(4)],
        )
        pathway_embedding.insert(0, "node_id", pathways)
        pathway_embedding.insert(0, "node_index", range(len(pathways)))
        lnc_embedding_path = fold_root / "lncRNA.parquet"
        pathway_embedding_path = fold_root / "pathway.parquet"
        lnc_embedding.to_parquet(lnc_embedding_path, index=False)
        pathway_embedding.to_parquet(pathway_embedding_path, index=False)
        manifest["folds"][str(fold)] = {  # type: ignore[index]
            "patient_fold": fold,
            "checkpoint_sha256": f"{fold + 1:x}" * 64,
            "core_parameter_sha256": f"{fold + 6:x}" * 64,
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "exports": {
                "lncRNA": {
                    "path": str(lnc_embedding_path),
                    "sha256": artifact_sha256(lnc_embedding_path),
                    "rows": len(lncrnas),
                    "features": 4,
                },
                "pathway": {
                    "path": str(pathway_embedding_path),
                    "sha256": artifact_sha256(pathway_embedding_path),
                    "rows": len(pathways),
                    "features": 4,
                },
            },
        }
    manifest_path = core_root / "CORE_EMBEDDING_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "candidates": candidate_path,
        "folds": fold_path,
        "membership": membership_path,
        "gene_mutation": gene_path,
        "lnc_mutation": lnc_path,
        "cnv_gene": cnv_gene_path,
        "cnv_lnc": cnv_lnc_path,
        "mc3": mc3_path,
        "core_manifest": manifest_path,
    }


def _partition_records(candidates: pd.DataFrame, root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    row_start = 0
    for cancer, local in candidates.groupby("cancer_id", observed=True, sort=False):
        partition = root / f"cancer_id={cancer}"
        partition.mkdir(parents=True)
        path = partition / "part-0.parquet"
        local.to_parquet(path, index=False)
        records.append(
            {
                "cancer_id": str(cancer),
                "rows": len(local),
                "global_row_start": row_start,
                "global_row_stop_exclusive": row_start + len(local),
                "path": str(path),
                "sha256": artifact_sha256(path),
            }
        )
        row_start += len(local)
    return records


def _ordered_candidate_key_sha256(
    candidates: pd.DataFrame, cancers: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    for cancer in cancers:
        local = (
            candidates.loc[candidates.cancer_id.eq(cancer), keys]
            .astype(str)
            .sort_values(keys, kind="stable")
        )
        for row in local.itertuples(index=False, name=None):
            digest.update(("\t".join(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def _write_partition_resource_contract(
    root: Path, paths: dict[str, Path]
) -> tuple[Path, Path, str]:
    cancers = ("BRCA", "COAD")
    candidates = pd.read_parquet(paths["candidates"])
    rows_per_cancer = len(candidates) // len(cancers)
    cnv_success = root / "cnv_aggregate_SUCCESS.json"
    cnv_success.write_text(
        json.dumps(
            {
                "format": "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_V1",
                "status": "SUCCESS",
                "cancers": {cancer: {} for cancer in cancers},
                "required_cancers": list(cancers),
                "patient_folds": 5,
                "typed_unavailable_is_never_zero": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cnv_sha = artifact_sha256(cnv_success)
    budget = root / "partition_resource_budget.json"
    budget.write_text(
        json.dumps(
            {
                "format": "CC_HHGT_V3_2_GENOMIC_RESOURCE_BUDGET_V1",
                "resource_budget_id": "synthetic-partition-trainer-e2e",
                "input_bindings": {
                    "candidate_authority": {
                        "path": str(paths["candidates"].resolve()),
                        "sha256": artifact_sha256(paths["candidates"]),
                        "ordered_exact_candidate_key_sha256": (
                            _ordered_candidate_key_sha256(candidates, cancers)
                        ),
                    },
                    "cnv_streaming_aggregate_success": {
                        "path": str(cnv_success.resolve()),
                        "sha256_policy": "CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH",
                        "format": "CC_HHGT_V3_2_STREAMING_SEGMENT_CNV_V1",
                        "status": "SUCCESS",
                    },
                },
                "scope": {
                    "formal_cancers": list(cancers),
                    "candidate_rows": len(candidates),
                    "candidate_rows_per_cancer": rows_per_cancer,
                },
                "adapter_budget": {
                    "max_candidate_partition_rows": rows_per_cancer,
                    "max_declared_adapter_peak_rss_bytes": 2 * 1024**3,
                    "max_memmap_bytes": 512 * 1024**2,
                    "max_staging_disk_bytes": 5 * 1024**3,
                },
                "downstream_budget": {
                    "forbid_launch_without_resource_success": True,
                    "max_declared_process_rss_bytes": 12 * 1024**3,
                    "minimum_host_available_memory_bytes": 1,
                    "minimum_host_free_disk_bytes": 1,
                },
                "memmap_layout": {
                    "modalities": ["mutation", "cnv"],
                    "patient_folds": 5,
                    "probability_dtype": "float32",
                    "available_dtype": "bool",
                    "processed_dtype": "bool",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return budget, cnv_success, cnv_sha


def _stage_partition_resource(
    root: Path, paths: dict[str, Path], name: str
) -> Path:
    budget, cnv_gate, cnv_gate_sha = _write_partition_resource_contract(root, paths)
    stage_root = root / name
    stage_genomic_cancer_memmaps(
        candidate_authority_path=paths["candidates"],
        genomic_training_source_path=Path(__file__).resolve().parents[1]
        / "cc_hhgt"
        / "v32"
        / "genomic_training.py",
        cnv_streaming_success_path=cnv_gate,
        cnv_streaming_success_sha256=cnv_gate_sha,
        budget_contract_path=budget,
        output_root=stage_root,
        enforce_host_budget=False,
    )
    return stage_root


def test_partition_collector_is_byte_equal_to_original_global_collector(
    tmp_path: Path,
) -> None:
    from cc_hhgt.v32.genomic_training import (
        _collect_examples,
        _group_calls,
        _split_patients,
        _validate_core_manifest,
        build_exact_pathway_calls,
        load_fold_core_embeddings,
        normalise_cnv_calls,
        normalise_exact_membership,
        normalise_gene_mutation_calls,
        normalise_lncrna_mutation_calls,
        normalise_patient_folds,
    )

    paths = _write_inputs(tmp_path)
    candidates = normalise_candidates(pd.read_parquet(paths["candidates"]))
    records = _partition_records(candidates, tmp_path / "partitions")
    folds = normalise_patient_folds(pd.read_csv(paths["folds"]))
    membership = normalise_exact_membership(pd.read_parquet(paths["membership"]))
    fold_keys = folds[["cancer_id", "patient_id"]]
    gene_mutation = normalise_gene_mutation_calls(
        pd.read_parquet(paths["gene_mutation"])
    ).merge(fold_keys, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    lnc_mutation = normalise_lncrna_mutation_calls(
        pd.read_parquet(paths["lnc_mutation"])
    ).merge(fold_keys, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    pathway_mutation = build_exact_pathway_calls(gene_mutation, membership)
    gene_cnv = normalise_cnv_calls(
        pd.read_parquet(paths["cnv_gene"]), entity_kind="gene", event_threshold=0.3
    ).merge(fold_keys, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    lnc_cnv = normalise_cnv_calls(
        pd.read_parquet(paths["cnv_lnc"]), entity_kind="lncrna", event_threshold=0.3
    ).merge(fold_keys, on=["cancer_id", "patient_id"], how="inner", validate="many_to_one")
    pathway_cnv = build_exact_pathway_calls(gene_cnv, membership)
    sources = {
        "mutation": (_group_calls(lnc_mutation), _group_calls(pathway_mutation)),
        "cnv": (_group_calls(lnc_cnv), _group_calls(pathway_cnv)),
    }
    core_manifest, _, _ = _validate_core_manifest(paths["core_manifest"])
    core = load_fold_core_embeddings(paths["core_manifest"], core_manifest, 0)
    cases = (("train", 17, 500), ("validation", 50_017, 300))
    for modality, (lnc_by_cancer, pathway_by_cancer) in sources.items():
        for split, seed, maximum in cases:
            patients = _split_patients(folds, 0, split)
            expected = _collect_examples(
                candidates,
                lnc_by_cancer,
                pathway_by_cancer,
                patients,
                core,
                modality=modality,
                min_pair_callable=2,
                maximum=maximum,
                seed=seed,
            )
            observed = collect_examples_partitioned(
                records,
                lnc_by_cancer,
                pathway_by_cancer,
                patients,
                core,
                modality=modality,
                min_pair_callable=2,
                maximum=maximum,
                seed=seed,
            )
            for original, partitioned in zip(expected, observed, strict=True):
                assert original.dtype == partitioned.dtype
                assert original.shape == partitioned.shape
                assert original.tobytes(order="C") == partitioned.tobytes(order="C")


def test_partitioned_global_trainer_writes_ten_validation_selected_heads_and_no_oof_success(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    stage_root = _stage_partition_resource(
        tmp_path, paths, "partition_resource_stage"
    )
    stage_success_before = json.loads(
        (stage_root / "SUCCESS.json").read_text(encoding="utf-8")
    )
    output = tmp_path / "partition_training"
    result = run_partitioned_genomic_training(
        resource_staging_success_path=stage_root / "SUCCESS.json",
        patient_folds_path=paths["folds"],
        pathway_gene_membership_path=paths["membership"],
        sample_gene_mutation_path=paths["gene_mutation"],
        sample_lncrna_mutation_path=paths["lnc_mutation"],
        mc3_path=paths["mc3"],
        core_embedding_manifest_path=paths["core_manifest"],
        cnv_gene_calls_path=paths["cnv_gene"],
        cnv_lncrna_calls_path=paths["cnv_lnc"],
        allow_development_long_cnv=True,
        output_root=output,
        training_run_id="v32-partition-global-head-e2e",
        config=GenomicTrainingConfig(
            seed=23,
            epochs=2,
            batch_size=32,
            hidden_features=8,
            dropout=0.0,
            learning_rate=0.01,
            patience=1,
            min_pair_callable=2,
            max_train_rows=500,
            max_validation_rows=500,
            prediction_batch_size=64,
            cnv_event_threshold=0.3,
        ),
        enforce_host_budget=False,
    )
    assert result["status"] == "TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED"
    assert result["checkpoint_count"] == 10
    assert result["formal_genomic_oof_success_emitted"] is False
    assert result["release_ready"] is False
    assert (output / "TRAINING_SUCCESS.json").is_file()
    assert not (output / "SUCCESS.json").exists()
    assert (stage_root / "PREDICTIONS_READY.json").is_file()
    assert json.loads((stage_root / "SUCCESS.json").read_text(encoding="utf-8")) == (
        stage_success_before
    )
    assert stage_success_before["formal_training_started"] is False
    checkpoint_manifest = json.loads(
        (output / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["checkpoint_count"] == 10
    assert checkpoint_manifest["global_head_scope"] == (
        "ONE_CROSS_CANCER_HEAD_PER_MODALITY_AND_FOLD"
    )
    assert {
        (row["modality"], row["patient_fold"])
        for row in checkpoint_manifest["records"]
    } == {(modality, fold) for modality in ("mutation", "cnv") for fold in range(5)}
    assert all(
        row["checkpoint_selection_split"] == "validation"
        and row["scaler_fit_split"] == "train"
        and row["test_rows_consumed_by_fit"] == 0
        and row["source_checkpoint_sha256"] is None
        for row in checkpoint_manifest["records"]
    )
    split_policy = json.loads((output / "SPLIT_POLICY.json").read_text(encoding="utf-8"))
    assert split_policy["test_keys_or_labels_enter_scaling_early_stop_or_checkpoint"] is False
    assert len(split_policy["traces"]) == 10
    available = np.load(
        stage_root / "memmaps" / "fold_available.bool.npy", mmap_mode="r"
    )
    probability = np.load(
        stage_root / "memmaps" / "fold_probability.float32.npy", mmap_mode="r"
    )
    processed = np.load(
        stage_root / "memmaps" / "fold_processed.bool.npy", mmap_mode="r"
    )
    # Candidate partitions are BRCA then COAD; the synthetic long CNV calls
    # intentionally cover BRCA only.  COAD must stay typed-null, never zero.
    assert processed.all()
    assert not available[1, :, 32:].any()
    assert np.isnan(probability[1, :, 32:]).all()


def test_partitioned_trainer_fails_closed_on_unapproved_long_cnv_and_dirty_stage(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)
    stage_root = _stage_partition_resource(tmp_path, paths, "fail_closed_stage")
    common = {
        "resource_staging_success_path": stage_root / "SUCCESS.json",
        "patient_folds_path": paths["folds"],
        "pathway_gene_membership_path": paths["membership"],
        "sample_gene_mutation_path": paths["gene_mutation"],
        "sample_lncrna_mutation_path": paths["lnc_mutation"],
        "mc3_path": paths["mc3"],
        "core_embedding_manifest_path": paths["core_manifest"],
        "cnv_gene_calls_path": paths["cnv_gene"],
        "cnv_lncrna_calls_path": paths["cnv_lnc"],
        "training_run_id": "v32-partition-fail-closed",
        "config": GenomicTrainingConfig(epochs=1, min_pair_callable=2),
        "enforce_host_budget": False,
    }
    forbidden_output = tmp_path / "forbidden_long_mode"
    with pytest.raises(
        RuntimeError, match="requires the aggregate streaming CNV store"
    ):
        run_partitioned_genomic_training(
            **common,
            output_root=forbidden_output,
        )
    assert not forbidden_output.exists()
    assert not forbidden_output.with_name(f".{forbidden_output.name}.staging").exists()
    assert not (stage_root / "write_receipts").exists()

    processed_path = stage_root / "memmaps" / "fold_processed.bool.npy"
    processed = np.load(processed_path, mmap_mode="r+")
    processed[0, 0, 0] = True
    processed.flush()
    del processed
    dirty_output = tmp_path / "dirty_stage_output"
    with pytest.raises(RuntimeError, match="contains processed prediction rows"):
        run_partitioned_genomic_training(
            **common,
            allow_development_long_cnv=True,
            output_root=dirty_output,
        )
    assert not dirty_output.exists()
    assert not dirty_output.with_name(f".{dirty_output.name}.staging").exists()


def test_partitioned_trainer_forbids_train_loss_fallback_when_validation_is_one_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cc_hhgt.v32.genomic_partition_training as partition_training

    paths = _write_inputs(tmp_path)
    stage_root = _stage_partition_resource(tmp_path, paths, "one_class_stage")
    original_collector = partition_training.collect_examples_partitioned

    def force_one_class_validation(*args, **kwargs):
        core_values, domain_values, labels = original_collector(*args, **kwargs)
        if int(kwargs["seed"]) >= 50_000:
            labels = np.zeros_like(labels)
        return core_values, domain_values, labels

    monkeypatch.setattr(
        partition_training, "collect_examples_partitioned", force_one_class_validation
    )
    output = tmp_path / "one_class_validation_output"
    with pytest.raises(
        RuntimeError, match="train-loss checkpoint fallback is forbidden"
    ):
        partition_training.run_partitioned_genomic_training(
            resource_staging_success_path=stage_root / "SUCCESS.json",
            patient_folds_path=paths["folds"],
            pathway_gene_membership_path=paths["membership"],
            sample_gene_mutation_path=paths["gene_mutation"],
            sample_lncrna_mutation_path=paths["lnc_mutation"],
            mc3_path=paths["mc3"],
            core_embedding_manifest_path=paths["core_manifest"],
            cnv_gene_calls_path=paths["cnv_gene"],
            cnv_lncrna_calls_path=paths["cnv_lnc"],
            allow_development_long_cnv=True,
            output_root=output,
            training_run_id="v32-partition-one-class-validation",
            config=GenomicTrainingConfig(
                seed=23,
                epochs=1,
                batch_size=32,
                hidden_features=8,
                dropout=0.0,
                learning_rate=0.01,
                patience=1,
                min_pair_callable=2,
                max_train_rows=500,
                max_validation_rows=500,
                prediction_batch_size=64,
            ),
            enforce_host_budget=False,
        )
    assert not output.exists()
    assert not (output.with_name(f".{output.name}.staging") / "TRAINING_SUCCESS.json").exists()
    assert not (stage_root / "write_receipts").exists()


def test_exact_pathway_rebuild_never_turns_missing_gene_calls_into_wildtype() -> None:
    calls = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "patient_id": "A", "entity_id": "G1", "event": True, "callable": True, "burden": 1.0},
            {"cancer_id": "BRCA", "patient_id": "B", "entity_id": "G1", "event": False, "callable": True, "burden": 0.0},
            {"cancer_id": "BRCA", "patient_id": "C", "entity_id": "G1", "event": False, "callable": True, "burden": 0.0},
            {"cancer_id": "BRCA", "patient_id": "C", "entity_id": "G2", "event": False, "callable": True, "burden": 0.0},
        ]
    )
    membership = pd.DataFrame(
        [{"pathway_id": "P1", "gene_id": "G1"}, {"pathway_id": "P1", "gene_id": "G2"}]
    )
    result = build_exact_pathway_calls(calls, membership).set_index("patient_id")
    assert bool(result.loc["A", "callable"])
    assert bool(result.loc["A", "event"])
    assert not bool(result.loc["B", "callable"])
    assert np.isnan(result.loc["B", "burden"])
    assert bool(result.loc["C", "callable"])
    assert not bool(result.loc["C", "event"])

    with pytest.raises(GenomicTrainingError, match="Family-level"):
        build_exact_pathway_calls(
            calls, pd.DataFrame([{"pathway_family_id": "F1", "gene_id": "G1"}])
        )


def test_mc3_assay_callable_sentinel_is_explicit_and_propagates_to_exact_pathways() -> None:
    calls = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": "A",
                "entity_id": ASSAY_CALLABLE_SENTINEL,
                "event": False,
                "callable": True,
                "burden": 0.0,
            },
            {
                "cancer_id": "BRCA",
                "patient_id": "B",
                "entity_id": ASSAY_CALLABLE_SENTINEL,
                "event": False,
                "callable": True,
                "burden": 0.0,
            },
            {
                "cancer_id": "BRCA",
                "patient_id": "A",
                "entity_id": "G1",
                "event": True,
                "callable": True,
                "burden": 1.0,
            },
        ]
    )
    membership = pd.DataFrame(
        [{"pathway_id": "P1", "gene_id": "G1"}, {"pathway_id": "P1", "gene_id": "G2"}]
    )
    pathway_calls = build_exact_pathway_calls(calls, membership)
    sentinel_patients = set(
        pathway_calls.loc[
            pathway_calls.entity_id.eq(ASSAY_CALLABLE_SENTINEL), "patient_id"
        ]
    )
    assert sentinel_patients == {"A", "B"}

    lncrna_calls = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": patient,
                "entity_id": ASSAY_CALLABLE_SENTINEL,
                "event": False,
                "callable": True,
                "burden": 0.0,
            }
            for patient in ("A", "B")
        ]
        + [
            {
                "cancer_id": "BRCA",
                "patient_id": "B",
                "entity_id": "L1",
                "event": True,
                "callable": True,
                "burden": 1.0,
            }
        ]
    )
    candidates = pd.DataFrame(
        [{"cancer_id": "BRCA", "lncrna_id": "L1", "pathway_id": "P1"}]
    )
    stats = candidate_statistics(
        candidates,
        lncrna_calls,
        pathway_calls,
        ["A", "B"],
        modality="mutation",
        min_pair_callable=2,
    )
    assert stats.available.tolist() == [True]
    assert np.isfinite(stats.labels[0])
    assert stats.reasons.tolist() == [""]


def test_unavailable_candidate_statistics_are_null_not_negative() -> None:
    candidates = pd.DataFrame(
        [{"cancer_id": "KIRC", "lncrna_id": "L1", "pathway_id": "P1"}]
    )
    empty = pd.DataFrame()
    stats = candidate_statistics(
        candidates,
        empty,
        empty,
        ["S1", "S2"],
        modality="cnv",
        min_pair_callable=2,
    )
    assert not stats.available.any()
    assert np.isnan(stats.labels).all()
    assert stats.reasons.tolist() == ["CNV_NOT_AVAILABLE_FOR_CANCER"]


def test_real_typed_lncrna_ids_match_prefix_normalized_core_embeddings(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    candidates = pd.read_parquet(paths["candidates"])
    candidates["lncrna_id"] = "LNC:" + candidates.lncrna_id.astype(str)
    candidates.to_parquet(paths["candidates"], index=False)
    lnc_calls = pd.read_parquet(paths["lnc_mutation"])
    lnc_calls["lncrna_id"] = "LNC:" + lnc_calls.lncrna_id.astype(str)
    lnc_calls.to_parquet(paths["lnc_mutation"], index=False)
    cnv_lnc_calls = pd.read_parquet(paths["cnv_lnc"])
    cnv_lnc_calls["lncrna_id"] = "LNC:" + cnv_lnc_calls.lncrna_id.astype(str)
    cnv_lnc_calls.to_parquet(paths["cnv_lnc"], index=False)
    for fold in range(5):
        manifest = json.loads(paths["core_manifest"].read_text(encoding="utf-8"))
        declaration = manifest["folds"][str(fold)]["exports"]["lncRNA"]
        embedding_path = Path(declaration["path"])
        embedding = pd.read_parquet(embedding_path)
        embedding["node_id"] = "LNC:" + embedding.node_id.astype(str)
        embedding.to_parquet(embedding_path, index=False)
        declaration["sha256"] = artifact_sha256(embedding_path)
        paths["core_manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    output = tmp_path / "typed-output"
    result = run_genomic_training(
        candidates_path=paths["candidates"],
        patient_folds_path=paths["folds"],
        pathway_gene_membership_path=paths["membership"],
        sample_gene_mutation_path=paths["gene_mutation"],
        sample_lncrna_mutation_path=paths["lnc_mutation"],
        mc3_path=paths["mc3"],
        core_embedding_manifest_path=paths["core_manifest"],
        cnv_gene_calls_path=paths["cnv_gene"],
        cnv_lncrna_calls_path=paths["cnv_lnc"],
        output_root=output,
        training_run_id="v32-genomic-typed-id-test",
        config=GenomicTrainingConfig(
            seed=19,
            epochs=2,
            batch_size=32,
            hidden_features=8,
            dropout=0.0,
            learning_rate=0.01,
            patience=1,
            min_pair_callable=2,
            max_train_rows=500,
            max_validation_rows=500,
            prediction_batch_size=64,
            cnv_event_threshold=0.3,
        ),
    )
    assert result["mutation_available_rows"] > 0


def test_old_prediction_columns_are_rejected_even_with_exact_keys() -> None:
    frame = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "lncrna_id": "L1",
                "pathway_id": "P1",
                "adjusted_probability": 0.8,
            }
        ]
    )
    with pytest.raises(GenomicTrainingError, match="forbidden result columns"):
        normalise_candidates(frame)


def test_raw_segments_produce_explicit_callable_gene_and_lncrna_calls(tmp_path: Path) -> None:
    segment_root = tmp_path / "segments" / "BRCA"
    segment_root.mkdir(parents=True)
    segment = segment_root / "BRCA_S00__fixture.masked_cnv_grch38.seg.v2.txt"
    pd.DataFrame(
        [
            {"GDC_Aliquot": "A", "Chromosome": "1", "Start": 1, "End": 500, "Num_Probes": 10, "Segment_Mean": 0.7},
            {"GDC_Aliquot": "A", "Chromosome": "1", "Start": 501, "End": 1000, "Num_Probes": 10, "Segment_Mean": 0.0},
        ]
    ).to_csv(segment, sep="\t", index=False)
    intervals = pd.DataFrame(
        [
            {"entity_type": "gene", "entity_id": "G1", "chromosome": "1", "start": 100, "end": 200},
            {"entity_type": "lncRNA", "entity_id": "L1", "chromosome": "1", "start": 800, "end": 900},
        ]
    )
    folds = pd.DataFrame(
        [{"cancer_id": "BRCA", "patient_id": "BRCA_S00", "patient_fold_id": 0}]
    )
    gene, lnc, audit = segment_calls_from_raw(tmp_path / "segments", intervals, folds)
    gene_call = normalise_cnv_calls(gene, entity_kind="gene", event_threshold=0.3)
    lnc_call = normalise_cnv_calls(lnc, entity_kind="lncrna", event_threshold=0.3)
    assert audit["matched_segment_files"] == 1
    assert bool(gene_call.iloc[0].callable) and bool(gene_call.iloc[0].event)
    assert bool(lnc_call.iloc[0].callable) and not bool(lnc_call.iloc[0].event)
    assert lnc_call.iloc[0].burden == 0.0


def test_five_fold_fresh_genomic_training_preserves_null_cnv_cancers(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    output = tmp_path / "output"
    result = run_genomic_training(
        candidates_path=paths["candidates"],
        patient_folds_path=paths["folds"],
        pathway_gene_membership_path=paths["membership"],
        sample_gene_mutation_path=paths["gene_mutation"],
        sample_lncrna_mutation_path=paths["lnc_mutation"],
        mc3_path=paths["mc3"],
        core_embedding_manifest_path=paths["core_manifest"],
        cnv_gene_calls_path=paths["cnv_gene"],
        cnv_lncrna_calls_path=paths["cnv_lnc"],
        output_root=output,
        training_run_id="v32-genomic-test",
        config=GenomicTrainingConfig(
            seed=17,
            epochs=4,
            batch_size=32,
            hidden_features=8,
            dropout=0.0,
            learning_rate=0.01,
            patience=3,
            min_pair_callable=2,
            max_train_rows=500,
            max_validation_rows=500,
            prediction_batch_size=64,
            cnv_event_threshold=0.3,
        ),
    )
    assert result["status"] == "SUCCESS"
    predictions = pd.read_parquet(output / "mutation_cnv_typed_predictions.parquet")
    original = pd.read_parquet(paths["candidates"])
    assert len(predictions) == len(original)
    assert predictions[list(("cancer_id", "lncrna_id", "pathway_id"))].duplicated().sum() == 0
    assert predictions.mutation_available.all()
    brca = predictions.loc[predictions.cancer_id.eq("BRCA")]
    coad = predictions.loc[predictions.cancer_id.eq("COAD")]
    assert brca.cnv_available.all()
    assert brca.cnv_context_probability.notna().all()
    assert not coad.cnv_available.any()
    assert coad.cnv_context_probability.isna().all()
    assert coad.cnv_unavailable_reason.eq("CNV_NOT_AVAILABLE_FOR_CANCER").all()
    assert coad.mutation_cnv_context_probability.notna().all()
    assert not predictions.changes_primary_ranking.any()

    lineage = json.loads((output / "LINEAGE.json").read_text(encoding="utf-8"))
    validate_module_lineage("mutation_cnv", lineage)
    validate_public_module_frame("mutation_cnv", predictions)
    assert lineage["folds"] == 5
    assert lineage["private_head_trained_from_scratch"] is True
    assert lineage["core_parameters_before_sha256"] == lineage["core_parameters_after_sha256"]
    assert lineage["old_checkpoint_loaded"] is False
    assert lineage["old_predictions_used_as_features"] is False
    assert lineage["family_pathway_statistics_loaded"] is False
    assert lineage["missing_mutation_assumed_wildtype"] is False
    assert lineage["missing_cnv_assumed_neutral"] is False
    assert lineage["cnv_cancers_null_with_reason"] == ["COAD"]

    checkpoint_manifest = json.loads(
        (output / "CHECKPOINT_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["core_detached_and_frozen"] is True
    assert checkpoint_manifest["all_private_heads_random_initialization"] is True
    assert len(checkpoint_manifest["records"]) == 10
    assert all(row["source_checkpoint_sha256"] is None for row in checkpoint_manifest["records"])
