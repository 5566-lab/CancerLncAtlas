from __future__ import annotations

import json
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.multimodal_fusion import pair_blocked_fold
from cc_hhgt.v32.patient_fold_authority import (
    FORMAL_PREPARED_BINDING_FORMAT,
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.sealed_test_inference import (
    ACCEPTANCE_STATUS,
    SEALED_TEST_MANIFEST_FORMAT,
    SEALED_TEST_PAYLOAD_FORMAT,
    WINNER_DECLARATION_FORMAT,
    SealedTestInferenceError,
    materialize_winner_locked_sealed_test,
    sha256_file,
)
from cc_hhgt.v32.training import CHECKPOINT_FORMAT, PREPARED_FORMAT


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_ROOT = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"
PAIR_SEED = 20260826


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _binding(authority_root: Path) -> dict:
    audit = validate_frozen_v32_patient_fold_binding(
        authority_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        authority_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    return {
        "format": FORMAL_PREPARED_BINDING_FORMAT,
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "authority": audit,
        "sample_patient_fold_map": {
            "path": str(authority_root / "SAMPLE_PATIENT_FOLD_MAP.tsv"),
            "sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        },
        "authority_receipt": {
            "path": str(authority_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"),
            "sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "sample_id_patient_fallback_used": False,
        "legacy_patient_fold_manifest_used": False,
    }


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    # Graph-binding validation is exercised by its own formal-graph tests.  The
    # focused materializer fixture keeps only node maps and observes the call.
    graph_calls: list[tuple[int, str]] = []

    def graph_gate(binding, *, outer_fold, variant, bundle):
        assert binding == {"fixture": True, "fold": outer_fold}
        graph_calls.append((outer_fold, variant))
        return {"status": "PASS"}

    monkeypatch.setattr(
        "cc_hhgt.v32.formal_graph_authority.validate_formal_graph_payload_binding",
        graph_gate,
    )

    authority = tmp_path / "authority"
    authority.mkdir(parents=True)
    for name in ("SAMPLE_PATIENT_FOLD_MAP.tsv", "PATIENT_FOLD_AUTHORITY_RECEIPT.json"):
        shutil.copyfile(AUTHORITY_ROOT / name, authority / name)
    binding = _binding(authority)

    prepared = tmp_path / "prepared" / "G2"
    prepared.mkdir(parents=True)
    for name in ("SAMPLE_PATIENT_FOLD_MAP.tsv", "PATIENT_FOLD_AUTHORITY_RECEIPT.json"):
        shutil.copyfile(authority / name, prepared / name)
    _write_json(prepared / "PATIENT_FOLD_BINDING.json", binding)
    _write_json(
        prepared / "FORMAL_GRAPH_VARIANT.json",
        {
            "format": "CANCERLNCATLAS_V32_FORMAL_GRAPH_VARIANT_ROOT_V1",
            "variant": "G2",
            "graph_authority_receipt_sha256": "a" * 64,
            "legacy_root_fold_payloads_allowed": False,
        },
    )

    # The full sealed patient fold and the pair-blocked fold are deliberately
    # different concepts.  Every sealed patient fold sees every candidate;
    # only its matching pair fold is selected into the final OOF frame.
    candidates = pd.DataFrame(
        [
            ("ACC", f"L{lnc:03d}", f"P{pathway:02d}")
            for lnc in range(40)
            for pathway in range(7)
        ],
        columns=["cancer_id", "lncrna_id", "pathway_id"],
    ).sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable")
    candidate_path = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    candidates.to_parquet(candidate_path, index=False)
    pair_folds = np.array(
        [
            pair_blocked_fold(lnc, pathway, seed=PAIR_SEED)
            for lnc, pathway in candidates[["lncrna_id", "pathway_id"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=np.int8,
    )
    assert set(pair_folds) == set(range(5))

    maps = {
        "cancer": {"ACC": 0},
        "lncRNA": {value: index for index, value in enumerate(candidates.lncrna_id.unique())},
        "pathway": {value: index for index, value in enumerate(candidates.pathway_id.unique())},
    }
    bundle = SimpleNamespace(node_maps=maps)
    payloads: dict[Path, dict] = {}
    prepared_records = []
    checkpoint_records = []
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    load_order: list[str] = []
    for fold in range(5):
        prepared_path = prepared / f"PATIENT_FOLD_{fold}.pt"
        prepared_path.write_bytes(f"prepared-g2-{fold}".encode())
        prepared_payload = {
            "prepared_format": PREPARED_FORMAT,
            "patient_fold": fold,
            "formal_graph_variant": "G2",
            "formal_graph_authority": {"fixture": True, "fold": fold},
            "bundle": bundle,
            "feature_dim": 4,
            "legacy_model_config": {},
            "conservation_context_features": 4,
            "train_batches": [{"candidate_batch": {}}],
            "validation_batches": [{"candidate_batch": {}}],
            "label_contract": {
                "test_labels_in_training_payload": False,
                "test_logits_in_training_payload": False,
                "test_metrics_computed_before_winner_lock": False,
            },
            "input_scope": {
                "sealed_test_accessed": False,
                "replication_labels_splits": ["validation"],
                "held_out_effects_as_features": False,
            },
            "contains_optimizer_state": False,
            "contains_trained_parameters": False,
            "patient_fold_authority": binding,
            "artifact_hashes": {"fold": str(fold), "code": "b" * 64},
        }
        payloads[prepared_path.resolve()] = prepared_payload
        prepared_sha = sha256_file(prepared_path)
        prepared_records.append(
            {
                "patient_fold": fold,
                "graph_variant": "G2",
                "path": str(prepared_path),
                "sha256": prepared_sha,
            }
        )
        checkpoint_path = checkpoint_root / f"FOLD_{fold}.pt"
        checkpoint_path.write_bytes(f"checkpoint-g2-{fold}".encode())
        checkpoint = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model_state": {"fixture": fold},
            "artifact_hashes": {
                "code_sha256": "c" * 64,
                "config_sha256": "d" * 64,
                "input_manifest_sha256": "e" * 64,
                "task_manifest_sha256": f"{fold:x}" * 64,
            },
            "input_authority_hashes": prepared_payload["artifact_hashes"],
            "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
        }
        payloads[checkpoint_path.resolve()] = checkpoint
        checkpoint_records.append(
            {
                "patient_fold": fold,
                "graph_variant": "G2",
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "prepared_fold_sha256": prepared_sha,
                "architecture_id": "HHGT_FORMAL_CORE_EXTERNAL_ROUTER",
                "selected_using": "VALIDATION_ONLY",
                "heldout_test_metrics_used": False,
                "training_artifact_hashes": checkpoint["artifact_hashes"],
            }
        )

    config_path = tmp_path / "winner_config.json"
    _write_json(config_path, {"primary_model": {"hidden_channels": 8}})
    declaration_path = tmp_path / "WINNER_DECLARATION.json"
    declaration = {
        "format": WINNER_DECLARATION_FORMAT,
        "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
        "compared_graph_variants": ["G0", "G1", "G2"],
        "winner_id": "G2",
        "graph_variant": "G2",
        "selection_scope": "VALIDATION_ONLY",
        "heldout_test_metrics_used_for_selection": False,
        "test_inputs_opened_before_winner_lock": False,
        "winner_or_checkpoint_changed_by_test": False,
        "declaration_locked": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "candidate_authority_sha256": sha256_file(candidate_path),
        "pair_fold_seed": PAIR_SEED,
        "prepared_folds": prepared_records,
        "checkpoints": checkpoint_records,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
    }
    _write_json(declaration_path, declaration)
    declaration_sha = sha256_file(declaration_path)

    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    authority_frame = pd.read_csv(
        authority / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )
    sealed_records = []
    sealed_paths = []
    for fold in range(5):
        keys = authority_frame.loc[
            pd.to_numeric(authority_frame.patient_fold_id).eq(fold)
        ].copy()
        keys_path = sealed_root / f"TEST_SAMPLE_PATIENT_KEYS_{fold}.tsv"
        keys.to_csv(keys_path, sep="\t", index=False, lineterminator="\n")
        sealed_path = sealed_root / f"SEALED_TEST_FOLD_{fold}.pt"
        sealed_path.write_bytes(f"sealed-test-fold-{fold}".encode())
        # The target encodes the patient-heldout fold.  The final assertions
        # prove that pair-fold k obtains labels from patient fold k only.
        payloads[sealed_path.resolve()] = {
            "sealed_test_format": SEALED_TEST_PAYLOAD_FORMAT,
            "patient_fold": fold,
            "formal_graph_variant": "G2",
            "candidate_authority_sha256": sha256_file(candidate_path),
            "training_payload_sha256": prepared_records[fold]["sha256"],
            "contains_test_labels": True,
            "contains_train_or_validation_batches": False,
            "winner_lock_required_before_deserialization": True,
            "accessed_before_winner_lock": False,
            "patient_fold_authority": binding,
            "test_batches": [
                {
                    "candidate_batch": {"fixture": np.arange(len(candidates))},
                    "base_logit": np.zeros(len(candidates), np.float32),
                    "conservation_context": np.zeros((len(candidates), 4), np.float32),
                    "proxy_label": ((np.arange(len(candidates)) + fold) % 2).astype(np.uint8),
                }
            ],
        }
        sealed_paths.append(sealed_path.resolve())
        sealed_records.append(
            {
                "patient_fold": fold,
                "training_payload_sha256": prepared_records[fold]["sha256"],
                "test_patient_keys": {"path": str(keys_path), "sha256": sha256_file(keys_path)},
                "test_payload": {"path": str(sealed_path), "sha256": sha256_file(sealed_path)},
            }
        )
    sealed_manifest_path = sealed_root / "SEALED_TEST_MANIFEST.json"
    sealed_manifest = {
        "format": SEALED_TEST_MANIFEST_FORMAT,
        "status": "PASS_SEALED_TEST_AUTHORITY_UNOPENED",
        "graph_variant": "G2",
        "winner_lock_required_before_payload_deserialization": True,
        "payloads_opened_before_winner_lock": False,
        "test_labels_absent_from_training_payload": True,
        "old_fold_outputs_used": False,
        "pair_fold_seed": PAIR_SEED,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "winner_declaration_sha256": declaration_sha,
        "candidate_authority_sha256": sha256_file(candidate_path),
        "candidate_authority": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
        "folds": sealed_records,
    }
    _write_json(sealed_manifest_path, sealed_manifest)

    def loader(path: Path):
        resolved = path.resolve()
        load_order.append(resolved.name)
        return payloads[resolved]

    inference_folds: list[int] = []

    def inferencer(*, prepared, sealed, checkpoint, checkpoint_record, config):
        fold = int(sealed["patient_fold"])
        inference_folds.append(fold)
        index = np.arange(len(candidates))
        return candidates.assign(
            base_logit=np.zeros(len(candidates), dtype=np.float32),
            final_logit=(index / 100.0 + fold).astype(np.float32),
            direction_logit=(fold - index / 1000.0).astype(np.float32),
            fusion_target=((index + fold) % 2).astype(np.uint8),
        )

    return {
        "authority": authority,
        "prepared": prepared,
        "candidate": candidates.reset_index(drop=True),
        "candidate_path": candidate_path,
        "pair_folds": pair_folds,
        "payloads": payloads,
        "load_order": load_order,
        "sealed_paths": sealed_paths,
        "inference_folds": inference_folds,
        "declaration": declaration,
        "declaration_path": declaration_path,
        "declaration_sha": declaration_sha,
        "sealed_manifest": sealed_manifest,
        "sealed_manifest_path": sealed_manifest_path,
        "loader": loader,
        "inferencer": inferencer,
        "graph_calls": graph_calls,
    }


def _run(fixture: dict, output: Path):
    return materialize_winner_locked_sealed_test(
        prepared_root=fixture["prepared"],
        winner_declaration_path=fixture["declaration_path"],
        winner_declaration_sha256=fixture["declaration_sha"],
        sealed_test_manifest_path=fixture["sealed_manifest_path"],
        patient_folds_path=fixture["authority"] / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        patient_fold_authority_receipt_path=(
            fixture["authority"] / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
        ),
        output_root=output,
        expected_graph_variant="G2",
        payload_loader=fixture["loader"],
        fold_inferencer=fixture["inferencer"],
    )


def test_materializes_exact_oof_only_after_validation_winner_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "result"
    acceptance = _run(fixture, output)
    assert acceptance["status"] == ACCEPTANCE_STATUS
    assert acceptance["graph_variant"] == "G2"
    assert acceptance["test_labels_absent_from_training_payload"] is True
    assert fixture["graph_calls"] == [(fold, "G2") for fold in range(5)]
    assert fixture["inference_folds"] == list(range(5))

    first_sealed = min(fixture["load_order"].index(path.name) for path in fixture["sealed_paths"])
    assert first_sealed == 10  # five prepared + five checkpoints, all winner-gated first

    primary = pd.read_parquet(output / "PRIMARY_FUSION_FRAME.PRIVATE.parquet")
    logits = pd.read_parquet(output / "TEST_LOGITS.PRIVATE.parquet")
    assert primary[["cancer_id", "lncrna_id", "pathway_id"]].equals(
        fixture["candidate"][["cancer_id", "lncrna_id", "pathway_id"]]
    )
    assert np.array_equal(primary.fusion_pair_fold.to_numpy(), fixture["pair_folds"])
    assert np.array_equal(primary.source_patient_fold.to_numpy(), fixture["pair_folds"])
    index = np.arange(len(primary))
    # Patient-fold labels are selected only after the independent pair-fold
    # mask.  Each full sealed fold also contained many pair_fold != patient_fold rows.
    assert np.array_equal(
        primary.fusion_target.to_numpy(),
        ((index + fixture["pair_folds"]) % 2).astype(np.uint8),
    )
    assert np.allclose(
        logits.final_logit.to_numpy(), index / 100.0 + fixture["pair_folds"]
    )
    lineage = json.loads((output / "LINEAGE.json").read_text(encoding="utf-8"))
    for record in lineage["sealed_folds"]:
        assert record["full_candidate_rows_inferred"] == len(fixture["candidate"])
        assert 0 < record["outer_pair_test_rows_selected"] < len(fixture["candidate"])
    audit = json.loads(
        (output / "SEALED_TEST_INFERENCE_AUDIT.json").read_text(encoding="utf-8")
    )
    assert audit["checks"]["five_outer_pair_folds_form_exact_candidate_universe"] is True
    assert not (output / "SUCCESS.json").exists()
    assert not (output.stat().st_mode & stat.S_IWUSR) or acceptance["artifacts_read_only"]


def test_rejects_training_payload_test_label_before_any_sealed_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    prepared_zero = (fixture["prepared"] / "PATIENT_FOLD_0.pt").resolve()
    fixture["payloads"][prepared_zero]["test_batches"] = [{"proxy_label": [1]}]
    with pytest.raises(SealedTestInferenceError, match="sealed-test firewall"):
        _run(fixture, tmp_path / "bad")
    assert not any(name.startswith("SEALED_TEST_FOLD") for name in fixture["load_order"])


def test_rejects_checkpoint_sha_or_variant_before_sealed_payload_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    checkpoint = Path(fixture["declaration"]["checkpoints"][2]["path"])
    checkpoint.write_bytes(b"mutated-after-winner-lock")
    with pytest.raises(SealedTestInferenceError, match="SHA-256 drift"):
        _run(fixture, tmp_path / "bad-checkpoint")
    assert not any(name.startswith("SEALED_TEST_FOLD") for name in fixture["load_order"])

    fixture = _fixture(tmp_path / "variant", monkeypatch)
    fixture["declaration"]["graph_variant"] = "G1"
    _write_json(fixture["declaration_path"], fixture["declaration"])
    fixture["declaration_sha"] = sha256_file(fixture["declaration_path"])
    with pytest.raises(SealedTestInferenceError, match="not locked"):
        _run(fixture, tmp_path / "bad-variant")


def test_rejects_independent_input_or_training_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path / "input", monkeypatch)
    checkpoint_path = Path(fixture["declaration"]["checkpoints"][1]["path"]).resolve()
    fixture["payloads"][checkpoint_path]["input_authority_hashes"] = {"drift": True}
    with pytest.raises(SealedTestInferenceError, match="input-authority hashes"):
        _run(fixture, tmp_path / "bad-input-authority")
    assert not any(name.startswith("SEALED_TEST_FOLD") for name in fixture["load_order"])

    fixture = _fixture(tmp_path / "training", monkeypatch)
    checkpoint_path = Path(fixture["declaration"]["checkpoints"][1]["path"]).resolve()
    fixture["payloads"][checkpoint_path]["artifact_hashes"] = {"drift": True}
    with pytest.raises(SealedTestInferenceError, match="training-authority hashes"):
        _run(fixture, tmp_path / "bad-training-authority")
    assert not any(name.startswith("SEALED_TEST_FOLD") for name in fixture["load_order"])


def test_rejects_missing_test_partition_lineage_and_output_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["sealed_manifest"]["folds"][3].pop("test_patient_keys")
    _write_json(fixture["sealed_manifest_path"], fixture["sealed_manifest"])
    with pytest.raises(SealedTestInferenceError, match="lacks a path"):
        _run(fixture, tmp_path / "missing-lineage")

    fixture = _fixture(tmp_path / "reuse", monkeypatch)
    occupied = tmp_path / "already-exists"
    occupied.mkdir()
    with pytest.raises(SealedTestInferenceError, match="refuses output reuse"):
        _run(fixture, occupied)
    assert fixture["load_order"] == []


def test_rejects_unpinned_winner_declaration_without_loading_any_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    fixture["declaration_sha"] = "f" * 64
    with pytest.raises(SealedTestInferenceError, match="caller-pinned SHA drift"):
        _run(fixture, tmp_path / "bad-pin")
    assert fixture["load_order"] == []
