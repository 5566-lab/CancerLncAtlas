from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cla_engine.common import sha
from cla_engine import modules
from cc_hhgt.v32 import drug_training, genomic_training, single_cell_training
from cc_hhgt.v32.core_export_contract_r4 import ARCHITECTURE, CHECKPOINT_FORMAT, EXPORT_FORMAT
from cc_hhgt.v32.evidence_streaming_training import materialize_streaming_evidence_stage
from cc_hhgt.v32.evidence_training import validate_core_embedding_lineage
from scripts.run_v32_evidence_streaming_training import (
    HEAD_RECEIPT_FORMAT,
    _completed_fold,
    _fresh_output_root,
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _r62_export(root, *, include_gene=True):
    root.mkdir()
    prepared = {str(fold): f"{fold + 20:064x}" for fold in range(5)}
    graph = root / "GRAPH_AUTHORITY_RECEIPT.json"
    _write_json(graph, {"format": "R62_CONTEXT_GRAPH_AUTHORITY_V1", "variant": "G2",
                        "prepared_manifest_sha256": prepared})
    candidate = root / "CANDIDATE_UNIVERSE.parquet"
    pd.DataFrame({"cancer_id": ["C0"], "lncrna_id": ["L0"], "pathway_id": ["P0"]}).to_parquet(candidate, index=False)
    folds = {}
    for fold in range(5):
        folder = root / f"patient_fold={fold}"; folder.mkdir()
        exports = {}
        nodes = {"cancer": ["C0"], "lncRNA": ["L0"], "pathway": ["P0"]}
        if include_gene: nodes["gene"] = ["ENSG000001"]
        for node, identifiers in nodes.items():
            path = folder / f"{node}.parquet"
            pd.DataFrame({"node_id": identifiers, "core_feature_0000": np.ones(len(identifiers), np.float32)}).to_parquet(path, index=False)
            exports[node] = {"path": str(path.relative_to(root)), "sha256": sha(path),
                             "rows": len(identifiers), "features": 1}
        folds[str(fold)] = {
            "patient_fold": fold, "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_sha256": f"{fold + 1:064x}", "prepared_sha256": prepared[str(fold)],
            "training_receipt_sha256": f"{fold + 6:064x}", "core_parameter_sha256": f"{fold + 11:064x}",
            "trained_from_scratch": True, "old_checkpoint_loaded": False, "exports": exports,
        }
    manifest = root / "CORE_EMBEDDING_MANIFEST.json"
    _write_json(manifest, {
        "export_format": EXPORT_FORMAT, "core_architecture": ARCHITECTURE,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_PYG_R62", "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True, "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False, "test_labels_used_in_export": False,
        "task_weights": {"pathway": 1.0, "state": 0.5}, "formal_graph_variant": "G2",
        "folds": folds, "graph_authority_receipt": {"path": graph.name, "sha256": sha(graph)},
        "candidates": {"path": candidate.name, "sha256": sha(candidate), "rows": 1, "cancers": 33},
    })
    return manifest, sha(graph)


def test_all_catalog_flags_exist_in_real_runners():
    for name, recipe in modules.RECIPES.items():
        flags = modules.parser_flags(modules.SCRIPTS / recipe["script"])
        requested = (set(recipe["paths"]) | set(recipe.get("optional_paths", {}))
                     | set(recipe["values"]) | set(recipe.get("hash_roles", {}))
                     | {recipe.get("output_flag", "--output-root")})
        assert requested <= flags, (name, sorted(requested - flags))
    training = modules.parser_flags(modules.SCRIPTS / modules.RECIPES["evidence_train"]["script"])
    assert {"--expected-core-manifest-sha256", "--expected-graph-authority-receipt-sha256",
            "--allow-training", "--deadline", "--resume"} <= training
    assert "--checkpoint-steps" in training
    prediction = modules.parser_flags(modules.SCRIPTS / modules.RECIPES["evidence_predict"]["script"])
    assert {"--expected-core-manifest-sha256", "--expected-graph-authority-receipt-sha256"} <= prediction
    parameters = inspect.signature(materialize_streaming_evidence_stage).parameters
    assert {"direct_evidence_event_path", "r62_core_authority_path"} <= set(parameters)


def test_r62_export_is_accepted_by_each_catalog_core_reader(tmp_path):
    manifest, graph_sha = _r62_export(tmp_path / "core")
    expected = sha(manifest)
    for reader in (single_cell_training.validate_core_manifest,
                   genomic_training._validate_core_manifest,
                   drug_training._validate_core_manifest):
        payload, observed, composite = reader(manifest)
        assert payload["export_format"] == EXPORT_FORMAT
        assert observed == expected and len(composite) == 64
    single = single_cell_training.load_fold_core_embeddings(manifest, single_cell_training.validate_core_manifest(manifest)[0], 0)
    genomic = genomic_training.load_fold_core_embeddings(manifest, genomic_training._validate_core_manifest(manifest)[0], 0)
    drug = drug_training.load_fold_core_embeddings(manifest, drug_training._validate_core_manifest(manifest)[0], 0)
    assert single.lncrna.values.shape == genomic.lncrna.values.shape == drug.lncrna.values.shape == (1, 1)
    lineage = validate_core_embedding_lineage(
        manifest.parent, 0, require_current_g2_authority=True,
        expected_manifest_sha256=expected, expected_graph_authority_receipt_sha256=graph_sha,
    )
    assert lineage.formal_graph_variant == "G2"
    assert lineage.graph_authority_receipt_sha256 == graph_sha
    assert set(lineage.export_paths) == {"cancer", "lncRNA", "pathway"}


def test_r62_graph_receipt_drift_and_missing_drug_gene_are_rejected(tmp_path):
    manifest, _ = _r62_export(tmp_path / "drift")
    graph = manifest.parent / "GRAPH_AUTHORITY_RECEIPT.json"
    payload = json.loads(graph.read_text()); payload["variant"] = "G1"; _write_json(graph, payload)
    for reader in (single_cell_training.validate_core_manifest,
                   genomic_training._validate_core_manifest,
                   drug_training._validate_core_manifest):
        with pytest.raises(Exception, match="R6.2 core export failed validation"):
            reader(manifest)

    missing, _ = _r62_export(tmp_path / "missing_gene", include_gene=False)
    payload = drug_training._validate_core_manifest(missing)[0]
    with pytest.raises(drug_training.DrugTrainingError, match="lacks lncRNA/gene/cancer"):
        drug_training.load_fold_core_embeddings(missing, payload, 0)


def test_evidence_requires_external_r62_hash_bindings(tmp_path):
    manifest, graph_sha = _r62_export(tmp_path / "evidence")
    with pytest.raises(Exception, match="external core-manifest and graph-receipt"):
        validate_core_embedding_lineage(manifest.parent, 0, require_current_g2_authority=True)
    with pytest.raises(Exception, match="SHA"):
        validate_core_embedding_lineage(
            manifest.parent, 0, require_current_g2_authority=True,
            expected_manifest_sha256="0" * 64, expected_graph_authority_receipt_sha256=graph_sha,
        )

def test_evidence_resume_reuses_only_completed_bound_fold(tmp_path):
    output, existed = _fresh_output_root(tmp_path / "training", resume=False)
    assert existed is False
    fold_dir = output / "patient_fold=0"
    fold_dir.mkdir()
    state = fold_dir / "private_head_state.pt"
    state.write_bytes(b"completed fold state")
    binding_sha = "1" * 64
    receipt = {
        "format": HEAD_RECEIPT_FORMAT,
        "status": "TRAINED_NOT_DEPLOYED",
        "patient_fold": 0,
        "pair_evaluation_fold": 0,
        "pair_validation_fold": 1,
        "pair_training_folds": [2, 3, 4],
        "evaluation_used_for_selection": False,
        "execution_binding_sha256": binding_sha,
        "state_sha256": sha(state),
        "optimizer_steps": 11,
    }
    _write_json(fold_dir / "HEAD_FIT.json", receipt)

    resumed = _completed_fold(
        output, 0, execution_binding_sha256=binding_sha,
    )
    assert resumed["optimizer_steps"] == 11
    assert resumed["receipt"] == "patient_fold=0/HEAD_FIT.json"

    with pytest.raises(RuntimeError, match="does not match this execution"):
        _completed_fold(
            output, 0, execution_binding_sha256="4" * 64,
        )

    state.write_bytes(b"mutated state")
    with pytest.raises(RuntimeError, match="does not match this execution"):
        _completed_fold(
            output, 0, execution_binding_sha256=binding_sha,
        )


def test_evidence_resume_distinguishes_step_checkpoint_from_completed_fold(tmp_path):
    output, existed = _fresh_output_root(tmp_path / "training", resume=False)
    assert existed is False
    with pytest.raises(FileExistsError, match="without --resume"):
        _fresh_output_root(output, resume=False)
    reused, existed = _fresh_output_root(output, resume=True)
    assert reused == output.resolve() and existed is True

    partial = output / "patient_fold=1"
    (partial / "resume_state").mkdir(parents=True)
    # A real fitter checkpoint is resumable but is not a completed fold.
    assert _completed_fold(
        output, 1, execution_binding_sha256="1" * 64,
    ) is None

    _write_json(
        partial / "HEAD_FIT.json",
        {
            "format": HEAD_RECEIPT_FORMAT,
            "status": "TRAINED_NOT_DEPLOYED",
            "patient_fold": 1,
        },
    )
    with pytest.raises(RuntimeError, match="lacks its model state"):
        _completed_fold(
            output, 1, execution_binding_sha256="1" * 64,
        )
