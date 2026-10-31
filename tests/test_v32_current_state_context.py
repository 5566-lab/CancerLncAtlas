"""Joint State publication adapter tests; no fitting or real cohort data."""
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_state_context import (
    CurrentStateContextQuery,
    INPUT_FORMAT,
    _adapt_engine_state_oof,
    _aggregate_state_oof,
    materialize_current_state_context,
)


def record(path):
    return {"path": str(path), "sha256": sha256(path)}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def no_training(*args, **kwargs):
    raise AssertionError("Training is forbidden in current-source serialization fixtures")


@pytest.fixture(scope="module")
def current_sources(tmp_path_factory):
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.selected_genesets import materialize_selected_analysis
    from scripts.export_v32_current_core_embeddings import main as export_main
    root = tmp_path_factory.mktemp("current_state_parent")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam", "AdamW", "SGD"):
            monkeypatch.setattr(torch.optim, name, no_training)
        path = current_fixture(root / "inputs")
        manifest = json.loads(path.read_text())
        run = manifest["core_run_id"]
        materialize_current_predictions(path, root / "predictions", expected_run_id=run)
        primary = root / "predictions" / "CURRENT_PREDICTION_BINDING.json"
        manifest["format"] = "CURRENT_CORE_EXPORT_INPUT_V1"
        export_input = write(root / "EXPORT_INPUT.json", manifest)
        assert export_main(["--manifest", str(export_input), "--expected-core-run-id", run,
                            "--output", str(root / "core")]) == 0
        spec = {"model_id": "current", "run_id": run,
            "producer_kind": "current_core_release", "producer_binding": record(primary),
            "score_column": "association_membership_probability"}
        selected = {"format": "CURRENT_SELECTED_ANALYSIS_INPUT_V1", "selected_model": spec,
            "reference_exact_model": spec,
            "selection": {"kind": "EXPLICIT_SOURCE_REGISTRATION", "rationale": "Synthetic fixture"},
            "geneset": {"min_members": 1, "max_members": 1, "score_min": .1},
            "subtype": {"score_column": "selected_score", "min_members": 1,
                        "min_cancers": 2, "bootstrap_replicates": 2}}
        selected_path = write(root / "SELECTED_INPUT.json", selected)
        result = materialize_selected_analysis(selected_path, root / "selected")
    return root / "core" / "CORE_EMBEDDING_MANIFEST.json", Path(result["binding_path"]), run


def test_joint_adapter_rejects_retired_manifest_before_writing(tmp_path):
    old = write(tmp_path / "old.json", {"format": "CURRENT_STATE_CONTEXT_INPUT_V1"})
    with pytest.raises(ValueError, match="five explicit folds"):
        materialize_current_state_context(old, tmp_path / "output")


def test_joint_adapter_requires_exact_five_fold_bindings(tmp_path, current_sources):
    _, selected, _ = current_sources
    manifest = write(tmp_path / "joint.json", {"format": INPUT_FORMAT, "release_id": "test",
        "selected_analysis": record(selected), "folds": []})
    with pytest.raises(ValueError, match="five explicit folds"):
        materialize_current_state_context(manifest, tmp_path / "output")


def test_actual_r62_evaluate_schema_is_adapted_without_frozen_fields():
    producer = pd.DataFrame({"patient_fold_id": [2, 2], "cancer_id": ["BRCA", "BRCA"],
        "task": ["state", "pathway"], "lncrna_id": ["LNC:A", "LNC:A"],
        "target_id": ["STATE:S0", "PW:1"], "model_logit": [0.0, 1.0],
        "baseline_logit": [-1.0, -1.0], "predicted_direction": ["positive", "negative"],
        "heldout_effect": [.2, -.3], "heldout_p": [.01, .02], "heldout_q": [.03, .04],
        "heldout_n": [20, 20], "availability": [True, True]})
    adapted = _adapt_engine_state_oof(producer, 2)
    assert adapted[["cancer_id", "lncrna_id", "state_id"]].to_dict("records") == [
        {"cancer_id": "BRCA", "lncrna_id": "LNC:A", "state_id": "STATE:S0"}]
    assert adapted.state_membership_probability.iloc[0] == pytest.approx(.5)
    assert adapted.model_family.iloc[0] == "cc_hhgt_shared_state_head"


def test_all_unavailable_candidate_key_remains_explicit():
    rows = []
    for fold in range(5):
        rows.append({"patient_fold_id": fold, "cancer_id": "BRCA", "lncrna_id": "LNC:A",
            "state_id": "STATE:S0", "state_membership_probability": pd.NA,
            "baseline_probability": pd.NA, "state_effect": pd.NA, "heldout_n": 0,
            "availability": False})
    aggregate = _aggregate_state_oof(pd.DataFrame(rows))
    assert len(aggregate) == 1
    assert aggregate.folds_available.iloc[0] == 0
    assert aggregate.availability.iloc[0] == False
    assert pd.isna(aggregate.state_membership_probability.iloc[0])
    assert aggregate.association_direction.iloc[0] is None


def test_synthetic_joint_verifier_double_exercises_publish_audit_and_query(
    tmp_path, current_sources, monkeypatch
):
    """Exercise adapter plumbing without masquerading a fixture as formal training."""
    import cc_hhgt.v32.current_state_context as context
    import cla_engine.evaluate as evaluate

    _, selected, _ = current_sources
    fold_records = []
    for fold in range(5):
        root = tmp_path / f"fold_{fold}"
        prepared = root / "prepared" / "MANIFEST.json"
        training = root / "training" / "TRAINING_RECEIPT.json"
        evaluation = root / "evaluation" / "EVALUATION_RECEIPT.json"
        for path, payload in ((prepared, {"fold": fold}), (training, {"fold": fold}),
                              (evaluation, {"fold": fold})):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        fold_records.append({"patient_fold_id": fold, "prepared_manifest": record(prepared),
            "training_receipt": record(training), "evaluation_receipt": record(evaluation)})
    manifest = write(tmp_path / "joint.json", {"format": INPUT_FORMAT,
        "release_id": "synthetic-adapter-plumbing-only", "selected_analysis": record(selected),
        "folds": fold_records})

    oof = pd.DataFrame({"patient_fold_id": range(5), "cancer_id": ["BRCA"] * 5,
        "lncrna_id": ["LNC:Z"] * 5, "state_id": ["STATE:S0"] * 5,
        "state_membership_probability": pd.array([.6, .7, .8, .9, .5], dtype="Float64"),
        "baseline_probability": pd.array([.5] * 5, dtype="Float64"),
        "predicted_direction": ["positive"] * 5,
        "state_effect": pd.array([.2, .3, .4, .5, .1], dtype="Float64"),
        "heldout_p": [.01] * 5, "heldout_q": [.02] * 5, "heldout_n": [20] * 5,
        "availability": [True] * 5, "analysis_version": [context.ANALYSIS_VERSION] * 5,
        "model_family": ["cc_hhgt_shared_state_head"] * 5,
        "state_gradients_into_core": [True] * 5, "changes_primary_ranking": [False] * 5})
    release = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"],
        "state_id": ["STATE:S0"], "state_membership_probability": [.7],
        "baseline_probability": [.5], "state_effect": [.3], "folds_available": [5],
        "heldout_n_total": [100], "association_direction": ["positive"],
        "folds_expected": [5], "availability": [True],
        "analysis_version": [context.ANALYSIS_VERSION],
        "model_family": ["cc_hhgt_shared_state_head"], "state_gradients_into_core": [True],
        "changes_primary_ranking": [False]})
    lineage = [{"patient_fold_id": fold, "synthetic_fixture_test_only": True}
               for fold in range(5)]
    catalog = pd.DataFrame({"gene_set_id": ["JOINT_STATE::CANCER::BRCA::STATE:S0"],
        "gene_set_name": ["JOINT_STATE::CANCER::BRCA::STATE:S0"],
        "gene_set_type": ["joint_state_oof_descriptive"], "scope": ["CANCER_SPECIFIC"],
        "cancer_id": ["BRCA"], "state_id": ["STATE:S0"], "direction": ["positive"],
        "member_count": [1], "selection_rule": ["R62_DESCRIPTIVE_OOF_CONSENSUS_TOP50"]})
    members = pd.DataFrame({"gene_set_id": catalog.gene_set_id, "member_rank": [1],
        "member_id": ["LNC:Z"], "member_gene_id": ["Z"], "membership_score": [.7],
        "state_effect": [.3], "cancer_count": [1], "direction_consistency": [1.0],
        "source_cancer_id": ["BRCA"]})
    empty = pd.DataFrame({"synthetic_fixture_test_only": pd.Series(dtype=bool)})
    tables = {"state_oof_predictions": oof, "state_release": release,
        "selected_member_states": empty.copy(), "selected_geneset_state_profiles": empty.copy(),
        "selected_state_coverage": empty.copy(), "state_relation_consensus": empty.copy(),
        "state_cross_cancer_candidates_not_final": empty.copy(),
        "state_target_context_subtypes": empty.copy(), "state_geneset_catalog": catalog,
        "state_geneset_members": members}
    for frame in tables.values():
        frame["analysis_version"] = context.ANALYSIS_VERSION
        frame["model_family"] = "cc_hhgt_shared_state_head"
        frame["state_gradients_into_core"] = True
        frame["changes_primary_ranking"] = False
    monkeypatch.setattr(context, "_validate_fold", lambda item: (
        oof.loc[oof.patient_fold_id.eq(item["patient_fold_id"])],
        lineage[item["patient_fold_id"]], {"seed": 1}))
    monkeypatch.setattr(context, "_tables", lambda folds, selected_path, engine_sets_output: (
        tables, "JOINT_STATE::CANCER::BRCA::STATE:S0\tsynthetic_fixture_test_only\tZ\n",
        lineage, [{"seed": 1}] * 5))

    def synthetic_gate(roots, config, output):
        output = Path(output); output.mkdir(parents=True)
        (output / "GLOBAL_GATE.json").write_text(json.dumps({
            "state": "PREDICTION_GATE_PASS_NOT_DEPLOYED",
            "tasks": {"state": {"pass": True}},
            "synthetic_fixture_test_only": True,
        }))
    monkeypatch.setattr(evaluate, "gate", synthetic_gate)

    result = materialize_current_state_context(manifest, tmp_path / "release")
    binding = json.loads(Path(result["binding_path"]).read_text())
    assert all(row["synthetic_fixture_test_only"] for row in binding["fold_lineage"])
    query = CurrentStateContextQuery(result["binding_path"],
        expected_sha=result["binding_sha256"], selected_analysis_sha=record(selected)["sha256"])
    assert query.query(filters={"cancer_id": "BRCA"})["rows"][0][
        "state_membership_probability"] == pytest.approx(.7)
    assert Path(query.download("state_oof_predictions")["path"]).is_file()
    from cc_hhgt.v32.state_gene_set_query import StateGeneSetReleaseQuery
    gene_sets = StateGeneSetReleaseQuery(result["binding_path"],
        expected_binding_sha256=result["binding_sha256"])
    assert gene_sets.query_gene_sets(state_id="STATE:S0")["rows"][0]["member_count"] == 1
    assert gene_sets.query_members(gene_set_id=catalog.gene_set_id.iloc[0])["rows"][0][
        "member_id"] == "LNC:Z"
