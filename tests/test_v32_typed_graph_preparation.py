"""Fresh five-fold typed authority, real prepare writers and frozen forwards.

Only the frozen full-cohort control audit, reduced-size closeout and L1 logits
are synthetic substitutes. Graph authority loading and source hashes are real.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import socket
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.formal_graph_authority import load_formal_graph_input_authority
from test_v32_fold_local_expression import normalized, forbidden
from test_v32_fold_local_scope import scoped, graph_fixture, P1
from test_v32_formal_prepare_graph_authority import _declaration
import scripts.prepare_v32_typed_graph_authority as typed


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def patch_controls(patch, patient_map, receipt):
    import cc_hhgt.v32.patient_fold_authority as patient
    import cc_hhgt.v32.formal_graph_authority as graph
    from test_v32_current_binding_expansion import verified_payload
    # Import/build the generic synthetic control before patching module
    # constants, so tests that import constants by value cannot capture ours.
    audit = verified_payload()["patient_fold_authority"]["authority"]
    for module in (patient, graph):
        patch.setattr(module, "FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256", record(patient_map)["sha256"])
        patch.setattr(module, "FROZEN_V32_RECEIPT_SHA256", record(receipt)["sha256"])
    audit.update(manifest_sha256=record(patient_map)["sha256"], receipt_sha256=record(receipt)["sha256"])
    patch.setattr(patient, "validate_frozen_v32_patient_fold_binding", lambda *a: audit)
    patch.setattr(typed, "validate_frozen_v32_patient_fold_binding", lambda *a: audit)
    return audit


@pytest.fixture(scope="module")
def native_authority(scoped, tmp_path_factory):
    import cc_hhgt.v32.formal_graph_authority as graph
    root = tmp_path_factory.mktemp("fresh_typed_graph")
    controls = root / "patients.json"; write(controls, {"synthetic_frozen_control": True})
    inputs = [graph_fixture(root / f"fold_{f}", scoped, f)[0] for f in range(5)]
    source = copy.deepcopy(inputs[0].receipt)
    source.update(format=graph.GRAPH_INPUT_RECEIPT_FORMAT, status=graph.GRAPH_INPUT_RECEIPT_STATUS,
        patient_fold_authority={"manifest_sha256": record(scoped.parent / "patient_map.tsv")["sha256"],
                                "receipt_sha256": record(controls)["sha256"]},
        artifacts={k: _declaration(p, source_role="outcome_free_expression_detection" if k == "detection"
                                   else "static_annotation") for k, p in inputs[0].static_paths.items()},
        gates=dict(historical_graph_rows_used=False, legacy_graph_root_fallback_allowed=False,
            toy_or_synthetic_fallback_allowed=False, outer_train_expression_only=True,
            outer_train_coexpression_only=True, static_evidence_outcome_free=True,
            same_node_and_relation_schema_all_variants=True))
    for f in range(5):
        declaration = copy.deepcopy(inputs[f].receipt["fold_artifacts"][str(f)])
        expression = root / f"expression_{f}.parquet"
        expression.write_bytes(Path(declaration["train_expression"]["path"]).read_bytes())
        declaration["train_expression"].update(record(expression))
        source["fold_artifacts"][str(f)] = declaration
    source_path = root / "source.json"; write(source_path, source)
    table = pd.concat([pd.read_parquet(inputs[0].static_paths["lnc_protein_binding"])] * 3, ignore_index=True)
    table["graph_assay_class"] = ["eclip", "rip", "predicted"]
    table["source_database"] = "synthetic_explicit_static_annotation"
    table.loc[2, "protein_id"] = "PREDICTED_ONLY"
    binding = root / "binding.parquet"; table.to_parquet(binding, index=False)
    spec = dict(format=typed.INPUT, source_receipt=record(source_path),
        patient_map=record(scoped.parent / "patient_map.tsv"), patient_receipt=record(controls),
        binding_declaration=_declaration(binding, source_role="static_annotation"),
        fold_expression_pattern=str(root / "expression_{fold}.parquet"),
        fold_coexpression_pattern=inputs[0].fold_coexpression_pattern)
    manifest = root / "input.json"; write(manifest, spec)
    with pytest.MonkeyPatch.context() as patch:
        patch_controls(patch, scoped.parent / "patient_map.tsv", controls)
        assert typed.main(["--manifest", str(manifest), "--output", str(root / "typed")]) == 0
    assert record(source_path) == spec["source_receipt"]
    result = root / "typed/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"
    return root, manifest, result


@pytest.fixture(scope="module")
def prepared(native_authority, scoped):
    from test_v32_local_cnv_source_ancestry import native_inputs
    import scripts.prepare_v32_formal as prepare
    import cc_hhgt.v32.local_cnv_core_preparation as cnv
    root, manifest, receipt = native_authority
    spec = read(manifest); document = read(receipt)
    candidates = scoped / "normal/candidates.parquet"
    output = root / "prepared"
    with pytest.MonkeyPatch.context() as patch:
        audit = patch_controls(patch, scoped.parent / "patient_map.tsv", root / "patients.json")
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        from sklearn.linear_model import LogisticRegression
        patch.setattr(LogisticRegression, "fit", forbidden)
        args, vargs, _, _ = native_inputs(root / "native", patch,
            fold_expression_root=scoped.parent / "converted", patient_map_source=scoped.parent / "patient_map.tsv",
            candidate_ids=["LNC:LEAK", "LNC:LOW", "LNC:Z"], pathway_ids=[P1], candidate_source=candidates)
        for key in ("SUCCESS", "INDEPENDENT_AUDIT"):
            write(vargs.output_root / (key + ".json"), {"synthetic_control": True})
        patch.setattr(cnv, "validate_validation_a1_authority", lambda *a, **kw: {
            "root": str(vargs.output_root), "success_sha256": record(vargs.output_root / "SUCCESS.json")["sha256"],
            "independent_audit_sha256": record(vargs.output_root / "INDEPENDENT_AUDIT.json")["sha256"],
            "candidate_sha256": record(candidates)["sha256"]})
        from frozen_formal_baseline import frozen_l1
        patch.setattr(prepare, "_fast_l1", frozen_l1)
        patch.setattr(prepare, "load_scope", forbidden)
        authorization = root / "authorization.json"; write(authorization, {"synthetic_control": True})
        argv = ["prepare", "--hashes", str(authorization), "--patient-folds", spec["patient_map"]["path"],
            "--patient-fold-authority-receipt", spec["patient_receipt"]["path"],
            "--graph-authority-receipt", str(receipt), "--graph-authority-receipt-sha256", record(receipt)["sha256"],
            "--graph-relation-schema-generation", "TYPED_ASSAY_CLASS_V1",
            "--graph-fold-expression-pattern", spec["fold_expression_pattern"],
            "--graph-fold-coexpression-pattern", spec["fold_coexpression_pattern"],
            "--formal-candidates", str(candidates), "--formal-candidates-sha256", record(candidates)["sha256"],
            "--association-mode", "local-cnv-adjusted", "--local-cnv-audit-root", str(args.output_root),
            "--local-cnv-audit-success-sha256", record(args.output_root / "SUCCESS.json")["sha256"],
            "--local-cnv-validation-root", str(vargs.output_root),
            "--local-cnv-validation-success-sha256", record(vargs.output_root / "SUCCESS.json")["sha256"],
            "--fold-local-scope", str(scoped / "normal/SCOPE_RELEASE.json"),
            "--fold-local-scope-sha256", record(scoped / "normal/SCOPE_RELEASE.json")["sha256"],
            "--budget-per-cancer", "3", "--fold-count", "5", "--output", str(output)]
        for key, item in document["artifacts"].items():
            name = key.replace("_", "-")
            argv += ["--graph-" + name, item["path"], "--graph-" + name + "-sha256", item["sha256"]]
        patch.setattr(sys, "argv", argv)
        assert prepare.main() == 0
    return output


@pytest.mark.parametrize("fold", range(5))
@pytest.mark.parametrize("arm", ["G0", "G1", "G2"])
def test_actual_typed_prepared_load_and_model_forward(prepared, native_authority, scoped, monkeypatch, fold, arm):
    from cc_hhgt.v32.training import _validate_prepared, _runtime_graph_for_chunk
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from cc_hhgt.v32.integrated_model import module_state_sha256
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", native_authority[0] / "patients.json")
    payload = torch.load(prepared / arm / f"PATIENT_FOLD_{fold}.pt", weights_only=False)
    _validate_prepared(payload, fold=fold, artifact_hashes=payload["artifact_hashes"])
    assert payload["contains_optimizer_state"] is False and payload["contains_trained_parameters"] is False
    binding = payload["formal_graph_authority"]
    assert binding["binding_generation"] == "TYPED_ASSAY_CLASS_V1"
    assert binding["binding_context_policy"] == "GLOBAL_PHYSICAL_BINDING"
    assert binding["typed_binding_preparation"]["excluded_predicted_source_rows"] == 1
    assert binding["fold_normalization_match"]["fold"] == fold
    bundle = payload["bundle"]
    assert "PREDICTED_ONLY" not in bundle.node_maps["protein"]
    relations = set(bundle.edges.relation_type)
    assert "binds_protein" not in relations and "binds_protein_predicted" not in relations
    assert ({"binds_protein_eclip", "binds_protein_rip"} <= relations) == (arm in {"G1", "G2"})
    assert ("physical_interaction" in relations) == (arm == "G2")
    encoder = build_model("cc_hhgt", bundle, payload["feature_dim"], payload["legacy_model_config"])
    model = build_v32_cc_hhgt_residual(encoder, hidden_channels=96, context_features=4, dropout=.2).eval()
    state = module_state_sha256(model)
    with torch.inference_mode():
        graph = _runtime_graph_for_chunk(bundle, 0, device="cpu")
        embeddings = model.encoder.encode(graph)
        batch = payload["train_batches"][0]
        predictions = model(graph, batch["candidate_batch"], batch["base_logit"],
                            batch["conservation_context"], encoded=embeddings)
    assert all(torch.isfinite(value).all() for value in embeddings.values())
    assert predictions["final_logit"].shape == (3,) and all(torch.isfinite(v).all() for v in predictions.values())
    assert module_state_sha256(model) == state
    summary = read(prepared / "PREP_INPUT_SUMMARY.json")
    assert set(summary["scope_by_fold"]) == set(map(str, range(5)))
    assert summary["relation_schema_generation"] == "TYPED_ASSAY_CLASS_V1"


@pytest.mark.parametrize("fault,reason", [
    ("context", "must be global"), ("cancer", "must be global"), ("class", "closed vocabulary"),
    ("unknown", "Unknown assay"), ("identifier", "nonempty"), ("weight", "finite positive"),
    ("hash", "SHA256"), ("outcome", "unsafe"), ("fold", "all five outer folds")])
def test_typed_authority_rejects_repinned_invalid_inputs(native_authority, scoped, tmp_path, monkeypatch, fault, reason):
    root, manifest, _ = native_authority
    patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", root / "patients.json")
    spec = copy.deepcopy(read(manifest))
    if fault in {"context", "cancer", "class", "unknown", "identifier", "weight"}:
        table = pd.read_parquet(spec["binding_declaration"]["path"])
        column, value = {"context": ("is_context_specific", True), "cancer": ("cancer_id", "BRCA"),
            "class": ("graph_assay_class", "unregistered"), "unknown": ("graph_assay_class", "unknown"),
            "identifier": ("protein_id", None), "weight": ("weight", np.nan)}[fault]
        table.loc[0, column] = value
        path = tmp_path / "binding.parquet"; table.to_parquet(path, index=False)
        spec["binding_declaration"].update(record(path))
    elif fault == "hash": spec["binding_declaration"]["sha256"] = "a" * 64
    elif fault == "outcome": spec["binding_declaration"]["outcome_derived"] = True
    else:
        document = read(spec["source_receipt"]["path"]); del document["fold_artifacts"]["4"]
        path = tmp_path / "source.json"; write(path, document); spec["source_receipt"] = record(path)
    path = tmp_path / "input.json"; write(path, spec)
    with pytest.raises((ValueError, RuntimeError), match=reason): typed.prepare(path, tmp_path / "output")
    assert not (tmp_path / "output/SUCCESS.json").exists()


def test_old_generation_loader_rejects_new_typed_receipt(native_authority, scoped, monkeypatch):
    root, manifest, path = native_authority
    audit = patch_controls(monkeypatch, scoped.parent / "patient_map.tsv", root / "patients.json")
    spec = read(manifest); document = read(path)
    with pytest.raises(RuntimeError, match="generation differs"):
        load_formal_graph_input_authority(receipt_path=path, receipt_sha256=record(path)["sha256"],
            patient_authority_audit=audit,
            static_inputs={k: (v["path"], v["sha256"]) for k, v in document["artifacts"].items()},
            fold_expression_pattern=spec["fold_expression_pattern"],
            fold_coexpression_pattern=spec["fold_coexpression_pattern"])


def test_typed_authority_is_immutable_and_host_guarded(native_authority, tmp_path, monkeypatch):
    root, manifest, _ = native_authority
    with pytest.raises(FileExistsError): typed.prepare(manifest, root / "typed")
    monkeypatch.setattr(socket, "gethostname", lambda: "local")
    with pytest.raises(RuntimeError, match="host 149"): typed.prepare(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()
