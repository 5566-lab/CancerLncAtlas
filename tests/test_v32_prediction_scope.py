"""Native count-to-scope-to-frozen-prediction-to-GeneSet checks on host 149.

All measurements are synthetic. Frozen historical control metadata is a
serialization fixture; no real cohort authority or trained checkpoint is claimed.
"""
import copy
import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.prediction_scope import (summarize_scopes, validate_annotations,
    geneset_eligibility, validate_prepared_scope, FOLD_COLUMNS)
from cc_hhgt.v32.formal_expression_sources import record, read, write
from cc_hhgt.v32.current_core_prediction import (candidate_table, prediction_batches,
    materialize_current_predictions, load_current_predictions)
from test_v32_fold_local_expression import normalized, forbidden
from test_v32_fold_local_scope import scoped, graph_fixture, P1


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(KMeans, "fit", forbidden)


def examples():
    return {f: pd.DataFrame({"cancer_id": "BRCA", "lncrna_id": ["A", "B", "C", "D"],
        "fold_id": f, "fit_split": "train", "shared_or_local_scope": ["shared", "cancer_local",
            "shared" if f < 3 else "cancer_local", "not_expression_eligible"]}) for f in range(5)}


def test_agreement_preserves_mixed_and_all_ineligible_states_and_explicit_policy():
    frame = summarize_scopes(examples())
    assert frame.shared_or_local_scope.tolist() == ["shared", "cancer_local", "fold_variable", "not_expression_eligible"]
    assert frame.scope_shared_fold_count.tolist() == [5, 0, 3, 0]
    assert geneset_eligibility(frame, 1).tolist() == [True, True, True, False]
    for threshold in (None, 0, 6, True, 1.5):
        with pytest.raises(ValueError, match="explicit"): geneset_eligibility(frame, threshold)


@pytest.mark.parametrize("fault", ["missing_fold", "heldout", "different_grid", "relabel", "counts", "partial"])
def test_scope_annotation_corruption_rejected(fault):
    frames = examples()
    if fault == "missing_fold": del frames[4]
    elif fault == "heldout": frames[0]["fit_split"] = "test"
    elif fault == "different_grid": frames[0].loc[0, "lncrna_id"] = "other"
    if fault in {"missing_fold", "heldout", "different_grid"}:
        with pytest.raises(ValueError): summarize_scopes(frames)
        return
    frame = summarize_scopes(frames)
    if fault == "relabel": frame.loc[2, "shared_or_local_scope"] = "shared"
    elif fault == "counts": frame.loc[2, "scope_shared_fold_count"] = 5
    else: frame = frame.drop(columns="scope_basis")
    with pytest.raises(ValueError): validate_annotations(frame)


@pytest.fixture(scope="module")
def native_current(scoped, tmp_path_factory):
    from test_v32_local_cnv_source_ancestry import native_inputs
    from test_v32_preparation_sources import adjusted_capture
    import test_v32_current_binding_expansion as fixtures
    from test_v32_current_core_prediction import current_fixture
    from cc_hhgt.v32.fold_local_scope import scope_input
    from cc_hhgt.v32.formal_graph_authority import build_bound_formal_graph, bind_formal_graph_variant
    from cc_hhgt.v32.formal_graph import build_variant_runtime_bundle, materialize_typed_lnc_protein_binding
    from cc_hhgt.v32.preparation_sources import attach_sources
    from scripts.prepare_v32_formal import make_batches, attach_replication_labels, load_adjusted_associations
    root = tmp_path_factory.mktemp("native_scope_prediction")
    candidate_path = scoped/"normal/candidates.parquet"
    patient_path = scoped.parent/"patient_map.tsv"
    candidates = pd.read_parquet(candidate_path)
    source = read(scoped/"normal/SCOPE_RELEASE.json")
    payloads = {}
    hierarchy = root/"hierarchy.parquet"
    pd.DataFrame({"pathway_id": [P1], "pathway_family_id": ["F1"], "weight": [1.]}).to_parquet(hierarchy, index=False)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        from sklearn.linear_model import LogisticRegression
        patch.setattr(LogisticRegression, "fit", forbidden)
        args, vargs, _, _ = native_inputs(root/"native", patch,
            fold_expression_root=scoped.parent/"converted", patient_map_source=patient_path,
            candidate_ids=candidates.lncrna_id.tolist(), pathway_ids=[P1], candidate_source=candidate_path)
        original = fixtures.verified_payload
        def payload(fold=0):
            result = original(fold)
            capture, _, _ = adjusted_capture(root/f"sources{fold}", fold)
            replaced = {"detection", "pathway_hierarchy", "candidate_authority", "patient_map",
                        "train_associations", "validation_associations"}
            capture.reads = [r for r in capture.reads if r["role"] not in replaced]
            capture.read(candidate_path, pd.read_parquet, role="candidate_authority")
            capture.read(patient_path, lambda p: pd.read_csv(p, sep="\t"), role="patient_map", kind="control")
            capture.read(hierarchy, pd.read_parquet, role="pathway_hierarchy")
            scope, detection = scope_input(source, scoped, fold, capture=capture)
            inputs, patients = graph_fixture(root/f"graph{fold}", scoped, fold)
            inputs.static_paths["pathway_hierarchy"] = hierarchy
            inputs.static_sha256["pathway_hierarchy"] = record(hierarchy)["sha256"]
            binding_path = inputs.static_paths["lnc_protein_binding"]
            pd.read_parquet(binding_path).assign(graph_assay_class="eclip").to_parquet(binding_path, index=False)
            inputs.static_sha256["lnc_protein_binding"] = record(binding_path)["sha256"]
            graph = build_bound_formal_graph(inputs, outer_fold=fold, split_manifest=patients,
                candidate_pairs=candidates, detection_override=detection,
                binding_materializer=materialize_typed_lnc_protein_binding, binding_generation="TYPED_ASSAY_CLASS_V1")
            result["bundle"] = build_variant_runtime_bundle(graph.authority, "G2", edge_chunk_size=2)
            result["formal_graph_authority"] = bind_formal_graph_variant(graph.binding, graph.authority, "G2")
            frames = load_adjusted_associations(capture, args.output_root, vargs.output_root, fold=fold, cancers=["BRCA"])
            for split in frames:
                frames[split] = frames[split].merge(scope, on=["cancer_id", "lncrna_id"], validate="many_to_one")
            frames["validation"] = attach_replication_labels(frames["train"], frames["validation"])
            for split in frames:
                result[split+"_batches"] = make_batches(frames[split], np.zeros(len(candidates)), result["bundle"], split=split)
            result["input_scope"].update(association_mode="local-cnv-adjusted", candidate_rows_per_split=len(candidates))
            result["label_contract"].update(train_direction_label_source="train_local_cnv_adjusted_effect",
                validation_direction_label_source="validation_local_cnv_adjusted_replication_effect",
                association_target="A1_LOCAL", association_available_mask="association_available")
            attach_sources(result, capture.receipt(fold=fold, mode="local-cnv-adjusted", cancers=["BRCA"]))
            payloads[fold] = result
            return result
        patch.setattr(fixtures, "verified_payload", payload)
        manifest = current_fixture(root/"current", pathway_id=P1)
    value = read(manifest); del value["lncrna_scope"]
    value.update(candidate_authority=record(candidate_path), fold_local_scope=record(scoped/"normal/SCOPE_RELEASE.json"),
                 patient_map=record(patient_path), pathway_hierarchy=record(hierarchy))
    write(manifest, value)
    binding = materialize_current_predictions(manifest, root/"prediction", expected_run_id="CURRENT-V4-SYNTHETIC")
    return root, manifest, binding, payloads


def test_native_five_fold_forward_preserves_all_candidates_and_score_estimand(native_current):
    root, manifest, binding, _ = native_current
    scores, value = load_current_predictions(binding, expected_sha=record(binding)["sha256"])
    assert scores.set_index("lncrna_id").shared_or_local_scope.to_dict() == {
        "LNC:LOW": "not_expression_eligible", "LNC:LEAK": "fold_variable", "LNC:Z": "cancer_local"}
    folds = [pd.read_parquet(r["prediction_path"]).sort_values(["cancer_id", "lncrna_id", "pathway_id"])
             for r in value["ensemble_source_records"]]
    expected = np.mean([f.association_membership_probability.to_numpy() for f in folds], axis=0)
    np.testing.assert_array_equal(scores.association_membership_probability, expected)
    assert all(r["training_detection_scope"] and not r["optimizer_started"] for r in value["ensemble_source_records"])
    from scripts.audit_v32_current_candidates import main
    assert main(["--binding", str(binding), "--expected-sha256", record(binding)["sha256"],
                 "--replay-models", "--output", str(root/"deep_audit.json")]) == 0


@pytest.mark.parametrize("fault", ["legacy", "fold", "feature", "patient_map", "hierarchy"])
def test_native_annotations_must_match_actual_prepared_inputs(native_current, tmp_path, fault):
    _, manifest_path, _, payloads = native_current
    manifest = read(manifest_path); candidates = candidate_table(manifest, manifest_path.parent)
    payload = copy.deepcopy(payloads[0]); batches = prediction_batches(payload, candidates)
    if fault == "legacy":
        del manifest["fold_local_scope"]
    elif fault == "fold":
        detection = next(r for r in payload["preparation_sources"]["source_reads"] if r["role"] == "detection")
        detection["fold"] = 1
    elif fault == "feature": batches[0]["conservation_context"][0, 1] = .7
    else:
        role = "patient_map" if fault == "patient_map" else "pathway_hierarchy"
        alternate = tmp_path/("copy"+Path(manifest[role]["path"]).suffix)
        alternate.write_bytes(Path(manifest[role]["path"]).read_bytes()); manifest[role] = record(alternate)
    with pytest.raises(ValueError):
        validate_prepared_scope(manifest, manifest_path.parent, payload, candidates, batches, 0)


def test_native_selected_genesets_query_gmt_scope_policy_and_model_use(native_current, tmp_path):
    from cc_hhgt.v32.selected_genesets import materialize_selected_analysis, SelectedAnalysisQuery
    from cc_hhgt.v32.current_model_use import materialize_current_model_use
    _, _, binding, _ = native_current
    spec = {"model_id": "native", "producer_kind": "current_core_release", "producer_binding": record(binding),
            "run_id": "CURRENT-V4-SYNTHETIC", "score_column": "association_membership_probability"}
    selected = {"format": "CURRENT_SELECTED_ANALYSIS_INPUT_V1", "selected_model": spec, "reference_exact_model": spec,
        "selection": {"kind": "EXPLICIT_SOURCE_REGISTRATION", "rationale": "Synthetic frozen forward; no optimizer or winner claim"},
        "geneset": {"min_members": 1, "max_members": 3, "score_min": 0., "min_expression_folds": 1},
        "subtype": {"score_column": "selected_score", "min_members": 1, "min_cancers": 2, "bootstrap_replicates": 2}}
    source = tmp_path/"SELECTED.json"; write(source, selected)
    result = materialize_selected_analysis(source, tmp_path/"selected")
    query = SelectedAnalysisQuery(result["binding_path"], expected_sha=result["binding_sha256"])
    rows = query.query()["rows"]; json.dumps(rows)
    assert {r["lncrna_id"] for r in rows} == {"LNC:LEAK", "LNC:Z"}
    variable = next(r for r in rows if r["lncrna_id"] == "LNC:LEAK")
    assert variable["shared_or_local_scope"] == "fold_variable"
    assert variable["selected_fold_rank_eligible_count"] == variable["scope_eligible_fold_count"] < 5
    assert variable["selected_fold_selection_frequency"] == variable["scope_eligible_fold_count"]/5
    text = Path(query.download("gmt")["path"]).read_text()
    assert "LNC:LEAK" in text and "LNC:Z" in text and "LNC:LOW" not in text
    from cc_hhgt.v32.selected_score import load_selected_scores
    sb = read(result["binding_path"])["selected_score_binding"]
    all_scores, _ = load_selected_scores(sb["path"], expected_sha=sb["sha256"])
    assert len(all_scores) == 3
    from cc_hhgt.v32.selected_genesets import selected_gene_sets, SelectedGeneSetConfig
    master, members, coverage = selected_gene_sets(all_scores, SelectedGeneSetConfig(1, 3, 0., 5))
    assert set(members.lncrna_id) == {"LNC:Z"} and coverage.candidate_rows.sum() == 3
    missing_policy = copy.deepcopy(selected); del missing_policy["geneset"]["min_expression_folds"]
    write(tmp_path/"NO_POLICY.json", missing_policy)
    with pytest.raises(ValueError, match="explicit min_expression_folds"):
        materialize_selected_analysis(tmp_path/"NO_POLICY.json", tmp_path/"invalid_selected")
    inventory_source = tmp_path/"model_use.json"
    write(inventory_source, {"format": "CURRENT_MODEL_USE_INPUT_V1", "models": [spec]})
    inventory = materialize_current_model_use(inventory_source, tmp_path/"inventory")
    inv = read(inventory["binding_path"])
    usage = pd.read_parquet(inv["outputs"]["usage"]["path"])
    for pmid in ("12344321", "99881122", "77889900"):
        assert usage.study_tokens_json.str.contains(pmid).any()
    assert not inv["model_use_inventory_completeness_verified"] and not inv["model_forwards_executed"]
    _check_http(tmp_path, Path(result["binding_path"]))


def _check_http(root, selected):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from cc_hhgt.v32.current_analysis_web import (COMPONENTS, INPUT_FORMAT, materialize_current_analysis_web, CurrentAnalysisWebStore)
    from website.backend.current_analysis_api import current_analysis_router
    source = root/"WEB.json"
    write(source, {"format": INPUT_FORMAT, "release_id": "native-fold-scope-synthetic", "selected_analysis": record(selected),
        "components": {key: {"status": "NOT_PRODUCED", "reason": "Not part of synthetic scope test"} for key in COMPONENTS}})
    result = materialize_current_analysis_web(source, root/"web")
    store = CurrentAnalysisWebStore(result["binding_path"], expected_sha=result["binding_sha256"])
    app = FastAPI(); app.include_router(current_analysis_router(store)); client = TestClient(app)
    catalog = client.get("/api/current/catalog").json()
    result = client.post("/api/current/tables/selected.geneset_members/query", json={"version": catalog["version"], "limit": 10})
    assert result.status_code == 200, result.text
    rows = result.json()["rows"]
    assert any(r["shared_or_local_scope"] == "fold_variable" for r in rows)
    assert all(set(FOLD_COLUMNS) <= set(r) and r["geneset_min_expression_folds"] == 1 for r in rows)
    all_rows = client.post("/api/current/tables/scores.candidates/query", json={"version": catalog["version"]}).json()["rows"]
    assert len(all_rows) == 3 and any(r["shared_or_local_scope"] == "not_expression_eligible" for r in all_rows)
    _check_browser(client)


def _check_browser(client):
    """Real page assets and actual HTTP handlers; intercepted transport only."""
    import os
    from urllib.parse import urlsplit
    from playwright.sync_api import sync_playwright, expect
    repo = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(repo/".verification_packages/web_browsers"))
    def serve(route):
        request = route.request; url = urlsplit(request.url)
        if url.path.startswith("/api/"):
            response = client.request(request.method, url.path+("?"+url.query if url.query else ""),
                content=request.post_data, headers={"content-type": "application/json"})
            route.fulfill(status=response.status_code, content_type=response.headers.get("content-type"), body=response.content)
        else:
            asset = repo/"website/frontend"/("current-analysis.html" if url.path == "/" else url.path.lstrip("/"))
            types = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
            route.fulfill(status=200 if asset.is_file() else 404, content_type=types.get(asset.suffix, "text/plain"),
                          body=asset.read_bytes() if asset.is_file() else b"")
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(); errors = []; page.on("pageerror", lambda e: errors.append(str(e)))
            page.route("http://native-scope.test/**", serve)
            page.goto("http://native-scope.test/", wait_until="networkidle")
            page.locator("#table-select").select_option("selected.geneset_members")
            expect(page.locator("#table-container tbody tr")).to_have_count(2)
            expect(page.locator("#table-container")).to_contain_text("训练折表达分类")
            expect(page.locator("#table-container")).to_contain_text("折间变化")
            expect(page.locator("#table-container")).to_contain_text("入集所需检测折数")
            assert not errors
        finally: browser.close()
