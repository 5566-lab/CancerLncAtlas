"""Actual current producers, HTTP routes and browser flow on synthetic data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_analysis_web import (COMPONENTS, INPUT_FORMAT, CurrentAnalysisWebStore,
    materialize_current_analysis_web, source_tables)
from website.backend.current_analysis_api import current_analysis_router
from test_v32_current_state_context import current_sources, native_fixture as state_fixture
from test_v32_current_clinical_context import clinical_fixture
from test_v32_current_drug_context import native_fixture as drug_fixture
from test_v32_current_single_cell_context import native_fixture as single_cell_fixture
from test_v32_current_patient_context import native_fixture as patient_fixture
from test_v32_current_omics_context import genomic_fixture,distal_fixture

ROOT=Path(__file__).resolve().parents[1]


def record(path): return {"path":str(path),"sha256":sha256(path)}
def write(path,value): path.write_text(json.dumps(value),encoding="utf-8"); return path
def forbidden(*args,**kwargs): raise AssertionError("No optimizer in current web checks")


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)


def evidence_fixture(root,selected,*,input_only=False):
    from test_v32_current_rule_evidence import native_fixture
    from scripts.materialize_v32_current_rule_evidence import main
    root.mkdir(); events,lineage,members=native_fixture(); events["lncrna_id"]="Z"
    frames={"events":events,"lineage":lineage,"members":members,
        "candidates":pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]})}
    artifacts={}
    for name,frame in frames.items():
        path=root/(name+".parquet"); frame.to_parquet(path,index=False); artifacts[name]=record(path)
    manifest={"format":"CURRENT_RULE_EVIDENCE_INPUT_V1","candidates":artifacts["candidates"],"exact_membership":artifacts["members"],
        "sources":[{"source_id":"native","kind":"native_exact_events","events":artifacts["events"],"lineage":artifacts["lineage"]}]}
    path=write(root/"INPUT.json",manifest)
    if input_only:return path
    assert main(["--manifest",str(path),"--output",str(root/"release")])==0
    return root/"release/RULE_EVIDENCE_BINDING.json"


def external_fixture(root,selected,*,input_only=False):
    from cc_hhgt.v32.external_validation_current import materialize_current_external
    root.mkdir(); sb=json.loads(selected.read_text()); score_path=Path(sb["selected_score_binding"]["path"])
    score=json.loads(score_path.read_text())
    frames={"crosswalk":pd.DataFrame([("lncrna","Z","LNC:Z"),("lncrna","OTHER","LNC:OTHER"),
        ("cancer","breast","BRCA"),("cancer","lung","LUAD"),("pathway","p1","P1")],columns=["namespace","raw_id","canonical_id"]),
        "used":pd.DataFrame({"cancer_id":["LUAD"],"lncrna_id":["LNC:OTHER"],"pathway_id":["P1"],"pmid":["99999999"]}),
        "observed":pd.DataFrame({"record_id":["OBS1"],"lncrna":["Z"],"cancer":["breast"],"pathway":["p1"],
            "label":[1],"pmid":["12345678"],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic measured assay"]})}
    files={}
    for name,frame in frames.items():
        path=root/(name+".parquet"); frame.to_parquet(path,index=False); files[name]=record(path)
    spec={"model_id":"selected","producer_kind":"selected_release","producer_binding":record(score_path),
        "run_id":score["training_run_id"],"score_column":"selected_score"}
    manifest={"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V1","identity_crosswalk":files["crosswalk"],"models":[spec],
        "model_use_sources":[{"kind":"canonical_usage","source_id":"used","phase":"TRAINING","model_ids":["selected"],"table":files["used"]}],
        "external_sources":[{"kind":"exact_assay","source_id":"external","table":files["observed"],"measured_negative_definition":"Measured null effect"}],
        "evaluation":{"primary_model":"selected","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}}
    path=write(root/"INPUT.json",manifest)
    if input_only:return path
    result=materialize_current_external(path,root/"release")
    return Path(result["binding_path"])


@pytest.fixture(scope="module")
def web_release(tmp_path_factory,current_sources):
    from cc_hhgt.v32.current_analysis_workflow import INPUT_FORMAT as WORKFLOW_INPUT
    from scripts.run_v32_current_analysis_workflow import main
    source_root=tmp_path_factory.mktemp("current_web");root=source_root/"workflow"
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
        components={}
        for key,fixture in [("state",state_fixture),("clinical",clinical_fixture),("drug",drug_fixture),
            ("single_cell",single_cell_fixture),("patient",patient_fixture),("genomic",genomic_fixture),("distal",distal_fixture)]:
            template=fixture(source_root/(key+"_inputs"),current_sources);data=read_json(template)
            data["selected_analysis"]={"workflow_ref":"selected_analysis"};write(template,data)
            components[key]={"status":"BUILD","input_manifest":record(template)}
        template=evidence_fixture(source_root/"evidence",current_sources[1],input_only=True)
        components["evidence"]={"status":"BUILD","input_manifest":record(template)}
        template=external_fixture(source_root/"external",current_sources[1],input_only=True);data=read_json(template)
        data["models"][0]["producer_binding"]={"workflow_ref":"selected_score"};write(template,data)
        components["external"]={"status":"BUILD","input_manifest":record(template)}
        selected_input=read_json(Path(read_json(current_sources[1])["input_manifest"]["path"]))
        selected_input["rule_evidence_binding"]={"workflow_ref":"evidence"}
        selected_template=write(source_root/"SELECTED_TEMPLATE.json",selected_input)
        manifest={"format":WORKFLOW_INPUT,"release_id":"synthetic-current-web",
            "selected":{"status":"BUILD","input_manifest":record(selected_template)},"components":components}
        source=write(source_root/"WORKFLOW_INPUT.json",manifest)
        assert main(["--manifest",str(source),"--output",str(root)])==0
    path=root/"resolved_inputs/web.json"
    binding=root/"web/CURRENT_WEB_BINDING.json"
    store=CurrentAnalysisWebStore(binding,expected_sha=sha256(binding))
    app=FastAPI(); app.include_router(current_analysis_router(store))
    return root,path,binding,store,TestClient(app)


def test_real_current_components_are_served_and_every_public_part_downloads(web_release):
    _,_,_,store,client=web_release
    response=client.get("/api/current/catalog"); assert response.status_code==200
    catalog=response.json(); version=catalog["version"]
    assert set(catalog["components"])==set(COMPONENTS)|{"selected","scores"}
    assert all(c["status"]=="AVAILABLE" for c in catalog["components"].values())
    assert not catalog["independent_external_validation_attested"]
    assert "path\"" not in json.dumps(catalog)
    for table in catalog["tables"]:
        if table["kind"]=="parquet":
            result=client.post(f"/api/current/tables/{table['id']}/query",json={"version":version,"limit":3})
            assert result.status_code==200,(table["id"],result.text)
            assert result.json()["selected_analysis_sha256"]==catalog["selected_analysis_sha256"]
            assert result.json()["has_more"]==(table["rows"]>3)
        for part in table["parts"]:
            result=client.get(f"/api/current/downloads/{table['id']}/{part['part']}",params={"version":version})
            assert result.status_code==200 and hashlib.sha256(result.content).hexdigest()==part["sha256"]
            assert result.headers["x-analysis-version"]==version
            assert result.headers["x-selected-analysis-sha256"]==catalog["selected_analysis_sha256"]
    events=client.post("/api/current/tables/evidence.events/query",json={"version":version}).json()["rows"]
    assert events and all("path" not in json.loads(r["provenance_json"])["artifacts"]["events"] for r in events)
    assert all("sha256" in json.loads(r["provenance_json"])["artifacts"]["events"] for r in events)
    native=read_json(Path(store.binding["component_bindings"]["evidence"]["path"]))
    original=pd.read_parquet(native["event_parts"][0]["path"])
    assert "path" in json.loads(original.provenance_json.iloc[0])["artifacts"]["events"]


def read_json(path): return json.loads(path.read_text())


def test_workflow_builds_native_contexts_and_records_actual_public_schemas(web_release):
    from cc_hhgt.v32.current_analysis_workflow import ORDER
    root,_,_,store,_=web_release
    workflow=read_json(root/"CURRENT_WORKFLOW_BINDING.json");success=read_json(root/"SUCCESS.json")
    assert success["binding_sha256"]==sha256(root/"CURRENT_WORKFLOW_BINDING.json")
    assert success["stage_status_sha256"]==sha256(root/"STAGE_STATUS.json")
    assert tuple(s["stage"] for s in workflow["stages"])==ORDER
    assert all(s["status"]=="BUILT" for s in workflow["stages"])
    assert not workflow["model_use_inventory_completeness_verified"] and not workflow["training_started"]
    catalog=read_json(root/"PUBLIC_TABLE_CATALOG.json")
    assert catalog["web_binding_sha256"]==store.sha and "path\"" not in json.dumps(catalog)
    assert {t["table_id"] for t in catalog["tables"]}==set(store.binding["tables"])
    for table in catalog["tables"]:
        native=store.binding["tables"][table["table_id"]]
        assert [f["name"] for f in table["fields"]]==native["columns"]
        assert table["rows"]==native["rows"] and table["table_semantics"]
        assert all(f["arrow_type"] and type(f["nullable"]) is bool for f in table["fields"])


def test_workflow_builds_selected_scores_and_keeps_unproduced_modules_explicit(current_sources,tmp_path):
    from cc_hhgt.v32.current_analysis_workflow import INPUT_FORMAT as WF,run_current_analysis_workflow
    selected=read_json(current_sources[1])
    manifest={"format":WF,"release_id":"selected-workflow-synthetic",
        "selected":{"status":"BUILD","input_manifest":selected["input_manifest"]},
        "components":{k:{"status":"NOT_PRODUCED","reason":"Synthetic no-native-result case"} for k in COMPONENTS}}
    path=write(tmp_path/"INPUT.json",manifest)
    result=run_current_analysis_workflow(path,tmp_path/"workflow")
    binding=read_json(Path(result["binding_path"]))
    assert next(r for r in binding["stages"] if r["stage"]=="selected")["status"]=="BUILT"
    store=CurrentAnalysisWebStore(result["web_binding"]["path"],expected_sha=result["web_binding"]["sha256"])
    assert all(store.binding["component_status"][k]["status"]=="NOT_PRODUCED" for k in COMPONENTS)
    assert {t["component"] for t in store.binding["tables"].values()}=={"selected","scores"}
    assert len(store.query("selected.geneset_members",version=store.sha)["rows"])==1


def test_workflow_preserves_failed_stage_and_rejects_forward_references(web_release,tmp_path):
    from cc_hhgt.v32.current_analysis_workflow import run_current_analysis_workflow
    root,_,_,store,_=web_release;manifest=read_json(root.parent/"WORKFLOW_INPUT.json")
    manifest["selected"]={"status":"REUSE","binding":store.binding["component_bindings"]["selected"]}
    manifest["components"]={k:{"status":"NOT_PRODUCED","reason":"Synthetic absent stage"} for k in COMPONENTS}
    # This is a real context manifest except for a reference to a later stage.
    template=read_json(root/"resolved_inputs/state.json")
    template["selected_analysis"]={"workflow_ref":"external"}
    manifest["components"]["state"]={"status":"BUILD","input_manifest":record(write(tmp_path/"state.json",template))}
    path=write(tmp_path/"INPUT.json",manifest);output=tmp_path/"failed"
    with pytest.raises(ValueError,match="earlier workflow reference"):run_current_analysis_workflow(path,output)
    status=read_json(output/"STAGE_STATUS.json")
    assert status["status"]=="FAILED" and status["failed_stage"]=="state"
    assert any(s["stage"]=="selected" and s["status"]=="REUSED" for s in status["stages"])
    assert not (output/"SUCCESS.json").exists() and not (output/"CURRENT_WORKFLOW_BINDING.json").exists()
    before=(output/"STAGE_STATUS.json").read_bytes()
    with pytest.raises(FileExistsError):run_current_analysis_workflow(path,output)
    assert (output/"STAGE_STATUS.json").read_bytes()==before


def test_workflow_reuses_pinned_components_and_rejects_changed_inputs(web_release,tmp_path):
    from cc_hhgt.v32.current_analysis_workflow import INPUT_FORMAT as WF,run_current_analysis_workflow
    root,_,_,store,_=web_release
    manifest={"format":WF,"release_id":"reused-workflow-synthetic",
        "selected":{"status":"REUSE","binding":store.binding["component_bindings"]["selected"]},
        "components":{k:{"status":"REUSE","binding":store.binding["component_bindings"][k]} for k in COMPONENTS}}
    path=write(tmp_path/"INPUT.json",manifest)
    result=run_current_analysis_workflow(path,tmp_path/"reused")
    workflow=read_json(Path(result["binding_path"]))
    assert all(r["status"]=="REUSED" for r in workflow["stages"] if r["stage"]!="web")
    manifest["selected"]["binding"]={**manifest["selected"]["binding"],"sha256":"f"*64}
    write(path,manifest)
    with pytest.raises(ValueError,match="SHA256 mismatch"):run_current_analysis_workflow(path,tmp_path/"invalid")
    assert not (tmp_path/"invalid").exists()
    manifest["selected"]["status"]="TRAIN"
    with pytest.raises(ValueError,match="stage status"):run_current_analysis_workflow(write(path,manifest),tmp_path/"no_training")


def test_version_mismatch_invalid_queries_and_private_downloads_are_rejected(web_release):
    _,_,_,store,client=web_release; version=store.sha
    route="/api/current/tables/selected.geneset_members/query"
    assert client.post(route,json={"version":"f"*64}).status_code==409
    assert client.post(route,json={"version":version,"filters":{"lncrna_id":"' OR 1=1 --"}}).json()["rows"]==[]
    for payload in ({"limit":False},{"limit":1001},{"offset":-1},{"filters":{"patient_id":"x"}},{"extra":"x"}):
        assert client.post(route,json={"version":version,**payload}).status_code==422
    assert client.get("/api/current/downloads/selected.gmt/0").status_code==422
    assert client.get("/api/current/downloads/selected.gmt/0",params={"version":"f"*64}).status_code==409
    for identity in ("drug.PRIVATE_DOSE_MEASUREMENT_DISPOSITION","scores.checkpoint","clinical.patient_risk","single_cell.donor_cell_context","patient.patient_oof","distal.nested_outer_expression_components","genomic.checkpoint","../INPUT.json"):
        assert client.get(f"/api/current/downloads/{identity}/0",params={"version":version}).status_code in {404,422}


def test_http_semantics_keep_null_zero_and_drug_missingness(web_release):
    _,_,_,store,client=web_release; version=store.sha
    states=client.post("/api/current/tables/state.selected_geneset_state_profiles/query",json={"version":version}).json()["rows"]
    assert any(r["mean_held_out_state_effect"]==0 for r in states)
    assert any(r["mean_held_out_state_effect"] is None and not r["availability"] for r in states)
    clinical=client.post("/api/current/tables/clinical.selected_geneset_clinical_support/query",
        json={"version":version,"filters":{"clinical_endpoint":"DFS"}}).json()["rows"]
    assert clinical and clinical[0]["supported_risk_weight_fraction"] is None
    gs=client.post("/api/current/tables/selected.geneset_master/query",json={"version":version}).json()["rows"][0]["geneset_id"]
    data=client.get("/api/current/drug-members",params={"version":version,"geneset_id":gs,"drug_id":"ASSAY:unmapped"}).json()
    assert data["rows"][0]["drug_response_association_probability"] is None and not data["rows"][0]["availability"]
    assert data["rows"][0]["failure_reason"]=="OUTSIDE_CONCEPTUAL_UNIVERSE"


def test_modified_artifact_fails_without_disclosing_source_paths(web_release):
    _,_,_,store,client=web_release
    path=Path(store.binding["tables"]["selected.geneset_members"]["files"][0]["path"]); original=path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        response=client.get("/api/current/downloads/selected.geneset_members/0",params={"version":store.sha})
        assert response.status_code==503 and str(path) not in response.text
    finally:path.write_bytes(original)


def test_missing_components_are_explicit_but_cross_selected_components_fail(web_release,tmp_path):
    _,path,_,_,_=web_release; manifest=read_json(path)
    manifest["components"]["drug"]={"status":"NOT_PRODUCED","reason":"当前版本药物分析尚未生成"}
    partial=write(tmp_path/"partial.json",manifest)
    result=materialize_current_analysis_web(partial,tmp_path/"web")
    catalog=CurrentAnalysisWebStore(result["binding_path"],expected_sha=result["binding_sha256"]).catalog()
    assert catalog["components"]["drug"]["status"]=="NOT_PRODUCED" and not any(t["component"]=="drug" for t in catalog["tables"])
    # Rehashing a mismatched selected identity cannot be registered as current.
    original=Path(manifest["components"]["state"]["binding"]["path"]); bad=read_json(original)
    bad["selected_analysis"]["sha256"]="f"*64
    wrong=write(tmp_path/"wrong_state.json",bad);manifest["components"]["state"]["binding"]=record(wrong)
    with pytest.raises(ValueError,match="different selected"):
        source_tables(write(tmp_path/"wrong.json",manifest))


@pytest.fixture(scope="module")
def live_site(web_release):
    _,_,binding,store,_=web_release
    environment=os.environ.copy()
    environment.update(CANCERLNCATLAS_CURRENT_WEB_BINDING=str(binding),CANCERLNCATLAS_CURRENT_WEB_SHA256=store.sha,
        CANCERLNCATLAS_ENABLE_V32_STAGING="0",CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY="1")
    environment["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"src"),environment.get("PYTHONPATH","")])
    sock=socket.socket();sock.bind(("127.0.0.1",0));sock.listen(128);port=sock.getsockname()[1]
    environment["CURRENT_WEB_TEST_FD"]=str(sock.fileno())
    program="""import os,socket,threading,torch
assert socket.gethostname()=='149'
def forbidden(*a,**k):raise AssertionError('No training in web server')
for name in ('Adam','AdamW','SGD'):setattr(torch.optim,name,forbidden)
timer=threading.Timer(90,lambda:os._exit(0));timer.daemon=True;timer.start()
import uvicorn
from website.backend.app import app
server=uvicorn.Server(uvicorn.Config(app,log_level='warning'))
server.run(sockets=[socket.socket(fileno=int(os.environ['CURRENT_WEB_TEST_FD']))])
timer.cancel()
"""
    process=subprocess.Popen([sys.executable,"-c",program],cwd=ROOT,env=environment,pass_fds=(sock.fileno(),),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    sock.close(); url=f"http://127.0.0.1:{port}"
    import httpx
    try:
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if process.poll() is not None: raise AssertionError(process.communicate()[1])
            try:
                if httpx.get(url+"/api/current/catalog",trust_env=False,timeout=1).status_code==200:break
            except httpx.HTTPError:pass
            time.sleep(.2)
        else:raise AssertionError("Current full application did not become ready")
        yield url
    finally:
        process.terminate()
        try:process.communicate(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.communicate(timeout=5)


def test_full_application_uses_current_home_and_does_not_serve_old_results(live_site):
    import httpx
    with httpx.Client(base_url=live_site,trust_env=False) as client:
        page=client.get("/");assert page.status_code==200 and "current-analysis.js" in page.text
        assert client.get("/assets/current-analysis.js").content==(ROOT/"website/frontend/assets/current-analysis.js").read_bytes()
        assert client.get("/assets/current-analysis.css").status_code==200
        assert client.get("/api/site/genesets").status_code==404
        assert client.get("/v2.6/version").status_code==404
        assert client.get("/v3.2-staging/health").status_code==404
        assert client.get("/api/current/unknown").status_code==404
        assert client.get("/openapi.json").json()["info"]["version"]=="synthetic-current-web"


def test_real_browser_filters_contexts_nulls_downloads_and_mobile_layout(live_site,web_release,monkeypatch):
    from playwright.sync_api import sync_playwright, expect
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH",str(ROOT/".verification_packages/web_browsers"))
    out=ROOT/"provenance/code_completion_20261015"
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True)
        try:
            page=browser.new_page(viewport={"width":1440,"height":1000}); errors=[]
            page.on("pageerror",lambda error:errors.append(str(error)))
            page.goto(live_site,wait_until="networkidle")
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("#table-container .member-link").click()
            expect(page.locator("#table-select")).to_have_value("selected.geneset_members")
            expect(page.locator("#table-container")).to_contain_text("LNC:Z")
            page.locator("#filters-form input[name='lncrna_id']").fill("' OR 1=1 --")
            page.locator("#filters-form button[type='submit']").click()
            expect(page.locator("#row-status")).to_have_text("没有匹配记录")
            page.locator("#clear-filters").click();expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("button[data-component='state']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(7)
            expect(page.locator("#table-container")).to_contain_text("不可用")
            page.locator("button[data-component='clinical']").click();expect(page.locator("#table-container tbody tr")).to_have_count(6)
            page.locator("button[data-component='drug']").click();expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("#table-select").select_option("drug.selected_associations")
            expect(page.locator("#table-container")).to_contain_text("HIGHER_EXPRESSION_LOWER_SENSITIVITY")
            gs=web_release[3].query("selected.geneset_master",version=web_release[3].sha)["rows"][0]["geneset_id"]
            page.locator("#drug-lookup summary").click()
            page.locator("#drug-form input[name='geneset_id']").fill(gs)
            page.locator("#drug-form input[name='drug_id']").fill("ASSAY:unmapped")
            page.locator("#drug-form button").click()
            expect(page.locator("#drug-results")).to_contain_text("OUTSIDE_CONCEPTUAL_UNIVERSE")
            expect(page.locator("#drug-results")).to_contain_text("不可用")
            page.locator("button[data-component='evidence']").click();expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("button[data-component='external']").click();expect(page.locator("#row-status")).not_to_have_text("正在查询…")
            page.locator("button[data-component='single_cell']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(3)
            expect(page.locator("#table-container")).to_contain_text("-0.6")
            page.locator("#table-select").select_option("single_cell.native_dataset_context")
            page.locator("#extra-filters summary").click()
            page.locator("#filters-form input[name='dataset_id']").fill("S2")
            page.locator("#filters-form button[type='submit']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            expect(page.locator("#table-container")).to_contain_text("INSUFFICIENT_INDEPENDENT_DONORS")
            page.locator("button[data-component='patient']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(2)
            expect(page.locator("#table-container")).to_contain_text("INSUFFICIENT_GROUP_SUPPORT")
            page.locator("#extra-filters summary").click()
            page.locator("#filters-form input[name='group_id']").fill("declared")
            page.locator("#filters-form button[type='submit']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            expect(page.locator("#table-container")).to_contain_text("exploratory_current_core")
            page.locator("button[data-component='genomic']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(5)
            page.locator("#extra-filters summary").click()
            page.locator("#filters-form input[name='producer_kind']").fill("cnv")
            page.locator("#filters-form button[type='submit']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            expect(page.locator("#table-container")).to_contain_text("不可用")
            page.locator("button[data-component='distal']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("#table-select").select_option("distal.native_feature_coverage")
            page.locator("#extra-filters summary").click()
            page.locator("#filters-form input[name='feature']").fill("distal_methylation_beta")
            page.locator("#filters-form button[type='submit']").click()
            expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.locator("button[data-component='selected']").click();expect(page.locator("#table-container tbody tr")).to_have_count(1)
            page.screenshot(path=str(out/"current_web_desktop.png"),full_page=True)
            page.locator("#downloads-toggle").click()
            group=page.locator("#download-list details").first;group.locator("summary").click()
            with page.expect_download() as download_info:group.get_by_role("link",name="GMT 基因集",exact=True).click()
            download=download_info.value; assert download.suggested_filename=="selected.gmt"
            assert "LNC:Z" in Path(download.path()).read_text()
            page.locator("#downloads-toggle").click();page.set_viewport_size({"width":390,"height":844})
            page.evaluate("window.scrollTo(0,0)");page.screenshot(path=str(out/"current_web_mobile.png"),full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            assert errors==[]
            version=web_release[3].sha
            page.goto(live_site+"/?version="+"f"*64,wait_until="networkidle")
            expect(page.locator("#message")).to_contain_text("另一分析版本")
            page.goto(live_site+"/?filters=null",wait_until="networkidle")
            expect(page.locator("#message")).to_contain_text("筛选条件无效")
            assert errors==[]
            write(out/"current_web_browser_receipt.json",{"target_host":"149","status":"PASS","training_started":False,
                "real_browser":"Chromium via Playwright 1.52.0","version":version,"page_errors":errors,
                "screenshots":["current_web_desktop.png","current_web_mobile.png"],"production_deployed":False})
        finally:browser.close()
