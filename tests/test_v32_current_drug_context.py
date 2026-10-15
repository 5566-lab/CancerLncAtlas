"""Tiny native-format Drug/current-core integration with optimizers forbidden."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.current_drug_sources import INPUT_FORMAT, current_drug_sources, replay_native_reports
from cc_hhgt.v32.current_drug_context import (materialize_current_drug_context, CurrentDrugContextQuery,
    audit_current_drug_context)
from cc_hhgt.v32.drug_sparse_query import (write_drug_sparse_query_sidecars, factor_core_entity_availability,
    PROBABILITY_COLUMN, OUTSIDE_CONCEPTUAL_UNIVERSE)
from cc_hhgt.v32.drug_training_streaming import StreamingDrugTrainingConfig
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.integrated_model import build_private_auxiliary_head, module_state_sha256
from test_v32_current_state_context import current_sources


def record(path): return {"path":str(path),"sha256":artifact_sha256(path)}


def write(path,value): path.write_text(json.dumps(value),encoding="utf-8"); return path


def no_training(*args,**kwargs): raise AssertionError("No training in current Drug verification")


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,no_training)
    monkeypatch.setattr(base,"_fit_head",no_training)


def native_fixture(root,current_sources,*,pharmacology=False,dose=False):
    from test_v32_drug_response_analysis import fixture as response_fixture
    from test_v32_drug_pharmacology import feature_sources
    from cc_hhgt.v32.drug_pharmacology import load_pharmacology, FEATURES, TARGET_POOLS
    from scripts.analyze_v32_drug_model_attribution import main as ablation_main
    from scripts.analyze_v32_native_drug_response import main as scan_main
    root.mkdir(); core_path,selected_path,run = current_sources
    core_manifest = json.loads(core_path.read_text()); core = base.load_fold_core_embeddings(core_path,core_manifest,0)
    gene = next(iter(core.gene.index)); tables = response_fixture()
    tables["expression"]["lncrna_id"] = "LNC:Z"
    tables["exact"] = pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"pathway_id":["P1"]})
    tables["targets"]["gene_id"] = gene
    paths = {}
    for name,frame in tables.items():
        path = root/(name+".parquet"); frame.to_parquet(path,index=False); paths[name] = path
    membership = root/"membership.parquet"
    pd.DataFrame({"pathway_id":["P1"],"gene_id":[gene]}).to_parquet(membership,index=False)
    query_path = root/"queries.parquet"
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"drug_id":["DC:1"]}).to_parquet(query_path,index=False)
    roles = {"exact":"exact_candidates","response":"raw_drug_response","expression":"raw_lncrna_expression",
        "mapping":"cell_line_map","targets":"drug_gene_target"}
    artifacts = {roles[k]:record(p) for k,p in paths.items() if k in roles}
    artifacts.update(core_embedding_manifest=record(core_path),exact_pathway_membership=record(membership),queries=record(query_path))
    cfg = StreamingDrugTrainingConfig(hidden_features=4,min_total_cell_lines=4,min_cell_lines_per_dataset=4)
    config_path = write(root/"config.json",{"training":asdict(cfg)})
    artifacts["config"] = record(config_path)
    pharma = None; features = list(base._DOMAIN_FEATURES)
    if pharmacology:
        pharma_path = feature_sources(root/"pharma",tables["targets"],tables["response"])
        pharma = load_pharmacology(pharma_path,targets=base.normalise_drug_targets(tables["targets"]),
            response=base.normalise_raw_response(tables["response"]),mapping=base.normalise_cell_line_map(tables["mapping"]))
        artifacts["pharmacology_manifest"] = record(pharma_path); features += list(FEATURES)
    records = []
    for fold in range(5):
        parent = core_manifest["folds"][str(fold)]
        head,init = build_private_auxiliary_head("drug",core_features=core.lncrna.values.shape[1]*(3+len(TARGET_POOLS) if pharma else 3),
            domain_features=len(features),hidden_features=cfg.hidden_features,dropout=cfg.dropout,seed=cfg.seed+fold*101)
        # Emulate the serialization of a historical completed tiny head; no
        # optimizer was run to generate this synthetic fixture.
        with torch.no_grad(): next(head.parameters()).add_(.01)
        digest = module_state_sha256(head); path = root/f"head{fold}.pt"
        checkpoint = {"checkpoint_format":base.PRIVATE_CHECKPOINT_FORMAT,"module_id":"drug","cell_line_fold":fold,
            "core_checkpoint_sha256":parent["checkpoint_sha256"],"core_parameter_sha256":parent["core_parameter_sha256"],
            "initialization":init.as_dict(),"final_parameter_sha256":digest,"model_state":head.state_dict(),
            "domain_features":features,"domain_mean":np.zeros(len(features),np.float32),"domain_scale":np.ones(len(features),np.float32),
            "pharmacology_features":pharma.receipt() if pharma else None,"optimizer_steps":1,
            "old_checkpoint_loaded":False,"old_predictions_used":False,
            "synthetic_serialization_fixture":True,"no_optimizer_actually_run":True}
        torch.save(checkpoint,path)
        records.append({**record(path),"cell_line_fold":fold,"optimizer_steps":1,
            "initial_parameter_sha256":init.initial_parameter_sha256,"final_parameter_sha256":digest,
            "core_checkpoint_sha256":parent["checkpoint_sha256"],"core_parameter_sha256":parent["core_parameter_sha256"]})
    cp_path = write(root/"checkpoints.json",{"records":records}); artifacts["checkpoint_manifest"] = record(cp_path)
    hashes = {role:artifacts[role]["sha256"] for role in [*roles.values(),"exact_pathway_membership","core_embedding_manifest"]}
    lineage = {"module_id":"drug","training_run_id":"drug-synthetic-current","training_status":"SUCCESS","folds":5,
        "checkpoint_manifest_sha256":sha256(cp_path),"config_sha256":sha256(config_path),
        "input_hashes_start":hashes,"input_hashes_end":hashes,"pharmacology_features":pharma.receipt() if pharma else None,
        "private_head_trained_from_scratch":True,"core_parameters_frozen":True,"all_five_folds_have_optimizer_updates":True,
        "old_checkpoint_loaded":False,"old_predictions_used_as_features":False,"old_rankings_used_as_outputs":False,
        "old_gdsc_prism_association_tables_used":False,"changes_primary_ranking":False,
        "synthetic_serialization_fixture":True,"no_optimizer_actually_run":True}
    lp = write(root/"lineage.json",lineage); artifacts["drug_lineage"] = record(lp)
    am = {"format":"DRUG_MODEL_ATTRIBUTION_INPUT_V1","core_run_id":run,"drug_run_id":lineage["training_run_id"],"artifacts":artifacts}
    am_path = write(root/"ATTRIBUTION_INPUT.json",am)
    assert ablation_main(["--manifest",str(am_path),"--output",str(root/"bootstrap_ablation")]) == 0
    baseline = pd.concat([pd.read_parquet(p) for p in (root/"bootstrap_ablation").glob("baseline*.parquet")])
    available = baseline.loc[baseline.available]; assert not available.empty
    sparse_root = root/"sparse"; pred_root = sparse_root/"predictions"; (pred_root/"cancer-BRCA").mkdir(parents=True)
    pd.DataFrame({"cancer_id":["BRCA"],"lncrna_id":["LNC:Z"],"drug_id":["DC:1"],
        PROBABILITY_COLUMN:[available.baseline_probability.mean()],"availability":[True],
        "prediction_fold_mask":[sum(1<<int(f) for f in available.cell_line_fold)],
        "cell_line_folds_with_prediction":[len(available)]}).to_parquet(pred_root/"cancer-BRCA/part-00000.parquet",index=False)
    sparse = write_drug_sparse_query_sidecars(bundle_root=sparse_root,exact_candidates=tables["exact"],
        pathway_drug_edges=pd.DataFrame({"pathway_id":["P1"],"drug_id":["DC:1"]}),
        assay_fold_eligibility=available[["cancer_id","drug_id","cell_line_fold"]].rename(columns={"cell_line_fold":"cell_line_fold_id"}),
        expression_fold_coverage=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":"LNC:Z","fold_id":range(5)}),
        core_entity_availability=factor_core_entity_availability(lncrna_ids_by_fold={i:["LNC:Z"] for i in range(5)},
            cancer_ids_by_fold={i:["BRCA"] for i in range(5)},drug_target_ids_by_fold={i:["DC:1"] for i in range(5)}),
        available_predictions_path=pred_root,conceptual_candidate_rows=1,training_run_id=lineage["training_run_id"])
    lineage.update(drug_sparse_query_manifest_sha256=sparse["manifest_sha256"],prediction_sha256=artifact_sha256(pred_root))
    write(lp,lineage); artifacts["drug_lineage"] = record(lp); write(am_path,am)
    assert ablation_main(["--manifest",str(am_path),"--output",str(root/"ablation")]) == 0
    args = [item for key,path in paths.items() for item in ["--"+key,str(path)]]
    if dose:
        dose_path = point_source_fixture(root,paths,tables)
        args += ["--dose-manifest",str(dose_path)]
    assert scan_main([*args,"--output",str(root/"response_scan"),"--min-models","4","--min-models-per-dataset","4"]) == 0
    manifest = {"format":INPUT_FORMAT,"core_run_id":run,"drug_run_id":lineage["training_run_id"],
        "core_manifest":record(core_path),"selected_analysis":record(selected_path),
        "sparse_query_manifest":record(Path(sparse["manifest_path"])),"attribution_manifest":record(am_path),
        "attribution_audit":record(root/"ablation/AUDIT.json"),"response_scan_audit":record(root/"response_scan/AUDIT.json")}
    return write(root/"INPUT.json",manifest)


def point_source_fixture(root,paths,tables):
    """Synthetic prior staging receipt plus actual point-measurement reader.

    The real native staging->point CLI is independently covered by dose tests.
    This fixture isolates consumption with the current-core synthetic identities.
    """
    source_map = tables["response"][["dataset_id","drug_id"]].drop_duplicates().assign(source_drug_id=lambda x:x.drug_id)
    mapping_path = root/"source_map.parquet"; source_map.to_parquet(mapping_path,index=False)
    roles = {"response":"raw_cell_line_drug_response","expression":"raw_cell_line_lncrna_expression",
        "mapping":"cell_line_map","targets":"drug_gene_target","subtypes":"model_subtypes"}
    staging = {"status":"SUCCESS","canonical_drug_id_policy":"CURATED_STABLE_ID_OR_SOURCE_SCOPED_ASSAY_ID",
        "input_artifacts_unchanged":True,"staging_run_id":"synthetic-staging-receipt",
        "artifacts":{**{role:record(paths[k]) for k,role in roles.items()},"source_drug_map":record(mapping_path)},
        "input_artifact_snapshot":{"start":{"v32_exact_candidates":record(paths["exact"])}},"synthetic_serialization_fixture":True}
    staging_path = write(root/"STAGING_MANIFEST.json",staging)
    rows = []
    for i in range(100):
        for dose in (0,100):
            # Targetless drug has measured point response in both subtypes.
            # LumA has zero change: observed zero must remain available while
            # the correlation of a constant response is undefined.
            sensitivity = (i%50)/100 if (dose==0 or i>=50) else (i%50)/50
            rows.append({"cell":"GDSC_M"+str(i),"source_drug":"ASSAY:unmapped","well":str(i)+"-"+str(dose),
                "dose":dose,"viability":1-sensitivity})
    point_path = root/"points.csv"; pd.DataFrame(rows).to_csv(point_path,index=False)
    return write(root/"POINT_INPUT.json",{"format":"NATIVE_DRUG_POINT_RESPONSE_V1","staging_manifest":record(staging_path),
        "measurements":[{"source":record(point_path),"condition_kind":"POINT_CONCENTRATION","measurement_level":"OBSERVED_POINT_RESPONSE",
            "columns":{"cell_line_id":"cell","source_drug_id":"source_drug","assay_id":"well","dose_value":"dose","response_value":"viability"},
            "constants":{"dataset_id":"GDSC","dose_unit":"nM","exposure_hours":72,"higher_is_sensitive":False,
                "response_metric":"POINT_VIABILITY","response_unit":"fraction"}}]})


@pytest.fixture(scope="module",params=[False,True],ids=["native","pharmacology_and_dose"])
def released(request,tmp_path_factory,current_sources):
    root = tmp_path_factory.mktemp("current_drug")
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,no_training)
        monkeypatch.setattr(base,"_fit_head",no_training)
        manifest_path = native_fixture(root/"native",current_sources,pharmacology=request.param,dose=request.param)
        result = materialize_current_drug_context(manifest_path,root/"context")
    query = CurrentDrugContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha256=sha256(current_sources[1]))
    return root,manifest_path,result,query,request.param


def test_actual_current_drug_predictions_and_signed_subtypes_remain_distinct(released):
    _,_,_,query,_ = released
    predicted = query.query()["rows"]; assert len(predicted)==1
    row = predicted[0]
    assert row["member_coverage"]==1 and row["selected_weight_coverage"]==1
    assert 0<=row["mean_association_strength_probability"]<=1
    positive = query.query(table="selected_associations_direction_profiles",filters={"drug_id":"ASSAY:unmapped","subtype_id":"Basal"})["rows"][0]
    negative = query.query(table="selected_associations_direction_profiles",filters={"drug_id":"ASSAY:unmapped","subtype_id":"LumA"})["rows"][0]
    assert positive["positive_available_weight_fraction"]==1 and negative["negative_available_weight_fraction"]==1
    missing = query.query(table="selected_associations_direction_profiles",filters={"drug_id":"DC:1","subtype_id":"LumA"})["rows"][0]
    assert missing["available_members"]==0 and missing["positive_available_weight_fraction"] is None
    resolved = query.member_predictions(geneset_id=row["geneset_id"],drug_id="ASSAY:unmapped")["rows"][0]
    assert resolved[PROBABILITY_COLUMN] is None and not resolved["availability"]
    assert resolved["failure_reason"]==OUTSIDE_CONCEPTUAL_UNIVERSE
    available = query.member_predictions(geneset_id=row["geneset_id"],drug_id="DC:1")["rows"][0]
    assert available[PROBABILITY_COLUMN]==pytest.approx(row["mean_association_strength_probability"])


def test_native_ablation_is_consumed_without_inventing_target_pathway_attribution(released):
    _,_,_,query,pharma = released
    rows = query.query(table="selected_attribution")["rows"]
    assert len(rows)==10 and {r["ablation_kind"] for r in rows}=={"TARGET","EXACT_PATHWAY_TARGET_SET"}
    assert all(r["target_pharmacology_recomputed"]==pharma and not r["causal_claimed"] for r in rows)
    assert all("pathway_id" not in r for r in rows)
    profiles = query.query(table="selected_pathway_attribution_profiles")["rows"]
    assert len(profiles)==1 and profiles[0]["minimum_member_folds"]==5
    assert profiles[0]["pathway_id"]=="P1" and not profiles[0]["additive_attribution_claimed"]
    if pharma:
        chem = query.query(table="drug_pharmacology",filters={"drug_id":"DC:1"})["rows"][0]
        assert chem["structure_available"] and chem["ki_exact_negative_log10_molar_median"]==8
        assert not any("morgan" in k for k in chem)
        annotation = query.query(table="drug_target_annotations")["rows"][0]
        assert annotation["negative_log10_molar_activity"]==8 and "p_value_molar" not in annotation


def test_same_selected_version_exact_queries_and_downloads(released):
    _,_,result,query,_ = released
    with pytest.raises(ValueError,match="another selected"):
        CurrentDrugContextQuery(result["binding_path"],expected_sha=result["binding_sha256"],selected_analysis_sha256="0"*64)
    assert not query.query(filters={"drug_id":"DC:1' OR 1=1 --"})["rows"]
    for kwargs in ({"limit":0},{"limit":True},{"offset":-1},{"filters":{"response_value":".3"}}):
        with pytest.raises(ValueError): query.query(**kwargs)
    for name in query.paths:
        download = query.download(name)
        assert sha256(Path(download["path"]))==download["sha256"]
        assert download["binding_sha256"]==result["binding_sha256"]
        assert not {"cell_line_id","canonical_model_id","response_value"}&set(pd.read_parquet(download["path"]))
    with pytest.raises(ValueError): query.download("PRIVATE_DOSE_MEASUREMENT_DISPOSITION")


def test_dose_consumer_preserves_observed_zero_and_undefined_correlation(released):
    _,_,_,query,dose = released
    if not dose:
        assert not query.binding["dose_analysis_available"]
        with pytest.raises(ValueError): query.query(table="selected_dose_contrasts")
        return
    local = query.query(table="selected_dose_contrasts",filters={"drug_id":"ASSAY:unmapped","subtype_id":"LumA"})["rows"][0]
    assert local["measured_response_available"] and local["mean_observed_oriented_response"]==0
    assert local["signed_expression_sensitivity_rho"] is None and not local["association_available"]
    positive = query.query(table="selected_dose_contrasts_direction_profiles",filters={"subtype_id":"Basal"})["rows"][0]
    assert positive["positive_available_weight_fraction"]==1 and positive["low_dose_molar_key"]=="0"
    assert positive["high_dose_molar_key"]=="0.0000001" and positive["exposure_hours_key"]=="72"


def test_independent_replay_and_modified_public_download_fail_closed(released,tmp_path):
    root,_,result,query,_ = released
    assert audit_current_drug_context(result["binding_path"],expected_sha=result["binding_sha256"])["status"]=="PASS"
    # Restore the file because the same module fixture serves other checks.
    path = query.paths["selected_drug_predictions"]; original = path.read_bytes()
    try:
        path.write_bytes(original+b"tampered")
        with pytest.raises(ValueError,match="SHA256|hash"): query.download("selected_drug_predictions")
    finally: path.write_bytes(original)


@pytest.mark.parametrize("kind",["response_scan_audit","attribution_audit"])
def test_rehashed_native_output_forgery_fails_actual_replay(released,tmp_path,kind):
    _,manifest_path,_,_,_ = released
    manifest = json.loads(manifest_path.read_text()); original_audit_path = Path(manifest[kind]["path"])
    audit = json.loads(original_audit_path.read_text())
    prefix = "associations/" if kind=="response_scan_audit" else "attribution_"
    name = next(n for n in audit["outputs"] if n.startswith(prefix))
    original = Path(audit["outputs"][name]["path"]); data = pd.read_parquet(original)
    column = "signed_expression_sensitivity_rho" if kind=="response_scan_audit" else "delta_probability"
    data.loc[0,column] = .12345
    changed = tmp_path/"changed.parquet"; data.to_parquet(changed,index=False)
    audit["outputs"][name] = {**audit["outputs"][name],**record(changed)}
    audit_path = write(tmp_path/"AUDIT.json",audit)
    write(tmp_path/"SUCCESS.json",{"status":"PASS","audit_sha256":sha256(audit_path)})
    manifest[kind] = record(audit_path); altered = write(tmp_path/"INPUT.json",manifest)
    context = current_drug_sources(altered)
    with pytest.raises(ValueError,match="replay differs"): replay_native_reports(context)
