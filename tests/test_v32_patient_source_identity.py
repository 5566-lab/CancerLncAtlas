"""File-scoped patient identity reaches native consumers and public exclusion."""
import json
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_model_use import Inventory,record,read,write,derive_manifest,materialize_current_model_use,load_current_model_use
from cc_hhgt.v32.patient_source_identity import IdentityMap,PatientUse
from cc_hhgt.v32.current_core_export import sha256


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("Patient identity checks must not optimize")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)


def crosswalk(path,rows,namespace="SYNTHETIC_PATIENT_AUTHORITY"):
    pd.DataFrame(rows,columns=["source_file_sha256","identity_column","raw_id","canonical_patient_id"]).to_parquet(path,index=False)
    return {**record(path),"namespace":namespace}


def test_file_scopes_aliases_and_public_overlap_counts(tmp_path):
    a=tmp_path/"a.parquet";b=tmp_path/"b.parquet"
    pd.DataFrame({"patient_id":["P1","P2"]}).to_parquet(a,index=False)
    pd.DataFrame({"donor_id":["P1","ALIAS"]}).to_parquet(b,index=False)
    declaration=crosswalk(tmp_path/"identity.parquet",[
        (sha256(a),"patient_id","P1","PERSON_A"),(sha256(a),"patient_id","P2","PERSON_B"),
        (sha256(b),"donor_id","P1","UNRELATED_PERSON"),(sha256(b),"donor_id","ALIAS","PERSON_A")])
    use=PatientUse(IdentityMap(declaration,tmp_path))
    for path,context in ((a,"CORE"),(b,"SC")):
        use.table(path,digest=sha256(path),model="fusion",context=context,role="input",scope="WHOLE_FILE")
    tables=use.tables();row=tables["patient_context_overlap"].iloc[0]
    assert row.shared_patient_count==1
    assert tables["patient_mapping"].unmapped_identity_count.eq(0).all()
    serialized=json.dumps({k:v.to_dict("records") for k,v in tables.items()})
    assert all(value not in serialized for value in ("PERSON_A","PERSON_B","UNRELATED_PERSON","ALIAS"))
    assert "patient_token" not in tables["patient_context_overlap"]


@pytest.mark.parametrize("fault",["ambiguous","namespace","digest"])
def test_invalid_identity_authority_rejected(tmp_path,fault):
    rows=[("a"*64,"patient_id","P1","PERSON_A")]
    if fault=="ambiguous":rows.append(("a"*64,"patient_id","P1","PERSON_B"))
    if fault=="digest":rows[0]=("not_a_hash","patient_id","P1","PERSON_A")
    declaration=crosswalk(tmp_path/"identity.parquet",rows)
    if fault=="namespace":declaration.pop("namespace")
    with pytest.raises(ValueError):IdentityMap(declaration,tmp_path)


def test_numeric_looking_csv_ids_keep_leading_zeroes(tmp_path):
    from cc_hhgt.v32.patient_source_identity import ExternalPatientIsolation
    source=tmp_path/"patients.csv";source.write_text("patient_id\n001\n")
    mapping=tmp_path/"identity.tsv"
    pd.DataFrame([(sha256(source),"patient_id","001","PERSON_001")],
        columns=["source_file_sha256","identity_column","raw_id","canonical_patient_id"]).to_csv(mapping,sep="\t",index=False)
    declaration={**record(mapping),"namespace":"SYNTHETIC"};use=PatientUse(IdentityMap(declaration,tmp_path))
    use.table(source,digest=sha256(source),model="m",context="core",role="input",scope="WHOLE_FILE")
    tables=use.tables();assert tables["patient_mapping"].mapped_identity_count.tolist()==[1]
    checker=ExternalPatientIsolation(declaration,tmp_path,{"patient_identity_namespace":"SYNTHETIC"},tables)
    # The general outcome reader may infer a numeric dtype. Identity reading
    # preserves the original textual ID independently and aligns by row order.
    annotated=checker.annotate(pd.read_csv(source),{"table":record(source)},path=source)
    assert annotated.patient_overlap.tolist()==[True]


def test_native_genomic_alias_overlap_reaches_external_query_without_identifiers(tmp_path):
    from test_v32_auxiliary_model_use import setup
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    source,raw=setup(tmp_path/"native","genomic")
    external=tmp_path/"external.parquet"
    pd.DataFrame({"record_id":["r1","r2","r3"],"lncrna":["new"]*3,"cancer":["breast"]*3,"pathway":["p"]*3,
        "label":[1]*3,"pmid":["99881122"]*3,"is_experimental":[True]*3,"is_predicted":[False]*3,"assay":["Synthetic"]*3,
        "patient_id":["EXTERNAL_ALIAS","PRIVATE_TEST_PATIENT","UNMAPPED_ALIAS"]}).to_parquet(external,index=False)
    declaration=crosswalk(tmp_path/"identities.parquet",[
        (sha256(raw),"patient_id","PRIVATE_TEST_PATIENT","PRIVATE_CANONICAL_PERSON"),
        (sha256(external),"patient_id","EXTERNAL_ALIAS","PRIVATE_CANONICAL_PERSON"),
        (sha256(external),"patient_id","PRIVATE_TEST_PATIENT","DIFFERENT_PRIVATE_PERSON")])
    request=read(source);request["patient_identity_crosswalk"]=declaration;write(source,request)
    inventory=Path(materialize_current_model_use(source,tmp_path/"inventory")["binding_path"])
    binding,tables=load_current_model_use(inventory,expected_sha=sha256(inventory))
    assert binding["outputs"]["patient_usage_private"]["private"] is True
    assert tables["patient_context_overlap"].shared_patient_count.eq(1).all()
    entities=tmp_path/"entities.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","PATHWAY:P1")],
        columns=["namespace","raw_id","canonical_id"]).to_parquet(entities,index=False)
    external_input=tmp_path/"external.json"
    policy={"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":request["models"],"model_use_inventory":record(inventory),
        "identity_crosswalk":record(entities),"patient_identity_crosswalk":declaration,
        "external_sources":[{"kind":"exact_assay","source_id":"patient_only_overlap","table":record(external)}],
        "evaluation":{"primary_model":"fusion","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}}
    write(external_input,policy);result=materialize_current_external(external_input,tmp_path/"external")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows={r["raw_record_id"]:r for r in query.query(table="isolation")["rows"]}
    assert rows["r1"]["patient_overlap"] and rows["r1"]["overlapping_patient_count"]==1
    assert not rows["r1"]["study_overlap"] and not rows["r1"]["lncrna_overlap"]
    assert "PATIENT_OVERLAP" in rows["r1"]["isolation_reasons_json"]
    assert not rows["r2"]["patient_overlap"] and rows["r2"]["external_patient_identity_complete"]
    assert not rows["r3"]["external_patient_identity_complete"] and not rows["r3"]["provenance_held_out"]
    for table in query.binding["outputs"]:
        public=json.dumps(query.query(table=table)["rows"])
        assert not any(value in public for value in ("patient_token","EXTERNAL_ALIAS","PRIVATE_CANONICAL_PERSON","PRIVATE_TEST_PATIENT"))
    with pytest.raises(ValueError,match="Unknown"):query.query(table="patient_usage_private")
    leaked=read(Path(result["binding_path"]));leaked["outputs"]["patient_usage_private"]=binding["outputs"]["patient_usage_private"]
    leaked_path=tmp_path/"LEAKED.json";write(leaked_path,leaked)
    with pytest.raises(ValueError,match="approved aggregate tables"):
        CurrentExternalValidationQuery(leaked_path,expected_sha=sha256(leaked_path))
    leaked["outputs"].pop("patient_usage_private")
    leaked["outputs"]["patient_mapping"]={k:v for k,v in binding["outputs"]["patient_usage_private"].items() if k!="private"}
    write(leaked_path,leaked)
    with pytest.raises(ValueError,match="private patient identity columns"):
        CurrentExternalValidationQuery(leaked_path,expected_sha=sha256(leaked_path))
    forged=read(inventory);forged["outputs"]["patient_usage_private"]["private"]=False
    changed=tmp_path/"FORGED.json";write(changed,forged)
    with pytest.raises(ValueError,match="remain private"):load_current_model_use(changed,expected_sha=sha256(changed),replay=False)
    policy["patient_identity_crosswalk"]={**declaration,"namespace":"UNRELATED_AUTHORITY"};write(external_input,policy)
    with pytest.raises(ValueError,match="namespace differs"):materialize_current_external(external_input,tmp_path/"wrong_namespace")


def test_actual_cnv_compact_referenced_patients_are_collected(tmp_path):
    from test_v32_cnv_input_sources import attach_consumed
    from test_v32_native_auxiliary_fusion import fixture
    from cc_hhgt.v32.native_auxiliary_fusion import native_sources
    from cc_hhgt.v32.current_cnv_model_use import inspect_consumed_sources
    path,_=fixture(tmp_path,"cnv");attach_consumed(path)
    rows=[]
    for cancer in ("BRCA","LAML"):
        file=tmp_path/f"compact_{cancer}"/"patient_ids.json"
        rows.extend((sha256(file),"patient_ids",raw,"CANONICAL_"+raw) for raw in read(file))
    declaration=crosswalk(tmp_path/"identities.parquet",rows)
    inventory=Inventory([],tmp_path,declaration)
    context=native_sources(path,expected_core_run_id="current-G2")
    inspect_consumed_sources(inventory,context,"fusion","cnv")
    audit=inventory.patient_use.tables()["patient_mapping"]
    compact=audit.loc[audit.source_role.eq("CNV_COMPACT_REFERENCED_PATIENT_ROWS")]
    assert compact.observed_identity_count.tolist()==[5,5]
    assert compact.unmapped_identity_count.eq(0).all()


def test_native_sc_actual_episode_donors_are_mapped(tmp_path,monkeypatch):
    from test_v32_single_cell_model_use import native_fixture,fusion_fixture
    import scripts.train_single_cell_hhgt_149 as trainer
    binding,raw=native_fixture(tmp_path/"native")
    pointer=read(tmp_path/"native/pointer.json");episodes=read(Path(pointer["input_manifest"]["path"]))
    rows=[]
    for fold in episodes["folds"]:
        for item in fold["episodes"]:
            episode=torch.load(item["path"],weights_only=True)
            for column,values in (("source_donors",episode["source_donors"]),("query_donor_id",episode["row_metadata"]["query_donor_id"])):
                rows.extend((item["sha256"],column,v,"CANONICAL_"+v) for v in set(values))
    rows.append((sha256(raw),"patient_id","PRIVATE_RAW_PATIENT","CANONICAL_RAW_PERSON"))
    declaration=crosswalk(tmp_path/"identities.parquet",rows)
    source=fusion_fixture(tmp_path/"native",binding);request=read(source);request["patient_identity_crosswalk"]=declaration;write(source,request)
    def forbidden(*a,**kw):raise AssertionError("No model construction during donor inspection")
    monkeypatch.setattr(trainer,"_initialize_model",forbidden)
    _,_,tables,_=derive_manifest(source)
    audit=tables["patient_mapping"];episodes=audit.loc[audit.source_role.str.startswith("SC_HHGT_EPISODE")]
    assert len(episodes)==80 and episodes.status.eq("MAPPED").all()
    assert {"TRAINING","MODEL_SELECTION","FEATURE_CONSTRUCTION"} <= set(episodes.phase)
    assert tables["patient_context_overlap"].shared_patient_count.gt(0).all()


def test_core_preparation_read_uses_file_scoped_patient_mapping(tmp_path,monkeypatch):
    import test_v32_preparation_sources as fixtures
    from cc_hhgt.v32.preparation_sources import attach_sources
    original=fixtures.load_adjusted_associations
    def reader(capture,train,validation,**kw):
        for directory in (train,validation):
            for path in directory.rglob("*.parquet"):
                pd.read_parquet(path).assign(patient_id="CORE_PATIENT_ALIAS").to_parquet(path,index=False)
        return original(capture,train,validation,**kw)
    monkeypatch.setattr(fixtures,"load_adjusted_associations",reader)
    _,receipt,_=fixtures.adjusted_capture(tmp_path/"sources")
    rows=[(r["sha256"],"patient_id","CORE_PATIENT_ALIAS","CANONICAL_CORE_PERSON")
          for r in receipt["source_reads"] if r["role"].endswith("_associations")]
    declaration=crosswalk(tmp_path/"identities.parquet",rows)
    payload={"patient_fold":0,"input_scope":{"association_mode":"local-cnv-adjusted"}};attach_sources(payload,receipt)
    inventory=Inventory([],tmp_path,declaration);inventory.preparation(payload,tmp_path,"core",0,False)
    audit=inventory.patient_use.tables()["patient_mapping"]
    assert audit.loc[audit.observed_identity_count.gt(0),"status"].eq("MAPPED").all()
    assert set(audit.loc[audit.observed_identity_count.gt(0),"phase"])=={"FEATURE_CONSTRUCTION","MODEL_SELECTION"}
