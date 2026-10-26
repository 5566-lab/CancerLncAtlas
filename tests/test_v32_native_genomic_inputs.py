"""Actual raw CNV producer and actual partition training input path, no fit."""
from pathlib import Path
from copy import deepcopy
import hashlib
import shutil
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_genomic_cnv as native
from cc_hhgt.v32 import segment_cnv_streaming as streaming
from cc_hhgt.v32 import genomic_partition_training as trainer
from cc_hhgt.v32 import genomic_resource_staging as resources
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_directional_cnv import inputs as raw_inputs
from test_v32_current_core_export import export_fixture
from test_v32_local_cnv_cohort import CANCERS, LNCS, PATHWAY


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**k): raise AssertionError("No optimizer or genomic forward permitted")
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
    for module in (trainer,__import__("cc_hhgt.v32.genomic_training",fromlist=["_"])):
        monkeypatch.setattr(module,"_fit_head",forbidden); monkeypatch.setattr(module,"_predict_head",forbidden)


def inputs(root):
    raw_manifest, _, args = raw_inputs(root)
    spec=read(raw_manifest)
    # Smaller declared patient cohort; actual candidate annotation replay is retained.
    patients=pd.read_csv(args.patient_map,sep="\t"); patients=patients.loc[patients.patient_id.str[-3:].astype(int)<20]
    # The shared patient authority may include cancers outside this declared run.
    extra=pd.DataFrame({"cancer_id":"COAD","patient_id":[f"COAD_UNUSED_{i}" for i in range(5)],"patient_fold_id":range(5)})
    pd.concat([patients,extra],ignore_index=True).to_csv(args.patient_map,sep="\t",index=False)
    spec["patient_map"]=record(args.patient_map)
    spec["segments"]=[r for r in spec["segments"] if r["patient_id"] in set(patients.patient_id)]
    genes=[]; lncs=[]
    for row in spec["segments"]:
        index=int(row["patient_id"][-3:]); parity=index%2
        if index==19:
            row.pop("file");row.pop("study_metadata");row.update(status="typed_unavailable",unavailable_reason="NO_SELECTED_SEGMENT_FILE")
        else:
            path=Path(row["file"]["path"])
            pd.DataFrame([("chr1",0,24,.6*parity),("chr1",25,44,.6*(1-parity)),("chr1",45,74,0.),
                ("chr1",75,150,.6*parity),("chr1",200,224,-.6*parity),("chr2",0,100,0.)],
                columns=["Chromosome","Start","End","Segment_Mean"]).assign(Sample=row["patient_id"]).to_csv(path,sep="\t",index=False)
            row["file"]=record(path)
        for gene in [f"G{i}" for i in range(1,11)]:
            event=bool(parity and gene=="G1")
            genes.append(dict(cancer_id=row["cancer_id"],patient_id=row["patient_id"],gene_id=gene,
                              is_mutated=event,mutation_count=int(event),absence_is_wildtype=not event))
        for name in LNCS:
            event=bool(parity if name=="LNC:A" else 1-parity if name=="LNC:B" else 0)
            lncs.append(dict(cancer_id=row["cancer_id"],patient_id=row["patient_id"],lncrna_id=name,
                             lncrna_any_mutation=event,lncrna_mutation_available=True,
                             exonic_variant_count=int(event),absence_is_wildtype=not event))
    write(raw_manifest,spec)
    gene=root/"sample_gene_mutation.parquet"; pd.DataFrame(genes).to_parquet(gene,index=False)
    lnc=root/"sample_lncrna_mutation.parquet"; pd.DataFrame(lncs).to_parquet(lnc,index=False)
    mc3=root/"mc3.tsv"; mc3.write_text("Hugo_Symbol\tTumor_Sample_Barcode\nG1\tSYNTHETIC\n")
    core_root=root/"core";core_root.mkdir();core_path,core=export_fixture(core_root)
    for fold in core["folds"].values():
        for kind,ids in (("lncRNA",LNCS),("pathway",[PATHWAY])):
            path=core_root/fold["exports"][kind]["path"]
            pd.DataFrame(dict(node_index=range(len(ids)),node_id=ids,core_feature_000=np.arange(len(ids),dtype=float),core_feature_001=1.)).to_parquet(path,index=False)
            fold["exports"][kind]={**record(path),"rows":len(ids),"features":2}
    core["fixture_no_optimizer_run"]=True;write(core_path,core)
    manifest=root/"native_genomic.json"
    write(manifest,dict(format=native.INPUT,segment_manifest=record(raw_manifest),sample_gene_mutation=record(gene),
        sample_lncrna_mutation=record(lnc),mc3=record(mc3),core_manifest=record(core_path),core_run_id="current-G2",max_memory_bytes=10**12))
    return manifest,args


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp("native_genomic")
    manifest,args=inputs(root/"inputs")
    from scripts.materialize_native_genomic_cnv import main
    output=root/"streaming"
    assert main(["prepare","--manifest",str(manifest),"--output",str(output),"--cancers",CANCERS[0]])==2
    assert not (output/"SUCCESS.json").exists()
    assert main(["prepare","--manifest",str(manifest),"--output",str(output),"--cancers",CANCERS[1]])==0
    return manifest,args,output


def stage(completed, root):
    manifest,args,stream=completed; candidates=pd.read_parquet(args.candidates)
    counts={c:int(candidates.cancer_id.eq(c).sum()) for c in CANCERS}
    ordered=pd.concat([resources._canonical_partition(candidates.loc[candidates.cancer_id.eq(c)],c) for c in CANCERS])
    digest=hashlib.sha256("".join("\t".join(r)+"\n" for r in ordered.itertuples(index=False,name=None)).encode()).hexdigest()
    budget=root/"budget.json"
    write(budget,dict(format=resources.BUDGET_FORMAT,resource_budget_id="native-genomic-synthetic",
        scope=dict(formal_cancers=CANCERS,candidate_rows=len(candidates),candidate_rows_by_cancer=counts),
        adapter_budget=dict(max_candidate_partition_rows=max(counts.values()),max_declared_adapter_peak_rss_bytes=10**12,
                            max_memmap_bytes=10**8,max_staging_disk_bytes=10**8),
        downstream_budget=dict(max_declared_process_rss_bytes=10**12,minimum_host_available_memory_bytes=1,
                               minimum_host_free_disk_bytes=1,forbid_launch_without_resource_success=True),
        memmap_layout=dict(modalities=["mutation","cnv"],patient_folds=5,probability_dtype="float32",available_dtype="bool",processed_dtype="bool"),
        input_bindings=dict(candidate_authority={**record(args.candidates),"ordered_exact_candidate_key_sha256":digest},
            cnv_streaming_aggregate_success={**record(stream/"SUCCESS.json"),"format":resources.CNV_STREAMING_FORMAT,
                "status":"SUCCESS","sha256_policy":"CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH"})))
    resources.stage_genomic_cancer_memmaps(candidate_authority_path=args.candidates,genomic_training_source_path=Path(trainer.__file__).with_name("genomic_training.py"),
        cnv_streaming_success_path=stream/"SUCCESS.json",cnv_streaming_success_sha256=record(stream/"SUCCESS.json")["sha256"],
        budget_contract_path=budget,output_root=root/"stage")
    spec=read(manifest)
    return ["--resource-staging-success",str(root/"stage/SUCCESS.json"),"--patient-folds",str(args.patient_map),
        "--candidate-definition",str(manifest.parent/"CANDIDATE_SOURCES.json"),"--candidate-definition-sha256",record(manifest.parent/"CANDIDATE_SOURCES.json")["sha256"],
        "--expected-core-run-id","current-G2","--pathway-gene-membership",read(read(manifest)["segment_manifest"]["path"])["membership"]["path"],
        "--sample-gene-mutation",spec["sample_gene_mutation"]["path"],"--sample-lncrna-mutation",spec["sample_lncrna_mutation"]["path"],
        "--mc3",spec["mc3"]["path"],"--core-embedding-manifest",spec["core_manifest"]["path"],"--cnv-streaming-success",str(stream/"SUCCESS.json"),
        "--output-root",str(root/"prepared"),"--training-run-id","v32-native-genomic-fixture","--min-pair-callable","2","--prepare-only"]


def test_actual_raw_genomic_arrays_replay_and_canonical_event_estimand(completed,tmp_path):
    _,_,output=completed
    result=streaming.validate_streaming_store(output/"SUCCESS.json")
    assert result["required_cancers"]==CANCERS and not result["source_ancestry_complete"]
    assert read(output/"LAUNCH_PREFLIGHT.json")["target_host"]=="149"
    for cancer in CANCERS:
        part=output/f"cancer={cancer}"; info=read(part/"SUCCESS.json")
        assert info["processed_segment_patients"]==19 and info["typed_unavailable_patients"]==1
        lnc_ids=read(part/"lncrna_ids.json");a=lnc_ids.index("LNC:A");b=lnc_ids.index("LNC:B")
        burden=np.load(part/"lncrna_burden.npy");event=np.load(part/"lncrna_event.npy")
        # Opposite signed intervals retain .6 absolute burden and an event.
        np.testing.assert_allclose(burden[:19,a],.6*(np.arange(19)%2),atol=1e-7)
        np.testing.assert_array_equal(event[:19,a],np.arange(19)%2)
        np.testing.assert_array_equal(event[:19,b],1-np.arange(19)%2)
        assert np.isnan(burden[-1]).all() and not np.load(part/"lncrna_callable.npy")[-1].any()
        np.testing.assert_allclose(np.load(part/"pathway_burden.npy")[:19,0],.6*(np.arange(19)%2),atol=1e-7)
    from scripts.materialize_native_genomic_cnv import main
    audit=tmp_path/"audit.json"
    assert main(["audit","--success",str(output/"SUCCESS.json"),"--expected-sha256",record(output/"SUCCESS.json")["sha256"],"--output",str(audit)])==0
    assert read(audit)["arrays_compared"]==12


def test_actual_partition_training_cli_collects_all_inputs_without_fit_or_memmap_mutation(completed,tmp_path):
    from scripts.train_v32_partitioned_genomic_heads import main
    argv=stage(completed,tmp_path)
    before={p.name:record(p)["sha256"] for p in (tmp_path/"stage/memmaps").iterdir()}
    assert main(argv)==0
    ready=read(tmp_path/"prepared/HEAD_PREFLIGHT.json")
    assert ready["status"]=="PASS_READY" and ready["target_host"]=="149"
    assert not ready["training_started"] and not ready["model_forwards_executed"] and not ready["predictions_written"]
    assert ready["candidate_rows"]==6 and ready["cohort"]["cancers"]==CANCERS
    assert len(ready["folds"])==10
    for row in ready["folds"]:
        assert row["train_rows"]==row["validation_rows"]==4
        assert row["train_positive_rows"]==row["validation_positive_rows"]==2
        assert row["core_features"]==4 and row["domain_features"]==6
        assert row["split_policy"]["patient_counts"]=={"train":24,"validation":8,"test":8}
    assert before=={p.name:record(p)["sha256"] for p in (tmp_path/"stage/memmaps").iterdir()}
    assert not (tmp_path/"prepared/checkpoints").exists() and not (tmp_path/"stage/write_receipts").exists()
    assert not list(tmp_path.rglob("TRAINING_SUCCESS.json")) and not list(tmp_path.rglob("PREDICTIONS_READY.json"))


@pytest.mark.parametrize("fault",["threshold","wrong_core","definition_hash","missing_core","one_class","patient_map"])
def test_actual_trainer_native_preflight_rejects_inconsistent_inputs(completed,tmp_path,fault):
    from scripts.train_v32_partitioned_genomic_heads import main
    argv=stage(completed,tmp_path)
    if fault=="threshold":argv.extend(["--cnv-event-threshold","0.5"])
    elif fault=="wrong_core":argv[argv.index("--expected-core-run-id")+1]="another-core"
    elif fault=="definition_hash":argv[argv.index("--candidate-definition-sha256")+1]="0"*64
    elif fault=="missing_core":i=argv.index("--expected-core-run-id");del argv[i:i+2]
    elif fault=="one_class":argv.extend(["--min-pair-callable","1000"])
    elif fault=="patient_map":
        path=tmp_path/"changed_patients.tsv";frame=pd.read_csv(completed[1].patient_map,sep="\t")
        frame.loc[0,"patient_id"]="CHANGED";frame.to_csv(path,sep="\t",index=False)
        argv[argv.index("--patient-folds")+1]=str(path)
    with pytest.raises((ValueError,RuntimeError)):main(argv)
    assert not (tmp_path/"prepared/HEAD_PREFLIGHT.json").exists()
    assert not (tmp_path/"stage/write_receipts").exists()


@pytest.mark.parametrize("name",native.ARRAYS)
def test_rehashed_streaming_values_fail_actual_numerical_replay(completed,tmp_path,name):
    output=tmp_path/"copy";shutil.copytree(completed[2],output)
    part=output/f"cancer={CANCERS[0]}"; array=np.load(part/name)
    array[0,0]=.7 if "burden" in name else 1-array[0,0];np.save(part/name,array)
    info=read(part/"SUCCESS.json");info["files"][name]=record(part/name)["sha256"];write(part/"SUCCESS.json",info)
    value=read(output/"SUCCESS.json")
    for cancer,item in value["cancers"].items():
        p=output/f"cancer={cancer}";item.update(path=str(p),success_sha256=record(p/"SUCCESS.json")["sha256"],files_composite_sha256=streaming._json_sha(read(p/"SUCCESS.json")["files"]))
    write(output/"SUCCESS.json",value)
    native.validate(output/"SUCCESS.json")
    with pytest.raises(ValueError,match="numerical replay differs"):native.validate(output/"SUCCESS.json",replay=True)


@pytest.mark.parametrize("fault",["raw_hash","wrong_patient","missing_patient","candidate_definition","threshold","core_run","memory"])
def test_native_materializer_rejects_invalid_input_contract(completed,tmp_path,fault):
    spec=deepcopy(read(completed[0])); segments=deepcopy(read(spec["segment_manifest"]["path"]))
    if fault=="raw_hash":segments["segments"][0]["file"]["sha256"]="0"*64
    elif fault=="wrong_patient":segments["segments"][0]["patient_id"]="UNKNOWN"
    elif fault=="missing_patient":segments["segments"].pop()
    elif fault=="candidate_definition":segments["candidate_definition"]["sha256"]="0"*64
    elif fault=="threshold":segments["event_threshold"]=0.
    elif fault=="core_run":spec["core_run_id"]="OTHER"
    elif fault=="memory":spec["max_memory_bytes"]=1
    path=tmp_path/"segments.json";write(path,segments);spec["segment_manifest"]=record(path)
    manifest=tmp_path/"input.json";write(manifest,spec)
    with pytest.raises((ValueError,RuntimeError)):native.materialize(record(manifest),tmp_path,tmp_path/"out")
    assert not (tmp_path/"out/SUCCESS.json").exists()


def test_changed_raw_source_invalidates_existing_release(completed):
    spec=read(read(completed[0])["segment_manifest"]["path"]);path=Path(spec["segments"][0]["file"]["path"])
    original=path.read_bytes()
    try:
        path.write_bytes(original+b"\n")
        with pytest.raises(ValueError):streaming.validate_streaming_store(completed[2]/"SUCCESS.json")
    finally:path.write_bytes(original)


def test_actual_legacy_entry_still_rejects_a_nonfrozen_patient_authority(completed,tmp_path):
    from scripts.train_v32_partitioned_genomic_heads import main
    argv=stage(completed,tmp_path)
    for option in ("--candidate-definition","--candidate-definition-sha256","--expected-core-run-id"):
        i=argv.index(option);del argv[i:i+2]
    argv.extend(["--patient-fold-authority-receipt",str(completed[0].parent/"CANDIDATE_SOURCES.json")])
    with pytest.raises((ValueError,RuntimeError)):main(argv)
    assert not (tmp_path/"prepared").exists()


def frozen_completion(completed,root):
    from test_v32_native_auxiliary_fusion import fixture
    from test_v32_native_partitioned_genomic import partitioned
    manifest,args,stream=completed; spec=read(manifest)
    keys=pd.read_parquet(args.candidates)
    path,_=fixture(root,core_override=(Path(spec["core_manifest"]["path"]),keys,"current-G2"))
    declaration=read(path);declaration.update(candidate_authority=record(args.candidates),patient_folds=record(args.patient_map));write(path,declaration)
    lp=root/"native/LINEAGE.json";lineage=read(lp)
    lineage["input_artifacts"]=list(read(stream/"RUN_CONTEXT.json")["bindings"].values())+[record(stream/"SUCCESS.json")]
    write(lp,lineage)
    return partitioned(root,existing=path,streaming_success=stream/"SUCCESS.json")[0]


def test_completed_native_partitions_reach_fusion_adapter_and_actual_model_use_source_replay(completed,tmp_path):
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary,load_native_auxiliary
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32.current_auxiliary_model_use import input_source
    declaration=frozen_completion(completed,tmp_path)
    binding=materialize_native_auxiliary(declaration,tmp_path/"adapter",expected_core_run_id="current-G2")
    frame,audit=load_native_auxiliary(binding,expected_sha=record(binding)["sha256"],expected_core_run_id="current-G2")
    assert len(frame)==6 and not audit["training_started"]
    inventory=Inventory([],tmp_path)
    source=record(completed[2]/"SUCCESS.json")
    input_source(inventory,source,completed[2],"genomic","streaming")
    proofs=[r for r in inventory.transform_replays if r["format"]==native.REPLAY]
    assert len(proofs)==1 and proofs[0]["outputs_compared"]==12
    assert pd.DataFrame(inventory.usage).study_tokens_json.str.contains("99112233").any()
    assert "SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED" in {r["reason"] for r in inventory.gaps}


@pytest.mark.parametrize("fault",["cohort","core_run","threshold","omitted_input"])
def test_completed_native_consumer_rejects_repinned_training_context(completed,tmp_path,fault):
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    from test_v32_native_partitioned_genomic import repin_training
    path=frozen_completion(completed,tmp_path)
    lp=tmp_path/"native/LINEAGE.json";value=read(lp)
    if fault=="cohort":value["cohort"]["candidate_rows"]+=1
    elif fault=="core_run":value["core_run_id"]="OTHER"
    elif fault=="threshold":value["configuration"]["cnv_event_threshold"]=.7
    elif fault=="omitted_input":
        p=tmp_path/"native/INPUT_LINEAGE_AUDIT.json";audit=read(p)
        digest=read(completed[0])["sample_gene_mutation"]["sha256"]
        audit["artifacts"]=[r for r in audit["artifacts"] if r["sha256"]!=digest];write(p,audit)
    write(lp,value);repin_training(path)
    with pytest.raises((ValueError,RuntimeError)):
        materialize_native_auxiliary(path,tmp_path/"adapter",expected_core_run_id="current-G2")


def test_native_cohort_cannot_be_removed_to_enter_legacy_consumer_validation(completed,tmp_path):
    from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary
    from test_v32_native_partitioned_genomic import repin_training
    from cc_hhgt.v32.genomic_training import normalise_patient_folds,_split_patients
    path=frozen_completion(completed,tmp_path)
    patients=normalise_patient_folds(pd.read_csv(completed[1].patient_map,sep="\t"))
    traces=[trainer._patient_split_trace({s:_split_patients(patients,f,s) for s in ("train","validation","test")},f,m)
            for f in range(5) for m in ("mutation","cnv")]
    lp=tmp_path/"native/LINEAGE.json";value=read(lp);value.pop("cohort")
    policy_path=tmp_path/"native/SPLIT_POLICY.json";policy=read(policy_path);policy["traces"]=traces
    write(policy_path,policy);value["split_policy"]=policy;write(lp,value)
    cp_path=tmp_path/"native/CHECKPOINT_MANIFEST.json";checkpoints=read(cp_path)
    for row in checkpoints["records"]:
        checkpoint=Path(row["path"]);container=torch.load(checkpoint,weights_only=False)
        container["split_policy"]=next(t for t in traces if (t["patient_fold"],t["modality"])==(row["patient_fold"],row["modality"]))
        torch.save(container,checkpoint);row.update(record(checkpoint))
    write(cp_path,checkpoints);repin_training(path)
    with pytest.raises(ValueError,match="cohort declaration"):
        materialize_native_auxiliary(path,tmp_path/"adapter",expected_core_run_id="current-G2")
