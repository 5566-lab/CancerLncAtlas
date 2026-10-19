"""Real GSVA synthetic oracles and native consumer isolation; no fitting."""
from pathlib import Path
import socket
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from cc_hhgt.v32.fold_local_activity import convert, binding, validate, inspect, activity_path
from cc_hhgt.v32.formal_expression_sources import read, write, record, read_matrix
from cc_hhgt.v32.current_model_use import Inventory
from test_v32_fold_local_expression import normalized, run_r, forbidden

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT/"scripts/convert_fold_local_activity.py"
R_RUNNER = ROOT/"scripts/materialize_fold_local_ssgsea.R"
HELPER = ROOT/"scripts/frozen_ssgsea.R"
P1 = "MSIGDB:HALLMARK:P1"
SMALL = "MSIGDB:HALLMARK:SMALL"


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def activity(normalized):
    root = normalized/"activity_inputs"; root.mkdir()
    annotation = root/"annotation.tsv"
    # Two Ensembl genes share S1. G80 is expressed in held-out patients only
    # for fold 0; Z is explicitly noncoding and must never enter pathways.
    pd.DataFrame({"ensembl_gene_id":["Z", *[f"G{i}" for i in range(1,81)]],
        "gene_symbol":["S0", "S1", "S1", *[f"S{i}" for i in range(3,81)]],
        "gene_type":["lncRNA", *["protein_coding"]*80], "pmid":"99887711"}).to_csv(annotation,sep="\t",index=False)
    sets = {"P1":["S1", *[f"S{i}" for i in range(3,22)]],
            "P2":[f"S{i}" for i in range(30,55)], "SMALL":["S80", "S0"],
            "LARGE":[f"S{i}" for i in range(1,502)]}
    gmt = root/"hallmark.gmt"
    gmt.write_text("\n".join("\t".join([key,"explicit synthetic source",*genes]) for key,genes in sets.items())+"\n",encoding="utf-8")
    manifest = root/"gmt.json"
    write(manifest,{"format":"SSGSEA_GMT_INPUT_V1","collections":[
        {"collection":"HALLMARK","file":record(gmt),"study_metadata":{"pmid":"12344321"}}]})
    for name in ("normal", "changed"):
        run_r(R_RUNNER, normalized/name, annotation, manifest, root/name)
    result=subprocess.run([sys.executable,str(RUNNER),"--source",str(root/"normal"),"--output",str(root/"converted")],
                          capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    return root


def output(activity,fold=0): return activity/"converted"/f"fold={fold}"/"cancer_id=BRCA/part-0.parquet"


def test_real_gsva_oracle_native_symbol_average_and_training_filter(activity, tmp_path):
    oracle = tmp_path/"oracle.R"
    oracle.write_text(r'''
suppressPackageStartupMessages({library(GSVA);library(BiocParallel);library(data.table);library(SummarizedExperiment)})
a<-commandArgs(trailingOnly=TRUE);root<-a[[1]]
x<-readRDS(file.path(dirname(root),'normal/fold=0/gene.RDS'))
train<-colnames(x)[!as.integer(substr(colnames(x),9,12))%%5 %in% c(0,1)]
# Independently specified expected input: G80 fails the training CPM filter;
# mean of G1/G2 logCPM becomes S1, G3..G79 keep their own symbols.
means<-rbind(S1=colMeans(x[c('GENE:G1','GENE:G2'),,drop=FALSE]),x[paste0('GENE:G',3:79),,drop=FALSE])
rownames(means)<-c('S1',paste0('S',3:79))
sets<-list('MSIGDB:HALLMARK:P1'=c('S1',paste0('S',3:21)),'MSIGDB:HALLMARK:P2'=paste0('S',30:54))
raw<-gsva(ssgseaParam(means,sets,minSize=10,maxSize=500,alpha=.25,normalize=FALSE),verbose=FALSE,BPPARAM=SerialParam())
expected<-raw/diff(range(raw[,train,drop=FALSE]))
actual<-readRDS(file.path(root,'normal/fold=0/activity.RDS'))
stopifnot(max(abs(actual[rownames(expected),]-expected))<1e-12)
# Direct counts/effective-library CPM independently validates the inverse-prior
# formula in production; it is not calculated from its exported detection table.
counts<-as.matrix(assay(readRDS(file.path(dirname(root),'counts.RDS')),'unstranded'))
qc<-fread(file.path(dirname(root),'normal/fold=0/normalization_samples.tsv'))
at<-match(train,qc$sample_id);eff<-qc$library_size[at]*qc$normalization_factor[at]
cpm<-sweep(counts[4:83,match(train,colnames(counts)),drop=FALSE],2,eff,'/')*1e6
detected<-fread(file.path(root,'normal/fold=0/gene_detection.tsv'))
stopifnot(identical(as.numeric(rowSums(cpm>=1)),as.numeric(detected$training_cpm_ge1_n)))
''',encoding="utf-8")
    run_r(oracle,activity)


def test_heldout_perturbation_preserves_training_state_and_scores(activity):
    a=read_matrix(activity/"normal/fold=0/activity.RDS")
    b=read_matrix(activity/"changed/fold=0/activity.RDS")
    train=[s for s in a.columns if int(s[8:12])%5 not in (0,1)]
    pd.testing.assert_frame_equal(a[train],b[train],check_exact=True)
    for name in ("state.RDS","gene_detection.tsv","coverage.tsv"):
        assert record(activity/"normal/fold=0"/name)["sha256"]==record(activity/"changed/fold=0"/name)["sha256"]
    assert not np.allclose(a.loc[P1].drop(train),b.loc[P1].drop(train))
    detection=pd.read_csv(activity/"normal/fold=0/gene_detection.tsv",sep="\t")
    assert not detection.set_index("gene_id").loc["GENE:G80","analysis_eligible"]


def test_independent_query_batch_and_degenerate_range(activity,tmp_path):
    script=tmp_path/"query.R"
    script.write_text(r'''
suppressPackageStartupMessages({library(GSVA);library(BiocParallel)})
a<-commandArgs(trailingOnly=TRUE);source(a[[1]]);root<-a[[2]]
x<-readRDS(file.path(dirname(root),'normal/fold=0/gene.RDS'))
state<-readRDS(file.path(root,'normal/fold=0/state.RDS'))
one<-apply_frozen_ssgsea(x[,1,drop=FALSE],state)
group<-x[,c(7,1,12),drop=FALSE];group[,c(1,3)]<-group[,c(1,3)]*100
stopifnot(identical(one,apply_frozen_ssgsea(group,state)[,2,drop=FALSE]))
rownames(group)[1]<-'wrong';stopifnot(inherits(try(apply_frozen_ssgsea(group,state),silent=TRUE),'try-error'))
# Identical samples, one pathway: a zero training range has no finite scale.
y<-matrix(10,20,20,dimnames=list(paste0('GENE:G',1:20),paste0('S',1:20)))
ann<-data.frame(ensembl_gene_id=paste0('G',1:20),gene_symbol=paste0('G',1:20),gene_type='protein_coding')
zero<-fit_frozen_ssgsea(y,colnames(y),ann,list(P=paste0('G',1:10)),1e6)
stopifnot(zero$coverage$unavailable_reason=='DEGENERATE_TRAIN_SCORE_RANGE',all(is.na(apply_frozen_ssgsea(y,zero))))
empty<-fit_frozen_ssgsea(y,colnames(y),ann,list(P='G1'),1e6)
stopifnot(empty$coverage$unavailable_reason=='INSUFFICIENT_TRAIN_DETECTED_GENES',all(is.na(apply_frozen_ssgsea(y,empty))))
''',encoding="utf-8")
    run_r(script,HELPER,activity)


def test_full_pathway_closure_and_actual_inventory_uses_only_requested_fold(activity,tmp_path):
    for fold in range(5):
        path=output(activity,fold);frame=pd.read_parquet(path)
        assert len(frame)==4*96 and frame.patient_id.nunique()==96
        assert set(frame.split_role)=={"train","validation"}
        missing=frame.loc[frame.pathway_id.eq(SMALL)]
        assert missing.activity_score.isna().all() and not missing.analysis_eligible.any()
        assert missing.unavailable_reason.eq("INSUFFICIENT_TRAIN_DETECTED_GENES").all()
        validate(binding(path),path.parent,path,replay=True)
    inventory=Inventory([],tmp_path);path=output(activity)
    inspect(inventory,binding(path),path.parent,path,"core","source",0)
    assert any("12344321" in r["study_tokens_json"] for r in inventory.usage)
    assert any("99887711" in r["study_tokens_json"] for r in inventory.usage)
    assert any("77889900" in r["study_tokens_json"] for r in inventory.usage)
    assert not any("fold=1/" in r["path"] for r in inventory.dependencies)


@pytest.mark.parametrize("fault",["output_repin","fold","patient_map","scale","gmt"])
def test_changed_scores_scope_or_ancestors_rejected(activity,tmp_path,fault):
    path=output(activity)
    if fault=="output_repin":
        target=tmp_path/"part-0.parquet";pd.read_parquet(path).assign(activity_score=99.).to_parquet(target,index=False)
        value=read(binding(path)["path"]);value["output"]=record(target)
        receipt=tmp_path/"ACTIVITY_SOURCES.json";write(receipt,value)
        with pytest.raises(ValueError,match="differs from native conversion"):validate(record(receipt),tmp_path,target,replay=True)
    elif fault=="fold":
        with pytest.raises(ValueError,match="different model fold"):
            inspect(Inventory([],tmp_path),binding(path),path.parent,path,"core","source",1)
    elif fault=="patient_map":
        other=tmp_path/"map.tsv";other.write_bytes((activity.parent/"patient_map.tsv").read_bytes())
        with pytest.raises(ValueError,match="patient map differs"):
            activity_path(read(activity/"converted/SOURCE_CONFIG_FRAGMENT.json")["sources"],"BRCA",0,other)
    else:
        from cc_hhgt.v32.fold_local_activity import native,frame_from_native
        value=read(binding(path)["path"]);_,info,npath,norm,samples=native(value["native"],path.parent)
        if fault=="scale":
            info["training_score_range"]*=2
            with pytest.raises(ValueError,match="scaling differs"):
                frame_from_native(info,Path(value["native"]["path"]).parent,npath,norm,samples)
        else:
            manifest=read(info["inputs"]["gmt_manifest"]["path"])
            fake=tmp_path/"changed.gmt";fake.write_text("P1\tdescription\tX\n")
            manifest["collections"][0]["file"]["path"]=str(fake)
            new=tmp_path/"gmt.json";write(new,manifest);info["inputs"]["gmt_manifest"]=record(new)
            with pytest.raises(ValueError):frame_from_native(info,tmp_path,npath,norm,samples)


def test_actual_native_five_fold_statistics_preserve_missing_pathways_and_sources(activity,tmp_path,monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs,prepared_capture
    from cc_hhgt.v32.preparation_sources import attach_sources
    args,vargs,train,valid=native_inputs(tmp_path/"native",monkeypatch,
        fold_expression_root=activity.parent/"converted",patient_map_source=activity.parent/"patient_map.tsv",
        fold_activity_root=activity/"converted",pathway_ids=[P1,SMALL])
    for group in (train["pair_outputs"],valid["outputs"]):
        for item in group:
            frame=pd.read_parquet(item["path"]);assert len(frame)==2
            missing=frame.loc[frame.pathway_id.eq(SMALL)]
            assert missing.unavailable_reason.eq("INSUFFICIENT_TRAIN_DETECTED_GENES").all()
            assert missing.availability.eq("typed_unavailable").all()
            assert missing.filter(regex="^rho_A").isna().all().all()
    receipt,frames=prepared_capture(tmp_path/"prepared",args,vargs,0)
    assert all(not f.loc[f.pathway_id.eq(SMALL),"association_available"].any() for f in frames.values())
    payload={"patient_fold":0,"input_scope":{"association_mode":"local-cnv-adjusted"}};attach_sources(payload,receipt)
    inventory=Inventory([],tmp_path);inventory.preparation(payload,tmp_path,"core",0,False)
    assert any("12344321" in r["study_tokens_json"] for r in inventory.usage)
    assert str(output(activity,0)) in {r["path"] for r in inventory.dependencies}
    assert str(output(activity,1)) not in {r["path"] for r in inventory.dependencies}


def test_activity_cannot_mix_with_expression_from_other_counts(activity,tmp_path):
    from cc_hhgt.v32.fold_local_expression import convert as convert_expression
    convert_expression([activity.parent/"changed"],tmp_path/"other",runner=ROOT/"scripts/convert_fold_local_expression.py")
    sources=read(tmp_path/"other/SOURCE_CONFIG_FRAGMENT.json")["sources"]
    sources.update(read(activity/"converted/SOURCE_CONFIG_FRAGMENT.json")["sources"])
    with pytest.raises(ValueError,match="different normalization sources"):
        activity_path(sources,"BRCA",0,activity.parent/"patient_map.tsv")


def test_actual_core_forward_and_external_isolation_include_gmt_only_study(activity,tmp_path,monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs,test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs
    def factory(root,patcher):
        return native_inputs(root,patcher,fold_expression_root=activity.parent/"converted",
            patient_map_source=activity.parent/"patient_map.tsv",fold_activity_root=activity/"converted",pathway_ids=[P1])
    test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(
        tmp_path,monkeypatch,native_factory=factory,external_pmid="12344321",pathway_id=P1)


def test_no_eligible_pathways_still_preserves_native_candidate_closure(activity,tmp_path,monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs,prepared_capture
    args,vargs,train,valid=native_inputs(tmp_path/"native",monkeypatch,
        fold_expression_root=activity.parent/"converted",patient_map_source=activity.parent/"patient_map.tsv",
        fold_activity_root=activity/"converted",pathway_ids=[SMALL])
    for group in (train["pair_outputs"],valid["outputs"]):
        for item in group:
            frame=pd.read_parquet(item["path"])
            assert len(frame)==1 and frame.availability.eq("typed_unavailable").all()
            assert frame.unavailable_reason.eq("INSUFFICIENT_TRAIN_DETECTED_GENES").all()
    _,frames=prepared_capture(tmp_path/"prepared",args,vargs,0)
    assert all(len(f)==1 and not f.association_available.any() for f in frames.values())


def test_wrong_host_rejected_before_output(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,"gethostname",lambda:"not-149")
    with pytest.raises(RuntimeError,match="target host 149"):
        convert([],tmp_path/"forbidden",runner=RUNNER)
    assert not (tmp_path/"forbidden").exists()
