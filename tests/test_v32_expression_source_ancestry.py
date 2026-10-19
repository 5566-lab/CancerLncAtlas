"""Actual edgeR normalization -> RDS conversion -> native core source consumers."""
from pathlib import Path
import socket
import subprocess
import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.formal_expression_sources import convert, binding, validate, inspect, read, write, record, read_matrix, matrix_to_frame
from cc_hhgt.v32.current_model_use import Inventory

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"scripts/convert_formal_lncrna_rds.py"
R_RUNNER=ROOT/"scripts/materialize_formal_lncrna_logcpm.R"


def forbidden(*a,**kw): raise AssertionError("No optimizer or LASSO fitting in expression source verification")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def normalized(tmp_path_factory):
    assert socket.gethostname()=="149"
    root=tmp_path_factory.mktemp("native_tmm")
    write(root/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"synthetic_only":True})
    # An independent direct edgeR calculation uses the explicitly expected
    # sample/row indices; production code determines them from native metadata.
    script=root/"fixture.R"
    script.write_text(r'''
library(SummarizedExperiment);library(edgeR);library(data.table)
root<-commandArgs(trailingOnly=TRUE)[[1]]
dir.create(file.path(root,'counts'));dir.create(file.path(root,'threshold'))
samples<-c(sprintf('TCGA-ZZ-%04d-01A',0:119),'TCGA-ZZ-0000-11A','TCGA-ZZ-0000-02A','TCGA-ZZ-0000-10A')
counts<-outer(1:80,1:123,function(g,s) ((g*17+s*13)%%100)+50)
counts[1,]<-round(80+20*(sin((0:122)*.63)+cos((0:122)*.27)))
counts[,122:123]<-counts[,122:123]*5
colnames(counts)<-samples
genes<-c('Z.1','Z.2','ENSG_PAR_Y',paste0('G',4:80))
x<-SummarizedExperiment(assays=list(unstranded=counts),
 rowData=DataFrame(gene_id=genes,gene_type=c(rep('lncRNA',3),rep('protein_coding',77))),
 colData=DataFrame(pmid=rep('77889900',123),row.names=samples))
saveRDS(x,file.path(root,'counts/TCGA-BRCA.RDS'))
fwrite(data.table(cancer_id='BRCA',lncrna_id='LNC:Z',detection_rate_logcpm_gt0=.8),file.path(root,'threshold/full_gdc_lncrna_detection_rates.tsv.gz'),sep='\t')
y<-calcNormFactors(DGEList(counts=counts[,c(121,1:120)]),method='TMM')
expected<-cpm(y,log=TRUE,prior.count=.25)[1,2:121,drop=FALSE]
rownames(expected)<-'LNC:Z'
saveRDS(expected,file.path(root,'expected.RDS'))
''',encoding="utf-8")
    subprocess.run(["/usr/bin/Rscript",str(script),str(root)],check=True,capture_output=True,text=True,timeout=60)
    result=subprocess.run(["/usr/bin/Rscript",str(R_RUNNER),str(root/"counts"),str(root/"threshold"),str(root/"normalized")],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    return root


@pytest.fixture
def converted(normalized,tmp_path):
    convert(normalized/"normalized",tmp_path/"expression",cancers=["BRCA"],runner=RUNNER)
    return tmp_path/"expression/cancer_id=BRCA/part-0.parquet"


def test_actual_edgeR_normalization_and_conversion_preserve_expected_values(normalized,converted,tmp_path):
    expected=matrix_to_frame(read_matrix(normalized/"expected.RDS"),"BRCA")
    pd.testing.assert_frame_equal(pd.read_parquet(converted),expected,check_exact=True)
    native=read(normalized/"normalized/TCGA-BRCA_SOURCE_RECEIPT.json")
    samples=pd.read_csv(native["outputs"]["normalization_samples"]["path"],sep="\t")
    assert len(samples)==123 and samples.include_expression.sum()==121 and samples.include_association.sum()==120
    assert samples.loc[samples.sample_id.eq("TCGA-ZZ-0000-11A"),"include_expression"].all()
    assert not samples.loc[samples.sample_id.eq("TCGA-ZZ-0000-02A"),"include_expression"].any()
    assert native["normalization"]["fold_local"] is False
    inventory=Inventory([],tmp_path)
    inspect(inventory,binding(converted),tmp_path,converted,"core","source",0)
    assert any("77889900" in r["study_tokens_json"] for r in inventory.usage)
    assert any(r["reason"]=="COHORT_LEVEL_TMM_NORMALIZATION_NOT_FOLD_LOCAL" for r in inventory.gaps)
    assert any(r["path"]==str(normalized/"counts/TCGA-BRCA.RDS") for r in inventory.dependencies)
    assert not any("TCGA-ZZ" in str(r) for r in inventory.usage)


@pytest.mark.parametrize("fault",["output","repinned_output","normalization","raw_counts"])
def test_changed_expression_or_ancestor_is_rejected(converted,tmp_path,fault):
    item=binding(converted);path=Path(item["path"]);document=read(path)
    if fault in {"output","repinned_output"}:
        frame=pd.read_parquet(converted);frame.loc[0,"logcpm"]+=1;frame.to_parquet(converted,index=False)
        if fault=="repinned_output":document["output"]=record(converted)
    elif fault=="normalization":document["normalization"]["sha256"]="0"*64
    else:
        native=read(document["normalization"]["path"])
        changed=tmp_path/"changed_raw.RDS";changed.write_bytes(b"changed input")
        native["inputs"]["counts"]["path"]=str(changed)
        other=tmp_path/"normalization.json";write(other,native);document["normalization"]=record(other)
    write(path,document)
    with pytest.raises(ValueError,match="SHA256 mismatch|differs from native"):
        validate(record(path),tmp_path,converted,replay=True)


def test_actual_reads_bind_expression_receipt_and_detect_later_change(converted,tmp_path):
    from cc_hhgt.v32.cnv_input_sources import SourceCapture
    from cc_hhgt.v32.preparation_sources import SourceCapture as Preparation
    capture=SourceCapture();capture.read(converted,pd.read_parquet,role="lncrna_expression")
    assert next(iter(capture.records.values()))["upstream_expression"]==binding(converted)
    prepared=Preparation();prepared.read(converted,pd.read_parquet,role="expression",cancer_id="BRCA")
    item=prepared.reads[0]["members"][0]["upstream_expression"]
    write(item["path"],{**read(item["path"]),"changed":True})
    with pytest.raises(ValueError,match="SHA256 mismatch"):
        prepared.receipt(fold=0,mode="legacy",cancers=["BRCA"])


def test_actual_legacy_formal_reader_follows_conversion_sources(converted,tmp_path):
    from cc_hhgt.v32.preparation_sources import SourceCapture,attach_sources,COMMON_ROLES
    from scripts.prepare_v32_formal import load_expression_cancer
    capture=SourceCapture()
    for role in COMMON_ROLES:
        table=role in {"detection","pathway_hierarchy"}
        path=tmp_path/(role+(".parquet" if table else ".json"))
        if table:pd.DataFrame({"value":[1]}).to_parquet(path,index=False)
        else:write(path,{"synthetic":True})
        capture.read(path,pd.read_parquet if table else read,role=role,kind="table" if table else "control")
    capture.read(converted,lambda _:load_expression_cancer("BRCA",converted.parent.parent),role="expression",cancer_id="BRCA")
    activity=tmp_path/"activity.parquet";pd.DataFrame({"value":[1]}).to_parquet(activity,index=False)
    capture.read(activity,pd.read_parquet,role="activity")
    payload={"patient_fold":0,"input_scope":{"association_mode":"legacy"}}
    attach_sources(payload,capture.receipt(fold=0,mode="legacy",cancers=["BRCA"]))
    inventory=Inventory([],tmp_path);inventory.preparation(payload,tmp_path,"core",0,False)
    assert any("77889900" in r["study_tokens_json"] for r in inventory.usage)


@pytest.mark.parametrize("fault",["duplicate_lnc","duplicate_sample","nonfinite","invalid_sample"])
def test_invalid_native_matrix_does_not_silently_change_identity_or_missingness(fault):
    matrix=pd.DataFrame([[1.,2.],[3.,4.]],index=["LNC:X","LNC:Y"],columns=["TCGA-ZZ-0000-01A","TCGA-ZZ-0001-01A"])
    if fault=="duplicate_lnc":matrix.index=["LNC:X","LNC:X"]
    elif fault=="duplicate_sample":matrix.columns=["TCGA-ZZ-0000-01A"]*2
    elif fault=="nonfinite":matrix.iloc[0,0]=np.nan
    else:matrix.columns=["not-a-TCGA-sample","TCGA-ZZ-0001-01A"]
    with pytest.raises(ValueError,match="Invalid|nonfinite"):matrix_to_frame(matrix,"BRCA")


def test_normalizer_host_guard_and_converter_full_scope(normalized,tmp_path):
    result=subprocess.run(["/usr/bin/Rscript","-e",
        'Sys.info<-function() c(nodename="elsewhere");source(commandArgs(trailingOnly=TRUE)[[1]])',str(R_RUNNER)],
        capture_output=True,text=True,timeout=20)
    assert result.returncode!=0 and "target host 149" in result.stderr
    with pytest.raises(ValueError,match="Expected 33"):
        convert(normalized/"normalized",tmp_path/"incomplete",runner=RUNNER)
    assert not (tmp_path/"incomplete").exists()


def test_legacy_rds_keeps_missing_normalization_explicit(normalized,tmp_path):
    import shutil
    legacy=tmp_path/"legacy";legacy.mkdir()
    shutil.copyfile(normalized/"normalized/TCGA-BRCA_formal_logcpm.RDS",legacy/"TCGA-BRCA_formal_logcpm.RDS")
    convert(legacy,tmp_path/"new",cancers=["BRCA"],runner=RUNNER)
    output=tmp_path/"new/cancer_id=BRCA/part-0.parquet";inventory=Inventory([],tmp_path)
    inspect(inventory,binding(output),tmp_path,output,"core","source",0)
    assert any(r["reason"]=="EXPRESSION_NORMALIZATION_SOURCES_ABSENT" for r in inventory.gaps)
    assert not inventory.usage


def test_wrong_host_and_existing_output_rejected(normalized,converted,tmp_path,monkeypatch):
    with pytest.raises(FileExistsError):convert(normalized/"normalized",converted.parent.parent,cancers=["BRCA"],runner=RUNNER)
    monkeypatch.setattr(socket,"gethostname",lambda:"elsewhere")
    with pytest.raises(RuntimeError,match="target host 149"):convert(normalized/"normalized",tmp_path/"wrong",cancers=["BRCA"],runner=RUNNER)
    assert not (tmp_path/"wrong").exists()


def test_actual_raw_counts_reach_local_cnv_core_and_external_isolation(converted,tmp_path,monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs,test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs
    def native(root,patcher):return native_inputs(root,patcher,expression_source=converted)
    test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(
        tmp_path/"downstream",monkeypatch,native_factory=native,external_pmid="77889900")
