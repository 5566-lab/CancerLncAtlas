"""Build three-cancer Lasso demo inputs with the full shared lncRNA axis."""
from __future__ import annotations
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from run_m0_linear import load_case,source_read,write_json
from cla_engine.common import sha

def prepare(cfg_path,export_root,output,cancers):
    if socket.gethostname()!="149":raise RuntimeError("Run on 149")
    if not 1<=len(cancers)<=3 or len(set(cancers))!=len(cancers):raise ValueError("Demo requires 1-3 distinct cancers")
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new output directory")
    out.mkdir(parents=True)
    write_json(out/"LAUNCH_PREFLIGHT.json",dict(target_host="149",operation="M0_LASSO_INPUT_PREPARATION",cancers=cancers,cohort_pooling=False))
    cfg=json.loads(Path(cfg_path).read_text());er=Path(export_root)
    erc=json.loads((er/"COMPLETE.json").read_text())
    if erc["status"]!="COMPLETE" or erc["target_host"]!="149":raise ValueError("Unfinished export")
    result=[]
    for cancer in cancers:
        cc={"KIRC":"CCRCC","LUSC":"LSCC","HNSC":"HNSCC","PAAD":"PDAC"}.get(cancer,cancer)
        data={}
        for dataset,cohort in [("TCGA",cancer),("CPTAC",cc)]:
            sources=[]
            x,y,folds,meta,features,info=load_case(cfg,dict(dataset=dataset,cohort=cohort,cancer_id=cancer),sources)
            data[dataset]=[x,y,folds,meta,features,info,sources]
        tpm_path=er/cancer/"GENE_TPM.tsv.gz";annotation_path=er/cancer/"GENE_ANNOTATION.tsv"
        rec=erc["cancers"][cancer]
        tpm=source_read(tpm_path,lambda p:pd.read_csv(p,sep="\t").set_index("gene_id"),data["TCGA"][6],rec["outputs"]["tpm_sha256"])
        ann=source_read(annotation_path,lambda p:pd.read_csv(p,sep="\t"),data["TCGA"][6],rec["outputs"]["annotation_sha256"])
        lnc=set("LNC:"+g for g in ann.loc[ann.gene_type.eq("lncRNA"),"gene_id"])
        common=sorted(lnc & set(data["CPTAC"][0].columns))
        if len(common)<16000:raise ValueError("Unexpectedly truncated shared lncRNA axis")
        native_tcga=set(data["TCGA"][0].columns)
        full=np.log2(tpm.loc[[g.removeprefix("LNC:") for g in common]]+1).T
        full.columns=common
        data["TCGA"][0]=full.reindex(data["TCGA"][0].index)
        if not np.isfinite(data["TCGA"][0].to_numpy()).all():raise ValueError("TCGA full TPM patient alignment failed")
        for dataset in ["TCGA","CPTAC"]:
            x,y,folds,meta,features,info,sources=data[dataset]
            x=x.reindex(columns=common)
            d=out/(dataset+"_"+info["cohort"]);d.mkdir()
            info.update(expression_unit="log2(TPM+1)",n_features=len(common),
                        full_common_gene_axis=True,legacy_detection_filter_applied=False,
                        expression_harmonization="Same_units_and_GENCODE_ids;upstream_quantification_pipelines_differ",
                        batch_corrected=False,cohort_pooling=False,
                        labels="UNCHANGED_FROZEN_NATIVE_SCORES;NOT_CROSS_COHORT_CALIBRATED")
            x.to_parquet(d/"X.parquet",compression="zstd")
            y.to_parquet(d/"Y.parquet",compression="zstd")
            pd.DataFrame(dict(patient_id=x.index,outer_fold=folds)).to_csv(d/"PATIENT_FOLDS.tsv",sep="\t",index=False)
            ff=data["CPTAC"][4].set_index("lncrna_id").reindex(common).reset_index()
            ff.to_csv(d/"FEATURES.tsv",sep="\t",index=False)
            meta.to_csv(d/"TARGETS.tsv",sep="\t",index=False)
            write_json(d/"INPUTS.json",dict(info=info,sources=sources,raw_tcga=rec,
                        previous_tcga_feature_count=len(native_tcga),restored_tcga_features=len(set(common)-native_tcga),
                        original_Y_unchanged=True))
            files={p.name:sha(p) for p in d.iterdir() if p.is_file()}
            write_json(d/"COMPLETE.json",dict(status="COMPLETE",target_host="149",files=files,info=info))
            result.append(dict(case=d.name,n_patients=len(x),n_features=len(common),n_targets=y.shape[1]))
            print("PREPARED",d.name,len(x),len(common),y.shape[1],flush=True)
    pd.DataFrame(result).to_csv(out/"SUMMARY.tsv",sep="\t",index=False)
    write_json(out/"COMPLETE.json",dict(status="COMPLETE",target_host="149",cases=result,config_sha256=sha(cfg_path),
                export_complete_sha256=sha(er/"COMPLETE.json"),cohort_pooling=False))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--tcga-export",required=True);p.add_argument("--output",required=True)
    p.add_argument("--cancer-list",nargs="+",default=["BRCA","LUAD","COAD"])
    a=p.parse_args();prepare(a.config,a.tcga_export,a.output,a.cancer_list)
if __name__=="__main__":main()
