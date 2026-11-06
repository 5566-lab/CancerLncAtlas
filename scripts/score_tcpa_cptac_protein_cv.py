
"""Five frozen-fold within-cancer common-protein calibration diagnostics."""
import argparse,json,socket,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.score_tcpa_cptac_paired_protein_pilot import run,COHORT_CANCERS
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cv(rppa,bundle,mapping,retrospective,specimens,output,dataset,cancer_id):
    assert socket.gethostname()=="149"
    if dataset not in COHORT_CANCERS or cancer_id not in COHORT_CANCERS[dataset]:raise ValueError("INCOMPATIBLE_DATASET_CANCER")
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    code={p:digest(ROOT/p) for p in ["scripts/score_tcpa_cptac_protein_cv.py","scripts/score_tcpa_cptac_paired_protein_pilot.py"]}
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",action="WITHIN_CANCER_FIVE_FROZEN_FOLD_PROTEIN_CALIBRATION",dataset=dataset,cancer_id=cancer_id,folds=list(range(5)),code_sha256=code,tuning=False,additional_repeats=False,pooling_cancers=False,aggregation="Per-pathway equal mean of five fold R2; no pooling differently referenced fold Y"),indent=2))
    tables=[];test_ids=[];reference=None;sizes=[]
    for fold in range(5):
        dest=out/("fold"+str(fold))
        run(rppa,bundle,mapping,retrospective,dest,specimens,fold,dataset,cancer_id)
        split=pd.read_csv(dest/"PATIENT_SPLIT.tsv",sep="\t")
        ids=set(split.patient_id)
        if reference is None:reference=ids
        assert ids==reference and split.patient_id.is_unique and split.heldout.eq(split.patient_fold_id.eq(fold)).all()
        test_ids+=split.loc[split.heldout,"patient_id"].tolist();sizes.append(int(split.heldout.sum()))
        tables.append(pd.read_csv(dest/"HELDOUT_PATHWAY_METRICS.tsv",sep="\t").assign(fold=fold))
        print("FOLD_COMPLETE",cancer_id,dataset,fold,sizes[-1],flush=True)
    assert len(test_ids)==len(set(test_ids))==len(reference)
    allrows=pd.concat(tables,ignore_index=True);allrows.to_csv(out/"ALL_FOLD_PATHWAY_METRICS.tsv",sep="\t",index=False,na_rep="NA")
    summary=allrows.groupby("pathway_id").agg(completed_folds=("heldout_r2","count"),mean_fold_r2=("heldout_r2","mean"),median_fold_r2=("heldout_r2","median"),positive_folds=("heldout_r2",lambda x:int(x.gt(0).sum())),min_panel_genes=("panel_genes","min"),min_heldout_pairs=("heldout_pairs","min")).reset_index()
    summary.to_csv(out/"PATHWAY_CV_SUMMARY.tsv",sep="\t",index=False,na_rep="NA")
    stats=[]
    for n in [1,3,5,10]:
        selected=summary.loc[summary.completed_folds.eq(5)&summary.min_panel_genes.ge(n)]
        stats.append(dict(minimum_panel_genes_descriptive=n,all_five_fold_pathways=len(selected),positive_mean_fold_r2=int(selected.mean_fold_r2.gt(0).sum()),positive_in_all_five=int(selected.positive_folds.eq(5).sum()),median_mean_fold_r2=float(selected.mean_fold_r2.median()) if len(selected) else None))
    assert all(digest(ROOT/p)==h for p,h in code.items())
    rec=dict(target_host="149",status="COMPLETE_WITHIN_CANCER_FIVE_FOLD_DIAGNOSTIC",dataset=dataset,cancer_id=cancer_id,patients=len(reference),heldout_counts=sizes,all_patients_heldout_exactly_once=True,pathway_axis=len(summary),summary=stats,production_unified_Y=False,prospective_transfer_validated=False,code_sha256=code,files={str(p.relative_to(out)):{"sha256":digest(p)} for p in out.rglob("*") if p.is_file()})
    (out/"COMPLETE.json").write_text(json.dumps(rec,indent=2));print("CV_COMPLETE",json.dumps({k:v for k,v in rec.items() if k!="files"}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["rppa","bundle","mapping","retrospective","specimens","output"]:p.add_argument("--"+name,required=True)
    p.add_argument("--dataset",required=True,choices=list(COHORT_CANCERS));p.add_argument("--cancer-id",required=True)
    cv(**vars(p.parse_args()))
