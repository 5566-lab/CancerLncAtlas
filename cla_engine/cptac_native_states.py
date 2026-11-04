"""CPTAC-native RNA stemness and explicitly derived relative immune composition."""
import argparse,socket
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .common import atomic_json,read_json,sha,fail
from .state_protein_map import pinned
from .cptac_materialize import gene_id

CELL_MAP={
'B_cell_memory':'B.cells.memory','B_cell_naive':'B.cells.naive','B_cell_plasma':'Plasma.cells',
'Myeloid_dendritic_cell_activated':'Dendritic.cells.activated','Myeloid_dendritic_cell_resting':'Dendritic.cells.resting',
'Eosinophil':'Eosinophils','Macrophage_M0':'Macrophages.M0','Macrophage_M1':'Macrophages.M1','Macrophage_M2':'Macrophages.M2',
'Mast_cell_activated':'Mast.cells.activated','Mast_cell_resting':'Mast.cells.resting','Monocyte':'Monocytes',
'NK_cell_activated':'NK.cells.activated','NK_cell_resting':'NK.cells.resting','Neutrophil':'Neutrophils',
'T_cell_CD4+_memory_activated':'T.cells.CD4.memory.activated','T_cell_CD4+_memory_resting':'T.cells.CD4.memory.resting',
'T_cell_CD4+_naive':'T.cells.CD4.naive','T_cell_CD8+':'T.cells.CD8','T_cell_follicular_helper':'T.cells.follicular.helper',
'T_cell_gamma_delta':'T.cells.gamma.delta','T_cell_regulatory_(Tregs)':'T.cells.regulatory..Tregs.'}

def relative_composition(published):
    columns=['CIBERSORT_'+k for k in CELL_MAP]
    if not set(columns)<=set(published):fail('CPTAC_CIBERSORT_COMPONENTS','Require all 22 explicit source components')
    f=published[columns].apply(pd.to_numeric,errors='raise')
    valid=np.isfinite(f).all(axis=1)&f.ge(0).all(axis=1)&f.sum(axis=1).gt(0)
    answer=f.div(f.sum(axis=1),axis=0);answer.loc[~valid]=np.nan
    answer.columns=['cibersort_relative::'+v for v in CELL_MAP.values()]
    return answer

def spearman_weight_score(values,weights):
    ok=np.isfinite(weights)&np.isfinite(values).all(axis=1)
    if ok.sum()<3:return np.full(values.shape[1],np.nan),int(ok.sum())
    x=rankdata(values[ok],axis=0,method='average')
    y=rankdata(weights[ok],method='average')
    x=x-x.mean(axis=0);y=y-y.mean()
    den=np.sqrt((x*x).sum(axis=0)*(y*y).sum())
    with np.errstate(divide='ignore',invalid='ignore'):score=(y@x)/den
    score[~np.isfinite(score)]=np.nan
    return score,int(ok.sum())

def run(cfg,output):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','Run on 149')
    ap=pinned(cfg['aligned_complete']);ac=read_json(ap);root=ap.parent
    weights_path=pinned(cfg['malta'])
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('CPTAC_OUTPUT_EXISTS','Use a new versioned directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'CPTAC_NATIVE_STEMNESS_AND_RELATIVE_COMPOSITION'})
    signatures={}
    for name,sheet,header in [('RNAss','mRNAsi',0),('EREG.EXPss','EREG-mRNAsi',1)]:
        f=pd.read_excel(weights_path,sheet_name=sheet,header=header).dropna(subset=['ENSEMBL'])
        f['gene_id']=f.ENSEMBL.map(gene_id)
        if f.gene_id.duplicated().any():fail('CPTAC_OCLR_DUPLICATE_GENE',name)
        signatures[name]=f.set_index('gene_id').Weight.astype(float)
    parts=[];coverage=[]
    for cohort in ac['cohorts']:
        p=root/'matrices'/cohort/'rna_TumorNormal.parquet'
        if sha(p)!=ac['files'][str(p.relative_to(root))]['sha256']:fail('CPTAC_RNA_HASH',cohort)
        f=pd.read_parquet(p).set_index('feature_id');f=f[[c for c in f if c.endswith('::T')]]
        result=pd.DataFrame({'sample_id':f.columns,'source_cohort':cohort})
        for name,w in signatures.items():
            common=f.index.intersection(w.index);scores,n=spearman_weight_score(f.loc[common].to_numpy(),w.loc[common].to_numpy())
            result[name+'::raw_spearman']=scores
            coverage.append({'source_cohort':cohort,'state_id':name,'signature_genes':len(w),'observed_genes':n,
                             'gene_coverage_percent':100*n/len(w)})
        parts.append(result)
    combined=pd.concat(parts,ignore_index=True)
    scales={}
    for name in signatures:
        col=name+'::raw_spearman';low=float(combined[col].min());high=float(combined[col].max())
        if not np.isfinite(low) or high<=low:fail('CPTAC_OCLR_CONSTANT',name)
        combined[name+'::CPTAC_native_0_1']=(combined[col]-low)/(high-low)
        scales[name]={'minimum':low,'maximum':high,'reference':'All_selected_CPTAC_tumors_with_RNA',
                      'formula':'(Spearman_rho-minimum)/(maximum-minimum)','TCGA_calibration':'NOT_ESTABLISHED'}
    combined.to_parquet(out/'RNA_STEMNESS_NATIVE.parquet',index=False)
    pd.DataFrame(coverage).to_csv(out/'RNA_STEMNESS_GENE_COVERAGE.tsv',sep='\t',index=False)
    pp=root/'PUBLISHED_PHENOTYPES.parquet'
    if sha(pp)!=ac['files'][pp.name]['sha256']:fail('CPTAC_PHENOTYPE_HASH','Source changed')
    published=pd.read_parquet(pp).set_index('patient_id')
    relative=relative_composition(published)
    relative.reset_index().to_parquet(out/'CIBERSORT_RELATIVE_DERIVED.parquet',index=False)
    result={'format':'CLA_CPTAC_NATIVE_STATE_V1','status':'COMPLETE','target_host':'149','config':cfg,
       'OCLR_method':'Frozen_Malta_original_weights_Spearman;CPTAC_reference_minmax_not_TCGA_rescale','OCLR_reference':scales,
       'OCLR_source':'https://doi.org/10.1016/j.cell.2018.03.034',
       'CIBERSORT_method':'Published_22_absolute_components_divided_by_their_sample_total;not_a_new_deconvolution_run',
       'CIBERSORT_no_missing_or_negative_components_permitted':True,
       'CIBERSORT_interpretation':'Relative_composition_within_22_LM22_cell_types;not_percent_of_all_tumor_cells',
       'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()}}
    atomic_json(out/'COMPLETE.json',result)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
if __name__=='__main__':main()
