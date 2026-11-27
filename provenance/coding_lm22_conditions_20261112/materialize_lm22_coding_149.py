import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import concurrent.futures,json,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';OUT=BASE/'lm22_coding_r1';REPO=ROOT/'source_integration/repo'
sys.path.insert(0,str(REPO))
from cla_engine.relation_evidence_bindings import file_sha
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'MATERIALIZATION_PREFLIGHT.json',{'target_host':'149','operation':'patient_RNA_immune_composition_readout','lncrna_predictor_training':False,'lncrna_association':False})
import numpy as np,pandas as pd
from cla_engine.coding_lm22 import METHOD,score_coding_lm22
valid=json.loads((OUT/'REFERENCE_VALIDATION.json').read_text())
assert valid['R_vs_python_max_absolute_error']<1e-6
assert file_sha(OUT/'LM22_CODING_REFERENCE.tsv')==valid['reference_sha256']
reference=pd.read_csv(OUT/'LM22_CODING_REFERENCE.tsv',sep='\t',index_col=0)

def case(cohort,cancer):
    name=cohort+'_'+cancer;src=BASE/'protein_coding_inputs'/name;dest=OUT/name
    if dest.exists():raise RuntimeError('existing LM22 output '+name)
    dest.mkdir()
    save(dest/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'LM22_PC_relative_composition','cohort':cohort,'cancer':cancer,'lncrna_association':False})
    rec=json.loads((src/'COMPLETE.json').read_text())
    for k,h in rec['files'].items():
        if file_sha(src/k)!=h:raise RuntimeError('coding input drift '+name+'/'+k)
    e=pd.read_parquet(src/'SYMBOL_LOGTPM.parquet')
    split=json.loads((src/'PATIENT_PARTITIONS.json').read_text())
    assert set(e.index)==set(split['source_patients']+split['label_patients']) and split['outer_fold']==0
    ann=pd.read_csv(src/'SCORING_GENE_ANNOTATION.tsv',sep='\t')
    assert ann.gene_biotype.eq('protein_coding').all() and set(reference.index)<=set(ann.reference_symbol)
    scores,qc=score_coding_lm22(e,reference,progress=lambda n,total:print(name+' '+str(n)+'/'+str(total),flush=True))
    ids=['STATE_NATIVE_'+cohort+'|cibersort_relative::'+c for c in scores]
    metadata=pd.DataFrame({'source_score_id':ids,'readout_id':[c+'@@'+METHOD for c in ids],
                          'method':METHOD,'method_variant':METHOD,'measurement_layer':'RNA_DECONVOLUTION'})
    scores.columns=metadata.readout_id
    good=scores.notna().all(axis=1)
    assert (scores.loc[good]>=0).all().all()
    np.testing.assert_allclose(scores.loc[good].sum(axis=1),1,atol=1e-10)
    assert not (scores.notna().any(axis=1)&~good).any()
    scores.to_parquet(dest/'VALUES.parquet',compression='zstd')
    scores.notna().to_parquet(dest/'OBSERVED.parquet',compression='zstd')
    metadata.to_csv(dest/'READOUTS.tsv',sep='\t',index=False)
    qc.to_csv(dest/'PATIENT_QC.tsv',sep='\t',index=False)
    native=pd.read_parquet(BASE/'measurements_coding_r2'/cancer/'validation'/cohort/'STATE/VALUES.parquet',columns=ids).reindex(e.index)
    comparison=[]
    for original,new in zip(ids,scores):
        ok=native[original].notna()&scores[new].notna()
        delta=scores.loc[ok,new]-native.loc[ok,original]
        comparison.append({'source_score_id':original,'paired_patients':int(ok.sum()),
            'mean_absolute_difference':float(delta.abs().mean()),'median_absolute_difference':float(delta.abs().median())})
    pd.DataFrame(comparison).to_csv(dest/'PUBLISHED_READOUT_COMPARISON.tsv',sep='\t',index=False)
    report={'target_host':'149','cohort':cohort,'cancer':cancer,'method_variant':METHOD,
            'patients':len(e),'patients_computed':int(good.sum()),'cell_readouts':22,'coding_reference_genes':len(reference),
            'reference_receipt_sha256':file_sha(OUT/'REFERENCE_VALIDATION.json'),'coding_input_receipt_sha256':file_sha(src/'COMPLETE.json'),
            'code_sha256':file_sha(REPO/'cla_engine/coding_lm22.py'),'model_reference':'fixed_external_LM22',
            'cross_patient_reference_fitted':False,'lncrna_predictor_training':False,'lncrna_association_computation':False,
            'published_numerical_equivalence_claimed':False,
            'files':{p.name:file_sha(p) for p in sorted(dest.iterdir()) if p.is_file()}}
    save(dest/'COMPLETE.json',report);print('LM22_COMPLETE '+name,flush=True)
    return report

done=[];errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    jobs={pool.submit(case,cohort,cancer):cohort+'_'+cancer for cancer in ('BRCA','LUAD','COAD') for cohort in ('TCGA','CPTAC')}
    for job in concurrent.futures.as_completed(jobs):
        try:done.append(job.result())
        except Exception as e:errors[jobs[job]]=str(e);print('LM22_ERROR '+jobs[job]+' '+str(e),flush=True)
        save(OUT/'MATERIALIZATION_PROGRESS.json',{'target_host':'149','completed':done,'errors':errors})
if errors:raise SystemExit(1)
print(json.dumps({'completed_cases':len(done),'patients':sum(r['patients_computed'] for r in done)}),flush=True)
