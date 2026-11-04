"""Validated CPTAC reader. Callers select one cohort and assay explicitly."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_json,sha,fail
from .state_protein_map import pinned
from .cptac_scope import select_cohorts,CANCER_MAP
from .cptac_bundle import FORMAT,verify_tree

ROLE='ALIGNED_EXTERNAL_COHORT_NATIVE_SCORES'
def validate_binding(binding,targets=None,tcga_state_sha256=None):
    if binding is None:return None
    if binding.get('format')!=FORMAT or binding.get('role')!=ROLE:
        fail('CPTAC_BINDING_FORMAT','Expected the aligned CPTAC native-score binding')
    if binding.get('training_pooling') is not False:
        fail('CPTAC_UNVALIDATED_POOLING','Cross-cohort and cross-assay calibration has not been established')
    complete=pinned(binding['complete']);root=complete.parent;rec=verify_tree(complete)
    if rec.get('format')!=FORMAT or rec.get('target_host')!='149' or rec.get('training_pooling') is not False:
        fail('CPTAC_PRODUCT_FORMAT','Invalid CPTAC product contract')
    if tcga_state_sha256 and rec['config']['tcga_state']['sha256']!=tcga_state_sha256:
        fail('CPTAC_TCGA_REFERENCE_DRIFT','CPTAC alignment used a different TCGA State snapshot')
    for source in read_json(root/'BOUND_SOURCES.json'):pinned(source)
    # Assay values remain in a separately committed measured-input package.
    aligned=pinned(rec['config']['aligned_complete']);verify_tree(aligned)
    patients=pd.read_csv(root/'PATIENTS.tsv',sep='\t',dtype=str)
    if len(patients)!=rec['patients'] or not patients.patient_id.is_unique or not patients.native_case_id.is_unique:
        fail('CPTAC_PATIENT_AXIS','Invalid patient authority')
    expected=patients.apply(lambda r:'CPTAC:'+r.source_cohort+':'+r.native_case_id,axis=1)
    if not expected.equals(patients.patient_id) or not patients.cancer_id.equals(patients.source_cohort.map(CANCER_MAP)):
        fail('CPTAC_PATIENT_IDENTITY','Cohort, native-case or TCGA cancer aliases differ')
    state=pd.read_parquet(root/'STATE_PATIENT.parquet')
    identity=['cohort_id','source_cohort','cancer_id','native_case_id','patient_id']
    scorecols=[c for c in state if c not in identity]
    if targets is not None and scorecols!=[t['column'] for t in targets]:
        fail('CPTAC_STATE_AXIS','State axis differs from the current catalog')
    if state.patient_id.tolist()!=patients.patient_id.tolist() or np.isinf(state[scorecols].to_numpy(float)).any():
        fail('CPTAC_STATE_VALUES','Invalid State values or patient order')
    if len(scorecols)!=rec['state_axis_count'] or int(state[scorecols].notna().any().sum())!=rec['observed_state_count']:
        fail('CPTAC_STATE_COVERAGE','State coverage differs from receipt')
    index=pd.read_csv(root/'SCORE_INDEX.tsv',sep='\t')
    for row in index.to_dict('records'):
        p=(root/row['path']).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p)!=row['sha256']:fail('CPTAC_SCORE_HASH',row['path'])
        f=pd.read_parquet(p).set_index('target_id')
        allowed=patients.loc[patients.source_cohort.eq(row['source_cohort']),'patient_id'].tolist()
        if not f.index.is_unique or not f.columns.is_unique or list(f.columns)!=allowed or np.isinf(f.to_numpy()).any():
            fail('CPTAC_SCORE_AXIS',row['path'])
        if int(f.notna().sum().sum())!=row['finite_scores']:fail('CPTAC_SCORE_COVERAGE',row['path'])
    return {'status':'VERIFIED_CPTAC_ASSAYS_AND_NATIVE_SCORES_NOT_TRAINED','complete':binding['complete'],
        'patients':rec['patients'],'cohorts':rec['cohorts'],'state_axis_count':rec['state_axis_count'],
        'observed_state_count':rec['observed_state_count'],'MSigDB_pathway_axis':rec['MSigDB_pathway_axis'],
        'PTM_signature_axis':rec['PTM_signature_axis'],'used_by_existing_HHGT_heads':False,'training_pooling':False}

def _product(binding):
    if binding.get('format')!=FORMAT or binding.get('role')!=ROLE:
        fail('CPTAC_BINDING_FORMAT','Expected the aligned CPTAC native-score binding')
    if binding.get('training_pooling') is not False:fail('CPTAC_UNVALIDATED_POOLING','Explicit cohort and assay required')
    complete=pinned(binding['complete']);rec=read_json(complete)
    if rec.get('format')!=FORMAT or rec.get('status')!='COMPLETE':fail('CPTAC_PRODUCT_FORMAT','Invalid product')
    return complete.parent,rec

def load_scores(binding,*,cohort,layer):
    root,rec=_product(binding)
    selected=select_cohorts('list',[cohort],rec['cohorts'])
    if len(selected)!=1:fail('CPTAC_SCORE_SCOPE','Select exactly one CPTAC cohort')
    cohort=next(iter(selected))
    if layer=='STATE':
        p=root/'STATE_PATIENT.parquet';pinned({'path':str(p),'sha256':rec['files'][p.name]['sha256']})
        f=pd.read_parquet(p);f=f.loc[f.source_cohort.eq(cohort)].set_index('patient_id')
        f=f.drop(columns=['cohort_id','source_cohort','cancer_id','native_case_id']).T
    else:
        idxpath=root/'SCORE_INDEX.tsv';pinned({'path':str(idxpath),'sha256':rec['files'][idxpath.name]['sha256']})
        index=pd.read_csv(idxpath,sep='\t');row=index.loc[index.source_cohort.eq(cohort)&index.layer.eq(layer)]
        if len(row)!=1:fail('CPTAC_SCORE_UNAVAILABLE','No score matrix for this explicit assay/cohort',cohort=cohort,layer=layer)
        row=row.iloc[0];p=pinned({'path':str(root/row['path']),'sha256':row.sha256});f=pd.read_parquet(p).set_index('target_id')
    return {'values':f,'available':f.notna(),'cohort':cohort,'layer':layer,
            'calibration':'CPTAC_NATIVE_NOT_CALIBRATED_TO_TCGA_OR_OTHER_ASSAYS','pooled_training_allowed':False}

def load_assay(binding,*,cohort,layer,tissue='Tumor',mapped_genes_only=False):
    root,rec=_product(binding);selected=select_cohorts('list',[cohort],rec['cohorts'])
    if len(selected)!=1 or layer not in ['rna','protein','phosphorylation','methylation_gene'] or tissue not in ['Tumor','Normal']:
        fail('CPTAC_ASSAY_SCOPE','Select one cohort, measured layer and tissue')
    cohort=next(iter(selected));ap=pinned(rec['config']['aligned_complete']);ar=read_json(ap)
    rel='matrices/'+cohort+'/'+layer+('_TumorNormal.parquet' if layer=='rna' else '_'+tissue+'.parquet')
    if rel not in ar['files']:fail('CPTAC_ASSAY_UNAVAILABLE',rel)
    p=pinned({'path':str(ap.parent/rel),'sha256':ar['files'][rel]['sha256']})
    f=pd.read_parquet(p).set_index('feature_id');suffix='::T' if tissue=='Tumor' else '::N';f=f[[c for c in f if c.endswith(suffix)]]
    fp=ap.parent/'FEATURE_ALIGNMENT.parquet';pinned({'path':str(fp),'sha256':ar['files'][fp.name]['sha256']})
    features=pd.read_parquet(fp);features=features.loc[features.layer.eq(layer)].drop_duplicates('feature_id').set_index('feature_id').reindex(f.index)
    if mapped_genes_only:
        if layer=='phosphorylation':fail('CPTAC_NO_GENE_LEVEL_PTM_COLLAPSE','Phosphorylation must retain site identity')
        take=features.reference_presence.eq('GENCODE_V36');f=f.loc[take];features=features.loc[take]
        f.index=features.model_gene_id;f.index.name='gene_id'
    return {'values':f,'available':f.notna(),'features':features,'cohort':cohort,'layer':layer,'tissue':tissue,
       'source_unit':{'rna':'log2(TPM+1)','protein':'BCM_LOG2_REFERENCE_INTENSITY_NORMALIZED','phosphorylation':'BCM_SITE_LOG2_REFERENCE_INTENSITY_NORMALIZED','methylation_gene':'MEAN_PROMOTER_RELATED_CPG_ISLAND_BETA'}[layer],
       'methylation_scope':read_json(root/'METHOD_POLICY.json')['methylation'] if layer=='methylation_gene' else None}
