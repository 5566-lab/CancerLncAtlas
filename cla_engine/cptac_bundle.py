"""Assemble CPTAC measured assays and native scores on the current identifier axes."""
from __future__ import annotations
import argparse,csv,hashlib,json,re,socket
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import atomic_json,read_json,sha,fail
from .state_protein_map import pinned
from .state_catalog import expanded_targets,AUGMENTED_CATALOG_ID
from .cptac_scope import CANCER_MAP
from .cptac_omics_prepare import gmt
FORMAT='CLA_CPTAC_INTEGRATED_INPUT_V1'

def verify_tree(path):
    p=Path(path);rec=read_json(p)
    if rec.get('status',rec.get('state'))!='COMPLETE':fail('CPTAC_COMPONENT_INCOMPLETE',str(p))
    for rel,item in rec.get('files',{}).items():
        if sha(p.parent/rel)!=item['sha256']:fail('CPTAC_COMPONENT_HASH',rel)
    return rec

def md5(path):
    h=hashlib.md5()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_gct(path):
    with Path(path).open() as f:
        version=f.readline().strip();dims=[int(x) for x in f.readline().strip().split('\t')]
        if version=='#1.3':
            nr,nc,nrd,ncd=dims
            header=f.readline().rstrip('\n').split('\t')
            for _ in range(ncd):next(f)
            data=list(csv.reader(f,delimiter='\t'))
            identifiers=[r[0] for r in data];samples=header[1+nrd:]
            matrix=[r[1+nrd:] for r in data]
        elif version=='#1.2':
            nr,nc=dims;header=f.readline().rstrip('\n').split('\t');data=list(csv.reader(f,delimiter='\t'))
            identifiers=[r[0] for r in data];samples=header[2:];matrix=[r[2:] for r in data]
        else:fail('CPTAC_GCT_FORMAT',str(path))
    f=pd.DataFrame(matrix,index=identifiers,columns=samples).replace({'NA':np.nan,'NaN':np.nan,'nan':np.nan,'':np.nan})
    f=f.apply(pd.to_numeric,errors='raise')
    if f.shape!=(nr,nc) or not f.index.is_unique or not f.columns.is_unique:fail('CPTAC_GCT_DIMENSIONS',str(path))
    return f

def ptm_observed_mask(values,sets,minimum=5):
    counts={};usable={}
    for sid,tokens in sets.items():
        up={t.rsplit(';',1)[0] for t in tokens if t.endswith(';u')}
        down={t.rsplit(';',1)[0] for t in tokens if t.endswith(';d')}
        u=values.loc[values.index.intersection(up)].notna().sum()
        d=values.loc[values.index.intersection(down)].notna().sum()
        total=values.loc[values.index.intersection(up|down)].notna().sum()
        # Broad sets a zero when a single unsupported direction is combined
        # with too few sites in the other direction; expose that as unavailable.
        supported=(total>=minimum)&~((u.eq(1)&d.lt(minimum))|(d.eq(1)&u.lt(minimum)))
        counts[sid]=total;usable[sid]=supported
    return pd.DataFrame(counts).T,pd.DataFrame(usable).T

def sample_to_patient(sample,cohort,allowed):
    prefix='CPTAC:'+cohort+':'
    if not isinstance(sample,str) or not sample.startswith(prefix) or not sample.endswith('::T'):
        fail('CPTAC_SCORE_TISSUE','Only exact cohort tumor sample IDs are accepted',sample=sample)
    patient=sample[:-3]
    if patient not in allowed:fail('CPTAC_SCORE_PATIENT','Outside patient authority',sample=sample)
    return patient

def clinical_table(raw):
    cols=['patient_id','source_cohort','cancer_id','native_case_id']
    out=raw[cols].copy()
    out['age_years']=pd.to_numeric(raw.Age,errors='coerce')
    out['sex']=raw.Sex.map({'Female':'female','Male':'male'})
    out['stage']=raw.Stage
    for endpoint in ['OS','PFS']:
        days=pd.to_numeric(raw[endpoint+'_days'],errors='coerce')
        event=pd.to_numeric(raw[endpoint+'_event'],errors='coerce')
        valid=days.notna()&days.ge(0)&event.isin([0,1])
        out[endpoint+'_days']=days.where(valid)
        out[endpoint+'_event']=event.where(valid)
        out[endpoint+'_available']=valid
    return out

def unsupported_reason(target):
    sid=target['id']
    if sid=='Leukocyte_fraction':return 'ORIGINAL_METHYLATION_DECONVOLUTION_ENDPOINT_NOT_AVAILABLE'
    if target['measurement']=='DNA_METHYLATION':return 'REQUIRES_ORIGINAL_CPG_PROBES_NOT_GENE_PROMOTER_AGGREGATES'
    if target['family'] in ['mutational_signature'] or sid.startswith('SBS'):return 'PUBLISHED_SBS_NOT_EQUIVALENT_TO_CURRENT_60_SIGNATURE_QC_AND_FITTING'
    if sid.startswith('HRD') or sid=='Aneuploidy':return 'REQUIRES_APPROPRIATE_ALLELE_SPECIFIC_CNV_OR_ORIGINAL_PRECOMPUTED_ENDPOINT'
    if sid=='MSI':return 'PUBLISHED_MSI_H_BINARY_IS_NOT_CURRENT_CONTINUOUS_MSISENSOR2'
    if sid=='TLS12_Chemokine_Published':return 'FROZEN_ORIGINAL_PCA_PROJECTION_NOT_AVAILABLE'
    if sid=='Leukocyte_fraction':return 'ORIGINAL_METHYLATION_DECONVOLUTION_ENDPOINT_NOT_AVAILABLE'
    return 'NO_VALIDATED_EQUIVALENT_INPUT_OR_METHOD'

def assemble(cfg,output):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','Run on 149')
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('CPTAC_OUTPUT_EXISTS','Use a new versioned output')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'CPTAC_FORMAL_ALIGNMENT_AND_SCORE_ASSEMBLY'})
    aligned=pinned(cfg['aligned_complete']).parent;ac=verify_tree(aligned/'COMPLETE.json')
    omics=pinned(cfg['omics_input_complete']).parent;oc=verify_tree(omics/'COMPLETE.json')
    native=pinned(cfg['native_state_complete']).parent;nc=verify_tree(native/'COMPLETE.json')
    extra=pinned(cfg['extra_state_complete']).parent;ec=verify_tree(extra/'COMPLETE.json')
    rna=Path(cfg['rna_scores_root']);assay_scores=Path(cfg['omics_scores_root'])
    ptm_scores=Path(cfg.get('ptm_scores_root',cfg['omics_scores_root']))
    patients=pd.read_csv(aligned/'CPTAC_PATIENTS.tsv',sep='\t',dtype=str)
    patients=patients.sort_values(['source_cohort','patient_id']).reset_index(drop=True)
    if not patients.patient_id.is_unique or not patients.native_case_id.is_unique:fail('CPTAC_PATIENT_COLLISION','Duplicate biological native case')
    patient_set=set(patients.patient_id)
    targets=expanded_targets(AUGMENTED_CATALOG_ID);columns=[t['column'] for t in targets]
    state=pd.DataFrame(np.nan,index=patients.patient_id,columns=columns)
    source_records=[];state_methods={}
    def capture(p,role):
        p=Path(p);source_records.append({'path':str(p),'sha256':sha(p),'role':role,'bytes':p.stat().st_size})
    for name in ['aligned_complete','omics_input_complete','native_state_complete','extra_state_complete',
                 'pathway_gmt','ptm_gmt','state_protein_map_complete','tcga_state','methylation_definition']:
        capture(pinned(cfg[name]),name)
    if cfg.get('source_spec'):
        spec_path=pinned(cfg['source_spec']);capture(spec_path,'FROZEN_SCORING_SOURCE_SPEC')
        spec=read_json(spec_path)
        for role,source in spec['references'].items():
            capture(pinned(source),'REFERENCE::'+role)
        capture(pinned(spec['download_manifest']),'DOWNLOAD_MANIFEST')
    for source in cfg.get('execution_evidence',[]):capture(pinned(source),'EXECUTION_EVIDENCE')
    for p in [extra/'XCELL.tsv.gz',extra/'EXTEND_NATIVE.tsv.gz',native/'RNA_STEMNESS_NATIVE.parquet',native/'CIBERSORT_RELATIVE_DERIVED.parquet']:
        capture(p,'ADDITIONAL_NATIVE_STATES')
    for cohort in ac['cohorts']:
        p=rna/'rna_scores'/(cohort+'.tsv.gz');rec=read_json(p.with_name(cohort+'.json'))
        if rec['state']!='COMPLETE' or rec.get('failures'):fail('CPTAC_RNA_SCORE_INCOMPLETE',cohort)
        if md5(p)!=rec['output_md5'] or md5(aligned/'matrices'/cohort/'RNA_TUMOR_SYMBOL_TPM.tsv.gz')!=rec['input_md5']:
            fail('CPTAC_RNA_SCORE_HASH',cohort)
        f=pd.read_csv(p,sep='\t')
        ids=[sample_to_patient(x,cohort,patient_set) for x in f.sample_id]
        if len(ids)!=len(set(ids)):fail('CPTAC_SCORE_DUPLICATE',cohort)
        f.index=ids
        shared=[c for c in columns if c in f]
        state.loc[ids,shared]=f[shared].to_numpy()
        for c in shared:state_methods[c]='RECOMPUTED_CPTAC_RNA_NATIVE;SEE_ORIGINAL_METHOD_RECEIPT'
        capture(p,'RNA_NATIVE_STATES');capture(p.with_name(cohort+'.json'),'RNA_METHOD_RECEIPT')
    xs=pd.read_csv(extra/'XCELL.tsv.gz',sep='\t').set_index('sample_id')
    xs.index=xs.index.str.removesuffix('::T');xs.columns=['xCell::'+x for x in xs.columns]
    for c in columns:
        if c in xs:
            state.loc[xs.index,c]=xs[c];state_methods[c]='XCELL_POOLED_CPTAC_REFERENCE_NATIVE'
    ext=pd.read_csv(extra/'EXTEND_NATIVE.tsv.gz',sep='\t').set_index('sample_id')
    ext.index=ext.index.str.removesuffix('::T')
    state.loc[ext.index,'EXTEND::published_score']=ext.NormEXTENDScores
    state_methods['EXTEND::published_score']='RECOMPUTED_EXTEND_NATIVE_PER_CPTAC_CANCER_MINMAX;NOT_TCGA_PUBLISHED_SCALE'
    stem=pd.read_parquet(native/'RNA_STEMNESS_NATIVE.parquet').set_index('sample_id')
    stem.index=stem.index.str.removesuffix('::T')
    for name in ['RNAss','EREG.EXPss']:
        col='stemness_rna::'+name
        state.loc[stem.index,col]=stem[name+'::CPTAC_native_0_1']
        state_methods[col]='FROZEN_MALTA_OCLR_SPEARMAN_WITH_CPTAC_NATIVE_MINMAX;RAW_RHO_RETAINED'
    relative=pd.read_parquet(native/'CIBERSORT_RELATIVE_DERIVED.parquet').set_index('patient_id')
    for c in relative:
        if c not in columns:fail('CPTAC_STATE_AXIS',c)
        state.loc[relative.index,c]=relative[c];state_methods[c]='DERIVED_RELATIVE_COMPOSITION_FROM_PUBLISHED_22_ABSOLUTE_COMPONENTS'
    published=pd.read_parquet(aligned/'PUBLISHED_PHENOTYPES.parquet').set_index('patient_id')
    for src,dest in [('WES_purity','absolute_purity_ploidy::purity'),('WES_ploidy','absolute_purity_ploidy::ploidy'),
                     ('TMB','mutation_load::Non-silent per Mb')]:
        state.loc[published.index,dest]=pd.to_numeric(published[src],errors='raise')
        state_methods[dest]='PUBLISHED_LINKEDOMICS_BCM_DOABSOLUTE_OR_MAFTOOLS;ORIGINAL_COHORT_METHOD'
    if not state.index.is_unique or set(state.index)!=patient_set or np.isinf(state.to_numpy()).any():fail('CPTAC_STATE_CONTENT','Invalid patient/values')
    joined=patients.merge(state.reset_index(),on='patient_id',validate='one_to_one')
    joined.to_parquet(out/'STATE_PATIENT.parquet',index=False)
    state_coverage=[];state_alignment=[]
    for t in targets:
        count=int(state[t['column']].notna().sum())
        state_alignment.append({**t,'observed_patients':count,'patient_coverage_percent':100*count/len(patients),
            'cptac_method':state_methods.get(t['column'],''),'availability':'AVAILABLE_NATIVE' if count else 'UNAVAILABLE',
            'reason':'' if count else unsupported_reason(t),
            'TCGA_scale_calibration':'NOT_ESTABLISHED;SAME_ID_DOES_NOT_AUTHORIZE_SCORE_POOLING'})
        for cohort,g in patients.groupby('source_cohort'):
            n=int(state.loc[g.patient_id,t['column']].notna().sum())
            state_coverage.append({'source_cohort':cohort,'cancer_id':CANCER_MAP[cohort],'state_id':t['id'],
                                  'patients':len(g),'observed_patients':n,'coverage_percent':100*n/len(g)})
    pd.DataFrame(state_alignment).to_csv(out/'STATE_ALIGNMENT.tsv',sep='\t',index=False)
    pd.DataFrame(state_coverage).to_csv(out/'STATE_COVERAGE.tsv',sep='\t',index=False)
    clinical_source=pd.read_csv(aligned/'CLINICAL.tsv',sep='\t',dtype=str)
    clinical_source.to_csv(out/'CLINICAL_SOURCE.tsv.gz',sep='\t',index=False,compression='gzip')
    clinical=clinical_table(clinical_source)
    clinical.to_parquet(out/'CLINICAL_ALIGNED.parquet',index=False)
    fields=[]
    for cohort,g in clinical.groupby('source_cohort'):
        for c in ['age_years','sex','stage','OS_days','PFS_days']:
            n=int(g[c].notna().sum());fields.append({'source_cohort':cohort,'field':c,'observed_patients':n,'patients':len(g),'coverage_percent':100*n/len(g)})
    pd.DataFrame(fields).to_csv(out/'CLINICAL_COVERAGE.tsv',sep='\t',index=False)
    patients.to_csv(out/'PATIENTS.tsv',sep='\t',index=False)
    sets=gmt(pinned(cfg['pathway_gmt']));ptmsets=gmt(pinned(cfg['ptm_gmt']))
    index=[];score_coverage=[];concordance=[]
    def export_scores(frame,cohort,layer,target_ids,method,counts=None):
        if not frame.index.is_unique or not frame.columns.is_unique:fail('CPTAC_SCORE_DUPLICATE',layer+cohort)
        if not set(frame.index)<=set(target_ids):fail('CPTAC_SCORE_TARGET',layer+cohort)
        pids=[sample_to_patient(x,cohort,patient_set) for x in frame.columns]
        frame=frame.reindex(target_ids);frame.columns=pids;frame.index.name='target_id'
        cohort_patients=patients.loc[patients.source_cohort.eq(cohort),'patient_id'].tolist()
        frame=frame.reindex(columns=cohort_patients)
        if np.isinf(frame.to_numpy()).any():fail('CPTAC_SCORE_INFINITY',layer+cohort)
        folder=out/'scores'/layer;folder.mkdir(parents=True,exist_ok=True);path=folder/(cohort+'.parquet')
        frame.reset_index().to_parquet(path,index=False)
        index.append({'source_cohort':cohort,'cancer_id':CANCER_MAP[cohort],'layer':layer,'method':method,
                      'path':str(path.relative_to(out)),'sha256':sha(path),'patients':len(cohort_patients),'assayed_patients':len(pids),'targets':len(frame),
                      'observed_targets':int(frame.notna().any(axis=1).sum()),'finite_scores':int(frame.notna().sum().sum())})
        denom=int(patients.source_cohort.eq(cohort).sum())
        for sid,n in frame.notna().sum(axis=1).items():
            score_coverage.append({'source_cohort':cohort,'layer':layer,'target_id':sid,'observed_patients':int(n),
                                   'cohort_patients':denom,'coverage_percent':100*n/denom})
        return frame
    for cohort in ac['cohorts']:
        print('ASSEMBLE_SCORES',cohort,flush=True)
        rp=rna/'pathway_scores'/(cohort+'.tsv.gz')
        rf=pd.read_csv(rp,sep='\t').set_index('pathway_id');capture(rp,'RNA_PATHWAY')
        rmat=export_scores(rf,cohort,'RNA_PATHWAY',list(sets),'GSVA_ssGSEA_alpha0.25_cohort_range_normalized')
        pp=assay_scores/cohort/'protein';pr=verify_tree(pp/'COMPLETE.json')
        if md5(omics/cohort/'PROTEIN_SYMBOL.tsv.gz')!=pr['input_md5']:fail('CPTAC_PROTEIN_INPUT_DRIFT',cohort)
        pf=pp/'PROTEIN_PATHWAY_ES.tsv.gz';pframe=pd.read_csv(pf,sep='\t').set_index('pathway_id');capture(pf,'PROTEIN_PATHWAY');capture(pp/'COMPLETE.json','PROTEIN_METHOD')
        pmat=export_scores(pframe,cohort,'PROTEIN_PATHWAY',list(sets),'GSVA_ssGSEA_alpha0.25_raw_ES_NA_aware')
        for sid in sets:
            x=rmat.loc[sid];y=pmat.loc[sid];common=x.dropna().index.intersection(y.dropna().index)
            if len(common)<12:continue
            xv=x[common].to_numpy();yv=y[common].to_numpy()
            if np.std(xv)==0 or np.std(yv)==0:continue
            rho,pval=spearmanr(xv,yv)
            concordance.append({'source_cohort':cohort,'pathway_id':sid,'paired_patients':len(common),
                                'spearman_rho':float(rho),'p_value':float(pval),'interpretation':'CROSS_ASSAY_CONCORDANCE_NOT_ACTIVATION_GROUND_TRUTH'})
        mp=omics/cohort/'METHYLATION_PATHWAY_MEAN_BETA.tsv.gz'
        if mp.exists():
            export_scores(pd.read_csv(mp,sep='\t').set_index('pathway_id'),cohort,'METHYLATION_PATHWAY',list(sets),
                          'OBSERVED_PROMOTER_RELATED_GENE_MEAN_BETA_NOT_ACTIVITY')
            capture(mp,'METHYLATION_PATHWAY')
        else:
            export_scores(pd.DataFrame(index=list(sets),dtype=float),cohort,'METHYLATION_PATHWAY',list(sets),
                          'NOT_ASSAYED_IN_THIS_COHORT;ALL_NA')
        pt=ptm_scores/cohort/'phosphorylation';prec=verify_tree(pt/'COMPLETE.json')
        if md5(omics/cohort/'PHOSPHO_SITE_CENTERED.gct')!=prec['input_md5']:fail('CPTAC_PTM_INPUT_DRIFT',cohort)
        pfile=pt/'PTMSEA-scores.gct';ptmf=read_gct(pfile);capture(pfile,'PTMSEA');capture(pt/'COMPLETE.json','PTM_METHOD')
        centered=read_gct(omics/cohort/'PHOSPHO_SITE_CENTERED.gct')
        counts,usable=ptm_observed_mask(centered,ptmsets)
        if list(centered.columns)!=list(ptmf.columns):fail('CPTAC_PTM_SAMPLE_ORDER',cohort)
        ptmf=ptmf.where(usable.reindex(index=ptmf.index,columns=ptmf.columns))
        support=out/'ptm_coverage';support.mkdir(exist_ok=True)
        counts.index.name='target_id';counts.to_csv(support/(cohort+'_observed_sites.tsv.gz'),sep='\t',compression='gzip')
        export_scores(ptmf,cohort,'PHOSPHO_PTMSEA',list(ptmsets),'BROAD_PTM_SEA_DIRECTIONAL_ES_SITE_COHORT_CENTERED')

    # PTMsigDB has a distinct target ontology: preserve exact directional sites,
    # and bridge to current genes without inventing a one-to-one MSigDB match.
    signature_sites=pd.DataFrame([{'ptm_signature_id':sid,'ptm_sequence_id':token.rsplit(';',1)[0],
        'direction':token.rsplit(';',1)[1]} for sid,tokens in ptmsets.items() for token in tokens])
    ptm_bridges=[]
    for cohort in ac['cohorts']:
        bridge=pd.read_csv(omics/cohort/'PHOSPHO_SEQUENCE_SITE_BRIDGE.tsv.gz',sep='\t')
        bridge=signature_sites.merge(bridge,on='ptm_sequence_id',how='inner',validate='many_to_many')
        bridge.insert(0,'source_cohort',cohort)
        ptm_bridges.append(bridge)
    pd.concat(ptm_bridges,ignore_index=True).to_parquet(out/'PTM_SIGNATURE_SITE_GENE_BRIDGE.parquet',index=False)
    feature_map=pd.read_parquet(aligned/'FEATURE_ALIGNMENT.parquet')
    gene_map=feature_map[['gene_id','model_gene_id','reference_symbol','uniprot_accessions','reference_presence']].drop_duplicates()
    member_frame=pd.DataFrame([{'pathway_id':sid,'reference_symbol':symbol} for sid,symbols in sets.items() for symbol in symbols])
    member_frame.merge(gene_map,on='reference_symbol',how='left',validate='many_to_many').to_parquet(out/'PATHWAY_GENE_PROTEIN_BRIDGE.parquet',index=False)
    pd.DataFrame(index).to_csv(out/'SCORE_INDEX.tsv',sep='\t',index=False)
    pd.DataFrame(score_coverage).to_csv(out/'SCORE_COVERAGE.tsv.gz',sep='\t',index=False,compression='gzip')
    concord=pd.DataFrame(concordance)
    if not concord.empty:
        from scipy.stats import false_discovery_control
        concord['BH_FDR_within_cohort']=concord.groupby('source_cohort')['p_value'].transform(lambda x:false_discovery_control(x.to_numpy(),method='bh'))
    concord.to_csv(out/'RNA_PROTEIN_PATHWAY_CONCORDANCE.tsv.gz',sep='\t',index=False,compression='gzip')
    # Preserve original State membership plus explicit per-gene measured coverage, without converting associations to activation edges.
    smc=pinned(cfg['state_protein_map_complete']);sm=verify_tree(smc)
    mp=smc.parent/'STATE_PROTEIN_MAP.tsv.gz';members=pd.read_csv(mp,sep='\t',dtype=str,keep_default_na=False)
    keep=[x for x in ['state_id','gene_id','reference_symbol','uniprot_accessions','gene_biotype','mapping_status'] if x in members]
    members=members.drop_duplicates();members.to_parquet(out/'STATE_GENE_PROTEIN_BRIDGE.parquet',index=False)
    measured=[]
    for cohort in ac['cohorts']:
        for layer in ('rna','protein','phosphorylation','methylation_gene'):
            p=aligned/'matrices'/cohort/(layer+('_TumorNormal.parquet' if layer=='rna' else '_Tumor.parquet'))
            if not p.exists():continue
            f=pd.read_parquet(p).set_index('feature_id');f=f[[c for c in f if c.endswith('::T')]]
            for feature,n in f.notna().sum(axis=1).items():
                measured.append({'source_cohort':cohort,'layer':layer,'feature_id':feature,'gene_id':str(feature).split('|')[0],
                                 'observed_patients':int(n),'assayed_patients':f.shape[1]})
    pd.DataFrame(measured).to_parquet(out/'MEASURED_FEATURE_COVERAGE.parquet',index=False)
    for name in ['CPTAC_ASSAY_COVERAGE.tsv','COHORT_COVERAGE.tsv','CPTAC_EXCLUDED_SAMPLES.tsv','SOURCE_NONFINITE_MEASUREMENTS.tsv',
                 'SOURCE_INVALID_RNA_VALUES.tsv','RNA_SCORING_EXCLUDED_GENES.tsv','FEATURE_ALIGNMENT.parquet','CPTAC_SAMPLE_ASSAY_MAP.tsv']:
        # Small audit tables and the feature crosswalk are part of the portable result package.
        import shutil
        shutil.copyfile(aligned/name,out/name)
    capture(mp,'STATE_PROTEIN_MAPPING')
    atomic_json(out/'BOUND_SOURCES.json',source_records)
    atomic_json(out/'METHOD_POLICY.json',{'no_MOGSA':True,'no_95_percent_assay_gene_gate':True,'missing_values':'NA_NO_IMPUTATION',
       'native_scores':'Each assay and cohort retains its published/recomputed native scale',
       'alignment':'COMMON_GENE_PATHWAY_STATE_IDENTIFIERS;NOT_A_VALIDATION_OF_EQUAL_CALIBRATION',
       'task_weights':{'pathway':1.0,'state':0.5},'HHGT_training_pooling':False,
       'methylation':read_json(pinned(cfg['methylation_definition'])),'State_native_methods':state_methods,
       'PTM_significance':'ES_only;nperm_0;placeholder_pvalues_are_not_inferential_and_are_not_integrated',
       'PTM_normalization':'Frozen_median_for_each_site_within_its_CPTAC_tumor_cohort;not_protein_adjusted',
       'future_inductive_training':'Fit cohort-derived normalization only on training/reference cases, freeze before held-out application; do not pool these exploratory cohort scores as a calibrated Y'})
    producer=out/'PRODUCERS';producer.mkdir()
    repo=Path(__file__).resolve().parents[1]
    import shutil
    for source in list((repo/'cla_engine').glob('cptac_*.py'))+list((repo/'scripts').glob('score_cptac_*.R'))+list((repo/'scripts').glob('cptac_*.R')):
        shutil.copyfile(source,producer/source.name)
    result={'format':FORMAT,'status':'COMPLETE','target_host':'149','config':cfg,'patients':len(patients),
        'cohorts':ac['cohorts'],'state_catalog':AUGMENTED_CATALOG_ID,'state_axis_count':len(targets),
        'observed_state_count':sum(x['observed_patients']>0 for x in state_alignment),
        'MSigDB_pathway_axis':len(sets),'PTM_signature_axis':len(ptmsets),'training_pooling':False,
        'files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}}
    atomic_json(out/'COMPLETE.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=assemble(read_json(a.config),a.output);print('COMPLETE',r['patients'],r['observed_state_count'])
if __name__=='__main__':main()
