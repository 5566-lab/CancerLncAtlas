"""Cell-type expression and matched pathway/State context, without primary fusion.

A pair of cell-type averages is not a donor correlation. Missing observations
stay missing. Independent replication is not claimed merely because these
measurements were not used in R6 training: study/patient overlap must be audited.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import fail,sha,digest,atomic_json
from .tables import table,require,strict_bool
from .identity import frame_ids,unique_values
from .statistics import bh
from .interpretation import queries,output_directory,finish,KEYS

CONTEXT=['dataset_id','cancer_id','cell_type','cell_state']
ASSOCIATION_FEATURES=('rho','abs_rho','neglog10_fdr','available')

def rename_aliases(f,aliases,role):
    f=frame_ids(f)
    for dst,choices in aliases.items():
        present=[c for c in [dst,*choices] if c in f]
        if len(present)>1:
            for other in present[1:]:
                a=f[present[0]].astype('string').fillna('<NA>');b=f[other].astype('string').fillna('<NA>')
                if not a.equals(b):fail('SC_ALIAS_CONFLICT',role,columns=present)
        if present and dst not in f:f=f.rename(columns={present[0]:dst})
    if 'cell_state' not in f:f['cell_state']='__UNSTRATIFIED__'
    else:f['cell_state']=f.cell_state.fillna('__UNSTRATIFIED__').replace('','__UNSTRATIFIED__')
    # No lower-casing, fuzzy cell matching, or dropping supplied state labels.
    for col in ['cell_type','cell_state']:
        if col in f:f[col]=f[col].astype('string').str.strip()
    return f

def eligible_manifest(f):
    require(f,['dataset_id','formal_eligible','model_derived','outcome_derived'],'single-cell manifest')
    f=unique_values(frame_ids(f),['dataset_id'],role='single-cell manifest')
    for col in ['formal_eligible','model_derived','outcome_derived']:
        if f[col].isna().any():fail('SC_MANIFEST_FLAG_UNKNOWN',col)
    good=strict_bool(f.formal_eligible)&~strict_bool(f.model_derived)&~strict_bool(f.outcome_derived)
    return set(f.loc[good,'dataset_id']),{'datasets_total':len(f),'eligible_datasets':sorted(f.loc[good,'dataset_id']),'excluded_datasets':sorted(f.loc[~good,'dataset_id'])}

def normalise_associations(f,manifest):
    """Reuse the audited private-head contract; raw p-values never substitute for FDR."""
    f=frame_ids(f);manifest=frame_ids(manifest)
    fdr_columns=[name for name in f.columns if str(name).strip().lower() in {'fdr','q_value','padj'}]
    if not fdr_columns:fail('SC_ASSOCIATION_FDR_MISSING','Require fdr/q_value/padj; p_value is not an FDR substitute')
    reference=pd.to_numeric(f[fdr_columns[0]],errors='coerce')
    if (f[fdr_columns[0]].notna()&~np.isfinite(reference)).any() or (~reference.dropna().between(0,1)).any():
        fail('SC_ASSOCIATION_FDR_INVALID','FDR must be finite in [0,1]')
    for name in fdr_columns[1:]:
        other=pd.to_numeric(f[name],errors='coerce')
        if not np.allclose(reference.to_numpy(float),other.to_numpy(float),equal_nan=True):fail('SC_ASSOCIATION_FDR_CONFLICT','Multiple FDR columns disagree',columns=fdr_columns)
    from cc_hhgt.v32.single_cell_training import (SingleCellTrainingError,
        normalise_dataset_manifest,normalise_single_cell_associations)
    try:
        strict_manifest=normalise_dataset_manifest(manifest)
        values=normalise_single_cell_associations(f,strict_manifest)
    except SingleCellTrainingError as exc:
        fail('SC_ASSOCIATION_CONTRACT_INVALID',str(exc))
    allowed,_=eligible_manifest(manifest)
    values=values.loc[values.dataset_id.isin(allowed)&values.formal_row].copy()
    if (~values.association_effect.between(-1,1)).any():fail('SC_ASSOCIATION_EFFECT_RANGE','Spearman rho must be in [-1,1]')
    if (~values.association_fdr.between(0,1)).any():fail('SC_ASSOCIATION_FDR_INVALID','FDR must be in [0,1]')
    return values,strict_manifest

def association_candidate_features(frame,manifest,rows,cell_types,cancer):
    """Exact cancer/lncRNA/pathway/cell-type features; never pathway-family broadcast."""
    contexts=list(map(str,cell_types))
    if not contexts or len(contexts)!=len(set(contexts)):fail('SC_ASSOCIATION_CONTEXT_CONFIG','Configured cell types must be nonempty and unique')
    values,strict_manifest=normalise_associations(frame,manifest)
    qualified=set(strict_manifest.loc[strict_manifest.qualified&strict_manifest.cancer_id.eq(cancer),'dataset_id'])
    values=values.loc[values.cancer_id.eq(cancer)&values.dataset_id.isin(qualified)].copy()
    states=values.cell_state.fillna('').astype(str).str.strip().str.lower()
    if (~states.isin(['','overall','__unstratified__'])).any():
        fail('SC_ASSOCIATION_STATE_COLLAPSE_FORBIDDEN','Primary cell-type context accepts only declared overall rows; state-specific rows require an explicit dimension')
    unexpected=sorted(set(values.cell_type_major.astype(str))-set(contexts))
    if unexpected:fail('SC_ASSOCIATION_CONTEXT_UNDECLARED','No silent cell-type dropping',cell_types=unexpected)
    requested={(str(lnc),str(target)) for row_cancer,lnc,target in rows if str(row_cancer)==str(cancer)}
    values=values.loc[[pair in requested for pair in zip(values.lncrna_id.astype(str),values.pathway_id.astype(str))]].copy()
    keys=['lncrna_id','pathway_id','cell_type_major']
    if values.duplicated(keys).any():
        conflicts=values.groupby(keys,observed=True)[['association_effect','association_fdr']].nunique().max(axis=1).gt(1)
        if conflicts.any():fail('SC_ASSOCIATION_DUPLICATE_CONFLICT','Exact pair/cell-type rows disagree',rows=int(conflicts.sum()))
        values=values.drop_duplicates(keys)
    width=len(contexts)*len(ASSOCIATION_FEATURES);features=np.zeros((len(rows),width),np.float32)
    row_index={(str(lnc),str(target)):index for index,(row_cancer,lnc,target) in enumerate(rows) if str(row_cancer)==str(cancer)}
    context_index={name:index for index,name in enumerate(contexts)}
    per_context={name:0 for name in contexts}
    for row in values.itertuples(index=False):
        index=row_index.get((str(row.lncrna_id),str(row.pathway_id)));context=context_index.get(str(row.cell_type_major))
        if index is None or context is None:continue
        start=context*len(ASSOCIATION_FEATURES);rho=float(row.association_effect);fdr=float(row.association_fdr)
        features[index,start:start+4]=[rho,abs(rho),-np.log10(max(fdr,1e-12)),1.]
        per_context[contexts[context]]+=1
    report={'status':'EXACT_SINGLECELL_ASSOCIATION_INPUT_CONTEXT','feature_order':[f'{cell}:{feature}' for cell in contexts for feature in ASSOCIATION_FEATURES],
        'candidate_rows':len(rows),'candidate_rows_with_association':int(features[:,3::4].any(axis=1).sum()),'per_context_exact_rows':per_context,
        'qualified_datasets':sorted(qualified),'exact_pathway_join':True,'family_broadcast':False,'p_value_used_as_fdr':False,
        'association_is_primary_input_not_independent_validation':True}
    return features,report,values

def expression_table(f,allowed):
    f=rename_aliases(f,{'cell_type':['cell_type_major'],'mean_log_expression':['lnc_mean_log_expression'],
        'detection_rate':['lnc_detection_rate'],'n_cells':['lnc_n_cells']},'expression')
    require(f,CONTEXT+['lncrna_id','mean_log_expression','detection_rate'],'SC expression')
    excluded=len(f.loc[~f.dataset_id.isin(allowed)]);f=f.loc[f.dataset_id.isin(allowed)].copy()
    if f[CONTEXT+['lncrna_id']].isna().any().any():fail('SC_CONTEXT_MISSING','No implicit cell type or patient context')
    for col in ['mean_log_expression','detection_rate','n_cells','n_donors','n_detected']:
        if col not in f:f[col]=np.nan
        raw=f[col];f[col]=pd.to_numeric(raw,errors='coerce')
        if (raw.notna()&~np.isfinite(f[col])).any():fail('SC_NUMERIC_INVALID',col)
    if (~f.detection_rate.dropna().between(0,1)).any():fail('SC_DETECTION_RANGE','Expected fraction in [0,1], not percentages')
    for col in ['n_cells','n_donors','n_detected']:
        v=f[col].dropna()
        if (v.lt(0)|v.ne(np.floor(v))).any():fail('SC_COUNT_INVALID',col)
    both=f.n_cells.notna()&f.n_detected.notna()
    if (f.loc[both,'n_detected']>f.loc[both,'n_cells']).any():fail('SC_DETECTED_EXCEEDS_CELLS','Invalid detection denominator')
    enough=both&f.n_cells.gt(0)&f.detection_rate.notna()
    if (abs(f.loc[enough,'detection_rate']-f.loc[enough,'n_detected']/f.loc[enough,'n_cells'])>1e-4).any():fail('SC_DETECTION_COUNT_CONFLICT','Fraction contradicts counts')
    if 'feature_universe_status' in f:
        f['expression_available']=f.feature_universe_status.astype(str).str.lower().isin(['pass','passed','verified','available','complete'])
    else:f['expression_available']=True  # Descriptive observations only, not formal replication readiness.
    f['expression_available'] &= f.mean_log_expression.notna()&f.detection_rate.notna()
    f['expression_available'] &= ~f.n_cells.eq(0)
    f['feature_universe_verified']=('feature_universe_status' in f)
    if 'lnc_celltype_available' in f:f['expression_available'] &= strict_bool(f.lnc_celltype_available)
    if 'malignant_status' not in f:f['malignant_status']='UNSPECIFIED_NOT_INFERRED_FROM_EXPRESSION'
    retain=CONTEXT+['lncrna_id','mean_log_expression','detection_rate','n_cells','n_donors','n_detected','expression_available','feature_universe_verified','malignant_status']
    return unique_values(f,CONTEXT+['lncrna_id'],[x for x in retain if x not in CONTEXT+['lncrna_id']],'SC expression/context'),excluded

def activity_table(f,allowed):
    f=f.copy()
    if 'task' not in f:
        f['task']='state' if 'state_id' in f and 'pathway_id' not in f else 'pathway'
    if not f.task.isin(['pathway','state']).all():fail('SC_TASK_INVALID','Explicit pathway/state role required')
    f=rename_aliases(f,{'cell_type':['cell_type_major'],'target_id':['pathway_id','state_id'],
        'activity_score':['activity_value','activity','score','mean_ucell']},'activity')
    require(f,CONTEXT+['target_id','activity_score'],'SC activity; dataset-wide scores cannot be broadcast')
    f=f.loc[f.dataset_id.isin(allowed)].copy()
    if 'task' not in f:f['task']='pathway'
    if 'activity_kind' not in f:
        f['activity_kind']=f['score_method'] if 'score_method' in f else 'UNSPECIFIED_METHOD'
    if f[CONTEXT+['target_id','activity_kind']].isna().any().any():fail('SC_ACTIVITY_CONTEXT_MISSING','Context is required')
    raw=f.activity_score;f['activity_score']=pd.to_numeric(raw,errors='coerce')
    if (raw.notna()&~np.isfinite(f.activity_score)).any():fail('SC_ACTIVITY_NONFINITE','Score must be finite or unavailable')
    if 'coverage_rate' in f:
        original=f.coverage_rate
        f['coverage_rate']=pd.to_numeric(original,errors='coerce')
        if (original.notna()&~np.isfinite(f.coverage_rate)).any():fail('SC_PATHWAY_COVERAGE_INVALID','Invalid numeric coverage')
        if (~f.coverage_rate.dropna().between(0,1)).any():fail('SC_PATHWAY_COVERAGE_INVALID','Expected fraction')
    else:f['coverage_rate']=np.nan
    f['activity_coverage_verified']=f.coverage_rate.notna()
    # Without coverage it can be displayed as a measured source score, not formal validation.
    f['activity_available']=f.activity_score.notna()
    keys=CONTEXT+['task','target_id','activity_kind']
    if 'donor_id' in f:
        if f.donor_id.isna().any() or f.donor_id.astype('string').str.strip().eq('').any():fail('SC_DONOR_ID_MISSING','Do not treat missing donors as biological replicates')
        f=unique_values(f,keys+['donor_id'],['activity_score','coverage_rate'],'SC donor activity')
        f=f.groupby(keys,dropna=False,observed=True).agg(activity_score=('activity_score','mean'),coverage_rate=('coverage_rate','min'),activity_donors=('donor_id','nunique')).reset_index()
        f['activity_available']=f.activity_score.notna();f['activity_coverage_verified']=f.coverage_rate.notna()
    else:
        f=unique_values(f,keys,['activity_score','coverage_rate','activity_available','activity_coverage_verified'],'SC activity/context');f['activity_donors']=np.nan
    return f

def pair_context(q,expression,activity):
    joined=q.merge(expression,on=['cancer_id','lncrna_id'],how='left',validate='many_to_many')
    if activity is not None and not activity.empty:
        joined=joined.merge(activity,on=CONTEXT+['task','target_id'],how='left',validate='many_to_many')
    else:
        for name in ['activity_score','coverage_rate','activity_donors']:joined[name]=np.nan
        joined['activity_kind']='NOT_BOUND';joined['activity_available']=False;joined['activity_coverage_verified']=False
    joined['expression_available']=joined.expression_available.astype('boolean').fillna(False).astype(bool)
    joined['activity_available']=joined.activity_available.astype('boolean').fillna(False).astype(bool)
    joined['same_context_measurements']=joined.expression_available&joined.activity_available
    joined['correlation']=np.nan;joined['correlation_available']=False
    joined['interpretation']='CO_CONTEXT_OBSERVATIONS_NOT_REGULATORY_PROOF'
    joined['unavailable_reason']=np.select([~joined.expression_available,~joined.activity_available],['EXPRESSION_CONTEXT_UNAVAILABLE','TARGET_CONTEXT_UNAVAILABLE'],default='DONOR_CORRELATION_NOT_SUPPLIED')
    joined['independent_replication_verified']=False
    return joined

def donor_correlations(frame,q,minimum=8):
    """Only genuinely donor-paired observations; no single-cell pseudoreplication."""
    frame=frame.copy()
    if 'task' not in frame:frame['task']='state' if 'state_id' in frame and 'pathway_id' not in frame else 'pathway'
    f=rename_aliases(frame,{'cell_type':['cell_type_major'],'target_id':['pathway_id','state_id']},'donor')
    if 'task' not in f:f['task']='pathway'
    require(f,CONTEXT+['donor_id','lncrna_id','task','target_id','expression','activity','activity_kind'],'paired donor data')
    if f[CONTEXT+['donor_id']].isna().any().any() or f.donor_id.astype('string').str.strip().eq('').any():fail('SC_DONOR_ID_MISSING','Every biological replicate must be identified')
    if not f.task.isin(['pathway','state']).all():fail('SC_TASK_INVALID','Explicit task role required')
    f=f.merge(q[KEYS],on=KEYS,how='inner',validate='many_to_one')
    keys=CONTEXT+['lncrna_id','task','target_id','activity_kind']
    f=unique_values(f,keys+['donor_id'],['expression','activity'],'paired donor data')
    for col in ['expression','activity']:
        raw=f[col];f[col]=pd.to_numeric(raw,errors='coerce')
        if (raw.notna()&~np.isfinite(f[col])).any():fail('SC_DONOR_NUMERIC_INVALID',col)
    rows=[]
    for ident,g in f.groupby(keys,sort=True,dropna=False):
        good=g.dropna(subset=['expression','activity']);n=good.donor_id.nunique();valid=n>=minimum and good.expression.nunique()>1 and good.activity.nunique()>1
        rho,p=spearmanr(good.expression,good.activity) if valid else (np.nan,np.nan)
        rows.append(dict(zip(keys,ident))|{'n_donors':n,'rho':rho,'p_value':p,'available':valid,'donor_ids_sha256':digest(sorted(good.donor_id)),
                    'unit':'BIOLOGICAL_DONOR','independence_from_primary_not_verified':True})
    result=pd.DataFrame(rows)
    if len(result):
        result['fdr_within_requested_queries']=np.nan
        for _,g in result.groupby(['dataset_id','cancer_id','task','target_id','activity_kind']):result.loc[g.index,'fdr_within_requested_queries']=bh(g.p_value.to_numpy())
        result['fdr_scope']='REQUESTED_FROZEN_QUERY_SUBSET_NOT_ALL_GENOME_TESTS'
    return result

def interpret(query_path,cfg,output,*,activity_path=None,donor_path=None,association_path=None):
    q,qr=queries(query_path,cfg.get('interpretation',{}).get('max_explanation_candidates',5000));out=output_directory(output)
    root=Path(cfg['imports_root'])/'singlecell';mp=root/'dataset_manifest_33c.parquet'
    if cfg.get('synthetic_fixture') and not mp.exists():mp=mp.with_suffix('.tsv')
    if not mp.exists():fail('SC_MANIFEST_MISSING','Known imported metadata required for interpretation',path=str(mp))
    allowed,manifest_report=eligible_manifest(table(mp));sources={str(mp):sha(mp)};expressions=[];excluded=0
    for c in sorted(q.cancer_id.unique()):
        p=root/'lnc_celltype'/f'{c}.parquet'
        if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
        if not p.exists():continue
        e,n=expression_table(table(p),allowed);expressions.append(e);excluded+=n;sources[str(p)]=sha(p)
    expr=pd.concat(expressions,ignore_index=True) if expressions else pd.DataFrame(columns=CONTEXT+['lncrna_id','mean_log_expression','detection_rate','n_cells','n_donors','n_detected','expression_available','feature_universe_verified','malignant_status'])
    activity_paths=[Path(activity_path)] if activity_path else sorted((root/'activity').glob('*.parquet'))
    if cfg.get('synthetic_fixture') and not activity_paths:activity_paths=sorted((root/'activity').glob('*.tsv'))
    frames=[]
    for p in activity_paths:
        frames.append(activity_table(table(p),allowed));sources[str(p)]=sha(p)
    activity=pd.concat(frames,ignore_index=True) if frames else None
    if activity is not None:activity=unique_values(activity,CONTEXT+['task','target_id','activity_kind'],role='SC activity files')
    result=pair_context(q,expr,activity)
    result.to_csv(out/'LNCRNA_TARGET_CELL_CONTEXT.tsv.gz',sep='\t',index=False)
    expr.to_csv(out/'CELLTYPE_EXPRESSION.tsv.gz',sep='\t',index=False)
    if donor_path:
        donor=table(donor_path);donor=donor.loc[donor.dataset_id.isin(allowed)]
        dc=donor_correlations(donor,q,int(cfg.get('interpretation',{}).get('minimum_donors',8)));dc.to_csv(out/'DONOR_ASSOCIATIONS.tsv.gz',sep='\t',index=False);sources[str(donor_path)]=sha(donor_path)
    else:dc=pd.DataFrame()
    pathway_queries=q.loc[q.task.eq('pathway')].rename(columns={'target_id':'pathway_id'}).copy()
    if association_path:
        supplied=Path(association_path)
        if not supplied.exists():fail('SC_ASSOCIATION_INPUT_MISSING','Requested association path does not exist',path=str(supplied))
        association_paths=sorted(supplied.glob('*.parquet')) if supplied.is_dir() else [supplied]
        if cfg.get('synthetic_fixture') and supplied.is_dir() and not association_paths:association_paths=sorted(supplied.glob('*.tsv'))
    else:
        association_paths=[]
        for cancer in sorted(pathway_queries.cancer_id.unique()):
            p=root/'association'/f'{cancer}.parquet'
            if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
            if p.exists():association_paths.append(p)
    association_frames=[]
    for p in association_paths:
        values,_=normalise_associations(table(p),table(mp));association_frames.append(values);sources[str(p)]=sha(p)
    if association_frames:
        associations=pd.concat(association_frames,ignore_index=True)
        associations=unique_values(associations,['dataset_id','cancer_id','lncrna_id','pathway_id','cell_type_major','cell_state'],role='SC association files')
        association_result=pathway_queries.merge(associations,on=['cancer_id','lncrna_id','pathway_id'],how='left',validate='one_to_many')
    else:
        association_result=pathway_queries.copy()
        for name in ['dataset_id','cell_type_major','cell_state','association_effect','association_fdr']:association_result[name]=np.nan
    association_result['association_available']=association_result.association_effect.notna()&association_result.association_fdr.notna()
    association_result['interpretation']='EXTERNAL_CELLTYPE_LNCRNA_PATHWAY_ASSOCIATION_NOT_CAUSAL_PROOF'
    association_result['used_by_primary_config']=cfg.get('singlecell_association_role')=='input_context'
    association_result['p_value_used_as_fdr']=False
    association_result.to_csv(out/'SINGLECELL_ASSOCIATIONS.tsv.gz',sep='\t',index=False)
    return finish(out,{'state':'SINGLECELL_CONTEXT_EXPORTED_NOT_CAUSAL_VALIDATION','module':'singlecell_interpretation','query':qr,'source_files':sources,
        'dataset_policy':manifest_report,'excluded_input_rows':excluded,'context_rows':len(result),'donor_correlation_rows':len(dc),
        'association_rows':len(association_result),'association_available_rows':int(association_result.association_available.sum()),
        'association_queries_covered':int(association_result.loc[association_result.association_available,['cancer_id','task','lncrna_id','pathway_id']].drop_duplicates().shape[0]) if len(association_result) else 0,
        'association_used_by_primary_config':cfg.get('singlecell_association_role')=='input_context','association_exact_pathway_join':True,
        'association_family_broadcast':False,'p_value_used_as_fdr':False,
        'missing_donor_data':'NO_CORRELATION_COMPUTED' if donor_path is None else None,'independent_replication_verified':False,
        'zero_fill_applied':False,'cell_origin_percentages_inferred':False,'changes_primary_training':False})
