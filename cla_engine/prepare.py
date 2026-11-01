"""149 CPU preparation: raw sources -> patient-disjoint inputs/labels -> matched LASSO -> packs.

Diagnostic is an explicitly smaller candidate scope. Formal uses the same fixed
sampler and all declared cancers; no outcome-dependent candidate selection.
"""
from __future__ import annotations
import hashlib,json,math,warnings,inspect
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from .common import atomic_json,atomic_npz,commit_manifest,read_json,digest,sha,fail,status,verify_manifest,lock
from .tables import table,partition,patient_map,attach_patient,matrix,cnv,state_wide,require,strict_bool
from .statistics import design,pairs,labels,raw_features
from .graph import nodes_and_static,coexpression,graph_npz
from .identity import detection_eligibility,frame_ids,unique_values,coverage
from .graph_contracts import membership_join
from .lineage import candidate_hash,prepare_lineage,validate_prepared


def deterministic_pairs(lncs,targets,budget,seed,cancer):
    # At least one target per lncRNA; no labels determine candidate admission.
    if not lncs or not targets:fail('EMPTY_CANDIDATES',cancer)
    n=len(lncs)*len(targets);budget=min(max(int(budget),len(lncs)),n)
    offset=int(hashlib.sha256(f'{seed}|{cancer}'.encode()).hexdigest()[:16],16)%len(targets)
    stride=max(1,len(targets)//2)
    while math.gcd(stride,len(targets))!=1:stride+=1
    q=np.arange(budget,dtype=np.int64);li=q%len(lncs);ti=(q//len(lncs)*stride+li+offset)%len(targets)
    return li,ti

def feature_stats(A,ix):
    x=A[ix];finite=np.isfinite(x);cnt=finite.sum(0);mean=np.divide(np.nansum(x,0),cnt,out=np.zeros(x.shape[1]),where=cnt>0)
    variance=np.divide(np.nansum((x-mean)**2,0),cnt,out=np.zeros(x.shape[1]),where=cnt>0)
    return np.column_stack([mean,np.sqrt(variance),cnt/max(len(ix),1)]).astype(np.float32)

def load_singlecell(c,lncs,cfg,cell_types=None):
    """Compatibility input reader; R6 primary defaults to interpretation_only.
    Explicit per-cell-type masks; eligibility precedes vocabulary construction.
    """
    from .singlecell_interpretation import eligible_manifest,expression_table
    root=Path(cfg['imports_root'])/'singlecell';mp=root/'dataset_manifest_33c.parquet';p=root/'lnc_celltype'/f'{c}.parquet'
    if cfg.get('synthetic_fixture'):
        if not mp.exists():mp=mp.with_suffix('.tsv')
        if not p.exists():p=p.with_suffix('.tsv')
    if not mp.exists() or not p.exists():
        width=4*(len(cell_types) if cell_types else int(cfg['model']['sc_buckets']))
        return np.zeros((len(lncs),width),np.float32),np.zeros(len(lncs),np.float32),{'status':'MATERIAL_NOT_BOUND'}
    allowed,_=eligible_manifest(table(mp));f,_=expression_table(table(p),allowed)
    if cell_types is None:cell_types=sorted(set(f.cell_type))
    feat=np.zeros((len(lncs),len(cell_types)*4),np.float32);available=np.zeros(len(lncs),np.float32)
    li={l:i for i,l in enumerate(lncs)};ci={c:i for i,c in enumerate(cell_types)}
    # No state-mixed feature join. Non-stratified measurements only in this legacy context reader.
    f=f.loc[f.cell_state.eq('__UNSTRATIFIED__')&f.expression_available&f.n_donors.ge(2)]
    for (l,ct),g in f.groupby(['lncrna_id','cell_type'],sort=True):
        if l not in li or ct not in ci:continue
        j=ci[ct]*4;feat[li[l],j:j+4]=[g.mean_log_expression.mean(),g.detection_rate.mean(),np.log1p(g.n_donors.mean()),1.]
        available[li[l]]=1
    return feat,available,{'status':'LEGACY_CONTEXT_INPUT_NOT_INDEPENDENT_VALIDATION','cell_types':cell_types,'per_context_mask':True,'source_sha256':sha(p),'manifest_sha256':sha(mp)}

def load_singlecell_association(c,rows,cfg,cell_types):
    """Bind exact external lncRNA/pathway correlations as candidate context.

    The join is cancer + lncRNA + exact pathway + declared cell type.  An
    ineligible cancer is a structural missing observation (all availability
    masks stay zero); an eligible cancer with a missing table is a hard error.
    """
    from .singlecell_interpretation import association_candidate_features
    from cc_hhgt.v32.single_cell_training import SingleCellTrainingError,normalise_dataset_manifest
    root=Path(cfg['imports_root'])/'singlecell';mp=root/'dataset_manifest_33c.parquet'
    if cfg.get('synthetic_fixture') and not mp.exists():mp=mp.with_suffix('.tsv')
    width=4*len(cell_types);empty=np.zeros((len(rows),width),np.float32)
    if not mp.exists():fail('SC_ASSOCIATION_MANIFEST_MISSING','Association input context requires the audited dataset manifest',path=str(mp))
    manifest=table(mp)
    try:strict_manifest=normalise_dataset_manifest(frame_ids(manifest))
    except SingleCellTrainingError as exc:fail('SC_ASSOCIATION_CONTRACT_INVALID',str(exc))
    qualified=sorted(strict_manifest.loc[strict_manifest.qualified&strict_manifest.cancer_id.eq(c),'dataset_id'].astype(str))
    common={'feature_order':[f'{cell}:{feature}' for cell in cell_types for feature in ('rho','abs_rho','neglog10_fdr','available')],
        'candidate_rows':len(rows),'candidate_rows_with_association':0,'qualified_datasets':qualified,
        'exact_pathway_join':True,'family_broadcast':False,'p_value_used_as_fdr':False,
        'association_is_primary_input_not_independent_validation':True,'manifest_path':str(mp),'manifest_sha256':sha(mp)}
    if not qualified:
        return empty,{**common,'status':'EXPLICITLY_UNAVAILABLE_FOR_CANCER','per_context_exact_rows':{cell:0 for cell in cell_types}}
    p=root/'association'/f'{c}.parquet'
    if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
    if not p.exists():fail('SC_ASSOCIATION_INPUT_MISSING','Qualified cancer requires an exact rho/FDR association table',cancer=c,path=str(p))
    features,report,_=association_candidate_features(table(p),manifest,rows,cell_types,c)
    return features,{**report,'source_path':str(p),'source_sha256':sha(p),'manifest_path':str(mp),'manifest_sha256':sha(mp)}

def load_mutation(c,lncs,source_patients,cfg):
    """Retain R5's three rate features, but share strict R6 call normalization.
    Observed positives without a locus denominator remain in the audit, not rates.
    """
    from .genomics import normalize_mutation,read_cancer_source
    p=Path(cfg['imports_root'])/'genomic'/'sample_lncrna_mutation_mc3_assay_callable.parquet'
    out=np.zeros((len(lncs),3),np.float32);available=np.zeros(len(lncs),np.float32)
    if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
    if 'mutation' in cfg.get('disabled_modalities',[]):return out,available,{'status':'DISABLED_BY_FROZEN_CONFIG'}
    if not p.exists():fail('LNC_MUTATION_INPUT_MISSING','Raw measured input required',path=str(p))
    calls=normalize_mutation(read_cancer_source(p,c),source_patients,lncs,'lncRNA',c)
    nc=calls.callable.sum(0);minimum=int(cfg.get('mutation_minimum_callable_patients',cfg.get('minimum_pairs',2)))
    valid=nc>=minimum;available[valid]=1.
    out[:,0]=np.divide(((calls.events==1)&calls.callable).sum(0),nc,out=np.zeros(len(lncs)),where=valid)
    out[:,1]=np.divide(np.where(calls.callable,calls.counts,0).sum(0),nc,out=np.zeros(len(lncs)),where=valid)
    out[:,2]=np.where(valid,nc/max(len(source_patients),1),0)
    return out,available,{'status':'EXPLICIT_LOCUS_RATES_ONLY','lnc_available':int(valid.sum()),'minimum_callable_patients':minimum,
        'source_sha256':sha(p),'wildcard_expansion':False,'sentinel_rows':calls.report['assay_sentinel_rows'],
        'observed_positive_records':int((calls.events==1).sum()),'explicit_locus_rows':int(calls.callable.sum()),
        'records_without_locus_denominator':int(((calls.events==1)&~calls.callable).sum()),**calls.report}

from .exact_baseline import (identity_records, design_block as exact_design,
    resource_plan,fit_vocabulary,training_design,decision_batched,unseen_audit)


def macro_ap(y,score,cancer,available,groups=None,target_ids=None):
    values=[]
    for c in sorted(set(cancer)):
        take=(cancer==c)&available
        if groups is None:
            if np.unique(y[take]).size==2:values.append(average_precision_score(y[take],score[take]))
        else:
            parts=[]
            for g in sorted(set(groups[take])):
                group_mask=take&(groups==g)
                if target_ids is None:
                    unit_values=[average_precision_score(y[group_mask],score[group_mask])] if np.unique(y[group_mask]).size==2 else []
                else:
                    unit_values=[average_precision_score(y[group_mask&(target_ids==t)],score[group_mask&(target_ids==t)]) for t in sorted(set(target_ids[group_mask])) if np.unique(y[group_mask&(target_ids==t)]).size==2]
                if unit_values:parts.append(float(np.mean(unit_values)))
            if parts:values.append(float(np.mean(parts)))
    if not values:fail('VALIDATION_NOT_ESTIMABLE','No two-class cancer in task')
    return float(np.mean(values))

def prepare(cfg,out,fold=0,profile='diagnostic'):
    from .execution_contract import validate_config_contract,baseline_contract,ENGINE_ID
    from .controller import assert_controller
    from .inputs import verify_import_authority
    from .cancer_scope import selection,resolve_cancer_scope
    selection(cfg,profile)
    validate_config_contract(cfg);assert_controller(cfg)
    imported=verify_import_authority(cfg)
    out=Path(out);sources=cfg['sources'];targets=cfg['targets'];model=cfg['model'];scope=cfg['profiles'][profile]
    if out.exists() and (out/'MANIFEST.json').exists():
        old,_,_,_=validate_prepared(out)
        if old['metadata']['config_sha256']!=digest(cfg) or old['metadata']['fold']!=fold or old['metadata']['profile']!=profile:fail('PREPARED_DRIFT','Existing committed package has another contract')
        status(out,'PREPARED_REUSED_VERIFIED');return out
    if out.exists() and any(p.is_file() and p.name not in {'STATUS.json','.lock'} for p in out.rglob('*')):
        fail('UNCOMMITTED_PREPARE_EXISTS','Incomplete CPU materialization is isolated; use a new output directory, never merge old files')
    out.mkdir(parents=True,exist_ok=True);status(out,'CPU_PREPARING',fold=fold,profile=profile)
    # Each input hash is recorded; pinned sources additionally must match prior receipts.
    input_hashes={k:v for k,v in imported['bound_files'].items()}
    for role,spec in sources.items():
        if not isinstance(spec,dict) or 'path' not in spec:continue
        p=Path(spec['path'])
        if p.is_file():
            h=sha(p);input_hashes[role]={'path':str(p),'sha256':h}
            if spec.get('sha256') and h!=spec['sha256']:fail('INPUT_DRIFT','Pinned input differs',role=role,path=str(p))
    fm=patient_map(sources['patient_map']['path'])
    cancer_scope=resolve_cancer_scope(cfg,fm.cancer_id.unique(),profile)
    cancers=cancer_scope['selected_cancers']
    atomic_json(out/'CANCER_SCOPE.json',cancer_scope)
    fm=fm.loc[fm.cancer_id.isin(cancers)].copy()
    valfold=(fold+1)%5;trainfolds=[x for x in range(5) if x not in [fold,valfold]]
    state,p=state_wide(sources['state'],fm,targets);input_hashes['state']={'path':str(p),'sha256':sha(p)}
    if sources['state'].get('format')=='CLA_PATIENT_STATE_V1':
        sp=Path(sources['state']['materialization_manifest'])
        atomic_json(out/'STATE_MATERIALIZATION.json',read_json(sp))
        input_hashes['state_materialization']={'path':str(sp),'sha256':sha(sp)}
    rates=table(sources['detection']['path']);require(rates,['cancer_id','lncrna_id','detection_rate_logcpm_gt0'],'detection')
    rates,eligible,detection_report=detection_eligibility(rates,cfg['candidate_scope']['minimum_expression_cancers'])
    atomic_json(out/'DETECTION_AUDIT.json',detection_report)
    # This fixed expression-only universe is explicitly transductive, not strict LOCO.
    per_lnc={c:sorted(set(rates.loc[rates.cancer_id.eq(c)&rates.eligible,'lncrna_id'].astype(str))&eligible) for c in cancers}
    membership=table(sources['exact_membership']['path']);membership_report={'current_rows':len(membership)}
    if 'weighted_membership' in sources:
        weighted=table(sources['weighted_membership']['path'])
        membership,membership_report=membership_join(membership,weighted)
    atomic_json(out/'MEMBERSHIP_ALIGNMENT.json',membership_report)
    hier=table(sources['hierarchy']['path']);binding=table(sources['binding']['path']);pg=table(sources['protein_gene']['path']);ppi=table(sources['ppi']['path']);cov=table(sources['covariates']['path'])
    pathways=sorted(set(membership.pathway_id.astype(str)))
    if scope.get('max_pathways'):pathways=sorted(pathways,key=lambda v:hashlib.sha256(v.encode()).hexdigest())[:scope['max_pathways']];pathways.sort()
    for c in cancers:
        if scope.get('max_lncrnas'):per_lnc[c]=sorted(per_lnc[c],key=lambda v:hashlib.sha256(v.encode()).hexdigest())[:scope['max_lncrnas']];per_lnc[c].sort()
        if not per_lnc[c]:fail('EMPTY_LNCRNA_SCOPE','No eligible lncRNA',cancer=c)
    lncs=sorted(set().union(*map(set,per_lnc.values())))
    # Target cap restricts target/member/family nodes. PPI context stays declared and its full size is reported.
    membership=membership.loc[membership.pathway_id.isin(pathways)].copy()
    hier=hier.loc[hier.pathway_id.isin(pathways)].copy()
    alignment=[];cohort_catalog={};candidate_catalog={};patient_assignments={}
    primary=[x for x in targets if x['role']=='primary'];stateids=[x['id'] for x in primary];family={x['id']:x['family'] for x in primary}
    ids,maps,rels,static,graph_audit=nodes_and_static(membership,hier,binding,pg,ppi,lncs,pathways)
    ids['cancer']=list(cancers);maps['cancer']={c:i for i,c in enumerate(cancers)}
    for nt in ('lncRNA','gene'):
        rels.extend([(nt,'measured_in','cancer'),('cancer','rev_measured_in',nt)])
    variant=cfg['graph'].get('variant','G2')
    if variant not in ['G0','G1','G2']:fail('GRAPH_VARIANT_INVALID',variant)
    removed=[k for k in static if (variant=='G0' and 'reported_binding' in k) or (variant!='G2' and 'reported_PPI' in k)]
    for k in removed:static.pop(k)
    graph_audit.append({'variant':variant,'removed_relation_keys':removed,'node_identity_universe_shared_across_variants':True})
    cell_types=set()
    interpretation_only=cfg.get('singlecell_role','interpretation_only')=='interpretation_only'
    for cancer in ([] if interpretation_only else cancers):
        p=Path(cfg['imports_root'])/'singlecell'/'lnc_celltype'/f'{cancer}.parquet'
        if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
        if p.exists():
            from .singlecell_interpretation import eligible_manifest,expression_table
            mpath=Path(cfg['imports_root'])/'singlecell/dataset_manifest_33c.parquet'
            if cfg.get('synthetic_fixture') and not mpath.exists():mpath=mpath.with_suffix('.tsv')
            allowed,_=eligible_manifest(table(mpath));f,_=expression_table(table(p),allowed);cell_types.update(f.cell_type.dropna().astype(str));input_hashes['singlecell_'+cancer]={'path':str(p),'sha256':sha(p)}
    cell_types=sorted(cell_types) or ['__NO_CELLTYPE_MEASUREMENT__'];sc_dim=1 if interpretation_only else len(cell_types)*4
    sc_association_role=cfg.get('singlecell_association_role','disabled')
    if sc_association_role!='disabled':fail('SC_PRIMARY_INPUT_RETIRED','Use disabled primary SC association input and the independent cell-type fusion expert')
    sc_association_cell_types=list(map(str,cfg.get('singlecell_association_cell_types',[]))) if sc_association_role=='input_context' else []
    if sc_association_role=='input_context' and (not sc_association_cell_types or len(sc_association_cell_types)!=len(set(sc_association_cell_types))):
        fail('SC_ASSOCIATION_CONTEXT_CONFIG','Input context requires a nonempty unique cell-type list')
    sc_association_dim=4*len(sc_association_cell_types)
    gene_node_omics_cfg=cfg.get('gene_node_omics',{'enabled':False})
    if not isinstance(gene_node_omics_cfg,dict) or type(gene_node_omics_cfg.get('enabled',False)) is not bool:
        fail('GENE_NODE_OMICS_CONFIG_INVALID','gene_node_omics.enabled must be an explicit boolean')
    gene_node_omics_enabled=gene_node_omics_cfg.get('enabled',False)
    gene_node_omics=None
    if gene_node_omics_enabled:
        from .node_omics import open_verified_gene_node_omics
        gene_node_omics=open_verified_gene_node_omics(gene_node_omics_cfg,ids['gene'],cancers)
        input_hashes.update(gene_node_omics.input_hashes())
    from .genomics import FEATURES,load_gene_calls,project_gene_context,build_genomic_projection
    genetic_enabled=bool(cfg.get('genomics',{}).get('enabled',False))
    feature_dims={t:6 for t in ids}
    if genetic_enabled:
        feature_dims.update(lncRNA=6+len(FEATURES)+2,gene=6+len(FEATURES),protein=6+len(FEATURES)+2,pathway=6+len(FEATURES)+2+3)
    target_genetic_width=(len(FEATURES)+2+3) if genetic_enabled else 0
    sc_association_slice=[5,5+sc_association_dim] if sc_association_dim else None
    node_input_feature_order=['log1p_degree','constant']
    if gene_node_omics_enabled:
        from .node_omics import FEATURE_ORDER
        node_input_feature_order.extend(FEATURE_ORDER)
    spec={'architecture':'CC_HHGT_PYG_FULL_R62_JOINT','nodes':ids,'relations':rels,'cancers':cancers,'states':stateids,'state_families':family,'feature_dims':feature_dims,'sc_dim':sc_dim,'sc_cell_types':cell_types,'singlecell_role':'interpretation_only' if interpretation_only else 'input_context','singlecell_association_role':sc_association_role,'singlecell_association_cell_types':sc_association_cell_types,'singlecell_association_feature_order':[f'{cell}:{feature}' for cell in sc_association_cell_types for feature in ('rho','abs_rho','neglog10_fdr','available')],'singlecell_association_dim':sc_association_dim,'singlecell_association_slice':sc_association_slice,'genomics_enabled':genetic_enabled,'gene_node_omics_enabled':gene_node_omics_enabled,'node_input_dim':len(node_input_feature_order),'node_input_feature_order':node_input_feature_order,'numeric_features':5+sc_association_dim+feature_dims['lncRNA']+sc_dim+1+3+1+target_genetic_width,'fold':fold,'validation_fold':valfold,'train_folds':trainfolds}
    if gene_node_omics_enabled:
        spec['gene_node_omics_complete_sha256']=gene_node_omics.complete_record['sha256']
    spec['state_catalog']=cfg.get('state_catalog')
    spec['graph_variant']=variant;spec['backbone_source']='cc_hhgt/gnn.py::build_model(cc_hhgt)';spec['full_r1_checkpoint_compatible']=False
    spec['execution_contract']=ENGINE_ID;spec['precision_k']=cfg['gates']['precision_k'];spec['metrics_contract']=cfg['metrics_contract']
    atomic_json(out/'ADJUSTMENT_CONTRACT.json',cfg['adjustment_contract'])
    atomic_json(out/'BASELINE_CONTRACT.json',baseline_contract(cfg))
    atomic_json(out/'IMPORT_AUTHORITY_BINDING.json',imported)
    atomic_json(out/'SPEC.json',spec);atomic_json(out/'CONFIG.json',cfg);atomic_json(out/'GRAPH_SOURCE_AUDIT.json',graph_audit)
    records=[];sealed=[];cov_audit=[];coverage=[];rows_by_task={t:{'train':[],'validation':[]} for t in ['pathway','state']};views=[]
    for cancer in cancers:
        ci=cancers.index(cancer);patients=sorted(fm.loc[fm.cancer_id.eq(cancer),'patient_id'].unique());mp=fm.loc[fm.cancer_id.eq(cancer)].copy();pf=mp[['patient_id','patient_fold_id']].drop_duplicates().set_index('patient_id').reindex(patients).patient_fold_id.to_numpy(int)
        lframe,lpath=partition(sources['lnc_expression_root']['path'],cancer);pframe,ppath=partition(sources['pathway_activity_root']['path'],cancer);gframe,gpath=partition(sources['gene_expression_root']['path'],cancer)
        policy=cfg.get('alignment',{})
        X=matrix(lframe,'lncrna_id','logcpm',patients,lncs,mp,required_entities=per_lnc[cancer],report=alignment)
        # The activity universe only exists where the source actually scored a
        # pathway; unmeasured pathways remain in the global pool as unavailable
        # candidates (explicit per-view n_usable vs n_candidates audit), never
        # silently dropped or invented.
        pw_measured=set(pframe.pathway_id.astype(str))
        pw=[p for p in pathways if p in pw_measured]
        coverage.append({'cancer':cancer,'pathway_activity_scope':{'universe':len(pathways),'measured':len(pw),
            'unmeasured':sorted(set(pathways)-pw_measured),'policy':'MEASURED_REQUIRED_FOR_ESTIMATION; UNMEASURED_STAY_UNAVAILABLE_IN_POOL'}})
        Yp=matrix(pframe,'pathway_id','activity_score',patients,pathways,mp,required_entities=pw,report=alignment)
        # The coding-gene expression universe does not cover every graph gene
        # (membership/protein crosswalk genes can be unmeasured in bulk RNA).
        # Unmeasured genes stay as zero-feature static nodes (no coexpression
        # edges; feature_stats zero-fills) and are audited, never silently dropped.
        gm_measured=set(gframe.gene_id.astype(str))
        gm=[g for g in ids['gene'] if g in gm_measured]
        coverage.append({'cancer':cancer,'gene_expression_scope':{'universe':len(ids['gene']),'measured':len(gm),
            'unmeasured':sorted(set(ids['gene'])-gm_measured),'policy':'MEASURED_REQUIRED_FOR_FEATURES; UNMEASURED_VERTICES_ZERO_FEATURED_STATIC_ONLY'}})
        G=matrix(gframe,'gene_id','logcpm',patients,ids['gene'],mp,required_entities=gm,minimum_entity_coverage=policy.get('gene_identity_minimum',.95),report=alignment)
        Ys=matrix(state.loc[state.cancer_id.eq(cancer)],'target_id','value',patients,stateids,mp)
        C,call,cnvfiles=cnv(sources['cnv_root']['path'],cancer,patients,lncs,required_lncs=per_lnc[cancer],report=alignment)
        cc=attach_patient(cov.loc[cov.cancer_id.eq(cancer)],mp)
        if interpretation_only:
            sc=np.zeros((len(lncs),1),np.float32);scok=np.zeros(len(lncs),np.float32)
            sc_report={'status':'INTERPRETATION_ONLY_NOT_READ_OR_USED_BY_PRIMARY','no_external_validation_claim':True}
        else:
            sc,scok,sc_report=load_singlecell(cancer,lncs,cfg,cell_types)
        coverage.append({'cancer':cancer,'singlecell':sc_report})
        gene_calls=None;P=None;pcall=None
        if genetic_enabled:
            gene_calls=load_gene_calls(cancer,patients,ids['gene'],cfg)
            gp=gene_calls.report['source_path'];input_hashes[gp]={'path':gp,'sha256':gene_calls.report['source_sha256']}
            from .cnv_sensitivity import pathway_cnv
            P,pcall,p_report=pathway_cnv(sources['cnv_root']['path'],cancer,patients,pathways)
            for fp,h in p_report['files'].items():input_hashes[fp]={'path':fp,'sha256':h}
            coverage.append({'cancer':cancer,'coding_gene_mutation':gene_calls.report,'pathway_cnv':p_report})
        if sc_report.get('source_sha256'):
            sp=Path(cfg['imports_root'])/'singlecell'/'lnc_celltype'/f'{cancer}.parquet'
            if cfg.get('synthetic_fixture') and not sp.exists():sp=sp.with_suffix('.tsv')
            input_hashes[str(sp)]={'path':str(sp),'sha256':sc_report['source_sha256']}
        clncs=per_lnc[cancer];ll,tt=deterministic_pairs(clncs,pathways,scope['pathway_candidates_per_cancer'],cfg['seed'],cancer)
        lp=np.asarray([maps['lncRNA'][clncs[i]] for i in ll],np.int64)
        ls=np.repeat([maps['lncRNA'][l] for l in clncs],len(stateids)).astype(np.int64);ts=np.tile(np.arange(len(stateids)),len(clncs))
        candidates={'pathway':(lp,tt,pathways,Yp),'state':(ls,ts,stateids,Ys)}
        pathway_rows=[(cancer,ids['lncRNA'][int(l)],pathways[int(t)]) for l,t in zip(lp,tt)]
        if sc_association_role=='input_context':
            sc_association,sc_association_report=load_singlecell_association(cancer,pathway_rows,cfg,sc_association_cell_types)
        else:
            sc_association=np.zeros((len(lp),0),np.float32);sc_association_report={'status':'DISABLED_BY_FROZEN_CONFIG','candidate_rows':len(lp)}
        state_sc_association=np.zeros((len(ls),sc_association_dim),np.float32)
        coverage.append({'cancer':cancer,'singlecell_association':sc_association_report})
        for key in ['source','manifest']:
            path=sc_association_report.get(key+'_path');checksum=sc_association_report.get(key+'_sha256')
            if path and checksum:input_hashes[path]={'path':path,'sha256':checksum}
        patient_assignments[cancer]=[[p,int(f)] for p,f in zip(patients,pf)]
        cohort_catalog[cancer]={'all_patients':digest(patients),'folds':{str(f):digest([patients[i] for i in np.flatnonzero(pf==f)]) for f in range(5)}}
        candidate_catalog[cancer]={t:{'ordered_keys_sha256':candidate_hash(li,ti,lncs,tids),'rows':len(li)} for t,(li,ti,tids,_) in candidates.items()}
        for path in [lpath,ppath,gpath,*cnvfiles]:
            files=sorted(path.glob('*.parquet')) if path.is_dir() else [path]
            for f in files:input_hashes[str(f)]={'path':str(f),'sha256':sha(f)}
        # A topology-only comparison fixes the measured numeric background in
        # all graph variants. Historical configs retain their original behavior.
        context_variant=cfg.get('genomics',{}).get('context_graph_variant',variant)
        if context_variant not in ('G0','G1','G2'):
            fail('GENOMIC_CONTEXT_VARIANT_INVALID',str(context_variant))
        projector=build_genomic_projection(ids['gene'],pg,ids['protein'],membership,pathways,binding,lncs,cancer,context_variant) if genetic_enabled else None
        views_here=[(f'train{j}',[k for k in trainfolds if k!=j],[j]) for j in trainfolds]+[('validation',trainfolds,[valfold])]
        for view,sourcefolds,labelfolds in views_here:
            source_ix=np.flatnonzero(np.isin(pf,sourcefolds));label_ix=np.flatnonzero(np.isin(pf,labelfolds));source_pat=[patients[i] for i in source_ix]
            if np.intersect1d(source_ix,label_ix).size:fail('PATIENT_LEAK','Feature/label patients overlap')
            vid=f'{cancer}__{view}';viewdir=out/'views'/vid;viewdir.mkdir(parents=True,exist_ok=True)
            cohort_roles={'source':source_pat,'label':[patients[i] for i in label_ix]}
            Z,report=design(cc,patients,source_pat,family='pathway',target_id='__pathway__',cfg=cfg,admission_cohorts=cohort_roles);cov_audit.append({'view':vid,'role':'graph',**report})
            dyn,dynamic_report=coexpression(X,G,C,call,Z,source_ix,topk=cfg['graph']['coexpression_topk'],min_n=cfg['minimum_pairs'],cnv_transform=cfg.get('statistics',{}).get('cnv_transform','raw_signed'),return_audit=True)
            support=dynamic_report.pop('edge_support')
            dynamic_report['source_patient_hash']=digest(source_pat)
            atomic_json(viewdir/'COEXPRESSION_AUDIT.json',dynamic_report)
            atomic_npz(viewdir/'coexpression_support.npz',**{f'{sign}_{term}':values for sign,items in support.items() for term,values in items.items()})
            if dynamic_report['estimable_pairs']==0:
                fail('COEXPRESSION_NO_ESTIMABLE_PAIRS','No lncRNA-gene pair meets the fixed common-mask/rank requirements; no imputation or forced empty dynamic graph',view=vid,audit=dynamic_report)
            features={t:np.zeros((len(vs),feature_dims[t]),np.float32) for t,vs in ids.items()}
            features['lncRNA'][:,:3]=feature_stats(X,source_ix);features['lncRNA'][:,3:6]=feature_stats(C,source_ix);features['gene'][:,:3]=feature_stats(G,source_ix)
            genetics_report={'enabled':False}
            if genetic_enabled:
                gene_features,genetics_report=gene_calls.summaries(source_ix,minimum=cfg['minimum_pairs'])
                blocks=project_gene_context(gene_features,ids['gene'],pg,ids['protein'],membership,pathways,binding,lncs,cancer,context_variant,projection=projector)
                features['gene'][:,6:]=blocks['gene']
                features['protein'][:,6:]=blocks['protein']
                features['pathway'][:,6:6+len(FEATURES)+2]=blocks['pathway']
                features['pathway'][:,-3:]=feature_stats(P,source_ix)
                features['lncRNA'][:,6:]=blocks['partner']
                genetics_report={**genetics_report,**blocks['audit'],'enabled':True,'source_patients_only':True,'graph_variant':variant,'numeric_context_variant':context_variant,'topology_only_comparison':context_variant!=variant}
                atomic_json(viewdir/'GENOMIC_CONTEXT.json',genetics_report)
            # The original PyG input includes fold-view degree; no trainable node IDs are added.
            mut,mutok,mut_report=load_mutation(cancer,lncs,source_pat,cfg)
            graph=graph_npz(ids,static,cancer,dyn,features);graph.update(singlecell=sc,sc_available=scok,mutation=mut,mutation_available=mutok)
            node_omics_report={'enabled':False}
            if gene_node_omics_enabled:
                from .node_omics import graph_node_omics
                gene_block,node_omics_report=gene_node_omics.build_view(cancer,source_pat)
                for node_type,block in graph_node_omics(ids,gene_block).items():
                    graph['node_omics::'+node_type]=block
                atomic_json(viewdir/'NODE_OMICS_AUDIT.json',node_omics_report)
                coverage.append({'view':vid,'gene_node_omics':node_omics_report})
            atomic_npz(viewdir/'graph.npz',**graph)
            provenance={'feature_patient_hash':digest(source_pat),'label_patient_hash':digest([patients[i] for i in label_ix]),'feature_folds':sourcefolds,'label_folds':labelfolds,'outer_test_excluded':fold not in sourcefolds+labelfolds,'cancer':cancer,'mutation':mut_report,'gene_protein_pathway_genomics':genetics_report,'gene_node_omics':node_omics_report}
            atomic_json(viewdir/'PROVENANCE.json',provenance)
            for task,(li,ti,tids,Y) in candidates.items():
                # Distinct TME targets use a composition-preserving covariate contract.
                dstat={k:np.zeros(len(li),dtype=bool if k=='available' else (np.int32 if k in ['n','rank'] else float)) for k in ['r','p','q','n','rank','available']};lstat={k:v.copy() for k,v in dstat.items()}
                if task=='state':families={s:family[s] for s in tids}
                else:families={s:'pathway' for s in tids}
                groups_to_fit=[('pathway','__pathway__',np.ones(len(ti),bool))] if task=='pathway' else [(families[t],t,ti==j) for j,t in enumerate(tids)]
                fitted_designs={}
                for f,target_id,take in groups_to_fit:
                    z,rep=design(cc,patients,source_pat,family=f,target_id=target_id,cfg=cfg,admission_cohorts=cohort_roles)
                    fitted_designs[target_id]=z
                    cov_audit.append({'view':vid,'task':task,**rep})
                    ds=pairs(X,Y,C,call,z,li[take],ti[take],source_ix,cfg['minimum_pairs'],cnv_transform=cfg.get('statistics',{}).get('cnv_transform','raw_signed'));ys=pairs(X,Y,C,call,z,li[take],ti[take],label_ix,cfg['minimum_pairs'],cnv_transform=cfg.get('statistics',{}).get('cnv_transform','raw_signed'))
                    for k in ds:dstat[k][take]=ds[k];lstat[k][take]=ys[k]
                # Recompute BH over the complete tested family, not one computational chunk.
                from .statistics import bh
                dstat['q']=bh(dstat['p']);lstat['q']=bh(lstat['p']);label,weak=labels(lstat,task)
                ok=dstat['available']&lstat['available']
                association_context=sc_association if task=='pathway' else state_sc_association
                pieces=[raw_features(dstat),association_context,features['lncRNA'][li],sc[li],scok[li,None],mut[li],mutok[li,None]]
                if genetic_enabled:
                    pieces.append(features['pathway'][ti,6:] if task=='pathway' else np.zeros((len(li),target_genetic_width),np.float32))
                num=np.column_stack(pieces).astype(np.float32)
                groups=np.asarray([families[tids[i]] for i in ti]);weight=np.ones(len(li),np.float32)
                if task=='state':
                    for f in np.unique(groups):weight[groups==f]=1/max((ok&(groups==f)).sum(),1)
                weight=np.where(weak,weight*.5,weight)
                arrays={'lnc':li,'target':ti,'numeric':num,'base':np.zeros(len(li),np.float32),'label':label,'available':ok,'discovery_available':dstat['available'],'weight':weight,'direction_label':(lstat['r']>=0).astype(np.float32),'direction_available':ok&(np.abs(lstat['r'])>=.15),'discovery_effect':dstat['r'],'discovery_q':dstat['q']}
                from .readiness import validate_task_arrays
                # A single cancer/fold view may have no estimable pairs. Global
                # trainability is checked after cross-cancer assembly below.
                validate_task_arrays(arrays,vid,task,spec['numeric_features'],allow_empty=True)
                atomic_npz(viewdir/f'{task}.npz',**arrays)
                split='validation' if view=='validation' else 'train'
                rows_by_task[task][split].append((vid,cancer,li,ti,num,label,ok))
                coverage.append({'view':vid,'task':task,'n_candidates':len(li),'n_usable':int(ok.sum()),'n_positive':int(label[ok].sum())})
                if view=='validation':
                    # Test labels stay outside training capsule, no eval read on GPU.
                    teststat={k:v.copy() for k,v in lstat.items()}
                    for f,target_id,take in groups_to_fit:
                        z=fitted_designs[target_id]
                        st=pairs(X,Y,C,call,z,li[take],ti[take],np.flatnonzero(pf==fold),cfg['minimum_pairs'],cnv_transform=cfg.get('statistics',{}).get('cnv_transform','raw_signed'))
                        for k in st:teststat[k][take]=st[k]
                    teststat['q']=bh(teststat['p']);yl,_=labels(teststat,task)
                    sealed.append((vid,task,{'lnc':li.copy(),'target':ti.copy(),'outer_patient_hash':np.asarray(cohort_catalog[cancer]['folds'][str(fold)]),'label':yl,'available':dstat['available']&teststat['available'],'effect':teststat['r'],'p':teststat['p'],'q':teststat['q'],'n':teststat['n'],'rank':teststat['rank']}))
            views.append({'id':vid,'cancer_index':ci,'split':'validation' if view=='validation' else 'train'})
            status(out,'CPU_PREPARING',completed_view=vid,fold=fold,profile=profile)
    # Matched LASSO: no class-balanced validation and no training-loss fallback.
    baseline_receipts={}
    for task,splits in rows_by_task.items():
        bc=baseline_contract(cfg)
        pre_numeric_dim=splits['train'][0][4].shape[1]
        pre_resource=resource_plan(sum(int(i[6].sum()) for i in splits['train']),pre_numeric_dim,sum(len(i[2]) for i in splits['validation']),bc)
        # Check before allocating concatenated arrays and Python identity rows.
        pre_resource['candidate_identity_rows']=sum(len(i[2]) for split_items in splits.values() for i in split_items)
        assembled={}
        for split,items in splits.items():
            nums=np.concatenate([i[4] for i in items]);y=np.concatenate([i[5] for i in items]);ok=np.concatenate([i[6] for i in items]);cancer=np.concatenate([np.repeat(i[1],len(i[2])) for i in items])
            ident=[(c,ids['lncRNA'][int(l)],(pathways if task=='pathway' else stateids)[int(t)]) for _,c,li,ti,*_ in items for l,t in zip(li,ti)]
            assembled[split]=(nums,y,ok,cancer,ident)
        tr=assembled['train'];va=assembled['validation']
        if np.unique(tr[1][tr[2]]).size!=2 or np.unique(va[1][va[2]]).size!=2:fail('TASK_NOT_ESTIMABLE','Task has no two-class train/validation',task=task)
        bc=baseline_contract(cfg);batch_rows=bc['batch_rows']
        budget=resource_plan(int(tr[2].sum()),tr[0].shape[1],len(va[0]),bc)
        train_positions=np.flatnonzero(tr[2]);train_rows=[tr[4][i] for i in train_positions]
        train_numeric=tr[0][train_positions]
        scaler=StandardScaler().fit(train_numeric)
        vectorizer,vocabulary_report=fit_vocabulary(train_rows,bc)
        xt=training_design(train_rows,train_numeric,scaler,vectorizer,batch_rows)
        validation_identity_audit=unseen_audit(va[4],vectorizer)
        budget['pre_assembly_check']=pre_resource
        budget['actual_train_csr_nnz']=int(xt.nnz);budget['actual_train_csr_bytes']=int(xt.data.nbytes+xt.indices.nbytes+xt.indptr.nbytes)
        atomic_json(out/'baseline'/f'{task}_resources.json',budget)
        chosen=None;trials=[]
        for Cpen in cfg['baseline']['C']:
            l1args={'l1_ratio':1.0} if inspect.signature(LogisticRegression).parameters['penalty'].default=='deprecated' else {'penalty':'l1'}
            clf=LogisticRegression(C=Cpen,solver='liblinear',max_iter=cfg['baseline']['max_iter'],random_state=cfg['seed'],**l1args)
            try:
                weights=np.concatenate([np.load(out/'views'/item[0]/f'{task}.npz',allow_pickle=False)['weight'] for item in splits['train']])[tr[2]]
                weights=weights/max(float(weights.mean()),1e-12)
                with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);clf.fit(xt,tr[1][tr[2]],sample_weight=weights)
            except ConvergenceWarning:
                trials.append({'C':Cpen,'status':'NONCONVERGED_EXCLUDED'});continue
            eval_groups=np.asarray([family[i[2]] for i in va[4]]) if task=='state' else None
            score=macro_ap(va[1],decision_batched(clf,va[4],va[0],scaler,vectorizer,batch_rows),va[3],va[2],eval_groups,np.asarray([i[2] for i in va[4]]) if task=='state' else None);trials.append({'C':Cpen,'macro_cancer_auprc':score})
            if chosen is None or score>chosen[0]:chosen=(score,Cpen,clf)
        if chosen is None:fail('BASELINE_NONCONVERGENCE','No converged LASSO candidate',task=task)
        score,Cpen,clf=chosen;baseline_receipts[task]={'C':Cpen,'validation_macro_ap':score,'trials':trials,'fdr_feature_present':True,'scaler_fitted_on':'TRAIN_ONLY','input_semantics':'discovery statistics + exact cell-type lncRNA/pathway rho/FDR context when configured + bulk/local CNV + observed/callable coding-gene/partner/pathway summaries + sparse identities; node-level SC expression excluded in interpretation_only; graph topology exclusive to HGT','not_multimodal_linear_baseline':False,'same_numeric_multimodal_inputs_as_deep':True,'singlecell_association_role':sc_association_role,'singlecell_association_dim':sc_association_dim,'identity_encoding':baseline_contract(cfg)['identity_encoding'],'identity_feature_count':len(vectorizer.vocabulary_),'hash_collisions_possible':False,'vocabulary':vocabulary_report,'validation_identity_audit':validation_identity_audit,'resource_plan':budget}
        atomic_json(out/'baseline'/f'{task}_identity_vocabulary.json',{'encoding':'DictVectorizer','dimensions':len(vectorizer.vocabulary_),'feature_names':vectorizer.feature_names_,'fit_scope':'AVAILABLE_TRAIN_ROWS_ONLY','feature_names_sha256':vocabulary_report['feature_names_sha256']})
        atomic_npz(out/'baseline'/f'{task}.npz',coef=clf.coef_,intercept=clf.intercept_,scale=scaler.scale_,mean=scaler.mean_)
        for split,items in splits.items():
            assembled_split=assembled[split];offset=0
            logits=decision_batched(clf,assembled_split[4],assembled_split[0],scaler,vectorizer,batch_rows).astype(np.float32)
            for vid,c,li,ti,num,y,ok in items:
                p=out/'views'/vid/f'{task}.npz'
                with np.load(p,allow_pickle=False) as f:arr={k:f[k] for k in f.files}
                arr['base']=logits[offset:offset+len(li)];arr['numeric']=scaler.transform(num).astype(np.float32);offset+=len(li);atomic_npz(p,**arr)
    from .readiness import write_prepared_readiness
    write_prepared_readiness(out,spec,views,baseline_receipts,coverage)
    # Files in sealed/ are deliberately outside the pack root.
    sealedroot=out.parent/(out.name+'__sealed_test');sealedroot.mkdir(exist_ok=True)
    for vid,task,arr in sealed:atomic_npz(sealedroot/f'{vid}__{task}.npz',**arr)
    sealed_binding={'format':'CLA_R62_SEALED_BINDING_V1','fold':fold,'profile':profile,'files':{}}
    for vid,task,arr in sealed:
        cancer=vid.removesuffix('__validation');rel=f'{vid}__{task}.npz'
        sealed_binding['files'][rel]={'sha256':sha(sealedroot/rel),'task':task,'cancer_id':cancer,
            'candidate_sha256':candidate_catalog[cancer][task]['ordered_keys_sha256'],
            'outer_patient_hash':cohort_catalog[cancer]['folds'][str(fold)]}
    atomic_json(out/'SEALED_BINDING.json',sealed_binding)
    atomic_json(out/'PATIENT_ASSIGNMENTS.json',patient_assignments)
    atomic_json(out/'COHORTS.json',cohort_catalog);atomic_json(out/'CANDIDATES.json',candidate_catalog)
    atomic_json(out/'ALIGNMENT_AUDIT.json',alignment)
    atomic_json(out/'GRAPH_SIZE.json',{'node_counts':{k:len(v) for k,v in ids.items()},'static_edges':{k:int(v['edge_index'].shape[1]) for k,v in static.items()},
        'scope':'SELECTED_TARGET_MEMBERS_AND_FAMILIES_PLUS_DECLARED_PROTEIN_CODING_GENES_AND_PPI_CONTEXT','diagnostic_is_formal_memory_estimate':False})
    atomic_json(out/'VIEWS.json',views);atomic_json(out/'BASELINES.json',baseline_receipts);atomic_json(out/'COVARIATE_AUDIT.json',cov_audit);atomic_json(out/'COVERAGE.json',coverage);atomic_json(out/'INPUT_HASHES.json',input_hashes)
    atomic_json(out/'STATE_TARGET_ROLES.json',targets)
    for rel in (['genomic/sample_lncrna_mutation_mc3_assay_callable.parquet']+([] if interpretation_only and sc_association_role!='input_context' else ['singlecell/dataset_manifest_33c.parquet'])):
        p=Path(cfg['imports_root'])/rel
        if cfg.get('synthetic_fixture') and not p.exists():p=p.with_suffix('.tsv')
        if p.exists():input_hashes[rel]={'path':str(p),'sha256':sha(p)}
    atomic_json(out/'INPUT_HASHES.json',input_hashes)
    atomic_json(out/'TASK_CONTRACT.json',{'task_weights':{'pathway':1.,'state':.5},'local_cnv_adjustment':'LNC_ONLY_A1_WITH_ACTUAL_DESIGN_MATRIX_RANK','cnv_covariate_transform':cfg.get('statistics',{}).get('cnv_transform','raw_signed'),'not_r7_rank_cnv_a2_contract':True,'pathway_label':'abs(r)>=.20 & q<=.05 OR abs(r)>=.15 & q<=.10','state_label':'abs(r)>=.20','fdr_family':'CANCER_TASK_TESTED_CANDIDATES','population':'POOLED_KNOWN_CANCERS_PATIENT_HOLDOUT_NOT_LOCO','same_source_sc_validation_claimed':False,'sc_role':spec['singlecell_role'],'sc_expression_used_by_primary':not interpretation_only,'singlecell_association_role':sc_association_role,'singlecell_association_used_by_primary':sc_association_dim>0,'singlecell_association_semantics':'EXTERNAL_EXACT_CANCER_LNCRNA_PATHWAY_CELLTYPE_RHO_ABS_RHO_NEGLOG10_FDR_AVAILABILITY;NOT_INDEPENDENT_VALIDATION','singlecell_association_feature_order':spec['singlecell_association_feature_order'],'singlecell_association_slice':sc_association_slice,'genomics_role':'SOURCE_FOLD_GENE_PROTEIN_PARTNER_PATHWAY_BACKGROUND_NOT_LABEL','gene_node_omics_role':'SOURCE_PATIENT_FOLD_LOCAL_GENE_CNV_AND_PROMOTER_METHYLATION_GRAPH_INPUT' if gene_node_omics_enabled else 'DISABLED_HISTORICAL_TWO_COLUMN_NODE_INPUT','gene_node_omics_feature_order':node_input_feature_order,'gene_node_omics_missingness':'EXPLICIT_MASK_NO_IMPUTATION','cnv_a2_role':'SEPARATE_COMMON_MASK_SENSITIVITY_NOT_PRIMARY_TARGET','formal_input_contract_unchanged_from_r7':False,'adjustment_contract_sha256':digest(cfg['adjustment_contract']),'metrics_contract':cfg['metrics_contract'],'baseline_contract_sha256':digest(baseline_contract(cfg)),'coexpression_contract':'PAIRWISE_COMMON_MASK_COEXPRESSION_R62_V1','binding_partner_scope':'NORMALIZE_BINDING_GLOBAL_PLUS_EXACT_CANCER','direct_evidence_used_by_primary':False,'sc_used_by_primary':(not interpretation_only) or sc_association_dim>0})
    from .source_lock import record_sources
    atomic_json(out/'SOURCE_FREEZE_RECEIPT.json',record_sources(cfg,input_hashes))
    atomic_json(out/'LINEAGE.json',prepare_lineage(out))
    commit_manifest(out,{'config_sha256':digest(cfg),'fold':fold,'profile':profile,'tasks':['pathway','state'],'trainable_pack':True,'sealed_test_labels_included':False,'created_by':'CLA_FULL_PYG_R62_RAW_PREPARATION','synthetic_fixture':bool(cfg.get('synthetic_fixture',False))})
    commit_manifest(sealedroot,{'prepared_manifest_sha256':sha(out/'MANIFEST.json'),'binding_sha256':sha(out/'SEALED_BINDING.json'),'fold':fold,'profile':profile,'role':'OUTER_TEST_LABELS_NOT_IN_TRAINING_CAPSULE'})
    status(out,'PREPARED_NOT_TRAINED',fold=fold,profile=profile,sealed_labels=str(sealedroot));return out
