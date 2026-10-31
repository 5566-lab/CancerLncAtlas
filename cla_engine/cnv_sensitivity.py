"""CNV A1/A2 audit on explicit common patient masks; never a primary label writer.

Primary: existing raw-signed, lncRNA-side-only A1.
Sensitivity: ranked A1 on the same A2-eligible patients, then symmetric ranked A2
with both local-lncRNA and target-pathway CNV in the nuisance space.
A2 is not mutation adjustment, RNA causality, or claimed bitwise r7 equivalence.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,t as student_t
from .common import fail,atomic_json,sha,digest,commit_manifest
from .tables import table,partition,matrix,attach_patient,patient_map,cnv
from .identity import token,coverage
from .statistics import pairs,basis,bh,design


def pathway_cnv(root,cancer,patients,pathways,minimum_entity_coverage=.0):
    directory=Path(root)/f'cancer={cancer}'
    names=['patient_ids.json','pathway_ids.json','pathway_signed_mean.npy','pathway_callable.npy']
    for name in names:
        if not (directory/name).is_file():fail('PATHWAY_CNV_INPUT_MISSING','A2 requires actual pathway signed values and masks',path=str(directory/name))
    pi=[token(x,'patient_id') for x in json.loads((directory/names[0]).read_text())]
    ti=[token(x,'pathway_id') for x in json.loads((directory/names[1]).read_text())]
    if len(set(pi))!=len(pi) or len(set(ti))!=len(ti):fail('PATHWAY_CNV_DUPLICATE_IDS','CNV identity axis must be unique')
    coverage(patients,pi,role='A2 CNV patients',minimum=1.)
    rep=coverage(pathways,ti,role='A2 pathway IDs',minimum=minimum_entity_coverage)
    v=np.load(directory/names[2],mmap_mode='r',allow_pickle=False)
    m=np.load(directory/names[3],mmap_mode='r',allow_pickle=False)
    if v.shape!=m.shape or v.shape!=(len(pi),len(ti)):fail('PATHWAY_CNV_SHAPE_MISMATCH','Axes and matrix differ')
    if not np.isin(m,[0,1]).all():fail('INVALID_CALLABILITY','Pathway CNV mask is not binary')
    x=pd.DataFrame(v,index=pi,columns=ti).reindex(index=patients,columns=pathways).to_numpy(dtype=float,copy=True)
    ok=pd.DataFrame(m,index=pi,columns=ti).reindex(index=patients,columns=pathways).fillna(False).to_numpy(dtype=bool,copy=True)
    if np.any(ok&~np.isfinite(x)):fail('PATHWAY_CNV_CALLABLE_NONFINITE','Measured CNV must be finite')
    x[~ok]=np.nan
    return x,ok,{'alignment':rep,'files':{str(directory/n):sha(directory/n) for n in names}}


def common_mask_audit(X,Y,C,call,P,pcall,Z,li,ti,cohort,minimum=12):
    """Return three methods with equal valid patients, actual ranks and per-method BH."""
    li=np.asarray(li,int);ti=np.asarray(ti,int);cohort=np.asarray(cohort,int)
    n=len(li);methods={name:{k:np.full(n,np.nan) for k in ('r','p','q','n','rank')} for name in ('raw_A1_common','ranked_A1_common','ranked_A2_common')}
    for d in methods.values():d['available']=np.zeros(n,bool)
    for i,(l,tgt) in enumerate(zip(li,ti)):
        mask=(np.isfinite(X[cohort,l])&np.isfinite(Y[cohort,tgt])&np.isfinite(C[cohort,l])&call[cohort,l]
            &np.isfinite(P[cohort,tgt])&pcall[cohort,tgt]&np.isfinite(Z[cohort]).all(1))
        ix=cohort[mask];nn=len(ix)
        for d in methods.values():d['n'][i]=nn
        if nn<minimum:continue
        for transform,name in [('raw_signed','raw_A1_common'),('rank','ranked_A1_common')]:
            r=pairs(X,Y,C,call,Z,[l],[tgt],ix,minimum,cnv_transform=transform)
            for key in methods[name]:methods[name][key][i]=r[key][0]
        x=rankdata(X[ix,l]);y=rankdata(Y[ix,tgt])
        nuisance=np.column_stack([Z[ix],rankdata(C[ix,l]),rankdata(P[ix,tgt])])
        q,rank=basis(nuisance);xr=x-q@(q.T@x);yr=y-q@(q.T@y);df=nn-rank-1
        methods['ranked_A2_common']['rank'][i]=rank
        denominator=np.linalg.norm(xr)*np.linalg.norm(yr)
        if denominator<=1e-10 or df<=0:continue
        r=float(np.clip(xr@yr/denominator,-1,1));stat=abs(r)*np.sqrt(df/max(1-r*r,1e-15))
        dest=methods['ranked_A2_common'];dest['r'][i]=r;dest['p'][i]=2*student_t.sf(stat,df);dest['available'][i]=True
    for d in methods.values():d['q']=bh(d['p'])
    return methods


def run_audit(prepared,output,*,scope='source',training=None):
    """source=training-only; outer_test explicitly creates a separate held-out report.
    The output is outside the capsule and is not consumed by fitting or selection.
    """
    from .lineage import validate_prepared
    root=Path(prepared).resolve();out=Path(output).resolve()
    if out==root or out.is_relative_to(root):fail('SENSITIVITY_OUTPUT_INSIDE_CAPSULE','Never write held-out audits into training inputs')
    if out.exists() and any(out.iterdir()):fail('OUTPUT_EXISTS','Use a new audit directory')
    if scope not in {'source','outer_test'}:fail('CNV_AUDIT_SCOPE_INVALID',scope)
    m,spec,cfg,lin=validate_prepared(root)
    from .controller import assert_controller
    assert_controller(cfg)
    freeze_proof=None
    if scope=='outer_test':
        from .test_access import require_frozen_model,record_outer_open
        from .lineage import verify_sealed
        # Authorization precedes opening raw targets, patient values or sealed arrays.
        freeze_proof=require_frozen_model(root,training)
        verify_sealed(root,root.parent/(root.name+'__sealed_test'))
        record_outer_open(root,training,purpose='CNV_COMMON_MASK_SENSITIVITY',output=out)
    out.mkdir(parents=True,exist_ok=True)
    bound=json.loads((root/'INPUT_HASHES.json').read_text())
    for row in bound.values():
        if sha(row['path'])!=row['sha256']:fail('CNV_SENSITIVITY_SOURCE_DRIFT','Primary input changed after preparation',path=row['path'])
    fm=patient_map(cfg['sources']['patient_map']['path']);cov=table(cfg['sources']['covariates']['path'])
    assignments=json.loads((root/'PATIENT_ASSIGNMENTS.json').read_text());source_files={r['path']:r['sha256'] for r in bound.values()};summaries=[]
    for cancer in spec['cancers']:
        patients=[p for p,_ in assignments[cancer]];folds=np.asarray([f for _,f in assignments[cancer]])
        mp=fm.loc[fm.cancer_id.eq(cancer)];fit=np.flatnonzero(np.isin(folds,spec['train_folds']))
        cohort=fit if scope=='source' else np.flatnonzero(folds==spec['fold'])
        lframe,lp=partition(cfg['sources']['lnc_expression_root']['path'],cancer)
        pframe,pp=partition(cfg['sources']['pathway_activity_root']['path'],cancer)
        lncs=spec['nodes']['lncRNA'];targets=spec['nodes']['pathway']
        X=matrix(lframe,'lncrna_id','logcpm',patients,lncs,mp,minimum_entity_coverage=0.)
        Y=matrix(pframe,'pathway_id','activity_score',patients,targets,mp)
        C,call,files=cnv(cfg['sources']['cnv_root']['path'],cancer,patients,lncs,minimum_entity_coverage=0.)
        P,pcall,report=pathway_cnv(cfg['sources']['cnv_root']['path'],cancer,patients,targets)
        source_files.update(report['files']);source_files.update({str(f):sha(f) for f in files})
        cc=attach_patient(cov.loc[cov.cancer_id.eq(cancer)],mp)
        Z,adj=design(cc,patients,[patients[i] for i in fit],family='pathway',target_id='__pathway__',cfg=cfg)
        with np.load(root/'views'/f'{cancer}__validation'/'pathway.npz',allow_pickle=False) as a:li=a['lnc'];ti=a['target']
        result=common_mask_audit(X,Y,C,call,P,pcall,Z,li,ti,cohort,cfg['minimum_pairs'])
        rows=[]
        for method,r in result.items():
            rows.append(pd.DataFrame({'cancer_id':cancer,'lncrna_id':[lncs[i] for i in li],'pathway_id':[targets[i] for i in ti],
                'method':method,'scope':scope,'fold':spec['fold'],**r}))
        pd.concat(rows,ignore_index=True).to_csv(out/f'{cancer}.tsv.gz',sep='\t',index=False)
        summaries.append({'cancer_id':cancer,'patients':len(cohort),'pathway_cnv':report,'adjustment':adj})
    rec={'format':'R6_CNV_COMMON_MASK_SENSITIVITY_V1','prepared_sha256':sha(root/'MANIFEST.json'),'scope':scope,
         'model_freeze':freeze_proof,'primary_training_target_changed':False,'consumed_by_training':False,'a2_definition':'PARTIAL_RANK_CORRELATION_RESIDUALIZE_BOTH_X_Y_ON_CLINICAL_PLUS_RANK_LOCAL_PLUS_RANK_PATHWAY_CNV',
         'r7_bitwise_equivalence_claimed':False,'sources':source_files,'cohorts':summaries}
    atomic_json(out/'CNV_SENSITIVITY_RECEIPT.json',rec);commit_manifest(out,rec)
    return rec
