"""Materialize measured relation evidence after structural readout binding.

Equivalent pathway readouts and regulator-member states retain separate strata.
No model is fitted and no source association summary is constructed as input.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .relation_evidence_bindings import FORMAT as BINDING_FORMAT, file_sha, read_table
from .functional_fusion import FORMAT, validate_fused_arrays
from .fused_adjustment.design import design
from .fused_adjustment.adjustment_policy import resolve_policy

ASSAYS={'RNA':1,'PROTEIN':2,'PHOSPHO':4,'METHYLATION':8,'CNV':16,'VARIANT':32,'OTHER':64}


def assay_bit(value):
    value=str(value).upper()
    for names,bit in [(('PHOSPH',),4),(('PROTEIN','PROTEOM'),2),(('RNA','TRANSCRIPT'),1),
                      (('METHYL',),8),(('COPY','CNV'),16),(('VARIANT','SNV','MUTATION','SEQUENC'),32)]:
        if any(s in value for s in names):return bit
    return 64


def summarize_effect(rho,available,duplicate_groups):
    rho=np.asarray(rho,float);available=np.asarray(available,bool)
    if rho.shape!=available.shape or rho.ndim!=2 or not np.isfinite(rho[available]).all() or np.any(abs(rho[available])>1+1e-6):
        raise ValueError('MULTIMODAL_EFFECT_AXIS_OR_VALUE')
    if sorted(i for group in duplicate_groups for i in group)!=list(range(rho.shape[1])):
        raise ValueError('MULTIMODAL_EFFECT_DUPLICATE_PARTITION')
    weight=available.astype(float);distinct=np.zeros(len(rho),np.uint16)
    for group in duplicate_groups:
        n=available[:,group].sum(1);distinct+=n>0
        weight[:,group]=np.divide(weight[:,group],n[:,None],out=np.zeros((len(rho),len(group))),where=n[:,None]>0)
    clean=np.where(available,rho,0.);den=weight.sum(1)
    rms=np.sqrt(np.divide((weight*clean**2).sum(1),den,out=np.full(len(rho),np.nan),where=den>0))
    signed=np.divide((weight*clean).sum(1),den,out=np.full(len(rho),np.nan),where=den>0)
    return rms,signed,weight,distinct


def _json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
    temp.replace(path)


def _npz(path,arrays):
    path=Path(path);temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def materialize_context(*, root,catalog,binding_dir,output_root,cancer,context,empirical_fisher,bh,basis,algorithm_path,progress=print):
    from .local_execution import require_host_149
    require_host_149()
    if cancer not in {'BRCA','LUAD','COAD'} or context not in {'train_label_fold2','train_label_fold3','train_label_fold4','validation'}:
        raise ValueError('MULTIMODAL_Y_CONTEXT_OUTSIDE_DEMO')
    root,catalog,binding_dir,output_root=map(Path,(root,catalog,binding_dir,output_root))
    dest=output_root/cancer/context
    if dest.exists():raise ValueError('MULTIMODAL_Y_OUTPUT_EXISTS')
    dest.mkdir(parents=True)
    started=time.monotonic()
    _json(dest/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'materialize multimodal relation labels',
        'training':False,'sealed_test_opened':False,'cancer':cancer,'context':context,
        'binding_receipt_sha256':file_sha(binding_dir/'COMPLETE.json')})
    bound_receipt=json.loads((binding_dir/'COMPLETE.json').read_text())
    if bound_receipt['format']!=BINDING_FORMAT or bound_receipt['target_host']!='149':raise ValueError('MULTIMODAL_Y_BINDING_SCHEMA')
    for name,digest in bound_receipt['files'].items():
        if file_sha(binding_dir/name)!=digest:raise ValueError('MULTIMODAL_Y_BINDING_DRIFT')
    refs=json.loads((binding_dir/'INPUT_REFERENCES.json').read_text())
    for record in refs['structural'].values():
        if file_sha(record['path'])!=record['sha256']:raise ValueError('MULTIMODAL_Y_STRUCTURAL_DRIFT')
    bound=read_table(binding_dir/'READOUT_BINDINGS.tsv')
    cfg_path=root/'shared_relation_demo_r1/SCORING_ADJUSTMENT_CONFIG.json'
    cfg=json.loads(cfg_path.read_text())
    if cfg.get('target_host')!='149':raise ValueError('MULTIMODAL_Y_ADJUSTMENT_HOST')
    adoption_path=root/'shared_relation_demo_r1/fusion_method_r1/METHOD_ADOPTION.json'
    adoption=json.loads(adoption_path.read_text())
    if adoption['status']!='ADOPT_EMPIRICALLY_CALIBRATED_FISHER' or file_sha(algorithm_path)!=adoption['algorithm_sha256']:
        raise ValueError('MULTIMODAL_Y_FUSION_METHOD_DRIFT')
    # The explicit endpoint axis retains all477 identities, including mechanisms
    # without matching condition evidence, and the existing194 State columns.
    ids=json.loads((output_root/'AXIS_DEFINITION.json').read_text())['canonical_ids']
    policy={r['canonical_id']:r for r in read_table(catalog/'TARGET_POLICY.tsv')}
    if len(ids)!=len(set(ids)) or any(policy[t]['analysis_role'] not in {'primary_pathway','state_auxiliary'} for t in ids):
        raise ValueError('MULTIMODAL_Y_TARGET_AXIS')
    index={t:j for j,t in enumerate(ids)}
    roles=[policy[t]['roles'] for t in ids]
    tasks=['Pathway' if policy[t]['loss_group']=='pathway' else 'State' for t in ids]
    definition=json.loads((output_root/'Y_DEFINITION.json').read_text())
    unavailable=set(definition['conditional_targets_without_condition_evidence'])
    if not {r['output_canonical_id'] for r in bound if r['condition_required']=='True'}.issubset(unavailable):
        raise ValueError('MULTIMODAL_Y_CONDITION_POLICY_DRIFT')
    streams=[];groups=collections.defaultdict(list);references=[];splits={};patient_axes={};cov={};lncs=None
    for spec in refs['streams']:
        ds,layer=spec['cohort'],spec['layer'];path=Path(spec['directory'])
        if path.parent.name!=context or path.parent.parent.name!=cancer:raise ValueError('MULTIMODAL_Y_STREAM_CONTEXT')
        receipt=json.loads((path/'COMPLETE.json').read_text())
        if file_sha(path/'COMPLETE.json')!=spec['source_receipt_sha256'] or receipt['config_sha256']!=file_sha(cfg_path):
            raise ValueError('MULTIMODAL_Y_SOURCE_RECEIPT_DRIFT')
        for name in ['TARGETS.tsv','ASSOCIATIONS.npz','PATIENT_S.parquet']:
            if file_sha(path/name)!=receipt['files'][name]:raise ValueError('MULTIMODAL_Y_SOURCE_CONTENT_DRIFT: '+str(path/name))
        ax=json.loads((path.parent/'AXES.json').read_text());split=json.loads((path.parent/'SPLIT.json').read_text())
        if set(split['source_patients'])&set(split['label_patients']):raise ValueError('MULTIMODAL_Y_PATIENT_OVERLAP')
        if ds=='CPTAC' and cancer=='BRCA' and 'CPTAC:BRCA:604' in split['source_patients']+split['label_patients']:
            raise ValueError('MULTIMODAL_Y_UNRESOLVED_RETROSPECTIVE_CASE')
        if ds in splits and split!=splits[ds]:raise ValueError('MULTIMODAL_Y_SPLIT_DRIFT')
        splits[ds]=split;patients=ax['patient_ids'];patient_axes[ds]=patients
        if lncs is None:lncs=ax['lncrna_ids']
        if lncs!=ax['lncrna_ids']:raise ValueError('MULTIMODAL_Y_LNCRNA_AXIS')
        if ds not in cov:cov[ds]=pd.read_parquet(root/'shared_relation_demo_r1/cohorts'/f'{ds}_{cancer}/COVARIATES.parquet')
        md=read_table(path/'TARGETS.tsv');scores=pd.read_parquet(path/'PATIENT_S.parquet').reindex(patients)
        if list(scores)!=[r['target_id'] for r in md]:raise ValueError('MULTIMODAL_Y_SCORE_AXIS')
        selected=[r for r in bound if r['cohort']==ds and r['layer']==layer and r['output_canonical_id'] not in unavailable]
        keep=sorted({int(r['column_index']) for r in selected});compact={j:k for k,j in enumerate(keep)}
        with np.load(path/'ASSOCIATIONS.npz',allow_pickle=False) as z:
            stats={k:z[k][:,keep] for k in ['r','p','q','n','available']}
        if stats['r'].shape!=(len(lncs),len(keep)):raise ValueError('MULTIMODAL_Y_ASSOCIATION_AXIS')
        st={'dataset':ds,'layer':layer,'path':path,'metadata':md,'scores':scores.to_numpy(float),'stats':stats,
            'compact':compact,'label_index':pd.Index(patients).get_indexer(split['label_patients'])}
        if len(patients)!=len(set(patients)) or (st['label_index']<0).any():
            raise ValueError('MULTIMODAL_Y_PATIENT_AXIS')
        sid=len(streams);streams.append(st)
        for r in selected:
            j=int(r['column_index'])
            if md[j]['target_id']!=r['target_id'] or md[j]['canonical_id']!=r['source_canonical_id']:
                raise ValueError('MULTIMODAL_Y_READOUT_BINDING_AXIS')
            if r['output_canonical_id'] not in index:raise ValueError('MULTIMODAL_Y_BINDING_TARGET_OUTSIDE_AXIS')
            groups[r['output_canonical_id']].append((sid,j,r))
        references.append({'dataset':ds,'layer':layer,'directory':str(path),'receipt_sha256':spec['source_receipt_sha256']})
        progress(f'INPUT_READY {cancer} {context} {ds} {layer} retained_readouts={len(keep)}',flush=True)
    n,m=len(lncs),len(ids);shape=(n,m)
    arrays={k:np.full(shape,np.nan,np.float64) for k in ['combined_p','combined_q']}
    arrays.update({k:np.full(shape,np.nan,np.float32) for k in ['rms_effect','max_abs_effect']})
    arrays.update({k:np.zeros(shape,np.uint8) for k in ['observed','membership','observed_assay_bits','individually_strict_assay_bits',
        'individually_supported_assay_bits','cohort_count','individually_supported_cohort_count','calibration_method','direction']})
    arrays.update({k:np.zeros(shape,np.uint16) for k in ['readout_count','distinct_readout_count','positive_evidence_count','negative_evidence_count']})
    strata={k:np.full(shape,np.nan,np.float32) for k in ['signed_effect','equivalent_rms','component_rms']+[x+'_rms' for x in ASSAYS]}
    strata.update({k:np.zeros(shape,np.uint16) for k in ['equivalent_readouts','component_readouts']+[x+'_readouts' for x in ASSAYS]})
    z_cache={};basis_cache={};residual_cache={};signature_cache={};dependence=[];readout_index=[]
    def residual_signature(sid,j):
        ck=(sid,j)
        if ck in residual_cache:return residual_cache[ck],signature_cache[ck]
        st=streams[sid];ds=st['dataset'];row=st['metadata'][j];split=splits[ds]
        target=row['source_target_id'] if st['layer']=='STATE' else '__pathway__'
        family=row['family'] if st['layer']=='STATE' else 'pathway'
        _,pol=resolve_policy(target,contract=cfg['adjustment_contract']);key=(ds,json.dumps(pol,sort_keys=True))
        if key not in z_cache:
            z_cache[key]=design(cov[ds],patient_axes[ds],split['source_patients'],family=family,target_id=target,cfg=cfg,
                admission_cohorts={'source':split['source_patients'],'label':split['label_patients']})[0]
        Z=z_cache[key][st['label_index']];y=st['scores'][st['label_index'],j]
        ok=np.isfinite(Z).all(1)&np.isfinite(y);vec=np.full(len(y),np.nan)
        if ok.sum()>=12:
            bk=(key,ok.tobytes())
            if bk not in basis_cache:basis_cache[bk]=basis(Z[ok])
            Q,rank=basis_cache[bk]
            if ok.sum()-rank>=3:
                v=rankdata(y[ok]);v-=Q@(Q.T@v)
                if np.linalg.norm(v)>1e-8:vec[ok]=v
        raw_ok=np.isfinite(y);rv=np.zeros(len(y));rv[raw_ok]=rankdata(y[raw_ok])-(raw_ok.sum()+1)/2
        nz=np.flatnonzero(rv)
        if len(nz) and rv[nz[0]]>0:rv=-rv
        rv[rv==0]=0.
        signature=(ds,raw_ok.tobytes(),rv.tobytes(),key[1])
        residual_cache[ck]=vec;signature_cache[ck]=signature
        return vec,signature
    for number,cid in enumerate(ids,1):
        parts=groups.get(cid,[]);col=index[cid]
        if not parts:continue
        rho=np.column_stack([streams[s]['stats']['r'][:,streams[s]['compact'][j]] for s,j,r in parts])
        p=np.column_stack([streams[s]['stats']['p'][:,streams[s]['compact'][j]] for s,j,r in parts])
        av=np.column_stack([streams[s]['stats']['available'][:,streams[s]['compact'][j]] for s,j,r in parts]).astype(bool)
        av &= np.isfinite(rho)&np.isfinite(p)
        for s,j,r in parts:
            readout_index.append({**r,'canonical_id':cid,'dataset':streams[s]['dataset'],
                                  'directory':str(streams[s]['path']),'label_null_observations':int(np.isfinite(residual_signature(s,j)[0]).sum())})
        if not av.any():continue
        vectors=[];duplicate=collections.defaultdict(list)
        for a,(s,j,r) in enumerate(parts):
            v,signature=residual_signature(s,j);vectors.append(v);duplicate[signature].append(a)
        groups_dup=list(duplicate.values());k=len(parts);corr=np.eye(k);pair_n=np.zeros((k,k),int)
        for i in range(k):
            pair_n[i,i]=np.isfinite(vectors[i]).sum()
            for j in range(i):
                if streams[parts[i][0]]['dataset']!=streams[parts[j][0]]['dataset']:value=0.;count=0
                else:
                    mask=np.isfinite(vectors[i])&np.isfinite(vectors[j]);count=int(mask.sum())
                    vi,vj=np.nan_to_num(vectors[i]),np.nan_to_num(vectors[j]);den=np.linalg.norm(vi)*np.linalg.norm(vj)
                    value=float(vi@vj/den) if den>1e-12 else np.nan
                corr[i,j]=corr[j,i]=value;pair_n[i,j]=pair_n[j,i]=count
        fused=empirical_fisher(p,av,corr,groups=groups_dup,simulations=100000,seed=20261110,cache=dest/'null_references')
        valid=fused['available'];rms,signed,w,distinct=summarize_effect(rho,av,groups_dup);clean=np.where(av,rho,0.)
        arrays['observed'][:,col]=valid;arrays['combined_p'][:,col]=fused['p'];arrays['calibration_method'][:,col]=fused['calibration_method']
        arrays['rms_effect'][:,col]=rms;arrays['max_abs_effect'][:,col]=np.where(valid,np.max(abs(clean),axis=1),np.nan)
        arrays['readout_count'][:,col]=av.sum(1);arrays['distinct_readout_count'][:,col]=distinct;strata['signed_effect'][:,col]=signed
        rawq=np.column_stack([streams[s]['stats']['q'][:,streams[s]['compact'][j]] for s,j,r in parts])
        strict=av&(rawq<=.05)&(abs(clean)>=.20);support=strict|(av&(rawq<=.10)&(abs(clean)>=.15))
        signs=av&(p<=.05)&(abs(clean)>=.15);positive=((clean>0)&signs).sum(1);negative=((clean<0)&signs).sum(1)
        arrays['positive_evidence_count'][:,col]=positive;arrays['negative_evidence_count'][:,col]=negative
        arrays['direction'][:,col]=np.where(valid,np.where((positive>0)&(negative>0),3,np.where(positive>0,1,np.where(negative>0,2,np.where(signed>=0,1,2)))),0)
        for ds in splits:
            ii=[a for a,(s,j,r) in enumerate(parts) if streams[s]['dataset']==ds]
            if ii:arrays['cohort_count'][:,col]+=av[:,ii].any(1);arrays['individually_supported_cohort_count'][:,col]+=support[:,ii].any(1)
        readout_bits=np.array([assay_bit(r['measurement_layer']) for s,j,r in parts])
        component=np.array([r['evidence_role']=='molecular_component_readout' for s,j,r in parts])
        for a,bit in enumerate(readout_bits):
            arrays['observed_assay_bits'][:,col]|=np.where(av[:,a],bit,0).astype(np.uint8)
            arrays['individually_strict_assay_bits'][:,col]|=np.where(strict[:,a],bit,0).astype(np.uint8)
            arrays['individually_supported_assay_bits'][:,col]|=np.where(support[:,a],bit,0).astype(np.uint8)
        masks={'equivalent':~component,'component':component,**{name:readout_bits==bit for name,bit in ASSAYS.items()}}
        for name,mask in masks.items():
            active=av&mask[None,:]
            strata[name+'_rms'][:,col]=summarize_effect(rho,active,groups_dup)[0]
            strata[name+'_readouts'][:,col]=active.sum(1)
        dependence.append({'canonical_id':cid,'readouts':[{'dataset':streams[s]['dataset'],'target_id':r['target_id'],
            'evidence_role':r['evidence_role']} for s,j,r in parts],'duplicate_groups':groups_dup,
            'label_residual_gram_correlation':corr.tolist(),'label_pair_n':pair_n.tolist(),
            'calibration_references':fused['references']})
        if number%20==0 or component.any():
            report={'cancer':cancer,'context':context,'targets_processed':number,'targets_total':m,'current_readouts':k,
                'component_readouts':int(component.sum()),'seconds':round(time.monotonic()-started,1)}
            _json(dest/'PROGRESS.json',report);progress('Y_PROGRESS '+json.dumps(report),flush=True)
    multiplicity=[]
    for task in ['Pathway','State']:
        mask=np.asarray(tasks)==task;values=arrays['combined_p'][:,mask];arrays['combined_q'][:,mask]=bh(values)
        multiplicity.append({'task':task,'canonical_targets':int(mask.sum()),'tested_pairs':int(np.isfinite(values).sum())})
    strong=(arrays['observed']>0)&(arrays['combined_q']<=.05)&(arrays['max_abs_effect']>=.20)
    weak=(arrays['observed']>0)&(arrays['combined_q']<=.10)&(arrays['max_abs_effect']>=.15)&~strong
    arrays['membership']=np.where(strong,2,np.where(weak,1,0)).astype(np.uint8);arrays['direction'][arrays['membership']==0]=0
    validate_fused_arrays(arrays,shape)
    _json(dest/'AXES.json',{'lncrna_ids':lncs,'canonical_ids':ids,'target_roles':roles,'primary_tasks':tasks,
        'loss_groups':[policy[t]['loss_group'] for t in ids],'loss_weights':[float(policy[t]['loss_weight']) for t in ids]})
    _json(dest/'SPLIT.json',splits);_json(dest/'INPUT_REFERENCES.json',references)
    _json(dest/'DEPENDENCE_REFERENCES.json',dependence);_json(dest/'MULTIPLICITY.json',multiplicity)
    pd.DataFrame(readout_index).to_csv(dest/'READOUT_INDEX.tsv',sep='\t',index=False)
    _npz(dest/'Y.npz',arrays);_npz(dest/'EVIDENCE_STRATA.npz',strata)
    counts=[]
    for j,cid in enumerate(ids):
        counts.append({'canonical_id':cid,'loss_group':policy[cid]['loss_group'],'observed_relations':int(arrays['observed'][:,j].sum()),
           'equivalent_relations':int(np.isfinite(strata['equivalent_rms'][:,j]).sum()),
           'component_relations':int(np.isfinite(strata['component_rms'][:,j]).sum()),
           **{name+'_relations':int(np.isfinite(strata[name+'_rms'][:,j]).sum()) for name in ASSAYS}})
    pd.DataFrame(counts).to_csv(dest/'TARGET_COUNTS.tsv',sep='\t',index=False)
    record={'format':FORMAT,'target_host':'149','cancer_id':cancer,'context':context,'training':False,'sealed_test_opened':False,
        'y_definition_sha256':file_sha(output_root/'Y_DEFINITION.json'),
        'binding_receipt_sha256':file_sha(binding_dir/'COMPLETE.json'),'binding_directory':str(binding_dir),
        'method_adoption_sha256':file_sha(adoption_path),'algorithm_sha256':file_sha(algorithm_path),
        'canonical_targets':m,'observed_targets':int(arrays['observed'].any(0).sum()),'observed_relations':int(arrays['observed'].sum()),
        'unavailable_conditional_targets':sorted(unavailable),'seconds':time.monotonic()-started,
        'files':{p.name:file_sha(p) for p in dest.iterdir() if p.is_file() and p.name not in ['COMPLETE.json','PROGRESS.json']}}
    _json(dest/'COMPLETE.json',record)
    progress('MULTIMODAL_Y_COMPLETE '+json.dumps({k:v for k,v in record.items() if k not in ['files','unavailable_conditional_targets']}),flush=True)
    return record
