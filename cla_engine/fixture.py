"""End-to-end generated measurements. Never presented as patient or biological evidence."""
from __future__ import annotations
import copy,json,tempfile,time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json,sha,read_json,verify_manifest,status

def make_fixture(root,parquet=False,expanded_states=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);cfg=read_json(Path(__file__).resolve().parents[1]/'config/149.json');rng=np.random.default_rng(9041)
    # Synthetic fixtures explicitly exercise the established RNA/State executor.
    # They do not select or validate the future per-assay biological targets.
    cfg.pop('rppa_input', None)  # Generic fixtures do not claim real RPPA500 inputs.
    cfg['pathway_fusion']={'enabled':False}
    cfg['pathway_target_policy']={'status':'EXPLICIT_EXISTING_RNA'}
    if expanded_states:
        from .state_catalog import CATALOG_ID,expanded_targets
        from cc_hhgt.v32.adjustment_policy import default_adjustment_contract
        cfg['state_catalog']=CATALOG_ID;cfg['targets']=expanded_targets()
        cfg['adjustment_contract']=default_adjustment_contract([t['id'] for t in cfg['targets']])
    if not expanded_states:
        from .state_catalog import legacy_fixture_targets
        from cc_hhgt.v32.adjustment_policy import default_adjustment_contract
        cfg.pop('state_catalog',None);cfg['targets']=legacy_fixture_targets()
        cfg['adjustment_contract']=default_adjustment_contract()
    cancers=['C1','C2','C3'];lncs=[f'L{i}' for i in range(8)];genes=[f'G{i}' for i in range(6)];pathways=['P0','P1','P2','P3'];proteins=['R0','R1','R2','R3']
    cfg['gene_node_omics']={'enabled':False}  # Basic synthetic fixtures do not bind real patient assays.
    cfg['synthetic_fixture']=True;cfg['work_root']=str(root/'work');cfg['imports_root']=str(root/'imports');cfg['candidate_scope']['minimum_expression_cancers']=3;cfg['minimum_pairs']=12
    # Generic execution fixtures do not fabricate an association table. The
    # strict association path has its own explicit unit fixture.
    cfg['singlecell_association_role']='disabled';cfg['singlecell_association_cell_types']=[]
    cfg['profiles']['diagnostic']={'cancers':cancers,'max_lncrnas':8,'max_pathways':4,'pathway_candidates_per_cancer':32};cfg['baseline'].update(C=[.1,1.0],max_iter=2000,batch_rows=128)
    cfg['model'].update(hidden=16,heads=2,layers=3,dropout=.1,sc_buckets=2)
    cfg['training'].update(epochs=2,min_epochs=2,steps_per_epoch=3,batch_size=32,checkpoint_steps=2,checkpoint_seconds=1000,cpu_threads=1,amp=False)
    cfg['gates']['minimum_evaluable_cancers']=2;cfg['gates']['precision_k']=10
    def write(name,frame):
        p=root/(name+('.parquet' if parquet else '.tsv'));p.parent.mkdir(parents=True,exist_ok=True)
        if parquet:frame.to_parquet(p,index=False)
        else:frame.to_csv(p,sep='\t',index=False)
        return p
    def bind(role,name,frame):
        p=write(name,frame);cfg['sources'][role]={'path':str(p),'sha256':sha(p)};return p
    maps=[];scores=[];cov=[];mut=[];gene_mut=[]
    for c in cancers:
        n=100;patients=[f'{c}-PAT-{i:03}' for i in range(n)];samples=[p+'-S' for p in patients]
        latent=rng.normal(size=(n,4));X=np.column_stack([latent[:,i%4]+rng.normal(0,.18,n) for i in range(8)]);G=np.column_stack([latent[:,i%4]+rng.normal(0,.2,n) for i in range(6)]);Y=np.column_stack([latent[:,i]+rng.normal(0,.2,n) for i in range(4)])
        for i,(p,s) in enumerate(zip(patients,samples)):
            maps.append(dict(cancer_id=c,sample_id=s,patient_id=p,patient_fold_id=i%5,fold_seed=20260726))
            cv={'cancer_id':c,'patient_id':p,'sample_id':s,'age_years':45+rng.normal(0,10),'sex':'F' if i%2 else 'M','stage':'Stage IIA' if i%3 else 'Stage IIIB','purity':rng.uniform(.3,.9),'purity_source':'ABSOLUTE','leukocyte_fraction':rng.uniform(.1,.6),
                'cell_fraction_Macrophages.M0':rng.uniform(.01,.12),'cell_fraction_Macrophages.M1':rng.uniform(.01,.12),
                'cell_fraction_Macrophages.M2':rng.uniform(.01,.12),'molecular_subtype':'Subtype1' if i%2 else 'Subtype2',
                'clinical_subtype':'Subtype1' if i%2 else 'Subtype2','technical_batch':'B1' if i%2 else 'B2'};cov.append(cv)
            row={'cancer_id':c,'sample_id':s}
            for j,t in enumerate(cfg['targets']):row[t['column']]=latent[i,j%4]+rng.normal(0,.2)
            scores.append(row)
            for l in lncs:
                event=bool(rng.random()<.15);mut.append({'cancer_id':c,'patient_id':p,'lncrna_id':l,'lncrna_any_mutation':event,'lncrna_mutation_available':True,'exonic_variant_count':float(rng.integers(1,3)) if event else 0.,'absence_is_wildtype':False,'locus_callable':True})
            for g in genes:
                event=bool(rng.random()<.18)
                gene_mut.append({'cancer_id':c,'patient_id':p,'gene_id':g,'is_mutated':event,'mutation_callable':True,'mutation_count':int(event),'absence_is_wildtype':True,'consequence':'missense_variant' if event else ''})
        def long(A,ent,col,names):
            return pd.DataFrame({'cancer_id':c,'patient_id':np.repeat(patients,len(names)),'sample_id':np.repeat(samples,len(names)),ent:np.tile(names,n),col:A.ravel()})
        write(f'lnc/{c}',long(X,'lncrna_id','logcpm',lncs));write(f'gene/{c}',long(G,'gene_id','logcpm',genes));write(f'activity/{c}',long(Y,'pathway_id','activity_score',pathways))
        d=root/'cnv'/f'cancer={c}';d.mkdir(parents=True,exist_ok=True);atomic_json(d/'patient_ids.json',patients);atomic_json(d/'lncrna_ids.json',lncs)
        cn=rng.normal(0,.15,size=X.shape).astype('float32');cn[:,0]=0
        calls=np.ones(X.shape,np.uint8)
        for j in range(X.shape[1]):
            for f in range(5):calls[(j*5+f)%n,j]=0
        cn[calls==0]=np.nan
        np.save(d/'lncrna_signed_mean.npy',cn);np.save(d/'lncrna_callable.npy',calls)
        atomic_json(d/'pathway_ids.json',pathways)
        np.save(d/'pathway_signed_mean.npy',rng.normal(0,.12,size=(n,len(pathways))).astype('float32'))
        np.save(d/'pathway_callable.npy',np.ones((n,len(pathways)),np.uint8))
        f=pd.DataFrame([{'dataset_id':'DS_'+c,'cancer_id':c,'lncrna_id':l,'cell_type_major':ct,'mean_log_expression':float(rng.uniform(1,3)),'detection_rate':.6,'n_donors':5} for l in lncs for ct in ['malignant','Tcell']])
        dest=Path(cfg['imports_root'])/'singlecell'/'lnc_celltype';dest.mkdir(parents=True,exist_ok=True)
        if parquet:f.to_parquet(dest/f'{c}.parquet',index=False)
        else:f.to_csv(dest/f'{c}.tsv',sep='\t',index=False)
    bind('patient_map','patients',pd.DataFrame(maps));bind('state','state',pd.DataFrame(scores));cfg['sources']['state'].update(columns=len(pd.DataFrame(scores).columns),bytes=Path(cfg['sources']['state']['path']).stat().st_size)
    bind('covariates','covariates',pd.DataFrame(cov));bind('detection','detection',pd.DataFrame([{'cancer_id':c,'lncrna_id':l,'detection_rate_logcpm_gt0':.9} for c in cancers for l in lncs]))
    bind('exact_membership','membership',pd.DataFrame([{'pathway_id':p,'gene_id':g} for i,p in enumerate(pathways) for g in genes[i:i+2]]))
    bind('weighted_membership','weighted',pd.DataFrame([{'pathway_id':p,'gene_id':g,'weight':-1. if p=='P3' else 1.} for i,p in enumerate(pathways) for g in genes[i:i+2]]))
    bind('hierarchy','hierarchy',pd.DataFrame([{'pathway_id':p,'pathway_family_id':'F'+str(i//2),'membership_weight':1.} for i,p in enumerate(pathways)]))
    bind('binding','binding',pd.DataFrame([{'lncrna_id':l,'protein_id':proteins[i%4],'cancer_id':None,'is_context_specific':False,'weight':.8,'source_database':'fixture_experimental','relation_type':'physical_binding','mapping_status':'MAPPED','mapping_multiplicity':1,'edge_id':'B'+str(i)} for i,l in enumerate(lncs)]))
    bind('protein_gene','protein_gene',pd.DataFrame([{'protein_id':p,'gene_id':genes[i],'mapping_weight':1.,'mapping_multiplicity':1,'source_database':'UniProt_fixture'} for i,p in enumerate(proteins)]))
    bind('ppi','ppi',pd.DataFrame([{'protein_id_a':proteins[i],'protein_id_b':proteins[(i+1)%4],'weight':.7,'physical_score':700.,'source_database':'STRING_physical_fixture','relation_type':'physical_interaction','edge_id':'PPI'+str(i)} for i in range(4)]))
    for role,d in [('lnc_expression_root','lnc'),('gene_expression_root','gene'),('pathway_activity_root','activity'),('cnv_root','cnv')]:cfg['sources'][role]={'path':str(root/d)}
    m=pd.DataFrame([{'dataset_id':'DS_'+c,'formal_eligible':True,'model_derived':False,'outcome_derived':False} for c in cancers]);sr=Path(cfg['imports_root'])/'singlecell';gr=Path(cfg['imports_root'])/'genomic';gr.mkdir(parents=True,exist_ok=True)
    if parquet:m.to_parquet(sr/'dataset_manifest_33c.parquet',index=False);pd.DataFrame(mut).to_parquet(gr/'sample_lncrna_mutation_mc3_assay_callable.parquet',index=False)
    else:m.to_csv(sr/'dataset_manifest_33c.tsv',sep='\t',index=False);pd.DataFrame(mut).to_csv(gr/'sample_lncrna_mutation_mc3_assay_callable.tsv',sep='\t',index=False)
    if parquet:pd.DataFrame(gene_mut).to_parquet(gr/'sample_gene_mutation_mc3_assay_callable.parquet',index=False)
    else:pd.DataFrame(gene_mut).to_csv(gr/'sample_gene_mutation_mc3_assay_callable.tsv',sep='\t',index=False)
    from .inputs import commit_import_authority
    ir=Path(cfg['imports_root'])
    manifest={'format':'CLA_FULL_PYG_R62_LOCAL_MEASUREMENTS_V1','files':{q.relative_to(ir).as_posix():{'sha256':sha(q),'bytes':q.stat().st_size,'original_path':str(q)} for q in sorted(ir.rglob('*')) if q.is_file()},'synthetic_fixture':True}
    atomic_json(ir/'LOCAL_INPUT_MANIFEST.json',manifest)
    commit_import_authority(cfg,manifest,fixture=True)
    atomic_json(root/'fixture_config.json',cfg);return cfg

def selftest(workdir=None,parquet=False,device='cpu'):
    root=Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix='cla_full_pyg_r61_selftest_'))
    if (root/'SELFTEST_RESULT.json').exists():raise FileExistsError('Choose an unused selftest directory')
    cfg=make_fixture(root/'sources',parquet);prepared=root/'prepared'
    from .prepare import prepare
    from .train import train,load_checkpoint,model_signature
    from .evaluate import evaluate_fold
    import torch
    prepare(cfg,prepared,0,'diagnostic');verify_manifest(prepared)
    a=train(prepared,root/'uninterrupted',device=device,deadline=time.time()+1800,unsafe_local_debug=True)
    b=train(prepared,root/'resumed',device=device,deadline=time.time()+1800,unsafe_local_debug=True,interrupt_after=2)
    assert b['state']=='PAUSED_TEST_INTERRUPT'
    b=train(prepared,root/'resumed',device=device,deadline=time.time()+1800,unsafe_local_debug=True)
    sig=model_signature(prepared,cfg,read_json(prepared/'SPEC.json'));ca,_=load_checkpoint(root/'uninterrupted',sig);cb,_=load_checkpoint(root/'resumed',sig)
    exact=all(torch.equal(ca['model'][k],cb['model'][k]) for k in ca['model'])
    if device=='cpu':assert exact,'CPU stop/resume differs from uninterrupted training'
    ev=evaluate_fold(prepared,root/'resumed',root/'evaluation')
    report={'state':'SELFTEST_PASS','synthetic':True,'parquet_backend_tested':parquet,'device':device,'hgt_real_typed_attention_executed':True,'pathway_shared_grad':b['gradient_proof']['pathway_max_shared_grad'],'state_shared_grad':b['gradient_proof']['state_max_shared_grad'],'cpu_bitwise_resume_equal':exact if device=='cpu' else 'NOT_CLAIMED_CUDA_ATOMICS','test_labels_uploaded_to_worker':False,'output':str(root),'model_gate':a['state'],'not_real_patient_training':True}
    atomic_json(root/'SELFTEST_RESULT.json',report);print(json.dumps(report,indent=2));return report


def data_selftest(workdir=None,parquet=False):
    """Run real preparation/LASSO/sealed-label checks without importing PyG."""
    root=Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix='cla_full_r61_data_'))
    if (root/'DATA_SELFTEST.json').exists():raise FileExistsError('Choose an unused data-selftest directory')
    from .prepare import prepare
    cfg=make_fixture(root/'sources',parquet);p=prepare(cfg,root/'prepared',0,'diagnostic');manifest=verify_manifest(p)
    spec=read_json(p/'SPEC.json');views=read_json(p/'VIEWS.json')
    assert spec['architecture']=='CC_HHGT_PYG_FULL_R62_JOINT'
    assert 'cancer' in spec['nodes'] and 'pathway_family' in spec['nodes']
    assert cfg['model']['task_weights']=={'pathway':1.,'state':.5}
    assert manifest['metadata']['sealed_test_labels_included'] is False
    result={'state':'DATA_SELFTEST_PASS_NOT_TRAINED','synthetic':True,'parquet_backend_tested':parquet,
            'view_count':len(views),'pyg_executed':False,'lasso_fitted':True,'output':str(root)}
    atomic_json(root/'DATA_SELFTEST.json',result);print(json.dumps(result,indent=2));return result
