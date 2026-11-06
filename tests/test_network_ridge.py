import numpy as np
from scipy import sparse
from cc_hhgt.network_ridge import alignment_weights,weighted_path,extract_weighted_fit,group_kernels,base_kernel,center_trace,add_legacy_kernels
from cc_hhgt.graph_ridge import ridge_path,normalized_kernel,fit_transform,target_policy

def fixture():
    rng=np.random.default_rng(93);x=rng.normal(size=(45,16));z=np.column_stack([np.ones(45),rng.normal(size=45)]);y=x[:32,:2]@np.array([[2.,-1.],[-3.,4.]])+z[:32]@np.array([[2.,3.],[8.,-4.]])
    kg=np.stack([x[:,:7]@x[:,:7].T/7,x[:,7:]@x[:,7:].T/9]);return kg,z,y,np.arange(32),np.arange(32,45)

def test_weighted_dual_matches_existing_and_primal():
    kg,z,y,tr,te=fixture();w=np.array([[.3,.8],[.7,.2]]);ls=[.001,.1,2.];p,st=weighted_path(kg,w,z,tr,te,y,ls,return_fits=True)
    for j in range(2):
        k=np.einsum('g,gij->ij',w[:,j],kg);old,_=ridge_path(k,z,tr,te,y[:,j],ls);np.testing.assert_allclose(p[:,:,j],old[:,:,0],atol=1e-9)
        f=extract_weighted_fit(st[j],ls[1],w[:,j]);np.testing.assert_allclose(z[te]@f['nuisance']+k[np.ix_(te,tr)]@f['dual'],p[1,:,j],atol=1e-9)
        vals,u=np.linalg.eigh((k+k.T)/2);x=u*np.sqrt(np.maximum(vals,0));a=np.column_stack([z[tr],x[tr]]);pen=np.diag(np.r_[np.zeros(z.shape[1]),np.repeat(len(tr)*ls[1],len(k))]);beta=np.linalg.solve(a.T@a+pen,a.T@y[:,j]);np.testing.assert_allclose(np.column_stack([z[te],x[te]])@beta,p[1,:,j],atol=1e-7)

def test_zero_power_and_heldout_poison():
    kg,z,y,tr,te=fixture();mass=np.array([.2,.8]);w=alignment_weights(kg,mass,z,tr,y,0);np.testing.assert_array_equal(w,np.broadcast_to(mass[:,None],w.shape));a=alignment_weights(kg,mass,z,tr,y,1)
    poisoned=kg.copy();poisoned[:,te,:]=1e10;poisoned[:,:,te]=-1e10;zz=z.copy();zz[te]=1e12
    np.testing.assert_array_equal(a,alignment_weights(poisoned,mass,zz,tr,y,1))
    raw=np.random.default_rng(5).normal(size=(45,10));raw[0,1]=np.nan;x,sp=fit_transform(raw,tr);raw[te]=1e10;xx,sp2=fit_transform(raw,tr);np.testing.assert_array_equal(x[tr],xx[tr])

def test_modules_keep_opposite_coefficients():
    rng=np.random.default_rng(42);x=rng.normal(size=(80,2));y=x[:60]@np.array([3.,-4.]);kg=(x@x.T/2)[None];z=np.ones((80,1));p,st=weighted_path(kg,np.ones((1,1)),z,np.arange(60),np.arange(60,80),y,[1e-8],return_fits=True);f=extract_weighted_fit(st[0],1e-8,np.ones(1));beta=x[:60].T@f['dual']/2
    assert beta[0]>2.9 and beta[1]<-3.9
    assert np.mean((p[0,:,0]-x[60:]@np.array([3.,-4.]))**2)<1e-10

def test_legacy_connected_solver_exact():
    rng=np.random.default_rng(7);a=np.zeros((12,12));a[0,5]=a[5,0]=.8;a[1,6]=a[6,1]=.5;d=a.sum(1);lap=sparse.csr_matrix(np.diag(d)-a);x=rng.normal(size=(24,12));types=np.array(['lncrna']*5+['mrna']*7);ctx={'_rna_x':x,'_rna_index':np.arange(12),'_rna_types':types,'train':np.arange(18),'legacy':{}}
    add_legacy_kernels([ctx],{'ppi':lap})
    for arm in ['lncrna','mrna','combined']:
        sel=np.arange(12) if arm=='combined' else np.flatnonzero(types==arm)
        for eta in [.01,.1,1,10,100]:
            expected,_=normalized_kernel(x[:,sel],np.arange(18),lap,eta,sel);np.testing.assert_allclose(ctx['legacy'][('ppi',arm,eta)],expected,atol=1e-10)

def test_self_assay_exclusions():
    assert 'PROTEIN' not in target_policy({'layer':'PROTEIN_PATHWAY','target_id':'p'})['allowed_blocks']
    assert 'PHOSPHO' not in target_policy({'layer':'PHOSPHO_PTMSEA','target_id':'p'})['allowed_blocks']
    assert target_policy({'layer':'RNA_PATHWAY','target_id':'r'})['role']=='RNA_DERIVED_TARGET'


def test_feature_mapping_and_degree_preserving_null(tmp_path):
    import pandas as pd
    from scripts.prepare_network_validation import feature_map,rewire
    ft=pd.DataFrame({'feature_id':['LNC:ENSG00000000001','GENE:ENSG00000000002'],'gene_id':['ENSG00000000001','ENSG00000000002'],'feature_type':['lncrna','mrna']});ft.to_csv(tmp_path/'RNA_FEATURES.tsv',sep='\t',index=False)
    pd.DataFrame({'gene_id':ft.gene_id,'reference_symbol':['LINC_TEST','GENE_TEST']}).to_csv(tmp_path/'ann.tsv',sep='\t',index=False)
    columns={'RNA':ft.feature_id.tolist(),'CNV':['CNV:LINC_TEST','CNV:GENE_TEST','CNV:UNKNOWN'],'MUTATION':['MUTATION:LINC_TEST'],'PROTEIN':['PROTEIN:ENSG00000000002'],'PHOSPHO':['PHOSPHO:ENSG00000000002.1|P|S23|peptide|1']}
    for b,cols in columns.items():pd.DataFrame(np.ones((3,len(cols))),columns=cols,index=['a','b','c']).to_parquet(tmp_path/(b+'.parquet'))
    fm=feature_map(tmp_path,tmp_path/'ann.tsv');assert fm.loc[fm.feature_id.eq('CNV:LINC_TEST'),'anchor'].item()=='L:ENSG00000000001';assert fm.loc[fm.feature_id.eq('CNV:UNKNOWN'),'anchor'].isna().all();assert fm.loc[fm.block.eq('PHOSPHO'),'anchor'].item()=='G:ENSG00000000002'
    for bip in [False,True]:
        a=np.arange(30);b=np.roll(a,3)+(30 if bip else 0)
        if not bip:a,b=np.minimum(a,b),np.maximum(a,b)
        e=pd.DataFrame({'lo':a,'hi':b,'weight':np.arange(30)+1.});ee,audit=rewire(e,60,101,bipartite=bip);assert audit['degree_exact'];assert audit['weight_multiset_exact'];assert len(set(zip(ee.lo,ee.hi)))==len(e)
        if bip:assert (ee.lo<30).all() and (ee.hi>=30).all()

def test_nested_split_integrity():
    from scripts.run_network_validation import split_plan
    from sklearn.model_selection import KFold
    original={'outer'+str(i):{'train':tr.tolist(),'test':te.tolist()} for i,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=7).split(np.arange(60)))}
    plans=split_plan(original,60)
    for name,p in plans.items():
        assert not set(p['train'])&set(p['test'])
        assert len(p['inner'])==6
        for s in p['inner']:
            assert not set(s['train'])&set(s['validation'])
            assert set(s['train'])|set(s['validation'])==set(p['train'])
    for i in range(5):assert plans['repeat0_outer'+str(i)]['test']==original['outer'+str(i)]['test']


def test_end_to_end_nested_outer_and_final(tmp_path):
    import json,pandas as pd,joblib
    from pathlib import Path
    from sklearn.model_selection import KFold
    from scripts.run_network_validation import run
    from cla_engine.common import sha
    from cc_hhgt.network_ridge import MODES
    rng=np.random.default_rng(134);n=60;p=40;ids=['sample'+str(i) for i in range(n)];src=tmp_path/'input/COAD';net=tmp_path/'net/COAD';src.mkdir(parents=True);net.mkdir(parents=True)
    types=['lncrna']*15+['mrna']*25;cols=['LNC:'+str(i) if i<15 else 'GENE:'+str(i) for i in range(p)];ft=pd.DataFrame({'feature_id':cols,'gene_id':[str(i) for i in range(p)],'feature_type':types});ft.to_csv(src/'RNA_FEATURES.tsv',sep='\t',index=False)
    fmap=[];blocks={}
    for block,width in [('RNA',p),('CNV',8),('MUTATION',8),('PROTEIN',8),('PHOSPHO',8)]:
        data=rng.normal(size=(n,width));bc=cols if block=='RNA' else [block+':'+str(i) for i in range(width)];blocks[block]=data;pd.DataFrame(data,index=ids,columns=bc).to_parquet(src/(block+'.parquet'))
        for i in range(width):fmap.append(dict(block=block,position=i,feature_id=bc[i],**{'module_'+mode:(i%4 if i<width-1 else -1) for mode in MODES}))
    pd.DataFrame(fmap).to_parquet(net/'FEATURE_MAP.parquet');cov=pd.DataFrame({'cohort':'CPTAC','age_years':rng.uniform(40,70,n),'sex':['M','F']*(n//2),'stage':['II','III']*(n//2),'purity':rng.uniform(.4,.8,n)},index=ids);cov.to_parquet(src/'COVARIATES.parquet')
    tids=['RNA_PATHWAY|a','PROTEIN_PATHWAY|a','PHOSPHO_PTMSEA|a','STATE|published::WES_purity'];meta=pd.DataFrame({'target_id':tids,'layer':['RNA_PATHWAY','PROTEIN_PATHWAY','PHOSPHO_PTMSEA','STATE'],'family':['test']*4,'method':['test']*4});meta.to_csv(src/'TARGETS.tsv',sep='\t',index=False);yy=blocks['RNA'][:,:4]+rng.normal(size=(n,4))*.3;pd.DataFrame(yy,index=ids,columns=tids).to_parquet(src/'Y.parquet')
    splits={'outer'+str(i):{'train':tr.tolist(),'test':te.tolist()} for i,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=9).split(np.arange(n)))};(src/'SPLITS.json').write_text(json.dumps(splits));a=np.zeros((p,p));a[0,15]=a[15,0]=1;a[16,17]=a[17,16]=1;lap=sparse.csr_matrix(np.diag(a.sum(1))-a)
    for mode in MODES:sparse.save_npz(net/('LEGACY_'+mode+'.npz'),lap)
    physical_receipt={'rpi_policy':'EXPLICIT_PHYSICAL_METHOD_AND_FAMILY_REQUIRED; SYNTHETIC_TEST_GRAPH'};(src/'NETWORK_AUDIT.json').write_text(json.dumps(physical_receipt));(net/'CORRECTED_RNA_NETWORK_AUDIT.json').write_text(json.dumps(physical_receipt))
    for folder in [src,net]:(folder/'COMPLETE.json').write_text(json.dumps({'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}))
    for phase in ['repeat0_outer0','final']:
        run(src.parent,net.parent,tmp_path/'out','COAD',phase);out=tmp_path/'out/COAD'/phase;assert (out/'COMPLETE.json').exists();audit=pd.read_parquet(out/'AUDIT.parquet');assert audit.status.eq('FITTED').all();assert audit.variant.nunique()==20;assert len(audit)==4*58
        assert audit.loc[audit.target_id.eq(tids[1]),'allowed_blocks'].str.contains('PROTEIN').sum()==0
        if phase=='final':
            models=joblib.load(out/'MODELS.joblib');assert len(models)==len(audit)
            for key,model in models.items():assert set(model['train'])==set(range(n));assert np.isfinite(model['dual']).all()
            from cc_hhgt.network_ridge import make_context
            from cc_hhgt.network_ridge_predict import predict
            ctxs={}
            for key,model in models.items():
                if model['kind']=='legacy':continue
                purity=model['purity']
                if purity not in ctxs:ctxs[purity]=make_context(blocks,ft,pd.DataFrame(fmap),cov,np.arange(n),purity)
                ctx=ctxs[purity]
                if model['kind']=='module':
                    kg,mass,labels=group_kernels(ctx,model['arm'],model['allowed_blocks'],model['mode']);assert labels==model['group_labels'];kernel=np.einsum('g,gij->ij',model['group_weights'],kg)
                else:kernel=base_kernel(ctx,model['arm'],model['allowed_blocks'],model['variant'])
                expected=ctx['design']@model['nuisance']+kernel[:,model['train']]@model['dual'];got,supported=predict(model,ctx,blocks,blocks,cov,pd.DataFrame(fmap),ft);np.testing.assert_allclose(got,expected,atol=1e-8);assert supported.all()
        else:
            predictions=pd.read_parquet(out/'PREDICTIONS.parquet');assert set(predictions.patient_id)=={ids[i] for i in splits['outer0']['test']};assert np.isfinite(predictions.predicted).all()


def test_bootstrap_respects_patient_pairing_and_r2_units():
    from scripts.bootstrap_network_validation import family_mean_bootstrap
    y=np.array([[0.,1.,2.,3.],[3.,2.,1.,0.]])
    err_base=np.array([[1.,0.,2.,3.],[0.,1.,2.,4.]])
    err_method=err_base*.5;delta=(err_base-err_method)/np.var(y,axis=1)[:,None];valid=np.ones_like(y,bool);counts=np.ones((5,4));point,b=family_mean_bootstrap(delta,valid,np.array(['one','one']),counts)
    expected=np.mean((err_base-err_method).sum(1)/((y-y.mean(1)[:,None])**2).sum(1));np.testing.assert_allclose(point,expected);np.testing.assert_allclose(b,expected)
    point,b=family_mean_bootstrap(np.zeros_like(delta),valid,np.array(['one','one']),counts);assert point==0;assert (b==0).all()


def test_expression_or_reporter_evidence_is_not_physical_rpi():
    import pandas as pd
    from cc_hhgt.network_evidence import physical_rpi_mask
    f=pd.DataFrame([
        dict(graph_assay_class='experimental_unspecified',experiment_family='expression_or_abundance',assay_subtype='expression_or_abundance',relation_type='binds_protein_experimental_unspecified'),
        dict(graph_assay_class='experimental_unspecified',experiment_family='reporter_assay',assay_subtype='reporter_assay'),
        dict(graph_assay_class='other_physical',experiment_family='physical_binding',assay_subtype='other_experimental'),
        dict(graph_assay_class='eclip',experiment_family='physical_binding',assay_subtype='eclip'),
        dict(graph_assay_class='rna_capture',experiment_family='physical_binding',assay_subtype='chart|rna_pulldown'),
        dict(graph_assay_class='other_clip',experiment_family='physical_binding',assay_subtype='hits_clip|par_clip'),
        dict(graph_assay_class='eclip',experiment_family='expression_or_abundance',assay_subtype='eclip'),
        dict(graph_assay_class='predicted',experiment_family='physical_binding',assay_subtype='eclip')])
    assert physical_rpi_mask(f).tolist()==[False,False,False,True,True,True,False,False]


def test_historical_graph_inputs_are_refused():
    import pytest
    from cc_hhgt.network_evidence import require_physical_network_receipt
    with pytest.raises(ValueError,match='RPI_EVIDENCE_GATE_MISSING'):require_physical_network_receipt({'rpi_policy':'EXPERIMENTAL_CLASSES_ONLY'})
    require_physical_network_receipt({'rpi_policy':'EXPLICIT_PHYSICAL_METHOD_AND_FAMILY_REQUIRED; TEST'})


def test_saved_final_model_audit_and_checksum_gate(tmp_path):
    import json, pandas as pd, pytest
    from scripts.audit_network_final_models import audit_cancer, checked_receipt
    test_end_to_end_nested_outer_and_final(tmp_path)
    audit_cancer(tmp_path/"input", tmp_path/"net", tmp_path/"out", tmp_path/"artifact_audit", "COAD")
    receipt = json.loads((tmp_path/"artifact_audit/COAD/COMPLETE.json").read_text())
    assert receipt["status"] == "PASS" and receipt["models_checked"] == 88
    assert receipt["targets_checked"] == 4 and not receipt["new_external_validation"]
    checks = pd.read_csv(tmp_path/"artifact_audit/COAD/CHECKS.tsv", sep="\t")
    assert set(checks.arm) == {"lncrna", "mrna", "combined", "shared"}
    assert checks.purity.nunique() == 2
    assert checks.missing_clinical_rejected.all() and checks.reordered_query_pass.all()
    final = tmp_path/"out/COAD/final"
    with (final/"TARGETS.tsv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="ARTIFACT_CHECKSUM_DRIFT"):
        checked_receipt(final, ["TARGETS.tsv"])
