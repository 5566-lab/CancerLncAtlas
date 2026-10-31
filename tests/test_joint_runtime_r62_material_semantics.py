"""R6.2 audit regressions using real-source-shaped synthetic inputs.
No 149 patient measurements, CUDA or cloud actions occur in these tests.
"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from cla_engine.common import Stop,sha,read_json,atomic_json
from cla_engine.graph import coexpression,graph_npz
from cla_engine.graph_contracts import normalize_binding,binding_context
from cla_engine.genomics import project_gene_context,build_genomic_projection,read_cancer_source,normalize_mutation,FEATURES,SENTINEL
from cla_engine.statistics import pairs,design,COMPOSITION
from cla_engine.exact_baseline import identity_records,fit_vocabulary,training_design,decision_batched,unseen_audit,resource_plan
from cla_engine.execution_contract import baseline_contract
from cc_hhgt.v32.adjustment_policy import default_adjustment_contract,effective_policy
from cc_hhgt.v32.covariate_contract import CovariateEncoder

ROOT=Path(__file__).resolve().parents[1]


def bindings():
    return pd.DataFrame([
        {'lncrna_id':'L','protein_id':'P1','cancer_id':None,'is_context_specific':False,'weight':.5,'relation_type':'physical_binding','source_database':'measured','mapping_status':'mapped','edge_id':'A'},
        {'lncrna_id':'L','protein_id':'P1','cancer_id':'KIRC','is_context_specific':True,'weight':.8,'relation_type':'physical_binding','source_database':'measured','mapping_status':'mapped','edge_id':'B'},
        {'lncrna_id':'M','protein_id':'P2','cancer_id':'BRCA','is_context_specific':True,'weight':.7,'relation_type':'physical_binding','source_database':'measured','mapping_status':'mapped','edge_id':'C'},
    ])


def projection_inputs():
    v=np.zeros((2,len(FEATURES)),np.float32);v[:,0]=[2,4];v[:,-1]=1
    return v,['ENSG1','ENSG2'],pd.DataFrame({'protein_id':['P1','P2'],'gene_id':['GENE:ENSG1','GENE:ENSG2']}),['P1','P2'],pd.DataFrame({'pathway_id':['W','W'],'gene_id':['ENSG1','ENSG2']}),['W'],bindings(),['L','M']


@pytest.mark.parametrize('variant',['G1','G2'])
def test_global_binding_has_same_graph_and_partner_scope(variant):
    args=projection_inputs();p=project_gene_context(*args,'KIRC',variant)
    assert p['partner'][0,0]==2 and p['partner'][0,-1]==1
    assert not p['partner'][1].any()  # BRCA-only does not leak to KIRC
    assert p['audit']['binding_context']['global_rows']==1
    assert p['audit']['distinct_lnc_partner_gene_pairs']==1  # not GLOBAL+local counted twice
    scoped,_=binding_context(args[6],'KIRC',args[7])
    edge=np.array([[0,0],[0,0]])
    graph=graph_npz({}, {'lncRNA|b|protein':{'edge_index':edge,'weight':scoped.weight.to_numpy(),'context':scoped.cancer_id.to_numpy()}},'KIRC',{}, {})
    assert graph['edge::lncRNA|b|protein'].shape[1]==1


def test_g0_has_no_partner_binding_information():
    p=project_gene_context(*projection_inputs(),'KIRC','G0')
    assert not p['partner'].any() and not p['audit']['partner_context_uses_binding']


def test_binding_normalization_idempotent_and_order_invariant():
    f=bindings();pd.testing.assert_frame_equal(normalize_binding(f),normalize_binding(normalize_binding(f)))
    args=projection_inputs();a=project_gene_context(*args,'KIRC','G2');args=list(args);args[6]=f.iloc[::-1]
    b=project_gene_context(*args,'KIRC','G2')
    np.testing.assert_array_equal(a['partner'],b['partner'])


def test_unknown_null_binding_cannot_be_invented_global():
    args=list(projection_inputs());args[6]=args[6].drop(columns='is_context_specific')
    with pytest.raises(Stop):project_gene_context(*args,'KIRC','G2')


def test_projection_cache_preserves_values_and_rejects_cancer_drift():
    args=projection_inputs();plan=build_genomic_projection(*args[1:],'KIRC','G2')
    direct=project_gene_context(*args,'KIRC','G2');cached=project_gene_context(*args,'KIRC','G2',projection=plan)
    for name in ('gene','protein','pathway','partner'):np.testing.assert_array_equal(direct[name],cached[name])
    with pytest.raises(Stop):project_gene_context(*args,'BRCA','G2',projection=plan)


def graph_data(seed=930):
    rng=np.random.default_rng(seed);n=46;l=5;g=7
    z=np.column_stack([np.ones(n),rng.normal(size=n)])
    x=rng.normal(size=(n,l));y=np.column_stack([x[:,i%l]+rng.normal(0,.15,n) for i in range(g)])
    cn=rng.normal(size=x.shape);cn[:,0]=0;mask=np.ones(x.shape,bool)
    for j in range(l):mask[j,j]=False
    x[7,2]=np.nan;y[8,3]=np.nan;y[9,4]=np.nan
    z[10,1]=np.nan
    return x,y,cn,mask,z


def reference_edges(args,cohort,topk,mode,min_n=12):
    x,g,c,m,z=args;li=np.repeat(np.arange(x.shape[1]),g.shape[1]);gi=np.tile(np.arange(g.shape[1]),x.shape[1])
    st=pairs(x,g,c,m,z,li,gi,cohort,min_n,cnv_transform=mode)
    corr=st['r'].reshape(x.shape[1],g.shape[1]);answer={}
    for l in range(x.shape[1]):
        order=sorted(range(g.shape[1]),key=lambda j:(-abs(corr[l,j]) if np.isfinite(corr[l,j]) else np.inf,j))[:topk]
        for j in order:
            if np.isfinite(corr[l,j]) and abs(corr[l,j])>=.15:answer[(l,j)]=(float(corr[l,j]),int(st['n'][l*g.shape[1]+j]),int(st['rank'][l*g.shape[1]+j]))
    return answer


@pytest.mark.parametrize('mode',['raw_signed','rank'])
@pytest.mark.parametrize('blocks',[(1,1),(2,3),(64,256)])
def test_pairwise_graph_matches_every_scalar_pair_on_common_masks(mode,blocks):
    args=graph_data();ix=np.arange(41);e,a=coexpression(*args,ix,topk=4,min_n=12,cnv_transform=mode,return_audit=True,lnc_batch=blocks[0],gene_batch=blocks[1])
    want=reference_edges(args,ix,4,mode);got={}
    for s,(edge,w) in e.items():
        for j,(l,g) in enumerate(edge.T):got[(int(l),int(g))]=(float(w[j])*(1 if s=='positive' else -1),int(a['edge_support'][s]['n'][j]),int(a['edge_support'][s]['rank'][j]))
    assert set(got)==set(want)
    for key in got:
        assert got[key][0]==pytest.approx(want[key][0],abs=1e-6)
        assert got[key][1:]==want[key][1:]
    assert a['all_patients_callable_lncrnas']==0 and a['estimable_pairs']>0


def test_all_lnc_partially_missing_does_not_empty_dynamic_graph():
    args=graph_data();e,a=coexpression(*args,range(40),return_audit=True)
    assert a['all_patients_callable_lncrnas']==0
    assert a['lncrnas_with_selected_edges']==5 and sum(len(x[1]) for x in e.values())>0


def test_shared_mask_minimum_pairs_not_global_complete_case():
    args=list(graph_data());args[3][:,1]=False;args[3][:11,1]=True
    e,a=coexpression(*args,range(40),return_audit=True,min_n=12)
    assert all(1 not in edges[0][0] for edges in e.values())
    assert a['pairs_below_minimum_n']>=7


def test_no_held_out_measurement_can_change_graph():
    args=graph_data();copy_=[v.copy() for v in args]
    for v in copy_[:3]:v[35:]=999
    copy_[3][35:]=False;copy_[4][35:]=np.nan
    a=coexpression(*args,range(35));b=coexpression(*copy_,range(35))
    for s in a:
        for aa,bb in zip(a[s],b[s]):np.testing.assert_array_equal(aa,bb)


def test_empty_cohort_and_all_missing_do_not_fabricate_edges():
    args=graph_data();e,a=coexpression(*args,[],return_audit=True)
    assert all(len(v[1])==0 for v in e.values()) and a['estimable_pairs']==0
    args=list(args);args[3][:]=False;e,a=coexpression(*args,range(46),return_audit=True)
    assert all(len(v[1])==0 for v in e.values()) and a['empty_reason']=='NO_ESTIMABLE_COMMON_MASK_PAIRS'


@pytest.mark.parametrize('cohort',[[0,0],[-1],[46],[.1]])
def test_graph_rejects_invalid_source_indices(cohort):
    with pytest.raises(Stop):coexpression(*graph_data(),cohort)


def covariates(source='ABSOLUTE'):
    rng=np.random.default_rng(5);n=70
    f=pd.DataFrame({'patient_id':[f'P{i}' for i in range(n)],'age_years':rng.normal(50,5,n),'sex':['F','M']*35,'stage':['IIA','IVB']*35,'purity_source':source})
    for c in COMPOSITION:f[c]=rng.uniform(.2,.8,n)
    return f


@pytest.mark.parametrize('target',['ImmuneScore','StromalScore','CYT','RNAss'])
def test_independent_purity_and_all_composition_are_retained(target):
    f=covariates();ids=f.patient_id.tolist();z,r=design(f,ids,ids[:50],family='state',target_id=target)
    assert set(COMPOSITION)<=set(r['required']) and set(COMPOSITION)<=set(r['encoded'])
    assert not r['excluded'] and np.isfinite(z).all()


def test_leukocyte_target_excludes_only_itself():
    f=covariates();ids=f.patient_id.tolist();z,r=design(f,ids,ids[:50],family='state',target_id='Leukocyte_fraction')
    assert set(r['excluded'])=={'leukocyte_fraction'}
    assert set(COMPOSITION)-{'leukocyte_fraction'}<=set(r['encoded'])


@pytest.mark.parametrize('target',['ImmuneScore','StromalScore','ESTIMATEScore','TumorPurity'])
def test_same_source_estimate_purity_only_is_excluded(target):
    f=covariates('ESTIMATE');ids=f.patient_id.tolist();z,r=design(f,ids,ids[:50],family='state',target_id=target)
    assert set(r['excluded'])=={'purity'}
    assert set(COMPOSITION)-{'purity'}<=set(r['encoded'])


def test_cyt_not_automatically_treated_as_estimate_component():
    f=covariates('ESTIMATE');ids=f.patient_id.tolist();_,r=design(f,ids,ids[:50],family='state',target_id='CYT')
    assert not r['excluded'] and 'purity' in r['encoded']


def test_absolute_purity_own_source_not_other_composition_removed():
    f=covariates();ids=f.patient_id.tolist();_,r=design(f,ids,ids[:50],family='state',target_id='ABSOLUTE_purity')
    assert set(r['excluded'])=={'purity'} and 'leukocyte_fraction' in r['encoded']


@pytest.mark.parametrize('case',['missing','unknown','mixed'])
def test_purity_derivative_decision_requires_actual_source_metadata(case):
    f=covariates()
    if case=='missing':f=f.drop(columns='purity_source')
    elif case=='unknown':f['purity_source']='not_sourced'
    else:f.loc[0,'purity_source']='ESTIMATE'
    with pytest.raises((Stop,ValueError)):design(f,f.patient_id.tolist(),f.patient_id[:50].tolist(),family='state',target_id='ImmuneScore')


def test_derivative_source_change_cannot_silently_change_label_adjustment():
    f=covariates();ids=f.patient_id.tolist();f.loc[50:,'purity_source']='ESTIMATE'
    with pytest.raises(Stop):design(f,ids,ids[:50],family='state',target_id='ImmuneScore',admission_cohorts={'source':ids[:50],'label':ids[50:]})


def test_mutation_missing_cancer_rejected_but_sentinel_only_supported(tmp_path):
    f=pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['001'],'gene_id':[SENTINEL],'is_mutated':[False],'mutation_callable':[True],'mutation_count':[0]})
    p=tmp_path/'m.tsv';f.to_csv(p,sep='\t',index=False)
    with pytest.raises(Stop) as e:read_cancer_source(p,'KIRC')
    assert e.value.code=='MUTATION_CANCER_SCOPE_EMPTY'
    c=normalize_mutation(read_cancer_source(p,'BRCA'),['001'],['ENSG1'],'gene','BRCA')
    assert c.report['assay_sentinel_rows']==1 and not c.callable.any() and np.isnan(c.events).all()


def test_mutation_explicit_prefix_is_canonicalized_before_selection(tmp_path):
    f=pd.DataFrame({'cancer_id':['TCGA-BRCA'],'patient_id':['001'],'gene_id':['GENE:ENSG1'],'is_mutated':[True],'mutation_callable':[True],'mutation_count':[1]})
    p=tmp_path/'m.tsv';f.to_csv(p,sep='\t',index=False);o=read_cancer_source(p,'BRCA')
    assert o.patient_id.iloc[0]=='001' and o.cancer_id.iloc[0]=='BRCA'


def test_direct_empty_mutation_normalization_rejected():
    with pytest.raises(Stop) as e:normalize_mutation(pd.DataFrame(columns=['cancer_id']),['P'],['G'],'gene','BRCA')
    assert e.value.code=='MUTATION_CANCER_SCOPE_EMPTY'


def test_pyarrow_empty_and_prefixed_cancer_filter(tmp_path):
    pytest.importorskip('pyarrow',reason='Actual Parquet engine unavailable; not mocked')
    f=pd.DataFrame({'cancer_id':['TCGA-KIRC','BRCA'],'patient_id':['001','002']})
    p=tmp_path/'m.parquet';f.to_parquet(p,index=False)
    assert read_cancer_source(p,'KIRC').patient_id.tolist()==['001']
    with pytest.raises(Stop):read_cancer_source(p,'LUAD')


def baseline_cfg():return baseline_contract(read_json(ROOT/'config/149.json'))


def test_exact_identity_collision_free_and_no_validation_vocabulary_fit():
    rows=[('C',f'L{i}','T') for i in range(5000)];v,r=fit_vocabulary(rows,baseline_cfg())
    assert len(v.vocabulary_)==15003
    encoded=v.transform(identity_records(rows));assert (encoded.getnnz(axis=1)==6).all()
    before=dict(v.vocabulary_);report=unseen_audit([('NEW','L0','Z')],v)
    assert v.vocabulary_==before and report['unseen_terms_by_type']['c']==1
    assert report['unseen_terms_by_type']['l']==0
    # Illustrate actual fixed-hash collisions, not a claim of measured performance loss.
    hasher=FeatureHasher(n_features=4096,input_type='dict',alternate_sign=False)
    hashed=hasher.transform({name:1.} for name in v.feature_names_)
    assert len(set(hashed.indices))<len(v.vocabulary_)


def test_structured_identity_keys_do_not_alias_delimiters():
    rows=[('a|l:b','c','t'),('a','b|l:c','t')]
    v,_=fit_vocabulary(rows,baseline_cfg());x=v.transform(identity_records(rows))
    assert (x[0]!=x[1]).nnz>0


def test_exact_batched_design_and_predictions_equal_eager():
    from cla_engine.exact_baseline import design_block
    rng=np.random.default_rng(31);rows=[('C',f'L{i%7}',f'T{i%3}') for i in range(61)]
    n=rng.normal(size=(61,4));s=StandardScaler().fit(n);v,_=fit_vocabulary(rows,baseline_cfg())
    x=training_design(rows,n,s,v,9);e=design_block(rows,n,s,v)
    np.testing.assert_array_equal(x.toarray(),e.toarray())
    clf=LogisticRegression(solver='liblinear').fit(x,(n[:,0]>0).astype(int))
    np.testing.assert_allclose(decision_batched(clf,rows,n,s,v,7),clf.decision_function(e),atol=1e-12)


def test_exact_memory_and_vocabulary_cap_never_hash_fallback(monkeypatch):
    c=baseline_cfg();c['max_identity_features']=4
    with pytest.raises(Stop) as e:fit_vocabulary([('C','L','T')],c)
    assert e.value.code=='BASELINE_VOCABULARY_RESOURCE_LIMIT'
    monkeypatch.setattr('cla_engine.exact_baseline.available_memory_bytes',lambda:1000)
    with pytest.raises(Stop) as e:resource_plan(100,40,100,baseline_cfg())
    assert e.value.code=='BASELINE_RESOURCE_LIMIT_EXCEEDED'


@pytest.fixture(scope='module')
def real_cpu_prepared(tmp_path_factory):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    root=tmp_path_factory.mktemp('r62_cpu');cfg=make_fixture(root/'sources')
    p=prepare(cfg,root/'prepared',0,'diagnostic')
    return cfg,p


def test_end_to_end_real_source_shapes_do_not_pass_empty_graph(real_cpu_prepared):
    from cla_engine.lineage import validate_prepared,verify_sealed
    cfg,p=real_cpu_prepared
    validate_prepared(p);verify_sealed(p,p.with_name(p.name+'__sealed_test'))
    for view in read_json(p/'VIEWS.json'):
        root=p/'views'/view['id'];a=read_json(root/'COEXPRESSION_AUDIT.json')
        assert a['all_patients_callable_lncrnas']==0 and a['estimable_pairs']>0 and sum(a['selected_edges'].values())>0
        g=read_json(root/'GENOMIC_CONTEXT.json');assert g['binding_context']['global_rows']>0
    b=read_json(p/'BASELINES.json')
    for task in b:assert b[task]['hash_collisions_possible'] is False and b[task]['vocabulary']['fit_scope']=='AVAILABLE_TRAIN_ROWS_ONLY'
    for row in read_json(p/'COVARIATE_AUDIT.json'):
        if row['target_id']=='ImmuneScore':assert 'purity' in row['encoded'] and 'leukocyte_fraction' in row['encoded']
