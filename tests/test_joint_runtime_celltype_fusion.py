from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from cla_engine.common import Stop, atomic_json, sha, read_json
from cla_engine.celltype_fusion import (
    AVAILABLE, PROB, REASON, KEYS, celltype_table, expert_matrix,
    export_table, fit_fusion, contribution_table, ANALYSIS_VERSION,
)
from cc_hhgt.v32.multimodal_fusion import (
    ResidualFusionConfig, pair_blocked_fold, train_pair_blocked_crossfit_fusion,
    apply_residual_fusion_model,
)


def source():
    candidates=pd.DataFrame([['C','L','P'],['C','L2','P']],columns=KEYS)
    rows=[]
    for dataset,cell,probability in [('D1','T_cell',.9),('D2','T_cell',.7),('D1','B_cell',.2),('D1','T_cell::activated',.6)]:
        rows.append(dict(cancer_id='C',lncrna_id='L',pathway_id='P',dataset_id=dataset,cell_type=cell,
                         single_cell_replication_probability=probability,single_cell_available=True,
                         single_cell_unavailable_reason=None,analysis_version=ANALYSIS_VERSION))
    return candidates,pd.DataFrame(rows)


def test_contexts_not_averaged_and_missing_remains_missing():
    candidates,raw=source();long=celltype_table(candidates,raw)
    assert len(long)==3
    assert long.set_index('cell_type').loc['T_cell',PROB]==pytest.approx(.8)
    frame,specs,catalog=expert_matrix(candidates,long,['B_cell','T_cell','T_cell::activated'])
    assert len(frame)==len(candidates)==2
    for item in catalog:
        assert not frame.loc[1,item['expert_id']+'_available']
        assert pd.isna(frame.loc[1,item['expert_id']+'_probability'])
    assert len(set(s.expert_id for s in specs))==3


@pytest.mark.parametrize('problem',['duplicate','family','wrong_version','str_bool','bad_probability','numeric_missing'])
def test_bad_sources_fail_closed(problem):
    candidates,raw=source()
    if problem=='duplicate':raw=pd.concat([raw,raw.iloc[:1]],ignore_index=True)
    if problem=='family':raw.loc[0,'pathway_id']='FAMILY_NOT_EXACT'
    if problem=='wrong_version':raw.loc[0,'analysis_version']='OLD'
    if problem=='str_bool':raw[AVAILABLE]=raw[AVAILABLE].astype(object);raw.loc[0,AVAILABLE]='false'
    if problem=='bad_probability':raw.loc[0,PROB]=1.2
    if problem=='numeric_missing':raw.loc[0,AVAILABLE]=False;raw.loc[0,REASON]='missing'
    with pytest.raises(Stop):celltype_table(candidates,raw)


def test_declared_contexts_cannot_drop_a_celltype():
    candidates,raw=source()
    with pytest.raises(Stop,match='Undeclared'):expert_matrix(candidates,celltype_table(candidates,raw),['T_cell'])


def test_same_pair_all_cancers_has_same_fold_and_labels_not_replicated():
    rows=pd.DataFrame([['C1','L','P'],['C2','L','P']],columns=KEYS)
    assert len(set(pair_blocked_fold(l,p) for l,p in rows[['lncrna_id','pathway_id']].itertuples(index=False,name=None)))==1


def test_synthetic_original_optimizer_consumes_distinct_celltype_columns():
    # Synthetic CPU-only unit test, not a project model or real-data training run.
    rows=[];raw=[]
    for i in range(100):
        row=dict(cancer_id='C',lncrna_id='L'+str(i),pathway_id='P')
        y=float(i%2);rows.append(row|{'primary_probability':.45,'fusion_target':y,
                                    'fusion_pair_fold':pair_blocked_fold(row['lncrna_id'],'P')})
        for cell,prob in [('T_cell',.9 if y else .1),('B_cell',.5)]:
            raw.append(row|dict(dataset_id='D',cell_type=cell,analysis_version=ANALYSIS_VERSION,
                                single_cell_replication_probability=prob,single_cell_available=True,
                                single_cell_unavailable_reason=None))
    primary=pd.DataFrame(rows)
    frame,specs,catalog=expert_matrix(primary,celltype_table(primary[KEYS],pd.DataFrame(raw)),['T_cell','B_cell'])
    settings=ResidualFusionConfig(max_steps=6,evaluation_interval=1,patience=6)
    result=train_pair_blocked_crossfit_fusion(frame,specs,endpoint='discovery',config=settings)
    assert len(result.oof_prediction)==100
    assert len(result.fold_models)==5
    assert all(model.optimiser_steps>0 for model in result.fold_models)
    assert len(result.final_model.weights)==2
    pred=apply_residual_fusion_model(result.final_model,frame)
    np.testing.assert_array_equal(pred.primary_probability,frame.primary_probability)
    assert pred.primary_ranking_unchanged.all()
    long=contribution_table(result.oof_prediction,frame,catalog,'TEST_OOF')
    assert len(long)==200
    assert set(long.cell_type)=={'T_cell','B_cell'}
    assert long.celltype_fused_probability.between(0,1).all()
    assert not long[KEYS+['cell_type']].duplicated().any()
    # With no SC, exact fallback; no invented negative evidence.
    absent=frame.copy()
    for spec in specs:absent[spec.expert_id+'_available']=False;absent[spec.expert_id+'_probability']=np.nan
    fallback=apply_residual_fusion_model(result.final_model,absent)
    np.testing.assert_array_equal(fallback.discovery_adjusted_probability,primary.primary_probability)
    zero=replace(result.final_model,weights=(0.,0.))
    np.testing.assert_array_equal(apply_residual_fusion_model(zero,frame).discovery_adjusted_probability,primary.primary_probability)


def test_training_requires_confirmation_before_reading_any_input(monkeypatch,tmp_path):
    monkeypatch.setattr('cla_engine.celltype_fusion.socket.gethostname',lambda:'149')
    with pytest.raises(Stop) as error:fit_fusion(tmp_path/'missing',tmp_path/'out',deadline=9999999999)
    assert error.value.code=='TRAINING_NOT_AUTHORIZED'
    assert not (tmp_path/'out').exists()


def test_export_real_entrypoint_uses_bound_oof_predictions_without_training(monkeypatch,tmp_path):
    monkeypatch.setattr('cla_engine.celltype_fusion.socket.gethostname',lambda:'149')
    candidates,raw=source();cp=tmp_path/'candidates.parquet';pp=tmp_path/'typed.parquet'
    candidates.to_parquet(cp,index=False);raw.to_parquet(pp,index=False)
    lineage={'analysis_version':ANALYSIS_VERSION,'execution_complete':True,'all_five_fold_private_heads_trained':True,
             'private_head_trained_from_scratch':True,'old_checkpoint_loaded':False,'old_predictions_used_as_features':False,
             'core_parameters_frozen':True,'prediction_path':str(pp),'prediction_sha256':sha(pp)}
    lp=tmp_path/'LINEAGE.json';atomic_json(lp,lineage)
    record=lambda p:{'path':str(p),'sha256':sha(p)}
    bp=tmp_path/'BINDING.json';atomic_json(bp,{'singlecell_lineage':record(lp),'singlecell_predictions':record(pp),'candidates':record(cp)})
    out=tmp_path/'out';export_table(bp,out)
    assert len(pd.read_parquet(out/'SC_CELLTYPE_EXACT_FUSION.parquet'))==3
    assert not read_json(out/'CONTRACT.json')['training_started']
    # A raw rho table or edited score cannot bypass its lineage checksum.
    raw.loc[0,PROB]=.3;raw.to_parquet(pp,index=False)
    with pytest.raises(Stop) as error:export_table(bp,tmp_path/'out2')
    assert error.value.code=='SC_FUSION_SOURCE_DRIFT'


def test_retired_primary_context_is_rejected_before_loading_checkpoint():
    from cla_engine.model import JointHGT, ARCHITECTURE
    spec={'architecture':ARCHITECTURE,'nodes':{'cancer':['C'],'pathway_family':['F']},'singlecell_association_dim':28}
    config={'task_weights':{'pathway':1.,'state':.5},'hidden':8}
    with pytest.raises(Stop) as error:JointHGT(spec,config)
    assert error.value.code=='SC_PRIMARY_INPUT_RETIRED'


def test_synthetic_prepare_and_explicit_fit_end_to_end(monkeypatch,tmp_path):
    from cla_engine import celltype_fusion as module
    from cla_engine.common import commit_manifest, verify_manifest
    monkeypatch.setattr(module.socket,'gethostname',lambda:'149')
    primary=pd.DataFrame([['C','L'+str(i),'P'] for i in range(100)],columns=KEYS)
    typed=pd.DataFrame([dict(cancer_id='C',lncrna_id='L'+str(i),pathway_id='P',dataset_id='D',
        cell_type=cell,analysis_version=ANALYSIS_VERSION,single_cell_available=True,
        single_cell_unavailable_reason=None,single_cell_replication_probability=.9 if i%2 else .1)
        for i in range(100) for cell in ('T_cell','B_cell')])
    sc=tmp_path/'sc';sc.mkdir()
    celltype_table(primary,typed).to_parquet(sc/'SC_CELLTYPE_EXACT_FUSION.parquet',index=False)
    primary.to_parquet(sc/'CANDIDATES.parquet',index=False)
    atomic_json(sc/'CELL_TYPES.json',['T_cell','B_cell']);commit_manifest(sc,{'synthetic_fixture':True})
    p=tmp_path/'primary';p.mkdir();atomic_json(p/'MANIFEST.json',{});atomic_json(p/'VIEWS.json',[])
    spec={'singlecell_association_dim':0,'singlecell_role':'interpretation_only',
          'nodes':{'lncRNA':primary.lncrna_id.tolist(),'pathway':['P']}}
    cfg={'training':{'batch_size':16},'seed':20260909}
    monkeypatch.setattr('cla_engine.lineage.validate_prepared',lambda _:({},spec,cfg,{}))
    monkeypatch.setattr('cla_engine.modality_report._checkpoint_model',lambda *args:(None,1.,{'synthetic_fixture':True}))
    arr={'lnc':np.arange(100),'target':np.zeros(100,dtype=int),'label':np.arange(100)%2,
         'available':np.ones(100,dtype=bool)}
    monkeypatch.setattr('cla_engine.train.evaluate',lambda *args:({}, {'pathway':[('C',arr,np.zeros(100),np.zeros(100))]}))
    native=module.ResidualFusionConfig
    monkeypatch.setattr(module,'ResidualFusionConfig',lambda **kw:native(**(kw|{'max_steps':3,'evaluation_interval':1})))
    plan=tmp_path/'plan';module.prepare_fusion(p,tmp_path/'mock_run',sc,plan)
    assert not read_json(plan/'PLAN.json')['training_authorized']
    assert not read_json(plan/'PLAN.json')['outer_test_read']
    frame=pd.read_parquet(plan/'FUSION_TRAINING_FRAME.parquet')
    assert len(frame)==100
    import time
    out=tmp_path/'fit';module.fit_fusion(plan,out,allow_training=True,deadline=time.time()+60)
    verify_manifest(out)
    receipt=read_json(out/'RECEIPT.json')
    assert receipt['primary_ranking_unchanged'] and not receipt['formal_admission']
    assert all(count>0 for count in receipt['optimizer_steps'])
    oof=pd.read_parquet(out/'SC_CELLTYPE_OOF_CONTRIBUTIONS.parquet')
    assert len(oof)==200 and set(oof.cell_type)=={'T_cell','B_cell'}
