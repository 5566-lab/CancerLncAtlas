"""Selected scores must actually change downstream outputs; synthetic files only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.evidence_rule_summary import KEYS
from cc_hhgt.v32.selected_score import prepare_selected_scores, materialize_selected_scores, load_selected_scores
from cc_hhgt.v32.selected_genesets import (SelectedGeneSetConfig, selected_gene_sets, selected_subtypes,
    add_fold_support, materialize_selected_analysis, audit_selected_analysis, SelectedAnalysisQuery)
from cc_hhgt.v32.subtypes import RankedSubtypeConfig

CANCERS = ['BRCA', 'COAD', 'KIRC', 'LUAD', 'LUSC', 'STAD']


def record(path): return {'path': str(path), 'sha256': sha256(path)}


@pytest.fixture(autouse=True)
def forbid_optimization(monkeypatch):
    import torch
    import cc_hhgt.v32.multimodal_fusion as fusion
    def forbidden(*args, **kwargs): raise AssertionError('No training in selected-analysis checks')
    for name in ('Adam', 'AdamW', 'SGD'): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(fusion, 'fit_residual_fusion_model', forbidden)
    monkeypatch.setattr(fusion, 'train_pair_blocked_crossfit_fusion', forbidden)


def exact_fixture(root, *, unknown=False):
    from cc_hhgt.v32.exact_release import materialize_exact_release
    root.mkdir(parents=True)
    frame = pd.DataFrame([{'cancer_id': cancer, 'pathway_id': pathway, 'lncrna_id': f'LNC:L{i}',
        'association_membership_probability': .8 if i < 4 else .2,
        'l1_probability': .2 if i < 4 else .8, 'ridge_probability': .5,
        'pathway_family_id': 'F1', 'shared_or_local_scope': 'cancer_local' if i == 7 else 'shared',
        'association_direction_probability': .5 if unknown else .8,
        'graph_residual': 0., 'graph_gate': .5, 'held_out_proxy_label': i % 2}
        for cancer in CANCERS for pathway in ['P1','P2'] for i in range(8)])
    paths = {key: [] for key in ('fold_prediction_paths','fold_metrics_paths','training_success_paths','checkpoint_paths')}
    hashes = {key:'a'*64 for key in ('code_sha256','config_sha256','input_manifest_sha256','task_manifest_sha256')}
    for fold in range(5):
        values = frame.copy(); values['patient_fold_id'] = fold
        values['association_membership_probability'] += (fold-2)*.01
        path = root/f'fold{fold}.parquet'; values.to_parquet(path,index=False); paths['fold_prediction_paths'].append(path)
        metric = {'fold':fold,'candidate_alignment':'FULL_ROW_EXACT','checkpoint_cycle':0,'rows':len(frame),
            'artifact_hashes':hashes, **{k:{'rows':len(frame)} for k in ('cc_hhgt','l1','ridge')}}
        path=root/f'metrics{fold}.json';path.write_text(json.dumps(metric));paths['fold_metrics_paths'].append(path)
        success = {'status':'SUCCESS','patient_fold':fold,'did_not_hit_hard_cap':True,'completed_cycles':1,
            'seed':13,'artifact_hashes':hashes,'fixture_only':True}
        path=root/f'success{fold}.json';path.write_text(json.dumps(success));paths['training_success_paths'].append(path)
        path=root/f'checkpoint{fold}.txt';path.write_text(f'SYNTHETIC FOLD {fold}; NO MODEL FITTED');paths['checkpoint_paths'].append(path)
    candidate=root/'candidates.parquet';frame[KEYS].to_parquet(candidate,index=False)
    folds=root/'folds.json';folds.write_text('{}');prep=root/'prep.json';prep.write_text('{}')
    result=materialize_exact_release(**paths,candidate_path=candidate,fold_manifest_path=folds,
        prep_summary_path=prep,output_root=root/'exact',run_id='CURRENT_SYNTHETIC_SELECTED')
    return Path(result['lineage_path'])


def input_manifest(root, producer, *, selected=None, threshold=.5):
    reference={'model_id':'primary','producer_kind':'exact_release','producer_binding':record(producer),
        'run_id':'CURRENT_SYNTHETIC_SELECTED','score_column':'association_membership_probability'}
    payload={'format':'CURRENT_SELECTED_ANALYSIS_INPUT_V1','reference_exact_model':reference,
        'selected_model':selected or reference,
        'selection':{'kind':'EXPLICIT_SOURCE_REGISTRATION','rationale':'Synthetic user-selected source; no winner claim'},
        'geneset':{'min_members':2,'max_members':3,'score_min':threshold},
        'subtype':{'score_column':'selected_score','bootstrap_replicates':4,'min_members':2,'min_cancers':6}}
    path=root/'SELECTED_INPUT.json';path.write_text(json.dumps(payload));return path


def tables(binding_path):
    binding=json.loads(binding_path.read_text())
    return {name:pd.read_parquet(value['path']) for name,value in binding['outputs'].items() if name!='gmt'},binding


def fusion_fixture(root, exact, *, development=False):
    import cc_hhgt.v32.multimodal_fusion as fusion
    root.mkdir()
    lineage=json.loads(exact.read_text());primary=Path(lineage['prediction_path'])
    base=pd.read_parquet(primary)[KEYS]
    expert=base.assign(rho=np.where(base.lncrna_id.isin([f'LNC:L{i}' for i in range(4)]),-.9,.9),available=True,
        analysis_version=fusion.ANALYSIS_VERSION)
    expert.loc[expert.lncrna_id.eq('LNC:L7'),['rho','available']]=[np.nan,False]
    spec=fusion.ExpertSpec('sc','rho','available',('discovery','confidence'),'donor',value_transform='signed_fisher')
    expert_path=root/'sc.parquet';expert.to_parquet(expert_path,index=False)
    zero=base.assign(probability=.6,available=True,analysis_version=fusion.ANALYSIS_VERSION)
    zero_spec=fusion.ExpertSpec('zero','probability','available',('discovery','confidence'),'synthetic')
    zero_path=root/'zero.parquet';zero.to_parquet(zero_path,index=False)
    folds=[Path(r['prediction_path']) for r in lineage['ensemble_source_records']]
    frame,audit=fusion.build_pair_aggregated_fusion_frame_from_parquet(primary,folds,{spec:expert_path,zero_spec:zero_path})
    audit.update(core_run_id=lineage['training_run_id'],primary_source_sha256=sha256(primary),synthetic_only=True)
    if development:
        from cc_hhgt.v32.current_primary_fusion import SCOPE
        frame['fusion_supervision_scope']=SCOPE
        frame['fusion_target_available']=True
        frame['fusion_target_available_fold_count']=5
        frame['fusion_target_unavailable_reason']=None
        audit.update(supervision_scope=SCOPE,independent_validation_claimed=False)
    results=[]
    for endpoint in ('discovery','confidence'):
        model=fusion.ResidualFusionModel(endpoint,('sc','zero'),(0.,0.),(2.,0.),1,'0'*64,'1'*64,0,0.,
            feature_transforms=('signed_fisher','probability_logit'))
        results.append(fusion.CrossfitFusionResult(endpoint,(spec,zero_spec),fusion.apply_residual_fusion_model(model,frame),
            pd.DataFrame({'synthetic_only':[True]}),(model,)*5,model,
            'DEVELOPMENT_NO_INCREMENT' if development else 'SYNTHETIC_NOT_ESTIMATED',
            'DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED' if development else 'SYNTHETIC_NOT_ESTIMATED'))
    sources=[{**record(path),'source_role':'current_v32_oof_prediction','generation':'current_v32'}
             for path in [primary,*folds,expert_path,zero_path]]
    result=fusion.materialize_multimodal_fusion_release(root/'release',frame,*results,source_declarations=sources,
        training_run_id='V32-FUSION-SYNTHETIC_SELECTED',input_audit=audit)
    binding=Path(result['binding_path'])
    return {'model_id':'fusion','producer_kind':'fusion_release','producer_binding':record(binding),
        'run_id':'V32-FUSION-SYNTHETIC_SELECTED','score_column':'discovery_adjusted_probability'}


def test_existing_cli_exact_selected_score_rank_gmt_full_coverage_and_query(tmp_path):
    from scripts.materialize_v32_release import main
    producer=exact_fixture(tmp_path/'exact');manifest=input_manifest(tmp_path,producer)
    assert main(['--selected-manifest',str(manifest),'--output',str(tmp_path/'analysis')])==0
    binding_path=tmp_path/'analysis/SELECTED_ANALYSIS_BINDING.json'
    data,binding=tables(binding_path);members=data['geneset_members']
    assert set(members.lncrna_id)=={'LNC:L0','LNC:L1','LNC:L2'}
    assert len(data['geneset_master'])==12 and len(data['geneset_coverage'])==12
    assert data['geneset_coverage'].boundary_tie_count.eq(4).all()
    assert np.allclose(members.selected_score_fold_sd,np.std([.78,.79,.8,.81,.82],ddof=1))
    assert members.selected_fold_selection_frequency.eq(1).all()
    assert members.primary_direction_fold_agreement.eq(1).all()
    assert data['pathway_context_subtype'].classification_status.eq('CONSERVED_K1').all()
    assert not data['pathway_context_subtype'].subtype_feedback_to_model.any()
    query=SelectedAnalysisQuery(binding_path,expected_sha=sha256(binding_path))
    result=query.query(filters={'cancer_id':'BRCA','pathway_id':'P1'})
    assert [r['lncrna_id'] for r in result['rows']]==['LNC:L0','LNC:L1','LNC:L2']
    assert query.query(filters={'cancer_id':"BRCA' OR true --"})['rows']==[]
    gmt=Path(query.download('gmt')['path']).read_text()
    assert len(gmt.splitlines())==12 and all(line.endswith('\tLNC:L0\tLNC:L1\tLNC:L2') for line in gmt.splitlines())
    assert binding['selected_score_run_id'] in gmt
    assert not {'held_out_proxy_label','fusion_target'} & set(members)
    from scripts.audit_v32_selected_analysis import main as audit_main
    assert audit_main(['--binding',str(binding_path),'--expected-sha256',sha256(binding_path),
        '--output',str(tmp_path/'REAUDIT.json')])==0
    assert json.loads((tmp_path/'REAUDIT.json.preflight.json').read_text())['target_host']=='149'


def test_selecting_baseline_changes_members_without_rewriting_primary_or_direction(tmp_path):
    producer=exact_fixture(tmp_path/'exact')
    path=input_manifest(tmp_path,producer);manifest=json.loads(path.read_text())
    manifest['selected_model']={**manifest['selected_model'],'model_id':'baseline','score_column':'l1_probability'}
    path.write_text(json.dumps(manifest))
    result=materialize_selected_analysis(path,tmp_path/'analysis');data,binding=tables(Path(result['binding_path']))
    assert set(data['geneset_members'].lncrna_id)=={'LNC:L4','LNC:L5','LNC:L6'}
    assert np.allclose(data['geneset_members'].association_membership_probability,.2)
    assert np.allclose(data['geneset_members'].selected_score,.8)
    scores,score_binding=load_selected_scores(binding['selected_score_binding']['path'],expected_sha=binding['selected_score_binding']['sha256'])
    assert not score_binding['sources']['selection']['validated_winner_claimed']
    assert scores.association_direction.eq('positive').all()
    assert scores.direction_source_run_id.eq('CURRENT_SYNTHETIC_SELECTED').all()


@pytest.mark.parametrize('column',['discovery_adjusted_probability','fused_confidence_probability'])
def test_actual_frozen_fusion_drives_gene_sets_and_preserves_missing_fold_support(tmp_path,column):
    producer=exact_fixture(tmp_path/'exact');selected=fusion_fixture(tmp_path/'fusion',producer)
    selected['score_column']=column
    manifest=input_manifest(tmp_path,producer,selected=selected)
    result=materialize_selected_analysis(manifest,tmp_path/'analysis');data,binding=tables(Path(result['binding_path']))
    members=data['geneset_members']
    assert set(members.lncrna_id)=={'LNC:L4','LNC:L5','LNC:L6'}
    assert members.selected_score_fold_sd.isna().all() and members.selected_fold_selection_frequency.isna().all()
    assert members.selected_score_fold_count.eq(0).all() and not members.selected_fold_estimates_available.any()
    assert members.selected_aux_sc_contributed.all() and members.selected_aux_zero_available.all()
    assert not members.selected_aux_zero_contributed.any()
    scores,_=load_selected_scores(binding['selected_score_binding']['path'],expected_sha=binding['selected_score_binding']['sha256'])
    fallback=scores.loc[scores.lncrna_id.eq('LNC:L7')]
    assert fallback.selected_primary_fallback.all() and not fallback.selected_aux_sc_available.any()
    assert np.array_equal(fallback.selected_score,fallback.association_membership_probability)
    assert data['geneset_coverage'].selected_aux_sc_available_rows.eq(7).all()
    assert data['geneset_coverage'].selected_aux_zero_contributed_rows.eq(0).all()
    assert members.score_scientific_status.eq('SYNTHETIC_NOT_ESTIMATED').all()
    private=json.loads(Path(selected['producer_binding']['path']).read_text())['artifacts']['fusion_input_private']
    assert private['public'] is False


def test_development_validation_reuse_status_reaches_actual_selected_geneset_members(tmp_path):
    producer=exact_fixture(tmp_path/'exact')
    selected=fusion_fixture(tmp_path/'fusion',producer,development=True)
    manifest=input_manifest(tmp_path,producer,selected=selected)
    result=materialize_selected_analysis(manifest,tmp_path/'analysis')
    data,binding=tables(Path(result['binding_path']))
    members=data['geneset_members']
    assert len(members)>0 and members.score_scientific_status.eq('DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED').all()
    assert members.score_performance_outcome.eq('DEVELOPMENT_NO_INCREMENT').all()
    scores,_=load_selected_scores(binding['selected_score_binding']['path'],expected_sha=binding['selected_score_binding']['sha256'])
    assert scores.score_scientific_status.eq('DEVELOPMENT_ONLY_PRIMARY_VALIDATION_REUSED').all()
    assert 'validation_proxy_label' not in scores and 'fusion_target' not in members


@pytest.mark.parametrize('unknown,threshold',[(True,.5),(False,1.)])
def test_unknown_direction_and_empty_gene_sets_keep_full_unavailable_subtype_coverage(tmp_path,unknown,threshold):
    producer=exact_fixture(tmp_path/'exact',unknown=unknown);path=input_manifest(tmp_path,producer,threshold=threshold)
    result=materialize_selected_analysis(path,tmp_path/'analysis');data,binding=tables(Path(result['binding_path']))
    context=data['pathway_context_subtype'];program=data['cancer_program_subtype']
    assert len(context)==12 and len(program)==6
    assert not context.available.any() and context.selected_k.isna().all() and context.pathway_context_subtype_id.isna().all()
    assert not program.available.any() and program.cancer_program_subtype_id.isna().all()
    assert context.unavailable_reason.eq('NO_MODEL_DIRECTION_FOR_RANKED_SUBTYPE' if unknown else 'NO_GENESET_PASSING_THRESHOLDS').all()
    if unknown:
        assert data['geneset_members'].association_direction.eq('unknown').all()
        assert data['geneset_members'].primary_direction_fold_agreement.isna().all()
    else:
        assert data['geneset_master'].empty and data['geneset_members'].empty
        assert Path(binding['outputs']['gmt']['path']).read_text()==''


def test_selected_weights_drive_similarity_and_local_or_rule_evidence_cannot_change_it():
    frame=pd.DataFrame([{'cancer_id':cancer,'pathway_id':'P1','pathway_family_id':'F1','lncrna_id':f'L{i}',
        'selected_score':weight*scale,'association_membership_probability':.5,'association_direction':'positive',
        'shared_or_local_scope':'shared' if i<2 else 'cancer_local','selected_score_run_id':'SELECTED:fixture'}
        for cancer,scale in [('BRCA',1.),('LUAD',.5)] for i,weight in enumerate([.8,.4,1.])])
    config=SelectedGeneSetConfig(min_members=2,max_members=3,score_min=0.)
    _,members,_=selected_gene_sets(frame,config)
    subtype=RankedSubtypeConfig(score_column='selected_score',min_members=2,min_cancers=2,bootstrap_replicates=3)
    result=selected_subtypes(frame,members,subtype)
    pair=result['pathway_pairwise_similarity'].iloc[0]
    assert pair.weighted_jaccard_similarity==pytest.approx(.5)
    assert pair.rbo_similarity==pytest.approx(1.) and pair.combined_similarity==pytest.approx(.75)
    changed=frame.assign(rule_confidence=np.arange(len(frame)),association_membership_probability=.99)
    changed.loc[changed.shared_or_local_scope.eq('cancer_local'),'selected_score']=.01
    _,new_members,_=selected_gene_sets(changed,config)
    after=selected_subtypes(changed,new_members,subtype)
    pd.testing.assert_frame_equal(result['pathway_pairwise_similarity'],after['pathway_pairwise_similarity'])


@pytest.mark.parametrize('target',['member','gmt','selected_score','primary_direction'])
def test_rehashed_downstream_or_direction_edits_fail_source_replay(tmp_path,target):
    producer=exact_fixture(tmp_path/'exact');manifest=input_manifest(tmp_path,producer)
    result=materialize_selected_analysis(manifest,tmp_path/'analysis');path=Path(result['binding_path']);binding=json.loads(path.read_text())
    if target in {'member','gmt'}:
        output=binding['outputs']['geneset_members' if target=='member' else 'gmt'];p=Path(output['path'])
        if target=='member':
            data=pd.read_parquet(p);data.loc[0,'selected_score']=.999;data.to_parquet(p,index=False)
        else:p.write_text(p.read_text().replace('LNC:L0','LNC:FAKE'))
        output['sha256']=sha256(p);path.write_text(json.dumps(binding))
        with pytest.raises(ValueError,match='recomputation|GMT'):audit_selected_analysis(path,expected_sha=sha256(path))
    elif target=='selected_score':
        path=Path(binding['selected_score_binding']['path']);payload=json.loads(path.read_text());p=Path(payload['scores']['path'])
        data=pd.read_parquet(p);data.loc[0,'selected_score']=.999;data.to_parquet(p,index=False)
        payload['scores']['sha256']=sha256(p);path.write_text(json.dumps(payload))
        (path.parent/'SUCCESS.json').write_text(json.dumps({'status':'PASS','binding_sha256':sha256(path)}))
        with pytest.raises(ValueError,match='actual producers'):load_selected_scores(path,expected_sha=sha256(path))
    else:
        lineage=json.loads(producer.read_text());p=Path(lineage['prediction_path']);data=pd.read_parquet(p)
        data.loc[0,'association_direction_probability']=.1;data.to_parquet(p,index=False)
        lineage['prediction_sha256']=sha256(p);producer.write_text(json.dumps(lineage))
        payload=json.loads(manifest.read_text())
        payload['selected_model']['producer_binding']=record(producer);payload['reference_exact_model']['producer_binding']=record(producer)
        manifest.write_text(json.dumps(payload))
        with pytest.raises(ValueError,match='annotations'):prepare_selected_scores(manifest)


def test_selected_snapshot_is_consumed_by_actual_external_metrics_without_new_selection(tmp_path):
    from cc_hhgt.v32.external_validation_current import evaluate_manifest
    producer=exact_fixture(tmp_path/'exact');path=input_manifest(tmp_path,producer)
    selected=materialize_selected_scores(path,tmp_path/'score')
    source={'model_id':'selected','producer_kind':'selected_release','producer_binding':record(selected),
        'run_id':'CURRENT_SYNTHETIC_SELECTED','score_column':'selected_score'}
    frames={
        'crosswalk':pd.DataFrame([('lncrna',f'L{i}',f'LNC:L{i}') for i in range(8)]+[('cancer','breast','BRCA'),('pathway','p1','P1')],columns=['namespace','raw_id','canonical_id']),
        'used':pd.DataFrame({'lncrna_id':['LNC:L7'],'pathway_id':['P1'],'cancer_id':['BRCA'],'pmid':['99999999']}),
        'external':pd.DataFrame({'record_id':[f'OBS{i}' for i in range(6)],'lncrna':[f'L{i}' for i in range(6)],'cancer':'breast','pathway':'p1','label':[1,1,1,1,0,0],
            'pmid':[str(12345000+i) for i in range(6)],'is_experimental':True,'is_predicted':False,'assay':'Synthetic measured assay'})}
    files={}
    for name,frame in frames.items():
        p=tmp_path/(name+'.parquet');frame.to_parquet(p,index=False);files[name]=record(p)
    manifest={'format':'CURRENT_EXTERNAL_VALIDATION_INPUT_V1','identity_crosswalk':files['crosswalk'],'models':[source],
        'model_use_sources':[{'kind':'canonical_usage','source_id':'used','phase':'TRAINING','model_ids':['selected'],'table':files['used']}],
        'external_sources':[{'kind':'exact_assay','source_id':'assay','table':files['external'],'measured_negative_definition':'Measured null effect'}],
        'evaluation':{'primary_model':'selected','cutoffs':[3],'lncrna_aggregation':'MAX','bootstrap_replicates':0,'seed':0}}
    p=tmp_path/'EXTERNAL.json';p.write_text(json.dumps(manifest))
    _,outputs,sources=evaluate_manifest(p)
    metrics=outputs['metrics'];value=metrics.loc[metrics.metric.eq('auroc') & metrics.available,'value']
    assert not value.empty and value.eq(1.).all()
    assert sources[0]['producer_binding']['sha256']==sha256(selected)


def test_fold_membership_support_obeys_the_same_top_n_cap(tmp_path):
    producer=exact_fixture(tmp_path/'exact');manifest=input_manifest(tmp_path,producer)
    path=materialize_selected_scores(manifest,tmp_path/'score')
    scores,binding=load_selected_scores(path,expected_sha=sha256(path))
    supported=add_fold_support(scores,binding,SelectedGeneSetConfig(min_members=2,max_members=3,score_min=.5))
    assert supported.loc[supported.lncrna_id.eq('LNC:L2'),'selected_fold_selection_frequency'].eq(1).all()
    assert supported.loc[supported.lncrna_id.eq('LNC:L3'),'selected_fold_selection_frequency'].eq(0).all()


def validation_lock(root,producer):
    from cc_hhgt.v32 import sealed_test_inference as sealed
    lineage=json.loads(producer.read_text())
    lock={'format':sealed.WINNER_DECLARATION_FORMAT,'status':'PASS_VALIDATION_ONLY_WINNER_LOCK',
        'compared_graph_variants':['G0','G1','G2'],'winner_id':'G2','graph_variant':'G2','selection_scope':'VALIDATION_ONLY',
        'heldout_test_metrics_used_for_selection':False,'test_inputs_opened_before_winner_lock':False,
        'winner_or_checkpoint_changed_by_test':False,'declaration_locked':True,'old_checkpoint_loaded':False,'old_predictions_used':False,
        'sample_patient_fold_map_sha256':sealed.FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        'authority_receipt_sha256':sealed.FROZEN_V32_RECEIPT_SHA256,
        'candidate_authority_sha256':lineage['input_artifacts'][0]['sha256'],'pair_fold_seed':13,
        'selection_table':[{'graph_variant':f'G{i}','fold_validation_losses':[loss]*5,'mean_validation_loss':loss}
                           for i,loss in enumerate([.6,.5,.4])], 'prepared_folds':[],'checkpoints':[]}
    for item in lineage['ensemble_source_records']:
        fold=item['patient_fold'];prepared=root/f'prepared{fold}.txt';prepared.write_text(f'SYNTHETIC PREPARED FOLD {fold}')
        lock['prepared_folds'].append({**record(prepared),'patient_fold':fold,'graph_variant':'G2'})
        lock['checkpoints'].append({'path':item['checkpoint_path'],'sha256':item['checkpoint_sha256'],
            'patient_fold':fold,'graph_variant':'G2','prepared_fold_sha256':sha256(prepared),
            'selected_using':'VALIDATION_ONLY','heldout_test_metrics_used':False,
            'training_artifact_hashes':json.loads(Path(item['success_path']).read_text())['artifact_hashes']})
    return lock


@pytest.mark.parametrize('corruption',[None,'winner','candidate','training','external','mean'])
def test_existing_g012_lock_is_verified_against_actual_selected_sources(tmp_path,corruption):
    producer=exact_fixture(tmp_path/'exact');path=input_manifest(tmp_path,producer)
    lock=validation_lock(tmp_path,producer)
    if corruption=='winner': lock['selection_table'][0].update(mean_validation_loss=.1,fold_validation_losses=[.1]*5)
    if corruption=='candidate': lock['candidate_authority_sha256']='f'*64
    if corruption=='training': lock['checkpoints'][0]['training_artifact_hashes']['config_sha256']='f'*64
    if corruption=='external': lock['heldout_test_metrics_used_for_selection']=True
    if corruption=='mean': lock['selection_table'][2]['mean_validation_loss']=.1
    lock_path=tmp_path/'LOCK.json';lock_path.write_text(json.dumps(lock))
    manifest=json.loads(path.read_text());manifest['selection']={'kind':'G012_VALIDATION_WINNER','binding':record(lock_path)}
    path.write_text(json.dumps(manifest))
    if corruption:
        from cc_hhgt.v32.sealed_test_inference import SealedTestInferenceError
        error = SealedTestInferenceError if corruption == 'external' else ValueError
        with pytest.raises(error): prepare_selected_scores(path)
    else:
        _,_,sources=prepare_selected_scores(path)
        assert sources['selection']['winner_id']=='G2' and sources['selection']['validated_winner_claimed']
        assert not sources['selection']['new_model_selection_performed']


def test_selected_fusion_must_have_consumed_the_reference_primary(tmp_path):
    producer=exact_fixture(tmp_path/'exact');selected=fusion_fixture(tmp_path/'fusion',producer)
    binding_path=Path(selected['producer_binding']['path']);binding=json.loads(binding_path.read_text())
    audit_path=Path(binding['artifacts']['fusion_input_audit']['path']);audit=json.loads(audit_path.read_text())
    audit['primary_source_sha256']='f'*64;audit_path.write_text(json.dumps(audit))
    binding['artifacts']['fusion_input_audit']['sha256']=sha256(audit_path);binding_path.write_text(json.dumps(binding))
    selected['producer_binding']=record(binding_path)
    path=input_manifest(tmp_path,producer,selected=selected)
    with pytest.raises(ValueError,match='different primary runs'):prepare_selected_scores(path)


def test_rule_source_annotations_reach_members_without_changing_ranks_or_subtypes(tmp_path):
    from test_v32_current_rule_evidence import native_fixture
    from scripts.materialize_v32_current_rule_evidence import main as rule_main
    producer=exact_fixture(tmp_path/'exact');manifest=input_manifest(tmp_path,producer)
    first=materialize_selected_analysis(manifest,tmp_path/'before');before,_=tables(Path(first['binding_path']))
    raw,lineage,membership=native_fixture()
    universe=pd.read_parquet(json.loads(producer.read_text())['prediction_path'])[KEYS]
    records={}
    for name,frame in {'raw':raw,'lineage':lineage,'membership':membership,'universe':universe}.items():
        path=tmp_path/(name+'.parquet');frame.to_parquet(path,index=False);records[name]=record(path)
    rule_input={'format':'CURRENT_RULE_EVIDENCE_INPUT_V1','candidates':records['universe'],'exact_membership':records['membership'],
        'sources':[{'source_id':'native','kind':'native_exact_events','events':records['raw'],'lineage':records['lineage']}]}
    path=tmp_path/'RULE_INPUT.json';path.write_text(json.dumps(rule_input))
    assert rule_main(['--manifest',str(path),'--output',str(tmp_path/'rules')])==0
    payload=json.loads(manifest.read_text());payload['rule_evidence_binding']=record(tmp_path/'rules/RULE_EVIDENCE_BINDING.json')
    # Preserve the first manifest, which remains a dependency of its completed release.
    with_rules=tmp_path/'WITH_RULES.json';with_rules.write_text(json.dumps(payload))
    second=materialize_selected_analysis(with_rules,tmp_path/'after');after,_=tables(Path(second['binding_path']))
    columns=[*KEYS,'geneset_rank','selected_score','association_membership_probability','association_direction']
    pd.testing.assert_frame_equal(before['geneset_members'][columns],after['geneset_members'][columns])
    assert after['geneset_members'].rule_independent_studies.max()>0
    columns=[c for c in before['pathway_pairwise_similarity'] if c not in {'selected_score_run_id','selected_score_binding_sha256'}]
    pd.testing.assert_frame_equal(before['pathway_pairwise_similarity'][columns],after['pathway_pairwise_similarity'][columns])


def test_candidate_authority_cannot_be_replaced_by_a_smaller_declared_universe(tmp_path):
    producer=exact_fixture(tmp_path/'exact');payload=json.loads(producer.read_text())
    path=Path(payload['input_artifacts'][0]['path']);frame=pd.read_parquet(path).iloc[:-1];frame.to_parquet(path,index=False)
    payload['input_artifacts'][0]['sha256']=sha256(path);producer.write_text(json.dumps(payload))
    manifest=input_manifest(tmp_path,producer)
    with pytest.raises(ValueError,match='candidate authority'):prepare_selected_scores(manifest)
