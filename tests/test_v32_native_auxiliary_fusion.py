"""Native serializers and real fusion consumers; synthetic sources, no optimizer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256
from cc_hhgt.v32.native_auxiliary_fusion import (INPUT_FORMAT, materialize_native_auxiliary, load_native_auxiliary)
from cc_hhgt.v32.multimodal_fusion import ANALYSIS_VERSION, PRIMARY_COLUMN, ResidualFusionModel, apply_residual_fusion_model
from test_v32_current_core_export import export_fixture


def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n');return path
def read(path): return json.loads(path.read_text())
def record(path): return {'path':str(path),'sha256':sha256(path)}


@pytest.fixture(autouse=True)
def no_fitting(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('No optimization permitted in native adapter checks')
    for name in ('Adam','AdamW','SGD'):monkeypatch.setattr(torch.optim,name,forbidden)
    import cc_hhgt.v32.genomic_training as genomic
    import cc_hhgt.v32.atac_training as atac
    import scripts.run_v32_current_multimodal_fusion as runner
    monkeypatch.setattr(genomic,'_fit_head',forbidden)
    monkeypatch.setattr(atac,'_fit_logistic',forbidden)
    monkeypatch.setattr(runner,'train_pair_blocked_crossfit_fusion',forbidden)


def fixture(root,kind='genomic'):
    root.mkdir(parents=True,exist_ok=True)
    export=root/'core';export.mkdir();core_path,core=export_fixture(export)
    # These are frozen serialization fixtures; no source fit took place.
    core['fixture_no_optimizer_run']=True
    for fold in range(5):
        p=export/core['folds'][str(fold)]['exports']['lncRNA']['path']
        pd.DataFrame({'node_index':range(80),'node_id':[f'LNC:L{i}' for i in range(80)],
            'core_feature_000':np.arange(80,dtype=float),'core_feature_001':1.}).to_parquet(p,index=False)
        core['folds'][str(fold)]['exports']['lncRNA']={**record(p),'rows':80,'features':2}
    write(core_path,core)
    keys=pd.DataFrame([(c,f'LNC:L{i}','PATHWAY:P1') for c in ('BRCA','LAML') for i in range(80)],columns=['cancer_id','lncrna_id','pathway_id'])
    candidate=root/'candidates.parquet';keys.to_parquet(candidate,index=False)
    patients=root/'patients.parquet'
    pd.DataFrame([(c,f'{c}-patient-{fold}',fold) for c in ('BRCA','LAML') for fold in range(5)],
        columns=['cancer_id','patient_id','patient_fold_id']).to_parquet(patients,index=False)
    native=root/'native';native.mkdir();(native/'checkpoints').mkdir()
    modalities=['mutation','cnv'] if kind=='genomic' else ['cnv'] if kind=='cnv_event' else [kind]
    run=f'v32-{kind}-synthetic';cp_records=[];fold_records=[];fold_frames=[]
    inputs={'candidates':sha256(candidate),'folds':sha256(patients)}
    for fold in range(5):
        for modality in modalities:
            cp=native/'checkpoints'/f'{modality}_patient_fold_{fold}.pt'
            if kind=='atac':
                cp=cp.with_suffix('.json')
                write(cp,{'checkpoint_format':'CC_HHGT_V3_2_ATAC_FOLD_CALIBRATOR_V1','patient_fold':fold,
                    'modality':modality,'training_run_id':run,'outer_test_patients_used_for_fit':False,
                    'input_hashes':inputs,'fixture_no_optimizer_run':True,'feature_mean':[0.],
                    'feature_scale':[1.],'weights_with_intercept':[0.,1.]})
            else:
                from cc_hhgt.v32.integrated_model import build_private_auxiliary_head
                head,_=build_private_auxiliary_head(f'mutation_cnv_{modality}',core_features=4,
                    domain_features=2,hidden_features=4,dropout=0.,seed=fold)
                torch.save({'checkpoint_format':'CC_HHGT_V3_2_GENOMIC_PRIVATE_HEAD_V1','patient_fold':fold,
                    'modality':modality,'core_parameter_sha256':core['folds'][str(fold)]['core_parameter_sha256'],
                    'core_checkpoint_sha256':core['folds'][str(fold)]['checkpoint_sha256'],
                    'model_state':head.state_dict(),'history':[{'fixture_not_training':True}],
                    'fixture_no_optimizer_run':True},cp)
            cp_records.append({**record(cp),'patient_fold':fold,'fold':fold,'modality':modality,'status':'SUCCESS'})
        available=keys.cancer_id.eq('BRCA').to_numpy(copy=True)
        if fold==4:available &= ~keys.lncrna_id.eq('LNC:L1').to_numpy()
        values=np.full(len(keys),(.2+.15*fold),np.float32)
        if fold==0:values[keys.lncrna_id.eq('LNC:L0')]=0.
        values[~available]=np.nan
        reasons=np.where(available,None,'NATIVE_INPUT_NOT_AVAILABLE')
        if kind=='atac':
            from cc_hhgt.v32.atac_training import _fold_frame
            frame=_fold_frame(keys,fold,values,reasons,4,1)
        elif kind in {'cnv','cnv_event'}:
            frame=keys.assign(fold_id=fold,cnv_probability=values,cnv_available=available,
                unavailable_reason=reasons,run_id=run,checkpoint_sha256=cp_records[-1]['sha256'],
                core_manifest_sha256=sha256(core_path),candidate_authority_sha256=sha256(candidate),
                fold_manifest_sha256=sha256(patients),callable_patient_n=available.astype(int),
                train_patient_n=3,validation_patient_n=1,oof_patient_n=1)
            if kind=='cnv_event':
                frame=frame.rename(columns={'fold_id':'patient_fold','unavailable_reason':'cnv_unavailable_reason','run_id':'training_run_id'})
        else:
            frame=keys.assign(patient_fold_id=fold)
            for modality in modalities:
                frame[f'{modality}_context_probability']=values
                frame[f'{modality}_available']=available
        frame['unused_observed_label']='DO_NOT_CONSUME_THIS_COLUMN'
        fold_frames.append(frame)
        if kind in {'cnv','cnv_event'}:
            for cancer in ('BRCA','LAML'):
                path=native/f'fold_{fold}_{cancer}.parquet';frame.loc[frame.cancer_id.eq(cancer)].to_parquet(path,index=False)
                fold_records.append({**record(path),'fold':fold,'cancer':cancer,'rows':80})
        else:
            path=native/f'fold_{fold}.parquet';frame.to_parquet(path,index=False)
            fold_records.append({**record(path),'patient_fold_id':fold,'rows':len(frame)})
        if kind=='atac':
            cp_record=cp_records[-1]
            success=write(native/'checkpoints'/f'atac_patient_fold_{fold}.SUCCESS.json',{
                'status':'SUCCESS','patient_fold':fold,'checkpoint_sha256':cp_record['sha256'],
                'prediction_sha256':sha256(path),'heldout_patients_used_for_fit':False})
            cp_record.update(success_path=str(success),success_sha256=sha256(success),
                prediction_path=str(path),prediction_sha256=sha256(path))
    if kind in {'cnv','cnv_event'}:
        native_path=write(native/'OOF_MANIFEST.json',{
            'format':'CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_V2','status':'READY_FOR_INDEPENDENT_AUDIT',
            'training_run_id':run,'cnv_only':True,'five_fold_oof':True,'mutation_features_used':False,
            'target':'current_fold_local_unadjusted_lncrna_exact_pathway_association',
            'candidate_authority_sha256':sha256(candidate),'fold_manifest_sha256':sha256(patients),
            'core_manifest_sha256':sha256(core_path),'core_parameters_before_sha256':{'fixture':'a'*64},
            'core_parameters_after_sha256':{'fixture':'a'*64},'checkpoints':cp_records,
            'predictions':fold_records,'prediction_file_count':len(fold_records),'fixture_no_optimizer_run':True})
        if kind=='cnv_event':
            from cc_hhgt.v32.cnv_only_training import write_cnv_event_oof_manifest
            # Exercise the actual production serializer, without entering its trainer.
            native_path.unlink()
            write(native/'INPUT_PRECHECK.json',{'status':'PASS','candidates_path':str(candidate),
                'candidates_sha256':sha256(candidate),'folds_path':str(patients),'folds_sha256':sha256(patients),
                'core_manifest_path':str(core_path)})
            write_cnv_event_oof_manifest(native,training_run_id=run,checkpoints=cp_records,predictions=fold_records,
                core_manifest_sha=sha256(core_path),core_composite='a'*64,
                core_hashes_before={'fixture':'a'*64},core_hashes_after={'fixture':'a'*64})
    else:
        typed=keys.copy()
        for modality in modalities:
            col=f'{modality}_context_probability'
            matrix=np.stack([f[col].to_numpy(float) for f in fold_frames]);count=np.isfinite(matrix).sum(axis=0)
            typed[col]=np.divide(np.nansum(matrix,axis=0),count,out=np.full(len(keys),np.nan),where=count>0)
            typed[f'{modality}_available']=count>0
            typed[f'{modality}_patient_folds_with_prediction']=count
            typed[f'{modality}_unavailable_reason']=np.where(count>0,None,'NATIVE_INPUT_NOT_AVAILABLE')
        typed['analysis_version']=ANALYSIS_VERSION;typed['training_run_id']=run
        typed_path=native/'typed.parquet';typed.to_parquet(typed_path,index=False)
        cp_manifest=write(native/'CHECKPOINT_MANIFEST.json',{'records':cp_records})
        inputs_audit=write(native/'INPUT_LINEAGE_AUDIT.json',{'input_artifacts':[record(candidate),record(patients)]})
        lineage={'training_status':'SUCCESS','training_run_id':run,'folds':5,
            'patient_fold_oof_predictions_not_fold_averaged':True,'input_artifacts':[record(candidate),record(patients)],
            'patient_fold_oof_prediction_paths':fold_records,'prediction_sha256':sha256(typed_path),
            'predictions_sha256':sha256(typed_path),'checkpoint_manifest_path':str(cp_manifest),
            'checkpoint_manifest_sha256':sha256(cp_manifest),'v32_core_checkpoint_sha256':sha256(core_path),
            'modalities':{m:[{'patient_fold':f,'status':'SUCCESS'} for f in range(5)] for m in modalities},
            'outer_test_patients_used_in_any_upstream_fit':0,'input_lineage_audit_path':str(inputs_audit),
            'input_lineage_audit_sha256':sha256(inputs_audit),'fixture_no_optimizer_run':True}
        lineage_path=write(native/'LINEAGE.json',lineage)
        native_path=write(native/'SUCCESS.json',{'status':'SUCCESS','training_run_id':run,
            'lineage_path':str(lineage_path),'lineage_sha256':sha256(lineage_path),
            'prediction_path':str(typed_path),'prediction_sha256':sha256(typed_path),
            'patient_fold_oof_prediction_paths':fold_records,'release_ready':False,'fixture_no_optimizer_run':True})
    manifest={'format':INPUT_FORMAT,'producer_kind':kind,'modalities':modalities,'core_run_id':'current-G2',
        'native_run_id':run,'candidate_authority':record(candidate),'patient_folds':record(patients),
        'core_manifest':record(core_path),'native_receipt':record(native_path)}
    return write(root/'manifest.json',manifest),keys


def materialize(path,output):return materialize_native_auxiliary(path,output,expected_core_run_id='current-G2')


def repin_native(path):
    manifest=read(path);native_path=Path(manifest['native_receipt']['path']);native=read(native_path)
    if manifest['producer_kind']!='cnv':
        lp=Path(native['lineage_path']);lineage=read(lp)
        lineage['checkpoint_manifest_sha256']=sha256(Path(lineage['checkpoint_manifest_path']))
        lineage['prediction_sha256']=lineage['predictions_sha256']=sha256(Path(native['prediction_path']))
        write(lp,lineage);native['lineage_sha256']=sha256(lp);native['prediction_sha256']=lineage['prediction_sha256']
        native['patient_fold_oof_prediction_paths']=lineage['patient_fold_oof_prediction_paths'];write(native_path,native)
    manifest['native_receipt']=record(native_path);write(path,manifest)


@pytest.mark.parametrize('kind',['genomic','cnv','cnv_event','atac'])
def test_native_command_to_real_fusion_preserves_estimand_folds_missingness_and_exact_fallback(tmp_path,kind):
    from scripts.materialize_v32_native_auxiliary_fusion import main
    import scripts.run_v32_current_multimodal_fusion as runner
    path,keys=fixture(tmp_path,kind);output=tmp_path/'adapter'
    assert main(['--manifest',str(path),'--output',str(output),'--expected-core-run-id','current-G2'])==0
    data,audit=load_native_auxiliary(output/'AUDIT.json',expected_sha=sha256(output/'AUDIT.json'),expected_core_run_id='current-G2')
    modality=read(path)['modalities'][0]
    partial=data.loc[data.cancer_id.eq('BRCA') & data.lncrna_id.eq('LNC:L1')].iloc[0]
    assert partial[f'{modality}_context_probability']==pytest.approx(np.mean([.2,.35,.5,.65]))
    assert partial[f'{modality}_patient_folds_with_prediction']==4
    assert partial[f'{modality}_fold_sd']==pytest.approx(np.std([.2,.35,.5,.65],ddof=1))
    assert data.loc[data.cancer_id.eq('LAML'),f'{modality}_context_probability'].isna().all()
    context=pd.read_parquet(output/'NATIVE_FOLD_CONTEXT.parquet')
    assert len(context)==len(keys)*5*len(read(path)['modalities'])
    assert 'unused_observed_label' not in context
    assert context.loc[context.available,'probability'].eq(0).any()
    assert audit['native_model_forwards_replayed'] is False and not audit['independent_biological_validation']
    primary=tmp_path/'primary.parquet';keys.assign(**{PRIMARY_COLUMN:.25,'analysis_version':ANALYSIS_VERSION}).to_parquet(primary,index=False)
    folds=[]
    for f in range(5):
        p=tmp_path/f'labels_{f}.parquet';keys.assign(patient_fold_id=f,held_out_proxy_label=f%2).to_parquet(p,index=False)
        folds.append({'fold_id':f,'artifact':record(p)})
    binding=write(tmp_path/'primary_binding.json',{'status':'PASS','run_id':'current-G2','prediction':record(primary),
        'fold_prediction_sha256':[r['artifact']['sha256'] for r in folds]})
    fusion_manifest=write(tmp_path/'fusion.json',{'format':'CURRENT_MULTIMODAL_FUSION_MANIFEST_V1',
        'core_run_id':'current-G2','primary_prediction':record(primary),'primary_binding_audit':record(binding),
        'primary_folds':folds,'experts':read(output/'FUSION_EXPERTS.json')})
    assert runner.main(['--manifest',str(fusion_manifest),'--expected-core-run-id','current-G2','--output',str(tmp_path/'prepared')])==0
    frame=pd.read_parquet(tmp_path/'prepared/PRIVATE_FUSION_INPUT.parquet')
    ids=tuple(spec['expert_id'] for spec in audit['expert_specs'])
    frozen=ResidualFusionModel('discovery',ids,tuple(0. for _ in ids),tuple(.7 for _ in ids),1,'0'*64,'1'*64,0,0)
    result=apply_residual_fusion_model(frozen,frame)
    assert (result.loc[result.cancer_id.eq('BRCA'),'discovery_adjusted_probability']!=.25).any()
    assert result.loc[result.cancer_id.eq('LAML'),'discovery_adjusted_probability'].eq(.25).all()
    assert read(tmp_path/'prepared/PREPARED.json')['training_started'] is False


@pytest.mark.parametrize('fault',['native_run','old_core','missing_fold','duplicate','missing_candidate','unavailable_zero','invalid_probability','wrong_fold','boolean_string','averaged_score','checkpoint_parent','split_authority'])
def test_genomic_source_corruption_cannot_be_relabelled_as_current(tmp_path,fault):
    path,keys=fixture(tmp_path);manifest=read(path);native_path=Path(manifest['native_receipt']['path'])
    native=read(native_path);lp=Path(native['lineage_path']);lineage=read(lp)
    if fault=='native_run':manifest['native_run_id']='WRONG_RUN';write(path,manifest)
    elif fault=='old_core':lineage['v32_core_checkpoint_sha256']='a'*64;write(lp,lineage)
    elif fault=='missing_fold':lineage['patient_fold_oof_prediction_paths'].pop();write(lp,lineage)
    elif fault=='averaged_score':
        p=Path(native['prediction_path']);f=pd.read_parquet(p);f.loc[0,'mutation_context_probability']=.99;f.to_parquet(p,index=False)
    elif fault=='checkpoint_parent':
        cp_path=Path(lineage['checkpoint_manifest_path']);cps=read(cp_path);r=cps['records'][0]
        p=Path(r['path']);cp=torch.load(p,weights_only=False);cp['core_checkpoint_sha256']='f'*64;torch.save(cp,p)
        r['sha256']=sha256(p);write(cp_path,cps)
    elif fault=='split_authority':
        p=tmp_path/'other_patients.parquet';f=pd.read_parquet(manifest['patient_folds']['path']);f.patient_id+='X';f.to_parquet(p,index=False)
        manifest['patient_folds']=record(p);write(path,manifest)
    else:
        rec=lineage['patient_fold_oof_prediction_paths'][0];p=Path(rec['path']);f=pd.read_parquet(p)
        if fault=='duplicate':f=pd.concat([f,f.iloc[:1]])
        if fault=='missing_candidate':f=f.iloc[1:]
        if fault=='unavailable_zero':f.loc[f.cancer_id.eq('LAML'),'mutation_context_probability']=0.
        if fault=='invalid_probability':f.loc[0,'mutation_context_probability']=1.5
        if fault=='wrong_fold':f.loc[0,'patient_fold_id']=1
        if fault=='boolean_string':f.mutation_available=f.mutation_available.astype(str)
        f.to_parquet(p,index=False);rec['sha256']=sha256(p);write(lp,lineage)
    repin_native(path)
    with pytest.raises((ValueError,KeyError)):materialize(path,tmp_path/'bad')
    assert not (tmp_path/'bad/SUCCESS.json').exists()


@pytest.mark.parametrize('kind,fault',[('cnv','row_source'),('cnv','patient_count'),('cnv','duplicate_partition'),('atac','upstream_fit'),('atac','checkpoint_split')])
def test_native_specific_identity_and_patient_boundaries(tmp_path,kind,fault):
    path,_=fixture(tmp_path,kind);manifest=read(path);npth=Path(manifest['native_receipt']['path']);native=read(npth)
    if kind=='cnv':
        if fault=='duplicate_partition':native['predictions'][1]=dict(native['predictions'][0])
        else:
            r=native['predictions'][0];p=Path(r['path']);f=pd.read_parquet(p)
            f.loc[0,'checkpoint_sha256' if fault=='row_source' else 'callable_patient_n']='f'*64 if fault=='row_source' else 10
            f.to_parquet(p,index=False);r['sha256']=sha256(p)
        write(npth,native)
    else:
        lp=Path(native['lineage_path']);lineage=read(lp)
        if fault=='upstream_fit':lineage['outer_test_patients_used_in_any_upstream_fit']=1;write(lp,lineage)
        else:
            p=Path(lineage['checkpoint_manifest_path']);cps=read(p);r=cps['records'][0]
            cp=Path(r['path']);v=read(cp);v['input_hashes']['folds']='f'*64;write(cp,v);r['sha256']=sha256(cp);write(p,cps)
    repin_native(path)
    with pytest.raises(ValueError):materialize(path,tmp_path/'bad')


@pytest.mark.parametrize('table',['fusion','fold_context','coverage'])
def test_rehashed_adapter_results_must_match_native_fold_replay(tmp_path,table):
    path,_=fixture(tmp_path,'atac');binding=materialize(path,tmp_path/'adapter');audit=read(binding)
    p=Path(audit['outputs'][table]['path']);f=pd.read_parquet(p)
    column={'fusion':'atac_context_probability','fold_context':'probability','coverage':'available_rows'}[table]
    f.loc[0,column]=.99 if table!='coverage' else 3
    f.to_parquet(p,index=False);audit['outputs'][table]['sha256']=sha256(p);write(binding,audit)
    write(binding.parent/'SUCCESS.json',{'status':'PASS','binding_sha256':sha256(binding)})
    with pytest.raises(ValueError,match='replay'):load_native_auxiliary(binding,expected_sha=sha256(binding),expected_core_run_id='current-G2')


def test_same_canonical_patient_cannot_be_assigned_to_multiple_folds(tmp_path):
    from cc_hhgt.v32.native_auxiliary_fusion import _folds
    p=tmp_path/'patient_aliases.parquet'
    pd.DataFrame({'cancer_id':'BRCA','patient_id':['TCGA-AB-0001-01A','tcga.ab.0001.02A','P2','P3','P4'],
        'patient_fold_id':range(5)}).to_parquet(p,index=False)
    with pytest.raises(ValueError,match='crosses'):_folds(p)


@pytest.mark.parametrize('case',['untrained_fold','one_available_fold'])
def test_partial_native_training_or_coverage_does_not_invent_folds_or_uncertainty(tmp_path,case):
    path,_=fixture(tmp_path);manifest=read(path);npth=Path(manifest['native_receipt']['path']);native=read(npth)
    lp=Path(native['lineage_path']);lineage=read(lp)
    all_frames=[]
    for rec in lineage['patient_fold_oof_prediction_paths']:
        p=Path(rec['path']);f=pd.read_parquet(p);fold=rec['patient_fold_id']
        selected=np.full(len(f),fold==4) if case=='untrained_fold' else (f.lncrna_id.eq('LNC:L0').to_numpy() & (fold!=0))
        for m in ('mutation','cnv'):
            f.loc[selected,f'{m}_context_probability']=np.nan;f.loc[selected,f'{m}_available']=False
        f.to_parquet(p,index=False);rec['sha256']=sha256(p);all_frames.append(f)
    if case=='untrained_fold':
        cp=Path(lineage['checkpoint_manifest_path']);cps=read(cp)
        cps['records']=[r for r in cps['records'] if r['patient_fold']!=4];write(cp,cps)
        for m in ('mutation','cnv'):
            lineage['modalities'][m][4]={'patient_fold':4,'status':'NULL_WITH_REASON','reason':'ONE_CLASS_NATIVE_TRAINING_LABELS'}
    typed_path=Path(native['prediction_path']);typed=pd.read_parquet(typed_path)
    for m in ('mutation','cnv'):
        matrix=np.stack([f[f'{m}_context_probability'].to_numpy(float) for f in all_frames]);count=np.isfinite(matrix).sum(axis=0)
        typed[f'{m}_context_probability']=np.divide(np.nansum(matrix,axis=0),count,out=np.full(len(typed),np.nan),where=count>0)
        typed[f'{m}_available']=count>0;typed[f'{m}_patient_folds_with_prediction']=count
    typed.to_parquet(typed_path,index=False);write(lp,lineage);repin_native(path)
    binding=materialize(path,tmp_path/'adapter')
    data,_=load_native_auxiliary(binding,expected_sha=sha256(binding),expected_core_run_id='current-G2')
    row=data.loc[data.cancer_id.eq('BRCA') & data.lncrna_id.eq('LNC:L0')].iloc[0]
    if case=='untrained_fold':
        assert row.mutation_patient_folds_with_prediction==4
        assert 'ONE_CLASS_NATIVE_TRAINING_LABELS' in row.mutation_missing_fold_reasons
    else:
        assert row.mutation_patient_folds_with_prediction==1
        assert row.mutation_context_probability==0. and pd.isna(row.mutation_fold_sd)
    assert data.loc[data.cancer_id.eq('LAML'),'mutation_unavailable_reason'].eq('NATIVE_INPUT_NOT_AVAILABLE').all()


def test_two_cnv_estimands_remain_distinct_and_can_both_reach_frozen_fusion(tmp_path):
    from cc_hhgt.v32.multimodal_fusion import build_pair_aggregated_fusion_frame, ExpertSpec
    experts={}
    for kind in ('genomic','cnv','atac'):
        path,keys=fixture(tmp_path/kind,kind);binding=materialize(path,tmp_path/(kind+'_adapter'))
        data,audit=load_native_auxiliary(binding,expected_sha=sha256(binding),expected_core_run_id='current-G2')
        for item in audit['expert_specs']:
            item['endpoint_roles']=tuple(item['endpoint_roles']);experts[ExpertSpec(**item)]=data
    assert {s.expert_id for s in experts}=={'mutation','cnv_event','cnv','atac'}
    primary=keys.assign(**{PRIMARY_COLUMN:.25,'analysis_version':ANALYSIS_VERSION})
    labels=[keys.assign(patient_fold_id=f,held_out_proxy_label=f%2) for f in range(5)]
    frame=build_pair_aggregated_fusion_frame(primary,labels,experts)
    assert all(col in frame for col in ('cnv_event_probability','cnv_probability','mutation_probability','atac_probability'))
    model=ResidualFusionModel('discovery',tuple(s.expert_id for s in experts),(0.,)*4,(.3,)*4,1,'0'*64,'1'*64,0,0)
    result=apply_residual_fusion_model(model,frame)
    assert result.loc[result.cancer_id.eq('LAML'),'discovery_adjusted_probability'].eq(.25).all()
    assert result.loc[result.cancer_id.eq('BRCA'),'discovery_adjusted_probability'].ne(.25).any()
