import copy
from pathlib import Path
import pytest
from cla_engine.common import Stop,atomic_json,digest,read_json
from cla_engine.pathway_fusion import FORMAT,SCOPE,ASSAYS,FittedPathwayScores,validate_fusion_config,verify_fusion_view
from cla_engine.state_catalog import CATALOG_ID,ROUTED_CATALOG_ID,expanded_targets,validate_targets
from cc_hhgt.v32.adjustment_policy import default_adjustment_contract

def fusion_config():
    return {'state_catalog':ROUTED_CATALOG_ID,'synthetic_fixture':True,'targets':expanded_targets(ROUTED_CATALOG_ID),
      'pathway_fusion':{'enabled':True,'assays':sorted(ASSAYS),'format':FORMAT,'fit_scope':SCOPE,'nf':10,'r_libraries':['/r/lib'],
       'cache_root':'/source-fitted-cache','complete':{'path':'/raw/COMPLETE.json','sha256':'a'*64}}}

def test_program_routing_registry_preserves_prior_state_definitions():
    old=expanded_targets(CATALOG_ID);new=expanded_targets(ROUTED_CATALOG_ID)
    assert new[:len(old)]==old and len(new)==483
    moved=new[len(old):]
    assert len(moved)==285 and sum(t['id'].startswith('PROGENY:') for t in moved)==14
    assert all(t['role']=='primary' and t['unit']=='LEGACY_WEIGHTED_ZSCORE_UNCHANGED' for t in moved)
    assert all('not native PROGENy/VIPER' in t['interpretation'] for t in moved)
    cfg=fusion_config();validate_targets(cfg);validate_fusion_config(cfg)
    assert set(t['id'] for t in new)<=set(default_adjustment_contract([t['id'] for t in new])['policies'])

@pytest.mark.parametrize('change,code',[
 ({'fit_scope':'ALL_PATIENTS'},'FUSION_FIT_SCOPE'),({'nf':0},'FUSION_COMPONENTS'),
 ({'enabled':'yes'},'FUSION_CONFIG'),({'complete':{'path':'/unbound'}},'FUSION_INPUT_BINDING')])
def test_unbound_or_global_fitted_configs_rejected(change,code):
    cfg=fusion_config();cfg['pathway_fusion'].update(change)
    with pytest.raises(Stop) as e:validate_fusion_config(cfg)
    assert e.value.code==code

def test_programs_must_move_to_state_before_fusion():
    cfg=fusion_config();cfg['state_catalog']=CATALOG_ID
    with pytest.raises(Stop) as e:validate_fusion_config(cfg)
    assert e.value.code=='FUSION_STATE_ROUTING'

def receipt():
    return {'format':FORMAT,'target_host':'149','scope':SCOPE,'source_patients':['p1','p2'],
      'source_patients_sha256':digest(['p1','p2']),'input_complete_sha256':'a'*64,'nf':10,
      'assays':sorted(ASSAYS),'mogsa_status':'OK','projection_policy':{'fixed_coverage_percentage_applied':False},'target_ids_sha256':digest(['MSIGDB:X']),'mogsa_fit_patients':['p1','p2'],'global_RNA_consensus_used_as_training_label':False}

def test_prepared_receipt_rejects_holdout_fit_and_global_rna_label(tmp_path):
    cfg=fusion_config();r=receipt();p=tmp_path/'PATHWAY_FUSION.json'
    atomic_json(p,r);verify_fusion_view(tmp_path,cfg,['p1','p2'],['MSIGDB:X'])
    for change in ({'mogsa_fit_patients':['p1','test_patient']},{'global_RNA_consensus_used_as_training_label':True}):
        bad={**r,**change};atomic_json(p,bad)
        with pytest.raises(Stop) as e:verify_fusion_view(tmp_path,cfg,['p1','p2'],['MSIGDB:X'])
        assert e.value.code=='FUSION_GLOBAL_FIT_FORBIDDEN'
    atomic_json(p,{**r,'source_patients':['p1','test_patient']})
    with pytest.raises(Stop) as e:verify_fusion_view(tmp_path,cfg,['p1','p2'],['MSIGDB:X'])
    assert e.value.code=='FUSION_PREPARED_PATIENT_LEAK'


@pytest.mark.parametrize('assays',[['RNA','CNV','METHYLATION'],['RNA','PROTEIN','PHOSPHORYLATION']])
def test_five_assays_cannot_be_silently_substituted(assays):
    cfg=fusion_config();cfg['pathway_fusion']['assays']=assays
    with pytest.raises(Stop) as e:validate_fusion_config(cfg)
    assert e.value.code=='FUSION_ASSAY_SCOPE'


@pytest.mark.parametrize('change,code',[
    ({'assays':['RNA','CNV','METHYLATION']},'FUSION_PREPARED_ASSAY_SCOPE'),
    ({'projection_policy':{'fixed_coverage_percentage_applied':True}},'FUSION_UNREVIEWED_COVERAGE_GATE')])
def test_prepared_design_drift_rejected(tmp_path,change,code):
    atomic_json(tmp_path/'PATHWAY_FUSION.json',{**receipt(),**change})
    with pytest.raises(Stop) as e:verify_fusion_view(tmp_path,fusion_config(),['p1','p2'],['MSIGDB:X'])
    assert e.value.code==code


def test_old_three_assay_manifest_cannot_be_relabelled(tmp_path,monkeypatch):
    from cla_engine.common import sha
    monkeypatch.setattr('cla_engine.pathway_fusion.socket.gethostname',lambda:'149')
    p=tmp_path/'COMPLETE.json'
    atomic_json(p,{'target_host':'149','status':'COMPLETE','assays':['RNA','CNV','METHYLATION']})
    cfg=fusion_config();cfg['pathway_fusion']['complete']={'path':str(p),'sha256':sha(p)}
    with pytest.raises(Stop) as e:FittedPathwayScores(cfg,None)
    assert e.value.code=='FUSION_RAW_ASSAY_DESIGN'


def test_running_view_uses_frozen_source_bytes(tmp_path,monkeypatch):
    import json
    import types
    import pandas as pd
    reader=FittedPathwayScores.__new__(FittedPathwayScores)
    reader.cfg={'complete':{'sha256':'a'*64},'cache_root':str(tmp_path/'cache'),'nf':10,'r_libraries':['/r/lib']}
    reader.root=tmp_path/'raw';reader.catalog_path=tmp_path/'CATALOG.json'
    reader.complete={'catalog_sha256':'b'*64};reader.parts={'TEST':{'sha256':'c'*64}}
    reader.catalog={'gene_sets':{'MSIGDB:X':['G1']}}
    reader.core=tmp_path/'live_core.R';reader.runner=tmp_path/'live_runner.R'
    reader.core.write_text('original core\n');reader.runner.write_text('original runner\n')
    reader.inputs=lambda cancer:{'patient_ids':['p1','p2']}
    observed=[]
    def execute(argv,**kwargs):
        request=json.loads(Path(argv[-1]).read_text())
        frozen_runner=Path(argv[1]);frozen_core=Path(request['core_script'])
        assert frozen_runner!=reader.runner and frozen_core!=reader.core
        reader.core.write_text('edited core while the process runs\n')
        reader.runner.write_text('edited runner while the process runs\n')
        assert frozen_core.read_text()=='original core\n'
        assert frozen_runner.read_text()=='original runner\n'
        assert frozen_core.stat().st_mode & 0o222 == 0
        observed.append(request)
        out=Path(request['output_dir'])
        pd.DataFrame({'patient_id':['p1','p2'],'MSIGDB:X':[1.,2.]}).to_csv(out/'FINAL_PATHWAY_SCORES.tsv.gz',sep='\t',index=False)
        atomic_json(out/'SCORES_RECEIPT.json',{'status':'COMPLETE','scope':SCOPE,'fit_patients':['p1'],
            'mogsa_status':'OK','mogsa_fit_patients':['p1'],'mogsa_available_patients':2,'parents_supported':0,
            'assays':sorted(ASSAYS),'projection_policy':{'fixed_coverage_percentage_applied':False},'parent_policy':'test'})
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr('cla_engine.pathway_fusion.subprocess.run',execute)
    values,report=reader.view('TEST',['p1'],['p1','p2'],['MSIGDB:X'])
    assert len(observed)==1 and values[:,0].tolist()==[1.,2.]
    assert report['source_patients']==['p1']
