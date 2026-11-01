"""Cancer selection must change real prepared data, not only CLI metadata."""
import shutil,subprocess,sys
from pathlib import Path
import pandas as pd
import pytest
from cla import parser
from cla_engine.cancer_scope import override_selection,resolve_cancer_scope
from cla_engine.common import Stop,atomic_json,read_json,commit_manifest
from cla_engine.fixture import make_fixture
from cla_engine.inputs import preflight
from cla_engine.lineage import validate_prepared,prepare_lineage
from cla_engine.prepare import prepare

ROOT=Path(__file__).resolve().parents[1]

def test_all_overrides_old_profile_subset():
    cfg={'cancer_type':'all','profiles':{'diagnostic':{'cancers':['C1']}}}
    assert resolve_cancer_scope(cfg,['C3','C1','C2'])['selected_cancers']==['C1','C2','C3']

def test_normalized_list_and_configuration_is_not_mutated():
    original={'cancer_type':'all','cancer_list':[]}
    cfg=override_selection(original,cancer_type='list',cancer_list=[' c3 ','C1','c3'])
    assert resolve_cancer_scope(cfg,['C1','C2','C3'])['selected_cancers']==['C1','C3']
    assert original=={'cancer_type':'all','cancer_list':[]}
    cfg=override_selection(cfg,cancer_type='all')
    assert resolve_cancer_scope(cfg,['C1','C2','C3'])['selected_cancers']==['C1','C2','C3']

@pytest.mark.parametrize('cfg,code',[
    ({'cancer_type':'list'},'CANCER_LIST_EMPTY'),
    ({'cancer_type':'list','cancer_list':[]},'CANCER_LIST_EMPTY'),
    ({'cancer_type':'list','cancer_list':'C1'},'CANCER_LIST_INVALID'),
    ({'cancer_type':'list','cancer_list':['']},'CANCER_LIST_INVALID'),
    ({'cancer_type':'list','cancer_list':[1]},'CANCER_LIST_INVALID'),
    ({'cancer_type':'list','cancer_list':['UNKNOWN']},'CANCER_SCOPE'),
    ({'cancer_type':'all','cancer_list':['C1']},'CANCER_LIST_WITH_ALL'),
    ({'cancer_type':'bogus'},'CANCER_TYPE_INVALID'),
    ({'cancer_list':['C1']},'CANCER_TYPE_REQUIRED'),
])
def test_invalid_selection_fails(cfg,code):
    with pytest.raises(Stop) as error:resolve_cancer_scope(cfg,['C1','C2','C3'])
    assert error.value.code==code

def test_existing_profile_scope_survives_without_new_parameter():
    cfg={'profiles':{'diagnostic':{'cancers':['C2']},'formal':{'cancers':[]}}}
    assert resolve_cancer_scope(cfg,['C1','C2'])['selected_cancers']==['C2']
    assert resolve_cancer_scope(cfg,['C1','C2'],'formal')['selected_cancers']==['C1','C2']

@pytest.mark.parametrize('mode_arg,list_arg',[('--cancer_type','--cancer_list'),('--cancer-type','--cancer-list')])
def test_cli_aliases(mode_arg,list_arg):
    args=parser().parse_args(['prepare',mode_arg,'list',list_arg,'C1','C3'])
    assert args.cancer_type=='list' and args.cancer_list==['C1','C3']

@pytest.fixture(scope='module')
def scoped_prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp('cancer_scope')
    cfg=make_fixture(root/'source',parquet=True)
    # Fixture eligibility needs three cancers; a two-cancer demo must retain it.
    assert cfg['candidate_scope']['minimum_expression_cancers']==3
    cfg=override_selection(cfg,cancer_type='list',cancer_list=['C3','C1'])
    # Removing an unselected expression partition proves it is not processed.
    for role in ('lnc_expression_root','gene_expression_root','pathway_activity_root'):
        parent=Path(cfg['sources'][role]['path'])
        for path in parent.rglob('*'):
            if path.is_file() and 'C2' in str(path.relative_to(parent)):path.unlink()
    out=root/'prepared';prepare(cfg,out)
    return root,cfg,out

def test_list_limits_actual_views_patients_baselines_and_cancer_nodes(scoped_prepared):
    root,cfg,out=scoped_prepared
    manifest,spec,stored,lineage=validate_prepared(out)
    assert spec['cancers']==spec['nodes']['cancer']==['C1','C3']
    assert set(read_json(out/'PATIENT_ASSIGNMENTS.json'))=={'C1','C3'}
    assert set(read_json(out/'CANDIDATES.json'))=={'C1','C3'}
    assert {v['id'].split('__')[0] for v in read_json(out/'VIEWS.json')}=={'C1','C3'}
    assert read_json(out/'CANCER_SCOPE.json')['available_cancers']==['C1','C2','C3']
    assert spec['nodes']['lncRNA'], 'Global eligibility must not shrink to the demo cancer count'
    report=preflight(cfg,root/'preflight')
    assert report['cancer_scope']['selected_cancers']==['C1','C3']
    coverage=pd.read_csv(root/'preflight/STATE_SOURCE_COVERAGE.tsv',sep='\t')
    assert set(coverage.cancer_id)=={'C1','C3'}

def test_prepared_scope_cannot_disagree_with_config(scoped_prepared,tmp_path):
    _,cfg,original=scoped_prepared
    out=tmp_path/'changed';shutil.copytree(original,out)
    record=read_json(out/'CANCER_SCOPE.json');record['selected_cancers']=['C1']
    atomic_json(out/'CANCER_SCOPE.json',record)
    atomic_json(out/'LINEAGE.json',prepare_lineage(out))
    commit_manifest(out,read_json(out/'MANIFEST.json')['metadata'])
    with pytest.raises(Stop) as error:validate_prepared(out)
    assert error.value.code=='CANCER_SCOPE_MISMATCH'

@pytest.mark.parametrize('mode,selected,expected',[
    ('all',[],['C1','C2','C3']),('list',['C1','C3'],['C1','C3'])])
def test_cli_scope_overrides_old_diagnostic_scope(tmp_path,mode,selected,expected):
    cfg=make_fixture(tmp_path/'source',parquet=True)
    cfg.pop('cancer_type',None);cfg.pop('cancer_list',None)
    cfg['profiles']['diagnostic']['cancers']=['C1']
    source=tmp_path/'config.json';atomic_json(source,cfg)
    out=tmp_path/'all'
    result=subprocess.run([sys.executable,str(ROOT/'cla.py'),'prepare','--config',str(source),
        '--cancer_type',mode,'--output',str(out)]+(['--cancer_list',*selected] if selected else []),capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    _,spec,stored,_=validate_prepared(out)
    assert spec['cancers']==expected and stored['cancer_type']==mode
    assert read_json(out.with_name('all_preflight')/'CANCER_SCOPE.json')['selected_cancers']==spec['cancers']


def test_diagnostic_worker_rejects_scope_override_instead_of_ignoring_it(tmp_path,monkeypatch):
    from cla import main
    monkeypatch.setattr(sys,'argv',['cla.py','worker','--prepared',str(tmp_path/'unused'),
        '--output',str(tmp_path/'run'),'--deadline','1','--cancer_type','list','--cancer_list','C1'])
    assert main()==2
    assert read_json(tmp_path/'run/STATUS.json')['error_code']=='CANCER_SCOPE_ALREADY_PREPARED'
