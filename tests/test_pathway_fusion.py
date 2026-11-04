"""Retirement must block both regeneration and reuse, without changing State."""
from pathlib import Path
import subprocess
import sys
import pytest
from cla_engine.common import Stop, atomic_json, read_json
from cla_engine.pathway_fusion import validate_fusion_config, require_pathway_targets, reject_retired_prepared
from cla_engine.state_catalog import CATALOG_ID, ROUTED_CATALOG_ID, expanded_targets, validate_targets
from cc_hhgt.v32.adjustment_policy import default_adjustment_contract

ROOT = Path(__file__).resolve().parents[1]


def test_program_routing_registry_preserves_prior_state_definitions():
    old = expanded_targets(CATALOG_ID); new = expanded_targets(ROUTED_CATALOG_ID)
    assert new[:len(old)] == old and len(new) == 483
    moved = new[len(old):]
    assert len(moved) == 285 and sum(t['id'].startswith('PROGENY:') for t in moved) == 14
    assert all(t['role'] == 'primary' and t['unit'] == 'LEGACY_WEIGHTED_ZSCORE_UNCHANGED' for t in moved)
    assert all('not native PROGENy/VIPER' in t['interpretation'] for t in moved)
    cfg = read_json(ROOT/'config/149.json'); validate_targets(cfg)
    assert set(t['id'] for t in new) <= set(default_adjustment_contract([t['id'] for t in new])['policies'])


@pytest.mark.parametrize('setting', [
    {'enabled': True},
    {'enabled': False, 'format': 'CLA_SOURCE_FITTED_FIVE_ASSAY_PATHWAY_FUSION_V2'},
    {'enabled': False, 'complete': {'path': '/old/COMPLETE.json'}},
    {'enabled': False, 'cache_root': '/old/cache'},
])
def test_old_configs_cannot_enable_or_relabel_fusion(setting):
    with pytest.raises(Stop) as e:
        validate_fusion_config({'pathway_fusion': setting})
    assert e.value.code == 'MOGSA_RETIRED'


def test_public_config_blocks_implicit_rna_fallback():
    cfg = read_json(ROOT/'config/149.json')
    assert cfg['pathway_fusion']['enabled'] is False
    assert cfg['model']['task_weights'] == {'pathway': 1., 'state': .5}
    with pytest.raises(Stop) as e:
        require_pathway_targets(cfg)
    assert e.value.code == 'PATHWAY_TARGET_REPLACEMENT_REQUIRED'
    cfg.pop('pathway_fusion')
    with pytest.raises(Stop):
        require_pathway_targets(cfg)


@pytest.mark.parametrize('marker', ['config', 'contract', 'inputs', 'view'])
def test_prepared_rejected_even_when_fusion_flag_is_removed(tmp_path, marker):
    cfg = {'pathway_fusion': {'enabled': False}}
    if marker == 'config':
        cfg['pathway_fusion']['enabled'] = True
    atomic_json(tmp_path/'CONFIG.json', cfg)
    if marker == 'contract':
        atomic_json(tmp_path/'TASK_CONTRACT.json', {'pathway_score_source': 'SOURCE_PATIENT_FITTED_FIVE_ASSAY_MOGSA_AND_LAVAAN'})
    elif marker == 'inputs':
        atomic_json(tmp_path/'INPUT_HASHES.json', {'pathway_fusion::complete': {}})
    elif marker == 'view':
        atomic_json(tmp_path/'views/v1/PATHWAY_FUSION.json', {})
    from cla_engine.lineage import validate_prepared
    # No manifest or source arrays are needed to detect a retired target.
    with pytest.raises(Stop) as e:
        validate_prepared(tmp_path)
    assert e.value.code == 'MOGSA_RETIRED'


def test_preexisting_rna_path_remains_distinct(tmp_path):
    cfg = {'pathway_fusion': {'enabled': False},
           'pathway_target_policy': {'status': 'EXPLICIT_EXISTING_RNA'}}
    atomic_json(tmp_path/'CONFIG.json', cfg)
    atomic_json(tmp_path/'TASK_CONTRACT.json', {'pathway_score_source': 'EXISTING_RNA_ACTIVITY_SCORE'})
    reject_retired_prepared(tmp_path)


def test_retired_cli_does_not_read_inputs_or_create_output(tmp_path):
    out = tmp_path/'must_not_exist'
    result = subprocess.run([sys.executable, str(ROOT/'scripts/score_pathway_fusion_149.py'),
                             '--raw-inputs', str(tmp_path/'nonexistent'), '--output', str(out)],
                            capture_output=True, text=True)
    assert result.returncode == 2 and 'MOGSA_RETIRED' in result.stderr
    assert not out.exists()
