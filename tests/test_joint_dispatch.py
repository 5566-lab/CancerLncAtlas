from cla import result_exit_code
from cla_engine.modules import PACKAGE,SCRIPTS,RECIPES

def test_failed_local_formal_gate_has_failure_exit_code():
    assert result_exit_code({'state':'QUEUE_COMPLETE_LOCAL_149',
      'prediction_gates':{'G0':'FORMAL_GATE_FAIL','G1':'FORMAL_GATE_FAIL','G2':'PREDICTION_GATE_PASS_NOT_DEPLOYED'}})==3

def test_complete_local_grid_and_paused_budget_have_distinct_exit_codes():
    assert result_exit_code({'state':'QUEUE_COMPLETE_LOCAL_149',
      'prediction_gates':{v:'PREDICTION_GATE_PASS_NOT_DEPLOYED' for v in ['G0','G1','G2']}})==0
    assert result_exit_code({'state':'QUEUE_BUDGET_EXHAUSTED'})==75

def test_downstream_dispatch_uses_actual_unified_package():
    assert SCRIPTS==PACKAGE/'scripts'
    assert (PACKAGE/'cc_hhgt/v32').is_dir()
    assert not (PACKAGE/'code/cc_hhgt').exists()
    assert RECIPES['evidence_train']['values']['--checkpoint-steps']==20
