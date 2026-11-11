"""The historical RNA-score preparation cannot claim a fused relation objective."""
import pytest

from cla_engine.common import Stop
from cla_engine.prepare import prepare


def test_continuous_relation_objective_rejected_before_old_input_access(tmp_path):
    with pytest.raises(Stop) as exc:
        prepare({'relation_objective': 'observed_fused_rms_effect'}, tmp_path / 'prepared')
    assert exc.value.code == 'RELATION_STRENGTH_PREPARATION_NOT_BOUND'
    assert not (tmp_path / 'prepared').exists()
