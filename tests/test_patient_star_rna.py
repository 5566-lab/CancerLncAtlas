from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from materialize_patient_star_rna import write_views


def test_gdc_views_preserve_missing_specimens_and_sum_symbols_in_linear_space(tmp_path):
    genes = pd.Index(['G1','G2','L1'], name='feature_id')
    matrix = pd.DataFrame({'P::T': np.log2([4.,8.,1.]), 'P::N': [5.,5.,5.]}, index=genes)
    annotation = pd.DataFrame({'reference_symbol':['A','A','L']}, index=genes)
    assert write_views(matrix, pd.Index(['Q','P']), annotation, pd.Index(['L1']), tmp_path) == 1
    lnc = pd.read_parquet(tmp_path / 'LNC_LOGTPM_FULL.parquet')
    symbols = pd.read_parquet(tmp_path / 'SYMBOL_LOGTPM.parquet')
    assert list(lnc.index) == ['Q','P'] and lnc.index.name == 'patient_id'
    assert lnc.loc['P','LNC:L1'] == 0 and np.isnan(lnc.loc['Q','LNC:L1'])
    assert symbols.loc['P','A'] == pytest.approx(np.log2(1+3+7))
    assert symbols.loc['Q'].isna().all()
    with pytest.raises(ValueError, match='cover'):
        write_views(matrix.drop(index='L1'), pd.Index(['P']), annotation, pd.Index(['L1']), tmp_path)


@pytest.mark.skipif(sys.platform != 'linux', reason='Deployment uses Linux rename semantics')
@pytest.mark.parametrize('use_exchange',[False,True])
def test_switch_rollback_and_retry_at_original_path(tmp_path, monkeypatch, use_exchange):
    import activate_patient_star_rna as publication
    live, new = tmp_path / 'current', tmp_path / 'gdc'
    for path in (live, new):
        path.mkdir()
        (path / 'COMPLETE.json').write_text(path.name)
    old_sha, new_sha = publication.digest(live), publication.digest(new)
    if not use_exchange:
        monkeypatch.setattr(publication, 'exchange', lambda a, b: False)
    backup = publication.install(live,new,old_sha,new_sha)
    assert publication.install(live,new,old_sha,new_sha) == backup
    assert publication.digest(live) == new_sha and publication.digest(backup) == old_sha
    publication.restore(live,backup,new_sha)
    assert publication.digest(live) == old_sha and publication.digest(new) == new_sha
    assert not live.is_symlink()
