import numpy as np
from cla_engine.functional_statistics import empirical_fisher

def test_single_readout_and_missing_are_exact():
    p=np.array([[.03,np.nan],[np.nan,np.nan]])
    r=empirical_fisher(p,np.isfinite(p),np.eye(2),simulations=2000)
    assert r['p'][0]==.03 and np.isnan(r['p'][1])
    assert r['available'].tolist()==[True,False]

def test_duplicate_does_not_multiply_evidence():
    r=empirical_fisher([[.01,.01]],[[True,True]],np.ones((2,2)),groups=[[0,1]],simulations=2000)
    assert abs(r['p'][0]-.01)<1e-14
    assert r['calibration_method'][0]==4

def test_unknown_dependence_is_conservative():
    r=empirical_fisher([[.01,.02]],[[True,True]],[[1,np.nan],[np.nan,1]],simulations=2000)
    assert r['p'][0]==.02 and r['calibration_method'][0]==3

def test_cached_null_reproduces_p_and_valid_resolution(tmp_path):
    p=np.array([[.01,.1],[.2,.3],[.001,.001]])
    args=(p,np.ones_like(p,bool),np.array([[1,.4],[.4,1]]))
    a=empirical_fisher(*args,simulations=4000,cache=tmp_path)
    b=empirical_fisher(*args,simulations=4000,cache=tmp_path)
    np.testing.assert_array_equal(a['p'],b['p'])
    assert np.all(a['p']>=1/4001) and np.all(a['p']<=1)
    assert a['p'][2]<a['p'][0]<a['p'][1]
