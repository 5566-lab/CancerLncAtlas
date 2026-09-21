from cc_hhgt.common import stable_id

def test_stable_id_is_deterministic():
    assert stable_id('X','a',1)==stable_id('X','a',1)
    assert stable_id('X','a',1)!=stable_id('X','a',2)
