import pandas as pd
import pytest
from cla_engine.exact_member_groups import exact_member_groups, merge_redirects, grouped_applicability


def row(cid, typ='Variant'):
    return dict(canonical_id=cid, source_target_id=cid, network_type=typ)


def test_only_nonempty_exact_original_sets_merge():
    rows = [row(x) for x in 'abcde']
    g = exact_member_groups(rows, {'a': {'P1', 'P2'}, 'b': {'P2', 'P1'}, 'c': {'P1'}, 'd': set(), 'e': set()})
    assert len(g) == 4
    assert merge_redirects(g) == {'a': 'a', 'b': 'a', 'c': 'c', 'd': 'd', 'e': 'e'}


def test_reference_preference_and_input_order_invariance():
    rows = [row('a'), row('z', 'Reference')]
    p = {'a': {'P'}, 'z': {'P'}}
    assert exact_member_groups(rows, p) == exact_member_groups(rows[::-1], p)
    assert merge_redirects(exact_member_groups(rows, p)) == {'z': 'z', 'a': 'z'}


def test_source_unknown_events_stay_unknown_and_singletons_unchanged():
    source = pd.DataFrame({'a': [float('nan'), 0], 'b': [float('nan'), 1], 'c': [0, 1]}, index=['patient1', 'patient2'])
    original = source.copy(deep=True)
    g = exact_member_groups([row('a'), row('b'), row('c')], {'a': {'P'}, 'b': {'P'}, 'c': {'Q'}})
    active = grouped_applicability(source, g)
    pd.testing.assert_frame_equal(source, original)
    assert active.a.tolist() == [1.0, 1.0]
    assert active.c.tolist() == source.c.tolist()
    assert 'b' not in active


def test_missing_source_condition_is_rejected():
    g = exact_member_groups([row('a'), row('b')], {'a': {'P'}, 'b': {'P'}})
    with pytest.raises(ValueError, match='MISSING_SOURCE_CONDITION_AXIS'):
        grouped_applicability(pd.DataFrame({'a': [1]}), g)


def test_duplicate_primary_ids_rejected():
    with pytest.raises(ValueError, match='DUPLICATE_PRIMARY_ID'):
        exact_member_groups([row('a'), row('a')], {'a': {'P'}})
