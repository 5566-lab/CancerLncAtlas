import numpy as np
import pandas as pd

from cc_hhgt.v32.single_cell_metacell_screen import (
    donor_folds,
    exact_donor_targets,
    screen_source_pairs,
)


def synthetic(seed=8):
    rng = np.random.default_rng(seed)
    donors = np.repeat(["D1", "D2", "D3", "D4"], 30)
    latent = np.concatenate([rng.normal(size=30) for _ in range(4)])
    x = np.column_stack(
        [latent + rng.normal(scale=0.05, size=len(latent)), rng.normal(size=len(latent))]
    )
    x -= x.min(axis=0, keepdims=True)
    y = np.column_stack(
        [latent + rng.normal(scale=0.05, size=len(latent)), rng.normal(size=len(latent))]
    )
    y -= y.min(axis=0, keepdims=True)
    meta = pd.DataFrame({"donor_id": donors, "cell_type": "T"})
    return x, y, meta


def test_four_donor_cohort_is_not_dropped_by_five_fold_constant():
    folds = donor_folds("DLBC", ["D1", "D2", "D3", "D4"])
    assert len(folds) == 4
    assert set(folds.outer_fold_id) == {0, 1, 2, 3}
    assert folds.limited_replication.all()


def test_source_screen_finds_new_pair_without_query_donor():
    x, y, meta = synthetic()
    result = screen_source_pairs(
        x,
        y,
        meta,
        ["L1", "L2"],
        ["P1", "P2"],
        ["D1", "D2", "D3"],
        top_k=1,
        sketch_dimensions=32,
        min_source_donors=3,
        min_abs_meta_rho=0.0,
    )
    chosen = result.set_index("lncrna_id").pathway_id.to_dict()
    assert chosen["L1"] == "P1"
    assert not result.candidate_selection_query_donor_used.any()


def test_query_mutation_cannot_change_source_universe():
    x, y, meta = synthetic()
    args = dict(
        lnc_expression=x,
        pathway_activity=y,
        metadata=meta,
        lnc_ids=["L1", "L2"],
        pathway_ids=["P1", "P2"],
        source_donors=["D1", "D2", "D3"],
        top_k=1,
        sketch_dimensions=32,
        min_source_donors=3,
        min_abs_meta_rho=0.0,
    )
    first = screen_source_pairs(**args)
    changed = x.copy()
    changed[meta.donor_id.eq("D4"), :] = 1000.0
    second = screen_source_pairs(**{**args, "lnc_expression": changed})
    pd.testing.assert_frame_equal(first, second)


def test_query_targets_are_signed_and_donor_resolved():
    x, y, meta = synthetic()
    pairs = pd.DataFrame({"lncrna_id": ["L1"], "pathway_id": ["P1"]})
    target = exact_donor_targets(
        x, y, meta, ["L1", "L2"], ["P1", "P2"], pairs, ["D4"]
    )
    assert target.query_donor_id.tolist() == ["D4"]
    assert target.rho.iloc[0] > 0.9
    assert np.isclose(target.association_target.iloc[0], abs(target.rho.iloc[0]))
    assert target.donor_resolved_target.all()
