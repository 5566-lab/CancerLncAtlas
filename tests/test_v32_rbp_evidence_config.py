"""Phase 11 tests: the rbp_evidence switch block and four ablation modes."""

from __future__ import annotations

import pytest

from cc_hhgt.v32.rbp_evidence_config import (
    ABLATION_MODES,
    FAIRNESS_KEYS,
    MODE_A,
    MODE_B,
    MODE_C,
    MODE_D,
    RbpEvidenceConfig,
    assert_fair_comparison,
    config_from_mapping,
    config_to_yaml_block,
    resolve_mode,
)


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------


def test_exactly_four_modes_exist() -> None:
    assert ABLATION_MODES == (MODE_A, MODE_B, MODE_C, MODE_D)


@pytest.mark.parametrize("alias,expected", [
    ("a", MODE_A), ("b", MODE_B), ("c", MODE_C), ("d", MODE_D),
    (MODE_A, MODE_A), (MODE_C, MODE_C), ("  D  ", MODE_D),
])
def test_mode_aliases_resolve(alias: str, expected: str) -> None:
    assert resolve_mode(alias).mode == expected


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        resolve_mode("mode_z")


def test_mode_a_is_the_legacy_behaviour() -> None:
    a = resolve_mode(MODE_A)
    assert a.preserve_assay_type is False
    assert a.typed_binding_relations is False
    assert a.encode_eclip is False
    assert a.encode_rbp_kd is False


def test_mode_b_adds_typing_but_no_encode() -> None:
    b = resolve_mode(MODE_B)
    assert b.preserve_assay_type is True
    assert b.typed_binding_relations is True
    assert b.encode_eclip is False
    assert b.encode_rbp_kd is False


def test_mode_c_adds_eclip_only() -> None:
    c = resolve_mode(MODE_C)
    assert c.typed_binding_relations is True
    assert c.encode_eclip is True
    assert c.encode_rbp_kd is False


def test_mode_d_adds_kd_as_evidence_only() -> None:
    d = resolve_mode(MODE_D)
    assert d.encode_eclip is True
    assert d.encode_rbp_kd is True
    assert d.encode_rbp_kd_primary_graph is False
    assert d.encode_rbp_kd_evidence_only is True


def test_modes_are_nested() -> None:
    """A is a strict subset of B, B of C, C of D on the switch axes."""

    order = [resolve_mode(m) for m in ABLATION_MODES]
    for earlier, later in zip(order, order[1:]):
        for key, value in earlier.switch_dict().items():
            if isinstance(value, bool) and value:
                assert later.switch_dict()[key] is True, key


def test_no_mode_enables_rbns() -> None:
    for mode in ABLATION_MODES:
        assert resolve_mode(mode).rbns is False


def test_no_mode_admits_predicted_binding_by_default() -> None:
    for mode in ABLATION_MODES:
        assert resolve_mode(mode).include_predicted_binding is False


# ---------------------------------------------------------------------------
# Fairness enforcement
# ---------------------------------------------------------------------------


def test_all_four_modes_are_comparable() -> None:
    report = assert_fair_comparison([resolve_mode(m) for m in ABLATION_MODES])
    assert report["status"] == "FAIR"
    assert report["modes"] == list(ABLATION_MODES)
    assert set(report["frozen_axes"]) == set(FAIRNESS_KEYS)


@pytest.mark.parametrize("key", FAIRNESS_KEYS)
def test_tampering_with_any_frozen_axis_is_rejected(key: str) -> None:
    base = resolve_mode(MODE_A)
    configs = [resolve_mode(m) for m in ABLATION_MODES]
    original = getattr(base, key)
    tampered = "TAMPERED" if isinstance(original, str) else (
        999999 if isinstance(original, int) and not isinstance(original, bool)
        else float(original) + 1.0
    )
    configs[2] = type(base)(**{**base.as_dict(), "mode": MODE_C, key: tampered})
    with pytest.raises(RuntimeError):
        assert_fair_comparison(configs)


def test_a_changed_label_definition_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    b = type(a)(**{**a.as_dict(), "mode": MODE_B, "label_definition": "SOMETHING_ELSE"})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([a, b])


def test_a_changed_seed_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    b = type(a)(**{**a.as_dict(), "mode": MODE_B, "seed": 1})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([a, b])


def test_a_changed_fold_count_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    b = type(a)(**{**a.as_dict(), "mode": MODE_B, "n_folds": 3})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([a, b])


# ---------------------------------------------------------------------------
# Mode-independent invariants
# ---------------------------------------------------------------------------


def test_changing_primary_ranking_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    bad = type(a)(**{**a.as_dict(), "changes_primary_ranking": True})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([bad])


def test_auxiliary_gradients_into_core_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    bad = type(a)(**{**a.as_dict(), "auxiliary_gradients_into_core": True})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([bad])


def test_rbp_kd_as_lncrna_truth_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    bad = type(a)(**{**a.as_dict(), "rbp_kd_is_lncrna_function_truth": True})
    with pytest.raises(RuntimeError):
        assert_fair_comparison([bad])


def test_rbp_kd_in_the_primary_graph_is_rejected() -> None:
    a = resolve_mode(MODE_A)
    bad = type(a)(
        **{
            **a.as_dict(),
            "encode_rbp_kd": True,
            "encode_rbp_kd_primary_graph": True,
            "encode_rbp_kd_evidence_only": False,
        }
    )
    with pytest.raises(RuntimeError):
        assert_fair_comparison([bad])


def test_rbp_kd_must_be_evidence_only() -> None:
    a = resolve_mode(MODE_A)
    bad = type(a)(
        **{
            **a.as_dict(),
            "encode_rbp_kd": True,
            "encode_rbp_kd_primary_graph": False,
            "encode_rbp_kd_evidence_only": False,
        }
    )
    with pytest.raises(RuntimeError):
        assert_fair_comparison([bad])


def test_empty_config_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        assert_fair_comparison([])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_yaml_block_contains_the_plan_switch_names() -> None:
    block = config_to_yaml_block(resolve_mode(MODE_D))
    for token in (
        "rbp_evidence:", "preserve_assay_type:", "typed_binding_relations:",
        "encode_eclip:", "encode_rbp_kd:", "primary_graph:", "evidence_only:",
        "rbns:", "fairness:", "invariants:",
    ):
        assert token in block, token


def test_yaml_block_renders_booleans_as_yaml() -> None:
    block = config_to_yaml_block(resolve_mode(MODE_A))
    assert "true" in block and "false" in block
    assert "True" not in block and "False" not in block


def test_yaml_block_round_trips_through_the_mapping_loader() -> None:
    original = resolve_mode(MODE_C)
    parsed = {
        "rbp_evidence": {
            "mode": original.mode,
            "preserve_assay_type": original.preserve_assay_type,
            "typed_binding_relations": {"enabled": original.typed_binding_relations},
            "encode_eclip": {"enabled": original.encode_eclip},
            "encode_rbp_kd": {
                "enabled": original.encode_rbp_kd,
                "primary_graph": original.encode_rbp_kd_primary_graph,
                "evidence_only": original.encode_rbp_kd_evidence_only,
            },
            "rbns": {"enabled": original.rbns},
        }
    }
    loaded = config_from_mapping(parsed)
    assert loaded == original


def test_loader_keeps_fairness_defaults_when_absent() -> None:
    loaded = config_from_mapping({"rbp_evidence": {"mode": MODE_B}})
    base = resolve_mode(MODE_B)
    for key in FAIRNESS_KEYS:
        assert getattr(loaded, key) == getattr(base, key)


def test_loader_cannot_relax_an_invariant() -> None:
    loaded = config_from_mapping(
        {
            "rbp_evidence": {
                "mode": MODE_D,
                "invariants": {"rbp_kd_is_lncrna_function_truth": True},
            }
        }
    )
    assert loaded.rbp_kd_is_lncrna_function_truth is True
    with pytest.raises(RuntimeError):
        assert_fair_comparison([loaded])


def test_a_full_fair_config_set_can_be_loaded_from_yaml_shaped_mappings() -> None:
    loaded = [
        config_from_mapping({"rbp_evidence": {"mode": mode}}) for mode in ABLATION_MODES
    ]
    report = assert_fair_comparison(loaded)
    assert report["status"] == "FAIR"


# ---------------------------------------------------------------------------
# The fairness axes must be bound to the REAL artifacts, not placeholders
# ---------------------------------------------------------------------------


def test_candidate_universe_hash_matches_the_authoritative_constant() -> None:
    from cc_hhgt.v32.evidence_training import FORMAL_CANDIDATE_SHA256

    assert resolve_mode(MODE_A).candidate_universe_sha256 == FORMAL_CANDIDATE_SHA256


def test_patient_fold_hash_matches_the_frozen_authority() -> None:
    from cc_hhgt.v32.patient_fold_authority import (
        FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    )

    assert (
        resolve_mode(MODE_A).patient_fold_manifest_sha256
        == FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
    )


def test_seed_and_fold_count_match_the_authoritative_constants() -> None:
    from cc_hhgt.v32.patient_fold_authority import DEFAULT_SEED, N_FOLDS

    config = resolve_mode(MODE_A)
    assert config.seed == DEFAULT_SEED
    assert config.n_folds == N_FOLDS


def test_lasso_base_is_bound_to_a_real_digest() -> None:
    value = resolve_mode(MODE_A).lasso_base_sha256
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)


def test_no_fairness_axis_is_left_as_a_placeholder() -> None:
    for mode in ABLATION_MODES:
        config = resolve_mode(mode)
        for key in FAIRNESS_KEYS:
            value = getattr(config, key)
            assert "PLACEHOLDER" not in str(value).upper(), key
            assert "REUSED_EXISTING" not in str(value).upper(), key