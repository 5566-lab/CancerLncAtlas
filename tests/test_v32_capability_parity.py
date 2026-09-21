from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from cc_hhgt.v32.capability_parity import (
    EVIDENCE_SCHEMA_VERSION,
    EXPECTED_CAPABILITY_IDS,
    EXPECTED_CLINICAL_ENDPOINTS,
    EXPECTED_STATE_PROGRAMS,
    CapabilityParityError,
    load_parity_config,
    validate_capability_parity,
    validate_parity_config,
)


def _valid_manifest(config: dict[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for capability_id, required in config["capabilities"].items():
        if required.get("required") is not True:
            continue
        target_level = required["target_level"]
        artifacts: dict[str, Any] = {}
        for artifact_id in required["gates"]["artifact"]["ids"]:
            artifact = {
                "present": True,
                "path": f"artifacts/{capability_id}/{artifact_id}.json",
                "sha256": "a" * 64,
                "rows": 1,
                "model_version": "V3.2",
                "provenance": "v32_new_training",
                "target_level": target_level,
                "availability_encoding": "null_with_reason",
                "unavailable_fill_value": None,
            }
            if target_level == "exact_pathway":
                artifact["source_target_level"] = "exact_pathway"
                artifact["family_broadcast"] = False
            artifacts[artifact_id] = artifact
        observed[capability_id] = {
            "status": "READY",
            "performance": {"outcome": "INCREMENT"},
            "new_training": {
                "completed": True,
                "model_version": "V3.2",
                "run_id": f"v32-{capability_id}-fresh",
                "initialization": "random",
                "new_parameters_from_scratch": True,
                "checkpoint_sha256": "b" * 64,
                "legacy_derived_inputs": [],
                "input_roles": ["raw_source_data", "task_definition"],
            },
            "artifacts": artifacts,
            "api": {
                "implemented": True,
                "ids": list(required["gates"]["api"]["ids"]),
            },
            "ui": {
                "implemented": True,
                "ids": list(required["gates"]["ui"]["ids"]),
            },
            "download": {
                "implemented": True,
                "ids": list(required["gates"]["download"]["ids"]),
            },
        }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "release_id": "V3.2-fresh-parity-test",
        "model_version": "V3.2",
        "target_level": "exact_pathway",
        "capabilities": observed,
    }


def test_canonical_config_covers_the_complete_historical_surface() -> None:
    config = load_parity_config()
    validate_parity_config(config)

    assert EXPECTED_CAPABILITY_IDS.issubset(config["capabilities"])
    assert len(EXPECTED_CAPABILITY_IDS) == 25
    assert {
        "expression_landscape",
        "survival_kaplan_meier",
        "bulk_coexpression",
        "external_validation",
        "continuous_pathway_activity",
    }.issubset(EXPECTED_CAPABILITY_IDS)
    assert set(config["required_clinical_endpoints"]) >= EXPECTED_CLINICAL_ENDPOINTS
    states = {
        row["capability_id"]: (row["state_id"], row["mainline_required"])
        for row in config["required_state_programs"]
    }
    assert states == EXPECTED_STATE_PROGRAMS
    assert {key for key, (_, required) in states.items() if required} == {
        "state_rnass",
        "state_dnass",
        "state_extend",
        "state_ereg_expss",
    }
    for capability in config["capabilities"].values():
        assert capability["new_training"]["required"] is True
        assert capability["gates"]["artifact"]["require_nonempty"] is True
        assert all(capability["gates"][gate]["required"] for gate in ("api", "ui", "download"))

    mixed = config["capabilities"]["mixed_lncrna_protein_pathway_query"]
    assert {
        "v32_mixed_set_encoder",
        "v32_custom_gene_set_encoder",
        "v32_protein_set_encoder",
    }.issubset(mixed["gates"]["artifact"]["ids"])
    assert {
        "POST /api/site/query/mixed-lncrna-protein-pathway",
        "POST /api/site/query/custom-gene-set-lncrna",
        "POST /api/site/query/protein-set-lncrna",
    }.issubset(mixed["gates"]["api"]["ids"])

    single_cell = config["capabilities"]["single_cell"]
    assert {"v32_sc_ucell", "v32_sc_pseudotime"}.issubset(
        single_cell["gates"]["artifact"]["ids"]
    )


def test_full_multitask_config_cannot_drop_state_or_clinical_contract() -> None:
    path = Path(__file__).parents[1] / "config" / "model_v3_2_full_multitask.yaml"
    model = yaml.safe_load(path.read_text(encoding="utf-8"))
    state_targets = set(model["modules"]["state"]["targets"])
    required_state_targets = {
        ("EXTEND" if state_id.startswith("EXTEND::") else state_id.rsplit("::", 1)[-1])
        for state_id, _mainline in EXPECTED_STATE_PROGRAMS.values()
    }
    assert required_state_targets.issubset(state_targets)
    assert set(model["modules"]["clinical"]["endpoints"]) == EXPECTED_CLINICAL_ENDPOINTS


def test_historical_descriptive_and_validation_surfaces_are_explicit() -> None:
    config = load_parity_config()

    expression = config["capabilities"]["expression_landscape"]
    assert {
        "GET /api/site/lncrna/{lncrna}/expression",
        "GET /api/site/lncrna/{lncrna}/coverage",
    }.issubset(expression["gates"]["api"]["ids"])
    assert config["capabilities"]["survival_kaplan_meier"]["endpoints"] == [
        "OS", "DSS", "PFI", "PFS", "DFI", "DFS"
    ]
    assert config["capabilities"]["continuous_pathway_activity"][
        "independent_from_exact_primary"
    ] is True
    physical = config["capabilities"]["physical_interaction"]
    assert "GET /api/site/relationship/{relationship_id}/evidence" in physical[
        "gates"
    ]["api"]["ids"]


def test_kaplan_meier_must_keep_every_historical_endpoint() -> None:
    config = load_parity_config()
    broken = copy.deepcopy(config)
    broken["capabilities"]["survival_kaplan_meier"]["endpoints"].remove("DFS")

    with pytest.raises(CapabilityParityError, match="Kaplan-Meier.*endpoint"):
        validate_parity_config(broken)


def test_continuous_activity_cannot_replace_exact_primary() -> None:
    config = load_parity_config()
    broken = copy.deepcopy(config)
    broken["capabilities"]["continuous_pathway_activity"][
        "independent_from_exact_primary"
    ] = False

    with pytest.raises(CapabilityParityError, match="independent.*exact-pathway"):
        validate_parity_config(broken)


def test_complete_fresh_v32_release_passes() -> None:
    config = load_parity_config()
    report = validate_capability_parity(config, _valid_manifest(config))

    assert report.status == "PASS"
    assert report.capability_count == len(EXPECTED_CAPABILITY_IDS)
    assert report.diagnostic_only_capabilities == ()


def test_missing_capability_fails_closed() -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    del manifest["capabilities"]["drug"]

    with pytest.raises(CapabilityParityError, match="missing required capabilities.*drug"):
        validate_capability_parity(config, manifest)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("checkpoint_path", "releases/V2.9/best.pt"),
        ("prediction_path", "releases/v3_0/clinical_predictions.parquet"),
        ("reused_legacy_checkpoint", True),
    ],
)
def test_legacy_checkpoint_or_prediction_is_rejected(key: str, value: object) -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    manifest["capabilities"]["clinical"]["new_training"][key] = value

    with pytest.raises(CapabilityParityError, match="Legacy|legacy"):
        validate_capability_parity(config, manifest)


@pytest.mark.parametrize("sentinel", [0, 0.0, 0.5, "0.5"])
def test_unavailable_cannot_be_disguised_as_zero_or_half(sentinel: object) -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    manifest["capabilities"]["cnv"]["coverage_example"] = {
        "availability": "UNAVAILABLE",
        "reason": "source coverage absent",
        "probability": sentinel,
    }

    with pytest.raises(CapabilityParityError, match="Unavailable value"):
        validate_capability_parity(config, manifest)


def test_unavailable_null_with_reason_is_allowed() -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    manifest["capabilities"]["cnv"]["coverage_example"] = {
        "availability": "UNAVAILABLE",
        "reason": "source coverage absent",
        "probability": None,
    }

    assert validate_capability_parity(config, manifest).status == "PASS"


@pytest.mark.parametrize(
    "tamper",
    [
        {"family_broadcast": True},
        {"source_target_level": "pathway_family"},
        {"derivation": "family_score_broadcast"},
    ],
)
def test_family_score_cannot_be_broadcast_to_exact_pathway(tamper: dict[str, object]) -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    artifact = manifest["capabilities"]["evidence_transformer"]["artifacts"][
        "v32_neural_evidence_probability"
    ]
    artifact.update(tamper)

    with pytest.raises(CapabilityParityError, match="[Ff]amily|exact-pathway source"):
        validate_capability_parity(config, manifest)


def test_no_increment_retains_capability_as_diagnostic_only() -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    mutation = manifest["capabilities"]["mutation"]
    mutation["status"] = "DIAGNOSTIC_ONLY"
    mutation["performance"]["outcome"] = "NO_INCREMENT"

    report = validate_capability_parity(config, manifest)
    assert report.diagnostic_only_capabilities == ("mutation",)

    mutation["status"] = "READY"
    with pytest.raises(CapabilityParityError, match="NO_INCREMENT.*DIAGNOSTIC_ONLY"):
        validate_capability_parity(config, manifest)


@pytest.mark.parametrize("gate", ["artifact", "api", "ui", "download"])
def test_missing_publication_gate_fails_closed(gate: str) -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    capability = manifest["capabilities"]["physical_interaction"]
    if gate == "artifact":
        capability["artifacts"].pop("v32_interaction_provenance")
    else:
        capability[gate]["implemented"] = False

    with pytest.raises(CapabilityParityError, match=gate):
        validate_capability_parity(config, manifest)


def test_evidence_confidence_cannot_reuse_old_probabilities() -> None:
    config = load_parity_config()
    manifest = _valid_manifest(config)
    training = manifest["capabilities"]["evidence_transformer"]["new_training"]
    training["legacy_derived_inputs"] = ["probability"]

    with pytest.raises(CapabilityParityError, match="legacy derived inputs"):
        validate_capability_parity(config, manifest)
