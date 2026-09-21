from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNTIME = _load_script(
    "v32_independent_runtime_smoke_hardening",
    "scripts/smoke_v32_independent_head_runtime_bindings_server.py",
)
DRUG = _load_script(
    "v32_drug_runtime_smoke_hardening",
    "scripts/smoke_v32_authorized_drug_partitioned_server.py",
)
CLOSED_RUNTIME = _load_script(
    "v32_closed_runtime_manifest_test",
    "scripts/build_v32_independent_head_runtime_bindings_closed_server.py",
)


def _json(path: Path, payload: dict) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_manifest(
    tmp_path: Path, *, gene_status: str = RUNTIME.GENE_SET_METADATA_CLOSED
) -> dict:
    def declaration(role: str, *, audited: bool = False) -> dict:
        value = {
            "binding_path": f"./data/CancerLncAtlas/runtime/authorized_bindings/test/{role}/binding.json",
            "binding_sha256": "a" * 64,
            "scientific_status": RUNTIME.EXPECTED_COMPONENTS[role],
            "production_deployed": False,
            "release_ready": False,
        }
        if audited:
            value.update(
                {
                    "audit_binding_path": f"./data/CancerLncAtlas/runtime/authorized_bindings/test/{role}/audit.json",
                    "audit_binding_sha256": "b" * 64,
                }
            )
        return value

    gene_declaration = declaration("gene_set_ranked_subtype", audited=True)
    closure = {
        "format": RUNTIME.GENE_SET_CLOSURE_FORMAT,
        "status": "PASS",
        "runtime_binding_declaration": gene_declaration,
        "metadata_closure": {
            "transitive_metadata_closed": True,
            "release_records_hash_validated": 47,
            "upstream_gene_set_records_hash_validated": 47,
            "upstream_gene_set_unique_records": 44,
            "legacy_gene_set_transitive_unresolved": 24,
            "legacy_ranked_subtype_transitive_unresolved": 72,
            "legacy_local_unresolved": 0,
            "portable_rebinder_pending_was_runtime_false_negative": True,
        },
        "exact_probes": {
            "representative_member_rows": 1,
            "cancers_typed": 33,
            "available_cancers": 31,
            "typed_unavailable": dict(RUNTIME.TYPED_UNAVAILABLE),
        },
        "resource_gate": {
            "passed": True,
            "kernel_peak_rss_kB": 100_000,
            "max_peak_rss_kB": 2 * 1024 * 1024,
        },
        "family_to_exact_broadcast": False,
        "changes_primary_ranking": False,
        "production_deployed": False,
        "release_ready": False,
    }
    closure_path = tmp_path / "GENE_SET_RUNTIME_CLOSURE_RECEIPT.json"
    closure_sha = _json(closure_path, closure)
    return {
        "format": RUNTIME.RUNTIME_BINDING_FORMAT,
        "status": RUNTIME.RUNTIME_BINDING_STATUS,
        "server_root": "./data/CancerLncAtlas/runtime/authorized_bindings/test",
        "bindings": {
            "evidence": declaration("evidence"),
            "evidence_direction": declaration(
                "evidence_direction", audited=True
            ),
            "clinical": declaration("clinical"),
            "state_gene_set": declaration("state_gene_set"),
            "gene_set_ranked_subtype": gene_declaration,
        },
        "gene_set_ranked_subtype": {
            "status": gene_status,
            "closure_receipt": {
                "path": str(closure_path),
                "sha256": closure_sha,
                "format": RUNTIME.GENE_SET_CLOSURE_FORMAT,
                "status": "PASS",
            },
            "production_deployed": False,
            "release_ready": False,
        },
        "main_score_changed": False,
        "production_deployed": False,
        "release_ready": False,
    }


def test_runtime_manifest_fails_closed_when_geneset_metadata_is_pending(
    tmp_path: Path,
) -> None:
    manifest = _runtime_manifest(
        tmp_path,
        gene_status="DIRECT_PAYLOAD_VERIFIED_TRANSITIVE_METADATA_PENDING"
    )
    path = tmp_path / "RUNTIME_BINDINGS.json"
    digest = _json(path, manifest)
    with pytest.raises(RUNTIME.RuntimeSmokeError, match="metadata is not closed"):
        RUNTIME.load_runtime_manifest(path, digest)


def test_closed_runtime_builder_adds_only_pinned_gene_set_declaration(
    tmp_path: Path,
) -> None:
    existing = {
        role: {
            "binding_path": f"${PRIVATE_WORK_ROOT}/existing/{role}.json",
            "binding_sha256": "a" * 64,
            "scientific_status": "unchanged",
            "production_deployed": False,
            "release_ready": False,
        }
        for role in CLOSED_RUNTIME.EXPECTED_EXISTING_COMPONENTS
    }
    base_path = tmp_path / "BASE_RUNTIME.json"
    base_sha = _json(
        base_path,
        {
            "format": CLOSED_RUNTIME.FORMAT,
            "status": CLOSED_RUNTIME.STATUS,
            "server_root": "${PRIVATE_WORK_ROOT}/old",
            "bindings": existing,
            "gene_set_ranked_subtype": {
                "status": "DIRECT_PAYLOAD_VERIFIED_TRANSITIVE_METADATA_PENDING"
            },
            "main_score_changed": False,
            "production_deployed": False,
            "release_ready": False,
        },
    )
    declaration = {
        "binding_path": "./data/CancerLncAtlas/runtime/authorized_acceptance/test/release/binding.json",
        "binding_sha256": "b" * 64,
        "audit_binding_path": "./data/CancerLncAtlas/runtime/authorized_acceptance/test/audit/binding.json",
        "audit_binding_sha256": "c" * 64,
        "scientific_status": "deterministic_v32_exact_pathway_derived_functional_head",
        "production_deployed": False,
        "release_ready": False,
    }
    closure_path = tmp_path / "CLOSURE.json"
    closure_sha = _json(
        closure_path,
        {
            "format": CLOSED_RUNTIME.CLOSURE_FORMAT,
            "status": "PASS",
            "runtime_binding_declaration": declaration,
            "metadata_closure": {
                "transitive_metadata_closed": True,
                "portable_rebinder_pending_was_runtime_false_negative": True,
            },
            "exact_probes": {
                "cancers_typed": 33,
                "available_cancers": 31,
                "typed_unavailable": {"CHOL": "x", "UCS": "y"},
            },
            "family_to_exact_broadcast": False,
            "changes_primary_ranking": False,
            "production_deployed": False,
            "release_ready": False,
        },
    )
    result = CLOSED_RUNTIME.build(
        base_path,
        base_sha,
        closure_path,
        closure_sha,
        server_root="./data/CancerLncAtlas/runtime/authorized_acceptance/test/runtime",
    )
    assert set(result["bindings"]) == {
        *CLOSED_RUNTIME.EXPECTED_EXISTING_COMPONENTS,
        "gene_set_ranked_subtype",
    }
    assert result["bindings"]["gene_set_ranked_subtype"] == declaration
    assert result["gene_set_ranked_subtype"]["status"] == (
        "RUNTIME_TRANSITIVE_METADATA_CLOSED"
    )
    assert result["gene_set_ranked_subtype"]["closure_receipt"][
        "sha256"
    ] == closure_sha


class _Evidence:
    def __init__(self, _path, *, expected_binding_sha256):
        self.binding_sha256 = expected_binding_sha256

    def query_confidence(self, **_kwargs):
        return {
            "returned_rows": 1,
            "rows": [
                {
                    "cancer_id": "BRCA",
                    "lncrna_id": RUNTIME.PROBE_LNCRNA,
                    "pathway_id": "P1",
                    "event_count": 2,
                }
            ],
        }

    def query_events(self, **kwargs):
        return {
            "returned_rows": 1,
            "rows": [
                {
                    "cancer_id": kwargs["cancer_id"],
                    "lncrna_id": kwargs["lncrna_id"],
                    "pathway_id": kwargs["pathway_id"],
                    "family_broadcast_used": False,
                }
            ],
        }


class _Direction:
    def __init__(
        self,
        _path,
        *,
        expected_binding_sha256,
        audit_binding_path,
        expected_audit_binding_sha256,
    ):
        assert audit_binding_path
        self.binding_sha256 = expected_binding_sha256
        self.audit_binding_sha256 = expected_audit_binding_sha256

    def query(self, **_kwargs):
        return {
            "returned_rows": 1,
            "rows": [
                {
                    "direction_negative_probability": 0.2,
                    "direction_neutral_probability": 0.3,
                    "direction_positive_probability": 0.5,
                    "changes_primary_ranking": False,
                    "changes_discovery_ranking": False,
                    "used_for_fusion": False,
                }
            ],
        }


class _Clinical:
    events = 4
    patients = 10

    def __init__(self, _path, *, expected_binding_sha256):
        self.binding_sha256 = expected_binding_sha256

    def query_pair(self, **_kwargs):
        row = {
            "availability": True,
            "n_events": self.events,
            "n_endpoint_patients": self.patients,
            "events_high": 2,
            "events_low": self.events - 2,
            "n_high": 5,
            "n_low": 5,
        }
        return {
            "returned_statistics_rows": 1,
            "returned_curve_rows": 12,
            "statistics": [row],
            "curves": [{"point": index} for index in range(12)],
        }


class _State:
    def __init__(self, _path, *, expected_binding_sha256):
        self.binding_sha256 = expected_binding_sha256

    def query_gene_sets(self, **_kwargs):
        return {"returned_rows": 1, "rows": [{"gene_set_id": "STATE:ONE"}]}

    def query_members(self, **kwargs):
        return {
            "returned_rows": 1,
            "rows": [{"gene_set_id": kwargs["gene_set_id"], "member_id": "L1"}],
        }


class _GeneSet:
    def __init__(
        self,
        _path,
        *,
        expected_binding_sha256,
        audit_binding_path,
        expected_audit_binding_sha256,
    ):
        assert audit_binding_path
        self.binding_sha256 = expected_binding_sha256
        self.audit_binding_sha256 = expected_audit_binding_sha256

    def query_gene_set_catalog(self, **_kwargs):
        return {
            "returned_rows": 1,
            "rows": [{"geneset_id": "GS:ONE"}],
            "availability": {"available": True},
        }

    def query_gene_set_detail(self, **_kwargs):
        return {
            "member_returned_rows": 1,
            "gene_set": {"pathway_target_level": "exact_pathway"},
        }

    def query_subtype_overview(self):
        unavailable = RUNTIME.TYPED_UNAVAILABLE
        rows = [
            {
                "cancer_id": f"C{index:02d}",
                "available": True,
                "unavailable_reason": None,
            }
            for index in range(31)
        ]
        rows.extend(
            {
                "cancer_id": cancer,
                "available": False,
                "unavailable_reason": reason,
            }
            for cancer, reason in unavailable.items()
        )
        return {"returned_rows": 33, "rows": rows}


def _patch_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUNTIME, "EvidenceBindingQuery", _Evidence)
    monkeypatch.setattr(RUNTIME, "EvidenceDirectionProbabilityQuery", _Direction)
    monkeypatch.setattr(RUNTIME, "ClinicalKMReleaseQuery", _Clinical)
    monkeypatch.setattr(RUNTIME, "StateGeneSetReleaseQuery", _State)
    monkeypatch.setattr(RUNTIME, "GeneSetSubtypeReleaseQuery", _GeneSet)


def test_runtime_smoke_requires_nonempty_exact_probes_and_33c_typing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_queries(monkeypatch)
    path = tmp_path / "RUNTIME_BINDINGS.json"
    digest = _json(path, _runtime_manifest(tmp_path))
    payload = RUNTIME.run_smoke(path, digest)
    assert payload["status"] == "PASS"
    assert payload["components"]["evidence"]["event_probe_rows"] == 1
    assert payload["components"]["clinical"]["curve_rows"] == 12
    assert payload["components"]["state_gene_set"]["member_probe_rows"] == 1
    gene = payload["components"]["gene_set_ranked_subtype"]
    assert gene["cancers_typed"] == 33
    assert gene["available_cancers"] == 31
    assert set(gene["typed_unavailable"]) == {"CHOL", "UCS"}
    assert payload["manual_assay_exactness_closed"] is False
    assert payload["family_level_assay_gap_closed"] is False


def test_runtime_smoke_rejects_clinical_events_above_patients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class InvalidClinical(_Clinical):
        events = 11
        patients = 10

    _patch_queries(monkeypatch)
    monkeypatch.setattr(RUNTIME, "ClinicalKMReleaseQuery", InvalidClinical)
    path = tmp_path / "RUNTIME_BINDINGS.json"
    digest = _json(path, _runtime_manifest(tmp_path))
    with pytest.raises(RUNTIME.RuntimeSmokeError, match="event count exceeds"):
        RUNTIME.run_smoke(path, digest)


def _core_unreachable_receipt(
    tmp_path: Path, manifest_sha: str, binding_sha: str
) -> tuple[Path, str]:
    path = tmp_path / "DRUG_CORE_UNREACHABLE.json"
    digest = _json(
        path,
        {
            "format": DRUG.CORE_REACHABILITY_FORMAT,
            "status": "PASS",
            "failure_reason": DRUG.CURRENT_V32_CORE_UNAVAILABLE,
            "reachability_status": DRUG.CORE_REACHABILITY_STATUS,
            "formal_manifest": {"sha256": manifest_sha},
            "authorized_binding": {"sha256": binding_sha},
            "artifact_root_authority": {"path": str(tmp_path.resolve())},
            "exhaustive_audit": {
                "semantic_sampling": False,
                "duckdb_used": False,
                "exact_rows_examined": 3_300_000,
                "conceptual_nonempty_assay_intersect_expression_combinations_examined": 44_344_518,
                "all_eligible_folds_missing_at_least_one_core_type": 0,
                "reachable_current_v32_core_unavailable_keys": 0,
            },
            "resource_gate": {
                "passed": True,
                "kernel_peak_rss_kB": 100_000,
                "max_peak_rss_kB": 2 * 1024 * 1024,
            },
            "api_defensive_enum_retained": True,
            "changes_primary_ranking": False,
            "production_deployed": False,
            "release_ready": False,
        },
    )
    return path, digest


def _drug_contract(
    manifest_sha: str,
    core_receipt_path: Path,
    core_receipt_sha: str,
    *,
    binding_sha: str,
) -> dict:
    probes = {}
    for index, (outcome, (available, reason)) in enumerate(
        DRUG.SUCCESS_PROBE_OUTCOMES.items()
    ):
        probes[outcome] = {
            "cancer_id": "BRCA",
            "lncrna_id": f"LNC:L{index}",
            "drug_id": f"DRUG:D{index}",
            "expected_availability": available,
            "expected_failure_reason": reason,
        }
    return {
        "format": DRUG.PROBE_FORMAT,
        "manifest_sha256": manifest_sha,
        "artifact_root": str(core_receipt_path.parent.resolve()),
        "authorized_binding_sha256": binding_sha,
        "model_run_status": "SUCCESS",
        "semantic_sampling": False,
        "non_applicable_typed_absence": {
            DRUG.MODEL_RUN_AUDITED_UNAVAILABLE: (
                DRUG.MODEL_RUN_REASON_NOT_APPLICABLE
            )
        },
        "exhaustive_unreachable_typed_absence": {
            DRUG.CURRENT_V32_CORE_UNAVAILABLE: {
                "status": DRUG.CORE_REACHABILITY_STATUS,
                "receipt": {
                    "path": str(core_receipt_path),
                    "sha256": core_receipt_sha,
                },
            }
        },
        "probes": probes,
    }


class _DrugBundle:
    def __init__(self, contract: dict, *, drift_outcome: str | None = None):
        self.lookup = {
            tuple(probe[key] for key in ("cancer_id", "lncrna_id", "drug_id")): (
                outcome,
                probe,
            )
            for outcome, probe in contract["probes"].items()
        }
        self.drift_outcome = drift_outcome

    def resolve(self, cancer_id, lncrna_id, drug_id):
        outcome, probe = self.lookup[(cancer_id, lncrna_id, drug_id)]
        reason = probe["expected_failure_reason"]
        if outcome == self.drift_outcome:
            reason = DRUG.OUTSIDE_CONCEPTUAL_UNIVERSE
        return {
            "cancer_id": cancer_id,
            "lncrna_id": lncrna_id,
            "drug_id": drug_id,
            "drug_response_association_probability": (
                0.75 if probe["expected_availability"] else None
            ),
            "availability": probe["expected_availability"],
            "failure_reason": reason,
            "scientific_status": "diagnostic_only",
            "tcga_patient_response_claimed": False,
        }


def test_drug_probe_contract_covers_every_success_bundle_reason(
    tmp_path: Path,
) -> None:
    manifest_sha = "c" * 64
    binding_sha = "b" * 64
    receipt_path, receipt_sha = _core_unreachable_receipt(
        tmp_path, manifest_sha, binding_sha
    )
    contract = _drug_contract(
        manifest_sha,
        receipt_path,
        receipt_sha,
        binding_sha=binding_sha,
    )
    path = tmp_path / "DRUG_TYPED_PROBES.json"
    digest = _json(path, contract)
    loaded = DRUG.load_probe_contract(path, digest, manifest_sha, tmp_path)
    observed = DRUG.validate_exact_probes(_DrugBundle(loaded), loaded)
    assert set(observed) == set(DRUG.SUCCESS_PROBE_OUTCOMES)
    assert loaded["non_applicable_typed_absence"] == {
        DRUG.MODEL_RUN_AUDITED_UNAVAILABLE: DRUG.MODEL_RUN_REASON_NOT_APPLICABLE
    }
    assert loaded["exhaustive_unreachable_typed_absence"] == {
        DRUG.CURRENT_V32_CORE_UNAVAILABLE: {
            "status": DRUG.CORE_REACHABILITY_STATUS,
            "receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        }
    }


def test_drug_probe_contract_and_result_drift_fail_closed(tmp_path: Path) -> None:
    manifest_sha = "d" * 64
    binding_sha = "b" * 64
    receipt_path, receipt_sha = _core_unreachable_receipt(
        tmp_path, manifest_sha, binding_sha
    )
    contract = _drug_contract(
        manifest_sha,
        receipt_path,
        receipt_sha,
        binding_sha=binding_sha,
    )
    missing = dict(contract)
    missing["probes"] = dict(contract["probes"])
    missing["probes"].pop(DRUG.NO_MATCHED_LNCRNA_EXPRESSION)
    path = tmp_path / "MISSING.json"
    digest = _json(path, missing)
    with pytest.raises(DRUG.DrugSmokeError, match="outcome set"):
        DRUG.load_probe_contract(path, digest, manifest_sha, tmp_path)

    valid_path = tmp_path / "VALID.json"
    valid_digest = _json(valid_path, contract)
    loaded = DRUG.load_probe_contract(
        valid_path, valid_digest, manifest_sha, tmp_path
    )
    with pytest.raises(DRUG.DrugSmokeError, match="reason failed"):
        DRUG.validate_exact_probes(
            _DrugBundle(
                loaded,
                drift_outcome=DRUG.MODEL_OUTPUT_MISSING_FAIL_CLOSED,
            ),
            loaded,
        )


def test_drug_probe_contract_rejects_false_core_unreachable_receipt(
    tmp_path: Path,
) -> None:
    manifest_sha = "e" * 64
    binding_sha = "b" * 64
    receipt_path, _ = _core_unreachable_receipt(
        tmp_path, manifest_sha, binding_sha
    )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["exhaustive_audit"]["reachable_current_v32_core_unavailable_keys"] = 1
    receipt_sha = _json(tmp_path / "FALSE_CORE_RECEIPT.json", payload)
    contract = _drug_contract(
        manifest_sha,
        tmp_path / "FALSE_CORE_RECEIPT.json",
        receipt_sha,
        binding_sha=binding_sha,
    )
    contract_path = tmp_path / "FALSE_CORE_CONTRACT.json"
    contract_sha = _json(contract_path, contract)
    with pytest.raises(DRUG.DrugSmokeError, match="exhaustive receipt"):
        DRUG.load_probe_contract(
            contract_path, contract_sha, manifest_sha, tmp_path
        )


def test_drug_runtime_spill_tree_is_removed_and_environment_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spill_parent = tmp_path / "formal-spill-parent"
    env_name = "CC_HHGT_DRUG_SPARSE_DUCKDB_TEMP_DIRECTORY"
    previous = str(tmp_path / "previous-spill-base")
    monkeypatch.setenv(env_name, previous)
    run_path: Path | None = None
    with pytest.raises(RuntimeError, match="deliberate smoke failure"):
        with DRUG.dedicated_duckdb_runtime_environment(
            spill_parent,
            memory_limit="512MB",
            threads="1",
            max_temp_size="4GB",
        ) as runtime_path:
            run_path = runtime_path
            (runtime_path / "failure-path-sentinel").write_text(
                "spill", encoding="utf-8"
            )
            raise RuntimeError("deliberate smoke failure")
    assert run_path is not None
    assert not run_path.exists()
    assert list(spill_parent.iterdir()) == []
    assert __import__("os").environ[env_name] == previous


def test_drug_runtime_cli_requires_explicit_artifact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_v32_authorized_drug_partitioned_server.py",
            "--manifest",
            "manifest.json",
            "--sha256",
            "0" * 64,
            "--probes",
            "probes.json",
            "--probes-sha256",
            "1" * 64,
            "--temp-directory",
            "temp",
            "--output",
            "out.json",
        ],
    )
    with pytest.raises(SystemExit) as error:
        DRUG.main()
    assert error.value.code == 2
