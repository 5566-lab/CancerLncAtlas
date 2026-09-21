from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.runtime_web_acceptance import (  # noqa: E402
    HTML_51_ROUTE_CONTRACT,
    TCGA_CANCERS,
    ProbeOutcome,
    ProbeSpec,
    build_download_specs,
    check_candidate_classification,
    check_clinical_invariant,
    check_downloads,
    check_final_binding_attestation,
    check_mutation_status,
    check_optional_components,
    check_runtime_version_lineage,
    check_single_cell_science,
    execute_runtime_audit,
    load_final_binding_evidence,
    render_markdown,
    run_specs,
    validate_candidate_base_url,
)


def _outcome(
    probe_id: str,
    payload: object,
    *,
    status: int = 200,
    group: str = "test",
    cancer: str | None = None,
    expected: tuple[int, ...] = (200,),
) -> ProbeOutcome:
    return ProbeOutcome(
        spec=ProbeSpec(
            probe_id,
            "/test",
            "/test",
            group=group,
            cancer_id=cancer,
            expected_statuses=expected,
        ),
        status=status,
        payload=payload,
        content_type="application/json",
    )


def test_contract_is_exactly_51_routes_and_33_cancers() -> None:
    assert len(HTML_51_ROUTE_CONTRACT) == 51
    assert len(set(HTML_51_ROUTE_CONTRACT)) == 51
    assert len(TCGA_CANCERS) == 33
    assert len(set(TCGA_CANCERS)) == 33


def test_single_cell_contract_probe_supplies_required_cancer_lncrna_pair() -> None:
    from cc_hhgt.v32.runtime_web_acceptance import build_route_contract_specs

    context = {
        "candidate_lncrna": "LNC:ENSG00000123456",
        "candidate_pathway": "PATH:1",
        "geneset_id": "PATH:1",
        "figure_id": "umap",
        "download_key": "model-card",
        "fallbacks": {},
    }
    probe = next(
        spec for spec in build_route_contract_specs(context) if spec.probe_id == "route.17"
    )
    assert probe.path.startswith("/api/site/single-cell?")
    assert "cancer=BRCA" in probe.path
    assert "lnc=LNC%3AENSG00000123456" in probe.path
    assert "limit=1" in probe.path


def test_discovery_uses_materialized_exact_gene_set_key() -> None:
    from cc_hhgt.v32.runtime_web_acceptance import derive_context

    discovery = [
        _outcome(
            "discovery.candidate",
            {
                "results": [
                    {
                        "lncrna_id": "LNC:ENSG00000123456",
                        "pathway_family_id": "PF:0001",
                    }
                ]
            },
        ),
        _outcome(
            "discovery.geneset",
            {"gene_sets": [{"geneset_id": "LGS:fresh-v32-exact"}]},
        ),
        _outcome("discovery.figure", {"figures": [{"figure_id": "umap"}]}),
        _outcome(
            "discovery.download",
            {"files": [{"key": "model-card"}]},
        ),
    ]
    context = derive_context(discovery)
    assert context["geneset_id"] == "LGS:fresh-v32-exact"
    assert context["fallbacks"]["geneset"] is False


def test_candidate_url_is_loopback_and_production_port_is_rejected() -> None:
    assert validate_candidate_base_url("http://127.0.0.1:8262/") == (
        "http://127.0.0.1:8262"
    )
    with pytest.raises(ValueError, match="production"):
        validate_candidate_base_url("http://127.0.0.1:8260")
    with pytest.raises(ValueError, match="loopback"):
        validate_candidate_base_url("https://example.org:8262")
    with pytest.raises(ValueError, match="application path"):
        validate_candidate_base_url("http://localhost:8262/api")


def test_run_specs_obeys_worker_bound_and_preserves_order() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_requester(
        _base: str, spec: ProbeSpec, timeout: float, max_body: int
    ) -> ProbeOutcome:
        nonlocal active, peak
        assert timeout == 1.5
        assert max_body == 4096
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return ProbeOutcome(spec=spec, status=200, payload={"status": "available"})

    specs = [ProbeSpec(f"p{index}", "/test", f"/test/{index}") for index in range(20)]
    outcomes = run_specs(
        "http://127.0.0.1:8262",
        specs,
        workers=3,
        timeout_seconds=1.5,
        max_body_bytes=4096,
        requester=fake_requester,
    )
    assert 1 < peak <= 3
    assert [row.spec.probe_id for row in outcomes] == [f"p{index}" for index in range(20)]


def test_clinical_invariant_detects_duplicate_oof_counting() -> None:
    good = _outcome(
        "clinical",
        {
            "status": "available",
            "endpoints": [
                {"cancer_id": "BLCA", "endpoint": "OS", "n_patients": 404, "n_events": 180},
                {"cancer_id": "GBM", "endpoint": "OS", "n_patients": 287, "n_events": 210},
            ],
        },
    )
    assert check_clinical_invariant(good)["pass"] is True

    bad = _outcome(
        "clinical",
        {
            "status": "available",
            "endpoints": [
                {"cancer_id": "BLCA", "endpoint": "OS", "n_patients": 404, "n_events": 531},
                {"cancer_id": "BLCA", "endpoint": "OS", "n_patients": 404, "n_events": 531},
            ],
        },
    )
    check = check_clinical_invariant(bad)
    assert check["pass"] is False
    assert check["violation_count"] == 2
    assert check["duplicate_key_count"] == 1


def test_predicted_candidate_runtime_check_requires_nonzero_and_correct_class() -> None:
    overview_rows = [
        {
            "cancer_id": cancer,
            "predicted_candidate_relations": 7 if cancer == "BRCA" else 0,
        }
        for cancer in TCGA_CANCERS
    ]
    overview = _outcome("overview", {"cancers": overview_rows})
    sweep = []
    for cancer in TCGA_CANCERS:
        total = 7 if cancer == "BRCA" else 0
        rows = [{"confidence_tier": "predicted_candidate"}] if total else []
        sweep.append(
            _outcome(
                f"predicted.{cancer}",
                {"status": "available" if total else "no_evidence", "total": total, "results": rows},
                group="cancer_sweep:predicted_candidate",
                cancer=cancer,
            )
        )
    check = check_candidate_classification(overview, sweep, expected_bound_total=7)
    assert check["pass"] is True
    assert check["overview_predicted_total"] == 7
    assert check["sweep_predicted_total"] == 7
    assert check["nonzero_cancers"] == ["BRCA"]
    assert check["binding_total_matches"] is True

    mismatched = list(sweep)
    mismatched[0] = _outcome(
        "predicted.ACC",
        {"status": "available", "total": 2, "results": []},
        group="cancer_sweep:predicted_candidate",
        cancer="ACC",
    )
    failed = check_candidate_classification(
        overview, mismatched, expected_bound_total=7
    )
    assert failed["pass"] is False
    assert failed["count_disagreements"][0]["cancer_id"] == "ACC"


def test_optional_typed_unavailable_passes_http_contract_not_science() -> None:
    lnc = _outcome(
        "lnc",
        {
            "status": "available",
            "partial_components_do_not_fail_request": True,
            "pathway_heatmap": {"status": "source_unavailable", "reason": "missing"},
            "bulk_scatter": {"status": "available", "points": []},
            "survival_curve": {"status": "source_unavailable", "reason": "missing"},
            "drug_evidence": {"status": "no_evidence", "rows": []},
        },
    )
    clinical = _outcome(
        "clinical",
        {
            "status": "available",
            "kaplan_meier": {"status": "source_unavailable", "reason": "missing"},
            "event_count_status": "available",
        },
    )
    check = check_optional_components(lnc, clinical)
    assert check["pass"] is True
    assert check["scientific_available"] is False
    assert check["typed_unavailable_component_count"] == 3


def test_mutation_requires_v32_fresh_attestation_separately_from_http() -> None:
    sweep = []
    for family in ("mutation_site", "mutation_compat"):
        for cancer in TCGA_CANCERS:
            sweep.append(
                _outcome(
                    f"{family}.{cancer}",
                    {"status": "available"},
                    group=f"cancer_sweep:{family}",
                    cancer=cancer,
                )
            )
    available = _outcome(
        "mutation.status",
        {
            "status": "available",
            "mutation_release_available": True,
            "mutation_model_release": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "new_training": True,
            "old_predictions_used": False,
        },
    )
    check = check_mutation_status(available, sweep)
    assert check["http_pass"] is True
    assert check["scientific_available"] is True
    assert check["pass"] is True

    unavailable = _outcome(
        "mutation.status",
        {
            "status": "release_unavailable",
            "mutation_release_available": False,
            "mutation_model_release": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "new_training": True,
            "old_predictions_used": False,
        },
    )
    check = check_mutation_status(unavailable, sweep)
    assert check["http_pass"] is True
    assert check["scientific_available"] is False
    assert check["pass"] is False


def test_download_catalog_probes_every_key_and_rejects_external_url() -> None:
    catalog = _outcome(
        "catalog",
        {
            "files": [
                {"key": "one", "download_url": "/api/site/download/one"},
                {"key": "two", "download_url": "/api/site/download/two"},
            ]
        },
    )
    specs, unsafe = build_download_specs(catalog, "http://127.0.0.1:8262")
    assert unsafe == []
    assert len(specs) == 2
    outcomes = [ProbeOutcome(spec=spec, status=206) for spec in specs]
    download_check = check_downloads(catalog, outcomes, unsafe)
    assert download_check["pass"] is True
    assert download_check["http_pass"] is True
    assert download_check["scientific_available"] is False

    attested_catalog = _outcome(
        "catalog.attested",
        {
            "files": [
                {
                    "key": "one",
                    "download_url": "/api/site/download/one",
                    "status": "available",
                    "scientifically_attested": True,
                    "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                    "sha256": "a" * 64,
                }
            ]
        },
    )
    attested_catalog.body_sample_sha256 = "f" * 64
    attested_specs, attested_unsafe = build_download_specs(
        attested_catalog, "http://127.0.0.1:8262"
    )
    attested_outcomes = [ProbeOutcome(spec=attested_specs[0], status=206)]
    attested_check = check_downloads(
        attested_catalog,
        attested_outcomes,
        attested_unsafe,
        expected_catalog_payload_sha256="f" * 64,
        expected_file_count=1,
    )
    assert attested_check["pass"] is True
    assert attested_check["scientific_available"] is True

    external = _outcome(
        "catalog",
        {"files": [{"key": "bad", "download_url": "https://example.org/bad"}]},
    )
    specs, unsafe = build_download_specs(external, "http://127.0.0.1:8262")
    assert specs == []
    assert unsafe == ["bad:external_or_wrong_origin_url"]


def test_single_cell_http_coverage_does_not_promote_typed_partial_science() -> None:
    outcomes = []
    for family in ("sc_summary", "sc_umap"):
        for cancer in TCGA_CANCERS:
            payload = {"status": "available"}
            if family == "sc_summary" and cancer == "ACC":
                payload = {"status": "source_unavailable", "reason": "pending"}
            outcomes.append(
                _outcome(
                    f"{family}.{cancer}",
                    payload,
                    group=f"cancer_sweep:{family}",
                    cancer=cancer,
                )
            )
    check = check_single_cell_science(outcomes)
    assert check["http_pass"] is True
    assert check["scientific_available"] is False
    assert check["families"]["sc_summary"]["scientifically_available_count"] == 32


def test_final_binding_rehashes_roles_and_runtime_marker_echoes_lineage(
    tmp_path: Path,
) -> None:
    roles = (
        "winner_selection_authority",
        "web_relationship_materialization",
        "download_catalog_binding",
        "clinical_binding",
        "exact_geneset_binding",
        "mutation_binding",
        "single_cell_binding",
    )
    artifacts = []
    for role in roles:
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps({"role": role}), encoding="utf-8")
        artifacts.append(
            {
                "role": role,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    binding = {
        "format": "CANCERLNCATLAS_V32_FINAL_WEB_BINDING_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "model_version": "V3.2",
        "status": "FINAL_CANDIDATE_BOUND",
        "release_ready": True,
        "production_deployed": False,
        "fresh_training": True,
        "old_predictions_used": False,
        "old_rankings_used": False,
        "winner_selection": {
            "winner_id": "hierarchical_end_to_end",
            "selection_split": "validation_only",
            "heldout_test_used_for_selection": False,
            "fair_comparison": True,
            "compared_models": ["external_router", "hierarchical_end_to_end"],
            "winner_score": 0.81,
            "fold_prediction_sha256": [f"{index:064x}" for index in range(1, 6)],
            "checkpoint_sha256": [f"{index:064x}" for index in range(6, 11)],
        },
        "modules": {
            "web_relationships": {
                "status": "AVAILABLE",
                "cancer_count": 33,
                "predicted_candidate_total": 1,
                "exclusive_relationship_routing": True,
            },
            "downloads": {
                "status": "AVAILABLE",
                "scientifically_attested": True,
                "catalog_payload_sha256": "f" * 64,
                "file_count": 1,
            },
            "clinical": {
                "status": "AVAILABLE",
                "events_le_patients_invariant": True,
            },
            "exact_geneset": {
                "status": "AVAILABLE",
                "target_level": "exact_pathway",
                "family_broadcast_used": False,
            },
            "mutation": {
                "status": "AVAILABLE",
                "cancer_count": 33,
                "fresh_training": True,
                "old_predictions_used": False,
                "old_rankings_used": False,
            },
            "single_cell": {
                "status": "AVAILABLE",
                "cancer_count": 33,
                "fresh_derived_cancer_count": 33,
                "fresh_recompute": True,
                "historical_assets_relabelled_fresh": False,
            },
        },
        "artifacts": artifacts,
    }
    binding_path = tmp_path / "FINAL_BINDING.json"
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    evidence = load_final_binding_evidence(
        binding_path,
        expected_sha256=binding_sha,
        artifact_root=tmp_path,
    )
    attestation = check_final_binding_attestation(evidence)
    assert attestation["pass"] is True

    marker = _outcome(
        "marker",
        {
            "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
            "model_version": "V3.2",
            "final_binding_sha256": binding_sha,
            "winner_id": "hierarchical_end_to_end",
            "fresh_training": True,
            "old_predictions_used": False,
            "old_rankings_used": False,
            "production_deployed": False,
        },
    )
    assert check_runtime_version_lineage(marker, attestation)["pass"] is True

    artifacts[0]["sha256"] = "0" * 64
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    drifted_binding_sha = hashlib.sha256(binding_path.read_bytes()).hexdigest()
    drifted = load_final_binding_evidence(
        binding_path,
        expected_sha256=drifted_binding_sha,
        artifact_root=tmp_path,
    )
    assert check_final_binding_attestation(drifted)["pass"] is False


def test_end_to_end_fake_transport_is_json_serializable_and_separates_verdicts() -> None:
    openapi_paths: dict[str, dict[str, object]] = {}
    for method, path in HTML_51_ROUTE_CONTRACT:
        openapi_paths.setdefault(path, {})[method.lower()] = {"responses": {"200": {}}}

    def fake_requester(
        _base: str, spec: ProbeSpec, _timeout: float, _max_body: int
    ) -> ProbeOutcome:
        payload: object = {"status": "available"}
        if spec.probe_id == "preflight.candidate_marker":
            payload = {"status": "ok", "production_deployed": False}
        elif spec.probe_id == "preflight.openapi":
            payload = {"openapi": "3.1.0", "paths": openapi_paths}
        elif spec.probe_id in {"discovery.candidate", "route.29"}:
            payload = {
                "status": "available",
                "total": 1,
                "results": [
                    {"lncrna_id": "MALAT1", "pathway_family_id": "PF:0001"}
                ],
            }
        elif spec.probe_id in {"discovery.geneset", "route.13"}:
            payload = {"status": "available", "results": [{"geneset_id": "PATH:1"}]}
        elif spec.probe_id in {"discovery.figure", "route.23"}:
            payload = {"status": "available", "figures": [{"figure_id": "umap"}]}
        elif spec.probe_id in {"discovery.download", "route.46"}:
            payload = {
                "status": "available",
                "files": [
                    {"key": "model-card", "download_url": "/api/site/download/model-card"}
                ],
            }
        elif spec.probe_id == "route.09":
            payload = {"query": "肺癌", "results": [{"id": "LUAD"}]}
        elif spec.probe_id == "route.11":
            payload = {
                "status": "available",
                "partial_components_do_not_fail_request": True,
                "pathway_heatmap": {"status": "available"},
                "bulk_scatter": {"status": "available"},
                "survival_curve": {"status": "available"},
                "drug_evidence": {"status": "no_evidence"},
            }
        elif spec.probe_id == "route.14":
            payload = {
                "status": "available",
                "pathway_target_level": "exact_pathway",
                "family_broadcast_used": False,
            }
        elif spec.probe_id == "route.26":
            payload = {
                "status": "available",
                "cancers": [
                    {
                        "cancer_id": cancer,
                        "predicted_candidate_relations": 1 if cancer == "BRCA" else 0,
                    }
                    for cancer in TCGA_CANCERS
                ],
            }
        elif spec.probe_id == "route.32":
            payload = {
                "status": "available",
                "endpoints": [
                    {
                        "cancer_id": cancer,
                        "endpoint": "OS",
                        "n_patients": 10,
                        "n_events": 4,
                    }
                    for cancer in TCGA_CANCERS
                ],
            }
        elif spec.probe_id == "route.36":
            payload = {
                "status": "available",
                "kaplan_meier": {"status": "available"},
                "event_count_status": "available",
            }
        elif spec.probe_id == "route.37":
            payload = {
                "status": "release_unavailable",
                "mutation_release_available": False,
                "mutation_model_release": "CancerLncAtlas_V3.2_FULL_MULTITASK",
                "new_training": True,
                "old_predictions_used": False,
            }
        elif spec.group == "cancer_sweep:predicted_candidate":
            total = 1 if spec.cancer_id == "BRCA" else 0
            payload = {
                "status": "available" if total else "no_evidence",
                "total": total,
                "results": (
                    [{"confidence_tier": "predicted_candidate"}] if total else []
                ),
            }
        elif spec.group.startswith("cancer_sweep:"):
            payload = {"status": "available"}
        status = 206 if spec.group == "download_keys" else 200
        return ProbeOutcome(
            spec=spec,
            status=status,
            payload=payload,
            content_type="application/json",
        )

    report = execute_runtime_audit(
        "http://127.0.0.1:8262",
        workers=4,
        timeout_seconds=1,
        max_body_bytes=4096,
        requester=fake_requester,
    )
    assert report["coverage"]["historical_routes_observed"] == 51
    assert report["coverage"]["cancer_matrix_probes_observed"] == 297
    assert report["verdicts"]["http_remediation_pass"] is True
    assert report["verdicts"]["data_processing_invariants_pass"] is True
    assert report["verdicts"]["scientific_artifacts_available"] is False
    assert report["checks"]["mutation"]["http_pass"] is True
    assert report["checks"]["mutation"]["scientific_available"] is False
    json.dumps(report, ensure_ascii=False)
    markdown = render_markdown(report)
    assert "Historical 51-route results" in markdown
    assert "Scientific artifacts available | FAIL" in markdown
