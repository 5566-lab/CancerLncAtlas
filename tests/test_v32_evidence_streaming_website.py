from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.evidence_streaming_website import (
    ANALYSIS_VERSION,
    EVENT_COUNT_CAP,
    INDEPENDENT_AUDIT_FORMAT,
    NO_EVENT_REASON,
    StreamingEvidenceAssetError,
    StreamingEvidenceWebsiteQuery,
    materialize_streaming_evidence_website_binding,
)
from website.backend.v32_evidence_candidate_api import create_evidence_candidate_app


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(root: Path, *, publication_allowed: bool = False) -> dict[str, Path | str]:
    root.mkdir()
    prediction = root / "EVIDENCE_PREDICTIONS.parquet"
    pd.DataFrame(
        [
            {
                "cancer_id": "ACC",
                "lncrna_id": "ENSG000001",
                "pathway_id": "HALLMARK:A",
                "evidence_confidence_probability": 0.8,
                "direction": "positive",
                "uncertainty": 0.2,
                "availability": True,
                "unavailable_reason": "",
                "event_count": EVENT_COUNT_CAP,
                "evidence_fold": 0.0,
                "failure_reason": "",
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": "V32-EVIDENCE-STREAMING-R1",
                "changes_primary_ranking": False,
                "main_ranking_modified": False,
            },
            {
                "cancer_id": "ACC",
                "lncrna_id": "ENSG000001",
                "pathway_id": "HALLMARK:NO_EVENT",
                "evidence_confidence_probability": None,
                "direction": None,
                "uncertainty": None,
                "availability": False,
                "unavailable_reason": NO_EVENT_REASON,
                "event_count": 0,
                "evidence_fold": None,
                "failure_reason": NO_EVENT_REASON,
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": "V32-EVIDENCE-STREAMING-R1",
                "changes_primary_ranking": False,
                "main_ranking_modified": False,
            },
            {
                "cancer_id": "BRCA",
                "lncrna_id": "LNC:ENSG000001",
                "pathway_id": "HALLMARK:A",
                "evidence_confidence_probability": 0.6,
                "direction": "negative",
                "uncertainty": 0.4,
                "availability": True,
                "unavailable_reason": "",
                "event_count": 1,
                "evidence_fold": 1.0,
                "failure_reason": "",
                "analysis_version": ANALYSIS_VERSION,
                "training_run_id": "V32-EVIDENCE-STREAMING-R1",
                "changes_primary_ranking": False,
                "main_ranking_modified": False,
            },
        ]
    ).to_parquet(prediction, index=False)

    events = root / "candidate_exact_events.parquet"
    event_rows = []
    for index in range(70):
        even = index % 2 == 0
        event_rows.append(
            {
                "event_id": f"BAGEV:{index + 1}",
                "source_event_id": f"EV:{index + 1}",
                "cancer_id": "ACC",
                "lncrna_id": "ENSG000001",
                "pathway_id": "HALLMARK:A",
                "partner_id": f"ENSGP{index + 1}",
                "member_type": "gene",
                "route_type": "PARTNER_EXACT_MEMBER",
                "source_database": "RNAInter" if even else "ENCORI",
                "source_dataset": "experimental" if even else "binding",
                "source_record_id": f"SRC:{index + 1}",
                "pmid": "123" if even else "",
                "experiment_type": "knockdown" if even else "binding",
                "relation_type": "regulation" if even else "interaction",
                "direction_raw": "up" if even else "unknown",
                "direction_target": 2 if even else -1,
                "confidence_target": 1.0 if even else None,
                "tissue": "adrenal" if even else "unknown",
                "cell_line": "x" if even else "unknown",
                "species": "Homo sapiens",
                "is_experimental": True,
                "is_computational": False,
                "is_physical": True,
                "is_model_prediction": False,
            }
        )
    event_rows.append(
        {
            "event_id": "BAGEV:71",
            "source_event_id": "EV:71",
            "cancer_id": "BRCA",
            "lncrna_id": "LNC:ENSG000001",
            "pathway_id": "HALLMARK:A",
            "partner_id": "ENSGP71",
            "member_type": "gene",
            "route_type": "PARTNER_EXACT_MEMBER",
            "source_database": "RNAInter",
            "source_dataset": "experimental",
            "source_record_id": "SRC:71",
            "pmid": "456",
            "experiment_type": "knockdown",
            "relation_type": "regulation",
            "direction_raw": "down",
            "direction_target": 0,
            "confidence_target": 1.0,
            "tissue": "breast",
            "cell_line": "y",
            "species": "Homo sapiens",
            "is_experimental": True,
            "is_computational": False,
            "is_physical": True,
            "is_model_prediction": False,
        }
    )
    pd.DataFrame(event_rows).to_parquet(events, index=False)

    lineage = root / "event_lineage.parquet"
    pd.DataFrame(
        [
            {
                "lineage_id": f"LIN:{index + 1}",
                "event_id": f"EV:{index + 1}",
                "source_kind": "interaction_relation",
                "source_row_sha256": f"{index + 1:064x}",
                "mapping_route": "PARTNER_EXACT_MEMBER",
                "family_broadcast_used": False,
            }
            for index in range(71)
        ]
    ).to_parquet(lineage, index=False)

    training = root / "TRAINING_MANIFEST.json"
    _write_json(
        training,
        {
            "format": "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_TRAINING_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_FRESH_V32_EVIDENCE_STREAMING_TRAINING",
            "production_deployed": False,
            "port_8260_touched": False,
        },
    )
    stage = root / "STAGING_MANIFEST.json"
    _write_json(
        stage,
        {
            "format": "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_STAGE_V1",
            "strict_formal_3_3m_contract": True,
            "production_deployed": False,
            "production_port_8260_touched": False,
            "counts": {"candidate_exact_events": 71, "event_lineage": 71},
            "artifacts": {
                "candidate_exact_events.parquet": {"sha256": _sha(events)},
                "event_lineage.parquet": {"sha256": _sha(lineage)},
            },
        },
    )
    manifest = root / "PREDICTION_MANIFEST.json"
    _write_json(
        manifest,
        {
            "format": "CANCERLNCATLAS_V32_EVIDENCE_STREAMING_PREDICTION_V1",
            "analysis_version": ANALYSIS_VERSION,
            "status": "PASS_FRESH_V32_EVIDENCE_STREAMING_PREDICTION",
            "production_deployed": False,
            "port_8260_touched": False,
            "prediction_rows": 3,
            "available_rows": 2,
            "unavailable_rows": 1,
            "predictions_sha256": _sha(prediction),
            "training_manifest_sha256": _sha(training),
        },
    )
    audit = root / "AUDIT.json"
    _write_json(
        audit,
        {
            "format": INDEPENDENT_AUDIT_FORMAT,
            "status": (
                "PASS_INDEPENDENT_AUDIT_PUBLICATION_ALLOWED"
                if publication_allowed
                else "BLOCKED_INDEPENDENT_AUDIT_NOT_PUBLISHABLE"
            ),
            "analysis_version": ANALYSIS_VERSION,
            "auditor_independent_of_training_implementation": True,
            "training_module_imported": False,
            "sealed_test_opened": False,
            "production_deployed": False,
            "port_8260_touched": False,
            "universe_rows": 3,
            "prediction_rows": 3,
            "available_rows": 2,
            "unavailable_rows": 1,
            "technical_acceptance": True,
            "technical_prediction_pass": True,
            "technical_failures": [],
            "publication_allowed": publication_allowed,
            "closure_allowed": publication_allowed,
            "publication_blockers": (
                []
                if publication_allowed
                else ["event_cap_quality_priority", "fresh_current_g2_core_lineage"]
            ),
            "checks": {
                "event_cap_quality_priority": {
                    "passed": publication_allowed,
                    "capped_bags": 1,
                    "max_raw_event_count": 70,
                }
            },
            "prediction": {"predictions_sha256": _sha(prediction)},
            "core_lineage_verdict": (
                "CURRENT_PATIENT_FIRST_V2"
                if publication_allowed
                else "SUPERSEDED_FORMAL_RELEASE_PREPARED_1SEED_V1"
            ),
        },
    )
    output = root.parent / f"{root.name}_binding"
    result = materialize_streaming_evidence_website_binding(
        prediction_manifest_path=manifest,
        predictions_path=prediction,
        training_manifest_path=training,
        stage_manifest_path=stage,
        candidate_events_path=events,
        event_lineage_path=lineage,
        independent_audit_path=audit,
        expected_independent_audit_sha256=_sha(audit),
        output_root=output,
        strict_formal=False,
    )
    return {
        "prediction": prediction,
        "events": events,
        "lineage": lineage,
        "audit": audit,
        "prediction_manifest": manifest,
        "training_manifest": training,
        "stage_manifest": stage,
        "binding": Path(result["binding_path"]),
        "binding_sha256": str(result["binding_sha256"]),
        "output": output,
    }


def test_aliases_typed_unavailable_events_and_candidate_boundary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "source")
    query = StreamingEvidenceWebsiteQuery(
        fixture["binding"], expected_binding_sha256=fixture["binding_sha256"]
    )
    bare = query.query_confidence(
        cancer_id="acc", lncrna_id="ENSG000001", pathway_id="HALLMARK:A"
    )
    prefixed = query.query_confidence(
        cancer_id="ACC", lncrna_id="LNC:ENSG000001.9", pathway_id="HALLMARK:A"
    )
    assert bare["rows"] == prefixed["rows"]
    available_row = bare["rows"][0]
    assert available_row["lncrna_id"] == "LNC:ENSG000001"
    assert available_row["availability"] is True
    assert "event_count" not in available_row
    assert available_row["model_consumed_event_count"] == EVENT_COUNT_CAP
    assert available_row["event_count_cap"] == EVENT_COUNT_CAP
    assert available_row["event_count_capped"] is True
    assert available_row["total_exact_event_count"] == 70

    unavailable = query.query_confidence(
        cancer_id="ACC",
        lncrna_id="LNCRNA:ENSG000001",
        pathway_id="HALLMARK:NO_EVENT",
    )
    row = unavailable["rows"][0]
    assert row["availability"] is False
    assert row["evidence_confidence_probability"] is None
    assert row["uncertainty"] is None
    assert row["direction"] is None
    assert row["failure_reason"] == NO_EVENT_REASON
    assert row["model_consumed_event_count"] == 0
    assert row["event_count_cap"] == EVENT_COUNT_CAP
    assert row["event_count_capped"] is False
    assert row["total_exact_event_count"] == 0
    assert query.query_events(
        cancer_id="ACC",
        lncrna_id="ENSG000001",
        pathway_id="HALLMARK:NO_EVENT",
    )["rows"] == []

    events = query.query_events(
        cancer_id="ACC", lncrna_id="LNC:ENSG000001", pathway_id="HALLMARK:A"
    )
    assert events["returned_rows"] == 70
    assert {row["source_database"] for row in events["rows"]} == {"RNAInter", "ENCORI"}
    assert {row["direction_target"] for row in events["rows"]} == {-1, 2}
    assert all(row["family_broadcast_used"] is False for row in events["rows"])

    capability = query.capability()
    assert capability["candidate_only"] is True
    assert capability["release_ready"] is False
    assert capability["publication_blocked_by_core_lineage"] is True
    assert query.download("evidence_predictions")["sha256"] == _sha(
        fixture["prediction"]
    )


def test_isolated_candidate_api_and_download_schema(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "source")
    app = create_evidence_candidate_app(
        fixture["binding"], binding_sha256=fixture["binding_sha256"]
    )
    client = TestClient(app)
    capability = client.get("/v3.2-candidate/evidence/capability")
    assert capability.status_code == 200
    assert capability.json()["publication_blocked_by_core_lineage"] is True
    response = client.get(
        "/v3.2-candidate/evidence/confidence",
        params={
            "cancer_id": "ACC",
            "lncrna_id": "LNC:ENSG000001",
            "pathway_id": "HALLMARK:NO_EVENT",
        },
    )
    assert response.status_code == 200
    response_row = response.json()["rows"][0]
    assert response_row["evidence_confidence_probability"] is None
    assert response_row["model_consumed_event_count"] == 0
    assert response_row["total_exact_event_count"] == 0
    events = client.get(
        "/v3.2-candidate/evidence/events",
        params={
            "cancer_id": "ACC",
            "lncrna_id": "ENSG000001",
            "pathway_id": "HALLMARK:A",
        },
    )
    assert events.status_code == 200
    assert events.json()["returned_rows"] == 70
    download = client.get(
        "/v3.2-candidate/evidence/download/evidence_predictions"
    )
    assert download.status_code == 200
    assert download.headers["x-artifact-sha256"] == _sha(fixture["prediction"])
    assert download.headers["x-candidate-only"] == "true"
    assert download.headers["x-release-ready"] == "false"
    assert client.get("/v3.2-candidate/evidence/download/not-allowed").status_code == 404
    openapi = client.get("/openapi.json").json()
    confidence_response = openapi["paths"]["/v3.2-candidate/evidence/confidence"][
        "get"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    response_name = confidence_response["$ref"].rsplit("/", 1)[-1]
    response_schema = openapi["components"]["schemas"][response_name]
    row_reference = response_schema["properties"]["rows"]["items"]["$ref"]
    row_schema = openapi["components"]["schemas"][row_reference.rsplit("/", 1)[-1]]
    row_fields = set(row_schema["properties"])
    assert {
        "model_consumed_event_count",
        "event_count_cap",
        "event_count_capped",
        "total_exact_event_count",
    } <= row_fields
    assert "event_count" not in row_fields


def test_binding_fails_closed_on_audit_or_artifact_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "source")
    with pytest.raises(FileExistsError):
        materialize_streaming_evidence_website_binding(
            prediction_manifest_path=tmp_path / "source" / "PREDICTION_MANIFEST.json",
            predictions_path=fixture["prediction"],
            training_manifest_path=tmp_path / "source" / "TRAINING_MANIFEST.json",
            stage_manifest_path=tmp_path / "source" / "STAGING_MANIFEST.json",
            candidate_events_path=fixture["events"],
            event_lineage_path=fixture["lineage"],
            independent_audit_path=fixture["audit"],
            expected_independent_audit_sha256=_sha(fixture["audit"]),
            output_root=fixture["output"],
            strict_formal=False,
        )

    frame = pd.read_parquet(fixture["prediction"])
    frame.loc[0, "evidence_confidence_probability"] = 0.1
    frame.to_parquet(fixture["prediction"], index=False)
    with pytest.raises(StreamingEvidenceAssetError, match="SHA drift"):
        StreamingEvidenceWebsiteQuery(
            fixture["binding"], expected_binding_sha256=fixture["binding_sha256"]
        )


def test_r2_audit_contract_is_hash_pinned_and_technical_failures_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "source")
    audit_path = Path(fixture["audit"])
    base_audit = json.loads(audit_path.read_text(encoding="utf-8"))

    with pytest.raises(StreamingEvidenceAssetError, match="independent audit SHA mismatch"):
        materialize_streaming_evidence_website_binding(
            prediction_manifest_path=fixture["prediction_manifest"],
            predictions_path=fixture["prediction"],
            training_manifest_path=fixture["training_manifest"],
            stage_manifest_path=fixture["stage_manifest"],
            candidate_events_path=fixture["events"],
            event_lineage_path=fixture["lineage"],
            independent_audit_path=audit_path,
            expected_independent_audit_sha256="0" * 64,
            output_root=tmp_path / "wrong-hash-binding",
            strict_formal=False,
        )

    failed_audit = dict(base_audit)
    failed_audit["technical_acceptance"] = False
    failed_audit["technical_prediction_pass"] = False
    failed_audit["technical_failures"] = ["prediction_partition_and_stage_alignment"]
    _write_json(audit_path, failed_audit)
    with pytest.raises(StreamingEvidenceAssetError, match="technical failures"):
        materialize_streaming_evidence_website_binding(
            prediction_manifest_path=fixture["prediction_manifest"],
            predictions_path=fixture["prediction"],
            training_manifest_path=fixture["training_manifest"],
            stage_manifest_path=fixture["stage_manifest"],
            candidate_events_path=fixture["events"],
            event_lineage_path=fixture["lineage"],
            independent_audit_path=audit_path,
            expected_independent_audit_sha256=_sha(audit_path),
            output_root=tmp_path / "technical-failure-binding",
            strict_formal=False,
        )

    missing_count_audit = dict(base_audit)
    missing_count_audit.pop("prediction_rows")
    _write_json(audit_path, missing_count_audit)
    with pytest.raises(StreamingEvidenceAssetError, match="canonical integer prediction_rows"):
        materialize_streaming_evidence_website_binding(
            prediction_manifest_path=fixture["prediction_manifest"],
            predictions_path=fixture["prediction"],
            training_manifest_path=fixture["training_manifest"],
            stage_manifest_path=fixture["stage_manifest"],
            candidate_events_path=fixture["events"],
            event_lineage_path=fixture["lineage"],
            independent_audit_path=audit_path,
            expected_independent_audit_sha256=_sha(audit_path),
            output_root=tmp_path / "missing-count-binding",
            strict_formal=False,
        )

    blocked_with_pass_label = dict(base_audit)
    blocked_with_pass_label["status"] = "PASS_INDEPENDENT_AUDIT_PUBLICATION_ALLOWED"
    _write_json(audit_path, blocked_with_pass_label)
    with pytest.raises(StreamingEvidenceAssetError, match="BLOCKED/publication verdict"):
        materialize_streaming_evidence_website_binding(
            prediction_manifest_path=fixture["prediction_manifest"],
            predictions_path=fixture["prediction"],
            training_manifest_path=fixture["training_manifest"],
            stage_manifest_path=fixture["stage_manifest"],
            candidate_events_path=fixture["events"],
            event_lineage_path=fixture["lineage"],
            independent_audit_path=audit_path,
            expected_independent_audit_sha256=_sha(audit_path),
            output_root=tmp_path / "laundered-status-binding",
            strict_formal=False,
        )


def test_audit_eligible_candidate_still_does_not_claim_release(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "source", publication_allowed=True)
    query = StreamingEvidenceWebsiteQuery(
        fixture["binding"], expected_binding_sha256=fixture["binding_sha256"]
    )
    capability = query.capability()
    assert capability["publication_allowed_by_independent_audit"] is True
    assert capability["publication_blocked_by_core_lineage"] is False
    assert capability["candidate_only"] is True
    assert capability["release_ready"] is False


def test_core_lineage_verdict_is_derived_from_audit_check_and_blocker(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "source")
    audit_path = Path(fixture["audit"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("core_lineage_verdict")
    audit["checks"]["fresh_current_g2_core_lineage"] = {
        "passed": False,
        "formal_lineage_present": False,
    }
    _write_json(audit_path, audit)

    result = materialize_streaming_evidence_website_binding(
        prediction_manifest_path=fixture["prediction_manifest"],
        predictions_path=fixture["prediction"],
        training_manifest_path=fixture["training_manifest"],
        stage_manifest_path=fixture["stage_manifest"],
        candidate_events_path=fixture["events"],
        event_lineage_path=fixture["lineage"],
        independent_audit_path=audit_path,
        expected_independent_audit_sha256=_sha(audit_path),
        output_root=tmp_path / "derived-core-verdict-binding",
        strict_formal=False,
    )
    binding = json.loads(Path(result["binding_path"]).read_text(encoding="utf-8"))
    assert binding["publication_blocked_by_core_lineage"] is True
    assert binding["core_lineage_verdict"] == "BLOCKED_FRESH_CURRENT_G2_CORE_LINEAGE"
