from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.single_cell_diagnostic_publication import (
    EXPECTED_FIGURES,
    FORMAL_CANCERS,
    SingleCellDiagnosticAssetError,
    SingleCellDiagnosticInputError,
    SingleCellDiagnosticPublicationQuery,
    materialize_publication_binding,
)
from website.backend.v32_staging_api import create_staging_app


def _write_tsv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    artifacts = {}
    for cancer in FORMAL_CANCERS:
        stage = tmp_path / "compute" / cancer
        (stage / "figures").mkdir(parents=True)
        activity = pd.DataFrame(
            {"pathway_id": ["P:1"], "activity_score": [0.3], "cancer_id": [cancer]}
        )
        pathway = pd.DataFrame(
            {
                "pathway_id": ["P:1"],
                "cancer_id": [cancer],
                "rho": [0.5],
                "p_value": [0.01],
            }
        )
        lncrna = pd.DataFrame(
            {
                "pathway_id": ["P:1"],
                "cancer_id": [cancer],
                "lncrna_id": ["LNC:1"],
                "rho": [0.4],
            }
        )
        pseudo = pd.DataFrame(
            {
                "cell_id": [f"{cancer}:1"],
                "cancer_id": [cancer],
                "pseudotime": [1.25],
                "pseudotime_0_1": [0.25],
            }
        )
        role_frames = {
            "activity": activity,
            "pathway_stats": pathway,
            "lncrna_pathway": lncrna,
        }
        declared = {}
        for role, frame in role_frames.items():
            path = stage / f"v32_exact2135_{role}.tsv.gz"
            _write_tsv(path, frame)
            declared[role] = {
                "path": str(path.resolve()),
                "sha256": artifact_sha256(path),
                "rows": 1,
                "exact_pathway_filtered": True,
            }
        _write_tsv(stage / "sc_malignant_pseudotime.tsv.gz", pseudo)
        pd.DataFrame(
            [
                {
                    "cancer_id": cancer,
                    "exact_unweighted_edges": 13153,
                    "signed_intersection_edges": 13153,
                    "excluded_signed_reference_edges": 70,
                    "missing_exact_signed_edges": 0,
                    "status": "PASS",
                }
            ]
        ).to_csv(stage / "dorothea_signed_intersection_qa.tsv", sep="\t", index=False)
        for name in EXPECTED_FIGURES:
            (stage / "figures" / name).write_bytes(b"%PDF-1.4\n% test\n")
        artifacts[cancer] = declared

    source = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_ARTIFACT_V1",
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "DIAGNOSTIC_COMPLETE_NOT_MODEL_FUSED",
        "cancers": list(FORMAL_CANCERS),
        "artifacts": artifacts,
        "root_provenance": "INFERRED_CYTOTRACE2_UCELL_CONSENSUS",
        "root_is_explicit": False,
        "exact_pathway_count": 2135,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "historical_sc_trajectory_outputs_used": False,
        "diagnostic_only": True,
        "model_fusion_permitted": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "production_deployed": False,
    }
    source_path = tmp_path / "V32_DIAGNOSTIC_MANIFEST.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    result = materialize_publication_binding(
        diagnostic_manifest_path=source_path,
        expected_diagnostic_manifest_sha256=artifact_sha256(source_path),
        output_root=tmp_path / "publication",
    )
    return Path(result["binding_path"]), result["binding_sha256"]


def test_materialize_query_figures_and_downloads(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    query = SingleCellDiagnosticPublicationQuery(
        binding, expected_binding_sha256=digest
    )
    trajectory = query.query_trajectory(
        cancer_id="hnsc", level="PATHWAY", pathway_id="P:1", limit=2
    )
    assert trajectory["availability"] is True
    assert trajectory["returned_rows"] == 1
    assert trajectory["pseudotime_numeric_values"] == 1
    assert trajectory["primary_score_weight"] == 0
    assert trajectory["root_is_explicit"] is False
    figures = query.figure_manifest(cancer_id="HNSC")
    assert figures["availability"] is True
    assert figures["returned_rows"] == 9
    assert query.resolve_figure(
        cancer_id="HNSC", figure_id="monocle3_pseudotime"
    )["sha256"]
    manifest = query.download_manifest(cancer_id="HNSC")
    assert manifest["file_count"] == 13
    assert manifest["request_time_payload_rehash"] is True


def test_fail_closed_on_traversal_and_payload_drift(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    query = SingleCellDiagnosticPublicationQuery(
        binding, expected_binding_sha256=digest
    )
    with pytest.raises(SingleCellDiagnosticInputError):
        query.resolve_download(cancer_id="HNSC", relative_path="../secret")
    target = Path(
        query.records["HNSC"]["artifacts"]["pathway_stats"]["path"]
    )
    target.write_bytes(target.read_bytes() + b"drift")
    with pytest.raises(SingleCellDiagnosticAssetError):
        query.query_trajectory(cancer_id="HNSC", level="PATHWAY")


def test_staging_compatibility_routes_serve_numeric_and_figures(tmp_path: Path) -> None:
    binding, digest = _fixture(tmp_path)
    root = Path(__file__).resolve().parents[1]
    app = create_staging_app(
        root
        / "artifacts/v32_staging/core_registry_refresh_20260826_r1/RELEASE_REGISTRY.json",
        single_cell_diagnostic_binding_path=binding,
        single_cell_diagnostic_binding_sha256=digest,
    )
    with TestClient(app) as client:
        trajectory = client.get(
            "/v3.2-staging/single-cell/trajectory/HNSC",
            params={"level": "PATHWAY", "limit": 1},
        )
        assert trajectory.status_code == 200
        assert trajectory.json()["pseudotime_numeric_values"] == 1
        figures = client.get("/v3.2-staging/single-cell/figures/HNSC")
        assert figures.status_code == 200
        assert figures.json()["returned_rows"] == 9
        figure = client.get(
            "/v3.2-staging/single-cell/figure/HNSC/monocle3_pseudotime"
        )
        assert figure.status_code == 200
        assert figure.headers["content-type"] == "application/pdf"
        assert len(figure.headers["x-artifact-sha256"]) == 64
        traversal = client.get(
            "/v3.2-staging/single-cell/diagnostic/HNSC/download/%2E%2E/secret"
        )
        assert traversal.status_code in {404, 422}
