from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.release_registry import artifact_sha256
from scripts.audit_v32_mixed_query_key_coverage import audit
import scripts.build_v32_mixed_query_staging_assets as builder


VERSION = "CancerLncAtlas_V3.2.builder_test"
RUN_ID = "V32_BUILDER_TEST"


def _association() -> pd.DataFrame:
    rows = []
    for lnc_id, probabilities in {
        "LNC:ENSG000001": [0.8, 0.2],
        "LNC:ENSG000002": [0.7, None],
    }.items():
        for pathway_id, probability in zip(
            ["PATHWAY:P1", "PATHWAY:P2"], probabilities, strict=True
        ):
            available = probability is not None
            rows.append(
                {
                    "cancer_id": "LUAD",
                    "lncrna_id": lnc_id,
                    "pathway_id": pathway_id,
                    "association_membership_probability": probability,
                    "analysis_version": VERSION,
                    "training_run_id": RUN_ID,
                    "pathway_target_level": "exact_pathway",
                    "availability": available,
                    "eligible_for_mixed_query": True,
                    "availability_reason": (
                        None if available else "NOT_EVALUATED_MODEL_FOLD_GAP"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_audit(source: Path, receipt: Path, scratch: Path) -> dict:
    return audit(
        Namespace(
            exact_association=source,
            output=receipt,
            scratch=scratch,
            threads=1,
            memory_limit="1GB",
        )
    )


def test_duckdb_receipt_proves_complete_keys_and_typed_nulls(tmp_path: Path) -> None:
    source = tmp_path / "association.parquet"
    _association().to_parquet(source, index=False)
    receipt_path = tmp_path / "KEY_COVERAGE.json"

    receipt = _run_audit(source, receipt_path, tmp_path / "duckdb_scratch")

    assert receipt["status"] == "PASS"
    assert receipt["eligible_pair_count"] == 2
    assert receipt["pathway_count"] == 2
    assert receipt["observed_key_count"] == receipt["expected_key_count"] == 4
    assert receipt["not_evaluated_key_count"] == 1
    assert receipt["missing_not_evaluated_reason_count"] == 0
    assert receipt["independent_of_asset_builder"] is True


def test_duckdb_receipt_writes_forensic_failure_for_missing_key(tmp_path: Path) -> None:
    source = tmp_path / "association.parquet"
    _association().iloc[:-1].to_parquet(source, index=False)
    receipt_path = tmp_path / "KEY_COVERAGE_FAIL.json"

    with pytest.raises(ValueError, match="Key coverage failed"):
        _run_audit(source, receipt_path, tmp_path / "duckdb_scratch")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "FAIL"
    assert receipt["complete_key_coverage"] is False
    assert receipt["missing_key_count"] == 1


def test_builder_streams_association_completes_lnc_ids_and_filters_ora_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "association.parquet"
    _association().to_parquet(source, index=False)
    receipt_path = tmp_path / "KEY_COVERAGE.json"
    _run_audit(source, receipt_path, tmp_path / "duckdb_scratch")
    lineage_path = tmp_path / "LINEAGE.json"
    lineage_path.write_text(
        json.dumps(
            {
                "analysis_version": VERSION,
                "training_run_id": RUN_ID,
                "prediction_sha256": artifact_sha256(source),
            }
        ),
        encoding="utf-8",
    )
    membership_path = tmp_path / "membership.parquet"
    pd.DataFrame(
        [
            {"pathway_id": "PATHWAY:P1", "gene_id": "ENSGP000001"},
            {"pathway_id": "PATHWAY:P1", "gene_id": "ENSG000001"},
            {"pathway_id": "PATHWAY:P2", "gene_id": "ENSGP000002"},
            {"pathway_id": "PATHWAY:P2", "gene_id": "ENSG000002"},
        ]
    ).to_parquet(membership_path, index=False)
    gene_map_path = tmp_path / "gene_map.parquet"
    pd.DataFrame(
        [
            {
                "gene_id": "ENSGP000001",
                "ensembl_gene_id": "ENSGP000001",
                "gene_symbol": "PROT1",
                "aliases": "P1",
                "entity_type": "protein_coding_gene",
            },
            {
                "gene_id": "ENSGP000002",
                "ensembl_gene_id": "ENSGP000002",
                "gene_symbol": "PROT2",
                "aliases": "P2",
                "entity_type": "protein_coding_gene",
            },
        ]
    ).to_parquet(gene_map_path, index=False)
    lnc_map_path = tmp_path / "lnc_map.parquet"
    pd.DataFrame(
        [
            {
                "lncrna_id": "LNC:ENSG000001",
                "ensembl_gene_id": "ENSG000001",
                "gene_symbol": "LINC1",
                "aliases": "L1",
                "entity_type": "lncRNA",
            }
        ]
    ).to_parquet(lnc_map_path, index=False)
    output = tmp_path / "built_assets"
    monkeypatch.setattr(builder, "validate_module_lineage", lambda *args: None)

    result = builder.build_assets(
        Namespace(
            exact_association=source,
            exact_lineage=lineage_path,
            key_coverage_receipt=receipt_path,
            exact_pathway_membership=membership_path,
            gene_identifier_map=gene_map_path,
            lncrna_identifier_map=lnc_map_path,
            batch_size=2,
            output_root=output,
        )
    )

    identifiers = pd.read_parquet(output / "identifier_map.parquet")
    lnc = identifiers.loc[identifiers.entity_type.eq("lncRNA")]
    assert set(lnc.canonical_id) == {"LNC:ENSG000001", "LNC:ENSG000002"}
    fallback = lnc.loc[lnc.canonical_id.eq("LNC:ENSG000002")].iloc[0]
    assert fallback.annotation_status == "CANONICAL_V32_PREDICTION_ID_FALLBACK"
    membership = pd.read_parquet(output / "exact_pathway_membership.parquet")
    assert set(membership.gene_id) == {"ENSGP000001", "ENSGP000002"}
    association = pd.read_parquet(output / "v32_lnc_exact_association.parquet")
    unavailable = association.loc[~association.availability]
    assert len(unavailable) == 1
    assert unavailable.association_membership_probability.isna().all()
    assert set(unavailable.availability_reason) == {"NOT_EVALUATED_MODEL_FOLD_GAP"}
    assert result["identifier_completion_audit"]["canonical_fallback_count"] == 1
    assert result["identifier_completion_audit"]["silent_drop_count"] == 0
    assert result["protein_membership_audit"]["excluded_nonprotein_or_unregistered_rows"] == 2
    assert result["association_streaming_audit"]["bounded_memory_streaming"] is True
    assert result["key_coverage"]["complete_key_coverage"] is True
