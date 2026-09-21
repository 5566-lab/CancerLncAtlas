from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.input_lineage import artifact_sha256
from scripts.materialize_v32_single_cell_r7_portable_supersession import (
    DERIVED_ASSET_FAMILIES,
    EXACT_MEMBERSHIP_SHA256,
    FORMAL_17,
    PortableSupersessionError,
    _assert_server_path,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "artifacts"
    / "single_cell_cell_level_r7_portable_supersession_20260829"
)


def _strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)
    elif isinstance(value, str):
        yield value


def test_authorized_server_path_gate_rejects_public0() -> None:
    assert _assert_server_path(
        "./data/CancerLncAtlas/processed/x", "raw"
    ).startswith("${DATA_ROOT}/")
    assert _assert_server_path(
        "./data/CancerLncAtlas/runtime/x", "portable"
    ).startswith("${PRIVATE_WORK_ROOT}/")
    for invalid in (
        "./data/CancerLncAtlas/results/x",
        "/tmp/x",
        r"D:\model\x",
    ):
        try:
            _assert_server_path(invalid, "invalid")
        except PortableSupersessionError:
            pass
        else:  # pragma: no cover - explicit fail message is more useful than parametrization
            raise AssertionError(f"Unauthorized path was admitted: {invalid}")


def test_materialized_r7_is_portable_and_fail_closed() -> None:
    run = json.loads((ARTIFACT / "RUN_STATUS.json").read_text(encoding="utf-8"))
    handoff = json.loads(
        (ARTIFACT / "TRAINING_HANDOFF.json").read_text(encoding="utf-8")
    )
    authority = json.loads(
        (ARTIFACT / "PORTABLE_AUTHORITY.json").read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (ARTIFACT / "RAW_INPUT_READABILITY_PREFLIGHT.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(run["per_cancer"]) == 33
    assert set(run["formal_eligible_cancers"]) == set(FORMAL_17)
    assert run["formal_eligible_count"] == 17
    assert run["derived_assets_bound"] is False
    assert preflight["formal_cancer_count"] == 17
    assert preflight["all_formal_h5_and_metadata_readable_nonempty"] is True
    assert preflight["heavy_recompute_started"] is False
    assert authority["public0_consumable_references"] == 0

    assert set(handoff["assets"]) == set(DERIVED_ASSET_FAMILIES)
    assert all(asset["path"] is None for asset in handoff["assets"].values())
    assert all(
        asset["status"] == "UNBOUND_FRESH_RECOMPUTE_REQUIRED"
        for asset in handoff["assets"].values()
    )
    assert handoff["historical_assets_relabelled_fresh"] is False
    assert handoff["private_head_training_ready_for_available_partitions"] is False

    for path in ARTIFACT.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not [
            value
            for value in _strings(payload)
            if value.replace("\\", "/").startswith("${DATA_ROOT}/")
        ]

    membership_path = ARTIFACT / "authority" / "exact_pathway_membership_2135.parquet"
    membership = pd.read_parquet(membership_path)
    assert artifact_sha256(membership_path) == EXACT_MEMBERSHIP_SHA256
    assert len(membership) == 352_205
    assert membership.pathway_id.nunique() == 2_135

    annotation_path = ARTIFACT / "authority" / "gencode_v50_gene_annotation.parquet"
    annotation = pd.read_parquet(annotation_path)
    assert len(annotation) == 54_960
    assert annotation.gene_id.is_unique
    assert int(annotation.gene_class.eq("lncRNA").sum()) == 34_866
    assert int(annotation.gene_class.eq("protein_coding").sum()) == 20_094

    checksums = pd.read_csv(ARTIFACT / "CONTROL_SHA256.tsv", sep="\t")
    assert len(checksums) == 11
    for row in checksums.itertuples(index=False):
        assert artifact_sha256(ARTIFACT / row.artifact) == row.sha256
