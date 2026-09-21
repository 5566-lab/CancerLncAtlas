from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from cc_hhgt.v32.single_cell_gap_audit import (
    BINDING_FORMAT,
    REPORT_FORMAT,
    SingleCellGapAuditError,
    audit_single_cell_gap_release,
    materialize_single_cell_gap_audit,
    sha256_file,
)
from scripts.materialize_v32_single_cell_gap_audit import default_inputs
from scripts.verify_v32_single_cell_gap_remote_readonly import (
    verify_remote_observation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_RELEASE = (
    REPO_ROOT / "artifacts" / "v32_single_cell_gap_audit_20260826_r2_codebound"
)
FORMAL_AUDIT = (
    REPO_ROOT
    / "artifacts"
    / "v32_single_cell_gap_audit_20260826_r2_codebound_independent_audit"
)


def test_materializes_typed_gaps_without_promoting_candidate(tmp_path: Path) -> None:
    release = tmp_path / "release"
    success = materialize_single_cell_gap_audit(
        inputs=default_inputs(REPO_ROOT), output_root=release
    )
    binding = json.loads(
        (release / "SINGLE_CELL_GAP_AUDIT_BINDING.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (release / "SINGLE_CELL_GAP_AUDIT.json").read_text(encoding="utf-8")
    )
    lncrna = json.loads(
        (release / "LNCRNA_DETECTION_33C.json").read_text(encoding="utf-8")
    )
    assert success["binding_sha256"] == sha256_file(
        release / "SINGLE_CELL_GAP_AUDIT_BINDING.json"
    )
    assert binding["format"] == BINDING_FORMAT
    assert binding["candidate_results_promoted"] is False
    assert report["format"] == REPORT_FORMAT
    assert report["pseudotime"]["numeric_rows"] == 0
    assert report["pseudotime"]["path"] is None
    assert (
        report["pseudotime"]["separate_read_candidate"]["dataset_id"]
        == "SC_GSE254249_READ"
    )
    assert (
        report["pseudotime"]["separate_read_candidate"][
            "same_cells_as_current_v32_read_authority"
        ]
        is False
    )
    assert report["ucell"]["covered_cancers"] == ["HNSC"]
    assert report["figures"]["current_v32_figure_files"] == 0
    assert len(lncrna["entries"]) == 33
    assert lncrna["counts"]["detected_union"] == 15_879
    assert lncrna["counts"]["detected_intersection"] == 4

    audit = audit_single_cell_gap_release(
        release_root=release,
        audit_root=tmp_path / "audit",
        expected_binding_sha256=success["binding_sha256"],
    )
    assert audit["status"] == "PASS_HASH_BOUND"
    assert audit["failed_checks"] == 0


def test_rejects_attempt_to_transfer_read_candidate_to_current_cells(
    tmp_path: Path,
) -> None:
    inputs = default_inputs(REPO_ROOT)
    observation = json.loads(
        inputs.remote_observation.read_text(encoding="utf-8")
    )
    observation["raw_metadata_header_audit"]["candidate_dataset"][
        "same_cells_as_current_v32_read_authority"
    ] = True
    tampered = tmp_path / "observation.json"
    tampered.write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SingleCellGapAuditError, match="cannot be transferred"):
        materialize_single_cell_gap_audit(
            inputs=replace(inputs, remote_observation=tampered),
            output_root=tmp_path / "release",
        )


def test_independent_audit_detects_materialized_file_drift(tmp_path: Path) -> None:
    release = tmp_path / "release"
    success = materialize_single_cell_gap_audit(
        inputs=default_inputs(REPO_ROOT), output_root=release
    )
    lncrna_path = release / "LNCRNA_DETECTION_33C.json"
    value = json.loads(lncrna_path.read_text(encoding="utf-8"))
    value["counts"]["detected_union"] += 1
    lncrna_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(SingleCellGapAuditError, match="Independent audit failed"):
        audit_single_cell_gap_release(
            release_root=release,
            audit_root=tmp_path / "audit",
            expected_binding_sha256=success["binding_sha256"],
        )


def test_remote_recheck_is_aggregate_only_and_reproduces_frozen_hashes() -> None:
    observation = json.loads(
        default_inputs(REPO_ROOT).remote_observation.read_text(encoding="utf-8")
    )
    metadata = observation["current_processed_metadata"]
    historical = observation["excluded_historical_results"]
    figures = observation["remote_figure_inventory"]
    candidate = observation["raw_metadata_header_audit"]["candidate_dataset"]

    def fake_run(command: str) -> str:
        if command.startswith("sha256sum -- ") and "cell_metadata.parquet" in command:
            return "\n".join(
                f"{row['sha256']}  {metadata['root']}/{row['cancer_id']}/cell_metadata.parquet"
                for row in metadata["files"]
            )
        if command == f"sha256sum -- {candidate['metadata_path']}":
            return f"{candidate['metadata_sha256']}  {candidate['metadata_path']}"
        if command == f"sha256sum -- {historical['trajectory_script']['path']}":
            return (
                f"{historical['trajectory_script']['sha256']}  "
                f"{historical['trajectory_script']['path']}"
            )
        if "sc_malignant_pseudotime.tsv.gz" in command and "wc -l" in command:
            return str(historical["pseudotime_files"])
        if "selected_pathway_ucell_aggregate.tsv.gz" in command and "wc -l" in command:
            return str(historical["ucell_aggregate_files"])
        if "sc_malignant_pseudotime.tsv.gz" in command and "sha256sum | sha256sum" in command:
            return historical["pseudotime_sha256_manifest_sha256"] + "  -"
        if "selected_pathway_ucell_aggregate.tsv.gz" in command and "sha256sum | sha256sum" in command:
            return historical["ucell_sha256_manifest_sha256"] + "  -"
        if "-iregex" in command and "wc -l" in command:
            return str(observation["current_v32_remote_inventory"]["matching_files"])
        for entry in (
            figures["historical_trajectory_figures"],
            figures["historical_web_umap_figures"],
        ):
            if command.startswith(f"find {entry['root']} "):
                if command.endswith("| wc -l"):
                    return str(entry["files"])
                if command.endswith("sha256sum | sha256sum"):
                    return entry["sha256_manifest_sha256"] + "  -"
        raise AssertionError(f"Unexpected live recheck command: {command}")

    result = verify_remote_observation(observation, fake_run)
    assert result["status"] == "PASS"
    assert result["failed_checks"] == 0
    assert result["remote_writes_performed"] is False
    assert result["patient_or_cell_records_returned"] is False


@pytest.mark.skipif(
    not FORMAL_RELEASE.is_dir() or not FORMAL_AUDIT.is_dir(),
    reason="formal single-cell gap audit is not materialized",
)
def test_formal_gap_audit_is_hash_bound() -> None:
    success = json.loads(
        (FORMAL_RELEASE / "SUCCESS.json").read_text(encoding="utf-8")
    )
    assert success["binding_sha256"] == sha256_file(
        FORMAL_RELEASE / "SINGLE_CELL_GAP_AUDIT_BINDING.json"
    )
    audit = json.loads(
        (FORMAL_AUDIT / "INDEPENDENT_AUDIT_BINDING.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["status"] == "PASS_HASH_BOUND"
    assert audit["failed_checks"] == 0
