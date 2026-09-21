from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.evidence_identifier_recovery import (
    EvidenceIdentifierRecoveryError,
    SourceSpec,
    audit_raw_identifier_recovery,
    fail_closed_resolution,
    historical_resolution,
    load_lncrna_authority,
    stable_source_record_id,
)


def _dimension(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {
                "lncrna_id": "LNC:1",
                "ensembl_gene_id": "ENSG00000111111",
                "gene_symbol": "LINC-A",
                "gene_name": "long A",
                "aliases": "ALIAS-A;SHARED",
            },
            {
                "lncrna_id": "LNC:2",
                "ensembl_gene_id": "ENSG00000222222",
                "gene_symbol": "LINC-B",
                "gene_name": "long B",
                "aliases": "ALIAS-B;SHARED",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep="\t", index=False, lineterminator="\n")
    return path


def test_fail_closed_resolution_uses_source_ensembl_and_rejects_ambiguity(tmp_path: Path) -> None:
    authority = load_lncrna_authority(_dimension(tmp_path / "dim.tsv"))
    assert historical_resolution(authority, raw_name="missing").identifier == ""
    recovered = fail_closed_resolution(
        authority,
        raw_name="renamed source token",
        source_ensembl="Ensembl:ENSG00000111111.7",
    )
    assert recovered.identifier == "LNC:1"
    assert recovered.route == "authoritative_source_ensembl"
    ambiguous = fail_closed_resolution(authority, raw_name="SHARED")
    assert ambiguous.identifier == ""
    assert ambiguous.route == "ambiguous_name_or_alias"
    tokenized = fail_closed_resolution(authority, raw_name="unknown|ALIAS-B")
    assert tokenized.identifier == "LNC:2"


def test_source_record_hash_exact_historical_vector() -> None:
    assert stable_source_record_id("RNAInter", "RP33978966") == (
        "SRC:07b82d0cac6a78157af83c3b146d03403cde521a"
    )


def test_audit_quantifies_recovery_without_reusing_output(tmp_path: Path) -> None:
    root = tmp_path / "project"
    _dimension(root / "processed/dimensions/dim_lncRNA.tsv")
    raw = root / "input/toy.tsv"
    raw.parent.mkdir(parents=True)
    with raw.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["record", "name", "ensembl", "pmid"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {"record": "R1", "name": "renamed", "ensembl": "ENSG00000111111", "pmid": "12345678"}
        )
        writer.writerow({"record": "R2", "name": "SHARED", "ensembl": "", "pmid": "12345679"})
    spec = SourceSpec(
        "ToyDB",
        "input/toy.tsv",
        "record",
        ("name",),
        ensembl_columns=("ensembl",),
        pmid_columns=("pmid",),
        historical_ensembl_used=False,
    )
    output = tmp_path / "audit"
    report = audit_raw_identifier_recovery(
        project_root=root,
        output_root=output,
        source_specs=(spec,),
        hash_inputs=True,
    )
    assert report["counts"]["rows"] == 2
    assert report["counts"]["newly_recoverable"] == 1
    assert report["counts"]["still_unresolved_or_ambiguous"] == 1
    assert report["interpretation"]["raw_data_missing"] is False
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "SUCCESS_AUDIT_ONLY_NOT_REBUILT"
    with pytest.raises(FileExistsError):
        audit_raw_identifier_recovery(
            project_root=root,
            output_root=output,
            source_specs=(spec,),
        )


def test_duplicate_source_record_ids_are_typed_for_collision_safe_rebuild(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    _dimension(root / "processed/dimensions/dim_lncRNA.tsv")
    raw = root / "input/toy.tsv"
    raw.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"record": "R1", "name": "LINC-A"},
            {"record": "R1", "name": "LINC-B"},
        ]
    ).to_csv(raw, sep="\t", index=False, lineterminator="\n")
    spec = SourceSpec("ToyDB", "input/toy.tsv", "record", ("name",))
    report = audit_raw_identifier_recovery(
        project_root=root,
        output_root=tmp_path / "audit",
        source_specs=(spec,),
        hash_inputs=False,
    )
    assert report["counts"]["duplicate_source_record_id"] == 1
    assert report["counts"]["unique_source_record_ids"] == 1
    assert report["interpretation"][
        "duplicate_source_record_ids_require_collision_safe_rematerialization"
    ] is True
