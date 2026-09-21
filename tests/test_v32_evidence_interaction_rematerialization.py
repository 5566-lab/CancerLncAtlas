from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.evidence_interaction_rematerialization import (
    RAW_SOURCES,
    load_gene_authority,
    rematerialize_evidence_interactions,
    resolve_gene,
)
from cc_hhgt.v32.evidence_training import (
    _iter_parquet_frames,
    build_exact_event_bags,
    read_input_table,
)
from scripts.validate_v32_evidence_interaction_rematerialization import (
    SOURCES as VALIDATOR_SOURCES,
    validate as independently_validate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    lnc = root / "processed/dimensions/dim_lncRNA.tsv"
    gene = root / "processed/dimensions/dim_gene.tsv"
    _write_tsv(
        lnc,
        ["lncrna_id", "ensembl_gene_id", "gene_symbol", "gene_name", "aliases"],
        [
            ["LNC:ENSG00000111111", "ENSG00000111111.3", "LINC_A", "Linc A", "OLD_A"],
            ["LNC:ENSG00000111112", "ENSG00000111112", "LINC_B", "Linc B", "DUP_LNC"],
            ["LNC:ENSG00000111113", "ENSG00000111113", "LINC_C", "Linc C", "DUP_LNC"],
        ],
    )
    _write_tsv(
        gene,
        ["gene_id", "ensembl_gene_id", "gene_symbol", "gene_name", "aliases"],
        [
            ["GENE:ENSG00000222221", "ENSG00000222221", "EWSR1", "EWS RNA binding protein 1", "DUP_GENE"],
            ["GENE:ENSG00000222222", "ENSG00000222222", "TP53", "tumor protein p53", "DUP_GENE"],
        ],
    )
    npinter = root / RAW_SOURCES[0].relative_path
    _write_tsv(
        npinter,
        [
            "ncRI_ID", "RNA_name", "RNA_NONCODE_ID", "RNA_type", "partner_name",
            "partner_ID", "partner_type", "description", "methods", "PMIDs", "species",
            "tissue_cell", "interaction_class", "interaction_type", "category", "source_db",
            "throughput_flag",
        ],
        [
            ["DUP-1", "LINC_A", "", "lncRNA", "EWSR1", "Q01844", "protein", "", "CLIP", "24813895", "Homo sapiens", "HeLa", "binding", "binding", "RNA-Protein", "test", "high"],
            ["DUP-1", "LINC_A", "", "lncRNA", "DUP_GENE", "", "protein", "", "qPCR", "", "Homo sapiens", "", "association", "association", "RNA-Protein", "test", "low"],
        ],
    )
    rnainter = root / RAW_SOURCES[1].relative_path
    _write_tsv(
        rnainter,
        [
            "RNAInterID", "Interactor1.Symbol", "Category1", "Species1",
            "Interactor2.Symbol", "Category2", "Species2", "Raw_ID1", "Raw_ID2",
            "score", "strong", "weak", "predict",
        ],
        [
            ["RP1", "not-a-symbol", "lncRNA", "Homo sapiens", "TP53", "protein", "Homo sapiens", "ENSG00000111111.9", "NCBI:7157", "1", "qRT-PCR", "", ""],
            ["RP2", "LINC_A", "unknown", "Homo sapiens", "EWSR1", "protein", "Homo sapiens", "ENSG00000111111", "", "1", "IP", "", ""],
        ],
    )
    audit = tmp_path / "RECOVERY_AUDIT.json"
    audit.write_text(
        json.dumps(
            {
                "status": "PASS_RAW_TOKENS_RETAINED_FAIL_CLOSED_RECOVERY_QUANTIFIED",
                "inputs": [
                    {"path": str(path), "sha256": _sha256(path)}
                    for path in (lnc, npinter, rnainter)
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, audit


def test_gene_mapping_is_fail_closed_for_ambiguous_alias(tmp_path: Path) -> None:
    root, _ = _project(tmp_path)
    authority = load_gene_authority(root / "processed/dimensions/dim_gene.tsv")
    exact = resolve_gene(authority, raw_name="TP53")
    ambiguous = resolve_gene(authority, raw_name="DUP_GENE")
    assert exact.identifier == "GENE:ENSG00000222222"
    assert exact.route == "unique_name_or_alias"
    assert ambiguous.identifier == ""
    assert ambiguous.route == "ambiguous_name_or_alias"
    assert ambiguous.candidate_count == 2


def test_rematerialization_recovers_raw_id_and_keeps_duplicate_records_distinct(
    tmp_path: Path,
) -> None:
    root, audit = _project(tmp_path)
    output = tmp_path / "rematerialized"
    report = rematerialize_evidence_interactions(
        project_root=root,
        output_root=output,
        recovery_audit_path=audit,
        raw_sources=RAW_SOURCES[:2],
        progress_every=0,
    )
    assert report["status"] == "PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION"
    relation_path = output / "interaction_relation_recovered.tsv.gz"
    event_path = output / "evidence_event_empty_by_contract.tsv"
    assert relation_path.is_file() and event_path.is_file()
    relation = read_input_table(relation_path)
    assert len(relation) == 4
    assert relation.interaction_id.nunique() == 4
    assert relation.source_row_id.nunique() == 4
    assert relation.source_row_sha256.nunique() == 4
    duplicates = relation.loc[relation.original_record_id.eq("DUP-1")]
    assert len(duplicates) == 2
    assert duplicates.source_record_id.nunique() == 1
    assert duplicates.historical_interaction_id.nunique() == 1
    assert duplicates.interaction_id.nunique() == 2
    recovered = relation.loc[relation.original_record_id.eq("RP1")].iloc[0]
    assert recovered.lncrna_id == "LNC:ENSG00000111111"
    assert recovered.lncrna_mapping_route == "authoritative_source_ensembl"
    inferred = relation.loc[relation.original_record_id.eq("RP2")].iloc[0]
    assert inferred.orientation_status == "inferred_unique_fail_closed_lncrna_mapping"
    ambiguous_partner = duplicates.loc[duplicates.partner_raw.eq("DUP_GENE")].iloc[0]
    assert pd.isna(ambiguous_partner.partner_id) or ambiguous_partner.partner_id == ""
    assert ambiguous_partner.mapping_status == "lncrna_mapped_partner_ambiguous"
    empty_event = pd.read_csv(event_path, sep="\t")
    assert empty_event.empty
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["outputs"].values():
        assert Path(item["path"]).is_file()
        assert ".rematerialized.tmp" not in item["path"]
    success = json.loads((output / "SUCCESS.json").read_text(encoding="utf-8"))
    assert Path(success["manifest"]["path"]) == output / "MANIFEST.json"


def test_header_only_event_authority_does_not_double_count_interaction_rows(
    tmp_path: Path,
) -> None:
    root, audit = _project(tmp_path)
    output = tmp_path / "rematerialized"
    rematerialize_evidence_interactions(
        project_root=root,
        output_root=output,
        recovery_audit_path=audit,
        raw_sources=RAW_SOURCES[:2],
        progress_every=0,
    )
    relation = read_input_table(output / "interaction_relation_recovered.tsv.gz")
    evidence = read_input_table(output / "evidence_event_empty_by_contract.tsv")
    members = pd.DataFrame(
        {
            "pathway_id": ["PATHWAY:P1", "PATHWAY:P1"],
            "member_id": ["GENE:ENSG00000222221", "GENE:ENSG00000222222"],
            "member_type": ["protein", "protein"],
            "static_member_id": ["MEMBER:1", "MEMBER:2"],
        }
    )
    built = build_exact_event_bags(evidence, relation, members)
    # Three rows have both a canonical lncRNA and a uniquely mapped partner;
    # the ambiguous partner row is rejected rather than broadcast.
    assert len(built.events) == 3
    assert int(built.events.source_occurrence_count.sum()) == 3
    assert set(built.lineage.source_kind) == {"interaction_relation"}


def test_independent_validator_replays_every_selected_raw_row(tmp_path: Path) -> None:
    root, audit = _project(tmp_path)
    rematerialized = tmp_path / "rematerialized"
    rematerialize_evidence_interactions(
        project_root=root,
        output_root=rematerialized,
        recovery_audit_path=audit,
        raw_sources=RAW_SOURCES[:2],
        progress_every=0,
    )
    validation_root = tmp_path / "independent_validation"
    report = independently_validate(
        project_root=root,
        rematerialization_root=rematerialized,
        output_root=validation_root,
        progress_every=0,
        sources=VALIDATOR_SOURCES[:2],
    )
    assert report["status"] == "PASS_INDEPENDENT_REMATERIALIZATION_VALIDATION"
    assert report["counts"]["rows"] == 4
    assert report["contract"]["production_module_imported"] is False
    success = json.loads((validation_root / "SUCCESS.json").read_text(encoding="utf-8"))
    assert success["rows"] == 4


def test_bounded_memory_formal_scanner_accepts_recovered_tsv_gz(tmp_path: Path) -> None:
    path = tmp_path / "interaction_relation_recovered.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("interaction_id", "lncrna_id", "ignored"))
        writer.writerows(
            (
                ("INT32:S:1", "LNC:ENSG00000111111", "x"),
                ("INT32:S:2", "LNC:ENSG00000111112", "y"),
                ("INT32:S:3", "LNC:ENSG00000111113", "z"),
            )
        )
    frames = list(
        _iter_parquet_frames(
            path,
            columns=("interaction_id", "lncrna_id"),
            batch_size=2,
        )
    )
    assert [len(frame) for frame in frames] == [2, 1]
    assert [list(frame.columns) for frame in frames] == [
        ["interaction_id", "lncrna_id"],
        ["interaction_id", "lncrna_id"],
    ]
    assert frames[1].iloc[0].interaction_id == "INT32:S:3"
