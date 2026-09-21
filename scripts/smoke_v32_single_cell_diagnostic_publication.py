#!/usr/bin/env python3
"""Exercise real V3.2 single-cell diagnostic query/download assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.single_cell_diagnostic_publication import (
    SingleCellDiagnosticInputError,
    SingleCellDiagnosticPublicationQuery,
)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query = SingleCellDiagnosticPublicationQuery(
        args.binding, expected_binding_sha256=args.binding_sha256
    )
    pathway = query.query_trajectory(cancer_id="HNSC", level="PATHWAY", limit=1)
    cell = query.query_trajectory(cancer_id="HNSC", level="CELL", limit=1)
    figures = query.figure_manifest(cancer_id="HNSC")
    figure_id = figures["rows"][0]["figure_id"]
    figure = query.resolve_figure(cancer_id="HNSC", figure_id=figure_id)
    if figure is None:
        raise RuntimeError("Declared real figure did not resolve")
    downloads = query.download_manifest(cancer_id="HNSC")
    pseudotime_download = query.resolve_download(
        cancer_id="HNSC", relative_path="pseudotime_cells.tsv.gz"
    )
    traversal_rejected = False
    try:
        query.resolve_download(cancer_id="HNSC", relative_path="../secret")
    except SingleCellDiagnosticInputError:
        traversal_rejected = True

    passed = bool(
        pathway["availability"] is True
        and pathway["returned_rows"] == 1
        and pathway["pseudotime_numeric_values"] > 0
        and pathway["root_is_explicit"] is False
        and pathway["primary_score_weight"] == 0
        and cell["returned_rows"] == 1
        and cell["rows"][0].get("pseudotime") is not None
        and figures["returned_rows"] == 9
        and artifact_sha256(figure["path"]) == figure["sha256"]
        and downloads["file_count"] == 13
        and downloads["request_time_payload_rehash"] is True
        and artifact_sha256(pseudotime_download["path"])
        == pseudotime_download["sha256"]
        and traversal_rejected
    )
    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_REAL_QUERY_SMOKE_V1",
        "status": "PASS" if passed else "FAIL",
        "binding_path": str(Path(args.binding).resolve()),
        "binding_sha256": args.binding_sha256.lower(),
        "cancer_id": "HNSC",
        "real_numeric_pathway_query_exercised": pathway["returned_rows"] == 1,
        "real_numeric_cell_query_exercised": cell["returned_rows"] == 1,
        "pseudotime_numeric_values": pathway["pseudotime_numeric_values"],
        "real_figure_manifest_exercised": figures["returned_rows"] == 9,
        "real_figure_file_rehash_exercised": True,
        "real_download_manifest_exercised": downloads["file_count"] == 13,
        "request_time_payload_rehash_exercised": True,
        "path_traversal_rejected": traversal_rejected,
        "root_provenance": pathway["root_provenance"],
        "root_is_explicit": False,
        "diagnostic_only": True,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "changes_primary_ranking": False,
        "historical_outputs_used": False,
        "production_deployed": False,
    }
    _atomic_json(Path(args.output).resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
