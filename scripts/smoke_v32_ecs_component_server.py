#!/usr/bin/env python3
"""Independent runtime query smokes for Evidence, Clinical, and State heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

FORMAT = "CANCERLNCATLAS_V32_ECS_COMPONENT_RUNTIME_QUERY_SMOKE_V1"
RUNTIME_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned(path: Path, expected: str, label: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked {label}")
    observed = sha256_file(path)
    require(observed == expected.lower(), f"{label} SHA256 mismatch: {observed}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    require(not path.exists(), f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def runtime_identity() -> dict[str, Any]:
    """Record the exact isolated query runtime used by every component."""
    import cc_hhgt
    import dateutil
    import duckdb
    import numpy
    import pandas
    import pyarrow
    import pytz
    import scipy
    import six
    import tzdata

    return {
        "python": {"executable": sys.executable, "version": sys.version},
        "modules": {
            "cc_hhgt": {"path": str(Path(cc_hhgt.__file__).resolve())},
            "dateutil": {"version": dateutil.__version__, "path": str(Path(dateutil.__file__).resolve())},
            "duckdb": {"version": duckdb.__version__, "path": str(Path(duckdb.__file__).resolve())},
            "numpy": {"version": numpy.__version__, "path": str(Path(numpy.__file__).resolve())},
            "pandas": {"version": pandas.__version__, "path": str(Path(pandas.__file__).resolve())},
            "pyarrow": {"version": pyarrow.__version__, "path": str(Path(pyarrow.__file__).resolve())},
            "pytz": {"version": pytz.__version__, "path": str(Path(pytz.__file__).resolve())},
            "scipy": {"version": scipy.__version__, "path": str(Path(scipy.__file__).resolve())},
            "six": {"version": six.__version__, "path": str(Path(six.__file__).resolve())},
            "tzdata": {"version": tzdata.__version__, "path": str(Path(tzdata.__file__).resolve())},
        },
    }


def first_lncrna_with_availability(path: Path, availability: bool) -> str:
    """Select a real probe key instead of assuming a hard-coded lncRNA has both states."""
    import duckdb

    sql_path = "'" + str(path).replace("'", "''") + "'"
    connection = duckdb.connect(":memory:")
    try:
        row = connection.execute(
            f"SELECT lncrna_id FROM read_parquet({sql_path}) WHERE availability = ? LIMIT 1",
            [availability],
        ).fetchone()
    finally:
        connection.close()
    require(row is not None and bool(row[0]), f"No lncRNA probe exists for availability={availability}")
    return str(row[0])


def evidence_smoke(bindings: Mapping[str, Any]) -> dict[str, Any]:
    # Keep component smokes genuinely independent: an unavailable dependency or
    # broken import in Clinical/State must not prevent the Evidence receipt from
    # being created (and vice versa).
    from cc_hhgt.v32.evidence_direction_query import EvidenceDirectionProbabilityQuery
    from cc_hhgt.v32.evidence_query import EvidenceBindingQuery

    declaration = bindings["evidence"]
    query = EvidenceBindingQuery(
        declaration["binding_path"],
        expected_binding_sha256=declaration["binding_sha256"],
    )
    available_lnc = first_lncrna_with_availability(query.paths["evidence_predictions"], True)
    unavailable_lnc = first_lncrna_with_availability(query.paths["evidence_predictions"], False)
    available = query.query_confidence(
        lncrna_id=available_lnc, availability=True, limit=1
    )
    unavailable = query.query_confidence(
        lncrna_id=unavailable_lnc, availability=False, limit=1
    )
    require(available["returned_rows"] == 1, "Evidence available probe did not return one row")
    require(unavailable["returned_rows"] == 1, "Evidence unavailable probe did not return one row")
    available_row = available["rows"][0]
    unavailable_row = unavailable["rows"][0]
    require(available_row.get("availability") is True, "Evidence available row is not typed true")
    require(unavailable_row.get("availability") is False, "Evidence unavailable row is not typed false")
    require(available_row.get("evidence_confidence_probability") is not None, "Evidence available probability is null")
    require(unavailable_row.get("evidence_confidence_probability") is None, "Evidence unavailable probability is not null")
    require(bool(unavailable_row.get("failure_reason")), "Evidence unavailable reason is empty")

    direction_decl = bindings["evidence_direction"]
    direction = EvidenceDirectionProbabilityQuery(
        direction_decl["binding_path"],
        expected_binding_sha256=direction_decl["binding_sha256"],
        audit_binding_path=direction_decl["audit_binding_path"],
        expected_audit_binding_sha256=direction_decl["audit_binding_sha256"],
    )
    direction_available = direction.query(available=True, limit=1)
    direction_unavailable = direction.query(available=False, limit=1)
    require(direction_available["returned_rows"] == 1, "Direction available probe did not return one row")
    require(direction_unavailable["returned_rows"] == 1, "Direction unavailable probe did not return one row")
    direction_available_row = direction_available["rows"][0]
    direction_unavailable_row = direction_unavailable["rows"][0]
    require(direction_available_row.get("direction_probability_available") is True, "Direction available row is not typed true")
    require(direction_unavailable_row.get("direction_probability_available") is False, "Direction unavailable row is not typed false")
    for column in ("direction_negative_probability", "direction_neutral_probability", "direction_positive_probability"):
        require(direction_available_row.get(column) is not None, f"Direction available {column} is null")
        require(direction_unavailable_row.get(column) is None, f"Direction unavailable {column} is not null")
    require(bool(direction_unavailable_row.get("direction_probability_unavailable_reason")), "Direction unavailable reason is empty")
    return {
        "component": "evidence",
        "scientific_status": "partial_not_publishable",
        "binding": {"path": declaration["binding_path"], "sha256": query.binding_sha256},
        "direction_binding": {"path": direction_decl["binding_path"], "sha256": direction.binding_sha256},
        "direction_audit_binding": {"path": direction_decl["audit_binding_path"], "sha256": direction.audit_binding_sha256},
        "probes": {
            "confidence_available": {"returned_rows": 1, "row": available_row},
            "confidence_unavailable": {"returned_rows": 1, "row": unavailable_row},
            "direction_available": {"returned_rows": 1, "row": direction_available_row},
            "direction_unavailable": {"returned_rows": 1, "row": direction_unavailable_row},
        },
        "typed_available_verified": True,
        "typed_unavailable_verified": True,
    }


def clinical_smoke(bindings: Mapping[str, Any]) -> dict[str, Any]:
    from cc_hhgt.v32.clinical_km_query import ClinicalKMReleaseQuery

    declaration = bindings["clinical"]
    query = ClinicalKMReleaseQuery(
        declaration["binding_path"],
        expected_binding_sha256=declaration["binding_sha256"],
    )
    available_lnc = first_lncrna_with_availability(query.paths["statistics"], True)
    unavailable_lnc = first_lncrna_with_availability(query.paths["statistics"], False)
    available = query.query_lncrna_survival(
        lncrna_id=available_lnc, availability=True,
        include_curves=False, limit=1,
    )
    unavailable = query.query_lncrna_survival(
        lncrna_id=unavailable_lnc, availability=False,
        include_curves=False, limit=1,
    )
    require(available["returned_statistics_rows"] == 1, "Clinical available probe did not return one row")
    require(unavailable["returned_statistics_rows"] == 1, "Clinical unavailable probe did not return one row")
    available_row = available["statistics"][0]
    unavailable_row = unavailable["statistics"][0]
    require(available_row.get("availability") is True, "Clinical available row is not typed true")
    require(unavailable_row.get("availability") is False, "Clinical unavailable row is not typed false")
    require(not available_row.get("failure_reason"), "Clinical available row has failure reason")
    require(bool(unavailable_row.get("failure_reason")), "Clinical unavailable row lacks failure reason")
    return {
        "component": "clinical",
        "scientific_status": "secondary_fresh_lncrna_survival_statistics",
        "binding": {"path": declaration["binding_path"], "sha256": query.binding_sha256},
        "clean_directory_runtime_view_bound": bool(declaration.get("clean_directory_runtime_view_bound")),
        "probes": {
            "available": {"returned_statistics_rows": 1, "returned_curve_rows": 0, "row": available_row},
            "unavailable": {"returned_statistics_rows": 1, "returned_curve_rows": 0, "row": unavailable_row},
        },
        "typed_available_verified": True,
        "typed_unavailable_verified": True,
    }


def state_smoke(bindings: Mapping[str, Any]) -> dict[str, Any]:
    from cc_hhgt.v32.state_gene_set_query import StateGeneSetReleaseQuery

    declaration = bindings["state_gene_set"]
    query = StateGeneSetReleaseQuery(
        declaration["binding_path"],
        expected_binding_sha256=declaration["binding_sha256"],
    )
    rnass = query.query_gene_sets(state_id="stemness_rna::RNAss", limit=1)
    dnass = query.query_gene_sets(state_id="stemness_dna::DNAss", limit=1)
    require(rnass["returned_rows"] == 1, "RNAss probe did not return one gene set")
    require(dnass["returned_rows"] == 1, "DNAss probe did not return one gene set")
    gene_set_id = rnass["rows"][0]["gene_set_id"]
    members = query.query_members(gene_set_id=gene_set_id, limit=1)
    require(members["returned_rows"] == 1 and members["total_rows"] >= 1, "State member probe did not return a member")
    return {
        "component": "state_gene_set",
        "scientific_status": "secondary_state_gene_set_and_report",
        "binding": {"path": declaration["binding_path"], "sha256": query.binding_sha256},
        "probes": {
            "RNAss": {"total_rows": rnass["total_rows"], "returned_rows": 1, "row": rnass["rows"][0]},
            "DNAss": {"total_rows": dnass["total_rows"], "returned_rows": 1, "row": dnass["rows"][0]},
            "member": {"gene_set_id": gene_set_id, "total_rows": members["total_rows"], "returned_rows": 1, "row": members["rows"][0]},
        },
        "typed_available_unavailable": "NOT_APPLICABLE_NO_AVAILABILITY_FIELD_IN_STATE_GENE_SET_QUERY_CONTRACT",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("evidence", "clinical", "state_gene_set"), required=True)
    parser.add_argument("--runtime-bindings", type=Path, required=True)
    parser.add_argument("--runtime-bindings-sha256", required=True)
    parser.add_argument("--quarantine-receipt", type=Path, required=True)
    parser.add_argument("--quarantine-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.absolute().exists(), f"Refusing to overwrite output: {args.output}")
    base = {
        "format": FORMAT,
        "component": args.component,
        "status": "FAIL",
        "runtime_bindings": {"path": str(args.runtime_bindings.absolute()), "sha256": args.runtime_bindings_sha256.lower()},
        "quarantine_receipt": {"path": str(args.quarantine_receipt.absolute()), "sha256": args.quarantine_receipt_sha256.lower()},
        "main_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }
    try:
        runtime = load_pinned(args.runtime_bindings, args.runtime_bindings_sha256, "runtime bindings")
        require(runtime.get("format") == RUNTIME_FORMAT, "Runtime binding format drifted")
        require(runtime.get("production_deployed") is False, "Runtime production flag drifted")
        require(runtime.get("release_ready") is False, "Runtime release flag drifted")
        quarantine = load_pinned(args.quarantine_receipt, args.quarantine_receipt_sha256, "quarantine receipt")
        require(quarantine.get("status") == "PASS" and quarantine.get("validated_files") == 59 and quarantine.get("moved_files") == 59, "Quarantine receipt is not complete PASS")
        runner = {"evidence": evidence_smoke, "clinical": clinical_smoke, "state_gene_set": state_smoke}[args.component]
        base["result"] = runner(runtime["bindings"])
        base["runtime_identity"] = runtime_identity()
        base["status"] = "PASS"
    except Exception as error:
        base["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        write_exclusive(args.output, base)
        raise
    write_exclusive(args.output, base)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
