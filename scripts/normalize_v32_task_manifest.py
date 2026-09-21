#!/usr/bin/env python3
"""Validate a V3.2 task manifest and emit Bash-safe unit-separated rows.

Tabs are shell IFS whitespace.  Consequently ``read`` collapses the empty
``blocked_reason`` field in an authorized manifest and shifts ``paid_task``
into the wrong variable.  This preflight parses the authoritative TSV with
``csv``, validates the complete runnable-task contract, and only then writes a
headerless file separated by ASCII Unit Separator (0x1f), which Bash preserves
when adjacent fields are empty.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence


TASK_MANIFEST_COLUMNS = (
    "task_id",
    "run_id",
    "task_type",
    "model",
    "patient_fold",
    "seed",
    "owner",
    "hardware_class",
    "status",
    "blocked_reason",
    "paid_task",
)
OUTPUT_SEPARATOR = "\x1f"


class ManifestPreflightError(RuntimeError):
    """A task manifest is not safe to hand to a paid launcher."""


def _parse_expected_folds(value: str) -> tuple[int, ...]:
    try:
        folds = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected folds must be comma-separated integers") from exc
    if not folds or len(set(folds)) != len(folds) or any(fold < 0 for fold in folds):
        raise argparse.ArgumentTypeError("expected folds must be unique non-negative integers")
    return folds


def validate_task_manifest(
    path: str | Path,
    *,
    run_id: str,
    owner: str,
    hardware_class: str,
    task_type: str = "CC_HHGT_PATIENT_FOLD",
    model: str = "CC-HHGT",
    status: str = "PENDING",
    blocked_reason: str = "",
    paid_task: str = "true",
    expected_folds: Sequence[int] = tuple(range(5)),
) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ManifestPreflightError(f"TASK_MANIFEST_MISSING={source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        if tuple(reader.fieldnames or ()) != TASK_MANIFEST_COLUMNS:
            raise ManifestPreflightError(
                "TASK_MANIFEST_HEADER_DRIFT=" + repr(reader.fieldnames)
            )
        try:
            rows = list(reader)
        except csv.Error as exc:
            raise ManifestPreflightError(f"TASK_MANIFEST_TSV_INVALID={exc}") from exc

    expected_fold_tuple = tuple(int(item) for item in expected_folds)
    if len(rows) != len(expected_fold_tuple):
        raise ManifestPreflightError(
            f"TASK_MANIFEST_ROW_COUNT_DRIFT={len(rows)}!={len(expected_fold_tuple)}"
        )
    expected = {
        "run_id": str(run_id),
        "task_type": str(task_type),
        "model": str(model),
        "owner": str(owner),
        "hardware_class": str(hardware_class),
        "status": str(status),
        "blocked_reason": str(blocked_reason),
        "paid_task": str(paid_task).lower(),
    }
    observed_folds: list[int] = []
    task_ids: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        if None in row or set(row) != set(TASK_MANIFEST_COLUMNS):
            raise ManifestPreflightError(
                f"TASK_MANIFEST_COLUMN_COUNT_DRIFT=line_{row_number}"
            )
        if any(OUTPUT_SEPARATOR in str(value) for value in row.values()):
            raise ManifestPreflightError(
                f"TASK_MANIFEST_CONTROL_SEPARATOR=line_{row_number}"
            )
        for key, required in expected.items():
            if row[key] != required:
                raise ManifestPreflightError(
                    f"TASK_MANIFEST_FIELD_DRIFT=line_{row_number}:{key}:"
                    f"{row[key]!r}!={required!r}"
                )
        try:
            fold = int(row["patient_fold"])
            seed = int(row["seed"])
        except ValueError as exc:
            raise ManifestPreflightError(
                f"TASK_MANIFEST_INTEGER_INVALID=line_{row_number}"
            ) from exc
        if str(fold) != row["patient_fold"] or seed < 0:
            raise ManifestPreflightError(
                f"TASK_MANIFEST_INTEGER_NONCANONICAL=line_{row_number}"
            )
        if not row["task_id"]:
            raise ManifestPreflightError(f"TASK_MANIFEST_TASK_ID_EMPTY=line_{row_number}")
        expected_task_id = f"{run_id}|PATIENT_FOLD_{fold}|CC-HHGT|{seed}"
        if row["task_id"] != expected_task_id:
            raise ManifestPreflightError(
                f"TASK_MANIFEST_TASK_ID_DRIFT=line_{row_number}:"
                f"{row['task_id']!r}!={expected_task_id!r}"
            )
        observed_folds.append(fold)
        task_ids.append(row["task_id"])

    if tuple(sorted(observed_folds)) != tuple(sorted(expected_fold_tuple)):
        raise ManifestPreflightError(
            f"TASK_MANIFEST_FOLD_DRIFT={sorted(observed_folds)!r}"
        )
    if len(set(task_ids)) != len(task_ids):
        raise ManifestPreflightError("TASK_MANIFEST_TASK_ID_DUPLICATE")
    return rows


def write_normalized_rows(
    rows: Sequence[dict[str, str]],
    output: str | Path,
    *,
    empty_blocked_marker: str | None = None,
) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            values = []
            for column in TASK_MANIFEST_COLUMNS:
                value = row[column]
                if column == "blocked_reason" and value == "" and empty_blocked_marker:
                    value = empty_blocked_marker
                values.append(value)
            handle.write(OUTPUT_SEPARATOR.join(values) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--task-type", default="CC_HHGT_PATIENT_FOLD")
    parser.add_argument("--model", default="CC-HHGT")
    parser.add_argument("--status", default="PENDING")
    parser.add_argument("--blocked-reason", default="")
    parser.add_argument("--paid-task", choices=("true", "false"), required=True)
    parser.add_argument(
        "--empty-blocked-marker",
        help="Encode an empty blocked_reason for shells that collapse adjacent delimiters.",
    )
    parser.add_argument("--expected-folds", type=_parse_expected_folds, default=tuple(range(5)))
    args = parser.parse_args(argv)
    rows = validate_task_manifest(
        args.manifest,
        run_id=args.run_id,
        owner=args.owner,
        hardware_class=args.hardware_class,
        task_type=args.task_type,
        model=args.model,
        status=args.status,
        blocked_reason=args.blocked_reason,
        paid_task=args.paid_task,
        expected_folds=args.expected_folds,
    )
    write_normalized_rows(rows, args.output, empty_blocked_marker=args.empty_blocked_marker)
    print(f"PASS_TASK_MANIFEST_PREFLIGHT={args.manifest} ROWS={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
