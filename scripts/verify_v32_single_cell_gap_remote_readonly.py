"""Live read-only recheck of the frozen V3.2 single-cell remote observation.

Only file hashes, file counts, and aggregate manifest hashes are returned over
SSH.  No patient/cell records are printed and no remote file is created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Callable


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CANCER_RE = re.compile(r"^[A-Z0-9]+$")
SCOPE = "./data/CancerLncAtlas"


class RemoteReadOnlyVerificationError(RuntimeError):
    """Raised when the live remote inventory differs from the frozen audit."""


def _ssh_runner(binary: str, host: str) -> Callable[[str], str]:
    def run(command: str) -> str:
        completed = subprocess.run(
            [
                binary,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ConnectionAttempts=1",
                host,
                command,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if completed.returncode != 0:
            raise RemoteReadOnlyVerificationError(
                f"read-only SSH command failed ({completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    return run


def _single_sha(output: str) -> str:
    token = output.split()[0] if output.split() else ""
    if not SHA_RE.fullmatch(token):
        raise RemoteReadOnlyVerificationError(f"Invalid SHA output: {output!r}")
    return token


def _integer(output: str) -> int:
    try:
        return int(output.strip())
    except ValueError as exc:
        raise RemoteReadOnlyVerificationError(
            f"Invalid integer output: {output!r}"
        ) from exc


def verify_remote_observation(
    observation: dict[str, Any], run: Callable[[str], str]
) -> dict[str, Any]:
    if observation.get("remote_scope") != SCOPE:
        raise RemoteReadOnlyVerificationError("Observation is outside authorised scope")
    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, expected: Any) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": expected,
                "passed": observed == expected,
            }
        )

    metadata = observation["current_processed_metadata"]
    metadata_root = str(metadata["root"])
    if not metadata_root.startswith(SCOPE + "/"):
        raise RemoteReadOnlyVerificationError("Metadata root escaped authorised scope")
    paths: list[str] = []
    expected_hashes: dict[str, str] = {}
    for row in metadata["files"]:
        cancer = str(row["cancer_id"])
        if not CANCER_RE.fullmatch(cancer):
            raise RemoteReadOnlyVerificationError(f"Unsafe cancer ID: {cancer}")
        path = f"{metadata_root}/{cancer}/cell_metadata.parquet"
        paths.append(path)
        expected_hashes[path] = str(row["sha256"])
    command = "sha256sum -- " + " ".join(shlex.quote(path) for path in paths)
    observed_lines = run(command).splitlines()
    observed_hashes: dict[str, str] = {}
    for line in observed_lines:
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not SHA_RE.fullmatch(fields[0]):
            raise RemoteReadOnlyVerificationError(f"Invalid metadata SHA line: {line!r}")
        observed_hashes[fields[1].lstrip(" *")] = fields[0]
    check("current_metadata_file_count", len(observed_hashes), 23)
    for path in paths:
        check(
            f"metadata_sha256.{Path(path).parent.name}",
            observed_hashes.get(path),
            expected_hashes[path],
        )

    candidate = observation["raw_metadata_header_audit"]["candidate_dataset"]
    candidate_path = str(candidate["metadata_path"])
    if not candidate_path.startswith(SCOPE + "/"):
        raise RemoteReadOnlyVerificationError("Candidate path escaped authorised scope")
    check(
        "read_candidate_metadata_sha256",
        _single_sha(run(f"sha256sum -- {shlex.quote(candidate_path)}")),
        candidate["metadata_sha256"],
    )

    historical = observation["excluded_historical_results"]
    script = historical["trajectory_script"]
    check(
        "historical_trajectory_script_sha256",
        _single_sha(run(f"sha256sum -- {shlex.quote(str(script['path']))}")),
        script["sha256"],
    )
    trajectory_root = str(historical["root"])
    if not trajectory_root.startswith(SCOPE + "/"):
        raise RemoteReadOnlyVerificationError("Trajectory root escaped authorised scope")
    pseudo_count_command = (
        f"find {shlex.quote(trajectory_root)} -mindepth 2 -maxdepth 2 -type f "
        "-name sc_malignant_pseudotime.tsv.gz | wc -l"
    )
    ucell_count_command = (
        f"find {shlex.quote(trajectory_root)} -mindepth 2 -maxdepth 2 -type f "
        "-name selected_pathway_ucell_aggregate.tsv.gz | wc -l"
    )
    check(
        "historical_pseudotime_file_count",
        _integer(run(pseudo_count_command)),
        int(historical["pseudotime_files"]),
    )
    check(
        "historical_ucell_file_count",
        _integer(run(ucell_count_command)),
        int(historical["ucell_aggregate_files"]),
    )
    pseudo_manifest_command = (
        f"find {shlex.quote(trajectory_root)} -mindepth 2 -maxdepth 2 -type f "
        "-name sc_malignant_pseudotime.tsv.gz -print0 | sort -z | "
        "xargs -0 sha256sum | sha256sum"
    )
    ucell_manifest_command = (
        f"find {shlex.quote(trajectory_root)} -mindepth 2 -maxdepth 2 -type f "
        "-name selected_pathway_ucell_aggregate.tsv.gz -print0 | sort -z | "
        "xargs -0 sha256sum | sha256sum"
    )
    check(
        "historical_pseudotime_manifest_sha256",
        _single_sha(run(pseudo_manifest_command)),
        historical["pseudotime_sha256_manifest_sha256"],
    )
    check(
        "historical_ucell_manifest_sha256",
        _single_sha(run(ucell_manifest_command)),
        historical["ucell_sha256_manifest_sha256"],
    )

    current = observation["current_v32_remote_inventory"]
    current_root = str(current["root"])
    if not current_root.startswith(SCOPE + "/"):
        raise RemoteReadOnlyVerificationError("Current V3.2 root escaped scope")
    current_count_command = (
        f"find {shlex.quote(current_root)} -xdev -type f -regextype posix-extended "
        "-iregex '.*(ucell|pseudotime|trajectory|figure|umap|pathway.?score).*' | wc -l"
    )
    check(
        "current_v32_related_file_count",
        _integer(run(current_count_command)),
        int(current["matching_files"]),
    )

    figures = observation["remote_figure_inventory"]
    for name, entry, find_args in (
        (
            "historical_trajectory_figures",
            figures["historical_trajectory_figures"],
            "-mindepth 3 -path '*/figures/*' -type f",
        ),
        (
            "historical_web_umap_figures",
            figures["historical_web_umap_figures"],
            "-mindepth 2 -maxdepth 2 -type f",
        ),
    ):
        root = str(entry["root"])
        if not root.startswith(SCOPE + "/"):
            raise RemoteReadOnlyVerificationError("Figure root escaped scope")
        check(
            f"{name}.count",
            _integer(run(f"find {shlex.quote(root)} {find_args} | wc -l")),
            int(entry["files"]),
        )
        manifest_command = (
            f"find {shlex.quote(root)} {find_args} -print0 | sort -z | "
            "xargs -0 sha256sum | sha256sum"
        )
        check(
            f"{name}.manifest_sha256",
            _single_sha(run(manifest_command)),
            entry["sha256_manifest_sha256"],
        )

    failed = [item for item in checks if not item["passed"]]
    result = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_GAP_REMOTE_LIVE_RECHECK_V1",
        "status": "PASS" if not failed else "FAIL",
        "checks": len(checks),
        "failed_checks": len(failed),
        "check_results": checks,
        "remote_scope": SCOPE,
        "observation_content_sha256": hashlib.sha256(
            json.dumps(
                observation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "remote_writes_performed": False,
        "patient_or_cell_records_returned": False,
    }
    if failed:
        raise RemoteReadOnlyVerificationError(
            "Remote observation drift: "
            + ", ".join(item["name"] for item in failed)
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--ssh-host", default="COMPUTE_HOST")
    parser.add_argument("--ssh-binary", default="ssh.exe")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    observation = json.loads(args.observation.read_text(encoding="utf-8"))
    result = verify_remote_observation(
        observation, _ssh_runner(args.ssh_binary, args.ssh_host)
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
