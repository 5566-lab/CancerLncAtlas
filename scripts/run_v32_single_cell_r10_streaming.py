#!/usr/bin/env python3
"""R10 control plane for the frozen R9 single-cell streaming algorithm.

R9 remains byte-for-byte frozen.  R10 reuses its already-equivalence-tested
numeric implementation, but replaces only the association spill location with
an explicit, private local-ext4 directory.  Durable artifacts are still staged
and atomically published on ``${PRIVATE_WORK_ROOT}``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.single_cell_association_scratch import (  # noqa: E402
    AssociationScratchContractError,
    association_work_path,
    canonical_contract_sha256,
    create_association_scratch_contract,
    estimate_association_scratch_bytes,
    preflight_association_scratch_root,
    remove_empty_association_scratch_root,
    validate_association_scratch_contract,
)


R9_RUNNER = ROOT / "scripts" / "run_v32_single_cell_r7_streaming.py"
R10_RUN_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_STREAMING_RUN_V1"
R10_PLAN_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R10_STREAMING_PLAN_V1"
R10_POST_BH_HANDOFF_FORMAT = (
    "CC_HHGT_V3_2_SINGLE_CELL_R10_POST_BH_EXEC_HANDOFF_V1"
)
R10_SOURCE_GENERATION = "V3.2_R10_FRESH_FROM_RAW_H5_R9_NUMERIC_KERNEL"
DELL2_DURABLE_ROOT = Path("./data/CancerLncAtlas")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REAL_EXECVE = os.execve


class R10StreamingRunError(RuntimeError):
    """Raised when R10 cannot prove a control-plane invariant."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise R10StreamingRunError(f"cannot hash absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise R10StreamingRunError(f"JSON input is absent/unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise R10StreamingRunError(f"JSON input is not an object: {path}")
    return payload


def _atomic_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.r10.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise R10StreamingRunError(f"unsafe R10 JSON temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_r9():
    spec = importlib.util.spec_from_file_location("v32_r10_frozen_r9_kernel", R9_RUNNER)
    if spec is None or spec.loader is None:
        raise R10StreamingRunError(f"cannot load frozen R9 runner: {R9_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _require_dell2_durable_output(path: Path) -> None:
    """Formal Linux publication is restricted to the authorized dell_2 tree."""

    if os.name != "posix":
        return
    if path.is_symlink():
        raise R10StreamingRunError("durable output path is a symlink")
    resolved = path.resolve()
    durable = DELL2_DURABLE_ROOT.resolve(strict=True)
    if resolved != durable and not _is_within(resolved, durable):
        raise R10StreamingRunError(
            f"R10 durable output must remain below {durable}: {resolved}"
        )


def _promote_bundle_to_r10(bundle: dict[str, Any], r9) -> dict[str, Any]:
    """Bind the R10 controller without changing the frozen numeric kernel."""

    controller_sha = _sha256(Path(__file__).resolve())
    kernel_sha = _sha256(R9_RUNNER)
    code_shas = dict(bundle["code_shas"])
    code_shas.pop("runner_sha256", None)
    code_shas.update(
        {
            "r10_control_runner_sha256": controller_sha,
            "r9_numeric_kernel_sha256": kernel_sha,
        }
    )
    contract = dict(bundle["contract"])
    contract.update(
        {
            "format": R10_RUN_FORMAT,
            "source_generation": R10_SOURCE_GENERATION,
            "code_shas": code_shas,
            "numeric_kernel_generation": "R9_PROCESS_ISOLATED_BH_EQUIVALENCE_FROZEN",
            "association_scratch_policy": (
                "EXPLICIT_EXCLUSIVE_0700_LOCAL_EXT4_POST_EXEC_REVALIDATED"
            ),
            "association_scratch_minimum_free_bytes": 8 * 1024**3,
            "association_scratch_estimate_multiplier": 16,
            "association_scratch_path_in_scientific_contract": False,
            "association_scratch_identity_bound_in_exec_handoff_and_lineage": True,
            "server_tmp_fallback_permitted": False,
            "general_output_allowlist_extended_for_scratch": False,
            "shared_expression_source_for_lnc_and_pathway": True,
            "association_is_exploratory_not_causal": True,
            "independent_replication_required_for_causal_claim": True,
            "pseudotime_computed": False,
            "pseudotime_root_status": "NOT_APPLICABLE_NO_PSEUDOTIME_IN_R10",
            "diagnostic_inferred_pseudotime_root_consumed": False,
            "cell_level_ucell_matrix_persisted": False,
            "ucell_published_unit": "DONOR_CELLTYPE_MEAN",
            "reference_catalog_2547_is_formal_pathway_universe": False,
            "formal_exact_pathway_count": 2_135,
            "processed_seurat_rds_fallback_permitted": False,
            "source_reuse_decision_uses_mtime_or_nonempty_only": False,
        }
    )
    promoted = dict(bundle)
    promoted["contract"] = contract
    promoted["contract_sha256"] = r9._canonical_sha(contract)
    promoted["code_shas"] = code_shas
    return promoted


def build_parser(r9) -> argparse.ArgumentParser:
    parser = r9.build_parser()
    parser.description = (
        "R10 fresh single-cell streaming with explicit local-ext4 association scratch"
    )
    parser.add_argument(
        "--association-scratch-root",
        type=Path,
        help=(
            "Exact per-run directory to create exclusively on local ext4; "
            "shared /tmp, pre-existing paths and output-tree overlap are rejected"
        ),
    )
    return parser


def _validate_public_args(args: argparse.Namespace, r9) -> None:
    r9._validate_args(args)
    if args.mode == "run" and args.association_scratch_root is None:
        raise AssociationScratchContractError(
            "ASSOCIATION_SCRATCH_ARGUMENT_REQUIRED",
            "formal R10 run requires --association-scratch-root",
            phase="ARGUMENT_VALIDATION",
        )


def _rewrite_handoff_and_exec(
    *,
    r9,
    scratch_contract: Mapping[str, Any],
    executable: str,
    command: list[str],
    environment: Mapping[str, str],
) -> None:
    """Replace the frozen R9 internal target with the R10 validated target."""

    if len(command) != 7 or command[2] != "--internal-post-bh-r9":
        raise R10StreamingRunError(f"unexpected R9 exec command: {command}")
    if Path(command[1]).resolve() != R9_RUNNER.resolve():
        raise R10StreamingRunError("R9 exec command runner path drift")
    if command[3] != "--handoff-json" or command[5] != "--expected-handoff-sha256":
        raise R10StreamingRunError("R9 exec command argument order drift")
    handoff_path = Path(command[4]).resolve()
    observed_sha = _sha256(handoff_path)
    if observed_sha != command[6]:
        raise R10StreamingRunError("R9 handoff changed before R10 promotion")
    handoff = _load_json(handoff_path)
    if handoff.get("format") != r9.POST_BH_HANDOFF_FORMAT:
        raise R10StreamingRunError("R9 handoff format drift before promotion")
    if int(handoff.get("producer_pid", -1)) != os.getpid():
        raise R10StreamingRunError("R9 handoff producer PID drift before promotion")

    scratch_work = association_work_path(scratch_contract, phase="PRE_EXEC")
    scratch_sha = canonical_contract_sha256(scratch_contract)
    controller = Path(__file__).resolve()
    semantic_fields = {
        "association_scratch_contract_sha256": scratch_sha,
        "association_scratch_root": str(scratch_contract["root"]),
        "association_scratch_filesystem": "ext4",
        "association_scratch_exact_mode": "0700",
        "association_scratch_uid": int(scratch_contract["uid"]),
        "association_scratch_gid": int(scratch_contract["gid"]),
        "association_scratch_st_dev": int(scratch_contract["st_dev"]),
        "association_scratch_st_ino": int(scratch_contract["st_ino"]),
        "association_scratch_required_free_bytes": int(
            scratch_contract["required_free_bytes"]
        ),
        "association_scratch_pre_exec_revalidated": True,
        "server_tmp_fallback_used": False,
        "general_output_allowlist_extended_for_scratch": False,
        "durable_publication_remains_on_dell_2": True,
        "shared_expression_source_for_lnc_and_pathway": True,
        "association_is_exploratory_not_causal": True,
        "pseudotime_computed": False,
        "pseudotime_root_status": "NOT_APPLICABLE_NO_PSEUDOTIME_IN_R10",
        "cell_level_ucell_matrix_persisted": False,
        "ucell_published_unit": "DONOR_CELLTYPE_MEAN",
        "reference_catalog_2547_is_formal_pathway_universe": False,
        "formal_exact_pathway_count": 2_135,
        "processed_seurat_rds_fallback_used": False,
        "mtime_or_nonempty_only_reuse_used": False,
    }
    lineage_base = dict(handoff["lineage_base"])
    lineage_base.update(semantic_fields)
    success_base = dict(handoff["success_base"])
    success_base.update(semantic_fields)
    handoff.update(
        {
            "format": R10_POST_BH_HANDOFF_FORMAT,
            "analysis_version": r9.ANALYSIS_VERSION,
            "source_generation": R10_SOURCE_GENERATION,
            "scratch": str(scratch_work),
            "association_scratch_contract": dict(scratch_contract),
            "association_scratch_contract_sha256": scratch_sha,
            "lineage_base": lineage_base,
            "success_base": success_base,
            "source_runner_path": str(controller),
            "source_runner_sha256": _sha256(controller),
            "r9_numeric_kernel_path": str(R9_RUNNER.resolve()),
            "r9_numeric_kernel_sha256": _sha256(R9_RUNNER),
            "formal_transition": (
                "OS_EXECVE_REPLACES_EXPRESSION_PROCESS_IMAGE_AND_REVALIDATES_"
                "EXTERNAL_EXT4_SCRATCH"
            ),
            "no_parent_helper_overlap": True,
        }
    )
    _atomic_json_replace(handoff_path, handoff)
    promoted_sha = _sha256(handoff_path)
    promoted_command = [
        executable,
        str(controller),
        "--internal-post-bh-r10",
        "--handoff-json",
        str(handoff_path),
        "--expected-handoff-sha256",
        promoted_sha,
    ]
    _REAL_EXECVE(executable, promoted_command, dict(environment))
    raise R10StreamingRunError("os.execve unexpectedly returned")


def _materialize_with_r10_scratch(
    bundle: dict[str, Any],
    accumulator: Any,
    work: Path,
    args: argparse.Namespace,
    r9,
) -> dict[str, Any]:
    estimate = estimate_association_scratch_bytes(bundle["resource_estimate"])
    output_parent = args.output_parent.resolve()
    scratch_contract = create_association_scratch_contract(
        args.association_scratch_root,
        estimate_bytes=estimate,
        forbidden_output_roots=(output_parent,),
    )

    def intercepted_execve(
        executable: str,
        command: list[str],
        environment: Mapping[str, str],
    ) -> None:
        _rewrite_handoff_and_exec(
            r9=r9,
            scratch_contract=scratch_contract,
            executable=executable,
            command=command,
            environment=environment,
        )

    # r9 imports the process-global os module.  Retain and restore the real
    # function if materialization fails before the expected exec boundary.
    os.execve = intercepted_execve  # type: ignore[assignment]
    try:
        return r9._materialize_outputs(bundle, accumulator, work, args)
    finally:
        os.execve = _REAL_EXECVE  # type: ignore[assignment]


def _post_bh_exec_publish_r10(handoff: dict[str, Any], r9) -> dict[str, Any]:
    """Revalidate local scratch, run frozen BH, then atomically publish dell_2."""

    if handoff.get("format") != R10_POST_BH_HANDOFF_FORMAT:
        raise R10StreamingRunError("post-BH R10 handoff format drift")
    if handoff.get("association_engine") != r9.ASSOCIATION_ENGINE:
        raise R10StreamingRunError("post-BH association engine drift")
    cancer = str(handoff.get("cancer_id", ""))
    lineage_base = handoff.get("lineage_base")
    success_base = handoff.get("success_base")
    if not isinstance(lineage_base, dict) or not isinstance(success_base, dict):
        raise R10StreamingRunError("post-BH lineage/success base is not an object")
    if (
        lineage_base.get("cancer_id") != cancer
        or success_base.get("cancer_id") != cancer
    ):
        raise R10StreamingRunError("post-BH cancer ID differs across payloads")
    contract_sha = str(lineage_base.get("contract_sha256", ""))
    if not _SHA256.fullmatch(contract_sha):
        raise R10StreamingRunError("post-BH scientific contract SHA-256 is invalid")
    producer_pid = int(handoff.get("producer_pid", -1))
    declared_publish = Path(handoff["publish"])
    declared_final = Path(handoff["final"])
    declared_raw = Path(handoff["raw_path"])
    declared_output = Path(handoff["evidence_path"])
    declared_scratch = Path(handoff["scratch"])
    if any(
        path.is_symlink()
        for path in (
            declared_publish,
            declared_final,
            declared_raw,
            declared_output,
            declared_scratch,
        )
    ):
        raise R10StreamingRunError("post-BH declared path is a symlink")
    publish = declared_publish.resolve()
    final = declared_final.resolve()
    raw_path = declared_raw.resolve()
    output_path = declared_output.resolve()
    for path in (publish, final, raw_path, output_path):
        r9._authorized_output(path)
    _require_dell2_durable_output(final)
    expected_publish_name = (
        f".cancer_id={cancer}.{contract_sha[:16]}.publish.{producer_pid}"
    )
    if (
        publish.parent != final.parent
        or final.name != f"cancer_id={cancer}"
        or publish.name != expected_publish_name
    ):
        raise R10StreamingRunError("post-BH publish/final topology or name drift")
    if (
        raw_path != publish / ".association_evidence_raw.parquet"
        or output_path != publish / "association_evidence.parquet"
    ):
        raise R10StreamingRunError("post-BH raw/evidence fixed path drift")

    scratch_contract = handoff.get("association_scratch_contract")
    if not isinstance(scratch_contract, dict):
        raise R10StreamingRunError("post-BH scratch contract is not an object")
    if (
        canonical_contract_sha256(scratch_contract)
        != handoff.get("association_scratch_contract_sha256")
    ):
        raise R10StreamingRunError("post-BH scratch contract SHA-256 drift")
    scratch = association_work_path(scratch_contract, phase="POST_EXEC")
    if declared_scratch.resolve() != scratch:
        raise R10StreamingRunError("post-BH external scratch work path drift")
    scratch_root = validate_association_scratch_contract(
        scratch_contract, phase="POST_EXEC", require_empty=True
    )
    if (
        scratch_root == publish
        or _is_within(scratch_root, publish)
        or _is_within(publish, scratch_root)
        or scratch_root == final
        or _is_within(scratch_root, final)
        or _is_within(final, scratch_root)
    ):
        raise R10StreamingRunError("post-BH scratch overlaps durable publication")

    relative_paths = tuple(map(str, handoff.get("relative_paths", [])))
    if relative_paths != r9.POST_BH_RELATIVE_PATHS:
        raise R10StreamingRunError("post-BH fixed release file set/order drift")
    for relative in relative_paths:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or len(relative_path.parts) != 1
        ):
            raise R10StreamingRunError("post-BH release path is not a safe basename")
        staged = publish / relative
        if staged != output_path and (staged.is_symlink() or not staged.is_file()):
            raise R10StreamingRunError(
                f"post-BH staged release file absent/unsafe: {relative}"
            )
    if publish.is_symlink() or not publish.is_dir():
        raise R10StreamingRunError("post-BH publish staging is absent/unsafe")
    if final.exists() or final.is_symlink():
        raise R10StreamingRunError("post-BH immutable final already exists")
    if output_path.exists() or output_path.is_symlink():
        raise R10StreamingRunError("post-BH evidence output unexpectedly exists")

    # On interruption or failure, this function never reaches atomic publish.
    # The private scratch inode is deliberately retained for audit; it is never
    # recursively deleted and never interpreted as a completed result.
    evidence_rows = r9._finalize_association_evidence_in_process(
        raw_path=raw_path,
        output_path=output_path,
        scratch=scratch,
        raw_rows=int(handoff["raw_rows"]),
    )
    validate_association_scratch_contract(
        scratch_contract, phase="POST_BH", require_empty=True
    )
    remove_empty_association_scratch_root(scratch_contract, phase="SUCCESS_CLEANUP")
    r9._memory_trace(
        "ASSOCIATION_BH_SORT_COMPLETE_R10",
        cancer_id=cancer,
        evidence_rows=evidence_rows,
    )
    _, manifest_sha = r9._write_manifest(publish, list(relative_paths))
    post_fields = {
        "association_scratch_post_exec_revalidated": True,
        "association_scratch_post_bh_revalidated": True,
        "association_scratch_root_removed_after_success": True,
        "association_scratch_contract_sha256": handoff[
            "association_scratch_contract_sha256"
        ],
        "server_tmp_fallback_used": False,
        "durable_publication_remained_on_dell_2": True,
    }
    lineage = {
        **lineage_base,
        **post_fields,
        "file_manifest_sha256": manifest_sha,
        "post_bh_exec_handoff_sha256": handoff["_verified_handoff_sha256"],
    }
    lineage_path = publish / "LINEAGE.json"
    r9._exclusive_json(lineage_path, lineage)
    success = {
        **success_base,
        **post_fields,
        "association_evidence_rows": int(evidence_rows),
        "file_manifest_sha256": manifest_sha,
        "lineage_sha256": r9._sha256(lineage_path),
        "post_bh_exec_handoff_sha256": handoff["_verified_handoff_sha256"],
    }
    success_path = publish / "SUCCESS.json"
    r9._exclusive_json(success_path, success)
    r9._memory_trace("ATOMIC_PUBLISH_START_R10", cancer_id=cancer)
    r9.atomic_publish_directory(publish, final)
    return {**success, "output_root": str(final)}


def main() -> int:
    r9 = _load_r9()
    args = build_parser(r9).parse_args()
    _validate_public_args(args, r9)
    bundle = _promote_bundle_to_r10(r9._load_contract(args), r9)
    if args.mode == "plan":
        output = args.plan_output_json.resolve()
        r9._authorized_output(output)
        if output.exists() or output.is_symlink():
            raise R10StreamingRunError(f"plan output reuse is forbidden: {output}")
        plan = r9._plan_summary(bundle)
        plan.update(
            {
                "format": R10_PLAN_FORMAT,
                "source_generation": R10_SOURCE_GENERATION,
                "formal_run_requires_explicit_association_scratch_root": True,
                "server_tmp_fallback_permitted": False,
                "scratch_created_during_plan": False,
            }
        )
        r9._exclusive_json(output, plan)
        print(json.dumps({**plan, "plan_output_json": str(output)}, indent=2, sort_keys=True))
        return 0

    output_parent = args.output_parent.resolve()
    _require_dell2_durable_output(output_parent)
    estimate = estimate_association_scratch_bytes(bundle["resource_estimate"])
    # Early read-only gate avoids spending hours in expression accumulation only
    # to discover an invalid scratch location.  Exclusive creation is delayed
    # until the association materialization boundary.
    preflight_association_scratch_root(
        args.association_scratch_root,
        estimate_bytes=estimate,
        forbidden_output_roots=(output_parent,),
    )
    accumulator, work = r9._run_stream(bundle, args)
    result = _materialize_with_r10_scratch(bundle, accumulator, work, args, r9)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _internal_main(argv: list[str]) -> int:
    if argv[0] != "--internal-post-bh-r10":
        raise R10StreamingRunError(f"unknown internal mode: {argv[0]}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff-json", required=True, type=Path)
    parser.add_argument("--expected-handoff-sha256", required=True)
    args = parser.parse_args(argv[1:])
    r9 = _load_r9()
    handoff_path = args.handoff_json.resolve()
    r9._authorized_output(handoff_path)
    observed_sha = _sha256(handoff_path)
    if observed_sha != args.expected_handoff_sha256:
        raise R10StreamingRunError("post-BH exec handoff SHA-256 drift")
    handoff = _load_json(handoff_path)
    declared_runner = Path(str(handoff.get("source_runner_path", "")))
    current_runner = Path(__file__).resolve()
    if declared_runner.is_symlink() or declared_runner.resolve() != current_runner:
        raise R10StreamingRunError("post-BH exec R10 runner path drift")
    if handoff.get("source_runner_sha256") != _sha256(current_runner):
        raise R10StreamingRunError("post-BH exec R10 runner SHA-256 drift")
    if handoff.get("r9_numeric_kernel_sha256") != _sha256(R9_RUNNER):
        raise R10StreamingRunError("post-BH exec frozen R9 numeric kernel drift")
    if int(handoff.get("producer_pid", -1)) != os.getpid():
        raise R10StreamingRunError("post-BH exec did not preserve producer PID")
    handoff["_verified_handoff_sha256"] = observed_sha
    result = _post_bh_exec_publish_r10(handoff, r9)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1].startswith("--internal-"):
            raise SystemExit(_internal_main(sys.argv[1:]))
        raise SystemExit(main())
    except AssociationScratchContractError as exc:
        print(json.dumps(exc.as_payload(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from None
