"""Validation-only winner lock for fresh patient-first G0/G1/G2 runs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    validate_frozen_v32_prepared_fold_binding,
)
from .sealed_test_inference import GRAPH_VARIANTS, WINNER_DECLARATION_FORMAT
from .training_guard import code_tree_sha256

COMPARISON_SOURCES_FORMAT = "G012_COMPARISON_SOURCES_V1"


class G012WinnerLockError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise G012WinnerLockError(f"Unreadable {label}: {path}") from exc
    if not isinstance(value, dict):
        raise G012WinnerLockError(f"{label} must be a JSON object: {path}")
    return value


def _task_fold_map(path: Path) -> dict[int, str]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise G012WinnerLockError(f"Unreadable task manifest: {path}") from exc
    result: dict[int, str] = {}
    for row in rows:
        fold = int(row.get("patient_fold", -1))
        if fold in result or fold not in range(5):
            raise G012WinnerLockError("Task manifest fold inventory is invalid")
        if row.get("status") != "PENDING" or row.get("task_type") != "CC_HHGT_PATIENT_FOLD":
            raise G012WinnerLockError("Task manifest is not an authorized five-fold training plan")
        result[fold] = str(row.get("task_id", ""))
    if set(result) != set(range(5)):
        raise G012WinnerLockError("Task manifest does not cover folds 0..4")
    return result


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_comparison_sources(declaration, root):
    """Replay all fifteen recorded validation losses from their native receipts.

    Old locks lack this inventory and return None. New locks cannot silently
    discard a losing arm or replace its receipt with a winner-only assertion.
    This validates selection lineage, not optimizer execution or raw ancestry.
    """
    from .current_core_export import verified_path
    from .sealed_test_inference import _validate_winner_declaration, _records_by_fold
    if "comparison_sources_format" not in declaration and "compared_runs" not in declaration:
        return None
    if declaration.get("comparison_sources_format") != COMPARISON_SOURCES_FORMAT:
        raise G012WinnerLockError("Unknown G012 comparison source inventory")
    runs = declaration.get("compared_runs", {})
    if set(runs) != set(GRAPH_VARIANTS):
        raise G012WinnerLockError("G012 comparison sources must include all three arms")
    rows = declaration.get("selection_table", [])
    if len(rows) != 3 or {r["graph_variant"] for r in rows} != set(GRAPH_VARIANTS):
        raise G012WinnerLockError("G012 comparison loss table is incomplete")
    selected_rows = {row["graph_variant"]: row for row in rows}
    if declaration.get("selection_metric") != "MEAN_FIVE_FOLD_BEST_VALIDATION_LOSS":
        raise G012WinnerLockError("G012 comparison metric changed")
    candidate = declaration["candidate_authority"]
    verified_path(candidate, root)
    if candidate["sha256"] != declaration["candidate_authority_sha256"]:
        raise G012WinnerLockError("G012 comparison candidate authority differs")
    codes = set()
    for variant in GRAPH_VARIANTS:
        run = runs[variant]
        # Reuse the sealed-inference authority checks without opening test data.
        _validate_winner_declaration({**declaration, "winner_id": variant, "graph_variant": variant,
            "prepared_folds": run["prepared_folds"], "checkpoints": run["checkpoints"]}, expected_graph_variant=variant)
        prepared = _records_by_fold(run["prepared_folds"], "comparison prepared folds")
        checkpoints = _records_by_fold(run["checkpoints"], "comparison checkpoints")
        paths = {key: verified_path(run[key], root) for key in ("config", "input_manifest", "task_manifest")}
        tasks = _task_fold_map(paths["task_manifest"])
        manifest = _json(paths["input_manifest"], "comparison input manifest")
        inputs = manifest.get("fold_inputs", [])
        if len(inputs) != 5 or any(type(r.get("fold")) is not int for r in inputs) or {r["fold"] for r in inputs} != set(range(5)):
            raise G012WinnerLockError("G012 comparison input manifest is incomplete")
        inputs = {r["fold"]: r for r in inputs}
        losses = []
        for fold in range(5):
            p, c = prepared[fold], checkpoints[fold]
            verified_path(p, root); verified_path(c, root)
            success_path = verified_path(c["training_receipt"], root)
            success = _json(success_path, "comparison training receipt")
            expected_hashes = {"code_sha256": declaration["training_code_sha256"],
                "config_sha256": run["config"]["sha256"], "input_manifest_sha256": run["input_manifest"]["sha256"],
                "task_manifest_sha256": run["task_manifest"]["sha256"]}
            if (inputs[fold].get("sha256") != p["sha256"]
                    or success.get("status") != "SUCCESS" or success.get("patient_fold") != fold
                    or success.get("task_id") != tasks[fold]
                    or success.get("architecture_id") != "HHGT_FORMAL_CORE_EXTERNAL_ROUTER"
                    or success.get("evidence_integration_mode") != "external_router"
                    or success.get("test_labels_read") is not False or success.get("test_metrics_computed") is not False
                    or success.get("artifact_hashes") != expected_hashes
                    or c["training_artifact_hashes"] != expected_hashes
                    or run["training_artifact_hashes"] != expected_hashes):
                raise G012WinnerLockError("G012 comparison receipt/input authority differs")
            for key, expected in (("best_model_state_sha256", c["sha256"]), ("prepared_artifact_sha256", p["sha256"]), ("graph_variant", variant)):
                if key in success and success[key] != expected:
                    raise G012WinnerLockError("G012 comparison selected checkpoint differs: " + key)
            loss = float(success.get("best_validation_loss", math.nan))
            if not math.isfinite(loss) or loss < 0: raise G012WinnerLockError("Invalid G012 comparison validation loss")
            losses.append(loss); codes.add(expected_hashes["code_sha256"])
        row = selected_rows[variant]
        if (row["fold_validation_losses"] != losses or not math.isclose(row["mean_validation_loss"], sum(losses)/5, rel_tol=0, abs_tol=1e-12)):
            raise G012WinnerLockError("G012 comparison losses differ from actual training receipts")
    winner = min(rows, key=lambda r: (r["mean_validation_loss"], GRAPH_VARIANTS.index(r["graph_variant"])))["graph_variant"]
    if winner != declaration["winner_id"] or declaration["graph_variant"] != winner or len(codes) != 1:
        raise G012WinnerLockError("G012 comparison winner or shared code differs")
    for field in ("prepared_folds", "checkpoints", "config"):
        if declaration[field] != runs[winner][field]:
            raise G012WinnerLockError("G012 winner aliases differ from comparison inventory")
    return runs


def lock_g012_validation_winner(
    *,
    repo_root: str | Path | None = None,
    prepared_parent: str | Path,
    run_parent: str | Path,
    candidate_authority_path: str | Path,
    output_root: str | Path,
    pair_fold_seed: int,
) -> dict[str, Any]:
    """Select one graph arm from five-fold validation loss without test access.

    ``run_parent/<variant>`` must contain ``config.yaml``, ``authorization``
    and ``training``.  Every physical hash is re-derived before the immutable
    declaration is written.
    """

    import socket
    if socket.gethostname() != "149": raise G012WinnerLockError("G012 winner selection requires host 149")
    prepared_parent = Path(prepared_parent).resolve()
    run_parent = Path(run_parent).resolve()
    candidate = Path(candidate_authority_path).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise G012WinnerLockError(f"Winner-lock output reuse is forbidden: {output}")
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise G012WinnerLockError(f"Candidate authority is missing: {candidate}")
    if not isinstance(pair_fold_seed, int) or pair_fold_seed < 0:
        raise G012WinnerLockError("pair_fold_seed must be a non-negative integer")
    output.mkdir(parents=True)
    _atomic_json(output / "LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False,
        "test_inputs_opened":False, "stage":"VALIDATION_SELECTION_SOURCE_CAPTURE"})
    expected_code_sha256 = (
        code_tree_sha256(Path(repo_root).resolve()) if repo_root is not None else None
    )

    variant_rows: list[dict[str, Any]] = []
    variant_records: dict[str, dict[str, Any]] = {}
    observed_code_hashes: set[str] = set()
    for variant in GRAPH_VARIANTS:
        prepared = prepared_parent / variant
        binding = validate_frozen_v32_prepared_fold_binding(prepared)
        marker = _json(prepared / "FORMAL_GRAPH_VARIANT.json", "graph variant marker")
        if marker.get("variant") != variant:
            raise G012WinnerLockError(f"Prepared root variant drift: {variant}")

        run = run_parent / variant
        config = run / "config.yaml"
        authorization = run / "authorization"
        input_manifest_path = authorization / "INPUT_MANIFEST.json"
        task_manifest_path = authorization / "TASK_MANIFEST.tsv"
        for label, path in (
            ("config", config),
            ("input manifest", input_manifest_path),
            ("task manifest", task_manifest_path),
        ):
            if not path.is_file() or path.stat().st_size <= 0:
                raise G012WinnerLockError(f"Missing {variant} {label}: {path}")
        input_manifest = _json(input_manifest_path, f"{variant} input manifest")
        fold_inputs = input_manifest.get("fold_inputs")
        if not isinstance(fold_inputs, list) or len(fold_inputs) != 5:
            raise G012WinnerLockError(f"{variant} input manifest lacks five folds")
        declared_inputs = {int(item["fold"]): item for item in fold_inputs}
        tasks = _task_fold_map(task_manifest_path)

        prepared_records: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        losses: list[float] = []
        training_hashes: dict[str, Any] | None = None
        for fold in range(5):
            prepared_path = (prepared / f"PATIENT_FOLD_{fold}.pt").resolve()
            prepared_sha = sha256_file(prepared_path)
            declared = declared_inputs.get(fold, {})
            if (
                Path(str(declared.get("path", ""))).resolve() != prepared_path
                or declared.get("sha256") != prepared_sha
            ):
                raise G012WinnerLockError(f"{variant} fold {fold} input-manifest drift")
            task_id = tasks[fold]
            task_dir = run / "training" / task_id.replace("|", "__")
            success = _json(task_dir / "SUCCESS.json", f"{variant} fold {fold} SUCCESS")
            checkpoint_path = task_dir / "best_model_state.pt"
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
                raise G012WinnerLockError(f"Missing {variant} fold {fold} checkpoint")
            expected_hashes = {
                "code_sha256": success.get("artifact_hashes", {}).get("code_sha256"),
                "config_sha256": sha256_file(config),
                "input_manifest_sha256": sha256_file(input_manifest_path),
                "task_manifest_sha256": sha256_file(task_manifest_path),
            }
            if (
                success.get("status") != "SUCCESS"
                or int(success.get("patient_fold", -1)) != fold
                or success.get("task_id") != task_id
                or success.get("architecture_id") != "HHGT_FORMAL_CORE_EXTERNAL_ROUTER"
                or success.get("evidence_integration_mode") != "external_router"
                or success.get("test_labels_read") is not False
                or success.get("test_metrics_computed") is not False
                or success.get("artifact_hashes") != expected_hashes
            ):
                raise G012WinnerLockError(f"{variant} fold {fold} SUCCESS lineage drift")
            code_sha = str(expected_hashes["code_sha256"] or "")
            if len(code_sha) != 64:
                raise G012WinnerLockError(f"{variant} fold {fold} code hash is invalid")
            if expected_code_sha256 is not None and code_sha != expected_code_sha256:
                raise G012WinnerLockError(
                    f"{variant} fold {fold} executed a different code tree"
                )
            observed_code_hashes.add(code_sha)
            loss = float(success.get("best_validation_loss", math.nan))
            if not math.isfinite(loss) or loss < 0:
                raise G012WinnerLockError(f"{variant} fold {fold} validation loss is invalid")
            losses.append(loss)
            if training_hashes is None:
                training_hashes = expected_hashes
            elif expected_hashes != training_hashes:
                raise G012WinnerLockError(
                    f"{variant} folds did not share one authorization hash closure"
                )
            prepared_records.append(
                {
                    "patient_fold": fold,
                    "graph_variant": variant,
                    "path": str(prepared_path),
                    "sha256": prepared_sha,
                }
            )
            checkpoint_records.append(
                {
                    "patient_fold": fold,
                    "graph_variant": variant,
                    "path": str(checkpoint_path.resolve()),
                    "sha256": sha256_file(checkpoint_path),
                    "prepared_fold_sha256": prepared_sha,
                    "architecture_id": success["architecture_id"],
                    "selected_using": "VALIDATION_ONLY",
                    "heldout_test_metrics_used": False,
                    "training_artifact_hashes": expected_hashes,
                    "training_receipt": {"path": str((task_dir / "SUCCESS.json").resolve()),
                                         "sha256": sha256_file(task_dir / "SUCCESS.json")},
                }
            )
        mean_loss = float(sum(losses) / len(losses))
        variant_rows.append(
            {
                "graph_variant": variant,
                "fold_validation_losses": losses,
                "mean_validation_loss": mean_loss,
                "selection_rank_key": [mean_loss, GRAPH_VARIANTS.index(variant)],
            }
        )
        variant_records[variant] = {
            "prepared_folds": prepared_records,
            "checkpoints": checkpoint_records,
            "config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
            "input_manifest": {"path": str(input_manifest_path.resolve()), "sha256": sha256_file(input_manifest_path)},
            "task_manifest": {"path": str(task_manifest_path.resolve()), "sha256": sha256_file(task_manifest_path)},
            "patient_fold_authority": binding,
            "training_artifact_hashes": training_hashes,
        }

    if len(observed_code_hashes) != 1:
        raise G012WinnerLockError("G0/G1/G2 folds did not execute one shared code tree")

    winner_row = min(
        variant_rows,
        key=lambda row: (row["mean_validation_loss"], GRAPH_VARIANTS.index(row["graph_variant"])),
    )
    winner = str(winner_row["graph_variant"])
    selected = variant_records[winner]
    declaration = {
        "format": WINNER_DECLARATION_FORMAT,
        "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
        "compared_graph_variants": list(GRAPH_VARIANTS),
        "winner_id": winner,
        "graph_variant": winner,
        "selection_scope": "VALIDATION_ONLY",
        "selection_metric": "MEAN_FIVE_FOLD_BEST_VALIDATION_LOSS",
        "selection_table": variant_rows,
        "heldout_test_metrics_used_for_selection": False,
        "test_inputs_opened_before_winner_lock": False,
        "winner_or_checkpoint_changed_by_test": False,
        "declaration_locked": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
        "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        "authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        "candidate_authority_sha256": sha256_file(candidate),
        "candidate_authority": {"path": str(candidate), "sha256": sha256_file(candidate)},
        "pair_fold_seed": pair_fold_seed,
        "training_code_sha256": next(iter(observed_code_hashes)),
        "prepared_folds": selected["prepared_folds"],
        "checkpoints": selected["checkpoints"],
        "config": selected["config"],
        "comparison_sources_format": COMPARISON_SOURCES_FORMAT,
        "compared_runs": variant_records,
    }
    validate_comparison_sources(declaration, output)
    _atomic_json(output / "G012_VALIDATION_WINNER_LOCK.json", declaration)
    receipt = {
        "status": "PASS_VALIDATION_ONLY_WINNER_LOCK",
        "winner": winner,
        "winner_declaration": {
            "path": str((output / "G012_VALIDATION_WINNER_LOCK.json").resolve()),
            "sha256": sha256_file(output / "G012_VALIDATION_WINNER_LOCK.json"),
        },
        "test_inputs_opened": False,
        "variants_compared": list(GRAPH_VARIANTS),
    }
    _atomic_json(output / "SUCCESS.json", receipt)
    return receipt


__all__ = ["G012WinnerLockError", "lock_g012_validation_winner", "sha256_file"]
