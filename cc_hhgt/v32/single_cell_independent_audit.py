"""Independent, fail-closed audit for the formal V3.2 single-cell head.

This module deliberately does not import :mod:`single_cell_training`.  It
reconstructs hashes, split blocks, exact-key sets, aggregate semantics and the
fusion-adapter closure directly from frozen artifacts.  The training module is
read only as code-bound evidence for the constant-lncRNA mask; none of its
validation helpers are called.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
AUDIT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_INDEPENDENT_AUDIT_V1"
BINDING_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_INDEPENDENT_AUDIT_BINDING_V1"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_PRIVATE_HEAD_V1"
CORE_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
PREDICTION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_TYPED_PREDICTIONS_V1"
DEFAULT_EXPECTED_TRAINING_RUN_ID = "V32-SINGLE-CELL-FRESH-20260826-R1"
EXPECTED_TYPED_ROWS = 7_814_014
EXPECTED_AVAILABLE_ROWS = 6_214_014
EXPECTED_NULL_ROWS = 1_600_000
EXPECTED_CANDIDATE_ROWS = 3_300_000
EXPECTED_FORMAL_CANCERS = 17
EXPECTED_UNAVAILABLE_CANCERS = 16
N_FOLDS = 5
FORMAL_TIERS = {
    "formal", "approved", "released", "primary", "primary_raw",
    "primary_raw_count", "raw",
}
FORMAL_QUALITY = {"pass", "qualified", "approved", "formal"}
DOMAIN_FEATURES = (
    "lnc_detection_rate",
    "lnc_mean_log_expression",
    "lnc_specificity_tau",
    "pathway_activity",
    "pathway_mean_ucell",
    "pathway_mean_pseudotime",
    "log1p_n_cells",
    "lnc_feature_available",
    "pathway_feature_available",
)
TARGET_KEYS = ("dataset_id", "cell_type", "lncrna_id", "pathway_id")
EXACT_KEYS = ("cancer_id", "lncrna_id", "pathway_id")


class IndependentAuditError(RuntimeError):
    """Raised when an audit cannot complete without weakening its contract."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink():
        raise IndependentAuditError(f"symlink artifact is forbidden: {source}")
    if source.is_file():
        return file_sha256(source)
    if not source.is_dir():
        raise IndependentAuditError(f"artifact is missing: {source}")
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise IndependentAuditError(f"artifact directory is empty: {source}")
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            raise IndependentAuditError(f"symlink artifact is forbidden: {item}")
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor content, dtype, shape and key without Torch serialization."""

    import torch

    digest = hashlib.sha256()
    if not state:
        raise IndependentAuditError("empty state_dict")
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise IndependentAuditError(f"state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def independently_rebuild_initial_state_sha256(
    *, seed: int, core_features: int, domain_features: int,
    hidden_features: int, dropout: float,
) -> str:
    """Recreate the declared architecture without importing trainer code."""

    import torch
    from torch import nn

    class IndependentPrivateHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(int(core_features) + int(domain_features), int(hidden_features)),
                nn.LayerNorm(int(hidden_features)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_features), 1),
            )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        head = IndependentPrivateHead()
    return state_dict_sha256(head.state_dict())


def deterministic_block_folds(block_ids: Iterable[str], seed: int) -> dict[str, int]:
    unique = sorted(
        set(map(str, block_ids)),
        key=lambda value: (
            hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    return {block: index % N_FOLDS for index, block in enumerate(unique)}


def split_blocks(mapping: Mapping[str, int], fold: int) -> dict[str, set[str]]:
    validation = (int(fold) + 1) % N_FOLDS
    result = {
        "test": {block for block, assigned in mapping.items() if assigned == int(fold)},
        "validation": {block for block, assigned in mapping.items() if assigned == validation},
        "train": {
            block for block, assigned in mapping.items()
            if assigned not in {int(fold), validation}
        },
    }
    if (
        result["train"] & result["validation"]
        or result["train"] & result["test"]
        or result["validation"] & result["test"]
    ):
        raise IndependentAuditError(f"fold {fold} block overlap")
    if set().union(*result.values()) != set(mapping):
        raise IndependentAuditError(f"fold {fold} block partition is incomplete")
    return result


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _canonical_node(value: Any, kind: str) -> str:
    text = str(value).strip()
    prefixes = ("LNC:", "LNCRNA:") if kind == "lncrna" else ("PATHWAY:", "PW:")
    for prefix in prefixes:
        if text.upper().startswith(prefix):
            return text[len(prefix):]
    return text


def _resolve(repo_root: Path, manifest_path: Path, value: str | Path) -> Path:
    candidate = Path(value)
    trials = [candidate] if candidate.is_absolute() else [repo_root / candidate]
    trials.extend(parent / candidate for parent in manifest_path.parents)
    for trial in trials:
        if trial.exists():
            return trial.resolve()
    raise IndependentAuditError(f"cannot resolve declared artifact: {value}")


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _sql_strings(values: Iterable[str]) -> str:
    return ",".join("'" + str(value).replace("'", "''") + "'" for value in values)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IndependentAuditError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def embedding_stats(path: Path, kind: str) -> tuple[dict[str, Any], set[str]]:
    frame = pd.read_parquet(path)
    if "node_id" not in frame:
        raise IndependentAuditError(f"{kind} export lacks node_id: {path}")
    columns = sorted(column for column in frame if str(column).startswith("core_feature_"))
    if not columns:
        raise IndependentAuditError(f"{kind} export lacks core features: {path}")
    ids = frame.node_id.map(lambda value: _canonical_node(value, kind)).astype(str)
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if ids.duplicated().any() or not np.isfinite(values).all():
        raise IndependentAuditError(f"{kind} export has duplicate IDs or non-finite values")
    unique = int(len(np.unique(values, axis=0)))
    varying = int(np.count_nonzero(np.ptp(values, axis=0) > 1.0e-12))
    usable = bool(len(values) == 1 or (unique > 1 and varying > 0))
    return (
        {
            "rows": int(values.shape[0]),
            "features": int(values.shape[1]),
            "unique_embedding_vectors": unique,
            "varying_feature_count": varying,
            "usable_for_node_discrimination": usable,
            "status": "USABLE" if usable else "CONSTANT_EMBEDDING_MASKED",
        },
        set(ids),
    )


def inspect_constant_mask_source(path: Path) -> dict[str, Any]:
    """AST-bind the source semantics needed to prove the constant branch mask."""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    function = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "candidate_core"),
        None,
    )
    if function is None:
        raise IndependentAuditError("candidate_core function is absent from bound trainer")
    source = ast.unparse(function)
    tokens = {
        "zero_initialization": "np.zeros" in source,
        "lncrna_usability_guard": (
            'embedding_usability(core.lncrna)["usable_for_node_discrimination"]' in source
            or "embedding_usability(core.lncrna)['usable_for_node_discrimination']" in source
        ),
        "lncrna_first_half_assignment": "values[available, :width]" in source,
        "pathway_second_half_assignment": "values[available, width:]" in source,
    }
    return {
        "trainer_path": str(path),
        "trainer_sha256": file_sha256(path),
        "function_ast_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "semantic_tokens": tokens,
        "mask_semantics_present": all(tokens.values()),
    }


@dataclass
class AuditRecorder:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def check(
        self, name: str, condition: bool, *, observed: Any = None,
        expected: Any = None, details: Any = None,
    ) -> bool:
        row: dict[str, Any] = {
            "check": str(name),
            "status": "PASS" if bool(condition) else "FAIL",
        }
        if observed is not None:
            row["observed"] = observed
        if expected is not None:
            row["expected"] = expected
        if details is not None:
            row["details"] = details
        self.checks.append(row)
        return bool(condition)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(row["status"] == "PASS" for row in self.checks)


def _qualified_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset_id", "cancer_id", "formal_eligible", "source_tier",
        "quality_status", "feature_universe_status", "donor_metadata_available",
    }
    if missing := sorted(required - set(frame.columns)):
        raise IndependentAuditError(f"dataset manifest lacks {missing}")
    result = frame.copy()
    result["source_token"] = result.source_tier.map(_normalise_token)
    result["quality_token"] = result.quality_status.map(_normalise_token)
    result["feature_token"] = result.feature_universe_status.map(_normalise_token)
    mask = (
        result.formal_eligible.astype(bool)
        & result.source_token.isin(FORMAL_TIERS)
        & result.quality_token.isin(FORMAL_QUALITY)
        & result.feature_token.eq("pass")
        & result.donor_metadata_available.astype(bool)
    )
    return result.loc[mask].copy()


def _snapshot(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        source = path.resolve()
        if source.is_file():
            files = 1
            size = source.stat().st_size
        elif source.is_dir():
            members = [item for item in source.rglob("*") if item.is_file()]
            files = len(members)
            size = sum(item.stat().st_size for item in members)
        else:
            raise IndependentAuditError(f"snapshot artifact missing: {source}")
        result[name] = {
            "path": str(source),
            "sha256": artifact_sha256(source),
            "files": int(files),
            "bytes": int(size),
        }
    return result


def verify_report_binding(report_path: str | Path, binding: Mapping[str, Any]) -> bool:
    source = Path(report_path)
    return (
        source.is_file()
        and str(binding.get("status")) == "PASS"
        and str(binding.get("audit_report_sha256")) == file_sha256(source)
    )


def run_independent_audit(
    *, repo_root: str | Path, training_root: str | Path,
    input_root: str | Path, candidate_universe: str | Path,
    core_manifest: str | Path, output_root: str | Path,
    expected_training_run_id: str = DEFAULT_EXPECTED_TRAINING_RUN_ID,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    training = Path(training_root).resolve()
    inputs = Path(input_root).resolve()
    candidates = Path(candidate_universe).resolve()
    core_manifest_path = Path(core_manifest).resolve()
    output = Path(output_root).resolve()
    if output.exists():
        raise IndependentAuditError(f"independent audit refuses output reuse: {output}")
    output.mkdir(parents=True)
    recorder = AuditRecorder()
    started = datetime.now(timezone.utc).isoformat()

    required_training_files = {
        "typed_predictions": training / "single_cell_typed_predictions.parquet",
        "exact_aggregate": training / "lnc_exact_pathway.parquet",
        "lnc_celltype": training / "lnc_celltype_summary.parquet",
        "activity": training / "activity.parquet",
        "checkpoint_manifest": training / "CHECKPOINT_MANIFEST.json",
        "run_config": training / "RUN_CONFIG.json",
        "input_lineage": training / "INPUT_LINEAGE_AUDIT.json",
        "lineage": training / "LINEAGE.json",
        "success": training / "SUCCESS.json",
        "contract": training / "FULL_MODEL_CONTRACT_VALIDATION.json",
        "figures": training / "FIGURE_MANIFEST.json",
    }
    required_input_paths = {
        "candidate_universe": candidates,
        "dataset_manifest": inputs / "dataset_manifest_33c.parquet",
        "association": inputs / "association",
        "lnc_celltype_input": inputs / "lnc_celltype",
        "activity_input": inputs / "activity",
        "core_manifest": core_manifest_path,
    }
    for name, path in {**required_training_files, **required_input_paths}.items():
        recorder.check(f"path_exists::{name}", path.exists(), observed=str(path))
    if not all(path.exists() for path in {**required_training_files, **required_input_paths}.values()):
        raise IndependentAuditError("required authoritative artifacts are missing")

    source_paths = {**required_training_files, **required_input_paths}
    source_before = _snapshot(source_paths)
    run_config = _json(required_training_files["run_config"])
    checkpoint_manifest = _json(required_training_files["checkpoint_manifest"])
    lineage = _json(required_training_files["lineage"])
    success = _json(required_training_files["success"])
    input_lineage = _json(required_training_files["input_lineage"])
    contract = _json(required_training_files["contract"])
    core_payload = _json(core_manifest_path)

    recorder.check("analysis_version::config", run_config.get("analysis_version") == ANALYSIS_VERSION)
    recorder.check("analysis_version::lineage", lineage.get("analysis_version") == ANALYSIS_VERSION)
    recorder.check("training_run_id", run_config.get("training_run_id") == expected_training_run_id,
                   observed=run_config.get("training_run_id"), expected=expected_training_run_id)
    recorder.check("formal_success", success.get("status") == "SUCCESS" and success.get("release_ready") is True)
    recorder.check("contract_pass", contract.get("status") == "PASS")
    recorder.check("input_lineage_pass", input_lineage.get("status") == "PASS")
    recorder.check("core_manifest_format", core_payload.get("export_format") == CORE_FORMAT)
    recorder.check("core_new_v32", core_payload.get("all_embeddings_from_newly_trained_v32_core") is True)
    recorder.check("core_no_historical_checkpoint", core_payload.get("historical_checkpoint_loaded") is False)
    recorder.check("core_no_historical_prediction", core_payload.get("historical_prediction_loaded") is False)

    training_code = repo / "cc_hhgt" / "v32" / "single_cell_training.py"
    training_runner = repo / "scripts" / "run_v32_single_cell_training.py"
    training_code_files = {
        "single_cell_training.py": file_sha256(training_code),
        "run_v32_single_cell_training.py": file_sha256(training_runner),
    }
    recomputed_training_code_sha = canonical_sha256(training_code_files)
    recorder.check("training_code_hash_bound", recomputed_training_code_sha == lineage.get("code_sha256"),
                   observed=recomputed_training_code_sha, expected=lineage.get("code_sha256"))
    recorder.check("run_config_hash_bound", file_sha256(required_training_files["run_config"]) == lineage.get("config_sha256"))
    recorder.check("checkpoint_manifest_hash_bound", file_sha256(required_training_files["checkpoint_manifest"]) == lineage.get("checkpoint_manifest_sha256"))
    recorder.check("input_lineage_hash_bound", input_lineage.get("lineage_sha256") == lineage.get("input_manifest_sha256"))
    recorder.check("lineage_success_hash_bound", file_sha256(required_training_files["lineage"]) == success.get("lineage_sha256"))
    mask_source = inspect_constant_mask_source(training_code)
    recorder.check("constant_lncrna_mask_is_code_bound", mask_source["mask_semantics_present"])

    declared_inputs = {Path(row["path"]).resolve(): row for row in lineage.get("input_artifacts", [])}
    expected_declared = set(path.resolve() for path in required_input_paths.values())
    recorder.check("lineage_declares_exact_authoritative_inputs", set(declared_inputs) == expected_declared,
                   observed=sorted(map(str, declared_inputs)), expected=sorted(map(str, expected_declared)))
    input_hash_rows: list[dict[str, Any]] = []
    for path in sorted(expected_declared, key=str):
        declaration = declared_inputs.get(path, {})
        observed_hash = artifact_sha256(path)
        declared_hash = declaration.get("sha256")
        recorder.check(f"declared_input_hash::{path.name}", observed_hash == declared_hash,
                       observed=observed_hash, expected=declared_hash)
        input_hash_rows.append({"path": str(path), "observed_sha256": observed_hash,
                                "declared_sha256": declared_hash})

    config = run_config.get("training", {})
    base_seed = int(config.get("seed", -1))
    hidden = int(config.get("hidden_features", -1))
    dropout = float(config.get("dropout", -1))
    epochs = int(config.get("epochs", -1))
    batch_size = int(config.get("batch_size", -1))
    patience = int(config.get("patience", -1))
    recorder.check("five_checkpoint_manifest_records", len(checkpoint_manifest.get("records", [])) == N_FOLDS)
    recorder.check("checkpoint_manifest_format", checkpoint_manifest.get("checkpoint_format") == CHECKPOINT_FORMAT)
    recorder.check("checkpoint_manifest_all_five", checkpoint_manifest.get("all_five_fold_private_heads_trained") is True)

    import torch

    checkpoint_audit: list[dict[str, Any]] = []
    final_parameter_hashes: list[str] = []
    total_optimizer_steps_lower_bound = 0
    total_best_checkpoint_steps_lower_bound = 0
    manifest_records = {
        int(row.get("single_cell_fold", -1)): row for row in checkpoint_manifest.get("records", [])
    }
    expected_shapes = {
        "network.0.weight": (hidden, 96 * 2 + len(DOMAIN_FEATURES)),
        "network.0.bias": (hidden,),
        "network.1.weight": (hidden,),
        "network.1.bias": (hidden,),
        "network.4.weight": (1, hidden),
        "network.4.bias": (1,),
    }
    for fold in range(N_FOLDS):
        record = manifest_records.get(fold, {})
        checkpoint_path = training / "checkpoints" / f"single_cell_fold_{fold}.pt"
        declared_path = Path(record.get("path", "")).resolve() if record.get("path") else None
        observed_file_hash = file_sha256(checkpoint_path)
        recorder.check(f"checkpoint::{fold}::path", declared_path == checkpoint_path.resolve())
        recorder.check(f"checkpoint::{fold}::file_hash", observed_file_hash == record.get("sha256"),
                       observed=observed_file_hash, expected=record.get("sha256"))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        initialization = payload.get("initialization", {})
        model_state = payload.get("model_state", {})
        final_hash = state_dict_sha256(model_state)
        final_parameter_hashes.append(final_hash)
        expected_seed = base_seed + fold * 101
        rebuilt_initial_hash = independently_rebuild_initial_state_sha256(
            seed=expected_seed, core_features=192, domain_features=len(DOMAIN_FEATURES),
            hidden_features=hidden, dropout=dropout,
        )
        stored_initial_hash = initialization.get("initial_parameter_sha256")
        history = payload.get("history", [])
        epoch_values = [float(row.get("epoch", float("nan"))) for row in history]
        loss_values = [float(row.get("train_loss", float("nan"))) for row in history]
        validation_values = [float(row.get("validation_loss", float("nan"))) for row in history]
        best_epoch = int(np.argmin(validation_values)) if validation_values else -1
        train_rows = int(initialization.get("train_rows", -1))
        batches_per_epoch = math.ceil(train_rows / batch_size) if train_rows > 0 and batch_size > 0 else 0
        executed_steps_lb = len(history) * batches_per_epoch
        best_steps_lb = (best_epoch + 1) * batches_per_epoch
        total_optimizer_steps_lower_bound += executed_steps_lb
        total_best_checkpoint_steps_lower_bound += best_steps_lb
        tensor_shapes = {name: tuple(value.shape) for name, value in model_state.items()}
        tensors_finite = all(
            bool(torch.isfinite(value).all()) if value.is_floating_point() else True
            for value in model_state.values()
        )
        history_finite = bool(history) and bool(np.isfinite(loss_values + validation_values).all())
        expected_epochs = [float(value) for value in range(len(history))]
        early_stop_consistent = (
            len(history) == epochs
            or (len(history) < epochs and best_epoch >= 0 and (len(history) - 1 - best_epoch) >= patience)
        )
        recorder.check(f"checkpoint::{fold}::format_fold_module",
                       payload.get("checkpoint_format") == CHECKPOINT_FORMAT
                       and int(payload.get("single_cell_fold", -1)) == fold
                       and payload.get("module_id") == "single_cell")
        recorder.check(f"checkpoint::{fold}::seed", int(initialization.get("seed", -1)) == expected_seed,
                       observed=initialization.get("seed"), expected=expected_seed)
        recorder.check(f"checkpoint::{fold}::initial_hash_rebuilt", stored_initial_hash == rebuilt_initial_hash,
                       observed=stored_initial_hash, expected=rebuilt_initial_hash)
        recorder.check(f"checkpoint::{fold}::trained_parameter_change", final_hash != stored_initial_hash,
                       observed=final_hash, expected=f"different from {stored_initial_hash}")
        recorder.check(f"checkpoint::{fold}::state_shapes", tensor_shapes == expected_shapes,
                       observed=tensor_shapes, expected=expected_shapes)
        recorder.check(f"checkpoint::{fold}::finite_state", tensors_finite)
        recorder.check(f"checkpoint::{fold}::history", history_finite and epoch_values == expected_epochs
                       and 1 <= len(history) <= epochs and early_stop_consistent)
        recorder.check(f"checkpoint::{fold}::best_loss",
                       bool(validation_values)
                       and math.isclose(float(initialization.get("best_validation_loss", float("nan"))),
                                        min(validation_values), rel_tol=0.0, abs_tol=1e-7))
        recorder.check(f"checkpoint::{fold}::optimizer_step_lower_bound", executed_steps_lb > 0,
                       observed=executed_steps_lb, expected=">0")
        recorder.check(f"checkpoint::{fold}::fresh_no_old",
                       initialization.get("source_checkpoint_sha256") is None
                       and initialization.get("initialization_policy") == "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH"
                       and payload.get("old_checkpoint_loaded") is False
                       and payload.get("old_predictions_used") is False)
        recorder.check(f"checkpoint::{fold}::frozen_parent",
                       initialization.get("core_embedding_detached") is True
                       and initialization.get("core_parameters_frozen") is True)
        recorder.check(f"checkpoint::{fold}::domain_contract",
                       list(payload.get("domain_features", [])) == list(DOMAIN_FEATURES)
                       and np.asarray(payload.get("domain_mean")).shape == (len(DOMAIN_FEATURES),)
                       and np.asarray(payload.get("domain_scale")).shape == (len(DOMAIN_FEATURES),)
                       and np.isfinite(np.asarray(payload.get("domain_mean"), dtype=float)).all()
                       and np.isfinite(np.asarray(payload.get("domain_scale"), dtype=float)).all()
                       and (np.asarray(payload.get("domain_scale"), dtype=float) > 0).all())
        checkpoint_audit.append({
            "fold": fold,
            "seed": expected_seed,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": observed_file_hash,
            "initial_parameter_sha256": stored_initial_hash,
            "independently_rebuilt_initial_parameter_sha256": rebuilt_initial_hash,
            "final_parameter_sha256": final_hash,
            "initial_differs_from_final": final_hash != stored_initial_hash,
            "history_epochs": len(history),
            "best_epoch": best_epoch,
            "train_rows": train_rows,
            "validation_rows": int(initialization.get("validation_rows", -1)),
            "batches_per_epoch": batches_per_epoch,
            "optimizer_steps_executed_lower_bound": executed_steps_lb,
            "best_checkpoint_contributing_steps_lower_bound": best_steps_lb,
        })
    recorder.check("final_parameter_hashes_unique", len(set(final_parameter_hashes)) == N_FOLDS)
    recorder.check("optimizer_total_lower_bound", total_optimizer_steps_lower_bound > 0,
                   observed=total_optimizer_steps_lower_bound, expected=">0")

    core_records: list[dict[str, Any]] = []
    core_parameter_hashes: list[str] = []
    core_lnc_ids: dict[int, set[str]] = {}
    core_pathway_ids: dict[int, set[str]] = {}
    core_folds = core_payload.get("folds", {})
    recorder.check("core_exact_folds", set(map(str, core_folds)) == set(map(str, range(N_FOLDS))))
    for fold in range(N_FOLDS):
        declaration = core_folds[str(fold)]
        checkpoint_path = _resolve(repo, core_manifest_path, declaration["checkpoint_path"])
        checkpoint_file_hash = file_sha256(checkpoint_path)
        core_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        core_parameter_hash = state_dict_sha256(core_checkpoint["model_state"])
        core_parameter_hashes.append(core_parameter_hash)
        recorder.check(f"core::{fold}::checkpoint_file_hash",
                       checkpoint_file_hash == declaration.get("checkpoint_sha256"))
        recorder.check(f"core::{fold}::parameter_hash_recomputed",
                       core_parameter_hash == declaration.get("core_parameter_sha256"),
                       observed=core_parameter_hash, expected=declaration.get("core_parameter_sha256"))
        recorder.check(f"core::{fold}::fresh",
                       declaration.get("trained_from_scratch") is True
                       and declaration.get("old_checkpoint_loaded") is False
                       and int(declaration.get("patient_fold", -1)) == fold)
        exports: dict[str, Any] = {}
        for node_type, kind in (("lncRNA", "lncrna"), ("pathway", "pathway")):
            export = declaration.get("exports", {}).get(node_type, {})
            export_path = _resolve(repo, core_manifest_path, export.get("path", ""))
            export_hash = file_sha256(export_path)
            stats, ids = embedding_stats(export_path, kind)
            if kind == "lncrna":
                core_lnc_ids[fold] = ids
            else:
                core_pathway_ids[fold] = ids
            recorder.check(f"core::{fold}::{kind}::file_hash", export_hash == export.get("sha256"))
            recorder.check(f"core::{fold}::{kind}::shape",
                           stats["rows"] == int(export.get("rows", -1))
                           and stats["features"] == int(export.get("features", -1)))
            if kind == "lncrna":
                recorder.check(f"core::{fold}::lncrna_constant_mask_required",
                               stats["unique_embedding_vectors"] == 1
                               and stats["varying_feature_count"] == 0
                               and stats["usable_for_node_discrimination"] is False)
            else:
                recorder.check(f"core::{fold}::pathway_discriminative",
                               stats["unique_embedding_vectors"] > 1
                               and stats["varying_feature_count"] > 0
                               and stats["usable_for_node_discrimination"] is True)
            exports[kind] = {"path": str(export_path), "sha256": export_hash, **stats}
        checkpoint_record = manifest_records[fold]
        private_payload = torch.load(
            training / "checkpoints" / f"single_cell_fold_{fold}.pt",
            map_location="cpu", weights_only=False,
        )
        recorder.check(f"core::{fold}::private_binding",
                       private_payload.get("core_checkpoint_sha256") == checkpoint_file_hash
                       and private_payload.get("core_parameter_sha256") == core_parameter_hash
                       and checkpoint_record.get("core_checkpoint_sha256") == checkpoint_file_hash
                       and checkpoint_record.get("core_parameter_sha256") == core_parameter_hash)
        core_records.append({
            "fold": fold,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_file_hash,
            "recomputed_core_parameter_sha256": core_parameter_hash,
            "exports": exports,
        })
    core_composite = canonical_sha256(core_parameter_hashes)
    recorder.check("core_before_after_equal",
                   lineage.get("core_parameters_before_sha256") == core_composite
                   and lineage.get("core_parameters_after_sha256") == core_composite,
                   observed={"recomputed": core_composite,
                             "before": lineage.get("core_parameters_before_sha256"),
                             "after": lineage.get("core_parameters_after_sha256")})
    recorder.check("core_manifest_hash_bound",
                   file_sha256(core_manifest_path) == lineage.get("v32_core_checkpoint_sha256"))

    manifest_frame = pd.read_parquet(required_input_paths["dataset_manifest"])
    qualified = _qualified_manifest(manifest_frame)
    qualified_cancers = set(qualified.cancer_id.astype(str).str.upper())
    qualified_datasets = set(qualified.dataset_id.astype(str))
    recorder.check("manifest_17_formal_cancers", len(qualified_cancers) == EXPECTED_FORMAL_CANCERS,
                   observed=len(qualified_cancers), expected=EXPECTED_FORMAL_CANCERS)
    recorder.check("formal_dataset_unique_per_cancer",
                   qualified.groupby("cancer_id", observed=True).dataset_id.nunique().max() == 1)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    typed_sql = _sql_path(required_training_files["typed_predictions"])
    exact_sql = _sql_path(required_training_files["exact_aggregate"])
    candidate_sql = _sql_path(candidates)
    association_glob = _sql_path(required_input_paths["association"] / "*.parquet")
    candidate_stats = con.execute(f"""
        SELECT count(*) AS rows,
               count(DISTINCT cancer_id) AS cancers,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
               count_if(cancer_id IS NULL OR lncrna_id IS NULL OR pathway_id IS NULL) AS null_keys
        FROM read_parquet('{candidate_sql}')
    """).fetchone()
    candidate_by_cancer = con.execute(f"""
        SELECT cancer_id, count(*) AS rows
        FROM read_parquet('{candidate_sql}') GROUP BY cancer_id ORDER BY cancer_id
    """).fetchdf()
    recorder.check("candidate_3_3m", int(candidate_stats[0]) == EXPECTED_CANDIDATE_ROWS,
                   observed=int(candidate_stats[0]), expected=EXPECTED_CANDIDATE_ROWS)
    recorder.check("candidate_33_cancers", int(candidate_stats[1]) == 33)
    recorder.check("candidate_unique_nonnull",
                   int(candidate_stats[2]) == int(candidate_stats[0]) and int(candidate_stats[3]) == 0)
    recorder.check("candidate_100k_per_cancer",
                   candidate_by_cancer.rows.min() == 100_000 and candidate_by_cancer.rows.max() == 100_000)

    typed_stats = con.execute(f"""
        SELECT count(*) AS rows,
               count_if(single_cell_available) AS available_rows,
               count_if(NOT single_cell_available) AS null_rows,
               count(DISTINCT cancer_id) AS cancers,
               count_if(single_cell_available AND
                        (single_cell_replication_probability IS NULL OR
                         NOT isfinite(single_cell_replication_probability) OR
                         single_cell_replication_probability < 0 OR
                         single_cell_replication_probability > 1)) AS bad_available_probability,
               count_if(NOT single_cell_available AND single_cell_replication_probability IS NOT NULL) AS bad_null_probability,
               count_if(NOT single_cell_available AND
                        (single_cell_unavailable_reason IS NULL OR trim(single_cell_unavailable_reason) = '')) AS bad_null_reason,
               count_if(changes_primary_ranking) AS changes_primary,
               min(single_cell_replication_probability) AS probability_min,
               max(single_cell_replication_probability) AS probability_max,
               stddev_pop(single_cell_replication_probability) AS probability_sd
        FROM read_parquet('{typed_sql}')
    """).fetchone()
    typed_dup = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT dataset_id, cell_type, lncrna_id, pathway_id, count(*) AS n
          FROM read_parquet('{typed_sql}')
          GROUP BY dataset_id, cell_type, lncrna_id, pathway_id HAVING count(*) > 1
        )
    """).fetchone()[0])
    typed_null_keys = int(con.execute(f"""
        SELECT count(*) FROM read_parquet('{typed_sql}')
        WHERE dataset_id IS NULL OR trim(dataset_id)='' OR cell_type IS NULL OR trim(cell_type)=''
           OR lncrna_id IS NULL OR trim(lncrna_id)='' OR pathway_id IS NULL OR trim(pathway_id)=''
    """).fetchone()[0])
    typed_extra = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT DISTINCT cancer_id, lncrna_id, pathway_id FROM read_parquet('{typed_sql}')
          EXCEPT
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{candidate_sql}')
        )
    """).fetchone()[0])
    cancer_stats = con.execute(f"""
        SELECT cancer_id, count(*) AS rows,
               count_if(single_cell_available) AS available_rows,
               count_if(NOT single_cell_available) AS null_rows,
               stddev_pop(single_cell_replication_probability) AS probability_sd
        FROM read_parquet('{typed_sql}') GROUP BY cancer_id ORDER BY cancer_id
    """).fetchdf()
    available_cancers = set(cancer_stats.loc[cancer_stats.available_rows.gt(0), "cancer_id"].astype(str))
    unavailable_cancers = set(cancer_stats.loc[cancer_stats.available_rows.eq(0), "cancer_id"].astype(str))
    variable_each_available = bool(
        cancer_stats.loc[cancer_stats.available_rows.gt(0), "probability_sd"].fillna(0).gt(0).all()
    )
    recorder.check("typed_exact_row_counts",
                   tuple(map(int, typed_stats[:3])) ==
                   (EXPECTED_TYPED_ROWS, EXPECTED_AVAILABLE_ROWS, EXPECTED_NULL_ROWS),
                   observed=tuple(map(int, typed_stats[:3])),
                   expected=(EXPECTED_TYPED_ROWS, EXPECTED_AVAILABLE_ROWS, EXPECTED_NULL_ROWS))
    recorder.check("typed_33_cancers", int(typed_stats[3]) == 33)
    recorder.check("typed_probability_semantics",
                   int(typed_stats[4]) == 0 and int(typed_stats[5]) == 0
                   and int(typed_stats[6]) == 0 and float(typed_stats[10]) > 0)
    recorder.check("typed_probability_varies_each_formal_cancer", variable_each_available)
    recorder.check("typed_does_not_change_primary", int(typed_stats[7]) == 0)
    recorder.check("typed_target_keys_unique_nonnull", typed_dup == 0 and typed_null_keys == 0,
                   observed={"duplicate_groups": typed_dup, "null_key_rows": typed_null_keys})
    recorder.check("typed_inside_candidate_universe", typed_extra == 0, observed=typed_extra, expected=0)
    recorder.check("typed_17_formal_cancers", available_cancers == qualified_cancers,
                   observed=sorted(available_cancers), expected=sorted(qualified_cancers))
    recorder.check("typed_16_unavailable_cancers",
                   len(unavailable_cancers) == EXPECTED_UNAVAILABLE_CANCERS
                   and available_cancers.isdisjoint(unavailable_cancers),
                   observed=sorted(unavailable_cancers))
    prediction_metadata = con.execute(f"""
        SELECT count(DISTINCT training_run_id), min(training_run_id),
               count(DISTINCT prediction_format), min(prediction_format),
               count(DISTINCT module_id), min(module_id)
        FROM read_parquet('{typed_sql}')
    """).fetchone()
    recorder.check("typed_current_run_only",
                   int(prediction_metadata[0]) == 1
                   and prediction_metadata[1] == expected_training_run_id
                   and int(prediction_metadata[2]) == 1
                   and prediction_metadata[3] == PREDICTION_FORMAT
                   and int(prediction_metadata[4]) == 1
                   and prediction_metadata[5] == "single_cell")

    association_files = sorted(required_input_paths["association"].glob("*.parquet"))
    association_schemas = {
        path.name: list(pd.read_parquet(path).columns) for path in association_files
    }
    donor_columns = {
        path: [name for name in columns if name in {"donor_id", "patient_id", "subject_id"}]
        for path, columns in association_schemas.items()
    }
    recorder.check("dataset_block_fallback_is_authoritative",
                   all(not columns for columns in donor_columns.values()),
                   observed=donor_columns, expected="no explicit donor ID in association artifacts")
    formal_dataset_sql = _sql_strings(sorted(qualified_datasets))
    formal_tier_sql = _sql_strings(sorted(FORMAL_TIERS))
    con.execute(f"""
        CREATE TEMP VIEW formal_exact AS
        SELECT trim(a.dataset_id) AS dataset_id,
               upper(trim(a.cancer_id)) AS cancer_id,
               CASE
                 WHEN coalesce(trim(a.analysis_context), '') = ''
                   OR lower(trim(a.analysis_context)) = lower(trim(a.cell_population))
                 THEN trim(a.cell_population)
                 ELSE trim(a.cell_population) || '::' || trim(a.analysis_context)
               END AS cell_type,
               trim(a.lncrna_id) AS lncrna_id,
               trim(a.pathway_id) AS pathway_id
        FROM read_parquet('{association_glob}', union_by_name=true) AS a
        INNER JOIN read_parquet('{candidate_sql}') AS c
          ON upper(trim(a.cancer_id)) = c.cancer_id
         AND trim(a.lncrna_id) = c.lncrna_id
         AND trim(a.pathway_id) = c.pathway_id
        WHERE trim(a.dataset_id) IN ({formal_dataset_sql})
          AND regexp_replace(lower(trim(a.source_tier)), '[^a-z0-9]+', '_', 'g') IN ({formal_tier_sql})
          AND try_cast(a.rho AS DOUBLE) IS NOT NULL
          AND try_cast(a.fdr AS DOUBLE) IS NOT NULL
          AND trim(a.cell_population) <> ''
    """)
    raw_stats = con.execute("""
        SELECT count(*) AS rows,
               count(DISTINCT (dataset_id, cell_type, lncrna_id, pathway_id)) AS unique_keys,
               count(DISTINCT dataset_id) AS datasets,
               count(DISTINCT cancer_id) AS cancers
        FROM formal_exact
    """).fetchone()
    raw_minus_typed = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT dataset_id, cell_type, lncrna_id, pathway_id FROM formal_exact
          EXCEPT
          SELECT dataset_id, cell_type, lncrna_id, pathway_id
          FROM read_parquet('{typed_sql}') WHERE single_cell_available
        )
    """).fetchone()[0])
    typed_minus_raw = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT dataset_id, cell_type, lncrna_id, pathway_id
          FROM read_parquet('{typed_sql}') WHERE single_cell_available
          EXCEPT
          SELECT dataset_id, cell_type, lncrna_id, pathway_id FROM formal_exact
        )
    """).fetchone()[0])
    recorder.check("raw_formal_exact_rows_rebuilt",
                   int(raw_stats[0]) == EXPECTED_AVAILABLE_ROWS
                   and int(raw_stats[1]) == EXPECTED_AVAILABLE_ROWS
                   and int(raw_stats[2]) == EXPECTED_FORMAL_CANCERS
                   and int(raw_stats[3]) == EXPECTED_FORMAL_CANCERS,
                   observed=tuple(map(int, raw_stats)))
    recorder.check("raw_to_typed_available_bidirectional_exact",
                   raw_minus_typed == 0 and typed_minus_raw == 0,
                   observed={"raw_minus_typed": raw_minus_typed,
                             "typed_minus_raw": typed_minus_raw})

    block_counts_frame = con.execute("""
        SELECT cancer_id, dataset_id, count(*) AS rows
        FROM formal_exact GROUP BY cancer_id, dataset_id ORDER BY cancer_id, dataset_id
    """).fetchdf()
    block_counts = {
        f"{row.cancer_id}|dataset:{row.dataset_id}": int(row.rows)
        for row in block_counts_frame.itertuples(index=False)
    }
    block_mapping = deterministic_block_folds(block_counts, base_seed)
    recorder.check("recomputed_fold_ids_cover_0_to_4", set(block_mapping.values()) == set(range(N_FOLDS)))
    fold_audit: list[dict[str, Any]] = []
    raw_lnc_ids = {_canonical_node(value, "lncrna") for value in
                   con.execute("SELECT DISTINCT lncrna_id FROM formal_exact").fetchnumpy()["lncrna_id"]}
    raw_pathway_ids = {_canonical_node(value, "pathway") for value in
                       con.execute("SELECT DISTINCT pathway_id FROM formal_exact").fetchnumpy()["pathway_id"]}
    for fold in range(N_FOLDS):
        split = split_blocks(block_mapping, fold)
        train_full = sum(block_counts[block] for block in split["train"])
        validation_full = sum(block_counts[block] for block in split["validation"])
        test_full = sum(block_counts[block] for block in split["test"])
        expected_train = min(train_full, int(config.get("max_train_rows", -1)))
        expected_validation = min(validation_full, int(config.get("max_validation_rows", -1)))
        record = checkpoint_audit[fold]
        lnc_missing = raw_lnc_ids - core_lnc_ids[fold]
        pathway_missing = raw_pathway_ids - core_pathway_ids[fold]
        recorder.check(f"fold::{fold}::core_covers_raw_targets",
                       not lnc_missing and not pathway_missing,
                       observed={"missing_lncrna": len(lnc_missing),
                                 "missing_pathway": len(pathway_missing)})
        recorder.check(f"fold::{fold}::selected_row_counts",
                       record["train_rows"] == expected_train
                       and record["validation_rows"] == expected_validation,
                       observed={"train": record["train_rows"],
                                 "validation": record["validation_rows"]},
                       expected={"train": expected_train,
                                 "validation": expected_validation})
        recorder.check(f"fold::{fold}::no_block_overlap",
                       not (split["train"] & split["validation"])
                       and not (split["train"] & split["test"])
                       and not (split["validation"] & split["test"]))
        fold_audit.append({
            "fold": fold,
            "test_blocks": sorted(split["test"]),
            "validation_blocks": sorted(split["validation"]),
            "train_blocks": sorted(split["train"]),
            "test_rows": test_full,
            "validation_rows_before_cap": validation_full,
            "train_rows_before_cap": train_full,
            "validation_rows_after_cap": expected_validation,
            "train_rows_after_cap": expected_train,
        })
    recorder.check("five_test_folds_partition_all_available_rows",
                   sum(row["test_rows"] for row in fold_audit) == EXPECTED_AVAILABLE_ROWS)

    exact_stats = con.execute(f"""
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
               count_if(single_cell_available) AS available_rows,
               count_if(NOT single_cell_available) AS unavailable_rows,
               count_if(NOT exact_pathway_only) AS nonexact_rows
        FROM read_parquet('{exact_sql}')
    """).fetchone()
    exact_extra = int(con.execute(f"""
        SELECT count(*) FROM (
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{exact_sql}')
          EXCEPT
          SELECT cancer_id, lncrna_id, pathway_id FROM read_parquet('{candidate_sql}')
        )
    """).fetchone()[0])
    aggregate_diff = int(con.execute(f"""
        WITH rebuilt AS (
          SELECT cancer_id, lncrna_id, pathway_id,
                 avg(single_cell_replication_probability) AS probability,
                 bool_or(single_cell_available) AS available,
                 first(single_cell_unavailable_reason ORDER BY dataset_id) AS reason,
                 count(DISTINCT dataset_id) AS n_targets
          FROM read_parquet('{typed_sql}') GROUP BY cancer_id, lncrna_id, pathway_id
        ), stored AS (
          SELECT cancer_id, lncrna_id, pathway_id,
                 single_cell_replication_probability AS probability,
                 single_cell_available AS available,
                 single_cell_unavailable_reason AS reason,
                 n_dataset_celltype_targets AS n_targets
          FROM read_parquet('{exact_sql}')
        )
        SELECT count(*) FROM rebuilt FULL OUTER JOIN stored
        USING (cancer_id, lncrna_id, pathway_id)
        WHERE rebuilt.cancer_id IS NULL OR stored.cancer_id IS NULL
           OR NOT (rebuilt.probability IS NOT DISTINCT FROM stored.probability)
           OR rebuilt.available != stored.available
           OR NOT (rebuilt.reason IS NOT DISTINCT FROM stored.reason)
           OR rebuilt.n_targets != stored.n_targets
    """).fetchone()[0])
    recorder.check("exact_aggregate_unique_inside_candidate",
                   int(exact_stats[0]) == int(exact_stats[1])
                   and int(exact_stats[4]) == 0 and exact_extra == 0,
                   observed={"stats": tuple(map(int, exact_stats)), "extra": exact_extra})
    recorder.check("exact_aggregate_independently_reproduced", aggregate_diff == 0,
                   observed=aggregate_diff, expected=0)
    adapter_stats = con.execute(f"""
        WITH sparse AS (
          SELECT cancer_id, lncrna_id, pathway_id,
                 single_cell_replication_probability AS probability,
                 single_cell_available AS available,
                 single_cell_unavailable_reason AS reason
          FROM read_parquet('{exact_sql}')
        ), adapted AS (
          SELECT c.cancer_id, c.lncrna_id, c.pathway_id,
                 s.probability,
                 coalesce(s.available, false) AS available,
                 CASE WHEN s.cancer_id IS NULL
                      THEN 'NO_SINGLE_CELL_EXACT_PAIR_MEASUREMENT'
                      ELSE s.reason END AS reason
          FROM read_parquet('{candidate_sql}') AS c
          LEFT JOIN sparse AS s USING (cancer_id, lncrna_id, pathway_id)
        )
        SELECT count(*) AS rows,
               count(DISTINCT (cancer_id, lncrna_id, pathway_id)) AS unique_keys,
               count_if(available) AS available_rows,
               count_if(NOT available) AS unavailable_rows,
               count_if(NOT available AND probability IS NOT NULL) AS bad_probability,
               count_if(NOT available AND (reason IS NULL OR trim(reason)='')) AS bad_reason
        FROM adapted
    """).fetchone()
    adapter_compatible = (
        int(adapter_stats[0]) == EXPECTED_CANDIDATE_ROWS
        and int(adapter_stats[1]) == EXPECTED_CANDIDATE_ROWS
        and int(adapter_stats[4]) == 0
        and int(adapter_stats[5]) == 0
    )
    recorder.check("full_universe_adapter_semantics", adapter_compatible,
                   observed=tuple(map(int, adapter_stats)))
    con.close()

    recorder.check("lineage_old_artifacts_false",
                   lineage.get("old_checkpoint_loaded") is False
                   and lineage.get("old_predictions_used_as_features") is False
                   and lineage.get("old_rankings_used_as_outputs") is False
                   and lineage.get("old_family_support_used") is False)
    recorder.check("lineage_counts_match",
                   int(lineage.get("prediction_rows", -1)) == EXPECTED_TYPED_ROWS
                   and int(lineage.get("available_rows", -1)) == EXPECTED_AVAILABLE_ROWS
                   and int(lineage.get("null_rows", -1)) == EXPECTED_NULL_ROWS)
    declared_output_hashes = {
        "typed_predictions": lineage.get("prediction_sha256"),
        "exact_aggregate": lineage.get("lnc_exact_pathway_sha256"),
        "lnc_celltype": lineage.get("lnc_celltype_sha256"),
        "activity": lineage.get("activity_sha256"),
        "figures": lineage.get("figure_manifest_sha256"),
    }
    output_hash_rows: list[dict[str, Any]] = []
    for name, declared_hash in declared_output_hashes.items():
        observed_hash = source_before[name]["sha256"]
        recorder.check(f"declared_output_hash::{name}", observed_hash == declared_hash,
                       observed=observed_hash, expected=declared_hash)
        output_hash_rows.append({"artifact": name, "path": source_before[name]["path"],
                                 "observed_sha256": observed_hash,
                                 "declared_sha256": declared_hash})

    source_after = _snapshot(source_paths)
    recorder.check("all_authoritative_sources_unchanged_during_audit", source_before == source_after)
    status = "PASS" if recorder.passed else "FAIL"
    report = {
        "format": AUDIT_FORMAT,
        "status": status,
        "fail_closed": True,
        "analysis_version": ANALYSIS_VERSION,
        "audit_id": "V32-SINGLE-CELL-FRESH-20260826-R1-INDEPENDENT-AUDIT",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "independence": {
            "training_invoked": False,
            "training_module_imported": False,
            "training_artifacts_modified": False,
            "remote_read_or_write": False,
            "deployment_performed": False,
            "validation_method": "INDEPENDENT_CONTENT_HASH_AND_EXACT_SET_RECONSTRUCTION",
        },
        "paths": {
            "repo_root": str(repo), "training_root": str(training),
            "input_root": str(inputs), "candidate_universe": str(candidates),
            "core_manifest": str(core_manifest_path),
        },
        "summary": {
            "candidate_rows": int(candidate_stats[0]),
            "typed_rows": int(typed_stats[0]),
            "available_rows": int(typed_stats[1]),
            "null_rows": int(typed_stats[2]),
            "formal_cancers": sorted(available_cancers),
            "typed_unavailable_cancers": sorted(unavailable_cancers),
            "exact_aggregate_rows": int(exact_stats[0]),
            "adapter_full_universe_rows": int(adapter_stats[0]),
            "adapter_available_rows": int(adapter_stats[2]),
            "adapter_unavailable_rows": int(adapter_stats[3]),
            "probability_min": float(typed_stats[8]),
            "probability_max": float(typed_stats[9]),
            "probability_sd": float(typed_stats[10]),
            "optimizer_steps_executed_lower_bound": total_optimizer_steps_lower_bound,
            "best_checkpoint_contributing_steps_lower_bound": total_best_checkpoint_steps_lower_bound,
        },
        "checkpoint_audit": checkpoint_audit,
        "core_audit": {
            "records": core_records,
            "parameter_composite_sha256": core_composite,
            "before_sha256": lineage.get("core_parameters_before_sha256"),
            "after_sha256": lineage.get("core_parameters_after_sha256"),
            "constant_lncrna_mask_source_proof": mask_source,
        },
        "fold_audit": {
            "seed": base_seed,
            "split_unit": "dataset_fallback_no_explicit_donor_id_in_association",
            "block_to_fold": dict(sorted(block_mapping.items())),
            "folds": fold_audit,
        },
        "prediction_audit": {
            "typed_target_duplicate_groups": typed_dup,
            "typed_target_null_key_rows": typed_null_keys,
            "typed_keys_outside_candidate": typed_extra,
            "raw_minus_typed_available": raw_minus_typed,
            "typed_available_minus_raw": typed_minus_raw,
            "available_probability_finite_bounded_and_variable": bool(
                int(typed_stats[4]) == 0 and float(typed_stats[10]) > 0
            ),
            "unavailable_is_null_with_reason": bool(
                int(typed_stats[5]) == 0 and int(typed_stats[6]) == 0
            ),
        },
        "aggregation_adapter_audit": {
            "typed_to_exact_aggregate_difference_rows": aggregate_diff,
            "exact_rows_outside_candidate": exact_extra,
            "adapter_stats": {
                "rows": int(adapter_stats[0]), "unique_keys": int(adapter_stats[1]),
                "available_rows": int(adapter_stats[2]),
                "unavailable_rows": int(adapter_stats[3]),
                "unavailable_nonnull_probability_rows": int(adapter_stats[4]),
                "unavailable_missing_reason_rows": int(adapter_stats[5]),
            },
            "full_universe_materialization_compatible": adapter_compatible,
        },
        "hash_evidence": {
            "training_code_files": training_code_files,
            "recomputed_training_code_sha256": recomputed_training_code_sha,
            "declared_training_code_sha256": lineage.get("code_sha256"),
            "input_hashes": input_hash_rows,
            "output_hashes": output_hash_rows,
            "authoritative_snapshot_before": source_before,
            "authoritative_snapshot_after": source_after,
        },
        "release_decision": {
            "formal_single_cell_head_accepted": status == "PASS",
            "full_universe_fusion_adapter_materialization": (
                "AUTHORIZED" if status == "PASS" and adapter_compatible else "PROHIBITED"
            ),
            "primary_ranking_may_be_changed": False,
        },
        "checks": recorder.checks,
        "failed_checks": [row["check"] for row in recorder.checks if row["status"] != "PASS"],
    }
    report_path = output / "INDEPENDENT_AUDIT.json"
    _atomic_json(report_path, report)
    report_hash = file_sha256(report_path)
    audit_code_path = Path(__file__).resolve()
    audit_runner_path = repo / "scripts" / "audit_v32_single_cell_fresh_independent.py"
    audit_test_path = repo / "tests" / "test_v32_single_cell_fresh_independent_audit.py"
    audit_code_hashes = {
        "single_cell_independent_audit.py": file_sha256(audit_code_path),
        "audit_v32_single_cell_fresh_independent.py": (
            file_sha256(audit_runner_path) if audit_runner_path.is_file() else None
        ),
        "test_v32_single_cell_fresh_independent_audit.py": (
            file_sha256(audit_test_path) if audit_test_path.is_file() else None
        ),
    }
    binding = {
        "format": BINDING_FORMAT,
        "status": status,
        "fail_closed": True,
        "analysis_version": ANALYSIS_VERSION,
        "audited_training_run_id": expected_training_run_id,
        "audit_report_path": str(report_path),
        "audit_report_sha256": report_hash,
        "audit_code_hashes": audit_code_hashes,
        "training_artifact_sha256": artifact_sha256(training),
        "candidate_universe_sha256": artifact_sha256(candidates),
        "single_cell_input_root_sha256": artifact_sha256(inputs),
        "core_manifest_sha256": file_sha256(core_manifest_path),
        "checkpoint_final_parameter_sha256": {
            str(row["fold"]): row["final_parameter_sha256"] for row in checkpoint_audit
        },
        "core_parameter_composite_sha256": core_composite,
        "release_decision": report["release_decision"],
    }
    binding_path = output / "INDEPENDENT_AUDIT_BINDING.json"
    _atomic_json(binding_path, binding)
    binding_hash = file_sha256(binding_path)
    completion = {
        "status": status,
        "audit_report_path": str(report_path),
        "audit_report_sha256": report_hash,
        "binding_path": str(binding_path),
        "binding_sha256": binding_hash,
        "all_checks_passed": status == "PASS",
        "failed_checks": report["failed_checks"],
        "full_universe_fusion_adapter_materialization":
            report["release_decision"]["full_universe_fusion_adapter_materialization"],
    }
    _atomic_json(output / ("SUCCESS.json" if status == "PASS" else "FAILURE.json"), completion)
    if status != "PASS":
        raise IndependentAuditError(f"independent audit failed: {report['failed_checks']}")
    return completion


__all__ = [
    "AUDIT_FORMAT", "BINDING_FORMAT", "IndependentAuditError",
    "artifact_sha256", "canonical_sha256", "deterministic_block_folds",
    "file_sha256", "independently_rebuild_initial_state_sha256",
    "run_independent_audit", "split_blocks", "state_dict_sha256",
    "verify_report_binding",
]
