"""Fail-closed label-contract validation for formal Pathway evaluation.

The historical candidate column ``label`` is intentionally not accepted as a
model target here.  Callers must name the estimand explicitly.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


class LabelContractError(RuntimeError):
    """Raised before compute starts when a label/manifest contract is ambiguous."""


REQUIRED_LABEL_COLUMNS = {
    "label_class",
    "association_proxy_label",
    "strong_evidence_label",
}
FORBIDDEN_IMPLICIT_TARGETS = {"label", "target", "y", "proxy_label"}
REQUIRED_HASH_ROLES = (
    "training_target_contract_sha256",
    "evaluation_target_contract_sha256",
    "baseline_target_contract_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_label_contract(path: str | Path, sidecar: str | Path | None = None) -> dict[str, Any]:
    contract_path = Path(path)
    sidecar_path = Path(sidecar) if sidecar else contract_path.with_suffix(".sha256")
    if not contract_path.is_file() or not sidecar_path.is_file():
        raise LabelContractError(f"Label contract or SHA256 sidecar is missing: {contract_path}")
    actual = sha256_file(contract_path)
    expected = sidecar_path.read_text(encoding="utf-8").split()[0].strip().lower()
    if actual != expected:
        raise LabelContractError(
            f"Label contract hash mismatch: expected={expected}, actual={actual}, path={contract_path}"
        )
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LabelContractError(f"Label contract is not a mapping: {contract_path}")
    payload["_contract_path"] = str(contract_path.resolve())
    payload["_contract_sha256"] = actual
    validate_label_contract(payload)
    return payload


def validate_label_contract(contract: Mapping[str, Any]) -> None:
    required = {"contract_id", "contract_version", "status", "task_id", "unit", "target", "candidate", "split"}
    missing = required - set(contract)
    if missing:
        raise LabelContractError(f"Label contract lacks required keys: {sorted(missing)}")
    if contract["status"] != "frozen":
        raise LabelContractError(f"Formal label contract must be frozen, got {contract['status']!r}")
    target = contract["target"]
    if not isinstance(target, Mapping) or target.get("column") not in {
        "association_proxy_label", "strong_evidence_label"
    }:
        raise LabelContractError(f"Unsupported or ambiguous target definition: {target!r}")
    unit = list(contract["unit"])
    allowed_units = {
        ("cancer_id", "lncrna_id", "pathway_family_id"),
        ("cancer_id", "lncrna_id", "pathway_id"),
    }
    if tuple(unit) not in allowed_units:
        raise LabelContractError(f"Unexpected Pathway unit: {unit}")
    candidate = contract["candidate"]
    if int(candidate.get("size_per_cancer", -1)) <= 0:
        raise LabelContractError("Formal Pathway candidate size must be a positive frozen value")
    if candidate.get("sampling") != "identity-only deterministic" or int(candidate.get("seed", -1)) != 20260726:
        raise LabelContractError(f"Candidate sampling contract drift: {candidate!r}")
    split = contract["split"]
    if int(split.get("seed", -1)) != 20260726:
        raise LabelContractError(f"Split seed drift: {split!r}")
    fractions = [float(split.get(k, -1)) for k in ("train_fraction", "validation_fraction", "test_fraction")]
    if fractions != [0.70, 0.15, 0.15]:
        raise LabelContractError(f"Split fractions drift: {fractions}")


def attach_explicit_pathway_labels(frame: pd.DataFrame) -> pd.DataFrame:
    if "label_class" not in frame:
        raise LabelContractError("Cannot attach explicit Pathway labels without label_class")
    out = frame.copy()
    label_class = out.label_class.astype(str)
    allowed = {"strong_positive", "weak_positive", "unlabeled"}
    unexpected = sorted(set(label_class.unique()) - allowed)
    if unexpected:
        raise LabelContractError(f"Unexpected Pathway label_class values: {unexpected}")
    out["association_proxy_label"] = label_class.isin(
        ["strong_positive", "weak_positive"]
    ).astype(np.int8)
    out["strong_evidence_label"] = label_class.eq("strong_positive").astype(np.int8)
    return out


def stable_stratified_split(
    candidate_id: pd.Series, target: pd.Series | np.ndarray, seed: int = 20260726
) -> np.ndarray:
    ids = candidate_id.astype(str).reset_index(drop=True)
    labels = np.asarray(target, dtype=np.int8)
    if len(ids) != len(labels) or not set(np.unique(labels)).issubset({0, 1}):
        raise LabelContractError("Split inputs must be aligned binary labels")
    result = np.empty(len(ids), dtype=object)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(f"{seed}|{ids.iat[index]}".encode()).digest(),
        )
        n = len(ordered)
        train_n, validation_n = int(n * 0.70), int(n * 0.15)
        result[ordered[:train_n]] = "train"
        result[ordered[train_n:train_n + validation_n]] = "validation"
        result[ordered[train_n + validation_n:]] = "test"
    return result.astype(str)


def _framed_hash(rows: pd.DataFrame, columns: list[str], sort_columns: list[str]) -> str:
    missing = set(columns) - set(rows)
    if missing:
        raise LabelContractError(f"Cannot hash manifest; missing columns: {sorted(missing)}")
    ordered = rows.sort_values(sort_columns, kind="stable")[columns]
    digest = hashlib.sha256()
    digest.update(("\t".join(columns) + "\n").encode())
    for row in ordered.itertuples(index=False, name=None):
        for value in row:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def candidate_manifest_sha256(frame: pd.DataFrame) -> str:
    target_column = "pathway_id" if "pathway_id" in frame else "pathway_family_id"
    return _framed_hash(
        frame,
        ["cancer_id", "lncrna_id", target_column, "candidate_id"],
        ["candidate_id"],
    )


def candidate_order_sha256(frame: pd.DataFrame) -> str:
    return _framed_hash(
        frame.assign(_order=np.arange(len(frame))),
        ["_order", "candidate_id"],
        ["_order"],
    )


def split_manifest_sha256(frame: pd.DataFrame) -> str:
    return _framed_hash(frame, ["candidate_id", "split"], ["candidate_id"])


def validate_candidate_manifest(
    frame: pd.DataFrame,
    contract: Mapping[str, Any],
    expected_contract_sha256: str | None = None,
) -> dict[str, str]:
    validate_label_contract(contract)
    target_column = list(contract["unit"])[-1]
    required = {
        "cancer_id", "lncrna_id", target_column, "candidate_id",
        "candidate_sampling_seed", "candidate_sampling_policy", "label_contract_version",
        "label_contract_sha256", *REQUIRED_LABEL_COLUMNS,
    }
    missing = required - set(frame)
    if missing:
        raise LabelContractError(f"Candidate label manifest lacks columns: {sorted(missing)}")
    if frame.empty or frame.duplicated(["cancer_id", "candidate_id"]).any():
        raise LabelContractError("Candidate label manifest is empty or contains duplicate identities")
    expected_size = int(contract["candidate"]["size_per_cancer"])
    sizes = frame.groupby("cancer_id", observed=True).size()
    if not sizes.eq(expected_size).all():
        raise LabelContractError(f"Candidate size drift: {sizes.loc[~sizes.eq(expected_size)].to_dict()}")
    if set(pd.to_numeric(frame.candidate_sampling_seed, errors="raise").astype(int)) != {20260726}:
        raise LabelContractError("Candidate sampling seed drift")
    if set(frame.candidate_sampling_policy.astype(str)) != {"identity-only deterministic"}:
        raise LabelContractError("Candidate sampling is not identity-only deterministic")
    explicit = attach_explicit_pathway_labels(frame[["label_class"]])
    for column in ("association_proxy_label", "strong_evidence_label"):
        observed = pd.to_numeric(frame[column], errors="raise").astype(np.int8).reset_index(drop=True)
        if not observed.equals(explicit[column].reset_index(drop=True)):
            raise LabelContractError(f"Row-level {column} mapping is inconsistent with label_class")
    contract_hash = expected_contract_sha256 or str(contract.get("_contract_sha256", ""))
    if not contract_hash or set(frame.label_contract_sha256.astype(str)) != {contract_hash}:
        raise LabelContractError("Candidate label contract hash mismatch")
    if set(frame.label_contract_version.astype(str)) != {str(contract["contract_id"])}:
        raise LabelContractError("Candidate label contract version mismatch")
    hashes = {}
    for cancer, group in frame.groupby("cancer_id", observed=True, sort=True):
        observed_hashes = set(group.candidate_manifest_sha256.astype(str)) if "candidate_manifest_sha256" in group else set()
        actual = candidate_manifest_sha256(group)
        if observed_hashes != {actual}:
            raise LabelContractError(f"Candidate manifest hash mismatch for {cancer}")
        hashes[str(cancer)] = actual
    return hashes


def validate_split_manifest(frame: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, str]:
    required = {"cancer_id", "candidate_id", "association_proxy_label", "split", "split_seed", "split_manifest_sha256"}
    missing = required - set(frame)
    if missing:
        raise LabelContractError(f"Split manifest lacks columns: {sorted(missing)}")
    if set(pd.to_numeric(frame.split_seed, errors="raise").astype(int)) != {20260726}:
        raise LabelContractError("Split seed drift")
    hashes = {}
    for cancer, group in frame.groupby("cancer_id", observed=True, sort=True):
        expected = stable_stratified_split(
            group.candidate_id, group.association_proxy_label, seed=int(contract["split"]["seed"])
        )
        observed = group.split.astype(str).to_numpy()
        if not np.array_equal(expected, observed):
            raise LabelContractError(f"PATHWAY_PROXY_V1 split assignment mismatch for {cancer}")
        actual_hash = split_manifest_sha256(group)
        if set(group.split_manifest_sha256.astype(str)) != {actual_hash}:
            raise LabelContractError(f"Split manifest hash mismatch for {cancer}")
        hashes[str(cancer)] = actual_hash
    return hashes


def validate_contract_hashes(hashes: Mapping[str, str]) -> str:
    missing = set(REQUIRED_HASH_ROLES) - set(hashes)
    if missing:
        raise LabelContractError(f"Missing target-contract hashes: {sorted(missing)}")
    values = {str(hashes[key]).lower() for key in REQUIRED_HASH_ROLES}
    if len(values) != 1 or "" in values:
        raise LabelContractError(
            "training/evaluation/baseline label-contract hashes are not identical; DO_NOT_START_GPU"
        )
    return values.pop()


def validate_model_target(
    frame: pd.DataFrame,
    target_column: str,
    contract: Mapping[str, Any],
    role_hashes: Mapping[str, str],
) -> pd.Series:
    validate_label_contract(contract)
    if target_column in FORBIDDEN_IMPLICIT_TARGETS:
        raise LabelContractError(f"Ambiguous model target is forbidden: {target_column}")
    expected = str(contract["target"]["column"])
    if target_column != expected:
        raise LabelContractError(f"Model target {target_column!r} does not match contract target {expected!r}")
    if target_column not in frame:
        raise LabelContractError(f"Model target column is missing: {target_column}")
    validate_contract_hashes(role_hashes)
    labels = pd.to_numeric(frame[target_column], errors="raise").astype(np.int8)
    if not set(labels.unique()).issubset({0, 1}):
        raise LabelContractError(f"Model target is not binary: {sorted(labels.unique())}")
    return labels


def validate_positive_counts(
    frame: pd.DataFrame,
    cancers: tuple[str, ...] = ("HNSC", "LGG", "UCEC"),
    minimums: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    minimums = dict(minimums or {"train": 500, "validation": 100, "test": 100})
    required = {"cancer_id", "association_proxy_label", "split"}
    missing = required - set(frame)
    if missing:
        raise LabelContractError(f"Positive-count preflight lacks columns: {sorted(missing)}")
    rows = []
    for cancer in cancers:
        group = frame.loc[frame.cancer_id.astype(str).eq(cancer)]
        if group.empty:
            raise LabelContractError(f"Positive-count preflight cancer is missing: {cancer}")
        row: dict[str, Any] = {"cancer": cancer, "N_candidates": len(group)}
        labels = pd.to_numeric(group.association_proxy_label, errors="raise").astype(np.int8)
        row["prevalence"] = float(labels.mean())
        passed = True
        for split, threshold in minimums.items():
            count = int(labels.loc[group.split.astype(str).eq(split)].sum())
            row[f"{split}_pos"] = count
            passed &= count >= int(threshold)
        row["gate"] = "PASS" if passed else "FAIL"
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.gate.eq("PASS").all():
        raise LabelContractError(
            f"Positive-count gate failed; DO_NOT_START_GPU: {result.to_dict('records')}"
        )
    return result
