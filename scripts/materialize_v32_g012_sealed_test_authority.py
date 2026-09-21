#!/usr/bin/env python3
"""Build the fresh G0/G1/G2 sealed-test authority after winner lock.

The training preparer intentionally cannot serialize test labels.  This
post-lock materializer first validates the immutable validation-only winner,
then reconstructs the fold-train L1 baseline, verifies its train logits
against the guarded prepared payload, and only then opens each patient-heldout
test split.  Its output is the exact input contract consumed by
``materialize_v32_winner_locked_sealed_test.py``.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON authority is not a mapping: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _stored_logits(payload: Mapping[str, Any], split: str) -> np.ndarray:
    batches = payload.get(f"{split}_batches")
    if not isinstance(batches, (list, tuple)) or not batches:
        raise RuntimeError(f"Prepared payload lacks {split}_batches")
    return np.concatenate(
        [_numpy(batch["base_logit"]).reshape(-1) for batch in batches]
    ).astype(np.float64, copy=False)


def _test_batches(
    frame: pd.DataFrame,
    logits: np.ndarray,
    bundle: Any,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch

    maps = bundle.node_maps
    rows: list[dict[str, Any]] = []
    for start in range(0, len(frame), batch_size):
        sub = frame.iloc[start : start + batch_size]
        effect = (
            pd.to_numeric(sub.discovery_effect, errors="coerce")
            .fillna(0)
            .to_numpy(float)
        )
        context = np.column_stack(
            [
                sub.shared_or_local_scope.astype(str).eq("shared").to_numpy(float),
                pd.to_numeric(sub.detection_rate, errors="coerce")
                .fillna(0)
                .to_numpy(float),
                np.clip(np.abs(effect), 0, 5),
                pd.to_numeric(sub.detected_cancers, errors="coerce")
                .fillna(0)
                .to_numpy(float)
                / 33.0,
            ]
        )
        rows.append(
            {
                "candidate_batch": {
                    "l": torch.tensor(
                        [maps["lncRNA"][str(value)] for value in sub.lncrna_id],
                        dtype=torch.long,
                    ),
                    "p": torch.tensor(
                        [maps["pathway"][str(value)] for value in sub.pathway_id],
                        dtype=torch.long,
                    ),
                    "c": torch.tensor(
                        [maps["cancer"][str(value)] for value in sub.cancer_id],
                        dtype=torch.long,
                    ),
                },
                "base_logit": torch.tensor(
                    logits[start : start + len(sub)], dtype=torch.float32
                ),
                "conservation_context": torch.tensor(context, dtype=torch.float32),
                "proxy_label": torch.tensor(
                    sub.proxy_label.to_numpy(float), dtype=torch.float32
                ),
                "direction_label": torch.tensor(
                    sub.replication_direction_label.to_numpy(float),
                    dtype=torch.float32,
                ),
                "direction_available": torch.tensor(
                    sub.replication_direction_available.to_numpy(bool),
                    dtype=torch.bool,
                ),
                "graph_available": torch.ones(len(sub), dtype=torch.bool),
            }
        )
    if not rows:
        raise RuntimeError("Sealed-test batching produced no rows")
    return rows


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo_root).resolve(strict=True)
    scripts_root = repo / "scripts"
    for value in (str(repo), str(scripts_root)):
        if value not in sys.path:
            sys.path.insert(0, value)

    from cc_hhgt.v32.baselines import build_safe_hashed_design
    from cc_hhgt.v32.patient_fold_authority import (
        validate_frozen_v32_patient_fold_binding,
    )
    from cc_hhgt.v32.patient_folds import assign_outer_split
    from cc_hhgt.v32.sealed_test_inference import (
        SEALED_TEST_MANIFEST_FORMAT,
        SEALED_TEST_PAYLOAD_FORMAT,
        _validate_winner_declaration,
    )
    from prepare_v32_formal import (
        attach_replication_labels,
        load_activity,
        load_expression_cancer,
        sampled_associations,
    )
    from sklearn.linear_model import LogisticRegression

    winner_path = Path(args.winner_declaration).resolve(strict=True)
    observed_winner_sha = _sha256(winner_path)
    expected_winner_sha = str(args.winner_declaration_sha256).lower()
    if observed_winner_sha != expected_winner_sha:
        raise RuntimeError("Winner declaration SHA256 drift")
    winner = _json(winner_path)
    variant = str(winner.get("graph_variant", ""))
    prepared_records, _ = _validate_winner_declaration(
        winner, expected_graph_variant=variant
    )

    prepared_parent = Path(args.prepared_parent).resolve(strict=True)
    prepared_root = prepared_parent / variant
    if not prepared_root.is_dir():
        raise RuntimeError(f"Winner prepared root is missing: {prepared_root}")
    candidate_record = winner.get("candidate_authority")
    if not isinstance(candidate_record, Mapping):
        raise RuntimeError("Winner declaration lacks candidate_authority")
    candidate_path = Path(str(candidate_record.get("path"))).resolve(strict=True)
    candidate_sha = _sha256(candidate_path)
    if candidate_sha != winner.get("candidate_authority_sha256"):
        raise RuntimeError("Candidate authority SHA256 drift")
    candidates = pd.read_parquet(candidate_path)
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    if sorted(set(keys) - set(candidates)):
        raise RuntimeError("Candidate authority lacks exact-pair keys")
    candidates = candidates.sort_values(keys, kind="stable").reset_index(drop=True)

    patient_folds_path = Path(args.patient_folds).resolve(strict=True)
    authority_receipt_path = Path(
        args.patient_fold_authority_receipt
    ).resolve(strict=True)
    authority = validate_frozen_v32_patient_fold_binding(
        patient_folds_path, authority_receipt_path
    )
    fold_map = pd.read_csv(
        patient_folds_path,
        sep="\t",
        dtype={"cancer_id": str, "sample_id": str, "patient_id": str},
    )

    hierarchy = pd.read_parquet(prepared_parent / "PATHWAY_HIERARCHY.parquet")
    scope = pd.read_parquet(prepared_parent / "FORMAL_LNCRNA_SCOPE.parquet")

    # No expression or pathway-activity file is opened before the immutable
    # winner and every binding above have passed validation.
    activity = load_activity(args.formal_activity_root)
    cancers = sorted(activity.cancer_id.astype(str).unique())
    activity_by = {
        cancer: activity.loc[activity.cancer_id.astype(str).eq(cancer)].copy()
        for cancer in cancers
    }
    expr_by = {
        cancer: load_expression_cancer(cancer, args.formal_expression_root)
        for cancer in cancers
    }

    output = Path(args.output_root).resolve()
    staging = output.with_name(output.name + f".partial.{os.getpid()}")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Refusing sealed-test output reuse: {output}")
    staging.mkdir(parents=True)
    fold_records: list[dict[str, Any]] = []
    try:
        for fold in range(5):
            prepared_path = Path(str(prepared_records[fold]["path"])).resolve(
                strict=True
            )
            prepared_sha = _sha256(prepared_path)
            if prepared_sha != prepared_records[fold]["sha256"]:
                raise RuntimeError(f"Prepared fold {fold} SHA256 drift")
            import torch

            prepared = torch.load(
                prepared_path, map_location="cpu", weights_only=False
            )
            if (
                int(prepared.get("patient_fold", -1)) != fold
                or prepared.get("formal_graph_variant") != variant
            ):
                raise RuntimeError(f"Prepared fold {fold} identity drift")
            split_manifest = assign_outer_split(
                fold_map, fold, n_folds=5, validation_offset=1
            )
            train = sampled_associations(
                expr_by,
                activity_by,
                candidates,
                split_manifest,
                "train",
                hierarchy,
                scope,
            )
            heldout = sampled_associations(
                expr_by,
                activity_by,
                candidates,
                split_manifest,
                "test",
                hierarchy,
                scope,
            )
            sealed_frame = attach_replication_labels(train, heldout)
            for frame in (train, sealed_frame):
                frame["cross_cancer_support_frequency"] = (
                    frame.lncrna_id.map(
                        scope.groupby("lncrna_id").detected_cancers.max()
                    )
                    .fillna(0)
                    .to_numpy(float)
                    / 33.0
                )
                frame["cross_cancer_direction_consistency"] = 0.0
                frame["cross_cancer_i2"] = 0.0
            x_train = build_safe_hashed_design(train, n_features=4096)
            y_train = train.proxy_label.to_numpy(int)
            weight = np.where(
                train.label_class.astype(str).eq("weak_positive"),
                0.35,
                np.where(y_train == 0, 0.12, 1.0),
            )
            baseline = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=0.1,
                max_iter=250,
                random_state=int(args.seed),
            )
            baseline.fit(x_train, y_train, sample_weight=weight)
            reconstructed_train = baseline.decision_function(x_train).astype(
                np.float64, copy=False
            )
            guarded_train = _stored_logits(prepared, "train")
            if guarded_train.shape != reconstructed_train.shape:
                raise RuntimeError(f"Fold {fold} L1 train-logit shape drift")
            max_error = float(
                np.max(np.abs(guarded_train - reconstructed_train), initial=0.0)
            )
            if max_error > float(args.logit_tolerance):
                raise RuntimeError(
                    f"Fold {fold} L1 reconstruction drift: max_abs={max_error}"
                )
            x_test = build_safe_hashed_design(sealed_frame, n_features=4096)
            test_logits = baseline.decision_function(x_test)
            batches = _test_batches(
                sealed_frame,
                test_logits,
                prepared["bundle"],
                batch_size=int(args.batch_size),
            )
            payload = {
                "sealed_test_format": SEALED_TEST_PAYLOAD_FORMAT,
                "patient_fold": fold,
                "formal_graph_variant": variant,
                "candidate_authority_sha256": candidate_sha,
                "training_payload_sha256": prepared_sha,
                "contains_test_labels": True,
                "contains_train_or_validation_batches": False,
                "winner_lock_required_before_deserialization": True,
                "accessed_before_winner_lock": False,
                "patient_fold_authority": prepared["patient_fold_authority"],
                "test_batches": batches,
                "l1_reconstruction_max_abs_error": max_error,
            }
            payload_path = staging / f"SEALED_TEST_FOLD_{fold}.pt"
            payload_tmp = payload_path.with_suffix(".pt.partial")
            torch.save(payload, payload_tmp)
            payload_tmp.replace(payload_path)
            test_keys = fold_map.loc[
                pd.to_numeric(fold_map.patient_fold_id).eq(fold)
            ].sort_values(
                ["cancer_id", "patient_id", "sample_id"], kind="stable"
            )
            keys_path = staging / f"TEST_SAMPLE_PATIENT_KEYS_{fold}.tsv"
            test_keys.to_csv(
                keys_path, sep="\t", index=False, lineterminator="\n"
            )
            fold_records.append(
                {
                    "patient_fold": fold,
                    "training_payload_sha256": prepared_sha,
                    "test_patient_keys": {
                        "path": str(output / keys_path.name),
                        "sha256": _sha256(keys_path),
                    },
                    "test_payload": {
                        "path": str(output / payload_path.name),
                        "sha256": _sha256(payload_path),
                    },
                    "l1_reconstruction_max_abs_error": max_error,
                }
            )
            del prepared, train, heldout, sealed_frame, batches, x_train, x_test
            gc.collect()

        manifest = {
            "format": SEALED_TEST_MANIFEST_FORMAT,
            "status": "PASS_SEALED_TEST_AUTHORITY_UNOPENED",
            "graph_variant": variant,
            "winner_lock_required_before_payload_deserialization": True,
            "payloads_opened_before_winner_lock": False,
            "test_labels_absent_from_training_payload": True,
            "old_fold_outputs_used": False,
            "pair_fold_seed": int(winner["pair_fold_seed"]),
            "sample_patient_fold_map_sha256": authority["manifest_sha256"],
            "authority_receipt_sha256": authority["receipt_sha256"],
            "winner_declaration_sha256": observed_winner_sha,
            "candidate_authority_sha256": candidate_sha,
            "candidate_authority": {
                "path": str(candidate_path),
                "sha256": candidate_sha,
            },
            "folds": fold_records,
        }
        _atomic_json(staging / "SEALED_TEST_MANIFEST.json", manifest)
        for path in staging.iterdir():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "PASS_G012_SEALED_TEST_AUTHORITY",
        "graph_variant": variant,
        "winner_declaration_sha256": observed_winner_sha,
        "manifest": {
            "path": str(output / "SEALED_TEST_MANIFEST.json"),
            "sha256": _sha256(output / "SEALED_TEST_MANIFEST.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--prepared-parent", required=True)
    parser.add_argument("--winner-declaration", required=True)
    parser.add_argument("--winner-declaration-sha256", required=True)
    parser.add_argument("--patient-folds", required=True)
    parser.add_argument("--patient-fold-authority-receipt", required=True)
    parser.add_argument("--formal-expression-root", required=True)
    parser.add_argument("--formal-activity-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--logit-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    result = materialize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
