"""Bridge completed partition training and actual memmap reassembly, without fit.

A resource-stage SUCCESS or a standalone reassembly is not a trained expert.
The bridge requires both producer receipts and replays the bounded reassembler.
It verifies serialization/split contracts, not model forwards or study ancestry.
"""
from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
import socket
import tempfile

import pandas as pd

from .current_core_export import sha256


def compare_parquet(actual, replay):
    """Compare in bounded batches, independent of original row-group sizes."""
    import pyarrow.parquet as pq
    a, b = pq.ParquetFile(actual), pq.ParquetFile(replay)
    if a.schema_arrow.remove_metadata() != b.schema_arrow.remove_metadata():
        raise ValueError("Partitioned reassembly schema differs from numerical replay")
    for left, right in zip_longest(a.iter_batches(batch_size=65536), b.iter_batches(batch_size=65536)):
        if left is None or right is None:
            raise ValueError("Partitioned reassembly rows differ from numerical replay")
        try:
            pd.testing.assert_frame_equal(left.to_pandas(), right.to_pandas(), check_exact=True)
        except AssertionError as error:
            raise ValueError("Partitioned reassembly differs from numerical replay") from error


def load_partitioned(sources, manifest, native_path, native, patient_path, *, manifest_root):
    """Return the existing genomic adapter contract only after checking both stages."""
    if socket.gethostname() != "149":
        raise RuntimeError("Partitioned genomic replay requires host 149")
    from .genomic_partition_training import TRAINING_FORMAT, SPLIT_POLICY_FORMAT, _patient_split_trace
    from .genomic_resource_staging import REASSEMBLY_FORMAT, _load_stage, reassemble_genomic_memmap_outputs
    from .genomic_training import _split_patients, normalise_patient_folds
    from .evidence_rule_sources import read_table
    root = native_path.parent
    if (native.get("format") != TRAINING_FORMAT
            or native.get("status") != "TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED"
            or native.get("checkpoint_count") != 10 or native.get("release_ready") is not False
            or native.get("formal_genomic_oof_success_emitted") is not False):
        raise ValueError("Partitioned training is not complete; resource staging is not training")
    def document(path, digest, base=root):
        return sources.document({"path": path, "sha256": digest}, base)
    lineage_path, original = document(native["lineage_path"], native["lineage_sha256"])
    if (original.get("format") != TRAINING_FORMAT
            or original.get("status") != "TRAINING_COMPLETE_PREDICTIONS_SEALED_NOT_RELEASED"
            or original.get("training_run_id") != manifest["native_run_id"]
            or original.get("patient_folds") != 5 or original.get("modalities") != ["mutation", "cnv"]
            or original.get("checkpoint_count") != 10 or original.get("global_cross_cancer_heads") is not True
            or not original.get("core_parameters_before_sha256")
            or original["core_parameters_before_sha256"] != original.get("core_parameters_after_sha256")
            or original.get("core_manifest_sha256") != manifest["core_manifest"]["sha256"]):
        raise ValueError("Partitioned lineage run, folds or frozen core differ")
    records = original.get("completion_records", {})
    names = {"INPUT_LINEAGE_AUDIT.json", "SPLIT_POLICY.json", "CHECKPOINT_MANIFEST.json", "RESOURCE_AUDIT.json"}
    if set(records) != names:
        raise ValueError("Partitioned training lacks hash-bound completion records")
    completed = {name: sources.document(item, lineage_path.parent)[1] for name, item in records.items()}
    if (records["CHECKPOINT_MANIFEST.json"]["sha256"] != native.get("checkpoint_manifest_sha256")
            or completed["RESOURCE_AUDIT.json"].get("status") != "RESOURCE_BUDGET_PASS"):
        raise ValueError("Partitioned checkpoint/resource completion differs")
    source_audit = completed["INPUT_LINEAGE_AUDIT.json"]
    if source_audit.get("status") != "PASS" or not source_audit.get("artifacts"):
        raise ValueError("Partitioned training input audit is incomplete")
    inputs = []
    for item in source_audit["artifacts"]:
        if item.get("status") != "PASS":
            raise ValueError("Partitioned training input artifact did not pass")
        path = sources.bind(item, root)
        inputs.append({**item, "path": str(path)})
    if not {manifest[k]["sha256"] for k in ("candidate_authority", "patient_folds")} <= {r["sha256"] for r in inputs}:
        raise ValueError("Partitioned candidate/patient inputs differ")
    # Recompute train/validation/test identities from the actual patient map.
    patients = normalise_patient_folds(read_table(patient_path))
    traces = [_patient_split_trace({s: _split_patients(patients, fold, s) for s in ("train", "validation", "test")}, fold, modality)
              for fold in range(5) for modality in ("mutation", "cnv")]
    policy = completed["SPLIT_POLICY.json"]
    expected_policy = {"format": SPLIT_POLICY_FORMAT,
        "outer_test_policy": "PREDICTION_ONLY_AFTER_VALIDATION_CHECKPOINT_SELECTION",
        "scaling_fit": "TRAIN_PATIENTS_ONLY", "optimizer_fit": "TRAIN_PATIENTS_ONLY",
        "early_stopping": "VALIDATION_PATIENTS_ONLY", "checkpoint_choice": "VALIDATION_PATIENTS_ONLY",
        "test_keys_or_labels_enter_scaling_early_stop_or_checkpoint": False, "traces": traces}
    if policy != expected_policy or original.get("split_policy") != policy:
        raise ValueError("Partitioned split policy differs from actual patient folds")
    checkpoints = completed["CHECKPOINT_MANIFEST.json"]
    cp_records = checkpoints.get("records", [])
    if (len(cp_records) != 10 or {(r["modality"], r["patient_fold"]) for r in cp_records}
            != {(m, f) for m in ("mutation", "cnv") for f in range(5)}):
        raise ValueError("Partitioned training requires ten distinct native heads")
    import torch
    for item in cp_records:
        if (item.get("checkpoint_selection_split") != "validation" or item.get("scaler_fit_split") != "train"
                or item.get("test_rows_consumed_by_fit") != 0 or item.get("source_checkpoint_sha256") is not None):
            raise ValueError("Partitioned checkpoint selection/fit policy differs")
        checkpoint = torch.load(sources.bind(item, root), map_location="cpu", weights_only=False)
        trace = next(t for t in traces if (t["modality"], t["patient_fold"]) == (item["modality"], item["patient_fold"]))
        if checkpoint.get("split_policy") != trace:
            raise ValueError("Partitioned checkpoint split identity differs")
    ready_path, ready = document(native["predictions_ready_path"], native["predictions_ready_sha256"])
    if (original.get("predictions_ready_sha256") != native["predictions_ready_sha256"]
            or Path(original["predictions_ready_path"]).resolve() != ready_path):
        raise ValueError("Partitioned training prediction seal differs")
    stage_path = sources.bind({"path": original["resource_stage_success_path"],
                               "sha256": original["resource_stage_success_sha256"]}, root)
    _, stage = _load_stage(stage_path, verify_external_hashes=True)
    if (ready_path.parent != stage_path.parent
            or stage["candidate_authority"]["sha256"] != manifest["candidate_authority"]["sha256"]
            or original.get("ordered_exact_candidate_key_sha256") != stage["ordered_exact_candidate_key_sha256"]):
        raise ValueError("Partitioned stage and training candidate/seal identity differ")
    sources.bind({"path": ready["resource_manifest_path"], "sha256": ready["resource_manifest_sha256"]}, root)
    for item in [stage["resource_budget"], stage["candidate_authority"], stage["cnv_streaming_aggregate_success"],
                 *stage["candidate_partitions"], *ready["memmaps"].values()]:
        sources.bind(item, root)
    reassembly_path, reassembly = sources.document(manifest["reassembly_receipt"], manifest_root)
    if reassembly.get("format") != REASSEMBLY_FORMAT or reassembly.get("status") != "SUCCESS":
        raise ValueError("Partitioned predictions have not been reassembled")
    _, assembled = document(reassembly["lineage_path"], reassembly["lineage_sha256"], reassembly_path.parent)
    if (assembled.get("format") != REASSEMBLY_FORMAT or assembled.get("training_run_id") != manifest["native_run_id"]
            or assembled.get("ordered_exact_candidate_key_sha256") != stage["ordered_exact_candidate_key_sha256"]
            or assembled.get("candidate_rows") != stage["candidate_rows"]):
        raise ValueError("Partitioned reassembly run/candidate identity differs")
    typed = sources.bind(assembled["combined_predictions"], reassembly_path.parent)
    if sha256(typed) != reassembly.get("prediction_sha256"):
        raise ValueError("Partitioned reassembly typed prediction differs")
    folds = [{**r, "path": str(sources.bind(r, reassembly_path.parent)), "patient_fold_id": r["patient_fold"]}
             for r in assembled["patient_fold_predictions"]]
    if len(folds) != 5 or {r["patient_fold_id"] for r in folds} != set(range(5)):
        raise ValueError("Partitioned reassembly requires five unaveraged folds")
    # Actual bounded producer replay catches a repinned, internally consistent
    # replacement of all five tables and their average.
    with tempfile.TemporaryDirectory(prefix="native-genomic-replay-", dir=root) as temporary:
        replay_root = Path(temporary)/"reassembled"
        replay = reassemble_genomic_memmap_outputs(predictions_ready_path=ready_path,
            output_root=replay_root, training_run_id=manifest["native_run_id"])
        compare_parquet(typed, replay["prediction_path"])
        for item in folds:
            compare_parquet(item["path"], replay_root/"patient_fold_oof_predictions"/f"patient_fold={item['patient_fold_id']}"/"part-0.parquet")
    for item in list(sources.records.values()): sources.bind(item, root)
    # An in-memory compatibility view, never a replacement training receipt.
    lineage = {**original, "input_artifacts": inputs, "v32_core_checkpoint_sha256": original["core_manifest_sha256"],
        "modalities": {m: [{"patient_fold": f, "status": "SUCCESS"} for f in range(5)] for m in ("mutation", "cnv")}}
    return lineage, typed, checkpoints, folds
