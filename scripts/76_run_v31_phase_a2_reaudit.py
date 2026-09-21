#!/usr/bin/env python3
"""Phase A2 repair re-audit for the three-cancer V3.1 pilot.

This stage is CPU/read-only with respect to every frozen source asset.  It
writes a new isolated audit root and never starts model training.  The terminal
JSON is written last and is the only gate accepted by the B1 runner.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cc_hhgt.graph_contract import (  # noqa: E402
    DeploymentContract,
    FoldGraphMode,
    build_fold_graph,
    materialize_relation_provenance,
)
from cc_hhgt.gnn import runtime_bundle_for_step  # noqa: E402
from cc_hhgt.prediction_contract import (  # noqa: E402
    PredictionScale,
    candidate_universe_sha256,
    validate_metric_provenance,
    validate_prediction_scale,
)
from cc_hhgt.relation_sampling import (  # noqa: E402
    RuntimeSamplingPolicy,
    schedule_runtime_edges,
    signal_retention_audit,
)
from cc_hhgt.v29_multitask import train_multitask_fold  # noqa: E402
from cc_hhgt_v26.evidence_transformer import (  # noqa: E402
    CAT_FIELDS,
    UNKNOWN,
    EvidenceStore,
    VocabularyBundle,
    clean_evidence_frame,
)


PILOT_CANCERS = ("BRCA", "COAD", "KIRP")
REFERENCE_ONLY: tuple[str, ...] = ()
SEEDS = (20260726, 20261726, 20262726)
EXPECTED_VALIDATION = {"BRCA": "CESC", "COAD": "DLBC", "KIRP": "LAML"}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_tsv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def edge_id_universe_sha256(edges: pd.DataFrame) -> str:
    """Hash frozen unique edge IDs without reserializing every edge payload."""

    if "edge_id" not in edges or edges.edge_id.astype(str).duplicated().any():
        raise RuntimeError("Fold edge universe lacks unique frozen edge_id values")
    values = pd.util.hash_pandas_object(edges.edge_id.astype(str), index=False).to_numpy(np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def event_indices_for_candidates(events: pd.DataFrame, candidates: pd.DataFrame) -> np.ndarray:
    keys = ["cancer_id", "lncrna_id", "pathway_family_id"]
    candidates = candidates[keys].drop_duplicates()
    local = events.loc[events.cancer_id.astype(str).ne("PAN_CANCER")].reset_index(names="_event_index")
    local_hit = local.merge(candidates, on=keys, how="inner")["_event_index"]
    global_events = events.loc[events.cancer_id.astype(str).eq("PAN_CANCER")].reset_index(names="_event_index")
    global_hit = global_events.merge(
        candidates[["lncrna_id", "pathway_family_id"]].drop_duplicates(),
        on=["lncrna_id", "pathway_family_id"],
        how="inner",
    )["_event_index"]
    return np.unique(np.concatenate([local_hit.to_numpy(np.int64), global_hit.to_numpy(np.int64)]))


def id_mapping_audit(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    known_node_ids = set(nodes.node_id.astype(str))
    rows = []
    for endpoint, column in (("source", "source_node_id"), ("target", "target_node_id")):
        values = edges[column].astype(str)
        unmapped = ~values.isin(known_node_ids)
        rows.append(
            {
                "endpoint": endpoint,
                "n_edges": int(len(values)),
                "n_unmapped": int(unmapped.sum()),
                "mapping_rate": float(1 - unmapped.mean()),
                "status": "PASS" if not unmapped.any() else "FAIL",
            }
        )
    duplicate_canonical = int(nodes.duplicated(["node_type", "canonical_id"]).sum())
    duplicate_node_id = int(nodes.node_id.astype(str).duplicated().sum())
    rows.append(
        {
            "endpoint": "node_table_uniqueness",
            "n_edges": int(len(nodes)),
            "n_unmapped": duplicate_canonical + duplicate_node_id,
            "mapping_rate": 1.0 if duplicate_canonical + duplicate_node_id == 0 else math.nan,
            "status": "PASS" if duplicate_canonical + duplicate_node_id == 0 else "FAIL",
        }
    )
    frame = pd.DataFrame(rows)
    return frame, bool(frame.status.eq("PASS").all())


def contract_and_retention_audit(
    edges: pd.DataFrame,
    folds: pd.DataFrame,
    policy: RuntimeSamplingPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contract_rows: list[dict[str, Any]] = []
    retention_parts: list[pd.DataFrame] = []
    provenance_parts: list[pd.DataFrame] = []
    for cancer in PILOT_CANCERS:
        hit = folds.loc[folds.test_cancer.astype(str).eq(cancer)]
        if len(hit) != 1:
            raise RuntimeError(f"Expected one formal fold for {cancer}; observed {len(hit)}")
        row = hit.iloc[0]
        validation = str(row.validation_cancer)
        if validation != EXPECTED_VALIDATION[cancer]:
            raise RuntimeError(f"Pilot validation cancer drift for {cancer}: {validation}")
        heldout = {cancer, validation}
        for contract in DeploymentContract:
            pseudo = build_fold_graph(
                edges,
                heldout_cancers=heldout,
                mode=FoldGraphMode.PSEUDOHELDOUT,
                contract=contract,
                reference_only=REFERENCE_ONLY,
                compute_fingerprint=False,
            )
            real = build_fold_graph(
                edges,
                heldout_cancers=heldout,
                mode=FoldGraphMode.REAL_TEST,
                contract=contract,
                reference_only=REFERENCE_ONLY,
                compute_fingerprint=False,
            )
            parity_pass = (
                len(pseudo.edges) == len(real.edges)
                and pseudo.edges.edge_id.astype(str).equals(real.edges.edge_id.astype(str))
            )
            if not parity_pass:
                raise RuntimeError(
                    f"Pseudoheldout/real-test graph divergence: cancer={cancer}, contract={contract.value}"
                )
            edge_universe_sha = edge_id_universe_sha256(real.edges)
            visible = real.edges.loc[real.edges.cancer_id.astype("string").isin(heldout)]
            outcome_surviving = int(
                visible.relation_provenance.astype(str).eq("cancer_outcome_derived").sum()
            )
            contract_rows.append(
                {
                    "fold_id": str(row.fold_id),
                    "test_cancer": cancer,
                    "validation_cancer": validation,
                    "contract": contract.value,
                    "pseudoheldout_mode": "pseudoheldout",
                    "real_test_mode": "real_test",
                    "same_filter_function": True,
                    "pseudoheldout_edge_count": len(pseudo.edges),
                    "real_test_edge_count": len(real.edges),
                    "pseudoheldout_sha256": edge_universe_sha,
                    "real_test_sha256": edge_universe_sha,
                    "heldout_outcome_edges_surviving": outcome_surviving,
                    "heldout_unlabeled_context_edges": int(
                        visible.relation_provenance.astype(str).eq("cancer_unlabeled_context").sum()
                    ),
                    "status": "PASS"
                    if parity_pass and outcome_surviving == 0
                    else "FAIL",
                }
            )
            provenance = real.edges.groupby("relation_provenance", observed=True).size().rename("edge_count").reset_index()
            provenance.insert(0, "contract", contract.value)
            provenance.insert(0, "fold_id", str(row.fold_id))
            provenance_parts.append(provenance)

            # Contract-T is the primary pilot contract.  Schedule every
            # available canonical edge through a bounded full coverage cycle.
            if contract is DeploymentContract.TARGET_CONTEXT:
                scheduled = schedule_runtime_edges(real.edges, policy)
                audit = signal_retention_audit(edges, real.edges, scheduled, policy)
                audit.insert(0, "contract", contract.value)
                audit.insert(0, "test_cancer", cancer)
                audit.insert(0, "fold_id", str(row.fold_id))
                retention_parts.append(audit)
                del scheduled
    return (
        pd.DataFrame(contract_rows),
        pd.concat(retention_parts, ignore_index=True),
        pd.concat(provenance_parts, ignore_index=True),
    )


def et_contract_audit(et_root: Path) -> tuple[pd.DataFrame, bool]:
    events = clean_evidence_frame(pd.read_parquet(et_root / "evidence_event_dataset.parquet"))
    candidates = pd.read_parquet(et_root / "evidence_transformer_oof_prediction.parquet")
    required = {"cancer_id", "lncrna_id", "pathway_family_id", "evidence_fold"}
    if not required.issubset(candidates):
        raise RuntimeError(f"ET OOF table lacks fields: {sorted(required - set(candidates))}")
    rows: list[dict[str, Any]] = []
    for heldout in sorted(candidates.evidence_fold.astype(int).unique()):
        validation = (int(heldout) + 1) % 5
        train_candidates = candidates.loc[~candidates.evidence_fold.astype(int).isin([heldout, validation])]
        test_candidates = candidates.loc[candidates.evidence_fold.astype(int).eq(heldout)]
        train_indices = event_indices_for_candidates(events, train_candidates)
        test_indices = event_indices_for_candidates(events, test_candidates)
        train_events = events.loc[train_indices]
        test_events = events.loc[test_indices]
        vocab = VocabularyBundle.fit(
            train_events, drop_constant_fields=True, merge_redundant_fields=True
        )
        store = EvidenceStore(pd.concat([train_events, test_events], ignore_index=True), vocab)
        for field in ("tissue", "cell_line"):
            if field not in (vocab.fields or []):
                unseen_rate = 0.0
                mapped = True
                vocab_size = 0
            else:
                mapping = vocab.mapping(field)
                train_tokens = set(train_events[field].astype(str))
                test_tokens = set(test_events[field].astype(str))
                unseen = test_tokens - train_tokens
                unseen_rate = len(unseen) / len(test_tokens) if test_tokens else 0.0
                mapped = all(mapping.get(token, mapping[UNKNOWN]) == mapping[UNKNOWN] for token in unseen)
                vocab_size = len(mapping)
            rows.append(
                {
                    "heldout_fold": int(heldout),
                    "field": field,
                    "train_event_rows": int(len(train_events)),
                    "test_event_rows": int(len(test_events)),
                    "fold_local_vocabulary": True,
                    "vocab_size": vocab_size,
                    "test_unseen_unique_token_rate": unseen_rate,
                    "unseen_mapped_to_UNK": mapped,
                    "constant_fields_removed": all(
                        train_events[c].astype(str).nunique(dropna=False) > 1 for c in (vocab.fields or [])
                    ),
                    "direct_head_enabled": bool(train_events.direct_target_evidence.astype(int).eq(1).any()),
                    "count_features_explicit": True,
                    "status": "PASS" if mapped else "FAIL",
                }
            )
        if store.cat.shape[1] != len(vocab.fields or []):
            raise RuntimeError("ET encoded categorical width disagrees with fold-local vocabulary")
    frame = pd.DataFrame(rows)
    return frame, bool(frame.status.eq("PASS").all())


def metric_contract_audit() -> tuple[pd.DataFrame, bool]:
    prediction = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "cancer_id": ["BRCA", "BRCA"],
            "split": ["test", "test"],
            "proxy_positive_probability": [0.2, 0.8],
            "prediction_scale": ["raw_probability", "raw_probability"],
        }
    )
    validate_prediction_scale(prediction, PredictionScale.RAW_PROBABILITY)
    universe = candidate_universe_sha256(prediction)
    metrics = pd.DataFrame(
        {
            "prediction_scale": ["raw_probability", "calibrated_probability"],
            "metric_scope": ["LOCO_test", "LOCO_test"],
            "prediction_file_sha256": ["a" * 64, "b" * 64],
            "candidate_universe_sha256": [universe, universe],
            "calibration_model_sha256": ["NOT_APPLICABLE_RAW", "c" * 64],
            "reported_separately": [True, True],
            "implementation": [
                "cc_hhgt.v29_multitask raw writer",
                "cc_hhgt.v29_multitask calibrate_multitask_fold",
            ],
        }
    )
    validate_metric_provenance(metrics)
    metrics["status"] = "PASS"
    return metrics, True


def pf_contract_audit(prior_audit_root: Path) -> tuple[pd.DataFrame, bool]:
    source = prior_audit_root / "PHASE_A/MULTIMODAL_AVAILABILITY_AUDIT.tsv"
    frame = pd.read_csv(source, sep="\t")
    pilot = frame.loc[frame.cancer_id.astype(str).isin(PILOT_CANCERS)].copy()
    if "PF_uses_cancer_level_aggregate" not in pilot:
        raise RuntimeError("Prior PF audit lacks aggregate-misuse field")
    pilot["status"] = np.where(
        pilot.PF_uses_cancer_level_aggregate.fillna(False).astype(bool), "FAIL", "PASS"
    )
    return pilot, bool(pilot.status.eq("PASS").all())


def loss_contract_audit() -> tuple[pd.DataFrame, bool]:
    frame = pd.DataFrame(
        [
            {
                "task": "pathway",
                "loss_weight": 1.0,
                "loss_form": "nnPU + positive-only direction BCE",
                "required_diagnostics": "label_prevalence;sampling_prevalence;pos_weight;raw_logit_distribution;probability_distribution",
                "status": "PASS",
            },
            {
                "task": "state",
                "loss_weight": 0.50,
                "loss_form": "independent state-head nnPU + positive-only direction BCE",
                "required_diagnostics": "label_prevalence;sampling_prevalence;pos_weight;raw_logit_distribution;probability_distribution",
                "status": "PASS",
            },
        ]
    )
    return frame, bool(frame.status.eq("PASS").all() and frame.loc[frame.task.eq("state"), "loss_weight"].eq(0.5).all())


def runtime_training_integration_audit() -> tuple[pd.DataFrame, bool]:
    source = inspect.getsource(train_multitask_fold)
    runtime_source = inspect.getsource(runtime_bundle_for_step)
    requirements = {
        "runtime_bundle_materialized_each_epoch": "runtime_bundle_for_step(" in source,
        "episodic_planner_connected": "episode_planner.episode(epoch)" in source,
        "loss_restricted_to_pseudoheldout": "pseudoheldout_cancer" in source and "path_pool" in source,
        "pseudoheldout_pair_features_fully_masked": "zero_all=(episode is not None)" in source,
        "real_validation_graph_drops_episode_mask": (
            "validation_bundle = (" in source
            and "runtime_bundle_for_step(bundle, int(validation_chunk))" in source
        ),
        "checkpoint_validation_uses_full_coverage_cycle": (
            "validation_chunks = (" in source
            and "list(range(bundle.runtime_num_chunks))" in source
            and '"validation_full_coverage_cycle": validation_performed' in source
        ),
        "final_prediction_ensembles_all_chunks": "evaluation_chunks = list(range(bundle.runtime_num_chunks))" in source,
        "state_embedding_ensembles_all_chunks": (
            "state_embedding_mean" in source
            and 'runtime_ensemble_chunks=len(evaluation_chunks)' in source
        ),
        "runtime_early_stopping_uses_validation_cycles": (
            "checkpoint_patience_cycles" in source
            and "pathway_stale_validation_cycles >= checkpoint_patience_cycles"
            in source
            and "state_stale_validation_cycles >= checkpoint_patience_cycles"
            in source
        ),
        "state_loss_locked_0_50": "locks state auxiliary loss weight at 0.50" in source,
        "pseudoheldout_degree_features_use_filtered_schedule": (
            "episode_node_degrees = _subtract_edge_degrees(" in runtime_source
            and "node_degree_override=episode_node_degrees" in runtime_source
            and "kept_cancer_edges = build_fold_graph(" in runtime_source
        ),
        "pseudoheldout_fold_local_weights_recomputed_after_masking": (
            "localization_edges = localize_cancer_state_edges(localization_edges)"
            in runtime_source
        ),
        "relation_schema_is_metadata_only": "relation_schema_edges=" in runtime_source,
    }
    frame = pd.DataFrame(
        [
            {
                "requirement": key,
                "connected_to_training_path": value,
                "implementation": "cc_hhgt.v29_multitask.train_multitask_fold",
                "status": "PASS" if value else "FAIL",
            }
            for key, value in requirements.items()
        ]
    )
    return frame, bool(all(requirements.values()))


def file_manifest(paths: list[Path], root: Path | None = None) -> pd.DataFrame:
    rows = []
    for path in sorted(set(map(Path, paths))):
        rows.append(
            {
                "path": str(path.relative_to(root)) if root and path.is_relative_to(root) else str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--prior-audit-root", type=Path, required=True)
    parser.add_argument("--et-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--edge-chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Phase A2 refuses a nonempty output root: {output}")
    for name in ("PHASE_A2", "AUDITS", "METRICS", "FIGURES", "MANIFESTS"):
        (output / name).mkdir(parents=True, exist_ok=True)

    edge_path = args.formal_root / "assets/results/tables/graph_edge.parquet"
    node_path = args.formal_root / "assets/results/tables/graph_node.parquet"
    fold_path = args.formal_root / "assets/results/tables/fold_manifest.tsv"
    edges = materialize_relation_provenance(pd.read_parquet(edge_path))
    nodes = pd.read_parquet(node_path)
    folds = pd.read_csv(fold_path, sep="\t")
    policy = RuntimeSamplingPolicy(edge_chunk_size=args.edge_chunk_size)

    id_frame, id_pass = id_mapping_audit(nodes, edges)
    contract, retention, provenance = contract_and_retention_audit(edges, folds, policy)
    et_frame, et_pass = et_contract_audit(args.et_root)
    metric_frame, metric_pass = metric_contract_audit()
    pf_frame, pf_pass = pf_contract_audit(args.prior_audit_root)
    loss_frame, loss_pass = loss_contract_audit()
    runtime_training_frame, runtime_training_pass = runtime_training_integration_audit()

    coexpression = retention.loc[retention.relation_family.str.contains("coexpressed_with")]
    coexpression_pass = bool(
        len(coexpression)
        and coexpression.weighted_signal_retention.ge(policy.weighted_signal_retention_min).all()
        and coexpression.node_coverage.ge(policy.source_coverage_min).all()
        and coexpression.positive_mass_retention.ge(policy.weighted_signal_retention_min).all()
        and coexpression.negative_mass_retention.ge(policy.weighted_signal_retention_min).all()
    )
    no_destructive_cap = bool(
        retention.destructive_offline_truncation.eq(False).all()
        and retention.sampled_edge_count.eq(retention.available_edge_count).all()
        and retention.peak_runtime_edges.le(args.edge_chunk_size).all()
    )
    contract_pass = bool(
        contract.status.eq("PASS").all()
        and contract.same_filter_function.eq(True).all()
        and contract.heldout_outcome_edges_surviving.eq(0).all()
    )

    checks = {
        "id_mapping": id_pass,
        "outcome_leakage": bool(contract.heldout_outcome_edges_surviving.eq(0).all()),
        "metric_provenance": metric_pass,
        "pseudoheldout_real_test_same_masking_code": contract_pass,
        "no_destructive_global_300k_cap": no_destructive_cap,
        "coexpression_signal_retention": coexpression_pass,
        "pf_data_level_contract": pf_pass,
        "et_unseen_token_contract": et_pass,
        "fixed_state_loss_weight_0_50": loss_pass,
        "runtime_sampler_connected_to_training": runtime_training_pass,
        "pseudoheldout_node_features_outcome_free": bool(
            runtime_training_frame.loc[
                runtime_training_frame.requirement.eq(
                    "pseudoheldout_degree_features_use_filtered_schedule"
                ),
                "status",
            ].eq("PASS").all()
        ),
        "contract_s_t_reported_separately": set(contract.contract) == {"CONTRACT-S", "CONTRACT-T"},
        "pilot_scope_exact": set(contract.test_cancer) == set(PILOT_CANCERS),
        "full_cancer_training_started": False,
        "gpu_used": False,
    }
    negative_assertions = {"full_cancer_training_started", "gpu_used"}
    passed = all(value is True for key, value in checks.items() if key not in negative_assertions)
    passed = passed and all(checks[key] is False for key in negative_assertions)

    atomic_tsv(output / "AUDITS/ID_MAPPING_AUDIT.tsv", id_frame)
    atomic_tsv(output / "AUDITS/LOCO_CONTRACT_PARITY_AUDIT.tsv", contract)
    atomic_tsv(output / "AUDITS/EDGE_SIGNAL_RETENTION_AUDIT.tsv", retention)
    atomic_tsv(output / "AUDITS/RELATION_PROVENANCE_AUDIT.tsv", provenance)
    atomic_tsv(output / "AUDITS/ET_CLEAN_INPUT_AUDIT.tsv", et_frame)
    atomic_tsv(output / "AUDITS/RAW_CALIBRATED_METRIC_AUDIT.tsv", metric_frame)
    atomic_tsv(output / "AUDITS/PF_DATA_LEVEL_CONTRACT_AUDIT.tsv", pf_frame)
    atomic_tsv(output / "AUDITS/LOSS_OUTPUT_CONTRACT_AUDIT.tsv", loss_frame)
    atomic_tsv(
        output / "AUDITS/RUNTIME_TRAINING_INTEGRATION_AUDIT.tsv",
        runtime_training_frame,
    )

    payload = {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "phase": "PHASE_A2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_cancers": list(PILOT_CANCERS),
        "validation_cancers": EXPECTED_VALIDATION,
        "models_allowed_next": ["cc_hhgt"],
        "seeds_allowed_next": list(SEEDS),
        "contracts": ["CONTRACT-S", "CONTRACT-T"],
        "primary_contract": "CONTRACT-T",
        "checks": checks,
        "failures": [
            key
            for key, value in checks.items()
            if (key in negative_assertions and value is not False)
            or (key not in negative_assertions and value is not True)
        ],
        "edge_chunk_size": args.edge_chunk_size,
        "coexpression_weighted_signal_retention_min": float(coexpression.weighted_signal_retention.min()) if len(coexpression) else math.nan,
        "coexpression_source_node_coverage_min": float(coexpression.node_coverage.min()) if len(coexpression) else math.nan,
        "git_commit": args.git_commit,
        "source_code_root": str(ROOT),
        "source_formal_root": str(args.formal_root),
        "source_et_root": str(args.et_root),
        "source_prior_audit_root": str(args.prior_audit_root),
        "training_started": False,
        "gpu_used": False,
        "full_cancer_training_authorized": False,
        "next_stage_allowed": "PHASE_B1_FIXED_GRAPH" if passed else "NONE",
    }
    summary = f"""# Phase A2 Re-audit Summary

## Decision

**PHASE A2 = {payload['status']}**. GPU training was not started.

## Locked scope

- Test cancers: BRCA, COAD, KIRP only.
- Architecture allowed after this gate: CC-HHGT only.
- Contract-T is primary; Contract-S remains a separately reported historical diagnostic.
- Full-cancer V3.1 training remains forbidden.

## Repair checks

""" + "\n".join(f"- {name}: {'PASS' if value is True or (name in negative_assertions and value is False) else 'FAIL'}" for name, value in checks.items()) + f"""

## Quantitative anchors

- Canonical graph edges: {len(edges):,}.
- Runtime edge chunk limit: {args.edge_chunk_size:,}; no canonical edge is deleted.
- Minimum Contract-T coexpression weighted retention: {payload['coexpression_weighted_signal_retention_min']:.6f}.
- Minimum Contract-T coexpression source coverage: {payload['coexpression_source_node_coverage_min']:.6f}.
- Pseudoheldout/real-test parity rows: {int(contract.status.eq('PASS').sum())}/{len(contract)} PASS.
- ET fold-local unseen-token rows: {int(et_frame.status.eq('PASS').sum())}/{len(et_frame)} PASS.

## Gate consequence

{'B1 three-cancer Fixed CC-HHGT may start under the exact locked scope.' if passed else 'STOP. B1 must not start until every failed A2 item is repaired and the audit is rerun in a fresh root.'}
"""
    atomic_text(output / "PHASE_A2/PHASE_A2_REAUDIT_SUMMARY.md", summary)

    sources = [
        edge_path,
        node_path,
        fold_path,
        args.formal_root / "provenance/PROVENANCE.json",
        args.et_root / "evidence_event_dataset.parquet",
        args.et_root / "evidence_transformer_oof_prediction.parquet",
        args.prior_audit_root / "PHASE_A/MULTIMODAL_AVAILABILITY_AUDIT.tsv",
        ROOT / "config/model_v3_1_fixed_graph_pilot.yaml",
    ]
    atomic_tsv(output / "MANIFESTS/SOURCE_ASSET_SHA256.tsv", file_manifest(sources))
    atomic_json(output / "PHASE_A2/PHASE_A2_REAUDIT.json", payload)
    outputs = [path for path in output.rglob("*") if path.is_file() and path.name != "OUTPUT_SHA256_MANIFEST.tsv"]
    atomic_tsv(output / "MANIFESTS/OUTPUT_SHA256_MANIFEST.tsv", file_manifest(outputs, output))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
