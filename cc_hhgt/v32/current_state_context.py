"""Native State OOF results joined to one current selected-analysis snapshot.

State membership is a private-head prediction; the signed State effect is a
held-out Pearson correlation. Selected-score-weighted profiles are descriptive
member summaries, not predictions of joint GeneSet activity or causal effects.
No optimizer, training entry point or patient label is exposed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import duckdb
import numpy as np
import pandas as pd

from .contracts import canonical_json_sha256
from .current_core_export import sha256, verified_path, validate_current_export, checked_sha
from .integrated_model import module_state_sha256
from .multimodal_fusion import _duckdb_literal as literal
from .native_auxiliary_fusion import Sources
from .selected_genesets import SelectedAnalysisQuery, audit_selected_analysis
from .state_training import (HISTORICAL_STATE_IDS, STATE_TRAINING_FORMAT,
    STATE_CHECKPOINT_FORMAT, StateTrainingConfig, load_v32_core_embeddings, build_state_private_head)

INPUT_FORMAT = "CURRENT_STATE_CONTEXT_INPUT_V1"
FORMAT = "CURRENT_STATE_CONTEXT_RELEASE_V1"
KEYS = ["cancer_id", "lncrna_id", "state_id"]
EFFECT_SOURCE = "held_out_patient_fold_measurement"
CATALOG_COLUMNS = ["gene_set_id", "gene_set_name", "gene_set_type", "scope", "cancer_id",
    "state_id", "direction", "member_count", "selection_rule"]
MEMBER_COLUMNS = ["gene_set_id", "member_rank", "member_id", "member_gene_id", "membership_score",
    "state_effect", "cancer_count", "direction_consistency", "source_cancer_id"]


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def _assert_same_table(observed, expected):
    # Arrow strings, pandas nullable values and NumPy encode null differently.
    # Normalize representations, never the presence of a missing observation.
    pd.testing.assert_frame_equal(observed.isna(), expected.isna(), check_dtype=False)
    left = observed.astype(object).where(observed.notna(), None)
    right = expected.astype(object).where(expected.notna(), None)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=0, atol=1e-12)


@dataclass(frozen=True)
class StateSetConfig:
    cancer_max_members: int = 100
    pan_cancer_max_members: int = 200
    minimum_set_members: int = 3
    pan_cancer_min_cancers: int = 5
    pan_cancer_min_direction_consistency: float = .60

    def validate(self):
        integers = (self.cancer_max_members, self.pan_cancer_max_members,
                    self.minimum_set_members, self.pan_cancer_min_cancers)
        if (any(type(n) is not int or n < 1 for n in integers)
                or self.minimum_set_members > min(integers[:2])
                or not np.isfinite(self.pan_cancer_min_direction_consistency)
                or not .5 < self.pan_cancer_min_direction_consistency <= 1):
            raise ValueError("Invalid current State GeneSet policy")


def bind_state_input(sources, record, root):
    """Honor the native producer's file-or-dataset hash contract.

    Keep individual files in the public context's private source inventory;
    the complete directory membership hash remains bound by training lineage.
    """
    from .state_training import path_sha256
    path = (root / record["path"]).resolve()
    if not path.is_dir(): return sources.bind(record, root)
    if record["role"] == "v32_core_embedding_manifest": raise ValueError("State core manifest must be a file")
    expected = checked_sha(record.get("sha256"), "State dataset")
    if path_sha256(path) != expected: raise ValueError("State dataset SHA256 mismatch")
    for part in sorted(p for p in path.rglob("*") if p.is_file()):
        sources.bind({"path": str(part), "sha256": sha256(part)}, path)
    if path_sha256(path) != expected: raise ValueError("State dataset changed during binding")
    return path


def state_core_sources(manifest, root, sources):
    """Bind selected scores and all five actual core parents before any State run."""
    selected_path = sources.bind(manifest["selected_analysis"], root)
    selected = SelectedAnalysisQuery(selected_path, expected_sha=manifest["selected_analysis"]["sha256"])
    audit_selected_analysis(selected_path, expected_sha=selected.binding_sha256)
    score = _json(selected.score_binding_path)
    reference = score["sources"]["reference_exact_producer"]
    primary_path, primary = sources.document(reference["producer_binding"], selected.score_binding_path.parent)
    from .current_core_prediction import FORMAT as CURRENT_PREDICTION_FORMAT
    if primary.get("format") != CURRENT_PREDICTION_FORMAT:
        raise ValueError("Current State context requires a native current-core reference")
    if primary.get("training_run_id") != manifest["core_run_id"]:
        raise ValueError("State and selected analysis have different current core runs")
    core_path = sources.bind(manifest["core_manifest"], root)
    core, _, _ = validate_current_export(core_path, expected_run_id=manifest["core_run_id"])
    cores, core_lineage, _ = load_v32_core_embeddings(core_path, repo_root=core_path.parent)
    primary_folds = {r["patient_fold"]: r for r in primary["ensemble_source_records"]}
    if set(primary_folds) != set(range(5)):
        raise ValueError("Selected current core lacks five folds")
    for fold, parent in core_lineage.items():
        row = primary_folds[fold]
        if (row["checkpoint_sha256"] != parent["v32_core_checkpoint_sha256"]
                or row["core_parameter_sha256"] != parent["core_parameter_sha256"]):
            raise ValueError("State and selected scores use different current core checkpoints")
        sources.bind({"path": parent["core_embedding_path"], "sha256": parent["core_embedding_sha256"]}, core_path.parent)
        sources.bind(core["folds"][str(fold)]["training_receipt"], core_path.parent)
    return selected, score, core, cores, core_lineage


def state_sources(manifest_path):
    """Bind actual State heads, native tables and their selected current core."""
    import torch
    path = Path(manifest_path).resolve(); manifest = _json(path); root = path.parent
    if manifest.get("format") != INPUT_FORMAT or not manifest.get("core_run_id") or not manifest.get("state_run_id"):
        raise ValueError("Invalid current State manifest")
    sources = Sources()
    selected, score, core, cores, core_lineage = state_core_sources(manifest, root, sources)
    lineage_path, lineage = sources.document(manifest["state_training_lineage"], root)
    if (lineage.get("training_format") != STATE_TRAINING_FORMAT or lineage.get("module_id") != "state"
            or lineage.get("training_status") != "SUCCESS" or lineage.get("folds") != 5
            or lineage.get("training_run_id") != manifest["state_run_id"]
            or lineage.get("historical_state_ids") != list(HISTORICAL_STATE_IDS)
            or lineage.get("lineage_sha256") != canonical_json_sha256({k:v for k,v in lineage.items() if k!="lineage_sha256"})):
        raise ValueError("Invalid native State training lineage")
    for key in ("private_head_trained_from_scratch", "core_parameters_frozen", "core_embeddings_detached",
                "fold_local_association_features_and_labels"):
        if lineage.get(key) is not True: raise ValueError("Missing State isolation: "+key)
    for key in ("old_checkpoint_loaded", "old_predictions_used_as_features", "old_rankings_used_as_outputs",
                "test_measurements_used_for_training", "unavailable_sentinel_values_used"):
        if lineage.get(key) is not False: raise ValueError("Invalid State isolation: "+key)
    cfg = StateTrainingConfig(**lineage["config"]); cfg.validate()
    if canonical_json_sha256(asdict(cfg)) != lineage.get("config_sha256"):
        raise ValueError("Native State configuration hash differs")
    input_records = lineage["input_artifacts"]
    inputs = {r["role"]: bind_state_input(sources, r, lineage_path.parent) for r in input_records}
    required = {"tumor_state_long_measurement", "v32_patient_fold_manifest", "lncrna_expression", "v32_core_embedding_manifest"}
    if len(inputs) != len(input_records) or set(inputs) != required:
        raise ValueError("Native State inputs are incomplete or duplicated")
    if sha256(inputs["v32_core_embedding_manifest"]) != manifest["core_manifest"]["sha256"]:
        raise ValueError("Native State used a different core export")
    authority = lineage.get("patient_authority")
    if authority is not None:
        from .state_training import state_patient_authority
        patient_path = sources.bind(authority["patient_map"], lineage_path.parent)
        receipt = sources.bind(authority["receipt"], lineage_path.parent)
        if (patient_path != inputs["v32_patient_fold_manifest"]
                or authority != state_patient_authority(patient_path, receipt, authority["mode"])):
            raise ValueError("State completed patient authority differs")
    if "workflow_preparation" in manifest:
        from .native_state_workflow import validate_completed
        validate_completed(manifest["workflow_preparation"], manifest, lineage_path, lineage, sources, root)
    from .state_training import read_table, validate_fold_manifest
    raw_patients = read_table(inputs["v32_patient_fold_manifest"], "V3.2 patient fold manifest")
    patients = validate_fold_manifest(raw_patients)
    if patients.groupby("patient_id").patient_fold_id.nunique().gt(1).any():
        raise ValueError("One canonical patient crosses native State folds")
    if set(lineage["fold_lineage"]) != {str(i) for i in range(5)} or set(lineage["new_private_head_checkpoints"]) != {str(i) for i in range(5)}:
        raise ValueError("Native State requires five actual private heads")
    heads = {}
    for fold in range(5):
        row = lineage["fold_lineage"][str(fold)]; parent = core_lineage[fold]
        for key in ("core_run_id", "core_parameter_sha256", "v32_core_checkpoint_sha256", "core_checkpoint_format"):
            if row.get(key) != parent[key]: raise ValueError("State fold parent differs: "+key)
        for key in ("core_parameters_before_sha256", "core_parameters_after_sha256"):
            if row.get(key) != parent["core_parameter_sha256"]: raise ValueError("State changed its frozen core")
        for key in ("core_embedding_sha256_before", "core_embedding_sha256_after"):
            if row.get(key) != parent["core_embedding_sha256"]: raise ValueError("State core embedding changed")
        if (row.get("patient_fold") != fold or row.get("core_embedding_artifact_unchanged") is not True
                or row.get("fold_local_features_and_labels") is not True
                or row.get("test_measurements_used_for_training") is not False
                or not row.get("history") or row.get("epochs_completed", 0) < 1):
            raise ValueError("Incomplete native State fold training receipt")
        checkpoint_path = sources.bind(lineage["new_private_head_checkpoints"][str(fold)], lineage_path.parent)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (checkpoint.get("checkpoint_format") != STATE_CHECKPOINT_FORMAT or checkpoint.get("module_id") != "state"
                or checkpoint.get("patient_fold") != fold or checkpoint.get("run_id") != manifest["state_run_id"]
                or checkpoint.get("config") != asdict(cfg) or checkpoint.get("state_ids") != list(HISTORICAL_STATE_IDS)
                or checkpoint.get("source_checkpoint_sha256") is not None
                or checkpoint.get("private_head_trained_from_scratch") is not True
                or checkpoint.get("core_parameters_frozen") is not True
                or checkpoint.get("historical_predictions_used") is not False
                or checkpoint.get("initialization") != row.get("initialization")):
            raise ValueError("Native State checkpoint identity or provenance differs")
        tensors = checkpoint.get("model_state", {})
        if not tensors or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in tensors.values()):
            raise ValueError("Invalid State private-head tensors")
        digest = module_state_sha256(SimpleNamespace(state_dict=lambda: tensors))
        if digest != checkpoint.get("final_parameter_sha256") or digest != row.get("final_parameter_sha256"):
            raise ValueError("State private-head tensor fingerprint differs")
        head, initialization = build_state_private_head(fold, row["core_features"], row["domain_features"], cfg)
        if (row.get("seed") != cfg.seed + fold or initialization.as_dict() != row["initialization"]
                or digest == initialization.initial_parameter_sha256):
            raise ValueError("Native State initialization or parameter update receipt differs")
        head.load_state_dict(tensors, strict=True)
        heads[fold] = head
    native_paths = {name: sources.bind(record, lineage_path.parent) for name,record in lineage["outputs"].items()}
    if set(native_paths) != {"oof_predictions", "lncrna_state_release"}:
        raise ValueError("Unknown native State output inventory")
    policy = StateSetConfig(**manifest.get("state_sets", {})); policy.validate()
    from .state_patient_use import Trace
    patient_trace = Trace(raw_patients, next(r for r in input_records if r["role"] == "v32_patient_fold_manifest"),
        cores, core_lineage, cfg.validation_offset)
    replay = replay_native_state(inputs, cores, heads, cfg, lineage, native_paths, patient_trace=patient_trace)
    from .current_state_model_use import derive
    scopes = patient_trace.table()
    input_use, input_coverage = derive(manifest, root, sources, lineage_path, lineage, raw_patients, scopes)
    return dict(manifest=manifest, sources=sources, selected=selected, score=score, lineage=lineage,
        native_paths=native_paths, min_folds=cfg.min_available_folds, patients=patients, policy=policy, replay=replay,
        statistical_patient_scopes=scopes, input_use=input_use, input_coverage=input_coverage)


def replay_native_state(inputs, cores, heads, cfg, lineage, native_paths, *, patient_trace=None):
    """Replay the native measurement-to-OOF calculation with frozen CPU heads.

    This verifies the recorded transformation, not acquisition ancestry or
    model performance. The original training statistics/forward code is used;
    no optimizer, label fitting or independent-statistic claim is involved.
    """
    from .state_training import (read_table, prepare_state_inputs, state_fold_statistics,
        build_fold_oof_predictions, _model_arrays, dataframe_sha256)
    state, expression, folds, cores, alignment = prepare_state_inputs(
        read_table(inputs["tumor_state_long_measurement"], "tumor_state_long measurement"),
        read_table(inputs["v32_patient_fold_manifest"], "V3.2 patient fold manifest"), cores,
        read_table(inputs["lncrna_expression"], "lncRNA expression"), cfg)
    observed = [s for s in HISTORICAL_STATE_IDS if state.state_id.eq(s).any()]
    if (lineage.get("observed_state_ids") != observed
            or lineage.get("missing_state_ids") != [s for s in HISTORICAL_STATE_IDS if s not in observed]):
        raise ValueError("State measured-program inventory differs")
    replayed = []; fold_checks = []
    for fold in range(5):
        split, statistics = state_fold_statistics(state, expression, folds, fold, cfg,
            patient_observer=None if patient_trace is None else patient_trace.observer(fold))
        maximum = int(statistics["train"].n_pairs.max())
        counts = {}; dimensions = None
        for role in ("train", "validation"):
            frame, columns, _, domain, labels = _model_arrays(statistics[role], cores[fold], max_train_pairs=maximum)
            eligible = frame.available.to_numpy(bool) & frame.core_embedding_available.to_numpy(bool) & np.isfinite(labels)
            counts[role] = int(eligible.sum())
            if role == "train":
                positives = int(labels[eligible].sum()); dimensions = (len(columns), domain.shape[1])
        if counts["train"] < 2: raise ValueError("State replay has fewer than two trainable candidates")
        fallback = not counts["validation"]
        expected = {"max_train_pairs": maximum, "core_features": dimensions[0], "domain_features": dimensions[1],
            "trainable_candidates": counts["train"], "validation_candidates": counts["train"] if fallback else counts["validation"],
            "validation_fallback_to_training": fallback, "positive_candidates": positives,
            "negative_candidates": counts["train"] - positives, "alignment_counts": alignment,
            "split_manifest_sha256": dataframe_sha256(split, ["cancer_id", "sample_id", "outer_fold"])}
        row = lineage["fold_lineage"][str(fold)]
        if any(row.get(k) != value for k, value in expected.items()):
            raise ValueError("State replay split, alignment or feature counts differ at fold " + str(fold))
        predictions = build_fold_oof_predictions(heads[fold].cpu(), statistics["train"], statistics["test"], cores[fold],
            fold=fold, max_train_pairs=maximum, config=cfg)
        replayed.append(predictions)
        fold_checks.append({"fold": fold, **expected, "oof_rows": len(predictions)})
    expected = pd.concat(replayed, ignore_index=True).sort_values(["patient_fold_id", *KEYS]).reset_index(drop=True)
    observed = pd.read_parquet(native_paths["oof_predictions"])
    if set(observed.columns) != set(expected.columns): raise ValueError("State replay OOF schema differs")
    observed = observed[expected.columns].sort_values(["patient_fold_id", *KEYS]).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(observed.isna(), expected.isna(), check_dtype=False)
        # Allow float32 CPU/device roundoff for neural probability only. Effects,
        # pair counts, availability and all identities retain stricter comparison.
        probability = "state_membership_probability"
        _assert_same_table(observed.drop(columns=probability), expected.drop(columns=probability))
        np.testing.assert_allclose(observed[probability].to_numpy(dtype=float, na_value=np.nan),
            expected[probability].to_numpy(dtype=float, na_value=np.nan), atol=1e-6, rtol=1e-6, equal_nan=True)
    except AssertionError as error:
        raise ValueError("State raw measurement or frozen-head OOF replay differs") from error
    return {"format": "NATIVE_STATE_NUMERIC_REPLAY_V1", "folds": fold_checks, "rows": len(expected),
        "raw_measurement_statistics_replayed": True, "private_head_predictions_replayed": True,
        "same_implementation_replay": True, "source_acquisition_ancestry_verified": False,
        "independent_predictive_performance_verified": False, "training_started": False}


def native_aggregate(context):
    """Independently recompute the native pair-count-weighted five-fold release."""
    connection = duckdb.connect()
    try:
        path = context["native_paths"]["oof_predictions"]
        connection.execute(f"CREATE VIEW oof AS SELECT * FROM read_parquet({literal(str(path))})")
        schema = {row[0]:row[1] for row in connection.execute("DESCRIBE oof").fetchall()}
        required = set(KEYS)|{"patient_fold_id", "state_membership_probability", "state_effect", "association_direction",
            "availability", "availability_reason", "test_n_pairs", "train_n_pairs", "model_version", "probability_source", "effect_source"}
        if not required.issubset(schema) or schema["availability"] != "BOOLEAN":
            raise ValueError("Invalid native State schema or non-boolean availability")
        states = ",".join("'"+s.replace("'", "''")+"'" for s in HISTORICAL_STATE_IDS)
        invalid = f"""
            availability IS NULL OR patient_fold_id IS NULL OR patient_fold_id NOT IN (0,1,2,3,4)
            OR state_id IS NULL OR state_id NOT IN ({states}) OR cancer_id IS NULL OR trim(cancer_id)=''
            OR lncrna_id IS NULL OR trim(lncrna_id)='' OR test_n_pairs IS NULL OR train_n_pairs IS NULL
            OR NOT isfinite(test_n_pairs) OR NOT isfinite(train_n_pairs)
            OR test_n_pairs < 0 OR test_n_pairs != floor(test_n_pairs) OR train_n_pairs < 0 OR train_n_pairs != floor(train_n_pairs)
            OR model_version IS DISTINCT FROM 'V3.2' OR probability_source IS DISTINCT FROM 'fresh_fold_private_state_head'
            OR effect_source IS DISTINCT FROM '{EFFECT_SOURCE}'
            OR (availability AND (state_membership_probability IS NULL OR state_effect IS NULL
                OR NOT isfinite(state_membership_probability) OR NOT isfinite(state_effect)
                OR state_membership_probability NOT BETWEEN 0 AND 1 OR state_effect NOT BETWEEN -1 AND 1
                OR availability_reason IS NOT NULL OR association_direction IS DISTINCT FROM
                    CASE WHEN state_effect>0 THEN 'positive' WHEN state_effect<0 THEN 'negative' ELSE 'neutral' END))
            OR (NOT availability AND (state_membership_probability IS NOT NULL OR state_effect IS NOT NULL
                OR association_direction IS NOT NULL OR availability_reason IS NULL OR trim(availability_reason)=''))"""
        if connection.execute("SELECT count(*) FROM oof WHERE "+invalid).fetchone()[0]:
            raise ValueError("Invalid native State values, direction or nullable availability")
        keys = ",".join(KEYS)
        if connection.execute(f"SELECT count(*) FROM (SELECT {keys} FROM oof GROUP BY {keys} HAVING count(*)!=5 OR count(DISTINCT patient_fold_id)!=5)").fetchone()[0]:
            raise ValueError("State candidate lacks exactly five distinct native folds")
        count = connection.execute("SELECT count(*) FROM oof").fetchone()[0]
        if count == 0 or count != context["lineage"]["outputs"]["oof_predictions"]["rows"]:
            raise ValueError("Native State OOF row count differs")
        n = context["min_folds"]
        frame = connection.execute(f"""
            WITH summary AS (SELECT {keys}, count(*) FILTER (WHERE availability) AS folds_available,
                sum(state_membership_probability*greatest(test_n_pairs,1)) FILTER (WHERE availability)
                    /sum(greatest(test_n_pairs,1)) FILTER (WHERE availability) AS p,
                sum(state_effect*greatest(test_n_pairs,1)) FILTER (WHERE availability)
                    /sum(greatest(test_n_pairs,1)) FILTER (WHERE availability) AS e,
                string_agg(DISTINCT availability_reason, '|' ORDER BY availability_reason) FILTER (WHERE NOT availability) AS reasons
                FROM oof GROUP BY {keys})
            SELECT {keys}, CASE WHEN folds_available>={n} THEN p END AS state_membership_probability,
                CASE WHEN folds_available>={n} THEN e END AS state_effect,
                CASE WHEN folds_available>={n} THEN CASE WHEN e>0 THEN 'positive' WHEN e<0 THEN 'negative' ELSE 'neutral' END END AS association_direction,
                folds_available>={n} AS availability,
                CASE WHEN folds_available<{n} THEN 'fewer_than_{n}_available_folds:' || coalesce(reasons,'unknown') END AS availability_reason,
                folds_available, 5 AS folds_expected, 'V3.2' AS model_version,
                'five_fold_oof_fresh_private_state_heads' AS probability_source,
                '{EFFECT_SOURCE}' AS effect_source FROM summary ORDER BY {keys}
        """).df()
    finally:
        connection.close()
    original = pd.read_parquet(context["native_paths"]["lncrna_state_release"]).sort_values(KEYS).reset_index(drop=True)
    if len(original) != context["lineage"]["outputs"]["lncrna_state_release"]["rows"]:
        raise ValueError("Native State aggregate row count differs")
    if not pd.api.types.is_bool_dtype(original.availability):
        raise ValueError("Native State aggregate availability is not boolean")
    try:
        _assert_same_table(original, frame)
    except AssertionError as error:
        raise ValueError("Native State aggregate disagrees with five-fold recomputation") from error
    cancers = set(context["patients"].cancer_id.astype(str))
    if not set(frame.cancer_id).issubset(cancers): raise ValueError("State predictions contain a cancer outside patient authority")
    return frame


def selected_state_tables(members, coverage, aggregate):
    """Compute summaries using actual selected members and nonnegative scores."""
    selected_columns = ["geneset_id", "geneset_rank", "cancer_id", "lncrna_id", "pathway_id", "selected_score", "association_direction"]
    if not set(selected_columns).issubset(members): raise ValueError("Selected member schema is incomplete")
    values = pd.to_numeric(members.selected_score, errors="coerce")
    if not np.isfinite(values).all() or not values.between(0,1).all() or members.duplicated(["geneset_id","lncrna_id"]).any():
        raise ValueError("Invalid or duplicate selected State members")
    states = pd.DataFrame({"state_id": list(HISTORICAL_STATE_IDS)})
    data = members[selected_columns].rename(columns={"association_direction": "pathway_association_direction"}).merge(states, how="cross")
    native = aggregate.rename(columns={"association_direction": "state_association_direction"})
    data = data.merge(native, how="left", on=KEYS, validate="many_to_one")
    absent = data.availability.isna()
    data.loc[absent, "availability_reason"] = "NO_NATIVE_STATE_OOF_RESULT_FOR_SELECTED_MEMBER"
    data["availability"] = data.availability.eq(True)
    data["folds_available"] = data.folds_available.fillna(0).astype(int)
    data["folds_expected"] = 5
    data["state_effect_semantics"] = "HELD_OUT_PEARSON_CORRELATION_NOT_CAUSAL"
    # Each member appears once per State; no duplication over pathways in a profile.
    data["available_score_weight"] = data.selected_score.where(data.availability, 0.)
    data["weighted_membership"] = data.state_membership_probability.fillna(0)*data.available_score_weight
    data["weighted_effect"] = data.state_effect.fillna(0)*data.available_score_weight
    profile_keys = ["geneset_id", "cancer_id", "pathway_id", "state_id", "pathway_association_direction"]
    profiles = data.groupby(profile_keys, sort=True, dropna=False).agg(
        member_count=("lncrna_id","size"), available_members=("availability","sum"),
        total_score_weight=("selected_score","sum"), available_score_weight=("available_score_weight","sum"),
        weighted_membership=("weighted_membership","sum"), weighted_effect=("weighted_effect","sum")).reset_index()
    denom = profiles.available_score_weight.where(profiles.available_score_weight.gt(0))
    profiles["mean_state_membership_probability"] = profiles.weighted_membership/denom
    profiles["mean_held_out_state_effect"] = profiles.weighted_effect/denom
    profiles["member_coverage"] = profiles.available_members/profiles.member_count
    profiles["selected_weight_coverage"] = profiles.available_score_weight/profiles.total_score_weight.where(profiles.total_score_weight.gt(0))
    profiles["availability"] = profiles.available_score_weight.gt(0)
    profiles["availability_reason"] = np.select([profiles.available_members.eq(0), ~profiles.availability],
        ["NO_MEMBERS_WITH_AVAILABLE_STATE", "ZERO_AVAILABLE_SELECTED_SCORE_WEIGHT"], default=None)
    effect = profiles.mean_held_out_state_effect
    profiles["descriptive_effect_direction"] = np.select([effect.gt(0), effect.lt(0), effect.eq(0)],
        ["positive", "negative", "neutral"], default=None)
    profiles["summary_semantics"] = "SELECTED_SCORE_WEIGHTED_MEMBER_DESCRIPTIVE_MEAN_NOT_JOINT_GENESET_EFFECT"
    profiles["p_value_available"] = False
    profiles = profiles.drop(columns=["weighted_membership","weighted_effect"])
    data = data.drop(columns=["available_score_weight","weighted_membership","weighted_effect"])
    # Failed GeneSet groups also have seven explicit coverage rows.
    cov = coverage[["cancer_id","pathway_id","association_direction","geneset_available","unavailable_reason","exported_members"]].copy()
    cov = cov.rename(columns={"association_direction":"pathway_association_direction", "unavailable_reason":"geneset_unavailable_reason"}).merge(states, how="cross")
    cov = cov.merge(profiles[["geneset_id", *profile_keys[1:], "available_members", "member_coverage", "selected_weight_coverage", "availability", "availability_reason"]],
        on=profile_keys[1:], how="left", validate="one_to_one")
    missing_set = ~cov.geneset_available
    cov.loc[missing_set,"availability_reason"] = "SELECTED_GENESET_UNAVAILABLE"
    cov["availability"] = cov.availability.eq(True)
    cov["available_members"] = cov.available_members.fillna(0).astype(int)
    return {"selected_member_states": data, "selected_geneset_state_profiles": profiles, "selected_state_coverage": cov}


def native_state_sets(aggregate, config):
    from .state_gene_set_release import _cancer_specific_sets, _pan_cancer_sets
    # A true zero correlation is neutral. It is retained in context tables but
    # cannot be counted as negative evidence in signed State sets.
    signed = aggregate.loc[aggregate.availability & aggregate.association_direction.isin(["positive","negative"])].copy()
    cancer, cancer_members = _cancer_specific_sets(signed, max_members=config.cancer_max_members, min_members=config.minimum_set_members)
    pan, pan_members = _pan_cancer_sets(signed, max_members=config.pan_cancer_max_members,
        min_members=config.minimum_set_members, min_cancers=config.pan_cancer_min_cancers,
        minimum_direction_consistency=config.pan_cancer_min_direction_consistency)
    return {"state_geneset_catalog": pd.DataFrame(cancer+pan, columns=CATALOG_COLUMNS),
        "state_geneset_members": pd.DataFrame(cancer_members+pan_members, columns=MEMBER_COLUMNS)}


def _tables(context):
    aggregate = native_aggregate(context)
    selected = context["selected"]
    members = pd.read_parquet(selected.paths["geneset_members"])
    coverage = pd.read_parquet(selected.paths["geneset_coverage"])
    tables = {"native_state_aggregate": aggregate, "state_input_coverage": context["input_coverage"].copy(),
              **selected_state_tables(members, coverage, aggregate),
              **native_state_sets(aggregate, context["policy"])}
    for data in tables.values():
        data["state_training_run_id"] = context["manifest"]["state_run_id"]
        data["selected_analysis_sha256"] = selected.binding_sha256
        data["selected_score_run_id"] = selected.binding["selected_score_run_id"]
        data["changes_primary_ranking"] = False
    return tables


def _gmt(tables, context):
    members = tables["state_geneset_members"]
    lines = []
    for row in tables["state_geneset_catalog"].itertuples(index=False):
        ids = members.loc[members.gene_set_id.eq(row.gene_set_id)].sort_values("member_rank").member_gene_id.tolist()
        description = "native_state="+context["manifest"]["state_run_id"]+";selected_analysis="+context["selected"].binding_sha256
        values = [row.gene_set_id, description, *ids]
        if any(any(c in str(v) for c in "\t\r\n") for v in values): raise ValueError("Unsafe State GMT identity")
        lines.append("\t".join(map(str,values))+"\n")
    return "".join(lines)


def implementation_hashes():
    root = Path(__file__).resolve().parent
    from .current_model_use import implementations
    return {**implementations(), **{name: sha256(root/name) for name in ["current_state_context.py", "state_training.py", "state_gene_set_release.py",
        "integrated_model.py", "current_core_export.py", "native_auxiliary_fusion.py",
        "native_state_workflow.py", "patient_folds.py", "patient_fold_authority.py", "state_patient_use.py", "current_state_model_use.py"]}}


def materialize_current_state_context(manifest_path, output):
    if socket.gethostname() != "149": raise RuntimeError("Current State context must execute on target host 149")
    manifest_path = Path(manifest_path).resolve(); output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output/"LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False, "stage":"CURRENT_STATE_CONTEXT",
        "input_manifest_sha256":sha256(manifest_path)})
    context = state_sources(manifest_path); tables = _tables(context); outputs = {}
    from .state_patient_use import TABLE
    private_outputs = {}
    for name, data in {TABLE: context["statistical_patient_scopes"], **context["input_use"]}.items():
        private_path = output/(name+".parquet"); data.to_parquet(private_path, index=False)
        private_outputs[name] = {"path": str(private_path), "sha256": sha256(private_path), "rows": len(data), "private": True}
    for name, data in tables.items():
        path = output/(name+".parquet"); data.to_parquet(path,index=False)
        outputs[name] = {"path":str(path), "sha256":sha256(path), "rows":len(data)}
    path = output/"state_genesets.gmt"; path.write_text(_gmt(tables,context),encoding="utf-8")
    outputs["gmt"] = {"path":str(path),"sha256":sha256(path)}
    binding = {"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
        "input_manifest":{"path":str(manifest_path),"sha256":sha256(manifest_path)},
        "selected_analysis":context["manifest"]["selected_analysis"],"selected_score_run_id":context["selected"].binding["selected_score_run_id"],
        "core_run_id":context["manifest"]["core_run_id"],"state_training_run_id":context["manifest"]["state_run_id"],
        "state_set_config":asdict(context["policy"]),"outputs":outputs,"private_outputs":private_outputs,
        "source_artifacts":list(context["sources"].records.values()),
        "state_fold_aggregate_recomputed":True,"raw_measurement_statistics_replayed":True,"private_head_predictions_replayed":True,
        "native_numeric_replay":context["replay"],
        "context_changes_ranking":False,"gene_set_effect_inference_performed":False,"production_deployed":False,
        "implementation_sha256":implementation_hashes()}
    path = output/"CURRENT_STATE_CONTEXT_BINDING.json"; _write(path,binding)
    audit = audit_current_state_context(path,expected_sha=sha256(path)); audit_path = output/"RECOMPUTATION_AUDIT.json"; _write(audit_path,audit)
    result = {"status":"PASS","target_host":"149","training_started":False,"binding_path":str(path),"binding_sha256":sha256(path),"audit_sha256":sha256(audit_path)}
    _write(output/"SUCCESS.json", result)
    return result


def _load(binding_path, expected_sha):
    path = verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); binding = _json(path)
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("training_started") is not False or binding.get("context_changes_ranking") is not False
            or binding.get("gene_set_effect_inference_performed") is not False
            or binding.get("state_fold_aggregate_recomputed") is not True
            or binding.get("raw_measurement_statistics_replayed") is not True
            or binding.get("private_head_predictions_replayed") is not True
            or binding.get("production_deployed") is not False):
        raise ValueError("Invalid current State context binding")
    from .state_patient_use import TABLE
    from .current_state_model_use import PRIVATE_NAMES
    private = binding.get("private_outputs", {})
    if (set(private) != {TABLE}|PRIVATE_NAMES or any(r.get("private") is not True for r in private.values())
            or set(private) & set(binding["outputs"])
            or {r.get("sha256") for r in private.values()} & {r.get("sha256") for r in binding["outputs"].values()}):
        raise ValueError("State patient scopes must remain private")
    for item in private.values(): verified_path(item, path.parent)
    paths = {name:verified_path(r,path.parent) for name,r in binding["outputs"].items()}
    return path,binding,paths


def audit_current_state_context(binding_path, *, expected_sha):
    path,binding,outputs = _load(binding_path,expected_sha)
    if binding["implementation_sha256"] != implementation_hashes(): raise ValueError("State replay implementation changed")
    manifest_path = verified_path(binding["input_manifest"],path.parent); context = state_sources(manifest_path)
    if (binding["selected_analysis"] != context["manifest"]["selected_analysis"]
            or binding["core_run_id"] != context["manifest"]["core_run_id"]
            or binding["state_training_run_id"] != context["manifest"]["state_run_id"]
            or binding["selected_score_run_id"] != context["selected"].binding["selected_score_run_id"]
            or binding["state_set_config"] != asdict(context["policy"])
            or binding["source_artifacts"] != list(context["sources"].records.values())
            or binding.get("native_numeric_replay") != context["replay"]):
        raise ValueError("State context input identity or policy drift")
    tables = _tables(context)
    from .state_patient_use import TABLE
    for name, expected in {TABLE: context["statistical_patient_scopes"], **context["input_use"]}.items():
        private = binding["private_outputs"][name]
        observed = pd.read_parquet(verified_path(private, path.parent))
        if len(observed) != private["rows"]: raise ValueError("State private patient scope row count drift")
        try: _assert_same_table(observed, expected)
        except AssertionError as error: raise ValueError("State private patient scope replay differs: "+name) from error
    if set(outputs) != set(tables)|{"gmt"}: raise ValueError("State context output inventory drift")
    for name,expected in tables.items():
        observed = pd.read_parquet(outputs[name])
        if len(observed) != binding["outputs"][name]["rows"]: raise ValueError("State context row count drift")
        try: _assert_same_table(observed,expected)
        except AssertionError as error: raise ValueError("State context source replay differs: "+name) from error
    if outputs["gmt"].read_text(encoding="utf-8") != _gmt(tables,context): raise ValueError("State GMT source replay differs")
    return {"status":"PASS","binding_sha256":expected_sha,"native_fold_aggregate_independently_recomputed":True,
        "selected_member_profiles_recomputed":True,"profile_same_implementation_replay":True,
        "private_head_predictions_replayed":True,"raw_measurement_statistics_replayed":True,
        "native_numeric_replay":context["replay"],"training_started":False}


class CurrentStateContextQuery:
    """Version-bound State context queries alongside the selected analysis."""
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha):
        path,self.binding,self.paths = _load(binding_path,expected_sha)
        if self.binding["selected_analysis"]["sha256"] != selected_analysis_sha:
            raise ValueError("State query requested a different selected analysis")
        success = _json(path.parent/"SUCCESS.json"); audit_path = path.parent/"RECOMPUTATION_AUDIT.json"
        audit = _json(audit_path)
        if (success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha
                or success.get("audit_sha256")!=sha256(audit_path) or audit.get("status")!="PASS"
                or audit.get("binding_sha256")!=expected_sha):
            raise ValueError("State context lacks completed source replay")
        self.sha = expected_sha

    def download(self,name):
        if name not in self.paths: raise ValueError("Unknown current State download")
        record = self.binding["outputs"][name]; verified_path(record,Path.cwd())
        return {**record,"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"]}

    def query(self,*,table="selected_geneset_state_profiles",filters=None,limit=100,offset=0):
        if table=="gmt" or table not in self.paths: raise ValueError("Unknown current State table")
        if type(limit) is not int or type(offset) is not int or not 1<=limit<=1000 or not 0<=offset<=1000000:
            raise ValueError("Invalid State query pagination")
        filters = filters or {}; allowed = {"cancer_id","lncrna_id","pathway_id","state_id","geneset_id","gene_set_id",
            "state_association_direction","pathway_association_direction","association_direction","direction"}
        if not set(filters).issubset(allowed) or not all(isinstance(v,str) for v in filters.values()):
            raise ValueError("State filters require exact string identities")
        path = self.download(table)["path"]; connection = duckdb.connect()
        try:
            columns = [r[0] for r in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)",[path]).fetchall()]
            if not set(filters).issubset(columns): raise ValueError("State filter is not a column of this table")
            where = " AND ".join('"'+key+'"=?' for key in filters) or "true"
            order = ",".join('"'+key+'"' for key in ["cancer_id","pathway_id","geneset_id","gene_set_id","state_id","geneset_rank","member_rank","lncrna_id"] if key in columns) or "1"
            data = connection.execute(f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [path,*filters.values(),limit,offset]).df()
        finally: connection.close()
        return {"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"],
            "rows":data.astype(object).where(data.notna(),None).to_dict("records")}
