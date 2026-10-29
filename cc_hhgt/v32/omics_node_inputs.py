"""Produce and load measured node inputs without fitting any parameters.

Input values are already assay-level measurements, not raw sequencing reads.
Feature-to-node mapping and sample eligibility must be supplied explicitly.
The private source inventory remains necessary for subsequent ancestry review.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd

from . import omics_node_encoding as encoding
from .current_core_export import sha256, verified_path

FORMAT = "CANCERLNCATLAS_NODE_INPUT_PREPARATION_V1"
POOL = "PANCAN_TRAINING_POOL"
UNITS = {
    "cnv_signed": {"log2_copy_ratio", "gistic_thresholded"},
    "methylation_promoter_beta": {"beta"},
    "mutation_damage": {"declared_damage_score"},
    "atac_accessibility": {"normalized_accessibility"},
}
KEYS = ["node_type", "canonical_id", "channel"]


def _read(path, columns):
    frame = encoding._read_table(Path(path))
    if not set(columns).issubset(frame.columns):
        raise ValueError("NODE_INPUT_COLUMNS_MISSING: " + str(columns))
    return frame.loc[:, columns].copy()


def _identifiers(frame, columns):
    for column in columns:
        if frame[column].isna().any():
            raise ValueError("NODE_INPUT_IDENTIFIER_MISSING: " + column)
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError("NODE_INPUT_IDENTIFIER_MISSING: " + column)


def _artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def _assay(entry, source):
    """Read existing native patient products without inventing sample rows."""
    channel = entry["channel"]
    path = source(entry["measurements"], channel + ":measurements")
    layout = entry.get("layout", "sample_feature_long")
    if layout == "sample_feature_long":
        return _read(path, ["sample_id", "feature_id", "raw_value"]), False
    if layout == "native_methylation_patient":
        if channel != "methylation_promoter_beta" or entry["unit"] != "beta":
            raise ValueError("NODE_INPUT_NATIVE_METHYLATION_CHANNEL")
        from .native_methylation_workflow import validate
        _, binding = validate(_artifact(path), path.parent)
        for name in ("preflight", "audit"):
            item = binding[name]
            resolved = verified_path(item, path.parent)
            source(_artifact(resolved), channel + ":" + name)
        table = verified_path(binding["outputs"]["patient_features"], path.parent)
        source(_artifact(table), channel + ":patient_features")
        field = "promoter_methylation_beta"
        value = _read(table, ["cancer_id", "patient_id", "patient_fold_id", "lncrna_id",
                              field, field + "__available"])
        available = value[field + "__available"]
        numeric = pd.to_numeric(value[field], errors="coerce")
        if (available.dtype != bool or not np.array_equal(available, np.isfinite(numeric))):
            raise ValueError("NODE_INPUT_NATIVE_METHYLATION_AVAILABILITY")
        value = value.rename(columns={"patient_id": "sample_id", "lncrna_id": "feature_id", field: "raw_value"})
        return value.drop(columns=[field + "__available"]), True
    if layout == "native_directional_cnv":
        if channel != "cnv_signed" or entry["unit"] != "log2_copy_ratio":
            raise ValueError("NODE_INPUT_NATIVE_CNV_CHANNEL")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (receipt.get("format") != "CC_HHGT_V3_2_DIRECTIONAL_SEGMENT_CNV_V1"
                or receipt.get("status") != "SUCCESS" or receipt.get("signed_values_preserved") is not True
                or receipt.get("amplification_deletion_separate") is not True
                or receipt.get("absolute_burden_used_as_continuous") is not False
                or not receipt.get("cancer_id")):
            raise ValueError("NODE_INPUT_NATIVE_CNV_INCOMPLETE")
        files = {}
        for name in ("patient_ids.json", "lncrna_ids.json", "lncrna_signed_mean.npy", "lncrna_callable.npy"):
            files[name] = source({"path": str(path.parent / name), "sha256": receipt.get("files", {}).get(name)},
                                 channel + ":" + name)
        patients, nodes = [json.loads(files[name].read_text(encoding="utf-8"))
                           for name in ("patient_ids.json", "lncrna_ids.json")]
        for ids in (patients, nodes):
            if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids)
                    or len(ids) != len(set(ids))):
                raise ValueError("NODE_INPUT_NATIVE_CNV_IDENTIFIERS")
        values = np.load(files["lncrna_signed_mean.npy"], mmap_mode="r", allow_pickle=False)
        available = np.load(files["lncrna_callable.npy"], mmap_mode="r", allow_pickle=False)
        shape = (len(patients), len(nodes))
        if (values.shape != shape or available.shape != shape or not np.isin(available, [0, 1]).all()
                or not np.array_equal(available.astype(bool), np.isfinite(values))):
            raise ValueError("NODE_INPUT_NATIVE_CNV_SHAPE_OR_AVAILABILITY")
        return pd.DataFrame({"sample_id": np.repeat(patients, len(nodes)),
            "feature_id": np.tile(nodes, len(patients)), "raw_value": np.asarray(values).reshape(-1),
            "cancer_id": receipt["cancer_id"]}), True
    raise ValueError("NODE_INPUT_LAYOUT_UNKNOWN")


def _prepare(config, base):
    """Return evidence plus private lineage; no output and no fitting."""
    if config.get("format") != FORMAT or config.get("target_host") != "149":
        raise ValueError("NODE_INPUT_FORMAT_OR_HOST")
    fold = config.get("outer_fold")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("NODE_INPUT_OUTER_FOLD")
    scope, cancer = config.get("scope"), config.get("cancer_id")
    subject = config.get("subject_id") or ""
    if scope not in encoding.SCOPES or not isinstance(cancer, str) or not cancer.strip():
        raise ValueError("NODE_INPUT_SCOPE")
    if (scope == "subject") != bool(subject) or (scope == "subject" and cancer == POOL):
        raise ValueError("NODE_INPUT_SUBJECT_SCOPE")
    sources = {}

    def source(record, role):
        path = verified_path(record, Path(base))
        sources[role] = _artifact(path)
        return path

    authority = _read(source(config["sample_authority"], "sample_authority"),
                      ["sample_id", "patient_id", "cancer_id", "is_tumor", "partition"])
    _identifiers(authority, ["sample_id", "patient_id", "cancer_id", "partition"])
    if authority.empty or authority.sample_id.duplicated().any():
        raise ValueError("NODE_INPUT_SAMPLE_AUTHORITY_NOT_UNIQUE")
    if not authority.partition.isin(["train", "validation", "test"]).all():
        raise ValueError("NODE_INPUT_PARTITION_INVALID")
    if not authority.is_tumor.isin([True, False, 0, 1]).all():
        raise ValueError("NODE_INPUT_TUMOR_FLAG_INVALID")
    patients = authority.groupby("patient_id")[["cancer_id", "partition"]].nunique()
    if (patients > 1).any().any():
        raise ValueError("NODE_INPUT_PATIENT_SPLIT_OR_CANCER_CONFLICT")
    selected = authority.loc[authority.is_tumor.astype(bool)].copy()
    if cancer != POOL:
        selected = selected.loc[selected.cancer_id.eq(cancer)]
    if scope == "cohort":
        selected = selected.loc[selected.partition.eq("train")]
    else:
        selected = selected.loc[selected.patient_id.eq(str(subject))]
    if selected.empty:
        raise ValueError("NODE_INPUT_ELIGIBLE_PATIENTS_EMPTY")
    assays = config.get("assays") or []
    channels = [entry.get("channel") for entry in assays]
    if (not channels or len(channels) != len(set(channels))
            or set(channels) - set(UNITS)):
        raise ValueError("NODE_INPUT_CHANNELS_INVALID")
    values, lineage, counts = [], [], {}
    for entry in assays:
        channel = entry["channel"]
        unit = entry.get("unit")
        if unit not in UNITS[channel]:
            raise ValueError("NODE_INPUT_UNIT_INVALID: " + channel)
        raw, patient_level = _assay(entry, source)
        mapping = _read(source(entry["mapping"], channel + ":mapping"),
                        ["feature_id", "node_type", "canonical_id"])
        _identifiers(raw, ["sample_id", "feature_id"])
        _identifiers(mapping, ["feature_id", "node_type", "canonical_id"])
        if raw.duplicated(["sample_id", "feature_id"]).any() or mapping.duplicated().any():
            raise ValueError("NODE_INPUT_DUPLICATE_MEASUREMENT_OR_MAPPING")
        if not mapping.node_type.isin(["gene", "lncRNA", "protein"]).all():
            raise ValueError("NODE_INPUT_MAPPING_NODE_TYPE")
        if patient_level and not mapping.node_type.eq("lncRNA").all():
            raise ValueError("NODE_INPUT_NATIVE_LNCRNA_MAPPING_REQUIRED")
        eligible = selected
        known_samples = set(authority.sample_id)
        if patient_level:
            # One upstream patient value must never be repeated once per aliquot.
            eligible = selected.drop(columns=["sample_id", "is_tumor"]).drop_duplicates()
            eligible["sample_id"] = eligible.patient_id
            known_samples = set(authority.patient_id)
            cancers = authority.groupby("patient_id").cancer_id.first()
            if not raw.cancer_id.eq(raw.sample_id.map(cancers)).all():
                raise ValueError("NODE_INPUT_NATIVE_PATIENT_CANCER_MISMATCH")
            raw = raw.drop(columns=["cancer_id"])
            if "patient_fold_id" in raw:
                train_ids = set(eligible.loc[eligible.partition.eq("train"), "patient_id"])
                if (raw.sample_id.isin(train_ids) & raw.patient_fold_id.eq(fold)).any():
                    raise ValueError("NODE_INPUT_NATIVE_HELDOUT_PATIENT_IN_TRAIN")
        if set(raw.sample_id) - known_samples:
            raise ValueError("NODE_INPUT_SAMPLE_UNMAPPED")
        if set(raw.feature_id) - set(mapping.feature_id):
            raise ValueError("NODE_INPUT_FEATURE_UNMAPPED")
        numeric = pd.to_numeric(raw.raw_value, errors="coerce")
        missing = raw.raw_value.isna() | raw.raw_value.astype(str).isin(["", "NA", "NaN", "nan"])
        if ((numeric.isna() & ~missing) | np.isinf(numeric)).any():
            raise ValueError("NODE_INPUT_VALUE_INVALID")
        finite = numeric.dropna()
        if channel == "methylation_promoter_beta" and not finite.between(0, 1).all():
            raise ValueError("NODE_INPUT_BETA_OUT_OF_RANGE")
        if unit == "gistic_thresholded" and not finite.isin([-2, -1, 0, 1, 2]).all():
            raise ValueError("NODE_INPUT_GISTIC_VALUE_INVALID")
        raw["raw_value"] = numeric
        raw["source_row"] = np.arange(len(raw), dtype=np.int64)
        used = raw.dropna(subset=["raw_value"]).merge(eligible, on="sample_id", validate="many_to_one")
        used = used.merge(mapping, on="feature_id", validate="many_to_many")
        used["channel"] = channel
        # Equal feature weight inside a sample, equal sample weight per patient,
        # then equal patient weight. Technical replicates never weight patients.
        sample = used.groupby(["patient_id", "sample_id", *KEYS], as_index=False).raw_value.mean()
        patient = sample.groupby(["patient_id", *KEYS], as_index=False).raw_value.mean()
        result = patient.groupby(KEYS, as_index=False).raw_value.mean()
        values.append(result)
        lineage.append(used)
        counts[channel] = {"input_rows": len(raw), "missing_rows": int(missing.sum()),
                           "contributing_rows": len(used), "contributing_patients": used.patient_id.nunique(),
                           "observed_nodes": len(result), "unit": unit,
                           "input_granularity": "patient" if patient_level else "sample"}
    evidence = pd.concat(values, ignore_index=True).sort_values(KEYS, kind="stable").reset_index(drop=True)
    if evidence.empty:
        raise ValueError("NODE_INPUT_NO_MEASURED_NODES")
    manifest = {"format": encoding.EVIDENCE_FORMAT, "scope": scope, "cancer_id": cancer,
                "subject_id": subject or None, "channels": channels, "outer_fold": fold,
                "cancer_ids": sorted(selected.cancer_id.unique().tolist()), "clipped": True,
                "standardisation": {channel: {"center": 0.0, "scale": 1.0} for channel in channels},
                "standardisation_policy": "IDENTITY_NO_PARAMETER_FIT",
                "aggregation": "MEAN_FEATURES_PER_SAMPLE_THEN_SAMPLES_PER_PATIENT_THEN_PATIENTS",
                "cohort_eligibility": "TRAINING_TUMORS_ONLY" if scope == "cohort" else "DECLARED_SUBJECT_TUMORS",
                "source_artifacts": sources, "coverage": counts,
                "producer_format": FORMAT, "training_started": False,
                "raw_assay_preprocessing_verified": False}
    return evidence, pd.concat(lineage, ignore_index=True), manifest


def produce(config_path, output):
    """Materialize a new versioned evidence directory on 149 only."""
    if socket.gethostname() != "149":
        raise RuntimeError("NODE_INPUT_EXECUTION_REQUIRES_149")
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({
        "target_host": "149", "format": FORMAT, "training_started": False,
        "source_config": _artifact(config_path)}, indent=2), encoding="utf-8")
    frame, lineage, manifest = _prepare(config, config_path.parent)
    table = output / "node_evidence.parquet"
    frame.to_parquet(table, index=False)
    lineage.to_parquet(output / "contributions_private.parquet", index=False)
    manifest.update(table_path=table.name, table_sha256=sha256(table), row_count=len(frame),
                    source_config=_artifact(config_path),
                    private_contributions=_artifact(output / "contributions_private.parquet"))
    path = output / "node_evidence.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    encoding.read_evidence(table, path)
    (output / "COMPLETE.json").write_text(json.dumps({"target_host": "149", "format": FORMAT,
        "manifest": _artifact(path), "training_started": False}, indent=2), encoding="utf-8")
    return path


def load_declared_node_inputs(payload, config, input_manifest_path, fold):
    """Shared training/export loader; no declaration leaves old graphs untouched.

    A pooled cohort is a *global* training-patient feature. It is not a patient
    embedding or a cancer-specific feature on the shared gene node universe.
    """
    authority_path = Path(input_manifest_path).resolve()
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    rows = [r for r in authority.get("fold_inputs", []) if r.get("fold") == fold]
    if len(rows) != 1:
        raise ValueError("NODE_INPUT_AUTHORITY_FOLD")
    record = rows[0].get("node_omics")
    channels = config.get("task_contract", {}).get("node_omics_channels")
    if (record is not None) != (channels is not None):
        raise ValueError("NODE_INPUT_CONFIG_MANIFEST_DISAGREE")
    if record is None:
        if getattr(payload["bundle"], "node_input_dim", 2) != 2:
            raise ValueError("NODE_INPUT_UNDECLARED_AUGMENTATION")
        return payload
    path = verified_path(record, authority_path.parent)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("producer_format") != FORMAT or manifest.get("outer_fold") != fold
            or manifest.get("scope") != "cohort" or manifest.get("channels") != channels):
        raise ValueError("NODE_INPUT_MANIFEST_SCOPE_FOLD_CHANNELS")
    source_config = verified_path(manifest["source_config"], path.parent)
    producer_config = json.loads(source_config.read_text(encoding="utf-8"))
    expected, contributions, replay = _prepare(producer_config, source_config.parent)
    if any(manifest.get(key) != value for key, value in replay.items()):
        raise ValueError("NODE_INPUT_MANIFEST_REPLAY_MISMATCH")
    table = verified_path({"path": manifest["table_path"], "sha256": manifest["table_sha256"]}, path.parent)
    evidence = encoding.read_evidence(table, path)
    pd.testing.assert_frame_equal(evidence.frame.reset_index(drop=True), expected, check_dtype=False)
    private = verified_path(manifest["private_contributions"], path.parent)
    pd.testing.assert_frame_equal(pd.read_parquet(private), contributions, check_dtype=False)
    bundle = payload["bundle"]
    if set(bundle.node_maps.get("cancer", {})) != set(manifest["cancer_ids"]):
        raise ValueError("NODE_INPUT_GRAPH_CANCER_SCOPE_MISMATCH")
    block = encoding.node_feature_block(bundle.nodes, evidence)
    # Copy only modified stores; do not duplicate million-edge frames or labels.
    bundle = copy.copy(bundle)
    bundle.hetero_data = copy.copy(bundle.hetero_data)
    bundle.homogeneous = dict(bundle.homogeneous)
    encoding.attach_node_omics(bundle, block["blocks"])
    bundle.node_omics_channels = tuple(evidence.channels)
    result = dict(payload, bundle=bundle, graph=None)
    from .training import _prepared_input_authority_hashes
    result["input_authority_hashes"] = {**_prepared_input_authority_hashes(payload),
        "node_omics_manifest_sha256": sha256(path), "node_omics_table_sha256": sha256(table)}
    result["node_omics_authority"] = block["report"]
    return result
