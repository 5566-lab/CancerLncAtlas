"""Replay native formal features, supervision and the frozen L1 linear score.

This does not fit L1, construct a neural model, or certify the upstream assays.
Association/normalization ancestry remains the responsibility of their readers.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .preparation_sources import canonical_sha
from .formal_expression_sources import record

FORMAT = "NATIVE_FORMAL_FEATURE_LABEL_REPLAY_V1"
AUTHORITY = "formal_feature_label_construction_sha256"
ROOT = Path(__file__).resolve().parents[2]
CODE = ("cc_hhgt/v32/prepared_construction.py", "scripts/prepare_v32_formal.py",
        "cc_hhgt/v32/baselines.py", "cc_hhgt/v32/local_cnv_core_preparation.py")
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def adjusted_frames(candidates, scope, associations, *, cancers, fold_local, baseline_only=False):
    """The producer and replay use identical source joins and label-only attachment."""
    from scripts.prepare_v32_formal import attach_replication_labels
    columns = ["cancer_id", "lncrna_id", "detection_rate", "detected_cancers", "shared_or_local_scope"]
    if "assessed_cancers" in scope: columns.append("assessed_cancers")
    metadata = candidates.merge(scope[columns], on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    discovery = metadata.merge(associations["train"], on=KEYS, how="left", validate="one_to_one")
    if baseline_only:
        if discovery.association_available.isna().any(): raise ValueError("Training association candidate coverage differs")
        return add_cross_cancer_features({"train": discovery}, scope, denominator=len(cancers) if fold_local else 33.)
    replication = metadata.merge(associations["validation"], on=KEYS, how="left", validate="one_to_one")
    if discovery.association_available.isna().any() or replication.association_available.isna().any():
        raise ValueError("Adjusted association authority does not close over candidates")
    frames = {"train": discovery, "validation": attach_replication_labels(discovery, replication)}
    return add_cross_cancer_features(frames, scope, denominator=len(cancers) if fold_local else 33.)


def add_cross_cancer_features(frames, scope, *, denominator):
    for frame in frames.values():
        frame["cross_cancer_support_frequency"] = frame.lncrna_id.map(
            scope.groupby("lncrna_id").detected_cancers.max()).fillna(0).to_numpy(float)/denominator
        frame["cross_cancer_direction_consistency"] = 0.
        frame["cross_cancer_i2"] = 0.
    return frames


def training_weights(frame):
    y = frame.proxy_label.to_numpy(int)
    weight = np.where(frame.label_class.astype(str).eq("weak_positive"), .35, np.where(y == 0, .12, 1.))
    return np.where(frame.association_available.to_numpy(bool), weight, 0.)


def array_hash(value, dtype="<f8"):
    value = np.asarray(value, dtype=dtype)
    return hashlib.sha256(str(value.shape).encode()+value.tobytes(order="C")).hexdigest()


def design_hash(value):
    value = value.copy().tocsr(); value.sum_duplicates(); value.sort_indices()
    return canonical_sha({"shape": list(value.shape), "data": array_hash(value.data),
        "indices": array_hash(value.indices, "<i8"), "indptr": array_hash(value.indptr, "<i8")})


def fit_inputs(frames, train_design):
    from .contracts import candidate_key_sha256
    frame = frames["train"]
    return {"candidate_sha256": candidate_key_sha256(frame), "rows": len(frame),
        "design_sha256": design_hash(train_design), "labels_sha256": array_hash(frame.proxy_label.to_numpy()),
        "sample_weights_sha256": array_hash(training_weights(frame)),
        "available_rows": int(frame.association_available.sum())}


def contract(source_receipt, frames, model, train_design, *, seed):
    coefficients = np.asarray(model.coef_, dtype=float); intercept = np.asarray(model.intercept_, dtype=float)
    if (coefficients.shape != (1,4096) or intercept.shape != (1,)
            or list(model.classes_) != [0,1] or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()):
        raise ValueError("Native formal baseline must expose finite binary 4096-feature L1 state")
    return {"format": FORMAT, "patient_fold": source_receipt["patient_fold"],
        "preparation_sources_sha256": canonical_sha(source_receipt), "implementations": {p: record(ROOT/p) for p in CODE},
        "baseline": {"method": "LOGISTIC_L1_FIXED_C", "C": .1, "solver": "liblinear", "n_features": 4096,
            "seed": seed, "fit_split": "train", "validation_used_for_fitting": False, "classes": [0,1],
            "coefficients": coefficients[0].tolist(), "intercept": float(intercept[0])},
        "fit_inputs": fit_inputs(frames, train_design), "parameter_fitting_replayed": False}


def attach(payload, value):
    payload["feature_label_construction"] = value
    payload["artifact_hashes"][AUTHORITY] = value["sha256"]
    if "input_authority_hashes" in payload: payload["input_authority_hashes"][AUTHORITY] = value["sha256"]


def validate(payload, receipt):
    value = payload.get("feature_label_construction")
    hashes = [payload.get(k, {}) for k in ("artifact_hashes", "input_authority_hashes")]
    declared = receipt is not None and receipt.get("feature_construction_format") is not None
    if value is None:
        if declared or any(AUTHORITY in h for h in hashes): raise ValueError("Native feature/label construction receipt missing")
        return None
    if (not declared or receipt["feature_construction_format"] != FORMAT or value.get("format") != FORMAT
            or set(value) != {"format", "path", "sha256"} or not value.get("path")
            or not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64):
        raise ValueError("Invalid native feature/label construction reference")
    authority = payload.get("input_authority_hashes", payload.get("artifact_hashes", {}))
    if authority.get(AUTHORITY) != value["sha256"] or any(AUTHORITY in h and h[AUTHORITY] != value["sha256"] for h in hashes):
        raise ValueError("Native feature/label construction authority hash differs")
    return value


def validate_contract(value, receipt, fold):
    if (value.get("format") != FORMAT or value.get("patient_fold") != fold
            or value.get("preparation_sources_sha256") != canonical_sha(receipt)
            or value.get("parameter_fitting_replayed") is not False
            or set(value.get("implementations", {})) != set(CODE)):
        raise ValueError("Invalid native feature/label construction authority")
    baseline = value["baseline"]
    fixed = {"method": "LOGISTIC_L1_FIXED_C", "C": .1, "solver": "liblinear", "n_features": 4096,
             "seed": 20260726, "fit_split": "train", "validation_used_for_fitting": False, "classes": [0,1]}
    if (set(baseline) != set(fixed)|{"coefficients", "intercept"}
            or any(baseline.get(k) != v for k,v in fixed.items())
            or np.shape(baseline["coefficients"]) != (4096,)
            or not np.isfinite(np.asarray(baseline["coefficients"],float)).all()
            or not np.isfinite(baseline["intercept"])):
        raise ValueError("Invalid native frozen L1 state/fit policy")
    return value


def replay(payload, root, *, inventory=None, model=None, baseline_only=False):
    from .cnv_input_sources import require_host
    from .preparation_sources import validate_payload_sources
    from .input_lineage import artifact_sha256
    from .local_cnv_core_preparation import adjusted_association_frame
    from .baselines import build_safe_hashed_design
    from scripts.prepare_v32_formal import build_hierarchy, make_batches
    require_host()
    receipt = validate_payload_sources(payload)
    reference = validate(payload, receipt)
    if reference is None: return None
    fold = payload["patient_fold"]
    if (receipt["association_mode"] != "local-cnv-adjusted"
            or payload["input_scope"].get("candidate_definition") != "annotation_and_gmt_only"):
        raise ValueError("Native construction replay requires annotation-defined adjusted inputs")
    def locate(item, role):
        source = item
        if inventory is not None:
            source = inventory.locations.get(item["sha256"], item)
            if item["sha256"] in inventory.locations: inventory.used_locations.add(item["sha256"])
            return inventory.bind(source, root, model, "prepared.construction."+role, fold)
        path = Path(source["path"]); path = path if path.is_absolute() else Path(root)/path
        if artifact_sha256(path) != source["sha256"]: raise ValueError("Construction source hash differs")
        return path
    from .formal_expression_sources import read
    value = read(locate(reference, "linear_baseline_and_fit_inputs"))
    validate_contract(value, receipt, fold)
    for name, item in value["implementations"].items():
        locate(item, "implementation."+name)
        if record(ROOT/name)["sha256"] != item["sha256"]: raise ValueError("Construction implementation changed")
    sources = receipt["source_reads"]
    def source(role):
        rows = [r for r in sources if r["role"] == role]
        if len(rows) != 1: raise ValueError("Construction source role is not unique: "+role)
        return rows[0]
    candidate_item, scope_item = source("candidate_authority"), source("detection")
    candidates = pd.read_parquet(locate(candidate_item, "candidates"))
    scope = pd.read_parquet(locate(scope_item, "scope"))
    if (len(scope_item["members"]) != 1 or not scope_item["members"][0].get("upstream_scope")
            or len(candidate_item["members"]) != 1 or not candidate_item["members"][0].get("upstream_candidates")):
        raise ValueError("Construction lacks native candidate/detection definitions")
    from .fold_local_scope import validate_candidates, validate_scope
    validate_candidates(candidate_item["members"][0]["upstream_candidates"], root,
                        locate(candidate_item, "candidates"), replay=True)
    validate_scope(scope_item["members"][0]["upstream_scope"], root, locate(scope_item, "scope"), replay=True)
    hierarchy_path = locate(source("pathway_hierarchy"), "hierarchy")
    from .formal_graph_authority import _read_table
    hierarchy = build_hierarchy(sorted(candidates.pathway_id.unique()), _read_table(hierarchy_path))
    candidates = candidates.drop(columns=["pathway_family_id"], errors="ignore").merge(hierarchy, on="pathway_id", how="left", validate="many_to_one")
    associations = {}
    for split, prefix in (("train", "A1_local"), ("validation", "A1_validation")):
        if baseline_only and split != "train": continue
        parts = []
        for cancer in receipt["cancer_ids"]:
            rows = [r for r in sources if r["role"] == split+"_associations" and r["cancer_id"] == cancer]
            if len(rows) != 1: raise ValueError("Construction association cancer scope differs")
            raw = pd.read_parquet(locate(rows[0], split+"."+cancer))
            parts.append(adjusted_association_frame(raw, prefix=prefix, expected_cancer=cancer, expected_fold=fold))
        associations[split] = pd.concat(parts, ignore_index=True)
    frames = adjusted_frames(candidates, scope, associations, cancers=receipt["cancer_ids"], fold_local=True, baseline_only=baseline_only)
    designs = {k: build_safe_hashed_design(frame, n_features=4096) for k,frame in frames.items()}
    if fit_inputs(frames, designs["train"]) != value["fit_inputs"]:
        raise ValueError("Native L1 training design/labels/weights differ from consumed sources")
    compared = 0
    import torch
    for split, frame in frames.items():
        if baseline_only and split != "train": continue
        logits = designs[split] @ np.asarray(value["baseline"]["coefficients"]) + value["baseline"]["intercept"]
        batches = payload[split+"_batches"]
        # Preserve actual batching; compare tensors by row slices without making
        # one additional full-cohort tensor pack.
        offset = 0
        for actual in batches:
            n = len(actual["base_logit"])
            if n <= 0: raise ValueError("Empty native construction batch")
            expected = make_batches(frame.iloc[offset:offset+n], logits[offset:offset+n], payload["bundle"], split=split, batch_size=n)
            if len(expected) != 1 or set(actual) != set(expected[0]): raise ValueError("Native constructed batch fields differ")
            def compare(a, b):
                nonlocal compared
                if isinstance(b, dict):
                    if not isinstance(a, dict) or set(a) != set(b): raise ValueError("Native candidate tensor fields differ")
                    for key in b: compare(a[key], b[key])
                else:
                    if not torch.is_tensor(a) or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
                        raise ValueError("Native constructed feature/label/base-logit tensor differs")
                    compared += 1
            compare(actual, expected[0]); offset += n
        if offset != len(frame): raise ValueError("Native batch candidate coverage differs")
    return {"format": FORMAT, "patient_fold": fold, "source_receipt_sha256": canonical_sha(receipt),
        "construction_sha256": reference["sha256"], "tensor_fields_compared": compared,
        "linear_baseline_replayed": True, "parameter_fitting_replayed": False, "training_started": False}
