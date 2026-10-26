"""Actual State head scopes joined to the original frozen-core input inspector.

Inspection does not construct neural models or fit parameters. Its tables remain
private; the public State context receives only aggregate coverage counts.
Unknown source ancestry and unmapped identities are not waived by file integrity.
"""
from pathlib import Path
import json

from .current_model_use import Inventory, inventory_tables, verified_source, record
from .current_core_model_use import inspect_export

PREFIX = "state_model_use_"
NAMES = {"usage", "dependencies", "gaps", "patient_usage_private", "patient_mapping",
    "patient_context_overlap", "local_cnv_patient_scopes_private", "transformation_replays", "completeness"}
PRIVATE_NAMES = {PREFIX + name for name in NAMES}


def bind_sources(sources, records, root):
    """Preserve the State file inventory and independently validate dataset hashes."""
    for item in records:
        path = verified_source(item, root)
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files: sources.bind(record(file), root)
        verified_source(item, root)


def options(value, root, sources):
    """Resolve only explicit source relocations and a file-qualified crosswalk."""
    if not isinstance(value, dict) or set(value)-{"artifact_locations", "patient_identity_crosswalk"}:
        raise ValueError("Invalid State source inventory options")
    locations = value.get("artifact_locations", [])
    if not isinstance(locations, list): raise ValueError("Invalid State artifact locations")
    normalized = {"artifact_locations": [{**item, "path": str(verified_source(item, root))} for item in locations]}
    if "patient_identity_crosswalk" in value:
        from .patient_source_identity import IdentityMap
        declaration = value["patient_identity_crosswalk"]
        mapper = IdentityMap(declaration, root)
        normalized["patient_identity_crosswalk"] = {**declaration, "path": str(mapper.path)}
    bind_sources(sources, [*normalized["artifact_locations"],
        *([normalized["patient_identity_crosswalk"]] if "patient_identity_crosswalk" in normalized else [])], root)
    return normalized


def derive(manifest, root, sources, lineage_path, lineage, raw_patients, scopes):
    """Follow the actual core export, then map exact State statistical cohorts."""
    config = options(manifest.get("source_inventory", {}), root, sources)
    inventory = Inventory(config["artifact_locations"], root, config.get("patient_identity_crosswalk"))
    model = manifest["state_run_id"]
    with inventory.context("STATE_CORE"):
        core = inventory.bind(manifest["core_manifest"], root, model, "state.core")
        if not inspect_export(inventory, core, model, "state.core"):
            inventory.gap(model, "state.core", "STATE_CORE_TRAINING_SOURCE_FILES_INCOMPLETE")
    with inventory.context("STATE_HEAD"):
        inventory.bind(manifest["state_training_lineage"], root, model, "state.lineage")
        inputs = {item["role"]: item for item in lineage["input_artifacts"]}
        for role, item in inputs.items():
            inventory.bind(item, lineage_path.parent, model, "state.input."+role)
        for fold, item in sorted(lineage["new_private_head_checkpoints"].items()):
            inventory.bind(item, lineage_path.parent, model, "state.private_head", int(fold))
        patient_pin = inputs["v32_patient_fold_manifest"]
        seen = set()
        for event in scopes.to_dict("records"):
            fold = event["patient_fold_id"]; split = event["split_role"]
            # The original estimator can inspect a population without being
            # estimable. Include that statistical exposure, never optimizer use.
            key = (fold, split, event["cohort_id"])
            if key in seen: continue
            seen.add(key)
            rows = json.loads(event["patient_map_row_indices_json"])
            if event["patient_map_sha256"] != patient_pin["sha256"]:
                raise ValueError("State statistical scope uses a different authority")
            phase = {"train": "FEATURE_CONSTRUCTION", "validation": "MODEL_SELECTION", "test": "EVALUATION"}[split]
            inventory.patient_use.observe([raw_patients.iloc[rows][["patient_id"]]],
                digest=patient_pin["sha256"], model=model, context=inventory.context_id,
                role="state.paired_cohort."+event["cohort_id"], fold=fold, phase=phase,
                scope="NATIVE_STATE_STATISTICAL_PAIRED_PATIENT_ROWS")
        for role in ("tumor_state_long_measurement", "lncrna_expression"):
            inventory.gap(model, "state.input."+role, "STATE_RAW_MEASUREMENT_ANCESTRY_NOT_RECONSTRUCTED",
                digest=inputs[role]["sha256"])
    if set(inventory.locations) != inventory.used_locations:
        raise ValueError("Unused State model-use artifact locator is not a proven dependency")
    tables = inventory_tables(inventory, {model})
    if set(tables) != NAMES: raise ValueError("State model-use table inventory changed")
    bind_sources(sources, inventory.sources.values(), root)
    summary = tables["completeness"].copy()
    audit = tables["patient_mapping"]
    observed = []; unmapped = []
    for row in summary.itertuples():
        part = audit.loc[audit.model_id.eq(row.model_id) & audit.dependency_context.eq(row.dependency_context)]
        observed.append(int(part.observed_identity_count.sum()))
        unmapped.append(int(part.unmapped_identity_count.sum()))
    summary["observed_identity_context_count"] = observed
    summary["unmapped_identity_context_count"] = unmapped
    summary["registered_patient_mapping_complete"] = [count > 0 and missing == 0 for count, missing in zip(observed, unmapped)]
    summary["patient_identity_namespace"] = inventory.patient_use.mapper.namespace
    summary["optimizer_patient_use_attested"] = False
    summary["independent_performance_verified"] = False
    return {PREFIX+name: frame for name, frame in tables.items()}, summary
