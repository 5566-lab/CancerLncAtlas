"""Explicit native G2 input mode shared by preparation, training and export."""
from pathlib import Path
import json

NATIVE_MODE = "NATIVE_GLOBAL_PHYSICAL_BINDING"
C_OVERLAY_MODES = frozenset({"C_TYPED_ENCODE_V2", "C_GLOBAL_PHYSICAL_BINDING"})


def validate_native_base(payload, fold, *, bundle=True):
    from .formal_graph_authority import validate_formal_graph_payload_binding
    from .current_core_export import checked_sha
    binding = payload.get("formal_graph_authority", {})
    preparation = binding.get("typed_binding_preparation") or {}
    gates = binding.get("gates", {})
    if (payload.get("formal_graph_variant") != "G2" or payload.get("patient_fold") != fold
            or payload.get("binding_expansion_authority") or payload.get("graph_overlay_authority")
            or payload.get("contains_optimizer_state") is not False
            or payload.get("contains_trained_parameters") is not False
            or binding.get("binding_generation") != "TYPED_ASSAY_CLASS_V1"
            or binding.get("binding_context_policy") != "GLOBAL_PHYSICAL_BINDING"
            or preparation.get("format") != "TYPED_GLOBAL_GRAPH_AUTHORITY_PREPARATION_V1"
            or preparation.get("target_host") != "149"
            or preparation.get("binding_source") != binding.get("static_artifacts", {}).get("lnc_protein_binding")
            or preparation.get("predicted_binding_included") is not False
            or gates.get("training_patient_detection") is not True
            or gates.get("cohort_detection_used") is not False
            or not binding.get("fold_normalization_match")):
        raise ValueError("Native direct G2 requires fresh typed global inputs and training-fold normalization")
    for name in ("source_receipt", "source_manifest"):
        item = preparation.get(name, {})
        if not item.get("path"):
            raise ValueError("Native typed preparation lacks source ancestry")
        checked_sha(item.get("sha256"), "native typed preparation " + name)
    return validate_formal_graph_payload_binding(binding, outer_fold=fold, variant="G2",
        bundle=payload.get("bundle") if bundle else None)


def validate_source_mode(payload, config, fold, *, has_overlay):
    mode = str(config.get("task_contract", {}).get("rbp_evidence_mode", ""))
    if mode == NATIVE_MODE:
        if has_overlay or config.get("training_io", {}).get("c_graph_overlay_pattern"):
            raise ValueError("Native direct G2 cannot also declare a historical C overlay")
        if config.get("task_contract", {}).get("graph_variant") != "G2":
            raise ValueError("Native direct binding mode requires G2")
        validate_native_base(payload, fold, bundle=False)
    elif mode in C_OVERLAY_MODES:
        if not has_overlay:
            raise ValueError("Historical C mode requires its authorized graph overlay")
    elif mode:
        raise ValueError(f"Unsupported RBP evidence training mode: {mode}")


def load_training_graph_inputs(torch, *, payload, config, input_manifest_path,
                               fold, root, prepared_sha256, task_seed=None):
    """The trainer's actual graph-loading stage, callable without optimization."""
    mode = str(config.get("task_contract", {}).get("rbp_evidence_mode", ""))
    overlay_active = mode in C_OVERLAY_MODES
    manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
    rows = [r for r in manifest.get("fold_inputs", []) if r.get("fold") == fold]
    if len(rows) != 1:
        raise ValueError("Exactly one authorized graph input per fold is required")
    if (mode == NATIVE_MODE) != (rows[0].get("graph_input_mode") == NATIVE_MODE):
        raise ValueError("Native graph mode differs from its input manifest")
    if mode == NATIVE_MODE and (rows[0].get("graph_overlay") or rows[0].get("graph_input_mode") != NATIVE_MODE):
        raise ValueError("Native graph mode differs from its input manifest")
    if overlay_active:
        from .c_graph_overlay import load_c_graph_overlay
        pattern = config.get("training_io", {}).get("c_graph_overlay_pattern")
        if not isinstance(pattern, str) or pattern.count("{fold}") != 1:
            raise ValueError("C training requires one explicit graph overlay fold pattern")
        path = Path(pattern.format(fold=fold))
        if not path.is_absolute(): path = Path(root)/path
        payload, _ = load_c_graph_overlay(torch, input_manifest_path=input_manifest_path,
            overlay_path=path, fold=fold, parent_sha256=prepared_sha256, old_payload=payload)
    validate_source_mode(payload, config, fold, has_overlay=overlay_active)
    from .current_binding_overlay import load_declared_binding_expansion
    return load_declared_binding_expansion(torch, payload=payload, config=config,
        input_manifest_path=input_manifest_path, fold=fold, root=root, task_seed=task_seed), overlay_active
