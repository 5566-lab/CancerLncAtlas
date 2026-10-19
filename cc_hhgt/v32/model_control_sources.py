"""Validate native control files without inventing biological study evidence.

These functions are entered from the actual prepared/overlay loaders. No input
flag can declare a generic table to be a verified control or waive its ancestry.
Patients present in a split authority are not thereby treated as fitted rows.
"""
from pathlib import Path

from .formal_expression_sources import read


def locate(inventory,item,root,model,role,fold):
    digest=item.get("sha256","")
    source=inventory.locations.get(digest,item)
    path=Path(source.get("path",""));path=path if path.is_absolute() else root/path
    if not digest or not source.get("path") or not path.is_file():
        inventory.authority(item,root,model,role,fold)
        return None
    if digest in inventory.locations:inventory.used_locations.add(digest)
    return inventory.bind(source,root,model,role,fold)


def inspect_patients(inventory,binding,root,model,fold):
    from .patient_fold_authority import (validate_frozen_v32_patient_fold_payload_binding,
                                         validate_frozen_v32_patient_fold_binding)
    validate_frozen_v32_patient_fold_payload_binding(binding)
    records={name:binding[name] for name in ("sample_patient_fold_map","authority_receipt")}
    paths={name:locate(inventory,item,root,model,"patient."+name,fold) for name,item in records.items()}
    if all(path is not None for path in paths.values()):
        validate_frozen_v32_patient_fold_binding(paths["sample_patient_fold_map"],paths["authority_receipt"])
    else:
        inventory.gap(model,"patient.control","PATIENT_CONTROL_FILES_NOT_VERIFIED",fold)
    return {item["sha256"] for item in records.values()}


def inspect_graph_receipt(inventory,binding,root,model,role,fold):
    from .formal_graph_authority import (GRAPH_INPUT_RECEIPT_FORMAT,STATIC_ARTIFACT_IDS,FOLD_ARTIFACT_IDS,
                                         RELATION_SCHEMA_GENERIC_GLOBAL,validate_graph_control_receipt)
    from .current_core_export import checked_sha
    item=binding.get("receipt",{})
    path=locate(inventory,item,root,model,role,fold)
    if path is None:return None
    value=read(path)
    if value.get("format")!=GRAPH_INPUT_RECEIPT_FORMAT:
        inventory.gap(model,role,"UNSUPPORTED_GRAPH_CONTROL_RECEIPT",fold,item["sha256"])
        return path
    # A source-only E1 inspection may carry a real graph receipt without the
    # complete consumed training binding. Retain its sources and a blocking
    # gap; the strict prepared-input loaders still require the full contract.
    sections = {"patient_fold_authority": (), "static_artifacts": STATIC_ARTIFACT_IDS,
                "fold_artifacts": FOLD_ARTIFACT_IDS,
                "train_identity": ("train_sample_patient_sha256", "train_patient_sha256")}
    if ("outer_fold" not in binding or any(not isinstance(binding.get(name),dict)
            or not binding[name] or not set(keys)<=set(binding[name]) for name,keys in sections.items())):
        inventory.gap(model,role,"GRAPH_CONTROL_CONSUMED_BINDING_INCOMPLETE",fold,item["sha256"])
        return path
    # The frozen payload validator also checks these patient hashes. Actual
    # patient files are checked independently by inspect_patients.
    patient=binding["patient_fold_authority"]
    _,_,value=validate_graph_control_receipt(receipt_path=path,receipt_sha256=item["sha256"],
        patient_authority_audit={**patient,"explicit_patient_id":True,"sample_fallback_used":False},
        relation_schema_generation=binding.get("binding_generation",RELATION_SCHEMA_GENERIC_GLOBAL))
    if (set(value.get("artifacts",{}))!=set(STATIC_ARTIFACT_IDS)
            or set(value.get("fold_artifacts",{}))!=set(map(str,range(5)))
            or binding.get("outer_fold")!=fold):
        raise ValueError("Graph control receipt artifact/fold coverage differs")
    def declaration(source,expected_role):
        checked_sha(source.get("sha256"),"graph control artifact")
        if (not source.get("path") or source.get("source_role")!=expected_role
                or any(source.get(k) is not False for k in ("outcome_derived","historical_model_output","toy_or_synthetic"))):
            raise ValueError("Graph control artifact declaration is unsafe")
    for name in STATIC_ARTIFACT_IDS:
        actual=binding["static_artifacts"][name];declared=value["artifacts"][name]
        declaration(declared,"outcome_free_expression_detection" if name=="detection" else "static_annotation")
        # Native detection is a validated per-fold replacement of the receipt's
        # cohort declaration; graph_sources replays its normalization binding.
        if name=="detection" and actual.get("upstream_scope"):continue
        if actual["sha256"]!=declared.get("sha256"):
            raise ValueError("Graph control receipt differs from consumed static input: "+name)
    for f in range(5):
        declared=value["fold_artifacts"][str(f)]
        if declared.get("outer_fold")!=f or not set(FOLD_ARTIFACT_IDS)<=set(declared):
            raise ValueError("Graph control receipt fold declaration differs")
        for name in FOLD_ARTIFACT_IDS:
            source=declared[name]
            declaration(source,"outer_train_expression" if name=="train_expression" else "outer_train_coexpression")
            if (source.get("outer_fold")!=f or source.get("source_split")!="train"
                    or source.get("patient_first") is not True):
                raise ValueError("Graph control receipt is not patient-first outer-train")
            if f==fold and source.get("sha256")!=binding["fold_artifacts"][name]["sha256"]:
                raise ValueError("Graph control receipt differs from consumed fold input")
            for identity in ("train_sample_patient_sha256","train_patient_sha256"):
                checked_sha(source.get(identity),"graph control train identity")
                if f==fold and source[identity]!=binding.get("train_identity",{}).get(identity):
                    raise ValueError("Graph control receipt train identity differs from consumed graph")
    return path


def inspect_expansion_receipt(inventory,binding,verified,source,root,model,role,fold):
    """Logical audit digest checked by apply_binding_overlay on actual loading."""
    from .current_binding_overlay import canonical_json_sha
    if not source or not verified:
        inventory.gap(model,role,"EXPANSION_CONTROL_SIDECAR_NOT_VERIFIED",fold,binding["receipt"]["sha256"])
        return set()
    path=locate(inventory,source,root,model,role+".sidecar",fold)
    if path is None:return set()
    if (verified.get("verified") is not True or verified.get("sidecar_sha256")!=source["sha256"]
            or verified.get("logical_receipt_sha256")!=binding["receipt"]["sha256"]
            or verified.get("comparison_sha256")!=binding["comparison_sha256"]
            or canonical_json_sha(binding["comparison"])!=binding["comparison_sha256"]
            or verified.get("arm")!=binding["binding_expansion_arm"] or binding["outer_fold"]!=fold):
        raise ValueError("Expansion control differs from the actual loaded sidecar")
    return {binding["receipt"]["sha256"],binding["comparison_sha256"],source["sha256"]}
