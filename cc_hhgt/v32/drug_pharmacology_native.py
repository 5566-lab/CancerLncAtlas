"""Adapt native DrugCentral and explicit assay protocols to the existing head.

The actual native staging manifest supplies compound/model identities. Activity
value and censoring scales must be declared separately. A fitted concentration
range is never materialized as measured point doses or a response curve.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pandas as pd

from . import drug_training as base
from .current_core_export import checked_sha, sha256
from .drug_identity import DrugIdentityResolver, text
from .drug_pharmacology import FORMAT, ENDPOINTS, normalize_targets, normalize_design, structure_table
from .drug_staging import _stage_drug_targets, artifact_sha256

NATIVE_FORMAT = "NATIVE_DRUG_PHARMACOLOGY_V1"
AUDIT_FORMAT = "NATIVE_DRUG_PHARMACOLOGY_AUDIT_V1"
TARGET_COLUMNS = ["drug_id", "gene_id", "action_type", "affinity_type", "affinity_value",
                  "affinity_unit", "affinity_relation", "source_id"]
DESIGN_COLUMNS = ["dataset_id", "cell_line_id", "drug_id", "assay_id", "dose_value", "dose_unit",
                  "exposure_hours", "source_id"]
REVERSE = {"=":"=", "<":">", ">":"<", "<=":">=", ">=":"<="}
SCALES = {"NEGATIVE_LOG10_MOLAR", "CONCENTRATION"}


def verify_artifact(record, root):
    path = Path(record["path"])
    path = (root/path).resolve() if not path.is_absolute() else path.resolve()
    if artifact_sha256(path) != checked_sha(record.get("sha256"), str(path)):
        raise ValueError("Native pharmacology source SHA256 mismatch: "+str(path))
    return path


def _strings(path):
    if path.is_dir() or path.suffix == ".parquet": return pd.read_parquet(path)
    logical = path.suffixes[-2] if path.suffix == ".gz" else path.suffix
    return pd.read_csv(path, sep="\t" if logical in {".tsv", ".txt"} else ",", dtype=str, low_memory=False)


def activity_rows(staged, authority, declaration, *, source_sha):
    """Keep target/action rows even when quantitative activity is unavailable."""
    value_scale, relation_scale = declaration.get("value_scale"), declaration.get("relation_scale")
    if value_scale not in SCALES or relation_scale not in SCALES:
        raise ValueError("Native activity requires explicit value_scale and relation_scale")
    if not text(declaration.get("schema_reference")):
        raise ValueError("Native activity scale declaration requires its schema reference")
    allowed = set(authority[["drug_id", "gene_id"]].itertuples(index=False, name=None))
    rows, audits = [], []
    for row in staged.to_dict("records"):
        index = int(row["native_source_row"])
        audit = {"source_row":index, "drug_id":row["drug_id"], "gene_id":row["gene_id"],
                 "action_raw":text(row.get("native_action_type")) or None,
                 "activity_raw":text(row.get("native_act_value")) or None,
                 "unit_raw":text(row.get("native_act_unit")) or None,
                 "endpoint_raw":text(row.get("native_act_type")) or None,
                 "relation_raw":text(row.get("native_relation")) or None,
                 "target_retained":False, "activity_available":False, "reason":None}
        if (row["drug_id"], row["gene_id"]) not in allowed:
            audit["reason"] = "OUTSIDE_CURRENT_TARGET_AUTHORITY"; audits.append(audit); continue
        output = {"drug_id":row["drug_id"], "gene_id":row["gene_id"], "action_type":audit["action_raw"],
                  "source_id":f"DrugCentral:{source_sha}:row:{index}",
                  **{k:None for k in TARGET_COLUMNS[3:7]}}
        audit["target_retained"] = True
        endpoint = text(audit["endpoint_raw"]).upper()
        if audit["activity_raw"] is None:
            audit["reason"] = "NO_QUANTITATIVE_ACTIVITY"
        elif endpoint not in ENDPOINTS:
            audit["reason"] = "UNSUPPORTED_ACTIVITY_ENDPOINT"
        elif audit["relation_raw"] not in REVERSE:
            audit["reason"] = "MISSING_OR_UNSUPPORTED_CENSORING_RELATION"
        else:
            unit = audit["unit_raw"]
            if value_scale == "NEGATIVE_LOG10_MOLAR":
                if unit not in {None, "p_molar"}:
                    raise ValueError("Native ACT_UNIT conflicts with declared negative-log molar scale")
                unit = "p_molar"
            elif unit == "p_molar":
                raise ValueError("Native ACT_UNIT conflicts with declared concentration scale")
            if not unit:
                audit["reason"] = "MISSING_CONCENTRATION_UNIT"
            else:
                relation = audit["relation_raw"]
                if value_scale != relation_scale: relation = REVERSE[relation]
                output.update(affinity_type=endpoint, affinity_value=float(audit["activity_raw"]),
                              affinity_unit=unit, affinity_relation=relation)
                audit["activity_available"] = True
        rows.append(output); audits.append(audit)
    frame = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    # Validate once against the authority; rebuilding its full pair set for
    # every native measurement would make this extraction quadratic.
    normalize_targets(frame, authority)
    return frame, pd.DataFrame(audits)


def native_structures(frame, resolver, drugs, *, source_sha):
    if not {"ID", "INN", "SMILES"}.issubset(frame):
        raise ValueError("DrugCentral structure export requires ID, INN and SMILES")
    rows, audit = [], []
    for index, row in enumerate(frame.to_dict("records")):
        if not text(row["ID"]): raise ValueError("Native structure lacks DrugCentral ID")
        identity = resolver.resolve("DRUGCENTRAL", row["ID"], row["INN"],
            identifiers={"inchi_key":row.get("InChIKey")})
        retained = identity["identity_available"] and identity["drug_id"] in drugs
        audit.append({"source_row":index, "native_drug_id":text(row["ID"]), **identity,
            "structure_retained":retained,
            "reason":None if retained else identity["identity_unavailable_reason"] or "OUTSIDE_CURRENT_DRUG_UNIVERSE"})
        if retained:
            rows.append({"drug_id":identity["drug_id"], "smiles":text(row["SMILES"]) or None,
                         "source_id":f"DrugCentralStructures:{source_sha}:row:{index}"})
    return pd.DataFrame(rows, columns=["drug_id", "smiles", "source_id"]), pd.DataFrame(audit)


def native_protocols(frame, spec, source_map, response, *, source_sha):
    """Read explicitly mapped point protocols using the staging native ID map."""
    if spec.get("condition_kind") not in {"POINT_CONCENTRATION", "FITTED_CONCENTRATION_RANGE"}:
        raise ValueError("Assay metadata must declare point concentration or fitted range semantics")
    columns, constants = spec.get("columns", {}), spec.get("constants", {})
    fields = {"dataset_id", "cell_line_id", "source_drug_id", "assay_id", "dose_value", "dose_unit", "exposure_hours"}
    if (set(columns) | set(constants))-fields or set(columns) & set(constants):
        raise ValueError("Unknown or multiply assigned native protocol fields")
    if not set(columns.values()).issubset(frame): raise ValueError("Declared native protocol columns do not exist")
    if not {"dataset_id", "cell_line_id", "source_drug_id", "assay_id"}.issubset(set(columns) | set(constants)):
        raise ValueError("Native protocol lacks exact assay/model/compound identities")
    identity = source_map[["dataset_id", "source_drug_id", "drug_id"]].drop_duplicates().copy()
    for key in identity: identity[key] = identity[key].map(text)
    if identity.groupby(["dataset_id", "source_drug_id"]).drug_id.nunique().gt(1).any():
        raise ValueError("Staged native drug identity is ambiguous")
    identity = identity.set_index(["dataset_id", "source_drug_id"]).drug_id.to_dict()
    observed = set(response[["dataset_id", "cell_line_id", "drug_id"]].itertuples(index=False, name=None))
    rows, audits = [], []
    for index, native in enumerate(frame.to_dict("records")):
        values = {key:native[column] for key,column in columns.items()}; values.update(constants)
        for key in ("dataset_id", "cell_line_id", "source_drug_id", "assay_id"):
            values[key] = text(values[key])
            if not values[key]: raise ValueError("Empty native assay identity: "+key)
        drug = identity.get((values["dataset_id"], values["source_drug_id"]))
        reason = None
        if drug is None: reason = "NO_STAGED_NATIVE_DRUG_ID"
        elif (values["dataset_id"], values["cell_line_id"], drug) not in observed: reason = "NO_STAGED_RESPONSE_IDENTITY"
        elif spec["condition_kind"] == "FITTED_CONCENTRATION_RANGE": reason = "RANGE_IS_NOT_POINT_DOSE"
        audits.append({"source_sha256":source_sha, "source_row":index, "dataset_id":values["dataset_id"],
            "cell_line_id":values["cell_line_id"], "native_drug_id":values["source_drug_id"],
            "drug_id":drug, "assay_id":values["assay_id"], "retained":reason is None, "reason":reason})
        if reason: continue
        rows.append({"dataset_id":values["dataset_id"], "cell_line_id":values["cell_line_id"], "drug_id":drug,
            "assay_id":values["assay_id"], "dose_value":values.get("dose_value"), "dose_unit":values.get("dose_unit"),
            "exposure_hours":values.get("exposure_hours"), "source_id":f"NativeProtocol:{source_sha}:row:{index}"})
    return pd.DataFrame(rows, columns=DESIGN_COLUMNS), pd.DataFrame(audits)


def materialize_native_pharmacology(manifest_path, output):
    if socket.gethostname() != "149": raise RuntimeError("Native pharmacology preparation requires host 149")
    path = Path(manifest_path).resolve(); manifest_sha = sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != NATIVE_FORMAT: raise ValueError("Unknown native pharmacology format")
    sources = {"native_manifest":{"path":str(path), "sha256":manifest_sha}}
    def bound(role, record, root=path.parent):
        target = verify_artifact(record, root)
        sources[role] = {"path":str(target), "sha256":record["sha256"]}
        return target
    staging_path = bound("staging_manifest", manifest["staging_manifest"])
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    if (staging.get("status") != "SUCCESS" or staging.get("canonical_drug_id_policy") != "CURATED_STABLE_ID_OR_SOURCE_SCOPED_ASSAY_ID"
            or staging.get("input_artifacts_unchanged") is not True):
        raise ValueError("Native pharmacology requires successful stable-ID native staging")
    original = staging["input_artifact_snapshot"]["start"]
    native = {role:bound(role, original[role], staging_path.parent) for role in
              ("drugcentral_target_native", "hgnc_native", "drug_identity_crosswalk")}
    staged = {role:bound(role, staging["artifacts"][role], staging_path.parent) for role in
              ("drug_gene_target", "raw_cell_line_drug_response", "cell_line_map", "source_drug_map")}
    structures_path = bound("drugcentral_structures", manifest["drugcentral_structures"])
    protocols = [(spec, bound("assay_protocol_"+str(i), spec["source"])) for i,spec in enumerate(manifest.get("assay_protocols", []))]
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149", "training_started":False,
        "native_manifest_sha256":manifest_sha}))
    resolver = DrugIdentityResolver(_strings(native["drug_identity_crosswalk"]))
    authority = base.normalise_drug_targets(base._read_table(staged["drug_gene_target"]))
    response = base.normalise_raw_response(base._read_table(staged["raw_cell_line_drug_response"]))
    mapping = base.normalise_cell_line_map(base._read_table(staged["cell_line_map"]))
    source_map = _strings(staged["source_drug_map"])
    # Replay the actual existing HGNC/compound mapper rather than reimplement it.
    mapped, target_audit = _stage_drug_targets(native["drugcentral_target_native"], native["hgnc_native"],
        output/"native_mapped_targets.parquet", resolver)
    if set(mapped[["drug_id", "gene_id"]].itertuples(index=False, name=None)) != set(authority.itertuples(index=False, name=None)):
        raise ValueError("Replayed native target identities differ from the selected staging")
    annotations, annotation_audit = activity_rows(mapped, authority, manifest["drugcentral_activity"],
        source_sha=sources["drugcentral_target_native"]["sha256"])
    raw_targets = _strings(native["drugcentral_target_native"])
    mapped_rows = set(mapped.native_source_row)
    disposition = pd.DataFrame({"source_row":np.arange(len(raw_targets)),
        "native_drug_id":raw_targets.get("STRUCT_ID", pd.Series(None,index=raw_targets.index,dtype=object)),
        "native_gene":raw_targets.GENE, "organism":raw_targets.ORGANISM})
    disposition["target_mapped"] = disposition.source_row.isin(mapped_rows)
    disposition["reason"] = pd.Series(None,index=disposition.index,dtype=object)
    missing = ~disposition.target_mapped
    disposition.loc[missing,"reason"] = "MISSING_OR_AMBIGUOUS_NATIVE_COMPOUND_OR_GENE"
    nonhuman = ~disposition.organism.fillna("").str.casefold().eq("homo sapiens")
    disposition.loc[missing & nonhuman,"reason"] = "NON_HUMAN_OR_UNSPECIFIED_ORGANISM"
    drugs = set(authority.drug_id) | set(response.drug_id)
    structures, structure_audit = native_structures(_strings(structures_path), resolver, drugs,
        source_sha=sources["drugcentral_structures"]["sha256"])
    parts, protocol_audits = [], []
    for spec, source in protocols:
        part, audit = native_protocols(_strings(source), spec, source_map, response, source_sha=artifact_sha256(source))
        parts.append(part); protocol_audits.append(audit)
    design = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=DESIGN_COLUMNS)
    # Validate actual downstream schemas, measurements and molecular graphs.
    normalized = normalize_targets(annotations, authority)
    actual_structures = structure_table(structures, drugs)
    actual_design = normalize_design(design, response, mapping)
    frames = {"target_annotations":annotations, "structures":structures, "assay_conditions":design,
        "target_adaptation":annotation_audit, "native_target_disposition":disposition, "structure_adaptation":structure_audit,
        "protocol_adaptation":pd.concat(protocol_audits, ignore_index=True) if protocol_audits else
            pd.DataFrame(columns=["source_sha256", "source_row", "retained", "reason"])}
    outputs = {}
    for name, frame in frames.items():
        target = output/(name+".parquet"); frame.to_parquet(target, index=False)
        outputs[name] = {"path":str(target), "sha256":sha256(target), "rows":len(frame)}
    for record in sources.values(): verify_artifact(record, path.parent)
    audit = {"format":AUDIT_FORMAT, "status":"PASS", "target_host":"149", "training_started":False,
        "sources":sources, "outputs":outputs, "native_target_mapping":target_audit,
        "activity_declaration":manifest["drugcentral_activity"], "staging_run_id":staging["staging_run_id"],
        "quantitative_target_rows":int(normalized.p_value_molar.notna().sum()),
        "structure_available_drugs":int(actual_structures.structure_available.sum()),
        "protocol_rows":len(actual_design), "dose_response_analysis_performed":False,
        "assay_response_values_used_as_features":False,
        "code_sha256":{p.name:sha256(p) for p in (Path(__file__), Path(__file__).with_name("drug_staging.py"),
            Path(__file__).with_name("drug_identity.py"), Path(__file__).with_name("drug_pharmacology.py"))}}
    audit_path = output/"NATIVE_ADAPTATION_AUDIT.json"; audit_path.write_text(json.dumps(audit, indent=2)+"\n")
    result = {"format":FORMAT, **{k:outputs[k] for k in ("target_annotations", "structures", "assay_conditions")},
        "native_adaptation":{"path":str(audit_path), "sha256":sha256(audit_path)}, "native_sources":sources}
    result_path = output/"PHARMACOLOGY_INPUT.json"; result_path.write_text(json.dumps(result, indent=2)+"\n")
    (output/"SUCCESS.json").write_text(json.dumps({"status":"PREPARED_NOT_TRAINED",
        "input_manifest":{"path":str(result_path), "sha256":sha256(result_path)},
        "audit_sha256":sha256(audit_path)}))
    return result_path
