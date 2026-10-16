"""Raw external observations and declared model-use provenance, without labels inferred from scores."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import verified_path
from .evidence_rule_sources import read_table, strict_bool
from .evidence_rule_summary import provenance_tokens, stable_id, text, assign_study_components

PHASES = {"TRAINING", "MODEL_SELECTION", "FEATURE_CONSTRUCTION", "CALIBRATION"}
SOURCE_COLUMNS = ["source_id", "source_event_id", "source_file_sha256", "phase", "endpoint", "cancer_id", "lncrna_id",
    "pathway_id", "label", "is_experimental", "is_predicted", "mapped", "assay", "biosample", "study_tokens_json",
    "raw_record_id", "raw_row_sha256", "raw_identifiers_json"]
KNOWN_POSITIVE = {
    "lnc2cancer": ("name", "cancer type", "pubmed id", "methods"),
    "lncrnadisease": ("ncRNA Symbol", "Disease Name", "PubMed ID", "Validated Method//Prediction Method"),
    "rnadisease_experimental": ("RNA Symbol", "Disease Name", "PMID", None),
    "rnadisease_predicted": ("RNA_symbol", "disease_name", None, "method_name"),
}


def identity_maps(frame):
    if not {"namespace", "raw_id", "canonical_id"}.issubset(frame) or frame.empty:
        raise ValueError("External validation requires an explicit identity crosswalk")
    if frame[["namespace", "raw_id", "canonical_id"]].isna().any().any(): raise ValueError("Missing crosswalk identity")
    result = {}
    for row in frame.itertuples():
        if row.namespace not in {"lncrna", "cancer", "pathway"}: raise ValueError("Unknown crosswalk namespace")
        if any(not isinstance(v, str) or not v or v.strip() != v for v in (row.raw_id, row.canonical_id)):
            raise ValueError("Crosswalk IDs must be canonical strings")
        mapping = result.setdefault(row.namespace, {})
        for alias in (row.raw_id, row.canonical_id):
            if alias in mapping and mapping[alias] != row.canonical_id: raise ValueError("Ambiguous external identity")
            mapping[alias] = row.canonical_id
    return result


def _tokens(row):
    pmid = row.get("pmid")
    if isinstance(pmid, (float, np.floating)) and np.isfinite(pmid) and pmid == int(pmid): pmid = str(int(pmid))
    tokens = provenance_tokens({**row, "pmid": pmid,
        "source_dataset": text(row.get("dataset_id")) or text(row.get("source_dataset")) or text(row.get("dataset_accession"))})
    accession = text(row.get("study_accession"))
    if accession and accession.lower() not in {"unknown", "na", "none"}:
        tokens.append("REGISTERED_STUDY:"+accession)
    return json.dumps(sorted(set(tokens)))


def normalize_source(frame, spec, maps, *, usage=False):
    kind = spec["kind"]; source_id = spec["source_id"]
    if not isinstance(source_id, str) or not source_id.strip() or source_id == "ALL_SOURCES":
        raise ValueError("Source IDs must be nonempty strings distinct from the pooled cohort name")
    if usage:
        if kind not in {"canonical_usage", "native_exact_events"} or spec.get("phase") not in PHASES:
            raise ValueError("Model-use sources need a supported schema and explicit use phase")
        if not {"lncrna_id", "pathway_id"}.issubset(frame): raise ValueError("Model-use inventory lacks entity identities")
    elif kind == "exact_assay":
        required = {"record_id", "lncrna", "cancer", "pathway", "label", "is_experimental", "is_predicted", "assay"}
        if not required.issubset(frame): raise ValueError("Exact assay source lacks measured outcomes or identities")
        strict_bool(frame, "is_experimental"); strict_bool(frame, "is_predicted")
        values = pd.to_numeric(frame.label, errors="coerce")
        if (frame.label.notna() & values.isna()).any() or not values.dropna().isin([0, 1]).all():
            raise ValueError("Exact external outcomes must be measured 0/1 or explicitly unknown")
        if values.eq(0).any() and not text(spec.get("measured_negative_definition")):
            raise ValueError("Measured negative outcomes require their experimental definition")
        if frame.loc[values.notna() & frame.is_experimental, "assay"].map(text).eq("").any():
            raise ValueError("Measured outcomes require an assay description")
    elif kind == "gse85011_followup":
        required = {"gsm", "target_raw", "cell_line", "dataset_accession", "processed_data_file", "is_control"}
        if not required.issubset(frame): raise ValueError("GSE85011 follow-up metadata is incomplete")
        strict_bool(frame, "is_control")
        if not frame.dataset_accession.eq("GSE85011").all(): raise ValueError("Unexpected follow-up dataset")
        if spec.get("followup_selection_semantics") != "PUBLISHED_GROWTH_MODIFIER_HITS":
            raise ValueError("Follow-up sample selection must have its explicit published interpretation")
    elif kind in KNOWN_POSITIVE:
        columns = [c for c in KNOWN_POSITIVE[kind] if c]
        if not set(columns).issubset(frame): raise ValueError("Native known-positive source schema differs")
        # LncRNADisease mixes prediction and experimental methods in one export.
        # Its classification must be supplied by typed raw flags, never guessed
        # from a substring such as 'not experimentally validated'.
        if kind == "lncrnadisease" and {"is_experimental", "is_predicted"}.issubset(frame):
            strict_bool(frame, "is_experimental"); strict_bool(frame, "is_predicted")
        elif kind == "lncrnadisease":
            classes = spec.get("method_classes", {})
            if not classes or not set(classes.values()).issubset({"experimental", "predicted", "unknown"}):
                raise ValueError("Native mixed-method source requires an explicit exact-method classification")
    else: raise ValueError("Unsupported external observation source")
    rows = []
    for index, raw in enumerate(frame.to_dict("records")):
        if usage:
            ids = {k: text(raw.get(k+"_id")) for k in ("lncrna", "cancer", "pathway")}
            label = np.nan; experimental = False; predicted = False; endpoint = "MODEL_USE"
        elif kind == "exact_assay":
            ids = {k: text(raw.get(k)) for k in ("lncrna", "cancer", "pathway")}
            label = float(raw["label"]) if pd.notna(raw["label"]) else np.nan
            experimental = bool(raw["is_experimental"]); predicted = bool(raw["is_predicted"]); endpoint = "EXACT_PATHWAY"
        elif kind == "gse85011_followup":
            ids = {"lncrna": text(raw["target_raw"]), "cancer": text(raw["cell_line"]), "pathway": ""}
            selected = "rnaseq_tpm" in text(raw["processed_data_file"]).lower() and not raw["is_control"]
            label = 1. if selected else np.nan
            experimental = True; predicted = False; endpoint = "CANCER_LNCRNA_KNOWN_POSITIVE"
            raw = {**raw, "record_id": raw["gsm"], "biosample": raw["cell_line"],
                "assay": "CRISPRI_GROWTH_MODIFIER_SELECTED_FOR_RNASEQ_FOLLOWUP"}
        else:
            lnc, cancer, pmid, method = KNOWN_POSITIVE[kind]
            ids = {"lncrna": text(raw[lnc]), "cancer": text(raw[cancer]), "pathway": ""}
            raw = {**raw, "pmid": raw.get(pmid, ""), "assay": raw.get(method, "")}
            label = 1.; predicted = kind == "rnadisease_predicted"; experimental = not predicted
            if kind == "lncrnadisease":
                if {"is_experimental", "is_predicted"}.issubset(raw):
                    experimental = bool(raw["is_experimental"]); predicted = bool(raw["is_predicted"])
                else:
                    method_class = spec["method_classes"].get(text(raw[method]), "unknown")
                    experimental = method_class == "experimental"; predicted = method_class == "predicted"
            endpoint = "CANCER_LNCRNA_KNOWN_POSITIVE"
        canonical = {k+"_id": maps.get(k, {}).get(v, "") if v else "" for k, v in ids.items()}
        if usage and any(ids[k] and not canonical[k+"_id"] for k in ("lncrna", "pathway", "cancer")):
            raise ValueError("Unmapped model-use entity would invalidate entity isolation")
        mapped = bool(canonical["lncrna_id"] and (usage or canonical["cancer_id"]) and
                      (endpoint != "EXACT_PATHWAY" or canonical["pathway_id"]))
        raw_hash = stable_id("", {str(k): str(v) if isinstance(v, (list, tuple, dict, np.ndarray)) else text(v) for k, v in raw.items()})
        rows.append({**canonical, "source_id": source_id, "source_event_id": stable_id("EXT:", [source_id, index, raw_hash]),
            "source_file_sha256": spec["table"]["sha256"], "phase": spec["phase"] if usage else "EXTERNAL_EVALUATION",
            "endpoint": endpoint, "label": label, "is_experimental": experimental, "is_predicted": predicted,
            "mapped": mapped, "assay": text(raw.get("assay", raw.get("experiment_type"))), "biosample": text(raw.get("biosample")),
            "study_tokens_json": _tokens(raw), "raw_record_id": text(raw.get("record_id", raw.get("source_record_id", index))),
            "raw_row_sha256": raw_hash, "raw_identifiers_json": json.dumps(ids, sort_keys=True)})
    result = pd.DataFrame(rows, columns=SOURCE_COLUMNS)
    for col in ("is_experimental", "is_predicted", "mapped"): result[col] = result[col].astype(bool)
    result["label"] = pd.to_numeric(result.label).astype(float)
    return result


def read_sources(specs, maps, root, *, usage=False,patients=None):
    if not specs or len({s["source_id"] for s in specs}) != len(specs): raise ValueError("Source declarations must be nonempty and unique")
    frames = []
    for spec in specs:
        if not text(spec["source_id"]): raise ValueError("Missing source identity")
        path = verified_path(spec["table"], root)
        raw=read_table(path);normalized=normalize_source(raw,spec,maps,usage=usage)
        if patients is not None:
            annotations=patients.annotate(raw,spec,path=path)
            for column in annotations:normalized[column]=annotations[column].to_numpy()
        frames.append(normalized)
    return pd.concat(frames, ignore_index=True)


def isolate_sources(external, usage, *, dependency_complete=None, dependency_hashes=()):
    """Connected study exclusion plus cancer-agnostic pair and lncRNA novelty."""
    if set(external.source_id) & set(usage.source_id): raise ValueError("External and used source identities must be distinct")
    combined = assign_study_components(pd.concat([usage, external], ignore_index=True))
    used = combined.loc[combined.phase.ne("EXTERNAL_EVALUATION")].copy()
    evaluated = combined.loc[combined.phase.eq("EXTERNAL_EVALUATION")].copy()
    used_studies = set(used.loc[used.provenance_verified, "study_id"])
    used_lnc = set(used.loc[used.lncrna_id.ne(""), "lncrna_id"])
    used_pairs = set(used.loc[used.lncrna_id.ne("") & used.pathway_id.ne(""), ["lncrna_id", "pathway_id"]].itertuples(index=False, name=None))
    used_unspecified_pair = set(used.loc[used.pathway_id.eq(""), "lncrna_id"])
    complete = bool(len(used) and used.provenance_verified.all() and dependency_complete is not False)
    entity_complete = bool(len(used) and used.mapped.all())
    evaluated["model_usage_provenance_complete"] = complete
    evaluated["model_usage_entity_complete"] = entity_complete
    evaluated["study_overlap"] = evaluated.study_id.isin(used_studies)
    evaluated["source_file_overlap"] = evaluated.source_file_sha256.isin(set(used.source_file_sha256)|set(dependency_hashes))
    evaluated["lncrna_overlap"] = evaluated.lncrna_id.isin(used_lnc)
    evaluated["pair_overlap"] = [lnc in used_unspecified_pair or (lnc, pathway) in used_pairs
        or (not pathway and lnc in used_lnc) for lnc, pathway in evaluated[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None)]
    evaluated["descriptive_eligible"] = evaluated.mapped & evaluated.is_experimental & ~evaluated.is_predicted & evaluated.label.notna()
    patient_checked="patient_overlap" in evaluated
    patient_ready=(evaluated.external_patient_identity_complete.fillna(False)
                   & evaluated.registered_patient_mapping_complete.fillna(False)
                   & ~evaluated.patient_overlap.fillna(False)) if patient_checked else True
    evaluated["provenance_held_out"] = (evaluated.descriptive_eligible & evaluated.provenance_verified & complete
        & ~evaluated.study_overlap & ~evaluated.source_file_overlap & patient_ready)
    evaluated["pair_held_out"] = evaluated.provenance_held_out & entity_complete & ~evaluated.pair_overlap
    evaluated["lncrna_held_out"] = evaluated.provenance_held_out & entity_complete & ~evaluated.lncrna_overlap
    def reasons(row):
        return json.dumps([reason for reason, condition in (
            ("IDENTITY_UNMAPPED", not row.mapped), ("NONEXPERIMENTAL_OR_PREDICTED", not row.is_experimental or row.is_predicted),
            ("OUTCOME_UNKNOWN", pd.isna(row.label)), ("EXTERNAL_PROVENANCE_MISSING", not row.provenance_verified),
            ("MODEL_USE_PROVENANCE_INCOMPLETE", not complete), ("STUDY_OVERLAP", row.study_overlap),
            ("NATIVE_DEPENDENCIES_INCOMPLETE", dependency_complete is False),
            ("MODEL_USE_ENTITY_INCOMPLETE", not entity_complete),
            ("SOURCE_FILE_OVERLAP", row.source_file_overlap), ("PAIR_OVERLAP", row.pair_overlap),
            ("LNCRNA_OVERLAP", row.lncrna_overlap),
            ("PATIENT_OVERLAP",patient_checked and bool(row.patient_overlap)),
            ("EXTERNAL_PATIENT_IDENTITY_INCOMPLETE",patient_checked and not bool(row.external_patient_identity_complete)),
            ("REGISTERED_PATIENT_MAPPING_INCOMPLETE",patient_checked and not bool(row.registered_patient_mapping_complete))) if condition])
    evaluated["isolation_reasons_json"] = [reasons(row) for row in evaluated.itertuples()]
    used=used.drop(columns=["patient_overlap","overlapping_patient_count","external_patient_identity_complete",
        "registered_patient_mapping_complete","patient_identity_status"],errors="ignore")
    return evaluated.reset_index(drop=True), used.reset_index(drop=True)
