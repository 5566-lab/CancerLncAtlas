"""Source-specific rule evidence readers, preserving assay and provenance."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import verified_path, sha256
from .evidence_rule_summary import EVENT_COLUMNS, stable_id, text, review_value, observed_direction, provenance_tokens


def read_table(path):
    return pd.read_parquet(path) if Path(path).suffix in {".parquet", ".pq"} else pd.read_csv(path, sep="\t")


def strict_bool(frame, column):
    if column not in frame or frame[column].isna().any() or not pd.api.types.is_bool_dtype(frame[column].dtype):
        raise ValueError(f"Evidence requires typed complete {column}")
    return frame[column]


def base_event(row, *, source_id, source_event_id, route, source_cancer, provenance, detail):
    event = {column: "" for column in EVENT_COLUMNS}
    for key in ("cancer_id", "lncrna_id", "pathway_id", "partner_id", "assay", "biosample", "tissue", "cell_line",
                "species", "experiment_id", "intervention", "readout", "mediator_id"):
        event[key] = text(row.get(key))
    event.update(source_id=source_id, source_event_id=str(source_event_id), source_cancer_id=str(source_cancer).upper(),
        route_type=route, direction="unknown", review_value=np.nan, raw_occurrences=1,
        study_tokens_json=json.dumps(provenance_tokens(row)), provenance_json=json.dumps(provenance, sort_keys=True),
        detail_json=json.dumps(detail, sort_keys=True, default=str))
    return event


def _membership(members):
    if not {"pathway_id", "member_id"}.issubset(members):
        raise ValueError("Rule evidence needs exact pathway_id/member_id membership, never a family table")
    result = members[["pathway_id", "member_id"]].drop_duplicates()
    if result.isna().any().any(): raise ValueError("Missing exact membership identity")
    return result


def native_events(events, lineage, members, *, source_id, artifacts):
    required = {"event_id", "cancer_id", "lncrna_id", "pathway_id", "partner_id", "route_type", "static_member_id",
        "is_experimental", "is_computational", "is_model_prediction", "experiment_type", "source_database", "source_record_id"}
    if not required.issubset(events): raise ValueError("Native event table lacks exact routing/source identity")
    if events.event_id.isna().any() or events.event_id.duplicated().any(): raise ValueError("Native source events must be unique")
    required_lineage = {"event_id", "lineage_id", "source_row_sha256", "source_sha256", "family_broadcast_used",
                        "mapping_route", "static_member_id"}
    if not required_lineage.issubset(lineage) or set(events.event_id) != set(lineage.event_id):
        raise ValueError("Every native event requires its full lineage universe")
    if lineage.lineage_id.isna().any() or lineage.lineage_id.duplicated().any(): raise ValueError("Native lineage IDs must be unique")
    if strict_bool(lineage, "family_broadcast_used").any(): raise ValueError("Family broadcast cannot enter exact evidence")
    for col in ("source_row_sha256", "source_sha256"):
        if not lineage[col].astype(str).str.fullmatch("[a-f0-9]{64}").all(): raise ValueError("Native lineage lacks source hashes")
    for col in ("is_experimental", "is_computational", "is_model_prediction"): strict_bool(events, col)
    membership = set(_membership(members).itertuples(index=False, name=None)); provenance = {}
    for event_id, group in lineage.groupby("event_id", sort=False):
        provenance[event_id] = group[["lineage_id", "source_row_sha256", "source_sha256", "mapping_route", "static_member_id"]].fillna("").to_dict("records")
    rows = []
    for raw in events.to_dict("records"):
        route = raw["route_type"]
        if route not in {"DIRECT_EXACT_ASSERTION", "PARTNER_EXACT_MEMBER"}: raise ValueError("Unknown native exact routing")
        links = provenance[raw["event_id"]]
        if any(link["mapping_route"] != route or text(link["static_member_id"]) != text(raw["static_member_id"]) for link in links):
            raise ValueError("Native event lineage relabelled its pathway route")
        if route == "PARTNER_EXACT_MEMBER" and (raw["pathway_id"], raw["partner_id"]) not in membership:
            raise ValueError("Native partner does not belong to the declared exact pathway")
        raw["assay"] = next((text(raw.get(key)) for key in ("experiment_raw", "experiment_type", "graph_assay_class")
                             if text(raw.get(key)).lower() not in {"", "unknown"}), "unknown")
        predicted = raw["is_computational"] or raw["is_model_prediction"]
        emitted_route = "COMPUTATIONAL_SUPPORT" if predicted else route if raw["is_experimental"] else "UNVERIFIED_ASSERTION"
        row = base_event(raw, source_id=source_id, source_event_id=raw["event_id"], route=emitted_route,
            source_cancer=text(raw["cancer_id"]), provenance={"artifacts": artifacts, "lineage": links},
            detail={key: raw.get(key) for key in ("experiment_type", "graph_assay_class", "source_database", "source_dataset", "source_record_id", "pmid", "doi",
                "direction_raw", "manual_review_status", "is_experimental", "is_computational", "is_model_prediction", "static_member_id")})
        row["raw_occurrences"] = raw.get("source_occurrence_count", 1)
        if emitted_route == "DIRECT_EXACT_ASSERTION":
            row["direction"] = observed_direction(raw.get("direction_raw"))
            row["review_value"] = review_value(raw.get("manual_review_status"))
        rows.append(row)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def functional_events(frame, audit, *, source_id, artifacts):
    if frame.empty: return pd.DataFrame(columns=EVENT_COLUMNS)
    required = {"cancer_id", "lncrna_id", "pathway_id", "rbp_id", "biosample", "contrast_id", "file_accession",
        "experiment_accession", "assay", "direction", "binding_source_id", "graph_assay_class", "qvalue", "available",
        "lncrna_perturbation_claimed", "physical_binding_inferred_from_kd", "independent_external_validation"}
    if not required.issubset(frame): raise ValueError("Functional bridge lacks context and claim semantics")
    for col in ("lncrna_perturbation_claimed", "physical_binding_inferred_from_kd", "independent_external_validation"):
        if strict_bool(frame, col).any(): raise ValueError("RBP depletion is not lncRNA perturbation or independent external validation")
    if not strict_bool(frame, "available").all() or not frame.qvalue.between(0, audit.get("analysis", {}).get("alpha", .05)).all():
        raise ValueError("Functional bridge contains unavailable/non-significant pathway results")
    if not frame.graph_assay_class.isin({"eclip", "other_clip", "rip", "rna_capture", "other_physical"}).all():
        raise ValueError("Functional bridge requires physical lncRNA-RBP binding")
    rows = []
    for raw in frame.to_dict("records"):
        if raw["direction"] not in {"altered", "up", "down"}: raise ValueError("Unknown KD pathway direction")
        source = audit["source_files"].get(raw["file_accession"])
        if not source: raise ValueError("Functional contrast is absent from its source audit")
        row = {**raw, "cancer_id": "", "partner_id": raw["rbp_id"], "mediator_id": raw["rbp_id"],
            "experiment_id": raw["experiment_accession"], "intervention": "RBP_DEPLETION",
            "readout": raw["assay"]+":"+raw["direction"]}
        event_id = stable_id("FUNCTIONAL:", [raw[k] for k in ("lncrna_id", "pathway_id", "contrast_id", "binding_source_id", "direction")])
        rows.append(base_event(row, source_id=source_id, source_event_id=event_id, route="RBP_PERTURBATION_PATHWAY_BRIDGE",
            source_cancer="GLOBAL", provenance={"artifacts": artifacts, "contrast_source": source, "binding_source_id": raw["binding_source_id"]},
            detail={k: raw.get(k) for k in ("contrast_id", "file_accession", "direction", "qvalue", "pvalue", "affected_gene_ids",
                "background_tested_genes", "pathway_tested_genes", "overlap_genes", "graph_assay_class")}))
    result = pd.DataFrame(rows, columns=EVENT_COLUMNS).drop_duplicates()
    if result.source_event_id.duplicated().any(): raise ValueError("Conflicting functional source rows across candidate cancers")
    return result


def rbns_events(genes, audit, members, *, source_id, artifacts):
    required = {"lncrna_id", "rbp_id", "profile_id", "file_accession", "experiment_accession", "available",
                "prediction_score", "mean_log2_enrichment", "is_predicted", "is_experimental"}
    if not required.issubset(genes): raise ValueError("RBNS requires versioned gene sequence preferences")
    if strict_bool(genes, "is_experimental").any() or not strict_bool(genes, "is_predicted").all():
        raise ValueError("Sequence predictions cannot claim observed lncRNA binding")
    available = strict_bool(genes, "available")
    settings = audit["scoring"]
    selected = genes.loc[available & genes.prediction_score.ge(settings["min_excess"])
        & genes.mean_log2_enrichment.ge(settings["min_mean_log2_enrichment"])].copy()
    membership = _membership(members)
    # This is an explicit gene-level RBP -> exact membership bridge. No family
    # expansion, protein-symbol guessing or molecular binding claim is made.
    alias = membership.member_id.astype(str).str.replace(r"^GENE:", "", regex=True)
    mapped = membership.assign(_member_alias=alias)
    if mapped.groupby("_member_alias").member_id.nunique().gt(1).any(): raise ValueError("Ambiguous RBP member namespace")
    selected["_member_alias"] = selected.rbp_id.astype(str).str.replace(r"^GENE:", "", regex=True)
    joined = selected.merge(mapped, on="_member_alias", how="inner", validate="many_to_many")
    rows = []
    for raw in joined.to_dict("records"):
        source = audit["source_files"].get(raw["file_accession"])
        if not source: raise ValueError("RBNS profile is absent from its source audit")
        row = {**raw, "partner_id": raw["rbp_id"], "experiment_id": raw["experiment_accession"],
            "assay": "RBNS_SEQUENCE_PREFERENCE", "readout": "PREDICTED_RNA_SEQUENCE_PREFERENCE"}
        event_id = stable_id("RBNSPRED:", [raw[k] for k in ("lncrna_id", "pathway_id", "profile_id")])
        rows.append(base_event(row, source_id=source_id, source_event_id=event_id, route="SEQUENCE_PREDICTED_MEMBER",
            source_cancer="GLOBAL", provenance={"artifacts": artifacts, "rbns_source": source, "annotation_id": audit["annotation_id"]},
            detail={k: raw.get(k) for k in ("profile_id", "file_accession", "concentration", "k", "registered_transcripts",
                "scorable_transcripts", "transcript_coverage", "prediction_score", "mean_log2_enrichment")}))
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def load_rule_sources(specs, root, members):
    if not specs or len({s["source_id"] for s in specs}) != len(specs): raise ValueError("Rule source IDs must be unique and nonempty")
    frames = []; coverage = []
    for spec in specs:
        source_id = spec["source_id"]; kind = spec["kind"]
        if not source_id: raise ValueError("Empty source identity")
        if kind == "native_exact_events":
            paths = {k: verified_path(spec[k], root) for k in ("events", "lineage")}
            frame = native_events(read_table(paths["events"]), read_table(paths["lineage"]), members,
                source_id=source_id, artifacts={k: spec[k] for k in paths})
        elif kind in {"encode_functional", "rbns_sequence"}:
            audit_path = verified_path(spec["audit"], root)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            expected = "ENCODE_FUNCTIONAL_ANALYSIS_AUDIT_V1" if kind == "encode_functional" else "RBNS_SEQUENCE_ANALYSIS_AUDIT_V1"
            if audit.get("format") != expected or audit.get("status") != "PASS": raise ValueError("Incomplete upstream evidence analysis")
            name = "FUNCTIONAL_EVIDENCE.parquet" if kind == "encode_functional" else "GENE_SEQUENCE_PREFERENCE.parquet"
            path = verified_path(spec["table"], root)
            if spec["table"]["sha256"] != audit["outputs"][name]["sha256"]: raise ValueError("Evidence table belongs to a different upstream run")
            table = read_table(path)
            artifacts = {"audit": spec["audit"], "table": spec["table"], "run_id": audit["run_id"]}
            frame = (functional_events(table, audit, source_id=source_id, artifacts=artifacts) if kind == "encode_functional"
                else rbns_events(table, audit, members, source_id=source_id, artifacts=artifacts))
        else: raise ValueError(f"Unsupported current evidence source: {kind}")
        coverage.append({"source_id": source_id, "kind": kind, "normalized_events": len(frame),
            "no_support_means_negative": False})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), coverage
