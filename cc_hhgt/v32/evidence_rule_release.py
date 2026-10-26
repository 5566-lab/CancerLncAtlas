"""Audited rule evidence consumers for exact releases, fusion and queries."""
from __future__ import annotations
import json
from pathlib import Path

import duckdb
import pandas as pd

from .current_core_export import verified_path, sha256, checked_sha
from .evidence_rule_summary import (KEYS, SUMMARY_COLUMNS, attach_candidates, summarize_events,
    update_source_matches, complete_source_coverage)
from .evidence_rule_audit import sql_summary, assert_summary_equal


def _read_binding(path, expected_sha):
    path = verified_path({"path": str(path), "sha256": expected_sha}, Path.cwd())
    binding = json.loads(path.read_text(encoding="utf-8"))
    if (binding.get("format") != "CURRENT_RULE_EVIDENCE_RELEASE_V1" or binding.get("status") != "PASS"
            or binding.get("training_started") is not False or binding.get("neural_score_available") is not False
            or binding.get("changes_primary_ranking") is not False or binding.get("independent_external_validation") is not False
            or binding.get("target_host") != "149"):
        raise ValueError("Invalid observed rule evidence binding")
    artifacts = {k: verified_path(r, path.parent) for k, r in binding["outputs"].items()}
    events = [verified_path(r, path.parent) for r in binding["event_parts"]]
    contexts = [verified_path(r, path.parent) for r in binding["context_parts"]]
    return path, binding, artifacts, events, contexts


def _same_frame(left, right, keys):
    if set(left) != set(right): raise ValueError("Rule lineage table schema changed")
    try:
        pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True)[sorted(left)],
            right.sort_values(keys).reset_index(drop=True)[sorted(right)], check_dtype=False, check_exact=True)
    except AssertionError as error: raise ValueError("Rule lineage differs from its source recomputation") from error


def audit_rule_release(path, *, expected_sha):
    from scripts.materialize_v32_current_rule_evidence import prepare_sources
    path, binding, artifacts, event_paths, context_paths = _read_binding(path, expected_sha)
    manifest_path = verified_path(binding["input_manifest"], path.parent)
    _, candidates, sources, coverage = prepare_sources(manifest_path)
    if len(candidates) != binding["candidate_rows"]: raise ValueError("Rule candidate count changed")
    _same_frame(pd.read_parquet(artifacts["source_events"]), sources, ["source_id", "source_event_id"])
    observed = pd.read_parquet(artifacts["summary"])
    if set(observed.cancer_id) != set(candidates.cancer_id): raise ValueError("Rule summary omitted a cancer")
    expected_parts = []; contexts = []; event_rows = 0; matches = {}
    # Read each cancer's event part without assembling a many-million-row ledger.
    declared_parts = {}
    for part in event_paths:
        frame = pd.read_parquet(part)
        if frame.empty: continue
        if frame.cancer_id.nunique() != 1: raise ValueError("Rule event partitions must each contain one cancer")
        cancer = frame.cancer_id.iloc[0]
        if cancer in declared_parts: raise ValueError("Duplicate rule event cancer partition")
        declared_parts[cancer] = part
    if set(declared_parts)-set(candidates.cancer_id): raise ValueError("Rule event partitions include an undeclared cancer")
    for cancer, universe in candidates.groupby("cancer_id", sort=True):
        expected_events = attach_candidates(sources, universe)
        update_source_matches(matches, expected_events)
        actual = pd.read_parquet(declared_parts[cancer]) if cancer in declared_parts else expected_events.iloc[:0]
        _same_frame(actual, expected_events, ["evidence_record_id"])
        expected_parts.append(sql_summary(actual, universe)); event_rows += len(actual)
        _, context = summarize_events(expected_events, universe)
        if not context.empty: contexts.append(context)
    assert_summary_equal(observed, pd.concat(expected_parts, ignore_index=True))
    _same_frame(pd.read_parquet(artifacts["source_coverage"]), complete_source_coverage(coverage, matches), ["source_id"])
    if contexts:
        if not context_paths: raise ValueError("Rule release omitted experimental contexts")
        _same_frame(pd.concat([pd.read_parquet(p) for p in context_paths], ignore_index=True),
            pd.concat(contexts, ignore_index=True), ["context_id"])
    elif context_paths: raise ValueError("Rule release invented context rows")
    return {"format": "CURRENT_RULE_EVIDENCE_INDEPENDENT_AGGREGATION_AUDIT_V1", "status": "PASS",
        "binding_sha256": expected_sha, "run_id": binding["run_id"], "candidate_rows": len(candidates),
        "event_rows": event_rows, "source_normalization_recomputed": True,
        "summary_recomputed_by_independent_sql": True, "contexts_recomputed": True,
        "biological_external_validation_claimed": False, "training_started": False}


def load_rule_release(path, *, expected_sha):
    path, binding, artifacts, events, contexts = _read_binding(path, expected_sha)
    success = json.loads((path.parent/"SUCCESS.json").read_text(encoding="utf-8"))
    audit_path = path.parent/"INDEPENDENT_AUDIT.json"
    if (success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha
            or sha256(audit_path) != checked_sha(success.get("independent_audit_sha256"), "rule independent audit")):
        raise ValueError("Rule release lacks its completed independent audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit.get("status") != "PASS" or audit.get("binding_sha256") != expected_sha
            or audit.get("summary_recomputed_by_independent_sql") is not True):
        raise ValueError("Rule summary independent audit did not pass")
    # A cached aggregate audit cannot hide changes to the KD/RBNS raw inputs.
    # Legacy source audits retain their historical scope; native releases bind
    # the source manifest and must reconstruct it at subsequent consumption.
    source_path=verified_path(binding["input_manifest"],path.parent)
    for source in json.loads(source_path.read_text(encoding="utf-8"))["sources"]:
        if source["kind"] not in {"encode_functional","rbns_sequence"}:continue
        upstream_path=verified_path(source["audit"],source_path.parent)
        upstream=json.loads(upstream_path.read_text(encoding="utf-8"))
        if "source_contract" in upstream or "input_manifest" in upstream:
            from .native_rbp_workflow import replay_audit
            replay_audit(source["audit"],source_path.parent)
            name="FUNCTIONAL_EVIDENCE.parquet" if source["kind"]=="encode_functional" else "GENE_SEQUENCE_PREFERENCE.parquet"
            verified_path(source["table"],source_path.parent)
            if source["table"]["sha256"]!=upstream["outputs"][name]["sha256"]:
                raise ValueError("Rule source table differs from replayed native analysis")
    return binding, artifacts, events, contexts


def attach_rule_annotations(frame, *, binding_path, expected_sha):
    binding, artifacts, _, _ = load_rule_release(binding_path, expected_sha=expected_sha)
    summary = pd.read_parquet(artifacts["summary"])
    if list(summary) != SUMMARY_COLUMNS or summary.duplicated(KEYS).any() or frame.duplicated(KEYS).any():
        raise ValueError("Rule annotations require unique canonical exact keys")
    keys = frame[KEYS].merge(summary[KEYS], on=KEYS, how="outer", indicator=True)
    if not keys._merge.eq("both").all(): raise ValueError("Rule annotations must cover the exact full release candidate universe")
    annotations = summary.rename(columns={c: "rule_"+c for c in summary if c not in KEYS})
    if (set(annotations)-set(KEYS)) & set(frame): raise ValueError("Rule columns already exist in the selected release")
    result = frame.merge(annotations, on=KEYS, how="left", sort=False, validate="one_to_one")
    # This also catches an accidental reorder or a cast of an existing score.
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), result[list(frame)].reset_index(drop=True))
    result["rule_evidence_run_id"] = binding["run_id"]
    return result


class RuleEvidenceQuery:
    """Parameter-bound access to the same audited summary/context/download files."""
    def __init__(self, binding_path, *, expected_sha):
        self.binding, self.artifacts, self.events, self.contexts = load_rule_release(binding_path, expected_sha=expected_sha)
        self.binding_sha256 = expected_sha

    def query(self, *, cancer_id, lncrna_id, pathway_id=None, table="summary", limit=100, offset=0):
        if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 1000 or not 0 <= offset <= 1000000:
            raise ValueError("Invalid rule evidence pagination")
        if table not in {"summary", "events", "contexts"}: raise ValueError("Unknown rule evidence table")
        paths = [self.artifacts["summary"]] if table == "summary" else self.events if table == "events" else self.contexts
        if not paths: return {"run_id": self.binding["run_id"], "binding_sha256": self.binding_sha256, "rows": []}
        if not isinstance(cancer_id, str) or not isinstance(lncrna_id, str) or not cancer_id or not lncrna_id:
            raise ValueError("Exact cancer and lncRNA identifiers are required")
        filters = "cancer_id=? AND lncrna_id=?"; values = [cancer_id, lncrna_id]
        if pathway_id is not None: filters += " AND pathway_id=?"; values.append(pathway_id)
        connection = duckdb.connect()
        try:
            sort = "pathway_id" + (",evidence_record_id" if table == "events" else ",context_id" if table == "contexts" else "")
            frame = connection.execute(f"SELECT * FROM read_parquet(?, union_by_name=true) WHERE {filters} ORDER BY {sort} LIMIT ? OFFSET ?",
                [[str(p) for p in paths], *values, limit, offset]).df()
        finally: connection.close()
        return {"run_id": self.binding["run_id"], "binding_sha256": self.binding_sha256,
            "rows": json.loads(frame.to_json(orient="records"))}
