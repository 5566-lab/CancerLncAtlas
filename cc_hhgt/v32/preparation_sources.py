"""Capture the files actually read by formal core preparation.

This is immediate input provenance, not a claim that standardized tables have
complete raw-study ancestry. Study identifiers are read only from explicit
source columns, conservatively over the whole consumed file before filtering.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .input_lineage import artifact_sha256

FORMAT = "FORMAL_PREPARATION_SOURCES_V1"
AUTHORITY_KEY = "formal_preparation_sources_sha256"
METADATA_COLUMNS = ("pmid", "doi", "source_dataset", "dataset_id", "dataset_accession", "study_accession",
                    "study_dataset_id", "experiment_id", "experiment_accession")
COMMON_ROLES = {"authorization", "patient_map", "patient_receipt", "detection", "pathway_hierarchy"}
ADJUSTED_ROLES = {"candidate_authority", "train_audit_success", "train_audit_independent", "train_audit_decision",
                  "validation_audit_success", "validation_audit_independent", "train_associations", "validation_associations"}
LEGACY_ROLES = {"expression", "activity"}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def metadata_tokens(row):
    from .external_validation_sources import _tokens
    tokens = set(json.loads(_tokens(row)))
    # The normalization reader prioritizes one dataset/experiment column. Here
    # every explicit column must be retained when a source carries several.
    for key in ("source_dataset", "dataset_id", "dataset_accession", "study_dataset_id"):
        if key in row: tokens.update(json.loads(_tokens({"source_dataset": row[key]})))
    for key in ("experiment_id", "experiment_accession"):
        if key in row: tokens.update(json.loads(_tokens({"experiment_id": row[key]})))
    return tokens


def parquet_metadata(path):
    """No filename inference and no patient identifiers in the returned record."""
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path)
    columns = sorted(set(table.schema_arrow.names) & set(METADATA_COLUMNS))
    tokens = set()
    if columns:
        for batch in table.iter_batches(columns=columns, batch_size=65536):
            for row in batch.to_pandas().drop_duplicates().to_dict("records"):
                tokens.update(metadata_tokens(row))
    return {"metadata_columns": columns, "study_tokens": sorted(tokens)}


def table_metadata(path):
    if Path(path).suffix == ".parquet": return parquet_metadata(path)
    import pandas as pd
    separator = "," if ".csv" in Path(path).suffixes else "\t"
    columns = sorted(set(pd.read_csv(path, sep=separator, nrows=0).columns) & set(METADATA_COLUMNS))
    tokens = set()
    if columns:
        for chunk in pd.read_csv(path, sep=separator, usecols=columns, dtype=str, chunksize=65536):
            for row in chunk.drop_duplicates().to_dict("records"): tokens.update(metadata_tokens(row))
    return {"metadata_columns": columns, "study_tokens": sorted(tokens)}


def source_members(path, kind):
    path = Path(path).resolve(strict=True)
    if kind == "table":
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        if not files: raise ValueError("Preparation input contains no Parquet files")
    elif kind == "control" and path.is_file(): files = [path]
    else: raise ValueError("Unsupported preparation input kind")
    result = []
    for file in files:
        item = {"path": str(file), "sha256": artifact_sha256(file),
                **(table_metadata(file) if kind == "table" else {"metadata_columns": [], "study_tokens": []})}
        if kind == "table" and file.suffix == ".parquet":
            from .formal_expression_sources import binding
            upstream = binding(file)
            if upstream: item["upstream_expression"] = upstream
            from .fold_local_activity import binding as activity_binding
            activity = activity_binding(file)
            if activity: item["upstream_activity"] = activity
            from .fold_local_scope import scope_binding, candidate_binding
            scope = scope_binding(file); candidates = candidate_binding(file)
            if scope: item["upstream_scope"] = scope
            if candidates: item["upstream_candidates"] = candidates
        result.append(item)
    return result


class SourceCapture:
    """Wrap real reads; changes during or between reads fail before serialization."""
    def __init__(self): self.reads = []

    def attach_transformation(self,path,receipt):
        from .local_cnv_sources import partition_binding
        binding=partition_binding(path,receipt)
        if binding is not None:
            selected=[r for r in self.reads if r["path"]==str(Path(path).resolve())]
            if not selected:raise ValueError("Transformation has no actual preparation read")
            for item in selected:item["upstream_transformation"]=binding

    def read(self, path, reader, *, role, kind="table", fold=-1, cancer_id="", expected_sha=None):
        path = Path(path).resolve(strict=True)
        before = artifact_sha256(path)
        if expected_sha is not None and before != expected_sha:
            raise ValueError("Preparation source SHA256 mismatch: " + role)
        members = source_members(path, kind)
        value = reader(path)
        if artifact_sha256(path) != before:
            raise ValueError("Preparation source changed during read: " + role)
        for member in members:
            if member.get("upstream_expression"):
                from .formal_expression_sources import validate
                validate(member["upstream_expression"],path.parent,member["path"])
            if member.get("upstream_activity"):
                from .fold_local_activity import validate
                validate(member["upstream_activity"],path.parent,member["path"])
            from .fold_local_scope import validate_scope, validate_candidates
            for key,check in (("upstream_scope",validate_scope),("upstream_candidates",validate_candidates)):
                if member.get(key):check(member[key],path.parent,member["path"])
        record = {"role": role, "path": str(path), "sha256": before, "kind": kind,
                  "fold": fold, "cancer_id": str(cancer_id), "members": members}
        for old in self.reads:
            if (old["role"], old["path"], old["fold"], old["cancer_id"]) == (role, str(path), fold, str(cancer_id)):
                if {k:v for k,v in old.items() if k!="upstream_transformation"} != record:
                    raise ValueError("Preparation source changed between reads: " + role)
                return value
        self.reads.append(record)
        return value

    def receipt(self, *, fold, mode, cancers):
        records = [r for r in self.reads if r["fold"] in (-1, fold)]
        for item in records:
            if artifact_sha256(item["path"]) != item["sha256"]:
                raise ValueError("Preparation source changed before output: " + item["role"])
            for member in item["members"]:
                if member.get("upstream_expression"):
                    from .formal_expression_sources import validate
                    validate(member["upstream_expression"],Path(item["path"]).parent,member["path"])
                if member.get("upstream_activity"):
                    from .fold_local_activity import validate
                    validate(member["upstream_activity"],Path(item["path"]).parent,member["path"])
                from .fold_local_scope import validate_scope, validate_candidates
                for key,check in (("upstream_scope",validate_scope),("upstream_candidates",validate_candidates)):
                    if member.get(key):check(member[key],Path(item["path"]).parent,member["path"])
            if item.get("upstream_transformation"):
                for source in item["upstream_transformation"].values():
                    if artifact_sha256(source["path"])!=source["sha256"]:
                        raise ValueError("Preparation transformation source changed before output")
        value = {"format": FORMAT, "patient_fold": fold, "association_mode": mode,
                 "cancer_ids": sorted(set(map(str, cancers))), "source_reads": records,
                 "metadata_scope": "WHOLE_CONSUMED_FILE_BEFORE_ROW_FILTERING",
                 "raw_study_ancestry_complete": False,
                 "baseline_fit": {"method": "LOGISTIC_L1_FIXED_C", "C": 0.1,
                                  "fit_split": "train", "validation_used_for_fitting": False}}
        validate_receipt(value, fold=fold)
        return value


def validate_receipt(value, *, fold):
    if (value.get("format") != FORMAT or value.get("patient_fold") != fold
            or value.get("raw_study_ancestry_complete") is not False
            or value.get("metadata_scope") != "WHOLE_CONSUMED_FILE_BEFORE_ROW_FILTERING"
            or value.get("baseline_fit") != {"method": "LOGISTIC_L1_FIXED_C", "C": 0.1,
                                             "fit_split": "train", "validation_used_for_fitting": False}):
        raise ValueError("Invalid formal preparation source receipt")
    mode = value.get("association_mode")
    if mode not in {"legacy", "local-cnv-adjusted"}: raise ValueError("Invalid preparation association mode")
    cancers = value.get("cancer_ids")
    if not isinstance(cancers, list) or not cancers or cancers != sorted(set(cancers)) or any(not isinstance(c, str) or not c for c in cancers):
        raise ValueError("Invalid preparation cancer scope")
    expected = COMMON_ROLES | (ADJUSTED_ROLES if mode == "local-cnv-adjusted" else LEGACY_ROLES)
    reads = value.get("source_reads", [])
    if {r["role"] for r in reads} != expected: raise ValueError("Preparation source roles are incomplete or unexpected")
    seen = set()
    for item in reads:
        role = item["role"]
        per_cancer = role in {"expression", "train_associations", "validation_associations"}
        per_fold = role in {"train_associations", "validation_associations"} or (
            role == "detection" and any(m.get("upstream_scope") for m in item["members"]))
        identity = (role, item["cancer_id"])
        if identity in seen: raise ValueError("Duplicate preparation source role/cancer")
        seen.add(identity)
        if item["fold"] != (fold if per_fold else -1) or item["cancer_id"] not in (cancers if per_cancer else [""]):
            raise ValueError("Preparation source fold/cancer differs")
        kind = "table" if role in {"detection", "pathway_hierarchy", "candidate_authority", "train_associations", "validation_associations", "expression", "activity"} else "control"
        if item["kind"] != kind or not item["members"]: raise ValueError("Preparation source reader differs")
        for source in [item, *item["members"]]:
            digest = source.get("sha256", "")
            if not source.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Preparation source lacks file identity")
        transformation=item.get("upstream_transformation")
        if transformation is not None:
            if set(transformation)!={"receipt","sources","output"} or transformation["output"]!={"path":item["path"],"sha256":item["sha256"]}:
                raise ValueError("Preparation transformation output differs from actual read")
        for member in item["members"]:
            if (member["metadata_columns"] != sorted(set(member["metadata_columns"]))
                    or not set(member["metadata_columns"]).issubset(METADATA_COLUMNS)
                    or member["study_tokens"] != sorted(set(member["study_tokens"]))
                    or any(not isinstance(t, str) or not t for t in member["study_tokens"])
                    or (kind == "control" and (member["metadata_columns"] or member["study_tokens"]))):
                raise ValueError("Invalid source metadata declaration")
    for role in expected:
        required_cancers = cancers if role in {"expression", "train_associations", "validation_associations"} else [""]
        if {c for r, c in seen if r == role} != set(required_cancers):
            raise ValueError("Preparation source cancer coverage is incomplete")
    return value


def attach_sources(payload, receipt):
    validate_receipt(receipt, fold=payload["patient_fold"])
    payload["preparation_sources"] = receipt
    digest = canonical_sha(receipt)
    payload["artifact_hashes"] = {**payload.get("artifact_hashes", {}), AUTHORITY_KEY: digest}
    if "input_authority_hashes" in payload:
        payload["input_authority_hashes"] = {**payload["input_authority_hashes"], AUTHORITY_KEY: digest}


def validate_payload_sources(payload):
    """Portable structural check; source files need not be on a training host."""
    mappings = [payload.get(k, {}) for k in ("artifact_hashes", "input_authority_hashes")]
    receipt = payload.get("preparation_sources")
    if receipt is None:
        if any(AUTHORITY_KEY in m for m in mappings): raise ValueError("Hash-bound preparation sources are missing")
        return None  # Historical payloads remain explicitly unresolved downstream.
    validate_receipt(receipt, fold=payload["patient_fold"])
    digest = canonical_sha(receipt)
    authority = payload.get("input_authority_hashes", payload.get("artifact_hashes", {}))
    if authority.get(AUTHORITY_KEY) != digest or any(AUTHORITY_KEY in m and m[AUTHORITY_KEY] != digest for m in mappings):
        raise ValueError("Preparation source receipt SHA256 differs from native authority")
    if payload.get("input_scope", {}).get("association_mode") != receipt["association_mode"]:
        raise ValueError("Preparation source mode differs from payload")
    return receipt
