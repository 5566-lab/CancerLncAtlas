"""Capture files at native CNV reads and verify them before downstream reuse.

These private receipts describe consumed files/columns, not fitted rows. Array
integrity and a fold-local label file do not prove upstream biological isolation.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd

from .current_core_export import sha256

FORMAT = "NATIVE_CNV_CONSUMED_SOURCES_V1"
IDS = ("patient_ids.json", "lncrna_ids.json", "pathway_ids.json")
ARRAYS = {
    "cnv": tuple(f"{entity}_{feature}.npy" for entity in ("lncrna", "pathway")
                 for feature in ("signed_mean", "callable", "amplification", "deletion")),
    "cnv_event": tuple(f"{entity}_{feature}.npy" for entity in ("lncrna", "pathway")
                       for feature in ("event", "callable", "burden")),
}


def require_host():
    if socket.gethostname() != "149":
        raise RuntimeError("CNV execution requires target host 149")


class SourceCapture:
    def __init__(self):
        self.records = {}

    def read(self, path, reader, *, role, cancer="", fold=-1, split="", columns=()):
        path = Path(path).resolve()
        before = sha256(path)
        value = reader(path)
        if sha256(path) != before:
            raise ValueError("CNV input changed while reading: " + str(path))
        item = dict(path=str(path), sha256=before, role=role, cancer=cancer,
                    fold=fold, split=split, columns=list(columns))
        key = (str(path), role, cancer, fold, split)
        previous = self.records.get(key)
        if previous is not None and previous != item:
            raise ValueError("CNV input changed between reads: " + str(path))
        self.records[key] = item
        return value

    def json(self, path, **context):
        return self.read(path, lambda p: json.loads(p.read_text(encoding="utf-8")), **context)

    def array(self, path, **context):
        return self.read(path, lambda p: np.load(p, mmap_mode="r", allow_pickle=False), **context)

    def check_partition(self, root, receipt):
        """Reject disagreement with existing materialization hashes before fit."""
        root = Path(root).resolve()
        declared = receipt.get("files", {})
        for item in self.records.values():
            path = Path(item["path"])
            if path.parent == root and path.name in declared and item["sha256"] != declared[path.name]:
                raise ValueError("CNV materializer and consumed matrix hashes differ")

    def write(self, path, *, kind, run_id):
        require_host()
        if kind not in ARRAYS or not run_id:
            raise ValueError("CNV source receipt requires native kind and run")
        path = Path(path)
        if path.exists():
            raise ValueError("Refuse to overwrite CNV consumed sources")
        records = list(self.records.values())
        for item in records:
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("CNV consumed file changed before receipt: " + item["path"])
        payload = dict(format=FORMAT, target_host="149", producer_kind=kind,
                       training_run_id=run_id, records=records,
                       exposure_scope="WHOLE_CONSUMED_FILE_NOT_ALL_ROWS_FITTED")
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {"path": str(path.resolve()), "sha256": sha256(path)}


def validate_consumed_sources(sources, item, root, *, kind, run_id, candidates, folds, core):
    """Validate actual native reads; no model initialization, fit or forward."""
    receipt_path, receipt = sources.document(item, root)
    if (receipt.get("format") != FORMAT or receipt.get("target_host") != "149"
            or receipt.get("producer_kind") != kind or receipt.get("training_run_id") != run_id
            or receipt.get("exposure_scope") != "WHOLE_CONSUMED_FILE_NOT_ALL_ROWS_FITTED"):
        raise ValueError("CNV consumed-source identity differs")
    records = receipt.get("records", [])
    grouped = {}
    seen = set()
    for record in records:
        path = sources.bind(record, receipt_path.parent)
        identity = (str(path), record["role"], record["cancer"], record["fold"], record["split"])
        if identity in seen:
            raise ValueError("CNV duplicate consumed-source read")
        seen.add(identity)
        grouped.setdefault(record["role"], []).append((record, path))
    expected_roles = {"candidates", "folds", "core", "compact", "materialization_success"}
    if kind == "cnv":
        expected_roles |= {"association_success", "formal_labels"}
    if set(grouped) != expected_roles:
        raise ValueError("CNV consumed-source roles are incomplete or unexpected")
    for role, expected in (("candidates", candidates), ("folds", folds), ("core", core)):
        if len(grouped[role]) != 1 or grouped[role][0][0]["sha256"] != expected["sha256"]:
            raise ValueError("CNV consumed authority differs: " + role)
    candidate_path = grouped["candidates"][0][1]
    candidate_frame = pd.read_parquet(candidate_path, columns=["cancer_id", "lncrna_id", "pathway_id"])
    cancers = set(candidate_frame.cancer_id.astype(str).str.upper())
    compact = {}
    for record, path in grouped["compact"]:
        if record["fold"] != -1 or record["split"] or record["columns"]:
            raise ValueError("CNV compact read scope differs")
        cancer = record["cancer"]
        members = compact.setdefault(cancer, {})
        if path.name in members:
            raise ValueError("CNV compact file identity is duplicated")
        members[path.name] = path
    if set(compact) != cancers:
        raise ValueError("CNV consumed cancer partitions differ from candidates")
    for cancer, members in compact.items():
        if set(members) != set(IDS + ARRAYS[kind] + ("SUCCESS.json",)):
            raise ValueError("CNV consumed matrix/identifier partition is incomplete")
        if len({p.parent for p in members.values()}) != 1:
            raise ValueError("CNV compact files come from different partitions")
        ids = [json.loads(members[name].read_text(encoding="utf-8")) for name in IDS]
        if any(not isinstance(v, list) or len(v) != len(set(v)) for v in ids):
            raise ValueError("CNV compact identifiers are invalid or duplicated")
        for name in ARRAYS[kind]:
            value = np.load(members[name], mmap_mode="r", allow_pickle=False)
            if value.shape != (len(ids[0]), len(ids[1 if name.startswith("lncrna_") else 2])):
                raise ValueError("CNV compact array/identifier shape differs")
        success = json.loads(members["SUCCESS.json"].read_text(encoding="utf-8"))
        if success.get("status") != "SUCCESS":
            raise ValueError("CNV compact materialization is incomplete")
        for name, digest in success.get("files", {}).items():
            if name in members and sha256(members[name]) != digest:
                raise ValueError("CNV materializer and consumed matrix hashes differ")
        if kind == "cnv" and (success.get("signed_values_preserved") is not True
                or success.get("amplification_deletion_separate") is not True
                or success.get("absolute_burden_used_as_continuous") is not False):
            raise ValueError("CNV directional semantics differ")
    for role in expected_roles & {"materialization_success", "association_success"}:
        if len(grouped[role]) != 1 or json.loads(grouped[role][0][1].read_text()).get("status") != "SUCCESS":
            raise ValueError("CNV aggregate input is incomplete")
    aggregate_path = grouped["materialization_success"][0][1]
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if set(aggregate.get("cancers", {})) != cancers:
        raise ValueError("CNV materializer cancer partitions differ")
    for cancer, member in aggregate["cancers"].items():
        declared = Path(member["path"])
        declared = declared if declared.is_absolute() else aggregate_path.parent/declared
        actual = compact[cancer]["SUCCESS.json"]
        if declared.resolve() != actual.parent or member.get("success_sha256") != sha256(actual):
            raise ValueError("CNV aggregate and consumed compact partition differ")
    if kind == "cnv":
        from .cnv_only_training_v2 import _formal_labels
        association_root = grouped["association_success"][0][1].parent
        labels = {}
        for record, path in grouped["formal_labels"]:
            key = (record["cancer"], record["fold"], record["split"])
            if key in labels:
                raise ValueError("CNV formal label consumption is duplicated")
            labels[key] = path
            column = "label_A0_full" if record["split"] == "train" else "label_A0_validation"
            if record["columns"] != ["lncrna_id", "pathway_id", column, "run_id"]:
                raise ValueError("CNV consumed label columns differ")
            cancer, fold, split = key
            expected_suffix = Path("pair_level") / f"cancer={cancer}" / f"fold={fold}" / "part.parquet"
            if tuple(path.parts[-4:]) != expected_suffix.parts:
                raise ValueError("CNV fold-local label partition differs")
            if path.parents[3] != association_root:
                raise ValueError("CNV consumed label association root differs")
            local = candidate_frame.loc[candidate_frame.cancer_id.eq(cancer)]
            _formal_labels(path.parents[3], cancer, fold, local, split)
        if set(labels) != {(c, f, s) for c in cancers for f in range(5) for s in ("train", "validation")}:
            raise ValueError("CNV consumed fold-local labels are incomplete")
    return receipt_path, receipt
