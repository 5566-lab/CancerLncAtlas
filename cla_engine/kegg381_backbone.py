"""Freeze the user-selected 381 historical KEGG MEDICUS definitions.

Cancer annotations describe each definition and never filter this backbone.
Reference, variant, pathogen, and environmental mechanisms retain separate IDs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

FORMAT = "CLA_KEGG381_BACKBONE_V1"
FIELDS = (
    "backbone_id", "source_target_id", "source_name", "systematic_id",
    "network_id", "canonical_id", "display_name", "network_name", "network_type",
    "original_gene_count", "original_members_sha256",
    "official_cancer_supported", "cancer_scope_status", "cancer_map_ids",
    "backbone_inclusion_reason", "is_backbone",
    "exact_source_evidence", "metadata_archive_version",
)
MEMBER_FIELDS = ("source_target_id", "backbone_id", "network_id",
                 "canonical_id", "ensembl_gene_id")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def members_sha256(members):
    """Digest original ordered member list as compact UTF-8 JSON."""
    raw = json.dumps(list(members), ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unique_index(rows, key, subset=None):
    result = {}
    for row in rows:
        value = row.get(key, "")
        if subset is not None and value not in subset:
            continue
        if not value:
            raise ValueError("missing_" + key)
        if value in result:
            raise ValueError("duplicate_" + key + ":" + value)
        result[value] = row
    return result


def build_backbone(source_ids, mapping_rows, scope_rows, gene_sets,
                   catalog_rows, expected_count=381):
    source_ids = list(source_ids)
    if len(source_ids) != expected_count:
        raise ValueError("unexpected_backbone_count:" + str(len(source_ids)))
    if len(set(source_ids)) != expected_count:
        raise ValueError("duplicate_source_target_id")
    selected = set(source_ids)
    mappings = _unique_index(mapping_rows, "source_target_id", selected)
    scopes = _unique_index(scope_rows, "source_target_id", selected)
    catalog = _unique_index(catalog_rows, "canonical_id")
    if selected - mappings.keys():
        raise ValueError("missing_source_mapping:" + "|".join(sorted(selected - mappings.keys())))
    if selected - scopes.keys():
        raise ValueError("missing_scope_annotation:" + "|".join(sorted(selected - scopes.keys())))
    output, members = [], {}
    for sid in source_ids:
        row, scope = mappings[sid], scopes[sid]
        if row.get("mapping_status") != "mapped":
            raise ValueError("unverified_source_mapping:" + sid)
        nid, cid = row["network_id"], row["canonical_id"]
        if not re.fullmatch(r"N[0-9]{5}", nid):
            raise ValueError("invalid_network_id:" + sid)
        if scope.get("network_id") != nid or scope.get("canonical_id") != cid:
            raise ValueError("scope_identity_conflict:" + sid)
        if cid not in catalog:
            raise ValueError("canonical_absent_from_current_catalog:" + cid)
        if sid not in gene_sets or not gene_sets[sid]:
            raise ValueError("missing_original_members:" + sid)
        original = list(gene_sets[sid])
        if len(original) != len(set(original)):
            raise ValueError("duplicate_original_members:" + sid)
        if any(not re.fullmatch(r"ENSG[0-9]{11}(?:\.[0-9]+)?", g) for g in original):
            raise ValueError("invalid_original_ensembl_member:" + sid)
        members[sid] = original
        supported = scope.get("admitted_by_network_scope", "").lower() == "true"
        if supported != (scope.get("cancer_scope_status") == "direct_cancer_network"):
            raise ValueError("cancer_scope_annotation_conflict:" + sid)
        output.append({
            "backbone_id": "KEGG.NETWORK:" + nid,
            "source_target_id": sid, "source_name": row["source_name"],
            "systematic_id": row["systematic_id"], "network_id": nid,
            "canonical_id": cid, "display_name": catalog[cid]["display_name"],
            "network_name": scope.get("network_name", ""),
            "network_type": scope.get("network_type", ""),
            "original_gene_count": len(original),
            "original_members_sha256": members_sha256(original),
            "official_cancer_supported": str(supported),
            "cancer_scope_status": scope["cancer_scope_status"],
            "cancer_map_ids": scope.get("cancer_map_ids", ""),
            "backbone_inclusion_reason": "user_fixed_historical_381",
            "is_backbone": "True",
            "exact_source_evidence": row.get("exact_source_evidence", ""),
            "metadata_archive_version": row.get("metadata_archive_version", ""),
        })
    for key in ("network_id", "canonical_id", "backbone_id"):
        _unique_index(output, key)
    summary = {
        "backbone_count": len(output),
        "official_cancer_supported_count": sum(r["official_cancer_supported"] == "True" for r in output),
        "cancer_scope_status_counts": dict(sorted(Counter(r["cancer_scope_status"] for r in output).items())),
        "network_type_counts": dict(sorted(Counter(r["network_type"] or "missing" for r in output).items())),
        "original_member_rows": sum(len(v) for v in members.values()),
        "unique_original_genes": len({g for values in members.values() for g in values}),
        "selection_rule": "exact user-fixed historical 381 source definitions",
        "official_cancer_annotation_is_filter": False,
        "reference_variant_identity_preserved": True,
        "original_members_preserved": True,
    }
    return output, members, summary


def materialize_backbone(output, rows, gene_sets, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "KEGG381_BACKBONE.tsv"
    genes = output / "GENE_SETS.json"
    member_table = output / "MEMBERS.tsv"
    source_list = output / "SOURCE_IDS.txt"
    write_tsv(table, rows, FIELDS)
    genes.write_text(json.dumps(gene_sets, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    member_rows = [
        {"source_target_id": row["source_target_id"], "backbone_id": row["backbone_id"],
         "network_id": row["network_id"], "canonical_id": row["canonical_id"],
         "ensembl_gene_id": gene}
        for row in rows for gene in gene_sets[row["source_target_id"]]
    ]
    write_tsv(member_table, member_rows, MEMBER_FIELDS)
    source_list.write_text("\n".join(row["source_target_id"] for row in rows) + "\n", encoding="utf-8")
    receipt = dict(manifest)
    receipt["format"] = FORMAT
    receipt["files"] = {path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                        for path in (table, genes, member_table, source_list)}
    receipt["ordered_source_network_canonical_sha256"] = members_sha256([
        [r["source_target_id"], r["network_id"], r["canonical_id"]] for r in rows])
    receipt["member_digest_definition"] = "SHA256 of ordered original members as compact ensure_ascii UTF-8 JSON"
    (output / "MANIFEST.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def load_backbone(path, verify=True):
    path = Path(path)
    manifest = json.loads((path / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported_backbone_format")
    if verify:
        for name in ("KEGG381_BACKBONE.tsv", "GENE_SETS.json", "MEMBERS.tsv", "SOURCE_IDS.txt"):
            expected = manifest.get("files", {}).get(name, {}).get("sha256")
            if not expected or file_sha256(path / name) != expected:
                raise ValueError("backbone_checksum_mismatch:" + name)
    rows = read_tsv(path / "KEGG381_BACKBONE.tsv")
    gene_sets = json.loads((path / "GENE_SETS.json").read_text(encoding="utf-8"))
    source_ids = (path / "SOURCE_IDS.txt").read_text(encoding="utf-8").splitlines()
    if len(rows) != manifest["backbone_count"] or [r["source_target_id"] for r in rows] != source_ids:
        raise ValueError("backbone_axis_mismatch")
    for key in ("source_target_id", "network_id", "canonical_id", "backbone_id"):
        _unique_index(rows, key)
    if set(gene_sets) != set(source_ids):
        raise ValueError("backbone_gene_set_axis_mismatch")
    for row in rows:
        original = gene_sets[row["source_target_id"]]
        if len(original) != int(row["original_gene_count"]) or members_sha256(original) != row["original_members_sha256"]:
            raise ValueError("backbone_members_mismatch:" + row["source_target_id"])
        if row["is_backbone"] != "True" or row["backbone_id"] != "KEGG.NETWORK:" + row["network_id"]:
            raise ValueError("backbone_identity_mismatch")
    mapping_digest = members_sha256([[r["source_target_id"], r["network_id"], r["canonical_id"]] for r in rows])
    if mapping_digest != manifest["ordered_source_network_canonical_sha256"]:
        raise ValueError("backbone_mapping_digest_mismatch")
    if verify:
        expected_members = [(r["source_target_id"], r["backbone_id"], r["network_id"],
                             r["canonical_id"], gene)
                            for r in rows for gene in gene_sets[r["source_target_id"]]]
        actual_members = [tuple(r[k] for k in MEMBER_FIELDS) for r in read_tsv(path / "MEMBERS.tsv")]
        if actual_members != expected_members:
            raise ValueError("backbone_member_table_mismatch")
    return {"manifest": manifest, "rows": rows, "gene_sets": gene_sets}
