"""Traceable MSigDB source-definition to KEGG NETWORK mapping.

Identifiers come from verified source metadata or explicit reviewed crosswalks.
Names, member overlap, and patient associations never supply an identifier.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

NETWORK_RE = re.compile(r"(?<![A-Za-z0-9])N[0-9]{5}(?![A-Za-z0-9])")
FIELDS = (
    "source_target_id", "source_name", "systematic_id", "network_id",
    "canonical_id", "original_canonical_id", "exact_source_evidence",
    "metadata_source_url", "metadata_sha256", "metadata_archive_version", "metadata_status",
    "mapping_status", "mapping_issue",
)


def network_ids(value):
    """Extract explicit N accessions, never infer them from a biological name."""
    return sorted(set(NETWORK_RE.findall(str(value or ""))))


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _redirect_graph(redirects, definitions):
    graph = defaultdict(set)
    for row in redirects:
        old, new = row.get("previous_internal_id", ""), row.get("canonical_id", "")
        if old and new and old != new:
            graph[old].add(new)
    for row in definitions:
        new = row.get("integrated_canonical_id", "")
        for key in ("canonical_id", "node_id"):
            old = row.get(key, "")
            if old and new and old != new:
                graph[old].add(new)
    return graph


def _resolve(identifier, graph, trail=()):
    if not identifier:
        return set(), ["missing_canonical"]
    if identifier in trail:
        return set(), ["canonical_redirect_cycle:" + "|".join((*trail, identifier))]
    if identifier not in graph:
        return {identifier}, []
    output, issues = set(), []
    for target in sorted(graph[identifier]):
        terminal, branch_issues = _resolve(target, graph, (*trail, identifier))
        output.update(terminal)
        issues.extend(branch_issues)
    return output, issues


def _metadata_evidence(row, source_name, verify_cache):
    ids = network_ids(row.get("exact_source"))
    external_ids = sorted(set(itertools.chain.from_iterable(
        network_ids(item) for item in row.get("external_details", []))))
    issues = []
    valid = row.get("status") == "VERIFIED_SOURCE_METADATA"
    if not valid:
        return [], [], False
    if external_ids and ids and external_ids != ids:
        issues.append("exact_source_external_url_conflict")
        ids = sorted(set(ids + external_ids))
    if verify_cache:
        path = row.get("path", "")
        try:
            actual = sha256_file(path)
            if not row.get("sha256") or actual != row["sha256"]:
                issues.append("metadata_checksum_mismatch")
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            entry = raw.get(source_name)
            if not isinstance(entry, dict):
                issues.append("raw_metadata_missing_source")
            else:
                raw_ids = network_ids(entry.get("exactSource"))
                if raw_ids != network_ids(row.get("exact_source")):
                    issues.append("raw_exact_source_conflict")
                    ids = sorted(set(ids + raw_ids))
                if entry.get("systematicName", "") != row.get("systematic_name", ""):
                    issues.append("raw_systematic_id_conflict")
        except (OSError, ValueError, TypeError) as error:
            issues.append("raw_metadata_unreadable:" + type(error).__name__)
    return ids, issues, not issues


def build_mapping(registry_rows, metadata_rows, bindings, definition_rows,
                  redirects, overrides=(), verify_cache=True):
    """Return one row per source / candidate N / canonical and a summary.

    Only status 'mapped' is safe for automatic classification. Multiple source
    definitions of the same N remain separate rows. Ambiguities are preserved.
    """
    metadata = defaultdict(list)
    binding_index = defaultdict(set)
    override_index = defaultdict(list)
    for row in metadata_rows:
        metadata[row.get("source_name", "")].append(row)
    for row in bindings:
        if row.get("canonical_id"):
            binding_index[row.get("source_target_id", "")].add(row["canonical_id"])
    for row in overrides:
        override_index[row.get("source_name", "")].append(row)
    graph = _redirect_graph(redirects, definition_rows)
    output = []
    sources = [row for row in registry_rows
               if row.get("source_database") == "KEGG_MEDICUS"]
    counts = Counter(row["source_target_id"] for row in sources)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError("duplicate_source_target_id:" + "|".join(duplicates))
    for source in sources:
        sid, name = source["source_target_id"], source["source_name"]
        metas = metadata.get(name, [])
        networks, systematic, evidence, urls, hashes, meta_status = set(), set(), set(), set(), set(), set()
        issues, valid_evidence = [], False
        archive_versions = set()
        for row in metas:
            if row.get("archive_version"):
                archive_versions.add(row["archive_version"])
            if row.get("systematic_name"):
                systematic.add(row["systematic_name"])
            meta_status.add(row.get("status", ""))
            ids, row_issues, valid = _metadata_evidence(row, name, verify_cache)
            networks.update(ids)
            issues.extend(row_issues)
            if ids:
                evidence.add("msigdb_exact_source")
                valid_evidence |= valid
            if row.get("url"):
                urls.add(row["url"])
            if row.get("sha256"):
                hashes.add(row["sha256"])
        for row in override_index.get(name, []):
            ids = network_ids(row.get("network_id"))
            url_ids = network_ids(row.get("source_url"))
            if len(ids) != 1 or ids != url_ids or not row.get("evidence", "").strip():
                issues.append("invalid_reviewed_override")
                continue
            networks.update(ids)
            evidence.add("reviewed_xref:" + row["evidence"])
            urls.add(row["source_url"])
            if row.get("metadata_sha256"):
                hashes.add(row["metadata_sha256"])
            valid_evidence = True
        originals = binding_index.get(sid) or {source.get("canonical_id", "")}
        canonical = set()
        canonical_issues = []
        for identifier in sorted(originals):
            resolved, found_issues = _resolve(identifier, graph)
            canonical.update(resolved)
            canonical_issues.extend(found_issues)
        issues.extend(canonical_issues)
        if len(systematic) > 1:
            issues.append("multiple_systematic_ids")
        if len(networks) > 1:
            status = "ambiguous_network"
        elif issues:
            status = "unresolved_canonical" if canonical_issues else "invalid_metadata"
        elif len(canonical) > 1:
            status = "ambiguous_canonical"
        elif not canonical:
            status = "unresolved_canonical"
        elif not networks:
            status = "unmapped_network" if metas else "missing_metadata"
        elif not valid_evidence:
            status = "invalid_metadata"
        else:
            status = "mapped"
        for nid, cid in itertools.product(sorted(networks) or [""], sorted(canonical) or [""]):
            output.append({
                "source_target_id": sid, "source_name": name,
                "systematic_id": "|".join(sorted(systematic)), "network_id": nid,
                "canonical_id": cid, "original_canonical_id": source.get("canonical_id", ""),
                "exact_source_evidence": " | ".join(sorted(evidence)),
                "metadata_source_url": "|".join(sorted(urls)),
                "metadata_sha256": "|".join(sorted(hashes)),
                "metadata_archive_version": "|".join(sorted(archive_versions)),
                "metadata_status": "|".join(sorted(meta_status)),
                "mapping_status": status, "mapping_issue": "|".join(sorted(set(issues))),
            })
    source_status = {}
    for row in output:
        source_status[row["source_target_id"]] = row["mapping_status"]
    return output, {
        "source_definition_count": len(sources),
        "output_row_count": len(output),
        "source_status_counts": dict(sorted(Counter(source_status.values()).items())),
        "mapped_distinct_network_ids": len({r["network_id"] for r in output if r["mapping_status"] == "mapped"}),
        "mapped_distinct_canonical_ids": len({r["canonical_id"] for r in output if r["mapping_status"] == "mapped"}),
        "identifier_inferred_from_name": False,
        "raw_metadata_verified": verify_cache,
    }


def write_mapping(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_msigdb_archive(path, source_url, version):
    """Read official release JSON as source metadata, preserving its version.

    This returns metadata records only. It does not update gene-set membership.
    """
    path = Path(path)
    archive = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(archive, dict):
        raise ValueError("archive must be a gene-set-name keyed object")
    digest = sha256_file(path)
    rows = []
    for name, entry in archive.items():
        if not name.startswith("KEGG_MEDICUS_"):
            continue
        if not isinstance(entry, dict):
            raise ValueError("invalid archive entry:" + name)
        external = entry.get("externalDetailsURL", [])
        if isinstance(external, str):
            external = [external]
        rows.append({
            "source_name": name, "exact_source": entry.get("exactSource", ""),
            "systematic_name": entry.get("systematicName", ""),
            "collection": entry.get("collection", ""),
            "members": entry.get("geneSymbols", []),
            "external_details": external, "url": source_url,
            "sha256": digest, "path": str(path),
            "status": "VERIFIED_SOURCE_METADATA", "archive_version": version,
        })
    return rows


def audit_archive_membership(registry_rows, archive_rows, source_gene_sets,
                             hgnc_rows, previous_metadata):
    """Compare members without altering the original catalog or score inputs.

    Exact approved HGNC symbol to Ensembl mapping is used. Unmapped or ambiguous
    symbols are listed so annotation differences are not hidden.
    """
    symbol_map = defaultdict(set)
    for row in hgnc_rows:
        if row.get("status") == "Approved" and row.get("ensembl_gene_id"):
            symbol_map[row["symbol"]].add(row["ensembl_gene_id"].split(".")[0])
    archives = {row["source_name"]: row for row in archive_rows}
    previous = {row["source_name"]: row for row in previous_metadata}
    rows = []
    for source in registry_rows:
        if source.get("source_database") != "KEGG_MEDICUS":
            continue
        name, sid = source["source_name"], source["source_target_id"]
        if name not in archives:
            continue
        archive = archives[name]
        symbols = set(archive.get("members", []))
        mapped = set()
        unmapped, ambiguous = [], []
        for symbol in sorted(symbols):
            ids = symbol_map.get(symbol, set())
            if len(ids) == 1:
                mapped.update(ids)
            elif len(ids) > 1:
                ambiguous.append(symbol)
            else:
                unmapped.append(symbol)
        original = {g.split(".")[0] for g in source_gene_sets.get(sid, [])}
        old_symbols = set(previous.get(name, {}).get("members", []))
        check = ("missing_original_gene_set" if sid not in source_gene_sets else
                 "exact_ensembl_match" if original == mapped and not unmapped and not ambiguous else
                 "difference_or_mapping_gap")
        rows.append({
            "source_target_id": sid, "source_name": name,
            "archive_version": archive["archive_version"],
            "original_ensembl_count": len(original),
            "archive_symbol_count": len(symbols),
            "archive_mapped_ensembl_count": len(mapped),
            "shared_ensembl_count": len(original & mapped),
            "original_only_ensembl": "|".join(sorted(original - mapped)),
            "archive_only_ensembl": "|".join(sorted(mapped - original)),
            "unmapped_archive_symbols": "|".join(unmapped),
            "ambiguous_archive_symbols": "|".join(ambiguous),
            "member_comparison_status": check,
            "existing_metadata_symbol_comparison": (
                "same" if old_symbols == symbols else "different") if old_symbols else "unavailable",
            "original_members_preserved": "True",
        })
    return rows
