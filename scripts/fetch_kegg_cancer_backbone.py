#!/usr/bin/env python3
"""Freeze official human KEGG Cancer overview/specific-type definitions on host 149.

This is a pathway-definition snapshot, independent of patient data and expression.
The official BRITE category membership determines scope; all selected pathways must
have human definitions and at least one verified Entrez member. Existing complete
snapshots are immutable and verified before reuse.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import socket
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES = ('Cancer: overview', 'Cancer: specific types')
SOURCES = {
    'br08901.json': 'https://rest.kegg.jp/get/br:br08901/json',
    'human_pathways.tsv': 'https://rest.kegg.jp/list/pathway/hsa',
    'human_pathway_gene_links.tsv': 'https://rest.kegg.jp/link/hsa/pathway',
    'human_genes.tsv': 'https://rest.kegg.jp/list/hsa',
    'human_entrez_conversion.tsv': 'https://rest.kegg.jp/conv/ncbi-geneid/hsa',
    'human_uniprot_conversion.tsv': 'https://rest.kegg.jp/conv/uniprot/hsa',
    'pathway_release.txt': 'https://rest.kegg.jp/info/pathway',
    'human_release.txt': 'https://rest.kegg.jp/info/hsa',
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cancer_paths(tree):
    """Return explicit BRITE category leaves, rejecting missing/duplicate scopes."""
    found = {name: [] for name in CATEGORIES}
    def walk(node):
        name = re.sub(r'^\d+\s+', '', node.get('name', '')).strip()
        if name in found:
            found[name].append(node)
        for child in node.get('children', []):
            walk(child)
    walk(tree)
    rows = []
    seen = set()
    for category in CATEGORIES:
        if len(found[category]) != 1:
            raise ValueError(f'Expected one official category {category!r}; found {len(found[category])}')
        for node in found[category][0].get('children', []):
            match = re.match(r'^(\d{5})\s+(.+?)(?:\s+\[PATH:.*)?$', node.get('name', ''))
            if not match or node.get('children'):
                raise ValueError(f'Unexpected pathway category leaf: {node!r}')
            pathway_id = 'hsa' + match.group(1)
            if pathway_id in seen:
                raise ValueError(f'Duplicate cancer pathway {pathway_id}')
            seen.add(pathway_id)
            rows.append({'kegg_id': pathway_id, 'name': match.group(2).strip(), 'category': category})
        if not found[category][0].get('children'):
            raise ValueError(f'Empty official category {category}')
    return rows


def tab_pairs(text):
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 2:
            raise ValueError(f'Expected two tab fields at line {lineno}')
        yield fields


def build_membership(backbone, human_pathways, links, entrez_text, human_genes):
    names = {key.removeprefix('path:'): value for key, value in tab_pairs(human_pathways)}
    selected = {row['kegg_id'] for row in backbone}
    missing = selected - names.keys()
    if missing:
        raise ValueError(f'Cancer pathways absent from official human pathway list: {sorted(missing)}')
    entrez = {}
    for gene, external in tab_pairs(entrez_text):
        if not gene.startswith('hsa:') or not external.startswith('ncbi-geneid:'):
            raise ValueError(f'Unexpected Entrez conversion {gene}, {external}')
        mapped = external.removeprefix('ncbi-geneid:')
        if gene in entrez and entrez[gene] != mapped:
            raise ValueError(f'Ambiguous Entrez gene mapping: {gene}')
        entrez[gene] = mapped
    annotations = {}
    for lineno, line in enumerate(human_genes.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) not in (2, 4):
            raise ValueError(f'Unexpected KEGG human gene list format at line {lineno}')
        gene, description = parts[0], parts[-1]
        gene_type = parts[1] if len(parts) == 4 else ''
        symbols = [s.strip() for s in description.split(';', 1)[0].split(',')] if ';' in description else ['']
        annotations[gene] = (symbols[0], '|'.join(symbols), gene_type)
    membership = set()
    for pathway, gene in tab_pairs(links):
        pathway = pathway.removeprefix('path:')
        if pathway in selected:
            if gene not in entrez:
                raise ValueError(f'Missing verified Entrez ID: {pathway}, {gene}')
            if gene not in annotations:
                raise ValueError(f'Missing human gene annotation: {gene}')
            membership.add((pathway, gene, entrez[gene], *annotations[gene]))
    covered = {row[0] for row in membership}
    if selected != covered:
        raise ValueError(f'Cancer pathways without gene definitions: {sorted(selected-covered)}')
    fields = ('kegg_id', 'hsa_gene_id', 'entrez_gene_id', 'gene_symbol', 'gene_aliases', 'gene_biotype')
    return [dict(zip(fields, row)) for row in sorted(membership)], names


def write_tsv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def verify_existing(root, manifest):
    for item in manifest['sources'] + manifest['outputs']:
        path = root / item['path']
        if not path.is_file() or digest(path) != item['sha256']:
            raise ValueError(f'Frozen snapshot failed checksum: {path}')


def freeze(output_root, proxy=None, timeout=40):
    hostname = socket.gethostname()
    if hostname != '149':
        raise RuntimeError(f'CancerLncAtlas preparation must run on 149, got {hostname}')
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = root / 'sources'
    result = root / 'backbone_snapshot'
    sources.mkdir(exist_ok=True)
    result.mkdir(exist_ok=True)
    manifest_path = result / 'KEGG_SNAPSHOT_MANIFEST.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        verify_existing(root, manifest)
        print(json.dumps({'status': 'verified_existing_snapshot', **manifest['counts']}, ensure_ascii=False))
        return manifest
    write_json(result / 'LAUNCH_PREFLIGHT.json', {
        'target_host': '149', 'verified_hostname': hostname,
        'timestamp_utc': utc_now(), 'scope': list(CATEGORIES),
        'policy': 'all official human pathways in both categories; missing definitions fail',
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': proxy, 'http': proxy})) if proxy else urllib.request.build_opener()
    source_records = []
    last_request = 0.0
    for filename, url in SOURCES.items():
        payload = None
        for attempt in range(3):
            time.sleep(max(0.0, .45-(time.monotonic()-last_request)))
            last_request = time.monotonic()
            try:
                request = urllib.request.Request(url, headers={'User-Agent':'CancerLncAtlas-KEGG-Snapshot/1.0'})
                with opener.open(request, timeout=timeout) as response:
                    payload = response.read()
                    http_status = response.status
                    content_type = response.headers.get('Content-Type', '')
                if not payload:
                    raise ValueError(f'Empty official source {url}')
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        path = sources / filename
        path.write_bytes(payload)
        source_records.append({'path': str(path.relative_to(root)), 'url': url,
                               'fetched_at_utc': utc_now(), 'sha256': digest(path),
                               'bytes': len(payload), 'http_status': http_status,
                               'content_type': content_type})
        print(f'Fetched {filename}: {len(payload)} bytes', flush=True)
    read = lambda filename: (sources / filename).read_text(encoding='utf-8')
    backbone = select_cancer_paths(json.loads(read('br08901.json')))
    members, names = build_membership(backbone, read('human_pathways.tsv'),
                                     read('human_pathway_gene_links.tsv'),
                                     read('human_entrez_conversion.tsv'), read('human_genes.tsv'))
    counts_per_path = {row['kegg_id']: 0 for row in backbone}
    for row in members:
        counts_per_path[row['kegg_id']] += 1
    for row in backbone:
        row['human_name'] = names[row['kegg_id']]
        row['gene_count'] = counts_per_path[row['kegg_id']]
        row['definition_url'] = 'https://www.kegg.jp/pathway/' + row['kegg_id']
    unique_genes = {row['hsa_gene_id'] for row in members}
    uniprot_rows = []
    for gene, external in tab_pairs(read('human_uniprot_conversion.tsv')):
        if gene in unique_genes:
            if not external.startswith('up:'):
                raise ValueError(f'Unexpected UniProt mapping {external}')
            uniprot_rows.append({'hsa_gene_id': gene, 'entrez_gene_id': gene.removeprefix('hsa:'),
                                 'uniprot_id': external.removeprefix('up:')})
    write_tsv(result / 'KEGG_BACKBONE.tsv', backbone,
              ['kegg_id','name','category','human_name','gene_count','definition_url'])
    write_tsv(result / 'KEGG_GENE_MEMBERSHIP.tsv', members,
              ['kegg_id','hsa_gene_id','entrez_gene_id','gene_symbol','gene_aliases','gene_biotype'])
    uniprot_rows = sorted({tuple(row.items()) for row in uniprot_rows})
    write_tsv(result / 'KEGG_BACKBONE_UNIPROT.tsv', [dict(row) for row in uniprot_rows],
              ['hsa_gene_id','entrez_gene_id','uniprot_id'])
    counts = {'pathways': len(backbone), 'category_counts': {c: sum(r['category']==c for r in backbone) for c in CATEGORIES},
              'pathway_gene_memberships': len(members), 'unique_entrez_genes': len(unique_genes),
              'uniprot_mappings': len(uniprot_rows),
              'unique_genes_by_biotype': {kind: len({r['hsa_gene_id'] for r in members if r['gene_biotype']==kind}) for kind in sorted({r['gene_biotype'] for r in members})}}
    manifest = {'schema_version': 1, 'target_host': hostname, 'frozen_at_utc': utc_now(),
                'categories': list(CATEGORIES), 'counts': counts, 'sources': source_records,
                'scope_rule': 'all human pathways in the two official cancer categories',
                'gene_identity': 'KEGG hsa gene ID verified through official ncbi-geneid conversion',
                'outputs': []}
    for path in sorted(result.glob('*.tsv')):
        manifest['outputs'].append({'path': str(path.relative_to(root)), 'sha256': digest(path), 'bytes': path.stat().st_size})
    write_json(manifest_path, manifest)
    print(json.dumps({'status':'frozen', **counts}, ensure_ascii=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--proxy', default=None, help='Optional transient HTTP CONNECT proxy; never stored in manifest')
    parser.add_argument('--timeout', type=float, default=40)
    args = parser.parse_args()
    freeze(args.output_root, args.proxy, args.timeout)


if __name__ == '__main__':
    main()
