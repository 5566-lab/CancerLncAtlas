"""Preserve source-specific pathway composition for later graph construction."""
from __future__ import annotations
import collections
import csv
import gzip
import json
from pathlib import Path
import shutil
from .relation_evidence_bindings import file_sha, read_table

SOURCE_FILES = (
    'DEFINITION_TO_IDENTITY.tsv', 'SOURCE_ID_BINDINGS.tsv', 'SOURCE_ALIASES.tsv',
    'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz', 'PATHWAY_GENE_COMPOSITION.tsv.gz',
    'PATHWAY_REACTION_COMPOSITION.tsv.gz', 'STATE_MOLECULAR_COMPOSITION.tsv.gz',
    'PHOSPHO_READOUT_DEFINITIONS.tsv', 'PHOSPHO_READOUT_COMPOSITION.tsv.gz',
    'PTM_READOUT_BINDINGS.tsv', 'KINASE_MOLECULAR_COMPONENTS.tsv',
    'KINASE_CONSTRUCT_RELATIONS.tsv', 'CURATED_EXTERNAL_RELATIONS.tsv',
    'BACKBONE_ALIGNMENT.tsv', 'MECHANISM_CONTEXT_LINKS.tsv', 'TARGET_POLICY.tsv',
)


def materialize_graph_sources(catalog, backbone, output, *, readout_binding_dirs=()):
    from .local_execution import require_host_149
    require_host_149()
    catalog, backbone, output = Path(catalog), Path(backbone), Path(output)
    if output.exists():
        raise ValueError('GRAPH_SOURCE_OUTPUT_EXISTS')
    output.mkdir(parents=True)
    (output/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','training':False,
        'operation':'preserve original database memberships and typed alignment for graph construction'},indent=2))
    files={}
    for name in SOURCE_FILES:
        original=catalog/name;destination=output/name
        shutil.copy2(original,destination)
        digest=file_sha(original)
        if file_sha(destination)!=digest:
            raise ValueError('GRAPH_SOURCE_COPY_MISMATCH: '+name)
        files[name]={'source':str(original),'sha256':digest,'bytes':destination.stat().st_size}
    for name in ['KEGG_CANCER_BACKBONE.tsv','MEMBERS.tsv','GENE_SETS.json','MANIFEST.json']:
        original=backbone/name
        if original.exists():
            shutil.copy2(original,output/name)
            digest=file_sha(original)
            if file_sha(output/name)!=digest:raise ValueError('GRAPH_SOURCE_BACKBONE_COPY_MISMATCH')
            files[name]={'source':str(original),'sha256':digest,'bytes':original.stat().st_size}
    binding_references=[]
    for directory in map(Path,readout_binding_dirs):
        receipt=json.loads((directory/'COMPLETE.json').read_text())
        if receipt.get('format')!='CLA_MULTIMODAL_RELATION_BINDINGS_V1' or receipt.get('target_host')!='149':
            raise ValueError('GRAPH_SOURCE_READOUT_BINDING_SCHEMA')
        name='READOUT_BINDINGS.tsv';digest=file_sha(directory/name)
        if digest!=receipt['files'][name]:raise ValueError('GRAPH_SOURCE_READOUT_BINDING_DRIFT')
        relative='readout_bindings/'+digest+'.tsv'
        destination=output/relative;destination.parent.mkdir(exist_ok=True)
        if not destination.exists():shutil.copy2(directory/name,destination)
        if file_sha(destination)!=digest:raise ValueError('GRAPH_SOURCE_READOUT_COPY_MISMATCH')
        files[relative]={'source':str(directory/name),'sha256':digest,'bytes':destination.stat().st_size}
        binding_references.append({'directory':str(directory),'receipt_sha256':file_sha(directory/'COMPLETE.json'),
            'preserved_readout_binding_file':relative})
    definitions=read_table(catalog/'DEFINITION_TO_IDENTITY.tsv')
    node_map={r['node_id']:r for r in definitions}
    memberships=collections.Counter();databases=collections.Counter();rows=0
    with gzip.open(output/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz','rt',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            node=row['node_id']
            if node not in node_map:raise ValueError('GRAPH_SOURCE_UNBOUND_DEFINITION: '+node)
            memberships[node]+=1;databases[node_map[node]['source_database']]+=1;rows+=1
    # Keep the unmerged source definition as the membership owner. Identity and
    # backbone alignment are separate edges, never an intersection replacement.
    summary={'format':'CLA_SOURCE_PATHWAY_GRAPH_FOUNDATION_V1','target_host':'149',
        'source_definition_count':len(definitions),'definitions_with_protein_rows':len(memberships),
        'protein_membership_rows':rows,'membership_rows_by_database':dict(databases),
        'source_memberships_preserved_byte_for_byte':True,
        'protein_lists_intersected_with_backbone':False,'canonical_members_silently_expanded':False,
        'edge_semantics':{'source_definition_to_protein':'original source membership',
            'source_definition_to_canonical_identity':'explicit identity mapping',
            'source_identity_to_KEGG_mechanism':'typed alignment with evidence and admission fields',
            'regulator_program_to_protein':'explicit TF or kinase molecular identity',
            'phospho_readout_to_site':'original residue/site and direction metadata'},
        'classification_adds_loss':False,'readout_binding_contexts':binding_references,'files':files}
    (output/'COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
