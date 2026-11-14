"""Fixed exact-pathway supervision and explicit display/training roles."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv, gzip, hashlib, json, shutil, socket
from .kegg381_backbone import load_backbone
from .target_policy import policy_from_rows, POLICY_SCHEMA, REQUIRED_FIELDS

KINASE_TYPES = frozenset({'kinase_substrate_program','kinase_complex_program',
                         'isoform_kinase_program','source_kinase_program','fusion_kinase_program'})
MECHANISM_FAMILIES = frozenset({'transcription_factor','pathway_footprint'})

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle,delimiter='\t'))

def write_tsv(path, rows, fields=None):
    if fields is None:
        fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,delimiter='\t',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def compile_policy(master, state_definitions, backbone, protein_sets, *, excluded_kegg=(), aligned_contexts=None):
    """Scope is defined without patient expression or lncRNA association labels."""
    identities={r['canonical_id']:r for r in master}
    if len(identities)!=len(master):
        raise ValueError('RELATION_CATALOG_DUPLICATE_IDENTITY')
    anchors={r['canonical_id']:r for r in backbone}
    if len(anchors)!=len(backbone) or set(anchors)-set(identities):
        raise ValueError('RELATION_CATALOG_BACKBONE_IDENTITY_MISMATCH')
    families=defaultdict(set)
    for row in state_definitions:
        families[row['canonical_id']].add(row['state_family'])
    backbone_proteins=set().union(*(protein_sets.get(cid,set()) for cid in anchors))
    if not backbone_proteins:
        raise ValueError('RELATION_CATALOG_EMPTY_BACKBONE_PROTEINS')
    policy=[];updated=[]
    for raw in master:
        row=dict(raw);cid=row['canonical_id'];entity=row['entity_type']
        fs=families[cid]
        tf='transcription_factor' in fs
        footprint='pathway_footprint' in fs
        kinase=entity in KINASE_TYPES
        special=tf or footprint or kinase
        genuine_state=bool(fs-MECHANISM_FAMILIES) or (
            'State' in row['roles'].split('|') and not special)
        kind=('transcription_factor_program' if tf else
              'pathway_footprint' if footprint and not genuine_state else
              'shared_pathway_state' if footprint else entity)
        if special:
            row['roles']='Pathway|State' if genuine_state else 'Pathway'
            row['entity_type']=kind
        shared=protein_sets.get(cid,set()) & backbone_proteins
        anchor=anchors.get(cid)
        if anchor:
            role,group,weight,active='primary_pathway','pathway',1.0,True
            scope=anchor.get('backbone_inclusion_reason','user_fixed_historical_381')
            row['display_name']=anchor.get('network_name') or row['display_name']
            row['roles']='Pathway|State' if genuine_state else 'Pathway'
            anchor_ids=anchor['backbone_id']
        elif genuine_state:
            role,group,weight,active='state_auxiliary','state',.5,True
            scope='retained_state_task'
            anchor_ids=''
        elif cid in set(excluded_kegg):
            role,group,weight,active='excluded','none',0.,False
            scope='outside_evidence_defined_cancer_scope'
            anchor_ids=''
        elif (bool(shared) if aligned_contexts is None else cid in aligned_contexts):
            role,group,weight,active='context_only','none',0.,True
            scope='backbone_component_context' if aligned_contexts is None else 'supported_backbone_component_context'
            anchor_ids=''
        else:
            role,group,weight,active='excluded','none',0.,False
            scope='excluded_zero_overlap' if not shared else 'excluded_insufficient_backbone_evidence'
            anchor_ids=''
        policy.append(dict(canonical_id=cid,roles=row['roles'],biological_type=kind,
            analysis_role=role,loss_group=group,loss_weight=weight,active_in_analysis=active,
            cancer_scope_status=scope,backbone_anchor_ids=anchor_ids,
            backbone_shared_protein_count=len(shared),
            mechanism_kind=('TF' if tf else 'kinase' if kinase else 'PROGENy' if footprint else ''),
            official_cancer_supported=anchor.get('official_cancer_supported','') if anchor else ''))
        updated.append(row)
    policy_from_rows(policy)
    if {r['canonical_id'] for r in policy if r['analysis_role']=='primary_pathway'}!=set(anchors):
        raise ValueError('RELATION_CATALOG_PRIMARY_AXIS_DRIFT')
    return updated,policy


def source_display_category(source, state_definitions, policy):
    definition = state_definitions.get(source['source_target_id'])
    if definition is not None:
        return 'Pathway' if definition['state_family'] in MECHANISM_FAMILIES else 'State'
    if (source['source_target_id'].startswith('MSIGDB:')
            or source.get('source_database') in {'KEGG_MEDICUS','REACTOME','HALLMARK','WikiPathways'}
            or policy.get('mechanism_kind') in {'TF','kinase','PROGENy'}):
        return 'Pathway'
    if policy['roles'] in {'Pathway','State'}:
        return policy['roles']
    raise ValueError('RELATION_CATALOG_SOURCE_ROLE_AMBIGUOUS: '+source['source_target_id'])

def materialize_catalog(source_catalog, backbone_path, output, protein_annotation=None):
    if socket.gethostname()!='149':
        raise RuntimeError('CancerLncAtlas preparation must run on 149')
    src,out=Path(source_catalog).resolve(),Path(output).resolve()
    if out==src or out.exists():
        raise ValueError('RELATION_CATALOG_USE_NEW_OUTPUT_DIRECTORY')
    backbone=load_backbone(backbone_path,verify=True)
    master=read_tsv(src/'TARGET_MASTER.tsv')
    states=read_tsv(src/'STATE_SCORE_DEFINITIONS.tsv')
    proteins=defaultdict(set);protein_genes=defaultdict(set);source_proteins=defaultdict(set)
    with gzip.open(src/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz','rt') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            if row['uniprot_id']:
                proteins[row['integrated_canonical_id']].add(row['uniprot_id'])
                source_proteins[row['source_definition']].add(row['uniprot_id'])
                if row.get('source_namespace')=='Ensembl_gene' and row.get('source_member','').startswith('ENSG'):
                    protein_genes[row['uniprot_id']].add(row['source_member'].split('.')[0])

    regulator_proteins=defaultdict(set)
    for row in read_tsv(src/'KINASE_MOLECULAR_COMPONENTS.tsv'):
        regulator_proteins[row['canonical_id']].update(x for x in row['uniprot_accessions'].split('|') if x)
    annotation_path=Path(protein_annotation) if protein_annotation else src/'PROTEIN_ANNOTATION.tsv'
    if backbone.get('decisions') and not annotation_path.is_file():
        raise ValueError('RELATION_CATALOG_REGULATOR_ANNOTATION_REQUIRED')
    if annotation_path.is_file():
        by_symbol=defaultdict(set)
        for row in read_tsv(annotation_path):
            by_symbol[row['symbol']].add(row['uniprot_id'])
            if row['ensembl_gene_id']:protein_genes[row['uniprot_id']].add(row['ensembl_gene_id'])
        for row in read_tsv(src/'SOURCE_ID_BINDINGS.tsv'):
            if row['source_target_id'].startswith('DOROTHEA:'):
                regulator_proteins[row['canonical_id']].update(by_symbol.get(row['source_target_id'].split(':',1)[1],set()))
    decisions=backbone.get('decisions',[])
    excluded_kegg={x['canonical_id'] for x in decisions if x['cancer_relevant']!='True'}
    alignment=top_alignment=alignment_summary=None; aligned_contexts=None
    if decisions:
        from .backbone_alignment import build_alignment,pin_anchor_components
        proteins=pin_anchor_components(proteins,backbone['rows'],source_proteins)
        alignment,top_alignment,aligned_contexts,alignment_summary=build_alignment(master,backbone['rows'],proteins,excluded_kegg,regulator_proteins)
    updated,policy=compile_policy(master,states,backbone['rows'],proteins,excluded_kegg=excluded_kegg,aligned_contexts=aligned_contexts)
    shutil.copytree(src,out)
    if annotation_path.is_file():shutil.copy2(annotation_path,out/'PROTEIN_ANNOTATION.tsv')
    (out/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'evidence_defined_relation_catalog'},indent=2))
    write_tsv(out/'TARGET_MASTER.tsv',updated)
    if (out/'IDENTITY_MASTER.tsv').exists():
        old=read_tsv(out/'IDENTITY_MASTER.tsv')
        lookup={r['canonical_id']:r for r in updated}
        for row in old:
            if row['canonical_id'] in lookup:
                for key in ('roles','entity_type','display_name'):
                    row[key]=lookup[row['canonical_id']][key]
        write_tsv(out/'IDENTITY_MASTER.tsv',old)
    write_tsv(out/'TARGET_POLICY.tsv',policy)
    if alignment is not None:
        write_tsv(out/'BACKBONE_ALIGNMENT.tsv',alignment)
        write_tsv(out/'BACKBONE_ALIGNMENT_TOP3.tsv',top_alignment)
        (out/'ALIGNMENT_SUMMARY.json').write_text(json.dumps(alignment_summary,indent=2)+'\n')
        write_tsv(out/'KEGG_SOURCE_SCOPE_DECISIONS.tsv',decisions)
        from .mechanism_context import catalog_context
        context_links,context_components=catalog_context(policy,alignment,proteins,protein_genes,
            {r['canonical_id']:r for r in backbone['rows']},regulator_proteins)
        write_tsv(out/'MECHANISM_CONTEXT_LINKS.tsv',context_links,
            ['mechanism_id','pathway_source_id','network_id','weight','relation'])
        write_tsv(out/'MECHANISM_CONTEXT_COMPONENTS.tsv',context_components,
            ['mechanism_id','uniprot_id','gene_id'])

    write_tsv(out/'PRIMARY_PATHWAY_TARGETS.tsv',[r for r in updated if r['canonical_id'] in {x['canonical_id'] for x in backbone['rows']}])
    write_tsv(out/'MECHANISM_TARGETS.tsv',[r for r in policy if r['mechanism_kind']])
    bindings=read_tsv(src/'SOURCE_ID_BINDINGS.tsv')
    write_tsv(out/'SOURCE_IDENTITY_MAP.tsv',bindings)
    byid={r['canonical_id']:r for r in policy}
    source_roles=[]
    state_by_id={r['state_id']:r for r in states}
    for row in bindings:
        pol=byid.get(row['canonical_id'])
        if pol is None:
            raise ValueError('RELATION_CATALOG_UNRESOLVED_BINDING: '+row['source_target_id'])
        category=source_display_category(row,state_by_id,pol)
        source_roles.append({**row,'display_category':category,'analysis_role':pol['analysis_role'],'loss_group':pol['loss_group']})
    write_tsv(out/'SOURCE_TASK_ROLES.tsv',source_roles)
    definition={'format':('CLA_CANCER_RELATION_CATALOG_V2' if decisions else 'CLA_FIXED381_RELATION_CATALOG_V1'),'unit':'cancer x lncRNA x exact pathway',
      'primary_pathway_count':len(backbone['rows']),'target_policy_schema':POLICY_SCHEMA,
      'target_policy_file':'TARGET_POLICY.tsv','target_policy_sha256':sha(out/'TARGET_POLICY.tsv'),
      'target_policy_required':True,'backbone_path':str(Path(backbone_path).resolve()),
      'backbone_manifest_sha256':sha(Path(backbone_path)/'MANIFEST.json'),
      'source_catalog':str(src),'state_loss_weight':.5,
      'regulator_independent_loss_enabled':False,
      'display_category_determines_loss':False}
    (out/'CATALOG_DEFINITION.json').write_text(json.dumps(definition,indent=2)+'\n')
    binding={'path':str(out/'TARGET_POLICY.tsv'),'sha256':sha(out/'TARGET_POLICY.tsv'),
      'backbone_path':str(Path(backbone_path).resolve()),
      'backbone_manifest_sha256':sha(Path(backbone_path)/'MANIFEST.json'),
      'source_identity_map':str(out/'SOURCE_IDENTITY_MAP.tsv'),
      'source_identity_map_sha256':sha(out/'SOURCE_IDENTITY_MAP.tsv')}
    if decisions:
        binding['context_graph']={name:{'path':str(out/name),'sha256':sha(out/name)}
            for name in ['MECHANISM_CONTEXT_LINKS.tsv','MECHANISM_CONTEXT_COMPONENTS.tsv']}
    (out/'PREPARE_POLICY_BINDING.json').write_text(json.dumps({'relation_target_policy':binding},indent=2)+'\n')
    summary={'target_host':'149','format':definition['format'],'primary_pathways':len(backbone['rows']),
      'analysis_roles':dict(Counter(r['analysis_role'] for r in policy)),
      'display_roles':dict(Counter(r['roles'] for r in policy)),
      'mechanism_kinds':dict(Counter(r['mechanism_kind'] for r in policy if r['mechanism_kind'])),
      'backbone_official_cancer_supported':sum(str(r.get('official_cancer_supported','')).lower() in ('true','1') for r in backbone['rows']),
      'state_loss_weight':.5,'regulator_independent_loss_enabled':False,
      'label_values_changed':False,'source_catalog_complete_sha256':sha(src/'COMPLETE.json')}
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='COMPLETE.json'}
    summary['files']=files
    (out/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def demo_configurations(base_config, catalog, cancers=('BRCA','LUAD','COAD')):
    """One independently fitted cancer per capsule; preserve the existing Y contract."""
    from copy import deepcopy
    root=Path(catalog).resolve()
    binding=json.loads((root/'PREPARE_POLICY_BINDING.json').read_text())['relation_target_policy']
    result={}
    for cancer in cancers:
        config=deepcopy(base_config)
        config['relation_target_policy']=dict(binding)
        config['relation_training_scope']='within_cancer'
        config['cancer_type']='list';config['cancer_list']=[cancer]
        config['baseline']['identity_mode']='numeric_only'
        for profile in config['profiles'].values():
            profile['max_pathways']=None
        result[cancer]=config
    return result
