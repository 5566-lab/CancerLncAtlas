"""Evidence-defined KEGG cancer mechanisms, independent of assay coverage."""
from __future__ import annotations
import csv, hashlib, json, re, socket
from collections import Counter
from pathlib import Path
from .kegg381_backbone import build_backbone, members_sha256, _unique_index

FORMAT='CLA_KEGG_CANCER_BACKBONE_V2'
POLICY_VERSION='cancer_mechanisms_20261111_v2'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter='	'))

def write_tsv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='	',extrasaction='ignore');w.writeheader();w.writerows(rows)

def validate_evidence(curation,references,verify_cached_sources=False):
    """Bind every literature-based admission to a retrieved named publication."""
    required={str(p) for f in curation['families'] for p in f['pmids']}
    for pmid in sorted(required):
        item=references.get(pmid)
        if not item or str(item.get('pmid'))!=pmid or not item.get('title') or not item.get('abstract'):
            raise ValueError('CANCER_SCOPE_REFERENCE_MISSING:'+pmid)
        if re.search(r'\bretract(?:ed|ion)\b',item['title'],re.I):
            raise ValueError('CANCER_SCOPE_RETRACTED_REFERENCE:'+pmid)
        if item.get('source_url')!='https://pubmed.ncbi.nlm.nih.gov/'+pmid+'/':
            raise ValueError('CANCER_SCOPE_REFERENCE_URL:'+pmid)
        if not re.fullmatch(r'[0-9a-f]{64}',item.get('cached_source_sha256','')):
            raise ValueError('CANCER_SCOPE_REFERENCE_DIGEST:'+pmid)
        if verify_cached_sources and sha(item['cached_source'])!=item['cached_source_sha256']:
            raise ValueError('CANCER_SCOPE_REFERENCE_SOURCE_CHANGED:'+pmid)
    return required

def classify_sources(sources,raw_records,curation):
    """Every source receives a decision; gene overlap/statistics never admit it."""
    _unique_index(sources,'source_target_id');_unique_index(sources,'network_id')
    families=curation['families'];by_network={}
    for family in families:
        if not family.get('pmids') or not family.get('rationale'):
            raise ValueError('CANCER_SCOPE_EVIDENCE_REQUIRED:'+family['id'])
        for nid in family['network_ids']:
            if nid in by_network:raise ValueError('CANCER_SCOPE_DUPLICATE_CURATION:'+nid)
            by_network[nid]=family
    universe={x['network_id'] for x in sources}
    if set(by_network)-universe:raise ValueError('CANCER_SCOPE_UNKNOWN_CURATED_ID')
    malignant=set(curation['malignant_disease_ids'])
    decisions=[]
    for row in sources:
        nid=row['network_id'];record=raw_records.get(nid,{});fields=record.get('fields',{})
        kind=row.get('network_type','')
        diseases=fields.get('DISEASE',[])
        disease_ids={x.split()[0] for x in diseases if x.split()}
        matched=sorted(disease_ids & malignant)
        family=by_network.get(nid)
        official=row.get('cancer_scope_status')=='direct_cancer_network'
        if official:
            admitted=True;basis='official_cancer_class';biofamily='KEGG cancer mechanisms'
            rationale='Official NETWORK Cancer / Cancer signaling membership.'
            evidence='KEGG:'+row.get('cancer_map_ids','');level='official_classification'
        elif matched:
            admitted=True;basis='official_malignant_disease';biofamily='Cancer-associated infection or alteration'
            rationale='Exact NETWORK record has a curated malignant DISEASE association.'
            evidence='KEGG:'+ '|'.join(matched);level='official_disease_annotation'
        elif family:
            if not record:raise ValueError('CANCER_SCOPE_CURATED_RECORD_MISSING:'+nid)
            if family['allowed_type']!=kind:raise ValueError('CANCER_SCOPE_CONTEXT_TYPE_MISMATCH:'+nid)
            admitted=True;basis='curated_mechanism_evidence';biofamily=family['name']
            rationale=family['rationale'];evidence='|'.join('PMID:'+p for p in family['pmids']);level=family['evidence_level']
        else:
            admitted=False;biofamily='';evidence='KEGG:'+nid
            if not record:
                basis='archival_definition_without_current_record'
                rationale='MSigDB historical identity retained; no current authoritative NETWORK mechanism record.'
            elif kind in {'Variant','Pathogen','Env factor'}:
                basis='specific_context_not_supported_for_cancer'
                rationale='The exact alteration, pathogen or exposure mechanism is outside the supported cancer evidence set; general host-gene relevance is insufficient.'
            else:
                basis='reference_process_not_in_supported_scope'
                rationale='The exact physiological or tissue-specific mechanism is not covered by the reviewed cancer-process evidence; retained in the source inventory.'
            level='not_admitted_under_this_definition'
        applicability={'Reference':'reference_biological_process','Variant':'matching_molecular_alteration','Pathogen':'matching_pathogen_or_viral_protein','Env factor':'matching_exposure'}.get(kind,'historical_definition')
        if family and family.get('applicability'):applicability=family['applicability']
        decisions.append({**row,'cancer_relevant':str(admitted),'cancer_evidence_basis':basis,
          'biological_family':biofamily,'cancer_rationale':rationale,'cancer_evidence':evidence,
          'evidence_level':level,'disease_annotations':'|'.join(diseases),
          'applicability':applicability,'generic_host_gene_score_proves_context':'False',
          'curation_family_id':family['id'] if family else '',
          'scope_definition_version':curation['version']})
    return decisions

def materialize_scope(source_table,raw_records_file,curation_file,mapping_file,catalog,gene_catalog,output):
    if socket.gethostname()!='149':raise RuntimeError('CancerLncAtlas preparation must run on 149')
    out=Path(output)
    if out.exists():raise ValueError('CANCER_SCOPE_USE_NEW_OUTPUT')
    sources=read_tsv(source_table);raw=json.loads(Path(raw_records_file).read_text());curation=json.loads(Path(curation_file).read_text())
    registry_path=Path(curation_file).parent/curation['evidence_registry']
    references=json.loads(registry_path.read_text())
    validate_evidence(curation,references,verify_cached_sources=True)
    decisions=classify_sources(sources,raw,curation)
    accepted=[x for x in decisions if x['cancer_relevant']=='True']
    if not accepted:raise ValueError('CANCER_SCOPE_EMPTY')
    genes=json.loads(Path(gene_catalog).read_text())['gene_sets']
    rows,members,summary=build_backbone([x['source_target_id'] for x in accepted],read_tsv(mapping_file),sources,genes,read_tsv(Path(catalog)/'TARGET_MASTER.tsv'),expected_count=len(accepted))
    index={x['network_id']:x for x in accepted}
    for row in rows:
        d=index[row['network_id']]
        for key in ['cancer_evidence_basis','biological_family','cancer_rationale','cancer_evidence','evidence_level','applicability','curation_family_id']:
            row[key]=d[key]
        row['backbone_inclusion_reason']=d['cancer_evidence_basis']
        row['display_name']=d['network_name']
    out.mkdir(parents=True)
    (out/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'evidence_defined_cancer_scope'},indent=2)+'\n')
    write_tsv(out/'KEGG_CANCER_BACKBONE.tsv',rows)
    write_tsv(out/'SOURCE_SCOPE_DECISIONS.tsv',decisions)
    (out/'GENE_SETS.json').write_text(json.dumps(members,indent=2)+'\n')
    write_tsv(out/'MEMBERS.tsv',[{'source_target_id':x['source_target_id'],'backbone_id':x['backbone_id'],'network_id':x['network_id'],'canonical_id':x['canonical_id'],'ensembl_gene_id':g} for x in rows for g in members[x['source_target_id']]])
    (out/'SOURCE_IDS.txt').write_text('\n'.join(x['source_target_id'] for x in rows)+'\n')
    (out/'CURATION.json').write_text(json.dumps(curation,indent=2)+'\n')
    (out/'EVIDENCE_REFERENCES.json').write_text(json.dumps(references,indent=2)+'\n')
    summary.update({'format':FORMAT,'target_host':'149','scope_definition_version':curation['version'],
      'source_universe_count':len(sources),'backbone_count':len(rows),'not_admitted_count':len(decisions)-len(rows),
      'selection_rule':'official cancer class OR exact malignant disease annotation OR reviewed mechanism evidence',
      'official_cancer_annotation_is_filter':False,'patient_coverage_used_for_selection':False,
      'lncrna_association_used_for_selection':False,'fixed_target_count_required':False,
      'evidence_basis_counts':dict(Counter(x['cancer_evidence_basis'] for x in accepted)),
      'all_decision_counts':dict(Counter(x['cancer_evidence_basis'] for x in decisions)),
      'inputs':{str(p):sha(p) for p in [source_table,raw_records_file,curation_file,registry_path,mapping_file,gene_catalog]},
      'files':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()},
      'ordered_source_network_canonical_sha256':members_sha256([[x['source_target_id'],x['network_id'],x['canonical_id']] for x in rows])})
    (out/'MANIFEST.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def load_cancer_backbone(path,verify=True):
    root=Path(path);m=json.loads((root/'MANIFEST.json').read_text())
    if m.get('format')!=FORMAT:raise ValueError('CANCER_BACKBONE_FORMAT')
    if verify:
        for name,item in m['files'].items():
            if Path(name).name!=name or sha(root/name)!=item['sha256']:raise ValueError('CANCER_BACKBONE_CHECKSUM:'+name)
    rows=read_tsv(root/'KEGG_CANCER_BACKBONE.tsv');genes=json.loads((root/'GENE_SETS.json').read_text())
    ids=(root/'SOURCE_IDS.txt').read_text().splitlines()
    if len(rows)!=m['backbone_count'] or ids!=[x['source_target_id'] for x in rows]:raise ValueError('CANCER_BACKBONE_AXIS')
    for key in ['source_target_id','network_id','canonical_id','backbone_id']:_unique_index(rows,key)
    if set(ids)!=set(genes):raise ValueError('CANCER_BACKBONE_MEMBERSHIP_AXIS')
    for row in rows:
        values=genes[row['source_target_id']]
        if len(values)!=int(row['original_gene_count']) or members_sha256(values)!=row['original_members_sha256']:raise ValueError('CANCER_BACKBONE_MEMBERSHIP_DIGEST')
    validate_evidence(json.loads((root/'CURATION.json').read_text()),json.loads((root/'EVIDENCE_REFERENCES.json').read_text()))
    decisions=read_tsv(root/'SOURCE_SCOPE_DECISIONS.tsv')
    if {x['source_target_id'] for x in decisions if x['cancer_relevant']=='True'}!=set(ids):raise ValueError('CANCER_BACKBONE_DECISION_DRIFT')
    return {'manifest':m,'rows':rows,'gene_sets':genes,'decisions':decisions}
