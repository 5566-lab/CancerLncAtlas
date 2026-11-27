"""Compile the frozen dictionary's documented semantics, not its overlap edges."""
import json
import pandas as pd


def semantic_catalog(catalog, rules):
    def read(name): return pd.read_csv(catalog/name,sep='\t',keep_default_na=False)
    definitions = read('DEFINITION_TO_IDENTITY.tsv')
    aliases = {}
    for r in definitions.to_dict('records'):
        cid = r['integrated_canonical_id'] or r['canonical_id']
        for key in (r['node_id'],r['canonical_id'],r['authority_id']):
            if key: aliases.setdefault(key,set()).add(cid)
    resolve = lambda key: next(iter(aliases[key])) if key in aliases and len(aliases[key])==1 else key
    primary = read('PRIMARY_PATHWAY_TARGETS.tsv')
    targets = set(primary.canonical_id)
    rows = []
    def add(a,b,relation,evidence,basis,reason):
        if a and b and a!=b:
            rows.append(dict(source_scope=a,target_id=b,relation=relation,evidence=evidence,
                             evidence_basis=basis,biological_reason=reason))
    decisions = read('FUNCTIONAL_SCOPE_DECISIONS.tsv')
    for r in decisions.to_dict('records'):
        if r['decision_source'] not in {'ComPath_hierarchy','ComPath_scope_mapping','functional_scope_review','official_ID'}:
            continue
        a,b = resolve(r['node_a']),resolve(r['node_b'])
        if r['relation']=='is_part_of':
            add(a,b,'narrower_scope_context',r['evidence'],'curated_functional_scope',r['biological_reason'])
            add(b,a,'broader_scope_context',r['evidence'],'curated_functional_scope',r['biological_reason'])
        elif r['relation'] in {'same_functional_identity','shares_functional_core'}:
            relation = 'same_scope' if r['relation']=='same_functional_identity' else 'shared_process_context'
            add(a,b,relation,r['evidence'],'curated_functional_scope',r['biological_reason'])
            add(b,a,relation,r['evidence'],'curated_functional_scope',r['biological_reason'])
    networks = read('KEGG_SOURCE_SCOPE_DECISIONS.tsv')
    reviewed = []
    for rule in rules['rules']:
        scope = 'PROGENY:'+rule['score']
        hits = 0
        for r in networks.to_dict('records'):
            cid = resolve(r['canonical_id'])
            if cid not in targets: continue
            maps = set(r['all_map_ids'].split('|')) & set(rule['parent_maps'])
            if r['network_id'] not in rule['network_ids'] and not maps: continue
            evidence = '|'.join([rules['source_reference'],
                'https://www.kegg.jp/entry/'+r['network_id']] +
                ['https://www.kegg.jp/network/'+m for m in sorted(maps)])
            add(scope,cid,'broader_scope_context',evidence,'reviewed_perturbation_scope',rule['reason'])
            reviewed.append(dict(score=rule['score'],target_id=cid,network_id=r['network_id'],
                network_name=r['network_name'],mechanism_definition=r['mechanism_definition'],
                matched_official_parents='|'.join(sorted(maps)),reason=rule['reason'],evidence=evidence))
            hits += 1
    scope_table = pd.DataFrame(rows).drop_duplicates(['source_scope','target_id','relation'])
    source_lookup = {}
    for r in read('SOURCE_ID_BINDINGS.tsv').to_dict('records'):
        for key in [r['source_target_id'],r['source_column']]:
            if key: source_lookup.setdefault(key,set()).add(resolve(r['canonical_id']))
    for r in read('PTM_READOUT_BINDINGS.tsv').to_dict('records'):
        source_lookup.setdefault(r['source_readout_id'],set()).add(resolve(r['canonical_id']))
    return primary, scope_table, pd.DataFrame(reviewed), aliases, source_lookup


def observation_binding(record, aliases, source_lookup):
    """Exact stored IDs only. Unknown dependencies are kept but not input-enabled."""
    candidates = [record.get('definition_id',''),record.get('feature_id',''),record.get('source_target_id','')]
    try: definition = json.loads(record.get('definition_json','{}'))
    except (ValueError,TypeError): definition = {}
    if isinstance(definition,dict):
        candidates += [str(definition.get(k,'')) for k in ('canonical_id','source_target_id','target_id','source_readout_id')]
    resolved = set()
    progeny = None
    for key in candidates:
        key = str(key).split('@@')[0]
        if 'PROGENY:' in key: progeny='PROGENY:'+key.split('PROGENY:',1)[1]
        options = [key, key.split('|',1)[-1]]
        for item in options:
            if item in source_lookup: resolved |= source_lookup[item]
            if item in aliases: resolved |= aliases[item]
    scope = progeny or (next(iter(resolved)) if len(resolved)==1 else '')
    layer = record.get('layer','')
    # Functional category STATE is not a measurement modality.
    if progeny: layer='RNA'
    if layer not in {'RNA','PROTEIN','PHOSPHO','METHYLATION','CNV','MUTATION'}: layer='UNRESOLVED'
    key = record['observation_id']
    return dict(score_id=key,source_scope=scope,input_layers=json.dumps([layer]),
        lineage_ids=json.dumps(sorted({str(x) for x in candidates+[key] if x})),
        binding_evidence='existing frozen source ID binding' if scope else 'unresolved exact source identity',
        source_method=record.get('method',''),source_feature_id=record.get('feature_id',''),
        source_layer=record.get('layer',''),input_modality=layer,
        independent_loss_enabled=False,copy_to_target_y=False)
