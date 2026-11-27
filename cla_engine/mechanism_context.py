"""Static molecular-program context; never creates a supervised target."""
from collections import defaultdict
import numpy as np

def catalog_context(policy, alignment, proteins, gene_map, anchors, regulator_proteins=None):
    regulator_proteins=regulator_proteins or {}
    admitted={r['canonical_id']:r for r in policy if r['analysis_role']=='context_only' and str(r['active_in_analysis']) in {'True','true'}}
    links=[]
    used=set()
    for r in alignment:
        cid=r['source_canonical_id']
        if cid not in admitted or r['admitted_context_link']!='True':continue
        target=anchors[r['target_canonical_id']]
        links.append(dict(mechanism_id=cid,pathway_source_id=target['source_target_id'],
            network_id=target['network_id'],weight=1. if r.get('direct_regulator_proteins') else r['jaccard'],relation=r.get('relation','shared_protein_components')))
        used.add(cid)
    components=[]
    for cid in sorted(used):
        for protein in sorted(proteins.get(cid,set()) | regulator_proteins.get(cid,set())):
            genes=gene_map.get(protein,set())
            # Preserve ambiguous protein identity without forcing a gene mapping.
            gene=next(iter(genes)) if len(genes)==1 else ''
            components.append(dict(mechanism_id=cid,uniprot_id=protein,gene_id=gene))
    return links,components

def add_context_graph(ids,maps,relations,edges,links,components):
    """Join fixed composition through exact protein IDs or unambiguous gene IDs."""
    target=set(ids['pathway'])
    links=[r for r in links if r['pathway_source_id'] in target]
    mechanisms=sorted({r['mechanism_id'] for r in links})
    if not mechanisms:return {'mechanism_nodes':0,'links':0,'component_edges':0}
    if 'mechanism' in ids:raise ValueError('MECHANISM_NODE_TYPE_ALREADY_EXISTS')
    ids['mechanism']=mechanisms;maps['mechanism']={v:i for i,v in enumerate(mechanisms)}
    records=defaultdict(dict);matched=set();unmapped=0
    def add(src,rel,dst,a,b,w):
        records[(src,rel,dst)][(a,b)]=max(w,records[(src,rel,dst)].get((a,b),0))
    for row in components:
        mid=row['mechanism_id']
        if mid not in maps['mechanism']:continue
        if row['uniprot_id'] in maps['protein']:
            kind,value='protein',row['uniprot_id']
        elif row['gene_id'] and row['gene_id'] in maps['gene']:
            kind,value='gene',row['gene_id']
        else:
            unmapped+=1;continue
        add(kind,'mechanism_component','mechanism',maps[kind][value],maps['mechanism'][mid],1.)
        matched.add(mid)
    if not matched:raise ValueError('MECHANISM_COMPONENT_IDENTITY_COVERAGE_ZERO')
    for row in links:
        if row['mechanism_id'] not in matched:continue
        weight=float(row['weight'])
        if not np.isfinite(weight) or not 0<weight<=1:raise ValueError('MECHANISM_LINK_WEIGHT')
        relation='mechanism_regulator_context' if row.get('relation')=='regulator_component_in_mechanism' else 'mechanism_context'
        add('mechanism',relation,'pathway',maps['mechanism'][row['mechanism_id']],maps['pathway'][row['pathway_source_id']],weight)
    edge_counts={}
    for rel,pairs in records.items():
        order=sorted(pairs);index=np.asarray(order,dtype=np.int64).T
        entry={'edge_index':index,'weight':np.asarray([pairs[k] for k in order],np.float32),
               'context':np.repeat('',len(order)),'sign':1}
        key='|'.join(rel)
        if key in edges:raise ValueError('MECHANISM_RELATION_COLLISION')
        edges[key]=entry;relations.append(rel)
        reverse=(rel[2],'rev_'+rel[1],rel[0])
        edges['|'.join(reverse)]={**entry,'edge_index':index[::-1].copy()};relations.append(reverse)
        edge_counts[key]=len(order)
    return {'mechanism_nodes':len(mechanisms),'mapped_mechanism_nodes':len(matched),
        'unmapped_component_records':unmapped,'edge_counts':edge_counts,
        'independent_loss':False,'label_values_read':False,
        'relation_semantics':'molecular composition and shared components; no activity or causality direction'}
