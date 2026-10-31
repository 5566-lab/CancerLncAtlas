"""Bounded, provenance-preserving TC-KPNN-style POST-HOC path enumeration.

This is NOT a trained KPNN or an added HHGT loss. A path is a graph-supported
hypothesis, not a causal mediation proof. No unsupported protein->gene
regulatory edge is invented from a protein-coding identity mapping.
"""
from pathlib import Path
from collections import defaultdict
import heapq,json
import numpy as np
import pandas as pd
from .common import fail,sha,read_json,digest
from .tables import table,require
from .identity import frame_ids,gene,unique_values
from .graph_contracts import normalize_binding,normalize_ppi,membership_join
from .interpretation import queries,output_directory,finish


def knowledge_paths(query_path,cfg,output,*,prepared=None):
    limit=cfg.get('interpretation',{});q,qr=queries(query_path,limit.get('max_explanation_candidates',5000));out=output_directory(output)
    count_limit=int(limit.get('max_paths_per_candidate',20));total_limit=int(limit.get('max_total_path_records',100000))
    if not 1<=count_limit<=1000 or total_limit<1:fail('PATH_LIMIT_INVALID','Finite positive limits required')
    sources=cfg['sources'];hashes={}
    def get(role):
        p=Path(sources[role]['path']);h=sha(p)
        if sources[role].get('sha256') and h!=sources[role]['sha256']:fail('KNOWLEDGE_SOURCE_DRIFT',role)
        hashes[str(p)]=h;return table(p)
    binding=normalize_binding(get('binding'));ppi=normalize_ppi(get('ppi'));pg=frame_ids(get('protein_gene'))
    require(pg,['protein_id','gene_id','mapping_weight','mapping_multiplicity','source_database'],'protein-gene')
    mult=pd.to_numeric(pg.mapping_multiplicity,errors='coerce')
    if not np.isfinite(mult).all() or mult.lt(1).any():fail('PROTEIN_GENE_AMBIGUOUS','No invalid mapping branch metadata')
    if not np.isclose(pd.to_numeric(pg.mapping_weight,errors='coerce').to_numpy(float),1/mult.to_numpy(float)).all():
        fail('PROTEIN_GENE_ATTRIBUTION_INCONSISTENT','mapping_weight must equal 1/multiplicity')
    pg=unique_values(pg,['protein_id','gene_id'],role='protein-gene')
    if pg.duplicated('protein_id').any():fail('PROTEIN_GENE_AMBIGUOUS','Protein maps to multiple coding genes')
    membership,_=membership_join(get('exact_membership'),get('weighted_membership'))
    p2g={str(r.protein_id):str(r.gene_id) for r in pg.itertuples()}
    pgrows={str(r['protein_id']):r for r in pg.to_dict('records')}
    members=defaultdict(dict)
    for row in membership.to_dict('records'):members[str(row['pathway_id'])][str(row['gene_id'])]=row
    adj=defaultdict(list)
    for row in ppi.to_dict('records'):
        a,b=str(row['protein_id_a']),str(row['protein_id_b']);adj[a].append((b,row));adj[b].append((a,row))
    for p in adj:adj[p].sort(key=lambda x:x[0])
    binds=defaultdict(list)
    for row in binding.to_dict('records'):binds[str(row['lncrna_id'])].append(row)
    for l in binds:binds[l].sort(key=lambda r:(r['protein_id'],r['cancer_id'],r.get('edge_id','')))
    view_cache={};prepared_info=None
    if prepared:
        from .lineage import validate_prepared
        pm,spec,pcfg,lineage=validate_prepared(prepared)
        bound=read_json(Path(prepared)/'INPUT_HASHES.json')
        for role in ('binding','ppi','protein_gene','exact_membership','weighted_membership'):
            old=pcfg['sources'][role];old_path=str(Path(old['path']))
            old_records=[r for r in bound.values() if str(Path(r['path']))==old_path]
            expected={r['sha256'] for r in old_records}
            if len(expected)!=1 or hashes[str(Path(sources[role]['path']))] not in expected:
                fail('KNOWLEDGE_PREPARED_SOURCE_MISMATCH','Cannot combine source paths with genotype context from another input authority',role=role)
        prepared_info={'manifest_sha256':sha(Path(prepared)/'MANIFEST.json'),'fold':spec['fold'],'context_source':'INNER_TRAIN_PATIENTS_ONLY'}
        for c in q.cancer_id.unique():
            if c not in spec['cancers']:continue
            p=Path(prepared)/'views'/f'{c}__validation'/'graph.npz'
            # The actual capsule graph filename is checked, never guessed silently.
            if not p.exists():fail('KNOWLEDGE_PREPARED_GRAPH_MISSING','Expected graph.npz',path=str(p))
            with np.load(p,allow_pickle=False) as arr:
                g=np.array(arr['x::gene'],copy=True);pr=np.array(arr['x::protein'],copy=True)
            view_cache[c]=(spec,{k:i for i,k in enumerate(spec['nodes']['gene'])},g,{k:i for i,k in enumerate(spec['nodes']['protein'])},pr)
    records=[];summary=[]
    for query in q.to_dict('records'):
        c,l,t,task=query['cancer_id'],query['lncrna_id'],query['target_id'],query['task'];candidates={};seen=0
        if task!='pathway':
            summary.append({**query,'path_count':0,'availability':'STATE_MEMBERSHIP_SIGNATURE_NOT_BOUND_NO_PATHWAY_SUBSTITUTION'});continue
        selected=members.get(t,{})
        # Each path includes a physical contact, then optionally ONE PPI hop,
        # then a coding-identity edge, then target membership. No exhaustive walks.
        for br in binds.get(l,[]):
            if seen>=int(limit.get('max_walks_per_candidate',10000)):break
            if br['cancer_id'] not in ('GLOBAL',c):continue
            p=str(br['protein_id']);options=[(p,None)]
            options.extend(adj.get(p,[]) if int(limit.get('maximum_ppi_hops',1))>=1 else [])
            for endpoint,edge in options:
                if seen>=int(limit.get('max_walks_per_candidate',10000)):break
                seen+=1
                g=p2g.get(endpoint)
                if g not in selected:continue
                ids=[l,p]+([] if edge is None else [endpoint])+[g,t]
                key=tuple(ids)
                if key in candidates:continue
                # Keep lexicographically earliest short paths with a finite cap.
                confidence=[float(br['weight'])]+([] if edge is None else [float(edge['weight'])])
                mr=selected[g];source_records={'binding':br,'ppi':edge,'protein_coding_identity':pgrows[endpoint],'pathway_membership':mr}
                rec={**query,'path_id':digest({'cancer':c,'nodes':ids})[:24],'nodes_json':json.dumps(ids),'edge_semantics':
                    'physical_binding'+(' -> physical_PPI' if edge else '')+' -> encoded_by_identity -> pathway_membership',
                    'ppi_hops':int(edge is not None),'edge_confidences_json':json.dumps(confidence),'minimum_edge_confidence':min(confidence),
                    'minimum_confidence_is_not_path_probability':True,'membership_weight':mr.get('weight'),
                    'path_direction':'UNKNOWN_NOT_INFERRED_FROM_MEMBERSHIP_OR_BINDING','source_records_json':json.dumps(source_records,default=str,sort_keys=True),
                    'causal_mediation_proven':False,'pathway_gene_mutation_features_json':None,'partner_protein_mutation_features_json':None}
                if c in view_cache:
                    spec,gi,gv,pri,prv=view_cache[c]
                    from .genomics import FEATURES
                    if spec.get('genomics_enabled') and g in gi:rec['pathway_gene_mutation_features_json']=json.dumps(dict(zip(FEATURES,map(float,gv[gi[g],6:6+len(FEATURES)]))))
                    if spec.get('genomics_enabled') and p in pri:rec['partner_protein_mutation_features_json']=json.dumps(dict(zip([*FEATURES,'known_member_fraction','aggregate_available'],map(float,prv[pri[p],6:]))))
                candidates[key]=rec
                if len(candidates)>count_limit:
                    last=max(candidates,key=lambda k:(len(k),k));del candidates[last]
        ordered=sorted(candidates.values(),key=lambda r:(r['ppi_hops'],r['nodes_json']))
        if len(records)+len(ordered)>total_limit:fail('PATH_OUTPUT_CAP','Cannot expand additional queries inside fixed total budget',cap=total_limit)
        for i,r in enumerate(ordered,1):r['display_order']=i;records.append(r)
        summary.append({**query,'path_count':len(ordered),'eligible_walks_examined':seen,'truncated':seen>len(ordered),
                        'availability':'OBSERVED_GRAPH_SUPPORT_NOT_CAUSAL' if ordered else 'NO_QUALIFIED_KNOWLEDGE_PATH_NOT_NEGATIVE'})
    columns=[*q.columns,'path_id','nodes_json','edge_semantics','ppi_hops','edge_confidences_json','minimum_edge_confidence','minimum_confidence_is_not_path_probability','membership_weight','path_direction','source_records_json','causal_mediation_proven','pathway_gene_mutation_features_json','partner_protein_mutation_features_json','display_order']
    pd.DataFrame(records,columns=columns).to_csv(out/'KNOWLEDGE_PATHS.tsv.gz',sep='\t',index=False)
    pd.DataFrame(summary).to_csv(out/'QUERY_PATH_AVAILABILITY.tsv',sep='\t',index=False)
    return finish(out,{'state':'POSTHOC_KNOWLEDGE_PATHS_EXPORTED','module':'tc_kpnn_style_posthoc_not_neural','query':qr,'source_files':hashes,'paths':len(records),
        'maximum_paths_per_query':count_limit,'maximum_ppi_hops':1,'finite_search':True,'genomic_context':prepared_info,'uses_test_labels':False,
        'causal_direction_inferred':False,'no_path_does_not_mean_no_function':True})
