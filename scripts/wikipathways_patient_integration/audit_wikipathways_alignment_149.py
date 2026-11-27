import socket,json,sys,hashlib,datetime
from pathlib import Path
import pandas as pd
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';N=R/'canonical_targets_r11_state_groups_20261114_r1';REPO=R/'source_integration/repo';O=W/'wiki_patient_integration_20261114_r1/source_audit';O.mkdir(parents=True,exist_ok=True)
assert socket.gethostname()=='149'
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'WikiPathways context adapter and source routing audit','input_catalog':str(N),'no_patient_data_read':True},indent=2))
sys.path.insert(0,str(REPO));from cla_engine.wikipathways_alignment import build_wikipathways_alignment
read=lambda f:pd.read_csv(N/f,sep='\t',keep_default_na=False)
defs=read('DEFINITION_TO_IDENTITY.tsv');composition=read('INTEGRATED_PROTEIN_COMPOSITION.tsv.gz');backbone=read('GROUPED_BACKBONE.tsv');old=read('BACKBONE_ALIGNMENT.tsv');previous=json.loads((N/'ALIGNMENT_SUMMARY.json').read_text());wp=defs[defs.database_family=='WIKIPATHWAYS'];assert len(wp)==193 and len(backbone)==476
candidates,routes,dispositions,summary=build_wikipathways_alignment(defs.to_dict('records'),composition.to_dict('records'),backbone.to_dict('records'),previous_alignment=old.to_dict('records'),previous_summary=previous)
route=pd.DataFrame(routes);disp=pd.DataFrame(dispositions);cand=pd.DataFrame(candidates)
cand.reindex(columns=old.columns).to_csv(O/'WP_BACKBONE_ALIGNMENT.tsv',sep='\t',index=False);route.to_csv(O/'WP_SOURCE_ROUTES.tsv',sep='\t',index=False);disp.to_csv(O/'WP_SOURCE_DISPOSITIONS.tsv',sep='\t',index=False);cand.to_csv(O/'WP_NONZERO_ALIGNMENT_CANDIDATES.tsv.gz',sep='\t',index=False)
# Also materialize the zero-overlap audit rows, so all 193 x 476 comparisons are explicit.
lookup={(r['source_canonical_id'],r['target_canonical_id']):r for r in candidates};counts=composition.groupby('source_definition').uniprot_id.nunique().to_dict();full=[]
for s in wp.to_dict('records'):
 for t in backbone.to_dict('records'):
  key=(s['node_id'],t['canonical_id']);r=dict(lookup.get(key,{}))
  if not r:r={'source_canonical_id':s['node_id'],'source_name':s['display_name'],'source_database':'WIKIPATHWAYS','target_canonical_id':t['canonical_id'],'network_id':t['network_id'],'target_name':t['display_name'],'source_protein_count':s['protein_count'],'target_protein_count':counts[t['source_target_id']],'shared_proteins':0,'jaccard':0,'source_coverage':0,'target_coverage':0,'overlap_p':1,'overlap_q':1,'overlap_q_wp_family':1,'admitted_context_link':'False','relation':'protein_component_association','identity_merge':'False','mechanism_context_inherited':'False'}
  full.append(r)
pd.DataFrame(full).to_csv(O/'WP_ALL_ALIGNMENT_COMPARISONS.tsv.gz',sep='\t',index=False)
route[route.relation=='same_functional_identity'].to_csv(O/'WP_EQUIVALENT_ROUTES.tsv',sep='\t',index=False)
disp[disp.disposition=='zero_backbone_overlap'].to_csv(O/'WP_ZERO_BACKBONE_OVERLAP.tsv',sep='\t',index=False)
# Existing protein clusters are candidate context evidence, not an identity gate.
membership=read('ALL_GROUP_MEMBERSHIP.tsv');pids=set(backbone.canonical_id);anchor_nodes=set(defs[defs.integrated_canonical_id.isin(pids)].node_id);anchor_clusters=set(membership[membership.node_id.isin(anchor_nodes)].protein_cluster_id);wpclusters=membership[membership.node_id.isin(set(wp.node_id))].copy();wpclusters['cluster_contains_existing_primary_definition']=wpclusters.protein_cluster_id.isin(anchor_clusters);wpclusters.to_csv(O/'WP_EXISTING_PROTEIN_CLUSTERS.tsv',sep='\t',index=False)
primary=read('PRIMARY_PATHWAY_TARGETS.tsv');eq=route[route.relation=='same_functional_identity'].merge(primary[['canonical_id','display_name','authority_id']],left_on='target_canonical_id',right_on='canonical_id',validate='many_to_one');eq.to_csv(O/'WP_EQUIVALENT_IDENTITIES.tsv',sep='\t',index=False)
summary.update({'context_target_count':int(route[route.relation=='protein_component_association'].target_canonical_id.nunique()),'all_routed_target_count':int(route.target_canonical_id.nunique()),'existing_WP_alignment_rows':int(old.source_canonical_id.isin(set(wp.node_id)).sum()),'existing_WP_definitions_with_empty_canonical_id':int(wp.canonical_id.eq('').sum()),'existing_protein_cluster_sources_with_primary_definition':int(wpclusters.cluster_contains_existing_primary_definition.sum()),'why_existing_context_count_was_zero':'External WikiPathways definitions have REF node IDs and blank canonical_id; the previous alignment did not include these REF sources. Zero means not tested, not no biological overlap.','full_source_lists_retained':True,'whole_source_score_routing':'Only the three previously reviewed same-functional-identity mappings can route whole-source scores to an exact target. Other accepted links contribute shared-member states and graph context; no identity averaging.'})
(O/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
inputs=[N/x for x in ['DEFINITION_TO_IDENTITY.tsv','INTEGRATED_PROTEIN_COMPOSITION.tsv.gz','GROUPED_BACKBONE.tsv','BACKBONE_ALIGNMENT.tsv','ALIGNMENT_SUMMARY.json','ALL_GROUP_MEMBERSHIP.tsv']];inputs += [REPO/'cla_engine/wikipathways_alignment.py',REPO/'cla_engine/backbone_alignment.py']
(O/'AUDIT_RECEIPT.json').write_text(json.dumps({'target_host':'149','inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file() and p.name!='AUDIT_RECEIPT.json'}},indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));print(eq[['source_external_id','source_name','target_canonical_id','display_name','network_id']].to_string(index=False));print('ZERO',disp[disp.disposition=='zero_backbone_overlap'][['source_external_id','source_name']].to_json(orient='records'))
