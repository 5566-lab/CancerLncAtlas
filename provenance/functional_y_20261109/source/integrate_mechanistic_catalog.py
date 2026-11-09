import csv,gzip,pathlib,json,collections,hashlib,socket,itertools
assert socket.gethostname()=="149"
w=pathlib.Path("/dsk2/user/dengsc/cla_functional_y_20261109_r1");s=w/"structure_reactome98";out=w/"catalog_inventory"
def read(p):
 with (gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)) as f:yield from csv.DictReader(f,delimiter="\t")
def write(name,rows,cols=None):
 path=out/name;count=0
 with (gzip.open(path,"wt",newline="") if name.endswith(".gz") else open(path,"w",newline="")) as f:
  it=iter(rows)
  try:first=next(it)
  except StopIteration:first=None
  cols=cols or (list(first) if first else [])
  wr=csv.DictWriter(f,fieldnames=cols,delimiter="\t");wr.writeheader()
  if first is not None:
   wr.writerow(first);count+=1
   for r in it:wr.writerow(r);count+=1
 return count
def canonical(st):return "CLA-P-"+hashlib.sha256(("REACTOME:"+st).encode()).hexdigest()[:16].upper()
nodes={r["node_id"]:r for r in read(s/"BIOPAX_NODES.tsv.gz")}
path_nodes={k:v for k,v in nodes.items() if v["class"]=="Pathway"}
byid={v["reactome_stable_id"]:k for k,v in path_nodes.items() if v["reactome_stable_id"]}
assert len(byid)==len(path_nodes), "Ambiguous pathway stable IDs"
reactions=collections.defaultdict(set)
for r in read(s/"PATHWAY_REACTION.tsv.gz"):reactions[r["pathway_node"]].add(r["reaction_node"])
roles=collections.defaultdict(list)
for r in read(s/"REACTION_ENTITY_ROLES.tsv.gz"):roles[r["reaction_node"]].append(r)
source=list(read(out/"PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv"));coverage=[]
for r in source:
 st=r["reactome_id"];p=byid.get(st);rs=reactions.get(p,set());rr=[x for rx in rs for x in roles.get(rx,[])]
 controls={x["control_node"] for x in rr if x["role"]=="regulator"}
 cats={x["control_node"] for x in rr if x["role"]=="catalyst"}
 complexes={x["entity_node"] for x in rr if x["entity_class"]=="Complex"}
 coverage.append(dict(canonical_id=r["canonical_id"],source_target_id=r["source_target_id"],source_database=r["source_database"],reactome_id=st,reactome98_name=nodes[p]["name"] if p else "",structure_status="DIRECT_VERIFIED_ID_BRIDGE" if p else ("UNRESOLVED_REACTOME_ID" if r["source_database"]=="REACTOME" else "CROSS_DATABASE_MECHANISM_REVIEW_REQUIRED"),descendant_reactions=len(rs),catalysis_records=len(cats),regulation_records=len(controls),direct_role_complex_entities=len(complexes),scope="CURRENT_REACTOME_DEFINITION;OLD_SCORE_GENESET_REMAINS_VERSIONED",score_values_generated=False))
write("PATHWAY_MECHANISM_COVERAGE.tsv",coverage)
catalog_ids={r["reactome_id"] for r in source if r["reactome_id"]}
pathmaster=[]
for k,n in path_nodes.items():
 st=n["reactome_stable_id"]
 pathmaster.append(dict(canonical_id=canonical(st),reactome_id=st,name=n["name"],authority_version="98",in_existing_score_catalog=st in catalog_ids,role="existing_scored_definition" if st in catalog_ids else "mechanistic_context_until_score_registered",descendant_reaction_count=len(reactions[k]),definition_node=k))
write("REACTOME_PATHWAY_AUTHORITY.tsv",pathmaster)
alias_rows=[]
for r in read(s/"PATHWAY_ALIASES.tsv"):
 alias_rows.append(dict(canonical_id=canonical(r["reactome_stable_id"]),reactome_id=r["reactome_stable_id"],alias=r["alias"],relation="synonym_of",evidence="NAMES_OF_SAME_OFFICIAL_REACTOME98_OBJECT",identity_merge=True))
write("REACTOME_SYNONYMS.tsv",alias_rows)
hr=list(read(s/"PATHWAY_HIERARCHY.tsv"));hrows=[]
children=collections.defaultdict(set)
for r in hr:
 p=r["parent_stable_id"];c=r["child_stable_id"];children[p].add(c)
 hrows.append(dict(parent_id=canonical(p),parent_reactome_id=p,child_id=canonical(c),child_reactome_id=c,relation="parent_of",identity_merge=False,evidence=r["evidence"],source_version="Reactome98"))
write("REACTOME_PARENT_CHILD.tsv",hrows)
def siblings():
 for p,ch in children.items():
  if len(ch)<2:continue
  for a,b in itertools.combinations(sorted(ch),2):
   yield dict(parent_id=canonical(p),left_id=canonical(a),right_id=canonical(b),relation="co_child_of",evidence="EXPLICIT_SHARED_PARENT",independent_functional_branch_claimed=False,identity_merge=False)
n_siblings=write("REACTOME_HIERARCHY_SIBLINGS.tsv.gz",siblings())
# Keep biochemical forks separate from mere siblings in a hierarchy.
nxt=collections.defaultdict(set);step_events=collections.defaultdict(set)
for r in read(s/"BIOPAX_EDGES.tsv.gz"):
 if r["relation"]=="nextStep":nxt[r["subject_id"]].add(r["object_id"])
 if r["relation"] in {"stepProcess","stepConversion"}:step_events[r["subject_id"]].add(r["object_id"])
branch=[]
for step,nexts in nxt.items():
 if len(nexts)>1:
  for child in sorted(nexts):
   branch.append(dict(fork_step=step,upstream_processes="|".join(sorted(step_events[step])),downstream_step=child,downstream_processes="|".join(sorted(step_events[child])),relation="reaction_order_fork",evidence="EXPLICIT_BIOPAX_NEXTSTEP",n_next_steps=len(nexts),independent_flux_claimed=False))
write("REACTION_ORDER_BRANCHES.tsv.gz",branch)
summary=dict(target_host="149",reactome_release="98",source_scored_pathways=len(source),reactome_source_pathways=sum(r["source_database"]=="REACTOME" for r in source),scored_pathways_bridged=sum(r["structure_status"]=="DIRECT_VERIFIED_ID_BRIDGE" for r in coverage),coverage_status=dict(collections.Counter(r["structure_status"] for r in coverage)),authority_pathways=len(pathmaster),official_aliases=len(alias_rows),explicit_parent_child=len(hr),hierarchy_sibling_links=n_siblings,reaction_order_fork_steps=sum(len(x)>1 for x in nxt.values()),reaction_order_branch_edges=len(branch),not_claimed=["final_crossdatabase_deduplication","functional_equivalence_from_shared_proteins","patient_activity_scores","final_relationship_Y"])
(out/"MECHANISM_INTEGRATION_SUMMARY.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
