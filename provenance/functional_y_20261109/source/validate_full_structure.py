import pathlib,csv,gzip,json,collections,xml.etree.ElementTree as ET,socket,hashlib
assert socket.gethostname()=="149"
w=pathlib.Path("/dsk2/user/dengsc/cla_functional_y_20261109_r1")
s=w/"structure_reactome98";receipt=json.loads((s/"STRUCTURE_RECEIPT.json").read_text())
depth=0;counter=collections.Counter()
for event,e in ET.iterparse(w/"sources/Reactome98_Homo_sapiens.owl",events=("start","end")):
 if event=="start":
  depth+=1
  if depth==1:root=e
 else:
  if depth==2:
   if e.tag.rsplit("}",1)[-1]!="Ontology":counter[e.tag.rsplit("}",1)[-1]]+=1
   e.clear();root.clear()
  depth-=1
assert dict(counter)==receipt["classes"]
assert receipt["unresolved_structural_references"]==0
def read(p):
 with (gzip.open(p,"rt") if str(p).endswith(".gz") else open(p)) as f:yield from csv.DictReader(f,delimiter="\t")
for name,info in receipt["files"].items():
 assert sum(1 for _ in read(s/name))==info["rows"],name
parents=list(read(w/"catalog_inventory/REACTOME_PARENT_CHILD.tsv"))
degree=collections.Counter();adj=collections.defaultdict(set);nodes=set()
for row in parents:
 a,b=row["parent_id"],row["child_id"];assert a!=b
 nodes.update([a,b])
 if b not in adj[a]:adj[a].add(b);degree[b]+=1
queue=collections.deque(n for n in nodes if degree[n]==0);visited=[]
while queue:
 a=queue.popleft();visited.append(a)
 for b in adj[a]:
  degree[b]-=1
  if degree[b]==0:queue.append(b)
assert len(visited)==len(nodes),"Pathway hierarchy has cycle"
coverage=list(read(w/"catalog_inventory/PATHWAY_MECHANISM_COVERAGE.tsv"))
assert len(coverage)==2547 and len({r["source_target_id"] for r in coverage})==2547
assert all(int(r["descendant_reactions"])>0 for r in coverage if r["structure_status"]=="DIRECT_VERIFIED_ID_BRIDGE")
master=list(read(w/"catalog_inventory/REACTOME_PATHWAY_AUTHORITY.tsv"))
assert len({r["canonical_id"] for r in master})==len(master)
result={"target_host":"149","status":"PASS_FULL_REACTOME98_STRUCTURE","independent_XML_objects":sum(counter.values()),"checked_export_tables":len(receipt["files"]),"hierarchy_is_acyclic":True,"source_pathways":len(coverage),"source_pathways_bridged":sum(r["structure_status"]=="DIRECT_VERIFIED_ID_BRIDGE" for r in coverage),"validated_scope":"Downloaded BioPAX, parsing, canonical ID bridge, hierarchy and export integrity","excluded_scope":["Cross-database functional equivalence","Final retained target count","Patient score validity","Relationship Y and per-cancer supervision"]}
(w/"STRUCTURE_FULL_VALIDATION.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
