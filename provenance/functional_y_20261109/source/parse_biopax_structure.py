"""Parse BioPAX structure on host149; structural evidence is not patient activity."""
import argparse, collections, csv, gzip, hashlib, json, re, socket, time
from pathlib import Path
import xml.etree.ElementTree as ET

RDF="{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
BP="http://www.biopax.org/release/biopax-level3.owl#"
REACTIONS={"BiochemicalReaction","Transport","TransportWithBiochemicalReaction","ComplexAssembly","Degradation","Conversion","TemplateReaction"}
CONTROLS={"Catalysis","Control","Modulation","TemplateReactionRegulation"}
STRUCTURAL={"pathwayComponent","pathwayOrder","stepProcess","stepConversion","nextStep","left","right","product","template","controller","controlled","component","componentStoichiometry","participantStoichiometry","physicalEntity","entityReference","memberPhysicalEntity","memberEntityReference","cellularLocation","feature","notFeature","featureLocation","modificationType","sequenceIntervalBegin","sequenceIntervalEnd","organism"}
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def read_graph(path):
 nodes={};depth=0;root=None
 def norm(s):return s[1:] if s.startswith("#") else s
 for event,e in ET.iterparse(path,events=("start","end")):
  if event=="start":
   depth+=1
   if root is None:root=e
  else:
   if depth==2:
    uid=e.get(RDF+"about") or e.get(RDF+"ID")
    if uid:
     uid=norm(uid);p=collections.defaultdict(list);refs=collections.defaultdict(list)
     for c in e:
      key=c.tag.rsplit("}",1)[-1]
      if c.get(RDF+"resource") is not None:refs[key].append(norm(c.get(RDF+"resource")))
      elif c.text is not None:p[key].append(c.text.strip())
     assert uid not in nodes,("duplicate URI",uid)
     nodes[uid]={"type":e.tag.rsplit("}",1)[-1],"p":dict(p),"r":dict(refs)}
    e.clear();root.clear()
   depth-=1
 return nodes
def main():
 assert socket.gethostname()=="149"
 a=argparse.ArgumentParser();a.add_argument("--input",required=True);a.add_argument("--output",required=True);a.add_argument("--source",required=True);a.add_argument("--version",required=True);args=a.parse_args()
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
 n=read_graph(args.input);print("PARSED",len(n),flush=True)
 def vals(uid,key):return n.get(uid,{}).get("p",{}).get(key,[])
 def refs(uid,key):return n.get(uid,{}).get("r",{}).get(key,[])
 def label(uid):return (vals(uid,"displayName") or vals(uid,"standardName") or vals(uid,"name") or [uid])[0]
 def xrefs(uid):
  return [(x,(vals(x,"db") or [""])[0],(vals(x,"id") or [""])[0]) for x in refs(uid,"xref") if x in n]
 def stable(uid):
  candidates=set()
  if "bioregistry.io/reactome:" in uid:
   candidates.update(re.findall(r"R-(?:HSA|ALL)-\d+",uid))
  for x,db,i in xrefs(uid):
   if "reactome" in db.lower(): candidates.update(re.findall(r"R-(?:HSA|ALL)-\d+",i))
  return "|".join(sorted(candidates))
 def writer(name,cols,rows):
  path=output/name
  opener=gzip.open if str(path).endswith(".gz") else open
  count=0
  with opener(path,"wt",encoding="utf8",newline="") as f:
   w=csv.DictWriter(f,fieldnames=cols,delimiter="\t",extrasaction="raise");w.writeheader()
   for row in rows:w.writerow(row);count+=1
  return {"path":str(path),"rows":count,"sha256":sha(path),"bytes":path.stat().st_size}
 files={}
 def write(name,cols,rows):files[name]=writer(name,cols,rows)
 nodes_for_export={k for k,v in n.items() if v["type"] not in {"UnificationXref","RelationshipXref","PublicationXref"}}
 write("BIOPAX_NODES.tsv.gz",["node_id","class","name","reactome_stable_id","source","version"],({"node_id":k,"class":n[k]["type"],"name":label(k),"reactome_stable_id":stable(k),"source":args.source,"version":args.version} for k in sorted(nodes_for_export)))
 write("BIOPAX_LITERAL_PROPERTIES.tsv.gz",["node_id","property","value"],({"node_id":k,"property":prop,"value":value} for k,v in n.items() for prop,values in v["p"].items() for value in values))
 edges=[]
 for uid,v in n.items():
  for prop,targets in v["r"].items():
   if prop in STRUCTURAL:
    for target in targets:edges.append({"subject_id":uid,"relation":prop,"object_id":target,"subject_class":v["type"],"object_class":n.get(target,{}).get("type","UNRESOLVED_REFERENCE"),"object_resolved":int(target in n),"source":args.source,"version":args.version})
 write("BIOPAX_EDGES.tsv.gz",["subject_id","relation","object_id","subject_class","object_class","object_resolved","source","version"],edges)
 write("SOURCE_XREFS.tsv.gz",["node_id","xref_id","database","identifier","xref_class"],({"node_id":k,"xref_id":x,"database":db,"identifier":i,"xref_class":n[x]["type"]} for k in sorted(nodes_for_export) for x,db,i in xrefs(k)))
 reactions={k for k,v in n.items() if v["type"] in REACTIONS}
 pathways={k for k,v in n.items() if v["type"]=="Pathway"}
 controls={k for k,v in n.items() if v["type"] in CONTROLS}
 def event_children(k):
  return refs(k,"pathwayComponent")+refs(k,"pathwayOrder")+refs(k,"stepProcess")+refs(k,"stepConversion")
 def descendants(k):
  found=set();todo=list(event_children(k));seen={k}
  while todo:
   v=todo.pop()
   if v in seen:continue
   seen.add(v)
   if v in reactions:found.add(v)
   if v in n and n[v]["type"] in {"Pathway","PathwayStep","BiochemicalPathwayStep"}:todo.extend(event_children(v))
  return found
 pathway_reactions={k:descendants(k) for k in pathways}
 write("PATHWAY_REACTION.tsv.gz",["pathway_node","pathway_stable_id","pathway_name","reaction_node","reaction_stable_id","reaction_name","membership_scope"],({"pathway_node":p,"pathway_stable_id":stable(p),"pathway_name":label(p),"reaction_node":r,"reaction_stable_id":stable(r),"reaction_name":label(r),"membership_scope":"DIRECT_OR_DESCENDANT_EXPLICIT_BIOPAX"} for p in sorted(pathways) for r in sorted(pathway_reactions[p])))
 hierarchy=[]
 for p in sorted(pathways):
  for c in refs(p,"pathwayComponent"):
   if c in pathways:hierarchy.append({"parent_node":p,"parent_stable_id":stable(p),"parent_name":label(p),"child_node":c,"child_stable_id":stable(c),"child_name":label(c),"relation":"parent_of","evidence":"EXPLICIT_BIOPAX_PATHWAY_COMPONENT"})
 write("PATHWAY_HIERARCHY.tsv",["parent_node","parent_stable_id","parent_name","child_node","child_stable_id","child_name","relation","evidence"],hierarchy)
 roles=[]
 for r in reactions:
  for prop,role in [("left","input"),("right","output"),("product","product"),("template","template")]:
   for e in refs(r,prop):roles.append({"reaction_node":r,"reaction_stable_id":stable(r),"role":role,"entity_node":e,"entity_name":label(e),"entity_class":n.get(e,{}).get("type","UNRESOLVED_REFERENCE"),"control_node":"","control_type":"","role_evidence":"BIOPAX_"+prop})
 def controlled_reactions(c):
  todo=refs(c,"controlled");seen=set();found=set()
  while todo:
   t=todo.pop()
   if t in seen:continue
   seen.add(t)
   if t in reactions:found.add(t)
   elif t in controls:todo+=refs(t,"controlled")
  return found
 for c in controls:
  for r in controlled_reactions(c):
   for e in refs(c,"controller"):roles.append({"reaction_node":r,"reaction_stable_id":stable(r),"role":"catalyst" if n[c]["type"]=="Catalysis" else "regulator","entity_node":e,"entity_name":label(e),"entity_class":n.get(e,{}).get("type","UNRESOLVED_REFERENCE"),"control_node":c,"control_type":"|".join(vals(c,"controlType")),"role_evidence":"BIOPAX_CONTROL_CHAIN;SIGN_APPLIES_TO_DIRECT_CONTROLLED_OBJECT"})
 write("REACTION_ENTITY_ROLES.tsv.gz",["reaction_node","reaction_stable_id","role","entity_node","entity_name","entity_class","control_node","control_type","role_evidence"],roles)
 aliases=[]
 for p in sorted(pathways):
  for name in sorted(set(vals(p,"displayName")+vals(p,"standardName")+vals(p,"name"))):aliases.append({"pathway_node":p,"reactome_stable_id":stable(p),"alias":name,"evidence":"SAME_SOURCE_OBJECT_NAME","source":args.source,"version":args.version})
 write("PATHWAY_ALIASES.tsv",["pathway_node","reactome_stable_id","alias","evidence","source","version"],aliases)
 # Keep complex membership and alternative entity sets distinct. No conversion to PPI evidence.
 write("COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz",["container_node","container_class","member_node","member_class","relation","stoichiometry_nodes"],({"container_node":k,"container_class":v["type"],"member_node":m,"member_class":n.get(m,{}).get("type","UNRESOLVED_REFERENCE"),"relation":prop,"stoichiometry_nodes":"|".join(refs(k,"componentStoichiometry"))} for k,v in n.items() for prop in ["component","memberPhysicalEntity","memberEntityReference"] for m in refs(k,prop)))
 def protein_ids(k):
  result=set()
  for er in refs(k,"entityReference")+[k]:
   if "bioregistry.io/uniprot:" in er:result.add(er.rsplit(":",1)[-1])
   for _,db,ident in xrefs(er):
    if db.lower() in {"uniprot","uniprotkb","uniprot knowledgebase"}:result.add(ident)
  return sorted(result)
 features=[]
 protein_rows=[]
 for k,v in n.items():
  if v["type"]!="Protein":continue
  identifiers=protein_ids(k)
  protein_rows.append({"protein_state_node":k,"name":label(k),"reactome_stable_id":stable(k),"uniprot_accessions":"|".join(identifiers),"cellular_location_nodes":"|".join(refs(k,"cellularLocation")),"feature_nodes":"|".join(refs(k,"feature"))})
  for f in refs(k,"feature"):
   if n.get(f,{}).get("type")!="ModificationFeature":continue
   for mod in refs(f,"modificationType") or [""]:
    for site in refs(f,"featureLocation") or [""]:
     features.append({"protein_state_node":k,"uniprot_accessions":"|".join(identifiers),"feature_node":f,"modification_type_node":mod,"modification_terms":"|".join(vals(mod,"term")),"location_node":site,"sequence_position":"|".join(vals(site,"sequencePosition")),"position_status":"|".join(vals(site,"positionStatus")),"location_class":n.get(site,{}).get("type","UNSPECIFIED"),"site_inferred_from_name":False})
 write("PROTEIN_STATES.tsv.gz",["protein_state_node","name","reactome_stable_id","uniprot_accessions","cellular_location_nodes","feature_nodes"],protein_rows)
 write("PROTEIN_MODIFICATIONS.tsv.gz",["protein_state_node","uniprot_accessions","feature_node","modification_type_node","modification_terms","location_node","sequence_position","position_status","location_class","site_inferred_from_name"],features)
 summary={"target_host":"149","source":args.source,"version":args.version,"input_path":args.input,"input_sha256":sha(args.input),"script_sha256":sha(__file__),"classes":dict(collections.Counter(v["type"] for v in n.values())),"pathways_with_reactions":sum(bool(v) for v in pathway_reactions.values()),"reaction_count":len(reactions),"control_count":len(controls),"unresolved_structural_references":sum(not e["object_resolved"] for e in edges),"role_counts":dict(collections.Counter(r["role"] for r in roles)),"files":files,"semantics":{"patient_activity_scores_generated":False,"relationship_Y_generated":False,"physical_PPI_evidence_claimed":False,"global_pathway_sign_inferred":False}}
 (output/"STRUCTURE_RECEIPT.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k!="files"}),flush=True)
if __name__=="__main__":main()
