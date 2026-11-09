import pathlib,json,csv,gzip,hashlib,socket,collections,importlib.util,sys
assert socket.gethostname()=="149"
b=pathlib.Path("/dsk2/user/dengsc");w=b/"cla_functional_y_20261109_r1"
out=w/"catalog_inventory";out.mkdir(exist_ok=True)
cat_path=b/"cla_pathway_fusion_20261101_r1/raw_inputs/CATALOG.json"
xref_path=b/"cla_pathway_fusion_20261101_r1/REACTOME_XREFS.json"
state_path=b/"cla_state_signature_update_20261104_r1/repo/config/state_catalog_augmented_20261104.json"
cat=json.loads(cat_path.read_text());xref=json.loads(xref_path.read_text());state=json.loads(state_path.read_text())
look={}
for x in xref["rows"]:
 if x.get("status")=="VERIFIED_DIRECT_XREF":look[x["pathway_id"]]=x["reactome_id"]
def sid(kind,key):return "CLA-"+kind+"-"+hashlib.sha256(key.encode()).hexdigest()[:16].upper()
rows=[]
for pid,genes in cat["gene_sets"].items():
 rx=look.get(pid,"");key="REACTOME:"+rx if rx else pid
 rows.append(dict(canonical_id=sid("P",key),target_type="pathway" if ":HALLMARK:" not in pid else "functional_program",source_target_id=pid,authority_id=key,reactome_id=rx,source_database=pid.split(":")[1] if pid.startswith("MSIGDB:") else pid.split(":")[0],source_name=pid.split(":")[-1],member_count=len(genes),identity_evidence="VERIFIED_DIRECT_REACTOME_XREF" if rx else "SOURCE_DEFINITION_PRESERVED",integration_status="INVENTORY_NOT_FINAL_FUSION",score_mix_allowed=False))
assert len({r["source_target_id"] for r in rows})==len(rows)
def write(name,rows):
 if not rows:return
 with (out/name).open("w",newline="",encoding="utf8") as f:
  d=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t");d.writeheader();d.writerows(rows)
write("PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv",rows)
targets=state["targets"];print("STATE_TARGET_EXAMPLE",targets[:2]);print("STATE_FAMILIES",dict(collections.Counter(x["family"] for x in targets)))
states=[]
for t in targets:
 states.append(dict(canonical_id=sid("S","STATE:"+t["id"]),target_type="state",source_target_id=t["id"],source_column=t.get("column",""),family=t.get("family",""),role=t.get("role",""),original_definition=json.dumps(t,ensure_ascii=False,sort_keys=True),identity_evidence="EXISTING_SOURCE_DEFINITION_PRESERVED",integration_status="INVENTORY_NOT_FINAL_FUSION"))
write("STATE_SOURCE_ID_REGISTRY_DRAFT.tsv",states)
bridge=[]
with open(w/"structure_pc14_akt/PATHWAY_ALIASES.tsv") as f:
 parsed=list(csv.DictReader(f,delimiter="\t"))
byreact=collections.defaultdict(list)
for r in rows:
 if r["reactome_id"]:byreact[r["reactome_id"]].append(r)
for item in parsed:
 for r in byreact.get(item["reactome_stable_id"],[]):
  bridge.append(dict(canonical_id=r["canonical_id"],source_target_id=r["source_target_id"],reactome_id=item["reactome_stable_id"],biopax_node=item["pathway_node"],biopax_name=item["alias"],source="PathwayCommons14",upstream_version="Reactome81",relationship="SAME_STABLE_ID_VERSIONED_DEFINITION",score_merge_authorized=False))
write("AKT_EXISTING_TARGET_BRIDGE.tsv",bridge)
input_records=[]
for p in [cat_path,xref_path,state_path]:
 input_records.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
summary=dict(target_host="149",status="SOURCE_INVENTORY_AND_MECHANISM_PILOT;NOT_FINAL_Y",pathway_source_definitions=len(rows),pathway_identity_count_before_crossdatabase_review=len({r["canonical_id"] for r in rows}),verified_reactome_source_definitions=sum(bool(r["reactome_id"]) for r in rows),state_source_definitions=len(states),mechanism_pilot_existing_targets=len({r["source_target_id"] for r in bridge}),source_counts=dict(collections.Counter(r["source_database"] for r in rows)),inputs=input_records,ID_policy="SHA256_16_OF_AUTHORITY_NAMESPACE_AND_STABLE_ID;VERSIONED_DEFINITIONS_RETAINED;DO_NOT_RENUMBER",remaining=["Full Reactome98 parsing","Mechanistic synonym/extension/parent/branch mapping","Cross-database and State semantic review","All score source registry and patient coverage","Actual cancer-lncRNA-target relation label construction and supervision counts","Leakage and missing-label audits","Final figures and PPT and dictionary delivery"],main_task="Cancer x lncRNA x precise canonical target",retired_methods=["MOGSA"],PathFA_adopted=False)
(w/"BUILD_PROGRESS.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
for p in [b/"cla_m0_linear_20261104_r1/python_deps",b/"cla_m0_lasso_20261104_r1/python_deps",b/"cla_m0_joint_20261105_r1/python_deps"]:sys.path.insert(0,str(p))
print("PACKAGES",{n:bool(importlib.util.find_spec(n)) for n in ["numpy","pandas","matplotlib","pptx","openpyxl","pyarrow"]})
