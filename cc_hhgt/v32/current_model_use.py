"""Derive model input exposure from native selected/core/fusion producers.

This inventory inspects authorized tensors and dependencies without constructing
an optimizer or doing model forwards. It records unresolved upstream study
provenance explicitly; file integrity is not proof of biological independence.
"""
from __future__ import annotations
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd

from .current_core_export import sha256,verified_path,validate_checkpoint
from .evidence_rule_summary import KEYS,stable_id
from .external_validation_sources import SOURCE_COLUMNS

INPUT_FORMAT="CURRENT_MODEL_USE_INPUT_V1"
FORMAT="CURRENT_MODEL_USE_RELEASE_V1"
SCOPE="NATIVE_INPUT_EXPOSURE_WITH_UNRESOLVED_UPSTREAM_PROVENANCE"
EXTRA=["model_id","patient_fold_id","use_view","candidate_rows","entity_scope"]
DEPENDENCIES=["model_id","patient_fold_id","role","path","sha256","status"]
GAPS=["model_id","patient_fold_id","role","sha256","reason"]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def record(path):return {"path":str(Path(path).resolve()),"sha256":sha256(path)}
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def verified_source(item,root):
    from .input_lineage import artifact_sha256
    path=Path(item["path"]);path=(root/path).resolve() if not path.is_absolute() else path.resolve()
    if artifact_sha256(path)!=item["sha256"]:raise ValueError("Model-use dependency SHA256 mismatch: "+str(path))
    return path


def implementations():
    root=Path(__file__).resolve().parents[2]
    names=["cc_hhgt/v32/current_model_use.py","cc_hhgt/v32/current_core_prediction.py",
        "cc_hhgt/v32/current_core_export.py","cc_hhgt/v32/current_primary_fusion.py",
        "scripts/export_v32_current_core_embeddings.py","cc_hhgt/v32/training.py",
        "cc_hhgt/v32/external_validation_current.py"]
    return {name:sha256(root/name) for name in names}


def model_identities(models,root):
    result={}
    for spec in models:
        name=spec.get("model_id")
        if not isinstance(name,str) or not name or name in result:raise ValueError("Model-use IDs must be nonempty and unique")
        path=verified_path(spec["producer_binding"],root)
        result[name]={k:spec[k] for k in ("run_id","producer_kind","score_column")}
        result[name]["producer_binding_sha256"]=sha256(path)
    if not result:raise ValueError("Model-use inventory needs evaluated models")
    return result


class Inventory:
    def __init__(self,locations,root):
        self.sources={};self.dependencies=[];self.gaps=[];self.usage=[];self.visited=set();self.locations={};self.used_locations=set()
        for item in locations:
            path=verified_source(item,root);digest=item["sha256"]
            if digest in self.locations:raise ValueError("Duplicate model-use artifact location")
            self.locations[digest]={"path":str(path),"sha256":digest}

    def bind(self,item,root,model,role,fold=-1):
        path=verified_source(item,root);source={"path":str(path),"sha256":item["sha256"]};self.sources[str(path)]=source
        self.dependencies.append({"model_id":model,"patient_fold_id":fold,"role":role,**source,
            "status":"VERIFIED_DIRECTORY" if path.is_dir() else "VERIFIED_FILE"})
        return path

    def gap(self,model,role,reason,fold=-1,digest=""):
        self.gaps.append({"model_id":model,"patient_fold_id":fold,"role":role,"sha256":digest,"reason":reason})

    def authority(self,item,root,model,role,fold):
        digest=item.get("sha256","")
        if not digest:
            self.gap(model,role,"AUTHORITY_HAS_NO_FILE_HASH",fold);return
        source=self.locations.get(digest,item)
        if digest in self.locations:self.used_locations.add(digest)
        path=Path(source.get("path",""));path=path if path.is_absolute() else root/path
        if not source.get("path") or not (path.is_file() or path.is_dir()):
            self.dependencies.append({"model_id":model,"patient_fold_id":fold,"role":role,"path":str(path) if source.get("path") else "",
                "sha256":digest,"status":"UNRESOLVED_LOCATION"})
            self.gap(model,role,"HASH_BOUND_AUTHORITY_NOT_LOCATED",fold,digest);return
        self.bind(source,root,model,role,fold)
        self.gap(model,role,"RAW_STUDY_LINEAGE_NOT_RECONSTRUCTED",fold,digest)

    def entities(self,frame,*,model,fold,phase,view,digest):
        """Bounded by the lncRNA universe, not candidates times five folds.

        A lncRNA seen by a graph/feature/training input is conservatively treated
        as exposed for every pathway. Counts retain the native candidate support;
        they are not sample counts, effective gradients or confidence weights.
        """
        if frame.empty:return
        if frame.lncrna_id.isna().any() or frame.lncrna_id.astype(str).str.strip().eq("").any():
            raise ValueError("Native model-use entity identity is missing")
        counts=frame.groupby("lncrna_id",sort=True).size()
        source_id="NATIVE-USE:"+stable_id("",[model,fold,phase,view,digest])
        for lnc,count in counts.items():
            ids={"lncrna":str(lnc),"cancer":"","pathway":""}
            identity=stable_id("",[source_id,str(lnc)])
            self.usage.append({"cancer_id":"","lncrna_id":str(lnc),"pathway_id":"","source_id":source_id,
                "source_event_id":"MODEL-INPUT:"+identity,"source_file_sha256":digest,"phase":phase,"endpoint":"MODEL_USE",
                "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,
                "assay":"NATIVE_MODEL_INPUT_EXPOSURE","biosample":"","study_tokens_json":"[]",
                "raw_record_id":identity,"raw_row_sha256":stable_id("",[ids,int(count)]),"raw_identifiers_json":json.dumps(ids,sort_keys=True),
                "model_id":model,"patient_fold_id":fold,"use_view":view,"candidate_rows":int(count),
                "entity_scope":"LNCRNA_EXPOSURE_ALL_PATHWAYS"})

    def core(self,path,binding,model,score_column):
        import torch
        from .current_core_prediction import candidate_table,prediction_batches
        from scripts.export_v32_current_core_embeddings import _validated_payload
        source=self.bind(binding["input_manifest"],path.parent,model,"core_prediction_input")
        manifest=read(source);root=source.parent;candidates=candidate_table(manifest,root)
        for role in ("candidate_authority","lncrna_scope"):self.bind(manifest[role],root,model,role)
        for item in sorted(manifest["fold_inputs"],key=lambda r:r["fold"]):
            fold=item["fold"]
            roles={"config","input_manifest","prepared","checkpoint","training_receipt"}
            roles|={r for r in ("graph_overlay","global_binding_source","binding_expansion") if r in item}
            paths={r:self.bind(item[r],root,model,r,fold) for r in sorted(roles)}
            receipt=read(paths["training_receipt"])
            checkpoint=torch.load(paths["checkpoint"],map_location="cpu",weights_only=False)
            validate_checkpoint(checkpoint,receipt,fold=fold,prepared_sha=item["prepared"]["sha256"],
                config_sha=item["config"]["sha256"],input_manifest_sha=item["input_manifest"]["sha256"])
            payload=_validated_payload(paths,item,fold)
            # Reuse the exact candidate alignment and validation-only input firewall.
            features=prediction_batches(payload,candidates)
            digest=item["prepared"]["sha256"]
            self.entities(candidates,model=model,fold=fold,phase="FEATURE_CONSTRUCTION",view="CANDIDATE_FEATURES",digest=digest)
            if score_column=="l1_probability":
                # The published baseline is sigmoid(base_logit). The reference
                # neural head, graph and its checkpoint selection did not fit
                # those baseline values, even though they share a container.
                self.gap(model,"baseline.preparation","BASELINE_FITTING_STUDY_DEPENDENCIES_NOT_RECONSTRUCTED",fold,digest)
                del payload,checkpoint,features
                continue
            graph_nodes=pd.DataFrame({"lncrna_id":list(payload["bundle"].node_maps["lncRNA"])})
            self.entities(graph_nodes,model=model,fold=fold,phase="FEATURE_CONSTRUCTION",view="GRAPH_NODE_INPUT",digest=digest)
            for split,phase in (("train_batches","TRAINING"),("validation_batches","MODEL_SELECTION")):
                indices=[]
                if len(payload[split])!=len(features):raise ValueError("Model-use split alignment differs")
                for batch,feature in zip(payload[split],features):
                    positions=feature["candidate_positions"]
                    mask=batch.get("association_available",torch.ones(len(positions),dtype=torch.bool))
                    if not torch.is_tensor(mask) or mask.dtype!=torch.bool or tuple(mask.shape)!=(len(positions),):
                        raise ValueError("Model-use supervision requires an aligned boolean availability mask")
                    indices.extend(positions[mask.cpu().numpy()].tolist())
                self.entities(candidates.iloc[indices],model=model,fold=fold,phase=phase,view="AVAILABLE_ASSOCIATION_SUPERVISION",digest=digest)
            graph=payload["formal_graph_authority"]
            known=set()
            for group in ("static_artifacts","fold_artifacts"):
                for role,value in graph.get(group,{}).items():
                    self.authority(value,paths["prepared"].parent,model,"graph."+group+"."+role,fold);known.add(value.get("sha256"))
            for role,value in [("graph.receipt",graph.get("receipt",{})),
                *[("patient."+k,payload.get("patient_fold_authority",{}).get(k,{})) for k in ("sample_patient_fold_map","authority_receipt")]]:
                self.authority(value,paths["prepared"].parent,model,role,fold);known.add(value.get("sha256"))
            known.update(r["sha256"] for r in item.values() if isinstance(r,dict) and "sha256" in r)
            for role,digest in receipt["input_authority_hashes"].items():
                if isinstance(digest,str) and len(digest)==64 and digest not in known:
                    self.gap(model,"prepared."+role,"UNRESOLVED_INPUT_AUTHORITY_DIGEST",fold,digest)
            self.gap(model,"prepared.feature_and_label_construction","RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED",fold,item["prepared"]["sha256"])
            del payload,checkpoint,features

    def producer(self,spec,root,model):
        from .external_validation_current import producer_scores
        path=self.bind(spec["producer_binding"],root,model,"producer."+spec["producer_kind"])
        identity=(model,spec["producer_kind"],spec["score_column"],sha256(path))
        if identity in self.visited:return
        self.visited.add(identity)
        # Inspect actual producer semantics and reconstruct published scores.
        producer_scores(spec,root);binding=read(path);kind=spec["producer_kind"]
        if kind=="current_core_release":self.core(path,binding,model,spec["score_column"])
        elif kind=="selected_release":
            source=self.bind(binding["input_manifest"],path.parent,model,"selected_policy");manifest=read(source)
            self.producer(manifest["selected_model"],source.parent,model)
            # Reference directions and rule evidence annotate the release;
            # they do not fit or modify its selected numeric score.
            selection=manifest["selection"]
            if selection["kind"]=="G012_VALIDATION_WINNER":
                self.bind(selection["binding"],source.parent,model,"model_selection_lock")
                self.gap(model,"model_selection_lock","ALL_COMPETITOR_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED")
            # Rule annotations do not change scores or fit model parameters.
        elif kind=="fusion_release":
            artifacts={k:self.bind(v,path.parent,model,"fusion."+k) for k,v in binding["artifacts"].items()}
            frame=pd.read_parquet(artifacts["fusion_input_private"])
            self.entities(frame[KEYS],model=model,fold=-1,phase="FEATURE_CONSTRUCTION",view="FUSION_FEATURES",digest=sha256(artifacts["fusion_input_private"]))
            if "fusion_target_available" in frame:
                mask=frame.fusion_target_available
                if mask.isna().any() or not pd.api.types.is_bool_dtype(mask):raise ValueError("Invalid fusion target availability")
                self.entities(frame.loc[mask,KEYS],model=model,fold=-1,phase="CALIBRATION",view="FUSION_SUPERVISION",digest=sha256(artifacts["fusion_input_private"]))
            audit=read(artifacts["fusion_input_audit"])
            parent=audit.get("primary_label_binding")
            if parent:
                parent_path=self.bind(parent,path.parent,model,"fusion.primary_label_binding");native=read(parent_path)
                from .current_primary_fusion import FORMAT as PRIMARY_FORMAT,load_primary_fusion_inputs
                if native.get("format")==PRIMARY_FORMAT:
                    load_primary_fusion_inputs(parent_path,expected_sha=sha256(parent_path),expected_core_run_id=audit["core_run_id"])
                    child={"model_id":model,"producer_kind":"current_core_release","producer_binding":native["primary_source"],
                        "score_column":"association_membership_probability","run_id":audit["core_run_id"]}
                    self.producer(child,parent_path.parent,model)
                else:self.gap(model,"fusion.primary","PRIMARY_NATIVE_DEPENDENCIES_NOT_RECONSTRUCTED")
            else:self.gap(model,"fusion.primary","PRIMARY_NATIVE_BINDING_MISSING")
            for i,item in enumerate(binding["source_inputs"]):self.bind(item,path.parent,model,"fusion.source."+str(i))
            parameters=read(artifacts["checkpoint"])
            endpoint="confidence" if spec["score_column"]=="fused_confidence_probability" else "discovery"
            for expert in parameters[endpoint]["expert_ids"]:
                self.gap(model,"fusion.expert."+expert,"AUXILIARY_RAW_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED")
        else:
            self.gap(model,"producer."+kind,"LEGACY_NATIVE_INPUT_DEPENDENCIES_NOT_RECONSTRUCTED")


def derive_manifest(path):
    path=Path(path).resolve();manifest=read(path)
    if set(manifest)-{"format","models","artifact_locations"} or manifest.get("format")!=INPUT_FORMAT:
        raise ValueError("Invalid native model-use input")
    identities=model_identities(manifest["models"],path.parent)
    inventory=Inventory(manifest.get("artifact_locations",[]),path.parent)
    inventory.sources[str(path)]=record(path)
    for spec in manifest["models"]:inventory.producer(spec,path.parent,spec["model_id"])
    if set(inventory.locations)!=inventory.used_locations:raise ValueError("Unused model-use artifact locator is not a proven dependency")
    tables={"usage":pd.DataFrame(inventory.usage,columns=SOURCE_COLUMNS+EXTRA),
        "dependencies":pd.DataFrame(inventory.dependencies,columns=DEPENDENCIES).drop_duplicates().reset_index(drop=True),
        "gaps":pd.DataFrame(inventory.gaps,columns=GAPS).drop_duplicates().reset_index(drop=True)}
    for field in ("mapped","is_experimental","is_predicted"):tables["usage"][field]=tables["usage"][field].astype(bool)
    tables["usage"]["label"]=tables["usage"]["label"].astype(float)
    for source in inventory.sources.values():verified_source(source,path.parent)
    return manifest,identities,tables,list(inventory.sources.values())


def materialize_current_model_use(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Model-use preparation requires host 149")
    source=Path(manifest_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest":record(source)})
    _,models,tables,sources=derive_manifest(source);outputs={}
    for name,frame in tables.items():
        path=output/(name.upper()+".parquet");frame.to_parquet(path,index=False);outputs[name]={**record(path),"rows":len(frame)}
    binding={"format":FORMAT,"status":"PASS_DERIVED_INVENTORY","target_host":"149","input_manifest":record(source),
        "models":models,"outputs":outputs,"source_inputs":sources,"implementation_sha256":implementations(),
        "isolation_scope":SCOPE,"model_use_inventory_completeness_verified":False,"independent_external_validation_attested":False,
        "raw_study_provenance_reconstructed":False,"model_forwards_executed":False,"training_started":False}
    path=output/"MODEL_USE_BINDING.json";write(path,binding)
    audit=audit_current_model_use(path,expected_sha=sha256(path));ap=output/"RECOMPUTATION_AUDIT.json";write(ap,audit)
    result={"status":"PASS_DERIVED_INVENTORY","binding_path":str(path),"binding_sha256":sha256(path),
        "audit_sha256":sha256(ap),"training_started":False};write(output/"SUCCESS.json",result);return result


def load_current_model_use(path,*,expected_sha,models=None,root=None,replay=True):
    path=verified_path({"path":str(path),"sha256":expected_sha},Path.cwd());binding=read(path)
    if binding.get("format")!=FORMAT or binding.get("status")!="PASS_DERIVED_INVENTORY" or binding.get("target_host")!="149" or binding.get("isolation_scope")!=SCOPE:
        raise ValueError("Invalid native model-use release")
    for name in ("model_use_inventory_completeness_verified","independent_external_validation_attested","raw_study_provenance_reconstructed","model_forwards_executed","training_started"):
        if binding.get(name) is not False:raise ValueError("Native model-use inventory makes an unsupported completeness claim")
    if models is not None and binding["models"]!=model_identities(models,root or path.parent):raise ValueError("Model-use inventory belongs to different evaluated models")
    if binding["implementation_sha256"]!=implementations():raise ValueError("Model-use implementation changed")
    for source in binding["source_inputs"]:verified_source(source,path.parent)
    tables={name:pd.read_parquet(verified_path(item,path.parent)) for name,item in binding["outputs"].items()}
    if replay:
        audit_current_model_use(path,expected_sha=expected_sha)
        success=read(path.parent/"SUCCESS.json");ap=path.parent/"RECOMPUTATION_AUDIT.json"
        if success.get("binding_sha256")!=expected_sha or success.get("audit_sha256")!=sha256(ap):raise ValueError("Incomplete native model-use release")
    return binding,tables


def audit_current_model_use(path,*,expected_sha):
    if socket.gethostname()!="149":raise RuntimeError("Model-use replay requires host 149")
    path=Path(path).resolve();binding,tables=load_current_model_use(path,expected_sha=expected_sha,replay=False)
    _,models,expected,sources=derive_manifest(verified_path(binding["input_manifest"],path.parent))
    if models!=binding["models"] or sources!=binding["source_inputs"] or set(tables)!=set(expected):raise ValueError("Native model-use provenance drift")
    for name,frame in expected.items():
        try:pd.testing.assert_frame_equal(tables[name],frame,check_dtype=False,check_exact=True)
        except AssertionError as error:raise ValueError("Native model-use recomputation differs: "+name) from error
        if len(frame)!=binding["outputs"][name]["rows"]:raise ValueError("Native model-use row count differs")
    return {"status":"PASS","binding_sha256":expected_sha,"native_inputs_recomputed":True,"training_started":False}
