"""Derive model input exposure from native selected/core/fusion producers.

This inventory inspects authorized tensors and dependencies without constructing
an optimizer or constructing neural models. Bound native L1 linear scores may be replayed. It records unresolved upstream study
provenance explicitly; file integrity is not proof of biological independence.
"""
from __future__ import annotations
import json
from pathlib import Path
import socket
from contextlib import contextmanager

import numpy as np
import pandas as pd

from .current_core_export import sha256,verified_path,validate_checkpoint
from .evidence_rule_summary import KEYS,stable_id
from .external_validation_sources import SOURCE_COLUMNS

INPUT_FORMAT="CURRENT_MODEL_USE_INPUT_V1"
FORMAT="CURRENT_MODEL_USE_RELEASE_V1"
SCOPE="NATIVE_REGISTERED_INPUTS_WITH_EXPLICIT_PROVENANCE_LIMITS"
EXTRA=["model_id","patient_fold_id","use_view","candidate_rows","entity_scope","dependency_context","usage_record_kind"]
DEPENDENCIES=["model_id","patient_fold_id","role","path","sha256","status","dependency_context"]
GAPS=["model_id","patient_fold_id","role","sha256","reason","dependency_context"]
REPLAYS=["model_id","dependency_context","patient_fold_id","role","format","source_release_sha256",
         "normalization_sha256","activity_sha256","folds_recomputed","outputs_compared","model_training_started"]


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
        "cc_hhgt/v32/native_graph_input.py",
        "scripts/export_v32_current_core_embeddings.py","cc_hhgt/v32/training.py",
        "cc_hhgt/v32/preparation_sources.py","scripts/prepare_v32_formal.py",
        "cc_hhgt/v32/prepared_construction.py","cc_hhgt/v32/baselines.py",
        "cc_hhgt/v32/g012_winner_lock.py","cc_hhgt/v32/selected_score.py",
        "cc_hhgt/v32/current_auxiliary_model_use.py","cc_hhgt/v32/native_auxiliary_fusion.py",
        "cc_hhgt/v32/native_partitioned_genomic.py","cc_hhgt/v32/genomic_partition_training.py",
        "cc_hhgt/v32/genomic_resource_staging.py",
        "cc_hhgt/v32/cnv_head_association.py", "cc_hhgt/v32/atac_cohort.py",
        "cc_hhgt/v32/atac_materialization.py", "cc_hhgt/v32/atac_training.py",
        "scripts/run_v32_current_multimodal_fusion.py",
        "cc_hhgt/v32/multimodal_fusion.py",
        "cc_hhgt/v32/current_single_cell_model_use.py","cc_hhgt/v32/single_cell_signed_fusion.py",
        "cc_hhgt/v32/current_core_model_use.py","cc_hhgt/v32/integrated_model.py",
        "cc_hhgt/v32/patient_source_identity.py",
        "cc_hhgt/v32/local_cnv_sources.py","cc_hhgt/v32/local_cnv_confounding.py","cc_hhgt/v32/local_cnv_cohort.py",
        "cc_hhgt/v32/formal_expression_sources.py","scripts/convert_formal_lncrna_rds.py",
        "cc_hhgt/v32/gdc_star_counts.py","scripts/assemble_gdc_star_counts.R","scripts/materialize_gdc_star_counts.py",
        "cc_hhgt/v32/fold_local_expression.py","scripts/frozen_tmm.R","scripts/materialize_fold_local_expression.R",
        "cc_hhgt/v32/fold_local_activity.py","scripts/frozen_ssgsea.R","scripts/materialize_fold_local_ssgsea.R",
        "scripts/convert_fold_local_activity.py",
        "cc_hhgt/v32/fold_local_scope.py","scripts/materialize_fold_local_scope.py","cc_hhgt/v32/prediction_scope.py","cc_hhgt/v32/fold_local_graph.py",
        "cc_hhgt/v32/formal_graph.py","cc_hhgt/v32/formal_graph_authority.py","cc_hhgt/v32/safe_graph.py",
        "scripts/materialize_formal_lncrna_logcpm.R",
        "cc_hhgt/v32/local_cnv_core_preparation.py","cc_hhgt/stats.py","cc_hhgt/v32/directional_cnv.py",
        "cc_hhgt/v32/cnv_input_sources.py","cc_hhgt/v32/current_cnv_model_use.py",
        "cc_hhgt/v32/cnv_only_training.py","cc_hhgt/v32/cnv_only_training_v2.py",
        "scripts/materialize_v32_directional_cnv.py","cc_hhgt/v32/segment_cnv_streaming.py",
        "scripts/train_single_cell_hhgt_149.py","cc_hhgt/v32/single_cell_hhgt.py",
        "cc_hhgt/v32/external_validation_current.py","cc_hhgt/v32/external_validation_sources.py",
        "cc_hhgt/v32/model_use_completeness.py","cc_hhgt/v32/normalization_source_replay.py",
        "cc_hhgt/v32/activity_source_replay.py","cc_hhgt/v32/model_control_sources.py",
        "cc_hhgt/v32/patient_fold_authority.py","cc_hhgt/v32/current_binding_overlay.py",
        "scripts/compare_fold_local_normalization.R"]
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
    def __init__(self,locations,root,patient_crosswalk=None):
        self.sources={};self.dependencies=[];self.gaps=[];self.usage=[];self.visited=set();self.locations={};self.used_locations=set()
        self.transform_replays=[]
        self.context_id="SELECTED_SCORE"
        from .patient_source_identity import IdentityMap,PatientUse
        self.patient_use=PatientUse(IdentityMap(patient_crosswalk,root))
        if patient_crosswalk is not None:
            path=self.patient_use.mapper.path
            self.sources[str(path)]={"path":str(path),"sha256":patient_crosswalk["sha256"]}
        for item in locations:
            path=verified_source(item,root);digest=item["sha256"]
            if digest in self.locations:raise ValueError("Duplicate model-use artifact location")
            self.locations[digest]={"path":str(path),"sha256":digest}

    @contextmanager
    def context(self,value):
        previous=self.context_id;self.context_id=value
        try:yield
        finally:self.context_id=previous

    def bind(self,item,root,model,role,fold=-1):
        path=verified_source(item,root);source={"path":str(path),"sha256":item["sha256"]};self.sources[str(path)]=source
        self.dependencies.append({"model_id":model,"patient_fold_id":fold,"role":role,**source,
            "status":"VERIFIED_DIRECTORY" if path.is_dir() else "VERIFIED_FILE","dependency_context":self.context_id})
        return path

    def gap(self,model,role,reason,fold=-1,digest=""):
        self.gaps.append({"model_id":model,"patient_fold_id":fold,"role":role,"sha256":digest,"reason":reason,"dependency_context":self.context_id})

    def authority(self,item,root,model,role,fold):
        digest=item.get("sha256","")
        if not digest:
            self.gap(model,role,"AUTHORITY_HAS_NO_FILE_HASH",fold);return
        source=self.locations.get(digest,item)
        if digest in self.locations:self.used_locations.add(digest)
        path=Path(source.get("path",""));path=path if path.is_absolute() else root/path
        if not source.get("path") or not (path.is_file() or path.is_dir()):
            self.dependencies.append({"model_id":model,"patient_fold_id":fold,"role":role,"path":str(path) if source.get("path") else "",
                "sha256":digest,"status":"UNRESOLVED_LOCATION","dependency_context":self.context_id})
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
        source_id="NATIVE-USE:"+stable_id("",[model,fold,phase,view,digest,self.context_id])
        for lnc,count in counts.items():
            ids={"lncrna":str(lnc),"cancer":"","pathway":""}
            identity=stable_id("",[source_id,str(lnc)])
            self.usage.append({"cancer_id":"","lncrna_id":str(lnc),"pathway_id":"","source_id":source_id,
                "source_event_id":"MODEL-INPUT:"+identity,"source_file_sha256":digest,"phase":phase,"endpoint":"MODEL_USE",
                "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,
                "assay":"NATIVE_MODEL_INPUT_EXPOSURE","biosample":"","study_tokens_json":"[]",
                "raw_record_id":identity,"raw_row_sha256":stable_id("",[ids,int(count)]),"raw_identifiers_json":json.dumps(ids,sort_keys=True),
                "model_id":model,"patient_fold_id":fold,"use_view":view,"candidate_rows":int(count),
                "entity_scope":"LNCRNA_EXPOSURE_ALL_PATHWAYS","dependency_context":self.context_id,
                "usage_record_kind":"ENTITY_EXPOSURE"})

    def preparation(self,payload,root,model,fold,baseline):
        from .preparation_sources import validate_payload_sources,canonical_sha,source_members
        receipt=validate_payload_sources(payload)
        if receipt is None:return set()
        known={canonical_sha(receipt)}
        for item in receipt["source_reads"]:
            role=item["role"];digest=item["sha256"];known.add(digest)
            # Validation metrics were diagnostic; fixed-C L1 was fitted only
            # on train rows. They are not dependencies of its numeric score.
            if baseline and role.startswith("validation_"):continue
            source=self.locations.get(digest,item)
            if digest in self.locations:self.used_locations.add(digest)
            path=Path(source["path"]);path=path if path.is_absolute() else root/path
            label="prepared.sources."+role+("."+item["cancer_id"] if item["cancer_id"] else "")
            if not path.exists():
                self.authority(source,root,model,label,fold);continue
            path=self.bind(source,root,model,label,fold)
            observed=source_members(path,item["kind"])
            # Compare the complete actual file list, not a subset declared by
            # the receipt. Relocation preserves relative partition names.
            original=Path(item["path"])
            expected=[]
            for member in item["members"]:
                relocated=path/Path(member["path"]).relative_to(original) if path.is_dir() else path
                expected.append({**member,"path":str(relocated)})
            if observed!=expected:raise ValueError("Preparation consumed-file metadata differs: "+role)
            from .current_auxiliary_model_use import inherited_tokens
            start=len(self.usage)
            if item.get("upstream_transformation"):
                from .local_cnv_sources import inspect
                inspect(self,item["upstream_transformation"],root,model,label+".transformation",fold)
            item_tokens=inherited_tokens(self,start)
            for member in observed:
                known.add(member["sha256"])
                self.bind(member,path.parent,model,label+".consumed_file",fold)
                tokens=member["study_tokens"]
                if item["kind"]=="control":continue
                start=len(self.usage)
                if member.get("upstream_expression"):
                    from .formal_expression_sources import inspect
                    inspect(self,member["upstream_expression"],path.parent,member["path"],model,label+".expression",fold)
                if member.get("upstream_activity"):
                    from .fold_local_activity import inspect
                    inspect(self,member["upstream_activity"],path.parent,member["path"],model,label+".activity",fold)
                from .fold_local_scope import inspect_scope, inspect_candidates
                if member.get("upstream_scope"):
                    inspect_scope(self,member["upstream_scope"],path.parent,member["path"],model,label+".detection",fold)
                if member.get("upstream_candidates"):
                    inspect_candidates(self,member["upstream_candidates"],path.parent,member["path"],model,label+".candidates",fold)
                upstream_tokens=sorted(set(item_tokens)|set(inherited_tokens(self,start)))
                tokens=sorted(set(tokens)|set(upstream_tokens))
                self.patient_use.table(member["path"],digest=member["sha256"],model=model,context=self.context_id,
                    role=label,fold=fold,phase="MODEL_SELECTION" if role=="validation_associations" else "FEATURE_CONSTRUCTION",
                    scope="WHOLE_CONSUMED_FILE_NOT_ALL_ROWS_FITTED")
                if not tokens:self.gap(model,label,"SOURCE_STUDY_METADATA_ABSENT",fold,member["sha256"])
                phases=["MODEL_SELECTION"] if role=="validation_associations" else ["FEATURE_CONSTRUCTION"]
                if role in {"train_associations","expression","activity"}:phases.append("TRAINING")
                if not baseline and role in {"expression","activity"}:phases.append("MODEL_SELECTION")
                for phase in phases:
                    identity=stable_id("",[model,fold,label,phase,member["sha256"],self.context_id])
                    self.usage.append({"cancer_id":"","lncrna_id":"","pathway_id":"",
                        "source_id":"PREPARATION-SOURCE:"+identity,"source_event_id":identity,
                        "source_file_sha256":member["sha256"],"phase":phase,"endpoint":"MODEL_USE",
                        "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,
                        "assay":"VERIFIED_UPSTREAM_STUDY_METADATA" if upstream_tokens else "CONSUMED_SOURCE_METADATA","biosample":"","study_tokens_json":json.dumps(tokens),
                        "raw_record_id":identity,"raw_row_sha256":canonical_sha(member),"raw_identifiers_json":"{}",
                        "model_id":model,"patient_fold_id":fold,"use_view":"PREPARATION_SOURCE_METADATA",
                        "candidate_rows":0,"entity_scope":"WHOLE_CONSUMED_FILE_NO_ENTITY_ASSERTION","dependency_context":self.context_id})
            if (not item.get("upstream_transformation") and item["kind"]=="table"
                    and not all(any(m.get(k) for k in ("upstream_expression","upstream_activity","upstream_scope","upstream_candidates")) for m in observed)):
                self.gap(model,label,"RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED",fold,digest)
        return known

    def core(self,path,binding,model,score_column):
        from .current_core_prediction import candidate_table
        source=self.bind(binding["input_manifest"],path.parent,model,"core_prediction_input")
        manifest=read(source);root=source.parent;candidates=candidate_table(manifest,root)
        from .prediction_scope import manifest_roles
        for role in manifest_roles(manifest):self.bind(manifest[role],root,model,role)
        self.core_inputs(manifest,root,candidates,model,score_column)

    def core_inputs(self,manifest,root,candidates,model,score_column,*,comparison=False,exported_folds=None):
        import torch
        from .current_core_prediction import prediction_batches
        from .current_core_export import CHECKPOINT_FORMAT,validate_training_receipt
        from scripts.export_v32_current_core_embeddings import _validated_payload
        for item in sorted(manifest["fold_inputs"],key=lambda r:r["fold"]):
            fold=item["fold"]
            roles={"config","input_manifest","prepared","checkpoint","training_receipt"}
            roles|={r for r in ("graph_overlay","global_binding_source","binding_expansion") if r in item}
            paths={r:self.bind(item[r],root,model,r,fold) for r in sorted(roles)}
            receipt=read(paths["training_receipt"])
            checkpoint=torch.load(paths["checkpoint"],map_location="cpu",weights_only=False)
            if comparison and checkpoint.get("checkpoint_format")!=CHECKPOINT_FORMAT:
                self.gap(model,"comparison.checkpoint","UNSUPPORTED_COMPARISON_CHECKPOINT_FORMAT",fold,item["checkpoint"]["sha256"])
                continue
            validate_training_receipt(receipt,run_id=manifest["core_run_id"],fold=fold,
                checkpoint_sha=item["checkpoint"]["sha256"],variant=manifest["graph_variant"])
            validate_checkpoint(checkpoint,receipt,fold=fold,prepared_sha=item["prepared"]["sha256"],
                config_sha=item["config"]["sha256"],input_manifest_sha=item["input_manifest"]["sha256"])
            payload=_validated_payload(paths,item,fold)
            if payload["formal_graph_variant"]!=manifest["graph_variant"]:
                raise ValueError("Model-use prepared graph differs from the selected training arm")
            from .current_core_model_use import prepared_candidates,check_export_payload,graph_sources
            fold_candidates=prepared_candidates(payload) if candidates is None else candidates
            if exported_folds is not None:
                check_export_payload(self,exported_folds[str(fold)],root,payload,checkpoint,model,fold)
            # Reuse the exact candidate alignment and validation-only input firewall.
            features=prediction_batches(payload,fold_candidates)
            if "candidate_authority" in manifest:
                from .prediction_scope import validate_prepared_scope
                validate_prepared_scope(manifest,root,payload,fold_candidates,features,fold)
            digest=item["prepared"]["sha256"]
            self.entities(fold_candidates,model=model,fold=fold,phase="FEATURE_CONSTRUCTION",view="CANDIDATE_FEATURES",digest=digest)
            known=self.preparation(payload,paths["prepared"].parent,model,fold,score_column=="l1_probability")
            from .prepared_construction import replay as replay_construction
            construction = replay_construction(payload, paths["prepared"].parent, inventory=self, model=model,
                                                baseline_only=score_column=="l1_probability")
            if construction is not None:
                known.add(construction["construction_sha256"])
                self.transform_replays.append({"model_id":model,"dependency_context":self.context_id,
                    "patient_fold_id":fold,"role":"prepared.feature_and_label_construction",
                    "format":construction["format"],"source_release_sha256":construction["construction_sha256"],
                    "normalization_sha256":"","activity_sha256":"","folds_recomputed":1,
                    "outputs_compared":construction["tensor_fields_compared"],"model_training_started":False})
            if score_column=="l1_probability":
                # The published baseline is sigmoid(base_logit). The reference
                # neural head, graph and its checkpoint selection did not fit
                # those baseline values, even though they share a container.
                if known:
                    indices=[]
                    for batch,feature in zip(payload["train_batches"],features):
                        mask=batch.get("association_available",torch.ones(len(feature["candidate_positions"]),dtype=torch.bool))
                        if not torch.is_tensor(mask) or mask.dtype!=torch.bool or tuple(mask.shape)!=(len(feature["candidate_positions"]),):
                            raise ValueError("Baseline model-use supervision mask differs")
                        indices.extend(feature["candidate_positions"][mask.cpu().numpy()].tolist())
                    self.entities(fold_candidates.iloc[indices],model=model,fold=fold,phase="TRAINING",view="BASELINE_AVAILABLE_SUPERVISION",digest=digest)
                else:self.gap(model,"baseline.preparation","BASELINE_FITTING_STUDY_DEPENDENCIES_NOT_RECONSTRUCTED",fold,digest)
                del payload,checkpoint,features
                continue
            from .model_control_sources import inspect_patients
            known.update(inspect_patients(self,payload["patient_fold_authority"],paths["prepared"].parent,model,fold))
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
                self.entities(fold_candidates.iloc[indices],model=model,fold=fold,phase=phase,view="AVAILABLE_ASSOCIATION_SUPERVISION",digest=digest)
            graph=payload["formal_graph_authority"]
            known.update(graph_sources(self,graph,paths["prepared"].parent,model,fold,
                expansion_source=({**item["binding_expansion"],"path":str(paths["binding_expansion"])} if "binding_expansion" in item else None),
                verified_expansion=payload.get("binding_expansion_authority")))
            known.update(r["sha256"] for r in item.values() if isinstance(r,dict) and "sha256" in r)
            for role,digest in receipt["input_authority_hashes"].items():
                if isinstance(digest,str) and len(digest)==64 and digest not in known:
                    self.gap(model,"prepared."+role,"UNRESOLVED_INPUT_AUTHORITY_DIGEST",fold,digest)
            if construction is None:
                self.gap(model,"prepared.feature_and_label_construction","RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED",fold,item["prepared"]["sha256"])
            del payload,checkpoint,features

    def comparison(self,path,model):
        from .g012_winner_lock import validate_comparison_sources
        from .evidence_rule_summary import candidate_aliases
        lock=read(path);runs=validate_comparison_sources(lock,path.parent)
        if runs is None:
            self.gap(model,"model_selection_lock","ALL_COMPETITOR_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED")
            return
        candidate=self.bind(lock["candidate_authority"],path.parent,model,"comparison.candidate_authority")
        candidates=pd.read_parquet(candidate);candidate_aliases(candidates)
        candidates=candidates.sort_values(KEYS,kind="stable").reset_index(drop=True)
        for variant,run in sorted(runs.items()):
            with self.context("G012_COMPARISON:"+variant):
                manifests={k:self.bind(run[k],path.parent,model,"comparison."+k) for k in ("config","input_manifest","task_manifest")}
                inputs={r["fold"]:r for r in read(manifests["input_manifest"])["fold_inputs"]}
                prepared={r["patient_fold"]:r for r in run["prepared_folds"]}
                records=[];run_ids=set()
                for checkpoint in sorted(run["checkpoints"],key=lambda r:r["patient_fold"]):
                    fold=checkpoint["patient_fold"]
                    receipt=self.bind(checkpoint["training_receipt"],path.parent,model,"comparison.training_receipt",fold)
                    run_ids.add(read(receipt).get("run_id"))
                    item={"fold":fold,"prepared":prepared[fold],"checkpoint":checkpoint,
                          "training_receipt":record(receipt),"config":record(manifests["config"]),
                          "input_manifest":record(manifests["input_manifest"])}
                    for role in ("graph_overlay","global_binding_source","binding_expansion"):
                        if not inputs[fold].get(role):continue
                        declared=inputs[fold][role];raw=Path(declared["path"])
                        raw=raw if raw.is_absolute() else manifests["input_manifest"].parent/raw
                        observed=record(raw)
                        size_only=role=="graph_overlay" and declared.get("verification_mode")=="SIZE_AND_GRAPH_SEMANTICS_V1"
                        if (size_only and raw.stat().st_size!=declared.get("bytes")) or (not size_only and observed["sha256"]!=declared.get("sha256")):
                            raise ValueError("G012 comparison sidecar differs from training authority")
                        item[role]=observed
                    # Resolve top-level records relative to the lock, then use
                    # those exact files in the same native loader as prediction.
                    for role in ("prepared","checkpoint"):
                        item[role]=record(self.bind(item[role],path.parent,model,"comparison."+role,fold))
                    records.append(item)
                if len(run_ids)!=1 or None in run_ids:
                    self.gap(model,"comparison.run","COMPARISON_NATIVE_RUN_ID_UNRESOLVED")
                    continue
                self.core_inputs({"core_run_id":next(iter(run_ids)),"graph_variant":variant,"fold_inputs":records},
                    path.parent,candidates,model,"association_membership_probability",comparison=True)

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
                self.comparison(self.bind(selection["binding"],source.parent,model,"model_selection_lock"),model)
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
            from .current_auxiliary_model_use import inspect_fusion_expert
            for expert in parameters[endpoint]["expert_ids"]:
                with self.context("FUSION_EXPERT:"+expert):
                    inspect_fusion_expert(self,path,binding,audit,frame,expert,model,endpoint)
        else:
            self.gap(model,"producer."+kind,"LEGACY_NATIVE_INPUT_DEPENDENCIES_NOT_RECONSTRUCTED")


def inventory_tables(inventory,models):
    """Typed records and computed coverage; no caller-supplied completeness flag."""
    from .model_use_completeness import assess,STUDY
    tables={"usage":pd.DataFrame(inventory.usage,columns=SOURCE_COLUMNS+EXTRA),
        "dependencies":pd.DataFrame(inventory.dependencies,columns=DEPENDENCIES).drop_duplicates().reset_index(drop=True),
        "gaps":pd.DataFrame(inventory.gaps,columns=GAPS).drop_duplicates().reset_index(drop=True)}
    tables.update(inventory.patient_use.tables())
    tables["transformation_replays"]=pd.DataFrame(inventory.transform_replays,columns=REPLAYS).drop_duplicates().reset_index(drop=True)
    tables["usage"]["usage_record_kind"]=tables["usage"].usage_record_kind.fillna(STUDY)
    for field in ("mapped","is_experimental","is_predicted"):tables["usage"][field]=tables["usage"][field].astype(bool)
    tables["usage"]["label"]=tables["usage"]["label"].astype(float)
    tables["completeness"]=assess(models,tables)
    return tables


def derive_manifest(path):
    path=Path(path).resolve();manifest=read(path)
    if set(manifest)-{"format","models","artifact_locations","patient_identity_crosswalk"} or manifest.get("format")!=INPUT_FORMAT:
        raise ValueError("Invalid native model-use input")
    identities=model_identities(manifest["models"],path.parent)
    inventory=Inventory(manifest.get("artifact_locations",[]),path.parent,manifest.get("patient_identity_crosswalk"))
    inventory.sources[str(path)]=record(path)
    for spec in manifest["models"]:inventory.producer(spec,path.parent,spec["model_id"])
    if set(inventory.locations)!=inventory.used_locations:raise ValueError("Unused model-use artifact locator is not a proven dependency")
    tables=inventory_tables(inventory,identities)
    for source in inventory.sources.values():verified_source(source,path.parent)
    return manifest,identities,tables,list(inventory.sources.values())


def linear_replay_executed(tables):
    from .prepared_construction import FORMAT as CONSTRUCTION_FORMAT
    return bool(tables["transformation_replays"].format.eq(CONSTRUCTION_FORMAT).any())


def materialize_current_model_use(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Model-use preparation requires host 149")
    source=Path(manifest_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest":record(source)})
    manifest,models,tables,sources=derive_manifest(source);outputs={}
    for name,frame in tables.items():
        path=output/(name.upper()+".parquet");frame.to_parquet(path,index=False);outputs[name]={**record(path),"rows":len(frame),
            "private":name=="patient_usage_private"}
    from .model_use_completeness import is_complete
    binding={"format":FORMAT,"status":"PASS_DERIVED_INVENTORY","target_host":"149","input_manifest":record(source),
        "models":models,"outputs":outputs,"source_inputs":sources,"implementation_sha256":implementations(),
        "isolation_scope":SCOPE,"model_use_inventory_completeness_verified":is_complete(tables["completeness"]),"independent_external_validation_attested":False,
        "raw_study_provenance_reconstructed":False,"model_forwards_executed":linear_replay_executed(tables),
        "neural_model_forwards_executed":False,"training_started":False}
    binding["patient_identity_namespace"]=manifest.get("patient_identity_crosswalk",{}).get("namespace")
    path=output/"MODEL_USE_BINDING.json";write(path,binding)
    audit=audit_current_model_use(path,expected_sha=sha256(path));ap=output/"RECOMPUTATION_AUDIT.json";write(ap,audit)
    result={"status":"PASS_DERIVED_INVENTORY","binding_path":str(path),"binding_sha256":sha256(path),
        "audit_sha256":sha256(ap),"training_started":False};write(output/"SUCCESS.json",result);return result


def load_current_model_use(path,*,expected_sha,models=None,root=None,replay=True):
    path=verified_path({"path":str(path),"sha256":expected_sha},Path.cwd());binding=read(path)
    if binding.get("format")!=FORMAT or binding.get("status")!="PASS_DERIVED_INVENTORY" or binding.get("target_host")!="149" or binding.get("isolation_scope")!=SCOPE:
        raise ValueError("Invalid native model-use release")
    for name in ("independent_external_validation_attested","raw_study_provenance_reconstructed","training_started"):
        if binding.get(name) is not False:raise ValueError("Native model-use inventory makes an unsupported completeness claim")
    if models is not None and binding["models"]!=model_identities(models,root or path.parent):raise ValueError("Model-use inventory belongs to different evaluated models")
    if binding["implementation_sha256"]!=implementations():raise ValueError("Model-use implementation changed")
    if binding["outputs"].get("patient_usage_private",{}).get("private") is not True:
        raise ValueError("Patient identity usage must remain private")
    for source in binding["source_inputs"]:verified_source(source,path.parent)
    tables={name:pd.read_parquet(verified_path(item,path.parent)) for name,item in binding["outputs"].items()}
    if binding.get("model_forwards_executed") is not linear_replay_executed(tables):
        raise ValueError("Native model-use linear replay execution declaration differs")
    if binding.get("neural_model_forwards_executed", False) is not False:
        raise ValueError("Native model-use inspection does not run neural forwards")
    from .model_use_completeness import assess,is_complete
    coverage=assess(binding["models"],tables)
    if binding.get("model_use_inventory_completeness_verified") is not is_complete(coverage):
        raise ValueError("Native model-use inventory makes an unsupported completeness claim")
    try:pd.testing.assert_frame_equal(tables["completeness"],coverage,check_dtype=False,check_exact=True)
    except (KeyError,AssertionError) as error:raise ValueError("Native model-use recomputation differs: completeness") from error
    if replay:
        audit_current_model_use(path,expected_sha=expected_sha)
        success=read(path.parent/"SUCCESS.json");ap=path.parent/"RECOMPUTATION_AUDIT.json"
        if success.get("binding_sha256")!=expected_sha or success.get("audit_sha256")!=sha256(ap):raise ValueError("Incomplete native model-use release")
    return binding,tables


def audit_current_model_use(path,*,expected_sha):
    if socket.gethostname()!="149":raise RuntimeError("Model-use replay requires host 149")
    path=Path(path).resolve();binding,tables=load_current_model_use(path,expected_sha=expected_sha,replay=False)
    manifest,models,expected,sources=derive_manifest(verified_path(binding["input_manifest"],path.parent))
    if binding.get("patient_identity_namespace")!=manifest.get("patient_identity_crosswalk",{}).get("namespace"):
        raise ValueError("Native patient identity namespace differs from input")
    if models!=binding["models"] or sources!=binding["source_inputs"] or set(tables)!=set(expected):raise ValueError("Native model-use provenance drift")
    for name,frame in expected.items():
        try:pd.testing.assert_frame_equal(tables[name],frame,check_dtype=False,check_exact=True)
        except AssertionError as error:raise ValueError("Native model-use recomputation differs: "+name) from error
        if len(frame)!=binding["outputs"][name]["rows"]:raise ValueError("Native model-use row count differs")
    return {"status":"PASS","binding_sha256":expected_sha,"native_inputs_recomputed":True,"training_started":False}
