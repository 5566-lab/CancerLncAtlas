"""Inspect actual SC-HHGT checkpoints/episodes and bound upstream source files.

The native loader verifies graph construction and donor partitions. Inspection
does not construct a neural model, fit LASSO, optimize, or recompute predictions.
Canonical donor mappings are checked privately; public usage contains no donors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import sha256
from .evidence_rule_summary import stable_id
from .preparation_sources import metadata_tokens


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_signed_sources(inventory,native,inputs,outputs,model,role):
    import torch
    import scripts.train_single_cell_hhgt_149 as trainer
    from .current_auxiliary_model_use import input_source
    from .context_table_replay import relation
    from .drug_sparse_query import _bounded_duckdb
    pointer=read(inputs["pointer"])
    if not pointer.get("training_success"):
        inventory.gap(model,role,"SC_NATIVE_TRAINING_SUCCESS_NOT_BOUND");return

    def bind(item,root,label,fold=-1):return inventory.bind(item,root,model,role+"."+label,fold)
    success_path=bind(pointer["training_success"],inputs["pointer"].parent,"training_success")
    success=read(success_path);root=success_path.parent
    if (success.get("format")!=trainer.SUCCESS_FORMAT or success.get("state")!="TRAINING_COMPLETE"
            or success.get("target_host")!="149" or success.get("outer_folds")!=5
            or set(success.get("arms",[]))!=set(trainer.ARMS) or success.get("trained_model_count")!=10):
        raise ValueError("SC native training receipt is incomplete")
    files={k:bind(v,root,"training."+k) for k,v in success["files"].items()}
    if not {"run_config","preflight","input_lineage","checkpoint_manifest","oof_predictions","checksums"}.issubset(files):
        raise ValueError("SC training receipt lacks native files")
    if sha256(files["oof_predictions"])!=pointer["oof_predictions"]["sha256"]:
        raise ValueError("SC activated OOF differs from actual training output")
    validation_path=bind(pointer["independent_validation"],inputs["pointer"].parent,"validation_receipt")
    validation=read(validation_path)
    if (validation.get("format")!="CANCERLNCATLAS_V32_SC_HHGT_INDEPENDENT_VALIDATION_V1"
            or validation.get("status")!="PASS" or validation.get("training_success_sha256")!=sha256(success_path)
            or validation.get("checkpoint_manifest_sha256")!=sha256(files["checkpoint_manifest"])):
        raise ValueError("SC validation receipt belongs to another training run")
    config=read(files["run_config"]);lineage=read(files["input_lineage"])
    if (config.get("format")!=trainer.CONFIG_FORMAT or config.get("primary_arm")!=native["arm"]
            or lineage.get("format")!=trainer.LINEAGE_FORMAT or lineage.get("status")!="PASS"
            or lineage.get("target_host")!="149"):
        raise ValueError("SC training configuration or lineage differs from activation")
    for name,current in (("trainer_source",Path(trainer.__file__)),
                         ("model_source",Path(trainer.__file__).resolve().parents[1]/"cc_hhgt/v32/single_cell_hhgt.py")):
        if name not in lineage:
            inventory.gap(model,role+"."+name,"SC_NATIVE_IMPLEMENTATION_BINDING_ABSENT");continue
        original=bind(lineage[name],root,name)
        if sha256(original)!=sha256(current):
            inventory.gap(model,role+"."+name,"SC_NATIVE_CODE_VERSION_DIFFERS_FROM_INSPECTOR",digest=sha256(original))
    episode_path=bind(lineage["episode_manifest"],root,"episode_manifest")
    if pointer.get("input_manifest"):
        if pointer["input_manifest"]["sha256"]!=sha256(episode_path):
            raise ValueError("SC activated episode manifest differs from training")
        bind(pointer["input_manifest"],inputs["pointer"].parent,"activated_episode_manifest")
    graph_path=bind(lineage["graph_asset_manifest"],root,"graph_manifest")
    descriptors,observed=trainer.load_episode_manifest(trainer.TrainingSettings(
        episode_root=episode_path.parent,graph_assets_root=graph_path.parent,output_root=root))
    for name in ("episode_manifest","graph_asset_manifest","episode_assets","shared_static_assets"):
        if lineage.get(name)!=observed[name]:raise ValueError("SC training input inventory differs: "+name)

    manifest=read(files["checkpoint_manifest"])
    records=manifest.get("models",[])
    if (manifest.get("format")!=trainer.CHECKPOINT_MANIFEST_FORMAT or manifest.get("status")!="SUCCESS"
            or len(records)!=10 or {(r.get("arm"),r.get("outer_fold_id")) for r in records}!={(a,f) for a in trainer.ARMS for f in range(5)}):
        raise ValueError("SC native checkpoint manifest lacks both complete five-fold arms")
    selected={};prediction_paths=[];all_audits={}
    for item in records:
        arm=item["arm"];fold=item["outer_fold_id"];label="model."+arm
        fold_success=read(bind(item["success"],root,label+".success",fold))
        if (fold_success.get("state")!="FOLD_TRAINING_COMPLETE" or fold_success.get("format")!=trainer.SUCCESS_FORMAT
                or fold_success.get("arm")!=arm or fold_success.get("outer_fold_id")!=fold or fold_success.get("files")!=item["files"]):
            raise ValueError("SC per-fold training receipt differs from manifest")
        paths={k:bind(v,root,label+"."+k,fold) for k,v in item["files"].items()}
        checkpoint=torch.load(paths["checkpoint"],map_location="cpu",weights_only=True)
        if checkpoint.get("format")!=trainer.CHECKPOINT_FORMAT:
            raise ValueError("Unsupported SC native checkpoint")
        for key in ("arm","outer_fold_id","seed","model_config","continuous_feature_dims","edge_schema",
                    "best_epoch","best_validation_loss","initial_state_sha256","final_state_sha256",
                    "initial_hgt_state_sha256","final_hgt_state_sha256"):
            if checkpoint.get(key)!=item.get(key):raise ValueError("SC checkpoint/manifest identity differs: "+key)
        state=checkpoint.get("state_dict",{})
        if not state or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
            raise ValueError("SC checkpoint has invalid state tensors")
        if (trainer._state_hash(state)!=item["final_state_sha256"]
                or trainer._state_hash(state,"encoder.layers.")!=item["final_hgt_state_sha256"]):
            raise ValueError("SC checkpoint state digest differs")
        report=read(paths["report"])
        if (report.get("arm")!=arm or report.get("outer_fold_id")!=fold or report.get("status")!="PASS"
                or report.get("model_selection_source")!="inner_validation_only"
                or report.get("outer_query_used_for_model_selection") is not False):
            raise ValueError("SC training report changed its selection/OOF policy")
        if arm==native["arm"]:selected[fold]=(paths["predictions"],report)
        audit_rows=report.get("episode_audits",[])
        audits={r["episode_id"]:r for r in audit_rows}
        if len(audits)!=len(audit_rows) or set(audits)!={d.episode_id for d in descriptors[fold]}:
            raise ValueError("SC training report omits or duplicates consumed episodes")
        all_audits[arm,fold]=audits
        prediction_paths.append(paths["predictions"])

    # Reconcile all native fold outputs with the activated aggregate. This
    # verifies serialization and identity, not neural-forward correctness.
    with _bounded_duckdb() as con:
        keys=["arm","outer_fold_id","row_uid"]
        columns=keys+["episode_id",*trainer.ROW_ID_COLUMNS,"rho","predicted_rho"]
        projection=','.join(columns)
        con.execute("CREATE VIEW combined AS "+" UNION ALL ".join(f"SELECT {projection} FROM {relation(p)}" for p in prediction_paths))
        con.execute(f"CREATE VIEW activated AS SELECT {projection} FROM {relation(files['oof_predictions'])}")
        for name in ("combined","activated"):
            if con.execute(f"SELECT count(*) FROM (SELECT {','.join(keys)} FROM {name} GROUP BY ALL HAVING count(*)<>1)").fetchone()[0]:
                raise ValueError("SC duplicate native OOF row identity")
        if con.execute("SELECT count(*) FROM ((SELECT * FROM combined EXCEPT ALL SELECT * FROM activated) UNION ALL (SELECT * FROM activated EXCEPT ALL SELECT * FROM combined))").fetchone()[0]:
            raise ValueError("SC activated predictions differ from native fold files")

    visited=set()
    def walk(record,base,label,fold=-1,depth=0):
        if depth>80:raise ValueError("SC source receipt graph exceeds traversal depth")
        path=bind(record,base,label,fold)
        identity=(str(path),record["sha256"])
        if identity in visited:return
        visited.add(identity)
        if path.suffix==".json":
            document=read(path)
            def visit(value,prefix):
                if isinstance(value,dict):
                    if "path" in value and "sha256" in value:
                        walk(value,path.parent,prefix,fold,depth+1);return
                    # Upstream mappings can use donor IDs as keys. Public
                    # dependency roles retain structural indices, not raw keys.
                    for number,(key,item) in enumerate(value.items()):
                        child=prefix+".field."+str(number)
                        if key.endswith("_path") and isinstance(item,str):
                            digest=value.get(key[:-5]+"_sha256")
                            if digest:walk({"path":item,"sha256":digest},path.parent,child,fold,depth+1)
                            elif item:inventory.gap(model,role+"."+child,"SC_UPSTREAM_PATH_HAS_NO_BOUND_HASH",fold)
                        elif isinstance(item,(dict,list)):visit(item,child)
                elif isinstance(value,list):
                    for i,item in enumerate(value):visit(item,prefix+"."+str(i))
            visit(document,label)
        elif path.suffix in {".parquet",".tsv",".csv"}:
            input_source(inventory,record,base,model,role+"."+label)
        elif path.suffix not in {".pt",".py",".R"}:
            inventory.gap(model,role+"."+label,"SC_RAW_ARRAY_TRANSFORM_ANCESTRY_NOT_RECONSTRUCTED",fold,record["sha256"])
    walk(lineage["episode_manifest"],root,"upstream.episodes")
    walk(lineage["graph_asset_manifest"],root,"upstream.graph")

    crosswalk=pd.read_parquet(inputs["donor_crosswalk"])
    donor_sets=crosswalk.groupby("query_donor_id").canonical_donor_id.agg(lambda x:set(map(str,x)))
    canonical={str(k):next(iter(v)) for k,v in donor_sets.items() if len(v)==1}
    outer_folds={}
    for fold,episodes in sorted(descriptors.items()):
        prediction_path,report=selected[fold]
        for descriptor in episodes:
            bind(descriptor.record,descriptor.path.parent,"episode."+descriptor.role,fold)
            loaded=trainer.load_episode(descriptor,native["arm"])
            for arm in trainer.ARMS:
                checked=loaded if arm==native["arm"] else trainer.load_episode(descriptor,arm)
                if checked.audit(arm)!=all_audits[arm,fold][descriptor.episode_id]:
                    raise ValueError("SC episode input differs from its training audit")
                del checked
            payload=torch.load(descriptor.path,map_location="cpu",weights_only=True)
            static=torch.load(descriptor.static_path,map_location="cpu",weights_only=True)
            walk(descriptor.static_record,descriptor.static_path.parent,"upstream.static",fold)
            for key,item in (("matrix",descriptor.matrix_identity_record),("graph_static",descriptor.graph_asset_static_record)):
                walk(item,descriptor.static_path.parent,"upstream."+key,fold)
            for i,item in enumerate(payload.get("source_receipts",[])):
                walk(item,descriptor.path.parent,"upstream.episode_source."+str(i),fold)
            if not payload.get("source_receipts"):
                inventory.gap(model,role+".episode","SC_EPISODE_UPSTREAM_RECEIPTS_ABSENT",fold,descriptor.record["sha256"])
            for batch in static.get("external_edge_batches",[]):
                digest=batch.get("source_hash","")
                if len(digest)==64:
                    inventory.authority({"path":batch.get("provenance",""),"sha256":digest},descriptor.static_path.parent,
                        model,role+".external_graph_source",fold)
            phase={"fit":"TRAINING","validation":"MODEL_SELECTION","query":"FEATURE_CONSTRUCTION"}[descriptor.role]
            inventory.patient_use.observe([loaded.rows[["query_donor_id"]]],digest=descriptor.record["sha256"],
                model=model,context=inventory.context_id,role="SC_HHGT_EPISODE_TARGET_DONORS",fold=fold,phase=phase,
                scope="ACTUAL_NATIVE_EPISODE_TARGET_ROWS")
            inventory.patient_use.observe([pd.DataFrame({"source_donors":list(loaded.source_donors)})],
                digest=descriptor.record["sha256"],model=model,context=inventory.context_id,
                role="SC_HHGT_EPISODE_FEATURE_DONORS",fold=fold,scope="ACTUAL_NATIVE_EPISODE_FEATURE_DONORS")
            inventory.entities(loaded.rows,model=model,fold=fold,phase=phase,view="SC_HHGT_"+descriptor.role.upper(),digest=descriptor.record["sha256"])
            ids=static.get("node_identifiers",{}).get("lncRNA")
            if ids is not None:
                positions=payload["node_global_indices"]["lncRNA"].tolist()
                if len(ids)!=static["node_counts"]["lncRNA"] or len(set(ids))!=len(ids):raise ValueError("SC graph node identities differ")
                nodes=pd.DataFrame({"lncrna_id":[ids[i] for i in positions]})
                inventory.entities(nodes,model=model,fold=fold,phase="FEATURE_CONSTRUCTION",view="SC_HHGT_GRAPH_NODES",digest=descriptor.record["sha256"])
            else:inventory.gap(model,role+".graph","SC_STATIC_NODE_IDENTIFIERS_ABSENT",fold)
            tokens=set()
            for dataset in loaded.rows.dataset_id.unique():tokens.update(metadata_tokens({"dataset_id":dataset}))
            identity=stable_id("",[model,inventory.context_id,descriptor.record["sha256"],phase])
            inventory.usage.append({"cancer_id":"","lncrna_id":"","pathway_id":"","source_id":"SC-EPISODE:"+identity,
                "source_event_id":identity,"source_file_sha256":descriptor.record["sha256"],"phase":phase,"endpoint":"MODEL_USE",
                "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,"assay":"NATIVE_SC_EPISODE_METADATA",
                "biosample":"","study_tokens_json":json.dumps(sorted(tokens)),"raw_record_id":identity,
                "raw_row_sha256":stable_id("",[sorted(tokens),descriptor.role]),"raw_identifiers_json":"{}",
                "model_id":model,"patient_fold_id":fold,"use_view":"SC_HHGT_EPISODE_SOURCE_METADATA","candidate_rows":0,
                "entity_scope":"WHOLE_EPISODE_NO_ENTITY_ASSERTION","dependency_context":inventory.context_id})
            sets={key:set(getattr(loaded,key)) for key in ("outer_query_donors","outer_train_donors","source_donors","excluded_donors")}
            if any(not values.issubset(canonical) for values in sets.values()):
                inventory.gap(model,role+".donors","SC_TRAINING_DONORS_NOT_FULLY_CANONICALIZED",fold)
            mapped={key:{canonical[v] for v in values if v in canonical} for key,values in sets.items()}
            if mapped["outer_query_donors"] & mapped["outer_train_donors"] or mapped["source_donors"] & mapped["excluded_donors"]:
                raise ValueError("SC canonical donor aliases cross a native training partition")
            for donor in mapped["outer_query_donors"]:
                if donor in outer_folds and outer_folds[donor]!=fold:raise ValueError("SC canonical donor crosses outer folds")
                outer_folds[donor]=fold
            if descriptor.role=="query":
                with _bounded_duckdb() as con:
                    con.register("expected_rows",loaded.rows)
                    columns=','.join([*trainer.ROW_ID_COLUMNS,"row_uid","rho"])
                    con.execute(f"CREATE TEMP TABLE actual_rows AS SELECT {columns} FROM {relation(prediction_path)} WHERE episode_id=?",[descriptor.episode_id])
                    if con.execute(f"SELECT count(*) FROM ((SELECT {columns} FROM expected_rows EXCEPT ALL SELECT * FROM actual_rows) UNION ALL (SELECT * FROM actual_rows EXCEPT ALL SELECT {columns} FROM expected_rows))").fetchone()[0]:
                        raise ValueError("SC native query rows differ from their consumed episode")
            del payload,static,loaded
    inventory.gap(model,role,"SC_RAW_STUDY_AND_CROSS_MODULE_PATIENT_ANCESTRY_NOT_COMPLETE")
