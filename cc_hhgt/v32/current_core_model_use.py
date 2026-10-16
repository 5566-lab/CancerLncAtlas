"""Inspect native frozen-core exports back to their actual training inputs.

No model is constructed. State hashes, node identities, chunk contracts and
native prepared loaders are checked; exported embeddings are not recomputed.
"""
from pathlib import Path
import json
from types import SimpleNamespace

import pandas as pd


def prepared_candidates(payload):
    """Recover the core's full input keys, independent of downstream candidates."""
    import torch
    inverse = {}
    for kind in ("lncRNA", "pathway", "cancer"):
        mapping = payload["bundle"].node_maps[kind]
        if any(type(v) is not int for v in mapping.values()) or sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("Core input node maps are not one-to-one and zero-based")
        inverse[kind] = {v:str(k) for k,v in mapping.items()}
    frames = []
    for batch in payload["train_batches"]:
        keys = {}
        for field,kind,column in (("l","lncRNA","lncrna_id"),("p","pathway","pathway_id"),("c","cancer","cancer_id")):
            values = batch["candidate_batch"][field]
            if not torch.is_tensor(values) or values.dtype != torch.int64 or values.ndim != 1:
                raise ValueError("Core candidate indices are invalid")
            try:keys[column] = [inverse[kind][int(v)] for v in values]
            except KeyError as error:raise ValueError("Core candidate references an unknown node") from error
        frames.append(pd.DataFrame(keys))
    if not frames:raise ValueError("Core prepared input has no candidate batches")
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated().any():raise ValueError("Core prepared input repeats candidate identities")
    return result


def check_export_payload(inventory, part, root, payload, checkpoint, model, fold):
    import torch
    from .training import _runtime_chunk_contract
    from .integrated_model import module_state_sha256
    state = checkpoint.get("model_state", {})
    if not state or any(not torch.is_tensor(v) or not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Frozen core checkpoint contains invalid state tensors")
    observed = module_state_sha256(SimpleNamespace(state_dict=lambda:state))
    if observed != part["core_parameter_sha256"]:
        raise ValueError("Frozen core checkpoint state differs from exported parameter hash")
    chunks,weights = _runtime_chunk_contract(payload["bundle"])
    if len(chunks) != part["runtime_chunks_total"] or {str(c):weights[c] for c in chunks} != part["chunk_weights"]:
        raise ValueError("Frozen core prepared chunk contract differs from export")
    for kind in ("lncRNA", "pathway"):
        path = inventory.bind(part["exports"][kind],root,model,"core.export."+kind,fold)
        nodes = pd.read_parquet(path,columns=["node_id","node_index"]).sort_values("node_index")
        mapping = payload["bundle"].node_maps[kind]
        if (nodes.node_index.tolist() != list(range(len(mapping)))
                or nodes.node_id.astype(str).tolist() != list(map(str,sorted(mapping,key=mapping.get)))):
            raise ValueError("Frozen core exported node identities differ from prepared graph")


def inspect_export(inventory, path, model, role):
    from .current_core_export import validate_current_export
    path = Path(path).resolve()
    export,_,_ = validate_current_export(path)
    root = path.parent
    read = lambda p:json.loads(p.read_text(encoding="utf-8"))
    expected = None
    if export.get("source_inputs"):
        item = export["source_inputs"];digest = item["sha256"]
        located = inventory.locations.get(digest,item)
        if digest in inventory.locations:inventory.used_locations.add(digest)
        source = inventory.bind(located,root,model,role+".export_inputs")
        declaration = read(source)
        if (declaration.get("format") != "CURRENT_CORE_EXPORT_INPUT_V1"
                or declaration.get("core_run_id") != export["core_run_id"]
                or declaration.get("graph_variant") != export["graph_variant"]):
            raise ValueError("Frozen core export source manifest identity differs")
        expected = {r["fold"]:r for r in declaration["fold_inputs"]}
        if len(declaration["fold_inputs"]) != 5 or set(expected) != set(range(5)):
            raise ValueError("Frozen core export source manifest omits or repeats folds")
    records = []
    for fold in range(5):
        part = export["folds"][str(fold)]
        sources = part.get("sources")
        if not sources:
            if expected is not None:raise ValueError("Frozen core export omitted declared sources")
            inventory.gap(model,role,"CORE_EXPORT_TRAINING_SOURCES_NOT_BOUND",fold)
            continue
        required = {"prepared","config","input_manifest","checkpoint","training_receipt"}
        optional = {"graph_overlay","global_binding_source","binding_expansion"}
        if not required <= set(sources) or set(sources) - required - optional:
            raise ValueError("Frozen core export training sources are incomplete or unexpected")
        if expected is not None:
            declared = expected[fold]
            roles = required | (optional & set(declared))
            if set(sources) != roles or any(sources[k]["sha256"] != declared[k]["sha256"] for k in roles):
                raise ValueError("Frozen core export sources differ from actual export inputs")
        if (sources["checkpoint"]["sha256"] != part["checkpoint_sha256"]
                or sources["training_receipt"]["sha256"] != part["training_receipt"]["sha256"]):
            raise ValueError("Frozen core source checkpoint or copied training receipt differs")
        record = {"fold":fold}
        for key,item in sorted(sources.items()):
            digest = item["sha256"]
            resolved = inventory.locations.get(digest,item)
            if digest in inventory.locations:inventory.used_locations.add(digest)
            file = Path(resolved.get("path",""));file = file if file.is_absolute() else root/file
            if not resolved.get("path") or not file.is_file():
                inventory.authority(resolved,root,model,role+".source."+key,fold)
                continue
            file = inventory.bind(resolved,root,model,role+".source."+key,fold)
            record[key] = {"path":str(file),"sha256":digest}
        if set(record)-{"fold"} != set(sources):
            inventory.gap(model,role,"CORE_EXPORT_TRAINING_SOURCE_FILES_UNAVAILABLE",fold)
            continue
        records.append(record)
    if records:
        inventory.core_inputs({"core_run_id":export["core_run_id"],"graph_variant":export["graph_variant"],"fold_inputs":records},
            root,None,model,"association_membership_probability",exported_folds=export["folds"])
    return len(records)==5


def graph_sources(inventory, graph, root, model, fold):
    """Retain explicit table provenance, including the base of a C expansion."""
    from .current_auxiliary_model_use import input_source
    known = set()
    def visit(authority, prefix):
        parent = authority.get("base_authority")
        if parent is not None:visit(parent,prefix+"base.")
        for group in ("static_artifacts","fold_artifacts"):
            for name,item in authority.get(group,{}).items():
                input_source(inventory,item,root,model,prefix+group+"."+name,fold=fold)
                known.add(item.get("sha256"))
        item = authority.get("receipt",{})
        inventory.authority(item,root,model,prefix+"receipt",fold);known.add(item.get("sha256"))
        receipt=inventory.locations.get(item.get("sha256"),item)
        path=Path(receipt.get("path",""));path=path if path.is_absolute() else root/path
        graph_input=authority.get("fold_artifacts",{}).get("train_coexpression")
        if path.is_file() and graph_input:
            from .local_cnv_sources import inspect_corrected_graph
            actual=inventory.bind(graph_input,root,model,prefix+"train_coexpression",fold)
            inspect_corrected_graph(inventory,path,actual,model,prefix+"local_cnv",fold)
    visit(graph,"graph.")
    return known
