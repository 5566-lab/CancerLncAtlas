"""Follow verified native CNV reads and hash-bound materialization ancestry."""
from pathlib import Path
import json
import pandas as pd


def inspect_bound_sources(inventory, records, root, model, role, fold=-1,*,phase="FEATURE_CONSTRUCTION"):
    from .current_auxiliary_model_use import input_source
    visited = set(); reconstructed = set()

    def walk(item, base, label, depth=0):
        if depth > 80:
            raise ValueError("CNV source receipt graph exceeds traversal depth")
        digest = item.get("sha256")
        if not digest:
            inventory.gap(model, role+label, "CNV_UPSTREAM_PATH_HAS_NO_BOUND_HASH")
            return
        resolved = inventory.locations.get(digest, item)
        if digest in inventory.locations:
            inventory.used_locations.add(digest)
        declared = Path(resolved.get("path", ""))
        declared = declared if declared.is_absolute() else base/declared
        if not resolved.get("path") or not declared.exists():
            inventory.authority(resolved, base, model, role+label, fold)
            return
        path = inventory.bind(resolved, base, model, role+label, fold)
        identity = (str(path), digest)
        if identity in visited:
            return
        visited.add(identity)
        if path.suffix == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
            from .native_directional_cnv import PARTITION, inspect
            if isinstance(document, dict) and document.get("format") == PARTITION:
                inspect(inventory, resolved, base, model, role+label+".directional", fold, phase=phase)
                reconstructed.add(digest)
                return

            def visit(value, prefix):
                if isinstance(value, dict):
                    if "path" in value:
                        if value.get("sha256"):
                            walk(value, path.parent, prefix, depth+1)
                            return
                        if value.get("success_sha256"):
                            walk({"path":str(Path(value["path"])/"SUCCESS.json"),
                                  "sha256":value["success_sha256"]}, path.parent, prefix, depth+1)
                            return
                        inventory.gap(model, role+prefix, "CNV_UPSTREAM_PATH_HAS_NO_BOUND_HASH")
                    # JSON keys may contain patient IDs. Expose indices only.
                    for i, (key, child) in enumerate(value.items()):
                        name = prefix+".field."+str(i)
                        if key == "files" and isinstance(child, dict):
                            for j, (filename, file_sha) in enumerate(child.items()):
                                target = (path.parent/filename).resolve()
                                if not target.is_relative_to(path.parent):
                                    raise ValueError("CNV materialized member escapes its partition")
                                walk({"path":str(target), "sha256":file_sha}, path.parent, name+"."+str(j), depth+1)
                        elif key.endswith("_path") and isinstance(child, str):
                            walk({"path":child, "sha256":value.get(key[:-5]+"_sha256")}, path.parent, name, depth+1)
                        elif isinstance(child, (dict, list)):
                            visit(child, name)
                elif isinstance(value, list):
                    for i, child in enumerate(value):
                        visit(child, prefix+"."+str(i))
            visit(document, label)
        elif path.suffix in {".parquet", ".tsv", ".csv", ".maf", ".txt", ".seg"}:
            input_source(inventory, {**item, **resolved, "source_kind":"table"}, base, model, role+label,fold=fold,phase=phase)

    for i,item in enumerate(records):walk(item,root,".consumed."+str(i))
    return reconstructed


def inspect_consumed_sources(inventory, context, model, role):
    receipt_path, receipt = context["consumed"]
    biological=[r for r in receipt["records"] if r["role"] not in {"folds","core","candidates"}]
    reconstructed = inspect_bound_sources(inventory,biological,receipt_path.parent,model,role)

    for i, item in enumerate(receipt["records"]):
        if item["role"] in {"folds", "core", "candidates"}:
            continue  # Already bound by native_sources; not biological metadata.
        if item["role"]=="compact" and Path(item["path"]).name=="patient_ids.json":
            path=Path(item["path"]);path=path if path.is_absolute() else receipt_path.parent/path
            patients=json.loads(path.read_text(encoding="utf-8"))
            folds=context["patients"]
            referenced=set(folds.loc[folds.cancer_id.eq(item["cancer"]),"patient_id"])
            values=[str(v) for v in patients if str(v) in referenced]
            inventory.patient_use.observe([pd.DataFrame({"patient_ids":values})],digest=item["sha256"],
                model=model,context=inventory.context_id,role="CNV_COMPACT_REFERENCED_PATIENT_ROWS",
                scope="NATIVE_COMPACT_ROWS_REFERENCED_BY_FOLDS_NOT_ALL_ROWS_FITTED")
    compact_receipts = {r["sha256"] for r in biological if r["role"]=="compact" and Path(r["path"]).name=="SUCCESS.json"}
    reason = ("CNV_SEGMENT_ACQUISITION_AND_CROSS_MODULE_PATIENT_CLOSURE_UNVERIFIED"
              if compact_receipts and compact_receipts <= reconstructed else
              "CNV_RAW_TRANSFORMATION_AND_CROSS_MODULE_PATIENT_CLOSURE_NOT_VERIFIED")
    inventory.gap(model, role, reason)
