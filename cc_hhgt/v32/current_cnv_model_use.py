"""Follow verified native CNV reads and hash-bound materialization ancestry."""
from pathlib import Path
import json


def inspect_consumed_sources(inventory, context, model, role):
    from .current_auxiliary_model_use import input_source
    receipt_path, receipt = context["consumed"]
    visited = set()

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
            inventory.authority(resolved, base, model, role+label, -1)
            return
        path = inventory.bind(resolved, base, model, role+label)
        identity = (str(path), digest)
        if identity in visited:
            return
        visited.add(identity)
        if path.suffix == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))

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
            input_source(inventory, {**resolved, "source_kind":"table"}, base, model, role+label)

    for i, item in enumerate(receipt["records"]):
        if item["role"] in {"folds", "core", "candidates"}:
            continue  # Already bound by native_sources; not biological metadata.
        walk(item, receipt_path.parent, ".consumed."+str(i))
    inventory.gap(model, role, "CNV_RAW_TRANSFORMATION_AND_CROSS_MODULE_PATIENT_CLOSURE_NOT_VERIFIED")
