"""Publish a verified GDC raw replica and stage the native score bundle on the shared CPTAC volume."""
import argparse
import os
import stat
from pathlib import Path
import sys
import shutil
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.common import atomic_json, read_json, sha
from cla_engine.cptac_binding import validate_binding, load_assay, load_scores
from cla_engine.cptac_bundle import verify_tree


def pin(path):
    return {"path": str(path), "sha256": sha(path)}


def mirror_access(tree, reference):
    """Preserve access of the original published inputs for the corresponding artifacts."""
    directory_mode = stat.S_IMODE(reference.stat().st_mode)
    file_mode = stat.S_IMODE((reference / "COMPLETE.json").stat().st_mode)
    for path in [tree, *tree.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"Expected a physical shared product: {path}")
        prior = reference / path.relative_to(tree)
        expected = (stat.S_IMODE(prior.stat().st_mode) if prior.exists()
                    else directory_mode if path.is_dir() else file_mode)
        actual = path.stat()
        if stat.S_IMODE(actual.st_mode) != expected:
            if path.is_file() and actual.st_nlink > 1:
                raise ValueError(f"Refusing to change mode of a linked source file: {path}")
            path.chmod(expected)
            if stat.S_IMODE(path.stat().st_mode) != expected:
                raise ValueError(f"Shared access mode verification failed: {path}")


def raw_replica(raw, shared, live=None):
    digest = sha(raw / "COMPLETE.json")
    dest = shared / ("aligned_gdc_star_" + digest[:12])
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=shared, prefix=".gdc_raw.") as wd:
            tree = Path(wd) / "aligned"
            shutil.copytree(raw, tree)
            verify_tree(tree / "COMPLETE.json")
            if sha(tree / "COMPLETE.json") != digest:
                raise ValueError("GDC raw manifest changed during copy")
            tree.rename(dest)
    if sha(dest / "COMPLETE.json") != digest:
        raise ValueError("Unexpected raw replica")
    mirror_access(dest, live or shared / "aligned_r4")
    verify_tree(dest / "COMPLETE.json")
    return dest

def physical_file(source, destination):
    # Dereference legacy links so reads and permission updates stay in this release.
    shutil.copyfile(Path(source).resolve(), destination)
    return destination


def shared_bundle(native, raw, replica, binding, shared):
    private = Path(binding["complete"]["path"]).parent
    source_hash = binding["complete"]["sha256"]
    if sha(private / "COMPLETE.json") != source_hash:
        raise ValueError("Private native score release changed")
    dest = shared / ("bundle_gdc_star_" + source_hash[:12])
    if dest.exists():
        if read_json(dest / "COMPLETE.json").get("private_bundle_sha256") != source_hash:
            raise ValueError("Unexpected shared native score bundle")
        mirror_access(dest, shared / "bundle_r1")
        return dest
    with tempfile.TemporaryDirectory(dir=shared, prefix=".gdc_native.") as wd:
        stage = Path(wd) / "bundle"
        # The manifest and bound-sources file must have private inodes before rebinding.
        shutil.copytree(private, stage)
        support = stage / "SOURCES"
        support.mkdir()
        folders = ["scores", "native_states", "non_rna_inputs", "extra"]
        for folder in folders:
            shutil.copytree(native / folder, support / folder, copy_function=physical_file)
        # These two receipts contain an input reference. Replace only the reference to
        # the raw GDC matrix. Write by rename so a hard-linked input is never changed.
        for folder in ["native_states", "non_rna_inputs"]:
            receipt = support / folder / "COMPLETE.json"
            doc = read_json(receipt)
            old = doc["config"]["aligned_complete"]
            if old["sha256"] != sha(replica / "COMPLETE.json"):
                raise ValueError("Rebound native input references the wrong GDC release")
            doc["config"]["aligned_complete"] = pin(replica / "COMPLETE.json")
            atomic_json(receipt, doc)

        def relocate(path):
            path = Path(path)
            if path.is_relative_to(raw):
                return replica / path.relative_to(raw)
            if path.is_relative_to(native):
                relative = path.relative_to(native)
                if relative.parts[0] not in folders:
                    raise ValueError(f"Unrecognized private dependency: {path}")
                return dest / "SOURCES" / relative
            if path.is_relative_to("/dsk2") and path.is_file():
                # Legacy pinned references are small and can also be served from this release.
                relative = Path("SOURCES", "references", sha(path), path.name)
                local = stage / relative
                if not local.exists():
                    local.parent.mkdir(parents=True, exist_ok=True)
                    physical_file(path, local)
                return dest / relative
            return path

        def rewrite(obj):
            if isinstance(obj, list):
                return [rewrite(v) for v in obj]
            if isinstance(obj, dict):
                if "path" in obj and "sha256" in obj:
                    source = Path(obj["path"])
                    if sha(source) != obj["sha256"]:
                        raise ValueError(f"Native provenance drift: {source}")
                    target = relocate(source)
                    local = stage / target.relative_to(dest) if target.is_relative_to(dest) else target
                    record = {**obj, "path": str(target), "sha256": sha(local)}
                    if "bytes" in record:
                        record["bytes"] = local.stat().st_size
                    return record
                return {k: rewrite(v) for k, v in obj.items()}
            if isinstance(obj, str) and obj.startswith("/"):
                return str(relocate(obj))
            return obj

        sources = stage / "BOUND_SOURCES.json"
        atomic_json(sources, rewrite(read_json(sources)))
        rec = read_json(private / "COMPLETE.json")
        for name, info in rec["files"].items():
            if name != "BOUND_SOURCES.json" and sha(stage / name) != info["sha256"]:
                raise ValueError(f"Shared native payload differs: {name}")
        rec["config"] = rewrite(rec["config"])
        rec["private_bundle_sha256"] = source_hash
        rec["files"] = {p.relative_to(stage).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                        for p in stage.rglob("*") if p.is_file() and p != stage / "COMPLETE.json"}
        atomic_json(stage / "COMPLETE.json", rec)
        mirror_access(stage, shared / "bundle_r1")
        stage.rename(dest)
    return dest


def make_plan(raw, replica, native_plan, shared, out):
    plan = read_json(native_plan)
    if plan["status"] != "VERIFIED_STAGED":
        raise ValueError("Private validation has not completed")
    targets = {Path(t["active"]).name: t for t in plan["targets"]}
    runtime_name = "CURRENT_149_CONFIG.json"
    required = {runtime_name, "CPTAC_CURRENT.json", "STATE_CURRENT.json", "RPPA_CURRENT.json"}
    if set(targets) != required or len(plan["targets"]) != len(required):
        raise ValueError("Expected all four production bindings")
    out.mkdir(exist_ok=True, parents=True)
    for t in targets.values():
        if sha(t["staged"]["path"]) != t["staged"]["sha256"]:
            raise ValueError("Private staged configuration changed")
    rt = read_json(targets[runtime_name]["staged"]["path"])
    private_binding = rt["cptac_input"]
    bundle = shared_bundle(native_plan.parent, raw, replica, private_binding, shared)
    binding = {**private_binding, "complete": pin(bundle / "COMPLETE.json")}
    validation = validate_binding(binding)
    for cohort in validation["cohorts"]:
        assay = load_assay(binding, cohort=cohort, layer="rna")
        scores = load_scores(binding, cohort=cohort, layer="RNA_PATHWAY")
        if len(assay["values"]) != 60660 or not scores["available"].equals(scores["values"].notna()):
            raise ValueError(f"Invalid shared RNA reader output: {cohort}")
    status_file = shared / ("VALIDATED_GDC_" + binding["complete"]["sha256"][:12] + ".json")
    atomic_json(status_file, validation)
    status_file.chmod(stat.S_IMODE((shared / "bundle_r1/COMPLETE.json").stat().st_mode))
    rt["cptac_input"] = binding
    atomic_json(out / runtime_name, rt)
    runtime_pin = {"path": targets[runtime_name]["active"], "sha256": sha(out / runtime_name)}
    for name in required - {runtime_name}:
        conf = read_json(targets[name]["staged"]["path"])
        if conf["cptac_input"] != private_binding:
            raise ValueError(f"Native binding mismatch in {name}")
        conf.update(cptac_input=binding, current_config=runtime_pin)
        if name == "CPTAC_CURRENT.json":
            conf["validation"] = pin(status_file)
        atomic_json(out / name, conf)
    original = Path(targets["CPTAC_CURRENT.json"]["active"])
    old = read_json(original)
    if sha(original) != targets["CPTAC_CURRENT.json"]["previous_sha256"]:
        # A prior activation keeps the exact previous pointer at its backup location.
        previous_sha = targets["CPTAC_CURRENT.json"]["previous_sha256"]
        backup = original.with_name(original.name + ".before_gdc_" + previous_sha[:12])
        if sha(backup) != previous_sha:
            raise ValueError("Expected the previous CPTAC pointer or its verified backup")
        old = read_json(backup)
    prior = old["cptac_input"]["complete"]
    live = Path(prior["path"]).parent
    if live.parent != shared:
        raise ValueError("Original bundle is outside the CPTAC shared root")
    plan["bundle_switch"] = dict(active=str(live), target=str(bundle), previous_sha256=prior["sha256"],
                                 staged_sha256=binding["complete"]["sha256"])
    for t in plan["targets"]:
        t["staged"] = pin(out / Path(t["active"]).name)
    atomic_json(out / "STAGED_NATIVE_POINTERS.json", plan)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True, type=Path)
    p.add_argument("--raw-live", required=True, type=Path)
    p.add_argument("--native-plan", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if bool(a.native_plan) != bool(a.output):
        p.error("--native-plan and --output must be supplied together")
    raw = a.raw.resolve()
    shared = a.raw_live.parent.resolve()
    replica = raw_replica(raw, shared, a.raw_live)
    if a.native_plan:
        make_plan(raw, replica, a.native_plan.resolve(), shared, a.output.resolve())


if __name__ == "__main__":
    main()
