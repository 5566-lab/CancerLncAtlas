"""Stage native CPTAC scores and current-config payloads against harmonized patient RNA."""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cla_engine.common import atomic_json, read_json, sha
from cla_engine.cptac_binding import ROLE, load_assay, load_scores, validate_binding
from cla_engine.cptac_bundle import assemble, md5
from cla_engine.cptac_native_states import run as native_states


def pin(path):
    return {"path": str(path), "sha256": sha(path)}


def rjob(script, arguments, log):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with log.open("w") as output:
        subprocess.run(["/usr/bin/Rscript", "--vanilla", str(REPO / "scripts" / script), *arguments],
                       stdout=output, stderr=subprocess.STDOUT, check=True, env=env)


def rebind_non_rna(config, aligned, out):
    """Reuse existing PTM/protein/methylation bytes after comparing the RNA-independent inputs."""
    previous = Path(config["omics_input_complete"]["path"])
    if sha(previous) != config["omics_input_complete"]["sha256"]:
        raise ValueError("Unverified non-RNA inputs")
    rec = read_json(previous)
    original = rec["config"]["aligned_complete"]
    root = Path(original["path"]).parent
    if sha(root / "COMPLETE.json") != original["sha256"]:
        root = root.with_name(root.name + ".before_gdc_" + original["sha256"][:12])
    if sha(root / "COMPLETE.json") != original["sha256"]:
        raise ValueError("Missing previous alignment generation")
    old = read_json(root / "COMPLETE.json")
    new = read_json(aligned / "COMPLETE.json")
    features = "FEATURE_ALIGNMENT.parquet"
    oldf = pd.read_parquet(root / features)
    newf = pd.read_parquet(aligned / features, columns=oldf.columns.tolist())
    for rel in old["files"]:
        if rel.startswith("matrices/") and Path(rel).name.startswith(("protein_", "phosphorylation_", "methylation_gene_")):
            if old["files"][rel]["sha256"] != new["files"][rel]["sha256"]:
                raise ValueError(f"Cannot reuse non-RNA scores, changed {rel}")
    pd.testing.assert_frame_equal(oldf.loc[oldf.layer.ne("rna")].reset_index(drop=True),
                                  newf.loc[newf.layer.ne("rna")].reset_index(drop=True))
    dest = out / "non_rna_inputs"
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=out) as temp:
            build = Path(temp) / "data"
            build.mkdir()
            for rel, item in rec["files"].items():
                source = previous.parent / rel
                if sha(source) != item["sha256"]:
                    raise ValueError(f"Changed non-RNA score input {source}")
                target = build / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
            rec["config"]["aligned_complete"] = pin(aligned / "COMPLETE.json")
            rec["source_non_rna_preparation"] = dict(prior_receipt_sha256=sha(previous),
                prior_aligned_sha256=original["sha256"], basis="Identical non-RNA feature records and measured matrices")
            atomic_json(build / "COMPLETE.json", rec)
            build.rename(dest)
    return pin(dest / "COMPLETE.json")


def stage(aligned, inputs, pointer, out, jobs):
    out.mkdir(parents=True, exist_ok=True)
    aligned = aligned.resolve()
    base = read_json(aligned / "COMPLETE.json")
    quant = base["rna_quantification"]
    if (quant["workflow_type"], quant["gene_model"], quant["metric"]) != (
            "STAR - Counts", "GENCODE v36", "tpm_unstranded"):
        raise ValueError("The aligned dataset is not a GDC STAR / v36 release")
    identity = {"aligned": pin(aligned / "COMPLETE.json"), "source_config": pin(inputs / "BUNDLE_CONFIG.json")}
    identity_file = out / "SCORING_INPUT.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise ValueError("This staging directory belongs to another input generation")
    atomic_json(identity_file, identity)
    cfg = read_json(inputs / "BUNDLE_CONFIG.json")
    cfg["aligned_complete"] = identity["aligned"]
    source_spec = read_json(cfg["source_spec"]["path"])
    libraries = ":".join(source_spec["R_libraries"])
    rsrc = inputs / "SCORING_RESOURCES.json"
    resource = read_json(rsrc)

    def run_cohort(cohort):
        cohort_root = out / "individual" / cohort
        cohort_root.parent.mkdir(exist_ok=True)
        if not cohort_root.exists():
            with tempfile.TemporaryDirectory(dir=cohort_root.parent) as temp:
                build = Path(temp) / "output"
                rjob("score_cptac_rna.R", ["--input-root", str(aligned), "--r-library", libraries,
                     "--output-root", str(build), "--senmayo-json", resource["senmayo"],
                     "--resources", str(rsrc), "--cancer-type", "list", "--cancer-list", cohort],
                     out / (cohort + ".log"))
                build.rename(cohort_root)
        method = read_json(cohort_root / "rna_scores" / (cohort + ".json"))
        if (method["state"] != "COMPLETE" or method["failures"] or method["input_md5"] !=
                md5(aligned / "matrices" / cohort / "RNA_TUMOR_SYMBOL_TPM.tsv.gz")):
            raise ValueError("Stale native RNA score " + cohort)
        return cohort_root

    def run_extra():
        dest = out / "extra"
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=out) as temp:
                build = Path(temp) / "output"
                rjob("score_cptac_extra_states.R", ["--input-root", str(aligned), "--output", str(build),
                     "--r-library", libraries], out / "XCELL_EXTEND.log")
                build.rename(dest)
        return dest / "COMPLETE.json"

    def run_oclr():
        if importlib.util.find_spec("openpyxl") is None:
            # The historical 149 build installed this pure-Python reader separately.
            sys.path.append(str(inputs.parent / "cla_rppa500_upgrade_20261104_r1" / "python_deps"))
        dest = out / "native_states"
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=out) as temp:
                build = Path(temp) / "output"
                spec = read_json(inputs / "NATIVE_STATE_CONFIG.json")
                spec["aligned_complete"] = identity["aligned"]
                native_states(spec, build)
                build.rename(dest)
        return dest / "COMPLETE.json"

    with ThreadPoolExecutor(max_workers=2) as others, ThreadPoolExecutor(max_workers=jobs) as cohorts:
        extra = others.submit(run_extra)
        oclr = others.submit(run_oclr)
        scores_root = out / "scores"
        for single in cohorts.map(run_cohort, base["cohorts"]):
            for folder in ("rna_scores", "pathway_scores"):
                dest = scores_root / folder
                dest.mkdir(parents=True, exist_ok=True)
                for item in (single / folder).iterdir():
                    target = dest / item.name
                    if not target.exists():
                        os.link(item, target)
                    elif sha(target) != sha(item):
                        raise ValueError(f"Score conflict: {target}")
        cfg.update(extra_state_complete=pin(extra.result()), native_state_complete=pin(oclr.result()),
                   rna_scores_root=str(scores_root))
    cfg["omics_input_complete"] = rebind_non_rna(cfg, aligned, out)
    bundle = out / "bundle"
    if not bundle.exists():
        with tempfile.TemporaryDirectory(dir=out) as temp:
            build = Path(temp) / "bundle"
            assemble(cfg, build)
            build.rename(bundle)
    rec = read_json(bundle / "COMPLETE.json")
    if rec["config"] != cfg:
        raise ValueError("Native score bundle source configuration is stale")
    binding = dict(format=rec["format"], role=ROLE, training_pooling=False, complete=pin(bundle / "COMPLETE.json"))
    status = validate_binding(binding, tcga_state_sha256=cfg["tcga_state"]["sha256"])
    # Verify public readers against the new raw data and stored RNA pathway files.
    for cohort in base["cohorts"]:
        assay = load_assay(binding, cohort=cohort, layer="rna")["values"]
        raw = pd.read_parquet(aligned / "matrices" / cohort / "rna_TumorNormal.parquet").set_index("feature_id")
        if not assay.equals(raw.loc[:, raw.columns.str.endswith("::T")]):
            raise ValueError("Reader/source mismatch " + cohort)
        score = load_scores(binding, cohort=cohort, layer="RNA_PATHWAY")
        if not score["available"].equals(score["values"].notna()):
            raise ValueError("Invalid RNA pathway missingness " + cohort)
        load_scores(binding, cohort=cohort, layer="STATE")
    atomic_json(out / "VALIDATED.json", status)
    current = read_json(pointer)
    runtime_path = Path(current["current_config"]["path"])
    if sha(runtime_path) != current["current_config"]["sha256"]:
        raise ValueError("The old CPTAC runtime config changed")
    runtime = read_json(runtime_path)
    if runtime.get("cptac_input") != current["cptac_input"]:
        raise ValueError("Existing CPTAC config and pointer disagree")
    runtime["cptac_input"] = binding
    atomic_json(out / "CURRENT_149_CONFIG.json", runtime)
    scores = pd.read_csv(bundle / "SCORE_INDEX.tsv", sep="\t")
    scorecov = pd.read_csv(bundle / "SCORE_COVERAGE.tsv.gz", sep="\t")
    # Patient/clinical coverage is controlled by the original non-RNA axes.
    old = read_json(Path(current["cptac_input"]["complete"]["path"]))
    if (old["patients"] != rec["patients"] or old["cohorts"] != rec["cohorts"]
            or old["state_axis_count"] != rec["state_axis_count"]):
        raise ValueError("Unexpected patient or state-axis change")
    summary = current["coverage"]
    summary["state_observed"] = rec["observed_state_count"]
    summary["state_unavailable"] = rec["state_axis_count"] - rec["observed_state_count"]
    for line in summary["scores"]:
        match = scores[scores.layer == line["layer"]]
        assayed = match[match.assayed_patients > 0]
        hits = scorecov[(scorecov.layer == line["layer"]) & (scorecov.observed_patients > 0)]
        line.update(assayed_cohorts=len(assayed), finite_scores=int(match.finite_scores.sum()),
                    observed_targets_any_cohort=hits.target_id.nunique(),
                    per_assayed_cohort_min=int(assayed.observed_targets.min()),
                    per_assayed_cohort_max=int(assayed.observed_targets.max()))
    new_pointer = dict(format=current["format"], state="CPTAC_GDC_STAR_ASSAYS_AND_NATIVE_SCORES_ACTIVE",
                       cptac_input=binding, current_config={"path": str(runtime_path),
                       "sha256": sha(out / "CURRENT_149_CONFIG.json")}, coverage=summary,
                       target_host="149", training_pooling=False, new_training=False,
                       TCGA_inputs_unchanged=True, task_weights=current["task_weights"],
                       MOGSA=current["MOGSA"], rna_quantification=quant, validation=pin(out / "VALIDATED.json"),
                       previous_pointer_sha256=sha(pointer))
    atomic_json(out / "CPTAC_CURRENT.json", new_pointer)
    atomic_json(out / "STAGED_NATIVE_POINTERS.json", {"status":"VERIFIED_STAGED", "targets": [
        dict(active=str(runtime_path), previous_sha256=sha(runtime_path), staged=pin(out / "CURRENT_149_CONFIG.json")),
        dict(active=str(pointer), previous_sha256=sha(pointer), staged=pin(out / "CPTAC_CURRENT.json"))]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aligned", "original-stage", "current-pointer", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    stage(args.aligned, args.original_stage, args.current_pointer, args.output, args.jobs)
