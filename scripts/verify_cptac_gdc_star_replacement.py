"""Cross-check CPTAC RNA values and non-RNA payloads in a harmonized release."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def compare_values(expected, actual, label):
    if not actual.index.is_unique or set(actual.index) != set(expected.index):
        raise ValueError("Gene identities differ in " + label)
    differences = np.abs(actual.loc[expected.index].to_numpy() - expected.to_numpy())
    if not np.all(np.isfinite(differences)) or differences.max() > 1e-12:
        raise ValueError("Value mismatch in " + label + ": " + str(np.max(differences)))
    return float(np.max(differences))


def check_rna(stage, common, intake, selected):
    checks = {}
    gene_universe = pd.read_csv(
        common / "COMMON_GENE_ANNOTATION.tsv", sep="\t", usecols=["gene_id"]
    ).gene_id
    for cohort, group in selected.groupby("source_cohort"):
        samples = group.sort_values("gdc_file_id").groupby("tissue_type").head(2)
        source_path = stage / "matrices" / cohort / "rna_TumorNormal.parquet"
        matrix = pd.read_parquet(
            source_path, columns=["feature_id", *samples.sample_id]
        ).set_index("feature_id")
        count = 0
        internal_count = 0
        src_error = 0.0
        common_error = 0.0
        common_dir = common / "aligned" / ("CPTAC_" + cohort)
        for row in samples.itertuples():
            native = intake / "files" / (row.gdc_file_id + ".tsv")
            with native.open("rb") as f:
                if f.readline().strip() != b"# gene-model: GENCODE v36":
                    raise ValueError("Unexpected native model")
            data = pd.read_csv(
                native,
                sep="\t",
                comment="#",
                usecols=["gene_id", "gene_type", "tpm_unstranded"],
            )
            data = data.loc[data.gene_id.str.startswith("ENSG")].copy()
            data.index = data.gene_id.map(
                lambda s: re.sub(r"\.\d+(?=_PAR_Y$|$)", "", s)
            )
            data.index.name = "feature_id"
            expected = np.log2(1.0 + data.tpm_unstranded)
            src_error = max(
                src_error,
                compare_values(expected, matrix[row.sample_id], cohort + " source"),
            )
            count += len(expected)
            if row.tissue_type != "Tumor":
                continue
            # Cover expressed and zero-valued lncRNA/coding genes at different levels.
            common_data = data.loc[data.index.isin(gene_universe)]
            if internal_count == 0:
                expected_lncs = {
                    "LNC:" + g
                    for g in data.index[data.gene_type.eq("lncRNA")]
                    if not g.endswith("_PAR_Y")
                }
                actual_lncs = set(
                    pq.read_schema(common_dir / "LNC_LOGTPM_FULL.parquet").names
                )
                actual_genes = set(
                    pq.read_schema(common_dir / "GENE_LOGTPM.parquet").names
                )
                if actual_lncs - {"patient_id"} != expected_lncs or actual_genes - {
                    "patient_id"
                } != set(gene_universe):
                    raise ValueError("Unexpected patient RNA axis in " + cohort)
            records = []
            for kind in ("lncRNA", "protein_coding"):
                ranked = common_data.loc[common_data.gene_type.eq(kind)].sort_values(
                    ["tpm_unstranded", "gene_id"]
                )
                positions = np.linspace(0, len(ranked) - 1, num=24).astype(int)
                records.extend(ranked.iloc[positions].index)
            genes = list(dict.fromkeys(records))
            frame = pd.read_parquet(
                common_dir / "GENE_LOGTPM.parquet", columns=genes + ["patient_id"]
            )
            if "patient_id" in frame.columns:
                frame = frame.set_index("patient_id")
            common_error = max(
                common_error,
                compare_values(
                    expected.loc[genes],
                    frame.loc[row.patient_id],
                    cohort + " patient genes",
                ),
            )
            internal_count += len(genes)
            lncs = [g for g in genes if data.loc[g, "gene_type"] == "lncRNA"]
            frame = pd.read_parquet(
                common_dir / "LNC_LOGTPM_FULL.parquet",
                columns=["LNC:" + g for g in lncs] + ["patient_id"],
            )
            if "patient_id" in frame.columns:
                frame = frame.set_index("patient_id")
            observed = frame.loc[row.patient_id].rename(
                index=lambda s: s.removeprefix("LNC:")
            )
            common_error = max(
                common_error,
                compare_values(
                    expected.loc[lncs], observed, cohort + " patient lncRNAs"
                ),
            )
            internal_count += len(lncs)
        checks[cohort] = {
            "files_by_tissue": samples.tissue_type.value_counts().to_dict(),
            "source_values_checked": count,
            "source_max_abs_diff": src_error,
            "common_values_checked": internal_count,
            "common_max_abs_diff": common_error,
        }
    return checks


def check_other_layers(old, stage):
    first = json.loads((old / "COMPLETE.json").read_text())
    selected = {}
    for rel, record in first["files"].items():
        is_measurement = (
            rel.startswith("matrices/")
            and rel.endswith(".parquet")
            and "rna" not in Path(rel).name.lower()
        )
        if is_measurement or rel in (
            "CLINICAL.tsv",
            "PUBLISHED_PHENOTYPES.parquet",
            "CPTAC_PATIENTS.tsv",
        ):
            observed = digest(stage / rel)
            if record["sha256"] != observed:
                raise ValueError("Changed non-RNA payload: " + rel)
            selected[rel] = observed
    if not selected:
        raise ValueError("No non-RNA payloads checked")
    return selected


def check_other_layer_annotations(old, stage):
    counts = {}
    for name in (
        "CPTAC_SAMPLE_ASSAY_MAP.tsv",
        "CPTAC_ASSAY_COVERAGE.tsv",
        "COHORT_COVERAGE.tsv",
        "FEATURE_ALIGNMENT.parquet",
    ):

        def read(path):
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            else:
                frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            frame = frame.loc[frame.layer.ne("rna")].reset_index(drop=True)
            if "matrix_path" in frame:
                frame["matrix_path"] = frame.matrix_path.map(lambda v: Path(v).name)
            return frame

        before = read(old / name)
        after = read(stage / name)
        if before.empty:
            raise ValueError("Expected non-RNA annotations in " + name)
        pd.testing.assert_frame_equal(before, after[before.columns], check_dtype=False)
        counts[name] = len(before)
    return counts


def main():
    parser = argparse.ArgumentParser()
    for name in ("baseline", "stage", "common", "intake", "matches", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    selected = pd.read_csv(a.matches, sep="\t", dtype=str, keep_default_na=False)
    if (
        not selected.gdc_file_id.is_unique
        or not selected.match_status.eq("MATCHED").all()
    ):
        raise ValueError("Audit calls for a uniquely matched file per RNA sample")
    baseline_sha = digest(a.baseline / "COMPLETE.json")
    source = json.loads((a.stage / "COMPLETE.json").read_text())
    if source["rna_quantification"]["baseline_complete_sha256"] != baseline_sha:
        raise ValueError("The comparison baseline differs from the release input")
    res = {
        "baseline_sha256": baseline_sha,
        "source_complete_sha256": digest(a.stage / "COMPLETE.json"),
        "common_complete_sha256": digest(a.common / "COMPLETE.json"),
        "match_sha256": digest(a.matches),
        "samples": len(selected),
        "cohorts": check_rna(a.stage, a.common, a.intake, selected),
        "unchanged_non_rna_payloads": check_other_layers(a.baseline, a.stage),
        "unchanged_non_rna_annotation_rows": check_other_layer_annotations(
            a.baseline, a.stage
        ),
        "status": "PASS",
    }
    a.output.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print(
        "CPTAC validation passed",
        len(res["cohorts"]),
        len(res["unchanged_non_rna_payloads"]),
    )


if __name__ == "__main__":
    main()
