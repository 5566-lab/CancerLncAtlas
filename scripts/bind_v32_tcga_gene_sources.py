"""Bind existing GDC file identities/MD5s to the current formal patient folds.

This selects existing files only; raw bytes are verified by the materializer.
The old CNV download-time fold assignment is deliberately not consumed.
"""
import argparse
from pathlib import Path
import pandas as pd

from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.gene_patient_omics import INPUT, host


def bind(source_path, output):
    host(); source_path, output = Path(source_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
                                           "source_manifest": record(source_path)})
    cfg = read(source_path)
    if cfg.get("genome_build") != "GRCh38" or cfg.get("target_host") != "149":
        raise ValueError("TCGA_GENE_SOURCE_BUILD_OR_HOST")
    files = {k: verified_path(v, source_path.parent) for k, v in cfg["sources"].items()}
    genes = set()
    for key in ["signed_membership", "protein_gene_encoding"]:
        ids = pd.read_parquet(files[key], columns=["gene_id"]).gene_id
        if ids.isna().any() or not ids.str.fullmatch(r"GENE:ENSG\d+(?:_PAR_Y)?").all():
            raise ValueError("TCGA_GRAPH_GENE_IDS_NOT_CANONICAL")
        genes.update(ids.str.removeprefix("GENE:"))
    write(output / "genes.json", sorted(genes))
    folds = pd.read_csv(files["patient_folds"], sep="\t")
    patients = folds[["patient_id", "cancer_id", "patient_fold_id"]].drop_duplicates()
    if patients.patient_id.duplicated().any(): raise ValueError("TCGA_FORMAL_PATIENT_CONFLICT")
    known = patients.set_index("patient_id").cancer_id
    segment = pd.read_csv(files["cnv_manifest"], sep="\t", dtype=str)
    if not segment.selected_for_patient.isin(["True", "False"]).all():
        raise ValueError("TCGA_CNV_SELECTION_FLAG")
    segment = segment.loc[segment.selected_for_patient.eq("True") & segment.case_submitter_id.isin(known.index)].copy()
    if (segment.case_submitter_id.duplicated().any() or not segment.sample_metadata_ambiguous.eq("False").all()
            or not segment.workflow_type.eq("DNAcopy").all() or not segment.data_type.eq("Masked Copy Number Segment").all()
            or not segment.file_name.str.contains("grch38", case=False).all()):
        raise ValueError("TCGA_CNV_SELECTION_IDENTITY_OR_BUILD")
    staging = pd.read_csv(files["cnv_staging"], sep="\t", dtype=str)
    segment = segment.merge(staging, on=["file_id", "cancer_id"], how="left", validate="one_to_one")
    if (segment.staged_path.isna().any() or not segment.md5sum.eq(segment.md5).all()
            or not segment.file_size.eq(segment["size"]).all()):
        raise ValueError("TCGA_CNV_STAGING_DIFFERS_FROM_GDC")
    cnv = pd.DataFrame({"channel": "cnv_signed", "patient_id": segment.case_submitter_id,
        "cancer_id": segment.cancer_id, "sample_id": segment.sample_submitter_id,
        "sample_uuid": segment.sample_id, "file_id": segment.file_id, "path": segment.staged_path,
        "bytes": segment.file_size.astype(int), "md5": segment.md5sum, "sample_type": segment.sample_type})
    methyl = pd.read_csv(files["methylation_manifest"], sep="\t", dtype=str)
    methyl = methyl.loc[methyl.patient_id.isin(known.index)].copy()
    if (not methyl.platform.eq("Illumina Human Methylation 450").all()
            or not methyl.workflow_type.eq("SeSAMe Methylation Beta Estimation").all()
            or not methyl.data_type.eq("Methylation Beta Value").all()
            or not methyl.tissue_type.eq("Tumor").all() or not methyl.sample_type.eq("Primary Tumor").all()):
        raise ValueError("TCGA_METHYLATION_ASSAY_OR_TUMOR_TYPE")
    download = Path(cfg["methylation_download_root"]).resolve()
    paths = []
    for row in methyl.itertuples():
        path = (download / row.file_id / row.file_name).resolve()
        if not path.is_relative_to(download): raise ValueError("TCGA_METHYLATION_PATH_OUTSIDE_ROOT")
        paths.append(str(path))
    beta = pd.DataFrame({"channel": "methylation_promoter_beta", "patient_id": methyl.patient_id,
        "cancer_id": methyl.cancer_id, "sample_id": methyl.sample_id, "sample_uuid": methyl.sample_uuid,
        "file_id": methyl.file_id, "path": paths, "bytes": methyl.file_size.astype(int),
        "md5": methyl.md5sum, "sample_type": methyl.sample_type})
    raw = pd.concat([cnv, beta], ignore_index=True)
    if not raw.cancer_id.eq(raw.patient_id.map(known)).all(): raise ValueError("TCGA_PATIENT_CANCER_CONFLICT")
    # GDC identity joins are authoritative; barcodes provide an independent sanity check.
    if not raw.sample_id.str[:12].eq(raw.patient_id).all(): raise ValueError("TCGA_SAMPLE_PATIENT_CONFLICT")
    if raw.groupby("sample_uuid").patient_id.nunique().gt(1).any(): raise ValueError("TCGA_SAMPLE_UUID_CONFLICT")
    raw.to_parquet(output / "raw_files.parquet", index=False)
    result = {"format": INPUT, "target_host": "149", "genome_build": "GRCh38",
        "gtf": record(files["gtf"]), "hm450": record(files["hm450"]), "patient_folds": record(files["patient_folds"]),
        "genes": record(output / "genes.json"), "raw_files": record(output / "raw_files.parquet"),
        "source_authorities": {**{k: record(p) for k, p in files.items()}, "binding_spec": record(source_path),
                               "binding_implementation": record(Path(__file__))}}
    write(output / "gene_assays.json", result)
    write(output / "BINDING_REPORT.json", {"target_host": "149", "training_started": False,
        "raw_checksums_verified": False, "raw_verification_required_at_materialization": True,
        "patients": len(patients), "requested_genes": len(genes), "cnv_files": len(cnv), "methylation_files": len(beta),
        "cnv_sample_types": cnv.sample_type.value_counts().to_dict(),
        "old_cnv_fold_assignments_used": False, "current_patient_fold_source": record(files["patient_folds"])})
    return output / "gene_assays.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(bind(args.sources, args.output))
