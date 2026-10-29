"""Identity and fold gates on the GDC-to-gene input binding."""
from pathlib import Path
import pandas as pd
import pytest

from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts.bind_v32_tcga_gene_sources import bind
from scripts.prepare_v32_gene_node_folds import prepare
from test_v32_gene_patient_omics import config, product


def source_spec(root):
    base = read(config(root))
    patient = "TCGA-AA-0001"; sample = patient + "-01A"
    pd.DataFrame({"cancer_id": ["C1"], "patient_id": [patient], "sample_id": [sample],
                  "patient_fold_id": [2]}).to_csv(root / "formal.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": ["GENE:ENSG1", "GENE:ENSG2"]}).to_parquet(root / "members.parquet", index=False)
    pd.DataFrame([{ "cancer_id": "C1", "case_submitter_id": patient, "sample_id": "uuid1", "sample_submitter_id": sample,
        "sample_type": "Primary Tumor", "sample_metadata_ambiguous": False, "selected_for_patient": True,
        "workflow_type": "DNAcopy", "data_type": "Masked Copy Number Segment", "file_name": "x.grch38.seg.txt",
        "file_id": "cnv1", "md5sum": "a"*32, "file_size": "12", "patient_fold_id": 4}]).to_csv(root / "gdc.tsv", sep="\t", index=False)
    pd.DataFrame([{ "file_id": "cnv1", "cancer_id": "C1", "staged_path": str(root / "x.seg"),
        "size": "12", "md5": "a"*32}]).to_csv(root / "stage.tsv", sep="\t", index=False)
    pd.DataFrame([{ "cancer_id": "C1", "patient_id": patient, "sample_id": sample, "sample_uuid": "uuid1",
        "sample_type": "Primary Tumor", "tissue_type": "Tumor", "platform": "Illumina Human Methylation 450",
        "workflow_type": "SeSAMe Methylation Beta Estimation", "data_type": "Methylation Beta Value",
        "file_name": "x.beta.txt", "file_id": "beta1", "md5sum": "b"*32, "file_size": "24"}]).to_csv(root / "beta.tsv", sep="\t", index=False)
    sources = {"gtf": base["gtf"], "hm450": base["hm450"], **{k: record(root / p) for k, p in {
        "patient_folds": "formal.tsv", "signed_membership": "members.parquet", "protein_gene_encoding": "members.parquet",
        "cnv_manifest": "gdc.tsv", "cnv_staging": "stage.tsv", "methylation_manifest": "beta.tsv"}.items()}}
    path = root / "sources.json"
    write(path, {"target_host": "149", "genome_build": "GRCh38", "sources": sources, "methylation_download_root": str(root / "download")})
    return path


def test_binding_uses_current_folds_and_requires_later_raw_verification(tmp_path):
    spec = source_spec(tmp_path)
    result = read(bind(spec, tmp_path / "bound"))
    assert result["patient_folds"] == read(spec)["sources"]["patient_folds"]
    report = read(tmp_path / "bound/BINDING_REPORT.json")
    assert report["old_cnv_fold_assignments_used"] is False
    assert report["raw_checksums_verified"] is False
    raw = pd.read_parquet(result["raw_files"]["path"])
    assert set(raw.patient_id) == {"TCGA-AA-0001"}
    assert set(raw.md5) == {"a"*32, "b"*32}


@pytest.mark.parametrize("fault", ["checksum", "ambiguous", "sample", "normal"])
def test_binding_refuses_inconsistent_source_identity(tmp_path, fault):
    spec = source_spec(tmp_path); cfg = read(spec)
    key = "cnv_staging" if fault == "checksum" else ("methylation_manifest" if fault == "normal" else "cnv_manifest")
    path = Path(cfg["sources"][key]["path"]); data = pd.read_csv(path, sep="\t")
    column, value = {"checksum": ("md5", "c"*32), "ambiguous": ("sample_metadata_ambiguous", True),
                     "sample": ("sample_submitter_id", "TCGA-ZZ-0002-01A"), "normal": ("tissue_type", "Normal")}[fault]
    data[column] = value; data.to_csv(path, sep="\t", index=False)
    cfg["sources"][key] = record(path); write(spec, cfg)
    error = {"checksum": "TCGA_CNV_STAGING_DIFFERS_FROM_GDC", "ambiguous": "TCGA_CNV_SELECTION_IDENTITY_OR_BUILD",
             "sample": "TCGA_SAMPLE_PATIENT_CONFLICT", "normal": "TCGA_METHYLATION_ASSAY_OR_TUMOR_TYPE"}[fault]
    with pytest.raises(ValueError, match=error): bind(spec, tmp_path / "bad")


def test_formal_attachment_refuses_a_different_patient_fold_authority(product, tmp_path):
    parent = tmp_path / "old_prepared"; parent.mkdir()
    (parent / "SAMPLE_PATIENT_FOLD_MAP.tsv").write_text("different fold assignments\n")
    with pytest.raises(ValueError, match="Gene assay and prepared patient-fold authorities differ"):
        prepare(product, parent, tmp_path / "output")
    assert not (tmp_path / "output/INPUT_MANIFEST.json").exists()
