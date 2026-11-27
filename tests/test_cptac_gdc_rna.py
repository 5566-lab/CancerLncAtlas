import gzip
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cla_engine.common import sha
from cla_engine.cptac_gdc_rna import MATRIX, SYMBOLS, materialize_gdc
from cla_engine import cptac_pipeline
from cla_engine.common import Stop


def test_legacy_recipe_uses_current_gdc_source_and_fails_if_unavailable(tmp_path, monkeypatch):
    expected = dict(intake=str(tmp_path), matches=str(tmp_path / "match.tsv"))
    monkeypatch.setattr(cptac_pipeline, "read_json", lambda path: {"gdc_rna": expected})
    with pytest.raises(Stop, match="unavailable"):
        cptac_pipeline.rna_source({"download_manifest": {}})
    (tmp_path / "gdc_files.json").write_text("[]")
    (tmp_path / "match.tsv").touch()
    assert cptac_pipeline.rna_source({"download_manifest": {}}) == expected
    with pytest.raises(Stop, match="must be specified"):
        cptac_pipeline.rna_source({"gdc_rna": {}})


def write_tsv(path, rows, columns):
    pd.DataFrame(rows, columns=columns.split()).to_csv(path, sep="\t", index=False)


def test_gdc_replacement_preserves_missingness_identity_and_other_measurements(tmp_path):
    baseline = tmp_path / "baseline"
    intake = tmp_path / "intake"
    cache = intake / "files"
    matrix_dir = baseline / "matrices/BRCA"
    cache.mkdir(parents=True)
    matrix_dir.mkdir(parents=True)
    gtf = tmp_path / "genes.gtf.gz"
    with gzip.open(gtf, "wt") as stream:
        for gid in ["ENSG00001.1", "ENSG00001.1_PAR_Y"]:
            stream.write(f'X\tsrc\tgene\t1\t2\t.\t+\t.\tgene_id "{gid}"; gene_name "A"; gene_type "lncRNA";\n')
    hgnc, graph = tmp_path / "hgnc.tsv", tmp_path / "graph.parquet"
    write_tsv(hgnc, [], "ensembl_gene_id status")
    pd.DataFrame(columns=["gene_id", "protein_id"]).to_parquet(graph)
    cfg = {k: {"path": str(p), "sha256": sha(p)} for k, p in
           [("gencode", gtf), ("hgnc", hgnc), ("protein_gene", graph)]}
    spec = "cohort_id source_cohort cancer_id native_case_id patient_id sample_id tissue_type layer native_sample_id source_file"
    rows = []; matches = []; hits = []
    for i, (case, suffix, tissue) in enumerate([("c1", "T", "Tumor"), ("c1", "N", "Normal"), ("c2", "T", "Tumor")]):
        patient = "CPTAC:BRCA:" + case
        cols = ["CPTAC", "BRCA", "BRCA", case, patient, patient + "::" + suffix, tissue, "rna", case, "bcm"]
        rows.append(cols)
        extra = dict(match_status="AMBIGUOUS", match_basis="multiple specimens", gdc_file_id="", gdc_md5="",
                     gdc_aliquot_id="", gdc_sample_id="", gdc_sample_submitter_id="", gdc_aliquot_submitter_id="")
        if i < 2:
            uid = "00000000-0000-0000-0000-" + str(i).zfill(12)
            header = "# gene-model: GENCODE v36\ngene_id\tgene_name\tgene_type\ttpm_unstranded\n"
            data = [f"N_{n}\t\t\t" for n in ["unmapped", "multimapping", "noFeature", "ambiguous"]]
            data += [f"ENSG00001.1\tA\tlncRNA\t{1+i}", "ENSG00001.1_PAR_Y\tA\tlncRNA\t3"]
            body = (header + "\n".join(data)).encode()
            (cache / (uid + ".tsv")).write_bytes(body)
            md5 = hashlib.md5(body).hexdigest()
            extra.update(match_status="MATCHED", match_basis="case tissue", gdc_file_id=uid,
                         gdc_md5=md5, gdc_aliquot_id=uid, gdc_aliquot_submitter_id=uid,
                         gdc_sample_id=suffix, gdc_sample_submitter_id=suffix)
            sample = dict(sample_id=suffix, submitter_id=suffix, sample_type=("Primary Tumor" if i == 0 else "Solid Tissue Normal"),
                          portions=[{"analytes": [{"aliquots": [{"aliquot_id": uid}]}]}])
            parents = [sample]
            if i == 0:
                parents.append({**sample, "sample_id": "other", "submitter_id": "other"})
                extra.update(gdc_sample_id="T|other", gdc_sample_submitter_id="T|other")
            hits.append(dict(access="open", data_type="Gene Expression Quantification", file_id=uid,
                             file_name=uid + ".tsv", file_size=len(body), md5sum=md5, analysis={"workflow_type": "STAR - Counts"},
                             cases=[dict(submitter_id=case, project={"project_id": "CPTAC-2"}, samples=parents)],
                             associated_entities=[dict(entity_type="aliquot", entity_id=uid, entity_submitter_id=uid)]))
        matches.append({**dict(zip(spec.split(), cols)), **extra})
    write_tsv(baseline / "CPTAC_SAMPLE_ASSAY_MAP.tsv", rows, spec)
    columns = spec.split()[:8] + ["features", "observed_features", "source_unit", "matrix_path", "source_file"]
    pd.DataFrame(columns=columns).to_csv(baseline / "CPTAC_ASSAY_COVERAGE.tsv", sep="\t", index=False)
    write_tsv(baseline / "COHORT_COVERAGE.tsv", [["BRCA", "BRCA", "rna", 2, 2, 100]],
              "source_cohort cancer_id layer patients observed_patients coverage_percent")
    write_tsv(baseline / "SOURCE_INVALID_RNA_VALUES.tsv", [], "source_file raw_value")
    write_tsv(baseline / "SOURCE_NONFINITE_MEASUREMENTS.tsv", [], "source_file raw_value")
    write_tsv(baseline / "RNA_SCORING_EXCLUDED_GENES.tsv", [], "source_cohort gene_symbol reason")
    write_tsv(baseline / "CPTAC_EXCLUDED_SAMPLES.tsv", [], "source_file native_sample_id reason")
    pd.DataFrame({"layer": ["protein"], "feature_id": ["old"]}).to_parquet(baseline / "FEATURE_ALIGNMENT.parquet")
    protein = matrix_dir / "protein_Tumor.parquet"
    pd.DataFrame({"p": [15.]}).to_parquet(protein)
    files = {p.relative_to(baseline).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
             for p in baseline.rglob("*") if p.is_file()}
    (baseline / "COMPLETE.json").write_text(json.dumps({"format": "CLA_CPTAC_ALIGNED_INPUT_V1",
        "status": "COMPLETE", "cohorts": ["BRCA"], "config": cfg, "files": files}))
    (intake / "gdc_files.json").write_text(json.dumps(hits))
    m = tmp_path / "matches.tsv"
    pd.DataFrame(matches).to_csv(m, sep="\t", index=False)
    out = tmp_path / "out"
    receipt = materialize_gdc(baseline, m, intake, out)
    assert receipt["rna_quantification"]["cohort_counts"]["BRCA"] == {"MATCHED": 2, "AMBIGUOUS": 1}
    frame = pd.read_parquet(out / "matrices/BRCA" / MATRIX).set_index("feature_id")
    assert list(frame.index) == ["ENSG00001", "ENSG00001_PAR_Y"]
    assert list(frame.columns) == ["CPTAC:BRCA:c1::T", "CPTAC:BRCA:c1::N"]
    np.testing.assert_allclose(frame.iloc[:, 1], np.log2([3, 4]))
    assert pd.read_csv(out / "matrices/BRCA" / SYMBOLS, sep="\t").iloc[0, 1] == 4.
    assert sha(protein) == sha(out / "matrices/BRCA" / protein.name)
    count = pd.read_csv(out / "COHORT_COVERAGE.tsv", sep="\t")
    assert count.observed_patients.item() == 1
    # The same frozen inputs can be replayed from the already replaced generation.
    repeat = tmp_path / "rerun"
    materialize_gdc(out, m, intake, repeat)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(repeat / "matrices/BRCA" / MATRIX).set_index("feature_id"))
    (cache / (hits[0]["file_id"] + ".tsv")).write_bytes(b"drift")
    with pytest.raises(ValueError, match="MD5"):
        materialize_gdc(baseline, m, intake, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
