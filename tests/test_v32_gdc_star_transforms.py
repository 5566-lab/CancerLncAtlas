"""GDC-derived counts reach real scope, ssGSEA and retained cohort conversion."""
from pathlib import Path
import subprocess
import pandas as pd

from test_v32_gdc_star_counts import assembled, no_fit, GENES, LNCS, ROOT
from test_v32_fold_local_expression import run_r
from cc_hhgt.v32.formal_expression_sources import record, read, write
from cc_hhgt.v32.current_model_use import Inventory


def test_native_scope_and_activity_inventory_follow_actual_raw_counts(assembled, tmp_path):
    from cc_hhgt.v32 import fold_local_scope as scope, fold_local_activity as activity
    root, _, manifest, cache = assembled
    annotation = tmp_path / "annotation.tsv"
    pd.DataFrame({"ensembl_gene_id": [g.split(".")[0] for g in GENES],
        "gene_symbol": ["Z", "LOW", "LEAK", *[f"G{i}" for i in range(1, 81)]],
        "gene_type": ["lncRNA"] * 3 + ["protein_coding"] * 80}).to_csv(annotation, sep="\t", index=False)
    gmt = tmp_path / "hallmark.gmt"
    gmt.write_text("P1\tsynthetic\t" + "\t".join(f"G{i}" for i in range(1, 21)) + "\n")
    gmt_manifest = tmp_path / "gmt.json"
    write(gmt_manifest, {"format": "SSGSEA_GMT_INPUT_V1", "collections": [{"collection": "HALLMARK", "file": record(gmt)}]})
    spec = tmp_path / "scope.json"
    write(spec, {"format": scope.INPUT, "annotation": record(annotation), "gmt_manifest": record(gmt_manifest),
        "normalizations": {"BRCA": record(root / "normal/SUCCESS.json")}, "budget_per_cancer": 3, "seed": 20260726})
    scope.materialize(spec, tmp_path / "scope", runner=ROOT / "scripts/materialize_fold_local_scope.py")
    run_r(ROOT / "scripts/materialize_fold_local_ssgsea.R", root / "normal", annotation, gmt_manifest, tmp_path / "activity")
    activity.convert([tmp_path / "activity"], tmp_path / "activity_tables", runner=ROOT / "scripts/convert_fold_local_activity.py")
    inventory = Inventory([], tmp_path)
    scope_file = tmp_path / "scope/fold=0/scope.parquet"
    scope.inspect_scope(inventory, record(scope_file.parent / "SCOPE_SOURCES.json"), tmp_path, scope_file, "core", "scope", 0)
    table = tmp_path / "activity_tables/fold=0/cancer_id=BRCA/part-0.parquet"
    activity.inspect(inventory, activity.binding(table), tmp_path, table, "core", "activity", 0)
    assert not pd.read_parquet(table).empty
    assert set(pd.read_parquet(tmp_path / "scope/candidates.parquet").lncrna_id) == set(LNCS)
    assert {str(p) for p in cache.rglob("*.tsv")} <= {r["path"] for r in inventory.dependencies}
    assert str(manifest) in {r["path"] for r in inventory.dependencies}
    assert not any(r["reason"] == "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED" for r in inventory.gaps)
    assert any("GDC_PROJECT:TCGA-BRCA" in r["study_tokens_json"] for r in inventory.usage)


def test_legacy_normalization_retains_cohort_warning_and_gdc_ancestry(assembled, tmp_path):
    from cc_hhgt.v32 import formal_expression_sources as expression
    _, counts, manifest, _ = assembled
    threshold = tmp_path / "threshold"; threshold.mkdir()
    pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": [LNCS[0]], "detection_rate_logcpm_gt0": [.8]}).to_csv(
        threshold / "full_gdc_lncrna_detection_rates.tsv.gz", sep="\t", index=False)
    run_r(ROOT / "scripts/materialize_formal_lncrna_logcpm.R", counts.parent, threshold, tmp_path / "legacy")
    expression.convert(tmp_path / "legacy", tmp_path / "converted", cancers=["BRCA"], runner=ROOT / "scripts/convert_formal_lncrna_rds.py")
    output = tmp_path / "converted/cancer_id=BRCA/part-0.parquet"
    inventory = Inventory([], tmp_path)
    expression.inspect(inventory, expression.binding(output), tmp_path, output, "core", "legacy", 0)
    assert str(manifest) in {r["path"] for r in inventory.dependencies}
    assert any(r["reason"] == "COHORT_LEVEL_TMM_NORMALIZATION_NOT_FOLD_LOCAL" for r in inventory.gaps)
    assert not any(r["reason"] == "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED" for r in inventory.gaps)


def test_normalization_rejects_native_rds_without_completed_sidecar(assembled, tmp_path):
    root, counts, _, _ = assembled
    moved = tmp_path / counts.name; moved.write_bytes(counts.read_bytes())
    result = subprocess.run(["/usr/bin/Rscript", str(ROOT / "scripts/materialize_fold_local_expression.R"),
        str(moved), str(root / "patients.tsv"), "BRCA", str(tmp_path / "bad")], capture_output=True, text=True, timeout=40)
    assert result.returncode != 0 and "source receipt is missing" in result.stderr
    assert not (tmp_path / "bad/SUCCESS.json").exists()
