"""Real edgeR on synthetic counts and actual five-fold CNV/core consumers."""
from pathlib import Path
import socket
import subprocess
import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.fold_local_expression import convert, native, expression_path
from cc_hhgt.v32.formal_expression_sources import binding, validate, inspect, read, write, record, read_matrix
from cc_hhgt.v32.current_model_use import Inventory

ROOT = Path(__file__).resolve().parents[1]
R_RUNNER = ROOT / "scripts/materialize_fold_local_expression.R"
RUNNER = ROOT / "scripts/convert_fold_local_expression.py"


def forbidden(*a, **kw): raise AssertionError("No optimizer or model fitting in fold-local expression checks")


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def run_r(*args):
    result = subprocess.run(["/usr/bin/Rscript", *map(str, args)], capture_output=True, text=True, timeout=65)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def normalized(tmp_path_factory):
    assert socket.gethostname() == "149"
    root = tmp_path_factory.mktemp("fold_local_tmm")
    write(root / "LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False, "synthetic_only":True})
    fixture = root / "fixture.R"
    fixture.write_text(r'''
library(SummarizedExperiment);library(edgeR);library(data.table)
root<-commandArgs(trailingOnly=TRUE)[[1]]
samples<-c(sprintf('TCGA-ZZ-%04d-01A',0:119),'TCGA-ZZ-0000-11A','TCGA-ZZ-0000-02A','TCGA-ZZ-0000-10A')
counts<-outer(1:83,1:123,function(g,s) (((g*17+s*13)%%100)+50)*1000)
counts[1,]<-round(80000+20000*(sin((0:122)*.63)+cos((0:122)*.27)))
counts[2:3,]<-0;counts[3,which((0:119)%%5 %in% c(0,1))]<-250000
counts[83,]<-0;counts[83,which((0:119)%%5 %in% c(0,1))]<-500000
counts[,122:123]<-counts[,122:123]*5
colnames(counts)<-samples
x<-SummarizedExperiment(assays=list(unstranded=counts),
 rowData=DataFrame(gene_id=c('Z.1','LOW.1','LEAK.1',paste0('G',1:80)),gene_type=c(rep('lncRNA',3),rep('protein_coding',80))),
 colData=DataFrame(pmid=rep('77889900',123),row.names=samples))
saveRDS(x,file.path(root,'counts.RDS'))
fwrite(data.table(cancer_id='BRCA',sample_id=samples[1:120],patient_id=sprintf('TCGA-ZZ-%04d',0:119),patient_fold_id=(0:119)%%5),file.path(root,'patient_map.tsv'),sep='\t')
# Independent edgeR oracle on known training tumor indices. Normal patient 0
# and its alternate aliquots are held out in fold 0 and must not enter fit.
train<-which(!((0:119)%%5 %in% c(0,1)))
oracle<-cpm(calcNormFactors(DGEList(counts=counts[,train]),method='TMM'),log=TRUE,prior.count=.25)[1:3,,drop=FALSE]
rownames(oracle)<-c('LNC:Z','LNC:LOW','LNC:LEAK');saveRDS(oracle,file.path(root,'oracle.RDS'))
# Change ALL validation/test patients, including their normal/alternate
# aliquots; training patients and annotation are identical.
held<-which(substr(samples,1,12) %in% sprintf('TCGA-ZZ-%04d',(0:119)[(0:119)%%5 %in% c(0,1)]))
assay(x,'unstranded')[,held]<-(assay(x,'unstranded')[,held]+1)*rep(seq_len(83)*23,length(held))
saveRDS(x,file.path(root,'changed.RDS'))
''', encoding="utf-8")
    run_r(fixture, root)
    for name, raw in (("normal", "counts.RDS"), ("changed", "changed.RDS")):
        run_r(R_RUNNER, root / raw, root / "patient_map.tsv", "BRCA", root / name)
    convert([root / "normal"], root / "converted", runner=RUNNER)
    return root


def output(root, fold=0, kind="lncrna"):
    return root / "converted" / kind / f"fold={fold}" / "cancer_id=BRCA/part-0.parquet"


def test_training_transform_matches_independent_edgeR_and_heldout_changes_do_not_refit(normalized):
    oracle = read_matrix(normalized / "oracle.RDS")
    baseline = read_matrix(normalized / "normal/fold=0/lncrna.RDS")
    changed = read_matrix(normalized / "changed/fold=0/lncrna.RDS")
    np.testing.assert_allclose(baseline.loc[oracle.index, oracle.columns], oracle, rtol=0, atol=2e-13)
    pd.testing.assert_frame_equal(baseline[oracle.columns], changed[oracle.columns], check_exact=True)
    for name in ("normalization_state.RDS", "detection.tsv"):
        assert record(normalized / "normal/fold=0" / name)["sha256"] == record(normalized / "changed/fold=0" / name)["sha256"]
    assert not np.allclose(baseline.drop(columns=oracle.columns), changed.drop(columns=oracle.columns))


def test_frozen_application_is_independent_of_query_batch_and_order(normalized, tmp_path):
    script = tmp_path / "apply.R"
    script.write_text(r'''
library(SummarizedExperiment);library(edgeR)
args<-commandArgs(trailingOnly=TRUE);source(args[[1]])
x<-readRDS(file.path(args[[2]],'counts.RDS'));counts<-as.matrix(assay(x,'unstranded'))
rownames(counts)<-paste(seq_len(nrow(counts)),rowData(x)$gene_id,sep=':')
state<-readRDS(file.path(args[[2]],'normal/fold=0/normalization_state.RDS'))
one<-apply_frozen_tmm(counts[,2,drop=FALSE],state)$logcpm
group<-counts[,c(7,2,12),drop=FALSE];group[,c(1,3)]<-group[,c(1,3)]*100
together<-apply_frozen_tmm(group,state)$logcpm[,2,drop=FALSE]
stopifnot(identical(one,together))
rownames(group)[1]<-'wrong';stopifnot(inherits(try(apply_frozen_tmm(group,state),silent=TRUE),'try-error'))
''', encoding="utf-8")
    run_r(script, ROOT / "scripts/frozen_tmm.R", normalized)


def test_all_folds_export_only_train_validation_and_record_normal_training_patients(normalized, tmp_path):
    for fold in range(5):
        file = output(normalized, fold)
        frame = pd.read_parquet(file)
        assert len(frame) == 96 * 3 and frame.patient_id.nunique() == 96
        assert not frame.split_role.eq("test").any()
        assert not frame.patient_id.str[-4:].astype(int).mod(5).eq(fold).any()
        info = read(normalized / f"normal/fold={fold}/NORMALIZATION_SOURCES.json")
        _, _, samples = native(record(normalized / f"normal/fold={fold}/NORMALIZATION_SOURCES.json"), normalized)
        assert samples.used_for_fit.sum() == (72 if fold in (0, 4) else 73)
        assert not samples.loc[samples.split_role.isin(["test", "validation"]), "used_for_fit"].any()
        assert not samples.loc[samples.sample_id.str.endswith(("02A", "10A")), "used_for_fit"].any()
        assert info["raw_file_exposure"] == "WHOLE_COUNT_RDS_INCLUDING_HELDOUT_ROWS"
        validate(binding(file), file.parent, file, replay=True)
    inventory = Inventory([], tmp_path)
    file = output(normalized)
    inspect(inventory, binding(file), file.parent, file, "core", "source", 0)
    assert any("77889900" in r["study_tokens_json"] for r in inventory.usage)
    assert not any(r["reason"] == "COHORT_LEVEL_TMM_NORMALIZATION_NOT_FOLD_LOCAL" for r in inventory.gaps)
    assert any(r["path"] == str(normalized / "counts.RDS") for r in inventory.dependencies)


def test_training_detection_never_uses_validation_only_expression(normalized):
    frame = pd.read_parquet(output(normalized))
    excluded = frame.loc[frame.lncrna_id.isin(["LNC:LOW", "LNC:LEAK"])]
    assert not excluded.analysis_eligible.any()
    assert excluded.unavailable_reason.eq("BELOW_TRAIN_DETECTION_THRESHOLD").all()
    assert excluded.loc[excluded.lncrna_id.eq("LNC:LEAK") & excluded.split_role.eq("validation"), "logcpm"].gt(0).all()


@pytest.mark.parametrize("fault", ["output_repin", "fold", "patient_map"])
def test_fold_expression_mismatch_rejected(normalized, tmp_path, fault):
    file = output(normalized)
    if fault == "output_repin":
        changed = tmp_path / "part-0.parquet"
        pd.read_parquet(file).assign(logcpm=99.).to_parquet(changed, index=False)
        document = read(binding(file)["path"]); document["output"] = record(changed)
        receipt = tmp_path / "EXPRESSION_SOURCES.json"; write(receipt, document)
        with pytest.raises(ValueError, match="differs from native conversion"):
            validate(record(receipt), tmp_path, changed, replay=True)
    elif fault == "fold":
        with pytest.raises(ValueError, match="different model fold"):
            inspect(Inventory([], tmp_path), binding(file), tmp_path, file, "core", "source", 1)
    else:
        other = tmp_path / "folds.tsv"; other.write_bytes((normalized / "patient_map.tsv").read_bytes())
        sources = read(normalized / "converted/SOURCE_CONFIG_FRAGMENT.json")["sources"]
        with pytest.raises(ValueError, match="patient map differs"):
            expression_path(sources, "lnc_expression_root", "BRCA", 0, other)


def test_actual_native_statistics_preserve_excluded_candidates_and_fold_dependencies(normalized, tmp_path, monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs, prepared_capture
    from cc_hhgt.v32.preparation_sources import attach_sources
    from cc_hhgt.v32.local_cnv_sources import validate as validate_sources
    args, vargs, train, valid = native_inputs(tmp_path / "native", monkeypatch,
        fold_expression_root=normalized / "converted", patient_map_source=normalized / "patient_map.tsv",
        candidate_ids=["LNC:Z", "LNC:LOW", "LNC:LEAK"])
    for collection in (train["pair_outputs"], valid["outputs"]):
        for item in collection:
            frame = pd.read_parquet(item["path"])
            assert len(frame) == 3
            unavailable = frame.loc[frame.lncrna_id.eq("LNC:LOW")]
            assert unavailable.availability.eq("typed_unavailable").all()
            assert unavailable.unavailable_reason.eq("BELOW_TRAIN_DETECTION_THRESHOLD").all()
            assert frame.loc[frame.lncrna_id.eq("LNC:Z"), "availability"].eq("available").all()
            assert unavailable.filter(regex="^rho_A").isna().all().all()
    receipt, frames = prepared_capture(tmp_path / "prepared", args, vargs, 0)
    for frame in frames.values():
        assert len(frame) == 3
        assert not frame.loc[frame.lncrna_id.isin(["LNC:LOW", "LNC:LEAK"]), "association_available"].any()
    payload = {"patient_fold":0, "input_scope":{"association_mode":"local-cnv-adjusted"}}
    attach_sources(payload, receipt)
    inventory = Inventory([], tmp_path); inventory.preparation(payload, tmp_path, "core", 0, False)
    paths = {r["path"] for r in inventory.dependencies}
    assert str(output(normalized, 0)) in paths
    assert not any(str(output(normalized, f)) in paths for f in range(1, 5))
    _, sources = validate_sources(train["consumed_sources"], tmp_path)
    sources["records"] = [r for r in sources["records"] if not (r["role"] == "lncrna_expression" and r["fold"] == 4)]
    broken = tmp_path / "missing_fold.json"; write(broken, sources)
    with pytest.raises(ValueError, match="fold coverage"):
        validate_sources(record(broken), tmp_path)


def test_actual_fold_local_expression_reaches_core_forward_and_external_isolation(normalized, tmp_path, monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs, test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs
    def native_factory(root, patcher):
        return native_inputs(root, patcher, fold_expression_root=normalized / "converted",
                             patient_map_source=normalized / "patient_map.tsv")
    test_actual_core_and_external_consumer_exclude_studies_only_in_transformation_inputs(
        tmp_path / "downstream", monkeypatch, native_factory=native_factory, external_pmid="77889900")


def test_no_eligible_cnv_lncrnas_preserves_empty_graph_schema(normalized, tmp_path, monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs, prepared_capture
    from cc_hhgt.v32.local_cnv_core_preparation import e1_coexpression_frame
    args, vargs, train, valid = native_inputs(tmp_path / "native", monkeypatch,
        fold_expression_root=normalized / "converted", patient_map_source=normalized / "patient_map.tsv",
        candidate_ids=["LNC:LOW"])
    for item in train["g0_outputs"]:
        edges = e1_coexpression_frame(pd.read_parquet(item["path"]), expected_fold=item["fold"])
        assert edges.empty and edges.rho.dtype.kind == "f"
    _, frames = prepared_capture(tmp_path / "prepared", args, vargs, 0)
    assert all(len(frame) == 1 and not frame.association_available.any() for frame in frames.values())


@pytest.mark.parametrize("fault", ["duplicate_patient", "invalid_fold"])
def test_invalid_fold_authority_stops_before_normalization(normalized, tmp_path, fault):
    folds = pd.read_csv(normalized / "patient_map.tsv", sep="\t")
    if fault == "duplicate_patient": folds.loc[1, "patient_id"] = folds.loc[0, "patient_id"]
    else: folds.loc[0, "patient_fold_id"] = 5
    source = tmp_path / "bad.tsv"; folds.to_csv(source, sep="\t", index=False)
    result = subprocess.run(["/usr/bin/Rscript", str(R_RUNNER), str(normalized / "counts.RDS"), str(source), "BRCA", str(tmp_path / "bad")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0 and "Invalid patient folds" in result.stderr
    assert not (tmp_path / "bad/SUCCESS.json").exists()


def test_host_guard_and_existing_output(normalized, tmp_path, monkeypatch):
    with pytest.raises(FileExistsError): convert([normalized / "normal"], normalized / "converted", runner=RUNNER)
    result = subprocess.run(["/usr/bin/Rscript", "-e",
        'Sys.info<-function() c(nodename="elsewhere");source(commandArgs(trailingOnly=TRUE)[[1]])', str(R_RUNNER)],
        capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and "target host 149" in result.stderr
    monkeypatch.setattr(socket, "gethostname", lambda:"elsewhere")
    with pytest.raises(RuntimeError, match="target host 149"):
        convert([normalized / "normal"], tmp_path / "wrong", runner=RUNNER)
