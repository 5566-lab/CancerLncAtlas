"""Variable scope through actual disk staging and aggregation; no model fitting."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import genomic_resource_staging as stage
from cc_hhgt.v32.genomic_partition_training import _assert_empty_resource_stage
from test_v32_native_auxiliary_fusion import no_fitting, read, write, record


def inputs(root, *, legacy=False):
    counts = {"BRCA": 7, "COAD": 7, "LUAD": 7} if legacy else {"BRCA": 7, "COAD": 4, "LUAD": 9}
    keys = pd.DataFrame([(c, f"L{i}", f"P{i%3}") for c,n in counts.items() for i in range(n)], columns=stage.TARGET_KEYS)
    candidate = root/"candidates.parquet"; keys.to_parquet(candidate, index=False)
    ordered = pd.concat([stage._canonical_partition(keys.loc[keys.cancer_id.eq(c)], c) for c in counts])
    digest = hashlib.sha256("".join("\t".join(r)+"\n" for r in ordered.itertuples(index=False, name=None)).encode()).hexdigest()
    cnv = write(root/"streaming.json", {"format": stage.CNV_STREAMING_FORMAT, "status":"SUCCESS",
        "cancers": {c:{} for c in counts}, "required_cancers":list(counts), "patient_folds":5,
        "typed_unavailable_is_never_zero":True, "fixture_metadata_only_not_actual_cnv_preparation":True})
    budget = write(root/"budget.json", {"format":stage.BUDGET_FORMAT, "resource_budget_id":"variable-synthetic",
        "scope":{"formal_cancers":list(counts), "candidate_rows":sum(counts.values()),
                 **({"candidate_rows_per_cancer":7} if legacy else {"candidate_rows_by_cancer":counts})},
        "adapter_budget":{"max_candidate_partition_rows":9, "max_declared_adapter_peak_rss_bytes":10**12,
                          "max_memmap_bytes":10**8, "max_staging_disk_bytes":10**8},
        "downstream_budget":{"max_declared_process_rss_bytes":10**12, "minimum_host_available_memory_bytes":1,
                             "minimum_host_free_disk_bytes":1, "forbid_launch_without_resource_success":True},
        "memmap_layout":{"modalities":["mutation","cnv"], "patient_folds":5,
                         "probability_dtype":"float32", "available_dtype":"bool", "processed_dtype":"bool"},
        "input_bindings":{"candidate_authority":{**record(candidate), "ordered_exact_candidate_key_sha256":digest},
            "cnv_streaming_aggregate_success":{**record(cnv), "format":stage.CNV_STREAMING_FORMAT,
                "status":"SUCCESS", "sha256_policy":"CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH"}}})
    return dict(candidate_authority_path=candidate,
        genomic_training_source_path=Path(stage.__file__).with_name("genomic_training.py"),
        cnv_streaming_success_path=cnv, cnv_streaming_success_sha256=record(cnv)["sha256"],
        budget_contract_path=budget, output_root=root/"stage", enforce_host_budget=True), keys, counts


@pytest.mark.parametrize("legacy", [False, True])
def test_actual_cli_and_trainer_empty_stage_accept_scope_and_preserve_each_fold(tmp_path, legacy):
    import scripts.manage_v32_genomic_resource_memmaps as cli
    args, keys, counts = inputs(tmp_path, legacy=legacy)
    assert cli.main(["stage", "--candidate-authority", str(args["candidate_authority_path"]),
        "--genomic-training-source", str(args["genomic_training_source_path"]),
        "--cnv-streaming-success", str(args["cnv_streaming_success_path"]),
        "--cnv-streaming-success-sha256", args["cnv_streaming_success_sha256"],
        "--budget-contract", str(args["budget_contract_path"]), "--output-root", str(args["output_root"])]) == 0
    success = tmp_path/"stage/SUCCESS.json"
    _, manifest = stage._load_stage(success, verify_external_hashes=True)
    _assert_empty_resource_stage(manifest, success.parent)
    assert manifest["candidate_rows_by_cancer"] == counts
    assert manifest["candidate_rows_per_cancer"] == (7 if legacy else None)
    assert [r["global_row_start"] for r in manifest["candidate_partitions"]] == [0,7,14 if legacy else 11]
    assert not read(success)["formal_training_started"]
    expected = {}
    for fold in range(5):
        for mi, modality in enumerate(stage.MODALITIES):
            for ci, (cancer, local) in enumerate(keys.groupby("cancer_id", sort=False)):
                local = stage._canonical_partition(local, cancer)
                values = (.05 + mi*.01 + ci*.2 + fold*.1 + np.arange(len(local))*.001).astype(np.float32)
                available = np.full(len(local), cancer != "COAD")
                if fold == 4: available[0] = False
                if fold == 0 and cancer == "BRCA": values[0] = 0.
                values[~available] = np.nan
                expected[(fold,modality,cancer)] = values
                stage.write_genomic_memmap_partition(staging_success_path=success, modality=modality,
                    patient_fold=fold, cancer_id=cancer, candidate_keys=local, probability=values, available=available)
    stage.seal_genomic_memmap_predictions(staging_success_path=success)
    assert cli.main(["reassemble", "--predictions-ready", str(tmp_path/"stage/PREDICTIONS_READY.json"),
        "--output-root", str(tmp_path/"assembled"), "--training-run-id", "v32-scope-fixture"]) == 0
    combined = pd.read_parquet(tmp_path/"assembled/mutation_cnv_typed_predictions.parquet")
    assert len(combined) == sum(counts.values())
    for fold in range(5):
        frame = pd.read_parquet(tmp_path/f"assembled/patient_fold_oof_predictions/patient_fold={fold}/part-0.parquet")
        pd.testing.assert_frame_equal(frame[list(stage.TARGET_KEYS)], combined[list(stage.TARGET_KEYS)])
        for modality in stage.MODALITIES:
            for cancer in counts:
                local = frame.loc[frame.cancer_id.eq(cancer)]
                np.testing.assert_array_equal(local[modality+"_context_probability"], expected[(fold,modality,cancer)])
                np.testing.assert_array_equal(local[modality+"_available"], np.isfinite(expected[(fold,modality,cancer)]))
    for modality in stage.MODALITIES:
        for cancer in counts:
            values = np.stack([expected[(f,modality,cancer)] for f in range(5)])
            n = np.isfinite(values).sum(0)
            mean = np.divide(np.nansum(values.astype(float), axis=0), n, out=np.full(counts[cancer],np.nan), where=n>0).astype(np.float32)
            local = combined.loc[combined.cancer_id.eq(cancer)]
            np.testing.assert_array_equal(local[modality+"_context_probability"], mean)
            np.testing.assert_array_equal(local[modality+"_patient_folds_with_prediction"], n)
    assert not (tmp_path/"stage/TRAINING_SUCCESS.json").exists()


@pytest.mark.parametrize("fault", ["missing_cancer", "extra_cancer", "zero", "negative", "fractional", "bool",
    "wrong_total", "conflicting_uniform", "partition_budget", "unsafe_cancer"])
def test_invalid_budget_scope_fails_before_stage_creation(tmp_path, fault):
    args, _, _ = inputs(tmp_path); path = args["budget_contract_path"]; budget = read(path); scope = budget["scope"]
    if fault == "missing_cancer": scope["candidate_rows_by_cancer"].pop("COAD")
    elif fault == "extra_cancer": scope["candidate_rows_by_cancer"]["LAML"] = 1
    elif fault in {"zero", "negative", "fractional", "bool"}:
        scope["candidate_rows_by_cancer"]["COAD"] = {"zero":0,"negative":-1,"fractional":4.5,"bool":True}[fault]
    elif fault == "wrong_total": scope["candidate_rows"] += 1
    elif fault == "conflicting_uniform": scope["candidate_rows_per_cancer"] = 7
    elif fault == "partition_budget": budget["adapter_budget"]["max_candidate_partition_rows"] = 8
    elif fault == "unsafe_cancer": scope["formal_cancers"][0] = "../BRCA"
    write(path, budget)
    with pytest.raises(stage.GenomicResourceError): stage.stage_genomic_cancer_memmaps(**args)
    assert not args["output_root"].exists() and not (tmp_path/".stage.staging").exists()


@pytest.mark.parametrize("fault", ["extra_cancer", "missing_row", "null_cancer", "null_key", "empty_key", "duplicate_key"])
def test_candidate_authority_cannot_drop_or_relabel_rows_in_declared_scope(tmp_path, fault):
    args, keys, _ = inputs(tmp_path); path = args["candidate_authority_path"]
    if fault == "extra_cancer": keys.loc[len(keys)] = ["LAML", "L0", "P0"]
    elif fault == "missing_row": keys = keys.iloc[1:]
    elif fault == "null_cancer": keys.loc[0,"cancer_id"] = None
    elif fault == "null_key": keys.loc[0,"lncrna_id"] = None
    elif fault == "empty_key": keys.loc[0,"pathway_id"] = " "
    elif fault == "duplicate_key": keys.loc[1] = keys.loc[0]
    keys.to_parquet(path,index=False)
    budget = read(args["budget_contract_path"]); budget["input_bindings"]["candidate_authority"].update(record(path))
    write(args["budget_contract_path"],budget)
    with pytest.raises(stage.GenomicResourceError): stage.stage_genomic_cancer_memmaps(**args)
    assert not args["output_root"].exists()


@pytest.mark.parametrize("fault", ["overlap", "gap", "swapped_ranges", "wrong_count", "wrong_total", "wrong_map", "wrong_max", "wrong_shape", "wrong_dtype"])
def test_repinned_partition_scope_is_rejected_before_any_prediction_write(tmp_path, fault):
    args, keys, _ = inputs(tmp_path); stage.stage_genomic_cancer_memmaps(**args)
    manifest_path = tmp_path/"stage/RESOURCE_MANIFEST.json"; manifest = read(manifest_path)
    rows = manifest["candidate_partitions"]
    if fault == "overlap": rows[1]["global_row_start"] -= 1; rows[1]["global_row_stop_exclusive"] -= 1
    elif fault == "gap": rows[1]["global_row_start"] += 1; rows[1]["global_row_stop_exclusive"] += 1
    elif fault == "swapped_ranges":
        for key in ("global_row_start", "global_row_stop_exclusive"): rows[0][key], rows[1][key] = rows[1][key], rows[0][key]
    elif fault == "wrong_count": rows[1]["rows"] += 1
    elif fault == "wrong_total": manifest["candidate_rows"] += 1
    elif fault == "wrong_map": manifest["candidate_rows_by_cancer"]["BRCA"] += 1
    elif fault == "wrong_max": manifest["max_candidate_partition_rows"] -= 1
    elif fault == "wrong_shape": manifest["memmaps"]["fold_probability"]["shape"][-1] += 1
    elif fault == "wrong_dtype": manifest["memmaps"]["fold_available"]["dtype"] = "float32"
    write(manifest_path,manifest)
    success_path = tmp_path/"stage/SUCCESS.json"; success = read(success_path)
    success["resource_manifest_sha256"] = record(manifest_path)["sha256"]
    success["candidate_rows"] = manifest["candidate_rows"]; write(success_path,success)
    before = {name:record(Path(r["path"]))["sha256"] for name,r in manifest["memmaps"].items()}
    local = keys.loc[keys.cancer_id.eq("BRCA")]
    with pytest.raises(stage.GenomicResourceError):
        stage.write_genomic_memmap_partition(staging_success_path=success_path, modality="mutation",
            patient_fold=0, cancer_id="BRCA", candidate_keys=local, probability=np.full(len(local),.5), available=np.ones(len(local),dtype=bool))
    assert before == {name:record(Path(r["path"]))["sha256"] for name,r in manifest["memmaps"].items()}


def test_historical_uniform_manifest_without_new_optional_fields_remains_readable(tmp_path):
    args, _, _ = inputs(tmp_path, legacy=True); stage.stage_genomic_cancer_memmaps(**args)
    path = tmp_path/"stage/RESOURCE_MANIFEST.json"; manifest = read(path)
    manifest.pop("candidate_rows_by_cancer"); manifest.pop("max_candidate_partition_rows"); write(path,manifest)
    success = tmp_path/"stage/SUCCESS.json"; value = read(success)
    value["resource_manifest_sha256"] = record(path)["sha256"]; write(success,value)
    _, loaded = stage._load_stage(success, verify_external_hashes=True)
    _assert_empty_resource_stage(loaded, success.parent)


def test_unused_categorical_levels_do_not_create_phantom_cancer_rows(tmp_path):
    args, keys, counts = inputs(tmp_path)
    keys["cancer_id"] = pd.Categorical(keys.cancer_id, categories=[*counts,"LAML"])
    candidate = args["candidate_authority_path"]; keys.to_parquet(candidate,index=False)
    budget_path = args["budget_contract_path"]; budget = read(budget_path)
    budget["input_bindings"]["candidate_authority"].update(record(candidate)); write(budget_path,budget)
    stage.stage_genomic_cancer_memmaps(**args)
    _, manifest = stage._load_stage(tmp_path/"stage/SUCCESS.json", verify_external_hashes=True)
    assert manifest["candidate_rows_by_cancer"] == counts
    _assert_empty_resource_stage(manifest, tmp_path/"stage")
