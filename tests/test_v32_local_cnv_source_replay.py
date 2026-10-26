"""Real native worker/closeout outputs, numerical replay, and tamper rejection."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.local_cnv_sources import partition_binding, inspect, validate
from cc_hhgt.v32.local_cnv_source_replay import replay, FORMAT
from cc_hhgt.v32.current_model_use import Inventory
from test_v32_native_cnv_workflow import completed
from test_v32_local_cnv_cohort import no_fit, CANCERS, LNCS, PATHWAY


def binding(completed, *, decision="correction", kind="pair", fold=0, cancer="BRCA"):
    output = completed[decision][0]
    root = output/("validation" if kind == "validation" else "association")
    layout = {"pair": "pair_level", "graph": "g0_edge_stability", "validation": "validation_associations"}[kind]
    return partition_binding(root/layout/f"cancer={cancer}"/f"fold={fold}"/"part.parquet", root/"receipts"/(cancer+".json"))


@pytest.mark.parametrize("decision", ["correction", "small", "review"])
def test_all_native_partitions_replay_after_actual_workers_and_closeout(completed, decision):
    kinds = ["pair", "graph"]+(["validation"] if decision == "correction" else [])
    for kind in kinds:
        for cancer in CANCERS:
            for fold in range(5):
                value = replay(binding(completed, decision=decision, kind=kind, fold=fold, cancer=cancer), Path.cwd())
                assert value["format"] == FORMAT and value["outputs_compared"] == 1
                assert not value["training_started"] and not value["model_forwards_executed"]
                assert not value["raw_study_ancestry_complete"]
    # Explicitly unavailable rows must survive reconstruction with null effects.
    if decision != "small":
        frame = pd.read_parquet(binding(completed, decision=decision)["output"]["path"])
        assert frame.loc[frame.lncrna_id.eq("LNC:LOW"), "rho_A1_local"].isna().all()


def test_independent_one_sided_rank_projection_and_t_distribution_oracle(completed):
    from scipy.stats import rankdata, t as student_t
    _, _, args = completed["correction"]
    patient_map = pd.read_csv(args.patient_map, sep="\t")
    ids = patient_map.loc[patient_map.cancer_id.eq("BRCA") & ~patient_map.patient_fold_id.isin([0, 1]), "patient_id"].tolist()
    spec = read(args.source_config)["sources"]
    def values(role, entity, key, column):
        table = pd.read_parquet(Path(spec[role]["path"])/"cancer_id=BRCA/part-0.parquet")
        return table.loc[table[entity].eq(key)].set_index("patient_id").loc[ids, column].to_numpy(np.float32)
    x = values("lnc_expression_root", "lncrna_id", "LNC:A", "logcpm")
    y = values("pathway_activity_root", "pathway_id", PATHWAY, "activity_score")
    compact = args.directional_root/"cancer=BRCA"
    positions = {v:i for i,v in enumerate(read(compact/"patient_ids.json"))}
    z = np.load(compact/"lncrna_signed_mean.npy")[[positions[p] for p in ids], LNCS.index("LNC:A")]
    cov = pd.read_parquet(args.covariates).set_index("patient_id").loc[ids, "purity"].to_numpy()
    design = np.column_stack([np.ones(len(ids)), cov, rankdata(z)])
    # A1 removes local CNV only from expression; activity retains its C0
    # residual. Symmetric partial correlation would test a different estimand.
    designs = (design, design[:, :2])
    residuals = [rankdata(v)-d @ np.linalg.lstsq(d, rankdata(v), rcond=None)[0] for v,d in zip((x,y), designs)]
    rho = np.corrcoef(*residuals)[0,1]
    df = len(ids)-np.linalg.matrix_rank(design)-1
    p = 2*student_t.sf(abs(rho)*np.sqrt(df/(1-rho*rho)), df)
    observed = pd.read_parquet(binding(completed)["output"]["path"]).set_index("lncrna_id").loc["LNC:A"]
    assert observed.n_A1_local == len(ids) and observed.correlation_df_A1_local == df
    np.testing.assert_allclose([observed.rho_A1_local, observed.p_A1_local], [rho,p], rtol=1e-10, atol=1e-12)


def altered(completed, directory, kind, change):
    original = binding(completed, kind=kind)
    sources = read(original["sources"]["path"]); receipt = read(original["receipt"]["path"])
    target = next(r for r in sources["outputs"] if r["fold"] == 0 and r["output_kind"] == kind)
    frame = pd.read_parquet(target["path"])
    change(frame, target, sources)
    output = directory/"changed.parquet"; frame.to_parquet(output, index=False)
    target.update(record(output))
    source_path = directory/"sources.json"; write(source_path, sources)
    field = {"pair": "pair_outputs", "graph": "g0_outputs", "validation": "outputs"}[kind]
    receipt[field] = [deepcopy(target) if r["fold"] == 0 else r for r in receipt[field]]
    receipt["consumed_sources"] = record(source_path)
    receipt_path = directory/"receipt.json"; write(receipt_path, receipt)
    return {"receipt": record(receipt_path), "sources": record(source_path), "output": record(output)}


@pytest.mark.parametrize("kind,column", [("pair", "rho_A1_local"), ("pair", "p_A1_local"),
    ("pair", "label_A1_local"), ("pair", "n_A1_local"), ("pair", "availability"),
    ("pair", "rho_A0_validation"), ("pair", "rho_A2_local_pathway"),
    ("validation", "rho_A1_validation"), ("validation", "validation_patient_n"),
    ("validation", "validation_patient_sha256"), ("graph", "rho_E1"), ("graph", "present_E1")])
def test_simultaneously_repinned_output_and_receipts_still_fail(completed, tmp_path, kind, column):
    def change(frame, target, sources):
        assert len(frame)
        finite = frame.index[frame[column].notna()]
        i = finite[0] if len(finite) else frame.index[0]
        if pd.api.types.is_bool_dtype(frame[column]): frame.loc[i,column] = not frame.loc[i,column]
        elif pd.api.types.is_numeric_dtype(frame[column]): frame.loc[i,column] = 1 if pd.isna(frame.loc[i,column]) else frame.loc[i,column]+1
        else: frame.loc[i,column] = "counterfeit"
    item = altered(completed, tmp_path, kind, change)
    validate(item["sources"], tmp_path)  # Every ordinary file/hash check still passes.
    with pytest.raises(ValueError, match="numerical replay differs"):
        replay(item, tmp_path)


@pytest.mark.parametrize("fault", ["row", "column", "metadata", "missing_metadata", "implementation", "missing_replay", "fold"])
def test_contract_and_full_output_coverage_fail_closed(completed, tmp_path, fault):
    def change(frame, target, sources):
        if fault == "row": frame.drop(index=frame.index[-1], inplace=True)
        elif fault == "column": frame.drop(columns=["rho_A1_local"], inplace=True)
        elif fault == "metadata": target["base_design_rank"] += 1
        elif fault == "missing_metadata": target.pop("base_design_rank")
        elif fault == "implementation":
            substitute = tmp_path/"associations.py"; substitute.write_text("different implementation")
            for item in sources["implementations"]:
                if Path(item["path"]).name == "associations.py": item.update(record(substitute))
        elif fault == "missing_replay": sources.pop("numeric_replay_format")
        else: target["fold"] = 4
    item = altered(completed, tmp_path, "pair", change)
    with pytest.raises(ValueError): replay(item, tmp_path)


def test_inventory_replays_and_preserves_earlier_ancestry_gap(completed, tmp_path):
    inventory = Inventory([], tmp_path)
    for kind in ("pair", "graph", "validation"):
        inspect(inventory, binding(completed, kind=kind), tmp_path, "core", "local_cnv", 0)
    rows = pd.DataFrame(inventory.transform_replays)
    rows = rows.loc[rows.format.eq(FORMAT)]
    assert len(rows) == 3 and rows.outputs_compared.eq(1).all()
    assert {r["reason"] for r in inventory.gaps} >= {"LOCAL_CNV_EARLIER_RAW_ANCESTRY_AND_CROSS_MODULE_PATIENT_IDENTITY_UNVERIFIED"}
    assert inventory.statistical_patient_scopes
    assert "LOCAL_CNV_NUMERIC_REPLAY_NOT_BOUND" not in {r["reason"] for r in inventory.gaps}
    assert sum(".validation_parent." in r["role"] for r in inventory.dependencies) == 3


@pytest.mark.parametrize("kind", ["pair", "graph", "validation"])
def test_replay_only_reads_tables_for_the_consumed_output_kind(completed, tmp_path, monkeypatch, kind):
    item = binding(completed, kind=kind)
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        text = str(path)
        if kind == "graph": assert "pathway_activity_root" not in text and "candidates.parquet" not in text
        else: assert "gene_expression_root" not in text
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_parquet", guarded)
    assert replay(item, tmp_path)["output_kind"] == kind


def test_cache_rechecks_actual_bytes_and_command_reports_pass(completed, tmp_path, monkeypatch):
    item = binding(completed); cache = set()
    result = replay(item, tmp_path, cache=cache)
    import scripts.run_v32_local_cnv_confounding_audit_r2 as association
    def forbidden(*a, **k): raise AssertionError("Unchanged partition must reuse numerical proof")
    with monkeypatch.context() as patch:
        patch.setattr(association, "association_fold", forbidden)
        assert replay(item, tmp_path, cache=cache) == result
    path = Path(item["output"]["path"]); original = path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError): replay(item, tmp_path, cache=cache)
    finally: path.write_bytes(original)
    from scripts.audit_local_cnv_source_replay import main
    output = tmp_path/"audit.json"
    assert main(["--receipt", item["receipt"]["path"], "--receipt-sha256", item["receipt"]["sha256"],
                 "--partition", str(path), "--output", str(output)]) == 0
    assert read(output)["status"] == "PASS" and not read(output)["raw_study_ancestry_complete"]


def test_wrong_host_stops_before_reading_inputs(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda:"windows")
    with pytest.raises(RuntimeError, match="149"): replay({}, Path.cwd())


def test_historical_eight_implementation_receipt_keeps_explicit_gap(completed, tmp_path):
    excluded = {"local_cnv_source_replay.py", "local_cnv_patient_use.py", "associations.py", "fold_local_expression.py", "fold_local_activity.py",
                "local_cnv_core_preparation.py"}
    def change(frame, target, sources):
        sources.pop("numeric_replay_format")
        sources["implementations"] = [r for r in sources["implementations"] if Path(r["path"]).name not in excluded]
        assert len(sources["implementations"]) == 8
    item = altered(completed, tmp_path, "pair", change)
    assert replay(item, tmp_path) is None
    from cc_hhgt.v32.local_cnv_source_replay import inspect as inspect_replay
    inventory = Inventory([], tmp_path)
    assert not inspect_replay(inventory, item, tmp_path, "core", "local_cnv", 0)
    assert inventory.gaps[0]["reason"] == "LOCAL_CNV_NUMERIC_REPLAY_NOT_BOUND"


def test_failed_command_records_numerical_difference(completed, tmp_path):
    def change(frame, target, sources): frame.loc[0,"rho_A1_local"] = .987
    item = altered(completed, tmp_path, "pair", change)
    from scripts.audit_local_cnv_source_replay import main
    output = tmp_path/"audit.json"
    with pytest.raises(ValueError, match="numerical replay differs"):
        main(["--receipt", item["receipt"]["path"], "--receipt-sha256", item["receipt"]["sha256"],
              "--partition", item["output"]["path"], "--output", str(output)])
    assert read(output)["status"] == "FAILED" and not read(output)["training_started"]


def test_cached_validation_proof_rechecks_parent_closeout(completed, tmp_path):
    item = binding(completed, kind="validation"); cache = set()
    replay(item, tmp_path, cache=cache)
    parent = Path(read(item["receipt"]["path"])["local_cnv_authority"]["root"])/"SUCCESS.json"
    original = parent.read_bytes()
    try:
        parent.write_bytes(original+b"changed")
        with pytest.raises(RuntimeError, match="SUCCESS SHA256 drift"): replay(item, tmp_path, cache=cache)
    finally: parent.write_bytes(original)
