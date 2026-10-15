import numpy as np
import pandas as pd
import pytest
import torch
import json
from dataclasses import asdict

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.drug_attribution import ablate_drug_targets


class LinearFixtureHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.tensor([1.]))

    def forward(self, core, domain):
        assert not torch.is_grad_enabled()
        return ((core[:, 4]+domain[:, 4])*self.coefficient).unsqueeze(-1)


def fixture():
    lookup = lambda ids, values: base.EmbeddingLookup(np.asarray(values, np.float32), {key: i for i, key in enumerate(ids)})
    core = base.FoldCoreEmbeddings(0,
        lookup(["L1"], [[1., 2.]]), lookup(["G1", "G2"], [[1., 0.], [3., 0.]]),
        lookup(["BRCA"], [[4., 5.]]), "a"*64, "b"*64, {})
    queries = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["L1"], "drug_id": ["DC:1"]})
    domain = np.array([[np.log1p(20), 1, 2., 1., np.log1p(2), 1]], np.float32)
    targets = pd.DataFrame({"drug_id": ["DC:1", "DC:1", "DC:1"], "gene_id": ["G1", "G2", "G1"]})
    exact = pd.DataFrame({"cancer_id": ["BRCA"]*2, "lncrna_id": ["L1"]*2, "pathway_id": ["Pboth", "Pone"]})
    membership = pd.DataFrame({"pathway_id": ["Pboth", "Pboth", "Pone"], "gene_id": ["G1", "G2", "G1"]})
    return core, queries, domain, targets, exact, membership


def run(**changes):
    core, queries, domain, targets, exact, membership = fixture()
    args = dict(head=LinearFixtureHead(), core=core, queries=queries, domain=domain,
        assay_available=np.array([True]), targets=targets, exact=exact, membership=membership,
        domain_mean=np.zeros(6), domain_scale=np.ones(6))
    args.update(changes)
    return ablate_drug_targets(**args)


def test_target_and_pathway_effects_are_actual_forward_passes_not_a_join():
    baseline, effects, audit = run()
    assert baseline.baseline_logit.iloc[0] == pytest.approx(2+np.log(3), abs=1e-6)
    target = effects.loc[effects.ablation_kind.eq("TARGET")].set_index("ablation_entity_id")
    assert target.loc["G1", "delta_logit"] == pytest.approx(-1+np.log(3/2), abs=1e-6)
    assert target.loc["G2", "delta_logit"] == pytest.approx(1+np.log(3/2), abs=1e-6)
    pathway = effects.loc[effects.ablation_entity_id.eq("Pboth")].iloc[0]
    assert pathway.delta_logit == pytest.approx(2+np.log(3), abs=1e-6)
    assert pathway.delta_logit != pytest.approx(target.delta_logit.sum())
    assert pathway.targetless_counterfactual
    assert audit["parameters_unchanged"] and not audit["optimizer_started"]
    assert audit["forward_counterfactuals_executed"] == 4
    assert not effects.causal_claimed.any()


def test_same_membership_different_head_changes_attribution():
    first = run()[1]
    model = LinearFixtureHead()
    with torch.no_grad(): model.coefficient.fill_(2.)
    second = run(head=model)[1]
    np.testing.assert_allclose(second.delta_logit, first.delta_logit*2, atol=1e-6)


def test_no_assay_support_is_null_not_zero_contribution():
    baseline, effects, audit = run(assay_available=np.array([False]))
    assert not baseline.available.any() and baseline.baseline_probability.isna().all()
    assert effects.empty and audit["forward_counterfactuals_executed"] == 0


def test_wrong_target_count_cannot_change_baseline_silently():
    _, _, domain, _, _, _ = fixture(); domain[0, 4] = np.log1p(99)
    with pytest.raises(ValueError, match="actual target encoding"):
        run(domain=domain)


def test_invalid_scaler_or_assay_mask_rejected():
    with pytest.raises(ValueError, match="scaler"):
        run(domain_scale=np.zeros(6))
    with pytest.raises(ValueError, match="boolean mask"):
        run(assay_available=np.array(["False"]))


@pytest.mark.parametrize("pharmacology_enabled", [False, True])
def test_complete_command_loads_five_serialized_heads_without_optimizer(tmp_path, monkeypatch, pharmacology_enabled):
    from tests.test_v32_current_core_export import export_fixture
    from tests.test_v32_drug_response_analysis import fixture as response_fixture
    from cc_hhgt.v32.integrated_model import build_private_auxiliary_head
    from cc_hhgt.v32.drug_training_streaming import StreamingDrugTrainingConfig
    from cc_hhgt.v32.input_lineage import artifact_sha256
    from scripts.analyze_v32_drug_model_attribution import main

    core_path, core_manifest = export_fixture(tmp_path)
    data = response_fixture()
    data["expression"]["lncrna_id"] = "LNC:L1"
    exact = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:L1"], "pathway_id": ["P1"]})
    queries = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:L1"], "drug_id": ["DC:1"]})
    tables = {"exact_candidates": exact, "queries": queries,
        "exact_pathway_membership": pd.DataFrame({"pathway_id": ["P1"], "gene_id": ["GENE:G1"]}),
        "raw_drug_response": data["response"], "raw_lncrna_expression": data["expression"],
        "cell_line_map": data["mapping"], "drug_gene_target": pd.DataFrame({"drug_id": ["DC:1"], "gene_id": ["GENE:G1"]})}
    record = lambda p: {"path": str(p), "sha256": artifact_sha256(p)}
    artifacts = {"core_embedding_manifest": record(core_path)}
    for name, frame in tables.items():
        path = tmp_path/(name+".parquet"); frame.to_parquet(path, index=False); artifacts[name] = record(path)
    config = StreamingDrugTrainingConfig(hidden_features=4, min_total_cell_lines=4, min_cell_lines_per_dataset=4)
    config_path = tmp_path/"config.json"; config_path.write_text(json.dumps({"training": asdict(config)}))
    artifacts["config"] = record(config_path)
    pharmacology = None; domain_features = list(base._DOMAIN_FEATURES)
    if pharmacology_enabled:
        from test_v32_drug_pharmacology import feature_sources
        from cc_hhgt.v32.drug_pharmacology import load_pharmacology, FEATURES, TARGET_POOLS
        pharma_path = feature_sources(tmp_path/"pharma", tables["drug_gene_target"], data["response"])
        artifacts["pharmacology_manifest"] = record(pharma_path)
        pharmacology = load_pharmacology(pharma_path, targets=base.normalise_drug_targets(tables["drug_gene_target"]),
            response=base.normalise_raw_response(data["response"]), mapping=base.normalise_cell_line_map(data["mapping"]))
        domain_features += list(FEATURES)
    records = []
    # These are hand-constructed synthetic source checkpoints. Their test-only
    # receipts emulate a prior completed run; no optimizer is called here.
    for fold in range(5):
        head, _ = build_private_auxiliary_head("drug", core_features=6+(2*len(TARGET_POOLS) if pharmacology_enabled else 0), domain_features=len(domain_features), hidden_features=4,
            dropout=config.dropout, seed=config.seed+fold*101)
        path = tmp_path/f"head_{fold}.pt"
        torch.save({"checkpoint_format": base.PRIVATE_CHECKPOINT_FORMAT, "cell_line_fold": fold,
            "core_parameter_sha256": core_manifest["folds"][str(fold)]["core_parameter_sha256"],
            "core_checkpoint_sha256": core_manifest["folds"][str(fold)]["checkpoint_sha256"],
            "domain_features": domain_features, "domain_mean": np.zeros(len(domain_features), np.float32),
            "domain_scale": np.ones(len(domain_features), np.float32), "model_state": head.state_dict(),
            "pharmacology_features":pharmacology.receipt() if pharmacology is not None else None}, path)
        records.append({"cell_line_fold": fold, "optimizer_steps": 1, **record(path)})
    checkpoint_path = tmp_path/"checkpoints.json"; checkpoint_path.write_text(json.dumps({"records": records}))
    artifacts["checkpoint_manifest"] = record(checkpoint_path)
    bound_inputs = {name: artifacts[name]["sha256"] for name in [*tables, "core_embedding_manifest"] if name != "queries"}
    lineage = {"training_run_id": "drug-fixture", "training_status": "SUCCESS",
        "pharmacology_features":pharmacology.receipt() if pharmacology is not None else None,
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_path), "config_sha256": artifact_sha256(config_path),
        "input_hashes_start": bound_inputs, "input_hashes_end": bound_inputs}
    path = tmp_path/"lineage.json"; path.write_text(json.dumps(lineage)); artifacts["drug_lineage"] = record(path)
    manifest = {"format": "DRUG_MODEL_ATTRIBUTION_INPUT_V1", "core_run_id": "current-G2",
                "drug_run_id": "drug-fixture", "artifacts": artifacts}
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest))
    def no_optimizer(*args, **kwargs): raise AssertionError("Attribution must not start an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", no_optimizer)
    output = tmp_path/"attribution"
    assert main(["--manifest", str(path), "--output", str(output)]) == 0
    audit = json.loads((output/"AUDIT.json").read_text())
    assert len(audit["fold_audits"]) == 5 and not audit["training_started"]
    effects = pd.concat([pd.read_parquet(p) for p in output.glob("attribution_fold*.parquet")])
    assert len(effects) == 10
    assert effects.cell_line_fold.nunique() == 5
    assert set(effects.ablation_kind) == {"TARGET", "EXACT_PATHWAY_TARGET_SET"}
    assert effects.target_pharmacology_recomputed.eq(pharmacology_enabled).all()
