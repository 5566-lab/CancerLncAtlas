from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_detectable_lncRNA_state_universe():
    module = load_script("state_data_v29", "scripts/27_build_lncrna_state_data.py")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 80))
    y = rng.normal(size=(50, 4))
    y[:, 0] = 0.8 * x[:, 5] + rng.normal(scale=0.4, size=50)
    detection = np.ones(80)
    frame = module.build_pair_universe(
        "LUAD", x, y, [f"L{i}" for i in range(80)],
        ["EXTEND", "RNAss", "DNAss", "EREG"], detection,
    )
    assert len(frame) == 80 * 4
    assert frame.groupby("state_id").lncrna_id.nunique().eq(80).all()
    assert frame.selected_for_model.sum() > 0


def test_state_network_has_patient_and_three_graph_heads():
    from cc_hhgt_v26.state_model import StateExpertNetwork

    network = StateExpertNetwork(
        patient_dim=9,
        graph_dims={"rgcn": 12, "hgt": 12, "cc_hhgt_strict": 12},
    )
    output = network(
        torch.randn(13, 9),
        {
            "rgcn": torch.randn(13, 60),
            "hgt": torch.randn(13, 60),
            "cc_hhgt_strict": torch.randn(13, 60),
        },
    )
    assert output["patient_logit"].shape == (13,)
    for model in ("rgcn", "hgt", "cc_hhgt_strict"):
        assert output[f"{model}_logit"].shape == (13,)


def test_masked_moe_assigns_zero_to_unavailable_state_expert():
    from cc_hhgt_v26.moe_model import AvailabilityMaskedMoE

    network = AvailabilityMaskedMoE(n_experts=4, quality_dim=3)
    logits = torch.randn(11, 4)
    availability = torch.ones(11, 4)
    availability[:4, 2] = 0
    _, weights = network(logits, availability, torch.randn(11, 3))
    assert torch.all(weights[:4, 2] == 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(11), atol=1e-6)


def test_random_effects_meta_analysis_returns_finite_effect():
    import ast
    import math
    from scipy import stats

    source = (ROOT / "scripts/47_build_v29_lncrna_state_significance.py").read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "random_effects_correlation"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np, "math": math, "stats": stats, "dict": dict, "float": float, "int": int, "len": len, "max": max, "abs": abs}
    exec(compile(module, "<random_effects_correlation>", "exec"), namespace)
    result = namespace["random_effects_correlation"](
        np.array([0.20, 0.25, 0.18, 0.22, 0.24]),
        np.array([30, 35, 28, 40, 33]),
    )
    assert result["k"] == 5
    assert 0.15 < result["effect"] < 0.30
    assert result["ci_lower"] < result["effect"] < result["ci_upper"]
    assert 0 <= result["p_value"] <= 1
