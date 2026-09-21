from pathlib import Path

from cc_hhgt.common import input_path, load_config


def test_config_inheritance_deep_merges(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text(
        f"project_root: {tmp_path}\n"
        "results_dir: old\n"
        "training:\n"
        "  hidden_channels: 64\n"
        "  epochs: 60\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "results_dir: new\n"
        "training:\n"
        "  epochs: 10\n",
        encoding="utf-8",
    )
    cfg = load_config(child)
    assert cfg["training"]["hidden_channels"] == 64
    assert cfg["training"]["epochs"] == 10
    assert cfg["_results"].name == "new"
    assert cfg["_config_lineage"] == [base.resolve(), child.resolve()]


def test_optional_input_root_is_independent_from_output_project_root(tmp_path: Path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "covariates.tsv").write_text("x\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"project_root: {tmp_path}\n"
        "input_root: snapshot\n"
        "inputs:\n"
        "  bulk_covariates: covariates.tsv\n",
        encoding="utf-8",
    )
    cfg = load_config(config)
    assert input_path(cfg, "bulk_covariates") == snapshot / "covariates.tsv"
