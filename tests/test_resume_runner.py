import importlib.util
from pathlib import Path


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "21_resume_loco_models.py"
    spec = importlib.util.spec_from_file_location("resume_loco_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_complete_artifact_policy(tmp_path):
    runner = _load_runner()
    model_dir = tmp_path / "fold"
    model_dir.mkdir()
    assert not runner.is_complete(model_dir, "cc_hhgt")
    for name in runner.expected_artifacts("cc_hhgt"):
        (model_dir / name).write_text("x", encoding="utf-8")
    assert runner.is_complete(model_dir, "cc_hhgt")
    (model_dir / "metrics.tsv").write_text("", encoding="utf-8")
    assert not runner.is_complete(model_dir, "cc_hhgt")
