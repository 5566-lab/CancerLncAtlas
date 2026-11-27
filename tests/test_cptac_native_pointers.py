import json
from scripts.refresh_cptac_native_bundle import pin, stage_pointer_plan


def test_all_current_cptac_bindings_pin_the_replacement_config(tmp_path):
    active = tmp_path / "active"
    staged = tmp_path / "staged"
    active.mkdir()
    staged.mkdir()
    runtime = active / "CURRENT_CONFIG.json"
    runtime.write_text("{}")
    replacement = staged / "CURRENT_149_CONFIG.json"
    replacement.write_text('{"version":2}')
    old = {"cptac_input": {"id": "old"}, "current_config": pin(runtime)}
    new = {"cptac_input": {"id": "new"}, "current_config": {**pin(replacement), "path": str(runtime)},
           "validation": {"status": "PASS"}}
    for name in ["CPTAC_CURRENT.json", "RPPA_CURRENT.json", "STATE_CURRENT.json"]:
        (active / name).write_text(json.dumps({**old, "retained": name}))
    (staged / "CPTAC_CURRENT.json").write_text(json.dumps(new))
    stage_pointer_plan(active / "CPTAC_CURRENT.json", runtime, staged)
    plan = json.loads((staged / "STAGED_NATIVE_POINTERS.json").read_text())
    assert len(plan["targets"]) == 4
    for name in ["RPPA_CURRENT.json", "STATE_CURRENT.json"]:
        actual = json.loads((staged / name).read_text())
        assert actual["retained"] == name
        assert actual["current_config"] == new["current_config"]
        assert actual["cptac_input"] == new["cptac_input"]
        assert json.loads((active / name).read_text())["cptac_input"] == old["cptac_input"]
