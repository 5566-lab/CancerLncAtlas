"""Machine-checked code acceptance: positive case and every fail-closed path."""
import hashlib
import json
from pathlib import Path

import pytest

from cc_hhgt.v32.code_acceptance import EXPECTED, FORMAT, OPEN, VERIFIED, accept, validate, write_acceptance

SUITE = "provenance/code_completion_20261015/synthetic_tests.xml"
RECEIPT = "provenance/code_completion_20261015/synthetic_test_receipt.json"
CASES = [("tests.synthetic_a", "test_one"), ("tests.synthetic_a", "test_two"),
         ("tests.synthetic_b", "test_three")]
IMPLEMENTATION = "Synthetic implementation record. " * 10


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _suite_xml(path, cases=CASES, bad=None):
    rows = "".join(
        f'<testcase classname="{name}" name="{case}">'
        + ("<failure message=\"x\">boom</failure>" if bad == (name, case) else "")
        + "</testcase>"
        for name, case in cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<?xml version="1.0"?><testsuites><testsuite tests="3">{rows}</testsuite></testsuites>',
                    encoding="utf-8")


def fixture(tmp_path):
    """A minimal, self-consistent acceptance input set."""
    code = tmp_path / "code"
    pinned = code / "cc_hhgt/v32/dummy_source.py"
    pinned.parent.mkdir(parents=True, exist_ok=True)
    pinned.write_text("VALUE = 1\n", encoding="utf-8")
    provenance = code / "provenance/code_completion_20261015"
    _suite_xml(code / SUITE)
    _write(code / RECEIPT, {"target_host": "149", "exit_code": 0, "training_started": False})
    requirements = []
    for index in range(1, 22):
        identifier = f"C{index:02d}"
        if identifier == "C01":
            requirements.append({"id": identifier, "status": OPEN, "implementation": IMPLEMENTATION,
                                 "remaining": "one item still open"})
        else:
            requirements.append({"id": identifier, "status": VERIFIED,
                                 "implementation": IMPLEMENTATION})
    plan = {"objective": "synthetic", "base_commit": "0" * 40, "target_host_for_checks": "149",
            "training_launched": False, "requirements": requirements,
            "latest_validation": {"target_host": "149", "unique_fixture_tests_passed": len(CASES),
                                  "latest_change_tests_passed": len(CASES),
                                  "remaining_requirements_not_complete": ["C01"],
                                  "pending_validation": []}}
    validation = {"status": "PASS_SYNTHETIC_CODE_CHECKS", "target_host": "149",
                  "unique_test_cases_passed": len(CASES), "optimizer_jobs_started": 0,
                  "real_data_runs_started": 0, "paid_compute_started": False,
                  "production_changed": False,
                  "source_sha256": {"cc_hhgt/v32/dummy_source.py": _sha(pinned)},
                  "test_suites": [{"path": SUITE, "passed": len(CASES), "sha256": _sha(code / SUITE)}],
                  "verified_case_inventory": [{"module": name, "case": case, "suite": SUITE}
                                              for name, case in CASES],
                  "latest_change_validation": {"tags": ["synthetic"], "passed": len(CASES),
                                               "target_host": "149", "training_started": False}}
    paths = {"code": code, "plan": code / "CODE_COMPLETION_PLAN.json",
             "validation": provenance / "VALIDATION_SUMMARY.json", "provenance": provenance,
             "pinned": pinned, "suite": code / SUITE, "receipt": code / RECEIPT}
    _write(paths["plan"], plan)
    _write(paths["validation"], validation)
    return paths


def run(paths, **extra):
    return accept(paths["plan"], paths["validation"], paths["provenance"], paths["code"],
                  head_commit="a" * 40, worktree_status="", **extra)


def test_consistent_registry_is_accepted(tmp_path):
    paths = fixture(tmp_path)
    value = run(paths)
    assert value["format"] == FORMAT and value["status"] == "PASS"
    assert value["requirement_count"] == 21 and value["verified_count"] == 20
    assert value["open_requirements"] == ["C01"] and value["open_count"] == 1
    assert value["latest_case_count"] == len(CASES) and value["cumulative_case_count"] == len(CASES)
    assert value["source_pin_count"] == 1 and value["checks_applied"] > 20
    assert value["training_started"] is False
    assert value["real_cohort_analysis_attested"] is False
    assert set(EXPECTED) == {row["id"] for row in value["requirements"]}


def test_recorded_acceptance_must_match_a_fresh_check(tmp_path):
    paths = fixture(tmp_path)
    output = tmp_path / "CODE_ACCEPTANCE.json"
    write_acceptance(output, run(paths))
    assert validate(output, plan_path=paths["plan"], validation_path=paths["validation"],
                    provenance_dir=paths["provenance"], code_root=paths["code"],
                    worktree_status="")["status"] == "PASS"
    value = json.loads(output.read_text(encoding="utf-8"))
    value["verified_count"] = 4
    write_acceptance(output, value)
    with pytest.raises(ValueError):
        validate(output, plan_path=paths["plan"], validation_path=paths["validation"],
                 provenance_dir=paths["provenance"], code_root=paths["code"], worktree_status="")


def _fault(kind):
    def apply(paths):
        plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
        validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
        if kind == "training":
            plan["training_launched"] = True
        elif kind == "short_implementation":
            plan["requirements"][1]["implementation"] = "too short"
        elif kind == "verified_with_remaining":
            plan["requirements"][1]["remaining"] = "still open"
        elif kind == "open_without_remaining":
            plan["requirements"][0]["remaining"] = "  "
        elif kind == "missing_requirement":
            plan["requirements"] = plan["requirements"][:-1]
            plan["latest_validation"]["remaining_requirements_not_complete"] = ["C01"]
        elif kind == "open_list_mismatch":
            plan["latest_validation"]["remaining_requirements_not_complete"] = ["C01", "C13"]
        elif kind == "pending_validation":
            plan["latest_validation"]["pending_validation"] = ["something"]
        elif kind == "receipt_exit":
            _write(paths["receipt"], {"target_host": "149", "exit_code": 1, "training_started": False})
        elif kind == "receipt_training":
            _write(paths["receipt"], {"target_host": "149", "exit_code": 0, "training_started": True})
        elif kind == "suite_failure":
            _suite_xml(paths["suite"], bad=CASES[1])
            validation["test_suites"][0]["sha256"] = _sha(paths["suite"])
        elif kind == "suite_missing":
            paths["suite"].unlink()
        elif kind == "pin_drift":
            paths["pinned"].write_text("VALUE = 2\n", encoding="utf-8")
        elif kind == "case_count":
            validation["unique_test_cases_passed"] = 99
        elif kind == "latest_count":
            validation["latest_change_validation"]["passed"] = 99
        elif kind == "real_data":
            validation["real_data_runs_started"] = 1
        elif kind == "suite_unregistered":
            validation["test_suites"] = []
        elif kind == "duplicate_case":
            validation["verified_case_inventory"].append(dict(validation["verified_case_inventory"][0]))
        else:
            raise AssertionError(kind)
        _write(paths["plan"], plan)
        _write(paths["validation"], validation)
    return apply


@pytest.mark.parametrize("kind", [
    "training", "short_implementation", "verified_with_remaining", "open_without_remaining",
    "missing_requirement", "open_list_mismatch", "pending_validation", "receipt_exit",
    "receipt_training", "suite_failure", "suite_missing", "pin_drift", "case_count",
    "latest_count", "real_data", "suite_unregistered", "duplicate_case"])
def test_inconsistent_registry_fails_closed(tmp_path, kind):
    paths = fixture(tmp_path)
    _fault(kind)(paths)
    with pytest.raises(ValueError):
        run(paths)


def test_dirty_worktree_fails_closed(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError):
        accept(paths["plan"], paths["validation"], paths["provenance"], paths["code"],
               head_commit="a" * 40, worktree_status=" M cc_hhgt/v32/dummy_source.py\n")


def test_missing_receipt_fails_closed(tmp_path):
    paths = fixture(tmp_path)
    paths["receipt"].unlink()
    with pytest.raises(ValueError):
        run(paths)
