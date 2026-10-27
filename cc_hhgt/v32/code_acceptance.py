"""Machine-check the registered code requirements against recorded evidence.

This is a static bookkeeping check: it reads the completion plan, the validation
summary and the recorded pytest receipts, then refuses to declare acceptance when
a requirement has no evidence, when a pinned source no longer matches the working
tree, when a receipt records failures, skips or training, or when the declared
counters disagree with the receipts.

It performs no scientific computation, needs no host gate, and it cannot make a
requirement complete: ``in_progress`` stays ``in_progress`` until the plan itself
is updated with new evidence.
"""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

FORMAT = "CANCERLNCATLAS_CODE_ACCEPTANCE_V1"
VERIFIED = "code_implemented_fixture_verified"
OPEN = "in_progress"
STATUSES = {VERIFIED, OPEN}
EXPECTED = {f"C{index:02d}" for index in range(1, 17)}
IDENTIFIER = re.compile(r"^C\d{2}$")
PROVENANCE = "provenance/code_completion_20261015"
MINIMUM_IMPLEMENTATION_CHARS = 200


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _latest_cases(tags, provenance, code_root):
    """Unique (module, case) pairs from the latest listed suites per module."""
    latest = set()
    suites = []
    for tag in tags:
        path = Path(provenance) / (tag + "_tests.xml")
        if not path.is_file():
            raise ValueError("Acceptance suite is missing: " + str(path))
        cases = list(ET.parse(path).getroot().iter("testcase"))
        if not cases:
            raise ValueError("Acceptance suite recorded no cases: " + tag)
        broken = [case for case in cases
                  if any(case.find(tag_name) is not None for tag_name in ("failure", "error", "skipped"))]
        if broken:
            raise ValueError("Acceptance suite recorded failures, errors or skips: " + tag)
        modules = {case.get("classname") for case in cases}
        latest = {key for key in latest if key[0] not in modules}
        for case in cases:
            latest.add((case.get("classname"), case.get("name")))
        try:
            recorded = path.resolve().relative_to(Path(code_root).resolve()).as_posix()
        except ValueError:
            recorded = str(path)
        suites.append({"tag": tag, "path": recorded, "cases": len(cases), "sha256": _sha(path)})
    return latest, suites


def accept(plan_path, validation_path, provenance_dir, code_root, *, head_commit=None,
           worktree_status=None):
    """Return a machine-checked acceptance record or raise on any inconsistency."""
    plan = _read(plan_path)
    validation = _read(validation_path)
    provenance = Path(provenance_dir)
    code_root = Path(code_root)
    problems = []
    checks = []

    def require(name, condition, detail=""):
        checks.append({"check": name, "ok": bool(condition), "detail": detail})
        if not condition:
            problems.append(name + (": " + detail if detail else ""))

    require("plan.declares_no_training", plan.get("training_launched") is False)
    require("plan.target_host_is_149", plan.get("target_host_for_checks") == "149")
    require("validation.target_host_is_149", validation.get("target_host") == "149")
    require("validation.no_real_data_runs", validation.get("real_data_runs_started") == 0)
    require("validation.no_optimizer_jobs", validation.get("optimizer_jobs_started") == 0)
    require("validation.no_paid_compute", validation.get("paid_compute_started") is False)
    require("validation.no_production_change", validation.get("production_changed") is False)

    requirements = plan.get("requirements")
    require("plan.has_requirements", isinstance(requirements, list) and bool(requirements))
    rows = []
    identifiers = []
    for item in requirements or []:
        identifier = item.get("id")
        status = item.get("status")
        implementation = item.get("implementation") or ""
        remaining = (item.get("remaining") or "").strip()
        identifiers.append(identifier)
        valid_identifier = isinstance(identifier, str) and bool(IDENTIFIER.match(identifier))
        require("requirement.identity." + str(identifier), valid_identifier)
        require("requirement.status." + str(identifier), status in STATUSES, str(status))
        if status == VERIFIED:
            require("requirement.implementation." + str(identifier),
                    len(implementation) >= MINIMUM_IMPLEMENTATION_CHARS,
                    str(len(implementation)) + " chars")
            require("requirement.closed." + str(identifier), not remaining,
                    "verified requirement still declares remaining work")
        elif status == OPEN:
            require("requirement.remaining." + str(identifier), bool(remaining))
        rows.append({"id": identifier, "status": status,
                     "implementation_chars": len(implementation),
                     "remaining_declared": bool(remaining)})
    require("plan.requirement_identifiers_unique", len(identifiers) == len(set(identifiers)))
    require("plan.requirement_set", set(identifiers) == EXPECTED,
            str(sorted(EXPECTED - set(identifiers))) + " missing")

    tags = ((validation.get("latest_change_validation") or {}).get("tags")) or []
    require("validation.latest_tags_present", bool(tags) and isinstance(tags, list))
    latest, suites = _latest_cases(tags, provenance, code_root)
    for tag in tags:
        receipt_path = provenance / (tag + "_test_receipt.json")
        if not receipt_path.is_file():
            problems.append("receipt missing: " + tag)
            require("receipt.exists." + tag, False)
            continue
        receipt = _read(receipt_path)
        require("receipt.host." + tag, receipt.get("target_host") == "149")
        require("receipt.passed." + tag, receipt.get("exit_code") == 0)
        require("receipt.no_training." + tag, receipt.get("training_started") is False)

    declared_latest = (validation.get("latest_change_validation") or {}).get("passed")
    plan_latest = (plan.get("latest_validation") or {}).get("latest_change_tests_passed")
    require("counts.latest_matches_receipts", declared_latest == len(latest),
            str(declared_latest) + " vs " + str(len(latest)))
    require("counts.latest_matches_plan", plan_latest == len(latest),
            str(plan_latest) + " vs " + str(len(latest)))

    pinned = validation.get("source_sha256") or {}
    require("validation.source_pins_present", bool(pinned))
    drifted = []
    for name, digest in pinned.items():
        target = Path(code_root) / name
        if not target.is_file() or _sha(target) != digest:
            drifted.append(name)
    require("validation.source_pins_match_worktree", not drifted, ", ".join(sorted(drifted)[:5]))

    inventory = validation.get("verified_case_inventory") or []
    cases = {}
    for row in inventory:
        key = (row.get("module"), row.get("case"))
        if key in cases:
            problems.append("duplicate case: " + str(key))
        cases[key] = row.get("suite")
    require("inventory.cases_unique", len(cases) == len(inventory))
    suite_paths = {row.get("path") for row in validation.get("test_suites") or []}
    missing_suites = sorted({value for value in cases.values() if value not in suite_paths})
    require("inventory.suites_registered", not missing_suites, ", ".join(missing_suites[:5]))
    suite_drift = []
    for suite in validation.get("test_suites") or []:
        path = Path(code_root) / suite.get("path", "")
        if not path.is_file() or _sha(path) != suite.get("sha256"):
            suite_drift.append(str(path))
    require("inventory.suite_hashes_match", not suite_drift, ", ".join(suite_drift[:5]))
    require("inventory.case_count", len(cases) == validation.get("unique_test_cases_passed"),
            str(len(cases)) + " vs " + str(validation.get("unique_test_cases_passed")))
    require("counts.plan_cumulative", (plan.get("latest_validation") or {}).get("unique_fixture_tests_passed")
            == validation.get("unique_test_cases_passed"))

    open_ids = sorted(row["id"] for row in rows if row["status"] == OPEN)
    declared_open = sorted((plan.get("latest_validation") or {}).get("remaining_requirements_not_complete") or [])
    require("plan.open_requirements_match", declared_open == open_ids,
            str(declared_open) + " vs " + str(open_ids))
    pending = (plan.get("latest_validation") or {}).get("pending_validation") or []
    require("plan.no_pending_validation", not pending, str(pending))

    if worktree_status is not None:
        require("source.worktree_clean", not str(worktree_status).strip(), str(worktree_status)[:200])

    if problems:
        raise ValueError("CODE_ACCEPTANCE_FAILED: " + "; ".join(problems))
    return {
        "format": FORMAT,
        "status": "PASS",
        "target_host": "149",
        "training_started": False,
        "requirements": rows,
        "requirement_count": len(rows),
        "verified_count": sum(1 for row in rows if row["status"] == VERIFIED),
        "open_count": len(open_ids),
        "open_requirements": open_ids,
        "latest_tags": list(tags),
        "latest_case_count": len(latest),
        "cumulative_case_count": len(cases),
        "suite_count": len(suites),
        "source_pin_count": len(pinned),
        "latest_suites": suites,
        "plan_sha256": _sha(plan_path),
        "validation_sha256": _sha(validation_path),
        "head_commit": head_commit,
        "checks_applied": len(checks),
        "scope": "REGISTERED_CODE_EVIDENCE_AND_PIN_INTEGRITY_ONLY",
        "real_cohort_analysis_attested": False,
        "independent_biological_validation_attested": False,
        "production_deployment_attested": False,
    }


def write_acceptance(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate(path, *, plan_path, validation_path, provenance_dir, code_root, worktree_status=None):
    """Re-run acceptance and require the recorded artifact to match exactly."""
    recorded = _read(path)
    expected = accept(plan_path, validation_path, provenance_dir, code_root,
                      head_commit=recorded.get("head_commit"), worktree_status=worktree_status)
    if recorded != expected:
        raise ValueError("Recorded code acceptance differs from a fresh machine check")
    return expected


__all__ = ["EXPECTED", "FORMAT", "OPEN", "VERIFIED", "accept", "validate", "write_acceptance"]
