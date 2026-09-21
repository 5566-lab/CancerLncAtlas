from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v32_frozen_plan_complement_audit.py"
SPEC = importlib.util.spec_from_file_location("v32_frozen_plan_complement_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_plan(path: Path, target_root: Path, rows: list[dict]) -> str:
    payload = {
        "format": MODULE.PLAN_FORMAT,
        "target_root": str(target_root),
        "entry_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "entries": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(encoded)
    return _sha(encoded)


def _record(path: Path, payload: bytes) -> dict:
    return {
        "source_path": "test-only",
        "target_path": str(path),
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


class FrozenComplementAuditTests(unittest.TestCase):
    def _arguments(
        self,
        *,
        root: Path,
        full_path: Path,
        full_sha: str,
        full_rows: list[dict],
        remaining_path: Path,
        remaining_sha: str,
        remaining_rows: list[dict],
        output: Path,
    ) -> list[str]:
        remaining_targets = {row["target_path"] for row in remaining_rows}
        complement = [row for row in full_rows if row["target_path"] not in remaining_targets]
        return [
            "--full-plan",
            str(full_path),
            "--full-plan-sha256",
            full_sha,
            "--remaining-plan",
            str(remaining_path),
            "--remaining-plan-sha256",
            remaining_sha,
            "--target-root",
            str(root),
            "--allowed-root",
            str(root.parent),
            "--expected-full-count",
            str(len(full_rows)),
            "--expected-full-bytes",
            str(sum(row["bytes"] for row in full_rows)),
            "--expected-remaining-count",
            str(len(remaining_rows)),
            "--expected-remaining-bytes",
            str(sum(row["bytes"] for row in remaining_rows)),
            "--expected-complement-count",
            str(len(complement)),
            "--expected-complement-bytes",
            str(sum(row["bytes"] for row in complement)),
            "--output",
            str(output),
        ]

    def test_hashes_only_strict_complement_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "target"
            root.mkdir()
            payloads = [b"remaining-wrong-0", b"remaining-wrong-1", b"a", b"bb", b"ccc"]
            intended = [b"remaining-0", b"remaining-1", b"a", b"bb", b"ccc"]
            paths = [root / f"payload-{index}.bin" for index in range(5)]
            # Remaining targets are deliberately wrong.  A passing audit proves
            # they were excluded rather than rehashed by the complement verifier.
            for path, payload in zip(paths, payloads, strict=True):
                path.write_bytes(payload)
            full_rows = [_record(path, payload) for path, payload in zip(paths, intended, strict=True)]
            remaining_rows = full_rows[:2]
            full_path = base / "full.json"
            remaining_path = base / "remaining.json"
            full_sha = _write_plan(full_path, root, full_rows)
            remaining_sha = _write_plan(remaining_path, root, remaining_rows)
            output = base / "audit.json"
            arguments = self._arguments(
                root=root,
                full_path=full_path,
                full_sha=full_sha,
                full_rows=full_rows,
                remaining_path=remaining_path,
                remaining_sha=remaining_sha,
                remaining_rows=remaining_rows,
                output=output,
            )
            self.assertEqual(MODULE.main(arguments), 0)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["plan_relationship"]["complement_count"], 3)
            self.assertEqual(audit["hash_scope"]["verified_count"], 3)
            self.assertEqual(audit["hash_scope"]["remaining_targets_skipped_count"], 2)
            self.assertEqual(audit["hash_scope"]["remaining_targets_rehashed_count"], 0)
            self.assertEqual(audit["conflict_counts"]["total"], 0)

            before = output.read_bytes()
            self.assertEqual(MODULE.main(arguments), 2)
            self.assertEqual(output.read_bytes(), before)

    def test_overlap_expectation_conflict_fails_without_target_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "target"
            root.mkdir()
            first = root / "first.bin"
            second = root / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            full_rows = [_record(first, b"first"), _record(second, b"second")]
            remaining_rows = [dict(full_rows[0])]
            remaining_rows[0]["sha256"] = _sha(b"other")
            full_path = base / "full.json"
            remaining_path = base / "remaining.json"
            full_sha = _write_plan(full_path, root, full_rows)
            remaining_sha = _write_plan(remaining_path, root, remaining_rows)
            output = base / "audit.json"
            arguments = self._arguments(
                root=root,
                full_path=full_path,
                full_sha=full_sha,
                full_rows=full_rows,
                remaining_path=remaining_path,
                remaining_sha=remaining_sha,
                remaining_rows=remaining_rows,
                output=output,
            )
            self.assertEqual(MODULE.main(arguments), 3)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "FAIL")
            self.assertEqual(audit["conflict_counts"]["plan_relationship"], 1)
            self.assertEqual(audit["hash_scope"]["complement_targets_hashed_count"], 0)
            self.assertEqual(audit["hash_scope"]["remaining_targets_rehashed_count"], 0)

    def test_non_regular_complement_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "target"
            root.mkdir()
            target = root / "directory-not-file"
            target.mkdir()
            full_rows = [_record(target, b"expected")]
            remaining_rows: list[dict] = []
            full_path = base / "full.json"
            remaining_path = base / "remaining.json"
            full_sha = _write_plan(full_path, root, full_rows)
            remaining_sha = _write_plan(remaining_path, root, remaining_rows)
            output = base / "audit.json"
            arguments = self._arguments(
                root=root,
                full_path=full_path,
                full_sha=full_sha,
                full_rows=full_rows,
                remaining_path=remaining_path,
                remaining_sha=remaining_sha,
                remaining_rows=remaining_rows,
                output=output,
            )
            self.assertEqual(MODULE.main(arguments), 3)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(audit["conflicts"]["targets"][0]["status"], "non_regular")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support is unavailable")
    def test_symlink_complement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / "target"
            root.mkdir()
            real = base / "real.bin"
            real.write_bytes(b"payload")
            target = root / "link.bin"
            try:
                target.symlink_to(real)
            except OSError as exc:
                self.skipTest(f"symlink creation is not permitted: {exc}")
            full_rows = [_record(target, b"payload")]
            remaining_rows: list[dict] = []
            full_path = base / "full.json"
            remaining_path = base / "remaining.json"
            full_sha = _write_plan(full_path, root, full_rows)
            remaining_sha = _write_plan(remaining_path, root, remaining_rows)
            output = base / "audit.json"
            arguments = self._arguments(
                root=root,
                full_path=full_path,
                full_sha=full_sha,
                full_rows=full_rows,
                remaining_path=remaining_path,
                remaining_sha=remaining_sha,
                remaining_rows=remaining_rows,
                output=output,
            )
            self.assertEqual(MODULE.main(arguments), 3)
            audit = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                audit["conflicts"]["targets"][0]["status"],
                "symlink_or_unsafe_component",
            )


if __name__ == "__main__":
    unittest.main()
