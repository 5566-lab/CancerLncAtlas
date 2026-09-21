#!/usr/bin/env python3
"""Freeze the exact r9 code leaves, Python runtime, and direct dependencies."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
import sysconfig
from typing import Any


FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_R9_RUNTIME_BINDING_V1"
REQUIRED_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "h5py": "h5py",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
}
THREAD_PINS = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class RuntimeBindingError(RuntimeError):
    """Raised when code or runtime identity cannot be proved exactly."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeBindingError(f"absent or unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeBindingError(f"absent or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeBindingError(f"JSON root is not an object: {path}")
    return value


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeBindingError(f"output reuse forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_tool(tool_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeBindingError("copy manifest has no entries")
    verified: list[dict[str, Any]] = []
    resolved_root = tool_root.resolve(strict=True)
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("kind") != "file":
            raise RuntimeBindingError("copy manifest contains unsupported leaf")
        target = Path(str(entry.get("target_path", "")))
        resolved = target.resolve(strict=True)
        if resolved_root not in resolved.parents:
            raise RuntimeBindingError(f"manifest leaf escapes tool root: {target}")
        expected_bytes = int(entry.get("bytes", -1))
        expected_sha = str(entry.get("sha256", ""))
        actual_bytes = resolved.stat().st_size
        actual_sha = sha256_file(resolved)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise RuntimeBindingError(f"tool leaf drift: {target}")
        verified.append(
            {
                "path": str(resolved),
                "relative_path": str(resolved.relative_to(resolved_root)),
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        )
    if len(verified) != int(manifest.get("entry_count", -1)):
        raise RuntimeBindingError("manifest entry count drift")
    if sum(row["bytes"] for row in verified) != int(
        manifest.get("total_bytes", -1)
    ):
        raise RuntimeBindingError("manifest total bytes drift")
    return verified


def module_identity(module_name: str, distribution_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if not module_file.is_file():
        raise RuntimeBindingError(f"module file is absent: {module_name}")
    return {
        "module": module_name,
        "distribution": distribution_name,
        "version": importlib.metadata.version(distribution_name),
        "module_file": str(module_file),
        "module_file_bytes": module_file.stat().st_size,
        "module_file_sha256": sha256_file(module_file),
    }


def native_extension_identity(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if not module_file.is_file():
        raise RuntimeBindingError(f"native module file is absent: {module_name}")
    return {
        "module": module_name,
        "path": str(module_file),
        "bytes": module_file.stat().st_size,
        "sha256": sha256_file(module_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-root", required=True, type=Path)
    parser.add_argument("--copy-manifest-sha256", required=True)
    parser.add_argument("--final-runner-binding", required=True, type=Path)
    parser.add_argument("--final-runner-binding-sha256", required=True)
    parser.add_argument("--expected-python-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tool_root = args.tool_root.resolve(strict=True)
    manifest_path = tool_root / "COPY_MANIFEST.json"
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != args.copy_manifest_sha256:
        raise RuntimeBindingError("copy manifest SHA-256 drift")
    verified_tool_leaves = verify_tool(tool_root, manifest_path)

    binding = args.final_runner_binding.resolve(strict=True)
    binding_sha = sha256_file(binding)
    if binding_sha != args.final_runner_binding_sha256:
        raise RuntimeBindingError("final-runner binding SHA-256 drift")
    if load_json(binding).get("status") != (
        "PASS_FINAL_RUNNER_BH_IMPLEMENTATION_IDENTICAL_TO_TESTED_RUNNER"
    ):
        raise RuntimeBindingError("final-runner binding status drift")

    for key, expected in THREAD_PINS.items():
        if os.environ.get(key) != expected:
            raise RuntimeBindingError(
                f"thread pin {key}={os.environ.get(key)!r}, expected {expected!r}"
            )

    executable_link = Path(sys.executable)
    executable = executable_link.resolve(strict=True)
    executable_sha = sha256_file(executable)
    if executable_sha != args.expected_python_sha256:
        raise RuntimeBindingError("Python executable SHA-256 drift")
    python_stat = executable.stat()

    dependencies = {
        name: module_identity(name, distribution)
        for name, distribution in REQUIRED_DISTRIBUTIONS.items()
    }
    native_extensions = {
        name: native_extension_identity(name)
        for name in (
            "numpy._core._multiarray_umath",
            "scipy.linalg._fblas",
            "h5py.h5",
            "pyarrow.lib",
            "_duckdb",
        )
    }
    all_distributions = sorted(
        {
            (
                str(distribution.metadata.get("Name", "")).lower(),
                str(distribution.version),
            )
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )

    value: dict[str, Any] = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_CODE_DEPENDENCY_RUNTIME_BOUND",
        "tool_root": str(tool_root),
        "copy_manifest_path": str(manifest_path),
        "copy_manifest_sha256": manifest_sha,
        "verified_tool_leaf_count": len(verified_tool_leaves),
        "verified_tool_total_bytes": sum(
            row["bytes"] for row in verified_tool_leaves
        ),
        "verified_tool_leaves": verified_tool_leaves,
        "final_runner_binding_path": str(binding),
        "final_runner_binding_sha256": binding_sha,
        "python": {
            "invoked_path": str(executable_link),
            "resolved_path": str(executable),
            "sha256": executable_sha,
            "bytes": python_stat.st_size,
            "mode": stat.S_IFMT(python_stat.st_mode) | stat.S_IMODE(python_stat.st_mode),
            "uid": python_stat.st_uid,
            "gid": python_stat.st_gid,
            "dev": python_stat.st_dev,
            "inode": python_stat.st_ino,
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "platform": platform.platform(),
        },
        "direct_dependencies": dependencies,
        "native_extensions": native_extensions,
        "installed_distribution_count": len(all_distributions),
        "installed_distribution_set_sha256": canonical_sha256(all_distributions),
        "installed_distributions": [
            {"name": name, "version": version}
            for name, version in all_distributions
        ],
        "thread_pins": dict(THREAD_PINS),
        "production_deployed": False,
    }
    value["runtime_binding_contract_sha256"] = canonical_sha256(value)
    output = args.output.resolve()
    exclusive_json(output, value)
    print(
        json.dumps(
            {**value, "output_file_sha256": sha256_file(output)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
