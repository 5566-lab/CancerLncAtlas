#!/usr/bin/env python3
"""Install and bind the offline R9 CPython runtime without reading /dsk2.

The payload, runtime, and audit roots are all immutable, distinct children of
the authorized CancerLncAtlas ${PRIVATE_WORK_ROOT} tree.  Installation is performed in a
new sibling build root and atomically renamed.  Pip is offline and every
temporary/cache environment variable is pinned inside that build root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import tarfile
import time
from typing import Any, Mapping


AUTHORIZED_ROOT = Path("./data/CancerLncAtlas")
FORMAT = "CANCERLNCATLAS_V32_SINGLE_CELL_R9_STANDALONE_RUNTIME_BINDING_V1"
FAILURE_FORMAT = "CANCERLNCATLAS_V32_SINGLE_CELL_R9_STANDALONE_RUNTIME_FAILURE_V1"
EXPECTED_MANIFEST_FORMAT = (
    "CANCERLNCATLAS_V32_SINGLE_CELL_R9_STANDALONE_RUNTIME_INPUT_V1"
)
EXPECTED_PYTHON = "3.13.15"
EXPECTED_VERSIONS = {
    "numpy": "2.3.4",
    "pandas": "2.3.3",
    "scipy": "1.17.1",
    "h5py": "3.16.0",
    "pyarrow": "23.0.1",
    "duckdb": "1.5.5",
    "python-dateutil": "2.9.0.post0",
    "pytz": "2025.2",
    "tzdata": "2025.2",
    "six": "1.17.0",
    "packaging": "25.0",
}
IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "h5py": "h5py",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
    "python-dateutil": "dateutil",
    "pytz": "pytz",
    "tzdata": "tzdata",
    "six": "six",
    "packaging": "packaging",
}
FORBIDDEN_PATH_FRAGMENTS = ("/dsk2", "/tmp")


class RuntimeInstallError(RuntimeError):
    """Raised when an immutable runtime invariant cannot be proved."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeInstallError(f"absent/unsafe file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeInstallError(f"immutable JSON reuse forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeInstallError(f"absent/unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeInstallError(f"JSON root is not an object: {path}")
    return value


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_authorized(path: Path, *, may_exist: bool) -> Path:
    value = path.resolve(strict=may_exist)
    root = AUTHORIZED_ROOT.resolve(strict=True)
    if value == root or not is_within(value, root):
        raise RuntimeInstallError(f"path leaves authorized ${PRIVATE_WORK_ROOT} project: {path}")
    if any(fragment in str(value) for fragment in FORBIDDEN_PATH_FRAGMENTS):
        raise RuntimeInstallError(f"forbidden path fragment: {value}")
    return value


def validate_payload(
    payload_root: Path, manifest_path: Path, expected_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeInstallError("runtime input manifest SHA drift")
    manifest = load_json(manifest_path)
    if (
        manifest.get("format") != EXPECTED_MANIFEST_FORMAT
        or manifest.get("status") != "DOWNLOADED_AND_OFFICIAL_DIGESTS_VERIFIED"
    ):
        raise RuntimeInstallError("runtime input manifest semantics drift")
    policy = manifest.get("runtime_policy", {})
    expected_false = (
        "reads_dsk2",
        "writes_dsk2",
        "writes_tmp",
        "writes_root_filesystem",
        "network_install_permitted",
        "overwrite_permitted",
    )
    if any(policy.get(key) is not False for key in expected_false):
        raise RuntimeInstallError("runtime policy permits an unauthorized operation")
    if policy.get("offline_wheels_only") is not True:
        raise RuntimeInstallError("runtime policy lost offline-wheel requirement")
    python = manifest.get("python")
    wheels = manifest.get("wheels")
    licenses = manifest.get("licenses")
    if not isinstance(python, dict) or not isinstance(wheels, list) or not isinstance(
        licenses, list
    ):
        raise RuntimeInstallError("runtime manifest leaf collection is malformed")
    if python.get("version") != EXPECTED_PYTHON:
        raise RuntimeInstallError("standalone Python version drift")
    observed_versions = {
        str(row.get("name")): str(row.get("version"))
        for row in wheels
        if isinstance(row, dict)
    }
    if observed_versions != EXPECTED_VERSIONS:
        raise RuntimeInstallError("wheel version set is not exactly frozen R9")
    rows: list[dict[str, Any]] = []
    leaves = [(payload_root / str(python["filename"]), python, "python")]
    leaves.extend(
        (payload_root / "wheels" / str(row["filename"]), row, str(row["name"]))
        for row in wheels
    )
    leaves.extend(
        (payload_root / str(row["filename"]), row, "license") for row in licenses
    )
    for path, expected, role in leaves:
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeInstallError(f"payload byte drift: {role}:{path.name}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise RuntimeInstallError(f"payload SHA drift: {role}:{path.name}")
        rows.append(
            {
                "role": role,
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": digest,
            }
        )
    return manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "verified_leaves": rows,
        "verified_leaf_count": len(rows),
        "verified_payload_bytes": sum(row["bytes"] for row in rows),
    }


def validate_tar_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        raise RuntimeInstallError("standalone archive is empty")
    for member in members:
        name = PurePosixPath(member.name)
        if (
            name.is_absolute()
            or not name.parts
            or name.parts[0] != "python"
            or ".." in name.parts
        ):
            raise RuntimeInstallError(f"unsafe standalone archive member: {member.name}")
        if member.ischr() or member.isblk() or member.isfifo():
            raise RuntimeInstallError(f"special archive member forbidden: {member.name}")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or not member.linkname:
                raise RuntimeInstallError(
                    f"unsafe standalone archive link: {member.name}->{member.linkname}"
                )
            # A symbolic-link target is relative to the link's parent.  A tar
            # hard-link target is relative to the archive root.  Legitimate
            # python-build-standalone terminfo links contain ``..`` while
            # remaining inside ``python/``; reject based on the normalized
            # destination instead of rejecting every parent component.
            if member.issym():
                normalized_target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(member.name), member.linkname)
                )
            else:
                normalized_target = posixpath.normpath(member.linkname)
            normalized = PurePosixPath(normalized_target)
            if (
                normalized.is_absolute()
                or not normalized.parts
                or normalized.parts[0] != "python"
                or ".." in normalized.parts
            ):
                raise RuntimeInstallError(
                    f"archive link leaves python root: "
                    f"{member.name}->{member.linkname}=>{normalized_target}"
                )
    return members


def offline_environment(build_root: Path, runtime_bin: Path) -> dict[str, str]:
    temporary = build_root / "offline_temp"
    pip_cache = build_root / "offline_pip_cache"
    xdg_cache = build_root / "offline_xdg_cache"
    for path in (temporary, pip_cache, xdg_cache):
        path.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CONDA_PYTHON_EXE",
        "_CE_CONDA",
        "_CE_M",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "PATH": f"{runtime_bin}:/usr/bin:/bin",
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "PIP_CACHE_DIR": str(pip_cache),
            "XDG_CACHE_HOME": str(xdg_cache),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    if any(
        fragment in value
        for key, value in environment.items()
        for fragment in FORBIDDEN_PATH_FRAGMENTS
        if key in {"PATH", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "TMPDIR", "TMP", "TEMP", "PIP_CACHE_DIR", "XDG_CACHE_HOME"}
    ):
        raise RuntimeInstallError("sanitized runtime environment retains forbidden path")
    return environment


PROBE = r'''
import hashlib, importlib, importlib.metadata, json, os, pathlib, sys
expected = json.loads(os.environ["V32_EXPECTED_VERSIONS_JSON"])
imports = json.loads(os.environ["V32_IMPORT_NAMES_JSON"])
modules = {}
for distribution, import_name in imports.items():
    module = importlib.import_module(import_name)
    path = pathlib.Path(module.__file__).resolve()
    modules[distribution] = {
        "version": importlib.metadata.version(distribution),
        "module_file": str(path),
        "module_file_bytes": path.stat().st_size,
        "module_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
versions = {name: importlib.metadata.version(name) for name in expected}
maps = pathlib.Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
mapped_paths = sorted({part for line in maps.splitlines() for part in line.split() if part.startswith("/")})
payload = {
    "python_version": ".".join(map(str, sys.version_info[:3])),
    "python_executable": str(pathlib.Path(sys.executable).resolve()),
    "sys_prefix": str(pathlib.Path(sys.prefix).resolve()),
    "sys_path": [str(pathlib.Path(value).resolve()) for value in sys.path if value],
    "versions": versions,
    "modules": modules,
    "mapped_paths": mapped_paths,
    "forbidden_dsk2_found": any("/dsk2" in value for value in mapped_paths + sys.path),
    "forbidden_tmp_found": any(value == "/tmp" or value.startswith("/tmp/") for value in mapped_paths + sys.path),
}
print(json.dumps(payload, sort_keys=True))
'''


def run_checked(argv: list[str], *, environment: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        shell=False,
        env=dict(environment),
    )
    if result.returncode != 0:
        raise RuntimeInstallError(
            f"offline subprocess failed ({result.returncode}): {argv}; stderr={result.stderr[-4000:]}"
        )
    return result


def probe_runtime(python: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    probe_env = dict(environment)
    probe_env["V32_EXPECTED_VERSIONS_JSON"] = json.dumps(EXPECTED_VERSIONS)
    probe_env["V32_IMPORT_NAMES_JSON"] = json.dumps(IMPORT_NAMES)
    result = run_checked([str(python), "-I", "-s", "-c", PROBE], environment=probe_env)
    value = json.loads(result.stdout)
    if value["python_version"] != EXPECTED_PYTHON:
        raise RuntimeInstallError("installed Python version drift")
    if value["versions"] != EXPECTED_VERSIONS:
        raise RuntimeInstallError("installed scientific version set drift")
    if value["forbidden_dsk2_found"] or value["forbidden_tmp_found"]:
        raise RuntimeInstallError("runtime probe loaded a forbidden path")
    return value


def normalize_probe(value: Any, source: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(source, replacement)
    if isinstance(value, list):
        return [normalize_probe(item, source, replacement) for item in value]
    if isinstance(value, dict):
        return {
            key: normalize_probe(item, source, replacement)
            for key, item in value.items()
        }
    return value


def execute(args: argparse.Namespace) -> dict[str, Any]:
    payload_root = require_authorized(args.payload_root, may_exist=True)
    output_root = require_authorized(args.output_root, may_exist=False)
    audit_root = require_authorized(args.audit_root, may_exist=False)
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeInstallError(f"runtime output reuse forbidden: {output_root}")
    if audit_root.exists() or audit_root.is_symlink():
        raise RuntimeInstallError(f"runtime audit reuse forbidden: {audit_root}")
    audit_root.mkdir(parents=True, exist_ok=False)
    build_root = output_root.with_name(f".{output_root.name}.build.{os.getpid()}")
    if build_root.exists() or build_root.is_symlink():
        raise RuntimeInstallError(f"runtime build root reuse forbidden: {build_root}")
    manifest_path = payload_root / "RUNTIME_INPUT_MANIFEST.json"
    manifest, payload_audit = validate_payload(
        payload_root, manifest_path, args.expected_manifest_sha256
    )
    build_root.mkdir(parents=True, exist_ok=False)
    archive_path = payload_root / str(manifest["python"]["filename"])
    with tarfile.open(archive_path, "r:gz") as archive:
        members = validate_tar_members(archive)
        archive.extractall(build_root, members=members)
    python = build_root / "python" / "bin" / "python3.13"
    if python.is_symlink() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeInstallError("extracted standalone Python is absent/unsafe")
    environment = offline_environment(build_root, python.parent)
    wheel_root = payload_root / "wheels"
    requirements = [f"{name}=={version}" for name, version in EXPECTED_VERSIONS.items()]
    install = run_checked(
        [
            str(python),
            "-I",
            "-s",
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-cache-dir",
            "--disable-pip-version-check",
            "--no-compile",
            "--find-links",
            str(wheel_root),
            *requirements,
        ],
        environment=environment,
    )
    probe_before = probe_runtime(python, environment)
    license_files = sorted(
        path
        for path in (build_root / "python" / "lib" / "python3.13" / "site-packages").rglob("*")
        if path.is_file()
        and ("license" in path.name.lower() or "copying" in path.name.lower())
    )
    if len(license_files) < sum(
        int(row.get("embedded_license_files", 0)) for row in manifest["wheels"]
    ):
        raise RuntimeInstallError("installed runtime lost embedded wheel licenses")
    normalized_before = normalize_probe(
        probe_before, str(build_root), str(output_root)
    )
    os.rename(build_root, output_root)
    final_python = output_root / "python" / "bin" / "python3.13"
    final_environment = offline_environment_for_existing(output_root, final_python.parent)
    probe_after = probe_runtime(final_python, final_environment)
    if normalized_before != probe_after:
        raise RuntimeInstallError("runtime changed across atomic rename")
    binding: dict[str, Any] = {
        "format": FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "status": "PASS_OFFLINE_STANDALONE_RUNTIME_BOUND",
        "runtime_root": str(output_root),
        "python_path": str(final_python),
        "python_sha256": sha256_file(final_python),
        "python_bytes": final_python.stat().st_size,
        "expected_versions": EXPECTED_VERSIONS,
        "runtime_probe": probe_after,
        "payload": payload_audit,
        "license_file_count_installed": len(license_files),
        "license_filename_set_sha256": canonical_sha256(
            [str(path.relative_to(build_root / "python")) for path in license_files]
        ),
        "pip_install_stdout_sha256": hashlib.sha256(install.stdout.encode()).hexdigest(),
        "pip_install_stderr_sha256": hashlib.sha256(install.stderr.encode()).hexdigest(),
        "offline_install": True,
        "reads_dsk2": False,
        "writes_dsk2": False,
        "writes_tmp": False,
        "writes_root_filesystem": False,
        "production_deployed": False,
        "port_8260_touched": False,
        "timestamp_unix": time.time(),
    }
    binding["binding_contract_sha256"] = canonical_sha256(binding)
    binding_path = audit_root / "RUNTIME_BINDING.json"
    exclusive_json(binding_path, binding)
    success = {
        "format": FORMAT,
        "status": "PASS_OFFLINE_STANDALONE_RUNTIME_BOUND",
        "runtime_binding_path": str(binding_path),
        "runtime_binding_sha256": sha256_file(binding_path),
        "runtime_root": str(output_root),
        "runtime_root_created_atomically": True,
        "formal_scientific_run_started": False,
        "production_deployed": False,
        "port_8260_touched": False,
    }
    success["success_contract_sha256"] = canonical_sha256(success)
    exclusive_json(audit_root / "SUCCESS.json", success)
    return success


def offline_environment_for_existing(
    runtime_root: Path, runtime_bin: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "CONDA_PYTHON_EXE",
        "_CE_CONDA",
        "_CE_M",
    ):
        environment.pop(key, None)
    temporary = runtime_root / "offline_temp"
    pip_cache = runtime_root / "offline_pip_cache"
    xdg_cache = runtime_root / "offline_xdg_cache"
    if any(not path.is_dir() or path.is_symlink() for path in (temporary, pip_cache, xdg_cache)):
        raise RuntimeInstallError("pre-created offline runtime cache roots are absent/unsafe")
    environment.update(
        {
            "PATH": f"{runtime_bin}:/usr/bin:/bin",
            "TMPDIR": str(temporary),
            "TMP": str(temporary),
            "TEMP": str(temporary),
            "PIP_CACHE_DIR": str(pip_cache),
            "XDG_CACHE_HOME": str(xdg_cache),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        }
    )
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audit_root = args.audit_root.resolve()
    try:
        result = execute(args)
    except Exception as exc:
        try:
            if AUTHORIZED_ROOT.resolve() in audit_root.parents:
                audit_root.mkdir(parents=True, exist_ok=True)
                failure = {
                    "format": FAILURE_FORMAT,
                    "status": "TYPED_FAILURE_PRESERVED",
                    "reason": f"{type(exc).__name__}:{exc}",
                    "payload_root": str(args.payload_root),
                    "output_root": str(args.output_root),
                    "audit_root": str(args.audit_root),
                    "reads_dsk2": False,
                    "writes_dsk2": False,
                    "writes_tmp": False,
                    "production_deployed": False,
                    "port_8260_touched": False,
                    "timestamp_unix": time.time(),
                }
                failure["failure_contract_sha256"] = canonical_sha256(failure)
                exclusive_json(audit_root / "TYPED_FAILURE.json", failure)
        except Exception:
            pass
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
