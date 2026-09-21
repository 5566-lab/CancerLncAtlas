"""Read-only portability audit for V3.2 website bindings."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, deque
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_SERVER_ABSOLUTE = re.compile(r"^/(?:public\d+|dell_\d+|dsk\d+|home|mnt)(?:/|$)")
_PATH_KEY = re.compile(
    r"(?:^|_)(?:path|paths|root|roots|file|files|directory|directories|dir|table|"
    r"manifest|binding|report|success|checkpoint|script|asset|payload|source|output)$",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decorated_routes(path: str | Path) -> set[tuple[str, str]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    routes: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            method = decorator.func.attr.upper()
            if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
                continue
            if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
                continue
            route = decorator.args[0].value
            if isinstance(route, str):
                routes.add((method, route))
    return routes


def _walk_strings(value: Any, pointer: str = "") -> Iterable[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
            if isinstance(item, str):
                yield child, str(key), item
            else:
                yield from _walk_strings(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{pointer}/{index}"
            if isinstance(item, str):
                yield child, "", item
            else:
                yield from _walk_strings(item, child)


def _looks_path(key: str, value: str) -> bool:
    if _WINDOWS_ABSOLUTE.match(value) or _SERVER_ABSOLUTE.match(value):
        return True
    if _PATH_KEY.search(key):
        return "/" in value or "\\" in value or "." in Path(value).name
    return False


def _windows_relative(value: str, repo_root: Path) -> Path | None:
    win = PureWindowsPath(value)
    root = PureWindowsPath(str(repo_root))
    try:
        relative = win.relative_to(root)
    except ValueError:
        return None
    return Path(*relative.parts)


def audit_binding_portability(
    unified_path: str | Path,
    *,
    repo_root: str | Path,
    allowed_server_roots: Iterable[str],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    unified_source = Path(unified_path).resolve()
    unified = json.loads(unified_source.read_text(encoding="utf-8"))
    allowed = tuple(str(item).rstrip("/") for item in allowed_server_roots)
    observations: list[dict[str, Any]] = []
    json_files: dict[str, dict[str, Any]] = {}
    capability_summary: dict[str, Counter[str]] = {}

    starts: list[tuple[str, Path]] = [("registry", root / unified["registry"]["path"])]
    for capability, entry in unified.get("bindings", {}).items():
        raw = entry.get("path") or entry.get("deployment_binding_path")
        if raw:
            starts.append((str(capability), root / str(raw)))

    for capability, start in starts:
        counts: Counter[str] = Counter()
        capability_summary[capability] = counts
        queue: deque[Path] = deque([start.resolve()])
        seen: set[Path] = set()
        while queue:
            source = queue.popleft()
            if source in seen:
                continue
            seen.add(source)
            try:
                source.relative_to(root)
            except ValueError:
                counts["json_escape"] += 1
                continue
            if not source.is_file() or source.suffix.lower() != ".json":
                counts["json_missing"] += 1
                continue
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                counts["json_invalid"] += 1
                continue
            relative_json = source.relative_to(root).as_posix()
            json_files.setdefault(
                relative_json,
                {"path": relative_json, "sha256": sha256_file(source)},
            )
            for pointer, key, value in _walk_strings(payload):
                if not _looks_path(key, value):
                    continue
                record = {
                    "capability": capability,
                    "json": relative_json,
                    "pointer": pointer,
                    "key": key,
                    "value": value,
                }
                if _WINDOWS_ABSOLUTE.match(value):
                    relative = _windows_relative(value, root)
                    if relative is None:
                        classification = "windows_absolute_outside_repo"
                        record["source_exists"] = False
                    else:
                        mapped = (root / relative).resolve()
                        exists = mapped.exists()
                        classification = (
                            "windows_absolute_repo_mappable"
                            if exists
                            else "windows_absolute_repo_missing"
                        )
                        record["repo_relative"] = relative.as_posix()
                        record["source_exists"] = exists
                        if exists and mapped.is_file() and mapped.suffix.lower() == ".json":
                            queue.append(mapped)
                elif _SERVER_ABSOLUTE.match(value):
                    in_scope = any(
                        value == prefix or value.startswith(f"{prefix}/")
                        for prefix in allowed
                    )
                    classification = (
                        "server_absolute_authorized_scope"
                        if in_scope
                        else "server_absolute_outside_authorized_scope"
                    )
                elif value.startswith("/"):
                    # Slash-prefixed strings also occur as identifiers internal to
                    # containers (for example HDF5 dataset names such as
                    # ``/matrix``).  They are not host filesystem paths and must
                    # not be reported as unauthorized server mounts.
                    classification = "opaque_nonfilesystem_path"
                else:
                    classification = "relative_path"
                    candidates = ((source.parent / value).resolve(), (root / value).resolve())
                    target = next((item for item in candidates if item.exists()), None)
                    record["source_exists"] = target is not None
                    if target is not None:
                        try:
                            record["repo_relative"] = target.relative_to(root).as_posix()
                        except ValueError:
                            pass
                        if target.is_file() and target.suffix.lower() == ".json":
                            queue.append(target)
                record["classification"] = classification
                observations.append(record)
                counts[classification] += 1

    blockers = {
        "windows_absolute_repo_mappable",
        "windows_absolute_repo_missing",
        "windows_absolute_outside_repo",
        "server_absolute_outside_authorized_scope",
        "json_escape",
        "json_missing",
        "json_invalid",
    }
    per_capability = {}
    for capability, counts in capability_summary.items():
        per_capability[capability] = {
            "counts": dict(sorted(counts.items())),
            "portable_for_authorized_server": not any(counts[item] for item in blockers),
        }
    return {
        "format": "CANCERLNCATLAS_V32_PRODUCTION_BINDING_PORTABILITY_AUDIT_V1",
        "unified_binding": {
            "path": str(unified_source),
            "sha256": sha256_file(unified_source),
        },
        "repo_root": str(root),
        "allowed_server_roots": list(allowed),
        "reachable_json_count": len(json_files),
        "observation_count": len(observations),
        "classification_counts": dict(
            sorted(Counter(item["classification"] for item in observations).items())
        ),
        "capabilities": per_capability,
        "observations": observations,
        "production_portable": all(
            item["portable_for_authorized_server"] for item in per_capability.values()
        ),
    }


__all__ = [
    "audit_binding_portability",
    "decorated_routes",
    "sha256_file",
]
