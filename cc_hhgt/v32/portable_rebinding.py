"""Create a provenance-preserving Linux overlay for V3.2 JSON bindings.

Payload bytes are never modified.  JSON declarations are rewritten in a
dependency order, and every SHA-256 cross-reference to a rewritten JSON is
updated.  Unresolved or out-of-scope paths make the owning capability pending;
they are never silently converted to zero-valued data or a successful mount.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from .production_binding_audit import sha256_file


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_FILESYSTEM_POSIX = re.compile(
    r"^/(?:public\d+|dell_\d+|dsk\d+|home|mnt|opt|srv|var|tmp)(?:/|$)"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PortableRebindingError(RuntimeError):
    """Raised when a binding graph cannot be safely and deterministically rebound."""


@dataclass(frozen=True)
class SourceMap:
    source_prefix: PureWindowsPath
    local_root: Path
    target_relative_root: PurePosixPath


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_immutable(path: Path, payload: bytes, *, allow_identical: bool = False) -> None:
    if path.exists():
        if allow_identical and path.is_file() and path.read_bytes() == payload:
            return
        raise PortableRebindingError(f"Refusing to overwrite candidate artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def payload_summary(path: str | Path) -> dict[str, Any]:
    """Hash a file or directory without following any symlink."""

    source = Path(path)
    if source.is_symlink():
        raise PortableRebindingError(f"Payload may not be a symlink: {source}")
    resolved = source.resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "file_count": 1,
        }
    if not resolved.is_dir():
        raise PortableRebindingError(f"Payload is missing: {resolved}")
    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for item in sorted(resolved.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise PortableRebindingError(f"Payload tree contains a symlink: {item}")
        if not item.is_file():
            continue
        real = item.resolve()
        try:
            relative = real.relative_to(resolved)
        except ValueError as exc:
            raise PortableRebindingError(f"Payload escapes its root: {item}") from exc
        file_sha = sha256_file(real)
        size = real.stat().st_size
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha))
        digest.update(b"\0")
        total_bytes += size
        file_count += 1
    if file_count == 0:
        raise PortableRebindingError(f"Payload directory is empty: {resolved}")
    return {
        "kind": "directory",
        "tree_sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "file_count": file_count,
    }


def _walk(value: Any, pointer: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}"
            yield child, str(key), item
            yield from _walk(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{pointer}/{index}"
            yield child, "", item
            yield from _walk(item, child)


def _replace(value: Any, fn, pointer: str = "") -> Any:
    if isinstance(value, dict):
        return {
            key: _replace(
                item,
                fn,
                f"{pointer}/{str(key).replace('~', '~0').replace('/', '~1')}",
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace(item, fn, f"{pointer}/{index}")
            for index, item in enumerate(value)
        ]
    return fn(pointer, value)


class PortableRebinder:
    def __init__(
        self,
        *,
        source_repo_root: str | Path,
        overlay_root: str | Path,
        target_root: str,
        source_maps: Iterable[SourceMap],
        allowed_server_roots: Iterable[str],
    ) -> None:
        self.source_repo_root = Path(source_repo_root).resolve()
        self.overlay_root = Path(overlay_root).resolve()
        self.target_root = PurePosixPath(target_root)
        if not self.target_root.is_absolute():
            raise PortableRebindingError("target_root must be an absolute POSIX path")
        self.source_maps = tuple(
            sorted(source_maps, key=lambda item: len(item.source_prefix.parts), reverse=True)
        )
        self.allowed_server_roots = tuple(
            str(PurePosixPath(value)) for value in allowed_server_roots
        )
        if not any(
            str(self.target_root) == item
            or str(self.target_root).startswith(f"{item}/")
            for item in self.allowed_server_roots
        ):
            raise PortableRebindingError("target_root is outside the authorized roots")
        if self.overlay_root.exists():
            if self.overlay_root.is_symlink() or not self.overlay_root.is_dir():
                raise PortableRebindingError("overlay_root must be a real directory")
            if any(self.overlay_root.iterdir()):
                raise PortableRebindingError("overlay_root must be empty and immutable")
        else:
            self.overlay_root.mkdir(parents=True)
        self._visiting: set[Path] = set()
        self._rewritten: dict[Path, dict[str, Any]] = {}
        self._old_to_new_sha: dict[str, str] = {}
        self._path_rewrites: list[dict[str, Any]] = []
        self._unresolved: list[dict[str, Any]] = []
        self._payloads: dict[tuple[str, str], dict[str, Any]] = {}
        self._expanded_payload_dirs: set[tuple[str, str]] = set()

    def _map_windows(self, raw: str) -> tuple[Path, PurePosixPath] | None:
        source = PureWindowsPath(raw)
        for mapping in self.source_maps:
            try:
                relative = source.relative_to(mapping.source_prefix)
            except ValueError:
                continue
            local = mapping.local_root.joinpath(*relative.parts).resolve()
            target = self.target_root / mapping.target_relative_root / PurePosixPath(
                *relative.parts
            )
            return local, target
        return None

    def _target_for_local(self, local: Path) -> PurePosixPath | None:
        resolved = local.resolve()
        for mapping in self.source_maps:
            try:
                relative = resolved.relative_to(mapping.local_root.resolve())
            except ValueError:
                continue
            return self.target_root / mapping.target_relative_root / PurePosixPath(
                *relative.parts
            )
        return None

    def _record_payload(self, local: Path, target: PurePosixPath) -> None:
        local = local.resolve()
        # A serialized repo/workspace root is provenance about the build
        # environment, not a payload directory.  Treating it as data would
        # recursively copy the whole developer workspace and later collide
        # with the rebound JSON overlay.
        if any(local == mapping.local_root.resolve() for mapping in self.source_maps):
            return
        if local.is_dir():
            directory_key = (str(local), str(target))
            if directory_key in self._expanded_payload_dirs:
                return
            self._expanded_payload_dirs.add(directory_key)
            found = False
            for item in sorted(local.rglob("*"), key=lambda value: value.as_posix()):
                if item.is_symlink():
                    raise PortableRebindingError(
                        f"Payload tree contains a symlink: {item}"
                    )
                if not item.is_file():
                    continue
                found = True
                relative = item.relative_to(local)
                self._record_payload(
                    item,
                    target / PurePosixPath(*relative.parts),
                )
            if not found:
                raise PortableRebindingError(f"Payload directory is empty: {local}")
            return
        key = (str(local), str(target))
        if key in self._payloads:
            return
        record: dict[str, Any] = {
            "source_path": str(local),
            "target_path": str(target),
            "exists": local.exists(),
            "target_relative_path": str(target.relative_to(self.target_root)),
        }
        if local.exists():
            record.update(payload_summary(local))
        self._payloads[key] = record

    def _discover_json_dependencies(self, source: Path, payload: Any) -> list[Path]:
        dependencies: set[Path] = set()
        for pointer, key, item in _walk(payload):
            if not isinstance(item, str):
                continue
            local: Path | None = None
            target: PurePosixPath | None = None
            if _WINDOWS_ABSOLUTE.match(item):
                mapped = self._map_windows(item)
                if mapped is not None:
                    local, target = mapped
            elif not item.startswith("/") and ("path" in key.lower() or item.endswith(".json")):
                candidates = ((source.parent / item).resolve(), (self.source_repo_root / item).resolve())
                local = next((candidate for candidate in candidates if candidate.exists()), None)
                if local is not None:
                    target = self._target_for_local(local)
            if local is None or target is None:
                continue
            if local.is_file() and local.suffix.lower() == ".json":
                dependencies.add(local)
            elif local.exists():
                self._record_payload(local, target)
        return sorted(dependencies, key=str)

    def rebind_json(self, source_path: str | Path) -> dict[str, Any]:
        source = Path(source_path).resolve()
        if source in self._rewritten:
            return self._rewritten[source]
        if source in self._visiting:
            raise PortableRebindingError(f"JSON binding cycle detected at {source}")
        if not source.is_file() or source.suffix.lower() != ".json":
            raise PortableRebindingError(f"Binding JSON is missing: {source}")
        target = self._target_for_local(source)
        if target is None:
            raise PortableRebindingError(f"No target mapping for JSON: {source}")

        self._visiting.add(source)
        original_bytes = source.read_bytes()
        try:
            original = json.loads(original_bytes)
        except json.JSONDecodeError as exc:
            raise PortableRebindingError(f"Invalid binding JSON: {source}") from exc
        dependencies = self._discover_json_dependencies(source, original)
        cycle_dependencies = [item for item in dependencies if item in self._visiting]
        dependency_records = [
            self.rebind_json(item)
            for item in dependencies
            if item not in self._visiting
        ]

        local_rewrites: list[dict[str, Any]] = []
        local_unresolved: list[dict[str, Any]] = [
            {
                "pointer": "",
                "kind": "json_dependency_cycle",
                "value": str(item),
            }
            for item in cycle_dependencies
        ]

        def transform(pointer: str, item: Any) -> Any:
            if not isinstance(item, str):
                return item
            lowered = item.lower()
            if _SHA256.fullmatch(lowered) and lowered in self._old_to_new_sha:
                replacement = self._old_to_new_sha[lowered]
                local_rewrites.append(
                    {"pointer": pointer, "kind": "json_sha256", "old": item, "new": replacement}
                )
                return replacement
            if _WINDOWS_ABSOLUTE.match(item):
                mapped = self._map_windows(item)
                if mapped is None:
                    local_unresolved.append(
                        {"pointer": pointer, "kind": "windows_unmapped", "value": item}
                    )
                    return item
                local, rebound = mapped
                if not local.exists():
                    local_unresolved.append(
                        {
                            "pointer": pointer,
                            "kind": "windows_source_missing",
                            "value": item,
                            "mapped_source": str(local),
                        }
                    )
                    return item
                if local.is_file() and local.suffix.lower() != ".json":
                    self._record_payload(local, rebound)
                elif local.is_dir():
                    self._record_payload(local, rebound)
                replacement = str(rebound)
                local_rewrites.append(
                    {"pointer": pointer, "kind": "path", "old": item, "new": replacement}
                )
                return replacement
            if _FILESYSTEM_POSIX.match(item):
                in_scope = any(
                    item == root or item.startswith(f"{root}/")
                    for root in self.allowed_server_roots
                )
                if not in_scope:
                    local_unresolved.append(
                        {"pointer": pointer, "kind": "server_path_out_of_scope", "value": item}
                    )
            return item

        rebound_payload = _replace(copy.deepcopy(original), transform)
        rebound_bytes = _canonical_json(rebound_payload)
        source_sha = hashlib.sha256(original_bytes).hexdigest()
        rebound_sha = hashlib.sha256(rebound_bytes).hexdigest()
        target_relative = target.relative_to(self.target_root)
        overlay_path = self.overlay_root.joinpath(*target_relative.parts)
        _write_immutable(overlay_path, rebound_bytes)
        original_relative = PurePosixPath(
            f"_provenance/original_json/{source_sha}.json"
        )
        original_copy = self.overlay_root.joinpath(*original_relative.parts)
        _write_immutable(original_copy, original_bytes, allow_identical=True)
        dependency_unresolved_count = sum(
            int(item["transitive_unresolved_count"]) for item in dependency_records
        )
        record = {
            "source_path": str(source),
            "source_sha256": source_sha,
            "original_copy_target_path": str(self.target_root / original_relative),
            "original_copy_overlay_path": str(original_copy),
            "target_path": str(target),
            "overlay_path": str(overlay_path),
            "rebound_sha256": rebound_sha,
            "dependency_source_paths": [item["source_path"] for item in dependency_records],
            "rewrite_count": len(local_rewrites),
            "unresolved_count": len(local_unresolved),
            "dependency_unresolved_count": dependency_unresolved_count,
            "transitive_unresolved_count": (
                len(local_unresolved) + dependency_unresolved_count
            ),
            "rewrites": local_rewrites,
            "unresolved": local_unresolved,
        }
        self._rewritten[source] = record
        self._old_to_new_sha[source_sha] = rebound_sha
        self._path_rewrites.extend(
            {"json_source": str(source), **item} for item in local_rewrites
        )
        self._unresolved.extend(
            {"json_source": str(source), **item} for item in local_unresolved
        )
        self._visiting.remove(source)
        return record

    def build_unified(self, source_unified: str | Path) -> dict[str, Any]:
        source = Path(source_unified).resolve()
        manifest = json.loads(source.read_text(encoding="utf-8"))
        candidate = copy.deepcopy(manifest)
        capability_results: dict[str, Any] = {}

        registry_source = (self.source_repo_root / manifest["registry"]["path"]).resolve()
        registry_record = self.rebind_json(registry_source)
        registry_relative = PurePosixPath(registry_record["target_path"]).relative_to(
            self.target_root
        )
        candidate["registry"]["path"] = str(registry_relative)
        candidate["registry"]["sha256"] = registry_record["rebound_sha256"]

        atomic_pairs = (
            ("exact_pathway_report", "exact_pathway_report_independent_audit"),
            ("evidence_direction_probabilities", "evidence_direction_probabilities_independent_audit"),
            ("experiment_evidence_bridge", "experiment_evidence_bridge_audit"),
            ("single_cell_gap_audit", "single_cell_gap_independent_audit"),
            ("single_cell_formal_context", "single_cell_formal_context_independent_audit"),
            ("gene_set_ranked_subtype", "gene_set_ranked_subtype_independent_audit"),
            ("historical_artifact_remediation", "historical_artifact_remediation_independent_audit"),
            ("download_catalog", "download_catalog_independent_audit"),
            ("unified_network", "unified_network_independent_audit"),
        )
        blocked: set[str] = set()
        for capability, entry in manifest.get("bindings", {}).items():
            status = entry.get("status")
            if status == "SERVER_MOUNT_READY_HASH_PINNED":
                blocked.add(capability)
                capability_results[capability] = {
                    "status": "PENDING_FORMAL_SUCCESS",
                    "reason": "server-ready binding is not auto-activated by portable rebinding",
                }
                continue
            if status != "MOUNTED_HASH_PINNED":
                capability_results[capability] = {"status": status}
                continue
            binding_source = (self.source_repo_root / entry["path"]).resolve()
            record = self.rebind_json(binding_source)
            if record["transitive_unresolved_count"]:
                blocked.add(capability)
                capability_results[capability] = {
                    "status": "PENDING_FORMAL_SUCCESS",
                    "reason": "unresolved or out-of-authority paths remain after rebinding",
                    "unresolved_count": record["transitive_unresolved_count"],
                }
                continue
            relative = PurePosixPath(record["target_path"]).relative_to(self.target_root)
            candidate["bindings"][capability]["path"] = str(relative)
            candidate["bindings"][capability]["sha256"] = record["rebound_sha256"]
            capability_results[capability] = {
                "status": "MOUNTED_HASH_PINNED",
                "source_sha256": record["source_sha256"],
                "rebound_sha256": record["rebound_sha256"],
                "target_path": record["target_path"],
            }

        for left, right in atomic_pairs:
            if left in blocked or right in blocked:
                blocked.update((left, right))
        registry_blocked = bool(registry_record["transitive_unresolved_count"])
        if registry_blocked:
            for capability, result in capability_results.items():
                if result.get("status") == "MOUNTED_HASH_PINNED":
                    blocked.add(capability)
                    result["reason"] = (
                        "core registry retains transitive unresolved paths; "
                        "runtime registry validation is not provable"
                    )
        for capability in blocked:
            source_entry = manifest["bindings"][capability]
            candidate["bindings"][capability] = {
                "status": "PENDING_FORMAL_SUCCESS",
                "reason": capability_results.get(capability, {}).get(
                    "reason",
                    "paired source/audit binding was not safely rebound",
                ),
                "production_deployed": False,
            }
            capability_results[capability]["status"] = "PENDING_FORMAL_SUCCESS"

        candidate["environment"] = "staging"
        candidate["production_deployed"] = False
        candidate["release_ready"] = False
        candidate_relative = PurePosixPath("config/v32_portable_candidate_bindings.json")
        candidate_path = self.overlay_root.joinpath(*candidate_relative.parts)
        candidate_bytes = _canonical_json(candidate)
        _write_immutable(candidate_path, candidate_bytes)
        candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()

        # Rebound JSON is supplied by the immutable overlay, so its original
        # source bytes must not also be copied to the same target.  All data
        # directories were expanded to leaf files above, which makes this
        # target-level exclusion deterministic and avoids directory-tree hash
        # conflicts after JSON rebinding.
        rebound_json_targets = {
            item["target_path"] for item in self._rewritten.values()
        }
        materialized_payloads = sorted(
            (
                item
                for item in self._payloads.values()
                if item["target_path"] not in rebound_json_targets
            ),
            key=lambda item: item["target_path"],
        )
        copy_manifest = {
            "format": "CANCERLNCATLAS_V32_PORTABLE_COPY_MANIFEST_V1",
            "target_root": str(self.target_root),
            "payloads_are_byte_identical": True,
            "symlinks_permitted": False,
            "overwrite_permitted": False,
            "entries": materialized_payloads,
        }
        copy_manifest_path = self.overlay_root / "COPY_MANIFEST.json"
        copy_manifest_bytes = _canonical_json(copy_manifest)
        _write_immutable(copy_manifest_path, copy_manifest_bytes)
        copy_manifest_sha = hashlib.sha256(copy_manifest_bytes).hexdigest()

        lineage = {
            "format": "CANCERLNCATLAS_V32_PORTABLE_REBINDING_LINEAGE_V1",
            "production_deployed": False,
            "release_ready": False,
            "source_unified": {
                "path": str(source),
                "sha256": sha256_file(source),
            },
            "candidate_unified": {
                "target_path": str(self.target_root / candidate_relative),
                "overlay_path": str(candidate_path),
                "sha256": candidate_sha,
            },
            "target_root": str(self.target_root),
            "allowed_server_roots": list(self.allowed_server_roots),
            "capabilities": capability_results,
            "json_rewrites": sorted(self._rewritten.values(), key=lambda item: item["target_path"]),
            "payload_invariants": materialized_payloads,
            "unresolved": self._unresolved,
            "registry_transitive_unresolved_count": registry_record[
                "transitive_unresolved_count"
            ],
            "runtime_create_app_smoke_required": True,
            "copy_manifest": {
                "target_path": str(self.target_root / "COPY_MANIFEST.json"),
                "overlay_path": str(copy_manifest_path),
                "sha256": copy_manifest_sha,
                "entry_count": len(copy_manifest["entries"]),
            },
            "mounted_capability_count": sum(
                item["status"] == "MOUNTED_HASH_PINNED"
                for item in capability_results.values()
            ),
            "pending_capability_count": sum(
                item["status"] == "PENDING_FORMAL_SUCCESS"
                for item in capability_results.values()
            ),
        }
        lineage_path = self.overlay_root / "PORTABLE_REBINDING_LINEAGE.json"
        _write_immutable(lineage_path, _canonical_json(lineage))
        return lineage


__all__ = [
    "PortableRebinder",
    "PortableRebindingError",
    "SourceMap",
    "payload_summary",
]
