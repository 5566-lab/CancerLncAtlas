"""Read-only State GeneSet queries backed by a joint State context binding."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from .current_state_context import ANALYSIS_VERSION, CurrentStateContextQuery, FORMAT

MAX_GENE_SET_LIMIT = 500
MAX_MEMBER_LIMIT = 500
MAX_QUERY_OFFSET = 100_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANCER = re.compile(r"^[A-Z0-9_-]{2,16}$")


class StateGeneSetQueryError(RuntimeError): pass
class StateGeneSetQueryAssetError(StateGeneSetQueryError): pass
class StateGeneSetQueryInputError(StateGeneSetQueryError): pass


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    value = Path(path).resolve()
    if not value.is_file() or value.is_symlink():
        raise StateGeneSetQueryAssetError(f"{label} is missing or unsafe: {value}")
    return value


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, np.generic): value = value.item()
            if isinstance(value, float) and not math.isfinite(value): value = None
            try:
                if bool(pd.isna(value)): value = None
            except (TypeError, ValueError):
                pass
            clean[str(key)] = value
        records.append(clean)
    return records


def _bounds(limit: int, offset: int, *, maximum: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= maximum:
        raise StateGeneSetQueryInputError(f"limit must be 1..{maximum}")
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise StateGeneSetQueryInputError(f"offset must be 0..{MAX_QUERY_OFFSET}")
    return int(limit), int(offset)


class StateGeneSetReleaseQuery:
    """Compatibility API whose only accepted producer is the joint context."""

    def __init__(self, binding_path: str | Path, *, expected_binding_sha256: str | None,
                 require_formal_authority: bool = True) -> None:
        source = _safe_file(binding_path, "Joint State context binding")
        expected = str(expected_binding_sha256 or "").lower()
        if not _SHA256.fullmatch(expected):
            raise StateGeneSetQueryAssetError("State GeneSet query requires an expected binding SHA256")
        observed = artifact_sha256(source)
        if observed != expected:
            raise StateGeneSetQueryAssetError(f"State GeneSet binding SHA mismatch: {observed} != {expected}")
        try:
            import json
            binding = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise StateGeneSetQueryAssetError("Joint State binding is invalid JSON") from error
        if binding.get("format") != FORMAT:
            raise StateGeneSetQueryAssetError("Retired frozen State GeneSet bindings are not accepted")
        try:
            reader = CurrentStateContextQuery(source, expected_sha=observed,
                selected_analysis_sha=binding.get("selected_analysis", {}).get("sha256"))
        except Exception as error:
            raise StateGeneSetQueryAssetError(str(error)) from error
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.computation_run_id = str(binding.get("release_id", ""))
        self.paths = {"catalog": reader.paths["state_geneset_catalog"],
            "members": reader.paths["state_geneset_members"], "gmt": reader.paths["gmt"]}
        self.require_formal_authority = require_formal_authority
        with duckdb.connect(":memory:") as connection:
            self._state_ids = tuple(row[0] for row in connection.execute(
                f"SELECT DISTINCT state_id FROM read_parquet({_sql_path(self.paths['catalog'])}) ORDER BY state_id").fetchall())

    def _connect(self): return duckdb.connect(":memory:")

    def _provenance(self) -> dict[str, Any]:
        return {"binding_sha256": self.binding_sha256,
            "computation_run_id": self.computation_run_id,
            "source_model_family": "cc_hhgt_shared_state_head",
            "source_initialization_policy": "JOINT_SHARED_ENCODER_FROM_SCRATCH",
            "state_gradients_into_core": True,
            "joint_set_status": "DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE",
            "changes_primary_ranking": False, "release_ready": False,
            "production_deployed": False}

    def query_gene_sets(self, *, state_id: str | None = None,
                        cancer_id: str | None = None, scope: str | None = None,
                        direction: str | None = None, limit: int = 50,
                        offset: int = 0) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset, maximum=MAX_GENE_SET_LIMIT)
        values = {item.casefold(): item for item in self._state_ids}
        if state_id is not None:
            key = str(state_id).strip().casefold()
            if key not in values: raise StateGeneSetQueryInputError(f"Unknown state_id: {state_id}")
            state_id = values[key]
        cancer = None
        if cancer_id is not None:
            cancer = str(cancer_id).strip().upper()
            if not _CANCER.fullmatch(cancer): raise StateGeneSetQueryInputError("cancer_id is invalid")
        if scope is not None:
            scope = str(scope).strip().upper()
            if scope != "CANCER_SPECIFIC": raise StateGeneSetQueryInputError("scope must be CANCER_SPECIFIC")
        if direction is not None:
            direction = str(direction).strip().lower()
            if direction not in {"positive", "negative", "mixed"}:
                raise StateGeneSetQueryInputError("direction must be positive, negative or mixed")
        predicates, parameters = [], []
        for column, value in (("state_id", state_id), ("cancer_id", cancer),
                              ("scope", scope), ("direction", direction)):
            if value is not None:
                predicates.append(f"{column} = ?"); parameters.append(value)
        where = " WHERE " + " AND ".join(predicates) if predicates else ""
        relation = f"read_parquet({_sql_path(self.paths['catalog'])})"
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT count(*) FROM {relation}{where}", parameters).fetchone()[0])
            rows = connection.execute(f"SELECT * FROM {relation}{where} ORDER BY state_id,scope,cancer_id,direction,gene_set_id LIMIT ? OFFSET ?",
                [*parameters, limit, offset]).fetchdf()
        return {"analysis_version": ANALYSIS_VERSION, "query_type": "state_gene_sets",
            "total_rows": total, "returned_rows": len(rows), "limit": limit,
            "offset": offset, "rows": _records(rows), "provenance": self._provenance()}

    def query_members(self, *, gene_set_id: str, limit: int = 200,
                      offset: int = 0) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset, maximum=MAX_MEMBER_LIMIT)
        identifier = str(gene_set_id or "").strip()
        if not identifier or len(identifier) > 256 or any(ord(char) < 32 for char in identifier):
            raise StateGeneSetQueryInputError("gene_set_id is required and must be valid")
        relation = f"read_parquet({_sql_path(self.paths['members'])})"
        with self._connect() as connection:
            total = int(connection.execute(f"SELECT count(*) FROM {relation} WHERE gene_set_id=?", [identifier]).fetchone()[0])
            rows = connection.execute(f"SELECT * FROM {relation} WHERE gene_set_id=? ORDER BY member_rank LIMIT ? OFFSET ?",
                [identifier, limit, offset]).fetchdf()
        return {"analysis_version": ANALYSIS_VERSION,
            "query_type": "state_gene_set_members", "gene_set_id": identifier,
            "total_rows": total, "returned_rows": len(rows), "limit": limit,
            "offset": offset, "rows": _records(rows), "provenance": self._provenance()}


__all__ = ["MAX_GENE_SET_LIMIT", "MAX_MEMBER_LIMIT", "MAX_QUERY_OFFSET",
    "StateGeneSetQueryAssetError", "StateGeneSetQueryInputError",
    "StateGeneSetReleaseQuery", "artifact_sha256"]
