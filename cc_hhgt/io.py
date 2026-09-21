from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .common import LOGGER, first_existing_column, input_candidates, input_path, read_collection, read_table, resolve_input_path


def discover_preferred(cfg: dict[str, Any], formal_key: str, staging_glob_key: str) -> tuple[list[Path], str]:
    formal = input_path(cfg, formal_key)
    if formal and formal.exists():
        return [formal], "formal"
    pattern = cfg.get("inputs", {}).get(staging_glob_key)
    if not pattern:
        return [], "missing"
    full_pattern = str(resolve_input_path(cfg, pattern))
    paths = [Path(x) for x in sorted(glob.glob(full_pattern))]
    return paths, "staging" if paths else "missing"


def read_preferred_collection(cfg: dict[str, Any], formal_key: str, staging_glob_key: str, columns: Sequence[str] | None = None) -> tuple[pd.DataFrame, str]:
    paths, source = discover_preferred(cfg, formal_key, staging_glob_key)
    if not paths:
        return pd.DataFrame(), source
    if len(paths) == 1 and paths[0].is_dir():
        return read_table(paths[0], columns=columns), source
    return read_collection(paths, columns=columns, add_source_path=True), source


def read_glob_collection(cfg: dict[str, Any], key: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    import glob as glob_module
    pattern = cfg.get("inputs", {}).get(key)
    if not pattern:
        return pd.DataFrame()
    full = str(resolve_input_path(cfg, pattern))
    paths = [Path(x) for x in sorted(glob_module.glob(full))]
    return read_collection(paths, columns=columns, add_source_path=True)


def normalize_expression_columns(df: pd.DataFrame, entity_col: str, aliases: Sequence[str]) -> pd.DataFrame:
    value_col = first_existing_column(df, aliases)
    required = ["cancer_id", "sample_id", entity_col, value_col]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"Expression table missing {missing}")
    rename = {value_col: "value"}
    out = df.rename(columns=rename)
    keep = [x for x in ["cancer_id", "sample_id", "patient_id", entity_col, "value"] if x in out.columns]
    out = out[keep].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def cancer_partition_paths(path: Path, cancer_id: str) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = sorted(path.glob(f"cancer_id={cancer_id}/**/*.parquet"))
    if candidates:
        return candidates
    return sorted(path.glob(f"**/*{cancer_id}*.parquet"))


def read_cancer_partition(path: Path, cancer_id: str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns, filters=[("cancer_id", "==", cancer_id)])
    except Exception:
        paths = cancer_partition_paths(path, cancer_id)
        if not paths:
            return pd.DataFrame()
        frames = [pd.read_parquet(p, columns=columns) for p in paths]
        out = pd.concat(frames, ignore_index=True)
        if "cancer_id" in out.columns:
            out = out.loc[out["cancer_id"].astype(str) == cancer_id]
        return out
