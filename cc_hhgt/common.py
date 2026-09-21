from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml

# PyTorch deterministic CUDA matrix multiplications require this to be set
# before the first cuBLAS handle is created.  Formal training imports this
# module before model construction, so establish the process contract here in
# addition to the per-seed guard below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LOGGER = logging.getLogger("cc_hhgt")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(path: Path, seen: set[Path] | None = None) -> tuple[dict[str, Any], list[Path]]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular configuration inheritance detected at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    bases = raw.pop("extends", [])
    bases = [bases] if isinstance(bases, (str, Path)) else list(bases)
    cfg: dict[str, Any] = {}
    lineage: list[Path] = []
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base_cfg, base_lineage = _load_config_mapping(base_path, seen)
        cfg = _deep_merge(cfg, base_cfg)
        lineage.extend(base_lineage)
    cfg = _deep_merge(cfg, raw)
    lineage.append(path)
    seen.remove(path)
    return cfg, lineage


def load_config(
    path: str | Path,
    *,
    project_root_override: str | Path | None = None,
    create_dirs: bool = True,
) -> dict[str, Any]:
    path = Path(path).resolve()
    cfg, lineage = _load_config_mapping(path)
    root = Path(
        project_root_override
        if project_root_override is not None
        else cfg["project_root"]
    ).resolve()
    cfg["project_root"] = str(root)
    cfg["_config_path"] = path
    cfg["_config_lineage"] = lineage
    cfg["_root"] = root
    cfg["_results"] = resolve_path(root, cfg.get("results_dir", "results/model/cc_hhgt_v2"))
    cfg["_cache"] = resolve_path(root, cfg.get("cache_dir", "results/model/cc_hhgt_v2/cache"))
    cfg["_standardized"] = resolve_path(root, cfg.get("standardized_dir", "results/model/cc_hhgt_v2/standardized"))
    if create_dirs:
        for key in ("_results", "_cache", "_standardized"):
            cfg[key].mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_path(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def input_root(cfg: Mapping[str, Any]) -> Path:
    value = cfg.get("input_root")
    return resolve_path(cfg["_root"], value) if value else cfg["_root"]


def resolve_input_path(cfg: Mapping[str, Any], value: str | Path | None) -> Path | None:
    return resolve_path(input_root(cfg), value)


def input_path(cfg: Mapping[str, Any], key: str) -> Path | None:
    value = cfg.get("inputs", {}).get(key)
    if isinstance(value, list):
        for candidate in value:
            path = resolve_input_path(cfg, candidate)
            if path and path.exists():
                return path
        return resolve_input_path(cfg, value[0]) if value else None
    return resolve_input_path(cfg, value)


def input_candidates(cfg: Mapping[str, Any], key: str) -> list[Path]:
    value = cfg.get("inputs", {}).get(key)
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [resolve_input_path(cfg, x) for x in values if x is not None]


def glob_paths(cfg: Mapping[str, Any], key: str) -> list[Path]:
    value = cfg.get("inputs", {}).get(key)
    if value is None:
        return []
    import glob as glob_module
    pattern = str(resolve_input_path(cfg, value))
    return sorted(Path(p) for p in glob_module.glob(pattern))


def expand_collection(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)
    out: list[Path] = []
    for item in items:
        text = str(item)
        if ";" in text:
            out.extend(Path(x) for x in text.split(";") if x)
        else:
            out.append(Path(text))
    return out


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def require_columns(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing required columns {missing}; observed={list(df.columns)}")


def first_existing_column(df: pd.DataFrame, aliases: Sequence[str], required: bool = True) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    if required:
        raise ValueError(f"None of columns {aliases} found. Observed: {list(df.columns)}")
    return None


def sanitize_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text.strip("_") or "unknown"


def write_json(data: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def write_table(df: pd.DataFrame, path: str | Path, partition_cols: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    if partition_cols:
        import pyarrow as pa
        import pyarrow.parquet as pq
        path.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(table, root_path=str(path), partition_cols=list(partition_cols), compression="zstd")
    elif suffixes.endswith(".parquet"):
        df.to_parquet(path, index=False, compression="zstd")
    elif suffixes.endswith(".tsv.gz"):
        df.to_csv(path, sep="\t", index=False, compression="gzip")
    elif suffixes.endswith(".tsv") or suffixes.endswith(".txt"):
        df.to_csv(path, sep="\t", index=False)
    elif suffixes.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    elif suffixes.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")
    return path


def read_table(path: str | Path, columns: Sequence[str] | None = None, filters: Any = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffixes = "".join(path.suffixes).lower()
    if path.is_dir() or suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path, columns=columns, filters=filters)
        except pa.lib.ArrowTypeError:
            # Hive-partitioned datasets that also store the partition column
            # (e.g. cancer_id) inside the files fail schema unification
            # (string vs dictionary). Fall back to per-file reads.
            if not path.is_dir():
                raise
            frames = [
                pd.read_parquet(f, columns=columns, filters=filters)
                for f in sorted(path.rglob("*.parquet"))
            ]
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz") or suffixes.endswith(".txt") or suffixes.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported input format: {path}")


def read_collection(paths: Sequence[Path], columns: Sequence[str] | None = None, add_source_path: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frame = read_table(path, columns=columns)
        if add_source_path:
            frame["_source_path"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def parquet_row_count(path: Path) -> int:
    import pyarrow.dataset as ds
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return int(dataset.count_rows())


def parquet_schema(path: Path) -> list[dict[str, str]]:
    import pyarrow.dataset as ds
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    return [{"name": f.name, "type": str(f.type)} for f in dataset.schema]


def git_or_file_version(project_dir: Path) -> str:
    try:
        result = subprocess.run(["git", "-C", str(project_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unversioned"


def dataframe_key_duplicates(df: pd.DataFrame, key: Sequence[str]) -> int:
    existing = [x for x in key if x in df.columns]
    if not existing:
        return -1
    return int(df.duplicated(existing, keep=False).sum())


def available_device(requested: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "unavailable"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


@dataclass
class StagePaths:
    root: Path

    @property
    def tables(self) -> Path:
        path = self.root / "tables"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models(self) -> Path:
        path = self.root / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports(self) -> Path:
        path = self.root / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs(self) -> Path:
        path = self.root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path


def stage_paths(cfg: Mapping[str, Any]) -> StagePaths:
    return StagePaths(Path(cfg["_results"]))


@contextmanager
def stage_status(cfg: Mapping[str, Any], stage: str) -> Iterator[dict[str, Any]]:
    status_dir = Path(cfg["_results"]) / "status"
    status_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"stage": stage, "started_at": utc_now(), "status": "RUNNING"}
    write_json(record, status_dir / f"{stage}.json")
    try:
        yield record
    except Exception as exc:
        record.update({"finished_at": utc_now(), "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        write_json(record, status_dir / f"{stage}.json")
        raise
    else:
        record.update({"finished_at": utc_now(), "status": "SUCCESS"})
        write_json(record, status_dir / f"{stage}.json")


def parse_common_args(description: str):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to model_v2.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser
