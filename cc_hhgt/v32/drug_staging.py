"""Reproducible native-source staging for the V3.2 drug module.

Only native cell-line measurements and static identifiers are admitted.  No
historical lncRNA-drug association, probability, ranking, web table or model
checkpoint is read by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
STAGING_FORMAT = "CC_HHGT_V3_2_NATIVE_DRUG_STAGING_V2_FACTORED"
FACTORED_CANDIDATE_FORMAT = "CC_HHGT_V3_2_FACTORED_DRUG_CANDIDATES_V1"
FORMAL_MIN_NATIVE_TARGET_MAPPING_FRACTION = 0.15
FORMAL_MIN_NATIVE_TARGET_MAPPED_DRUGS = 1_000
NATIVE_TARGET_IDENTITY_SCOPE = (
    "EXACT_DRUG_NAME_AFTER_UPPERCASE_ALPHANUMERIC_NORMALIZATION_NO_SYNONYM_CROSSWALK"
)
TCGA_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
TISSUES = {
    "ACC": ("Adrenal Gland",), "BLCA": ("Bladder",), "BRCA": ("Breast",),
    "CESC": ("Cervix",), "CHOL": ("Biliary Tract",), "COAD": ("Large Intestine",),
    "DLBC": ("Haematopoietic and Lymphoid",), "ESCA": ("Esophagus",),
    "GBM": ("Central Nervous System",), "HNSC": ("Head and Neck",),
    "KICH": ("Kidney",), "KIRC": ("Kidney",), "KIRP": ("Kidney",),
    "LAML": ("Haematopoietic and Lymphoid",), "LGG": ("Central Nervous System",),
    "LIHC": ("Liver",), "LUAD": ("Lung",), "LUSC": ("Lung",),
    "MESO": ("Soft Tissue",), "OV": ("Ovary",), "PAAD": ("Pancreas",),
    "PCPG": ("Peripheral Nervous System",), "PRAD": ("Prostate",),
    "READ": ("Large Intestine",), "SARC": ("Soft Tissue",), "SKCM": ("Skin",),
    "STAD": ("Stomach",), "TGCT": ("Testis",), "THCA": ("Thyroid",),
    "THYM": (), "UCEC": ("Endometrium",), "UCS": ("Endometrium", "Uterus"),
    "UVM": ("Eye",),
}
_FORBIDDEN_SOURCE_TOKENS = (
    "lncrna_drug_response_association",
    "drug_response_replication",
    "oof_prediction",
    "probability",
    "ranking",
    "ranked",
    "checkpoint",
    "web_table",
    "release_table",
)
_FORBIDDEN_EXACT_COLUMNS = {
    "label", "score", "prediction", "probability", "ranking", "rank",
    "association_membership_probability", "sample_weight", "fdr", "rho", "beta",
}
_EXACT_BINDING_KEYS = (
    "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id",
)
_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


class DrugStagingError(RuntimeError):
    """Raised when native provenance or staging completeness fails."""


@dataclass(frozen=True)
class DrugStagingConfig:
    csv_chunk_rows: int = 100_000
    prism_chunk_drugs: int = 250
    expression_chunk_genes: int = 1_000
    max_prism_drugs: int | None = None
    max_expression_lncrnas: int | None = None
    max_models: int | None = None
    candidate_storage_mode: str = "auto"

    @property
    def formal(self) -> bool:
        return all(
            value is None
            for value in (self.max_prism_drugs, self.max_expression_lncrnas, self.max_models)
        )

    def validate(self) -> None:
        if min(self.csv_chunk_rows, self.prism_chunk_drugs, self.expression_chunk_genes) < 1:
            raise ValueError("Staging chunk sizes must be positive")
        for value in (self.max_prism_drugs, self.max_expression_lncrnas, self.max_models):
            if value is not None and int(value) < 1:
                raise ValueError("Smoke limits must be positive when supplied")
        if self.candidate_storage_mode not in {"auto", "factored", "materialized"}:
            raise ValueError(
                "candidate_storage_mode must be auto, factored, or materialized"
            )
        if self.formal and self.candidate_storage_mode == "materialized":
            raise ValueError(
                "Formal Drug staging forbids the dense materialized candidate universe"
            )

    @property
    def resolved_candidate_storage_mode(self) -> str:
        if self.candidate_storage_mode != "auto":
            return self.candidate_storage_mode
        # Formal runs must not materialise the tens-of-millions-row Cartesian
        # candidate relation. Tiny bounded smoke runs remain conventional.
        return "factored" if self.formal else "materialized"


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalise_drug_name(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())


def canonical_drug_id(value: Any) -> str:
    normalised = normalise_drug_name(value)
    if not normalised:
        return ""
    return "DRUG:" + hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:16]


def _canonical_lncrna(value: Any) -> str:
    text = str(value).strip()
    if text.upper().startswith("LNC:"):
        return "LNC:" + text.split(":", 1)[1].split(".", 1)[0].upper()
    if re.fullmatch(r"ENSG\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
        return "LNC:" + text.split(".", 1)[0].upper()
    return text


def _canonical_gene(value: Any) -> str:
    text = str(value).strip()
    if text.upper().startswith("GENE:"):
        return "GENE:" + text.split(":", 1)[1].split(".", 1)[0].upper()
    if re.fullmatch(r"ENSG\d+(?:\.\d+)?", text, flags=re.IGNORECASE):
        return "GENE:" + text.split(".", 1)[0].upper()
    return text


def _file_sha256(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_sha256(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return _file_sha256(source)
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [item for item in source.rglob("*") if item.is_file()]
    if not files:
        raise DrugStagingError(f"Cannot hash empty directory: {source}")
    relative_names = [item.relative_to(source).as_posix() for item in files]
    if len({name.casefold() for name in relative_names}) != len(relative_names):
        raise DrugStagingError(
            f"Cannot hash directory with case-colliding paths: {source}"
        )
    # Path ordering is case-insensitive on Windows and case-sensitive on Linux.
    # Formal Drug tree hashes were generated on Windows, so sorting Path objects
    # directly made the same byte-identical mirror fail on COMPUTE_HOST.
    files.sort(key=lambda item: item.relative_to(source).as_posix().casefold())
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(source).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _snapshot_artifacts(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash provenance artifacts without trusting an earlier manifest."""

    snapshot: dict[str, dict[str, Any]] = {}
    for role, raw_path in sorted(paths.items()):
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        snapshot[str(role)] = {
            "path": str(path),
            "artifact_type": "file" if path.is_file() else "directory",
            "bytes": _path_size_bytes(path),
            "sha256": artifact_sha256(path),
        }
    return snapshot


def _verify_artifact_snapshot(
    start: Mapping[str, Mapping[str, Any]], *, label: str
) -> dict[str, Any]:
    paths = {role: Path(str(record["path"])) for role, record in start.items()}
    end = _snapshot_artifacts(paths)
    changed = [
        role for role in sorted(start)
        if dict(start[role]) != dict(end.get(role, {}))
    ]
    if changed:
        details = {
            role: {"start": dict(start[role]), "end": end.get(role)}
            for role in changed
        }
        raise DrugStagingError(
            f"{label} changed during staging: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )
    return {"start": dict(start), "end": end, "unchanged": True}


def _runtime_fingerprint() -> dict[str, Any]:
    import duckdb
    import pyarrow

    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "numpy": str(np.__version__),
            "pandas": str(pd.__version__),
            "pyarrow": str(pyarrow.__version__),
            "duckdb": str(duckdb.__version__),
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "thread_environment": {
            key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_VARIABLES
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_native_path(path: Path, role: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    # Inspect the artifact name itself.  Repository ancestors can legitimately
    # contain words such as ``ranked_subtypes`` without making a native file a
    # historical ranking.
    token = _normalise_token(path.name)
    found = [forbidden for forbidden in _FORBIDDEN_SOURCE_TOKENS if forbidden in token]
    if found:
        raise DrugStagingError(f"Forbidden historical result supplied as {role}: {path}; {found}")


def _header(path: Path) -> list[str]:
    suffixes = [value.lower() for value in path.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] == ".gz" else path.suffix.lower()
    if logical == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if logical == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return sorted(map(str, payload.keys())) if isinstance(payload, Mapping) else []
    separator = "\t" if logical in {".tsv", ".txt"} else ","
    return list(pd.read_csv(path, sep=separator, nrows=0, compression="infer").columns.astype(str))


def audit_native_inputs(
    specs: Sequence[tuple[str, Path, str]],
    *,
    precomputed_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for role, path, source_kind in specs:
        _assert_native_path(path, role)
        columns = _header(path)
        cached = (precomputed_snapshot or {}).get(role)
        if cached is not None and Path(str(cached.get("path"))).resolve() != path.resolve():
            raise DrugStagingError(f"Precomputed native snapshot path mismatch for {role}")
        rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "source_kind": source_kind,
                "bytes": int(cached["bytes"]) if cached is not None else int(path.stat().st_size),
                "sha256": str(cached["sha256"]) if cached is not None else _file_sha256(path),
                "columns": columns,
                "historical_model_result": False,
            }
        )
    payload: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "audit_mode": "READ_ONLY_NATIVE_INPUTS",
        "status": "PASS",
        "old_association_tables_read": False,
        "inputs": rows,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


class _ParquetSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_name(f".{path.name}.tmp")
        self.temporary.unlink(missing_ok=True)
        self.writer = None
        self.schema = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(self.temporary, self.schema, compression="zstd")
        else:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise DrugStagingError(f"No rows were written to {self.path}")
        self.writer.close()
        os.replace(self.temporary, self.path)


def load_cmp_models(path: str | Path) -> pd.DataFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    included = {
        (str(item.get("type", "")), str(item.get("id", ""))): item
        for item in payload.get("included", [])
    }
    rows: list[dict[str, str]] = []
    for model in payload.get("data", []):
        relationships = model.get("relationships", {})
        identifier_refs = relationships.get("identifiers", {}).get("data", []) or []
        identifiers = [
            included.get((str(ref.get("type", "")), str(ref.get("id", ""))), {})
            .get("attributes", {}).get("identifier", "")
            for ref in identifier_refs
        ]
        depmap_ids = sorted({str(value) for value in identifiers if str(value).startswith("ACH-")})
        sample_ref = relationships.get("sample", {}).get("data")
        sample = (
            included.get((str(sample_ref.get("type", "")), str(sample_ref.get("id", ""))), {})
            if sample_ref else {}
        )
        tissue_ref = sample.get("relationships", {}).get("tissue", {}).get("data")
        tissue = (
            included.get((str(tissue_ref.get("type", "")), str(tissue_ref.get("id", ""))), {})
            if tissue_ref else {}
        )
        tissue_name = str(tissue.get("attributes", {}).get("name", ""))
        names = model.get("attributes", {}).get("names") or []
        sidm = str(model.get("id", ""))
        for depmap in depmap_ids or [""]:
            rows.append(
                {
                    "canonical_model_id": sidm,
                    "depmap_id": depmap,
                    "tissue": tissue_name,
                    "cell_line_name": str(names[0]) if names else "",
                }
            )
    result = pd.DataFrame(rows)
    result = result.loc[result.canonical_model_id.str.startswith("SIDM")].copy()
    if result.empty:
        raise DrugStagingError("Cell Model Passports models.json yielded no SIDM models")
    return result.drop_duplicates(["canonical_model_id", "depmap_id"]).reset_index(drop=True)


def _tissue_to_cancers() -> dict[str, tuple[str, ...]]:
    inverse: dict[str, list[str]] = {}
    for cancer, tissues in TISSUES.items():
        for tissue in tissues:
            inverse.setdefault(tissue, []).append(cancer)
    return {key: tuple(sorted(value)) for key, value in inverse.items()}


def _stage_gdsc(
    sources: Sequence[Path], response_root: Path, config: DrugStagingConfig
) -> tuple[list[dict[str, str]], pd.DataFrame, list[pd.DataFrame]]:
    sink = _ParquetSink(response_root / "gdsc_auc.parquet")
    source_maps: list[pd.DataFrame] = []
    map_rows: list[dict[str, str]] = []
    audits: list[dict[str, str]] = []
    seen_models: set[str] = set()
    for source in sources:
        required = ["DATASET", "SANGER_MODEL_ID", "TCGA_DESC", "DRUG_ID", "DRUG_NAME", "AUC"]
        for chunk in pd.read_csv(
            source, usecols=required, dtype={"SANGER_MODEL_ID": str, "DRUG_ID": str},
            chunksize=config.csv_chunk_rows, low_memory=False,
        ):
            chunk = chunk.loc[chunk.SANGER_MODEL_ID.notna() & chunk.DRUG_NAME.notna()].copy()
            if config.max_models is not None:
                allowed = sorted(chunk.SANGER_MODEL_ID.astype(str).unique())[: config.max_models]
                chunk = chunk.loc[chunk.SANGER_MODEL_ID.astype(str).isin(allowed)]
            chunk["response_value"] = pd.to_numeric(chunk.AUC, errors="coerce")
            chunk = chunk.loc[np.isfinite(chunk.response_value)].copy()
            chunk["dataset_id"] = chunk.DATASET.astype(str) + "_AUC"
            chunk["cell_line_id"] = chunk.SANGER_MODEL_ID.astype(str)
            chunk["drug_name"] = chunk.DRUG_NAME.astype(str).str.strip()
            chunk["drug_id"] = chunk.drug_name.map(canonical_drug_id)
            chunk["source_drug_id"] = chunk.DRUG_ID.astype(str)
            chunk["higher_is_sensitive"] = False
            sink.write(
                chunk[[
                    "dataset_id", "cell_line_id", "drug_id", "response_value",
                    "higher_is_sensitive", "source_drug_id", "drug_name",
                ]]
            )
            source_maps.append(
                chunk[["dataset_id", "source_drug_id", "drug_id", "drug_name"]].drop_duplicates()
            )
            for row in chunk[["dataset_id", "cell_line_id", "TCGA_DESC"]].drop_duplicates().itertuples(index=False):
                map_rows.append(
                    {"dataset_id": str(row.dataset_id), "cell_line_id": str(row.cell_line_id), "tcga_desc": str(row.TCGA_DESC)}
                )
            seen_models.update(chunk.cell_line_id.astype(str))
        audits.append({"source": str(source), "response_metric": "AUC", "lower_is_sensitive": "true"})
    sink.close()
    return map_rows, pd.concat(source_maps, ignore_index=True).drop_duplicates(), [pd.DataFrame(audits)]


def _stage_prism(
    matrix_path: Path,
    compound_path: Path,
    response_root: Path,
    config: DrugStagingConfig,
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    compound = pd.read_csv(compound_path, low_memory=False)
    id_col = next((column for column in ("IDs", "broad_id", "Broad_ID") if column in compound), None)
    name_col = next((column for column in ("Drug.Name", "name", "drug_name") if column in compound), None)
    if id_col is None or name_col is None:
        raise DrugStagingError("PRISM compound list lacks source ID/name")
    mapping = compound[[id_col, name_col]].dropna().copy()
    mapping.columns = ["source_drug_id", "drug_name"]
    mapping["source_drug_id"] = mapping.source_drug_id.astype(str).str.strip()
    mapping["drug_name"] = mapping.drug_name.astype(str).str.strip()
    mapping = mapping.loc[mapping.drug_name.ne("")].drop_duplicates("source_drug_id")
    name_map = dict(zip(mapping.source_drug_id, mapping.drug_name))
    header = pd.read_csv(matrix_path, nrows=0)
    source_col = str(header.columns[0])
    model_columns = [str(column) for column in header.columns[1:] if str(column).startswith("ACH-")]
    if config.max_models is not None:
        model_columns = model_columns[: config.max_models]
    usecols = [source_col, *model_columns]
    sink = _ParquetSink(response_root / "prism_24q2.parquet")
    source_frames: list[pd.DataFrame] = []
    rows_seen = 0
    rows_mapped = 0
    for chunk in pd.read_csv(matrix_path, usecols=usecols, chunksize=config.prism_chunk_drugs):
        if config.max_prism_drugs is not None:
            remaining = config.max_prism_drugs - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining].copy()
        rows_seen += len(chunk)
        chunk = chunk.rename(columns={source_col: "source_drug_id"})
        chunk["source_drug_id"] = chunk.source_drug_id.astype(str).str.strip()
        chunk["drug_name"] = chunk.source_drug_id.map(name_map)
        chunk = chunk.loc[chunk.drug_name.notna()].copy()
        rows_mapped += len(chunk)
        if chunk.empty:
            continue
        long = chunk.melt(
            id_vars=["source_drug_id", "drug_name"], var_name="cell_line_id", value_name="response_value"
        )
        long["response_value"] = pd.to_numeric(long.response_value, errors="coerce")
        long = long.loc[np.isfinite(long.response_value)].copy()
        long["dataset_id"] = "PRISM_24Q2"
        long["drug_id"] = long.drug_name.map(canonical_drug_id)
        long["higher_is_sensitive"] = False
        sink.write(
            long[[
                "dataset_id", "cell_line_id", "drug_id", "response_value",
                "higher_is_sensitive", "source_drug_id", "drug_name",
            ]]
        )
        source_frames.append(
            long[["dataset_id", "source_drug_id", "drug_id", "drug_name"]].drop_duplicates()
        )
    sink.close()
    source_map = pd.concat(source_frames, ignore_index=True).drop_duplicates()
    audit = {
        "matrix_drug_rows_seen": rows_seen,
        "matrix_drug_rows_mapped_by_native_compound_list": rows_mapped,
        "response_rows": sink.rows,
        "model_columns": len(model_columns),
        "lower_logfold_change_is_sensitive": True,
    }
    return model_columns, source_map, audit


def _stage_drug_targets(drugcentral_path: Path, hgnc_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    hgnc = pd.read_csv(
        hgnc_path, sep="\t", dtype=str, usecols=["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id"],
        low_memory=False,
    )
    hgnc = hgnc.loc[hgnc.ensembl_gene_id.notna()].copy()
    symbol_rows: list[tuple[str, str, str]] = []
    for row in hgnc.itertuples(index=False):
        gene_id = _canonical_gene(row.ensembl_gene_id)
        symbol_rows.append((str(row.symbol).upper(), gene_id, "approved_symbol"))
        for field, method in ((row.alias_symbol, "alias_symbol"), (row.prev_symbol, "previous_symbol")):
            if pd.isna(field):
                continue
            for symbol in str(field).split("|"):
                if symbol.strip():
                    symbol_rows.append((symbol.strip().upper(), gene_id, method))
    symbol_frame = pd.DataFrame(symbol_rows, columns=["target_gene_symbol", "gene_id", "mapping_method"])
    conflicts = symbol_frame.groupby("target_gene_symbol", observed=True).gene_id.nunique()
    allowed = set(conflicts[conflicts.eq(1)].index)
    symbol_frame = symbol_frame.loc[symbol_frame.target_gene_symbol.isin(allowed)].drop_duplicates("target_gene_symbol")
    targets = pd.read_csv(drugcentral_path, sep="\t", compression="infer", dtype=str, low_memory=False)
    required = {"DRUG_NAME", "GENE", "ORGANISM"}
    if not required.issubset(targets):
        raise DrugStagingError(f"DrugCentral lacks fields: {sorted(required - set(targets))}")
    human = targets.loc[
        targets.ORGANISM.astype(str).str.casefold().eq("homo sapiens")
        & targets.DRUG_NAME.notna() & targets.GENE.notna()
    ].copy()
    human["target_gene_symbol"] = human.GENE.astype(str).str.strip().str.upper()
    human["drug_name"] = human.DRUG_NAME.astype(str).str.strip()
    human["drug_id"] = human.drug_name.map(canonical_drug_id)
    mapped = human.merge(symbol_frame, on="target_gene_symbol", how="inner", validate="many_to_one")
    result = mapped[["drug_id", "gene_id", "drug_name", "target_gene_symbol", "mapping_method"]].drop_duplicates()
    result["source_database"] = "DrugCentral_native"
    _atomic_parquet(result, output_path)
    return result, {
        "native_human_target_rows": int(len(human)),
        "mapped_target_rows": int(len(result)),
        "mapped_drugs": int(result.drug_id.nunique()),
        "mapped_genes": int(result.gene_id.nunique()),
        "symbol_mapping_rate": float(len(mapped) / max(len(human), 1)),
    }


def _build_cell_line_map(
    models: pd.DataFrame,
    gdsc_rows: Sequence[Mapping[str, str]],
    prism_models: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inverse = _tissue_to_cancers()
    by_sidm = models.drop_duplicates("canonical_model_id").set_index("canonical_model_id")
    by_ach = models.loc[models.depmap_id.ne("")].drop_duplicates("depmap_id").set_index("depmap_id")
    output: list[dict[str, str]] = []
    for row in gdsc_rows:
        sidm = str(row["cell_line_id"])
        tissue = str(by_sidm.loc[sidm, "tissue"]) if sidm in by_sidm.index else ""
        cancers = set(inverse.get(tissue, ()))
        tcga = str(row.get("tcga_desc", "")).upper()
        if tcga in TCGA_CANCERS:
            cancers.add(tcga)
        for cancer in sorted(cancers):
            output.append(
                {
                    "dataset_id": str(row["dataset_id"]), "cell_line_id": sidm,
                    "canonical_model_id": sidm, "cancer_id": cancer, "tissue": tissue,
                    "mapping_method": "CMP_tissue_plus_native_GDSC_TCGA_DESC",
                }
            )
    for ach in prism_models:
        if ach not in by_ach.index:
            continue
        record = by_ach.loc[ach]
        sidm = str(record.canonical_model_id)
        tissue = str(record.tissue)
        for cancer in inverse.get(tissue, ()):
            output.append(
                {
                    "dataset_id": "PRISM_24Q2", "cell_line_id": str(ach),
                    "canonical_model_id": sidm, "cancer_id": cancer, "tissue": tissue,
                    "mapping_method": "CMP_ACH_to_SIDM_and_tissue",
                }
            )
    frame = pd.DataFrame(output).drop_duplicates(
        ["dataset_id", "cell_line_id", "canonical_model_id", "cancer_id"]
    )
    if frame.empty:
        raise DrugStagingError("No GDSC/PRISM cell line could be mapped to a TCGA cancer")
    coverage = pd.DataFrame({"cancer_id": TCGA_CANCERS}).merge(
        frame.groupby("cancer_id", observed=True).agg(
            mapped_models=("canonical_model_id", "nunique"),
            mapped_datasets=("dataset_id", "nunique"),
        ).reset_index(), on="cancer_id", how="left",
    )
    coverage[["mapped_models", "mapped_datasets"]] = coverage[["mapped_models", "mapped_datasets"]].fillna(0).astype(int)
    coverage["cell_line_assay_available"] = coverage.mapped_models.gt(0)
    coverage["unavailable_reason"] = np.where(
        coverage.cell_line_assay_available, pd.NA, "NO_NATIVE_GDSC_PRISM_MODEL_MAPPED_TO_CANCER"
    )
    return frame.sort_values(["cancer_id", "canonical_model_id", "dataset_id"]), coverage


def _parse_expression_values(raw: np.ndarray) -> np.ndarray:
    flat = raw.astype(str, copy=False).ravel()
    parsed = np.full(len(flat), np.nan, dtype=np.float32)
    simple = np.char.find(flat, " ") < 0
    nonempty = simple & (flat != "") & (flat != "nan") & (flat != "NA")
    parsed[nonempty] = pd.to_numeric(flat[nonempty], errors="coerce").astype(np.float32)
    for index in np.flatnonzero(~simple):
        values = np.fromstring(flat[index], sep=" ", dtype=np.float32)
        parsed[index] = values.mean() if len(values) else np.nan
    return parsed.reshape(raw.shape)


def _stage_expression(
    expression_path: Path,
    exact_candidates_path: Path,
    canonical_models: Iterable[str],
    output_path: Path,
    config: DrugStagingConfig,
) -> dict[str, Any]:
    exact = pd.read_parquet(exact_candidates_path, columns=["lncrna_id"])
    wanted = sorted({_canonical_lncrna(value).removeprefix("LNC:") for value in exact.lncrna_id})
    if config.max_expression_lncrnas is not None:
        wanted = wanted[: config.max_expression_lncrnas]
    wanted_set = set(wanted)
    header = pd.read_csv(expression_path, nrows=3, index_col=0)
    available_models = set(map(str, canonical_models)) & set(map(str, header.columns))
    model_columns = sorted(available_models)
    if config.max_models is not None:
        model_columns = model_columns[: config.max_models]
    if not model_columns:
        raise DrugStagingError("CMP expression has no model shared with staged drug response")
    usecols = ["model_id", "Unnamed: 1", *model_columns]
    sink = _ParquetSink(output_path)
    found_lncrnas: set[str] = set()
    for chunk in pd.read_csv(
        expression_path, skiprows=[1, 2, 3], usecols=usecols, dtype=str,
        na_values=["NA", "NaN", ""], chunksize=config.expression_chunk_genes,
    ):
        ensembl = chunk["Unnamed: 1"].astype(str).str.replace(r"\..*$", "", regex=True)
        hit_mask = ensembl.isin(wanted_set)
        if not hit_mask.any():
            continue
        hit = chunk.loc[hit_mask].copy()
        identifiers = ensembl.loc[hit_mask].to_numpy(str)
        values = _parse_expression_values(hit[model_columns].fillna("").to_numpy(dtype=str))
        row_indices, column_indices = np.where(np.isfinite(values))
        if not len(row_indices):
            continue
        output = pd.DataFrame(
            {
                "canonical_model_id": np.asarray(model_columns, dtype=object)[column_indices],
                "lncrna_id": np.char.add("LNC:", identifiers[row_indices]),
                "expression_value": values[row_indices, column_indices],
            }
        )
        sink.write(output)
        found_lncrnas.update(output.lncrna_id.astype(str))
    sink.close()
    return {
        "expression_rows": sink.rows,
        "candidate_lncrnas_requested": len(wanted),
        "candidate_lncrnas_found": len(found_lncrnas),
        "canonical_models_requested": len(model_columns),
    }


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _available_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    if Path("/proc/meminfo").is_file():
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    raise DrugStagingError("Available system memory could not be measured")


def _staging_resource_preflight(
    output: Path, paths: Mapping[str, Path], config: DrugStagingConfig
) -> dict[str, Any]:
    sizes = {key: _path_size_bytes(path) for key, path in paths.items()}
    source_total = int(sum(sizes.values()))
    usage = shutil.disk_usage(output.parent)
    disk_reserve = max(2 * 1024**3, int(usage.free * 0.15))
    estimated_output = int(source_total * 2.0 + 512 * 1024**2)
    disk_budget = max(0, int(usage.free) - disk_reserve)
    available_ram = _available_memory_bytes()
    nonstreamed = sum(
        sizes[key]
        for key in (
            "prism_compound_native", "cmp_models_native",
            "drugcentral_target_native", "hgnc_native",
        )
    )
    chunk_peak = max(
        int(config.csv_chunk_rows) * 1536,
        int(config.prism_chunk_drugs) * max(1, config.max_models or 2000) * 24,
        int(config.expression_chunk_genes) * max(1, config.max_models or 2000) * 24,
    )
    estimated_ram = int(nonstreamed * 8 + chunk_peak + 512 * 1024**2)
    ram_budget = max(
        0, min(int(available_ram * 0.50), int(available_ram - 2 * 1024**3))
    )
    residual_disk = max(0, disk_budget - estimated_output)
    residual_ram = max(0, ram_budget - estimated_ram)
    duckdb_temp_limit = min(128 * 1024**3, residual_disk)
    duckdb_memory_limit = min(16 * 1024**3, residual_ram)
    status = (
        "PASS"
        if estimated_output <= disk_budget
        and estimated_ram <= ram_budget
        and duckdb_temp_limit >= 1024**3
        and duckdb_memory_limit >= 256 * 1024**2
        else "FAIL"
    )
    return {
        "status": status,
        "guard_timing": "BEFORE_NATIVE_TABLE_READ_AND_BEFORE_STAGING_ARTIFACT_WRITE",
        "formal_fixed_policy_not_cli_overridable": True,
        "source_sizes_bytes": sizes,
        "source_total_bytes": source_total,
        "disk_free_bytes": int(usage.free),
        "disk_reserve_bytes": disk_reserve,
        "disk_budget_bytes": disk_budget,
        "estimated_peak_incremental_disk_bytes": estimated_output,
        "available_ram_bytes": available_ram,
        "ram_reserve_bytes": 2 * 1024**3,
        "ram_budget_bytes": ram_budget,
        "estimated_peak_ram_bytes": estimated_ram,
        "duckdb_temp_directory": str(output / ".duckdb_staging_tmp"),
        "duckdb_max_temp_directory_size_bytes": int(duckdb_temp_limit),
        "duckdb_memory_limit_bytes": int(duckdb_memory_limit),
        "duckdb_threads": int(max(1, min(os.cpu_count() or 1, 8))),
    }


@contextmanager
def _bounded_duckdb(
    output: Path, resource_audit: Mapping[str, Any], *, purpose: str
):
    """Open a resource-capped in-memory DuckDB and always remove its own spill."""

    import duckdb

    if resource_audit.get("status") != "PASS":
        raise DrugStagingError(f"{purpose} cannot start without resource PASS")
    temp_root = output / f".duckdb_{purpose}_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    connection = None
    try:
        connection = duckdb.connect()
        connection.execute(f"SET temp_directory='{_sql_path(temp_root)}'")
        connection.execute(
            "SET max_temp_directory_size='"
            f"{min(int(resource_audit['duckdb_max_temp_directory_size_bytes']), 128 * 1024**3)}B'"
        )
        connection.execute(
            "SET memory_limit='"
            f"{min(int(resource_audit['duckdb_memory_limit_bytes']), 16 * 1024**3)}B'"
        )
        connection.execute(
            f"SET threads={max(1, min(int(resource_audit['duckdb_threads']), 8))}"
        )
        yield connection
    finally:
        if connection is not None:
            connection.close()
        if temp_root.is_dir():
            shutil.rmtree(temp_root)


def _validate_exact_candidate_source_binding(
    *,
    exact_candidates_path: Path,
    exact_release_prediction_path: Path,
    exact_release_lineage_path: Path,
    output: Path,
    resource_audit: Mapping[str, Any],
    input_start_snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind candidate identities to the newly trained V3.2 exact R2 release."""

    try:
        lineage = json.loads(exact_release_lineage_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DrugStagingError(f"Invalid exact release lineage JSON: {error}") from error
    if not isinstance(lineage, Mapping):
        raise DrugStagingError("Exact release lineage must be a JSON object")
    expected = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "exact_pathway",
        "training_status": "SUCCESS",
        "trained_from_scratch": True,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
    }
    mismatches = {
        key: {"expected": value, "observed": lineage.get(key)}
        for key, value in expected.items()
        if lineage.get(key) != value
    }
    if mismatches:
        raise DrugStagingError(
            "Exact release lineage is not a successful newly trained V3.2 exact_pathway: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    prediction_snapshot = input_start_snapshot["exact_release_prediction_proof"]
    actual_prediction_sha = str(prediction_snapshot["sha256"])
    actual_prediction_rows = _parquet_rows(exact_release_prediction_path)
    if lineage.get("prediction_sha256") != actual_prediction_sha:
        raise DrugStagingError("Exact lineage prediction_sha256 does not match proof prediction")
    try:
        lineage_rows = int(lineage.get("prediction_rows"))
    except (TypeError, ValueError) as error:
        raise DrugStagingError("Exact lineage prediction_rows is invalid") from error
    if lineage_rows != actual_prediction_rows:
        raise DrugStagingError("Exact lineage prediction_rows does not match proof prediction")

    candidate_columns = {_normalise_token(value) for value in _header(exact_candidates_path)}
    prediction_columns = {
        _normalise_token(value) for value in _header(exact_release_prediction_path)
    }
    missing_candidate = sorted(set(_EXACT_BINDING_KEYS) - candidate_columns)
    missing_prediction = sorted(set(_EXACT_BINDING_KEYS) - prediction_columns)
    if missing_candidate or missing_prediction:
        raise DrugStagingError(
            "Exact four-key binding columns missing: "
            f"candidate={missing_candidate}; prediction={missing_prediction}"
        )

    candidate_sql = _sql_path(exact_candidates_path)
    prediction_sql = _sql_path(exact_release_prediction_path)
    with _bounded_duckdb(output, resource_audit, purpose="exact_binding") as con:
        con.execute(
            f"""
            CREATE TEMP VIEW staged_exact_keys AS
            SELECT
              CAST(cancer_id AS VARCHAR) AS cancer_id,
              CAST(lncrna_id AS VARCHAR) AS lncrna_id,
              CAST(pathway_id AS VARCHAR) AS pathway_id,
              CAST(pathway_family_id AS VARCHAR) AS pathway_family_id
            FROM read_parquet('{candidate_sql}')
            """
        )
        con.execute(
            f"""
            CREATE TEMP VIEW released_exact_keys AS
            SELECT
              CAST(cancer_id AS VARCHAR) AS cancer_id,
              CAST(lncrna_id AS VARCHAR) AS lncrna_id,
              CAST(pathway_id AS VARCHAR) AS pathway_id,
              CAST(pathway_family_id AS VARCHAR) AS pathway_family_id
            FROM read_parquet('{prediction_sql}')
            """
        )
        candidate_rows = int(con.execute("SELECT count(*) FROM staged_exact_keys").fetchone()[0])
        prediction_rows = int(con.execute("SELECT count(*) FROM released_exact_keys").fetchone()[0])
        candidate_unique = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM staged_exact_keys) q"
            ).fetchone()[0]
        )
        prediction_unique = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM released_exact_keys) q"
            ).fetchone()[0]
        )
        candidate_null_keys = int(
            con.execute(
                "SELECT count(*) FROM staged_exact_keys WHERE "
                + " OR ".join(f"{key} IS NULL OR {key} = ''" for key in _EXACT_BINDING_KEYS)
            ).fetchone()[0]
        )
        prediction_null_keys = int(
            con.execute(
                "SELECT count(*) FROM released_exact_keys WHERE "
                + " OR ".join(f"{key} IS NULL OR {key} = ''" for key in _EXACT_BINDING_KEYS)
            ).fetchone()[0]
        )
        candidate_minus_prediction = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM staged_exact_keys "
                "EXCEPT SELECT DISTINCT * FROM released_exact_keys) q"
            ).fetchone()[0]
        )
        prediction_minus_candidate = int(
            con.execute(
                "SELECT count(*) FROM (SELECT DISTINCT * FROM released_exact_keys "
                "EXCEPT SELECT DISTINCT * FROM staged_exact_keys) q"
            ).fetchone()[0]
        )
    counts = {candidate_rows, prediction_rows, candidate_unique, prediction_unique}
    if (
        len(counts) != 1
        or candidate_null_keys != 0
        or prediction_null_keys != 0
        or candidate_minus_prediction != 0
        or prediction_minus_candidate != 0
    ):
        raise DrugStagingError(
            "Exact candidate four-key source binding failed: "
            + json.dumps(
                {
                    "candidate_rows": candidate_rows,
                    "prediction_rows": prediction_rows,
                    "candidate_unique_keys": candidate_unique,
                    "prediction_unique_keys": prediction_unique,
                    "candidate_null_keys": candidate_null_keys,
                    "prediction_null_keys": prediction_null_keys,
                    "candidate_minus_prediction": candidate_minus_prediction,
                    "prediction_minus_candidate": prediction_minus_candidate,
                },
                sort_keys=True,
            )
        )
    return {
        "status": "PASS",
        "binding_policy": "LITERAL_FOUR_KEY_BIDIRECTIONAL_EXCEPT_AND_ROW_UNIQUENESS",
        "key_columns": list(_EXACT_BINDING_KEYS),
        "candidate_path": str(exact_candidates_path),
        "candidate_sha256": input_start_snapshot["v32_exact_candidates"]["sha256"],
        "candidate_rows": candidate_rows,
        "candidate_unique_keys": candidate_unique,
        "release_prediction_path": str(exact_release_prediction_path),
        "release_prediction_sha256": actual_prediction_sha,
        "release_prediction_rows": prediction_rows,
        "release_prediction_unique_keys": prediction_unique,
        "release_lineage_path": str(exact_release_lineage_path),
        "release_lineage_sha256": input_start_snapshot["exact_release_lineage_proof"]["sha256"],
        "lineage_training_run_id": lineage.get("training_run_id"),
        "lineage_newly_trained_v32": True,
        "candidate_null_keys": candidate_null_keys,
        "prediction_null_keys": prediction_null_keys,
        "candidate_minus_prediction": candidate_minus_prediction,
        "prediction_minus_candidate": prediction_minus_candidate,
        "duckdb_resource_limits": {
            "memory_limit_bytes": min(
                int(resource_audit["duckdb_memory_limit_bytes"]), 16 * 1024**3
            ),
            "max_temp_directory_size_bytes": min(
                int(resource_audit["duckdb_max_temp_directory_size_bytes"]),
                128 * 1024**3,
            ),
            "threads": max(1, min(int(resource_audit["duckdb_threads"]), 8)),
            "temporary_directory_removed": True,
        },
    }


def _stage_candidate_universe_with_connection(
    exact_path: Path,
    membership_path: Path,
    target_path: Path,
    native_assayed_drug_ids: Iterable[str],
    output_root: Path,
    *,
    storage_mode: str,
    native_target_mapping_coverage: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
    con: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if storage_mode not in {"factored", "materialized"}:
        raise ValueError(storage_mode)
    output_root.mkdir(parents=True, exist_ok=True)
    assayed = pd.DataFrame(
        {"drug_id": sorted({str(value) for value in native_assayed_drug_ids if str(value)})}
    )
    if assayed.empty:
        raise DrugStagingError("No native GDSC/PRISM drug identity was staged")
    if resource_audit.get("status") != "PASS":
        raise DrugStagingError("Candidate DISTINCT cannot start without resource PASS")
    con.register("native_assayed_drugs", assayed)
    exact_sql = _sql_path(exact_path)
    member_sql = _sql_path(membership_path)
    target_sql = _sql_path(target_path)
    con.execute(
        f"""
        CREATE TEMP VIEW pathway_drug AS
        SELECT DISTINCT m.pathway_id, t.drug_id
        FROM read_parquet('{member_sql}') m
        JOIN read_parquet('{target_sql}') t
          ON t.gene_id = CASE
            WHEN starts_with(m.gene_id, 'GENE:') THEN regexp_replace(m.gene_id, '\\.[0-9]+$', '')
            ELSE 'GENE:' || regexp_replace(m.gene_id, '\\.[0-9]+$', '')
          END
        JOIN native_assayed_drugs a ON a.drug_id = t.drug_id
        """
    )
    edge_path = output_root / "_factors" / "pathway_drug_native_assayed.parquet"
    edge_path.parent.mkdir(parents=True, exist_ok=True)
    edge_temporary = edge_path.with_name(f".{edge_path.stem}.tmp{edge_path.suffix}")
    edge_temporary.unlink(missing_ok=True)
    con.execute(
        f"""
        COPY (
          SELECT pathway_id, drug_id FROM pathway_drug
          ORDER BY pathway_id, drug_id
        ) TO '{_sql_path(edge_temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    os.replace(edge_temporary, edge_path)
    edge_rows = int(con.execute("SELECT count(*) FROM pathway_drug").fetchone()[0])
    rows: list[dict[str, Any]] = []
    for cancer in TCGA_CANCERS:
        relation = f"""
          SELECT DISTINCT c.cancer_id, c.lncrna_id, p.drug_id
          FROM read_parquet('{exact_sql}') c
          JOIN pathway_drug p USING(pathway_id)
          WHERE c.cancer_id = '{cancer}'
        """
        count, drugs, lncrnas = con.execute(
            f"""
            SELECT count(*), count(DISTINCT drug_id), count(DISTINCT lncrna_id)
            FROM ({relation}) q
            """
        ).fetchone()
        count, drugs, lncrnas = int(count), int(drugs), int(lncrnas)
        if storage_mode == "materialized":
            destination = output_root / f"part_{cancer}.parquet"
            temporary = destination.with_name(
                f".{destination.stem}.tmp{destination.suffix}"
            )
            temporary.unlink(missing_ok=True)
            con.execute(
                f"""
                COPY (
                  SELECT * FROM ({relation}) q
                  ORDER BY drug_id, lncrna_id
                ) TO '{_sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            os.replace(temporary, destination)
        rows.append({"cancer_id": cancer, "candidate_rows": count, "candidate_lncrnas": lncrnas, "candidate_drugs": drugs})
    coverage = pd.DataFrame(rows)
    missing = coverage.loc[coverage.candidate_rows.eq(0), "cancer_id"].tolist()
    if missing:
        raise DrugStagingError(f"Target-derived drug candidate universe lacks cancers: {missing}")
    definition = {
        "candidate_format": FACTORED_CANDIDATE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "storage_mode": storage_mode,
        "target_keys": ["cancer_id", "lncrna_id", "drug_id"],
        "conceptual_candidate_rows": int(coverage.candidate_rows.sum()),
        "candidate_cancers": int(coverage.cancer_id.nunique()),
        "pathway_drug_edge_path": str(edge_path),
        "pathway_drug_edge_sha256": artifact_sha256(edge_path),
        "pathway_drug_edge_rows": edge_rows,
        "exact_candidate_path": str(exact_path),
        "exact_candidate_sha256": artifact_sha256(exact_path),
        "relation": (
            "DISTINCT V3.2 exact(cancer_id,lncrna_id,pathway_id) JOIN "
            "native-assayed DrugCentral(pathway_id,drug_id)"
        ),
        "implicit_unavailable_policy": (
            "A conceptual target-derived key absent from the sparse trained public table "
            "is unavailable, never a zero probability. Reasons are resolved from the "
            "fold assay/core coverage audit; no historical prediction supplies a value."
        ),
        "native_assay_identity_and_finite_measurement_presence_only": True,
        "response_magnitude_not_used_for_candidate_eligibility": True,
        "native_assayed_to_target_mapping_coverage": dict(
            native_target_mapping_coverage
        ),
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
    }
    _atomic_json(output_root / "FACTORED_UNIVERSE.json", definition)
    return coverage, {
        "candidate_rows": int(coverage.candidate_rows.sum()),
        "cancers": int(coverage.cancer_id.nunique()),
        "pathway_drug_edge_rows": edge_rows,
        "storage_mode": storage_mode,
        "materialized_candidate_rows": (
            int(coverage.candidate_rows.sum()) if storage_mode == "materialized" else 0
        ),
        "derivation": "V3.2 exact candidates JOIN exact pathway-gene membership JOIN native-assayed DrugCentral human targets",
        "outcome_or_historical_prediction_used": False,
        "implicit_unavailable_policy": definition["implicit_unavailable_policy"],
        "native_assayed_to_target_mapping_coverage": dict(
            native_target_mapping_coverage
        ),
    }


def _stage_candidate_universe(
    exact_path: Path,
    membership_path: Path,
    target_path: Path,
    native_assayed_drug_ids: Iterable[str],
    output_root: Path,
    *,
    storage_mode: str,
    native_target_mapping_coverage: Mapping[str, Any],
    resource_audit: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stage candidate factors while closing DuckDB/spill even on failure."""

    with _bounded_duckdb(
        output_root.parent, resource_audit, purpose="candidate_staging"
    ) as con:
        return _stage_candidate_universe_with_connection(
            exact_path,
            membership_path,
            target_path,
            native_assayed_drug_ids,
            output_root,
            storage_mode=storage_mode,
            native_target_mapping_coverage=native_target_mapping_coverage,
            resource_audit=resource_audit,
            con=con,
        )
def _native_target_mapping_coverage(
    source_drug_map: pd.DataFrame, targets: pd.DataFrame
) -> dict[str, Any]:
    native = set(source_drug_map.drug_id.astype(str))
    targeted = set(targets.drug_id.astype(str))
    matched = native & targeted
    unmatched = native - targeted
    fraction = float(len(matched) / max(len(native), 1))
    threshold_pass = bool(
        len(matched) >= FORMAL_MIN_NATIVE_TARGET_MAPPED_DRUGS
        and fraction >= FORMAL_MIN_NATIVE_TARGET_MAPPING_FRACTION
    )
    return {
        "identity_scope": NATIVE_TARGET_IDENTITY_SCOPE,
        "synonym_crosswalk_applied": False,
        "native_assayed_unique_drugs": int(len(native)),
        "native_assayed_with_target_mapping": int(len(matched)),
        "native_assayed_without_target_mapping_silent_nonmatch": int(len(unmatched)),
        "mapping_fraction": fraction,
        "minimum_mapped_drugs": FORMAL_MIN_NATIVE_TARGET_MAPPED_DRUGS,
        "minimum_mapping_fraction": FORMAL_MIN_NATIVE_TARGET_MAPPING_FRACTION,
        "threshold_pass": threshold_pass,
        "silent_nonmatch_examples": sorted(unmatched)[:20],
        "policy": (
            "Only exact canonical-name identities enter target-derived candidates; "
            "silent nonmatches are counted and formal training fails closed below threshold."
        ),
    }


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    if path.is_file():
        return int(pq.ParquetFile(path).metadata.num_rows)
    return int(sum(pq.ParquetFile(item).metadata.num_rows for item in path.rglob("*.parquet")))


def _run_native_drug_staging_impl(
    *,
    gdsc1_path: str | Path,
    gdsc2_path: str | Path,
    prism_matrix_path: str | Path,
    prism_compound_path: str | Path,
    cmp_expression_path: str | Path,
    cmp_models_json_path: str | Path,
    drugcentral_target_path: str | Path,
    hgnc_path: str | Path,
    exact_candidates_path: str | Path,
    pathway_membership_path: str | Path,
    exact_release_prediction_path: str | Path | None = None,
    exact_release_lineage_path: str | Path | None = None,
    output_root: str | Path,
    staging_run_id: str,
    config: DrugStagingConfig | None = None,
    execution_code_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    settings = config or DrugStagingConfig()
    settings.validate()
    if not re.fullmatch(r"v32-[a-z0-9][a-z0-9._-]*", str(staging_run_id)):
        raise DrugStagingError("staging_run_id must be lowercase and start with 'v32-'")
    paths = {
        "gdsc1_native": Path(gdsc1_path).resolve(),
        "gdsc2_native": Path(gdsc2_path).resolve(),
        "prism_matrix_native": Path(prism_matrix_path).resolve(),
        "prism_compound_native": Path(prism_compound_path).resolve(),
        "cmp_expression_native": Path(cmp_expression_path).resolve(),
        "cmp_models_native": Path(cmp_models_json_path).resolve(),
        "drugcentral_target_native": Path(drugcentral_target_path).resolve(),
        "hgnc_native": Path(hgnc_path).resolve(),
        "v32_exact_candidates": Path(exact_candidates_path).resolve(),
        "exact_pathway_membership": Path(pathway_membership_path).resolve(),
    }
    proof_supplied = (
        exact_release_prediction_path is not None,
        exact_release_lineage_path is not None,
    )
    if any(proof_supplied) and not all(proof_supplied):
        raise DrugStagingError(
            "exact_release_prediction_path and exact_release_lineage_path must be supplied together"
        )
    if settings.formal and not all(proof_supplied):
        raise DrugStagingError(
            "Formal Drug staging requires exact release prediction and lineage proofs"
        )
    if all(proof_supplied):
        paths["exact_release_prediction_proof"] = Path(
            exact_release_prediction_path  # type: ignore[arg-type]
        ).resolve()
        paths["exact_release_lineage_proof"] = Path(
            exact_release_lineage_path  # type: ignore[arg-type]
        ).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_start_snapshot = _snapshot_artifacts(paths)
    code_paths: dict[str, Path] = {
        "drug_staging_module": Path(__file__).resolve(),
    }
    for index, code_path in enumerate(execution_code_paths or ()):
        resolved = Path(code_path).resolve()
        if resolved not in code_paths.values():
            code_paths[f"execution_entrypoint_{index}"] = resolved
    code_start_snapshot = _snapshot_artifacts(code_paths)
    runtime_fingerprint = _runtime_fingerprint()
    runtime_fingerprint_sha256 = _canonical_sha(runtime_fingerprint)
    resource_audit = _staging_resource_preflight(output, paths, settings)
    resource_audit_path = output / "STAGING_RESOURCE_AUDIT.json"
    _atomic_json(resource_audit_path, resource_audit)
    if resource_audit["status"] != "PASS":
        _atomic_json(
            output / "RUN_FAILED.json",
            {
                "status": "FAILED_RESOURCE_GUARD",
                "analysis_version": ANALYSIS_VERSION,
                "staging_run_id": staging_run_id,
                "release_ready": False,
                "partial_not_publishable": True,
                "staging_resource_audit_path": str(resource_audit_path),
                "staging_resource_audit_sha256": artifact_sha256(resource_audit_path),
            },
        )
        raise DrugStagingError("Native Drug staging resource guard failed")
    source_kinds = {
        key: ("current_v32_nonpredictive" if key == "v32_exact_candidates" else "static_annotation" if key in {"hgnc_native", "exact_pathway_membership"} else "native_raw")
        for key in paths
        if key not in {"exact_release_prediction_proof", "exact_release_lineage_proof"}
    }
    audit = audit_native_inputs(
        [(key, paths[key], source_kinds[key]) for key in source_kinds],
        precomputed_snapshot=input_start_snapshot,
    )
    exact_columns = {_normalise_token(column) for column in _header(paths["v32_exact_candidates"])}
    bad = sorted(exact_columns & _FORBIDDEN_EXACT_COLUMNS)
    if bad:
        raise DrugStagingError(f"V3.2 exact candidates contain result columns: {bad}")
    required_exact = {"cancer_id", "lncrna_id", "pathway_id"}
    if not required_exact.issubset(exact_columns):
        raise DrugStagingError(f"Exact candidates lack {sorted(required_exact - exact_columns)}")

    exact_binding: dict[str, Any]
    if all(proof_supplied):
        exact_binding = _validate_exact_candidate_source_binding(
            exact_candidates_path=paths["v32_exact_candidates"],
            exact_release_prediction_path=paths["exact_release_prediction_proof"],
            exact_release_lineage_path=paths["exact_release_lineage_proof"],
            output=output,
            resource_audit=resource_audit,
            input_start_snapshot=input_start_snapshot,
        )
    else:
        exact_binding = {
            "status": "NOT_REQUESTED_NONFORMAL",
            "formal_training_binding_pass": False,
        }

    _atomic_json(output / "NATIVE_INPUT_AUDIT.json", audit)
    response_root = output / "raw_cell_line_drug_response"
    response_root.mkdir()
    models = load_cmp_models(paths["cmp_models_native"])
    if settings.max_models is not None:
        keep = sorted(models.canonical_model_id.unique())[: settings.max_models]
        models = models.loc[models.canonical_model_id.isin(keep)].copy()

    gdsc_map_rows, gdsc_source_map, gdsc_audit_parts = _stage_gdsc(
        [paths["gdsc1_native"], paths["gdsc2_native"]], response_root, settings
    )
    prism_models, prism_source_map, prism_audit = _stage_prism(
        paths["prism_matrix_native"], paths["prism_compound_native"], response_root, settings
    )
    source_drug_map = pd.concat([gdsc_source_map, prism_source_map], ignore_index=True).drop_duplicates()
    _atomic_parquet(source_drug_map, output / "source_drug_map.parquet")

    target, target_audit = _stage_drug_targets(
        paths["drugcentral_target_native"], paths["hgnc_native"], output / "drug_gene_target.static.parquet"
    )
    native_target_coverage = _native_target_mapping_coverage(source_drug_map, target)
    target_drug_ids = set(target.drug_id.astype(str))
    unmapped_native = source_drug_map.loc[
        ~source_drug_map.drug_id.astype(str).isin(target_drug_ids),
        ["dataset_id", "source_drug_id", "drug_id", "drug_name"],
    ].drop_duplicates().sort_values(
        ["drug_id", "dataset_id", "source_drug_id"], kind="stable"
    )
    unmapped_native_path = output / "native_assayed_without_target_mapping.parquet"
    _atomic_parquet(unmapped_native, unmapped_native_path)
    native_target_coverage.update(
        {
            "candidate_scope": "TARGET_ANNOTATED_NATIVE_ASSAYED_DRUGS",
            "silent_nonmatch_sidecar_path": str(unmapped_native_path),
            "silent_nonmatch_sidecar_sha256": artifact_sha256(unmapped_native_path),
            "silent_nonmatch_sidecar_rows": int(len(unmapped_native)),
        }
    )
    cell_map, cell_coverage = _build_cell_line_map(models, gdsc_map_rows, prism_models)
    _atomic_parquet(cell_map, output / "cell_line_map.parquet")
    cell_coverage.to_csv(output / "CELL_LINE_CANCER_COVERAGE.tsv", sep="\t", index=False)
    canonical_models = set(cell_map.canonical_model_id)
    expression_audit = _stage_expression(
        paths["cmp_expression_native"], paths["v32_exact_candidates"], canonical_models,
        output / "raw_cell_line_lncrna_expression.parquet", settings,
    )
    candidate_coverage, candidate_audit = _stage_candidate_universe(
        paths["v32_exact_candidates"], paths["exact_pathway_membership"],
        output / "drug_gene_target.static.parquet", source_drug_map.drug_id.unique(),
        output / "drug_candidate_universe",
        storage_mode=settings.resolved_candidate_storage_mode,
        native_target_mapping_coverage=native_target_coverage,
        resource_audit=resource_audit,
    )
    coverage = candidate_coverage.merge(cell_coverage, on="cancer_id", how="left", validate="one_to_one")
    coverage.to_csv(output / "CANDIDATE_AND_ASSAY_COVERAGE.tsv", sep="\t", index=False)

    artifact_paths = {
        "raw_cell_line_drug_response": response_root,
        "raw_cell_line_lncrna_expression": output / "raw_cell_line_lncrna_expression.parquet",
        "cell_line_map": output / "cell_line_map.parquet",
        "drug_gene_target": output / "drug_gene_target.static.parquet",
        "drug_candidate_universe": output / "drug_candidate_universe",
        "source_drug_map": output / "source_drug_map.parquet",
        "native_assayed_without_target_mapping": unmapped_native_path,
    }
    artifacts = {
        key: {
            "path": str(path), "sha256": artifact_sha256(path), "rows": _parquet_rows(path),
        }
        for key, path in artifact_paths.items()
    }
    artifacts["drug_candidate_universe"]["logical_rows"] = candidate_audit["candidate_rows"]
    artifacts["drug_candidate_universe"]["storage_mode"] = candidate_audit["storage_mode"]
    artifacts["drug_candidate_universe"]["rows"] = candidate_audit["materialized_candidate_rows"]
    artifacts["drug_candidate_universe"]["factor_edge_rows"] = candidate_audit["pathway_drug_edge_rows"]
    input_artifact_snapshot = _verify_artifact_snapshot(
        input_start_snapshot, label="Input/static/proof artifacts"
    )
    execution_code_snapshot = _verify_artifact_snapshot(
        code_start_snapshot, label="Execution code artifacts"
    )
    provenance_snapshot = {
        "status": "PASS",
        "input_artifacts": input_artifact_snapshot,
        "execution_code": execution_code_snapshot,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "exact_candidate_source_binding": exact_binding,
    }
    provenance_snapshot_path = output / "STAGING_PROVENANCE_SNAPSHOT.json"
    _atomic_json(provenance_snapshot_path, provenance_snapshot)
    manifest = {
        "staging_format": STAGING_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "staging_run_id": staging_run_id,
        "status": "SUCCESS",
        "formal": settings.formal,
        "training_ready": bool(
            settings.formal
            and candidate_audit["cancers"] == 33
            and len(canonical_models) >= 5
            and native_target_coverage["threshold_pass"]
            and exact_binding.get("status") == "PASS"
            and input_artifact_snapshot["unchanged"] is True
            and execution_code_snapshot["unchanged"] is True
        ),
        "config": asdict(settings),
        "native_input_audit_sha256": artifact_sha256(output / "NATIVE_INPUT_AUDIT.json"),
        "staging_resource_audit_path": str(resource_audit_path),
        "staging_resource_audit_sha256": artifact_sha256(resource_audit_path),
        "staging_resource_audit": resource_audit,
        "staging_provenance_snapshot_path": str(provenance_snapshot_path),
        "staging_provenance_snapshot_sha256": artifact_sha256(provenance_snapshot_path),
        "input_artifact_snapshot": input_artifact_snapshot,
        "input_artifacts_unchanged": True,
        "execution_code_snapshot": execution_code_snapshot,
        "execution_code_unchanged": True,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "exact_candidate_source_binding": exact_binding,
        "native_inputs": audit["inputs"],
        "artifacts": artifacts,
        "gdsc": gdsc_audit_parts[0].to_dict(orient="records"),
        "prism": prism_audit,
        "expression": expression_audit,
        "drugcentral": target_audit,
        "native_assayed_to_target_mapping_coverage": native_target_coverage,
        "candidate_universe": candidate_audit,
        "candidate_cancers": sorted(candidate_coverage.cancer_id.tolist()),
        "cancers_without_native_cell_line_assay": sorted(
            cell_coverage.loc[~cell_coverage.cell_line_assay_available, "cancer_id"].tolist()
        ),
        "missing_assay_policy": (
            "SPARSE_AVAILABLE_ROWS_PLUS_EXPLICIT_IMPLICIT_UNAVAILABLE_POLICY"
            if candidate_audit["storage_mode"] == "factored"
            else "KEEP_TARGET_DERIVED_CANDIDATES_AND_EMIT_NULL_WITH_REASON"
        ),
        "old_association_tables_read": False,
        "old_predictions_used": False,
        "old_checkpoints_used": False,
        "canonical_drug_id_policy": "DRUG:sha1(uppercase_alphanumeric_drug_name)[:16]",
        "cell_line_fold_group": "canonical_SIDM_shared_across_GDSC_PRISM",
        "canonical_models_available_for_folding": len(canonical_models),
        "minimum_canonical_models_for_five_folds": 5,
    }
    manifest_path = output / "STAGING_MANIFEST.json"
    _atomic_json(manifest_path, manifest)
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "staging_run_id": staging_run_id,
        "formal": settings.formal,
        "training_ready": manifest["training_ready"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": artifact_sha256(manifest_path),
        "candidate_rows": candidate_audit["candidate_rows"],
        "candidate_cancers": candidate_audit["cancers"],
        "response_rows": artifacts["raw_cell_line_drug_response"]["rows"],
        "expression_rows": artifacts["raw_cell_line_lncrna_expression"]["rows"],
        "native_assayed_to_target_mapping_coverage": native_target_coverage,
        "staging_resource_audit_path": str(resource_audit_path),
        "staging_resource_audit_sha256": artifact_sha256(resource_audit_path),
        "staging_provenance_snapshot_path": str(provenance_snapshot_path),
        "staging_provenance_snapshot_sha256": artifact_sha256(provenance_snapshot_path),
        "input_artifact_snapshot": input_artifact_snapshot,
        "input_artifacts_unchanged": True,
        "execution_code_snapshot": execution_code_snapshot,
        "execution_code_unchanged": True,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "exact_candidate_source_binding": exact_binding,
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


def run_native_drug_staging(
    *,
    gdsc1_path: str | Path,
    gdsc2_path: str | Path,
    prism_matrix_path: str | Path,
    prism_compound_path: str | Path,
    cmp_expression_path: str | Path,
    cmp_models_json_path: str | Path,
    drugcentral_target_path: str | Path,
    hgnc_path: str | Path,
    exact_candidates_path: str | Path,
    pathway_membership_path: str | Path,
    exact_release_prediction_path: str | Path | None = None,
    exact_release_lineage_path: str | Path | None = None,
    output_root: str | Path,
    staging_run_id: str,
    config: DrugStagingConfig | None = None,
    execution_code_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Fail closed around all native-staging failures and partial artifacts."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise DrugStagingError(f"Refusing to overwrite non-empty staging directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    in_progress = output / "RUN_IN_PROGRESS.json"
    _atomic_json(
        in_progress,
        {
            "status": "RUN_IN_PROGRESS",
            "analysis_version": ANALYSIS_VERSION,
            "staging_run_id": str(staging_run_id),
            "release_ready": False,
            "partial_not_publishable": True,
        },
    )
    try:
        result = _run_native_drug_staging_impl(
            gdsc1_path=gdsc1_path,
            gdsc2_path=gdsc2_path,
            prism_matrix_path=prism_matrix_path,
            prism_compound_path=prism_compound_path,
            cmp_expression_path=cmp_expression_path,
            cmp_models_json_path=cmp_models_json_path,
            drugcentral_target_path=drugcentral_target_path,
            hgnc_path=hgnc_path,
            exact_candidates_path=exact_candidates_path,
            pathway_membership_path=pathway_membership_path,
            exact_release_prediction_path=exact_release_prediction_path,
            exact_release_lineage_path=exact_release_lineage_path,
            output_root=output,
            staging_run_id=staging_run_id,
            config=config,
            execution_code_paths=execution_code_paths,
        )
    except Exception as error:
        (output / "SUCCESS.json").unlink(missing_ok=True)
        _atomic_json(
            output / "RUN_FAILED.json",
            {
                "status": "FAILED",
                "analysis_version": ANALYSIS_VERSION,
                "staging_run_id": str(staging_run_id),
                "error_type": type(error).__name__,
                "error": str(error),
                "release_ready": False,
                "partial_not_publishable": True,
                "success_marker_present": False,
            },
        )
        in_progress.unlink(missing_ok=True)
        raise
    in_progress.unlink(missing_ok=True)
    return result


__all__ = [
    "ANALYSIS_VERSION",
    "DrugStagingConfig",
    "DrugStagingError",
    "FACTORED_CANDIDATE_FORMAT",
    "STAGING_FORMAT",
    "TCGA_CANCERS",
    "canonical_drug_id",
    "load_cmp_models",
    "normalise_drug_name",
    "run_native_drug_staging",
]
