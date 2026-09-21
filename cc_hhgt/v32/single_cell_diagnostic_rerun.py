"""Fail-closed V3.2 wrapper for the recovered single-cell diagnostic pipeline.

The recovered July-2026 pipeline is useful, but it was written against a
mutable ``results/sc_trajectory_staging`` tree and allowed processed Seurat
fallbacks.  This module turns it into a hash-bound, isolated diagnostic run:

* raw 10x H5 plus donor/cell metadata are the only expression inputs;
* a new output root is mandatory and historical result reuse is impossible;
* the inferred CytoTRACE2/UCell root is never described as an explicit root;
* pathway outputs are post-filtered to the pinned V3.2 exact-2,135 universe;
* all outputs remain diagnostic evidence with zero model-fusion weight.

The module deliberately does not start computation during preflight.  A
separate shell wrapper requires a passing preflight and an explicit execute
invocation on the server.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


CONFIG_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_RERUN_CONFIG_V1"
PREFLIGHT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_RERUN_PREFLIGHT_V1"
WORKSPACE_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_WORKSPACE_V1"
FINAL_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_ARTIFACT_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"

EXPECTED_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
FORMAL_CANCERS = (
    "ACC", "CHOL", "DLBC", "ESCA", "GBM", "HNSC", "KIRC", "LAML",
    "LGG", "LUSC", "MESO", "PCPG", "READ", "SARC", "SKCM", "THYM",
    "UCEC",
)
EXPECTED_EXACT_PATHWAYS = 2_135
EXPECTED_CANDIDATE_ROWS = 3_300_000
EXPECTED_EXACT_MEMBERSHIP_EDGES = 352_205
ROOT_PROVENANCE = "INFERRED_CYTOTRACE2_UCELL_CONSENSUS"

REQUIRED_SOURCE_FILES = {
    "13_sc_malignant_trajectory.R": (
        "55d9a0be39aaf2686e019267ba9902260f109f2de4a1e40413cf9a43bfc023ef"
    ),
    "14_sc_immune_subtype.R": (
        "cc7aef92c266db8e1d5833c3c1709483b5fd02f3832226759da43d110ee5cec1"
    ),
    "15_sc_patient_pseudobulk_ssgsea.R": (
        "88b47e1c50d1c436214a23893a8b1989bdb2c64376b5dcab527db7c1f9afaf34"
    ),
    "61_sc_trajectory_pathway_stats.py": (
        "096932aa7aa6b23f51513a6cf6f3fd100d43ae1d1e14bcacf3f1dbe6c0336502"
    ),
    "63_sc_lnc_pathway_association_ssgsea.py": (
        "aa560886e5cfa3d4a5c1b8bd5430dfb2deadcbaf67c120aba8cb1ee8048c9065"
    ),
    "run_sc_trajectory_one.sh": (
        "017a6084fc70fce89fd21bdaf36e180de4826df0bc0df0a9beb096964ec2f4b5"
    ),
    "sc_trajectory_ssgsea.yaml": (
        "56c9937a59192ed2a1f2c0379619962c1b29ac5e917eab1bdeb82b8d0b19bc56"
    ),
}

# Snapshot identities are explicit so a biologically material R-script change
# cannot pass merely because it was placed under an arbitrary directory name.
# The r4 snapshot differs only by normalizing DoRothEA gene symbols to upper
# case before the exact signed join, matching the already-pinned preflight
# comparison.  It keeps the fail-closed requirement that every one of the
# 13,153 current exact edges has a finite signed weight and audits the 70
# signed-reference-only edges that are excluded.
SOURCE_SNAPSHOT_SHA256_OVERRIDES = {
    "v32_single_cell_pipeline_source_snapshot_20260827_r3_exact_signed_intersection": {},
    "v32_single_cell_pipeline_source_snapshot_20260827_r4_uppercase_signed_contract": {
        "15_sc_patient_pseudobulk_ssgsea.R": (
            "f9dde21919e5b301fa1aa8cd1ec641f4f45186800a6032443e003815519ad268"
        ),
    },
}

REQUIRED_METADATA_COLUMNS = {
    "annotation": {
        "cell_id", "raw_cell_id", "dataset_id", "patient_id", "sample_id",
        "sample_class", "cell_type_major",
    },
    "qc": {"cell_id", "pass_qc"},
    "malignant": {"cell_id", "malignant_status", "primary_analysis_flag"},
}
FORBIDDEN_EXPRESSION_TOKENS = (
    "processed_seurat", "sc_reanalysis", "sc_trajectory",
    "pathway_association", "checkpoint", "prediction", "ranking",
)


class SingleCellDiagnosticError(RuntimeError):
    """Raised when the recovered pipeline cannot prove a fresh V3.2 run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent.parent / path).resolve()


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "yes", "y"}
    )


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("format") != CONFIG_FORMAT:
        raise SingleCellDiagnosticError("Unknown diagnostic rerun config format")
    if config.get("analysis_version") != ANALYSIS_VERSION:
        raise SingleCellDiagnosticError("The rerun must be labelled V3.2")
    if tuple(config.get("coverage_cancers", ())) != EXPECTED_CANCERS:
        raise SingleCellDiagnosticError("coverage_cancers must be the canonical 33")
    if tuple(config.get("formal_cancers", ())) != FORMAL_CANCERS:
        raise SingleCellDiagnosticError("formal_cancers must be the audited 17")
    exact = config.get("exact_pathway_authority", {})
    if int(exact.get("pathway_count", -1)) != EXPECTED_EXACT_PATHWAYS:
        raise SingleCellDiagnosticError("Exact-pathway authority must contain 2,135 IDs")
    if int(exact.get("candidate_rows", -1)) != EXPECTED_CANDIDATE_ROWS:
        raise SingleCellDiagnosticError("Candidate authority must contain 3,300,000 rows")
    if int(exact.get("membership_edges", -1)) != EXPECTED_EXACT_MEMBERSHIP_EDGES:
        raise SingleCellDiagnosticError("Exact membership edge-count contract drift")
    policy = config.get("policy", {})
    required = {
        "raw_h5_required": True,
        "processed_seurat_fallback_allowed": False,
        "historical_result_reuse_allowed": False,
        "root_is_explicit": False,
        "model_fusion_permitted": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "production_deployment": False,
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise SingleCellDiagnosticError(f"Unsafe policy value for {key}")
    if policy.get("root_provenance") != ROOT_PROVENANCE:
        raise SingleCellDiagnosticError("Root provenance must remain inferred")
    output = str(config.get("compute_output_root", "")).replace("\\", "/").lower()
    if "v32_full_multitask" not in output or "single_cell_diagnostic_fresh" not in output:
        raise SingleCellDiagnosticError("Compute output is not an isolated V3.2 diagnostic root")
    if "sc_trajectory_staging" in output:
        raise SingleCellDiagnosticError("Mutable historical staging root is forbidden")
    templates = config.get("input_templates", {})
    links = config.get("workspace_links", {})
    if not str(links.get("processed/pathway_gene_member_weighted.tsv", "")).strip():
        raise SingleCellDiagnosticError(
            "A separate signed pathway reference is required for DoRothEA"
        )
    expected_templates = {
        "raw_h5": str(links.get("processed/sc_tool_input", "")).rstrip("/")
        + "/{cancer}/raw_feature_bc_matrix.h5",
        "annotation": str(links.get("results/sc", "")).rstrip("/")
        + "/{cancer}/sc_cell_annotation.tsv.gz",
        "qc": str(links.get("results/sc", "")).rstrip("/")
        + "/{cancer}/sc_cell_qc.tsv.gz",
        "malignant": str(links.get("results/sc", "")).rstrip("/")
        + "/{cancer}/sc_malignant_call.tsv.gz",
    }
    if templates != expected_templates:
        raise SingleCellDiagnosticError(
            "Audited input templates must be identical to the workspace-linked sources"
        )
    nonformal = config.get("nonformal_cancer_status", {})
    if set(nonformal) != set(EXPECTED_CANCERS) - set(FORMAL_CANCERS):
        raise SingleCellDiagnosticError("Every non-formal cancer needs one typed reason")
    runtime = config.get("runtime", {})
    if not runtime.get("python_executable") or not runtime.get("rscript_executable"):
        raise SingleCellDiagnosticError("Server Python and Rscript executables must be declared")
    if not runtime.get("required_python_packages") or not runtime.get("required_r_packages"):
        raise SingleCellDiagnosticError("Runtime dependency gates cannot be empty")


def validate_source_snapshot(snapshot_root: str | Path) -> dict[str, Any]:
    root = Path(snapshot_root).resolve()
    discriminator = root / "15_sc_patient_pseudobulk_ssgsea.R"
    if not discriminator.is_file():
        raise SingleCellDiagnosticError(f"Recovered source is missing: {discriminator}")
    discriminator_sha = sha256_file(discriminator)
    matching_profiles = [
        profile
        for profile, overrides in SOURCE_SNAPSHOT_SHA256_OVERRIDES.items()
        if overrides.get(
            "15_sc_patient_pseudobulk_ssgsea.R",
            REQUIRED_SOURCE_FILES["15_sc_patient_pseudobulk_ssgsea.R"],
        ) == discriminator_sha
    ]
    if len(matching_profiles) != 1:
        raise SingleCellDiagnosticError(
            "Recovered source SHA drift for 15_sc_patient_pseudobulk_ssgsea.R: "
            f"unrecognized={discriminator_sha}"
        )
    profile = matching_profiles[0]
    expected_files = dict(REQUIRED_SOURCE_FILES)
    expected_files.update(SOURCE_SNAPSHOT_SHA256_OVERRIDES[profile])
    observed: dict[str, Any] = {}
    for name, expected in expected_files.items():
        path = root / name
        if not path.is_file():
            raise SingleCellDiagnosticError(f"Recovered source is missing: {path}")
        sha = sha256_file(path)
        if sha != expected:
            raise SingleCellDiagnosticError(
                f"Recovered source SHA drift for {name}: {sha} != {expected}"
            )
        observed[name] = {"path": str(path), "sha256": sha, "bytes": path.stat().st_size}
    return observed


def _assert_raw_h5_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").lower()
    matches = [token for token in FORBIDDEN_EXPRESSION_TOKENS if token in lowered]
    if matches or path.name != "raw_feature_bc_matrix.h5":
        raise SingleCellDiagnosticError(
            f"Expression source is not the admitted raw 10x H5: {path}; tokens={matches}"
        )


def _read_tsv_columns(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", usecols=list(columns), low_memory=False)


def _h5_barcodes(path: Path) -> tuple[list[str], tuple[int, int]]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - server dependency gate
        raise SingleCellDiagnosticError("h5py is required for raw-H5 preflight") from exc
    with h5py.File(path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        required = {"data", "indices", "indptr", "shape", "barcodes", "features"}
        if missing := sorted(required - set(group.keys())):
            raise SingleCellDiagnosticError(f"10x H5 lacks matrix members: {missing}")
        shape = tuple(int(item) for item in group["shape"][:])
        features = group["features"]
        if "id" not in features or "name" not in features:
            raise SingleCellDiagnosticError("10x H5 features lack id/name")
        feature_count = len(features["id"])
        if len(features["name"]) != feature_count:
            raise SingleCellDiagnosticError("10x H5 feature id/name lengths disagree")
        raw = group["barcodes"][:]
        barcodes = [
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item)
            for item in raw
        ]
    if (
        len(shape) != 2
        or shape[0] != feature_count
        or shape[1] != len(barcodes)
        or len(set(barcodes)) != len(barcodes)
    ):
        raise SingleCellDiagnosticError("10x H5 shape/barcode contract failed")
    return barcodes, (shape[0], shape[1])


def audit_cancer_inputs(
    *,
    cancer: str,
    config: Mapping[str, Any],
    require_available: bool,
) -> dict[str, Any]:
    templates = config["input_templates"]
    paths = {
        role: Path(str(template).format(cancer=cancer)).resolve()
        for role, template in templates.items()
        if role in {"raw_h5", "annotation", "qc", "malignant"}
    }
    _assert_raw_h5_path(paths["raw_h5"])
    missing = [role for role, path in paths.items() if not path.is_file()]
    if missing:
        if require_available:
            raise SingleCellDiagnosticError(f"{cancer} lacks required fresh inputs: {missing}")
        return {
            "cancer_id": cancer,
            "formal_expected": cancer in FORMAL_CANCERS,
            "diagnostic_runnable": False,
            "typed_unavailable_reason": "RAW_H5_OR_DONOR_CELLTYPE_METADATA_UNAVAILABLE",
            "missing_roles": missing,
            "paths": {key: str(value) for key, value in paths.items()},
        }

    annotation = _read_tsv_columns(paths["annotation"], REQUIRED_METADATA_COLUMNS["annotation"])
    qc_columns = set(pd.read_csv(paths["qc"], sep="\t", nrows=0).columns)
    malignant_columns = set(pd.read_csv(paths["malignant"], sep="\t", nrows=0).columns)
    if missing := sorted(REQUIRED_METADATA_COLUMNS["qc"] - qc_columns):
        raise SingleCellDiagnosticError(f"{cancer} QC metadata lacks columns: {missing}")
    if missing := sorted(REQUIRED_METADATA_COLUMNS["malignant"] - malignant_columns):
        raise SingleCellDiagnosticError(f"{cancer} malignant metadata lacks columns: {missing}")
    qc_use = list(REQUIRED_METADATA_COLUMNS["qc"] | ({"input_expression_scale"} & qc_columns))
    qc = _read_tsv_columns(paths["qc"], qc_use)
    malignant = _read_tsv_columns(paths["malignant"], REQUIRED_METADATA_COLUMNS["malignant"])
    for name, frame in (("annotation", annotation), ("qc", qc), ("malignant", malignant)):
        if frame.empty or frame.cell_id.astype(str).duplicated().any():
            raise SingleCellDiagnosticError(f"{cancer} {name} cell IDs are empty or duplicated")
    if annotation.raw_cell_id.astype(str).duplicated().any():
        raise SingleCellDiagnosticError(f"{cancer} annotation raw_cell_id values are duplicated")
    joined = annotation.merge(qc, on="cell_id", validate="one_to_one").merge(
        malignant, on="cell_id", validate="one_to_one"
    )
    if len(joined) != len(annotation) or len(qc) != len(annotation) or len(malignant) != len(annotation):
        raise SingleCellDiagnosticError(f"{cancer} annotation/QC/malignant cell sets differ")
    barcodes, shape = _h5_barcodes(paths["raw_h5"])
    cell_ids = set(annotation.cell_id.astype(str))
    raw_ids = set(annotation.raw_cell_id.astype(str))
    unmatched = [barcode for barcode in barcodes if barcode not in cell_ids and barcode not in raw_ids]
    if unmatched or len(barcodes) != len(annotation):
        raise SingleCellDiagnosticError(
            f"{cancer} H5/metadata one-to-one gate failed; unmatched={len(unmatched)}, "
            f"h5={len(barcodes)}, metadata={len(annotation)}"
        )
    pass_qc = _truthy(joined.pass_qc)
    primary = _truthy(joined.primary_analysis_flag)
    malignant_rows = joined.loc[pass_qc & primary]
    if malignant_rows.empty:
        raise SingleCellDiagnosticError(f"{cancer} has no pass-QC primary malignant cells")
    normal = malignant_rows.sample_class.fillna("").astype(str).str.lower().isin(
        {"normal", "healthy", "adjacent_normal"}
    )
    if normal.any():
        raise SingleCellDiagnosticError(f"{cancer} malignant truth contains normal-source cells")
    immune_text = (
        joined.cell_type_major.fillna("").astype(str).str.lower()
        + " "
        + joined.get("cell_type_raw", pd.Series("", index=joined.index)).fillna("").astype(str).str.lower()
    )
    immune = (
        pass_qc
        & ~primary
        & joined.malignant_status.fillna("").astype(str).str.lower().eq("non_malignant")
        & immune_text.str.contains(
            "lymph|myeloid|t cell|b cell|nk|monocyte|macrophage|dendritic|neutroph|mast|plasma",
            regex=True,
        )
    )
    donors = malignant_rows.patient_id.dropna().astype(str)
    donor_count = donors[donors.str.strip().ne("")].nunique()
    record = {
        "cancer_id": cancer,
        "formal_expected": cancer in FORMAL_CANCERS,
        "diagnostic_runnable": True,
        "typed_unavailable_reason": None,
        "paths": {key: str(value) for key, value in paths.items()},
        "sha256": {key: sha256_file(value) for key, value in paths.items()},
        "h5_genes": shape[0],
        "h5_cells": shape[1],
        "metadata_cells": int(len(joined)),
        "primary_malignant_cells": int(len(malignant_rows)),
        "primary_malignant_donors": int(donor_count),
        "immune_candidate_cells": int(immune.sum()),
        "trajectory_numeric_precondition": bool(len(malignant_rows) >= 200),
        "donor_association_precondition": bool(donor_count >= 5),
    }
    if cancer not in FORMAL_CANCERS:
        record["diagnostic_runnable"] = False
        record["typed_unavailable_reason"] = config["nonformal_cancer_status"][cancer]
    if donor_count < 3:
        record["diagnostic_limitations"] = ["INSUFFICIENT_DONOR_REPLICATION_FOR_PATHWAY_STATS"]
    return record


def validate_exact_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    authority = config["exact_pathway_authority"]
    candidate_path = Path(authority["candidate_path"]).resolve()
    membership_path = Path(authority["membership_path"]).resolve()
    for path, key in ((candidate_path, "candidate_sha256"), (membership_path, "membership_sha256")):
        if not path.is_file():
            raise SingleCellDiagnosticError(f"Missing exact authority: {path}")
        observed = sha256_file(path)
        if observed != authority[key]:
            raise SingleCellDiagnosticError(f"Exact authority SHA drift: {path}")
    candidates = pd.read_parquet(candidate_path, columns=["cancer_id", "pathway_id"])
    exact_ids = sorted(candidates.pathway_id.dropna().astype(str).unique())
    if (
        len(candidates) != EXPECTED_CANDIDATE_ROWS
        or set(candidates.cancer_id.astype(str).str.upper()) != set(EXPECTED_CANCERS)
        or len(exact_ids) != EXPECTED_EXACT_PATHWAYS
    ):
        raise SingleCellDiagnosticError("Candidate exact-pathway authority failed cardinality gates")
    membership = pd.read_parquet(membership_path, columns=["pathway_id", "gene_id"])
    membership["pathway_id"] = membership.pathway_id.astype(str)
    filtered = membership.loc[membership.pathway_id.isin(exact_ids)].drop_duplicates()
    if len(filtered) != EXPECTED_EXACT_MEMBERSHIP_EDGES or filtered.pathway_id.nunique() != EXPECTED_EXACT_PATHWAYS:
        raise SingleCellDiagnosticError("Exact membership failed 352,205-edge/2,135-pathway gate")
    return {
        "candidate_path": str(candidate_path),
        "candidate_sha256": authority["candidate_sha256"],
        "candidate_rows": int(len(candidates)),
        "candidate_cancers": int(candidates.cancer_id.nunique()),
        "membership_path": str(membership_path),
        "membership_sha256": authority["membership_sha256"],
        "exact_pathways": len(exact_ids),
        "exact_membership_edges": int(len(filtered)),
        "exact_pathway_id_sha256": _canonical_sha(exact_ids),
    }


def _stable_gene_id(value: Any) -> str:
    text = str(value).strip()
    for prefix in ("GENE:", "PROTEIN:"):
        if text.upper().startswith(prefix):
            text = text[len(prefix) :]
            break
    head, separator, tail = text.rpartition(".")
    return head if separator and tail.isdigit() else text


def _current_exact_symbol_membership(
    config: Mapping[str, Any],
) -> tuple[set[str], pd.DataFrame]:
    """Translate the pinned exact Ensembl membership to unique gene symbols."""

    authority = config["exact_pathway_authority"]
    candidates = pd.read_parquet(authority["candidate_path"], columns=["pathway_id"])
    exact_ids = set(candidates.pathway_id.dropna().astype(str))
    gene_path = Path(config["workspace_links"]["processed/dimensions"]).resolve() / "dim_gene.tsv"
    genes = pd.read_csv(
        gene_path,
        sep="\t",
        usecols=["ensembl_gene_id", "gene_symbol"],
        low_memory=False,
    ).dropna()
    genes["gene_id"] = genes.ensembl_gene_id.map(_stable_gene_id)
    genes["gene_symbol"] = genes.gene_symbol.astype(str).str.upper()
    conflicts = genes.groupby("gene_id", observed=True).gene_symbol.nunique()
    if (conflicts > 1).any():
        raise SingleCellDiagnosticError("dim_gene maps a stable Ensembl ID to multiple symbols")
    gene_map = genes.drop_duplicates("gene_id").set_index("gene_id").gene_symbol

    membership = pd.read_parquet(
        authority["membership_path"], columns=["pathway_id", "gene_id"]
    )
    membership["pathway_id"] = membership.pathway_id.astype(str)
    membership = membership.loc[membership.pathway_id.isin(exact_ids)].copy()
    membership["gene_id"] = membership.gene_id.map(_stable_gene_id)
    membership["gene_symbol"] = membership.gene_id.map(gene_map)
    if membership.gene_symbol.isna().any():
        raise SingleCellDiagnosticError(
            "Current exact membership contains Ensembl IDs absent from dim_gene"
        )
    symbol_membership = (
        membership[["pathway_id", "gene_symbol"]]
        .drop_duplicates()
        .sort_values(["pathway_id", "gene_symbol"], kind="stable")
        .reset_index(drop=True)
    )
    return exact_ids, symbol_membership


def validate_reference_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind static references and prove old catalog IDs/members match V3.2.

    The recovered ssGSEA script intentionally starts from the 2,547-pathway
    MSigDB reference catalog.  The formal V3.2 universe is the 2,135 members
    that survive its domain plus PROGENy/DoRothEA.  Literal-ID post-filtering
    is only safe if the exact member definitions also match, so this preflight
    compares the current Ensembl authority to the symbol-based R reference.
    """

    links = config["workspace_links"]
    observed: dict[str, Any] = {}
    for relative, raw in links.items():
        source = Path(raw).resolve()
        if not source.exists():
            raise SingleCellDiagnosticError(f"Missing workspace reference {relative}: {source}")
        record: dict[str, Any] = {
            "path": str(source),
            "kind": "directory" if source.is_dir() else "file",
        }
        if source.is_file():
            record.update({"sha256": sha256_file(source), "bytes": source.stat().st_size})
        observed[relative] = record

    dimensions = Path(links["processed/dimensions"]).resolve()
    pathway_path = dimensions / "dim_pathway.tsv"
    gene_path = dimensions / "dim_gene.tsv"
    lnc_path = dimensions / "dim_lncRNA.tsv"
    member_path = Path(links["processed/pathway_gene_member.tsv"]).resolve()
    weighted_member_path = Path(
        links["processed/pathway_gene_member_weighted.tsv"]
    ).resolve()
    for required in (pathway_path, gene_path, lnc_path, member_path, weighted_member_path):
        if not required.is_file():
            raise SingleCellDiagnosticError(f"Missing pathway reference table: {required}")

    exact_ids, exact_symbol_membership = _current_exact_symbol_membership(config)
    pathway = pd.read_csv(pathway_path, sep="\t", low_memory=False)
    required_pathway = {"pathway_id", "pathway_source", "pathway_collection"}
    if missing := sorted(required_pathway - set(pathway.columns)):
        raise SingleCellDiagnosticError(f"dim_pathway lacks columns: {missing}")
    msig = pathway.loc[
        pathway.pathway_source.eq("MSigDB")
        & pathway.pathway_collection.isin(["HALLMARK", "KEGG_MEDICUS", "REACTOME"])
    ]
    if len(msig) != 2_547:
        raise SingleCellDiagnosticError(
            f"Recovered ssGSEA catalog requires 2,547 MSigDB rows; observed={len(msig)}"
        )
    missing_ids = exact_ids - set(pathway.pathway_id.astype(str))
    if missing_ids:
        raise SingleCellDiagnosticError(
            f"Reference catalog lacks {len(missing_ids)} current exact pathway IDs"
        )

    expected_pairs = set(
        zip(
            exact_symbol_membership.pathway_id,
            exact_symbol_membership.gene_symbol,
            strict=True,
        )
    )
    member_columns = set(pd.read_csv(member_path, sep="\t", nrows=0).columns)
    if not {"pathway_id", "gene_symbol"}.issubset(member_columns):
        raise SingleCellDiagnosticError("pathway_gene_member lacks pathway_id/gene_symbol")
    reference = pd.read_csv(
        member_path,
        sep="\t",
        usecols=["pathway_id", "gene_symbol"],
        low_memory=False,
    )
    reference = reference.loc[reference.pathway_id.astype(str).isin(exact_ids)].copy()
    reference["pathway_id"] = reference.pathway_id.astype(str)
    reference["gene_symbol_upper"] = reference.gene_symbol.astype(str).str.upper()
    observed_pairs = set(zip(reference.pathway_id, reference.gene_symbol_upper, strict=True))
    missing_pairs = expected_pairs - observed_pairs
    extra_pairs = observed_pairs - expected_pairs
    if missing_pairs:
        raise SingleCellDiagnosticError(
            "Recovered symbol membership lacks current V3.2 exact members: "
            f"missing={len(missing_pairs)}"
        )
    weighted_columns = set(
        pd.read_csv(weighted_member_path, sep="\t", nrows=0).columns
    )
    if not {"pathway_id", "gene_symbol", "weight"}.issubset(weighted_columns):
        raise SingleCellDiagnosticError(
            "Signed pathway reference lacks pathway_id/gene_symbol/weight"
        )
    dorothea_ids = set(
        pathway.loc[pathway.pathway_source.eq("DoRothEA"), "pathway_id"].astype(str)
    ) & exact_ids
    if not dorothea_ids:
        raise SingleCellDiagnosticError("Exact V3.2 authority contains no DoRothEA IDs")
    weighted = pd.read_csv(
        weighted_member_path,
        sep="\t",
        usecols=["pathway_id", "gene_symbol", "weight"],
        low_memory=False,
    )
    weighted = weighted.loc[weighted.pathway_id.astype(str).isin(dorothea_ids)].copy()
    weighted["pathway_id"] = weighted.pathway_id.astype(str)
    weighted["gene_symbol_upper"] = weighted.gene_symbol.astype(str).str.upper()
    weighted["weight"] = pd.to_numeric(weighted.weight, errors="coerce")
    if weighted.empty or weighted.weight.isna().any():
        raise SingleCellDiagnosticError(
            "DoRothEA signed reference contains missing/non-numeric weights"
        )
    if weighted[["pathway_id", "gene_symbol_upper"]].duplicated().any():
        raise SingleCellDiagnosticError(
            "DoRothEA signed reference contains duplicate regulator-target rows"
        )
    expected_dorothea_pairs = {
        pair for pair in expected_pairs if pair[0] in dorothea_ids
    }
    weighted_pairs = set(
        zip(weighted.pathway_id, weighted.gene_symbol_upper, strict=True)
    )
    missing_weighted_pairs = expected_dorothea_pairs - weighted_pairs
    extra_weighted_pairs = weighted_pairs - expected_dorothea_pairs
    if missing_weighted_pairs:
        raise SingleCellDiagnosticError(
            "DoRothEA signed reference lacks current exact V3.2 members: "
            f"missing={len(missing_weighted_pairs)}"
        )
    observed["catalog_contract"] = {
        "reference_msigdb_pathways": int(len(msig)),
        "current_exact_pathways": int(len(exact_ids)),
        "current_exact_symbol_membership_edges": int(len(expected_pairs)),
        "literal_id_subset": True,
        "reference_contains_all_current_exact_members": True,
        "reference_extra_edges_excluded": int(len(extra_pairs)),
        "workspace_membership_rebuilt_from_exact_authority": True,
        "member_definition_equal_after_exact_rebuild": True,
        "postfilter_required": True,
        "dorothea_signed_reference_path": str(weighted_member_path),
        "dorothea_signed_reference_sha256": sha256_file(weighted_member_path),
        "dorothea_signed_pathways": len(dorothea_ids),
        "dorothea_signed_reference_edges": len(weighted_pairs),
        "dorothea_exact_weighted_edges": len(expected_dorothea_pairs),
        "dorothea_extra_reference_edges_excluded": len(extra_weighted_pairs),
        "dorothea_all_exact_members_have_signed_weights": True,
    }
    return observed


def validate_runtime_dependencies(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check server Python/R packages without starting biological computation."""

    runtime = config["runtime"]
    configured_python = Path(runtime["python_executable"]).resolve()
    running_python = Path(sys.executable).resolve()
    if not configured_python.is_file():
        raise SingleCellDiagnosticError(f"Pinned server Python is missing: {configured_python}")
    if configured_python != running_python:
        raise SingleCellDiagnosticError(
            f"Preflight must run with the pinned Python: {running_python} != {configured_python}"
        )
    python_versions: dict[str, str] = {}
    for package in runtime["required_python_packages"]:
        if find_spec(package) is None:
            raise SingleCellDiagnosticError(f"Missing Python package: {package}")
        try:
            python_versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            python_versions[package] = "importable_version_metadata_unavailable"

    rscript_token = str(runtime["rscript_executable"])
    absolute_rscript = Path(rscript_token) if Path(rscript_token).is_absolute() else None
    executable = (
        str(absolute_rscript.resolve()) if absolute_rscript is not None else shutil.which(rscript_token)
    )
    if not executable or not Path(executable).is_file():
        raise SingleCellDiagnosticError(f"Pinned Rscript is missing: {rscript_token}")
    packages = [str(item) for item in runtime["required_r_packages"]]
    package_vector = ",".join(json.dumps(item) for item in packages)
    expression = (
        f"p<-c({package_vector});"
        "m<-p[!vapply(p,requireNamespace,logical(1),quietly=TRUE)];"
        "if(length(m)){cat(paste(m,collapse=','));quit(status=71)};"
        "cat(paste(vapply(p,function(x)paste0(x,'=',as.character(packageVersion(x))),"
        "character(1)),collapse=';'))"
    )
    r_library_paths = [
        str(Path(item).resolve())
        for item in runtime.get(
            "r_library_paths", [config["workspace_links"]["r_libs"]]
        )
    ]
    if not r_library_paths or any(not Path(item).is_dir() for item in r_library_paths):
        raise SingleCellDiagnosticError(
            f"Pinned R library path is missing: {r_library_paths}"
        )
    environment = os.environ.copy()
    environment["R_LIBS_USER"] = os.pathsep.join(r_library_paths)
    result = subprocess.run(
        [str(executable), "--vanilla", "-e", expression],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    if result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr).strip()
        raise SingleCellDiagnosticError(f"R dependency gate failed: {detail}")
    r_versions = dict(
        item.split("=", 1) for item in result.stdout.strip().split(";") if "=" in item
    )
    pinned = runtime.get("pinned_r_package_versions", {})
    drift = {
        package: {"expected": expected, "observed": r_versions.get(package)}
        for package, expected in pinned.items()
        if r_versions.get(package) != str(expected)
    }
    if drift:
        raise SingleCellDiagnosticError(f"Pinned R package version drift: {drift}")
    return {
        "python_executable": str(running_python),
        "python_packages": python_versions,
        "rscript_executable": str(Path(executable).resolve()),
        "r_library_paths": r_library_paths,
        "r_packages": r_versions,
        "biological_computation_started": False,
    }


def build_preflight(
    config_path: str | Path,
    *,
    scope: str,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _load_json(path)
    validate_config(config)
    if scope not in {"formal17", "coverage33"}:
        raise SingleCellDiagnosticError("scope must be formal17 or coverage33")
    snapshot_root = _as_path(path, config["source_snapshot"]["path"])
    sources = validate_source_snapshot(snapshot_root)
    output_root = Path(config["compute_output_root"])
    if output_root.exists():
        raise SingleCellDiagnosticError(f"Fresh compute output already exists: {output_root}")
    exact = validate_exact_authority(config) if verify_inputs else {
        "status": "DEFERRED_TO_SERVER_FILE_PREFLIGHT",
        "exact_pathways": EXPECTED_EXACT_PATHWAYS,
    }
    references = validate_reference_inputs(config) if verify_inputs else {
        "status": "DEFERRED_TO_SERVER_FILE_PREFLIGHT"
    }
    runtime = validate_runtime_dependencies(config) if verify_inputs else {
        "status": "DEFERRED_TO_SERVER_FILE_PREFLIGHT"
    }
    selected = FORMAL_CANCERS if scope == "formal17" else EXPECTED_CANCERS
    records: list[dict[str, Any]] = []
    if verify_inputs:
        for cancer in selected:
            records.append(
                audit_cancer_inputs(
                    cancer=cancer,
                    config=config,
                    require_available=scope == "formal17",
                )
            )
    else:
        records = [
            {
                "cancer_id": cancer,
                "formal_expected": cancer in FORMAL_CANCERS,
                "diagnostic_runnable": cancer in FORMAL_CANCERS,
                "typed_unavailable_reason": (
                    None if cancer in FORMAL_CANCERS else "DEFERRED_33C_COVERAGE_GATE"
                ),
            }
            for cancer in selected
        ]
    runnable = [row["cancer_id"] for row in records if row["diagnostic_runnable"]]
    expected = list(FORMAL_CANCERS)
    status = "PASS" if runnable == expected else "FAIL"
    payload: dict[str, Any] = {
        "format": PREFLIGHT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "scope": scope,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "source_snapshot": sources,
        "exact_pathway_authority": exact,
        "static_reference_inputs": references,
        "runtime_dependencies": runtime,
        "cancer_records": records,
        "run_cancers": runnable,
        "coverage_cancers": list(selected),
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "historical_result_reuse_allowed": False,
        "processed_seurat_fallback_allowed": False,
        "all_expression_from_raw_h5": True,
        "computation_started": False,
        "training_started": False,
        "model_fusion_permitted": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "release_ready": False,
        "production_deployed": False,
        "known_statistical_risks": [
            "SAME_PSEUDOBULK_EXPRESSION_DRIVES_LNCRNA_AND_PATHWAY_ACTIVITY",
            "LEGACY_WITHIN_PATIENT_QUINTILE_CORRELATION_USES_GROUP_LEVEL_DF",
            "ALL_DONOR_UNSUPERVISED_PREPROCESSING_NOT_FOLD_LOCAL",
        ],
        "full_exact_pathway_cell_level_ucell_in_seven_file_snapshot": False,
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def write_preflight(
    config_path: str | Path,
    output_path: str | Path,
    *,
    scope: str,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    output = Path(output_path).resolve()
    if output.exists():
        raise SingleCellDiagnosticError(f"Refusing preflight output reuse: {output}")
    payload = build_preflight(config_path, scope=scope, verify_inputs=verify_inputs)
    _atomic_json(output, payload)
    return payload


def _safe_symlink(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise SingleCellDiagnosticError(f"Workspace target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
    except OSError:
        # Windows CI commonly lacks SeCreateSymbolicLinkPrivilege.  Copying is
        # limited to test/local preparation; the Linux server uses symlinks so
        # large immutable input trees are never duplicated.
        if os.name != "nt":
            raise
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def prepare_workspace(
    config_path: str | Path,
    preflight_path: str | Path,
    *,
    expected_preflight_sha256: str,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_json(config_file)
    validate_config(config)
    preflight_file = Path(preflight_path).resolve()
    observed_preflight_sha = sha256_file(preflight_file)
    if observed_preflight_sha != expected_preflight_sha256:
        raise SingleCellDiagnosticError("Preflight SHA drift")
    preflight = _load_json(preflight_file)
    if preflight.get("format") != PREFLIGHT_FORMAT or preflight.get("status") != "PASS":
        raise SingleCellDiagnosticError("Workspace requires a passing diagnostic preflight")
    if preflight.get("config_sha256") != sha256_file(config_file):
        raise SingleCellDiagnosticError("Preflight does not bind the current config SHA")
    if preflight.get("computation_started") is not False:
        raise SingleCellDiagnosticError("Preflight is not a pre-compute gate")
    required_preflight_policy = {
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "historical_result_reuse_allowed": False,
        "processed_seurat_fallback_allowed": False,
        "all_expression_from_raw_h5": True,
        "model_fusion_permitted": False,
    }
    for key, expected in required_preflight_policy.items():
        if preflight.get(key) != expected:
            raise SingleCellDiagnosticError(f"Preflight policy drift for {key}")
    if tuple(preflight.get("run_cancers", ())) != FORMAL_CANCERS:
        raise SingleCellDiagnosticError("Execution is restricted to the audited formal 17")
    output = Path(config["compute_output_root"]).resolve()
    if output.exists():
        raise SingleCellDiagnosticError(f"Refusing compute output reuse: {output}")
    workspace = output / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    snapshot = _as_path(config_file, config["source_snapshot"]["path"])
    validate_source_snapshot(snapshot)
    for name in REQUIRED_SOURCE_FILES:
        destination = workspace / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / name, destination)
    for name in ("13_sc_malignant_trajectory.R", "14_sc_immune_subtype.R", "15_sc_patient_pseudobulk_ssgsea.R"):
        destination = workspace / "R" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / name, destination)
    for name in ("61_sc_trajectory_pathway_stats.py", "63_sc_lnc_pathway_association_ssgsea.py"):
        destination = workspace / "python" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot / name, destination)
    runner = workspace / "scripts" / "run_sc_trajectory_one.sh"
    runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot / "run_sc_trajectory_one.sh", runner)
    runner.chmod(0o750)
    (workspace / "python" / "common.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "ROOT = Path(os.environ['CANCERLNCATLAS_ROOT']).resolve()\n"
        f"TCGA_CANCERS = frozenset({EXPECTED_CANCERS!r})\n",
        encoding="utf-8",
    )
    links = config["workspace_links"]
    for relative, source in links.items():
        if relative == "processed/pathway_gene_member.tsv":
            continue
        _safe_symlink(Path(source).resolve(), workspace / relative)
    _, exact_symbol_membership = _current_exact_symbol_membership(config)
    exact_member_path = workspace / "processed" / "pathway_gene_member.tsv"
    exact_member_path.parent.mkdir(parents=True, exist_ok=True)
    exact_symbol_membership.to_csv(exact_member_path, sep="\t", index=False)
    authority_dir = workspace / "authority"
    authority_dir.mkdir(parents=True, exist_ok=True)
    _safe_symlink(
        Path(config["exact_pathway_authority"]["candidate_path"]).resolve(),
        authority_dir / "FORMAL_CANDIDATE_UNIVERSE.parquet",
    )
    _safe_symlink(
        Path(config["exact_pathway_authority"]["membership_path"]).resolve(),
        authority_dir / "exact_pathway_gene_membership_ensembl.parquet",
    )
    manifest = {
        "format": WORKSPACE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "workspace": str(workspace),
        "preflight_path": str(preflight_file),
        "preflight_sha256": observed_preflight_sha,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "run_cancers": preflight["run_cancers"],
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "historical_output_tree_linked": False,
        "processed_seurat_tree_linked": False,
        "model_fusion_permitted": False,
        "exact_symbol_membership_path": str(exact_member_path),
        "exact_symbol_membership_sha256": sha256_file(exact_member_path),
        "exact_symbol_membership_edges": int(len(exact_symbol_membership)),
        "computation_started": False,
        "production_deployed": False,
    }
    manifest["contract_sha256"] = _canonical_sha(manifest)
    _atomic_json(output / "WORKSPACE_MANIFEST.json", manifest)
    return manifest


def _load_exact_ids(config: Mapping[str, Any]) -> set[str]:
    candidate = pd.read_parquet(
        config["exact_pathway_authority"]["candidate_path"], columns=["pathway_id"]
    )
    exact = set(candidate.pathway_id.dropna().astype(str))
    if len(exact) != EXPECTED_EXACT_PATHWAYS:
        raise SingleCellDiagnosticError("Exact-pathway ID set drift during finalization")
    return exact


def _filter_exact_tsv(source: Path, destination: Path, exact_ids: set[str]) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    rows = 0
    wrote = False
    for chunk in pd.read_csv(source, sep="\t", chunksize=250_000, low_memory=False):
        if "pathway_id" not in chunk.columns:
            raise SingleCellDiagnosticError(f"Pathway output lacks pathway_id: {source}")
        selected = chunk.loc[chunk.pathway_id.astype(str).isin(exact_ids)].copy()
        if selected.empty:
            continue
        selected["analysis_version"] = ANALYSIS_VERSION
        selected["evidence_tier"] = "diagnostic_inferred_root"
        selected["root_provenance"] = ROOT_PROVENANCE
        selected["root_is_explicit"] = False
        selected.to_csv(
            temporary,
            sep="\t",
            index=False,
            mode="a" if wrote else "w",
            header=not wrote,
            compression="gzip",
        )
        rows += len(selected)
        wrote = True
    if not wrote:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            handle.write("pathway_id\tanalysis_version\tevidence_tier\troot_provenance\troot_is_explicit\n")
    os.replace(temporary, destination)
    return int(rows)


def finalize_diagnostic(
    config_path: str | Path,
    *,
    expected_workspace_manifest_sha256: str,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config = _load_json(config_file)
    validate_config(config)
    output = Path(config["compute_output_root"]).resolve()
    workspace_manifest_path = output / "WORKSPACE_MANIFEST.json"
    if sha256_file(workspace_manifest_path) != expected_workspace_manifest_sha256:
        raise SingleCellDiagnosticError("Workspace manifest SHA drift")
    workspace_manifest = _load_json(workspace_manifest_path)
    exact_ids = _load_exact_ids(config)
    artifacts: dict[str, Any] = {}
    for cancer in workspace_manifest["run_cancers"]:
        stage = output / "workspace" / "results" / "sc_trajectory_staging" / cancer
        required = {
            "activity": stage / "sc_ssgsea_activity.tsv.gz",
            "pathway_stats": stage / "pseudotime_pathway_association.tsv.gz",
            "lncrna_pathway": stage / "sc_lncRNA_pathway_association.tsv.gz",
        }
        missing = [key for key, value in required.items() if not value.is_file()]
        if missing:
            raise SingleCellDiagnosticError(f"{cancer} lacks diagnostic outputs: {missing}")
        cancer_artifacts: dict[str, Any] = {}
        for role, source in required.items():
            destination = stage / f"v32_exact2135_{role}.tsv.gz"
            rows = _filter_exact_tsv(source, destination, exact_ids)
            cancer_artifacts[role] = {
                "path": str(destination),
                "sha256": sha256_file(destination),
                "rows": rows,
                "exact_pathway_filtered": True,
            }
        artifacts[cancer] = cancer_artifacts
    payload = {
        "format": FINAL_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "status": "DIAGNOSTIC_COMPLETE_NOT_MODEL_FUSED",
        "cancers": list(workspace_manifest["run_cancers"]),
        "artifacts": artifacts,
        "root_provenance": ROOT_PROVENANCE,
        "root_is_explicit": False,
        "exact_pathway_count": EXPECTED_EXACT_PATHWAYS,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
        "historical_sc_trajectory_outputs_used": False,
        "diagnostic_only": True,
        "model_fusion_permitted": False,
        "primary_score_weight": 0,
        "secondary_score_weight": 0,
        "full_exact_pathway_cell_level_ucell_complete": False,
        "full_exact_pathway_cell_level_ucell_typed_status": (
            "SEPARATE_CURRENT_V32_CELL_LEVEL_RUNNER_REQUIRED"
        ),
        "known_statistical_risks": [
            "SAME_PSEUDOBULK_EXPRESSION_DRIVES_LNCRNA_AND_PATHWAY_ACTIVITY",
            "LEGACY_WITHIN_PATIENT_QUINTILE_CORRELATION_USES_GROUP_LEVEL_DF",
            "ALL_DONOR_UNSUPERVISED_PREPROCESSING_NOT_FOLD_LOCAL",
        ],
        "release_ready": False,
        "production_deployed": False,
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    _atomic_json(output / "V32_DIAGNOSTIC_MANIFEST.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--output", required=True)
    preflight.add_argument("--scope", choices=("formal17", "coverage33"), required=True)
    preflight.add_argument("--defer-remote-input-checks", action="store_true")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--preflight", required=True)
    prepare.add_argument("--expected-preflight-sha256", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--expected-workspace-manifest-sha256", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = write_preflight(
            args.config,
            args.output,
            scope=args.scope,
            verify_inputs=not args.defer_remote_input_checks,
        )
    elif args.command == "prepare":
        result = prepare_workspace(
            args.config,
            args.preflight,
            expected_preflight_sha256=args.expected_preflight_sha256,
        )
    else:
        result = finalize_diagnostic(
            args.config,
            expected_workspace_manifest_sha256=args.expected_workspace_manifest_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
