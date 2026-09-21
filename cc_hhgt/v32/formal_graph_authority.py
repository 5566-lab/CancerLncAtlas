"""Hash-bound input authority for fresh, fold-local V3.2 G0/G1/G2 graphs.

The graph materializers intentionally accept data frames.  This module is the
missing security boundary between those pure transforms and the formal
preparation entrypoint: every source is an explicit path, every path is pinned
by SHA256 and one immutable receipt, and fold-local expression is proven to
contain exactly the outer-train sample/patient identities from the frozen
patient-first authority.

No historical graph table, toy fallback, or implicit default path is accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .contracts import dataframe_sha256
from .formal_graph import (
    FormalGraphAuthority,
    build_formal_graph_authority,
    materialize_expressed_in,
    materialize_global_lnc_protein_binding,
    materialize_pathway_hierarchy,
    materialize_protein_gene_encoding,
    materialize_signed_coexpression,
    materialize_signed_membership,
    materialize_symmetric_ppi,
    variant_edges,
)
from .input_lineage import artifact_sha256
from .patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
)
from .safe_graph import EDGE_KEYS


GRAPH_INPUT_RECEIPT_FORMAT = "CANCERLNCATLAS_V32_FRESH_G012_INPUT_AUTHORITY_V1"
GRAPH_INPUT_RECEIPT_STATUS = "PASS_FRESH_HASH_BOUND_G012_AUTHORITIES"
GRAPH_PAYLOAD_BINDING_FORMAT = "CANCERLNCATLAS_V32_FRESH_G012_PAYLOAD_BINDING_V1"
GRAPH_PAYLOAD_BINDING_STATUS = "PASS_FRESH_FOLD_LOCAL_G012_GRAPH"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

STATIC_ARTIFACT_IDS = (
    "detection",
    "signed_membership",
    "pathway_hierarchy",
    "lnc_protein_binding",
    "protein_gene_encoding",
    "ppi",
)
FOLD_ARTIFACT_IDS = ("train_expression", "train_coexpression")


class FormalGraphAuthorityError(RuntimeError):
    """Raised when a fresh graph authority cannot be proven safe."""


@dataclass(frozen=True)
class FormalGraphInputAuthority:
    receipt_path: Path
    receipt_sha256: str
    receipt: Mapping[str, Any]
    static_paths: Mapping[str, Path]
    static_sha256: Mapping[str, str]
    fold_expression_pattern: str
    fold_coexpression_pattern: str


@dataclass(frozen=True)
class BoundFormalGraph:
    authority: FormalGraphAuthority
    binding: Mapping[str, Any]


def _declared_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256.fullmatch(digest):
        raise FormalGraphAuthorityError(f"{label} is not a lowercase SHA256")
    return digest


def _resolved_file_or_tree(value: str | Path, label: str) -> Path:
    path = Path(value).resolve()
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise FormalGraphAuthorityError(f"{label} is missing or unsafe: {path}")
    return path


def _read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a pinned table without guessing a different source."""

    suffixes = [value.lower() for value in path.suffixes]
    if path.is_dir() or ".parquet" in suffixes:
        return pd.read_parquet(path, columns=columns)
    separator = "," if ".csv" in suffixes else "\t"
    return pd.read_csv(path, sep=separator, usecols=columns, compression="infer")


def read_formal_graph_table(
    inputs: FormalGraphInputAuthority, artifact_id: str, *, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read one already hash-validated static graph authority."""

    if artifact_id not in inputs.static_paths:
        raise FormalGraphAuthorityError(f"Unknown static graph artifact: {artifact_id}")
    return _read_table(inputs.static_paths[artifact_id], columns=columns)


def _artifact_record(
    *,
    artifact_id: str,
    explicit_path: str | Path,
    explicit_sha256: str,
    declaration: Mapping[str, Any],
    expected_source_role: str,
) -> tuple[Path, str]:
    path = _resolved_file_or_tree(explicit_path, artifact_id)
    expected = _declared_sha256(explicit_sha256, f"{artifact_id} explicit SHA256")
    declared = _declared_sha256(declaration.get("sha256"), f"{artifact_id} receipt SHA256")
    declared_path = Path(str(declaration.get("path", ""))).resolve()
    if declared_path != path:
        raise FormalGraphAuthorityError(
            f"{artifact_id} explicit path differs from receipt: {path} != {declared_path}"
        )
    if expected != declared:
        raise FormalGraphAuthorityError(f"{artifact_id} explicit/receipt SHA256 drift")
    observed = artifact_sha256(path)
    if observed != expected:
        raise FormalGraphAuthorityError(
            f"{artifact_id} content SHA256 drift: {observed} != {expected}"
        )
    if (
        declaration.get("source_role") != expected_source_role
        or declaration.get("outcome_derived") is not False
        or declaration.get("historical_model_output") is not False
        or declaration.get("toy_or_synthetic") is not False
    ):
        raise FormalGraphAuthorityError(f"{artifact_id} lineage declaration is unsafe")
    return path, expected


def _format_fold_pattern(pattern: str, outer_fold: int, label: str) -> Path:
    if pattern.count("{fold}") != 1:
        raise FormalGraphAuthorityError(
            f"{label} must contain exactly one literal {{fold}} placeholder"
        )
    try:
        rendered = pattern.format(fold=int(outer_fold))
    except (KeyError, IndexError, ValueError) as exc:
        raise FormalGraphAuthorityError(f"Invalid {label}: {pattern}") from exc
    return Path(rendered).resolve()


def _identity_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame.loc[:, columns].drop_duplicates().copy()
    for column in columns:
        values[column] = values[column].astype(str).str.strip()
    values = values.sort_values(columns, kind="stable")
    payload = "".join(
        "\t".join(map(str, row)) + "\n"
        for row in values.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def train_identity_hashes(
    split_manifest: pd.DataFrame, *, cancer_scope: list[str]
) -> dict[str, str | int]:
    required = {"cancer_id", "sample_id", "patient_id", "split"}
    if missing := sorted(required - set(split_manifest)):
        raise FormalGraphAuthorityError(f"Split manifest lacks identity columns: {missing}")
    scope = set(map(str, cancer_scope))
    train = split_manifest.loc[
        split_manifest.cancer_id.astype(str).isin(scope)
        & split_manifest.split.astype(str).eq("train"),
        ["cancer_id", "sample_id", "patient_id"],
    ].copy()
    if train.empty or train.isna().any().any():
        raise FormalGraphAuthorityError("Outer-train identity authority is empty or null")
    return {
        "train_sample_patient_sha256": _identity_hash(
            train, ["cancer_id", "sample_id", "patient_id"]
        ),
        "train_patient_sha256": _identity_hash(train, ["cancer_id", "patient_id"]),
        "train_samples": int(train[["cancer_id", "sample_id"]].drop_duplicates().shape[0]),
        "train_patients": int(train[["cancer_id", "patient_id"]].drop_duplicates().shape[0]),
    }


def load_formal_graph_input_authority(
    *,
    receipt_path: str | Path,
    receipt_sha256: str,
    patient_authority_audit: Mapping[str, Any],
    static_inputs: Mapping[str, tuple[str | Path, str]],
    fold_expression_pattern: str,
    fold_coexpression_pattern: str,
) -> FormalGraphInputAuthority:
    """Validate the immutable receipt and every non-fold graph input."""

    receipt_file = _resolved_file_or_tree(receipt_path, "graph authority receipt")
    if not receipt_file.is_file():
        raise FormalGraphAuthorityError("Graph authority receipt must be one regular file")
    receipt_digest = _declared_sha256(receipt_sha256, "graph authority receipt SHA256")
    if artifact_sha256(receipt_file) != receipt_digest:
        raise FormalGraphAuthorityError("Graph authority receipt SHA256 drift")
    try:
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalGraphAuthorityError("Graph authority receipt is invalid JSON") from exc
    gates = receipt.get("gates", {}) if isinstance(receipt, Mapping) else {}
    if (
        receipt.get("format") != GRAPH_INPUT_RECEIPT_FORMAT
        or receipt.get("status") != GRAPH_INPUT_RECEIPT_STATUS
        or gates.get("historical_graph_rows_used") is not False
        or gates.get("legacy_graph_root_fallback_allowed") is not False
        or gates.get("toy_or_synthetic_fallback_allowed") is not False
        or gates.get("outer_train_expression_only") is not True
        or gates.get("outer_train_coexpression_only") is not True
        or gates.get("static_evidence_outcome_free") is not True
        or gates.get("same_node_and_relation_schema_all_variants") is not True
    ):
        raise FormalGraphAuthorityError("Graph authority receipt contract failed")
    patient = receipt.get("patient_fold_authority", {})
    if (
        patient.get("manifest_sha256") != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or patient.get("receipt_sha256") != FROZEN_V32_RECEIPT_SHA256
        or patient_authority_audit.get("manifest_sha256")
        != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or patient_authority_audit.get("receipt_sha256") != FROZEN_V32_RECEIPT_SHA256
        or patient_authority_audit.get("explicit_patient_id") is not True
        or patient_authority_audit.get("sample_fallback_used") is not False
    ):
        raise FormalGraphAuthorityError("Graph receipt is not bound to frozen patient-first folds")
    if set(static_inputs) != set(STATIC_ARTIFACT_IDS):
        raise FormalGraphAuthorityError(
            f"Explicit graph inputs must be exactly {list(STATIC_ARTIFACT_IDS)}"
        )
    declarations = receipt.get("artifacts", {})
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for artifact_id in STATIC_ARTIFACT_IDS:
        declaration = declarations.get(artifact_id)
        if not isinstance(declaration, Mapping):
            raise FormalGraphAuthorityError(f"Receipt lacks graph artifact {artifact_id}")
        expected_role = "outcome_free_expression_detection" if artifact_id == "detection" else "static_annotation"
        paths[artifact_id], hashes[artifact_id] = _artifact_record(
            artifact_id=artifact_id,
            explicit_path=static_inputs[artifact_id][0],
            explicit_sha256=static_inputs[artifact_id][1],
            declaration=declaration,
            expected_source_role=expected_role,
        )
    # Resolve both patterns now so an old launcher cannot pass one shared path.
    expression_paths = {
        _format_fold_pattern(fold_expression_pattern, fold, "fold expression pattern")
        for fold in range(5)
    }
    coexpression_paths = {
        _format_fold_pattern(fold_coexpression_pattern, fold, "fold coexpression pattern")
        for fold in range(5)
    }
    if len(expression_paths) != 5 or len(coexpression_paths) != 5:
        raise FormalGraphAuthorityError("Every outer fold requires a distinct graph authority path")
    fold_declarations = receipt.get("fold_artifacts", {})
    if set(fold_declarations) != {str(value) for value in range(5)}:
        raise FormalGraphAuthorityError("Graph receipt must declare all five outer folds")
    for outer_fold in range(5):
        declaration = fold_declarations[str(outer_fold)]
        if not isinstance(declaration, Mapping) or int(declaration.get("outer_fold", -1)) != outer_fold:
            raise FormalGraphAuthorityError(f"Malformed fold {outer_fold} graph declaration")
        for artifact_id, path, role in (
            (
                "train_expression",
                _format_fold_pattern(
                    fold_expression_pattern, outer_fold, "fold expression pattern"
                ),
                "outer_train_expression",
            ),
            (
                "train_coexpression",
                _format_fold_pattern(
                    fold_coexpression_pattern, outer_fold, "fold coexpression pattern"
                ),
                "outer_train_coexpression",
            ),
        ):
            artifact = declaration.get(artifact_id)
            if not isinstance(artifact, Mapping):
                raise FormalGraphAuthorityError(
                    f"Fold {outer_fold} lacks {artifact_id} declaration"
                )
            digest = _declared_sha256(
                artifact.get("sha256"), f"fold {outer_fold} {artifact_id} SHA256"
            )
            _artifact_record(
                artifact_id=f"fold_{outer_fold}_{artifact_id}",
                explicit_path=path,
                explicit_sha256=digest,
                declaration=artifact,
                expected_source_role=role,
            )
            if (
                artifact.get("source_split") != "train"
                or int(artifact.get("outer_fold", -1)) != outer_fold
                or artifact.get("patient_first") is not True
                or not _SHA256.fullmatch(
                    str(artifact.get("train_sample_patient_sha256", ""))
                )
                or not _SHA256.fullmatch(str(artifact.get("train_patient_sha256", "")))
            ):
                raise FormalGraphAuthorityError(
                    f"Fold {outer_fold} {artifact_id} lineage is not patient-first outer-train"
                )
    return FormalGraphInputAuthority(
        receipt_path=receipt_file,
        receipt_sha256=receipt_digest,
        receipt=receipt,
        static_paths=paths,
        static_sha256=hashes,
        fold_expression_pattern=str(fold_expression_pattern),
        fold_coexpression_pattern=str(fold_coexpression_pattern),
    )


def _validate_fold_artifact(
    inputs: FormalGraphInputAuthority,
    *,
    outer_fold: int,
    artifact_id: str,
    path: Path,
) -> tuple[Path, str, Mapping[str, Any]]:
    folds = inputs.receipt.get("fold_artifacts", {})
    fold = folds.get(str(int(outer_fold)))
    if not isinstance(fold, Mapping) or int(fold.get("outer_fold", -1)) != int(outer_fold):
        raise FormalGraphAuthorityError(f"Receipt lacks outer-fold {outer_fold} graph authority")
    declaration = fold.get(artifact_id)
    if not isinstance(declaration, Mapping):
        raise FormalGraphAuthorityError(f"Fold {outer_fold} lacks {artifact_id}")
    expected = _declared_sha256(declaration.get("sha256"), f"fold {outer_fold} {artifact_id}")
    role = "outer_train_expression" if artifact_id == "train_expression" else "outer_train_coexpression"
    resolved, observed = _artifact_record(
        artifact_id=f"fold_{outer_fold}_{artifact_id}",
        explicit_path=path,
        explicit_sha256=expected,
        declaration=declaration,
        expected_source_role=role,
    )
    if (
        declaration.get("source_split") != "train"
        or int(declaration.get("outer_fold", -1)) != int(outer_fold)
        or declaration.get("patient_first") is not True
    ):
        raise FormalGraphAuthorityError(f"Fold {outer_fold} {artifact_id} is not outer-train-only")
    return resolved, observed, declaration


def build_bound_formal_graph(
    inputs: FormalGraphInputAuthority,
    *,
    outer_fold: int,
    split_manifest: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> BoundFormalGraph:
    """Materialize one fold's G2 master authority and G0/G1 masks."""

    if not 0 <= int(outer_fold) < 5:
        raise FormalGraphAuthorityError("Outer fold must be in [0, 4]")
    candidate_required = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(candidate_required - set(candidate_pairs)):
        raise FormalGraphAuthorityError(f"Candidate authority lacks: {missing}")
    cancer_scope = sorted(candidate_pairs.cancer_id.astype(str).unique().tolist())
    declared_scope = sorted(map(str, inputs.receipt.get("cancer_scope", [])))
    if not cancer_scope or cancer_scope != declared_scope:
        raise FormalGraphAuthorityError(
            f"Candidate cancer scope differs from graph receipt: {cancer_scope} != {declared_scope}"
        )
    expected_identity = train_identity_hashes(split_manifest, cancer_scope=cancer_scope)
    expression_path = _format_fold_pattern(
        inputs.fold_expression_pattern, outer_fold, "fold expression pattern"
    )
    coexpression_path = _format_fold_pattern(
        inputs.fold_coexpression_pattern, outer_fold, "fold coexpression pattern"
    )
    expression_path, expression_sha, expression_decl = _validate_fold_artifact(
        inputs, outer_fold=outer_fold, artifact_id="train_expression", path=expression_path
    )
    coexpression_path, coexpression_sha, coexpression_decl = _validate_fold_artifact(
        inputs, outer_fold=outer_fold, artifact_id="train_coexpression", path=coexpression_path
    )
    for declaration in (expression_decl, coexpression_decl):
        for key in ("train_sample_patient_sha256", "train_patient_sha256"):
            if declaration.get(key) != expected_identity[key]:
                raise FormalGraphAuthorityError(
                    f"Fold {outer_fold} {key} differs from frozen outer-train identities"
                )
    expression_identity = _read_table(
        expression_path, columns=["cancer_id", "sample_id", "patient_id"]
    )
    observed_identity = {
        "train_sample_patient_sha256": _identity_hash(
            expression_identity, ["cancer_id", "sample_id", "patient_id"]
        ),
        "train_patient_sha256": _identity_hash(
            expression_identity, ["cancer_id", "patient_id"]
        ),
    }
    if any(observed_identity[key] != expected_identity[key] for key in observed_identity):
        raise FormalGraphAuthorityError(
            f"Fold {outer_fold} expression includes non-train, missing, or relabelled identities"
        )

    detection = _read_table(inputs.static_paths["detection"])
    coexpression = _read_table(coexpression_path)
    membership = _read_table(inputs.static_paths["signed_membership"])
    hierarchy = _read_table(inputs.static_paths["pathway_hierarchy"])
    binding = _read_table(inputs.static_paths["lnc_protein_binding"])
    protein_gene = _read_table(inputs.static_paths["protein_gene_encoding"])
    ppi = _read_table(inputs.static_paths["ppi"])
    candidate_keys = candidate_pairs[["cancer_id", "lncrna_id"]].drop_duplicates()
    coexpression = coexpression.merge(
        candidate_keys, on=["cancer_id", "lncrna_id"], how="inner", validate="many_to_one"
    )
    pathways = sorted(candidate_pairs.pathway_id.astype(str).unique())
    hierarchy = hierarchy.loc[hierarchy.pathway_id.astype(str).isin(pathways)].copy()
    missing_hierarchy = sorted(set(pathways) - set(hierarchy.pathway_id.astype(str)))
    if missing_hierarchy:
        raise FormalGraphAuthorityError(
            f"Static hierarchy lacks candidate exact pathways: {missing_hierarchy[:10]}"
        )
    relations = [
        materialize_expressed_in(detection, candidate_pairs),
        materialize_signed_coexpression(coexpression, outer_fold=int(outer_fold)),
        materialize_signed_membership(membership, candidate_pathways=pathways),
        materialize_pathway_hierarchy(hierarchy),
        materialize_global_lnc_protein_binding(
            binding, candidate_lncrnas=candidate_pairs.lncrna_id.astype(str).unique()
        ),
        materialize_protein_gene_encoding(protein_gene),
        materialize_symmetric_ppi(ppi),
    ]
    required_nodes = pd.concat(
        [
            candidate_pairs[["lncrna_id"]].drop_duplicates().rename(
                columns={"lncrna_id": "canonical_id"}
            ).assign(node_type="lncRNA"),
            candidate_pairs[["pathway_id"]].drop_duplicates().rename(
                columns={"pathway_id": "canonical_id"}
            ).assign(node_type="pathway"),
            candidate_pairs[["cancer_id"]].drop_duplicates().rename(
                columns={"cancer_id": "canonical_id"}
            ).assign(node_type="cancer"),
        ],
        ignore_index=True,
    )[["node_type", "canonical_id"]]
    authority = build_formal_graph_authority(
        relations, outer_fold=int(outer_fold), required_nodes=required_nodes
    )
    fold_declarations = inputs.receipt["fold_artifacts"][str(int(outer_fold))]
    binding = {
        "format": GRAPH_PAYLOAD_BINDING_FORMAT,
        "status": GRAPH_PAYLOAD_BINDING_STATUS,
        "outer_fold": int(outer_fold),
        "receipt": {
            "path": str(inputs.receipt_path),
            "sha256": inputs.receipt_sha256,
        },
        "patient_fold_authority": {
            "manifest_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "static_artifacts": {
            key: {"path": str(inputs.static_paths[key]), "sha256": inputs.static_sha256[key]}
            for key in STATIC_ARTIFACT_IDS
        },
        "fold_artifacts": {
            "train_expression": {"path": str(expression_path), "sha256": expression_sha},
            "train_coexpression": {"path": str(coexpression_path), "sha256": coexpression_sha},
        },
        "train_identity": expected_identity,
        "graph": {
            "node_sha256": dataframe_sha256(
                authority.nodes, ["node_type", "canonical_id"]
            ),
            "edge_sha256": authority.safe_graph.manifest["edge_sha256"],
            "node_counts": authority.manifest["node_counts"],
            "variant_edge_counts": authority.manifest["variant_edge_counts"],
            "same_relation_schema_all_variants": authority.manifest[
                "same_relation_schema_all_variants"
            ],
        },
        "gates": {
            "frozen_patient_first_authority": True,
            "outer_train_expression_only": True,
            "outer_train_coexpression_only": True,
            "static_evidence_outcome_free": True,
            "historical_graph_rows_used": False,
            "legacy_graph_root_fallback_used": False,
            "toy_or_synthetic_fallback_used": False,
            "same_node_and_relation_schema_all_variants": True,
            "all_required_fold_declarations_present": set(fold_declarations)
            >= set(FOLD_ARTIFACT_IDS),
        },
    }
    return BoundFormalGraph(authority=authority, binding=binding)


def validate_formal_graph_payload_binding(
    binding: Mapping[str, Any], *, outer_fold: int, variant: str, bundle: Any | None = None
) -> dict[str, Any]:
    """Reject legacy prepared folds before a trainer can consume them."""

    if not isinstance(binding, Mapping):
        raise FormalGraphAuthorityError("Prepared fold graph authority binding failed")
    arm = str(variant).upper()
    gates = binding.get("gates", {})
    graph = binding.get("graph", {})
    patient = binding.get("patient_fold_authority", {})
    receipt = binding.get("receipt", {})
    static_artifacts = binding.get("static_artifacts", {})
    fold_artifacts = binding.get("fold_artifacts", {})
    provenance_records = [
        *(static_artifacts.values() if isinstance(static_artifacts, Mapping) else []),
        *(fold_artifacts.values() if isinstance(fold_artifacts, Mapping) else []),
    ]
    provenance_complete = bool(provenance_records) and all(
        isinstance(record, Mapping)
        and bool(str(record.get("path", "")).strip())
        and bool(_SHA256.fullmatch(str(record.get("sha256", ""))))
        for record in provenance_records
    )
    required_gates = (
        "frozen_patient_first_authority",
        "outer_train_expression_only",
        "outer_train_coexpression_only",
        "static_evidence_outcome_free",
        "same_node_and_relation_schema_all_variants",
        "all_required_fold_declarations_present",
    )
    if (
        binding.get("format") != GRAPH_PAYLOAD_BINDING_FORMAT
        or binding.get("status") != GRAPH_PAYLOAD_BINDING_STATUS
        or int(binding.get("outer_fold", -1)) != int(outer_fold)
        or arm not in {"G0", "G1", "G2"}
        or binding.get("variant") != arm
        or patient.get("manifest_sha256") != FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
        or patient.get("receipt_sha256") != FROZEN_V32_RECEIPT_SHA256
        or not _SHA256.fullmatch(str(receipt.get("sha256", "")))
        or not _SHA256.fullmatch(str(graph.get("node_sha256", "")))
        or not _SHA256.fullmatch(str(graph.get("edge_sha256", "")))
        or not _SHA256.fullmatch(str(graph.get("master_edge_sha256", "")))
        or int(graph.get("edge_count", -1)) < 0
        or graph.get("same_relation_schema_all_variants") is not True
        or any(gates.get(key) is not True for key in required_gates)
        or gates.get("historical_graph_rows_used") is not False
        or gates.get("legacy_graph_root_fallback_used") is not False
        or gates.get("toy_or_synthetic_fallback_used") is not False
        or not isinstance(static_artifacts, Mapping)
        or not isinstance(fold_artifacts, Mapping)
        or set(static_artifacts) != set(STATIC_ARTIFACT_IDS)
        or set(fold_artifacts) != set(FOLD_ARTIFACT_IDS)
        or not provenance_complete
    ):
        raise FormalGraphAuthorityError("Prepared fold graph authority binding failed")
    if bundle is not None:
        nodes = getattr(bundle, "nodes", None)
        edges = getattr(bundle, "edges", None)
        if not isinstance(nodes, pd.DataFrame) or not isinstance(edges, pd.DataFrame):
            raise FormalGraphAuthorityError("Prepared graph bundle lacks typed node/edge tables")
        observed_node_sha = dataframe_sha256(nodes, ["node_type", "canonical_id"])
        observed_edge_sha = dataframe_sha256(edges, list(EDGE_KEYS))
        if (
            observed_node_sha != graph.get("node_sha256")
            or observed_edge_sha != graph.get("edge_sha256")
            or len(edges) != int(graph.get("edge_count", -1))
        ):
            raise FormalGraphAuthorityError(
                "Prepared graph bundle does not match its declared G0/G1/G2 arm"
            )
    return {
        "status": "PASS_FRESH_G012_PAYLOAD_BINDING",
        "outer_fold": int(outer_fold),
        "variant": arm,
        "receipt_sha256": str(receipt["sha256"]),
        "edge_sha256": str(graph["edge_sha256"]),
    }


def bind_formal_graph_variant(
    binding: Mapping[str, Any], authority: FormalGraphAuthority, variant: str
) -> dict[str, Any]:
    """Bind one concrete masked edge table, preventing arm substitution."""

    arm = str(variant).upper()
    if arm not in {"G0", "G1", "G2"}:
        raise FormalGraphAuthorityError(f"Unknown formal graph variant: {variant}")
    edges = variant_edges(authority, arm)
    graph = dict(binding.get("graph", {}))
    master_sha = str(graph.get("edge_sha256", ""))
    if not _SHA256.fullmatch(master_sha):
        raise FormalGraphAuthorityError("Master G2 graph SHA256 is absent")
    graph.update(
        {
            "master_edge_sha256": master_sha,
            "edge_sha256": dataframe_sha256(edges, list(EDGE_KEYS)),
            "edge_count": int(len(edges)),
        }
    )
    return {**dict(binding), "variant": arm, "graph": graph}


__all__ = [
    "BoundFormalGraph",
    "FOLD_ARTIFACT_IDS",
    "FormalGraphAuthorityError",
    "FormalGraphInputAuthority",
    "GRAPH_INPUT_RECEIPT_FORMAT",
    "GRAPH_INPUT_RECEIPT_STATUS",
    "GRAPH_PAYLOAD_BINDING_FORMAT",
    "GRAPH_PAYLOAD_BINDING_STATUS",
    "STATIC_ARTIFACT_IDS",
    "build_bound_formal_graph",
    "bind_formal_graph_variant",
    "load_formal_graph_input_authority",
    "read_formal_graph_table",
    "train_identity_hashes",
    "validate_formal_graph_payload_binding",
]
