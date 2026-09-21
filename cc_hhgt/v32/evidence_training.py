"""Fresh V3.2 private evidence/physical-interaction training pipeline.

This module is intentionally separate from the exact-pathway ranking model.
It builds event bags only from raw evidence facts plus static exact-pathway
membership, freezes and detaches the matching-fold V3.2 core embeddings, and
trains five new private attention heads.  It never reads a historical Evidence
checkpoint, a family SPF, ``pair_evidence`` labels/weights/scores, an old
confidence value, or a primary ranking value.

The public output is auxiliary: confidence, direction, uncertainty and
availability.  Physical interaction facts and per-event lineage are written as
separate tables and are never represented as model predictions.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .rbp_assay import classify_assay


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
CURRENT_G2_CORE_LINEAGE_FORMAT = (
    "CANCERLNCATLAS_V32_CURRENT_G2_CORE_EMBEDDING_LINEAGE_V1"
)
PRIVATE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_PRIVATE_EVIDENCE_EVENTSET_V1"
EXACT_KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
N_FOLDS = 5
UNKNOWN = "unknown"
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_CANCER_COUNT = 33
FORMAL_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
FORMAL_SPLIT_FAILURE_REASON = (
    "STRICT_CONNECTED_COMPONENT_ISOLATION_NOT_TRAINABLE"
)
EVIDENCE_SPLIT_POLICY = (
    "LNCRNA_EXACT_PATHWAY_PAIR_BLOCKED_5FOLD__"
    "EVAL_PMID_AND_SOURCE_EVENT_EXCLUDED_FROM_TRAIN_V1"
)
EVALUATION_PROVENANCE_EXCLUSION_POLICY = (
    "FOR_EACH_OUTER_HEAD_REMOVE_TRAIN_EVENTS_SHARING_PMID_OR_CANONICAL_"
    "SOURCE_EVENT_WITH_VALIDATION_OR_HELDOUT"
)

# Normalization removes punctuation/underscores before this set is consulted.
# The ban applies to evidence/relation model inputs, not to static identifiers.
FORBIDDEN_RAW_COLUMNS = {
    "label",
    "sampleweight",
    "score",
    "confidence",
    "confidencescore",
    "oldconfidence",
    "pairscore",
    "pairevidencelabel",
    "pairevidencescore",
    "neural_evidence_probability".replace("_", ""),
    "evidence_confidence_probability".replace("_", ""),
    "familyspf",
    "pathwayfamilyspf",
}

LEGACY_EVENT_FEATURE_FIELDS = (
    "route_type",
    "source_database",
    "source_dataset",
    "experiment_type",
    "relation_type",
    "tissue",
    "cell_line",
    "species",
    "member_type",
)

#: Assay-aware field set.  ``experiment_type`` is deliberately replaced by the
#: normalised ``experiment_family`` plus the fine-grained ``assay_subtype`` so
#: that eCLIP, RIP, ChIRP and the remaining physical assays stop sharing one
#: hashed token.  ``experiment_raw`` is intentionally *not* a feature: it is
#: free text and would create unbounded, spelling-dependent buckets.  It is
#: retained for lineage only (see ``experiment_raw`` in the event frame).
EVENT_FEATURE_FIELDS = (
    "route_type",
    "source_database",
    "source_dataset",
    "experiment_family",
    "assay_subtype",
    "relation_type",
    "tissue",
    "cell_line",
    "species",
    "member_type",
)


def event_feature_fields(*, preserve_assay_type: bool) -> tuple[str, ...]:
    """Return the event feature field list for the requested ablation mode.

    ``preserve_assay_type=False`` reproduces the historical field list exactly,
    which is required for ablation mode A (``legacy_generic_binding``) to be
    provably equivalent to the pre-change model.  Because
    ``_event_feature_matrix`` hashes ``f"{field}={value}"``, renaming a field
    changes its bucket even when the value is unchanged, so the two lists must
    remain genuinely distinct rather than being merged.
    """

    return EVENT_FEATURE_FIELDS if preserve_assay_type else LEGACY_EVENT_FEATURE_FIELDS


class EvidenceTrainingContractError(RuntimeError):
    """Raised when the evidence-private-layer contract is violated."""


@dataclass(frozen=True)
class EvidenceBuildResult:
    """Canonical exact events and their non-predictive companion tables."""

    events: pd.DataFrame
    physical_facts: pd.DataFrame
    lineage: pd.DataFrame
    rejected: pd.DataFrame


@dataclass(frozen=True)
class CoreEmbeddingLineage:
    patient_fold: int
    manifest_path: Path
    manifest_sha256: str
    checkpoint_sha256: str
    core_parameter_sha256: str
    export_paths: Mapping[str, Path]
    export_sha256: Mapping[str, str]
    formal_graph_variant: str | None = None
    graph_authority_receipt_sha256: str | None = None
    patient_fold_authority_receipt_sha256: str | None = None


@dataclass(frozen=True)
class CoreFeatureBundle:
    lineage: CoreEmbeddingLineage
    feature_maps: Mapping[str, Mapping[str, np.ndarray]]
    feature_dims: Mapping[str, int]

    @property
    def combined_dim(self) -> int:
        return sum(self.feature_dims[node_type] for node_type in ("cancer", "lncRNA", "pathway"))

    def vector_for(self, cancer_id: object, lncrna_id: object, pathway_id: object) -> np.ndarray | None:
        values: list[np.ndarray] = []
        for node_type, node_id in (
            ("cancer", cancer_id),
            ("lncRNA", lncrna_id),
            ("pathway", pathway_id),
        ):
            mapping = self.feature_maps[node_type]
            found = next((mapping[key] for key in _node_aliases(node_type, node_id) if key in mapping), None)
            if found is None:
                return None
            values.append(found)
        return np.concatenate(values).astype(np.float32, copy=False)


@dataclass(frozen=True)
class BagExample:
    key: tuple[str, str, str]
    event_features: np.ndarray
    core_features: np.ndarray
    confidence_target: float
    direction_target: int
    leakage_fold: int
    event_count: int


@dataclass
class PrivateHeadFit:
    model: Any
    initial_parameter_sha256: str
    final_parameter_sha256: str
    history: list[dict[str, float]]
    optimizer_steps: int
    confidence_supervision_available: bool
    direction_supervision_available: bool


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _clean(value: object, default: str = "") -> str:
    if value is None or (not isinstance(value, (list, tuple, set, dict)) and pd.isna(value)):
        return default
    text = str(value).strip()
    return default if text.lower() in {"", "nan", "none", "null", "na", "n/a", "-"} else text


def _stable_sha256(*values: object) -> str:
    payload = "\x1f".join(_clean(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _column(frame: pd.DataFrame, aliases: Sequence[str], *, required: bool = False) -> str | None:
    lookup = {_normalized_name(column): str(column) for column in frame.columns}
    for alias in aliases:
        if _normalized_name(alias) in lookup:
            return lookup[_normalized_name(alias)]
    if required:
        raise EvidenceTrainingContractError(f"Missing required column; accepted aliases={list(aliases)}")
    return None


def _row_value(row: pd.Series, column: str | None, default: object = "") -> object:
    return default if column is None else row[column]


def _truth(value: object) -> bool | None:
    text = _clean(value).lower()
    if text in {"1", "true", "t", "yes", "y", "pass", "positive"}:
        return True
    if text in {"0", "false", "f", "no", "n", "fail", "negative"}:
        return False
    return None


def _identity(value: object, id_map: Mapping[str, str] | None = None) -> str:
    text = _clean(value).upper()
    if not text:
        return ""
    aliases = {text, re.sub(r"^(?:LNC|GENE|PROTEIN|PATHWAY|CANCER):", "", text)}
    if id_map:
        for alias in aliases:
            mapped = id_map.get(alias) or id_map.get(_normalized_name(alias))
            if mapped:
                text = _clean(mapped).upper()
                break
    text = re.sub(r"^(?:LNC|GENE|PROTEIN|PATHWAY|CANCER):", "", text)
    ensembl_match = re.search(r"ENSG\d+(?:\.\d+)?", text)
    if ensembl_match:
        text = ensembl_match.group(0)
    if re.fullmatch(r"ENSG\d+\.\d+", text):
        text = text.split(".", 1)[0]
    return text


def _cancer(value: object) -> str:
    text = _clean(value).upper()
    if not text or text in {"PAN", "PANCAN", "PAN-CANCER", "PAN_CANCER", "ALL"}:
        return "PAN_CANCER"
    return re.sub(r"^(?:CANCER):", "", text)


def _tokens(value: object) -> tuple[str, ...]:
    text = _clean(value)
    if not text:
        return ()
    return tuple(sorted({token.strip() for token in re.split(r"[;,|\s]+", text) if token.strip()}))


def assert_label_blind_raw_input(frame: pd.DataFrame, input_name: str) -> None:
    """Reject learned/legacy evidence inputs before any transformation."""

    if "pairevidence" in _normalized_name(Path(str(input_name)).name):
        raise EvidenceTrainingContractError("pair_evidence is forbidden for V3.2 private evidence training")
    forbidden = sorted(
        str(column)
        for column in frame.columns
        if _normalized_name(column) in FORBIDDEN_RAW_COLUMNS
        or "familyspf" in _normalized_name(column)
        or "oldconfidence" in _normalized_name(column)
    )
    if forbidden:
        raise EvidenceTrainingContractError(
            f"{input_name} contains forbidden learned/legacy columns: {forbidden}"
        )


def normalize_id_map(frame: pd.DataFrame | None) -> dict[str, str]:
    if frame is None or frame.empty:
        return {}
    source_col = _column(
        frame,
        [
            "source_id", "input_id", "raw_id", "alias", "protein_id", "uniprot_id",
            "entry",
        ],
        required=True,
    )
    target_col = _column(
        frame,
        [
            "target_id", "canonical_id", "gene_id", "ensembl_gene_id", "node_id",
            "ensembl",
        ],
        required=True,
    )
    result: dict[str, str] = {}
    for source, target in frame[[source_col, target_col]].itertuples(index=False, name=None):
        normalized_source = _identity(source)
        normalized_target = _identity(target)
        if normalized_source and normalized_target:
            result[normalized_source] = normalized_target
            result[_normalized_name(normalized_source)] = normalized_target
    return result


def normalize_exact_pathway_members(
    frame: pd.DataFrame, id_map: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Normalize a static member table without consulting pathway families."""

    pathway_col = _column(frame, ["pathway_id", "exact_pathway_id"])
    if pathway_col is None:
        family_col = _column(frame, ["pathway_family_id", "family_id"])
        if family_col is not None:
            raise EvidenceTrainingContractError(
                "A pathway-family member table cannot be broadcast to exact pathways"
            )
        raise EvidenceTrainingContractError("Static membership lacks exact pathway_id")
    member_col = _column(
        frame,
        ["member_id", "gene_id", "partner_id", "entity_id", "node_id", "gene_symbol"],
        required=True,
    )
    member_type_col = _column(frame, ["member_type", "partner_type", "entity_type"])
    status_col = _column(frame, ["mapping_status"])
    value = pd.DataFrame(
        {
            "pathway_id": frame[pathway_col].map(_identity),
            "member_id": frame[member_col].map(lambda item: _identity(item, id_map)),
            "member_type": (
                frame[member_type_col].map(lambda item: _clean(item, "gene"))
                if member_type_col
                else "gene"
            ),
            "mapping_status": (
                frame[status_col].map(lambda item: _clean(item, "mapped").lower())
                if status_col
                else "mapped"
            ),
            "static_member_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    value = value.loc[
        value.pathway_id.ne("")
        & value.member_id.ne("")
        & ~value.mapping_status.isin({"unmapped", "rejected", "invalid"})
    ].copy()
    value["static_member_id"] = [
        "MEM:" + _stable_sha256(pathway, member)[:24]
        for pathway, member in value[["pathway_id", "member_id"]].itertuples(index=False, name=None)
    ]
    return value.drop_duplicates(["pathway_id", "member_id"]).reset_index(drop=True)


def _quality_target(row: pd.Series, columns: Mapping[str, str | None], pmids: tuple[str, ...]) -> float:
    review = _clean(_row_value(row, columns.get("manual_review_status"))).lower()
    if re.search(r"accepted|validated|confirmed|curated|approved|pass", review):
        return 1.0
    if re.search(r"rejected|refuted|invalid|failed|false", review):
        return 0.0
    eligible = _truth(_row_value(row, columns.get("predictive_core_eligible")))
    if eligible is not None:
        return float(eligible)
    tier = _clean(_row_value(row, columns.get("evidence_tier"))).lower()
    if re.search(r"quarantine|rejected|unsupported|computational.only|invalid", tier):
        return 0.0
    if re.search(r"\ba1\b|\ba2\b|validated|high.confidence|causal.direct|physical.support", tier):
        return 1.0
    experimental = _truth(_row_value(row, columns.get("experimental")))
    computational = _truth(_row_value(row, columns.get("computational")))
    if experimental is True and pmids:
        return 1.0
    if experimental is False and computational is True:
        return 0.0
    return math.nan


def _direction_target(value: object, effect_sign: object = "") -> int:
    text = f"{_clean(value)} {_clean(effect_sign)}".lower()
    if re.search(r"negative|down|inhibit|repress|decreas|(?:^|\s)-1(?:\s|$)", text):
        return 0
    if re.search(r"positive|up|activat|stimulat|increas|(?:^|\s)\+?1(?:\s|$)", text):
        return 2
    if re.search(r"neutral|no.effect|unchanged", text):
        return 1
    return -1


def _is_physical(row: pd.Series, columns: Mapping[str, str | None]) -> bool:
    explicit = _truth(_row_value(row, columns.get("physical")))
    if explicit is not None:
        return explicit
    text = " ".join(
        _clean(_row_value(row, columns.get(name)))
        for name in ("relation_type", "experiment_type", "interaction_type", "directness")
    ).lower()
    return bool(re.search(r"physical|binding|pull.?down|\brip\b|\bclip\b|chirp|co.?ip", text))


def _source_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    return {
        "lncrna_id": _column(frame, ["lncrna_id", "lncrna", "lnc_id", "rna_id"]),
        "partner_id": _column(frame, ["partner_id", "target_id", "gene_id"]),
        "pathway_id": _column(frame, ["pathway_id", "exact_pathway_id"]),
        "pathway_family_id": _column(frame, ["pathway_family_id", "family_id"]),
        "cancer_id": _column(frame, ["cancer_id", "tcga_code", "cancer", "disease"]),
        "entity_a": _column(frame, ["entity_a", "source_node_id"]),
        "entity_b": _column(frame, ["entity_b", "target_node_id"]),
        "entity_a_type": _column(frame, ["entity_a_type", "source_type"]),
        "entity_b_type": _column(frame, ["entity_b_type", "target_type"]),
        "raw_event_id": _column(
            frame,
            ["evidence_event_id", "event_id", "interaction_row_id", "interaction_id", "source_record_id"],
        ),
        "source_record_id": _column(frame, ["source_record_id", "interaction_id", "record_id"]),
        "source_database": _column(frame, ["source_database", "source", "database"]),
        "source_dataset": _column(
            frame,
            ["source_dataset", "dataset_id", "dataset", "source_resource", "source_version"],
        ),
        "pmid": _column(frame, ["pmid", "pubmed_id", "independent_pmid"]),
        "experiment_raw": _column(
            frame,
            ["experiment_raw", "experiment", "experiment_method", "methods", "assay_detail"],
        ),
        "experiment_family": _column(frame, ["experiment_family"]),
        "assay_subtype": _column(frame, ["assay_subtype"]),
        "graph_assay_class": _column(frame, ["graph_assay_class"]),
        "experiment_type": _column(
            frame,
            ["experiment_type", "experiment_family", "assay_type", "experimental_system"],
        ),
        "relation_type": _column(frame, ["relation_type", "interaction_class", "assertion_type"]),
        "interaction_type": _column(frame, ["interaction_type", "experimental_system_type"]),
        "direction": _column(frame, ["direction", "regulation_direction"]),
        "effect_sign": _column(frame, ["effect_sign"]),
        "tissue": _column(frame, ["tissue"]),
        "cell_line": _column(frame, ["cell_line", "cellline"]),
        "species": _column(frame, ["species", "organism"]),
        "manual_review_status": _column(frame, ["manual_review_status", "review_status"]),
        "evidence_tier": _column(frame, ["evidence_tier", "evidence_level"]),
        "predictive_core_eligible": _column(frame, ["predictive_core_eligible"]),
        "experimental": _column(frame, ["experimental", "is_experimental"]),
        "computational": _column(frame, ["computational", "is_predicted"]),
        "physical": _column(frame, ["physical", "is_physical", "physical_interaction"]),
        "directness": _column(frame, ["directness"]),
        "source_row_sha256": _column(frame, ["source_row_sha256"]),
        "source_sha256": _column(frame, ["source_sha256"]),
    }


def _normalize_source_rows(
    frame: pd.DataFrame,
    *,
    source_kind: str,
    input_name: str,
    id_map: Mapping[str, str],
) -> Iterable[dict[str, Any]]:
    assert_label_blind_raw_input(frame, input_name)
    columns = _source_columns(frame)
    for source_row_index, row in frame.iterrows():
        entity_a = _row_value(row, columns["entity_a"])
        entity_b = _row_value(row, columns["entity_b"])
        entity_a_type = _clean(_row_value(row, columns["entity_a_type"])).lower()
        entity_b_type = _clean(_row_value(row, columns["entity_b_type"])).lower()
        lnc = _identity(_row_value(row, columns["lncrna_id"]), id_map)
        partner = _identity(_row_value(row, columns["partner_id"]), id_map)
        exact_pathway = _identity(_row_value(row, columns["pathway_id"]))
        if not lnc:
            if "lnc" in entity_a_type or _clean(entity_a).upper().startswith("LNC:"):
                lnc = _identity(entity_a, id_map)
                if not partner and "pathway" not in entity_b_type:
                    partner = _identity(entity_b, id_map)
            elif "lnc" in entity_b_type or _clean(entity_b).upper().startswith("LNC:"):
                lnc = _identity(entity_b, id_map)
                if not partner and "pathway" not in entity_a_type:
                    partner = _identity(entity_a, id_map)
        if not exact_pathway:
            if "pathway" in entity_b_type or _clean(entity_b).upper().startswith("PATHWAY:"):
                exact_pathway = _identity(entity_b)
            elif "pathway" in entity_a_type or _clean(entity_a).upper().startswith("PATHWAY:"):
                exact_pathway = _identity(entity_a)
        if not partner and not exact_pathway:
            if entity_b_type and "pathway" not in entity_b_type and "lnc" not in entity_b_type:
                partner = _identity(entity_b, id_map)
        pmids = _tokens(_row_value(row, columns["pmid"]))
        source_database = _clean(_row_value(row, columns["source_database"]), source_kind)
        source_dataset = _clean(_row_value(row, columns["source_dataset"]), source_database)
        raw_event_id = _clean(
            _row_value(row, columns["raw_event_id"]),
            _stable_sha256(source_kind, source_row_index)[:24],
        )
        source_record_id = _clean(
            _row_value(row, columns["source_record_id"]), raw_event_id
        )
        raw_row_hash = _clean(_row_value(row, columns["source_row_sha256"]))
        if not raw_row_hash:
            raw_row_hash = hashlib.sha256(
                json.dumps(
                    {str(key): _clean(value) for key, value in row.to_dict().items()},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
        # Assay taxonomy.  ``experiment_raw`` is authoritative when present;
        # otherwise the coarse value already resolved by the legacy path is
        # used so that pre-existing frames still classify sensibly.  Explicit
        # computational flags win over any textual match, so a prediction can
        # never be relabelled as an experimental observation.
        experiment_raw_value = _clean(_row_value(row, columns["experiment_raw"]))
        assay = classify_assay(
            experiment_raw_value or _clean(_row_value(row, columns["experiment_type"])),
            is_predicted=_row_value(row, columns["computational"]),
            is_experimental=_row_value(row, columns["experimental"]),
        )
        yield (
            {
                "source_kind": source_kind,
                "source_input": str(input_name),
                "source_row_index": int(source_row_index) if isinstance(source_row_index, (int, np.integer)) else str(source_row_index),
                "source_row_sha256": raw_row_hash,
                "source_sha256": _clean(_row_value(row, columns["source_sha256"])),
                "raw_event_id": raw_event_id,
                "source_record_id": source_record_id,
                "cancer_id": _cancer(_row_value(row, columns["cancer_id"])),
                "lncrna_id": lnc,
                "partner_id": partner,
                "pathway_id": exact_pathway,
                "pathway_family_id": _clean(_row_value(row, columns["pathway_family_id"])),
                "source_database": source_database,
                "source_dataset": source_dataset,
                "pmid": ";".join(pmids),
                "pmid_tokens": pmids,
                "experiment_type": _clean(_row_value(row, columns["experiment_type"]), UNKNOWN),
                "experiment_raw": experiment_raw_value,
                "experiment_family": assay.experiment_family,
                "assay_subtype": assay.assay_subtype,
                "graph_assay_class": assay.graph_assay_class,
                "relation_type": _clean(_row_value(row, columns["relation_type"]), UNKNOWN),
                "direction_raw": _clean(_row_value(row, columns["direction"]), UNKNOWN),
                "direction_target": _direction_target(
                    _row_value(row, columns["direction"]),
                    _row_value(row, columns["effect_sign"]),
                ),
                "tissue": _clean(_row_value(row, columns["tissue"]), UNKNOWN),
                "cell_line": _clean(_row_value(row, columns["cell_line"]), UNKNOWN),
                "species": _clean(_row_value(row, columns["species"]), UNKNOWN),
                "evidence_tier": _clean(_row_value(row, columns["evidence_tier"]), UNKNOWN),
                "confidence_target": _quality_target(row, columns, pmids),
                "is_experimental": bool(_truth(_row_value(row, columns["experimental"])) or False),
                "is_computational": bool(_truth(_row_value(row, columns["computational"])) or False),
                "is_physical": _is_physical(row, columns),
                "family_only": bool(_clean(_row_value(row, columns["pathway_family_id"]))) and not exact_pathway,
            }
        )
def build_exact_event_bags(
    evidence_event: pd.DataFrame,
    interaction_relation: pd.DataFrame,
    pathway_members: pd.DataFrame,
    *,
    id_map: pd.DataFrame | Mapping[str, str] | None = None,
    candidate_universe: pd.DataFrame | None = None,
    evidence_input_name: str = "evidence_event",
    interaction_input_name: str = "interaction_relation",
) -> EvidenceBuildResult:
    """Build deduplicated cancer×lncRNA×exact-pathway event facts.

    Exact assignments are allowed only when an event names an exact pathway or
    its partner maps through the static exact member table.  A family ID alone
    is recorded as rejected and is never expanded.
    """

    identifier_map = (
        normalize_id_map(id_map)
        if isinstance(id_map, pd.DataFrame)
        else dict(id_map or {})
    )
    members = normalize_exact_pathway_members(pathway_members, identifier_map)
    member_lookup: dict[str, list[dict[str, Any]]] = {}
    for row in members.to_dict("records"):
        member_lookup.setdefault(str(row["member_id"]), []).append(row)

    allowed_local: set[tuple[str, str, str]] | None = None
    allowed_global: set[tuple[str, str]] | None = None
    if candidate_universe is not None:
        normalized_universe = normalize_candidates(candidate_universe)
        allowed_local = set(
            normalized_universe[EXACT_KEYS].itertuples(index=False, name=None)
        )
        allowed_global = set(
            normalized_universe[["lncrna_id", "pathway_id"]].itertuples(index=False, name=None)
        )
    normalized = itertools.chain(
        _normalize_source_rows(
            evidence_event,
            source_kind="evidence_event",
            input_name=evidence_input_name,
            id_map=identifier_map,
        ),
        _normalize_source_rows(
            interaction_relation,
            source_kind="interaction_relation",
            input_name=interaction_input_name,
            id_map=identifier_map,
        ),
    )

    event_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    physical_rows: list[dict[str, Any]] = []
    for raw in normalized:
        if raw["is_physical"] and raw["lncrna_id"] and raw["partner_id"]:
            fact_signature = _stable_sha256(
                raw["source_database"],
                raw["source_dataset"],
                raw["source_record_id"],
                raw["pmid"],
                raw["lncrna_id"],
                raw["partner_id"],
                raw["relation_type"],
                raw["experiment_type"],
            )
            physical_rows.append(
                {
                    "physical_fact_id": "PHYS:" + fact_signature[:24],
                    "cancer_id": raw["cancer_id"],
                    "lncrna_id": raw["lncrna_id"],
                    "partner_id": raw["partner_id"],
                    "relation_type": raw["relation_type"],
                    "experiment_type": raw["experiment_type"],
                    "experiment_raw": raw["experiment_raw"],
                    "experiment_family": raw["experiment_family"],
                    "assay_subtype": raw["assay_subtype"],
                    "graph_assay_class": raw["graph_assay_class"],
                    "source_database": raw["source_database"],
                    "source_dataset": raw["source_dataset"],
                    "source_record_id": raw["source_record_id"],
                    "pmid": raw["pmid"],
                    "source_row_sha256": raw["source_row_sha256"],
                    "is_prediction": False,
                }
            )

        if not raw["lncrna_id"]:
            rejected_rows.append({**raw, "rejection_reason": "LNC_ID_UNMAPPED"})
            continue
        routes: dict[str, tuple[str, str, str]] = {}
        if raw["pathway_id"]:
            routes[str(raw["pathway_id"])] = ("DIRECT_EXACT_ASSERTION", "", "")
        if raw["partner_id"]:
            for membership in member_lookup.get(str(raw["partner_id"]), []):
                pathway = str(membership["pathway_id"])
                routes.setdefault(
                    pathway,
                    (
                        "PARTNER_EXACT_MEMBER",
                        str(membership["static_member_id"]),
                        str(membership["member_type"]),
                    ),
                )
        unfiltered_route_count = len(routes)
        if allowed_local is not None and allowed_global is not None:
            routes = {
                pathway: route
                for pathway, route in routes.items()
                if (
                    (raw["lncrna_id"], pathway) in allowed_global
                    if raw["cancer_id"] == "PAN_CANCER"
                    else (raw["cancer_id"], raw["lncrna_id"], pathway) in allowed_local
                )
            }
        if not routes:
            reason = (
                "FAMILY_ONLY_NOT_BROADCAST_TO_EXACT"
                if raw["family_only"]
                else "OUTSIDE_EXACT_CANDIDATE_UNIVERSE"
                if unfiltered_route_count > 0 and allowed_local is not None
                else "NO_EXACT_PATHWAY_OR_STATIC_MEMBER_MAPPING"
            )
            rejected_rows.append({**raw, "rejection_reason": reason})
            continue

        for pathway_id, (route_type, static_member_id, member_type) in sorted(routes.items()):
            canonical_signature = _stable_sha256(
                raw["cancer_id"],
                raw["lncrna_id"],
                pathway_id,
                raw["partner_id"],
                raw["source_database"],
                raw["source_dataset"],
                raw["source_record_id"],
                raw["pmid"],
                raw["relation_type"],
                raw["experiment_type"],
                raw["direction_raw"],
            )
            event_id = "EV32:" + canonical_signature[:24]
            physical_fact_id = (
                "PHYS:"
                + _stable_sha256(
                    raw["source_database"],
                    raw["source_dataset"],
                    raw["source_record_id"],
                    raw["pmid"],
                    raw["lncrna_id"],
                    raw["partner_id"],
                    raw["relation_type"],
                    raw["experiment_type"],
                )[:24]
                if raw["is_physical"] and raw["partner_id"]
                else ""
            )
            event_rows.append(
                {
                    "event_id": event_id,
                    "cancer_id": raw["cancer_id"],
                    "lncrna_id": raw["lncrna_id"],
                    "pathway_id": pathway_id,
                    "partner_id": raw["partner_id"],
                    "member_type": member_type or UNKNOWN,
                    "route_type": route_type,
                    "static_member_id": static_member_id,
                    "physical_fact_id": physical_fact_id,
                    "source_database": raw["source_database"],
                    "source_dataset": raw["source_dataset"],
                    "source_record_id": raw["source_record_id"],
                    "pmid": raw["pmid"],
                    "experiment_type": raw["experiment_type"],
                    "experiment_raw": raw["experiment_raw"],
                    "experiment_family": raw["experiment_family"],
                    "assay_subtype": raw["assay_subtype"],
                    "graph_assay_class": raw["graph_assay_class"],
                    "relation_type": raw["relation_type"],
                    "direction_raw": raw["direction_raw"],
                    "direction_target": int(raw["direction_target"]),
                    "confidence_target": float(raw["confidence_target"]),
                    "tissue": raw["tissue"],
                    "cell_line": raw["cell_line"],
                    "species": raw["species"],
                    "is_experimental": bool(raw["is_experimental"]),
                    "is_computational": bool(raw["is_computational"]),
                    "is_physical": bool(raw["is_physical"]),
                    "is_model_prediction": False,
                }
            )
            lineage_rows.append(
                {
                    "lineage_id": "LIN32:"
                    + _stable_sha256(
                        event_id,
                        raw["source_kind"],
                        raw["source_row_index"],
                        raw["source_row_sha256"],
                    )[:24],
                    "event_id": event_id,
                    "raw_event_id": raw["raw_event_id"],
                    "source_kind": raw["source_kind"],
                    "source_input": raw["source_input"],
                    "source_row_index": raw["source_row_index"],
                    "source_row_sha256": raw["source_row_sha256"],
                    "source_sha256": raw["source_sha256"],
                    "mapping_route": route_type,
                    "static_member_id": static_member_id,
                    "family_id_observed_but_unused": raw["pathway_family_id"],
                    "family_broadcast_used": False,
                }
            )

    event_columns = [
        "event_id", *EXACT_KEYS, "partner_id", "member_type", "route_type",
        "static_member_id", "physical_fact_id", "source_database", "source_dataset",
        "source_record_id", "pmid", "experiment_type", "experiment_raw",
        "experiment_family", "assay_subtype", "graph_assay_class", "relation_type",
        "direction_raw", "direction_target", "confidence_target", "tissue", "cell_line",
        "species", "is_experimental", "is_computational", "is_physical",
        "is_model_prediction",
    ]
    events = pd.DataFrame(event_rows, columns=event_columns)
    if not events.empty:
        duplicate_counts = events.groupby("event_id").size().rename("source_occurrence_count")
        events = (
            events.sort_values(["event_id", "route_type"])
            .drop_duplicates("event_id", keep="first")
            .merge(duplicate_counts, on="event_id", how="left", validate="one_to_one")
            .sort_values(EXACT_KEYS + ["event_id"])
            .reset_index(drop=True)
        )
    lineage_columns = [
        "lineage_id", "event_id", "raw_event_id", "source_kind", "source_input",
        "source_row_index", "source_row_sha256", "source_sha256", "mapping_route",
        "static_member_id", "family_id_observed_but_unused", "family_broadcast_used",
    ]
    lineage = (
        pd.DataFrame(lineage_rows, columns=lineage_columns)
        .drop_duplicates("lineage_id")
        .reset_index(drop=True)
    )
    rejected = pd.DataFrame(rejected_rows)
    if rejected.empty:
        rejected = pd.DataFrame(
            columns=["raw_event_id", "source_kind", "source_input", "rejection_reason"]
        )
    physical_columns = [
        "physical_fact_id", "cancer_id", "lncrna_id", "partner_id", "relation_type",
        "experiment_type", "experiment_raw", "experiment_family", "assay_subtype",
        "graph_assay_class", "source_database", "source_dataset", "source_record_id",
        "pmid", "source_row_sha256", "is_prediction",
    ]
    physical = pd.DataFrame(physical_rows, columns=physical_columns)
    if not physical.empty:
        fact_counts = physical.groupby("physical_fact_id").size().rename("source_occurrence_count")
        physical = (
            physical.sort_values("physical_fact_id")
            .drop_duplicates("physical_fact_id")
            .merge(fact_counts, on="physical_fact_id", how="left", validate="one_to_one")
            .reset_index(drop=True)
        )
    return EvidenceBuildResult(events, physical, lineage, rejected)


def normalize_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    cancer_col = _column(frame, ["cancer_id", "tcga_code", "cancer"], required=True)
    lnc_col = _column(frame, ["lncrna_id", "lncrna", "lnc_id"], required=True)
    pathway_col = _column(frame, ["pathway_id", "exact_pathway_id"])
    if pathway_col is None:
        if _column(frame, ["pathway_family_id", "family_id"]):
            raise EvidenceTrainingContractError(
                "Candidate family IDs cannot be broadcast to exact pathways"
            )
        raise EvidenceTrainingContractError("Candidates lack exact pathway_id")
    candidates = pd.DataFrame(
        {
            "cancer_id": frame[cancer_col].map(_cancer),
            "lncrna_id": frame[lnc_col].map(_identity),
            "pathway_id": frame[pathway_col].map(_identity),
            "candidate_order": np.arange(len(frame), dtype=np.int64),
        }
    )
    if candidates[EXACT_KEYS].eq("").any(axis=None):
        raise EvidenceTrainingContractError("Candidates contain empty exact identifiers")
    if candidates.duplicated(EXACT_KEYS).any():
        raise EvidenceTrainingContractError("Candidates must be unique by cancer/lncRNA/exact pathway")
    return candidates


def materialize_candidate_events(events: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach PAN_CANCER events to explicit cancers without any family expansion."""

    normalized_candidates = normalize_candidates(candidates)
    if events.empty:
        return events.copy()
    local = events.loc[events.cancer_id.ne("PAN_CANCER")].merge(
        normalized_candidates[EXACT_KEYS], on=EXACT_KEYS, how="inner", validate="many_to_one"
    )
    global_events = events.loc[events.cancer_id.eq("PAN_CANCER")].drop(columns="cancer_id")
    global_attached = normalized_candidates[EXACT_KEYS].merge(
        global_events, on=["lncrna_id", "pathway_id"], how="inner", validate="many_to_many"
    )
    output = pd.concat([local, global_attached], ignore_index=True, sort=False)
    if output.empty:
        return output
    output["source_event_id"] = output["event_id"].astype(str)
    output["event_id"] = [
        "BAGEV32:" + _stable_sha256(cancer, source_event)[:24]
        for cancer, source_event in output[["cancer_id", "source_event_id"]].itertuples(index=False, name=None)
    ]
    return output.drop_duplicates(EXACT_KEYS + ["event_id"]).reset_index(drop=True)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _biological_pair_token(row: Mapping[str, Any]) -> str:
    """Cancer-agnostic target pair used as the hard evidence split unit.

    Pan-cancer evidence is materialised into more than one cancer context.  Keeping
    the lncRNA--exact-pathway pair together across cancers prevents those copies
    from entering both train and evaluation bags.
    """

    return f"pair:{row['lncrna_id']}|{row['pathway_id']}"


def _provenance_audit_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return provenance identities that are audited but are not split groups."""

    values: list[str] = []
    source = _clean(row.get("source_database")).lower()
    dataset = _clean(row.get("source_dataset")).lower()
    if source not in {"", UNKNOWN}:
        values.append(f"source:{source}")
    if dataset not in {"", UNKNOWN}:
        values.append(f"dataset:{dataset}")
    values.extend(f"pmid:{pmid}" for pmid in _tokens(row.get("pmid")) if pmid.lower() != UNKNOWN)
    return tuple(values)


def _evaluation_provenance_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Identities forbidden from crossing evaluation into a fold's train events."""

    values = [
        f"pmid:{pmid.lower()}"
        for pmid in _tokens(row.get("pmid"))
        if pmid.lower() != UNKNOWN
    ]
    source_event = _clean(row.get("source_event_id"))
    if source_event not in {"", UNKNOWN}:
        values.append(f"source_event:{source_event}")
    source_record = _clean(row.get("source_record_id"))
    if source_record not in {"", UNKNOWN}:
        values.append(
            "source_record:"
            + "|".join(
                [
                    _clean(row.get("source_database"), UNKNOWN).lower(),
                    _clean(row.get("source_dataset"), UNKNOWN).lower(),
                    source_record,
                ]
            )
        )
    return tuple(dict.fromkeys(values))


def _legacy_strict_leakage_tokens(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Historical all-provenance component tokens retained for audit only."""

    return (_biological_pair_token(row), *_provenance_audit_tokens(row))


def assign_leakage_safe_folds(
    events: pd.DataFrame, *, n_folds: int = N_FOLDS, seed: int = 20260726
) -> pd.DataFrame:
    """Assign balanced folds with lncRNA--exact-pathway pairs as hard groups.

    Source databases and datasets are provenance covariates, not biological
    outcome identities.  Making either a hard group collapses a database such as
    RNAInter into one component.  PMID and canonical source-event leakage is
    instead removed from each fold's training event bags by
    :func:`exclude_evaluation_provenance_from_training`.
    """

    if n_folds < 2:
        raise EvidenceTrainingContractError("n_folds must be at least two")
    missing = sorted(set(EXACT_KEYS + ["event_id"]) - set(events.columns))
    if missing:
        raise EvidenceTrainingContractError(f"Events lack leakage-control columns: {missing}")
    output = events.reset_index(drop=True).copy()
    if output.empty:
        output["split_component_id"] = pd.Series(dtype="string")
        output["leakage_fold"] = pd.Series(dtype="int64")
        return output
    pair_sizes = (
        output.groupby(["lncrna_id", "pathway_id"], observed=True, sort=False)
        .size()
        .rename("event_rows")
        .reset_index()
    )
    pair_sizes["pair_token"] = (
        "pair:"
        + pair_sizes["lncrna_id"].astype(str)
        + "|"
        + pair_sizes["pathway_id"].astype(str)
    )
    ordered = pair_sizes.sort_values(
        ["event_rows", "pair_token"], ascending=[False, True], kind="mergesort"
    ).copy()
    ordered["tie_sha256"] = [
        _stable_sha256(seed, token) for token in ordered["pair_token"]
    ]
    ordered = ordered.sort_values(
        ["event_rows", "tie_sha256"], ascending=[False, True], kind="mergesort"
    )
    fold_sizes = [0] * n_folds
    pair_assignment: dict[str, int] = {}
    component_ids: dict[str, str] = {}
    for row in ordered.itertuples(index=False):
        smallest = min(fold_sizes)
        choices = [fold for fold, size in enumerate(fold_sizes) if size == smallest]
        tie_hash = int(str(row.tie_sha256)[:12], 16)
        fold = choices[tie_hash % len(choices)]
        pair_assignment[str(row.pair_token)] = fold
        component_ids[str(row.pair_token)] = (
            "SPLIT32:" + _stable_sha256(seed, row.pair_token)[:24]
        )
        fold_sizes[fold] += int(row.event_rows)
    row_pair_token = (
        "pair:"
        + output["lncrna_id"].astype(str)
        + "|"
        + output["pathway_id"].astype(str)
    )
    output["split_component_id"] = row_pair_token.map(component_ids).astype("string")
    output["leakage_fold"] = row_pair_token.map(pair_assignment).astype("int64")
    assert_pair_isolation(output, n_folds=n_folds)
    return output


def assert_pair_isolation(events: pd.DataFrame, *, n_folds: int = N_FOLDS) -> None:
    """Prove every cancer-agnostic lncRNA--exact-pathway pair owns one fold."""

    if "leakage_fold" not in events:
        raise EvidenceTrainingContractError("Events have no leakage_fold")
    folds = pd.to_numeric(events["leakage_fold"], errors="raise").astype(int)
    invalid = sorted(set(folds) - set(range(n_folds)))
    if invalid:
        raise EvidenceTrainingContractError(f"Invalid leakage folds: {invalid}")
    audit = events.assign(_fold=folds).groupby(
        ["lncrna_id", "pathway_id"], observed=True, sort=False
    )["_fold"].nunique()
    leaked = audit.loc[audit.ne(1)]
    if not leaked.empty:
        raise EvidenceTrainingContractError(
            "lncRNA--exact-pathway pair leakage detected: "
            f"{list(leaked.index[:10])}"
        )


def assert_source_isolation(events: pd.DataFrame, *, n_folds: int = N_FOLDS) -> None:
    """Audit the historical all-provenance isolation rule.

    This deliberately remains strict for backwards-compatible diagnostics, but
    the V3.2 V1 pair-blocked training policy no longer calls it.
    """

    if "leakage_fold" not in events:
        raise EvidenceTrainingContractError("Events have no leakage_fold")
    owners: dict[str, set[int]] = {}
    for row in events.to_dict("records"):
        fold = int(row["leakage_fold"])
        if fold not in range(n_folds):
            raise EvidenceTrainingContractError(f"Invalid leakage fold: {fold}")
        for token in _legacy_strict_leakage_tokens(row):
            owners.setdefault(token, set()).add(fold)
    leaked = sorted(token for token, folds in owners.items() if len(folds) != 1)
    if leaked:
        raise EvidenceTrainingContractError(
            f"PMID/source/dataset/pair leakage detected: {leaked[:10]}"
        )


def build_pair_blocked_split_audit(
    events: pd.DataFrame, *, n_folds: int = N_FOLDS
) -> dict[str, Any]:
    """Materialise hard-isolation proof and soft provenance overlap counts."""

    assert_pair_isolation(events, n_folds=n_folds)
    active_folds = sorted(
        set(pd.to_numeric(events["leakage_fold"], errors="raise").astype(int))
    ) if not events.empty else []
    pair_fold = events[["lncrna_id", "pathway_id", "leakage_fold"]].drop_duplicates()

    def overlap_counts(column: str, *, tokenize: bool = False) -> dict[str, int]:
        if column not in events:
            return {"tokens": 0, "tokens_spanning_folds": 0}
        values = events[[column, "leakage_fold"]].drop_duplicates().copy()
        values[column] = values[column].map(_clean)
        values = values.loc[values[column].str.lower().ne(UNKNOWN) & values[column].ne("")]
        if tokenize:
            values[column] = values[column].map(
                lambda value: tuple(token.lower() for token in _tokens(value))
            )
            values = values.explode(column).dropna().drop_duplicates()
        else:
            values[column] = values[column].str.lower()
        owners = values.groupby(column, observed=True)["leakage_fold"].nunique()
        return {
            "tokens": int(len(owners)),
            "tokens_spanning_folds": int(owners.gt(1).sum()),
        }

    by_type = {
        "source": overlap_counts("source_database"),
        "dataset": overlap_counts("source_dataset"),
        "pmid": overlap_counts("pmid", tokenize=True),
    }

    fold_rows = {
        str(fold): int((events["leakage_fold"] == fold).sum()) for fold in active_folds
    }
    fold_pairs = {
        str(fold): int((pair_fold["leakage_fold"] == fold).sum()) for fold in active_folds
    }
    return {
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "hard_split_unit": "cancer_agnostic_lncrna_exact_pathway_pair",
        "hard_pair_cross_fold_count": 0,
        "hard_pair_count": int(
            pair_fold[["lncrna_id", "pathway_id"]].drop_duplicates().shape[0]
        ),
        "active_folds": active_folds,
        "all_five_folds_populated": len(active_folds) == n_folds,
        "event_rows_by_fold": fold_rows,
        "biological_pairs_by_fold": fold_pairs,
        "audit_only_provenance_overlap": by_type,
        "source_database_and_dataset_are_hard_split_groups": False,
        "pmid_identity_used_as_model_feature": False,
        "pmid_presence_and_count_used_as_model_features": True,
        "evaluation_provenance_exclusion_policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
    }


def exclude_evaluation_provenance_from_training(
    events: pd.DataFrame,
    *,
    evaluation_folds: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove train events sharing PMID/source-event identity with evaluation.

    The validation and held-out rows are kept intact.  Only event rows belonging
    to other folds are filtered, so target-pair membership remains a hard split
    while direct study/event copies cannot enter training.
    """

    if "leakage_fold" not in events:
        raise EvidenceTrainingContractError("Events have no leakage_fold")
    evaluation_fold_set = {int(value) for value in evaluation_folds}
    if not evaluation_fold_set or not evaluation_fold_set.issubset(set(range(N_FOLDS))):
        raise EvidenceTrainingContractError("evaluation_folds must be non-empty fold IDs")
    fold_series = pd.to_numeric(events["leakage_fold"], errors="raise").astype(int)
    evaluation_mask = fold_series.isin(evaluation_fold_set)
    evaluation = events.loc[evaluation_mask]

    def normalized_pmid_tokens(values: pd.Series) -> set[str]:
        unique_values = values.dropna().astype(str).drop_duplicates()
        return {
            token.lower()
            for value in unique_values
            for token in _tokens(value)
            if token.lower() != UNKNOWN
        }

    evaluation_pmids = (
        normalized_pmid_tokens(evaluation["pmid"])
        if "pmid" in evaluation
        else set()
    )
    evaluation_source_events = (
        set(evaluation["source_event_id"].dropna().astype(str)) - {"", UNKNOWN}
        if "source_event_id" in evaluation
        else set()
    )

    def source_record_keys(frame: pd.DataFrame) -> pd.Series:
        if "source_record_id" not in frame:
            return pd.Series("", index=frame.index, dtype="string")
        source = (
            frame["source_database"].fillna(UNKNOWN).astype(str).str.lower()
            if "source_database" in frame
            else pd.Series(UNKNOWN, index=frame.index)
        )
        dataset = (
            frame["source_dataset"].fillna(UNKNOWN).astype(str).str.lower()
            if "source_dataset" in frame
            else pd.Series(UNKNOWN, index=frame.index)
        )
        record = frame["source_record_id"].fillna("").astype(str)
        return (source + "|" + dataset + "|" + record).where(record.ne(""), "")

    evaluation_source_records = set(source_record_keys(evaluation)) - {""}
    evaluation_token_count = (
        len(evaluation_pmids)
        + len(evaluation_source_events)
        + len(evaluation_source_records)
    )
    train = events.loc[~evaluation_mask].copy()
    train_pair_before = train[EXACT_KEYS].drop_duplicates()
    remove = pd.Series(False, index=train.index, dtype=bool)
    if evaluation_pmids and "pmid" in train:
        unique_train_pmid_values = train["pmid"].dropna().astype(str).drop_duplicates()
        bad_pmid_values = {
            value
            for value in unique_train_pmid_values
            if evaluation_pmids.intersection(
                token.lower() for token in _tokens(value)
            )
        }
        remove |= train["pmid"].fillna("").astype(str).isin(bad_pmid_values)
    if evaluation_source_events and "source_event_id" in train:
        remove |= train["source_event_id"].fillna("").astype(str).isin(
            evaluation_source_events
        )
    if evaluation_source_records:
        remove |= source_record_keys(train).isin(evaluation_source_records)
    filtered_train = train.loc[~remove].copy()

    residual_pmid_overlap = (
        evaluation_pmids.intersection(normalized_pmid_tokens(filtered_train["pmid"]))
        if evaluation_pmids and "pmid" in filtered_train
        else set()
    )
    residual_source_event_overlap = (
        evaluation_source_events.intersection(
            set(filtered_train["source_event_id"].dropna().astype(str))
        )
        if evaluation_source_events and "source_event_id" in filtered_train
        else set()
    )
    residual_source_record_overlap = (
        evaluation_source_records.intersection(set(source_record_keys(filtered_train)))
        if evaluation_source_records
        else set()
    )
    residual_overlap_count = (
        len(residual_pmid_overlap)
        + len(residual_source_event_overlap)
        + len(residual_source_record_overlap)
    )
    if residual_overlap_count:
        raise EvidenceTrainingContractError(
            "Evaluation PMID/source-event provenance remained in training"
        )
    output = pd.concat(
        [filtered_train, events.loc[evaluation_mask]], ignore_index=True, sort=False
    )
    train_pair_after = filtered_train[EXACT_KEYS].drop_duplicates()
    return output, {
        "policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
        "evaluation_folds": sorted(evaluation_fold_set),
        "evaluation_provenance_token_count": evaluation_token_count,
        "evaluation_pmid_count": len(evaluation_pmids),
        "evaluation_source_event_count": len(evaluation_source_events),
        "evaluation_source_record_count": len(evaluation_source_records),
        "train_event_rows_before": int(len(train)),
        "train_event_rows_removed": int(remove.sum()),
        "train_event_rows_after": int(len(filtered_train)),
        "train_pairs_before": int(len(train_pair_before)),
        "train_pairs_after": int(len(train_pair_after)),
        "train_pairs_lost_all_events": int(len(train_pair_before) - len(train_pair_after)),
        "evaluation_event_rows_unchanged": int(evaluation_mask.sum()),
        "residual_pmid_overlap_count": len(residual_pmid_overlap),
        "residual_source_event_overlap_count": len(residual_source_event_overlap),
        "residual_source_record_overlap_count": len(residual_source_record_overlap),
        "residual_train_evaluation_provenance_overlap_count": residual_overlap_count,
    }


def _is_sha256(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", _clean(value).lower()))


def _resolve_export_path(root: Path, fold: int, node_type: str, declared: object) -> Path:
    local = root / f"patient_fold={fold}" / f"{node_type}.parquet"
    if local.is_file():
        return local
    declared_path = Path(_clean(declared))
    if declared_path.is_absolute() and declared_path.is_file():
        return declared_path
    for ancestor in [root, *root.parents]:
        candidate = ancestor / declared_path
        if candidate.is_file():
            return candidate
    return local


def validate_core_embedding_lineage(
    core_embedding_root: Path,
    patient_fold: int,
    *,
    verify_files: bool = True,
    require_current_g2_authority: bool = False,
    expected_manifest_sha256: str | None = None,
    expected_graph_authority_receipt_sha256: str | None = None,
) -> CoreEmbeddingLineage:
    """Validate same-generation, freshly trained V3.2 core exports and hashes."""

    if patient_fold not in range(N_FOLDS):
        raise EvidenceTrainingContractError("patient_fold must be 0..4")
    root = Path(core_embedding_root)
    manifest_path = root / "CORE_EMBEDDING_MANIFEST.json"
    if not manifest_path.is_file():
        raise EvidenceTrainingContractError(f"Missing V3.2 core manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_manifest_sha256 = file_sha256(manifest_path)
    required_top = {
        "export_format": CORE_EXPORT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "training_generation": "V3.2",
        "all_embeddings_from_newly_trained_v32_core": True,
        "historical_checkpoint_loaded": False,
        "historical_prediction_loaded": False,
    }
    for field, expected in required_top.items():
        if payload.get(field) != expected or type(payload.get(field)) is not type(expected):
            raise EvidenceTrainingContractError(
                f"Core manifest has invalid {field}: {payload.get(field)!r}"
            )
    fold_payload = payload.get("folds", {}).get(str(patient_fold))
    if not isinstance(fold_payload, Mapping):
        raise EvidenceTrainingContractError(f"Core manifest lacks fold {patient_fold}")
    if int(fold_payload.get("patient_fold", -1)) != patient_fold:
        raise EvidenceTrainingContractError("Core manifest fold identity drift")
    if fold_payload.get("trained_from_scratch") is not True:
        raise EvidenceTrainingContractError("Core must be trained from scratch")
    if fold_payload.get("old_checkpoint_loaded") is not False:
        raise EvidenceTrainingContractError("Historical core checkpoint is forbidden")
    if _clean(fold_payload.get("checkpoint_format")) != "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1":
        raise EvidenceTrainingContractError("Core checkpoint is not the V3.2 training format")
    formal_graph_variant: str | None = None
    graph_receipt_sha256: str | None = None
    patient_receipt_sha256: str | None = None
    if require_current_g2_authority:
        from cc_hhgt.v32.patient_fold_authority import (
            FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256,
            FROZEN_V32_RECEIPT_SHA256,
            FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
        )

        expected_manifest = _clean(expected_manifest_sha256).lower()
        expected_graph = _clean(expected_graph_authority_receipt_sha256).lower()
        if not _is_sha256(expected_manifest) or observed_manifest_sha256 != expected_manifest:
            raise EvidenceTrainingContractError(
                "Current G2 core manifest is not externally SHA256-pinned"
            )
        if not _is_sha256(expected_graph):
            raise EvidenceTrainingContractError(
                "Current G2 graph authority receipt is not externally SHA256-pinned"
            )
        formal = payload.get("formal_lineage")
        if not isinstance(formal, Mapping):
            raise EvidenceTrainingContractError(
                "Current G2 core formal lineage is missing; legacy V1 self-declaration is insufficient"
            )
        required_formal = {
            "format": CURRENT_G2_CORE_LINEAGE_FORMAT,
            "formal_graph_variant": "G2",
            "graph_authority_receipt_sha256": expected_graph,
            "patient_fold_authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
            "sample_patient_fold_map_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "patient_authority_logical_sha256": FROZEN_V32_PATIENT_AUTHORITY_LOGICAL_SHA256,
            "sealed_test_opened": False,
        }
        for field, expected in required_formal.items():
            if formal.get(field) != expected or type(formal.get(field)) is not type(expected):
                raise EvidenceTrainingContractError(
                    f"Current G2 core formal lineage has invalid {field}: {formal.get(field)!r}"
                )
        required_fold = {
            "formal_graph_variant": "G2",
            "graph_authority_receipt_sha256": expected_graph,
            "patient_fold_authority_receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        }
        for field, expected in required_fold.items():
            if fold_payload.get(field) != expected:
                raise EvidenceTrainingContractError(
                    f"Current G2 core fold lineage has invalid {field}: {fold_payload.get(field)!r}"
                )
        if not _is_sha256(fold_payload.get("prepared_sha256")):
            raise EvidenceTrainingContractError(
                "Current G2 core fold lacks a hash-bound prepared G2 payload"
            )
        formal_graph_variant = "G2"
        graph_receipt_sha256 = expected_graph
        patient_receipt_sha256 = FROZEN_V32_RECEIPT_SHA256
    checkpoint_sha = _clean(fold_payload.get("checkpoint_sha256")).lower()
    parameter_sha = _clean(fold_payload.get("core_parameter_sha256")).lower()
    if not _is_sha256(checkpoint_sha) or not _is_sha256(parameter_sha):
        raise EvidenceTrainingContractError("Core checkpoint/parameter SHA256 is missing or invalid")
    exports = fold_payload.get("exports")
    if not isinstance(exports, Mapping):
        raise EvidenceTrainingContractError("Core fold has no exports")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for node_type in ("cancer", "lncRNA", "pathway"):
        entry = exports.get(node_type)
        if not isinstance(entry, Mapping) or not _is_sha256(entry.get("sha256")):
            raise EvidenceTrainingContractError(f"Core fold lacks hashed {node_type} export")
        path = _resolve_export_path(root, patient_fold, node_type, entry.get("path"))
        if verify_files:
            if not path.is_file():
                raise EvidenceTrainingContractError(f"Missing core {node_type} export: {path}")
            observed = file_sha256(path)
            if observed != str(entry["sha256"]).lower():
                raise EvidenceTrainingContractError(
                    f"Core {node_type} hash mismatch: {observed} != {entry['sha256']}"
                )
        paths[node_type] = path
        hashes[node_type] = str(entry["sha256"]).lower()
    return CoreEmbeddingLineage(
        patient_fold=patient_fold,
        manifest_path=manifest_path,
        manifest_sha256=observed_manifest_sha256,
        checkpoint_sha256=checkpoint_sha,
        core_parameter_sha256=parameter_sha,
        export_paths=paths,
        export_sha256=hashes,
        formal_graph_variant=formal_graph_variant,
        graph_authority_receipt_sha256=graph_receipt_sha256,
        patient_fold_authority_receipt_sha256=patient_receipt_sha256,
    )


def build_fresh_init_contract(
    *, patient_fold: int, seed: int, core_lineage: CoreEmbeddingLineage
) -> dict[str, Any]:
    if core_lineage.patient_fold != patient_fold:
        raise EvidenceTrainingContractError("Private head/core fold mismatch")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvidenceTrainingContractError("seed must be a non-negative integer")
    return {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "patient_fold": patient_fold,
        "seed": seed,
        "private_parameters_fresh_init": True,
        "initialized_from_checkpoint": False,
        "historical_evidence_checkpoint_allowed": False,
        "historical_evidence_result_allowed": False,
        "family_spf_allowed": False,
        "pair_evidence_supervision_allowed": False,
        "core_frozen": True,
        "core_detached": True,
        "core_checkpoint_sha256": core_lineage.checkpoint_sha256,
        "core_parameter_sha256": core_lineage.core_parameter_sha256,
        "core_manifest_sha256": core_lineage.manifest_sha256,
    }


def _node_aliases(node_type: str, value: object) -> tuple[str, ...]:
    raw = _clean(value)
    normalized = _identity(value) if node_type != "cancer" else _cancer(value)
    prefix = {"cancer": "CANCER", "lncRNA": "LNC", "pathway": "PATHWAY"}[node_type]
    return tuple(dict.fromkeys([raw, raw.upper(), normalized, f"{prefix}:{normalized}"]))


def load_core_feature_bundle(
    core_embedding_root: Path,
    patient_fold: int,
    *,
    require_current_g2_authority: bool = False,
    expected_manifest_sha256: str | None = None,
    expected_graph_authority_receipt_sha256: str | None = None,
) -> CoreFeatureBundle:
    lineage = validate_core_embedding_lineage(
        core_embedding_root,
        patient_fold,
        require_current_g2_authority=require_current_g2_authority,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_graph_authority_receipt_sha256=(
            expected_graph_authority_receipt_sha256
        ),
    )
    feature_maps: dict[str, dict[str, np.ndarray]] = {}
    feature_dims: dict[str, int] = {}
    for node_type, path in lineage.export_paths.items():
        frame = pd.read_parquet(path)
        id_col = _column(frame, ["node_id"], required=True)
        features = sorted(column for column in frame.columns if str(column).startswith("core_feature_"))
        if not features:
            raise EvidenceTrainingContractError(f"Core {node_type} export has no feature columns")
        matrix = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
        if not np.isfinite(matrix).all():
            raise EvidenceTrainingContractError(f"Core {node_type} features are non-finite")
        mapping: dict[str, np.ndarray] = {}
        for node_id, vector in zip(frame[id_col].astype(str), matrix, strict=True):
            for alias in _node_aliases(node_type, node_id):
                if alias in mapping and not np.array_equal(mapping[alias], vector):
                    raise EvidenceTrainingContractError(
                        f"Core {node_type} alias collision: {alias}"
                    )
                mapping[alias] = vector
        feature_maps[node_type] = mapping
        feature_dims[node_type] = len(features)
    return CoreFeatureBundle(lineage, feature_maps, feature_dims)


def _event_feature_matrix(
    frame: pd.DataFrame,
    feature_dim: int,
    *,
    preserve_assay_type: bool = False,
) -> np.ndarray:
    if feature_dim < 32:
        raise EvidenceTrainingContractError("event_feature_dim must be at least 32")
    fields = event_feature_fields(preserve_assay_type=preserve_assay_type)
    matrix = np.zeros((len(frame), feature_dim), dtype=np.float32)
    for row_index, row in enumerate(frame.to_dict("records")):
        for field in fields:
            token = f"{field}={_clean(row.get(field), UNKNOWN).lower()}"
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % (feature_dim - 8)
            sign = 1.0 if digest[4] & 1 else -1.0
            matrix[row_index, bucket] += sign
        matrix[row_index, feature_dim - 8] = float(bool(row.get("is_experimental", False)))
        matrix[row_index, feature_dim - 7] = float(bool(row.get("is_computational", False)))
        matrix[row_index, feature_dim - 6] = float(bool(row.get("is_physical", False)))
        matrix[row_index, feature_dim - 5] = float(bool(_tokens(row.get("pmid"))))
        matrix[row_index, feature_dim - 4] = min(len(_tokens(row.get("pmid"))), 5) / 5.0
        matrix[row_index, feature_dim - 3] = float(row.get("route_type") == "DIRECT_EXACT_ASSERTION")
        matrix[row_index, feature_dim - 2] = float(row.get("route_type") == "PARTNER_EXACT_MEMBER")
        matrix[row_index, feature_dim - 1] = 1.0
    return matrix


def build_bag_examples(
    events: pd.DataFrame,
    core: CoreFeatureBundle,
    *,
    event_feature_dim: int = 128,
    max_events: int = 64,
    preserve_assay_type: bool = False,
) -> tuple[list[BagExample], pd.DataFrame]:
    """Create private-head examples; core values are copied as immutable arrays."""

    if max_events < 1:
        raise EvidenceTrainingContractError("max_events must be positive")
    required = set(EXACT_KEYS + ["event_id", "leakage_fold", "confidence_target", "direction_target"])
    missing_columns = sorted(required - set(events.columns))
    if missing_columns:
        raise EvidenceTrainingContractError(f"Events lack training columns: {missing_columns}")
    examples: list[BagExample] = []
    missing_core: list[dict[str, Any]] = []
    for key, group in events.groupby(EXACT_KEYS, observed=True, sort=True):
        folds = sorted(set(pd.to_numeric(group.leakage_fold, errors="raise").astype(int)))
        if len(folds) != 1:
            raise EvidenceTrainingContractError(f"Pair spans leakage folds: {key} -> {folds}")
        group = group.sort_values("event_id").head(max_events)
        core_vector = core.vector_for(*key)
        if core_vector is None:
            missing_core.append(
                {
                    **dict(zip(EXACT_KEYS, map(str, key), strict=True)),
                    "leakage_fold": folds[0],
                    "event_count": int(len(group)),
                    "unavailable_reason": "CORE_EMBEDDING_ID_MISSING",
                }
            )
            continue
        confidence_values = pd.to_numeric(group.confidence_target, errors="coerce").to_numpy(float)
        confidence_target = (
            float(np.mean(confidence_values[np.isfinite(confidence_values)]))
            if np.isfinite(confidence_values).any()
            else math.nan
        )
        directions = pd.to_numeric(group.direction_target, errors="coerce").fillna(-1).astype(int)
        directions = directions[directions.isin([0, 1, 2])]
        if directions.empty:
            direction_target = -1
        else:
            counts = directions.value_counts()
            direction_target = int(counts.index[0]) if len(counts) == 1 or counts.iloc[0] > counts.iloc[1] else -1
        features = _event_feature_matrix(
            group, event_feature_dim, preserve_assay_type=preserve_assay_type
        )
        immutable_core = np.asarray(core_vector, dtype=np.float32).copy()
        immutable_core.setflags(write=False)
        examples.append(
            BagExample(
                key=tuple(map(str, key)),
                event_features=features,
                core_features=immutable_core,
                confidence_target=confidence_target,
                direction_target=direction_target,
                leakage_fold=folds[0],
                event_count=int(len(group)),
            )
        )
    return examples, pd.DataFrame(missing_core)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in dependency-poor deployments
        raise EvidenceTrainingContractError(
            "V3.2 private EventSet training requires the optional torch dependency"
        ) from exc
    return torch


def build_fresh_private_eventset_head(
    *,
    event_feature_dim: int,
    core_feature_dim: int,
    hidden_dim: int = 96,
    dropout: float = 0.20,
    seed: int = 20260726,
):
    """Create a newly initialized private attention head; no load API exists."""

    torch = _require_torch()
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvidenceTrainingContractError("seed must be a non-negative integer")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    class PrivateEventSetAttentionHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.event_projection = torch.nn.Sequential(
                torch.nn.Linear(event_feature_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
            )
            self.core_query = torch.nn.Sequential(
                torch.nn.Linear(core_feature_dim, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
            )
            self.attention_key = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.attention_value = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.private_fusion = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2 + 2, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            )
            self.confidence_head = torch.nn.Linear(hidden_dim, 1)
            self.direction_head = torch.nn.Linear(hidden_dim, 3)

        def forward(self, event_features, event_mask, frozen_core_features):
            if frozen_core_features.requires_grad:
                raise RuntimeError("V3.2 core embeddings must be detached")
            core = frozen_core_features.detach()
            event_hidden = self.event_projection(event_features)
            query = self.core_query(core).unsqueeze(1)
            logits = (
                self.attention_key(event_hidden) * query
            ).sum(dim=-1) / math.sqrt(event_hidden.shape[-1])
            logits = logits.masked_fill(~event_mask, -1.0e9)
            attention = torch.softmax(logits, dim=1) * event_mask.float()
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
            pooled = (
                self.attention_value(event_hidden) * attention.unsqueeze(-1)
            ).sum(dim=1)
            event_count = event_mask.sum(dim=1, keepdim=True).float()
            count_features = torch.cat(
                [torch.log1p(event_count) / math.log(65.0), (event_count > 1).float()], dim=1
            )
            fused = self.private_fusion(
                torch.cat([pooled, query.squeeze(1), count_features], dim=1)
            )
            return {
                "confidence_logit": self.confidence_head(fused).squeeze(-1),
                "direction_logits": self.direction_head(fused),
                "attention": attention,
            }

    model = PrivateEventSetAttentionHead()
    model.private_parameters_fresh_init = True
    model.core_frozen = True
    model.core_detached = True
    return model


def model_parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _batch_tensors(examples: Sequence[BagExample], indices: Sequence[int], device: Any):
    torch = _require_torch()
    selected = [examples[index] for index in indices]
    max_events = max(example.event_features.shape[0] for example in selected)
    event_dim = selected[0].event_features.shape[1]
    events = np.zeros((len(selected), max_events, event_dim), dtype=np.float32)
    mask = np.zeros((len(selected), max_events), dtype=bool)
    for index, example in enumerate(selected):
        count = example.event_features.shape[0]
        events[index, :count] = example.event_features
        mask[index, :count] = True
    core = np.stack([example.core_features for example in selected]).astype(np.float32)
    confidence = np.asarray([example.confidence_target for example in selected], dtype=np.float32)
    direction = np.asarray([example.direction_target for example in selected], dtype=np.int64)
    core_tensor = torch.from_numpy(core).to(device).detach()
    core_tensor.requires_grad_(False)
    return {
        "events": torch.from_numpy(events).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "core": core_tensor,
        "confidence": torch.from_numpy(confidence).to(device),
        "direction": torch.from_numpy(direction).to(device),
    }


def _loss_terms(torch: Any, result: Mapping[str, Any], batch: Mapping[str, Any]) -> list[Any]:
    terms: list[Any] = []
    confidence_mask = torch.isfinite(batch["confidence"])
    if bool(confidence_mask.any()):
        terms.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                result["confidence_logit"][confidence_mask],
                batch["confidence"][confidence_mask].clamp(0.0, 1.0),
            )
        )
    direction_mask = batch["direction"].ge(0)
    if bool(direction_mask.any()):
        terms.append(
            0.5
            * torch.nn.functional.cross_entropy(
                result["direction_logits"][direction_mask],
                batch["direction"][direction_mask],
            )
        )
    return terms


def _evaluate_examples(model: Any, examples: Sequence[BagExample], *, batch_size: int, device: Any) -> float:
    torch = _require_torch()
    if not examples:
        return math.nan
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            indices = list(range(start, min(start + batch_size, len(examples))))
            batch = _batch_tensors(examples, indices, device)
            result = model(batch["events"], batch["mask"], batch["core"])
            terms = _loss_terms(torch, result, batch)
            if terms:
                losses.append(float(torch.stack(terms).sum().detach().cpu()))
    return float(np.mean(losses)) if losses else math.nan


def fit_private_eventset_head(
    train_examples: Sequence[BagExample],
    validation_examples: Sequence[BagExample],
    *,
    event_feature_dim: int,
    core_feature_dim: int,
    seed: int,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 128,
    learning_rate: float = 2.0e-3,
    hidden_dim: int = 96,
    dropout: float = 0.20,
    device: str | None = None,
) -> PrivateHeadFit:
    """Fit a fresh private head; core arrays never enter model parameters."""

    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_fresh_private_eventset_head(
        event_feature_dim=event_feature_dim,
        core_feature_dim=core_feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        seed=seed,
    )
    initial_sha = model_parameter_sha256(model)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(torch_device)
    confidence_available = any(np.isfinite(example.confidence_target) for example in train_examples)
    direction_available = any(example.direction_target >= 0 for example in train_examples)
    if not train_examples or not (confidence_available or direction_available):
        return PrivateHeadFit(
            model=model,
            initial_parameter_sha256=initial_sha,
            final_parameter_sha256=initial_sha,
            history=[],
            optimizer_steps=0,
            confidence_supervision_available=confidence_available,
            direction_supervision_available=direction_available,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2.0e-3)
    rng = np.random.default_rng(seed)
    best_state: dict[str, Any] | None = None
    best_loss = math.inf
    stale = 0
    history: list[dict[str, float]] = []
    optimizer_steps = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_examples)).tolist()
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = _batch_tensors(train_examples, order[start : start + batch_size], torch_device)
            result = model(batch["events"], batch["mask"], batch["core"])
            terms = _loss_terms(torch, result, batch)
            if not terms:
                continue
            loss = torch.stack(terms).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else math.nan
        validation_loss = _evaluate_examples(
            model, validation_examples, batch_size=batch_size, device=torch_device
        )
        selection_loss = validation_loss if np.isfinite(validation_loss) else train_loss
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if np.isfinite(selection_loss) and selection_loss < best_loss - 1.0e-6:
            best_loss = selection_loss
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    final_sha = model_parameter_sha256(model)
    return PrivateHeadFit(
        model=model,
        initial_parameter_sha256=initial_sha,
        final_parameter_sha256=final_sha,
        history=history,
        optimizer_steps=optimizer_steps,
        confidence_supervision_available=confidence_available,
        direction_supervision_available=direction_available,
    )


def predict_private_eventset_head(
    fit: PrivateHeadFit,
    examples: Sequence[BagExample],
    *,
    batch_size: int = 256,
    mc_samples: int = 16,
    device: str | None = None,
) -> pd.DataFrame:
    torch = _require_torch()
    if mc_samples < 2:
        raise EvidenceTrainingContractError("mc_samples must be at least two")
    columns = EXACT_KEYS + [
        "evidence_confidence_probability",
        "direction",
        "uncertainty",
        "availability",
        "unavailable_reason",
        "event_count",
        "evidence_fold",
    ]
    if not examples:
        return pd.DataFrame(columns=columns)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = fit.model.to(torch_device)
    rows: list[dict[str, Any]] = []
    direction_names = np.asarray(["negative", "neutral", "positive"], dtype=object)
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            subset = list(examples[start : start + batch_size])
            batch = _batch_tensors(subset, list(range(len(subset))), torch_device)
            confidence_draws: list[np.ndarray] = []
            direction_draws: list[np.ndarray] = []
            for _ in range(mc_samples):
                model.train()  # activate only private-head dropout for MC uncertainty
                result = model(batch["events"], batch["mask"], batch["core"])
                confidence_draws.append(torch.sigmoid(result["confidence_logit"]).cpu().numpy())
                direction_draws.append(torch.softmax(result["direction_logits"], dim=-1).cpu().numpy())
            confidence_values = np.stack(confidence_draws)
            direction_values = np.stack(direction_draws)
            confidence_mean = confidence_values.mean(axis=0)
            confidence_var = confidence_values.var(axis=0)
            direction_mean = direction_values.mean(axis=0)
            for index, example in enumerate(subset):
                confidence = (
                    float(confidence_mean[index])
                    if fit.confidence_supervision_available and fit.optimizer_steps > 0
                    else math.nan
                )
                if np.isfinite(confidence):
                    entropy = -(
                        confidence * math.log(max(confidence, 1.0e-12))
                        + (1.0 - confidence) * math.log(max(1.0 - confidence, 1.0e-12))
                    ) / math.log(2.0)
                    epistemic = min(1.0, 4.0 * float(confidence_var[index]))
                    direction_entropy = -float(
                        np.sum(direction_mean[index] * np.log(np.clip(direction_mean[index], 1.0e-12, 1.0)))
                    ) / math.log(3.0)
                    uncertainty = float(np.clip(0.5 * entropy + 0.25 * epistemic + 0.25 * direction_entropy, 0, 1))
                    available = True
                    reason = ""
                else:
                    uncertainty = math.nan
                    available = False
                    reason = "NO_RAW_EVENT_CONFIDENCE_SUPERVISION"
                direction = (
                    str(direction_names[int(np.argmax(direction_mean[index]))])
                    if fit.direction_supervision_available and fit.optimizer_steps > 0
                    else None
                )
                rows.append(
                    {
                        **dict(zip(EXACT_KEYS, example.key, strict=True)),
                        "evidence_confidence_probability": confidence,
                        "direction": direction,
                        "uncertainty": uncertainty,
                        "availability": available,
                        "unavailable_reason": reason,
                        "event_count": example.event_count,
                        "evidence_fold": example.leakage_fold,
                    }
                )
    model.eval()
    return pd.DataFrame(rows, columns=columns)


def complete_prediction_frame(
    universe: pd.DataFrame,
    predictions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    training_run_id: str = "V32-EVIDENCE-PRIVATE",
) -> pd.DataFrame:
    """Return every exact pair; missing evidence is null with an explicit reason."""

    candidates = normalize_candidates(universe)
    if not predictions.empty and predictions.duplicated(EXACT_KEYS).any():
        raise EvidenceTrainingContractError("Evidence predictions are not unique by exact pair")
    event_counts = (
        events.groupby(EXACT_KEYS, observed=True)
        .event_id.nunique()
        .rename("observed_event_count")
        .reset_index()
        if not events.empty
        else pd.DataFrame(columns=EXACT_KEYS + ["observed_event_count"])
    )
    prediction_columns = EXACT_KEYS + [
        "evidence_confidence_probability",
        "direction",
        "uncertainty",
        "availability",
        "unavailable_reason",
        "event_count",
        "evidence_fold",
    ]
    available_predictions = predictions.reindex(columns=prediction_columns)
    output = (
        candidates.merge(available_predictions, on=EXACT_KEYS, how="left", validate="one_to_one")
        .merge(event_counts, on=EXACT_KEYS, how="left", validate="one_to_one")
        .sort_values("candidate_order")
        .reset_index(drop=True)
    )
    output["observed_event_count"] = output.observed_event_count.fillna(0).astype(int)
    output["event_count"] = output.event_count.fillna(output.observed_event_count).astype(int)
    output["availability"] = output.availability.fillna(False).astype(bool)
    no_event = output.observed_event_count.eq(0)
    missing_prediction = ~no_event & output.evidence_confidence_probability.isna()
    output.loc[no_event, "unavailable_reason"] = "NO_EXACT_PATHWAY_EVENT"
    output.loc[
        missing_prediction & output.unavailable_reason.fillna("").eq(""),
        "unavailable_reason",
    ] = "PRIVATE_HEAD_UNAVAILABLE"
    output.loc[~output.availability, "evidence_confidence_probability"] = np.nan
    output.loc[~output.availability, "uncertainty"] = np.nan
    output["failure_reason"] = output["unavailable_reason"].fillna("").astype(str)
    output["analysis_version"] = ANALYSIS_VERSION
    output["training_run_id"] = str(training_run_id)
    output["changes_primary_ranking"] = False
    output["main_ranking_modified"] = False
    return output.drop(columns=["candidate_order", "observed_event_count"])


def build_unavailable_prediction_frame(
    universe: pd.DataFrame,
    *,
    training_run_id: str,
    failure_reason: str,
) -> pd.DataFrame:
    """Build a contract-valid all-null frame without dropping any candidate."""

    reason = _clean(failure_reason)
    if not reason:
        raise EvidenceTrainingContractError("Unavailable predictions require a failure reason")
    candidates = normalize_candidates(universe)
    output = candidates[EXACT_KEYS].copy()
    output["evidence_confidence_probability"] = np.nan
    output["direction"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["uncertainty"] = np.nan
    output["availability"] = False
    output["failure_reason"] = reason
    output["unavailable_reason"] = reason
    output["analysis_version"] = ANALYSIS_VERSION
    output["training_run_id"] = str(training_run_id)
    output["changes_primary_ranking"] = False
    output["main_ranking_modified"] = False
    return output


def read_input_table(path: Path) -> pd.DataFrame:
    if Path(path).is_dir():
        files = sorted(Path(path).glob("*.parquet"))
        if not files:
            raise EvidenceTrainingContractError(f"Table directory has no parquet files: {path}")
        return pd.concat((pd.read_parquet(file) for file in files), ignore_index=True)
    suffix = Path(path).suffix.lower()
    lower_name = Path(path).name.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"} or lower_name.endswith((".tsv.gz", ".txt.gz")):
        return pd.read_csv(path, sep="\t", low_memory=False)
    if suffix == ".csv" or lower_name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)
    raise EvidenceTrainingContractError(f"Unsupported table format: {path}")


def read_candidate_table(path: Path) -> pd.DataFrame:
    """Read only exact IDs; primary ranks/scores never enter process memory."""

    value = Path(path)
    if value.is_dir():
        files = sorted(value.glob("*.parquet"))
        if not files:
            raise EvidenceTrainingContractError(f"Candidate directory has no parquet files: {value}")
        return pd.concat(
            (pd.read_parquet(file, columns=EXACT_KEYS) for file in files),
            ignore_index=True,
        )
    if value.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(value, columns=EXACT_KEYS)
        except ImportError as exc:
            raise EvidenceTrainingContractError(
                "A parquet engine is unavailable while reading the exact candidate "
                f"universe (install pyarrow or fastparquet): {value}"
            ) from exc
        except Exception as exc:
            raise EvidenceTrainingContractError(
                f"Candidate parquet must expose exact ID columns {EXACT_KEYS}: {value}"
            ) from exc
    if value.suffix.lower() in {".tsv", ".txt", ".csv"}:
        separator = "\t" if value.suffix.lower() in {".tsv", ".txt"} else ","
        try:
            return pd.read_csv(value, sep=separator, usecols=EXACT_KEYS, low_memory=False)
        except Exception as exc:
            raise EvidenceTrainingContractError(
                f"Candidate table must expose exact ID columns {EXACT_KEYS}: {value}"
            ) from exc
    raise EvidenceTrainingContractError(f"Unsupported candidate format: {value}")


def validate_exact_candidate_authority(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_sha256: str | None,
    expected_rows: int | None,
    expected_cancer_count: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fail closed on a pinned, exact-pathway candidate universe.

    The formal hash is deliberately a byte-level authority pin.  It prevents a
    drug-response candidate table (or any old prediction/ranking export) from
    being accepted merely because it happens to contain the three exact IDs.
    """

    value = Path(path)
    observed_sha256 = input_path_sha256(value)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise EvidenceTrainingContractError(
            "Exact candidate authority SHA-256 drift: "
            f"{observed_sha256} != {expected_sha256}: {value}"
        )
    candidates = normalize_candidates(frame)
    candidate_rows = int(len(candidates))
    cancer_count = int(candidates.cancer_id.nunique())
    if expected_rows is not None and candidate_rows != expected_rows:
        raise EvidenceTrainingContractError(
            f"Formal candidate count drift: {candidate_rows} != {expected_rows}"
        )
    if expected_cancer_count is not None and cancer_count != expected_cancer_count:
        raise EvidenceTrainingContractError(
            f"Formal cancer count drift: {cancer_count} != {expected_cancer_count}"
        )
    return candidates, {
        "path": str(value),
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "sha256_pinned": expected_sha256 is not None,
        "rows": candidate_rows,
        "cancers": cancer_count,
        "exact_id_columns_read": list(EXACT_KEYS),
        "old_prediction_or_ranking_columns_read": False,
    }


def input_path_sha256(path: Path) -> str:
    """Hash one input file or a deterministic flat parquet collection."""

    value = Path(path)
    if value.is_file():
        return file_sha256(value)
    if value.is_dir():
        files = sorted(value.glob("*.parquet"))
        if not files:
            raise EvidenceTrainingContractError(f"Input directory has no parquet files: {value}")
        digest = hashlib.sha256()
        for file in files:
            digest.update(file.name.encode("utf-8"))
            digest.update(file_sha256(file).encode("ascii"))
        return digest.hexdigest()
    raise EvidenceTrainingContractError(f"Input path does not exist: {value}")


def _safe_parquet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].map(lambda value: isinstance(value, (tuple, list, set))).any():
            output[column] = output[column].map(
                lambda value: ";".join(map(str, value)) if isinstance(value, (tuple, list, set)) else value
            )
    return output


def _fresh_output_root(path: Path) -> Path:
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise EvidenceTrainingContractError(
            f"Refusing to reuse a non-empty evidence output directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_string_series(
    frame: pd.DataFrame, column: str | None, default: str = ""
) -> pd.Series:
    if column is None:
        return pd.Series(default, index=frame.index, dtype="string")
    return frame[column].astype("string").fillna("").str.strip()


def _frame_bool_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(False, index=frame.index, dtype=bool)
    raw = frame[column]
    if pd.api.types.is_bool_dtype(raw.dtype):
        return raw.fillna(False).astype(bool)
    return (
        raw.astype("string")
        .fillna("")
        .str.strip()
        .str.lower()
        .isin({"1", "true", "t", "yes", "y", "positive"})
    )


def _normalized_identifier_series(
    values: pd.Series, identifier_map: Mapping[str, str] | None = None
) -> pd.Series:
    output = values.astype("string").fillna("").str.strip().str.upper()
    output = output.str.replace(
        r"^(?:LNC|GENE|PROTEIN|PATHWAY|CANCER):", "", regex=True
    )
    ensembl = output.str.extract(r"(ENSG\d+(?:\.\d+)?)", expand=False)
    output = ensembl.fillna(output)
    output = output.str.replace(r"^(ENSG\d+)\.\d+$", r"\1", regex=True)
    if identifier_map:
        mapped = output.map(identifier_map)
        normalized_key = output.str.replace(r"[^A-Z0-9]+", "", regex=True)
        mapped = mapped.fillna(normalized_key.map(identifier_map))
        output = mapped.fillna(output).astype("string").str.upper()
        output = output.str.replace(r"^(ENSG\d+)\.\d+$", r"\1", regex=True)
    return output.fillna("").astype("string")


def _iter_parquet_frames(
    path: Path, *, columns: Sequence[str], batch_size: int = 100_000
) -> Iterable[pd.DataFrame]:
    value = Path(path)
    if batch_size <= 0:
        raise EvidenceTrainingContractError("Evidence scan batch_size must be positive")
    if value.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - pandas parquet already needs an engine
            raise EvidenceTrainingContractError(
                "pyarrow is required for formal evidence dry-run"
            ) from exc
        parquet = pq.ParquetFile(value)
        available = set(parquet.schema_arrow.names)
        selected = [column for column in columns if column in available]
        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            yield batch.to_pandas()
        return

    # Fresh rematerialization intentionally emits a gzip-compressed TSV so it
    # can run in the minimal server Python without pyarrow. Keep the formal
    # dry-run bounded-memory for that authority as well; read_input_table()
    # support alone is insufficient because the dry-run uses this scanner.
    lower_name = value.name.lower()
    if (
        value.suffix.lower() in {".tsv", ".txt", ".csv"}
        or lower_name.endswith((".tsv.gz", ".txt.gz", ".csv.gz"))
    ):
        separator = "," if lower_name.endswith((".csv", ".csv.gz")) else "\t"
        requested = frozenset(map(str, columns))
        reader = pd.read_csv(
            value,
            sep=separator,
            usecols=lambda column: str(column) in requested,
            chunksize=batch_size,
            low_memory=False,
        )
        yield from reader
        return

    raise EvidenceTrainingContractError(
        f"Unsupported bounded-memory Evidence input format: {value}"
    )


def _standardize_raw_batch(
    frame: pd.DataFrame,
    *,
    source_kind: str,
    source_input: Path,
    row_offset: int,
    identifier_map: Mapping[str, str],
) -> pd.DataFrame:
    columns = _source_columns(frame)
    lnc = _normalized_identifier_series(
        _frame_string_series(frame, columns["lncrna_id"]), identifier_map
    )
    partner = _normalized_identifier_series(
        _frame_string_series(frame, columns["partner_id"]), identifier_map
    )
    pathway = _normalized_identifier_series(
        _frame_string_series(frame, columns["pathway_id"])
    )
    source_database = _frame_string_series(frame, columns["source_database"])
    source_database = source_database.mask(source_database.eq(""), source_kind)
    source_dataset = _frame_string_series(frame, columns["source_dataset"])
    source_dataset = source_dataset.mask(source_dataset.eq(""), source_database)
    raw_event_id = _frame_string_series(frame, columns["raw_event_id"])
    fallback_ids = pd.Series(
        [f"{source_kind}:{row_offset + index}" for index in range(len(frame))],
        index=frame.index,
        dtype="string",
    )
    raw_event_id = raw_event_id.mask(raw_event_id.eq(""), fallback_ids)
    source_record_id = _frame_string_series(frame, columns["source_record_id"])
    source_record_id = source_record_id.mask(source_record_id.eq(""), raw_event_id)
    cancer = _frame_string_series(frame, columns["cancer_id"]).str.upper()
    cancer = cancer.mask(
        cancer.isin({"", "PAN", "PANCAN", "PAN-CANCER", "PAN_CANCER", "ALL"}),
        "PAN_CANCER",
    )
    cancer = cancer.str.replace(r"^CANCER:", "", regex=True)
    relation = _frame_string_series(frame, columns["relation_type"], UNKNOWN)
    relation = relation.mask(relation.eq(""), UNKNOWN)
    experiment = _frame_string_series(frame, columns["experiment_type"], UNKNOWN)
    experiment = experiment.mask(experiment.eq(""), UNKNOWN)
    interaction_type = _frame_string_series(frame, columns["interaction_type"])
    directness = _frame_string_series(frame, columns["directness"])
    physical_text = (
        relation + " " + experiment + " " + interaction_type + " " + directness
    ).str.lower()
    is_physical = physical_text.str.contains(
        r"physical|binding|pull.?down|\brip\b|\bclip\b|chirp|co.?ip",
        regex=True,
        na=False,
    )
    explicit_physical = columns["physical"]
    if explicit_physical is not None:
        is_physical = is_physical | _frame_bool_series(frame, explicit_physical)
    is_experimental = _frame_bool_series(frame, columns["experimental"])
    if columns["experimental"] is not None:
        is_physical = is_physical & is_experimental
    is_computational = _frame_bool_series(frame, columns["computational"])
    is_physical = is_physical & ~is_computational & lnc.ne("") & partner.ne("")
    pmid = _frame_string_series(frame, columns["pmid"])
    physical_fact_id = pd.Series("", index=frame.index, dtype="string")
    physical_fact_id.loc[is_physical] = (
        "PHYSRAW32:" + source_kind + ":" + raw_event_id.loc[is_physical]
    )
    return pd.DataFrame(
        {
            "raw_event_id": raw_event_id,
            "source_kind": source_kind,
            "source_input": str(source_input),
            "source_row_index": np.arange(
                row_offset, row_offset + len(frame), dtype=np.int64
            ),
            "source_database": source_database,
            "source_dataset": source_dataset,
            "source_record_id": source_record_id,
            "cancer_id": cancer,
            "lncrna_id": lnc,
            "partner_id": partner,
            "pathway_id": pathway,
            "pmid": pmid,
            "relation_type": relation,
            "experiment_type": experiment,
            "is_experimental": is_experimental.astype(bool),
            "is_computational": is_computational.astype(bool),
            "is_physical": is_physical.astype(bool),
            "physical_fact_id": physical_fact_id,
        }
    )


def _eligible_exact_route_keys(
    standardized: pd.DataFrame,
    *,
    candidate_pathways_by_lnc: Mapping[str, frozenset[str]],
    member_pathways_by_partner: Mapping[str, frozenset[str]],
    pair_eligibility_cache: dict[tuple[str, str], bool],
) -> pd.Series:
    """Return a non-empty witness key only for a formal exact candidate route."""

    route = pd.Series("", index=standardized.index, dtype="string")
    plausible = standardized.lncrna_id.isin(candidate_pathways_by_lnc) & (
        standardized.pathway_id.ne("")
        | standardized.partner_id.isin(member_pathways_by_partner)
    )
    for index in standardized.index[plausible]:
        lnc = str(standardized.at[index, "lncrna_id"])
        pathway = str(standardized.at[index, "pathway_id"])
        partner = str(standardized.at[index, "partner_id"])
        candidate_pathways = candidate_pathways_by_lnc[lnc]
        if pathway and pathway in candidate_pathways:
            route.at[index] = f"DIRECT:{lnc}|{pathway}"
            continue
        if not partner:
            continue
        cache_key = (lnc, partner)
        eligible = pair_eligibility_cache.get(cache_key)
        if eligible is None:
            eligible = bool(
                candidate_pathways.intersection(
                    member_pathways_by_partner.get(partner, frozenset())
                )
            )
            pair_eligibility_cache[cache_key] = eligible
        if eligible:
            # Shared lnc/partner implies at least one shared exact pathway route.
            route.at[index] = f"MEMBER:{lnc}|{partner}"
    return route


def _first_sorted_intersection(left: np.ndarray, right: np.ndarray) -> int | None:
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_value = int(left[left_index])
        right_value = int(right[right_index])
        if left_value == right_value:
            return left_value
        if left_value < right_value:
            left_index += 1
        else:
            right_index += 1
    return None


def _finalize_source_connectivity_audit(
    chunks: Mapping[str, Mapping[str, list[np.ndarray]]],
    eligible_counts: Mapping[str, int],
) -> dict[str, Any]:
    sources = sorted(source for source, count in eligible_counts.items() if count > 0)
    tokens: dict[str, dict[str, np.ndarray]] = {}
    for source in sources:
        tokens[source] = {}
        for token_type in ("pair", "pmid"):
            values = list(chunks.get(source, {}).get(token_type, []))
            tokens[source][token_type] = (
                np.unique(np.concatenate(values)).astype(np.uint64, copy=False)
                if values
                else np.asarray([], dtype=np.uint64)
            )
    union_find = _UnionFind(len(sources))
    edges: list[dict[str, str]] = []
    for left_index, right_index in itertools.combinations(range(len(sources)), 2):
        left, right = sources[left_index], sources[right_index]
        witnesses: list[tuple[str, int]] = []
        for token_type in ("pair", "pmid"):
            witness = _first_sorted_intersection(
                tokens[left][token_type], tokens[right][token_type]
            )
            if witness is not None:
                witnesses.append((token_type, witness))
        if witnesses:
            union_find.union(left_index, right_index)
            for token_type, witness in witnesses:
                edges.append(
                    {
                        "left_source": left,
                        "right_source": right,
                        "token_type": token_type,
                        "witness_uint64_sha": f"{witness:016x}",
                    }
                )
    component_members: dict[int, list[str]] = {}
    for index, source in enumerate(sources):
        component_members.setdefault(union_find.find(index), []).append(source)
    components = sorted(
        (sorted(members) for members in component_members.values()),
        key=lambda members: (members[0], len(members)),
    )
    return {
        "split_policy": (
            "connected_components(PMID,source,dataset,"
            "cancer-lncRNA-exact_pathway_pair)"
        ),
        "audit_route_policy": (
            "raw direct exact route or static-member route intersecting the formal "
            "candidate universe; source nodes joined by PMID or shared eligible lnc-partner"
        ),
        "eligible_source_count": len(sources),
        "eligible_event_rows_by_source": {
            source: int(eligible_counts[source]) for source in sources
        },
        "connected_component_count": len(components),
        "connected_components": components,
        "strict_five_fold_trainable": len(components) >= N_FOLDS,
        "source_graph_edges": edges,
        "hash_witness_note": (
            "Witnesses are deterministic pandas uint64 hashes; no raw PMID is exposed."
        ),
    }


def _write_formal_raw_artifacts(
    *,
    evidence_event_path: Path,
    interaction_relation_path: Path,
    output_root: Path,
    identifier_map: Mapping[str, str],
    candidate_pathways_by_lnc: Mapping[str, frozenset[str]],
    member_pathways_by_partner: Mapping[str, frozenset[str]],
) -> tuple[dict[str, int], dict[str, Any]]:
    """Stream all raw rows to lineage and physical-fact tables while auditing split."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise EvidenceTrainingContractError("pyarrow is required for formal evidence dry-run") from exc

    lineage_columns = [
        "lineage_id", "raw_event_id", "source_kind", "source_input",
        "source_row_index", "source_database", "source_dataset", "source_record_id",
        "cancer_id", "lncrna_id", "partner_id", "pathway_id", "pmid",
        "physical_fact_id", "mapping_status", "family_broadcast_used",
        "prediction_available", "failure_reason",
    ]
    fact_columns = [
        "physical_fact_id", "cancer_id", "lncrna_id", "partner_id",
        "relation_type", "experiment_type", "source_database", "source_dataset",
        "source_record_id", "pmid", "source_kind", "source_input",
        "source_row_index", "is_experimental", "is_prediction",
        "prediction_available", "failure_reason", "source_occurrence_count",
    ]
    lineage_schema = pa.schema(
        [
            pa.field(column, pa.int64() if column == "source_row_index" else
                     pa.bool_() if column in {"family_broadcast_used", "prediction_available"}
                     else pa.string())
            for column in lineage_columns
        ]
    )
    fact_schema = pa.schema(
        [
            pa.field(column, pa.int64() if column in {"source_row_index", "source_occurrence_count"}
                     else pa.bool_() if column in {"is_experimental", "is_prediction", "prediction_available"}
                     else pa.string())
            for column in fact_columns
        ]
    )
    lineage_path = output_root / "event_lineage.parquet"
    fact_path = output_root / "physical_interaction_facts.parquet"
    lineage_writer = fact_writer = None
    raw_counts: dict[str, int] = {}
    physical_count = 0
    connectivity_chunks: dict[str, dict[str, list[np.ndarray]]] = {}
    eligible_counts: dict[str, int] = {}
    pair_eligibility_cache: dict[tuple[str, str], bool] = {}
    requested_columns = sorted(
        {
            alias
            for aliases in (
                ["lncrna_id", "lncrna", "lnc_id", "rna_id"],
                ["partner_id", "target_id", "gene_id"],
                ["pathway_id", "exact_pathway_id"],
                ["cancer_id", "tcga_code", "cancer", "disease", "disease_raw"],
                ["evidence_event_id", "event_id", "interaction_row_id", "interaction_id", "source_record_id"],
                ["source_database", "source", "database"],
                ["source_dataset", "dataset_id", "dataset", "source_resource", "source_version"],
                ["pmid", "pubmed_id", "independent_pmid"],
                ["experiment_type", "experiment_family", "assay_type", "experimental_system", "experiment_raw"],
                ["relation_type", "interaction_class", "assertion_type"],
                ["interaction_type", "experimental_system_type"],
                ["experimental", "is_experimental", "computational", "is_predicted"],
                ["physical", "is_physical", "physical_interaction", "directness"],
            )
            for alias in aliases
        }
    )
    try:
        for source_kind, input_path in (
            ("evidence_event", Path(evidence_event_path)),
            ("interaction_relation", Path(interaction_relation_path)),
        ):
            row_offset = 0
            for frame in _iter_parquet_frames(input_path, columns=requested_columns):
                standardized = _standardize_raw_batch(
                    frame,
                    source_kind=source_kind,
                    source_input=input_path,
                    row_offset=row_offset,
                    identifier_map=identifier_map,
                )
                row_offset += len(frame)
                raw_counts[source_kind] = raw_counts.get(source_kind, 0) + len(frame)
                route_keys = _eligible_exact_route_keys(
                    standardized,
                    candidate_pathways_by_lnc=candidate_pathways_by_lnc,
                    member_pathways_by_partner=member_pathways_by_partner,
                    pair_eligibility_cache=pair_eligibility_cache,
                )
                eligible = route_keys.ne("")
                for source, indices in standardized.loc[eligible].groupby(
                    "source_database", sort=False
                ).groups.items():
                    source_name = str(source)
                    positions = list(indices)
                    eligible_counts[source_name] = eligible_counts.get(source_name, 0) + len(positions)
                    store = connectivity_chunks.setdefault(
                        source_name, {"pair": [], "pmid": []}
                    )
                    pair_tokens = "pair:" + route_keys.loc[positions].astype("string")
                    store["pair"].append(
                        np.unique(
                            pd.util.hash_pandas_object(pair_tokens, index=False)
                            .to_numpy(dtype=np.uint64, copy=False)
                        )
                    )
                    pmid_tokens = (
                        standardized.loc[positions, "pmid"]
                        .astype("string")
                        .str.split(r"[;,|\s]+")
                        .explode()
                        .fillna("")
                        .str.strip()
                    )
                    pmid_tokens = pmid_tokens.loc[pmid_tokens.ne("")]
                    if not pmid_tokens.empty:
                        store["pmid"].append(
                            np.unique(
                                pd.util.hash_pandas_object(
                                    "pmid:" + pmid_tokens, index=False
                                ).to_numpy(dtype=np.uint64, copy=False)
                            )
                        )

                lineage = standardized[
                    [
                        "raw_event_id", "source_kind", "source_input", "source_row_index",
                        "source_database", "source_dataset", "source_record_id", "cancer_id",
                        "lncrna_id", "partner_id", "pathway_id", "pmid", "physical_fact_id",
                    ]
                ].copy()
                lineage.insert(
                    0,
                    "lineage_id",
                    "LINRAW32:" + lineage.source_kind + ":" + lineage.raw_event_id,
                )
                lineage["mapping_status"] = "NOT_MATERIALIZED_STRICT_SPLIT_BLOCKED"
                lineage["family_broadcast_used"] = False
                lineage["prediction_available"] = False
                lineage["failure_reason"] = FORMAL_SPLIT_FAILURE_REASON
                lineage = lineage[lineage_columns]
                lineage_table = pa.Table.from_pandas(
                    lineage, schema=lineage_schema, preserve_index=False, safe=False
                )
                if lineage_writer is None:
                    lineage_writer = pq.ParquetWriter(
                        lineage_path, lineage_schema, compression="zstd", use_dictionary=True
                    )
                lineage_writer.write_table(lineage_table)

                physical = standardized.loc[standardized.is_physical].copy()
                if not physical.empty:
                    facts = physical[
                        [
                            "physical_fact_id", "cancer_id", "lncrna_id", "partner_id",
                            "relation_type", "experiment_type", "source_database",
                            "source_dataset", "source_record_id", "pmid", "source_kind",
                            "source_input", "source_row_index", "is_experimental",
                        ]
                    ].copy()
                    facts["is_prediction"] = False
                    facts["prediction_available"] = False
                    facts["failure_reason"] = FORMAL_SPLIT_FAILURE_REASON
                    facts["source_occurrence_count"] = 1
                    facts = facts[fact_columns]
                    fact_table = pa.Table.from_pandas(
                        facts, schema=fact_schema, preserve_index=False, safe=False
                    )
                    if fact_writer is None:
                        fact_writer = pq.ParquetWriter(
                            fact_path, fact_schema, compression="zstd", use_dictionary=True
                        )
                    fact_writer.write_table(fact_table)
                    physical_count += len(facts)
    finally:
        if lineage_writer is not None:
            lineage_writer.close()
        if fact_writer is not None:
            fact_writer.close()
    if lineage_writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=lineage_schema), lineage_path)
    if fact_writer is None:
        pq.write_table(pa.Table.from_pylist([], schema=fact_schema), fact_path)
    counts = {
        **{f"raw_{source}_rows": int(count) for source, count in raw_counts.items()},
        "raw_lineage_rows": int(sum(raw_counts.values())),
        "physical_facts": int(physical_count),
    }
    return counts, _finalize_source_connectivity_audit(
        connectivity_chunks, eligible_counts
    )


def _core_feature_dimension(lineage: CoreEmbeddingLineage) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise EvidenceTrainingContractError("pyarrow is required for core schema audit") from exc
    dimension = 0
    for path in lineage.export_paths.values():
        dimension += sum(
            str(column).startswith("core_feature_")
            for column in pq.read_schema(path).names
        )
    if dimension <= 0:
        raise EvidenceTrainingContractError("V3.2 core exports have no feature columns")
    return int(dimension)


def _initialize_formal_blocked_heads(
    *,
    core_embedding_root: Path,
    output_root: Path,
    seed: int,
    event_feature_dim: int,
    hidden_dim: int,
    dropout: float,
) -> tuple[list[CoreEmbeddingLineage], Path, dict[str, Any]]:
    """Create five fresh initial states, with zero optimization steps, for lineage."""

    torch = _require_torch()
    core_lineages: list[CoreEmbeddingLineage] = []
    folds: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        lineage = validate_core_embedding_lineage(core_embedding_root, fold)
        core_lineages.append(lineage)
        fold_seed = seed + fold * 1009
        model = build_fresh_private_eventset_head(
            event_feature_dim=event_feature_dim,
            core_feature_dim=_core_feature_dimension(lineage),
            hidden_dim=hidden_dim,
            dropout=dropout,
            seed=fold_seed,
        )
        parameter_sha = model_parameter_sha256(model)
        fold_root = output_root / f"patient_fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        checkpoint_path = fold_root / "private_eventset_fresh_initialization.pt"
        torch.save(
            {
                **build_fresh_init_contract(
                    patient_fold=fold, seed=fold_seed, core_lineage=lineage
                ),
                "private_model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in model.state_dict().items()
                },
                "initial_parameter_sha256": parameter_sha,
                "final_parameter_sha256": parameter_sha,
                "optimizer_steps": 0,
                "statistical_training_status": "UNAVAILABLE_STRICT_COMPONENT_ISOLATION",
                "contains_core_parameters": False,
                "contains_primary_ranking_parameters": False,
            },
            checkpoint_path,
        )
        folds[str(fold)] = {
            "patient_fold": fold,
            "seed": fold_seed,
            "private_parameters_fresh_init": True,
            "optimizer_steps": 0,
            "initial_parameter_sha256": parameter_sha,
            "final_parameter_sha256": parameter_sha,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "core_checkpoint_sha256": lineage.checkpoint_sha256,
            "core_parameter_sha256": lineage.core_parameter_sha256,
        }
    payload = {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "folds": folds,
        "all_private_heads_freshly_initialized": True,
        "all_optimizer_steps_zero_due_to_strict_split_block": True,
        "historical_checkpoint_loaded": False,
    }
    manifest_path = output_root / "FRESH_HEAD_CHECKPOINT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return core_lineages, manifest_path, payload


def run_formal_evidence_dry_run(
    *,
    evidence_event_path: Path,
    interaction_relation_path: Path,
    pathway_member_path: Path,
    core_embedding_root: Path,
    output_root: Path,
    candidates_path: Path,
    id_map_path: Path | None = None,
    seed: int = 20260726,
    event_feature_dim: int = 128,
    hidden_dim: int = 96,
    dropout: float = 0.20,
    expected_candidate_rows: int | None = FORMAL_CANDIDATE_ROWS,
    expected_cancer_count: int | None = FORMAL_CANCER_COUNT,
    expected_candidate_sha256: str | None = FORMAL_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Publish a full null frame when the legal component split cannot train."""

    paths = {
        "evidence_event": Path(evidence_event_path),
        "interaction_relation": Path(interaction_relation_path),
        "pathway_members": Path(pathway_member_path),
        "candidates": Path(candidates_path),
    }
    for name, path in paths.items():
        if not path.exists():
            raise EvidenceTrainingContractError(f"Missing {name} input: {path}")
    if id_map_path is not None and not Path(id_map_path).is_file():
        raise EvidenceTrainingContractError(f"Missing static ID map: {id_map_path}")
    output = _fresh_output_root(Path(output_root))
    candidate_raw = read_candidate_table(paths["candidates"])
    candidates, candidate_authority = validate_exact_candidate_authority(
        candidate_raw,
        path=paths["candidates"],
        expected_sha256=expected_candidate_sha256,
        expected_rows=expected_candidate_rows,
        expected_cancer_count=expected_cancer_count,
    )
    candidate_count = len(candidates)
    cancer_count = int(candidates.cancer_id.nunique())
    identifier_map = (
        normalize_id_map(read_input_table(Path(id_map_path))) if id_map_path else {}
    )
    members = normalize_exact_pathway_members(
        read_input_table(paths["pathway_members"]), identifier_map
    )
    unique_candidates = candidates[EXACT_KEYS].drop_duplicates(
        ["lncrna_id", "pathway_id"]
    )
    candidate_pathways_by_lnc = {
        str(lnc): frozenset(map(str, group.pathway_id))
        for lnc, group in unique_candidates.groupby("lncrna_id", sort=False)
    }
    member_pathways_by_partner = {
        str(member): frozenset(map(str, group.pathway_id))
        for member, group in members.groupby("member_id", sort=False)
    }
    input_hashes = {
        name: input_path_sha256(path) for name, path in paths.items()
    }
    if id_map_path:
        input_hashes["id_map"] = file_sha256(Path(id_map_path))
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "seed": seed,
        "event_feature_dim": event_feature_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "strict_component_isolation": True,
        "expected_candidate_rows": expected_candidate_rows,
        "expected_cancer_count": expected_cancer_count,
        "expected_candidate_sha256": expected_candidate_sha256,
    }
    training_run_id = "V32-EVIDENCE-FORMAL-" + _json_sha256(
        {"inputs": input_hashes, "config": config}
    )[:16]

    core_lineages, checkpoint_manifest_path, checkpoint_manifest = (
        _initialize_formal_blocked_heads(
            core_embedding_root=Path(core_embedding_root),
            output_root=output,
            seed=seed,
            event_feature_dim=event_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    )
    raw_counts, split_audit = _write_formal_raw_artifacts(
        evidence_event_path=paths["evidence_event"],
        interaction_relation_path=paths["interaction_relation"],
        output_root=output,
        identifier_map=identifier_map,
        candidate_pathways_by_lnc=candidate_pathways_by_lnc,
        member_pathways_by_partner=member_pathways_by_partner,
    )
    split_audit_path = output / "SPLIT_CONNECTIVITY_AUDIT.json"
    split_audit_path.write_text(
        json.dumps(split_audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if split_audit["strict_five_fold_trainable"]:
        raise EvidenceTrainingContractError(
            "Formal unavailable release refused: strict audit found at least five components; "
            "run real five-fold training instead"
        )
    component_count = int(split_audit["connected_component_count"])
    failure_reason = f"STRICT_CONNECTED_COMPONENT_COUNT_{component_count}_LT_{N_FOLDS}"
    public = build_unavailable_prediction_frame(
        candidate_raw,
        training_run_id=training_run_id,
        failure_reason=failure_reason,
    )
    prediction_path = output / "evidence_private_predictions.parquet"
    public.to_parquet(prediction_path, index=False, compression="zstd")

    core_manifest_path = Path(core_embedding_root) / "CORE_EMBEDDING_MANIFEST.json"
    core_manifest_sha = file_sha256(core_manifest_path)
    core_parameter_set_sha = _json_sha256(
        {
            str(lineage.patient_fold): lineage.core_parameter_sha256
            for lineage in core_lineages
        }
    )
    artifacts = [
        {
            "path": str(core_manifest_path),
            "sha256": core_manifest_sha,
            "artifact_kind": "v32_core_checkpoint",
        },
        {
            "path": str(paths["evidence_event"]),
            "sha256": input_hashes["evidence_event"],
            "artifact_kind": "raw_data",
        },
        {
            "path": str(paths["interaction_relation"]),
            "sha256": input_hashes["interaction_relation"],
            "artifact_kind": "raw_data",
        },
        {
            "path": str(paths["pathway_members"]),
            "sha256": input_hashes["pathway_members"],
            "artifact_kind": "annotation",
        },
        {
            "path": str(paths["candidates"]),
            "sha256": input_hashes["candidates"],
            "artifact_kind": "standardized_input",
        },
    ]
    if id_map_path:
        artifacts.append(
            {
                "path": str(id_map_path),
                "sha256": input_hashes["id_map"],
                "artifact_kind": "annotation",
            }
        )
    module_lineage = {
        "module_id": "evidence",
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": training_run_id,
        "training_status": "AUDITED_UNAVAILABLE",
        "statistical_training_status": "UNAVAILABLE_STRICT_COMPONENT_ISOLATION",
        "failure_reason": failure_reason,
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": N_FOLDS,
        "seeds": [seed + fold * 1009 for fold in range(N_FOLDS)],
        "code_sha256": file_sha256(Path(__file__)),
        "config_sha256": _json_sha256(config),
        "input_manifest_sha256": _json_sha256(input_hashes),
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest_path),
        "input_artifacts": artifacts,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": False,
        "private_head_initialized_from_scratch": True,
        "private_head_optimizer_steps": 0,
        "trained_folds": 0,
        "checkpoint_files": 0,
        "release_ready": False,
        "all_probabilities_null": True,
        "all_unavailable_rows_have_reason": True,
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_manifest_sha,
        "core_parameters_before_sha256": core_parameter_set_sha,
        "core_parameters_after_sha256": core_parameter_set_sha,
        "candidate_universe_rows": candidate_count,
        "candidate_universe_cancers": cancer_count,
        "strict_split_audit_sha256": file_sha256(split_audit_path),
        "physical_facts_separate_from_predictions": True,
        "changes_primary_ranking": False,
        "fresh_head_checkpoints": checkpoint_manifest["folds"],
    }
    from cc_hhgt.v32.full_model_contract import (
        validate_module_lineage,
        validate_public_module_frame,
    )

    validate_module_lineage("evidence", module_lineage)
    validate_public_module_frame("evidence", public)
    module_lineage_path = output / "MODULE_LINEAGE.json"
    module_lineage_path.write_text(
        json.dumps(module_lineage, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validation = {
        "full_model_contract_module_lineage": "PASS",
        "full_model_contract_public_frame": "PASS",
        "validated_candidate_rows": candidate_count,
        "validated_cancer_count": cancer_count,
        "no_historical_ranking_columns": True,
        "all_unavailable_probabilities_null": bool(
            public.evidence_confidence_probability.isna().all()
        ),
        "all_failure_reasons_nonempty": bool(
            public.failure_reason.astype(str).str.strip().ne("").all()
        ),
    }
    (output / "FULL_MODEL_CONTRACT_VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    counts = {
        "candidate_rows": candidate_count,
        "candidate_cancers": cancer_count,
        "available_predictions": 0,
        "unavailable_predictions": candidate_count,
        **raw_counts,
    }
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "evidence",
        "training_run_id": training_run_id,
        "run_mode": "FORMAL_UNAVAILABLE_DRY_RUN",
        "execution_status": "SUCCESS",
        "training_status": "AUDITED_UNAVAILABLE",
        "statistical_training_status": "UNAVAILABLE",
        "failure_reason": failure_reason,
        "counts": counts,
        "data_gaps": {
            "strict_connected_component_count": component_count,
            "all_five_leakage_folds_populated": False,
            "confidence_supervision_available": False,
        },
        "split_audit": split_audit,
        "inputs": input_hashes,
        "candidate_authority": candidate_authority,
        "public_prediction_path": str(prediction_path),
        "physical_fact_path": str(output / "physical_interaction_facts.parquet"),
        "event_lineage_path": str(output / "event_lineage.parquet"),
        "main_ranking_modified": False,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "pair_evidence_loaded": False,
        "family_spf_loaded": False,
        "family_to_exact_broadcast_used": False,
    }
    (output / "TRAINING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_evidence_split_preflight(
    *,
    evidence_event_path: Path,
    interaction_relation_path: Path,
    pathway_member_path: Path,
    output_root: Path,
    id_map_path: Path | None = None,
    candidates_path: Path,
    seed: int = 20260726,
    expected_candidate_rows: int | None = FORMAL_CANDIDATE_ROWS,
    expected_cancer_count: int | None = FORMAL_CANCER_COUNT,
    expected_candidate_sha256: str | None = FORMAL_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Audit formal pair folds and per-head provenance exclusion without training.

    This recomputes event bags from raw evidence/interaction inputs and static
    exact-pathway membership.  It writes no checkpoint or prediction and cannot
    be registered as a public module result.
    """

    paths = {
        "evidence_event": Path(evidence_event_path),
        "interaction_relation": Path(interaction_relation_path),
        "pathway_members": Path(pathway_member_path),
        "candidates": Path(candidates_path),
    }
    for name, path in paths.items():
        if not path.exists():
            raise EvidenceTrainingContractError(f"Missing {name} input: {path}")
    if id_map_path is not None and not Path(id_map_path).is_file():
        raise EvidenceTrainingContractError(f"Missing static ID map: {id_map_path}")
    output = _fresh_output_root(Path(output_root))

    candidate_raw = read_candidate_table(paths["candidates"])
    candidates, candidate_authority = validate_exact_candidate_authority(
        candidate_raw,
        path=paths["candidates"],
        expected_sha256=expected_candidate_sha256,
        expected_rows=expected_candidate_rows,
        expected_cancer_count=expected_cancer_count,
    )
    cancer_count = int(candidates.cancer_id.nunique())

    evidence = read_input_table(paths["evidence_event"])
    interaction = read_input_table(paths["interaction_relation"])
    members = read_input_table(paths["pathway_members"])
    identifier_map = read_input_table(Path(id_map_path)) if id_map_path else None
    build = build_exact_event_bags(
        evidence,
        interaction,
        members,
        id_map=identifier_map,
        candidate_universe=candidate_raw,
        evidence_input_name=str(paths["evidence_event"]),
        interaction_input_name=str(paths["interaction_relation"]),
    )
    bag_events = materialize_candidate_events(build.events, candidate_raw)
    split_events = assign_leakage_safe_folds(bag_events, seed=seed)
    split_audit = build_pair_blocked_split_audit(split_events)
    if not split_audit["all_five_folds_populated"]:
        raise EvidenceTrainingContractError(
            "Formal pair-blocked evidence preflight did not populate all five folds"
        )

    fold_audits: dict[str, Any] = {}
    for patient_fold in range(N_FOLDS):
        validation_fold = (patient_fold + 1) % N_FOLDS
        fold_events, provenance_audit = exclude_evaluation_provenance_from_training(
            split_events,
            evaluation_folds={patient_fold, validation_fold},
        )
        fold_values = pd.to_numeric(
            fold_events["leakage_fold"], errors="raise"
        ).astype(int)
        train_mask = ~fold_values.isin({patient_fold, validation_fold})
        validation_mask = fold_values.eq(validation_fold)
        heldout_mask = fold_values.eq(patient_fold)

        def pair_count(mask: pd.Series) -> int:
            return int(fold_events.loc[mask, EXACT_KEYS].drop_duplicates().shape[0])

        fold_audits[str(patient_fold)] = {
            "patient_fold": patient_fold,
            "validation_fold": validation_fold,
            "train_bags_after_provenance_exclusion": pair_count(train_mask),
            "validation_bags": pair_count(validation_mask),
            "heldout_bags": pair_count(heldout_mask),
            "train_confidence_supervised_event_rows": int(
                pd.to_numeric(
                    fold_events.loc[train_mask, "confidence_target"], errors="coerce"
                ).notna().sum()
            ),
            "train_direction_supervised_event_rows": int(
                pd.to_numeric(
                    fold_events.loc[train_mask, "direction_target"], errors="coerce"
                ).isin([0, 1, 2]).sum()
            ),
            "provenance_exclusion_audit": provenance_audit,
        }
    viable = all(
        fold["train_bags_after_provenance_exclusion"] > 0
        and fold["validation_bags"] > 0
        and fold["heldout_bags"] > 0
        and fold["provenance_exclusion_audit"][
            "residual_train_evaluation_provenance_overlap_count"
        ]
        == 0
        for fold in fold_audits.values()
    )
    if not viable:
        raise EvidenceTrainingContractError(
            "Formal pair-blocked evidence preflight left a non-viable fold"
        )

    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "evidence_private_eventset",
        "status": "PREFLIGHT_SUCCESS",
        "training_status": "PREFLIGHT_ONLY_NO_TRAINING",
        "release_ready": False,
        "partial_not_publishable": True,
        "predictions_written": 0,
        "checkpoints_written": 0,
        "optimizer_steps": 0,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "old_ranking_loaded": False,
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "evaluation_provenance_exclusion_policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
        "inputs": {
            name: {"path": str(path), "sha256": input_path_sha256(path)}
            for name, path in paths.items()
        },
        "candidate_authority": candidate_authority,
        "optional_id_map": (
            {
                "path": str(id_map_path),
                "sha256": file_sha256(Path(id_map_path)),
            }
            if id_map_path
            else None
        ),
        "counts": {
            "candidate_rows": int(len(candidates)),
            "candidate_cancers": cancer_count,
            "canonical_events": int(len(build.events)),
            "materialized_bag_event_rows": int(len(split_events)),
            "physical_facts": int(len(build.physical_facts)),
            "rejected_event_mappings": int(len(build.rejected)),
        },
        "split_audit": split_audit,
        "folds": fold_audits,
    }
    manifest_path = output / "SPLIT_PREFLIGHT.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    success = {
        "status": "PREFLIGHT_SUCCESS",
        "release_ready": False,
        "partial_not_publishable": True,
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (output / "PREFLIGHT_SUCCESS.json").write_text(
        json.dumps(success, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_evidence_split_preflight_gate(
    manifest_path: Path,
    *,
    evidence_event_path: Path,
    interaction_relation_path: Path,
    pathway_member_path: Path,
    candidates_path: Path,
    expected_candidate_sha256: str | None = FORMAL_CANDIDATE_SHA256,
) -> dict[str, Any]:
    """Independently verify that a no-training preflight authorizes training."""

    path = Path(manifest_path)
    if not path.is_file() or path.name != "SPLIT_PREFLIGHT.json":
        raise EvidenceTrainingContractError(
            f"Formal Evidence training requires SPLIT_PREFLIGHT.json: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceTrainingContractError(f"Invalid split preflight manifest: {path}") from exc
    required = {
        "status": "PREFLIGHT_SUCCESS",
        "training_status": "PREFLIGHT_ONLY_NO_TRAINING",
        "release_ready": False,
        "partial_not_publishable": True,
        "predictions_written": 0,
        "checkpoints_written": 0,
        "optimizer_steps": 0,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "old_ranking_loaded": False,
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "evaluation_provenance_exclusion_policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
    }
    for field, expected in required.items():
        observed = payload.get(field)
        if observed != expected or type(observed) is not type(expected):
            raise EvidenceTrainingContractError(
                f"Split preflight gate has invalid {field}: {observed!r}"
            )

    success_path = path.parent / "PREFLIGHT_SUCCESS.json"
    if not success_path.is_file():
        raise EvidenceTrainingContractError("Split preflight has no success marker")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        success.get("status") != "PREFLIGHT_SUCCESS"
        or success.get("release_ready") is not False
        or success.get("partial_not_publishable") is not True
        or success.get("manifest_sha256") != file_sha256(path)
    ):
        raise EvidenceTrainingContractError("Split preflight success marker is invalid or stale")

    actual_inputs = {
        "evidence_event": Path(evidence_event_path),
        "interaction_relation": Path(interaction_relation_path),
        "pathway_members": Path(pathway_member_path),
        "candidates": Path(candidates_path),
    }
    declared_inputs = payload.get("inputs")
    if not isinstance(declared_inputs, Mapping):
        raise EvidenceTrainingContractError("Split preflight has no hashed inputs")
    for name, input_path in actual_inputs.items():
        declared = declared_inputs.get(name)
        if not isinstance(declared, Mapping):
            raise EvidenceTrainingContractError(f"Split preflight lacks input {name}")
        observed_sha = input_path_sha256(input_path)
        if declared.get("sha256") != observed_sha:
            raise EvidenceTrainingContractError(
                f"Split preflight input SHA-256 drift for {name}"
            )

    authority = payload.get("candidate_authority")
    if not isinstance(authority, Mapping) or authority.get("sha256_pinned") is not True:
        raise EvidenceTrainingContractError("Split preflight candidate authority was not pinned")
    candidate_sha = input_path_sha256(Path(candidates_path))
    if authority.get("sha256") != candidate_sha:
        raise EvidenceTrainingContractError("Split preflight candidate authority SHA-256 drift")
    if expected_candidate_sha256 is not None and candidate_sha != expected_candidate_sha256:
        raise EvidenceTrainingContractError(
            "Formal Evidence candidate is not the authoritative exact V3.2 universe"
        )

    split_audit = payload.get("split_audit")
    if (
        not isinstance(split_audit, Mapping)
        or split_audit.get("hard_pair_cross_fold_count") != 0
        or split_audit.get("all_five_folds_populated") is not True
        or sorted(split_audit.get("active_folds", [])) != list(range(N_FOLDS))
    ):
        raise EvidenceTrainingContractError("Split preflight pair-isolation proof failed")
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or set(folds) != {str(fold) for fold in range(N_FOLDS)}:
        raise EvidenceTrainingContractError("Split preflight does not contain five fold audits")
    for fold_id in range(N_FOLDS):
        fold = folds[str(fold_id)]
        provenance = fold.get("provenance_exclusion_audit", {})
        positive_counts = (
            fold.get("train_bags_after_provenance_exclusion", 0),
            fold.get("validation_bags", 0),
            fold.get("heldout_bags", 0),
            fold.get("train_confidence_supervised_event_rows", 0),
        )
        if any(not isinstance(value, int) or value <= 0 for value in positive_counts):
            raise EvidenceTrainingContractError(
                f"Split preflight fold {fold_id} has a non-positive training/evaluation count"
            )
        for field in (
            "residual_pmid_overlap_count",
            "residual_source_event_overlap_count",
            "residual_source_record_overlap_count",
            "residual_train_evaluation_provenance_overlap_count",
        ):
            if provenance.get(field) != 0:
                raise EvidenceTrainingContractError(
                    f"Split preflight fold {fold_id} has residual provenance leakage: {field}"
                )

    forbidden_files = [
        item
        for item in path.parent.rglob("*")
        if item.is_file()
        and (
            item.suffix.lower() in {".pt", ".pth", ".ckpt", ".parquet"}
            or "prediction" in item.name.lower()
            or "checkpoint" in item.name.lower()
        )
    ]
    if forbidden_files:
        raise EvidenceTrainingContractError(
            "Split preflight directory contains prediction/checkpoint artifacts"
        )
    if int(payload.get("counts", {}).get("physical_facts", 0)) <= 0:
        raise EvidenceTrainingContractError("Split preflight retained no physical interaction facts")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "success_marker_path": str(success_path),
        "success_marker_sha256": file_sha256(success_path),
        "pair_isolation_verified": True,
        "five_fold_provenance_exclusion_verified": True,
        "no_prediction_or_checkpoint_artifacts_verified": True,
        "candidate_sha256": candidate_sha,
    }


def run_evidence_training(
    *,
    evidence_event_path: Path,
    interaction_relation_path: Path,
    pathway_member_path: Path,
    core_embedding_root: Path,
    output_root: Path,
    id_map_path: Path | None = None,
    candidates_path: Path | None = None,
    split_preflight_manifest_path: Path | None = None,
    expected_candidate_rows: int | None = FORMAL_CANDIDATE_ROWS,
    expected_cancer_count: int | None = FORMAL_CANCER_COUNT,
    expected_candidate_sha256: str | None = FORMAL_CANDIDATE_SHA256,
    seed: int = 20260726,
    epochs: int = 40,
    patience: int = 6,
    batch_size: int = 128,
    max_events: int = 64,
    event_feature_dim: int = 128,
    hidden_dim: int = 96,
    dropout: float = 0.20,
    mc_samples: int = 16,
    preserve_assay_type: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """Run the complete five-fold fresh V3.2 private evidence workflow."""

    torch = _require_torch()
    paths = {
        "evidence_event": Path(evidence_event_path),
        "interaction_relation": Path(interaction_relation_path),
        "pathway_members": Path(pathway_member_path),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise EvidenceTrainingContractError(f"Missing {name} input: {path}")
    if id_map_path is not None and not Path(id_map_path).is_file():
        raise EvidenceTrainingContractError(f"Missing static ID map: {id_map_path}")
    if candidates_path is None or not Path(candidates_path).exists():
        raise EvidenceTrainingContractError(
            f"Formal Evidence training requires an exact candidate universe: {candidates_path}"
        )
    if split_preflight_manifest_path is None:
        raise EvidenceTrainingContractError(
            "Formal Evidence training requires a successful pair-blocked split preflight"
        )
    preflight_gate = validate_evidence_split_preflight_gate(
        Path(split_preflight_manifest_path),
        evidence_event_path=paths["evidence_event"],
        interaction_relation_path=paths["interaction_relation"],
        pathway_member_path=paths["pathway_members"],
        candidates_path=Path(candidates_path),
        expected_candidate_sha256=expected_candidate_sha256,
    )
    output = _fresh_output_root(Path(output_root))

    evidence = read_input_table(paths["evidence_event"])
    interaction = read_input_table(paths["interaction_relation"])
    members = read_input_table(paths["pathway_members"])
    identifier_map = read_input_table(Path(id_map_path)) if id_map_path else None
    candidate_raw = read_candidate_table(Path(candidates_path))
    candidates, candidate_authority = validate_exact_candidate_authority(
        candidate_raw,
        path=Path(candidates_path),
        expected_sha256=expected_candidate_sha256,
        expected_rows=expected_candidate_rows,
        expected_cancer_count=expected_cancer_count,
    )
    build = build_exact_event_bags(
        evidence,
        interaction,
        members,
        id_map=identifier_map,
        candidate_universe=candidate_raw,
        evidence_input_name=str(paths["evidence_event"]),
        interaction_input_name=str(paths["interaction_relation"]),
    )
    bag_events = materialize_candidate_events(build.events, candidate_raw)
    split_events = assign_leakage_safe_folds(bag_events, seed=seed)
    split_integrity_audit = build_pair_blocked_split_audit(split_events)
    split_integrity_audit_path = output / "PAIR_BLOCKED_SPLIT_AUDIT.json"
    split_integrity_audit_path.write_text(
        json.dumps(split_integrity_audit, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    active_split_folds = sorted(
        set(pd.to_numeric(split_events.leakage_fold, errors="coerce").dropna().astype(int))
    ) if not split_events.empty else []
    if len(active_split_folds) != N_FOLDS:
        raise EvidenceTrainingContractError(
            "Pair-blocked lncRNA--exact-pathway assignment did not populate five folds"
        )

    source_event_col = "source_event_id" if "source_event_id" in split_events else "event_id"
    lineage_summary = (
        build.lineage.groupby("event_id", observed=True)
        .agg(
            raw_lineage_ids=("lineage_id", lambda values: ";".join(sorted(set(map(str, values))))),
            raw_lineage_count=("lineage_id", "nunique"),
            family_broadcast_used=("family_broadcast_used", "any"),
        )
        .reset_index()
        if not build.lineage.empty
        else pd.DataFrame(columns=["event_id", "raw_lineage_ids", "raw_lineage_count", "family_broadcast_used"])
    )
    event_lineage = split_events.merge(
        lineage_summary,
        left_on=source_event_col,
        right_on="event_id",
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    if "event_id_source" in event_lineage:
        event_lineage = event_lineage.drop(columns="event_id_source")
    if event_lineage.get("family_broadcast_used", pd.Series(False, index=event_lineage.index)).fillna(False).any():
        raise EvidenceTrainingContractError("Family-to-exact broadcasting entered event lineage")

    _safe_parquet_frame(event_lineage).to_parquet(
        output / "event_lineage.parquet", index=False, compression="zstd"
    )
    _safe_parquet_frame(build.physical_facts).to_parquet(
        output / "physical_interaction_facts.parquet", index=False, compression="zstd"
    )
    _safe_parquet_frame(build.rejected).to_parquet(
        output / "rejected_event_mappings.parquet", index=False, compression="zstd"
    )

    all_predictions: list[pd.DataFrame] = []
    fold_manifests: dict[str, Any] = {}
    missing_core_frames: list[pd.DataFrame] = []
    for patient_fold in range(N_FOLDS):
        core = load_core_feature_bundle(core_embedding_root, patient_fold)
        validation_fold = (patient_fold + 1) % N_FOLDS
        fold_events, provenance_exclusion_audit = (
            exclude_evaluation_provenance_from_training(
                split_events,
                evaluation_folds={patient_fold, validation_fold},
            )
        )
        examples, missing_core = build_bag_examples(
            fold_events,
            core,
            event_feature_dim=event_feature_dim,
            max_events=max_events,
            preserve_assay_type=preserve_assay_type,
        )
        if not missing_core.empty:
            missing_core["patient_fold"] = patient_fold
            missing_core_frames.append(missing_core)
        train_examples = [
            example
            for example in examples
            if example.leakage_fold not in {patient_fold, validation_fold}
        ]
        validation_examples = [
            example for example in examples if example.leakage_fold == validation_fold
        ]
        heldout_examples = [
            example for example in examples if example.leakage_fold == patient_fold
        ]
        fold_seed = seed + patient_fold * 1009
        fresh_contract = build_fresh_init_contract(
            patient_fold=patient_fold,
            seed=fold_seed,
            core_lineage=core.lineage,
        )
        fit = fit_private_eventset_head(
            train_examples,
            validation_examples,
            event_feature_dim=event_feature_dim,
            core_feature_dim=core.combined_dim,
            seed=fold_seed,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            device=device,
        )
        if fit.optimizer_steps <= 0:
            raise EvidenceTrainingContractError(
                f"Evidence fold {patient_fold} performed no optimizer steps"
            )
        if not fit.confidence_supervision_available:
            raise EvidenceTrainingContractError(
                f"Evidence fold {patient_fold} has no confidence supervision"
            )
        if fit.initial_parameter_sha256 == fit.final_parameter_sha256:
            raise EvidenceTrainingContractError(
                f"Evidence fold {patient_fold} parameters did not change"
            )
        fold_predictions = predict_private_eventset_head(
            fit,
            heldout_examples,
            batch_size=batch_size,
            mc_samples=mc_samples,
            device=device,
        )
        all_predictions.append(fold_predictions)
        fold_root = output / f"patient_fold={patient_fold}"
        fold_root.mkdir(parents=True, exist_ok=False)
        checkpoint_path = fold_root / "private_eventset_state.pt"
        checkpoint_payload = {
            **fresh_contract,
            "private_model_state": {
                name: tensor.detach().cpu() for name, tensor in fit.model.state_dict().items()
            },
            "contains_core_parameters": False,
            "contains_primary_ranking_parameters": False,
            "initial_parameter_sha256": fit.initial_parameter_sha256,
            "final_parameter_sha256": fit.final_parameter_sha256,
            "optimizer_steps": fit.optimizer_steps,
            "confidence_supervision_available": fit.confidence_supervision_available,
            "direction_supervision_available": fit.direction_supervision_available,
            "provenance_exclusion_audit": provenance_exclusion_audit,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        history_path = fold_root / "training_history.tsv"
        pd.DataFrame(fit.history).to_csv(history_path, sep="\t", index=False)
        fold_manifests[str(patient_fold)] = {
            **fresh_contract,
            "initial_parameter_sha256": fit.initial_parameter_sha256,
            "final_parameter_sha256": fit.final_parameter_sha256,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "optimizer_steps": fit.optimizer_steps,
            "n_train_bags": len(train_examples),
            "n_validation_bags": len(validation_examples),
            "n_heldout_bags": len(heldout_examples),
            "n_missing_core_bags": int(len(missing_core)),
            "confidence_supervision_available": fit.confidence_supervision_available,
            "direction_supervision_available": fit.direction_supervision_available,
            "provenance_exclusion_audit": provenance_exclusion_audit,
        }

    predictions = (
        pd.concat(all_predictions, ignore_index=True)
        if all_predictions
        else pd.DataFrame(columns=EXACT_KEYS)
    )
    if missing_core_frames:
        missing_core_output = pd.concat(missing_core_frames, ignore_index=True)
        _safe_parquet_frame(missing_core_output).to_parquet(
            output / "missing_core_identifiers.parquet", index=False, compression="zstd"
        )
    else:
        missing_core_output = pd.DataFrame()
    training_run_id = "V32-EVIDENCE-TRAIN-" + _json_sha256(
        {
            "seed": seed,
            "evidence_event": file_sha256(paths["evidence_event"]),
            "interaction_relation": file_sha256(paths["interaction_relation"]),
            "pathway_members": file_sha256(paths["pathway_members"]),
            "candidates": (
                input_path_sha256(Path(candidates_path)) if candidates_path else None
            ),
        }
    )[:16]
    complete = complete_prediction_frame(
        candidate_raw,
        predictions,
        split_events,
        training_run_id=training_run_id,
    )
    complete.to_parquet(
        output / "evidence_private_predictions.parquet", index=False, compression="zstd"
    )

    rejection_counts = (
        build.rejected.rejection_reason.value_counts().to_dict()
        if not build.rejected.empty and "rejection_reason" in build.rejected
        else {}
    )
    active_folds = sorted(
        set(pd.to_numeric(split_events.leakage_fold, errors="coerce").dropna().astype(int))
    ) if not split_events.empty else []
    manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": "evidence_private_eventset",
        "status": "SUCCESS_NEWLY_TRAINED",
        "training_status": "SUCCESS_NEWLY_TRAINED",
        "release_ready": False,
        "partial_not_publishable": True,
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "training_generation": "V3.2",
        "five_fresh_private_heads_attempted": True,
        "historical_evidence_checkpoint_loaded": False,
        "historical_evidence_result_loaded": False,
        "old_confidence_loaded": False,
        "pair_evidence_loaded": False,
        "family_spf_loaded": False,
        "family_to_exact_broadcast_used": False,
        "core_frozen": True,
        "core_detached": True,
        "physical_facts_separate_from_predictions": True,
        "main_ranking_modified": False,
        "prediction_role": "AUXILIARY_CONFIDENCE_ONLY",
        "split_policy": EVIDENCE_SPLIT_POLICY,
        "evaluation_provenance_exclusion_policy": EVALUATION_PROVENANCE_EXCLUSION_POLICY,
        "split_integrity_audit": {
            "path": str(split_integrity_audit_path),
            "sha256": file_sha256(split_integrity_audit_path),
            **split_integrity_audit,
        },
        "split_preflight_gate": preflight_gate,
        "candidate_authority": candidate_authority,
        "inputs": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()
        },
        "optional_inputs": {
            "id_map": (
                {"path": str(id_map_path), "sha256": file_sha256(Path(id_map_path))}
                if id_map_path
                else None
            ),
            "candidates": (
                {"path": str(candidates_path), "sha256": input_path_sha256(Path(candidates_path))}
                if candidates_path
                else None
            ),
        },
        "counts": {
            "candidate_rows": int(len(candidates)),
            "candidate_cancers": int(candidates.cancer_id.nunique()),
            "canonical_events": int(len(build.events)),
            "materialized_bag_events": int(len(split_events)),
            "physical_facts": int(len(build.physical_facts)),
            "raw_lineage_rows": int(len(build.lineage)),
            "rejected_rows": int(len(build.rejected)),
            "available_predictions": int(complete.availability.sum()),
            "unavailable_predictions": int((~complete.availability).sum()),
            "optimizer_steps": int(
                sum(int(fold["optimizer_steps"]) for fold in fold_manifests.values())
            ),
            "trained_private_heads": int(
                sum(int(fold["optimizer_steps"]) > 0 for fold in fold_manifests.values())
            ),
        },
        "folds": fold_manifests,
        "data_gaps": {
            "rejection_reasons": {str(key): int(value) for key, value in rejection_counts.items()},
            "active_leakage_folds": active_folds,
            "all_five_leakage_folds_populated": len(active_folds) == N_FOLDS,
            "missing_core_identifier_bags": int(len(missing_core_output)),
            "confidence_supervised_events": int(
                pd.to_numeric(split_events.get("confidence_target"), errors="coerce").notna().sum()
                if not split_events.empty
                else 0
            ),
            "direction_supervised_events": int(
                pd.to_numeric(split_events.get("direction_target"), errors="coerce").isin([0, 1, 2]).sum()
                if not split_events.empty
                else 0
            ),
        },
    }
    manifest_path = output / "TRAINING_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    success = {
        "status": "SUCCESS_NEWLY_TRAINED",
        "release_ready": False,
        "partial_not_publishable": True,
        "trained_private_heads": manifest["counts"]["trained_private_heads"],
        "optimizer_steps": manifest["counts"]["optimizer_steps"],
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
    }
    (output / "TRAINING_SUCCESS.json").write_text(
        json.dumps(success, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
