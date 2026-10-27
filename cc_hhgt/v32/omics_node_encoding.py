"""Multi-omics node channels for the typed V3.2 graph encoder.

The question this module answers in code is narrow and checkable: can measured
mutation, signed CNV and promoter accessibility enter the graph *encoder*, i.e.
the first layer whose output is propagated by message passing, instead of being
attached to the decoder as a post-hoc expert?

The contract is:

* An evidence table records one standardised value per
  ``(node_type, canonical_id, channel)`` for one declared scope
  (``cohort`` = cancer-level aggregate, ``subject`` = one patient).
* Standardisation parameters are frozen in the manifest.  Nothing is fitted
  here, so the module cannot leak a training-set statistic into a fold.
* A node type without any measurement is *not* zero-filled in the graph layer.
  This module emits an explicit ``observed`` flag per channel, and the missing
  value stays exactly ``0.0``.  ``cc_hhgt.gnn.validate_node_omics`` then refuses
  to materialise a graph when a node type has no declared block at all.
* ``attach_node_omics`` widens every encoder input from the two historical
  degree columns to ``2 + 2 * len(channels)`` and re-verifies the result by
  reading the written tensors back.

What this module does **not** do, and what no fixture in this repository can
prove: that a multi-omics-conditioned graph predicts better than the current
G2 graph, that a patient's protein loss changes a real lncRNA-pathway relation,
that a protein is a mediating interaction node, or that a candidate binding edge
is biologically true.  It only makes multi-omics a legal graph input and records
the exact byte-level provenance of that input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

FORMAT = "CANCERLNCATLAS_V32_OMICS_NODE_ENCODING_V1"
EVIDENCE_FORMAT = "CANCERLNCATLAS_V32_OMICS_NODE_EVIDENCE_V1"
BLOCK_FORMAT = "CANCERLNCATLAS_V32_OMICS_NODE_BLOCK_V1"
CHANNELS = ("mutation_damage", "cnv_signed", "atac_accessibility")
SCOPES = ("cohort", "subject")
DEGREE_CHANNELS = 2
CLIP = 8.0
TOLERANCE = 1e-9
REQUIRED_EVIDENCE_COLUMNS = ("node_type", "canonical_id", "channel", "raw_value")


@dataclass(frozen=True)
class OmicsEvidence:
    """Validated evidence table plus the manifest that pins it."""

    frame: pd.DataFrame
    standardised: pd.DataFrame
    manifest: Mapping[str, Any]
    channels: tuple[str, ...]

    @property
    def scope(self) -> str:
        return str(self.manifest["scope"])

    @property
    def cancer_id(self) -> str:
        return str(self.manifest["cancer_id"])

    @property
    def subject_id(self) -> str:
        return str(self.manifest.get("subject_id") or "")

    @property
    def width(self) -> int:
        return len(self.channels) * 2


def channel_layout(channels: Iterable[str]) -> list[str]:
    """Column names of the appended block, in encoder input order."""

    return [
        f"{channel}::{field}"
        for channel in channels
        for field in ("value", "observed")
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(Path(path).suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    # Empty identifiers must stay empty instead of becoming a NaN that a
    # string cast would turn into the plausible-looking id "nan".
    if suffixes.endswith((".tsv", ".tsv.gz", ".txt")):
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    if suffixes.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, keep_default_na=False)
    raise RuntimeError(f"OMICS_EVIDENCE_TABLE_FORMAT_UNSUPPORTED: {path.name}")


def read_evidence(table_path, manifest_path) -> OmicsEvidence:
    """Load and validate one omics node evidence pair; fail closed on drift."""

    table_path = Path(table_path)
    manifest_path = Path(manifest_path)
    if not table_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"OMICS_EVIDENCE_FILE_MISSING: {table_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("format") != EVIDENCE_FORMAT:
        raise RuntimeError("OMICS_EVIDENCE_FORMAT_INVALID")
    scope = manifest.get("scope")
    if scope not in SCOPES:
        raise RuntimeError(f"OMICS_EVIDENCE_SCOPE_INVALID: {scope!r}")
    cancer_id = str(manifest.get("cancer_id") or "")
    if not cancer_id:
        raise RuntimeError("OMICS_EVIDENCE_CANCER_ID_MISSING")
    subject_id = str(manifest.get("subject_id") or "")
    if scope == "subject" and not subject_id:
        raise RuntimeError("OMICS_EVIDENCE_SUBJECT_ID_MISSING")
    if scope == "cohort" and subject_id:
        raise RuntimeError("OMICS_EVIDENCE_SUBJECT_ID_NOT_ALLOWED_FOR_COHORT")
    channels = tuple(manifest.get("channels") or ())
    if not channels or len(set(channels)) != len(channels):
        raise RuntimeError(f"OMICS_EVIDENCE_CHANNELS_INVALID: {channels}")
    unknown = sorted(set(channels) - set(CHANNELS))
    if unknown:
        raise RuntimeError(f"OMICS_EVIDENCE_CHANNEL_UNKNOWN: {unknown}")
    standardisation = manifest.get("standardisation") or {}
    for channel in channels:
        parameters = standardisation.get(channel)
        if not isinstance(parameters, Mapping):
            raise RuntimeError(f"OMICS_EVIDENCE_CENTER_INVALID: {channel}")
        center = parameters.get("center")
        scale = parameters.get("scale")
        if isinstance(center, bool) or not isinstance(center, (int, float)) or not np.isfinite(float(center)):
            raise RuntimeError(f"OMICS_EVIDENCE_CENTER_INVALID: {channel}")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not np.isfinite(float(scale))
            or float(scale) <= 0
        ):
            raise RuntimeError(f"OMICS_EVIDENCE_SCALE_INVALID: {channel}")
    if manifest.get("clipped") not in (True, None):
        raise RuntimeError("OMICS_EVIDENCE_CLIP_DECLARATION_INVALID")
    expected_sha = str(manifest.get("table_sha256") or "")
    if not expected_sha:
        raise RuntimeError("OMICS_EVIDENCE_TABLE_SHA256_MISSING")
    if _sha256(table_path) != expected_sha:
        raise RuntimeError("OMICS_EVIDENCE_TABLE_SHA256_MISMATCH")
    frame = _read_table(table_path)
    missing = [name for name in REQUIRED_EVIDENCE_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"OMICS_EVIDENCE_COLUMNS_MISSING: {missing}")
    declared_rows = manifest.get("row_count")
    if declared_rows is not None and int(declared_rows) != len(frame):
        raise RuntimeError(
            f"OMICS_EVIDENCE_ROW_COUNT_MISMATCH: {int(declared_rows)} != {len(frame)}"
        )
    frame = frame.loc[:, list(REQUIRED_EVIDENCE_COLUMNS)].copy()
    if frame[["node_type", "canonical_id", "channel"]].isna().any().any():
        raise RuntimeError("OMICS_EVIDENCE_EMPTY_IDENTIFIER")
    frame["node_type"] = frame.node_type.astype(str).str.strip()
    frame["canonical_id"] = frame.canonical_id.astype(str).str.strip()
    frame["channel"] = frame.channel.astype(str).str.strip()
    if (frame.node_type.eq("") | frame.canonical_id.eq("") | frame.channel.eq("")).any():
        raise RuntimeError("OMICS_EVIDENCE_EMPTY_IDENTIFIER")
    unexpected = sorted(set(frame.channel) - set(channels))
    if unexpected:
        raise RuntimeError(f"OMICS_EVIDENCE_CHANNEL_NOT_DECLARED: {unexpected}")
    values = pd.to_numeric(frame.raw_value, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise RuntimeError("OMICS_EVIDENCE_RAW_VALUE_INVALID")
    frame["raw_value"] = values.astype(float)
    duplicated = frame.duplicated(["node_type", "canonical_id", "channel"], keep=False)
    if duplicated.any():
        example = frame.loc[duplicated, ["node_type", "canonical_id", "channel"]].head(3)
        raise RuntimeError(
            "OMICS_EVIDENCE_DUPLICATE_NODE_CHANNEL: "
            + example.to_dict(orient="records").__repr__()
        )
    standardised = frame.copy()
    centers = standardised.channel.map(
        {channel: float(standardisation[channel]["center"]) for channel in channels}
    ).astype(float)
    scales = standardised.channel.map(
        {channel: float(standardisation[channel]["scale"]) for channel in channels}
    ).astype(float)
    standardised["value"] = np.clip(
        (standardised.raw_value.astype(float) - centers.to_numpy(float))
        / scales.to_numpy(float),
        -CLIP,
        CLIP,
    )
    return OmicsEvidence(
        frame=frame, standardised=standardised, manifest=manifest, channels=channels
    )


def _node_order(nodes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per node type frames in exactly the encoder's row order."""

    if "node_index_within_type" not in nodes.columns:
        raise RuntimeError("OMICS_NODES_LACK_INDEX_COLUMN")
    for column in ("node_type", "canonical_id"):
        if column not in nodes.columns:
            raise RuntimeError(f"OMICS_NODES_LACK_COLUMN: {column}")
    order: dict[str, pd.DataFrame] = {}
    for node_type, group in nodes.groupby("node_type", sort=False):
        group = group.sort_values("node_index_within_type", kind="stable")
        index = group.node_index_within_type.to_numpy(np.int64)
        if len(np.unique(index)) != len(index):
            raise RuntimeError(f"OMICS_NODE_INDEX_DUPLICATE: {node_type}")
        canonical = group.canonical_id.astype(str)
        if canonical.duplicated().any():
            raise RuntimeError(f"OMICS_NODE_CANONICAL_ID_DUPLICATE: {node_type}")
        order[str(node_type)] = group
    return order


def node_feature_block(nodes: pd.DataFrame, evidence: OmicsEvidence) -> dict[str, Any]:
    """Assemble the per-node-type appended block, row-aligned to the encoder.

    Unmeasured ``(node, channel)`` pairs keep ``value = 0.0`` and
    ``observed = 0.0``.  They are never imputed from another patient, another
    cancer or a cohort mean: the flag exists so a downstream reader can tell
    "measured zero" from "not measured".
    """

    order = _node_order(nodes)
    channels = evidence.channels
    frame = evidence.standardised
    known = {
        (node_type, str(node_id))
        for node_type, group in order.items()
        for node_id in group.canonical_id.astype(str)
    }
    positions = {
        (str(row.node_type), str(row.canonical_id), str(row.channel)): float(row.value)
        for row in frame.itertuples(index=False)
    }
    unknown_pairs = sorted(
        {
            f"{node_type}:{node_id}"
            for (node_type, node_id, channel) in positions
            if (node_type, node_id) not in known
        }
    )
    if unknown_pairs:
        raise RuntimeError(
            "OMICS_EVIDENCE_NODE_NOT_IN_GRAPH: " + ", ".join(unknown_pairs[:5])
        )
    covered: dict[str, int] = {}
    blocks: dict[str, np.ndarray] = {}
    for node_type, group in order.items():
        block = np.zeros((len(group), len(channels) * 2), dtype=np.float64)
        hit = 0
        for row_index, node_id in enumerate(group.canonical_id.astype(str)):
            for channel_index, channel in enumerate(channels):
                value = positions.get((node_type, node_id, channel))
                if value is None:
                    continue
                block[row_index, channel_index * 2] = value
                block[row_index, channel_index * 2 + 1] = 1.0
                hit += 1
        blocks[node_type] = block
        covered[node_type] = hit
    report = {
        "format": BLOCK_FORMAT,
        "scope": evidence.scope,
        "cancer_id": evidence.cancer_id,
        "subject_id": evidence.subject_id,
        "channels": list(channels),
        "columns": channel_layout(channels),
        "width": len(channels) * 2,
        "node_feature_width": DEGREE_CHANNELS + len(channels) * 2,
        "evidence_table_sha256": str(evidence.manifest.get("table_sha256")),
        "evidence_row_count": int(len(frame)),
        "node_types": sorted(blocks),
        "covered_pairs": covered,
        "nodes": {node_type: int(len(group)) for node_type, group in order.items()},
    }
    return {"blocks": blocks, "report": report}


def _bundle_order(bundle) -> dict[str, list[str]]:
    """Read back the canonical id order actually stored in the bundle."""

    order: dict[str, list[str]] = {}
    for node_type in bundle.hetero_data.node_types:
        ids = getattr(bundle.hetero_data[str(node_type)], "canonical_ids", None)
        if ids is None:
            raise RuntimeError(f"OMICS_BUNDLE_LACKS_CANONICAL_IDS: {node_type}")
        order[str(node_type)] = [str(value) for value in ids]
    return order


def attach_node_omics(bundle, blocks: Mapping[str, Any]) -> dict[str, Any]:
    """Widen the encoder inputs of an existing bundle in place.

    The bundle must be freshly materialised: this is the single moment in the
    pipeline where the two degree columns are the whole node input.  Every
    written tensor is read back and compared, so a silently dropped column or a
    misordered row raises instead of training on garbage.
    """

    import torch

    order = _bundle_order(bundle)
    graph_types = set(order)
    missing = sorted(graph_types - set(map(str, blocks)))
    if missing:
        raise RuntimeError(f"OMICS_NODE_TYPE_BLOCK_MISSING: {missing}")
    unknown = sorted(set(map(str, blocks)) - graph_types)
    if unknown:
        raise RuntimeError(f"OMICS_NODE_TYPE_BLOCK_UNKNOWN: {unknown}")
    widths = {int(np.asarray(blocks[str(node_type)]).shape[1]) for node_type in graph_types}
    if len(widths) != 1:
        raise RuntimeError(f"OMICS_BLOCK_WIDTH_INCONSISTENT: {sorted(widths)}")
    width = int(next(iter(widths)))
    if width == 0:
        raise RuntimeError("OMICS_BLOCK_EMPTY")
    node_input_dim = DEGREE_CHANNELS + width
    existing = {
        node_type: bundle.hetero_data[str(node_type)].x.detach().cpu().numpy()
        for node_type in graph_types
    }
    for node_type, value in existing.items():
        if value.ndim != 2 or value.shape[1] != DEGREE_CHANNELS:
            raise RuntimeError(
                f"OMICS_BUNDLE_ALREADY_AUGMENTED: {node_type} {value.shape}"
            )
        if value.shape[0] != len(order[node_type]):
            raise RuntimeError(f"OMICS_BUNDLE_ROW_MISMATCH: {node_type}")
    for node_type, node_ids in order.items():
        block = np.asarray(blocks[node_type], dtype=np.float64)
        if block.shape != (len(node_ids), width):
            raise RuntimeError(
                f"OMICS_BLOCK_ROW_MISMATCH: {node_type} {block.shape} != {(len(node_ids), width)}"
            )
        if not np.isfinite(block).all():
            raise RuntimeError(f"OMICS_BLOCK_NOT_FINITE: {node_type}")
    written: dict[str, np.ndarray] = {}
    for node_type, node_ids in order.items():
        features = np.column_stack(
            [existing[node_type], np.asarray(blocks[node_type], dtype=np.float64)]
        ).astype(np.float32)
        bundle.hetero_data[str(node_type)].x = torch.tensor(
            features, dtype=torch.float32
        )
        written[node_type] = features
    global_map = bundle.homogeneous["global_map"]
    homogeneous = bundle.homogeneous["node_x"].detach().cpu().numpy()
    if homogeneous.shape != (len(bundle.nodes), DEGREE_CHANNELS):
        raise RuntimeError(f"OMICS_HOMOGENEOUS_ALREADY_AUGMENTED: {homogeneous.shape}")
    widened = np.zeros((len(bundle.nodes), node_input_dim), dtype=np.float32)
    for node_type, node_ids in order.items():
        for row_index, node_id in enumerate(node_ids):
            position = global_map[(node_type, node_id)]
            widened[position] = written[node_type][row_index]
    bundle.homogeneous["node_x"] = torch.tensor(widened, dtype=torch.float32)
    bundle.node_omics = {key: np.array(value, dtype=np.float64) for key, value in blocks.items()}
    bundle.node_input_dim = node_input_dim
    return verify_attached(bundle, blocks)


def verify_attached(bundle, blocks: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read every stored node tensor back and confirm the omics columns landed."""

    order = _bundle_order(bundle)
    node_input_dim = int(bundle.node_input_dim)
    if node_input_dim <= DEGREE_CHANNELS:
        raise RuntimeError("OMICS_ATTACH_DID_NOT_WIDEN_INPUT")
    width = node_input_dim - DEGREE_CHANNELS
    checks: dict[str, Any] = {"node_input_dim": node_input_dim, "width": width}
    for node_type, node_ids in order.items():
        stored = bundle.hetero_data[str(node_type)].x.detach().cpu().numpy()
        if stored.shape != (len(node_ids), node_input_dim):
            raise RuntimeError(f"OMICS_ATTACH_SHAPE: {node_type} {stored.shape}")
        if not np.isfinite(stored).all():
            raise RuntimeError(f"OMICS_ATTACH_NOT_FINITE: {node_type}")
        if blocks is not None:
            expected = np.asarray(blocks[node_type], dtype=np.float64)
            error = float(
                np.max(np.abs(stored[:, DEGREE_CHANNELS:] - expected.astype(np.float32)))
            ) if stored.size else 0.0
            if not np.isfinite(error) or error > 1e-6:
                raise RuntimeError(
                    f"OMICS_ATTACH_VALUE_MISMATCH: {node_type} max_abs_error={error}"
                )
            checks[f"{node_type}_omics_max_abs_error"] = error
    homogeneous = bundle.homogeneous["node_x"].detach().cpu().numpy()
    if homogeneous.shape != (len(bundle.nodes), node_input_dim):
        raise RuntimeError(f"OMICS_ATTACH_HOMOGENEOUS_SHAPE: {homogeneous.shape}")
    global_map = bundle.homogeneous["global_map"]
    for node_type, node_ids in order.items():
        stored = bundle.hetero_data[str(node_type)].x.detach().cpu().numpy()
        for row_index, node_id in enumerate(node_ids[:8]):
            position = global_map[(node_type, node_id)]
            if not np.allclose(homogeneous[position], stored[row_index], atol=1e-6):
                raise RuntimeError(
                    f"OMICS_ATTACH_HOMOGENEOUS_ROW_MISMATCH: {node_type}:{node_id}"
                )
    checks["homogeneous_rows_verified"] = sum(
        min(8, len(node_ids)) for node_ids in order.values()
    )
    return checks


def assert_encoder_consumes_omics(model, bundle) -> dict[str, Any]:
    """Prove the first encoder layer reads the widened input.

    For the heterogeneous HGT model that is the per-node-type ``proj`` linear
    layer; for the homogeneous RGCN model it is ``input_projection``.  A model
    whose first layer still expects two columns would silently read only the
    degree features, which is exactly the "multi-omics as post-hoc expert"
    failure this check exists to catch.
    """

    node_input_dim = int(bundle.node_input_dim)
    if node_input_dim <= DEGREE_CHANNELS:
        raise RuntimeError("OMICS_ENCODER_NOT_WIDENED")
    observed: dict[str, Any] = {"node_input_dim": node_input_dim, "layers": {}}
    projection = getattr(model, "proj", None)
    if projection is not None:
        for node_type, layer in projection.items():
            first = layer if hasattr(layer, "in_features") else layer[0]
            in_features = int(getattr(first, "in_features", -1))
            observed["layers"][str(node_type)] = in_features
            if in_features != node_input_dim:
                raise RuntimeError(
                    f"OMICS_ENCODER_INPUT_WIDTH_MISMATCH: {node_type} {in_features} != {node_input_dim}"
                )
    else:
        projection = model.input_projection
        first = projection if hasattr(projection, "in_features") else projection[0]
        in_features = int(getattr(first, "in_features", -1))
        observed["layers"]["homogeneous"] = in_features
        if in_features != node_input_dim:
            raise RuntimeError(
                f"OMICS_ENCODER_INPUT_WIDTH_MISMATCH: homogeneous {in_features} != {node_input_dim}"
            )
    if int(bundle.homogeneous["node_x"].shape[1]) != node_input_dim:
        raise RuntimeError("OMICS_ENCODER_HOMOGENEOUS_WIDTH_MISMATCH")
    observed["verified"] = True
    return observed


def encode_forward(
    model,
    graph: Any,
    *,
    node_type: str,
    node_id: str,
    bundle,
) -> np.ndarray:
    """Return one node embedding without fitting anything.

    Used by the fixture checks to show that an omics value written on one node
    reaches another node's embedding through message passing, and to show that
    an isolated node is unaffected by an unrelated change.
    """

    import torch

    model.eval()
    with torch.no_grad():
        encoded = model.encode(graph)
    position = bundle.node_maps[str(node_type)][str(node_id)]
    if isinstance(encoded, Mapping):
        tensor = encoded[str(node_type)][position]
    else:
        position = bundle.homogeneous["global_map"][(str(node_type), str(node_id))]
        tensor = encoded[position]
    return tensor.detach().cpu().numpy().astype(float)


__all__ = [
    "BLOCK_FORMAT",
    "CHANNELS",
    "CLIP",
    "DEGREE_CHANNELS",
    "EVIDENCE_FORMAT",
    "FORMAT",
    "OmicsEvidence",
    "SCOPES",
    "TOLERANCE",
    "assert_encoder_consumes_omics",
    "attach_node_omics",
    "channel_layout",
    "encode_forward",
    "node_feature_block",
    "read_evidence",
    "verify_attached",
]
