"""Verified, fold-local gene CNV and promoter-methylation node inputs.

The source product is physically patient x gene.  Each graph view is built
directly from its declared feature patients; no outer-fold or inner-fold
aggregate is accepted.  Missingness remains an explicit mask, so an observed
zero is never confused with an unavailable assay value.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, fail, sha


PRODUCT_FORMAT = "GENE_PATIENT_ASSAYS_VERIFIED_V1"
CHANNELS = ("cnv_signed", "methylation_promoter_beta")
FEATURE_ORDER = (
    "cnv_signed_value",
    "cnv_signed_available",
    "methylation_promoter_beta_value",
    "methylation_promoter_beta_available",
)
NODE_INPUT_DIM = 2 + len(FEATURE_ORDER)


def _array_digest(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(b"\0")
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    h.update(b"\0")
    h.update(array.tobytes())
    return h.hexdigest()


def _resolve(record, base, role):
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        fail("GENE_NODE_OMICS_RECORD_INVALID", "Artifact record requires path and sha256", role=role)
    raw = Path(str(record["path"]))
    path = (raw if raw.is_absolute() else Path(base) / raw).resolve()
    expected = str(record["sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        fail("GENE_NODE_OMICS_HASH_INVALID", "Artifact SHA-256 is malformed", role=role)
    if not path.is_file() or sha(path) != expected:
        fail("GENE_NODE_OMICS_INPUT_DRIFT", "Artifact is missing or differs from its verified SHA-256", role=role, path=str(path))
    return path, {"path": str(path), "sha256": expected}


class VerifiedGeneNodeOmics:
    """Validated product reader with one-time hash and axis checks per cancer."""

    def __init__(self, config, graph_genes, cancers):
        if not isinstance(config, dict) or config.get("enabled") is not True:
            fail("GENE_NODE_OMICS_CONFIG_INVALID", "Enabled gene node omics requires an explicit mapping")
        if config.get("channels") != list(CHANNELS):
            fail("GENE_NODE_OMICS_CHANNEL_DRIFT", "CNV then promoter methylation are the frozen channel order")
        complete_record = config.get("complete")
        raw_complete = Path(str(complete_record.get("path", ""))) if isinstance(complete_record, dict) else Path("")
        complete_base = Path.cwd()
        self.complete_path, self.complete_record = _resolve(complete_record, complete_base, "complete")
        self.payload = json.loads(self.complete_path.read_text(encoding="utf-8"))
        p = self.payload
        if (p.get("format") != PRODUCT_FORMAT or p.get("status") != "PASS"
                or p.get("target_host") != "149" or p.get("training_started") is not False
                or p.get("node_type") != "gene" or p.get("gene_type") != "protein_coding"
                or p.get("genome_build") != "GRCh38" or p.get("matrix_orientation") != "patient_by_gene"
                or p.get("units") != {"cnv_signed": "log2_copy_ratio", "methylation_promoter_beta": "beta"}):
            fail("GENE_NODE_OMICS_PRODUCT_CONTRACT", "COMPLETE.json is not the verified Oct-29 gene assay product")
        base = self.complete_path.parent
        self.genes_path, self.genes_record = _resolve(p.get("genes"), base, "genes")
        self.patients_path, self.patients_record = _resolve(p.get("patients"), base, "patients")
        genes = pd.read_parquet(self.genes_path)
        patients = pd.read_parquet(self.patients_path)
        if ("gene_id" not in genes or genes.empty or genes.gene_id.isna().any()
                or genes.gene_id.duplicated().any()
                or not genes.gene_id.astype(str).str.fullmatch(r"ENSG\d+(?:_PAR_Y)?").all()):
            fail("GENE_NODE_OMICS_GENE_AXIS_INVALID", "Source gene axis must be unique stable Ensembl IDs")
        required_patient_columns = {"cancer_id", "patient_id", "patient_fold_id"}
        if (not required_patient_columns <= set(patients) or patients.empty
                or patients[list(required_patient_columns)].isna().any().any()
                or patients.patient_id.astype(str).duplicated().any()
                or not patients.patient_fold_id.isin(range(5)).all()):
            fail("GENE_NODE_OMICS_PATIENT_AXIS_INVALID", "Source patient axis is invalid")
        self.source_genes = genes.gene_id.astype(str).tolist()
        requested = list(map(str, graph_genes))
        if len(requested) != len(set(requested)):
            fail("GENE_NODE_OMICS_GRAPH_GENE_AXIS", "Graph genes must be unique")
        self.excluded_record = None
        exclusions = []
        if p.get("excluded_requested_genes") is not None:
            ep, self.excluded_record = _resolve(p["excluded_requested_genes"], base, "excluded_requested_genes")
            exclusions = json.loads(ep.read_text())
            if (not isinstance(exclusions, list) or len(exclusions) != len(set(exclusions))
                    or any(not isinstance(g, str) or not re.fullmatch(r"ENSG\d+(?:_PAR_Y)?", g) for g in exclusions)
                    or set(exclusions) & set(self.source_genes)):
                fail("GENE_NODE_OMICS_EXCLUSION_AXIS_INVALID", "Invalid reference-exclusion catalog")
        unknown = set(requested) - set(self.source_genes) - set(exclusions)
        if unknown:
            fail("GENE_NODE_OMICS_GRAPH_GENE_AXIS", "Genes are neither measured-axis members nor explicitly excluded", examples=sorted(unknown)[:8])
        positions = {gene: i for i, gene in enumerate(self.source_genes)}
        self.graph_source_positions = np.asarray([i for i, gene in enumerate(requested) if gene in positions], dtype=np.int64)
        self.gene_positions = np.asarray([positions[requested[i]] for i in self.graph_source_positions], dtype=np.int64)
        self.reference_excluded_graph_genes = sorted(set(requested) & set(exclusions))
        self.graph_genes = requested
        self.patients = patients.assign(cancer_id=patients.cancer_id.astype(str), patient_id=patients.patient_id.astype(str))
        partitions = p.get("partitions")
        if not isinstance(partitions, list):
            fail("GENE_NODE_OMICS_PARTITIONS_INVALID", "Product partitions are missing")
        self.partition_records = {}
        for row in partitions:
            cancer = str(row.get("cancer_id", "")) if isinstance(row, dict) else ""
            if not cancer or cancer in self.partition_records:
                fail("GENE_NODE_OMICS_PARTITIONS_INVALID", "Cancer partitions must be unique")
            self.partition_records[cancer] = row
        missing_cancers = sorted(set(map(str, cancers)) - set(self.partition_records))
        if missing_cancers:
            fail("GENE_NODE_OMICS_CANCER_MISSING", "Configured cancers are absent from the verified product", cancers=missing_cancers)
        self._partitions = {}
        self._input_hashes = {
            "gene_node_omics::complete": self.complete_record,
            "gene_node_omics::genes": self.genes_record,
            "gene_node_omics::patients": self.patients_record,
        }
        if self.excluded_record is not None:
            self._input_hashes["gene_node_omics::excluded_requested_genes"] = self.excluded_record

    def _partition(self, cancer):
        cancer = str(cancer)
        if cancer in self._partitions:
            return self._partitions[cancer]
        row = self.partition_records[cancer]
        base = self.complete_path.parent
        ids_path, ids_record = _resolve(row.get("patients"), base, cancer + ":patients")
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        expected = self.patients.loc[self.patients.cancer_id.eq(cancer), "patient_id"].tolist()
        if (not isinstance(ids, list) or list(map(str, ids)) != expected
                or not ids or len(ids) != len(set(ids))):
            fail("GENE_NODE_OMICS_PARTITION_AXIS_DRIFT", "Partition patient axis differs from patients.parquet", cancer=cancer)
        artifacts = {"patients": ids_record}
        matrices = {}
        channels = row.get("channels", {})
        if set(channels) != set(CHANNELS):
            fail("GENE_NODE_OMICS_CHANNEL_DRIFT", "Partition channel set differs", cancer=cancer)
        shape = (len(ids), len(self.source_genes))
        for channel in CHANNELS:
            values_path, values_record = _resolve(channels[channel].get("values"), base, cancer + ":" + channel + ":values")
            mask_path, mask_record = _resolve(channels[channel].get("available"), base, cancer + ":" + channel + ":available")
            values = np.load(values_path, mmap_mode="r", allow_pickle=False)
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if (values.shape != shape or mask.shape != shape or np.isinf(values).any()
                    or not np.isin(mask, [0, 1]).all()
                    or not np.array_equal(mask.astype(bool), np.isfinite(values))):
                fail("GENE_NODE_OMICS_MATRIX_INVALID", "Matrix shape or explicit availability mask is invalid", cancer=cancer, channel=channel)
            observed = np.asarray(values)[np.asarray(mask, dtype=bool)]
            if channel == "methylation_promoter_beta" and (np.any(observed < 0) or np.any(observed > 1)):
                fail("GENE_NODE_OMICS_BETA_INVALID", "Observed promoter methylation must be in [0,1]", cancer=cancer)
            matrices[channel] = (values, mask)
            artifacts[channel + ":values"] = values_record
            artifacts[channel + ":available"] = mask_record
        for role, record in artifacts.items():
            self._input_hashes["gene_node_omics::" + cancer + "::" + role] = record
        result = {"ids": ids, "positions": {patient: i for i, patient in enumerate(ids)},
                  "matrices": matrices, "artifacts": artifacts}
        self._partitions[cancer] = result
        return result

    def verify_cancers(self, cancers):
        for cancer in cancers:
            self._partition(cancer)
        return self

    def input_hashes(self):
        return dict(self._input_hashes)

    def build_view(self, cancer, source_patients):
        part = self._partition(cancer)
        source = list(map(str, source_patients))
        if not source or source != sorted(set(source)):
            fail("GENE_NODE_OMICS_SOURCE_PATIENTS_INVALID", "Source patients must be a nonempty sorted unique list", cancer=cancer)
        missing = sorted(set(source) - set(part["positions"]))
        if missing:
            fail("GENE_NODE_OMICS_SOURCE_PATIENT_MISSING", "View patient is absent from the verified matrix", cancer=cancer, examples=missing[:8])
        rows = np.asarray([part["positions"][patient] for patient in source], dtype=np.int64)
        block = np.zeros((len(self.graph_genes), len(FEATURE_ORDER)), dtype=np.float32)
        channel_audit = {}
        for offset, channel in enumerate(CHANNELS):
            values, available = part["matrices"][channel]
            selected_values = np.asarray(values[np.ix_(rows, self.gene_positions)], dtype=np.float32)
            selected_mask = np.asarray(available[np.ix_(rows, self.gene_positions)], dtype=bool)
            count = selected_mask.sum(axis=0)
            total = np.where(selected_mask, selected_values, 0.0).sum(axis=0, dtype=np.float64)
            measured = count > 0
            aggregate = np.divide(total, count, out=np.zeros(len(count), dtype=np.float64), where=measured).astype(np.float32)
            block[self.graph_source_positions, 2 * offset] = aggregate
            block[self.graph_source_positions, 2 * offset + 1] = measured.astype(np.float32)
            channel_audit[channel] = {
                "observed_patient_gene_values": int(selected_mask.sum()),
                "observed_graph_genes": int(measured.sum()),
                "missing_graph_genes": int(len(self.graph_genes) - measured.sum()),
                "measured_zero_graph_genes": int((measured & (aggregate == 0)).sum()),
            }
        audit = {
            "format": "CLA_R62_VIEW_GENE_NODE_OMICS_V1",
            "contract": "SOURCE_PATIENT_MEAN_EXPLICIT_MASK_NO_IMPUTATION",
            "cancer_id": str(cancer),
            "source_patient_count": len(source),
            "source_patient_sha256": digest(source),
            "source_gene_axis_sha256": digest(self.source_genes),
            "graph_gene_axis_sha256": digest(self.graph_genes),
            "reference_excluded_graph_genes": self.reference_excluded_graph_genes,
            "reference_exclusion_artifact": self.excluded_record,
            "feature_order": list(FEATURE_ORDER),
            "node_input_dim": NODE_INPUT_DIM,
            "block_sha256": _array_digest(block),
            "channels": channel_audit,
            "artifacts": part["artifacts"],
            "outer_or_label_patients_used": False,
            "parameter_fit": False,
            "missing_value_imputation": False,
        }
        return block, audit


def open_verified_gene_node_omics(config, graph_genes, cancers):
    return VerifiedGeneNodeOmics(config, graph_genes, cancers).verify_cancers(cancers)


def graph_node_omics(ids, gene_block):
    """Return one explicit four-column block for every graph node type."""
    block = np.asarray(gene_block, dtype=np.float32)
    if block.shape != (len(ids["gene"]), len(FEATURE_ORDER)) or not np.isfinite(block).all():
        fail("GENE_NODE_OMICS_BLOCK_INVALID", "Gene block has the wrong shape or nonfinite values")
    result = {kind: np.zeros((len(values), len(FEATURE_ORDER)), dtype=np.float32) for kind, values in ids.items()}
    result["gene"] = block
    return result


def verify_view_node_omics(root, spec, view, cfg):
    """Validate sealed view semantics without reopening private raw matrices."""
    from .common import read_json
    root=Path(root);folder=root/'views'/view['id']
    enabled=cfg.get('gene_node_omics',{}).get('enabled',False)
    if type(enabled) is not bool or spec.get('gene_node_omics_enabled',False)!=enabled:
        fail('NODE_OMICS_SPEC_DRIFT','Configuration and prepared channel role disagree')
    with np.load(folder/'graph.npz',allow_pickle=False) as graph:
        blocks={name[len('node_omics::'):]:graph[name] for name in graph.files if name.startswith('node_omics::')}
    if not enabled:
        if blocks or spec.get('node_input_dim',2)!=2 or (folder/'NODE_OMICS_AUDIT.json').exists():
            fail('NODE_OMICS_DISABLED_INPUT','Disabled configuration cannot carry undeclared assay inputs')
        return
    config=cfg['gene_node_omics'];complete=config.get('complete',{})
    expected_order=['log1p_degree','constant',*FEATURE_ORDER]
    if (config.get('channels')!=list(CHANNELS) or spec.get('node_input_dim')!=NODE_INPUT_DIM
            or spec.get('node_input_feature_order')!=expected_order
            or spec.get('gene_node_omics_complete_sha256')!=complete.get('sha256')):
        fail('NODE_OMICS_SPEC_DRIFT','Prepared node width, order or product binding differs')
    if set(blocks)!=set(spec['nodes']):
        fail('NODE_OMICS_NODE_TYPE_SET','Every node type needs an explicit value/mask block')
    for node,block in blocks.items():
        if block.shape!=(len(spec['nodes'][node]),len(FEATURE_ORDER)) or block.dtype!=np.float32 or not np.isfinite(block).all():
            fail('NODE_OMICS_SHAPE_OR_VALUE','Invalid sealed assay block',node_type=node)
        if not np.isin(block[:,[1,3]],[0,1]).all():
            fail('NODE_OMICS_MASK_INVALID','Availability is not binary',node_type=node)
        if np.any(block[:,0][block[:,1]==0]!=0) or np.any(block[:,2][block[:,3]==0]!=0):
            fail('NODE_OMICS_MISSING_VALUE_NONZERO','Missing measurements cannot carry values',node_type=node)
        if node!='gene' and np.any(block!=0):
            fail('NODE_OMICS_NON_GENE_MEASUREMENT','Gene assays cannot be relabelled as another node type',node_type=node)
        if np.any(block[:,2]<0) or np.any(block[:,2]>1):
            fail('NODE_OMICS_BETA_INVALID','Promoter beta outside [0,1]',node_type=node)
    audit_path=folder/'NODE_OMICS_AUDIT.json'
    if not audit_path.is_file():fail('NODE_OMICS_AUDIT_MISSING','Enabled assay inputs require a source-view audit')
    audit=read_json(audit_path);provenance=read_json(folder/'PROVENANCE.json')
    cancer=spec['cancers'][view['cancer_index']]
    if (audit.get('format')!='CLA_R62_VIEW_GENE_NODE_OMICS_V1'
            or audit.get('contract')!='SOURCE_PATIENT_MEAN_EXPLICIT_MASK_NO_IMPUTATION'
            or audit.get('cancer_id')!=cancer
            or audit.get('source_patient_sha256')!=provenance.get('feature_patient_hash')
            or audit.get('graph_gene_axis_sha256')!=digest(spec['nodes']['gene'])
            or audit.get('feature_order')!=list(FEATURE_ORDER)
            or audit.get('node_input_dim')!=NODE_INPUT_DIM
            or audit.get('block_sha256')!=_array_digest(blocks['gene'])
            or audit.get('outer_or_label_patients_used') is not False
            or audit.get('missing_value_imputation') is not False
            or audit.get('parameter_fit') is not False
            or provenance.get('gene_node_omics')!=audit):
        fail('NODE_OMICS_VIEW_AUDIT_DRIFT','Sealed graph and source-patient assay audit disagree')
    hashes=read_json(root/'INPUT_HASHES.json')
    if hashes.get('gene_node_omics::complete',{}).get('sha256')!=complete.get('sha256'):
        fail('NODE_OMICS_SOURCE_BINDING_DRIFT','COMPLETE product hash is not bound to prepared inputs')
    excluded = audit.get('reference_excluded_graph_genes', [])
    if excluded:
        record = audit.get('reference_exclusion_artifact')
        if (not isinstance(record, dict) or hashes.get('gene_node_omics::excluded_requested_genes') != record
                or excluded != sorted(set(excluded)) or not set(excluded) <= set(spec['nodes']['gene'])):
            fail('NODE_OMICS_REFERENCE_EXCLUSION_DRIFT', 'Excluded graph genes need a bound source receipt')
        positions = {gene: i for i, gene in enumerate(spec['nodes']['gene'])}
        if np.any(blocks['gene'][[positions[gene] for gene in excluded]] != 0):
            fail('NODE_OMICS_EXCLUDED_GENE_VALUE', 'Reference-excluded genes must have zero availability and no assay value')
    if not audit.get('artifacts'):
        fail('NODE_OMICS_SOURCE_BINDING_DRIFT','Per-cancer assay source records are absent')
    for role,record in audit['artifacts'].items():
        if hashes.get('gene_node_omics::'+cancer+'::'+role)!=record:
            fail('NODE_OMICS_SOURCE_BINDING_DRIFT','Per-view source differs from prepared source bindings',role=role)
