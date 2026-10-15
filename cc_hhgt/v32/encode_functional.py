"""ENCODE RBP depletion evidence, kept separate from physical RNA binding.

Each contrast retains its experiment, biosample, source file and effect
orientation. Gene-level and event-level tests are never pooled as independent
replicates. Over-representation uses the actually tested gene background.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import hashlib
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


@dataclass(frozen=True)
class FunctionalContrast:
    contrast_id: str
    file_accession: str
    experiment_accession: str
    rbp_id: str
    biosample: str
    schema: str
    effect_orientation: str
    member: str = ""
    gene_column: str = ""
    effect_column: str = ""
    pvalue_column: str = ""
    qvalue_column: str = ""
    event_column: str = "ID"
    event_type: str = ""
    quantification: str = ""

    def validate(self):
        if not all((self.contrast_id, self.file_accession, self.experiment_accession, self.rbp_id, self.biosample)):
            raise ValueError("A functional contrast requires experiment, biosample, RBP and file provenance")
        if self.schema not in {"deseq2", "cuffdiff", "rmats", "mapped_de"}:
            raise ValueError("Unsupported functional schema")
        if self.effect_orientation not in {"KD_MINUS_CONTROL", "CONTROL_MINUS_KD"}:
            raise ValueError("Declare native contrast orientation; never infer it from the sign")
        if self.schema == "mapped_de" and not all((self.gene_column, self.effect_column, self.pvalue_column, self.qvalue_column)):
            raise ValueError("mapped_de requires explicit native column names")


def validate_contrast_selection(contrasts):
    """Prevent alternate processing/JC/JCEC tables from acting as replicates."""
    used = set()
    for contrast in contrasts:
        contrast.validate()
        if contrast.schema == "rmats":
            if contrast.event_type not in {"SE", "MXE", "A3SS", "A5SS", "RI"} or contrast.quantification not in {"JC", "JCEC"}:
                raise ValueError("Declare the native rMATS event type and one JC/JCEC quantification")
            key = (contrast.experiment_accession, contrast.biosample, "SPLICING", contrast.event_type)
        else:
            key = (contrast.experiment_accession, contrast.biosample, "EXPRESSION")
        if key in used:
            raise ValueError("Select one processing/quantification version per biological experiment and event type")
        used.add(key)


def bh(pvalues, *, family_size=None):
    """BH over the complete declared family, including unavailable tests."""
    p = np.asarray(pvalues, float)
    total = len(p) if family_size is None else int(family_size)
    if total < len(p): raise ValueError("BH family cannot exclude declared tests")
    out = np.full(len(p), np.nan)
    good = np.flatnonzero(np.isfinite(p))
    if len(good):
        order = good[np.argsort(p[good], kind="stable")]
        out[order] = np.minimum(1., np.minimum.accumulate((p[order]*total/np.arange(1, len(order)+1))[::-1])[::-1])
    return out


def _md5(path):
    h = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def verified_encode_file(record, cache, *, download=False, proxy=None):
    """Resolve only a released, checksummed ENCODE file; atomically cache it.

The default performs no network download. Native archives are never extracted.
"""
    accession = str(record["file_accession"])
    href = str(record["href"])
    md5 = str(record["md5sum"]).lower(); size = int(record["file_size"])
    if (record["status"] != "released" or not re.fullmatch(r"ENCFF[A-Z0-9]{6}", accession)
            or not href.startswith(f"/files/{accession}/@@download/{accession}.")
            or any(x in href for x in ("..", "?", "#", "\\"))
            or not re.fullmatch(r"[0-9a-f]{32}", md5) or size <= 0):
        raise ValueError("Invalid released ENCODE file identity/checksum")
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache/PurePosixPath(href).name
    if path.exists():
        if path.stat().st_size != size or _md5(path) != md5:
            raise ValueError(f"Existing ENCODE file failed checksum: {accession}")
        return path
    if not download:
        raise FileNotFoundError(f"Missing selected ENCODE file {accession}; use --download in a future authorized run")
    partial = path.with_name(path.name+".partial")
    if partial.exists(): raise FileExistsError(partial)
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--connect-timeout", "15",
               "--max-time", "300", "--speed-time", "30", "--speed-limit", "1024", "--retry", "2"]
    if proxy: command += ["--proxy", str(proxy)]
    try:
        subprocess.run([*command, "--output", str(partial), "https://www.encodeproject.org"+href], check=True, timeout=960)
        if partial.stat().st_size != size or _md5(partial) != md5:
            raise ValueError(f"Downloaded ENCODE file failed checksum: {accession}")
        partial.replace(path)
    finally:
        if partial.exists(): partial.unlink()
    return path


def read_contrast_table(path, contrast, *, max_member_bytes=256*1024**2):
    contrast.validate()
    if contrast.member:
        member = PurePosixPath(contrast.member)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("Unsafe selected archive member")
        with tarfile.open(path, "r:*") as archive:
            matches = [x for x in archive.getmembers() if x.name == contrast.member]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > max_member_bytes:
                raise ValueError("Selected archive member must be one bounded regular table")
            with archive.extractfile(matches[0]) as stream:
                return pd.read_csv(stream, sep="\t", dtype=str)
    if tarfile.is_tarfile(path):
        raise ValueError("A splicing archive requires an explicit member; do not count JC/JCEC twice")
    return pd.read_csv(path, sep="\t", dtype=str, compression="infer")


def parse_functional_table(frame, contrast, gene_crosswalk):
    """Normalize DESeq2, Cuffdiff, explicit DE schemas and rMATS event tables."""
    contrast.validate()
    schemas = {
        "deseq2": ("gene_id", "log2FoldChange", "pvalue", "padj"),
        "cuffdiff": ("gene_id", "log2(fold_change)", "p_value", "q_value"),
        "rmats": ("GeneID", "IncLevelDifference", "PValue", "FDR"),
        "mapped_de": (contrast.gene_column, contrast.effect_column, contrast.pvalue_column, contrast.qvalue_column),
    }
    columns = list(schemas[contrast.schema])
    overrides = [contrast.gene_column, contrast.effect_column, contrast.pvalue_column, contrast.qvalue_column]
    columns = [new or old for old, new in zip(columns, overrides)]
    if not set(columns).issubset(frame): raise ValueError(f"Native functional table lacks columns: {columns}")
    crosswalk = gene_crosswalk[["source_gene_id", "gene_id"]].drop_duplicates().copy()
    if crosswalk.isna().any().any() or crosswalk.source_gene_id.duplicated().any():
        raise ValueError("Gene crosswalk requires unambiguous version-specific source identifiers")
    raw_ids = frame[columns[0]].astype("string").str.strip().str.strip('"')
    result = pd.DataFrame({"source_gene_id": raw_ids,
        "gene_id": raw_ids.map(crosswalk.set_index("source_gene_id").gene_id),
        "effect": pd.to_numeric(frame[columns[1]], errors="coerce"),
        "pvalue": pd.to_numeric(frame[columns[2]], errors="coerce"),
        "source_qvalue": pd.to_numeric(frame[columns[3]], errors="coerce")})
    if contrast.effect_orientation == "CONTROL_MINUS_KD": result["effect"] = -result.effect
    for column in ("pvalue", "source_qvalue"):
        values = result[column].dropna()
        if not values.between(0, 1).all(): raise ValueError("Functional p-values/FDR must lie in [0,1]")
    if contrast.schema == "rmats":
        if contrast.event_column not in frame: raise ValueError("Splicing table lacks event IDs")
        result["event_id"] = frame[contrast.event_column].astype("string")
        if result.effect.dropna().abs().gt(1).any(): raise ValueError("Splicing inclusion difference lies outside [-1,1]")
    else:
        result["event_id"] = raw_ids
    if result.event_id.isna().any() or result.event_id.duplicated().any():
        raise ValueError("Duplicate/missing native tests would inflate functional support")
    finite = np.isfinite(result.effect) & np.isfinite(result.pvalue) & np.isfinite(result.source_qvalue)
    result["unavailable_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[~finite, "unavailable_reason"] = "MISSING_OR_NONFINITE_TEST"
    if contrast.schema == "cuffdiff":
        if "status" not in frame: raise ValueError("Cuffdiff requires its native test status")
        result.loc[frame.status.ne("OK"), "unavailable_reason"] = "NATIVE_TEST_NOT_OK"
    result.loc[result.gene_id.isna(), "unavailable_reason"] = "GENE_CROSSWALK_UNAVAILABLE"
    result["available"] = result.unavailable_reason.isna()
    for key in ("contrast_id", "file_accession", "experiment_accession", "rbp_id", "biosample", "member", "event_type", "quantification"):
        result[key] = getattr(contrast, key)
    result["assay"] = "RBP_DEPLETION_SPLICING" if contrast.schema == "rmats" else "RBP_DEPLETION_EXPRESSION"
    result["effect_orientation"] = "KD_MINUS_CONTROL"
    result["physical_binding_claimed"] = False
    result["lncrna_perturbation_claimed"] = False
    return result.reset_index(drop=True)


def gene_level_tests(records, *, alpha=.05, min_abs_log2fc=1., min_abs_delta_psi=.1):
    """Collapse splicing events conservatively; preserve mixed event directions.

Splicing p-values use within-gene Simes and BH over all mapped tested genes.
An affected gene must also contain an event meeting native FDR/effect cutoffs.
DE uses the provider's full-family adjusted p-value, without re-adjusting a subset.
"""
    if not 0 < alpha < 1 or min_abs_log2fc < 0 or not 0 <= min_abs_delta_psi <= 1:
        raise ValueError("Invalid functional test thresholds")
    rows = []
    for contrast_id, contrast in records.groupby("contrast_id", sort=True):
        for key in ("file_accession", "experiment_accession", "rbp_id", "biosample", "assay"):
            if contrast[key].nunique() != 1: raise ValueError("One contrast mixes biological provenance")
        splicing = contrast.assay.iloc[0] == "RBP_DEPLETION_SPLICING"
        threshold = min_abs_delta_psi if splicing else min_abs_log2fc
        local = []
        for gene, group in contrast.dropna(subset=["gene_id"]).groupby("gene_id", sort=True):
            if not splicing and len(group) != 1:
                raise ValueError("Several DE records resolve to one gene; resolve the source crosswalk")
            tested = group.loc[group.available]
            p = np.sort(tested.pvalue.to_numpy(float))
            pvalue = min(1., float(np.min(p*len(group)/np.arange(1, len(p)+1)))) if len(p) else np.nan
            eligible = tested.loc[(tested.source_qvalue <= alpha) & (tested.effect.abs() >= threshold) & tested.effect.ne(0)]
            row = {key: contrast.iloc[0][key] for key in ("contrast_id", "file_accession", "experiment_accession", "rbp_id", "biosample", "assay")}
            local.append({**row, "gene_id": gene, "pvalue": pvalue,
                "qvalue": np.nan if splicing else group.source_qvalue.iloc[0], "available": bool(len(tested)),
                "native_tests": len(group), "available_tests": len(tested),
                "up_event_count": int(eligible.effect.gt(0).sum()), "down_event_count": int(eligible.effect.lt(0).sum())})
        if not local: continue
        table = pd.DataFrame(local)
        if splicing:
            unmapped_genes = contrast.loc[contrast.gene_id.isna(), "source_gene_id"].nunique(dropna=False)
            table["qvalue"] = bh(table.pvalue, family_size=len(table)+unmapped_genes)
        significant = table.available & table.qvalue.le(alpha)
        table["up"] = significant & table.up_event_count.gt(0)
        table["down"] = significant & table.down_event_count.gt(0)
        table["altered"] = table.up | table.down
        rows.append(table)
    columns = ["contrast_id", "file_accession", "experiment_accession", "rbp_id", "biosample", "assay", "gene_id",
               "pvalue", "qvalue", "available", "native_tests", "available_tests", "up_event_count", "down_event_count", "up", "down", "altered"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def pathway_support(genes, membership, *, min_tested_genes=3, contexts=None):
    if min_tested_genes < 1: raise ValueError("Minimum tested genes must be positive")
    membership = membership[["pathway_id", "gene_id"]].dropna().drop_duplicates()
    pathways = membership.groupby("pathway_id").gene_id.agg(set).to_dict()
    rows = []
    context_columns = ["contrast_id", "file_accession", "experiment_accession", "rbp_id", "biosample", "assay"]
    declared = (genes if contexts is None else contexts)[context_columns].drop_duplicates()
    if declared.contrast_id.duplicated().any(): raise ValueError("Pathway contrast mixes biological provenance")
    for base in declared.to_dict("records"):
        group = genes.loc[genes.contrast_id.eq(base["contrast_id"])]
        background = set(group.loc[group.available.astype(bool), "gene_id"])
        for direction in ("altered", "up", "down"):
            hit = set(group.loc[group[direction].astype(bool) & group.available.astype(bool), "gene_id"])
            local = []
            for pathway, members in sorted(pathways.items()):
                tested = background & members; overlap = tested & hit
                available = len(tested) >= min_tested_genes and len(background) > len(tested)
                p = float(hypergeom.sf(len(overlap)-1, len(background), len(hit), len(tested))) if available else np.nan
                local.append({**base, "pathway_id": pathway, "direction": direction,
                    "background_tested_genes": len(background), "background_affected_genes": len(hit),
                    "pathway_tested_genes": len(tested), "pathway_total_genes": len(members), "overlap_genes": len(overlap),
                    "affected_gene_ids": "|".join(sorted(overlap)), "pvalue": p, "available": available,
                    "unavailable_reason": None if available else "INSUFFICIENT_TESTED_PATHWAY_OR_COMPLEMENT",
                    "evidence_scope": "RBP_DEPLETION_IN_DECLARED_BIOSAMPLE", "causal_lncrna_effect_claimed": False})
            if local:
                q = bh([r["pvalue"] for r in local])
                for row, value in zip(local, q): row["qvalue"] = value
                rows.extend(local)
    return pd.DataFrame(rows)


def contextual_functional_evidence(pathways, physical_binding, exact, *, alpha=.05):
    """Trace lncRNA--RBP binding to an RBP depletion pathway result.

This is an explicit hypothesis bridge, never a lncRNA perturbation label.
Only matching assay biosamples are joined. Cancer context comes from exact
candidates; cell-line evidence is not promoted to a TCGA patient experiment.
"""
    required = {"lncrna_id", "rbp_id", "biosample", "graph_assay_class", "binding_source_id"}
    if not required.issubset(physical_binding): raise ValueError("Functional bridge requires source-bound physical binding with biosample")
    physical = {"eclip", "other_clip", "rip", "rna_capture", "other_physical"}
    binding = physical_binding.loc[physical_binding.graph_assay_class.isin(physical)].copy()
    binding = binding.dropna(subset=list(required)).drop_duplicates()
    if pathways.empty: return pd.DataFrame()
    support = pathways.loc[pathways.available & pathways.qvalue.le(alpha)].copy()
    joined = binding.merge(support, on=["rbp_id", "biosample"], how="inner", validate="many_to_many")
    joined = exact[["cancer_id", "lncrna_id", "pathway_id"]].drop_duplicates().merge(
        joined, on=["lncrna_id", "pathway_id"], how="inner", validate="many_to_many")
    joined["evidence_family"] = "RBP_PERTURBATION_PATHWAY_BRIDGE"
    joined["lncrna_perturbation_claimed"] = False
    joined["physical_binding_inferred_from_kd"] = False
    joined["independent_external_validation"] = False
    return joined
