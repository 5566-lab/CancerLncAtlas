"""Transcript-oriented RBNS sequence preference, explicitly predicted at RNA level.

Scores are averages of measured k-mer log2 enrichment over transcript windows.
The composition comparator assumes independent nucleotides with the observed
transcript base frequencies. It is neither a structure model nor a null-test
p-value. Isoforms remain separate before equal-isoform gene aggregation.
"""
from __future__ import annotations
from dataclasses import dataclass
import gzip
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .rbp_assay import classify_assay
from .rbp_typed_binding import BindingTyping, build_typed_lnc_protein_binding


@dataclass(frozen=True)
class RbnsProfile:
    profile_id: str
    rbp_id: str
    file_accession: str
    experiment_accession: str
    concentration: str
    k: int
    log2_enrichment: np.ndarray
    enriched: np.ndarray
    base_digits: np.ndarray


def _kmer_code(kmer):
    code = 0
    for base in kmer: code = 4*code+"ACGT".index(base)
    return code


def parse_rbns_profile(frame, *, profile_id, rbp_id, file_accession, experiment_accession,
                       concentration, source_rbp_label, min_motif_enrichment=2., zero_floor=1e-6):
    if not all((profile_id, rbp_id, file_accession, experiment_accession, source_rbp_label)):
        raise ValueError("RBNS profiles require complete source identity")
    if str(frame.columns[0]).strip() != f"[{source_rbp_label}]":
        raise ValueError("Native RBNS table header disagrees with its source RBP")
    if concentration not in frame: raise ValueError("Prespecify one native RBNS concentration column")
    if not 0 < zero_floor < 1 or min_motif_enrichment <= 1:
        raise ValueError("Invalid RBNS enrichment threshold/floor")
    kmers = frame.iloc[:, 0].astype(str).str.upper().str.replace("U", "T", regex=False)
    lengths = set(kmers.str.len())
    if len(lengths) != 1 or not 4 <= next(iter(lengths)) <= 7 or not kmers.str.fullmatch("[ACGT]+").all():
        raise ValueError("RBNS profile must contain one native 4-7-mer alphabet")
    k = next(iter(lengths))
    if kmers.duplicated().any() or len(kmers) != 4**k:
        raise ValueError("Incomplete/duplicate RBNS k-mers cannot be silently assigned neutral enrichment")
    values = pd.to_numeric(frame[concentration], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0): raise ValueError("Invalid native RBNS enrichment values")
    codes = np.array([_kmer_code(x) for x in kmers])
    ordered = np.empty(len(codes)); ordered[codes] = values
    digits = np.stack([(np.arange(4**k)//(4**i)) % 4 for i in range(k)], axis=1)
    return RbnsProfile(profile_id, rbp_id, file_accession, experiment_accession, concentration, k,
                       np.log2(np.maximum(ordered, zero_floor)), ordered >= min_motif_enrichment, digits)


def iter_fasta(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    header = None; chunks = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None: yield header, "".join(chunks)
                header, chunks = line[1:].split()[0], []
                if not header: raise ValueError("Empty FASTA identifier")
            else:
                if header is None: raise ValueError("Sequence before a FASTA header")
                sequence = line.upper().replace("U", "T")
                if not re.fullmatch("[ACGTRYSWKMBDHVN]+", sequence): raise ValueError("Invalid transcript sequence alphabet")
                chunks.append(sequence)
    if header is not None: yield header, "".join(chunks)


def _window_counts(sequence, k):
    lut = np.full(256, -1, np.int16)
    lut[[ord(x) for x in "ACGT"]] = np.arange(4)
    bases = lut[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    n = max(0, len(sequence)-k+1)
    codes = np.zeros(n, np.int32); valid = np.ones(n, bool)
    for offset in range(k):
        local = bases[offset:offset+n]
        codes = codes*4+np.maximum(local, 0); valid &= local >= 0
    counts = np.bincount(codes[valid], minlength=4**k)
    base_counts = np.bincount(bases[bases >= 0], minlength=4).astype(float)
    frequencies = base_counts/base_counts.sum() if base_counts.sum() else np.full(4, .25)
    return counts, n, frequencies


def iter_transcript_scores(fasta, crosswalk, profiles, *, annotation_id, min_valid_windows=50, min_window_fraction=.9):
    if min_valid_windows < 1 or not 0 < min_window_fraction <= 1:
        raise ValueError("Invalid transcript sequence coverage threshold")
    columns = ["transcript_id", "source_gene_id", "lncrna_id", "annotation_id"]
    mapping = crosswalk[columns].drop_duplicates()
    if mapping.isna().any().any() or mapping.transcript_id.duplicated().any() or not mapping.annotation_id.eq(annotation_id).all():
        raise ValueError("Transcript mapping must be unambiguous and match the selected annotation version")
    if (mapping.groupby("source_gene_id").lncrna_id.nunique().gt(1).any()
            or mapping.groupby("lncrna_id").source_gene_id.nunique().gt(1).any()):
        raise ValueError("Registered lncRNA and versioned source gene identity must agree across isoforms")
    if not profiles or len({p.profile_id for p in profiles}) != len(profiles): raise ValueError("RBNS profile identities must be unique")
    registered = mapping.set_index("transcript_id").to_dict("index")
    seen = set()
    by_k = {}
    for profile in profiles: by_k.setdefault(profile.k, []).append(profile)
    for header, sequence in iter_fasta(fasta):
        parts = header.split("|"); transcript = parts[0]
        if transcript not in registered: continue
        if transcript in seen: raise ValueError("Duplicate registered transcript in FASTA")
        seen.add(transcript); record = registered[transcript]
        if len(parts) > 1 and parts[1].startswith("ENSG") and parts[1] != record["source_gene_id"]:
            raise ValueError("FASTA gene version disagrees with the transcript crosswalk")
        for k, group in by_k.items():
            counts, possible, frequencies = _window_counts(sequence, k)
            valid = int(counts.sum()); fraction = valid/possible if possible else 0.
            available = valid >= min_valid_windows and fraction >= min_window_fraction
            weights = np.stack([p.log2_enrichment for p in group])
            raw = weights@counts/valid if valid else np.full(len(group), np.nan)
            composition = np.prod(frequencies[group[0].base_digits], axis=1)
            comparator = weights@composition
            motifs = np.stack([p.enriched for p in group])@counts/valid if valid else np.full(len(group), np.nan)
            for i, profile in enumerate(group):
                yield {"transcript_id": transcript, "lncrna_id": record["lncrna_id"], "source_gene_id": record["source_gene_id"],
                    "annotation_id": annotation_id, "profile_id": profile.profile_id, "rbp_id": profile.rbp_id,
                    "file_accession": profile.file_accession, "experiment_accession": profile.experiment_accession,
                    "concentration": profile.concentration, "k": k, "sequence_length": len(sequence),
                    "valid_windows": valid, "possible_windows": possible, "valid_window_fraction": fraction,
                    "available": available, "unavailable_reason": None if available else "INSUFFICIENT_UNAMBIGUOUS_SEQUENCE_WINDOWS",
                    "mean_log2_enrichment": float(raw[i]) if available else np.nan,
                    "composition_expected_log2_enrichment": float(comparator[i]) if available else np.nan,
                    "composition_excess_log2_enrichment": float(raw[i]-comparator[i]) if available else np.nan,
                    "enriched_window_fraction": float(motifs[i]) if available else np.nan,
                    "lncrna_binding_observed": False}


def aggregate_gene_scores(transcript_summary, crosswalk, lncrnas, profiles, *, min_transcript_coverage=1.):
    """Consume preaggregated isoform summaries without inventing missing sequences."""
    if not 0 < min_transcript_coverage <= 1: raise ValueError("Invalid gene transcript coverage threshold")
    lnc = lncrnas[["lncrna_id"]].drop_duplicates()
    if lnc.isna().any().any(): raise ValueError("Missing lncRNA identities")
    profile_rows = pd.DataFrame([{k: getattr(p, k) for k in ("profile_id", "rbp_id", "file_accession", "experiment_accession", "concentration", "k")} for p in profiles])
    universe = lnc.merge(profile_rows, how="cross")
    registered = crosswalk.groupby("lncrna_id").transcript_id.nunique().rename("registered_transcripts")
    universe = universe.merge(registered, on="lncrna_id", how="left", validate="many_to_one")
    universe["registered_transcripts"] = universe.registered_transcripts.fillna(0).astype(int)
    result = universe.merge(transcript_summary, on=["lncrna_id", "profile_id"], how="left", validate="one_to_one")
    for column in ("observed_transcripts", "scorable_transcripts"):
        result[column] = result[column].fillna(0).astype(int)
    if (result.observed_transcripts.gt(result.registered_transcripts).any()
            or result.scorable_transcripts.gt(result.observed_transcripts).any()):
        raise ValueError("Transcript summary exceeds its registered sequence universe")
    result["transcript_coverage"] = result.scorable_transcripts/result.registered_transcripts.replace(0, np.nan)
    result["available"] = result.scorable_transcripts.gt(0) & result.transcript_coverage.ge(min_transcript_coverage)
    result["unavailable_reason"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result.loc[~result.available, "unavailable_reason"] = "INCOMPLETE_OR_UNSCORABLE_TRANSCRIPT_SEQUENCE"
    result.loc[result.registered_transcripts.eq(0), "unavailable_reason"] = "NO_MATCHING_TRANSCRIPT_ANNOTATION"
    result["prediction_score"] = result.mean_composition_excess.where(result.available)
    result["evidence_family"] = "RBNS_SEQUENCE_PREFERENCE_PREDICTION"
    result["is_predicted"] = True; result["is_experimental"] = False
    result["affinity_estimated"] = False; result["score_is_probability"] = False
    return result


def predicted_binding_inputs(gene_scores, protein_gene_map, lncrnas, *, min_excess=.25, min_mean_log2_enrichment=0.):
    """Use the existing typed-binding materializer, preserving predicted status."""
    if not np.isfinite([min_excess, min_mean_log2_enrichment]).all() or min_excess <= 0:
        raise ValueError("Predicted RBNS relation requires a positive prespecified excess threshold")
    selected = gene_scores.loc[gene_scores.available & gene_scores.prediction_score.ge(min_excess)
                               & gene_scores.mean_log2_enrichment.ge(min_mean_log2_enrichment)].copy()
    relation = selected.rename(columns={"rbp_id": "partner_id"})
    relation["partner_type"] = "rbp"
    relation["source_database"] = "ENCODE_RBNS_SEQUENCE"
    relation["experiment_raw"] = "computational prediction from RBNS k-mer preference and transcript sequence"
    relation["pmid"] = pd.NA
    binding, audit = build_typed_lnc_protein_binding(relation, protein_gene_map, lncrnas,
        pd.DataFrame(columns=["cancer_id"]), classify=classify_assay,
        typing=BindingTyping(include_rbp_partner_type=True, group_by_assay_class=True,
                             emit_predicted_relation=True, globalize_binding_context=True))
    if not binding.empty and not binding.graph_assay_class.eq("predicted").all():
        raise AssertionError("RBNS sequence predictions became experimental edges")
    audit.update(main_training_graph_modified=False, relation_policy="PREDICTED_ONLY_SEPARATE_ARTIFACT",
                 graph_weight_semantics="EXISTING_FIXED_PREDICTED_PRIOR_TIMES_PROTEIN_MAPPING_WEIGHT_NOT_A_PROBABILITY")
    return relation, binding, audit
