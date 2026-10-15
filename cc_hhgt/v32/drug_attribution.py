"""Exact counterfactual forward passes through a frozen drug association head.

Deleting a target changes both its contribution to the mean target embedding
and the target-count/availability domain features. Pathway attribution deletes
the entire intersecting target set in one pass; overlapping or individual
target effects are not summed. These are model-input ablations, not biological
interventions, affinity estimates, or treatment-effect predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import drug_training as base


def ablate_drug_targets(head, core, queries, domain, assay_available, targets, exact, membership,
                       *, domain_mean, domain_scale, batch_size=256, pharmacology=None, models=()):
    """Return baselines and actual target/pathway deletion effects for one fold."""
    import torch
    from .integrated_model import freeze_v32_core, assert_v32_core_unchanged

    queries = queries[list(base.TARGET_KEYS)].copy().reset_index(drop=True)
    if batch_size < 1:
        raise ValueError("Attribution batch size must be positive")
    if queries.empty or queries.isna().any().any() or queries.duplicated(list(base.TARGET_KEYS)).any():
        raise ValueError("Attribution requires unique complete cancer/lncRNA/drug queries")
    domain = np.asarray(domain, dtype=np.float32)
    mean, scale = np.asarray(domain_mean, np.float32), np.asarray(domain_scale, np.float32)
    from .drug_pharmacology import FEATURES
    domain_width = len(base._DOMAIN_FEATURES) + (len(FEATURES) if pharmacology is not None else 0)
    if (domain.shape != (len(queries), domain_width) or mean.shape != scale.shape
            or mean.shape != (domain_width,) or not np.isfinite(domain).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError("Drug attribution domain features/scaler are invalid")
    if pharmacology is not None:
        expected = pharmacology.encode(queries, models, core=core)
        if not np.array_equal(domain[:,len(base._DOMAIN_FEATURES):], expected):
            raise ValueError("Baseline pharmacology features differ from the actual split/source encoder")
    available = np.asarray(assay_available)
    if available.shape != (len(queries),) or available.dtype != bool:
        raise ValueError("Attribution assay availability must be an aligned boolean mask")
    target = base.normalise_drug_targets(targets)
    genes = {str(drug): sorted(set(group.gene_id)) for drug, group in target.groupby("drug_id")}
    membership = membership[["pathway_id", "gene_id"]].dropna().copy()
    membership["gene_id"] = membership.gene_id.map(base._canonical_gene)
    pathway_genes = membership.groupby("pathway_id").gene_id.agg(set).to_dict()
    pair_pathways = exact.groupby(["cancer_id", "lncrna_id"]).pathway_id.agg(set).to_dict()
    vectors, counts, target_available = base._drug_target_vectors(target, core)
    core_values, core_available = base._candidate_core(queries, core, vectors, pharmacology=pharmacology)
    before = freeze_v32_core(head)
    baselines, effects = [], []
    count_index = base._DOMAIN_FEATURES.index("log1p_drug_target_count")
    available_index = base._DOMAIN_FEATURES.index("drug_target_embedding_available")

    def forward(x, d):
        result = []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                logits = head(torch.from_numpy(np.asarray(x[start:start+batch_size], np.float32)),
                              torch.from_numpy((np.asarray(d[start:start+batch_size], np.float32)-mean)/scale)).reshape(-1)
                if not torch.isfinite(logits).all(): raise ValueError("Attribution head produced non-finite logits")
                result.extend(zip(logits.cpu().tolist(), torch.sigmoid(logits).cpu().tolist()))
        return result

    for i, query in enumerate(queries.to_dict("records")):
        drug = query["drug_id"]
        all_genes = genes.get(drug, [])
        paths = pair_pathways.get((query["cancer_id"], query["lncrna_id"]), set())
        eligible_paths = {p: set(all_genes) & pathway_genes.get(p, set()) for p in paths}
        eligible_paths = {p: g for p, g in eligible_paths.items() if g}
        reason = ("NO_HELD_OUT_ASSAY_SUPPORT" if not available[i] else
                  "CURRENT_CORE_UNAVAILABLE" if not core_available[i] else
                  "NO_TARGET_EMBEDDING" if not target_available.get(drug, False) else
                  "OUTSIDE_TARGET_DERIVED_EXACT_CANDIDATES" if not eligible_paths else None)
        baseline = {**query, "cell_line_fold": core.fold, "available": reason is None,
                    "unavailable_reason": reason, "baseline_logit": np.nan, "baseline_probability": np.nan}
        if reason:
            baselines.append(baseline); continue
        if (not np.isclose(domain[i, count_index], np.log1p(counts[drug]))
                or not np.isclose(domain[i, available_index], 1.)):
            raise ValueError("Baseline domain does not match the actual target encoding")
        removals = [("TARGET", gene, {gene}) for gene in all_genes]
        removals += [("EXACT_PATHWAY_TARGET_SET", p, g) for p, g in sorted(eligible_paths.items())]
        xs = [core_values[i]]; ds = [domain[i]]; changed = []
        width = core.gene.values.shape[1]
        for kind, entity, removed in removals:
            remaining = sorted(set(all_genes)-removed)
            positions = [core.gene.index[g] for g in remaining if g in core.gene.index]
            x = core_values[i].copy()
            x[2*width:3*width] = core.gene.values[positions].mean(axis=0) if positions else 0.
            d = domain[i].copy()
            d[count_index] = np.log1p(len(remaining)); d[available_index] = bool(positions)
            if pharmacology is not None:
                x[3*width:] = pharmacology.target_core(queries.iloc[[i]], core, excluded_targets={drug:removed})[0][0]
                d[len(base._DOMAIN_FEATURES):] = pharmacology.encode(queries.iloc[[i]], models, core=core,
                    excluded_targets={drug:removed})[0]
            xs.append(x); ds.append(d)
            changed.append((kind, entity, removed, remaining, len(positions)))
        scored = forward(np.asarray(xs), np.asarray(ds))
        full_logit, full_probability = scored[0]
        baseline.update(baseline_logit=full_logit, baseline_probability=full_probability)
        baselines.append(baseline)
        for (kind, entity, removed, remaining, embedded), (logit, probability) in zip(changed, scored[1:]):
            effects.append({**query, "cell_line_fold": core.fold, "ablation_kind": kind, "ablation_entity_id": entity,
                "removed_target_ids": "|".join(sorted(removed)), "removed_target_count": len(removed),
                "remaining_target_count": len(remaining), "remaining_embedded_targets": embedded,
                "baseline_logit": full_logit, "ablated_logit": logit, "delta_logit": full_logit-logit,
                "baseline_probability": full_probability, "ablated_probability": probability,
                "delta_probability": full_probability-probability,
                "targetless_counterfactual": not bool(embedded), "additive_attribution_claimed": False,
                "estimand": "ABLATION_OF_ASSOCIATION_STRENGTH_MODEL_INPUT", "causal_claimed": False,
                "target_pharmacology_recomputed":pharmacology is not None})
    assert_v32_core_unchanged(head, before)
    return pd.DataFrame(baselines), pd.DataFrame(effects), {"head_parameter_sha256": before,
        "parameters_unchanged": True, "optimizer_started": False, "forward_counterfactuals_executed": len(effects),
        "baseline_rows": len(baselines), "target_gene_annotation_used_as_causal_truth": False}
