"""Source-bound compound identity and unambiguous cell-model context.

Aliases are curated data, never fuzzy guesses. Unresolved compounds retain a
source-scoped assay identity so their measurements remain usable independently
of target annotations. Molecular subtypes are never inferred from tissue.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict

import pandas as pd


def text(value) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def name_key(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", text(value)).casefold().split())


class DrugIdentityResolver:
    """Index long-form drug_id / identifier_type / identifier_value / source.

identifier_type=name admits preferred names and explicitly supplied synonyms.
source:<dataset_id> admits native assay identifiers. Other namespaces such as
inchi_key or chembl use exact values; names alone undergo Unicode/space/case
normalization. Salt, stereo, punctuation and mixture distinctions remain.
"""

    def __init__(self, identifiers: pd.DataFrame):
        required = {"drug_id", "identifier_type", "identifier_value", "source"}
        if not required.issubset(identifiers):
            raise ValueError("Drug identifiers require canonical ID, type, value and source")
        self.index = defaultdict(set)
        self.sources = defaultdict(set)
        self.cache = {}
        for row in identifiers.itertuples(index=False):
            drug, kind, value, source = map(text, (row.drug_id, row.identifier_type, row.identifier_value, row.source))
            if not all((drug, kind, value, source)) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*:[^\s]+", drug):
                raise ValueError("Drug identifier rows require complete namespaced identities and provenance")
            value = name_key(value) if kind == "name" else value
            self.index[kind, value].add(drug)
            self.sources[kind, value].add(source)

    def resolve(self, dataset_id, source_drug_id, drug_name, *, identifiers=None):
        dataset, native, name = map(text, (dataset_id, source_drug_id, drug_name))
        if not dataset or not native:
            raise ValueError("Native drug identity cannot be replaced by a missing source ID")
        strong = [("source:" + dataset, native)]
        strong += [(text(k), text(v)) for k, v in (identifiers or {}).items() if text(v)]
        cache_key = (dataset, native, name, tuple(sorted(strong)))
        if cache_key in self.cache:
            return dict(self.cache[cache_key])
        known = [(key, self.index[key]) for key in strong if self.index.get(key)]
        names = self.index.get(("name", name_key(name)), set())
        candidates = set.intersection(*(set(ids) for _, ids in known)) if known else set(names)
        reason = None
        if known and not candidates:
            reason = "CONFLICTING_STABLE_IDENTIFIERS"
        elif len(candidates) > 1:
            reason = "AMBIGUOUS_IDENTIFIER_OR_SYNONYM"
        elif not candidates:
            reason = "NO_CURATED_IDENTITY_MATCH"
        elif known and names and not candidates.issubset(names):
            reason = "STABLE_IDENTIFIER_NAME_CONFLICT"
        matched = reason is None
        fallback = "ASSAY:" + hashlib.sha256(json.dumps([dataset, native]).encode()).hexdigest()[:24]
        keys = [key for key, _ in known] if known else [("name", name_key(name))]
        result = {"drug_id": next(iter(candidates)) if matched else fallback,
                  "identity_available": matched, "identity_unavailable_reason": reason,
                  "identity_method": ("STABLE_IDENTIFIER" if known else "CURATED_NAME_OR_SYNONYM") if matched else "SOURCE_SCOPED_ASSAY_ID",
                  "identity_sources": "|".join(sorted(set().union(*(self.sources.get(key, set()) for key in keys)))),
                  "identity_candidate_ids": "|".join(sorted(candidates))}
        self.cache[cache_key] = result
        return dict(result)

    def resolve_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        keys = ["dataset_id", "source_drug_id", "drug_name"]
        if not set(keys).issubset(frame):
            raise ValueError("Compound rows lack their native identity")
        unique = frame[keys].drop_duplicates().reset_index(drop=True)
        resolved = pd.DataFrame([self.resolve(*row) for row in unique.itertuples(index=False, name=None)])
        mapping = pd.concat([unique, resolved], axis=1)
        # Conflicting names for one source ID must not split measurements into
        # several compounds after mapping.
        if mapping.groupby(keys[:2], dropna=False).drug_id.nunique().gt(1).any():
            raise ValueError("One native compound ID resolves to conflicting compound identities")
        return frame.drop(columns=list(resolved.columns), errors="ignore").merge(mapping, on=keys, how="left", validate="many_to_one")


def model_context(models, gdsc_rows, prism_models, *, tissue_cancers, cancer_universe, annotations=None):
    """Assign at most one cancer per canonical model and preserve subtype sources.

Annotations contain canonical_model_id, cancer_id, source and optionally
subtype_system/subtype_id. Different subtype systems are retained in a separate
long table; they never multiply the cell-line map used by drug training.
"""
    universe = set(cancer_universe)
    models = models.copy()
    for column in ["canonical_model_id", "depmap_id", "tissue"]:
        models[column] = models[column].map(text)
    if models.groupby("depmap_id").canonical_model_id.nunique().loc[lambda s: s.index != ""].gt(1).any():
        raise ValueError("A native DepMap ID maps to more than one canonical model")
    if models.groupby("canonical_model_id").tissue.nunique().gt(1).any():
        raise ValueError("Canonical model has conflicting tissue annotations")
    sidm_tissue = models.set_index("canonical_model_id").tissue.to_dict()
    ach_sidm = models.loc[models.depmap_id.ne("")].set_index("depmap_id").canonical_model_id.to_dict()
    native_cancers = defaultdict(set)
    registry = []
    for row in gdsc_rows:
        sidm = text(row["cell_line_id"])
        cancer = text(row.get("tcga_desc")).upper()
        if cancer in universe: native_cancers[sidm].add(cancer)
        registry.append((text(row["dataset_id"]), sidm, sidm))
    for ach in prism_models:
        registry.append(("PRISM_24Q2", text(ach), ach_sidm.get(text(ach), "")))
    annotation_rows = defaultdict(list)
    if annotations is not None:
        if not {"canonical_model_id", "cancer_id", "source"}.issubset(annotations):
            raise ValueError("Model annotations require model, cancer and provenance source")
        for row in annotations.to_dict("records"):
            sidm, cancer, source = map(text, (row["canonical_model_id"], row["cancer_id"], row["source"]))
            if not sidm or cancer not in universe or not source:
                raise ValueError("Invalid model/cancer/source annotation")
            system, subtype = text(row.get("subtype_system")), text(row.get("subtype_id"))
            if bool(system) != bool(subtype):
                raise ValueError("A molecular subtype requires its classification system")
            annotation_rows[sidm].append((cancer, source, system, subtype))
    contexts = {}; subtypes = []
    for sidm in sorted({r[2] for r in registry if r[2]}):
        tissue = sidm_tissue.get(sidm, "")
        annotated = {r[0] for r in annotation_rows[sidm]}
        declared = native_cancers[sidm] | annotated
        reason = None
        if len(declared) > 1:
            cancer = None; method = "CONFLICT"; reason = "CONFLICTING_EXPLICIT_CANCER_ANNOTATIONS"
        elif declared:
            cancer = next(iter(declared)); method = "EXPLICIT_MODEL_ANNOTATION"
        else:
            possible = tissue_cancers.get(tissue, ())
            cancer = possible[0] if len(possible) == 1 else None
            method = "UNIQUE_TISSUE_COMPATIBILITY" if cancer else "UNRESOLVED"
            reason = None if cancer else "AMBIGUOUS_OR_UNKNOWN_TISSUE_CANCER"
        sources = {r[1] for r in annotation_rows[sidm]}
        if native_cancers[sidm]: sources.add("NATIVE_GDSC_TCGA_DESC")
        contexts[sidm] = {"cancer_id": cancer, "tissue": tissue, "mapping_method": method,
                          "context_available": cancer is not None, "context_unavailable_reason": reason,
                          "context_source": "|".join(sorted(sources)) or "CMP_TISSUE"}
        grouped = defaultdict(set)
        for _, source, system, subtype in annotation_rows[sidm]:
            if system: grouped[system].add(subtype)
        for system, values in grouped.items():
            available = len(values) == 1 and cancer is not None
            subtypes.append({"canonical_model_id": sidm, "cancer_id": cancer, "subtype_system": system,
                "subtype_id": next(iter(values)) if available else None, "subtype_available": available,
                "subtype_unavailable_reason": None if available else "CONFLICTING_OR_UNRESOLVED_MODEL_SUBTYPE",
                "source": "|".join(sorted(r[1] for r in annotation_rows[sidm] if r[2] == system))})
    rows = []
    for dataset, native, sidm in sorted(set(registry)):
        context = contexts.get(sidm, {"cancer_id": None, "tissue": "", "mapping_method": "UNRESOLVED",
            "context_available": False, "context_unavailable_reason": "NO_CANONICAL_MODEL_CROSSWALK", "context_source": "CMP_IDENTIFIERS"})
        rows.append({"dataset_id": dataset, "cell_line_id": native, "canonical_model_id": sidm or None, **context})
    audit = pd.DataFrame(rows)
    columns = ["canonical_model_id", "cancer_id", "subtype_system", "subtype_id", "subtype_available", "subtype_unavailable_reason", "source"]
    return audit.loc[audit.context_available].copy(), audit, pd.DataFrame(subtypes, columns=columns)
