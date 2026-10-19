"""Evidence-derived coverage of registered native model inputs.

This boundary concerns registered study sources, not the truth of upstream
experiments. Unknown ancestry remains blocking. Only the explicit GDC alignment
and biospecimen-veracity limit is outside this boundary after native replay.
"""
import pandas as pd

from .evidence_rule_summary import assign_study_components

ENTITY = "ENTITY_EXPOSURE"
STUDY = "STUDY_SOURCE"
NONBLOCKING_LIMITS = {"GDC_UPSTREAM_ALIGNMENT_AND_BIOSPECIMEN_VERACITY_UNVERIFIED"}
COLUMNS = ["model_id", "dependency_context", "dependency_records", "unresolved_dependency_count",
           "entity_exposure_rows", "study_source_rows", "unverified_study_source_rows",
           "blocking_gap_count", "upstream_veracity_limit_count", "inventory_complete"]


def record_kinds(frame):
    """Manual source declarations cannot opt out of study provenance checks."""
    kinds = frame.get("usage_record_kind", pd.Series(STUDY, index=frame.index)).fillna(STUDY)
    if not kinds.isin([ENTITY, STUDY]).all():
        raise ValueError("Unknown model-use record kind")
    entities = kinds.eq(ENTITY)
    if entities.any():
        required = {"assay", "source_id", "entity_scope", "model_id", "lncrna_id", "study_tokens_json"}
        if not required <= set(frame): raise ValueError("Native entity exposure lacks typed identity")
        selected = frame.loc[entities]
        if (not selected.assay.eq("NATIVE_MODEL_INPUT_EXPOSURE").all()
                or not selected.source_id.str.startswith("NATIVE-USE:").all()
                or not selected.entity_scope.eq("LNCRNA_EXPOSURE_ALL_PATHWAYS").all()
                or selected.model_id.fillna("").eq("").any()
                or selected.lncrna_id.fillna("").eq("").any()
                or not selected.study_tokens_json.eq("[]").all()):
            raise ValueError("Invalid native entity exposure record")
    # A native entity record must not silently fall back to the study category.
    if "assay" in frame and (frame.assay.eq("NATIVE_MODEL_INPUT_EXPOSURE") & ~entities).any():
        raise ValueError("Native entity exposure is missing its record kind")
    return kinds


def assess(models, tables):
    """Every model and selection/expert context must independently be covered.

    The union cannot hide an uncovered losing comparator or zero-weight expert.
    Counts concern input records, not patients, gradients or independent studies.
    """
    names = set(models)
    if not names: raise ValueError("Model-use completeness needs evaluated models")
    usage, dependencies, gaps = (tables[k] for k in ("usage", "dependencies", "gaps"))
    kinds = record_kinds(usage)
    for frame in (usage, dependencies, gaps):
        if not set(frame.model_id) <= names:
            raise ValueError("Model-use coverage contains an undeclared model")
        if frame.dependency_context.isna().any() or frame.dependency_context.eq("").any():
            raise ValueError("Model-use coverage has an unidentified dependency context")
    annotated = assign_study_components(usage)
    rows = []
    for model in sorted(names):
        contexts = set().union(*(set(f.loc[f.model_id.eq(model), "dependency_context"])
                                 for f in (usage, dependencies, gaps))) or {"SELECTED_SCORE"}
        for context in sorted(contexts):
            select = usage.model_id.eq(model) & usage.dependency_context.eq(context)
            sources = annotated.loc[select & kinds.eq(STUDY)]
            dep = dependencies.loc[dependencies.model_id.eq(model) & dependencies.dependency_context.eq(context)]
            missing = gaps.loc[gaps.model_id.eq(model) & gaps.dependency_context.eq(context)]
            unresolved = int((~dep.status.isin(["VERIFIED_FILE", "VERIFIED_DIRECTORY"])).sum())
            blocking = int((~missing.reason.isin(NONBLOCKING_LIMITS)).sum())
            unverified = int((~sources.provenance_verified).sum())
            complete = bool(len(dep) and len(sources) and not (unresolved or blocking or unverified))
            rows.append(dict(zip(COLUMNS, [model, context, len(dep), unresolved,
                int((select & kinds.eq(ENTITY)).sum()), len(sources), unverified, blocking,
                int(missing.reason.isin(NONBLOCKING_LIMITS).sum()), complete])))
    return pd.DataFrame(rows, columns=COLUMNS)


def is_complete(coverage):
    return bool(len(coverage) and coverage.inventory_complete.all())
