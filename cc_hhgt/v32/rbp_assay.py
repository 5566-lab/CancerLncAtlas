"""Deterministic RBP / lncRNA-protein assay taxonomy.

Context
-------
The v3.2 evidence chain compresses every physical-binding assay into the single
token ``physical_binding`` (see ``evidence_interaction_rematerialization.experiment_family``).
That compression is lossy (eCLIP and RIP become indistinguishable) and, because the
historical regex matches bare substrings, also noisy: ``rip`` matches inside
``transcript`` / ``manuscript`` / ``description`` and ``rap`` matches inside
``graph`` / ``rapid``.

This module introduces a two-level, closed taxonomy that preserves the original
experiment string verbatim while exposing a stable, model-usable category.

Levels
------
``experiment_raw``
    The original string, byte-for-byte. Never normalised, never discarded.
``assay_subtype``
    Fine-grained normalised category. Used for provenance, audits and the
    Evidence layer.
``graph_assay_class``
    Small closed vocabulary. This is the *only* value allowed to influence
    graph relation typing, so that relation-type cardinality stays bounded.
``experiment_family``
    The historical coarse family, retained so that legacy behaviour can be
    reproduced exactly.

Determinism contract
--------------------
* Rules are evaluated in a fixed order; first match wins.
* Matching is pure ``re`` over a lowercased copy of the raw string.
* There is no randomness, no hashing of unordered containers, no model
  inference and no data-dependent state.
* Identical input always produces identical output.

Fail-closed contract
--------------------
* An empty or unrecognised string yields ``unspecified`` / ``unknown``.
  It is never silently promoted to an experimental class.
* An explicit ``predicted`` / ``not experimental`` flag always wins over any
  textual match, so a computational record can never be relabelled as an
  experimental observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

__all__ = [
    "ASSAY_SUBTYPES",
    "GRAPH_ASSAY_CLASSES",
    "AssayAssignment",
    "TAXONOMY_VERSION",
    "classify_assay",
    "classify_frame",
    "graph_assay_relation_type",
    "legacy_experiment_family",
    "taxonomy_table",
]

TAXONOMY_VERSION = "RBP_ASSAY_TAXONOMY_V1"

# --------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------

#: Fine-grained subtypes. Order is the declaration order used in exports.
ASSAY_SUBTYPES: tuple[str, ...] = (
    "eclip",
    "par_clip",
    "iclip",
    "hits_clip",
    "clip_unspecified",
    "rip",
    "chirp",
    "rap",
    "chart",
    "rna_pulldown",
    "emsa",
    "co_ip",
    "functional_perturbation",
    "reporter_assay",
    "expression_or_abundance",
    "computational_prediction",
    "other_experimental",
    "unspecified",
)

#: Coarse classes. The *only* taxonomy level permitted to type graph relations.
GRAPH_ASSAY_CLASSES: tuple[str, ...] = (
    "eclip",
    "other_clip",
    "rip",
    "rna_capture",
    "other_physical",
    "experimental_unspecified",
    "predicted",
    "unknown",
)

#: Coarse families, matching the historical ``experiment_family`` vocabulary.
EXPERIMENT_FAMILIES: tuple[str, ...] = (
    "physical_binding",
    "functional_perturbation",
    "reporter_assay",
    "expression_or_abundance",
    "computational_prediction",
    "other_experimental",
    "unspecified",
)

#: Graph relation types emitted for typed binding edges.
GRAPH_RELATION_PREFIX = "binds_protein_"

#: Mapping from graph assay class to the relation type used in the main graph.
GRAPH_ASSAY_RELATION_TYPES: Mapping[str, str] = {
    cls: f"{GRAPH_RELATION_PREFIX}{cls}" for cls in GRAPH_ASSAY_CLASSES
}

# --------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------
# Each rule: (rule_id, assay_subtype, graph_assay_class, experiment_family, regex)
#
# Boundary discipline: ``(?<![a-z0-9])`` / ``(?![a-z0-9])`` are used instead of
# ``\b`` because method strings frequently contain hyphens, digits and slashes
# (``PAR-CLIP``, ``HITS-CLIP``, ``RIP-Seq``, ``RNA:protein``) where ``\b``
# behaves inconsistently. This is what prevents the historical
# ``transcript``/``graph`` false positives.
_RULES: tuple[tuple[str, str, str, str, re.Pattern[str]], ...] = (
    # --- computational FIRST -------------------------------------------------
    # A string that self-describes as a prediction must never be recorded as an
    # observation, even when it *mentions* an assay dataset.  Real NPInter5 data
    # contains 116,499 rows of the form "Conserved miRNAs target sites predicted
    # by TargetScan and miRanda overlap with the AGO CLIP dataset" -- a target
    # prediction that merely references the AGO CLIP dataset.  The historical
    # regex classified every one of them as ``physical_binding``, and a
    # CLIP-first ordering classified them as ``clip_unspecified``.  Both are
    # wrong: prediction language therefore outranks assay keyword matching.
    # ``catrapid`` is required by real data: 30,276 lncRNA-RBP rows carry the
    # literal method string "catRAPID", a sequence-based interaction *prediction*
    # server.  Without it those rows would be recorded as experimental binding.
    ("computational_prediction", "computational_prediction", "predicted",
     "computational_prediction",
     re.compile(r"predict|computational|in[-\s_]?silico|inferred|algorithm|catrapid")),
    # --- CLIP family: specific variants must precede the generic form -------
    ("eclip", "eclip", "eclip", "physical_binding",
     re.compile(r"(?<![a-z0-9])e[-\s_]?clip(?![a-z0-9])")),
    ("par_clip", "par_clip", "other_clip", "physical_binding",
     re.compile(r"(?<![a-z0-9])par[-\s_]?clip(?![a-z0-9])")),
    ("iclip", "iclip", "other_clip", "physical_binding",
     re.compile(r"(?<![a-z0-9])i[-\s_]?clip(?![a-z0-9])")),
    ("hits_clip", "hits_clip", "other_clip", "physical_binding",
     re.compile(r"(?<![a-z0-9])hits[-\s_]?clip(?![a-z0-9])")),
    # Generic CLIP.  The optional short alphabetic prefix is required by real
    # data: easyCLIP, fCLIP, fPARCLIP, GoldCLIP, pCLIP, seCLIP-Seq, PARCLIP and
    # MY-CLIP are genuine CLIP variants (18,378 rows in NPInter5) that a strict
    # left boundary would wrongly demote out of the physical-binding classes.
    # It cannot match ordinary prose, because words such as "transcript" or
    # "description" do not contain the substring "clip" at all.
    ("clip_unspecified", "clip_unspecified", "other_clip", "physical_binding",
     re.compile(r"(?<![a-z0-9])[a-z]{0,6}[-\s_]?clip(?![a-z0-9])")),
    # --- RNA-capture family -------------------------------------------------
    ("chirp", "chirp", "rna_capture", "physical_binding",
     re.compile(r"(?<![a-z0-9])chirp(?![a-z0-9])")),
    ("chart", "chart", "rna_capture", "physical_binding",
     re.compile(r"(?<![a-z0-9])chart(?![a-z0-9])")),
    ("rap", "rap", "rna_capture", "physical_binding",
     re.compile(r"(?<![a-z0-9])rap(?:[-\s_]?(?:ms|rna|seq))?(?![a-z0-9])")),
    ("rna_pulldown", "rna_pulldown", "rna_capture", "physical_binding",
     re.compile(r"rna[-\s_]?pull[-\s_]?down|pull[-\s_]?down")),
    # --- RIP ----------------------------------------------------------------
    ("rip", "rip", "rip", "physical_binding",
     re.compile(r"(?<![a-z0-9])rip(?:[-\s_]?seq)?(?![a-z0-9])")),
    # --- other physical -----------------------------------------------------
    ("emsa", "emsa", "other_physical", "physical_binding",
     re.compile(r"(?<![a-z0-9])emsa(?![a-z0-9])")),
    ("co_ip", "co_ip", "other_physical", "physical_binding",
     re.compile(r"co[-\s_]?ip(?![a-z0-9])|co[-\s_]?immunoprecipitat")),
    ("immunoprecipitation", "other_experimental", "other_physical", "physical_binding",
     re.compile(r"immunoprecipitat|(?<![a-z0-9])ip(?![a-z0-9])")),
    # --- functional perturbation -------------------------------------------
    ("functional_perturbation", "functional_perturbation",
     "experimental_unspecified", "functional_perturbation",
     re.compile(r"knock|sirna|shrna|rnai|crispr|cas9|overexpress|transfect|deplet|silenc")),
    # --- reporter -----------------------------------------------------------
    ("reporter_assay", "reporter_assay", "experimental_unspecified", "reporter_assay",
     re.compile(r"luciferase|reporter")),
    # --- expression / abundance --------------------------------------------
    ("expression_or_abundance", "expression_or_abundance",
     "experimental_unspecified", "expression_or_abundance",
     re.compile(r"qpcr|rt[-\s_]?pcr|rna[-\s_]?seq|microarray|western|northern|\bchip[-\s_]?seq\b")),
    # --- already-collapsed family tokens ------------------------------------
    # These come LAST so that fine-grained text always wins.  They exist because
    # the upstream rematerialiser already emits coarse ``experiment_family``
    # values; without them a pre-collapsed row would fall through to
    # ``other_experimental`` and lose its physical semantics entirely.
    ("family_physical_binding", "other_experimental", "other_physical",
     "physical_binding",
     re.compile(r"physical[-\s_]?binding|physical[-\s_]?interaction")),
    ("family_functional_perturbation", "functional_perturbation",
     "experimental_unspecified", "functional_perturbation",
     re.compile(r"functional[-\s_]?perturbation")),
    ("family_reporter_assay", "reporter_assay", "experimental_unspecified",
     "reporter_assay",
     re.compile(r"reporter[-\s_]?assay")),
    ("family_expression_or_abundance", "expression_or_abundance",
     "experimental_unspecified", "expression_or_abundance",
     re.compile(r"expression[-\s_]?or[-\s_]?abundance")),
    ("family_computational_prediction", "computational_prediction", "predicted",
     "computational_prediction",
     re.compile(r"computational[-\s_]?prediction")),
)

#: The historical regex, reproduced verbatim for legacy-equivalence testing.
_LEGACY_PHYSICAL = re.compile(
    r"clip|chirp|chart|rap|rip|pull.?down|immunoprecip|\bip\b|emsa"
)
_LEGACY_PERTURBATION = re.compile(r"knock|sirna|shrna|crispr|overexpress|transfect|deplet")
_LEGACY_REPORTER = re.compile(r"luciferase|reporter")
_LEGACY_EXPRESSION = re.compile(r"qpcr|rt-pcr|rna-seq|microarray|western")

_UNSPECIFIED = "unspecified"
_UNKNOWN = "unknown"
_OTHER_EXPERIMENTAL = "other_experimental"
_EXPERIMENTAL_UNSPECIFIED = "experimental_unspecified"


def _clean(value: object) -> str:
    """Return the raw string with only surrounding whitespace removed.

    Interior characters (case, hyphens, slashes) are preserved verbatim so that
    provenance is never altered.
    """

    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def legacy_experiment_family(value: object) -> str:
    """Reproduce the historical ``experiment_family`` mapping exactly.

    Kept for legacy-ablation equivalence only. It intentionally retains the
    historical substring behaviour (including its false positives) so that mode
    A can be shown to reproduce the old semantics bit-for-bit.
    """

    text = _clean(value).lower()
    if _LEGACY_PHYSICAL.search(text):
        return "physical_binding"
    if _LEGACY_PERTURBATION.search(text):
        return "functional_perturbation"
    if _LEGACY_REPORTER.search(text):
        return "reporter_assay"
    if _LEGACY_EXPRESSION.search(text):
        return "expression_or_abundance"
    if not text:
        return _UNSPECIFIED
    return _OTHER_EXPERIMENTAL


@dataclass(frozen=True)
class AssayAssignment:
    """Immutable classification result for one experiment string."""

    experiment_raw: str
    assay_subtype: str
    graph_assay_class: str
    experiment_family: str
    matched_rule: str
    is_experimental: bool
    is_predicted: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "experiment_raw": self.experiment_raw,
            "assay_subtype": self.assay_subtype,
            "graph_assay_class": self.graph_assay_class,
            "experiment_family": self.experiment_family,
            "matched_rule": self.matched_rule,
            "is_experimental": self.is_experimental,
            "is_predicted": self.is_predicted,
        }


def _as_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def classify_assay(
    experiment_raw: object,
    *,
    is_predicted: object = None,
    is_experimental: object = None,
) -> AssayAssignment:
    """Classify one experiment string into the closed taxonomy.

    Parameters
    ----------
    experiment_raw
        The original experiment / method string. Preserved verbatim in the
        result.
    is_predicted, is_experimental
        Optional explicit flags from the source record. When ``is_predicted``
        is true, or ``is_experimental`` is explicitly false, the result is
        forced to ``computational_prediction`` / ``predicted`` regardless of
        any textual match. This guarantees a computational record can never be
        presented as an experimental observation.
    """

    raw = _clean(experiment_raw)
    text = raw.lower()

    predicted_flag = _as_optional_bool(is_predicted)
    experimental_flag = _as_optional_bool(is_experimental)

    forced_predicted = bool(predicted_flag) or (experimental_flag is False)

    if forced_predicted:
        return AssayAssignment(
            experiment_raw=raw,
            assay_subtype="computational_prediction",
            graph_assay_class="predicted",
            experiment_family="computational_prediction",
            matched_rule="explicit_computational_flag",
            is_experimental=False,
            is_predicted=True,
        )

    if not text:
        return AssayAssignment(
            experiment_raw=raw,
            assay_subtype=_UNSPECIFIED,
            graph_assay_class=_UNKNOWN,
            experiment_family=_UNSPECIFIED,
            matched_rule="empty",
            is_experimental=False,
            is_predicted=False,
        )

    for rule_id, subtype, graph_class, family, pattern in _RULES:
        if pattern.search(text):
            is_predicted_class = graph_class == "predicted"
            return AssayAssignment(
                experiment_raw=raw,
                assay_subtype=subtype,
                graph_assay_class=graph_class,
                experiment_family=family,
                matched_rule=rule_id,
                # Derived from the class, never hard-coded: a textual prediction
                # match must not be able to claim it is an observation.
                is_experimental=not is_predicted_class,
                is_predicted=is_predicted_class,
            )

    # Fail closed: an unrecognised non-empty string is an *experimental*
    # observation of unknown type, not a claim of binding and not a prediction.
    return AssayAssignment(
        experiment_raw=raw,
        assay_subtype=_OTHER_EXPERIMENTAL,
        graph_assay_class=_EXPERIMENTAL_UNSPECIFIED,
        experiment_family=_OTHER_EXPERIMENTAL,
        matched_rule="fallback_other_experimental",
        is_experimental=True,
        is_predicted=False,
    )


def classify_frame(
    frame: pd.DataFrame,
    *,
    raw_column: str = "experiment_raw",
    predicted_column: str | None = "is_predicted",
    experimental_column: str | None = "is_experimental",
    fallback_columns: Sequence[str] = ("experiment_family", "assay_type", "experimental_system"),
) -> pd.DataFrame:
    """Classify a frame, returning the taxonomy columns aligned to ``frame.index``.

    When ``raw_column`` is absent, ``fallback_columns`` are consulted in order so
    that frames produced before this module existed still classify sensibly.
    The original ``experiment_raw`` is *not* overwritten.
    """

    if raw_column in frame.columns:
        raw = frame[raw_column]
    else:
        raw = pd.Series([""] * len(frame), index=frame.index, dtype="object")
        for candidate in fallback_columns:
            if candidate in frame.columns:
                missing = raw.map(_clean).eq("")
                if missing.any():
                    raw = raw.where(~missing, frame[candidate])
                break

    predicted = (
        frame[predicted_column]
        if predicted_column and predicted_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index, dtype="object")
    )
    experimental = (
        frame[experimental_column]
        if experimental_column and experimental_column in frame.columns
        else pd.Series([None] * len(frame), index=frame.index, dtype="object")
    )

    if len(frame) == 0:
        empty = pd.DataFrame(
            {
                "experiment_raw": pd.Series(dtype="object"),
                "assay_subtype": pd.Series(dtype="object"),
                "graph_assay_class": pd.Series(dtype="object"),
                "experiment_family": pd.Series(dtype="object"),
                "matched_rule": pd.Series(dtype="object"),
                "is_experimental": pd.Series(dtype="bool"),
                "is_predicted": pd.Series(dtype="bool"),
            },
            index=frame.index,
        )
        return empty

    records = [
        classify_assay(r, is_predicted=p, is_experimental=e).as_dict()
        for r, p, e in zip(raw.tolist(), predicted.tolist(), experimental.tolist())
    ]
    out = pd.DataFrame.from_records(records, index=frame.index)
    out["experiment_raw"] = raw.map(_clean).to_numpy()
    return out


def graph_assay_relation_type(graph_assay_class: object) -> str:
    """Map a graph assay class to its relation type, failing closed to unknown."""

    cls = _clean(graph_assay_class).lower()
    if cls not in GRAPH_ASSAY_RELATION_TYPES:
        cls = _UNKNOWN
    return GRAPH_ASSAY_RELATION_TYPES[cls]


def taxonomy_table() -> pd.DataFrame:
    """Return the full rule table as a deterministic, exportable frame."""

    rows: list[dict[str, object]] = []
    for order, (rule_id, subtype, graph_class, family, pattern) in enumerate(_RULES):
        rows.append(
            {
                "rule_order": order,
                "matched_rule": rule_id,
                "assay_subtype": subtype,
                "graph_assay_class": graph_class,
                "experiment_family": family,
                "regex": pattern.pattern,
                "relation_type": GRAPH_ASSAY_RELATION_TYPES[graph_class],
                "taxonomy_version": TAXONOMY_VERSION,
            }
        )
    rows.append(
        {
            "rule_order": len(_RULES),
            "matched_rule": "empty",
            "assay_subtype": _UNSPECIFIED,
            "graph_assay_class": _UNKNOWN,
            "experiment_family": _UNSPECIFIED,
            "regex": "",
            "relation_type": GRAPH_ASSAY_RELATION_TYPES[_UNKNOWN],
            "taxonomy_version": TAXONOMY_VERSION,
        }
    )
    rows.append(
        {
            "rule_order": len(_RULES) + 1,
            "matched_rule": "fallback_other_experimental",
            "assay_subtype": _OTHER_EXPERIMENTAL,
            "graph_assay_class": _EXPERIMENTAL_UNSPECIFIED,
            "experiment_family": _OTHER_EXPERIMENTAL,
            "regex": "",
            "relation_type": GRAPH_ASSAY_RELATION_TYPES[_EXPERIMENTAL_UNSPECIFIED],
            "taxonomy_version": TAXONOMY_VERSION,
        }
    )
    return pd.DataFrame.from_records(rows)


def taxonomy_coverage(values: Iterable[object]) -> pd.DataFrame:
    """Count assay subtypes over an iterable of raw strings (audit helper)."""

    from collections import Counter

    counter: Counter[str] = Counter()
    for value in values:
        counter[classify_assay(value).assay_subtype] += 1
    frame = pd.DataFrame(
        sorted(counter.items()), columns=["assay_subtype", "n"]
    ).sort_values("n", ascending=False, ignore_index=True)
    return frame
