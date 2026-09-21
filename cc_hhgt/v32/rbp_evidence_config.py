"""Phase 11: the ``rbp_evidence`` configuration block and the four ablation modes.

The plan requires four mutually comparable modes:

===========  ==========================================================
mode         behaviour
===========  ==========================================================
``A``        ``legacy_generic_binding`` -- the historical flat
             ``binds_protein`` relation, no assay typing, no ENCODE
``B``        ``typed_binding_only`` -- assay typing preserved, no ENCODE
``C``        ``typed_binding_plus_encode_eclip`` -- B + ENCODE eCLIP
``D``        ``full_rbp_evidence`` -- C + RBP knockdown in the Evidence layer
===========  ==========================================================

Fairness is **enforced in code**, not by convention.  The plan states:

    四种模式：相同 patient folds / 相同 candidate universe / 相同 LASSO base /
    相同 label / 相同 random seeds / 相同训练预算
    禁止改变其他模态来配合结果

so :func:`assert_fair_comparison` compares every mode against mode A on exactly
those axes and raises if any of them moved.  A mode that silently changed the
label or the fold assignment to improve its own score cannot be constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

__all__ = [
    "ABLATION_MODES",
    "FAIRNESS_KEYS",
    "MODE_A",
    "MODE_B",
    "MODE_C",
    "MODE_D",
    "RbpEvidenceConfig",
    "assert_fair_comparison",
    "config_from_mapping",
    "config_to_yaml_block",
    "resolve_mode",
]

MODE_A = "legacy_generic_binding"
MODE_B = "typed_binding_only"
MODE_C = "typed_binding_plus_encode_eclip"
MODE_D = "full_rbp_evidence"

ABLATION_MODES: tuple[str, ...] = (MODE_A, MODE_B, MODE_C, MODE_D)

#: Axes that must be byte-identical across every ablation mode.  Anything a
#: reviewer would call "changing the experiment to help the result" lives here.
FAIRNESS_KEYS: tuple[str, ...] = (
    "seed",
    "n_folds",
    "candidate_universe_sha256",
    "lasso_base_sha256",
    "label_definition",
    "patient_fold_manifest_sha256",
    "training_budget_epochs",
    "learning_rate",
    "batch_size",
    "evaluation_metric",
)


@dataclass(frozen=True)
class RbpEvidenceConfig:
    """One ablation mode.

    The switch names mirror the YAML block the plan specifies, so the dataclass
    and the config file cannot drift apart.
    """

    mode: str

    # --- Phase 11 switches -------------------------------------------------
    preserve_assay_type: bool
    typed_binding_relations: bool
    include_predicted_binding: bool
    encode_eclip: bool
    encode_rbp_kd: bool
    encode_rbp_kd_primary_graph: bool
    encode_rbp_kd_evidence_only: bool
    rbns: bool

    # --- frozen axes (identical in every mode) ----------------------------
    seed: int = 20260726
    n_folds: int = 5
    # Mirrors cc_hhgt.v32.evidence_training.FORMAL_CANDIDATE_SHA256
    candidate_universe_sha256: str = (
        "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
    )
    # Composite SHA256 over the ALREADY-EXISTING LASSO baseline artifacts.  The
    # user's instruction is explicit: reuse them, never re-materialise.  The
    # digest is a leaf-hash of sorted "<sha256>  <name>\n" lines over
    # posttraining_lasso_full/; see manifests/LASSO_BASE_BINDING.json.
    lasso_base_sha256: str = (
        "6c1cd82c5ad14baa76b180befc4ed69aad5232759af0213d9e04926ad0cfb066"
    )
    label_definition: str = "V32_FORMAL_STRONG_PLUS_WEAK_POSITIVE"
    # Mirrors patient_fold_authority.FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256
    patient_fold_manifest_sha256: str = (
        "e05c20085be532159bb6a51f200a27923975c3caa6a90910290e10a5b60e6253"
    )
    training_budget_epochs: int = 40
    learning_rate: float = 2.0e-3
    batch_size: int = 128
    evaluation_metric: str = "AUPRC"

    # --- invariants that must not move in ANY mode ------------------------
    changes_primary_ranking: bool = False
    auxiliary_gradients_into_core: bool = False
    rbp_kd_is_lncrna_function_truth: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def switch_dict(self) -> dict[str, Any]:
        """The subset that actually differs between modes."""

        return {
            key: value
            for key, value in self.as_dict().items()
            if key not in FAIRNESS_KEYS and key != "mode"
        }


def resolve_mode(mode: str) -> RbpEvidenceConfig:
    """Return the canonical configuration for one ablation mode."""

    key = str(mode).strip().lower()
    if key in {"a", MODE_A}:
        return RbpEvidenceConfig(
            mode=MODE_A,
            preserve_assay_type=False,
            typed_binding_relations=False,
            include_predicted_binding=False,
            encode_eclip=False,
            encode_rbp_kd=False,
            encode_rbp_kd_primary_graph=False,
            encode_rbp_kd_evidence_only=True,
            rbns=False,
        )
    if key in {"b", MODE_B}:
        return RbpEvidenceConfig(
            mode=MODE_B,
            preserve_assay_type=True,
            typed_binding_relations=True,
            include_predicted_binding=False,
            encode_eclip=False,
            encode_rbp_kd=False,
            encode_rbp_kd_primary_graph=False,
            encode_rbp_kd_evidence_only=True,
            rbns=False,
        )
    if key in {"c", MODE_C}:
        return RbpEvidenceConfig(
            mode=MODE_C,
            preserve_assay_type=True,
            typed_binding_relations=True,
            include_predicted_binding=False,
            encode_eclip=True,
            encode_rbp_kd=False,
            encode_rbp_kd_primary_graph=False,
            encode_rbp_kd_evidence_only=True,
            rbns=False,
        )
    if key in {"d", MODE_D}:
        # RBP knockdown reaches the Evidence layer only.  It is never a lncRNA
        # functional truth and never a primary-graph edge.
        return RbpEvidenceConfig(
            mode=MODE_D,
            preserve_assay_type=True,
            typed_binding_relations=True,
            include_predicted_binding=False,
            encode_eclip=True,
            encode_rbp_kd=True,
            encode_rbp_kd_primary_graph=False,
            encode_rbp_kd_evidence_only=True,
            rbns=False,
        )
    raise ValueError(f"unknown ablation mode: {mode!r}; expected one of {ABLATION_MODES}")


def assert_fair_comparison(
    configs: Sequence[RbpEvidenceConfig],
    *,
    baseline: RbpEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Raise unless every mode shares the frozen axes with the baseline.

    Also enforces the mode-independent invariants: no mode may change primary
    ranking, push auxiliary gradients into the core, or treat RBP knockdown as
    lncRNA functional truth.
    """

    if not configs:
        raise ValueError("no configurations supplied")
    base = baseline or configs[0]

    mismatches: dict[str, dict[str, tuple[Any, Any]]] = {}
    for config in configs:
        differing: dict[str, tuple[Any, Any]] = {}
        for key in FAIRNESS_KEYS:
            mine = getattr(config, key)
            theirs = getattr(base, key)
            if mine != theirs:
                differing[key] = (theirs, mine)
        if differing:
            mismatches[config.mode] = differing
    if mismatches:
        raise RuntimeError(
            "ablation modes are not comparable: the frozen axes moved "
            f"(baseline={base.mode}): {mismatches}"
        )

    for config in configs:
        if config.changes_primary_ranking:
            raise RuntimeError(f"{config.mode}: changes_primary_ranking must stay false")
        if config.auxiliary_gradients_into_core:
            raise RuntimeError(
                f"{config.mode}: auxiliary_gradients_into_core must stay false"
            )
        if config.rbp_kd_is_lncrna_function_truth:
            raise RuntimeError(
                f"{config.mode}: RBP knockdown must never be lncRNA functional truth"
            )
        if config.encode_rbp_kd and config.encode_rbp_kd_primary_graph:
            raise RuntimeError(
                f"{config.mode}: RBP knockdown must not enter the primary graph"
            )
        if config.encode_rbp_kd and not config.encode_rbp_kd_evidence_only:
            raise RuntimeError(
                f"{config.mode}: RBP knockdown must be evidence_only"
            )

    return {
        "modes": [config.mode for config in configs],
        "baseline": base.mode,
        "frozen_axes": {key: getattr(base, key) for key in FAIRNESS_KEYS},
        "switch_matrix": {config.mode: config.switch_dict() for config in configs},
        "status": "FAIR",
    }


def config_to_yaml_block(config: RbpEvidenceConfig, *, indent: int = 0) -> str:
    """Render the plan's YAML block deterministically (no external dependency)."""

    pad = " " * indent

    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return repr(value)
        return str(value)

    lines = [
        f"{pad}rbp_evidence:",
        f"{pad}  mode: {config.mode}",
        f"{pad}  preserve_assay_type: {scalar(config.preserve_assay_type)}",
        f"{pad}  typed_binding_relations:",
        f"{pad}    enabled: {scalar(config.typed_binding_relations)}",
        f"{pad}    include_predicted: {scalar(config.include_predicted_binding)}",
        f"{pad}  encode_eclip:",
        f"{pad}    enabled: {scalar(config.encode_eclip)}",
        f"{pad}  encode_rbp_kd:",
        f"{pad}    enabled: {scalar(config.encode_rbp_kd)}",
        f"{pad}    primary_graph: {scalar(config.encode_rbp_kd_primary_graph)}",
        f"{pad}    evidence_only: {scalar(config.encode_rbp_kd_evidence_only)}",
        f"{pad}  rbns:",
        f"{pad}    enabled: {scalar(config.rbns)}",
        f"{pad}  fairness:",
    ]
    for key in FAIRNESS_KEYS:
        lines.append(f"{pad}    {key}: {scalar(getattr(config, key))}")
    lines.extend(
        [
            f"{pad}  invariants:",
            f"{pad}    changes_primary_ranking: {scalar(config.changes_primary_ranking)}",
            f"{pad}    auxiliary_gradients_into_core: "
            f"{scalar(config.auxiliary_gradients_into_core)}",
            f"{pad}    rbp_kd_is_lncrna_function_truth: "
            f"{scalar(config.rbp_kd_is_lncrna_function_truth)}",
        ]
    )
    return "\n".join(lines) + "\n"


def config_from_mapping(payload: Mapping[str, Any], *, mode: str | None = None) -> RbpEvidenceConfig:
    """Load a config from a parsed YAML mapping.

    Missing keys fall back to the canonical mode defaults, so a partial config
    cannot accidentally relax a fairness axis.
    """

    block = payload.get("rbp_evidence", payload)
    resolved_mode = str(mode or block.get("mode") or MODE_A)
    base = resolve_mode(resolved_mode)

    typed = block.get("typed_binding_relations") or {}
    eclip = block.get("encode_eclip") or {}
    kd = block.get("encode_rbp_kd") or {}
    rbns = block.get("rbns") or {}
    fairness = block.get("fairness") or {}
    invariants = block.get("invariants") or {}

    updates: dict[str, Any] = {
        "preserve_assay_type": bool(
            block.get("preserve_assay_type", base.preserve_assay_type)
        ),
        "typed_binding_relations": bool(
            typed.get("enabled", base.typed_binding_relations)
        ),
        "include_predicted_binding": bool(
            typed.get("include_predicted", base.include_predicted_binding)
        ),
        "encode_eclip": bool(eclip.get("enabled", base.encode_eclip)),
        "encode_rbp_kd": bool(kd.get("enabled", base.encode_rbp_kd)),
        "encode_rbp_kd_primary_graph": bool(
            kd.get("primary_graph", base.encode_rbp_kd_primary_graph)
        ),
        "encode_rbp_kd_evidence_only": bool(
            kd.get("evidence_only", base.encode_rbp_kd_evidence_only)
        ),
        "rbns": bool(rbns.get("enabled", base.rbns)),
        "changes_primary_ranking": bool(
            invariants.get("changes_primary_ranking", base.changes_primary_ranking)
        ),
        "auxiliary_gradients_into_core": bool(
            invariants.get(
                "auxiliary_gradients_into_core", base.auxiliary_gradients_into_core
            )
        ),
        "rbp_kd_is_lncrna_function_truth": bool(
            invariants.get(
                "rbp_kd_is_lncrna_function_truth", base.rbp_kd_is_lncrna_function_truth
            )
        ),
    }
    for key in FAIRNESS_KEYS:
        if key in fairness:
            updates[key] = fairness[key]
    return replace(base, **updates)
