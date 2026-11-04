"""Reject retired MOGSA targets; this module contains no scoring implementation."""
from pathlib import Path
from .common import fail, read_json

RETIREMENT_DATE = '2026-11-04'
MESSAGE = ('MOGSA and its derived lavaan parent targets were retired on 2026-11-04. '
           'Historical scores and prepared labels cannot be used for new training. '
           'No automatic RNA or alternative-scoring fallback is allowed.')


def retired():
    fail('MOGSA_RETIRED', MESSAGE)


def validate_fusion_config(cfg):
    c = cfg.get('pathway_fusion', {})
    if not isinstance(c, dict) or type(c.get('enabled', False)) is not bool:
        fail('FUSION_CONFIG', 'Explicit boolean enabled required')
    # Merely changing enabled to false must not relabel an old fusion config.
    if c.get('enabled') or c.get('format') or c.get('complete') or c.get('cache_root'):
        retired()


def require_pathway_targets(cfg):
    validate_fusion_config(cfg)
    policy = cfg.get('pathway_target_policy', {})
    if not isinstance(policy, dict):
        fail('PATHWAY_TARGET_POLICY', 'Expected a pathway target policy object')
    if policy.get('status') == 'REPLACEMENT_REQUIRED':
        fail('PATHWAY_TARGET_REPLACEMENT_REQUIRED',
             'MOGSA is retired. The per-assay replacement is not implemented or validated; '
             'do not silently promote existing RNA activity scores to replacement targets.')
    if policy.get('status') not in (None, 'EXPLICIT_EXISTING_RNA'):
        fail('PATHWAY_TARGET_POLICY', 'Unknown pathway target policy')


def reject_retired_prepared(root, cfg=None):
    root = Path(root)
    if cfg is None:
        cfg = read_json(root/'CONFIG.json')
    validate_fusion_config(cfg)
    contract = root/'TASK_CONTRACT.json'
    if contract.is_file() and 'MOGSA' in str(read_json(contract).get('pathway_score_source', '')).upper():
        retired()
    inputs = root/'INPUT_HASHES.json'
    if inputs.is_file() and any(key.startswith('pathway_fusion::') for key in read_json(inputs)):
        retired()
    if (root/'PATHWAY_FUSION.json').exists() or next(root.glob('views/*/PATHWAY_FUSION.json'), None) is not None:
        retired()
    require_pathway_targets(cfg)
