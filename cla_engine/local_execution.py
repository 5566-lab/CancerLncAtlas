"""Host-149 formal authorization for the unified joint worker."""
from __future__ import annotations

import socket
from pathlib import Path

from .common import digest, fail, read_json, sha, verify_manifest
from .controller import assert_controller
from .lineage import validate_prepared, verify_signed_file


PLAN_FORMAT = 'CLA_FULL_PYG_R62_15_JOB_PLAN_V1'


def require_host_149():
    actual = socket.gethostname()
    if actual != '149':
        fail('WRONG_EXECUTION_HOST', 'Joint execution is fixed to host 149', target_host='149', actual_host=actual)
    return actual


def validate_formal_plan(plan_path, cfg):
    """Verify the signed diagnostic admission and the complete 3x5 job grid."""
    require_host_149()
    assert_controller(cfg)
    plan_path = Path(plan_path).resolve()
    if not plan_path.is_file():
        fail('FORMAL_PLAN_REQUIRED', 'A signed formal JOBS.json is required')
    verify_signed_file(plan_path, cfg, 'FORMAL_JOB_PLAN')
    plan = read_json(plan_path)
    if plan.get('format') != PLAN_FORMAT or plan.get('config_sha256') != digest(cfg):
        fail('QUEUE_CONFIG_DRIFT', 'Wrong formal-plan generation or configuration')
    admission = plan.get('diagnostic_run')
    if not admission:
        fail('DIAGNOSTIC_BINDING_MISSING', 'Formal plan lacks diagnostic provenance')
    from .queue import check_admission
    check_admission(cfg, admission)
    receipt = Path(admission) / 'TRAINING_RECEIPT.json'
    if sha(receipt) != plan.get('diagnostic_receipt_sha256'):
        fail('DIAGNOSTIC_RECEIPT_CHANGED', 'Diagnostic admission changed after plan creation')
    expected = {(variant, fold) for variant in ('G0', 'G1', 'G2') for fold in range(5)}
    jobs = plan.get('jobs', [])
    observed = {(job.get('variant'), job.get('fold')) for job in jobs}
    expected_ids = {f'{variant}_PF{fold:02d}' for variant, fold in expected}
    if len(jobs) != 15 or observed != expected or {job.get('job_id') for job in jobs} != expected_ids:
        fail('FORMAL_GRID_DRIFT', 'Signed plan must contain exactly the named G0/G1/G2 x five-fold grid')
    return plan_path, plan


def validate_formal_job(prepared, plan_path, job_id, cfg=None):
    """Bind one formal worker to its signed plan entry before model creation."""
    prepared = Path(prepared).resolve()
    if not plan_path or not job_id:
        fail('FORMAL_PLAN_REQUIRED', 'Formal worker requires --plan and --job-id')
    if cfg is None:
        cfg = read_json(prepared / 'CONFIG.json')
    plan_path, plan = validate_formal_plan(plan_path, cfg)
    matches = [job for job in plan['jobs'] if job.get('job_id') == job_id]
    if len(matches) != 1:
        fail('FORMAL_JOB_NOT_AUTHORIZED', 'Job ID is absent or repeated in the signed plan', job_id=job_id)
    job = matches[0]
    if Path(job.get('prepared', '')).resolve() != prepared:
        fail('FORMAL_JOB_NOT_AUTHORIZED', 'Prepared directory is not the signed job input', job_id=job_id)
    verify_manifest(prepared, expected=job.get('sha256'))
    manifest, spec, prepared_cfg, _ = validate_prepared(prepared)
    from .queue import variant_config
    expected_cfg = variant_config(cfg, job['variant'])
    if (manifest['metadata'].get('profile') != 'formal'
            or spec.get('fold') != job['fold']
            or spec.get('graph_variant') != job['variant']
            or job.get('config_sha256') != digest(expected_cfg)
            or digest(prepared_cfg) != digest(expected_cfg)):
        fail('QUEUE_JOB_BINDING_MISMATCH', 'Prepared formal job differs from its signed entry', job_id=job_id)
    return {
        'state': 'FORMAL_JOB_AUTHORIZED_ON_149',
        'plan_path': str(plan_path),
        'plan_sha256': sha(plan_path),
        'job_id': job_id,
        'prepared_sha256': job['sha256'],
        'diagnostic_receipt_sha256': plan['diagnostic_receipt_sha256'],
    }
