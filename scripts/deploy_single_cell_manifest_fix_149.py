"""Activate validated head inputs and deploy source fixes, retaining backups."""
import hashlib
import json
import shutil
import socket
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if socket.gethostname() != "149":
        raise RuntimeError("Execution requires target_host=149")
    root = Path('/dsk2/user/dengsc/sc_manifest_fix_20261004_r1')
    inputs = Path('/dsk2/user/dengsc/v32_sc_head_inputs_20261004_r4')
    project = Path('/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1')
    receipt = json.loads((inputs / 'MANIFEST_RECONCILIATION.json').read_text())
    if receipt.get('status') != 'PASS' or receipt.get('target_host') != '149':
        raise RuntimeError('Real-data validation has not passed')
    if sha(inputs / 'dataset_manifest.training.parquet') != receipt['manifest_sha256']:
        raise RuntimeError('Validated manifest changed')
    tested = json.loads((root / 'receipts/CODE_AND_TEST_RECEIPT.json').read_text())
    changes = []
    for relative, expected in tested['sha256'].items():
        source = root / 'code' / relative
        destination = project / 'code' / relative
        baseline = project / 'runtime/c_graph_global_binding_20260928_r1/code' / relative
        if sha(source) != expected:
            raise RuntimeError(f'Tested source changed: {relative}')
        if destination.exists() and (not baseline.exists() or sha(destination) != sha(baseline)):
            raise RuntimeError(f'Canonical source changed during repair: {relative}')
        changes.append((source, destination, relative))
    for name in ('assemble_sc_inputs.py', 'launch_single_cell_head_149.sh'):
        changes.append((root / name, Path('/tmp') / name, 'entrypoints/' + name))
    backups = root / 'activation_backups'
    backups.mkdir(exist_ok=False)
    applied = []
    for source, destination, relative in changes:
        before = sha(destination) if destination.exists() else None
        if destination.exists():
            backup = backups / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != sha(source):
            raise RuntimeError(f'Deployment copy mismatch: {destination}')
        applied.append({'path': str(destination), 'before_sha256': before, 'after_sha256': sha(destination)})
    active = {'target_host': '149', 'status': 'PASS_ACTIVATED',
              'code_root': str(root / 'code'), 'inputs': str(inputs),
              'launcher': '/tmp/launch_single_cell_head_149.sh',
              'qualified_cancers': receipt['qualified_cancers'],
              'training_cancers': receipt['training_cancers'],
              'training_started': False,
              'changes': applied, 'backup_root': str(backups)}
    for destination in (root / 'receipts/ACTIVATION_RECEIPT.json',
                        project / 'manifests/CURRENT_SINGLE_CELL_HEAD.json'):
        if destination.exists():
            raise RuntimeError(f'Activation receipt already exists: {destination}')
        destination.write_text(json.dumps(active, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': active['status'], 'files': len(applied),
                      'qualified': len(active['qualified_cancers']), 'training_cancers': len(active['training_cancers'])}))


if __name__ == '__main__':
    main()
