"""Install verified expression matrices at the existing patient RNA paths on 149."""
import argparse
import ctypes
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.common import atomic_json, sha


def receipt(path):
    return path / 'COMPLETE.json'


def digest(path):
    return sha(receipt(path)) if receipt(path).is_file() else None


def metadata(path):
    return json.loads(receipt(path).read_text())


def verify_files(path, items):
    for name, entry in items.items():
        file = path / name
        expected = entry['sha256'] if isinstance(entry, dict) else entry
        if not file.resolve().is_relative_to(path.resolve()) or sha(file) != expected:
            raise ValueError(f'File differs from its manifest: {file}')


def verify(raw, common):
    raw_meta, common_meta = metadata(raw), metadata(common)
    if raw_meta['status'] != 'COMPLETE' or common_meta['status'] != 'COMPLETE':
        raise ValueError('Incomplete source')
    if common_meta['cptac_source']['sha256'] != digest(raw):
        raise ValueError('Patient RNA and source matrices are different releases')
    verify_files(raw, raw_meta['files'])
    records = common_meta['records']
    if (sum(k.startswith('TCGA_') for k in records) != 33 or
            {k[6:] for k in records if k.startswith('CPTAC_')} != set(raw_meta['cohorts'])
            or len(raw_meta['cohorts']) != 10):
        raise ValueError('Production cohort coverage changed')
    for case, row in records.items():
        for part in ('aligned', 'cache'):
            path = common / part / case
            if digest(path) != row[part + '_sha256']:
                raise ValueError(f'Invalid release record: {path}')
            verify_files(path, metadata(path)['files'])
        if metadata(common / 'cache' / case)['input_complete_sha256'] != row['aligned_sha256']:
            raise ValueError(f'Stale scores: {case}')
    return raw_meta, common_meta


def exchange(first, second):
    """XFS swaps the whole directory; SSHFS uses the recoverable rename path below."""
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, 'renameat2'):
        return False
    if libc.renameat2(-100, os.fsencode(first), -100, os.fsencode(second), 2) == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP, errno.EXDEV):
        return False
    raise OSError(error, os.strerror(error), str(first))


def install(current, target, old, new):
    backup = current.with_name(current.name + '.before_gdc_' + old[:12])
    pending = current.with_name('.' + current.name + '.switch_' + new[:12])
    active = digest(current)
    if active == new and digest(backup) == old:
        return backup
    if active is None and digest(backup) == old and digest(pending) == new:
        pending.rename(current)
        return backup
    if active != old or backup.exists() or backup.is_symlink():
        raise ValueError(f'Active release or existing backup differs: {current}')
    if not pending.is_symlink():
        if pending.exists():
            raise ValueError(f'Unrecognized staging path: {pending}')
        pending.symlink_to(os.path.relpath(target, start=current.parent), target_is_directory=True)
    if digest(pending) != new:
        raise ValueError(f'Staging link does not resolve to the verified release: {pending}')
    if exchange(current, pending):
        try:
            pending.rename(backup)
        except BaseException:
            if digest(pending) == old and digest(current) == new:
                exchange(current, pending)
            raise
    else:
        current.rename(backup)
        try:
            pending.rename(current)
        except BaseException:
            backup.rename(current)
            raise
    return backup


def restore(current, backup, new):
    if digest(current) != new or not current.is_symlink():
        raise ValueError(f'Refusing to roll back an unrecognized active release: {current}')
    pending = current.with_name('.' + current.name + '.switch_' + new[:12])
    if pending.exists() or pending.is_symlink():
        raise ValueError(f'Unexpected pending path: {pending}')
    if exchange(current, backup):
        backup.unlink()
    else:
        current.rename(pending)
        try:
            backup.rename(current)
        except BaseException:
            pending.rename(current)
            raise
        pending.unlink()


def replace_file(src, dest, expected):
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix='.' + dest.name + '.', delete=False) as f:
        temp = Path(f.name)
    try:
        shutil.copyfile(src, temp)
        shutil.copymode(dest if dest.exists() else src, temp)
        if sha(temp) != expected:
            raise ValueError(f'Incorrect staged bytes: {dest}')
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def native_updates(plan_file, raw_hash):
    plan = json.loads(plan_file.read_text())
    if plan['status'] != 'VERIFIED_STAGED':
        raise ValueError('Native scores have not been checked')
    targets = plan['targets']
    expected = {'CURRENT_149_CONFIG.json','CPTAC_CURRENT.json','STATE_CURRENT.json','RPPA_CURRENT.json'}
    by_name = {Path(t['active']).name: t for t in targets}
    if len(targets) != len(expected) or set(by_name) != expected:
        raise ValueError('Missing or duplicated runtime or assay pointers')
    for row in targets:
        current = Path(row['active'])
        if sha(Path(row['staged']['path'])) != row['staged']['sha256'] or sha(current) not in (
                row['previous_sha256'], row['staged']['sha256']):
            raise ValueError(f'Runtime config drift: {current}')
    runtime_row = by_name['CURRENT_149_CONFIG.json']
    runtime = json.loads(Path(runtime_row['staged']['path']).read_text())
    pointer = json.loads(Path(by_name['CPTAC_CURRENT.json']['staged']['path']).read_text())
    binding = pointer['cptac_input']
    bundle_file = Path(binding['complete']['path'])
    if (binding != runtime['cptac_input'] or sha(bundle_file) != binding['complete']['sha256'] or
            json.loads(bundle_file.read_text())['config']['aligned_complete']['sha256'] != raw_hash or
            pointer['current_config'] != dict(path=runtime_row['active'], sha256=runtime_row['staged']['sha256'])):
        raise ValueError('Native binding does not refer to the approved input generation')
    for name in ('RPPA_CURRENT.json','STATE_CURRENT.json'):
        other = json.loads(Path(by_name[name]['staged']['path']).read_text())
        if other['cptac_input'] != binding or other['current_config'] != pointer['current_config']:
            raise ValueError(f'Inconsistent shared input binding in {name}')
    switch = plan['bundle_switch']
    if Path(switch['active']).name != 'bundle_r1' or digest(Path(switch['target'])) != switch['staged_sha256']:
        raise ValueError('Missing shared native score bundle')
    if switch['staged_sha256'] != binding['complete']['sha256']:
        raise ValueError('Native score binding does not match shared bundle')
    return targets, binding, switch


def activate(raw, common, raw_live, common_live, record, native_plan):
    raw_meta, common_meta = verify(raw, common)
    old_raw = raw_meta['rna_quantification']['baseline_complete_sha256']
    old_common = common_meta['quantification']['baseline_manifest']
    new_raw, new_common = digest(raw), digest(common)
    updates, binding, bundle = native_updates(native_plan, new_raw)
    entry = common_live.parent / 'materialize_common_rna_components.py'
    launcher = None
    if entry.exists():
        repo = Path(__file__).resolve().parents[1]
        original = repo / 'provenance/functional_y_20261109/source' / entry.name
        wrapper = repo / 'scripts' / entry.name
        consumer = common_live.parent / 'source_integration/repo/scripts/materialize_patient_star_rna.py'
        if not consumer.is_file() or sha(consumer) != sha(repo / 'scripts' / consumer.name):
            raise ValueError('Deploy the tested patient RNA builder to the production checkout first')
        launcher = dict(active=str(entry), previous_sha256=sha(original),
                        staged=dict(path=str(wrapper), sha256=sha(wrapper)))
        if sha(entry) not in (launcher['previous_sha256'], launcher['staged']['sha256']):
            raise ValueError(f'Unrecognized existing materializer: {entry}')
    # Keep the replacement on the same shared mount as the legacy raw input path.
    # A link into a compute host's private /dsk2 would be broken on other mounts.
    replica = raw_live.parent / ('aligned_gdc_star_' + new_raw[:12])
    if not replica.exists():
        with tempfile.TemporaryDirectory(dir=raw_live.parent, prefix='.gdc_copy.') as tmp:
            staged = Path(tmp) / 'aligned'
            shutil.copytree(raw, staged)
            verify_files(staged, raw_meta['files'])
            if digest(staged) != new_raw:
                raise ValueError('Failed to verify source replica')
            staged.rename(replica)
    else:
        if digest(replica) != new_raw:
            raise ValueError('Incorrect published source manifest')
        verify_files(replica, raw_meta['files'])
    steps = [(raw_live, replica, old_raw, new_raw),
             (Path(bundle['active']), Path(bundle['target']), bundle['previous_sha256'], bundle['staged_sha256']),
             (common_live, common, old_common, new_common)]
    installed = []
    configs = []
    for live, _, old, new in steps:
        backup = live.with_name(live.name + '.before_gdc_' + old[:12])
        pending = live.with_name('.' + live.name + '.switch_' + new[:12])
        # Recover a process that stopped after the atomic exchange, before naming the backup.
        if digest(live) == new and digest(pending) == old and not backup.exists():
            pending.rename(backup)
        if not (digest(live) == old or
                (digest(live) in (new, None) and digest(backup) == old)):
            raise ValueError(f'Unrecognized live or backup version: {live}')
    try:
        for update in updates + ([launcher] if launcher else []):
            current = Path(update['active'])
            old = update['previous_sha256']
            backup = current.with_name(current.name + '.before_gdc_' + old[:12])
            if not backup.exists():
                if sha(current) != old:
                    raise ValueError(f'Original runtime config is unavailable: {current}')
                replace_file(current, backup, old)
            if sha(backup) != old:
                raise ValueError(f'Invalid configuration backup: {backup}')
            staged = update['staged']
            if sha(current) != staged['sha256']:
                replace_file(Path(staged['path']), current, staged['sha256'])
            configs.append((current, backup, old))
        for live, target, old, new in steps:
            backup = install(live, target, old, new)
            installed.append((live, backup, new))
            if digest(live) != new or digest(backup) != old:
                raise ValueError(f'Post-install verification failed: {live}')
        from cla_engine.cptac_binding import validate_binding
        validate_binding(binding)
        atomic_json(record, dict(status='ACTIVE', workflow='STAR - Counts', gene_model='GENCODE v36',
                                 metric='tpm_unstranded', scope='patient RNA inputs and RNA-derived native and common scores',
                                 native_binding=binding, native_configuration=updates, legacy_entry=launcher,
                                 paths=[dict(path=str(live), sha256=new, previous=str(backup))
                                        for live, backup, new in installed]))
    except BaseException:
        for live, backup, new in reversed(installed):
            restore(live, backup, new)
        for current, backup, old in reversed(configs):
            replace_file(backup, current, old)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'common', 'raw-live', 'common-live', 'record', 'native-plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with args.record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        activate(args.raw, args.common, args.raw_live, args.common_live, args.record, args.native_plan)


if __name__ == '__main__':
    main()
