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
        pending.rename(backup)
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


def activate(raw, common, raw_live, common_live, record):
    raw_meta, common_meta = verify(raw, common)
    old_raw = raw_meta['rna_quantification']['baseline_complete_sha256']
    old_common = common_meta['quantification']['baseline_manifest']
    new_raw, new_common = digest(raw), digest(common)
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
    steps = [(raw_live, replica, old_raw, new_raw), (common_live, common, old_common, new_common)]
    installed = []
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
        for live, target, old, new in steps:
            backup = install(live, target, old, new)
            installed.append((live, backup, new))
            if digest(live) != new or digest(backup) != old:
                raise ValueError(f'Post-install verification failed: {live}')
        atomic_json(record, dict(status='ACTIVE', workflow='STAR - Counts', gene_model='GENCODE v36',
                                 metric='tpm_unstranded', scope='patient RNA inputs and common expression score caches',
                                 paths=[dict(path=str(live), sha256=new, previous=str(backup))
                                        for live, backup, new in installed]))
    except BaseException:
        for live, backup, new in reversed(installed):
            restore(live, backup, new)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'common', 'raw-live', 'common-live', 'record'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with args.record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        activate(args.raw, args.common, args.raw_live, args.common_live, args.record)


if __name__ == '__main__':
    main()
