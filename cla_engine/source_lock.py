"""Incremental, controller-local source freeze before allowing GPU capsules.

This proves that the measured bytes do not change between preparations. It does
not certify biological correctness or replace authoritative published hashes.
"""
from pathlib import Path
from .common import atomic_json,read_json,sha,digest,lock,fail

def record_sources(cfg, records):
    root=Path(cfg['work_root'])/'source_freeze_r6'
    # All folds/variants share the same source freeze, while targets/cfg remain
    # bound separately in the prepared lineage. Never overwrite a conflicting SHA.
    with lock(root):
        p=root/'SOURCES.json';old=read_json(p) if p.exists() else {'format':'CLA_R62_OBSERVED_SOURCE_FREEZE_V1','files':{}}
        for role,r in records.items():
            path=str(Path(r['path']).resolve());expected=r['sha256']
            if sha(path)!=expected:fail('INPUT_CHANGED_DURING_PREPARE','Source changed while creating a payload',path=path)
            if path in old['files'] and old['files'][path]['sha256']!=expected:
                fail('FROZEN_SOURCE_CHANGED','Use a separately approved work root for a new data version',path=path)
            old['files'][path]={'sha256':expected}
        atomic_json(p,old)
    return {'role':'OBSERVED_BYTES_FREEZE_NOT_BIOLOGICAL_CERTIFICATION','bound_source_count':len(records),'sources_digest':digest(records)}
