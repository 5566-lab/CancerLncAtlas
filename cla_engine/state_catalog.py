"""Reviewed State definitions, source scales and biological interpretation."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import json
from .common import fail

CATALOG_ID='CLA_STATE_EXPANDED_20261101_V1'
ROUTED_CATALOG_ID='CLA_STATE_ROUTED_PROGRAMS_20261101_V2'
AUGMENTED_CATALOG_ID='CLA_STATE_AUGMENTED_20261104_V3'

@lru_cache(maxsize=3)
def _catalog(version=CATALOG_ID):
    filename={CATALOG_ID:'state_catalog_20261101.json',ROUTED_CATALOG_ID:'state_catalog_routed_20261101.json',AUGMENTED_CATALOG_ID:'state_catalog_augmented_20261104.json'}.get(version)
    if filename is None:fail('STATE_CATALOG_GENERATION','Unknown State registry')
    path=Path(__file__).resolve().parents[1]/'config'/filename
    value=json.loads(path.read_text())
    if value.get('catalog_id')!=version:fail('STATE_CATALOG_GENERATION','Wrong State registry')
    return value

def catalog(version=CATALOG_ID):
    return deepcopy(_catalog(version))

def expanded_targets(version=CATALOG_ID):
    return catalog(version)['targets']

def legacy_fixture_targets():
    return catalog()['legacy_fixture_targets']

def validate_targets(cfg):
    version=cfg.get('state_catalog')
    if version is None:
        # Existing bound configurations keep their recorded 15-target semantics.
        expected=legacy_fixture_targets()
        observed=cfg.get('targets',[])
        observed_roles={t.get('id'):t.get('role') for t in observed if isinstance(t,dict)}
        good=len(observed)==len(expected) and observed_roles=={t['id']:t['role'] for t in expected}
    elif version in (CATALOG_ID,ROUTED_CATALOG_ID,AUGMENTED_CATALOG_ID):
        expected=expanded_targets(version);good=cfg.get('targets')==expected
        if not cfg.get('synthetic_fixture'):
            source=cfg.get('sources',{}).get('state',{})
            if source.get('format')!='CLA_PATIENT_STATE_V1' or source.get('catalog_id')!=version:
                fail('STATE_MATERIALIZATION_REQUIRED','Expanded targets require a verified patient State matrix')
    else:fail('STATE_CATALOG_GENERATION','Unknown State catalog',catalog=version)
    if not good:fail('TARGET_ROLE_CATALOG_DRIFT','Target identity, method, source, family or role differs from the reviewed catalog')
    return expected
