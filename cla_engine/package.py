"""Verify the reviewed unified source tree, not a historical package snapshot."""
from pathlib import Path
from .common import read_json,sha,fail

MANIFEST='provenance/joint_mainline_20261101/SOURCE_MANIFEST.json'
SOURCE_DIRS=('cla_engine','cc_hhgt','scripts','config','tests','docs','requirements')
EXTENSIONS={'.py','.r','.sh','.yaml','.yml','.json','.rst','.md','.txt','.toml'}

def source_files(root):
    root=Path(root)
    result={p.relative_to(root).as_posix():p for folder in SOURCE_DIRS for p in (root/folder).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() in EXTENSIONS}
    for name in ('cla.py','README.md','run_model_workflow.sh','pyproject.toml','CODE_COMPLETION_PLAN.json','CODE_COMPLETION_NOTES.rst'):
        if (root/name).is_file():result[name]=root/name
    for p in (root/'provenance/joint_mainline_20261101').glob('*.json'):
        if p.name != 'SOURCE_MANIFEST.json' and p.is_file():
            result[p.relative_to(root).as_posix()]=p
    for name in ('LICENSE','NOTICE','CITATION.cff'):
        if (root/name).is_file():result[name]=root/name
    return result

def verify(root):
    root=Path(root).resolve();record=read_json(root/MANIFEST)
    if record.get('format')!='CLA_JOINT_MAINLINE_SOURCE_V1':fail('SOURCE_GENERATION_INVALID','Expected unified joint source manifest')
    actual=source_files(root);expected=record['files']
    if set(actual)!=set(expected):fail('SOURCE_FILE_SET_DRIFT','Missing or unlisted source files',missing=sorted(set(expected)-set(actual)),extra=sorted(set(actual)-set(expected)))
    for rel,p in actual.items():
        if p.is_symlink() or not p.resolve().is_relative_to(root) or sha(p)!=expected[rel]['sha256'] or p.stat().st_size!=expected[rel]['bytes']:
            fail('SOURCE_BYTES_DRIFT','Source no longer matches reviewed integration',path=rel)
    result={'state':'PACKAGE_HASH_CHECK_PASS_NOT_TRAINED','files':len(actual),'task_weights':{'pathway':1.,'state':.5},'production_release':False}
    print(result);return result
