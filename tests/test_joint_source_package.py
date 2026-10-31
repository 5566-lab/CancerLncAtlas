"""A reviewed source capsule must bind executable code and current documentation."""
import json
import pytest
from cla_engine.common import Stop,sha
from cla_engine.package import MANIFEST,source_files,verify

def reviewed(tmp_path):
    (tmp_path/'cla_engine').mkdir();(tmp_path/'docs').mkdir()
    (tmp_path/'cla.py').write_text('pass\n')
    (tmp_path/'cla_engine/example.py').write_text('VALUE = 1\n')
    (tmp_path/'docs/current.rst').write_text('State shares the encoder.\n')
    record={'format':'CLA_JOINT_MAINLINE_SOURCE_V1','files':{
        name:{'sha256':sha(path),'bytes':path.stat().st_size}
        for name,path in source_files(tmp_path).items()}}
    manifest=tmp_path/MANIFEST;manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(record))
    assert verify(tmp_path)['production_release'] is False
    return tmp_path

@pytest.mark.parametrize('relative',['cla_engine/example.py','docs/current.rst'])
def test_reviewed_code_and_documentation_bytes_cannot_drift(tmp_path,relative):
    root=reviewed(tmp_path);(root/relative).write_text('changed\n')
    with pytest.raises(Stop) as error:verify(root)
    assert error.value.code=='SOURCE_BYTES_DRIFT'

def test_added_executable_is_not_part_of_reviewed_source(tmp_path):
    root=reviewed(tmp_path);(root/'cla_engine/extra.py').write_text('pass\n')
    with pytest.raises(Stop) as error:verify(root)
    assert error.value.code=='SOURCE_FILE_SET_DRIFT'
