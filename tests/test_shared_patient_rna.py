import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage_shared_patient_rna import mirror_access, physical_file


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX publication permissions")
def test_shared_stage_mirrors_live_read_access_without_changing_sources(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(mode=0o755)
    old.chmod(0o755)
    new.mkdir(mode=0o700)
    for d in (old, new):
        (d / "COMPLETE.json").write_text("{}")
    (old / "COMPLETE.json").chmod(0o644)
    protected = old / "restricted"
    protected.write_text("x")
    protected.chmod(0o640)
    (new / "restricted").write_text("new")
    source = tmp_path / "pinned_source"
    source.write_text("measured")
    source.chmod(0o600)
    physical_file(source, new / "support.tsv")
    mirror_access(new, old)
    assert [stat.S_IMODE(p.stat().st_mode) for p in (new, new / "restricted", new / "support.tsv", source)] == [
        0o755, 0o640, 0o644, 0o600]
    assert protected.read_text() == "x"
