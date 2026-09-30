from pathlib import Path

import pytest

from scripts.rbp_encode_v33.collect_c_results_no_rehash_149 import (
    local_manifest,
    parse_manifest,
)


def test_result_manifest_compares_paths_and_sizes(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "model.pt").write_bytes(b"abc")
    (tmp_path / "nested" / "metrics.json").write_bytes(b"12345")
    expected = {"model.pt": 3, "nested/metrics.json": 5}
    assert parse_manifest("model.pt\t3\nnested/metrics.json\t5\n") == expected
    assert local_manifest(tmp_path) == expected


@pytest.mark.parametrize("name", ["../escape", "/absolute", "bad space"])
def test_result_manifest_rejects_unsafe_paths(name: str) -> None:
    with pytest.raises(RuntimeError, match="Unsafe result path"):
        parse_manifest(f"{name}\t1\n")
