import pytest
from scripts.audit_tcpa_cptac_epitopes import site_window,parse_fasta

def test_site_window_uses_residue_and_exact_coordinate_not_name_similarity():
    seq="ABCDEFGSJKLMNOP"
    assert site_window(seq,8,"S")==seq
    with pytest.raises(ValueError,match="RESIDUE_MISMATCH"):site_window(seq,8,"T")
    with pytest.raises(ValueError,match="WINDOW_OUTSIDE_SEQUENCE"):site_window(seq,7,"G")
    with pytest.raises(ValueError,match="AMBIGUOUS_REFERENCE_WINDOW"):site_window(seq+seq,8,"S")

def test_reference_version_is_retained_and_duplicates_rejected():
    assert parse_fasta(">NP_000001.1 reference\nABC\nDEF\n")=={"NP_000001.1":"ABCDEF"}
    with pytest.raises(ValueError,match="DUPLICATE_REFERENCE"):parse_fasta(">NP_000001.1\nAA\n>NP_000001.1\nBB")
