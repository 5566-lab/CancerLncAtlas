"""Behavior tests for official NETWORK snapshots without external requests."""
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/fetch_kegg_network_scope.py"
spec = importlib.util.spec_from_file_location("network_snapshot",SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def flat(identifier, fields):
    return "ENTRY       "+identifier+" Network\n"+"\n".join(f"{key:<12}{value}" for key,value in fields)+"\n///\n"

def test_preserves_direction_complex_and_expansion_without_guessing_genes():
    text = flat("N00056",[
        ("NAME","WNT reference"),("DEFINITION","A -> (B+C) -| D => E"),
        ("  EXPANDED","1 -> (2+3) -| 4 => 5"),("TYPE","Reference"),
        ("PATHWAY","hsa04310 WNT"),("CLASS","nt06215 WNT"),
        ("GENE","1 A; description"),("","5 E; description")])
    row = m.parse_network("N00056",m.parse_flat_records(text)["N00056"])
    assert row["definition"]=="A -> (B+C) -| D => E"
    assert row["expanded_definition"]=="1 -> (2+3) -| 4 => 5"
    assert row["gene_ids"]=="1|5"  # Intermediate numeric tokens are not inferred genes.
    assert row["class_map_ids"]=="nt06215"
    assert row["pathway_ids"]=="hsa04310"

@pytest.mark.parametrize("category,expected",[
    ("Disease view; Cancer",1),("Disease view; Cancer signaling",1),
    ("Disease view; Cancer-like",0),("Pathway view; Signal transduction",0)])
def test_only_exact_official_category_controls_admission(category,expected):
    fields = {"NAME":["Cancer-sounding name"],"CATEGORY":[category],"MEMBER":["N00056 WNT"]}
    assert m.parse_map("nt06215",fields)["is_cancer_map"]==expected

def test_missing_map_category_rejected():
    with pytest.raises(ValueError,match="Missing map"):
        m.parse_map("nt06215",{"NAME":["WNT"],"MEMBER":["N00056 WNT"]})

def test_duplicate_entry_is_error():
    block = flat("N00056",[("NAME","WNT")])
    with pytest.raises(ValueError,match="Duplicate"):
        m.parse_flat_records(block+block)

def test_brite_classification_not_leaf_name():
    tree = {"name":"br08907","children":[{"name":"Disease view","children":[
        {"name":"Cancer","children":[{"name":"06260  Colorectal cancer"}]},
        {"name":"Cancer signaling","children":[{"name":"06215  WNT signaling"}]},
        {"name":"Bacterial infection","children":[{"name":"06181  Cancer misleading name"}]}]}]}
    rows=m.parse_brite_maps(tree)
    assert rows["nt06215"]["category"]=="Cancer signaling"
    assert rows["nt06181"]["category"]=="Bacterial infection"
    assert rows["nt06260"]["category_path"]=="Disease view; Cancer"

def test_cached_batch_can_supply_changed_candidate_batch(tmp_path):
    f=m.Fetcher(tmp_path)
    payload=flat("N00056",[("NAME","WNT")])+flat("N00057",[("NAME","APC")])
    p=tmp_path/"raw/entries/N00056_N00057.txt"
    m.atomic_bytes(p,payload.encode())
    receipt={"path":str(p.relative_to(tmp_path)),"url":"https://rest.kegg.jp/get/N00056+N00057","sha256":m.sha(p)}
    m.atomic_json(p.with_name(p.name+".receipt.json"),receipt)
    records=f.records(["N00057"])
    assert set(records)=={"N00057"}
    assert receipt["path"] in f.sources
    p.write_text(payload+"\n")
    with pytest.raises(ValueError,match="checksum"):
        f.records(["N00057"])

def test_candidate_mapping_validation(tmp_path):
    p=tmp_path/"mapping.tsv"
    p.write_text("source_name\tnetwork_id\nA\tN00056\nB\tN00056\n")
    ids,receipts=m.candidate_networks(mapping_tsv=p)
    assert ids==["N00056"]
    assert receipts[0]["sha256"]==m.sha(p)
    p.write_text("source_name\tnetwork_id\nA\tnt06215\n")
    with pytest.raises(ValueError,match="Invalid mapped"):
        m.candidate_networks(mapping_tsv=p)

def test_host_guard_before_any_preparation(tmp_path,monkeypatch):
    monkeypatch.setattr(m.socket,"gethostname",lambda:"localhost")
    with pytest.raises(RuntimeError,match="must run on 149"):
        m.freeze(tmp_path)
    assert not list(tmp_path.iterdir())

def test_manifest_detects_missing_and_modified_sources(tmp_path):
    p=tmp_path/"source.txt"
    p.write_text("official text")
    manifest={"sources":[{"path":"source.txt","sha256":m.sha(p)}],"outputs":[]}
    m.verify_snapshot(tmp_path,manifest)
    p.write_text("changed")
    with pytest.raises(ValueError,match="checksum failure"):
        m.verify_snapshot(tmp_path,manifest)

def test_single_http404_requires_official_entry_confirmation(tmp_path,monkeypatch):
    import io
    import urllib.error
    f=m.Fetcher(tmp_path)
    def fake_fetch(relative,url):
        if relative.endswith(".txt"):
            raise urllib.error.HTTPError(url,404,"Not Found",{},io.BytesIO(b""))
        payload="<html>No such data was found.</html>"
        path=tmp_path/relative
        m.atomic_bytes(path,payload.encode())
        f.sources[relative]={"path":relative,"url":url,"sha256":m.sha(path)}
        return payload
    monkeypatch.setattr(f,"fetch",fake_fetch)
    assert f.confirm_missing("N00877") is None
    row=f.unresolved["N00877"]
    assert row["reason"]=="official_missing_confirmed"
    assert (tmp_path/row["http_receipt_path"]).is_file()
    assert (tmp_path/row["entry_response_path"]).is_file()

def test_server_failure_is_never_recorded_as_missing(tmp_path,monkeypatch):
    import io
    import urllib.error
    f=m.Fetcher(tmp_path)
    def fail(relative,url):
        raise urllib.error.HTTPError(url,503,"Unavailable",{},io.BytesIO(b"temporary"))
    monkeypatch.setattr(f,"fetch",fail)
    with pytest.raises(urllib.error.HTTPError):
        f.confirm_missing("N00877")
    assert not f.unresolved

def test_rest404_with_nonmissing_entry_is_error(tmp_path,monkeypatch):
    import io
    import urllib.error
    f=m.Fetcher(tmp_path)
    def conflict(relative,url):
        if relative.endswith(".txt"):
            raise urllib.error.HTTPError(url,404,"Not Found",{},io.BytesIO(b""))
        return "<html>Entry exists</html>"
    monkeypatch.setattr(f,"fetch",conflict)
    with pytest.raises(ValueError,match="not confirmed absent"):
        f.confirm_missing("N00877")
    assert not f.unresolved
