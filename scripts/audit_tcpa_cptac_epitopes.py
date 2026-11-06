"""Bind curated RPPA phospho epitope evidence to exact human reference windows."""
import argparse,json,hashlib,socket,re
from pathlib import Path
import pandas as pd

def site_window(sequence,position,residue):
    if position<8 or position+7>len(sequence):raise ValueError("WINDOW_OUTSIDE_SEQUENCE")
    if sequence[position-1]!=residue:raise ValueError("RESIDUE_MISMATCH")
    window=sequence[position-8:position+7]
    if sequence.count(window)!=1:raise ValueError("AMBIGUOUS_REFERENCE_WINDOW")
    return window

def parse_fasta(text):
    result={}
    for block in text.split(">"):
        if not block.strip():continue
        lines=block.splitlines();key=lines[0].split()[0]
        if key in result:raise ValueError("DUPLICATE_REFERENCE")
        result[key]="".join(lines[1:]).strip()
    return result

def run(config,source_root,rppa,bundle,output):
    assert socket.gethostname()=="149"
    config,root,rppa,bundle,out=map(Path,[config,source_root,rppa,bundle,output])
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Use new output directory")
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"CURATED_PHOSPHOSITE_SEQUENCE_AND_COVERAGE"}))
    sources={}
    def bind(p,expected=None):
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if expected and h!=expected:raise ValueError("Source checksum drift")
        sources[str(p)]=h;return p
    cfg=json.loads(bind(config).read_text())
    for rel in ["specimen_epitope_sources_r1","sequence_sources_r1"]:
        rec=json.loads(bind(root/rel/"FETCH_RECEIPT.json").read_text())
        for r in rec["records"]:
            if r["status"]=="DOWNLOADED":bind(root/rel/r["name"],r["sha256"])
    seq=parse_fasta((root/"sequence_sources_r1/refseq.fasta").read_text())
    rm=json.loads(bind(rppa/"COMPLETE.json").read_text())
    cat=pd.read_csv(bind(rppa/"FEATURE_CATALOG.tsv",rm["files"]["FEATURE_CATALOG.tsv"]["sha256"]),sep="\t").fillna("").set_index("feature_id")
    dm=json.loads(bind(root/"DOWNLOAD_COMPLETE.json").read_text())
    rec=next(r for r in dm["files"] if r["cancer"]=="BRCA" and r["layer"]=="phospho")
    measured=set(pd.read_csv(bind(Path(rec["path"]),rec["sha256"]),sep="\t",usecols=[0]).iloc[:,0])
    rows=[]
    for record in cfg["sites"]:
        d=dict(record)
        for feature,key in [(d["rppa_phospho"],"phospho_catalog"),(d["rppa_total"],"total_catalog")]:
            assert str(cat.loc[feature,"catalog_number"]).strip()==d[key]
            assert cat.loc[feature,"gene_ids"]==d["gene_id"] and cat.loc[feature,"validation_status"]=="Valid"
        u=json.loads((root/"sequence_sources_r1"/(d["uniprot"]+".json")).read_text())
        assert u["organism"]["taxonId"]==9606
        assert any(g.get("geneName",{}).get("value")==d["gene_symbol"] for g in u["genes"])
        wu=site_window(u["sequence"]["value"],d["human_position"],d["residue"])
        wr=site_window(seq[d["refseq"]],d["human_position"],d["residue"])
        if wu!=wr:raise ValueError("REFSEQ_UNIPROT_WINDOW_MISMATCH")
        site=d["residue"].lower()+str(d["human_position"])
        exact=d["gene_symbol"]+"_"+site+"__"+d["refseq"]+":"+site
        assert exact in measured,exact
        d.update(reference_window=wu,ptm_sequence_id=wu+"-p",retrospective_feature_id=exact,
            sequence_check="EXACT_UNIQUE_15AA_REFSEQ_UNIPROT_WINDOW",status="CURATED_PAIRED_SITE_DIAGNOSTIC",
            full_antibody_specificity_experimentally_revalidated=False)
        rows.append(d)
    pd.DataFrame(rows).to_csv(out/"CURATED_PAIRED_SITES.tsv",sep="\t",index=False)
    pd.DataFrame(cfg["exclusions"]).to_csv(out/"KNOWN_SINGLE_SITE_EXCLUSIONS.tsv",sep="\t",index=False)
    bm=json.loads(bind(bundle/"COMPLETE.json").read_text())
    bridge=pd.read_parquet(bind(bundle/"PTM_SIGNATURE_SITE_GENE_BRIDGE.parquet",bm["files"]["PTM_SIGNATURE_SITE_GENE_BRIDGE.parquet"]["sha256"]))
    # Score-definition membership, not observed data-dependent target selection.
    spec=json.loads((bundle/"COMPLETE.json").read_text())["config"]
    ptm_spec=spec.get("ptm_gmt")
    if ptm_spec is None:
        source_spec=json.loads(Path(spec["source_spec"]["path"]).read_text());ptm_spec=source_spec["references"]["ptm_gmt"]
    gmt=bind(Path(ptm_spec["path"]),ptm_spec["sha256"])
    available={r["ptm_sequence_id"] for r in rows};coverage=[]
    for line in gmt.read_text().splitlines():
        parts=line.split("\t");signature=parts[0];tokens=parts[2:]
        # PTMsigDB uses sequence;direction tokens. Preserve direction, never gene-level collapse.
        hit=[t for t in tokens if t.split(";")[0] in available]
        coverage.append(dict(signature_id=signature,total_signature_tokens=len(tokens),common_site_count=len(set(t.split(";")[0] for t in hit)),common_site_tokens="|".join(hit),
            existing_ptmsea_minimum=5,passes_existing_minimum=len(set(t.split(";")[0] for t in hit))>=5))
    pd.DataFrame(coverage).to_csv(out/"COMMON_PTM_SIGNATURE_COVERAGE.tsv",sep="\t",index=False)
    result=dict(target_host="149",status="COMPLETE_CURATED_SITE_SEQUENCE_AUDIT",sites=len(rows),
        max_common_sites_per_signature=max(r["common_site_count"] for r in coverage),
        signatures_passing_existing_minimum=sum(r["passes_existing_minimum"] for r in coverage),
        source_sha256=sources,limitations=cfg["limitations"])
    result["files"]={p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ["source_sha256","files"]}))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["config","source-root","rppa","bundle","output"]:p.add_argument("--"+key,required=True)
    a=p.parse_args();run(a.config,a.source_root,a.rppa,a.bundle,a.output)
