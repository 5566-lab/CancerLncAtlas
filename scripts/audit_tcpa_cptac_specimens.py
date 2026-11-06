"""Version-bound public PDC specimen concordance for retrospective bridge candidates."""
from __future__ import annotations
import argparse,json,re,socket,hashlib
from pathlib import Path
import pandas as pd

STUDIES={"BRCA_BI_protein":"PDC000173","BRCA_BI_phospho":"PDC000174","COADREAD_VU_protein":"PDC000111",
         "OV_JHU_protein":"PDC000113","OV_PNNL_protein":"PDC000114","OV_PNNL_phospho":"PDC000115"}

def specimen_match(rows, expected_vial):
    eligible=[r for r in rows if r.get("pool")=="No" and r.get("taxon")=="Homo sapiens"
              and r.get("sample_type")=="Primary Tumor"
              and all(r.get(k)=="Qualified" for k in ["sample_status","case_status","aliquot_status"])]
    vials=sorted(set(r["sample_submitter_id"] for r in eligible))
    if not eligible:status="NO_QUALIFIED_PRIMARY_TUMOR_METADATA"
    elif any(not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]",s) for s in vials):status="UNRESOLVED_SAMPLE_IDENTIFIER"
    elif len(vials)!=1:status="MULTIPLE_SAMPLE_VIALS_UNRESOLVED"
    elif vials[0]!=expected_vial:status="SAMPLE_VIAL_MISMATCH"
    else:status="SAME_VIAL_IN_PDC_COHORT_METADATA"
    return status,vials,sorted(set(r["aliquot_submitter_id"] for r in eligible))

def run(mapping,rppa,sources,out):
    assert socket.gethostname()=="149"
    mapping,rppa,sources,out=map(Path,[mapping,rppa,sources,out])
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Use new output directory")
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149","action":"PDC_VERSION_BOUND_SPECIMEN_CONCORDANCE"}))
    bound={}
    def record(p,h=None):
        s=hashlib.sha256(p.read_bytes()).hexdigest()
        if h and s!=h:raise ValueError("Changed source "+str(p))
        bound[str(p)]=s;return p
    manifests={root:json.loads(record(root/"COMPLETE.json").read_text()) for root in [mapping,rppa]}
    def product(root,rel):return record(root/rel,manifests[root]["files"][rel]["sha256"])
    pairs=pd.read_csv(product(mapping,"RETROSPECTIVE_PATIENT_PAIR_CANDIDATES.tsv"),sep="\t")
    alignment=pd.read_csv(product(rppa,"PATIENT_ALIGNMENT.tsv"),sep="\t")
    selected=alignment.loc[alignment.status.eq("MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL")].set_index("patient_id")
    assert selected.index.is_unique
    receipt=json.loads(record(sources/"FETCH_RECEIPT.json").read_text())
    datasets={}
    for dataset,pid in STUDIES.items():
        recs=[r for r in receipt["records"] if r["name"].startswith(pid+"_v") and r["name"].endswith("_biospecimen.json")]
        assert len(recs)==1 and recs[0]["status"]=="DOWNLOADED"
        rec=recs[0];d=json.loads(record(sources/rec["name"],rec["sha256"]).read_text())
        datasets[dataset]=(d["data"]["biospecimenPerStudy"],rec)
    rows=[]
    for r in pairs.loc[pairs.in_current_rppa_rna_scope].itertuples():
        data,meta=datasets[r.dataset]
        assert "study_id%3A" in meta["url"]
        records=[x for x in data if x["case_submitter_id"]==r.patient_id]
        expected=selected.loc[r.patient_id,"rppa_sample_id"]
        status,vials,aliquots=specimen_match(records,expected)
        rows.append(dict(dataset=r.dataset,patient_id=r.patient_id,rppa_sample_vial=expected,
            pdc_study_id=STUDIES[r.dataset],pdc_study_uuid=meta["study"]["study_id"],pdc_study_version=meta["study"]["study_version"],
            pdc_sample_vials="|".join(vials),pdc_aliquots="|".join(aliquots),status=status,
            pdc_vial_concordant=status=="SAME_VIAL_IN_PDC_COHORT_METADATA",
            processed_matrix_column_to_aliquot_lineage_verified=False,same_aliquot_verified=False))
    result=pd.DataFrame(rows);result.to_csv(out/"PATIENT_SPECIMEN_CONCORDANCE.tsv",sep="\t",index=False)
    summary=result.groupby(["dataset","status"]).size().rename("patients").reset_index()
    summary.to_csv(out/"SPECIMEN_CONCORDANCE_SUMMARY.tsv",sep="\t",index=False)
    complete=dict(target_host="149",status="COMPLETE_PDC_COHORT_METADATA_VIAL_CONCORDANCE",
        summary=summary.to_dict("records"),source_sha256=bound,
        limitations=["The matrix columns carry patient IDs only. Concordance uses the matched PDC study metadata.",
          "This is not direct processed-column-to-aliquot lineage; it does not prove identical lysate/portion.",
          "Excluded/nonprimary/pool entries and ambiguous multiple vials are not accepted as matches."])
    complete["files"]={p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.iterdir() if p.is_file()}
    (out/"COMPLETE.json").write_text(json.dumps(complete,indent=2));print(json.dumps(complete["summary"]))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["mapping","rppa","sources","out"]:p.add_argument("--"+key,required=True)
    a=p.parse_args();run(a.mapping,a.rppa,a.sources,a.out)
