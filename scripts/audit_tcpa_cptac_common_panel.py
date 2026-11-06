"""Audit RPPA500/CPTAC common measurements and retrospective patient bridges on 149.

Candidate gene/site matches are not sequence-validated epitope equivalence.
This command produces mapping/coverage evidence, never calibrated labels.
"""
from __future__ import annotations
import argparse, hashlib, json, re, socket
from pathlib import Path
import numpy as np
import pandas as pd
from cla_engine.rppa500 import phosphosite_labels

def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def tcga_patient(value):
    """Only undo the documented dotted patient barcode representation."""
    value=str(value).replace(".","-")
    if not re.fullmatch(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",value):
        raise ValueError("Not an exact patient barcode: "+value)
    return value

def run(rppa, cptac, bundle, retrospective, output):
    assert socket.gethostname().split(".")[0]=="149", "Execute on 149"
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Use a new output directory")
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149",
        "action":"COMMON_PANEL_AND_RETROSPECTIVE_PATIENT_AUDIT",
        "scope":"No scoring, no calibration claim, no change to bound training inputs"},indent=2))
    sources={}
    def record(p,expected=None):
        p=Path(p);h=sha(p)
        if expected and h!=expected:raise ValueError("Source hash mismatch: "+str(p))
        sources[str(p)]={"sha256":h,"bytes":p.stat().st_size}
        return p
    def product(root,rel):
        root=Path(root);mp=record(root/"COMPLETE.json")
        manifest=json.loads(mp.read_text())
        if manifest["status"]!="COMPLETE":raise ValueError("Incomplete source")
        return record(root/rel,manifest["files"][rel]["sha256"])
    cat=pd.read_csv(product(rppa,"FEATURE_CATALOG.tsv"),sep="\t").fillna("")
    cov=pd.read_csv(product(rppa,"FEATURE_COVERAGE.tsv"),sep="\t")
    cat=cat.merge(cov[["feature_id","observed_patients"]],validate="one_to_one")
    cat["corrected_site_labels"]=cat.apply(lambda r:"|".join(phosphosite_labels(r.source_column))
        if r.molecular_layer=="PHOSPHORYLATION" else "",axis=1)
    cat["parser_changed"]=cat.reported_phosphosite_labels.ne(cat.corrected_site_labels)
    cat.to_csv(out/"RPPA_FEATURE_MAPPING_AUDIT.tsv",sep="\t",index=False)
    protein=cat.loc[cat.molecular_layer.eq("PROTEIN") & cat.gene_count.eq(1)].copy()
    duplicates=set(protein.loc[protein.gene_ids.duplicated(False),"gene_ids"])
    protein["unique_gene_antibody"]=~protein.gene_ids.isin(duplicates)
    protein["primary_candidate"]=protein.unique_gene_antibody & protein.validation_status.eq("Valid") & protein.observed_patients.gt(0)
    protein.to_csv(out/"PROTEIN_ANTIBODY_CANDIDATES.tsv",sep="\t",index=False)
    ph=cat.loc[cat.molecular_layer.eq("PHOSPHORYLATION")].copy()
    pairrows=[]
    for r in ph.itertuples():
        totals=protein.loc[protein.gene_ids.eq(r.gene_ids)] if r.gene_count==1 else protein.iloc[:0]
        single_site=len(r.corrected_site_labels.split("|"))==1 and bool(r.corrected_site_labels)
        pairrows.append(dict(phospho_feature_id=r.feature_id,gene_ids=r.gene_ids,
            reported_sites_corrected=r.corrected_site_labels,single_gene=r.gene_count==1,single_reported_site=single_site,
            total_protein_candidates="|".join(totals.feature_id),
            unique_total_antibody=len(totals)==1,phospho_qc=r.validation_status,
            total_protein_qc="|".join(totals.validation_status),
            residual_calibration_fitted=False,epitope_sequence_validated=False))
    pd.DataFrame(pairrows).to_csv(out/"PHOSPHO_TOTAL_PROTEIN_CANDIDATES.tsv",sep="\t",index=False)
    bridge=pd.read_parquet(product(bundle,"PATHWAY_GENE_PROTEIN_BRIDGE.parquet"))
    sets=bridge.groupby("pathway_id").gene_id.apply(lambda x:set(x.dropna())).to_dict()
    features=pd.read_parquet(product(cptac,"FEATURE_ALIGNMENT.parquet"))
    protein_refs=features.loc[features.layer.eq("protein"),["gene_id","reference_symbol"]].drop_duplicates().dropna()
    symmap={g:sorted(set(d.reference_symbol)) for g,d in protein_refs.groupby("gene_id")}
    summary=[];links=[];pathcoverage=[];sitecandidates=[]
    cohorts=json.loads((Path(cptac)/"COMPLETE.json").read_text())["cohorts"]
    for cohort in cohorts:
        m=pd.read_parquet(product(cptac,"matrices/"+cohort+"/protein_Tumor.parquet")).set_index("feature_id")
        assert m.index.is_unique
        observed=set(m.index[np.isfinite(m.to_numpy(float)).any(axis=1)])
        present=protein.loc[protein.gene_ids.isin(observed)].copy()
        present["source_cohort"]=cohort;links.append(present)
        genes=set(present.loc[present.primary_candidate,"gene_ids"])
        for pathway,g in sets.items():
            pathcoverage.append(dict(source_cohort=cohort,pathway_id=pathway,full_gene_count=len(g),
                single_gene_valid_common_proteins=len(g & genes),common_gene_ids="|".join(sorted(g & genes))))
        pm=pd.read_parquet(product(cptac,"matrices/"+cohort+"/phosphorylation_Tumor.parquet")).set_index("feature_id")
        have=set(pm.index[np.isfinite(pm.to_numpy(float)).any(axis=1)])
        fmap=features.loc[features.layer.eq("phosphorylation") & features.feature_id.isin(have)]
        for r in ph.loc[ph.gene_count.eq(1) & ~ph.corrected_site_labels.str.contains("|",regex=False)].itertuples():
            hit=fmap.loc[fmap.gene_id.eq(r.gene_ids) & fmap.site.str.upper().eq(r.corrected_site_labels)].drop_duplicates("feature_id")
            for h in hit.itertuples():
                sitecandidates.append(dict(source_cohort=cohort,rppa_feature_id=r.feature_id,cptac_feature_id=h.feature_id,
                    gene_id=h.gene_id,site=h.site,protein_isoform=h.protein_isoform,flanking_sequence=h.flanking_sequence,
                    status="GENE_SITE_LABEL_CANDIDATE_NOT_SEQUENCE_VALIDATED"))
        summary.append(dict(source_cohort=cohort,common_single_gene_antibodies=len(present),
            primary_unique_valid_common_proteins=len(genes),
            protein_pathways_any=sum(bool(g & genes) for g in sets.values()),
            protein_pathways_ge3=sum(len(g & genes)>=3 for g in sets.values()),
            protein_pathways_ge5=sum(len(g & genes)>=5 for g in sets.values()),
            protein_pathways_ge10=sum(len(g & genes)>=10 for g in sets.values())))
        print("CURRENT_COHORT_AUDITED",cohort,flush=True)
    pd.concat(links).to_csv(out/"CURRENT_CPTAC_COMMON_PROTEIN_CANDIDATES.tsv",sep="\t",index=False)
    pd.DataFrame(summary).to_csv(out/"CURRENT_CPTAC_COMMON_PANEL_SUMMARY.tsv",sep="\t",index=False)
    pd.DataFrame(pathcoverage).to_csv(out/"COMMON_PROTEIN_PATHWAY_COVERAGE.tsv",sep="\t",index=False)
    pd.DataFrame(sitecandidates).to_csv(out/"CURRENT_CPTAC_PHOSPHOSITE_CANDIDATES.tsv",sep="\t",index=False)
    alignment=pd.read_csv(product(rppa,"PATIENT_ALIGNMENT.tsv"),sep="\t")
    authority=pd.read_parquet(product(rppa,"PATIENT_AUTHORITY.parquet")).set_index("patient_id")
    selected=set(alignment.loc[alignment.status.eq("MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL"),"patient_id"])
    raw=set(alignment.patient_id)
    mr=record(retrospective);manifest=json.loads(mr.read_text());assert manifest["status"]=="COMPLETE"
    pairing=[];retro_summary=[];retromap=[]
    for entry in manifest["files"]:
        p=record(entry["path"],entry["sha256"])
        d=pd.read_csv(p,sep="\t",index_col=0);assert d.index.is_unique and d.columns.is_unique
        assert not np.isinf(d.to_numpy(float)).any()
        ids=[tcga_patient(x) for x in d.columns];assert len(ids)==len(set(ids))
        for native,pid in zip(d.columns,ids):
            pairing.append(dict(dataset=entry["cancer"]+"_"+entry["center"]+"_"+entry["layer"],
                native_patient_id=native,patient_id=pid,in_rppa500_release=pid in raw,
                in_current_rppa_rna_scope=pid in selected,
                cancer_id=authority.loc[pid,"cancer_id"] if pid in authority.index else "",
                patient_fold_id=authority.loc[pid,"patient_fold_id"] if pid in authority.index else "",
                match_basis="EXACT_TCGA_PATIENT_BARCODE_DOT_TO_HYPHEN_ONLY",
                same_sample_vial_verified=False,same_aliquot_verified=False))
        if entry["layer"]=="protein":
            observed=set(d.index[np.isfinite(d.to_numpy(float)).any(axis=1)])
            for r in protein.itertuples():
                symbols=symmap.get(r.gene_ids,[])
                if len(symbols)==1 and symbols[0] in observed:
                    retromap.append(dict(dataset=entry["cancer"]+"_"+entry["center"],rppa_feature_id=r.feature_id,
                        gene_id=r.gene_ids,ms_gene_symbol=symbols[0],primary_candidate=bool(r.primary_candidate)))
        retro_summary.append(dict(cancer=entry["cancer"],center=entry["center"],layer=entry["layer"],
            source_samples=len(ids),source_features=len(d),rppa500_patient_overlap=len(set(ids)&raw),
            current_rppa_rna_patient_overlap=len(set(ids)&selected),
            same_sample_vial_verified=False))
    pd.DataFrame(pairing).to_csv(out/"RETROSPECTIVE_PATIENT_PAIR_CANDIDATES.tsv",sep="\t",index=False)
    pd.DataFrame(retro_summary).to_csv(out/"RETROSPECTIVE_BRIDGE_SUMMARY.tsv",sep="\t",index=False)
    pd.DataFrame(retromap).to_csv(out/"RETROSPECTIVE_COMMON_PROTEIN_CANDIDATES.tsv",sep="\t",index=False)
    actual=json.loads((Path(cptac)/"COMPLETE.json").read_text())
    result={"target_host":"149","status":"COMPLETE_MAPPING_AND_PATIENT_BRIDGE_AUDIT",
        "rppa_site_label_corrections":int(cat.parser_changed.sum()),"current_cohorts":summary,
        "retrospective_bridge":retro_summary,"source_bindings":sources,
        "total_protein_adjustment_fitted":False,"pathway_scores_computed":False,"calibration_established":False,
        "limitations":["Patient-level matches do not certify identical sample-vial or aliquot.",
            "Gene/site labels are candidate matches, not isoform-sequence/epitope equivalence.",
            "Coverage thresholds 3/5/10 are reported descriptively, not adopted as quality gates.",
            "No RNA input, current Y, patient folds or running network source was changed."]}
    result["files"]={str(p.relative_to(out)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out/"COMPLETE.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["rppa","cptac","bundle","retrospective","output"]:p.add_argument("--"+key,required=True)
    a=p.parse_args();run(a.rppa,a.cptac,a.bundle,a.retrospective,a.output)
