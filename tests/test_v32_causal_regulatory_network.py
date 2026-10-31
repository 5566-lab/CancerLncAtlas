"""Actual native source readers and GNN consumers; synthetic data, no fitting."""
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32.causal_regulatory_inputs import (
    SourceFiles, build_inputs, group_known_studies, sha256, write_json,
)
from cc_hhgt.v32.causal_regulatory_network import CausalRegulatoryNetwork


def table(root, name, rows):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def repin(root):
    write_json(root / "input_manifest.json", {"files":[
        {"path":p.relative_to(root).as_posix(), "bytes":p.stat().st_size, "sha256":sha256(p)}
        for p in sorted(root.rglob("*")) if p.is_file() and p.name != "input_manifest.json"]})
    return sha256(root / "input_manifest.json")


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    genes = [
        dict(gene_id="ENSG001", gene_name="L", hgnc_readout_symbol="L", hgnc_readout_class="lncRNA"),
        dict(gene_id="ENSG002", gene_name="B", hgnc_readout_symbol="B", hgnc_readout_class="protein_coding"),
        dict(gene_id="ENSG003", gene_name="C", hgnc_readout_symbol="C", hgnc_readout_class="protein_coding")]
    for source in ("PerturbAtlas", "GPSAdb2", "GEO_rescue_GSE154812"):
        rows = [{**r, "ensembl_gene_id":r["gene_id"]} for r in genes]
        table(root, "lncRNA_edge_audit/" + source + "_readout_annotation.tsv", rows)
    table(root, "PerturbAtlas/genes.tsv", genes)
    perturbations = []
    for i, (name, method) in enumerate([("P_KO","KO"), ("P_OE","OE"), ("P_BAD","KO|OE")]):
        perturbations.append(dict(perturb_id=name, condition=method, tissue_name="cell",
            study_accession="GSE101", split="train", row_index=i, canonical_target="GENE:ENSG001",
            target_hgnc_symbol="L", target_class="lncRNA", candidate_primary=method != "KO|OE",
            n_control_samples=2, n_treatment_samples=2))
    table(root, "PerturbAtlas/contrasts.tsv", perturbations)
    table(root, "PerturbAtlas/samples.tsv", [dict(contrast_id=x["perturb_id"], sample_id="GSM10", study="GSE101") for x in perturbations])
    with h5py.File(root / "PerturbAtlas/responses.h5", "w") as handle:
        handle["gene_id"] = np.array([r["gene_id"] for r in genes], dtype=h5py.string_dtype())
        handle["contrast_id"] = np.array([r["perturb_id"] for r in perturbations], dtype=h5py.string_dtype())
        handle["log2FoldChange"] = np.array([[0,2,np.nan], [0,-2,1], [0,1,1]], dtype=np.float32)
        handle["lfcSE"] = np.ones((3,3), dtype=np.float32)
        handle["padj"] = np.array([[1,.01,np.nan], [1,.02,.5], [1,.02,.03]], dtype=np.float32)
    table(root, "GPSAdb2/contrasts.tsv", [dict(dataset_id="D1", intervention="KD", cell_line="cell",
        study_gse="GSE101", split="test", canonical_target="GENE:ENSG001", target_hgnc_symbol="L",
        target_class="lncRNA", candidate_primary=True, source_method="shRNA", n_control=2, n_treatment=2)])
    table(root, "GPSAdb2/samples.tsv", [dict(dataset_id="D1", sample_id="s"+str(i+1),
        study="GSE101",group="control" if i<2 else "treat") for i in range(4)])
    counts = np.array([[2,2,1,1],[4,4,8,8],[0,0,0,0]], dtype=np.int64)
    logged = np.log2(counts / counts.sum(axis=0) * 1e6 + 1).astype(np.float32)
    path = root / "GPSAdb2/datasets"; path.mkdir()
    np.savez(path / "D1.npz", raw_counts=counts, log2cpm_plus1=logged,
        effect_mean_log2cpm_treat_minus_control=logged[:,2:].mean(1)-logged[:,:2].mean(1),
        gene_id=np.array([r["gene_id"] for r in genes]), sample_id=np.array(["s1","s2","s3","s4"]),
        control=np.array([True,True,False,False]), treatment=np.array([False,False,True,True]))
    oe = []
    for i, reviewed in enumerate([False,True]):
        oe.append(dict(signatureid="OE"+str(i), cellline="cell", plate="PLATE1", split_by_plate="train",
            matrix_signature_index=i, canonical_target="GENE:ENSG001" if reviewed else "GENE:ENSG002",
            target_hgnc_symbol="L" if reviewed else "B", target_class="lncRNA" if reviewed else "protein_coding",
            lncRNA_target_requires_construct_or_guide_review=reviewed))
    table(root, "LINCS_iLINCS_OE/signature_metadata.tsv", oe)
    table(root, "LINCS_iLINCS_OE/landmark_genes.tsv", [dict(ID_geneid="2",Name_GeneSymbol="B"),dict(ID_geneid="3",Name_GeneSymbol="C")])
    table(root,"lncRNA_edge_audit/LINCS_iLINCS_OE_readout_annotation.tsv",
          [dict(ID_geneid="2",hgnc_readout_symbol="B"),dict(ID_geneid="3",hgnc_readout_symbol="C")])
    np.savez(root/"LINCS_iLINCS_OE/landmark_signatures.npz", effect=np.array([[2,3],[-1,-2]],dtype=np.float32),
             entrez_gene_id=np.array(["2","3"]), signature_id=np.array(["OE0","OE1"]))
    table(root, "L2S2_KO/genes.tsv", [
        dict(gene_symbol="B",gene_index=0,is_direct_landmark=True),
        dict(gene_symbol="C",gene_index=1,is_direct_landmark=True),
        dict(gene_symbol="L",gene_index=2,is_direct_landmark=False)])
    table(root,"lncRNA_edge_audit/L2S2_KO_readout_annotation.tsv",
          [dict(gene_symbol=s,hgnc_readout_symbol=s) for s in ["B","C","L"]])
    table(root, "L2S2_KO/signatures.tsv", [dict(signature_id="KO1",signature_index=0,
        intervention="CRISPR_KO",cell_line="cell",plate="PLATE1",split="test",
        canonical_target="GENE:ENSG002",target_hgnc_symbol="B",target_class="protein_coding",
        candidate_primary=True)])
    sparse.save_npz(root/"L2S2_KO/up_gene_sets.npz", sparse.csr_matrix([[1,0,1]],dtype=np.int8))
    sparse.save_npz(root/"L2S2_KO/down_gene_sets.npz", sparse.csr_matrix([[0,0,0]],dtype=np.int8))
    table(root, "LncRNA2Target3/experiments_human.tsv", [
        dict(experiment_id=1,intervention_source="Overexpressed",perturbed_entity_id="GENE:ENSG001",
             perturbed_hgnc_symbol="L",perturbed_hgnc_class="lncRNA",cell_line="cell",tissue="",
             gse="GSE101",pmid="123.0",suspected_source_column_misalignment=False),
        dict(experiment_id=286,intervention_source="Knockdown",perturbed_entity_id="GENE:ENSG001",
             perturbed_hgnc_symbol="L",perturbed_hgnc_class="lncRNA",cell_line="cell",tissue="",
             gse="GSE999",pmid="999",suspected_source_column_misalignment=True)])
    table(root, "LncRNA2Target3/high_throughput_human_response.tsv.gz", [
        dict(experiment_id=x,response_entity_id="GENE:ENSG002",response_hgnc_symbol="B",
             response_direction_for_training=1 if x==1 else "", direction_training_eligible=x==1,
             p_value=.01 if x==1 else 50,adjusted_p_value=.02,effect_log2FC=99999,
             source_numeric_issue="" if x==1 else "misaligned",record_id="HT:"+str(x))
        for x in [1,286]])
    table(root, "LncRNA2Target3/low_throughput_human_relations.tsv.gz", [dict(record_id="LT:1", evidence="curated")])
    table(root, "LncTarD2/curated_relations_all.tsv.gz", [dict(record_id="RID1",
        regulator_entity_id="GENE:ENSG001",regulator_hgnc_symbol="L",regulator_hgnc_class="lncRNA",
        DiseaseName="context",PubMedID="123",study_group="PMID:123",intervention_type="unstructured",
        evidence_layer="curated_functional",levelOfRegulation="RNA-protein")])
    table(root, "GEO_rescue_GSE154812/contrasts.tsv", [
        dict(contrast_id=k,kind=kind,control=control,rescue_type=role)
        for k,kind,control,role in [
            ("NORAD_CRISPRi","CRISPRi","control","none"),
            ("circPRE2","low_valency_nonrescuing_comparator","NORAD_CRISPRi","negative_control"),
            ("circPRE8","functional_bypass","NORAD_CRISPRi","functional_bypass_not_WT")]])
    names = ["c1","c2","k1","k2","n1","n2","r1","r2"]
    groups = ["control"]*2+["NORAD_CRISPRi"]*2+["NORAD_CRISPRi_circPRE2"]*2+["NORAD_CRISPRi_circPRE8"]*2
    table(root, "GEO_rescue_GSE154812/samples.tsv", [dict(sample_id=n,group=g,study_gse="GSE154812",study_srp="SRP273076") for n,g in zip(names,groups)])
    values = np.array([[2.5,2.5,1,1,1,1,2.5,2.5],[4,4,8,8,8,8,4,4],[1,1,1,1,1,1,1,1]])
    log = np.log2(values+1)
    kd=log[:,2:4].mean(1)-log[:,:2].mean(1)
    rescue=log[:,6:].mean(1)-log[:,2:4].mean(1)
    table(root, "GEO_rescue_GSE154812/rescue_effects.tsv.gz", [
        dict(ensembl_gene_id=gene["gene_id"],gene_name=gene["gene_name"],
             NORAD_CRISPRi_minus_control=kd[i],circPRE2_construct_minus_CRISPRi=0,
             circPRE8_construct_minus_CRISPRi=rescue[i],circPRE8_residual_vs_control=0,
             circPRE8_minus_circPRE2=rescue[i]) for i,gene in enumerate(genes)])
    np.savez(root/"GEO_rescue_GSE154812/expression.npz", source_normalized_values=values,
        log2_source_plus_1=log, gene_id=np.array([r["gene_id"] for r in genes]),sample_id=np.array(names))
    (root/"CasRx2024").mkdir()
    fitness=np.array([[.5,.1],[np.nan,.2],[.3,.4]])
    np.savez(root/"CasRx2024/lncRNA_fitness.npz", fitness_lfc=fitness,
        observed_mask=np.isfinite(fitness),RRA_score=np.full((3,2),.1),p_value=np.full((3,2),.1),
        FDR=np.full((3,2),.1),record_id=np.array(["r1","r2","r3"]),family_id=np.array(["f1","f2","f3"]),
        library=np.array(["ALBAROSSA_library","ALBAROSSA_library","other"]),
        condition_id=np.array(["CELL","NO_CASRX"]),casrx_present=np.array([True,False]))
    output=tmp_path/"index"
    build_inputs(root,output,expected_manifest_sha256=repin(root))
    return root, output/"COMPLETE.json"


def test_catalog_preserves_all_sources_and_known_overlap(native):
    _, manifest=native; n=CausalRegulatoryNetwork(manifest)
    assert len(n.manifest["sources"])==8
    ids=n.index.loc[n.index.study_ids.str.contains("GSE101"),"known_study_group"]
    assert ids.nunique()==1
    assert n.experiment("LncRNA2Target3","1")["intervention_type"]=="OE"
    assert set(n.index.split)=={"unassigned","mechanism_validation"}
    assert not n.manifest["cross_source_split_ready"]


def test_known_sample_overlap_propagates_rescue_embargo():
    frame=pd.DataFrame([dict(study_ids="GSE1;GSM1",mechanism_holdout=True),
                        dict(study_ids="SRP1;GSM1",mechanism_holdout=False),
                        dict(study_ids="GSE2",mechanism_holdout=False)])
    result=group_known_studies(frame)
    assert result.split.tolist()==["mechanism_validation","mechanism_validation","unassigned"]
    assert result.known_study_group.iloc[0]==result.known_study_group.iloc[1]


def test_shrna_source_review_label_remains_kd_not_ko(native,tmp_path):
    root,_=native;path=root/"L2S2_KO/signatures.tsv"
    values=pd.read_csv(path,sep="\t")
    values["intervention"]="shRNA_label_requires_source_review_NOT_KO"
    values["candidate_primary"]=False
    values.to_csv(path,sep="\t",index=False)
    output=tmp_path/"reviewed_index"
    build_inputs(root,output,expected_manifest_sha256=repin(root))
    row=CausalRegulatoryNetwork(output/"COMPLETE.json").experiment("L2S2_KO","KO1")
    assert row["intervention_type"]=="KD" and not row["network_eligible"]
    assert row["source_method"]=="shRNA_label_requires_source_review_NOT_KO"


def test_source_hash_drift_fails(native):
    root,manifest=native
    (root/"PerturbAtlas/genes.tsv").write_text("altered")
    with pytest.raises(ValueError,match="bytes changed"):
        CausalRegulatoryNetwork(manifest).responses("PerturbAtlas","P_KO")


def test_manifest_hash_drift_fails(native):
    root,manifest=native
    (root/"input_manifest.json").write_text("{}")
    with pytest.raises(ValueError,match="manifest hash"):
        CausalRegulatoryNetwork(manifest)


def test_source_path_escape_fails(tmp_path):
    write_json(tmp_path/"input_manifest.json",{"files":[{"path":"../outside","bytes":1,"sha256":"a"*64}]})
    with pytest.raises(ValueError,match="Unsafe"): SourceFiles(tmp_path)


def test_ko_response_sign_is_not_activation_and_missing_is_not_zero(native):
    n=CausalRegulatoryNetwork(native[1]); rows=n.responses("PerturbAtlas","P_KO")
    assert rows.effect.tolist()[:2]==[0,2]
    assert np.isnan(rows.effect.iloc[2]) and not rows.effect_observed.iloc[2]
    assert rows.regulatory_activation_sign.isna().all()
    edge=n.edges("PerturbAtlas","P_KO",max_fdr=.05)
    assert len(edge)==1 and edge.response_direction.iloc[0]==1
    assert edge.source_canonical_id.iloc[0]=="LNC:ENSG001"
    assert edge.target_canonical_id.iloc[0]=="GENE:ENSG002"
    assert edge.intervention_type.iloc[0]=="KO" and set(edge.physical_binding_label)=={"not_determined"}


def test_oe_orientation_and_native_scale(native):
    rows=CausalRegulatoryNetwork(native[1]).responses("LINCS_iLINCS_OE","OE0")
    assert rows.effect.tolist()==[2,-1]
    assert rows.response_id.tolist()==["ENSG002","ENSG003"]
    assert set(rows.effect_scale)=={"iLINCS_source_MODZ_signature"}


def test_oe_uses_verified_readout_symbol_annotation(native):
    root,_=native;path=root/"LINCS_iLINCS_OE/landmark_genes.tsv"
    rows=pd.read_csv(path,sep="\t")
    rows.loc[0,"Name_GeneSymbol"]="OLD_B"
    rows.to_csv(path,sep="\t",index=False);rebind(native)
    rows=CausalRegulatoryNetwork(native[1]).responses("LINCS_iLINCS_OE","OE0")
    assert rows.response_id.iloc[0]=="ENSG002" and rows.readout.iloc[0]=="RNA_signature"


@pytest.mark.parametrize("source,identifier",[("PerturbAtlas","P_BAD"),("LINCS_iLINCS_OE","OE1"),("LncRNA2Target3","286")])
def test_review_records_retained_but_not_network_edges(native,source,identifier):
    n=CausalRegulatoryNetwork(native[1])
    assert not n.responses(source,identifier).empty
    assert n.edges(source,identifier).empty


def test_l2s2_unlisted_and_inferred_readouts_are_not_labels(native):
    n=CausalRegulatoryNetwork(native[1]); rows=n.responses("L2S2_KO","KO1")
    assert rows.effect.isna().all()
    assert np.isnan(rows.response_direction.iloc[1])
    assert not rows.directly_measured.iloc[2]
    edges=n.edges("L2S2_KO","KO1")
    assert len(edges)==1 and edges.weight.iloc[0]==1


def test_unverified_lncrna2target_magnitude_is_not_regression_label(native):
    n=CausalRegulatoryNetwork(native[1]); rows=n.responses("LncRNA2Target3","1")
    assert rows.effect.isna().all() and rows.source_reported_effect_for_audit.iloc[0]==99999
    assert n.edges("LncRNA2Target3","1").weight.iloc[0]==1
    bad=n.responses("LncRNA2Target3","286")
    assert not bad.p_value_available.any() and not bad.direction_available.any()


def test_gpsadb_native_group_replay(native):
    rows=CausalRegulatoryNetwork(native[1]).responses("GPSAdb2","D1")
    assert rows.effect.iloc[1]>0 and rows.effect.iloc[2]==0
    assert rows.effect_observed.iloc[2]
    assert rows.adjusted_p_value.isna().all()
    assert rows.source_method.iloc[0]=="shRNA"


def test_rescue_replays_fractional_expression_and_retains_control(native):
    n=CausalRegulatoryNetwork(native[1])
    rescue=n.responses("GEO_rescue_GSE154812","circPRE8")
    comparator=n.responses("GEO_rescue_GSE154812","circPRE2")
    assert set(rescue.intervention_type)=={"RESCUE"}
    assert set(comparator.intervention_type)=={"RESCUE_CONTROL"}
    assert rescue.mechanism_holdout.all() and rescue.rescue_success_label.isna().all()
    assert rescue.descriptive_reversal_only.tolist()==[True,True,False]
    assert not rescue.unperturbed_construct_control_available.any()
    assert np.all(comparator.effect==0)
    assert set(n.edges("GEO_rescue_GSE154812","circPRE8").source_canonical_id)=={"CONSTRUCT:circPRE8"}


def test_auxiliary_sources_are_consumed_with_native_semantics(native):
    n=CausalRegulatoryNetwork(native[1])
    assert len(n.auxiliary("LncTarD2"))==1 and len(n.auxiliary("LncRNA2Target3_low"))==1
    fitness=n.auxiliary("CasRx2024")
    assert fitness["fitness_lfc"].shape==(2,1)
    assert not fitness["observed_mask"][1,0] and np.isnan(fitness["fitness_lfc"][1,0])
    with pytest.raises(ValueError,match="separate evidence"): n.responses("CasRx2024","fitness")


@pytest.mark.parametrize("fdr",[-.1,0,1.1])
def test_invalid_fdr_policy_fails(native,fdr):
    with pytest.raises(ValueError,match="max_fdr"):
        CausalRegulatoryNetwork(native[1]).edges("PerturbAtlas","P_KO",max_fdr=fdr)


def test_absent_statistics_cannot_pass_fdr_filter(native):
    assert CausalRegulatoryNetwork(native[1]).edges("GPSAdb2","D1",max_fdr=.05).empty


def test_unreviewed_training_split_and_unknown_experiments_fail(native):
    n=CausalRegulatoryNetwork(native[1])
    with pytest.raises(ValueError,match="study split"): n.graph("PerturbAtlas","P_KO",purpose="training")
    with pytest.raises(ValueError,match="Unknown"): n.responses("PerturbAtlas","missing")


@pytest.mark.parametrize("kind", ["cc_hhgt", "rgcn"])
def test_real_graph_consumer_and_frozen_encoder(native,kind):
    import torch
    from cc_hhgt.gnn import build_model, move_graph
    from cc_hhgt.v32.patient_conditioning import experimental_regulatory_graph
    graph=experimental_regulatory_graph(native[1],"PerturbAtlas","P_OE",max_fdr=.05)
    assert graph.experimental_regulatory_authority["response_edges"]==1
    assert set(graph.edges.relation_polarity)=={-1}
    cfg={"training":{"hidden_channels":8,"num_layers":1,"num_heads":2,"dropout":0,
                     "pair_hidden_channels":4},"task_contract":{"pathway":{"target_level":"exact"}}}
    torch.manual_seed(13); model=build_model(kind,graph,2,cfg).eval()
    parameters={k:v.clone() for k,v in model.state_dict().items()}
    data=move_graph(graph,"cpu",kind)
    with torch.no_grad(): original=model.encode(data)
    if kind=="rgcn":
        altered={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
        altered["edge_weight"].neg_()
    else:
        altered=data.clone()
        for relation in altered.edge_types: altered[relation].edge_weight.neg_()
    with torch.no_grad(): changed=model.encode(altered)
    if kind=="rgcn": original={"all":original};changed={"all":changed}
    assert all(torch.isfinite(v).all() for v in original.values())
    assert any(not torch.equal(original[k],changed[k]) for k in original)
    assert all(torch.equal(parameters[k],v) for k,v in model.state_dict().items())


def rebind(native):
    root,manifest=native
    data=json.loads(manifest.read_text())
    data["source_manifest_sha256"]=repin(root)
    write_json(manifest,data)


@pytest.mark.parametrize("wrong_mask", [False, True])
def test_finite_fitness_without_fdr_does_not_become_observed(native, wrong_mask):
    path = native[0]/"CasRx2024/lncRNA_fitness.npz"
    with np.load(path) as z: values = {k:z[k] for k in z.files}
    values["FDR"][0,0] = np.nan
    values["observed_mask"][0,0] = wrong_mask
    np.savez(path, **values)
    rebind(native)
    reader = CausalRegulatoryNetwork(native[1])
    if wrong_mask:
        with pytest.raises(ValueError, match="missing-value mask"):
            reader.auxiliary("CasRx2024")
    else:
        result = reader.auxiliary("CasRx2024")
        assert result["fitness_lfc"][0,0] == .5
        assert result["fitness_lfc_available"][0,0]
        assert not result["FDR_available"][0,0]
        assert not result["observed_mask"][0,0]


def test_contrast_axis_mismatch_is_rejected_even_with_valid_file_hash(native):
    with h5py.File(native[0]/"PerturbAtlas/responses.h5","r+") as h: h["contrast_id"][0]="different"
    rebind(native)
    with pytest.raises(ValueError,match="contrast axis"):
        CausalRegulatoryNetwork(native[1]).responses("PerturbAtlas","P_KO")


def test_l2s2_conflicting_membership_is_unavailable(native):
    sparse.save_npz(native[0]/"L2S2_KO/down_gene_sets.npz",sparse.csr_matrix([[1,0,0]],dtype=np.int8))
    rebind(native);n=CausalRegulatoryNetwork(native[1])
    rows=n.responses("L2S2_KO","KO1")
    assert rows.set_membership_conflict.iloc[0] and np.isnan(rows.response_direction.iloc[0])
    assert n.edges("L2S2_KO","KO1").empty


def test_gpsadb_sample_identity_is_required(native):
    path=native[0]/"GPSAdb2/datasets/D1.npz"
    with np.load(path) as z: values={k:z[k] for k in z.files}
    values["sample_id"]=np.array(["wrong","s2","s3","s4"])
    np.savez(path,**values);rebind(native)
    with pytest.raises(ValueError,match="sample identity"):
        CausalRegulatoryNetwork(native[1]).responses("GPSAdb2","D1")


def test_rescue_effect_must_replay_from_groups(native):
    path=native[0]/"GEO_rescue_GSE154812/rescue_effects.tsv.gz"
    values=pd.read_csv(path,sep="\t")
    values["circPRE8_construct_minus_CRISPRi"]+=1
    values.to_csv(path,sep="\t",index=False);rebind(native)
    with pytest.raises(ValueError,match="replay"):
        CausalRegulatoryNetwork(native[1]).responses("GEO_rescue_GSE154812","circPRE8")


def test_duplicate_response_ids_are_reviewed_not_pooled(native):
    path=native[0]/"LncRNA2Target3/high_throughput_human_response.tsv.gz"
    values=pd.read_csv(path,sep="\t")
    duplicate=values.iloc[[0]].copy();duplicate["record_id"]="HT:repeat"
    pd.concat([values,duplicate]).to_csv(path,sep="\t",index=False);rebind(native)
    rows=CausalRegulatoryNetwork(native[1]).responses("LncRNA2Target3","1")
    assert len(rows)==2 and not rows.network_eligible.any()
    assert rows.review_reason.str.contains("DUPLICATE_RESPONSE").all()


def test_cli_prepares_native_graph_and_exact_artifact_hashes(native,tmp_path,monkeypatch):
    import sys,torch
    from scripts.prepare_v32_causal_regulatory_network import main
    output=tmp_path/"prepared"
    monkeypatch.setattr(sys,"argv",["prepare","network","--input-manifest",str(native[1]),
        "--source","PerturbAtlas","--experiment","P_KO","--max-fdr","0.05","--output",str(output)])
    main()
    receipt=json.loads((output/"COMPLETE.json").read_text())
    for item in receipt["files"].values(): assert sha256(output/item["path"])==item["sha256"]
    payload=torch.load(output/"GRAPH.pt",map_location="cpu",weights_only=False)
    assert payload["contains_trained_parameters"] is False
    assert len(payload["bundle"].edges)==1
