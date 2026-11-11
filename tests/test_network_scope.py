import json
from pathlib import Path
import pytest
from cla_engine.network_scope import decide_network_scope, cross_database_links, verify_snapshot, sha

def mapping(n="N00056", status="mapped", sid="src"):
    return dict(source_target_id=sid, network_id=n, mapping_status=status, canonical_id="canonical")

def record(n="N00056"):
    return dict(network_id=n,name="WNT canonical",definition="A -> B -| C",expanded_definition="A -> B -| C",network_type="Reference",pathway_ids="hsa04310")

def member(n="N00056", category="Cancer signaling", mid="nt06215"):
    return dict(network_id=n,map_id=mid,map_name="WNT cancer",category=category,is_cancer_map=category in {"Cancer","Cancer signaling"})

def definitions():
    return [
      dict(node_id="n",authority_id="KEGG.NETWORK:N00056",source_database="KEGG_MEDICUS",integrated_canonical_id="canonical"),
      dict(node_id="r",authority_id="Reactome:R-HSA-195721",source_database="Reactome",integrated_canonical_id="reactome",display_name="WNT module")
    ]

def admitted():
    return decide_network_scope([mapping()],[record()],[member()])

def test_direct_official_cancer_map_is_required():
    rows=decide_network_scope([mapping(),mapping("N12345",sid="other")],[record(),record("N12345")],[member(),member("N12345","Signal transduction","nt06505")])
    assert [r["admitted_by_network_scope"] for r in rows] == [True,False]
    assert rows[1]["cancer_scope_status"]=="outside_selected_network_categories"
    assert rows[0]["mechanism_definition"]=="A -> B -| C"

def test_missing_source_or_record_is_unknown_not_outside():
    rows=decide_network_scope([mapping(status="unmapped_network"),mapping("N12345",sid="missing")],[record()],[member()])
    assert [r["cancer_scope_status"] for r in rows] == ["unresolved_source_identity","unresolved_network_record"]
    assert not any(r["admitted_by_network_scope"] for r in rows)

def test_missing_classification_is_unknown():
    row=decide_network_scope([mapping()],[record()],[])[0]
    assert row["cancer_scope_status"]=="unresolved_network_classification"

def test_ambiguous_source_rows_preserve_candidates_without_admission():
    rows=decide_network_scope([mapping(status="ambiguous_network"),mapping("N12345",status="ambiguous_network")],[record(),record("N12345")],[member(),member("N12345")])
    assert len(rows)==2
    assert not any(r["admitted_by_network_scope"] for r in rows)

def test_conflicting_successful_mappings_fail():
    with pytest.raises(ValueError,match="DUPLICATE_SOURCE"):
        decide_network_scope([mapping(),mapping("N12345")],[record(),record("N12345")],[])

def test_category_flag_conflict_fails():
    bad=member(category="Infectious disease")
    bad["is_cancer_map"]=True
    with pytest.raises(ValueError,match="CATEGORY_FLAG_CONFLICT"):
        decide_network_scope([mapping()],[record()],[bad])

def test_reference_variant_networks_keep_identity():
    var=record("N00058")
    var["network_type"]="Variation"
    rows=decide_network_scope([mapping(),mapping("N00058",sid="variant")],[record(),var],[member(),member("N00058")])
    assert len({r["alignment_key"] for r in rows})==2
    assert {r["network_type"] for r in rows}=={"Reference","Variation"}

def test_parent_equivalence_cannot_merge_or_transfer_scope():
    external=[dict(source_resource="kegg.pathway",source_external_id="path:hsa04310",target_resource="reactome",target_external_id="R-HSA-195721",target_canonical_id="reactome",target_name="Signaling by WNT",relation="equivalentTo",mapping_source="ComPath")]
    links=cross_database_links(admitted(),definitions(),[],external)
    assert len(links)==1
    assert links[0]["alignment_relation"]=="shared_parent_pathway_candidate"
    assert not links[0]["scope_transfer"]
    assert not links[0]["identity_merge"]
    assert links[0]["shared_parent_pathway_id"]=="hsa04310"

def test_reviewed_exact_mechanism_can_reuse_scope():
    reviews=[dict(node_a="n",node_b="r",relation="same_functional_identity",evidence="reviewed reaction core",biological_reason="same reaction and perturbation context")]
    links=cross_database_links(admitted(),definitions(),reviews,[])
    assert len(links)==1
    assert links[0]["scope_transfer"]
    assert links[0]["alignment_relation"]=="reviewed_equivalent_mechanism"
    assert links[0]["target_canonical_id"]=="reactome"
    assert not links[0]["identity_merge"]

def test_similarity_cannot_merge_or_transfer_scope():
    pairs=[dict(node_a="n",node_b="r",shared_proteins=20,jaccard=1.0,overlap_coefficient=1.0)]
    links=cross_database_links(admitted(),definitions(),[],[],pairs)
    assert len(links)==1
    assert not links[0]["scope_transfer"]
    assert not links[0]["identity_merge"]
    assert links[0]["alignment_relation"]=="protein_similarity_candidate"

def test_nonadmitted_source_produces_no_links():
    scope=decide_network_scope([mapping()],[record()],[member(category="Signal transduction")])
    assert cross_database_links(scope,definitions(),[],[])==[]

def test_snapshot_hash_checks_every_source_and_output(tmp_path):
    src=tmp_path/"source.txt"
    out=tmp_path/"output.tsv"
    src.write_text("original")
    out.write_text("parsed")
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"sources":[{"path":src.name,"sha256":sha(src)}],"outputs":[{"path":out.name,"sha256":sha(out)}]}))
    assert verify_snapshot(tmp_path,manifest)
    src.write_text("changed")
    with pytest.raises(ValueError,match="CHECKSUM_MISMATCH"):
        verify_snapshot(tmp_path,manifest)

def test_snapshot_rejects_path_escape(tmp_path):
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps({"sources":[{"path":"../outside","sha256":"x"}],"outputs":[{"path":"output","sha256":"x"}]}))
    with pytest.raises(ValueError,match="CHECKSUM_MISMATCH"):
        verify_snapshot(tmp_path,manifest)

def test_recovered_source_can_link_old_definition_without_network_authority():
    scope=admitted()
    scope[0]["original_canonical_id"]="n"
    defs=definitions()
    defs[0]["authority_id"]="MSigDB:M12345"
    pairs=[dict(node_a="n",node_b="r",shared_proteins=2,jaccard=.5,overlap_coefficient=1)]
    links=cross_database_links(scope,defs,[],[],pairs)
    assert len(links)==1
    assert links[0]["alignment_relation"]=="protein_similarity_candidate"

def test_recovered_source_cannot_override_conflicting_network_identity():
    scope=admitted()
    scope[0]["original_canonical_id"]="n"
    defs=definitions()
    defs[0]["authority_id"]="KEGG.NETWORK:N99999"
    with pytest.raises(ValueError,match="RECOVERED_IDENTITY_CONFLICT"):
        cross_database_links(scope,defs,[],[])

def test_recovered_source_anchors_must_agree():
    scope=admitted()
    scope[0].update(original_canonical_id="n",systematic_id="M99999")
    defs=definitions()
    defs.append(dict(node_id="other",source_database="KEGG_MEDICUS",systematic_source_id="M99999"))
    with pytest.raises(ValueError,match="SOURCE_ANCHOR_CONFLICT"):
        cross_database_links(scope,defs,[],[])

def test_partof_original_direction_is_preserved():
    external=[dict(source_resource="reactome",source_external_id="R-HSA-195721",target_resource="kegg.pathway",target_external_id="hsa04310",relation="isPartOf",mapping_source="ComPath")]
    link=cross_database_links(admitted(),definitions(),[],external)[0]
    assert link["mapping_source_resource"]=="reactome"
    assert link["mapping_target_external_id"]=="hsa04310"
    assert not link["scope_transfer"]
