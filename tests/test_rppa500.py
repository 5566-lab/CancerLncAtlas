import json
from pathlib import Path
import pandas as pd
import pytest
from cla_engine.common import Stop, sha, atomic_json
from cla_engine.rppa500 import feature_catalog, align_samples, marker_key, bound_path, validate_binding, FORMAT, PANEL

GDC_COLUMNS = ['peptide_target', 'gene_id', 'gene_name', 'validation_status', 'source', 'catalog_number']
PAPER_COLUMNS = ['Protein', 'Gene_symbol', 'Type', 'Company', 'Catalog_Number']


def catalog(column, *, kind='total', symbols='G1', gene_id='ENSG00001.2', name=None, qc='Caution'):
    p = {'antibodies': [{'protein_label': column, 'protein_name': name or column, 'genes': symbols,
                         'validation_status': qc, 'flag': 0, 'source': 'vendor', 'catalog_number': '123'}]}
    s = pd.DataFrame([[marker_key(column), symbols, kind, 'vendor', '123']], columns=PAPER_COLUMNS)
    g = pd.DataFrame([[name or column, gene_id, symbols, qc, 'vendor', '123']], columns=GDC_COLUMNS)
    return feature_catalog([column], p, s, g, {'G1': {'ENSG00001'}, 'G2': {'ENSG00002'}})


@pytest.mark.parametrize('column,name,kind,expected', [
    ('X4EBP1PS65', '4E-BP1_pS65', 'ptm', 'PHOSPHORYLATION'),
    ('FRA1PS265', 'FRA1PS265', 'ptm', 'PHOSPHORYLATION'),
    ('RPS6', 'RPS6', 'total', 'PROTEIN'),
    ('CASPASE3CLEAVED', 'Caspase-3-cleaved', 'ptm', 'PTM_OTHER'),
    ('FRA1PS265CLEAVEDD22B1', 'FRA1PS265CLEAVEDD22B1', 'ptm', 'PTM_OTHER'),
    ('ACETYLATUBULINLYS40', 'Acetyl-a-Tubulin-Lys40', 'ptm', 'PTM_OTHER'),
    ('PAR', 'PAR', '', 'PTM_OTHER'),
    ('MITOCHONDRIA', 'MITOCHONDRIA', '', 'OTHER_MEASUREMENT'),
])
def test_types_do_not_collapse_other_modifications_into_phosphorylation(column, name, kind, expected):
    x = catalog(column, kind=kind, name=name).iloc[0]
    assert x.molecular_layer == expected
    assert x.validation_status == 'Caution'
    assert x.portal_flag == 0


def test_multigene_multisite_antibody_remains_one_measurement():
    x = catalog('AKTPT308T309', kind='ptm', symbols='G1,G2',
                gene_id='ENSG00001.2/ENSG00002.7', name='AKT_pT308T309')
    assert len(x) == 1 and x.iloc[0].gene_ids == 'ENSG00001|ENSG00002'
    assert x.iloc[0].reported_phosphosite_labels == 'T308|T309'
    assert x.iloc[0].site_resolution == 'ANTIBODY_LABEL_ONLY_NOT_ISOFORM_SEQUENCE_VALIDATED'


def test_missing_portal_marker_can_use_exact_gdc_annotation_without_claiming_portal_qc():
    g = pd.DataFrame([['IRF-3_pS396', 'ENSG00001.2', 'G1', 'Under Evaluation', 'vendor', '123']], columns=GDC_COLUMNS)
    x = feature_catalog(['IRF3PS396'], {'antibodies': [{'protein_label': 'UNRELATED'}]},
                        pd.DataFrame(columns=PAPER_COLUMNS), g, {'G1': {'ENSG00001'}}).iloc[0]
    assert x.molecular_layer == 'PHOSPHORYLATION'
    assert x.annotation_panel_mismatch and not x.portal_annotated
    assert x.validation_status == 'Under Evaluation'
    assert x.validation_status_source == 'GDC_ANNOTATION'


def test_non_gene_measurement_retained_without_fabricated_gene_edge():
    x = catalog('MITOCHONDRIA', kind='', symbols='', gene_id='Unknown').iloc[0]
    assert x.gene_ids == '' and x.gene_count == 0
    assert x.molecular_layer == 'OTHER_MEASUREMENT'


def test_paper_exact_symbol_fallback_is_not_fuzzy():
    p = {'antibodies': [{'protein_label': 'M', 'protein_name': 'M'}]}
    s = pd.DataFrame([['M', 'G1,G2', 'total', 'v', '1']], columns=PAPER_COLUMNS)
    g = pd.DataFrame(columns=GDC_COLUMNS)
    x = feature_catalog(['M'], p, s, g, {'G1': {'ENSG00001'}, 'G2': {'ENSG00002'}}).iloc[0]
    assert x.gene_ids == 'ENSG00001|ENSG00002'
    y = feature_catalog(['M'], p, s, g, {'G1': {'ENSG00001'}, 'G2_alias': {'ENSG00002'}}).iloc[0]
    assert y.gene_count == 0


def test_unknown_marker_and_duplicate_annotation_fail():
    p = {'antibodies': [{'protein_label': 'M'}]}
    s = pd.DataFrame(columns=PAPER_COLUMNS)
    g = pd.DataFrame(columns=GDC_COLUMNS)
    with pytest.raises(Stop, match='No exact official annotation'):
        feature_catalog(['UNLISTED'], p, s, g, {})
    p['antibodies'].append({'protein_label': 'M'})
    with pytest.raises(Stop, match='Duplicate portal'):
        feature_catalog(['M'], p, s, g, {})


def authority():
    return pd.DataFrame({'patient_id': ['TCGA-AA-0001', 'TCGA-AA-0002'],
                         'sample_id': ['TCGA-AA-0001-01A-RNA', 'TCGA-AA-0002-01A-RNA'],
                         'cancer_id': ['BRCA', 'LUAD'], 'patient_fold_id': [0, 1]})


def test_exact_sample_vial_join_never_accepts_patient_only_or_normal():
    d = pd.DataFrame({'sample_id': ['TCGA-AA-0001-01B', 'TCGA-AA-0002-11A', 'TCGA-AA-0003-01A']})
    a = align_samples(d, authority())
    assert a.status.tolist() == ['RNA_SAMPLE_VIAL_MISMATCH', 'NON_TUMOR_SAMPLE', 'NO_FROZEN_RNA_PATIENT']
    assert not a.aliquot_match_claimed.any()


def test_matched_sample_is_not_claimed_to_have_aliquot_identity():
    a = align_samples(pd.DataFrame({'sample_id': ['TCGA-AA-0001-01A']}), authority())
    assert a.status.iloc[0] == 'MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL'
    assert a.sample_quality_basis.iloc[0] == 'PUBLISHED_FINAL_RPPA500_RELEASE'
    assert not a.aliquot_match_claimed.iloc[0]


def test_repeated_patient_is_unresolved_not_averaged():
    d = pd.DataFrame({'sample_id': ['TCGA-AA-0001-01A', 'TCGA-AA-0001-01B']})
    a = align_samples(d, authority())
    assert not a.status.str.startswith('MATCHED').any()


@pytest.mark.parametrize('samples', [
    ['TCGA-AA-0001'],
    ['TCGA-AA-0001-01A', 'TCGA-AA-0001-01A'],
])
def test_invalid_or_duplicate_source_sample_keys_fail(samples):
    with pytest.raises(Stop):
        align_samples(pd.DataFrame({'sample_id': samples}), authority())


def test_changed_source_or_legacy_product_cannot_be_relabelled(tmp_path):
    f = tmp_path/'COMPLETE.json'
    atomic_json(f, {'format': 'CLA_FIVE_ASSAY_RAW_INPUTS_V1', 'status': 'COMPLETE', 'target_host': '149'})
    pinned = {'path': str(f), 'sha256': sha(f)}
    with pytest.raises(Stop) as e:
        validate_binding({'format': FORMAT, 'panel': PANEL, 'complete': pinned})
    assert e.value.code == 'RPPA500_PRODUCT_VERSION'
    f.write_text('{}')
    with pytest.raises(Stop) as e:
        bound_path(pinned)
    assert e.value.code == 'RPPA500_SOURCE_HASH'


def test_public_template_uses_rppa500_and_preserves_target_retirement():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'config/149.json').read_text())
    assert cfg['rppa_input']['panel'] == PANEL
    assert cfg['pathway_target_policy']['status'] == 'REPLACEMENT_REQUIRED'
    assert cfg['model']['task_weights'] == {'pathway': 1.0, 'state': .5}
    assert validate_binding(None) is None


def test_prior_official_exclusions_and_mixed_aliquots_are_preserved():
    d = pd.DataFrame({'sample_id': ['TCGA-AA-0001-01A', 'TCGA-AA-0002-01A']})
    q = pd.DataFrame({'platform': ['MDA_RPPA_Core'] * 3,
                      'aliquot_barcode': ['TCGA-AA-0001-01A-P1', 'TCGA-AA-0001-01A-P2', 'TCGA-AA-0002-01A-P1'],
                      'Do_not_use': ['False', 'True', 'False'],
                      'AWG_excluded_because_of_pathology': ['0.0', '', '1.0']})
    a = align_samples(d, authority(), q)
    assert a.status.eq('PRIOR_OFFICIAL_RPPA_SAMPLE_VIAL_EXCLUDED').all()
    assert a.prior_rppa_qc_mixed.tolist() == [True, False]
    q.loc[0, 'Do_not_use'] = 'unknown'
    with pytest.raises(Stop) as e:
        align_samples(d, authority(), q)
    assert e.value.code == 'RPPA500_QUALITY_FLAG'
