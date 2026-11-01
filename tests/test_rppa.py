import pandas as pd
from cla_engine.rppa import map_antibodies, align_rppa


def test_exact_mapping_preserves_multigene_targets_and_excludes_unresolved():
    annotation = pd.DataFrame([
        ['1', '4E-BP1', 'ENSG0001.3', 'Valid'],
        ['2', 'AKT_pS473', 'ENSG0002.1/ENSG0003.2', 'Caution'],
        ['3', 'AMBIG', 'ENSG0004', 'Valid'],
        ['4', 'AMBIG', 'ENSG0005', 'Valid'],
        ['5', 'CASPASE_CLEAVED', 'ENSG0006', 'Valid'],
    ], columns=['AGID', 'peptide_target', 'gene_id', 'validation_status'])
    result = map_antibodies(['X4EBP1', 'AKT_pS473', 'AMBIG', 'CASPASECLEAVED', 'UNKNOWN'], annotation).set_index('source_column')
    assert result.loc['X4EBP1', 'gene_ids'] == 'ENSG0001'
    assert result.loc['AKT_pS473', 'gene_ids'] == 'ENSG0002|ENSG0003'
    assert result.loc['AKT_pS473', 'molecular_layer'] == 'PHOSPHORYLATION'
    assert result.loc['AKT_pS473', 'validation_statuses'] == 'Caution'
    assert result.loc['AMBIG', 'mapping_status'] == 'AMBIGUOUS_ANNOTATION'
    assert result.loc['CASPASECLEAVED', 'mapping_status'] == 'OTHER_MODIFICATION_EXCLUDED'
    assert result.loc['UNKNOWN', 'mapping_status'] == 'UNMAPPED_NO_GUESS'
    assert result.index.is_unique and len(result) == 5


def test_pairing_requires_exact_vial_official_quality_and_unique_aliquot():
    patients = ['TCGA-AA-000' + str(i) for i in range(1, 7)]
    authority = pd.DataFrame({'patient_id': patients, 'sample_id': [p + '-01A-RNA' for p in patients],
                              'cancer_id': ['COAD', 'BRCA', 'BRCA', 'BRCA', 'BRCA', 'BRCA']})
    samples = [patients[0] + '-01A-P1', patients[1] + '-01B-P1', patients[2] + '-01A-P1',
               patients[3] + '-01A-P1', patients[4] + '-01A-P1', patients[4] + '-01A-P2', patients[5] + '-01A-P1']
    rppa = pd.DataFrame({'SampleID': samples, 'TumorType': ['CORE'] + ['BRCA'] * 6})
    quality = pd.DataFrame({'aliquot_barcode': [s for i, s in enumerate(samples) if i != 3],
        'Do_not_use': ['False', 'False', 'True', 'False', 'False', 'False'],
        'AWG_excluded_because_of_pathology': ['False'] * 6,
        'cancer type': ['COAD'] + ['BRCA'] * 5})
    aligned = align_rppa(rppa, authority, quality)
    assert aligned.status.tolist() == [
        'MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL', 'RNA_SAMPLE_VIAL_MISMATCH', 'OFFICIAL_SAMPLE_QUALITY_EXCLUDED',
        'NO_OFFICIAL_SAMPLE_QUALITY_RECORD', 'MULTIPLE_MATCHED_RPPA_ALIQUOTS_UNRESOLVED',
        'MULTIPLE_MATCHED_RPPA_ALIQUOTS_UNRESOLVED', 'MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL']
    assert aligned.loc[0, 'cancer_id'] == 'COAD'
    assert aligned.loc[0, 'combined_cancer_resolved_by_exact_QC']
    quality.loc[0, 'cancer type'] = 'READ'
    assert align_rppa(rppa, authority, quality).loc[0, 'status'] == 'CANCER_MISMATCH_OR_UNRESOLVED_COMBINED_LABEL'
