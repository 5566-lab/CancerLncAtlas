"""Exact RPPA feature identities and RNA-authority sample matching."""
import re
import numpy as np
import pandas as pd


def map_antibodies(columns, annotation):
    def normalize(value):
        return re.sub('[^A-Z0-9]', '', re.sub(r'^X(?=\d)', '', value).upper())
    ann = annotation.fillna('').copy()
    ann['join_key'] = ann.peptide_target.map(normalize)
    rows = []
    for feature in columns:
        sub = ann.loc[ann.join_key.eq(normalize(feature))]
        groups = {tuple(sorted(set(re.findall(r'ENSG\d+', x)))) for x in sub.gene_id}
        genes = sorted(set(re.findall(r'ENSG\d+', '/'.join(sub.gene_id))))
        if not len(sub):
            status = 'UNMAPPED_NO_GUESS'
        elif len(groups) != 1 or not genes:
            status = 'AMBIGUOUS_ANNOTATION'
        elif re.search(r'CLEAV|ACETYL', feature, re.I):
            status = 'OTHER_MODIFICATION_EXCLUDED'
        else:
            status = 'ANNOTATED'
        rows.append({
            'feature_id': 'RPPA:' + feature, 'source_column': feature,
            'molecular_layer': 'PHOSPHORYLATION' if re.search(r'_p[STY]\d+', feature, re.I) else 'PROTEIN',
            'mapping_status': status, 'gene_ids': '|'.join(genes), 'gene_count': len(genes),
            'peptide_targets': '|'.join(sorted(set(sub.peptide_target))),
            'AGIDs': '|'.join(sub.AGID), 'validation_statuses': '|'.join(sorted(set(sub.validation_status))),
            'measurement_unit': 'ONE_ANTIBODY_TARGET_NOT_DUPLICATED_PER_GENE',
            'mapping_rule': 'EXACT_ALPHANUMERIC_CASE_NORMALIZATION_WITH_R_NUMERIC_PREFIX_REMOVED',
        })
    return pd.DataFrame(rows)


def align_rppa(rppa, patient_authority, quality):
    if not rppa.SampleID.is_unique or not patient_authority.patient_id.is_unique:
        raise ValueError('RPPA samples and authority patients must be unique')
    qc = quality.fillna('').copy()
    truth = lambda x: x.astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
    qc['excluded'] = truth(qc.Do_not_use) | truth(qc.AWG_excluded_because_of_pathology)
    excluded = qc.groupby('aliquot_barcode').excluded.max()
    cancers = qc.groupby('aliquot_barcode')['cancer type'].agg(lambda x: set(x))
    authority = patient_authority.set_index('patient_id')
    rows = []
    for index, row in rppa.iterrows():
        sample = row.SampleID
        pid, vial = sample[:12], sample[:16]
        expected = authority.loc[pid, 'cancer_id'] if pid in authority.index else None
        resolved = (row.TumorType == 'CORE' and expected in ('COAD', 'READ')
                    and sample in cancers.index and cancers[sample] == {expected})
        found = sample in excluded.index
        bad = bool(excluded[sample]) if found else None
        if pid not in authority.index:
            status = 'NO_FROZEN_RNA_PATIENT'
        elif row.TumorType != expected and not resolved:
            status = 'CANCER_MISMATCH_OR_UNRESOLVED_COMBINED_LABEL'
        elif vial != authority.loc[pid, 'sample_id'][:16]:
            status = 'RNA_SAMPLE_VIAL_MISMATCH'
        elif not found:
            status = 'NO_OFFICIAL_SAMPLE_QUALITY_RECORD'
        elif bad:
            status = 'OFFICIAL_SAMPLE_QUALITY_EXCLUDED'
        else:
            status = 'MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL'
        rows.append({'rppa_row': index, 'rppa_sample_id': sample, 'patient_id': pid,
                     'cancer_id': expected if resolved else row.TumorType, 'sample_vial': vial,
                     'quality_annotation_found': found, 'quality_excluded': bad,
                     'source_cancer_id': row.TumorType,
                     'combined_cancer_resolved_by_exact_QC': resolved, 'status': status})
    frame = pd.DataFrame(rows)
    ok = frame.status.eq('MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL')
    duplicates = frame.loc[ok, 'patient_id'].duplicated(keep=False)
    duplicate_ids = set(frame.loc[ok].loc[duplicates, 'patient_id'])
    frame.loc[ok & frame.patient_id.isin(duplicate_ids), 'status'] = 'MULTIPLE_MATCHED_RPPA_ALIQUOTS_UNRESOLVED'
    return frame
