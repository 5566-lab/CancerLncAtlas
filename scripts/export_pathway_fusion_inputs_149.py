"""Add measured RPPA protein/phosphorylation to verified gene-level assay inputs."""
import argparse
import json
import shutil
import socket
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.common import atomic_json, read_json, sha
from cla_engine.cancer_scope import resolve_cancer_scope
from cla_engine.rppa import map_antibodies, align_rppa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gene-inputs', required=True, help='Directory produced by export_pathway_gene_inputs_149.py')
    parser.add_argument('--output', required=True)
    for key in ('rppa', 'antibody-annotation', 'sample-quality', 'patient-map'):
        parser.add_argument('--' + key, required=True)
        parser.add_argument('--' + key + '-sha256', required=True)
    parser.add_argument('--cancer-type', '--cancer_type', choices=['all', 'list'], default='all')
    parser.add_argument('--cancer-list', '--cancer_list', nargs='+', default=[])
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != '149':
        raise RuntimeError('Scientific materialization must execute on host 149')
    source_bindings = {}
    for key in ('rppa', 'antibody_annotation', 'sample_quality', 'patient_map'):
        path = Path(getattr(args, key)).resolve()
        expected = getattr(args, key + '_sha256')
        if len(expected) != 64 or sha(path) != expected:
            raise ValueError(key + '_HASH_MISMATCH')
        source_bindings[key] = {'path': str(path), 'sha256': expected}
    base_root = Path(args.gene_inputs).resolve()
    base = read_json(base_root / 'COMPLETE.json')
    if base['target_host'] != '149' or base['status'] != 'COMPLETE':
        raise ValueError('Incomplete gene assay inputs')
    catalog = read_json(base_root / 'CATALOG.json')
    if sha(base_root / 'CATALOG.json') != base['catalog_sha256']:
        raise ValueError('Gene catalog hash mismatch')
    fm = pd.read_csv(args.patient_map, sep='\t')
    scope = resolve_cancer_scope({'cancer_type': args.cancer_type, 'cancer_list': args.cancer_list}, fm.cancer_id.unique())
    selected = scope['selected_cancers']
    parts = {p['cancer_id']: p for p in base['partitions']}
    if set(selected) - set(parts):
        raise ValueError('Requested cancers missing from gene assay input')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError('Use a new output directory')
    atomic_json(out / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'operation': 'MATERIALIZE_VERIFIED_FIVE_ASSAY_INPUTS',
                'cancer_type': args.cancer_type, 'source_bindings': source_bindings})
    rppa = pd.read_csv(args.rppa, sep='\t')
    if list(rppa.columns[:2]) != ['SampleID', 'TumorType'] or np.isinf(rppa.iloc[:, 2:].to_numpy(float)).any():
        raise ValueError('RPPA schema or values invalid')
    mapping = map_antibodies(rppa.columns[2:], pd.read_csv(args.antibody_annotation, sep='\t', dtype=str))
    mapping.to_csv(out / 'RPPA_FEATURE_MAPPING.tsv', sep='\t', index=False)
    alignment = align_rppa(rppa, fm, pd.read_csv(args.sample_quality, sep='\t', dtype=str))
    alignment.to_csv(out / 'RPPA_PATIENT_ALIGNMENT.tsv', sep='\t', index=False)
    used = alignment.loc[alignment.status.eq('MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL')].set_index('patient_id')
    mapped = mapping.loc[mapping.mapping_status.eq('ANNOTATED')]
    catalog.update({
        'assay_feature_genes': {layer: dict(zip(f.feature_id, f.gene_ids.str.split('|'))) for layer, f in mapped.groupby('molecular_layer')},
        'protein_annotation_policy': 'ONE_TARGET_ONE_FEATURE;MULTIGENE_TARGET_ANNOTATES_PATHWAY_ONLY_IF_ALL_POSSIBLE_GENES_ARE_MEMBERS',
        'assay_measurement_types': {'RNA': 'protein_coding_gene_logCPM', 'CNV': 'gene_midpoint_signed_log2_copy_ratio',
            'METHYLATION': 'gene_promoter_beta_TSS_minus1000_plus100', 'PROTEIN': 'targeted_RPPA_protein_abundance',
            'PHOSPHORYLATION': 'targeted_RPPA_phosphorylation_abundance_NOT_total_protein_adjusted'},
        'rppa_feature_mapping': {'path': str(out / 'RPPA_FEATURE_MAPPING.tsv'), 'sha256': sha(out / 'RPPA_FEATURE_MAPPING.tsv')},
        'rppa_sources': source_bindings,
    })
    atomic_json(out / 'CATALOG.json', catalog)
    records = []
    for cancer in selected:
        part = parts[cancer]
        meta_path = Path(part['path'])
        if sha(meta_path) != part['sha256']:
            raise ValueError('Gene partition hash mismatch')
        m = read_json(meta_path)
        ids = m['patient_ids']
        auth = fm.loc[fm.cancer_id.eq(cancer)].sort_values('patient_id')
        if (ids != auth.patient_id.tolist() or m['patient_fold_ids'] != auth.patient_fold_id.tolist()
                or m['patient_map_sha256'] != source_bindings['patient_map']['sha256']):
            raise ValueError('Gene assay patient authority mismatch')
        dest = out / cancer
        dest.mkdir()
        blocks = {}
        paired = np.ones(len(ids), bool)
        for layer in ('RNA', 'CNV', 'METHYLATION'):
            b = dict(m['blocks'][layer])
            source = meta_path.parent / b['file']
            if not source.resolve().is_relative_to(meta_path.parent.resolve()) or sha(source) != b['sha256']:
                raise ValueError('Gene matrix hash or path mismatch')
            shutil.copyfile(source, dest / b['file'])
            if sha(dest / b['file']) != b['sha256']:
                raise ValueError('Copied gene matrix hash mismatch')
            x = np.memmap(dest / b['file'], mode='r', dtype='<f8', shape=tuple(b['shape']), order='F')
            present = np.isfinite(x).any(axis=0)
            paired &= present
            if 'genes' in b:
                b['feature_ids'] = b.pop('genes')
            b['patients_with_any_observed_feature'] = int(present.sum())
            blocks[layer] = b
        for layer in ('PROTEIN', 'PHOSPHORYLATION'):
            f = mapped.loc[mapped.molecular_layer.eq(layer)]
            if f.empty:
                raise ValueError('Required RPPA layer has no verified annotations: ' + layer)
            x = np.full((len(f), len(ids)), np.nan)
            source_samples = {}
            for i, pid in enumerate(ids):
                if pid not in used.index:
                    continue
                row = int(used.loc[pid, 'rppa_row'])
                if rppa.loc[row, 'SampleID'] != used.loc[pid, 'rppa_sample_id']:
                    raise ValueError('RPPA row alignment drift')
                x[:, i] = rppa.loc[row, f.source_column.tolist()].to_numpy(float)
                source_samples[pid] = rppa.loc[row, 'SampleID']
            path = dest / (layer + '.f64')
            np.ascontiguousarray(x.T, dtype='<f8').tofile(path)
            present = np.isfinite(x).any(axis=0)
            paired &= present
            blocks[layer] = {'file': path.name, 'sha256': sha(path), 'feature_ids': f.feature_id.tolist(),
                'shape': list(x.shape), 'dtype': 'little_endian_float64', 'orientation': 'R_column_major_feature_by_patient',
                'patients_with_any_observed_feature': int(present.sum()), 'measured_values': int(np.isfinite(x).sum()),
                'source_sample_ids': source_samples, 'source': source_bindings['rppa']}
        record = {'target_host': '149', 'status': 'COMPLETE', 'format': 'CLA_FIVE_ASSAY_RAW_INPUTS_V1',
            'cancer_id': cancer, 'patient_ids': ids, 'patient_fold_ids': m['patient_fold_ids'], 'blocks': blocks,
            'catalog_sha256': sha(out / 'CATALOG.json'), 'patient_map_sha256': m['patient_map_sha256'],
            'raw_RNA': m['raw_RNA'], 'raw_omics_source_artifacts': m['raw_omics_source_artifacts'],
            'raw_omics_complete': m['raw_omics_complete'], 'five_assay_paired_patients': int(paired.sum()),
            'base_raw_partition': {'path': str(meta_path), 'sha256': part['sha256']},
            'rppa_alignment_sha256': sha(out / 'RPPA_PATIENT_ALIGNMENT.tsv'),
            'rppa_annotation_sha256': sha(out / 'RPPA_FEATURE_MAPPING.tsv'),
            'missingness_policy': 'NAN_PRESERVED_NO_NEW_RAW_IMPUTATION', 'presence_is_not_a_final_score_estimability_gate': True}
        atomic_json(dest / 'COMPLETE.json', record)
        records.append({'cancer_id': cancer, 'path': str(dest / 'COMPLETE.json'), 'sha256': sha(dest / 'COMPLETE.json'),
                        'patients': len(ids), 'paired_patients': int(paired.sum())})
        print('FIVE_ASSAY_EXPORTED', cancer, len(ids), int(paired.sum()), flush=True)
    atomic_json(out / 'COMPLETE.json', {'target_host': '149', 'status': 'COMPLETE', 'format': 'CLA_FIVE_ASSAY_RAW_INPUTS_V1',
        'cancer_type': args.cancer_type, 'cancers': selected, 'patients': sum(r['patients'] for r in records),
        'paired_patients': sum(r['paired_patients'] for r in records),
        'assays': ['RNA', 'CNV', 'METHYLATION', 'PROTEIN', 'PHOSPHORYLATION'],
        'catalog_sha256': sha(out / 'CATALOG.json'), 'partitions': records, 'source_bindings': source_bindings,
        'rppa_mapping_status_counts': mapping.mapping_status.value_counts().to_dict(),
        'rppa_alignment_status_counts': alignment.status.value_counts().to_dict()})


if __name__ == '__main__':
    main()
