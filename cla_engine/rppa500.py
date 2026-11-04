"""Version-bound RPPA500 measurements; no pathway scoring or learned imputation."""
from __future__ import annotations
import gzip
import json
import re
import socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, read_json, sha, fail
from .cancer_scope import resolve_cancer_scope

FORMAT = 'CLA_RPPA500_PATIENT_INPUT_V1'
PANEL = 'TCPA_RPPA500'
LAYERS = ('PROTEIN', 'PHOSPHORYLATION', 'PTM_OTHER', 'OTHER_MEASUREMENT')


def marker_key(value):
    return re.sub('[^A-Z0-9]', '', re.sub(r'^X(?=\d)', '', str(value).upper()))


def bound_path(spec):
    p = Path(spec['path']).resolve()
    if not p.is_file() or len(spec.get('sha256', '')) != 64 or sha(p) != spec['sha256']:
        fail('RPPA500_SOURCE_HASH', 'Missing or changed pinned source', path=str(p))
    return p


def gtf_gene_names(path):
    names = {}
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] != 'gene':
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
            if 'gene_id' in attrs and 'gene_name' in attrs:
                gid = re.sub(r'\.\d+(?=_PAR_Y$|$)', '', attrs['gene_id'])
                names.setdefault(attrs['gene_name'], set()).add(gid)
    return names


def feature_catalog(columns, portal, paper, gdc, names):
    """Exact labels only. A multi-gene/multi-site antibody stays one measurement."""
    p = pd.DataFrame(portal['antibodies']).fillna('')
    if p.protein_label.duplicated().any():
        fail('RPPA500_ANNOTATION_DUPLICATE', 'Duplicate portal antibody labels')
    p = p.set_index('protein_label')
    paper = paper.fillna('').copy()
    paper['key'] = paper.Protein.map(marker_key)
    gdc = gdc.fillna('').copy()
    gdc['key'] = gdc.peptide_target.map(marker_key)
    if paper.key.duplicated().any():
        fail('RPPA500_ANNOTATION_DUPLICATE', 'Ambiguous supplement marker')
    paper = paper.set_index('key')
    known_ids = set().union(*names.values()) if names else set()
    rows = []
    for column in columns:
        key = marker_key(column)
        pp = p.loc[column].to_dict() if column in p.index else {}
        ss = paper.loc[key].to_dict() if key in paper.index else {}
        gg = gdc.loc[gdc.key.eq(key)]
        if not pp and not ss and gg.empty:
            fail('RPPA500_UNKNOWN_MARKER', 'No exact official annotation', marker=column)
        groups = {tuple(sorted(set(re.findall(r'ENSG\d+(?:\.\d+)?(?:_PAR_Y)?', x))))
                  for x in gg.gene_id.astype(str)}
        raw_ids = next(iter(groups)) if len(groups) == 1 else ()
        ids = sorted({re.sub(r'\.\d+(?=_PAR_Y$|$)', '', x) for x in raw_ids})
        gene_source = 'GDC_EXACT_ANTIBODY' if ids else 'UNRESOLVED'
        gene_symbols = str(ss.get('Gene_symbol', '')).strip()
        if not ids and gg.empty and gene_symbols:
            symbols = [v.strip() for v in gene_symbols.split(',')]
            if all(len(names.get(v, set())) == 1 for v in symbols):
                ids = sorted(set().union(*(names[v] for v in symbols)))
                gene_source = 'PAPER_EXACT_MARKER_AND_GENCODE_EXACT_SYMBOL'
        if ids and not set(ids) <= known_ids:
            fail('RPPA500_GENE_REFERENCE', 'Antibody gene ID absent from bound GENCODE', marker=column)
        reported_type = ss.get('Type', '')
        gdc_name = '|'.join(gg.peptide_target.astype(str))
        # Do not interpret composite/cleavage-labelled antibodies as a clean phosphosite.
        if key == 'MITOCHONDRIA':
            layer = 'OTHER_MEASUREMENT'
        elif 'CLEAVED' in key:
            layer = 'PTM_OTHER'
        elif (reported_type == 'ptm' and re.search(r'P[STY]\d+', key)) or (
                not reported_type and re.search(r'_p[STY]\d+', gdc_name, re.I)):
            layer = 'PHOSPHORYLATION'
        elif reported_type == 'ptm' or key == 'PAR':
            layer = 'PTM_OTHER'
        else:
            layer = 'PROTEIN'
        qc_source = 'TCPA_PORTAL' if pp else 'GDC_ANNOTATION'
        quality = str(pp.get('validation_status', '')) or '|'.join(sorted(set(gg.validation_status)))
        rows.append({
            'feature_id': 'RPPA500:' + column, 'source_column': column,
            'protein_name': pp.get('protein_name', '') or gdc_name or column,
            'molecular_layer': layer, 'paper_type': reported_type,
            'portal_annotated': bool(pp), 'paper_annotated': bool(ss),
            'gdc_annotated': not gg.empty, 'annotation_panel_mismatch': not bool(pp and ss),
            'gene_ids': '|'.join(ids), 'gene_count': len(ids),
            'gene_mapping_source': gene_source,
            'gene_mapping_status': 'RESOLVED' if ids else 'NO_GENE_EDGE_UNRESOLVED_OR_NON_GENE_TARGET',
            'paper_gene_symbols': gene_symbols, 'portal_gene_symbols': pp.get('genes', ''),
            'gdc_gene_symbols': '|'.join(sorted(set(gg.gene_name))),
            'validation_status': quality, 'validation_status_source': qc_source,
            'portal_flag': pp.get('flag', ''),  # Preserved; not assumed to be a QC exclusion.
            'company': pp.get('source', '') or ss.get('Company', '') or '|'.join(sorted(set(gg.source))),
            'catalog_number': str(pp.get('catalog_number', '') or ss.get('Catalog_Number', '') or '|'.join(sorted(set(gg.catalog_number.astype(str))))),
            'reported_phosphosite_labels': '|'.join(re.findall(r'[STY]\d+', key)) if layer == 'PHOSPHORYLATION' else '',
            'site_resolution': 'ANTIBODY_LABEL_ONLY_NOT_ISOFORM_SEQUENCE_VALIDATED',
            'measurement_unit': 'ONE_ANTIBODY_TARGET_NOT_DUPLICATED_PER_GENE_OR_SITE',
        })
    return pd.DataFrame(rows)


def align_samples(data, authority, quality=None):
    required = {'patient_id', 'sample_id', 'cancer_id', 'patient_fold_id'}
    if not required <= set(authority) or not authority.patient_id.is_unique:
        fail('RPPA500_PATIENT_AUTHORITY', 'Unique frozen patients and folds required')
    samples = data.sample_id.astype(str)
    if not samples.str.fullmatch(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}[A-Z]').all():
        fail('RPPA500_SAMPLE_ID', 'Expected exact TCGA sample-vial identifiers')
    if not samples.is_unique:
        fail('RPPA500_SAMPLE_DUPLICATE', 'Duplicate source sample-vial identifiers')
    ref = authority.set_index('patient_id')
    ids = samples.str[:12]
    repeated = set(ids[ids.duplicated(keep=False)])
    prior = {}
    if quality is not None:
        required_qc = {'platform', 'aliquot_barcode', 'Do_not_use', 'AWG_excluded_because_of_pathology'}
        if not required_qc <= set(quality):
            fail('RPPA500_QUALITY_SCHEMA', 'Bound legacy quality columns missing')
        q = quality.fillna('').copy()
        q = q.loc[q.platform.astype(str).str.contains('RPPA', case=False)].copy()
        bad = np.zeros(len(q), dtype=bool)
        for col in ['Do_not_use', 'AWG_excluded_because_of_pathology']:
            flag = q[col].astype(str).str.strip().str.lower()
            if not flag.isin(['', 'true', 'false', '0', '1', '0.0', '1.0']).all():
                fail('RPPA500_QUALITY_FLAG', 'Unknown official exclusion flag')
            bad |= flag.isin(['true', '1', '1.0']).to_numpy()
        q['bad'] = bad
        q['vial'] = q.aliquot_barcode.astype(str).str[:16]
        for vial, group in q.groupby('vial'):
            prior[vial] = {'excluded': bool(group.bad.any()), 'aliquots': len(group),
                           'mixed': bool(group.bad.any() and not group.bad.all())}
    records = []
    for index, sample in enumerate(samples):
        pid = sample[:12]
        if pid not in ref.index:
            status = 'NO_FROZEN_RNA_PATIENT'
        elif sample[13:15] not in {f'{i:02}' for i in range(1, 10)}:
            status = 'NON_TUMOR_SAMPLE'
        elif sample != str(ref.loc[pid, 'sample_id'])[:16]:
            status = 'RNA_SAMPLE_VIAL_MISMATCH'
        elif prior.get(sample, {}).get('excluded', False):
            status = 'PRIOR_OFFICIAL_RPPA_SAMPLE_VIAL_EXCLUDED'
        elif pid in repeated:
            status = 'MULTIPLE_PATIENT_SAMPLES_UNRESOLVED'
        else:
            status = 'MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL'
        records.append({'rppa_row': index, 'rppa_sample_id': sample, 'patient_id': pid,
                        'cancer_id': ref.loc[pid, 'cancer_id'] if pid in ref.index else '',
                        'status': status, 'sample_quality_basis': 'PUBLISHED_FINAL_RPPA500_RELEASE',
                        'prior_rppa_qc_found': sample in prior,
                        'prior_rppa_qc_excluded': prior.get(sample, {}).get('excluded', False),
                        'prior_rppa_qc_aliquots': prior.get(sample, {}).get('aliquots', 0),
                        'prior_rppa_qc_mixed': prior.get(sample, {}).get('mixed', False),
                        'aliquot_match_claimed': False})
    return pd.DataFrame(records)


def materialize(spec, output, cancer_type='all', cancer_list=None):
    if socket.gethostname().split('.')[0] != '149':
        fail('RPPA500_HOST', 'Scientific materialization must execute on host 149')
    if spec.get('panel') != PANEL or spec.get('format') != 'CLA_RPPA500_SOURCE_SPEC_V1':
        fail('RPPA500_SOURCE_VERSION', 'Use the pinned RPPA500 source specification')
    # Verify before producing any derived measurements.
    sources = {key: {'path': str(bound_path(value)), 'sha256': value['sha256']}
               for key, value in spec['sources'].items()}
    if spec.get('gene_assays_unchanged'):
        bound_path(spec['gene_assays_unchanged'])
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        fail('RPPA500_OUTPUT_EXISTS', 'Use a new versioned output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'operation': 'RPPA500_MEASUREMENT_REPLACEMENT',
                                            'sources': sources, 'training': False})
    data = pd.read_csv(sources['matrix']['path'], sep='\t')
    if data.columns[0] != 'sample_id' or data.columns.duplicated().any():
        fail('RPPA500_SCHEMA', 'Expected sample_id and unique assay columns')
    numeric = data.iloc[:, 1:].to_numpy(float)
    if np.isinf(numeric).any():
        fail('RPPA500_NONFINITE', 'Infinity is not a valid assay measurement')
    if list(data.shape) != [7828, 448] or list(data.shape) != spec['expected_matrix_shape']:
        fail('RPPA500_PANEL_DRIFT', 'Source matrix dimensions differ from pinned release')
    portal = read_json(sources['portal_annotation']['path'])
    paper = pd.read_excel(sources['paper_supplement']['path'], sheet_name='Table S3', header=1)
    gdc = pd.read_csv(sources['gdc_annotation']['path'], sep='\t', dtype=str)
    names = gtf_gene_names(sources['gtf']['path'])
    catalog = feature_catalog(data.columns[1:], portal, paper, gdc, names)
    mismatches = sorted(catalog.loc[catalog.annotation_panel_mismatch, 'source_column'])
    if mismatches != sorted(spec['expected_annotation_panel_mismatch']):
        fail('RPPA500_ANNOTATION_DRIFT', 'Unexpected matrix/annotation mismatch')
    authority = pd.read_csv(sources['patient_map']['path'], sep='\t')
    scope = resolve_cancer_scope({'cancer_type': cancer_type, 'cancer_list': cancer_list or []},
                                authority.cancer_id.unique())
    quality = pd.read_csv(sources['prior_sample_quality']['path'], sep='\t', dtype=str)
    alignment = align_samples(data, authority, quality)
    authority = authority.loc[authority.cancer_id.isin(scope['selected_cancers'])].sort_values(['cancer_id', 'patient_id']).reset_index(drop=True)
    matches = alignment.loc[alignment.status.eq('MATCHED_EXACT_PATIENT_AND_SAMPLE_VIAL')].set_index('patient_id')
    aligned = np.full((len(authority), len(catalog)), np.nan, dtype=np.float64)
    has = authority.patient_id.isin(matches.index).to_numpy()
    source_rows = matches.loc[authority.loc[has, 'patient_id'], 'rppa_row'].to_numpy(int)
    aligned[has] = numeric[source_rows]
    catalog.to_csv(out/'FEATURE_CATALOG.tsv', sep='\t', index=False)
    alignment.to_csv(out/'PATIENT_ALIGNMENT.tsv', sep='\t', index=False)
    authority.to_parquet(out/'PATIENT_AUTHORITY.parquet', index=False)
    coverage = catalog[['feature_id', 'molecular_layer']].copy()
    coverage['observed_patients'] = np.isfinite(aligned).sum(axis=0)
    coverage['patient_fraction'] = coverage.observed_patients / len(authority)
    coverage.to_csv(out/'FEATURE_COVERAGE.tsv', sep='\t', index=False)
    layers = {}
    cancer_rows = []
    for layer in LAYERS:
        take = catalog.molecular_layer.eq(layer).to_numpy()
        cols = catalog.loc[take, 'feature_id'].tolist()
        values = pd.DataFrame(aligned[:, take], columns=cols)
        mask = np.isfinite(values.to_numpy())
        values.insert(0, 'patient_id', authority.patient_id.to_numpy())
        available = pd.DataFrame(mask, columns=cols)
        available.insert(0, 'patient_id', authority.patient_id.to_numpy())
        values.to_parquet(out/(layer+'.parquet'), index=False)
        available.to_parquet(out/(layer+'_AVAILABLE.parquet'), index=False)
        layers[layer] = {'features': cols, 'dimensions': len(cols),
                         'features_with_any_measurement': int(mask.any(axis=0).sum()),
                         'patients_with_any_measurement': int(mask.any(axis=1).sum()),
                         'measured_cells': int(mask.sum())}
        for cancer, positions in authority.groupby('cancer_id').groups.items():
            block = mask[np.asarray(list(positions))]
            cancer_rows.append({'cancer_id': cancer, 'layer': layer, 'patients': len(positions),
                                'patients_observed': int(block.any(axis=1).sum()),
                                'features_observed': int(block.any(axis=0).sum())})
    pd.DataFrame(cancer_rows).to_csv(out/'CANCER_COVERAGE.tsv', sep='\t', index=False)
    atomic_json(out/'SOURCE_SPEC.json', spec)
    producer = out/'RPPA500_PRODUCER.py.txt'
    producer.write_bytes(Path(__file__).read_bytes())
    report = {'format': FORMAT, 'panel': PANEL, 'target_host': '149', 'status': 'COMPLETE',
              'sources': sources, 'patients': len(authority), 'cancers': scope['selected_cancers'],
              'cancer_scope': scope, 'source_samples': len(data), 'source_features': len(catalog),
              'features_with_any_measurement': int(np.isfinite(aligned).any(axis=0).sum()),
              'producer': {'file': producer.name, 'sha256': sha(producer)},
              'layers': layers, 'alignment_status_counts': alignment.status.value_counts().to_dict(),
              'matched_patients_in_scope': int(has.sum()), 'annotation_mismatch_features': mismatches,
              'gene_mapping_status_counts': catalog.gene_mapping_status.value_counts().to_dict(),
              'raw_missing_cells': int(np.isnan(numeric).sum()),
              'missingness_policy': 'SOURCE_NAN_PRESERVED_NO_IMPUTATION',
              'patient_match': 'EXACT_FROZEN_RNA_PATIENT_AND_SAMPLE_VIAL_NOT_ALIQUOT',
              'normalization': 'PRESERVE_OFFICIAL_RELEASE_VALUES_NO_RESCALE',
              'protein_adjusted_phosphorylation': False, 'scores_computed': False,
              'model_trained': False, 'MOGSA': 'RETIRED',
              'quality_policy': 'PUBLISHED_RELEASE_QC_PLUS_PRESERVED_PRIOR_RPPA_DO_NOT_USE_OR_PATHOLOGY_EXCLUSIONS_AT_EXACT_SAMPLE_VIAL;MIXED_ALIQUOT_FLAGS_EXCLUDED;ANTIBODY_STATUS_PRESERVED',
              'annotation_policy': 'EXACT_TCPA_AND_PAPER_MARKERS;GDC_EXACT_ANTIBODY_GENE_BRIDGE;NO_FUZZY_MATCH',
              'gene_assays_unchanged': spec.get('gene_assays_unchanged')}
    report['files'] = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size}
                       for p in out.iterdir() if p.is_file()}
    atomic_json(out/'COMPLETE.json', report)
    return report


def validate_binding(binding, patient_map_sha256=None):
    if binding is None:
        return None
    if binding.get('panel') != PANEL or binding.get('format') != FORMAT:
        fail('RPPA500_BINDING_VERSION', 'Old RPPA panels cannot satisfy the current input binding')
    complete = bound_path(binding['complete'])
    root = complete.parent
    report = read_json(complete)
    if (report.get('format') != FORMAT or report.get('panel') != PANEL or
            report.get('status') != 'COMPLETE' or report.get('target_host') != '149'):
        fail('RPPA500_PRODUCT_VERSION', 'Expected a completed RPPA500 product from host 149')
    if patient_map_sha256 and report['sources']['patient_map']['sha256'] != patient_map_sha256:
        fail('RPPA500_PATIENT_AUTHORITY', 'Frozen RNA patient authority differs')
    if report.get('gene_assays_unchanged'):
        bound_path(report['gene_assays_unchanged'])
    for role, source in report['sources'].items():
        bound_path(source)
    for name, rec in report['files'].items():
        p = (root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.stat().st_size != rec['bytes'] or sha(p) != rec['sha256']:
            fail('RPPA500_PRODUCT_HASH', 'Changed or missing product', file=name)
    if not report.get('producer') or report['producer'].get('sha256') != sha(root/report['producer']['file']):
        fail('RPPA500_PRODUCER_HASH', 'Producer snapshot changed')
    authority = pd.read_parquet(root/'PATIENT_AUTHORITY.parquet')
    if not authority.patient_id.is_unique or len(authority) != report['patients']:
        fail('RPPA500_PRODUCT_AXIS', 'Invalid patient axis')
    catalog = pd.read_csv(root/'FEATURE_CATALOG.tsv', sep='\t', keep_default_na=False)
    if not catalog.feature_id.is_unique or len(catalog) != report['source_features']:
        fail('RPPA500_PRODUCT_AXIS', 'Invalid feature axis')
    for layer in LAYERS:
        values = pd.read_parquet(root/(layer+'.parquet'))
        available = pd.read_parquet(root/(layer+'_AVAILABLE.parquet'))
        expected = ['patient_id'] + report['layers'][layer]['features']
        if (list(values) != expected or list(available) != expected or
                values.patient_id.tolist() != authority.patient_id.tolist() or
                available.patient_id.tolist() != authority.patient_id.tolist()):
            fail('RPPA500_PRODUCT_AXIS', 'Measurement axes differ', layer=layer)
        x = values.iloc[:, 1:].to_numpy(float)
        if np.isinf(x).any() or not np.array_equal(np.isfinite(x), available.iloc[:, 1:].to_numpy(bool)):
            fail('RPPA500_MISSINGNESS', 'Availability must equal observed finite measurements')
        if int(np.isfinite(x).any(axis=0).sum()) != report['layers'][layer]['features_with_any_measurement']:
            fail('RPPA500_COVERAGE', 'Measured feature count differs')
        if int(np.isfinite(x).sum()) != report['layers'][layer]['measured_cells']:
            fail('RPPA500_COVERAGE', 'Coverage receipt differs from measured cells')
    return {'status': 'VERIFIED_RPPA500_MEASUREMENTS_NOT_TRAINED', 'panel': PANEL,
            'complete': {'path': str(complete), 'sha256': sha(complete)},
            'patients': report['patients'], 'matched_patients': report['matched_patients_in_scope'],
            'layers': report['layers'], 'used_by_existing_HHGT_heads': False}


def cli(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-spec')
    parser.add_argument('--output')
    parser.add_argument('--validate-binding')
    parser.add_argument('--cancer-type', '--cancer_type', choices=['all', 'list'], default='all')
    parser.add_argument('--cancer-list', '--cancer_list', nargs='+', default=[])
    args = parser.parse_args(argv)
    if socket.gethostname().split('.')[0] != '149':
        fail('RPPA500_HOST', 'Run on host 149')
    if args.validate_binding:
        print(json.dumps(validate_binding(read_json(args.validate_binding)), indent=2))
    else:
        if not args.source_spec or not args.output:
            parser.error('--source-spec and --output are required for materialization')
        result = materialize(read_json(args.source_spec), args.output, args.cancer_type, args.cancer_list)
        validate_binding({'format': FORMAT, 'panel': PANEL, 'complete': {'path': str(Path(args.output)/'COMPLETE.json'), 'sha256': sha(Path(args.output)/'COMPLETE.json')}})
        print(json.dumps({k: result[k] for k in ['format', 'patients', 'matched_patients_in_scope', 'source_features', 'alignment_status_counts']}, indent=2))
