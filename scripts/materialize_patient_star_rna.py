"""Refresh the patient RNA inputs from GDC STAR Counts on the frozen v36 scoring background."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cla_engine.common import atomic_json, sha
from cache_m0_joint_scores import cache_case

FUNCTIONAL = Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
JOINT = Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1')
CPTAC = ('BRCA', 'CCRCC', 'COAD', 'GBM', 'HNSCC', 'LSCC', 'LUAD', 'OV', 'PDAC', 'UCEC')
AXES = ('COMMON_GENE_ANNOTATION.tsv', 'PROTEIN_CODING_SYMBOLS.tsv')


def read(path):
    return json.loads(path.read_text())


def check(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Checksum mismatch: {path}')


def verified_outputs(directory):
    receipt = read(directory / 'COMPLETE.json')
    for name, digest in receipt['files'].items():
        check(directory / name, digest)
    return receipt


def copy_file(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def tcga_case(cohort, *, records, before, baseline, output):
    case = 'TCGA_' + cohort
    expected = before['records'][case]
    exported = records[cohort]
    receipt = exported['receipt']
    if receipt['assay'] != 'tpm_unstrand':
        raise ValueError(f'{case}: expected GDC unstranded TPM')
    root = Path(exported['path'])
    check(root / 'GENE_ANNOTATION.tsv', receipt['outputs']['annotation_sha256'])
    check(root / 'GENE_TPM.tsv.gz', receipt['outputs']['tpm_sha256'])
    for part in ('aligned', 'cache'):
        src = baseline / part / case
        dest = output / part / case
        check(src / 'COMPLETE.json', expected[part + '_sha256'])
        meta = verified_outputs(src)
        if part == 'aligned':
            if (meta['source']['sha256'] != receipt['outputs']['tpm_sha256']
                    or meta['patients'] != receipt['patients']
                    or meta['full_lncrnas'] != receipt['source_lncRNAs']):
                raise ValueError(f'{case}: source or patient axis differs from the GDC export')
        elif meta['input_complete_sha256'] != expected['aligned_sha256']:
            raise ValueError(f'{case}: cached scores bound to a different expression release')
        if dest.exists():
            check(dest / 'COMPLETE.json', expected[part + '_sha256'])
            verified_outputs(dest)
        else:
            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                tree = Path(tmp) / 'verified'
                shutil.copytree(src, tree, copy_function=copy_file)
                tree.rename(dest)
    return case, dict(expected, quantification='GDC STAR Counts / GENCODE v36 / tpm_unstranded')


def write_views(matrix, patients, annotation, lnc_ids, dest):
    """Read only GDC values. A missing specimen produces a missing row in every RNA view."""
    if not matrix.index.is_unique or not matrix.columns.is_unique or not patients.is_unique:
        raise ValueError('Duplicated RNA gene or specimen identity')
    if not annotation.index.isin(matrix.index).all() or not lnc_ids.isin(matrix.index).all():
        raise ValueError('GDC file does not cover the expected v36 gene axis')
    tumor = matrix.loc[:, matrix.columns.str.endswith('::T')]
    tumor.columns = tumor.columns.str.removesuffix('::T')
    if not tumor.columns.isin(patients).all():
        raise ValueError('Unexpected tumor specimen outside the project patient axis')
    values = tumor.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid observed GDC log2(TPM+1)')
    tumor = tumor.reindex(columns=patients)
    lnc = tumor.loc[lnc_ids].T
    lnc.columns = 'LNC:' + lnc.columns
    genes = tumor.loc[annotation.index].T
    symbols = annotation.reference_symbol.replace('', np.nan).dropna()
    # Several v36 IDs can share a symbol. Sum on the linear TPM scale.
    linear = np.expm1(genes.loc[:, symbols.index].T * np.log(2))
    linear.index = symbols.to_numpy()
    by_symbol = np.log2(1 + linear.groupby(level=0, sort=True).sum(min_count=1)).T
    for name, frame in [('LNC_LOGTPM_FULL', lnc), ('GENE_LOGTPM', genes), ('SYMBOL_LOGTPM', by_symbol)]:
        frame.index.name = 'patient_id'
        frame.to_parquet(dest / (name + '.parquet'), compression='zstd')
    return int(tumor.isna().all(axis=0).sum())


def cptac_case(cohort, *, records, baseline, output, aligned, sources, old_records):
    case = 'CPTAC_' + cohort
    matrix_path = Path('matrices', cohort, 'rna_TumorNormal.parquet')
    digest = sources['files'][matrix_path.as_posix()]['sha256']
    check(aligned / matrix_path, digest)
    axis_source = Path(records['BRCA']['path']) / 'GENE_ANNOTATION.tsv'
    check(axis_source, records['BRCA']['receipt']['outputs']['annotation_sha256'])
    all_genes = pd.read_csv(axis_source, sep='\t', keep_default_na=False).set_index('gene_id')
    lnc_ids = all_genes.index[all_genes.gene_type.eq('lncRNA')]
    old = baseline / 'aligned' / case
    check(old / 'COMPLETE.json', old_records[case]['aligned_sha256'])
    old_meta = read(old / 'COMPLETE.json')
    check(old / 'LNC_LOGTPM_FULL.parquet', old_meta['files']['LNC_LOGTPM_FULL.parquet'])
    patients = pd.read_parquet(old / 'LNC_LOGTPM_FULL.parquet', columns=[]).index
    if patients.name != 'patient_id' or len(patients) != old_meta['patients']:
        raise ValueError(f'{case}: invalid baseline specimen axis')
    annotation = pd.read_csv(output / AXES[0], sep='\t', keep_default_na=False).set_index('gene_id')
    dest = output / 'aligned' / case
    if not dest.exists():
        with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
            work = Path(tmp) / case
            work.mkdir()
            matrix = pd.read_parquet(aligned / matrix_path).set_index('feature_id')
            missing = write_views(matrix, patients, annotation, lnc_ids, work)
            match_file = 'GDC_SAMPLE_MATCH.tsv'
            check(aligned / match_file, sources['files'][match_file]['sha256'])
            matches = pd.read_csv(aligned / match_file, sep='\t', dtype=str, keep_default_na=False)
            sample_support = matches.loc[matches.source_cohort.eq(cohort) & matches.tissue_type.eq('Tumor')]
            if sample_support.patient_id.duplicated().any() or set(sample_support.patient_id) != set(patients):
                raise ValueError(f'{case}: specimen provenance does not cover the patient axis')
            sample_support.set_index('patient_id').loc[patients].to_csv(
                work / 'RNA_SPECIMEN_SUPPORT.tsv', sep='\t', index_label='patient_id')
            shutil.copy2(output / AXES[1], work / AXES[1])
            meta = dict(dataset='CPTAC', cohort=cohort, target_host='149',
                        source=dict(dataset='CPTAC', cohort=cohort, source=str(aligned / matrix_path),
                                    sha256=digest, workflow='STAR - Counts', gene_model='GENCODE v36',
                                    metric='tpm_unstranded'),
                        patients=len(patients), full_lncrnas=len(lnc_ids), common_genes=len(annotation),
                        missing_rna_cells=missing * len(annotation), unmeasured_patients=missing,
                        files={p.name: sha(p) for p in work.iterdir()})
            atomic_json(work / 'COMPLETE.json', meta)
            work.rename(dest)
    data = verified_outputs(dest)
    if data['source']['sha256'] != digest:
        raise ValueError(f'{case}: staging directory belongs to a different GDC input')
    cache = output / 'cache' / case
    if not cache.exists():
        with tempfile.TemporaryDirectory(dir=cache.parent) as tmp:
            cache_case(output / 'aligned', case, tmp, JOINT / 'repo/config/cptac_149_sources_20261104.json',
                       Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/sources'),
                       JOINT / 'XCELL_FROZEN_REFERENCE.json',
                       Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/CURRENT_149_CONFIG.json'))
            made = Path(tmp) / case
            missing_ids = pd.read_parquet(dest / 'LNC_LOGTPM_FULL.parquet', columns=['LNC:' + lnc_ids[0]])
            missing_ids = missing_ids.index[missing_ids.iloc[:, 0].isna()]
            if len(missing_ids):
                # Empty measured views cannot contribute a finite pathway score, including clipped scores.
                creceipt = read(made / 'COMPLETE.json')
                for score_file in made.glob('*_RAW.parquet'):
                    scores = pd.read_parquet(score_file)
                    if scores.loc[missing_ids].notna().any().any():
                        scores.loc[missing_ids] = np.nan
                        scores.to_parquet(score_file, compression='zstd')
                        creceipt['files'][score_file.name] = sha(score_file)
                atomic_json(made / 'COMPLETE.json', creceipt)
            made.rename(cache)
    cache_meta = verified_outputs(cache)
    if cache_meta['input_complete_sha256'] != sha(dest / 'COMPLETE.json'):
        raise ValueError(f'{case}: scores are stale')
    row = {k: data[k] for k in ('dataset','cohort','patients','full_lncrnas','common_genes','missing_rna_cells','unmeasured_patients')}
    row.update(aligned_sha256=sha(dest / 'COMPLETE.json'), cache_sha256=sha(cache / 'COMPLETE.json'))
    return case, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=FUNCTIONAL / 'common_rna_scores_r1')
    parser.add_argument('--tcga', type=Path, default=FUNCTIONAL / 'full_tcga_tpm_r1')
    parser.add_argument('--cptac-aligned', type=Path, default=Path(
        '/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/aligned_r4'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('tcga','cptac','all'), default='all')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == args.baseline.resolve() or args.baseline.resolve() in out.parents:
        parser.error('Build in a separate staging directory before replacing the live input')
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / '.refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for part in ('aligned', 'cache'):
            (out / part).mkdir(exist_ok=True)
        for name in AXES:
            if not (out / name).exists():
                shutil.copy2(args.baseline / name, out / name)
            check(out / name, sha(args.baseline / name))
        tcga = read(args.tcga / 'COMPLETE.json')
        before = read(args.baseline / 'COMPLETE.json')
        records = tcga['records']
        if tcga['status'] != 'COMPLETE' or before['status'] != 'COMPLETE' or len(records) != 33:
            raise ValueError('Incomplete patient source manifest')
        plan = dict(tcga_manifest=sha(args.tcga / 'COMPLETE.json'), baseline_manifest=sha(args.baseline / 'COMPLETE.json'),
                    cases=sorted(before['records']), annotation='GENCODE_v36', workflow='STAR - Counts',
                    metric='tpm_unstranded', scoring_background_sha256=sha(out / AXES[0]),
                    scoring_background='FROZEN_PREVIOUS_COMMON_GENE_AXIS', expression_filter='NONE')
        pre = out / 'LAUNCH_PREFLIGHT.json'
        if pre.exists() and read(pre) != plan:
            raise ValueError('Staging inputs differ from the original plan')
        atomic_json(pre, plan)
        jobs = []
        if args.phase in ('tcga','all'):
            jobs.append(('TCGA', sorted(records), partial(tcga_case, records=records, before=before,
                                                        baseline=args.baseline, output=out)))
        if args.phase in ('cptac','all'):
            sources = read(args.cptac_aligned / 'COMPLETE.json')
            quant = sources.get('rna_quantification', {})
            if (quant.get('workflow_type'), quant.get('gene_model'), quant.get('metric')) != (
                    'STAR - Counts', 'GENCODE v36', 'tpm_unstranded'):
                raise ValueError('Expected a verified GDC v36 CPTAC source, found ' + str(quant))
            source_pin = dict(path=str(args.cptac_aligned.resolve() / 'COMPLETE.json'),
                              sha256=sha(args.cptac_aligned / 'COMPLETE.json'))
            pin_path = out / 'GDC_CPTAC_SOURCE.json'
            if pin_path.exists() and read(pin_path) != source_pin:
                raise ValueError('Staged expression belongs to a different CPTAC source')
            atomic_json(pin_path, source_pin)
            jobs.append(('CPTAC', CPTAC, partial(cptac_case, records=records, baseline=args.baseline, output=out,
                                               aligned=args.cptac_aligned, sources=sources, old_records=before['records'])))
        for dataset, cohorts, func in jobs:
            completed = {}
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for case, row in pool.map(func, cohorts):
                    completed[case] = row
                    print(f'{case}: {row["patients"]} patient rows verified', flush=True)
            atomic_json(out / (dataset + '_VERIFIED.json'), completed)
        if all((out / (d + '_VERIFIED.json')).is_file() for d in ('TCGA','CPTAC')):
            merged = {**read(out/'TCGA_VERIFIED.json'), **read(out/'CPTAC_VERIFIED.json')}
            if set(merged) != set(before['records']):
                raise ValueError('Missing or unexpected production cohort')
            final = dict(status='COMPLETE', target_host='149', records=merged, failures={}, quantification=plan,
                         cptac_source=read(out / 'GDC_CPTAC_SOURCE.json'))
            atomic_json(out/'PROGRESS.json', dict(final, completed=len(merged), total=len(merged)))
            atomic_json(out/'COMPLETE.json', final)


if __name__ == '__main__':
    main()
