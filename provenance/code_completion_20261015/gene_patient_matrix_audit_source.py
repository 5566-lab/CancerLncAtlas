"""Hash and numerically audit completed gene matrices on host149 only."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import socket
import traceback

ROOT=Path('/dell_2/DSC/CancerLncAtlas/inputs/gene_node_omics_20261029_r1')

def record(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return {'path':str(path),'sha256':h.hexdigest()}
def pin(item):
    assert record(item['path'])['sha256']==item['sha256'], item['path']
    return Path(item['path'])

def main():
    assert socket.gethostname()=='149'
    import numpy as np
    import pandas as pd
    product_path=ROOT/'patient_gene_matrices_r2/COMPLETE.json'
    product=json.loads(product_path.read_text())
    assert product['status']=='PASS'
    genes=pd.read_parquet(pin(product['genes']))
    patients=pd.read_parquet(pin(product['patients']))
    mapping=pd.read_parquet(pin(product['promoter_probe_map']))
    raw=pd.read_parquet(pin(product['verified_raw_files']))
    assert len(raw)==product['raw_file_count']==18600
    assert not raw.duplicated(['channel','file_id']).any()
    assert raw.sha256.str.fullmatch('[0-9a-f]{64}').all() and raw.md5.str.fullmatch('[0-9a-f]{32}').all()
    assert len(genes)==19149 and not genes.gene_id.duplicated().any()
    assert len(patients)==10432 and not patients.patient_id.duplicated().any()
    channels=list(product['units'])
    total={c:{'observed_cells':0,'missing_cells':0,'observed_zero_cells':0,'observed_patients':0,
              'minimum_observed':float('inf'),'maximum_observed':float('-inf')} for c in channels}
    gene_seen={c:np.zeros(len(genes),dtype=bool) for c in channels}
    seen=set(); both=0; hashes=[]
    for part in product['partitions']:
        ids=json.loads(pin(part['patients']).read_text()); assert not seen.intersection(ids); seen.update(ids)
        expected=patients.loc[patients.cancer_id.eq(part['cancer_id']),'patient_id'].tolist(); assert ids==expected
        patient_seen={}
        for c in channels:
            records=part['channels'][c]; vp=pin(records['values']); ap=pin(records['available'])
            hashes.extend([records['values'],records['available']])
            v=np.load(vp,mmap_mode='r',allow_pickle=False); m=np.load(ap,mmap_mode='r',allow_pickle=False)
            assert v.shape==m.shape==(len(ids),len(genes)) and v.dtype==np.float32 and m.dtype==np.bool_
            assert not np.isinf(v).any() and np.array_equal(np.isfinite(v),m)
            values=v[m]
            if c=='methylation_promoter_beta':assert ((values>=0)&(values<=1)).all()
            row_seen=m.any(axis=1); patient_seen[c]=row_seen
            gene_seen[c]|=m.any(axis=0)
            t=total[c]; t['observed_cells']+=int(m.sum()); t['missing_cells']+=int((~m).sum())
            t['observed_zero_cells']+=int((values==0).sum());t['observed_patients']+=int(row_seen.sum())
            if len(values):t['minimum_observed']=min(t['minimum_observed'],float(values.min()));t['maximum_observed']=max(t['maximum_observed'],float(values.max()))
            del v,m,values
        both+=int((patient_seen[channels[0]]&patient_seen[channels[1]]).sum())
        print('AUDITED_CANCER',part['cancer_id'],flush=True)
    assert seen==set(patients.patient_id)
    for c in channels:
        total[c]['observed_genes']=int(gene_seen[c].sum())
        total[c]['unobserved_genes_by_chromosome']=genes.loc[~gene_seen[c]].chromosome.value_counts().to_dict()
        total[c]['selected_raw_files']=int(raw.channel.eq(c).sum())
        total[c]['patients_with_selected_files']=int(raw.loc[raw.channel.eq(c)].patient_id.nunique())
        assert total[c]['observed_cells']+total[c]['missing_cells']==len(patients)*len(genes)
    result={'target_host':'149','status':'PASS','training_started':False,
        'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'matrix_product':record(product_path),'audit_script':record(__file__),
        'cancers':len(product['partitions']),'patients':len(patients),'matrix_genes':len(genes),
        'promoter_mapped_genes':int(mapping.gene_id.nunique()),'distinct_promoter_probes':int(mapping.probe_id.nunique()),
        'patients_observed_in_both_channels':both,'channels':total,'matrix_files_hashed':len(hashes),
        'array_hashes_verified':hashes,'raw_file_sha256_records_checked':len(raw),
        'raw_bytes_rehashed_by_this_audit':False,
        'raw_byte_validation':'All source MD5/size and consumed-byte SHA256 were checked by the completed materializer; this audit verifies its pinned raw-file receipt.',
        'matrix_orientation':'patient_by_gene','missing_cells_are_nan_with_false_mask':True}
    (ROOT/'GENE_MATRIX_COVERAGE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='array_hashes_verified'}),flush=True)

if __name__=='__main__':
    assert socket.gethostname()=='149'
    (ROOT/'GENE_MATRIX_AUDIT_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','training_started':False,'timeout_seconds':2400}))
    try:main()
    except BaseException as e:
        (ROOT/'GENE_MATRIX_AUDIT_FAILED.json').write_text(json.dumps({'error':str(e),'traceback':traceback.format_exc()}))
        raise
