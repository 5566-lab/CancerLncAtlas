"""Reuse unchanged assay files and replace only the verified network envelope."""
import argparse,json,socket,sys
from pathlib import Path
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from cc_hhgt.network_evidence import require_physical_network_receipt

def materialize(source,networks,output):
    assert socket.gethostname()=='149';source=Path(source);networks=Path(networks);out=Path(output);out.mkdir(parents=True,exist_ok=True);dump(out/'PREFLIGHT.json',dict(target_host='149',stage='VERIFIED_NETWORK_ENVELOPE',source=str(source),network=str(networks),code=sha(Path(__file__))))
    for cancer in ['BRCA','LUAD','COAD']:
        src=source/cancer;net=networks/cancer;d=out/cancer;d.mkdir(exist_ok=True)
        if (d/'COMPLETE.json').exists():raise ValueError('DO_NOT_OVERWRITE_VERIFIED_INPUTS')
        for folder in [src,net]:
            for name,h in json.loads((folder/'COMPLETE.json').read_text())['files'].items():
                if sha(folder/name)!=h:raise ValueError('SOURCE_DRIFT')
        for name in json.loads((src/'COMPLETE.json').read_text())['files']:
            if name in ['NETWORK_AUDIT.json','NETWORK_EDGES.parquet','RNA_LAPLACIAN.npz','PREFLIGHT.json','LAUNCH_PREFLIGHT.json']:continue
            (d/name).symlink_to((src/name).resolve())
        for old,new in [('CORRECTED_RNA_NETWORK_AUDIT.json','NETWORK_AUDIT.json'),('CORRECTED_RNA_NETWORK_EDGES.parquet','NETWORK_EDGES.parquet'),('LEGACY_union.npz','RNA_LAPLACIAN.npz')]:
            (d/new).write_bytes((net/old).read_bytes())
        require_physical_network_receipt(json.loads((d/'NETWORK_AUDIT.json').read_text()))
        dump(d/'PREFLIGHT.json',dict(target_host='149',stage='VERIFIED_NETWORK_ENVELOPE',assay_source=str(src),assay_receipt_sha256=sha(src/'COMPLETE.json'),physical_network_source=str(net),physical_network_receipt_sha256=sha(net/'COMPLETE.json'),assays_targets_covariates_and_splits_unchanged=True))
        dump(d/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}));print('VERIFIED_INPUT',cancer,flush=True)
    (out/'COVERAGE.tsv').write_bytes((source/'COVERAGE.tsv').read_bytes());dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'COMPLETE.json'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--networks',required=True);p.add_argument('--output',required=True);a=p.parse_args();materialize(a.source,a.networks,a.output)
