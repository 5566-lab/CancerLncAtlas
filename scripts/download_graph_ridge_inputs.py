"""Acquire or verify the six checksum-pinned CPTAC genomic demo sources."""
import argparse,concurrent.futures,hashlib,json,socket
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def acquire(work,sources,direct=False,verify_only=False):
    if socket.gethostname()!='149':raise RuntimeError('HOST_149_REQUIRED')
    work=Path(work);sources=Path(sources);items=json.loads(sources.read_text())
    if not verify_only:
        (work/'raw').mkdir(parents=True,exist_ok=True)
        (work/'DOWNLOAD_PREFLIGHT.json').write_text(json.dumps(dict(target_host='149',source_spec_sha256=digest(sources),files=len(items)),indent=2)+'\n')
    def one(e):
        p=work/'raw'/(e['cancer']+'_'+e['layer']+('.tsv.gz' if e['url'].endswith('.gz') else '.tsv'))
        if not p.exists():
            if verify_only:raise FileNotFoundError(p)
            q=p.with_suffix(p.suffix+'.part')
            with requests.Session() as session:
                session.trust_env=not direct
                with session.get(e['url'],timeout=(20,120),stream=True) as response:
                    response.raise_for_status()
                    with q.open('wb') as f:
                        for block in response.iter_content(1024*1024):f.write(block)
            if digest(q)!=e['sha256'] or q.stat().st_size!=e['bytes']:raise ValueError('DOWNLOADED_SOURCE_CHANGED: '+str(p))
            q.replace(p)
        if digest(p)!=e['sha256'] or p.stat().st_size!=e['bytes']:raise ValueError('EXISTING_SOURCE_CHANGED: '+str(p))
        print('VERIFIED',e['cancer'],e['layer'],flush=True)
        return dict(e,path=str(p))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:files=list(pool.map(one,items))
    if not verify_only:(work/'DOWNLOAD_COMPLETE.json').write_text(json.dumps(dict(target_host='149',files=files),indent=2)+'\n')
    return files
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--sources',default=str(ROOT/'config/graph_ridge_input_sources_20261105.json'));p.add_argument('--direct-network',action='store_true');p.add_argument('--verify-only',action='store_true');a=p.parse_args();acquire(a.work,a.sources,a.direct_network,a.verify_only)
