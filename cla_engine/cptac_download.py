"""Download the explicit CPTAC publication manifest on host 149."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, re, socket, time
from pathlib import Path
from urllib.request import Request, ProxyHandler, build_opener
from .common import atomic_json, read_json, sha, fail
from .cptac_scope import select_cohorts

def download_one(rec, root, attempts=3):
    if Path(rec['name']).name!=rec['name'] or Path(rec['cohort']).name!=rec['cohort']:
        fail('CPTAC_DOWNLOAD_PATH','Expected basename and explicit cohort ID')
    dest=Path(root)/rec['cohort']/rec['name']
    dest.parent.mkdir(parents=True,exist_ok=True)
    receipt=dest.with_name(dest.name+'.receipt.json')
    if dest.exists() and receipt.exists():
        done=read_json(receipt)
        if done['url']==rec['url'] and (not rec.get('sha256') or rec['sha256']==done['sha256']) and done['sha256']==sha(dest) and dest.stat().st_size==done['bytes']:
            return done
        fail('CPTAC_DOWNLOAD_DRIFT','Existing file differs from its receipt',path=str(dest))
    part=dest.with_name(dest.name+'.part')
    error=None
    for attempt in range(attempts):
        try:
            opener=build_opener(ProxyHandler({}))
            with opener.open(Request(rec['url'],headers={'User-Agent':'Mozilla/5.0 CancerLncAtlas-research'}),timeout=90) as response:
                length=int(response.headers['Content-Length'])
                etag=response.headers.get('ETag','').strip('"')
                if rec.get('bytes') and rec['bytes']!=length:raise ValueError('Published content length changed')
                if rec.get('etag') and rec['etag'].strip('"')!=etag:raise ValueError('Published ETag changed')
                md5=hashlib.md5();h=hashlib.sha256();total=0
                with part.open('wb') as handle:
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk:break
                        handle.write(chunk);total+=len(chunk);md5.update(chunk);h.update(chunk)
                if total!=length:raise ValueError('Incomplete download')
                md5_verified=bool(re.fullmatch(r'[0-9a-fA-F]{32}',etag))
                if md5_verified and md5.hexdigest().lower()!=etag.lower():raise ValueError('Object MD5 mismatch')
                if rec.get('sha256') and rec['sha256']!=h.hexdigest():raise ValueError('Frozen publication SHA256 changed')
                os.replace(part,dest)
                done={k:v for k,v in rec.items() if k!='error'}
                done.update(path=str(dest.resolve()),bytes=total,sha256=h.hexdigest(),etag=etag,
                            object_md5_verified=md5_verified,status='COMPLETE',target_host='149')
                atomic_json(receipt,done)
                print(json.dumps({'file':rec['name'],'status':'COMPLETE','bytes':total}),flush=True)
                return done
        except Exception as exc:
            error=str(exc)
            print(json.dumps({'file':rec['name'],'attempt':attempt+1,'error':error}),flush=True)
            if attempt+1<attempts:time.sleep(2**attempt)
    return {**rec,'status':'FAILED','error':error}

def run(manifest, output, cancer_type='all', cancer_list=None, workers=4):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','CPTAC preparation must run on 149')
    records=read_json(manifest)
    available={r['cohort'] for r in records}
    selected=select_cohorts(cancer_type,cancer_list,available)
    if not 1<=workers<=8:fail('CPTAC_DOWNLOAD_WORKERS','Use 1-8 download workers')
    if not selected or not selected<=available:fail('CPTAC_CANCER_SCOPE','Unknown or empty CPTAC cohort selection')
    records=[r for r in records if r['cohort'] in selected]
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'DOWNLOAD_CPTAC_PUBLICATION_FILES','cohorts':sorted(selected),'manifest_sha256':sha(manifest)})
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results=list(executor.map(lambda r:download_one(r,out),records))
    result={'format':'CLA_CPTAC_DOWNLOAD_V1','target_host':'149','status':'COMPLETE' if all(r['status']=='COMPLETE' for r in results) else 'INCOMPLETE',
            'cohorts':sorted(selected),'files':results}
    atomic_json(out/'DOWNLOAD_COMPLETE.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
    p.add_argument('--cancer-list','--cancer_list',nargs='+');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    r=run(a.manifest,a.output,a.cancer_type,a.cancer_list,a.workers)
    print(r['status'])
    if r['status']!='COMPLETE':raise SystemExit(1)

if __name__=='__main__':main()
