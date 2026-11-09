from pathlib import Path
from datetime import datetime,timezone
import socket,urllib.request,urllib.parse,hashlib,json,gzip,pandas as pd,re
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;D=R/'measurement_mechanism_r1';O=R/'sequence_validation_r1';O.mkdir(exist_ok=True)
assert not (O/'COMPLETE.json').exists()
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'CPTAC_FLANK_TO_REACTOME_REFERENCE_SITE_VALIDATION','reference_sequence_only_not_activation_validation':True},indent=2))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode({'compressed':'true','format':'fasta','query':'(organism_id:9606) AND (reviewed:true)'})
p=R/'sources/uniprot_reviewed_human_20261109.fasta.gz'
if not p.exists():
 req=urllib.request.Request(url,headers={'User-Agent':'CancerLncAtlas-reference-site-audit/1.0'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=90) as resp:
  headers=dict(resp.headers);data=resp.read()
 assert len(data)>1000000
 with gzip.GzipFile(fileobj=__import__('io').BytesIO(data)) as z:assert z.read(100).startswith(b'>')
 p.write_bytes(data)
 (R/'sources/UNIPROT_SEQUENCE_RECEIPT.json').write_text(json.dumps({'target_host':'149','url':url,'sha256':sha(p),'bytes':len(data),'headers':headers,'downloaded_utc':datetime.now(timezone.utc).isoformat()},indent=2))
seq={};acc=None
with gzip.open(p,'rt') as f:
 for line in f:
  if line.startswith('>'):
   acc=line.split('|')[1];seq[acc]=''
  else:seq[acc]+=line.strip()
print('REFERENCE_PROTEINS',len(seq),flush=True)
d=pd.read_parquet(D/'CPTAC_REACTOME_PHOSPHOSITE_CANDIDATES.parquet');status=[];canonical_positions=[]
for row in d.itertuples():
 s=seq.get(row.uniprot_accession);flank=str(row.flanking_sequence).upper();site=str(row.site).upper();m=re.fullmatch(r'([STY])(\d+)',site)
 pos=int(m.group(2)) if m else 0;canonical_positions.append(pos)
 if s is None:status.append('REFERENCE_ACCESSION_NOT_IN_REVIEWED_CANONICAL_FASTA');continue
 if not m or pos<1 or pos>len(s):status.append('INVALID_REFERENCE_POSITION');continue
 if s[pos-1]!=m.group(1):status.append('CANONICAL_RESIDUE_MISMATCH');continue
 if len(flank)%2==0 or len(flank)<3:status.append('FLANK_FORMAT_UNSUPPORTED');continue
 half=len(flank)//2;left=max(0,pos-1-half);right=min(len(s),pos+half)
 canonical='_'*max(0,half-(pos-1))+s[left:right]+'_'*max(0,pos+half-len(s))
 # Accept source terminal padding - or _, never wildcard a real residue.
 f=flank.replace('-','_')
 status.append('CANONICAL_SITE_AND_FULL_FLANK_VERIFIED' if f==canonical else 'FLANK_SEQUENCE_MISMATCH')
d['sequence_validation_status']=status;d['canonical_site_and_flank_verified']=d.sequence_validation_status.eq('CANONICAL_SITE_AND_FULL_FLANK_VERIFIED')
d['full_modified_protein_state_measured']=False
d['sequence_reference_sha256']=sha(p)
d.to_parquet(O/'CPTAC_REACTOME_SITE_SEQUENCE_VALIDATION.parquet',index=False)
d.to_csv(O/'CPTAC_REACTOME_SITE_SEQUENCE_VALIDATION.tsv.gz',sep='\t',index=False)
# Counts of features are unique within each status; a feature can have several candidate accessions.
counts=d.groupby('sequence_validation_status').agg(candidate_rows=('feature_id','size'),unique_features=('feature_id','nunique')).reset_index();counts.to_csv(O/'SEQUENCE_VALIDATION_COUNTS.tsv',sep='\t',index=False)
valid=d.loc[d.canonical_site_and_flank_verified]
summary={'target_host':'149','status':'COMPLETE_SEQUENCE_SITE_VALIDATION','candidate_features':d.feature_id.nunique(),'verified_features':valid.feature_id.nunique(),'verified_sites':len(valid[['uniprot_accession','site']].drop_duplicates()),'verified_reference_states_with_at_least_one_site':valid.protein_state_node.nunique(),'full_state_or_activity_equivalence_claimed':False,'uniprot_reference_proteins':len(seq),'source_candidate_sha256':sha(D/'CPTAC_REACTOME_PHOSPHOSITE_CANDIDATES.parquet'),'reference_sha256':sha(p),'counts':counts.to_dict('records'),'limitations':['Canonical peptide-site agreement is not full modified-state measurement','Does not validate antibody epitope specificity','Does not identify patient pathway activation','Candidate coverage still requires per-patient measurement masks']}
summary['files']={x.name:{'sha256':sha(x),'bytes':x.stat().st_size} for x in O.iterdir() if x.is_file() and x.name!='COMPLETE.json'}
(O/'COMPLETE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
