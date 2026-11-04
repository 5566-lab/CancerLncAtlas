"""Export verified RNA/gene CNV/promoter methylation measurements; no scoring method is selected."""
import collections, hashlib, json, socket, sys
from pathlib import Path
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description=__doc__)
for name in ('config','output','rna-root','reactome-hierarchy','reactome-xrefs','gene-dimension'):parser.add_argument('--'+name,required=True)
parser.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
parser.add_argument('--cancer-list','--cancer_list',nargs='+',default=[])
args=parser.parse_args()
W=Path(args.output).resolve().parent
assert socket.gethostname().split('.')[0]=='149'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.node_omics import open_verified_gene_node_omics
from cla_engine.tables import patient_map
from cla_engine.common import sha,atomic_json

cfg=json.loads(Path(args.config).read_text())
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise RuntimeError('Use a new output directory')
atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'VERIFIED_RAW_PATHWAY_FUSION_INPUTS','cancer_type':args.cancer_type})
from cla_engine.cancer_scope import resolve_cancer_scope
for key in ('patient_map','exact_membership'):
 assert sha(cfg['sources'][key]['path'])==cfg['sources'][key]['sha256'],key+'_HASH_DRIFT'
fm=patient_map(cfg['sources']['patient_map']['path'])
scope=resolve_cancer_scope({'cancer_type':args.cancer_type,'cancer_list':args.cancer_list},fm.cancer_id.unique())
cancers=scope['selected_cancers']
fm=fm.loc[fm.cancer_id.isin(cancers)]
bridge=open_verified_gene_node_omics(cfg['gene_node_omics'],[],cancers)
membership=pd.read_parquet(cfg['sources']['exact_membership']['path'])
membership=membership.loc[membership.pathway_id.str.startswith('MSIGDB:')].copy()
membership['gene_id']=membership.gene_id.str.removeprefix('GENE:')
sets={k:sorted(set(v.gene_id)) for k,v in membership.groupby('pathway_id',sort=True)}
assert len(sets)==2547
xref=json.loads(Path(args.reactome_xrefs).read_text())
mapping=collections.defaultdict(list)
for r in xref['rows']:
 if r['status']=='VERIFIED_DIRECT_XREF':mapping[r['reactome_id']].append(r['pathway_id'])
assert all(len(v)==1 for v in mapping.values()),'AMBIGUOUS_REACTOME_ID'
relation=Path(args.reactome_hierarchy)
parents=collections.defaultdict(set)
for line in relation.read_text().splitlines():
 a,b=line.split('\t')[:2]
 if a in mapping and b in mapping:
  parents[mapping[a][0]].add(mapping[b][0])
parents={k:sorted(v) for k,v in parents.items() if len(v)>=3}
catalog={'gene_sets':sets,'parent_members':parents,'parent_policy':'VERIFIED_REACTOME_DIRECT_CHILDREN_AT_LEAST_THREE; OTHER_PATHWAYS_UNFUSED',
 'source_gene_membership':{'path':cfg['sources']['exact_membership']['path'],'sha256':sha(cfg['sources']['exact_membership']['path'])},
 'reactome_hierarchy':{'path':str(relation),'sha256':sha(relation)},'reactome_xref_sha256':sha(Path(args.reactome_xrefs)),
 'missing_direct_xrefs':[r for r in xref['rows'] if r['status']!='VERIFIED_DIRECT_XREF'],
 'state_exclusion_prefixes':['PROGENY:','DOROTHEA:']}
atomic_json(out/'CATALOG.json',catalog)
print('CATALOG',len(sets),'PARENTS',len(parents),'MAX_CHILDREN',max(map(len,parents.values())),flush=True)
dim=pd.read_csv(args.gene_dimension,sep='\t')
coding=set(dim.loc[dim.gene_type.eq('protein_coding'),'ensembl_gene_id'])
records=[]
for cancer in cancers:
 dest=out/cancer;dest.mkdir(exist_ok=True)
 mp=fm.loc[fm.cancer_id.eq(cancer)].sort_values('patient_id')
 assert not mp.patient_id.duplicated().any(),'MULTIPLE_AUTHORITY_RNA_SAMPLES'
 ids=mp.patient_id.tolist();source=Path(args.rna_root)/cancer/'gene_logcpm.tsv.gz'
 rna=pd.read_csv(source,sep='\t',index_col=0)
 assert rna.index.is_unique and rna.columns.is_unique
 assert set(rna.index)<=coding,'RNA_BACKGROUND_MUST_BE_VERIFIED_PROTEIN_CODING'
 assert set(mp.sample_id)<=set(rna.columns),'RNA_AUTHORITY_SAMPLE_MISSING'
 rna=rna.loc[:,mp.sample_id];rna.columns=ids
 assert not np.isinf(rna.to_numpy(dtype=float)).any()
 part=bridge._partition(cancer)
 assert set(ids)<=set(part['ids'])
 omap=bridge.patients.loc[bridge.patients.cancer_id.eq(cancer)].set_index('patient_id')
 assert np.array_equal(omap.loc[ids,'patient_fold_id'].to_numpy(),mp.patient_fold_id.to_numpy()),'OMICS_PATIENT_FOLD_DRIFT'
 pi=[part['positions'][p] for p in ids]
 blocks={'RNA':(rna.index.tolist(),rna.to_numpy(dtype=float))}
 for channel,name in [('cnv_signed','CNV'),('methylation_promoter_beta','METHYLATION')]:
  values,mask=part['matrices'][channel]
  x=np.array(values[pi,:],dtype=float,copy=True);x[~np.asarray(mask[pi,:],dtype=bool)]=np.nan
  blocks[name]=(bridge.source_genes,x.T)
 specs={};paired=np.ones(len(ids),bool)
 for name,(genes,x) in blocks.items():
  present=np.isfinite(x).sum(axis=0)>=10;paired &= present
  p=dest/(name+'.f64');np.ascontiguousarray(x.T,dtype='<f8').tofile(p)
  specs[name]={'file':p.name,'sha256':sha(p),'genes':genes,'shape':list(x.shape),'dtype':'little_endian_float64','orientation':'R_column_major_gene_by_patient',
               'patients_with_at_least_10_measured_genes':int(present.sum()),'measured_values':int(np.isfinite(x).sum())}
 coverage={name:{'complete_variable_features_in_paired_patients':int((np.isfinite(x[:,paired]).all(axis=1)&(np.nanstd(x[:,paired],axis=1)>1e-12)).sum()) if paired.any() else 0} for name,(genes,x) in blocks.items()}
 record={'target_host':'149','status':'COMPLETE','cancer_id':cancer,'patient_ids':ids,'patient_fold_ids':mp.patient_fold_id.tolist(),'blocks':specs,
   'catalog_sha256':sha(out/'CATALOG.json'),'patient_map_sha256':sha(cfg['sources']['patient_map']['path']),
   'coding_gene_dimension':{'path':str(Path(args.gene_dimension).resolve()),'sha256':sha(args.gene_dimension)},
   'raw_RNA':{'path':str(source),'sha256':sha(source),'scale':'logCPM','sample_selection':'EXACT_FROZEN_AUTHORITY_SAMPLE'},
   'raw_omics_source_artifacts':part['artifacts'],'raw_omics_complete':bridge.complete_record,
   'paired_patients':int(paired.sum()),'complete_feature_coverage':coverage,'missingness_policy':'NAN_PRESERVED_NO_IMPUTATION'}
 atomic_json(dest/'COMPLETE.json',record);records.append(record)
 print('EXPORTED',cancer,len(ids),'PAIRED',int(paired.sum()),coverage,flush=True)
 bridge._partitions.pop(cancer,None)
summary={'target_host':'149','status':'COMPLETE','cancer_type':args.cancer_type,'cancers':cancers,'patients':sum(len(r['patient_ids']) for r in records),
 'paired_patients':sum(r['paired_patients'] for r in records),'catalog_sha256':sha(out/'CATALOG.json'),
 'partitions':[{'cancer_id':r['cancer_id'],'path':str(out/r['cancer_id']/'COMPLETE.json'),'sha256':sha(out/r['cancer_id']/'COMPLETE.json'),
 'patients':len(r['patient_ids']),'paired_patients':r['paired_patients'],'complete_feature_coverage':r['complete_feature_coverage']} for r in records]}
atomic_json(out/'COMPLETE.json',summary)
print('ALL_INPUTS_COMPLETE',summary['patients'],summary['paired_patients'],flush=True)
