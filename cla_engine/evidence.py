"""Source-preserving evidence audit. Observed assertion is NOT a learned causal probability.

Only a source explicitly typing its partner as pathway can produce a direct
pathway assertion here. Gene/protein partners are never broadcast to pathways.
"""
from __future__ import annotations
import csv,hashlib,json,re,sqlite3
from pathlib import Path
import pandas as pd
from .common import atomic_json,sha,fail,status,lock
from .tables import strict_bool,require

def chunks(path):
    p=Path(path)
    if p.suffix=='.parquet':
        import pyarrow.parquet as pq
        for b in pq.ParquetFile(p).iter_batches(batch_size=25000):yield b.to_pandas()
    else:
        yield from pd.read_csv(p,sep='\t',chunksize=25000,low_memory=False)

def text(v):return '' if pd.isna(v) else str(v).strip()
def direction(v):
    v=text(v).lower()
    return 1 if v in ['positive','up','upregulation','activate','activation','1'] else (-1 if v in ['negative','down','downregulation','inhibit','inhibition','-1'] else 0)
def review(v):
    v=text(v).lower()
    if v in ['approved','validated','accepted','pass']:return 'record_approved'
    if v in ['not approved','not validated','invalidated','rejected','fail']:return 'record_not_approved'
    return 'unknown'

def audit(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not source.exists():fail('EVIDENCE_SOURCE_MISSING',str(source))
    identity=sha(source)
    with lock(out):
        receipt=out/'EVIDENCE_RECEIPT.json'
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['source_sha256']!=identity:fail('EVIDENCE_SOURCE_DRIFT','Use a new output directory')
            return r
        dbpath=out/'evidence.sqlite';db=sqlite3.connect(dbpath)
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS events(event_key TEXT PRIMARY KEY,lncrna TEXT,partner TEXT,partner_type TEXT,cancer TEXT,study TEXT,pmid_raw TEXT,context TEXT,relation TEXT,direction INTEGER,experimental INTEGER,predicted INTEGER,record_review TEXT,source TEXT,source_row TEXT,assay TEXT,raw_json TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS progress(source_sha TEXT PRIMARY KEY,next_batch INTEGER,rows_seen INTEGER)')
        p=db.execute('SELECT source_sha,next_batch,rows_seen FROM progress').fetchone()
        if p and p[0]!=identity:fail('EVIDENCE_SOURCE_DRIFT','Partial SQLite belongs to another source')
        start=p[1] if p else 0;seen=p[2] if p else 0
        for bi,f in enumerate(chunks(source)):
            if bi<start:continue
            require(f,['lncrna_id','partner_id','partner_type','source_database','pmid'],'evidence raw relation')
            records=[]
            for row in f.to_dict('records'):
                l=text(row.get('lncrna_id'));partner=text(row.get('partner_id'))
                if not l or not partner:continue
                typ=text(row.get('partner_type')).lower();rawpmid=text(row.get('pmid'));pmids=sorted(set(re.findall(r'\b\d{5,9}\b',rawpmid)))
                # Ambiguous multi-PMID records retained as one unresolved study group, not expanded into independent experiments.
                study='PMID:'+pmids[0] if len(pmids)==1 else ('MULTI_PMID:'+','.join(pmids) if pmids else 'NO_STUDY_ID:'+text(row.get('source_database'))+':'+text(row.get('source_record_id',row.get('source_row_id'))))
                context='|'.join(text(row.get(k)) for k in ['cancer_id','tissue','cell_line','species']);assay=text(row.get('experiment_raw',row.get('experiment_family')));relation=text(row.get('relation_raw',row.get('relation_type')));dr=direction(row.get('direction'))
                raw=json.dumps({k:(None if pd.isna(v) else v) for k,v in row.items()},ensure_ascii=False,default=str)
                key=hashlib.sha256('|'.join([l,partner,typ,study,context,assay,relation,str(dr)]).encode()).hexdigest()
                ex=int(strict_bool(pd.Series([row.get('is_experimental',False)]))[0]);pred=int(strict_bool(pd.Series([row.get('is_predicted',False)]))[0])
                records.append((key,l,partner,typ,text(row.get('cancer_id')),study,rawpmid,context,relation,dr,ex,pred,review(row.get('manual_review_status')),text(row.get('source_database')),text(row.get('source_record_id',row.get('source_row_id'))),assay,raw))
            db.executemany('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',records);seen+=len(f)
            db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,?)',(identity,bi+1,seen));db.commit();status(out,'EVIDENCE_AUDITING',source_rows=seen,batch=bi+1)
        db.execute('CREATE INDEX IF NOT EXISTS pair_idx ON events(lncrna,partner,cancer)');db.commit()
        query="""SELECT cancer,lncrna,partner,partner_type,COUNT(*) events,
        COUNT(DISTINCT CASE WHEN study LIKE 'PMID:%' THEN study END) identified_studies,
        SUM(CASE WHEN direction=1 THEN 1 ELSE 0 END) positive_events,
        SUM(CASE WHEN direction=-1 THEN 1 ELSE 0 END) negative_events,
        SUM(CASE WHEN direction=0 THEN 1 ELSE 0 END) unknown_direction_events,
        SUM(experimental) experimental_events,SUM(predicted) predicted_events
        FROM events GROUP BY cancer,lncrna,partner,partner_type"""
        with (out/'OBSERVED_SUPPORT.tsv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream,delimiter='\t');cursor=db.execute(query);writer.writerow([d[0] for d in cursor.description]+['support_role','direction_conflict','learned_probability'])
            for r in cursor:
                direct=r[3] in ['pathway','exact_pathway'];writer.writerow([*r,'DIRECT_ASSERTION_SOURCE_TYPED' if direct else 'PARTNER_SUPPORT_NOT_PATHWAY_DIRECTION',r[6]>0 and r[7]>0,'NA'])
        counts={'source_rows':seen,'deduplicated_events':db.execute('SELECT COUNT(*) FROM events').fetchone()[0],'single_pmid_events':db.execute("SELECT COUNT(*) FROM events WHERE study LIKE 'PMID:%'").fetchone()[0],'explicit_pathway_events':db.execute("SELECT COUNT(*) FROM events WHERE partner_type IN ('pathway','exact_pathway')").fetchone()[0]}
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
        r={'state':'EVIDENCE_OBSERVED_AUDIT_COMPLETE_NOT_MODEL_TRAINED','source_sha256':identity,'counts':counts,'raw_lineage_preserved':True,'family_broadcast':False,'review_quality_is_biological_label':False,'model_probability':'NOT_ESTIMATED','independent_biological_supervision':'NOT_IDENTIFIED_IN_THESE_BOUND_COLUMNS','transformer_training':'NOT_RUN_NO_FABRICATED_GOLD','changes_discovery_ranking':False,'same_study_alias_resolution':'PARTIAL_PMID_ONLY_MULTI_PMID_UNRESOLVED','sqlite_path':str(dbpath)}
        atomic_json(receipt,r);status(out,r['state'],counts=counts);return r
