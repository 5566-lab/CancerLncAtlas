"""Cache common-universe RNA score components; learned references are deferred."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cc_hhgt.m0_joint_scores import raw_ssgsea
from cla_engine.common import sha
from run_m0_linear import write_json

def cache_case(aligned,cancer,output,refs,definitions,xcell_ref,config):
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    src=Path(aligned)/cancer;out=Path(output)/cancer
    if out.exists() and any(out.iterdir()):raise ValueError('New output required')
    out.mkdir(parents=True);write_json(out/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='JOINT_Y_RAW_COMPONENTS',cancer=cancer))
    rec=json.loads((src/'COMPLETE.json').read_text())
    for n,d in rec['files'].items():
        if sha(src/n)!=d:raise ValueError('Input drift '+n)
    ref=json.loads(Path(refs).read_text())['references']
    for v in ref.values():
        if 'path' in v and 'sha256' in v and sha(v['path'])!=v['sha256']:raise ValueError('Reference drift '+v['path'])
    defs=json.loads((Path(definitions)/'SIGNIFINDER_EXACT_DEFINITIONS.json').read_text())
    hs=json.loads((Path(definitions)/'HACKSIG_EXACT_DEFINITIONS.json').read_text())['signatures_data']
    xc=json.loads((Path(definitions)/'XCELL_ORIGINAL_DATA.json').read_text())
    e=pd.read_parquet(src/'SYMBOL_LOGTPM.parquet');pc=pd.read_csv(src/'PROTEIN_CODING_SYMBOLS.tsv',sep='\t').symbol.tolist();ep=e[pc];e.index.name='patient_id';ep.index.name='patient_id'
    def sets_gmt(path):
        z={}
        for line in Path(path).read_text().splitlines():
            t=line.split('\t');z[t[0]]=list(dict.fromkeys(t[2:]))
        return z
    def genes(name,klass=None):return [row['SYMBOL'] for row in defs[name] if klass is None or row.get('class')==klass]
    gs=sets_gmt(ref['pathway_gmt']['path']);pw,cov=raw_ssgsea(ep,{'RNA_PATHWAY|'+k:v for k,v in gs.items()},minimum=10,maximum=500)
    pw.to_parquet(out/'PATHWAY_RAW.parquet',compression='zstd');cov.to_csv(out/'PATHWAY_COVERAGE.tsv',sep='\t',index=False)
    print(cancer,'PATHWAY_RAW',pw.shape,flush=True)
    stsets={'STATE|signifinder::EMT_Miow_Epithelial':genes('EMT_Miow','Epithelial-like'),
       'STATE|signifinder::EMT_Miow_Mesenchymal':genes('EMT_Miow','Mesenchymal-like'),
       'STATE|signifinder::ECM_Chakravarthy_up':genes('ECM_Chakravarthy','ECMup'),
       'STATE|signifinder::ECM_Chakravarthy_down':genes('ECM_Chakravarthy','ECMdown'),
       'STATE|SenMayo::GSVA_ssGSEA':json.loads(Path(ref['senmayo']['path']).read_text())['SAUL_SEN_MAYO']['geneSymbols']}
    sr,sc=raw_ssgsea(e,stsets);sr.to_parquet(out/'STATE_SSGSEA_RAW.parquet',compression='zstd');sc.to_csv(out/'STATE_SSGSEA_COVERAGE.tsv',sep='\t',index=False)
    hist=json.loads(Path(ref['historical_networks']['path']).read_text());hr,hc=raw_ssgsea(ep,hist['hallmark']);hr.to_parquet(out/'HALLMARK_RAW.parquet',compression='zstd')
    raw=pd.DataFrame(index=e.index);coverage=[]
    def pick(name):
        gg=genes(name);present=[g for g in gg if g in e]
        coverage.append(dict(signature=name,total=len(gg),measured=len(present)))
        if not present:raise ValueError('Unobserved signature '+name)
        return e[present]
    raw['STATE|signifinder::Hypoxia_Buffa']=pick('Hypoxia_Buffa').median(axis=1,skipna=False)
    raw['STATE|signifinder::CellCycle_Lundberg']=pick('CellCycle_Lundberg').sum(axis=1,skipna=False)
    raw['STATE|signifinder::MitoticIndex_Yang']=np.expm1(pick('MitoticIndex_Yang')*np.log(2)).mean(axis=1,skipna=False)
    cin=pick('CIN_Carter');raw['STATE|signifinder::CIN_Carter_70']=(cin.sub(e.mean(axis=1,skipna=False),axis=0)).sum(axis=1,skipna=False)
    for name in ['Glycolysis_Zhang','LipidMetabolism_Zheng']:
        table=pd.DataFrame(defs[name]);table=table.loc[table.SYMBOL.isin(e.columns)];v=e[table.SYMBOL]
        if name=='LipidMetabolism_Zheng':v=np.expm1(v*np.log(2))
        raw['STATE|signifinder::'+name]=v.to_numpy()@table.coeff.to_numpy(float)
    raw['STATE|signifinder::VEGF_Hu']=pick('VEGF_Hu').mean(axis=1,skipna=False)
    raw['STATE|signifinder::IFN_Ayers']=pick('IFN_Ayers').mean(axis=1,skipna=False)*np.log10(2)
    tab=pd.DataFrame(defs['Tinflam_Ayers']);pred=tab.loc[tab['class'].eq('TInflam') & tab.SYMBOL.isin(e.columns)]
    house=[g for g in genes('Tinflam_Ayers','Housekeeping') if g in e]
    raw['STATE|signifinder::Tinflam_Ayers']=(e[pred.SYMBOL].sub(e[house].mean(axis=1,skipna=False),axis=0)*np.log10(2)).to_numpy()@pred.coeff.to_numpy(float)
    raw['STATE|signifinder::TLS_Cabrita']=pick('TLS_Cabrita').mean(axis=1,skipna=False)
    raw['STATE|CYT::mean_log_GZMA_PRF1']=ep[['GZMA','PRF1']].mean(axis=1,skipna=False)
    mcp=pd.read_csv(ref['mcp']['path'],sep='\t')
    for cell,g in mcp.groupby('Cell population'):
        present=[x for x in g['HUGO symbols'] if x in ep]
        raw['STATE|MCPcounter::'+cell]=ep[present].mean(axis=1,skipna=False)
    hh=pd.DataFrame(hs);estsets={name:hh.loc[hh.signature_id.eq(k),'gene_symbol'].tolist() for name,k in [('ImmuneScore','estimate_immune'),('StromalScore','estimate_stromal')]}
    es,_=raw_ssgsea(ep,estsets,integer_ranks=False)
    raw['STATE|ESTIMATE::ImmuneScore']=es.ImmuneScore;raw['STATE|ESTIMATE::StromalScore']=es.StromalScore
    total=es.sum(axis=1,skipna=False);raw['STATE|ESTIMATE::ESTIMATEScore']=total
    purity=np.cos(.6049872018+.0001467884*total);raw['STATE|ESTIMATE::TumorPurity']=purity.where(purity>=0)
    # hacksig scales each SAMPLE across its genes, not each gene across patients.
    z=ep.sub(ep.mean(axis=1,skipna=False),axis=0).div(ep.std(axis=1,ddof=1,skipna=False),axis=0)
    h=hh.loc[hh.signature_id.str.startswith('ips_') & hh.gene_symbol.isin(ep.columns)].copy()
    h['gene_type']=h.signature_keywords.str.replace(r'\|.*','',regex=True).str.replace(r' |\-','_',regex=True)
    types=[]
    for name,g in h.groupby('gene_type'):
        types.append(pd.DataFrame({'class':g.signature_id.iloc[0],'score':z[g.gene_symbol].mean(axis=1,skipna=False)*g.gene_weight.astype(float).mean()},index=e.index))
    ips=pd.concat(types).reset_index().groupby(['patient_id','class']).score.mean().unstack().sum(axis=1,skipna=False).reindex(e.index)
    raw['STATE|IPS::ips_score']=np.where(ips<=0,0,np.where(ips>=3,10,np.round(ips*10/3)))
    gene=pd.read_parquet(src/'GENE_LOGTPM.parquet')
    for name,sheet,header in [('RNAss','mRNAsi',0),('EREG.EXPss','EREG-mRNAsi',1)]:
        w=pd.read_excel(ref['malta']['path'],sheet_name=sheet,header=header).dropna(subset=['ENSEMBL']);w['gene_id']=w.ENSEMBL.astype(str).str.replace(r'\.[0-9]+$','',regex=True)
        w=w.set_index('gene_id').Weight;common=sorted(set(w.index)&set(gene.columns));v=gene[common].to_numpy(float)
        a=rankdata(v,axis=1,method='average',nan_policy='propagate');b=rankdata(w.loc[common].to_numpy());a-=a.mean(axis=1,keepdims=True);b-=b.mean()
        raw['STATE|stemness_rna::'+name]=(a@b)/np.sqrt((a*a).sum(axis=1)*(b*b).sum())
    raw.to_parquet(out/'STATE_FIXED_RAW.parquet',compression='zstd');pd.DataFrame(coverage).to_csv(out/'STATE_COVERAGE.tsv',sep='\t',index=False)
    xgenes=sorted(set(pc)&set(xc['genes']));xr,xcov=raw_ssgsea(e[xgenes],xc['signatures']);xr.to_parquet(out/'XCELL_RAW.parquet',compression='zstd');xcov.to_csv(out/'XCELL_COVERAGE.tsv',sep='\t',index=False)
    pg=pd.read_csv(ref['progeny']['path'],sep='\t');dn=pd.read_csv(ref['dorothea']['path'],sep='\t')
    net=pd.concat([pd.DataFrame({'gene':pg.gene,'state_id':'PROGENY:'+pg.pathway,'weight':pg.weight}),pd.DataFrame({'gene':dn.target,'state_id':'DOROTHEA:'+dn.tf,'weight':dn.mor})]).groupby(['gene','state_id'],as_index=False).weight.mean()
    net.to_csv(out/'PROGRAM_NETWORK.tsv',sep='\t',index=False)
    write_json(out/'DEFINITIONS.json',{k:defs[k] for k in ['DNArep_Kang','LRRC15CAF_Dominguez','APM_Wang']})
    write_json(out/'XCELL_REFERENCE.json',json.loads(Path(xcell_ref).read_text()))
    write_json(out/'CONFIG_TARGETS.json',json.loads(Path(config).read_text())['targets'])
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',cancer=cancer,input_complete_sha256=sha(src/'COMPLETE.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()},
      semantics='Common_background_raw_components;cohort_fitted_transforms_deferred_to_train_only',source_hashes={str(p):sha(p) for p in [Path(refs),Path(xcell_ref),Path(definitions)/'SIGNIFINDER_EXACT_DEFINITIONS.json',Path(definitions)/'HACKSIG_EXACT_DEFINITIONS.json',Path(definitions)/'XCELL_ORIGINAL_DATA.json']}))
    print('CACHE_COMPLETE',cancer,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['aligned','cancer','output','refs','definitions','xcell-ref','config']:p.add_argument('--'+n,required=True)
    a=p.parse_args();cache_case(a.aligned,a.cancer,a.output,a.refs,a.definitions,a.xcell_ref,a.config)
