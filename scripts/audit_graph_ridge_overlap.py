"""Audit RNA signature and symbol overlap without changing frozen scores."""
import argparse,socket,sys
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cc_hhgt.graph_ridge import target_policy

def audit(inputs,bundle,output):
    assert socket.gethostname()=='149';inputs=Path(inputs);b=Path(bundle);out=Path(output)
    bridge=pd.read_parquet(b/'STATE_GENE_PROTEIN_BRIDGE.parquet');align=pd.read_csv(b/'STATE_ALIGNMENT.tsv',sep='\t');records=[];collision=[]
    pro=pd.read_csv('/public8/DSC/CancerLncAtlas/input/progeny_human.tsv',sep='\t').rename(columns={'gene':'symbol'})
    pro['program']='PROGENY:'+pro.pathway
    dor=pd.read_csv('/public8/DSC/CancerLncAtlas/input/dorothea_human_ABC.tsv',sep='\t').rename(columns={'target':'symbol','mor':'weight'})
    dor['program']='DOROTHEA:'+dor.tf
    net=pd.concat([pro[['program','symbol','weight']],dor[['program','symbol','weight']]],ignore_index=True).groupby(['program','symbol'],as_index=False).weight.mean()
    net=net[np.isfinite(net.weight)&net.weight.ne(0)]
    for c in ['BRCA','LUAD','COAD']:
        f=pd.read_csv(inputs/c/'RNA_FEATURES.tsv',sep='\t');lnc=f[f.feature_type.eq('lncrna')];coding=f[f.feature_type.eq('mrna')];dual=set(lnc.reference_symbol)&set(coding.reference_symbol)
        collision.extend(dict(cancer_id=c,**row) for row in f[f.reference_symbol.isin(dual)].to_dict('records'))
        meta=pd.read_csv(inputs/c/'TARGETS.tsv',sep='\t').set_index('target_id')
        measured=pd.read_csv(b.parent/'aligned_r4'/'matrices'/c/'RNA_TUMOR_SYMBOL_TPM.tsv.gz',sep='\t',index_col=0)
        variable_symbols=set(measured.index[measured.var(axis=1).gt(0)])
        for a in align.to_dict('records'):
            tid='STATE|'+a['column'];p=target_policy(dict(target_id=tid,**meta.loc[tid].to_dict()));bb=bridge[bridge.state_id.eq(a['id'])]
            lncids=set(lnc.gene_id)&set(bb.gene_id.str.removeprefix('GENE:'));symbols=set(lnc.reference_symbol)&set(bb.reference_symbol);isrna=p['role']=='RNA_DERIVED_TARGET'
            basis='Existing State signature bridge; potential overlap, not proof of contribution to every native implementation'
            prefix='STATE|native_program::'
            if tid.startswith(prefix):
                program=tid[len(prefix):];nn=net[net.program.eq(program)&net.symbol.isin(variable_symbols)]
                symbols=set(lnc.reference_symbol)&set(nn.symbol);lncids=set(lnc.loc[lnc.reference_symbol.isin(symbols),'gene_id'])
                bb=pd.DataFrame({'gene_id':list(nn.symbol)})
                basis='Actual routed-program source weights after duplicate averaging; nonzero weights and variable measured RNA symbols'
            records.append(dict(cancer_id=c,overlap_basis=basis,target_id=tid,state_id=a['id'],measurement=a['measurement'],observed_patients=a['observed_patients'],mapped_signature_genes=int(bb.gene_id.fillna('').ne('').sum()),direct_lncrna_gene_ids=';'.join(sorted(lncids)),direct_lncrna_count=len(lncids),lncrna_signature_symbols=';'.join(sorted(symbols)),overlap_status='POTENTIAL_DIRECT_LNCRNA_REUSE' if isrna and (lncids or symbols) else 'SIGNATURE_MAPPING_PRESENT_NO_LNCRNA_MEMBER' if isrna and len(bb) and bb.gene_id.fillna('').ne('').any() else 'NO_FIXED_SIGNATURE_OR_UNRESOLVED' if isrna else 'NON_RNA_TARGET',background_limit='Some RNA scorers use all genes as ranking background; no member overlap does not prove no background dependence'))
    pd.DataFrame(records).to_csv(out/'STATE_LNCRNA_OVERLAP_AUDIT.tsv',sep='\t',index=False);pd.DataFrame(collision).to_csv(out/'RNA_SHARED_SYMBOL_AUDIT.tsv',sep='\t',index=False)
    print(pd.DataFrame(records).groupby(['cancer_id','overlap_status']).size().to_string(),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.inputs,a.bundle,a.output)
