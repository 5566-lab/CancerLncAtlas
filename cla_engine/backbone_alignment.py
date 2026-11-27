"""Cross-database evidence links to exact KEGG cancer mechanisms."""
from collections import defaultdict
from pathlib import Path
import csv,gzip,json
import numpy as np
from scipy.stats import hypergeom

def bh(values):
    values=np.asarray(values,float)
    order=np.argsort(values,kind='stable');n=len(values)
    ranked=values[order]*n/np.arange(1,n+1)
    result=np.empty(n,float);result[order]=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1])
    return result

def pin_anchor_components(protein_sets,backbone,source_sets):
    """Use original KEGG proteins as anchors, never the cross-library union."""
    result={cid:set(values) for cid,values in protein_sets.items()}
    for row in backbone:
        values=set(source_sets.get(row['source_target_id'],set()))-{''}
        if not values:
            raise ValueError('BACKBONE_ORIGINAL_PROTEIN_COMPOSITION_MISSING:'+row['source_target_id'])
        result[row['canonical_id']]=values
    return result

def build_alignment(master,backbone,protein_sets,excluded_kegg=(),regulator_proteins=None):
    """Enrichment gives context links, never biological identity equivalence.

    Universe is the frozen protein catalog. BH covers all tested non-backbone
    identities x backbone pairs, including zero-overlap pairs (p=1).
    """
    regulator_proteins=regulator_proteins or {}
    anchors={x['canonical_id']:x for x in backbone}
    excluded=set(excluded_kegg)
    universe=set().union(*protein_sets.values())
    size=len(universe)
    records=[];tested=0
    for row in master:
        cid=row['canonical_id']
        if cid in anchors or cid in excluded:continue
        a=protein_sets.get(cid,set())
        for aid,anchor in anchors.items():
            b=protein_sets.get(aid,set());common=len(a & b)
            direct=sorted(regulator_proteins.get(cid,set()) & b)
            p=float(hypergeom.sf(common-1,size,len(b),len(a))) if common and a and b else 1.
            records.append({'source_canonical_id':cid,'source_name':row['display_name'],
              'source_database':row['source_database'],'target_canonical_id':aid,
              'network_id':anchor['network_id'],'target_name':anchor.get('network_name') or anchor['display_name'],
              'target_network_type':anchor.get('network_type',''),
              'source_protein_count':len(a),'target_protein_count':len(b),
              'shared_proteins':common,'jaccard':common/(len(a|b)) if a or b else 0.,
              'source_coverage':common/len(a) if a else 0.,'target_coverage':common/len(b) if b else 0.,
              'overlap_p':p,'direct_regulator_proteins':'|'.join(direct)})
    q=bh([x['overlap_p'] for x in records]) if records else []
    output=[]
    for row,adjusted in zip(records,q):
        row['overlap_q']=float(adjusted)
        direct=bool(row['direct_regulator_proteins'])
        if not row['shared_proteins'] and not direct:continue
        significant=direct or (row['shared_proteins']>=3 and adjusted<=.05)
        row['admitted_context_link']=str(bool(significant))
        row['relation']='regulator_component_in_mechanism' if direct else 'protein_component_association'
        row['alignment_evidence_basis']='exact_regulator_molecular_identity' if direct else 'protein_set_enrichment'
        row['identity_merge']='False'
        row['mechanism_context_inherited']='False'
        output.append(row)
    bysource=defaultdict(list)
    for row in output:bysource[row['source_canonical_id']].append(row)
    top=[]
    for cid,rows in bysource.items():
        ranked=sorted(rows,key=lambda x:(not bool(x['direct_regulator_proteins']),x['overlap_q'],-x['jaccard'],-x['shared_proteins'],x['network_id']))
        for rank,row in enumerate(ranked,1):
            row['alignment_rank']=rank
            if rank<=3:top.append(row)
    supported={x['source_canonical_id'] for x in output if x['admitted_context_link']=='True'}
    return output,top,supported,{'protein_universe_size':size,'tested_pairs':len(records),
      'nonzero_overlap_pairs':len(output),'supported_context_identities':len(supported),
      'selection':'protein-set enrichment: BH q<=0.05 and >=3 shared proteins; explicit TF/kinase molecular identity: direct backbone membership; no identity merge',
      'patient_expression_used':False,'lncrna_association_used':False}
