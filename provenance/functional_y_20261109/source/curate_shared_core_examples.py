from pathlib import Path
import socket,sys,json,hashlib
import numpy as np,pandas as pd
assert socket.gethostname()=='149';R=Path(__file__).resolve().parent;O=R/'semantic_relations_r1';C=R/'canonical_targets_r2'
left='CLA-P-0DB8DD7072AB6B67';right='CLA-P-C3CACF9FE71B9E70';p=pd.read_csv(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz',sep='\t');d=p.loc[p.pathway_stable_id.eq('R-HSA-8963684')].drop_duplicates('reaction_stable_id')
core={'R-HSA-71155','R-HSA-71163','R-HSA-71164','R-HSA-71173','R-HSA-71181'};assert len(d)==7 and core<=set(d.reaction_stable_id)
q=d[['reaction_stable_id','reaction_name']].copy();q['functional_core_id']=left;q['expanded_pathway_id']=right;q['membership']=np.where(q.reaction_stable_id.isin(core),'shared_catabolic_core','Reactome_extension');q['left_source']='https://www.kegg.jp/entry/N00708';q['right_source']='https://reactome.org/content/detail/R-HSA-8963684';q.to_csv(O/'CURATED_TYROSINE_CORE_REACTIONS.tsv',sep='\t',index=False)
relations=[{'left_id':left,'right_id':right,'relation':'isPartOf','shared_function':'Tyrosine conversion to fumarate and acetoacetate','extension':'Reverse transamination and an additional HPP reaction','evidence':'Five forward catalytic steps correspond to the KEGG network; Reactome additionally includes two reactions','source_left':'https://www.kegg.jp/entry/N00708','source_right':'https://reactome.org/content/detail/R-HSA-8963684'}]
pd.DataFrame(relations).to_csv(O/'CROSS_DATABASE_CURATED_RELATIONS.tsv',sep='\t',index=False)
B=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t').fillna('');sources=['MSIGDB:KEGG_MEDICUS:KEGG_MEDICUS_VARIANT_MUTATION_ACTIVATED_SMO_TO_HEDGEHOG_SIGNALING_PATHWAY','MSIGDB:KEGG_MEDICUS:KEGG_MEDICUS_VARIANT_MUTATION_INACTIVATED_PTCH1_TO_HEDGEHOG_SIGNALING_PATHWAY']
s=pd.read_parquet(R/'common_rna_scores_r1/cache/TCGA_BRCA/PATHWAY_RAW.parquet');values=s[['RNA_PATHWAY|'+x for x in sources]].to_numpy(float);valid=np.isfinite(values).all(1);delta=np.max(np.abs(values[valid,0]-values[valid,1])) if valid.any() else None;assert valid.any() and delta==0
rec={'target_host':'149','cancer':'BRCA','patient_scores_checked':int(valid.sum()),'source_definitions':sources,'maximum_score_difference':float(delta),'network_identities':['N00017','N00010'],'shared_network_context':'nt06217','source_urls':['https://www.kegg.jp/entry/N00017','https://www.kegg.jp/entry/N00010','https://www.kegg.jp/network/nt06217'],'biological_difference':'Activated SMO variant versus inactivated PTCH1 variant','shared_RNA_observation':True};(O/'HEDGEHOG_SHARED_READOUT.json').write_text(json.dumps(rec,indent=2))
# Preserve the newly verified source-to-network identity for the next master revision.
(R/'sources/KEGG_CURATED_NETWORK_XREFS.json').write_text(json.dumps({'records':[{'source_name':sources[0].split(':')[-1],'network_id':'N00017','source_url':'https://www.kegg.jp/entry/N00017','evidence':'Exact network name, variant, molecular chain and network-variation family'}]},indent=2))
sys.path.insert(0,'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(15,8));ax.set_xlim(0,15);ax.set_ylim(0,8);ax.axis('off')
blue='#285B8C';green='#138778';orange='#C47725';gray='#566272'
ax.text(.35,7.7,'Shared functional core and database-specific extensions',fontsize=19,fontweight='bold',color='#21394D')
ax.text(.35,7.25,'Tyrosine catabolism: identical five-gene membership, different reaction scope',fontsize=12,color=gray)
xs=[.8,3.3,5.8,8.3,10.8,13.4];nodes=['Tyrosine','HPP','Homogentisate','4-MAA','4-FAA','Fumarate +\nacetoacetate'];enzymes=['TAT','HPD','HGD','GSTZ1','FAH']
def row(y,title,color):
 ax.add_patch(FancyBboxPatch((.25,y-.55),14.4,1.5,boxstyle='round,pad=0.05',facecolor=color+'0D',edgecolor=color,linewidth=1))
 ax.text(.45,y+.63,title,color=color,fontweight='bold',fontsize=12)
 for x,n in zip(xs,nodes):ax.text(x,y,n,ha='center',va='center',fontsize=10)
 for i,e in enumerate(enzymes):
  ax.add_patch(FancyArrowPatch((xs[i]+.58,y),(xs[i+1]-.69,y),arrowstyle='-|>',mutation_scale=13,color=color,linewidth=1.5));ax.text((xs[i]+xs[i+1])/2,y+.20,e,ha='center',color=color,fontweight='bold',fontsize=10)
row(5.5,'KEGG N00708  |  five forward reactions',blue)
row(3.3,'Reactome R-HSA-8963684  |  shared core plus two reactions',green)
ax.add_patch(FancyArrowPatch((3.15,2.98),(.95,2.98),connectionstyle='arc3,rad=-.20',arrowstyle='-|>',mutation_scale=13,color=orange,linewidth=1.5));ax.text(1.7,2.45,'Reverse TAT reaction',color=orange,ha='center',fontsize=10)
ax.add_patch(FancyArrowPatch((3.3,3.05),(3.3,1.98),arrowstyle='-|>',mutation_scale=13,color=orange,linewidth=1.5));ax.text(3.3,1.66,'Additional HPP reaction\n(unknown enzyme)',ha='center',va='center',color=orange,fontsize=10)
ax.text(8.0,2.0,'Mapping: KEGG core is part of the broader Reactome pathway',ha='center',fontsize=12,fontweight='bold',color='#21394D')
ax.text(.4,.75,'Gene-set overlap = 1.00     Shared forward reactions = 5     Reactome-specific reactions = 2',fontsize=11,color=gray)
ax.text(.4,.22,'Sources: kegg.jp/entry/N00708  |  reactome.org/content/detail/R-HSA-8963684  |  Reactome release 98',fontsize=9,color=gray)
fig.tight_layout();out=R/'reports/figures';out.mkdir(exist_ok=True);fig.savefig(out/'03_Tyrosine_shared_core.png',dpi=300,bbox_inches='tight');fig.savefig(out/'03_Tyrosine_shared_core.pdf',bbox_inches='tight');plt.close(fig)
print('CURATION_COMPLETE',json.dumps(rec),flush=True)
