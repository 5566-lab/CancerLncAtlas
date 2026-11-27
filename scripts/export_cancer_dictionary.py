"""Export the cancer mechanism dictionary from verified catalog files on host 149."""
from pathlib import Path
import argparse,csv,gzip,json,hashlib,socket
import xlsxwriter

def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as h:
        yield from csv.DictReader(h,delimiter='\t')

def export(catalog,backbone,source_tables,output,validation=None):
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    C,B,S,O=map(Path,(catalog,backbone,source_tables,output))
    O.parent.mkdir(parents=True,exist_ok=True)
    (O.parent/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'reader_dictionary_export'})+'\n')
    master={r['canonical_id']:r for r in rows(C/'TARGET_MASTER.tsv')}
    policy={r['canonical_id']:r for r in rows(C/'TARGET_POLICY.tsv')}
    active={cid for cid,p in policy.items() if p['active_in_analysis']=='True'}
    prim={r['canonical_id']:r for r in rows(B/'KEGG_CANCER_BACKBONE.tsv')}
    decisions={r['network_id']:r for r in rows(B/'SOURCE_SCOPE_DECISIONS.tsv')}
    refs=json.loads((B/'EVIDENCE_REFERENCES.json').read_text())
    manifest=json.loads((B/'MANIFEST.json').read_text())
    basis=manifest['evidence_basis_counts'];kinds=manifest['network_type_counts']
    source_roles={r['source_target_id']:r for r in rows(C/'SOURCE_TASK_ROLES.tsv')}
    memberships=list(rows(C/'ALL_GROUP_MEMBERSHIP.tsv'))
    groups={r['protein_cluster_id'] for r in memberships if r['integrated_canonical_id'] in active}
    nodes={r['node_id'] for r in memberships if r['protein_cluster_id'] in groups}
    definitions=list(rows(C/'DEFINITION_TO_IDENTITY.tsv'))
    node_ids={r['node_id'] for r in definitions if r['integrated_canonical_id'] in active}|nodes
    state_count=sum(p['analysis_role']=='state_auxiliary' for p in policy.values())
    wb=xlsxwriter.Workbook(O,{'constant_memory':True,'strings_to_formulas':False,'strings_to_urls':False,'nan_inf_to_errors':False})
    header=wb.add_format({'bold':True,'bg_color':'#17365D','font_color':'white','text_wrap':True,'valign':'top','font_size':11})
    cell=wb.add_format({'font_name':'Arial','font_size':10,'valign':'top'})
    counts={}
    def sheet(name,records,fields=None):
        records=iter(records);first=next(records,None)
        if fields is None:fields=list(first) if first is not None else ['内容']
        sh=wb.add_worksheet(name);sh.freeze_panes(1,0);sh.set_row(0,32)
        sh.set_column(0,len(fields)-1,24,cell)
        for j,key in enumerate(fields):
            if any(x in key.lower() for x in ('name','description','definition','rationale','interpretation','mechanism','说明','机制','证据','名称','方法')):sh.set_column(j,j,48,cell)
        sh.write_row(0,0,fields,header)
        n=0
        def write(row):
            nonlocal n
            n+=1
            if n>=1048576:raise ValueError('Excel row limit:'+name)
            sh.write_row(n,0,[row.get(k,'') for k in fields],cell)
        if first is not None:write(first)
        for record in records:write(record)
        if n:sh.autofilter(0,0,n,len(fields)-1)
        counts[name]=n
    notes=[
      ('研究单位','癌种 × lncRNA × 精确通路。以 BRCA、LUAD、COAD 为演示范围。'),
      ('癌症主干',f'{len(prim)} 个 KEGG NETWORK 精确机制；来源为 MSigDB KEGG_MEDICUS 2025.1.Hs，保留 N 编号、原始成员及条件。'),
      ('纳入依据',f"{basis['official_cancer_class']} 个命中 KEGG 癌症分类；{basis['official_malignant_disease']} 个另有明确恶性肿瘤疾病注释；{basis['curated_mechanism_evidence']} 个有经核对的癌症机制文献。"),
      ('机制颗粒度','NETWORK 条目包含反应模块、信号分支、突变、病原体及暴露机制，不等同于同数量的 KEGG 大通路图。'),
      ('命名','KEGG 主干使用官方 NETWORK 名称与 N 编号；其他来源保留正式名称和 ID。生物学身份 ID 保持稳定。'),
      ('跨库整合','正式等价映射支持身份归并；蛋白组成相似支持机制关联。父子关系、共同组成及特定扰动机制分别保留。'),
      ('主干对齐','在冻结的蛋白背景中检验组成富集，全部候选对执行 BH 校正；至少 3 个共同蛋白且 q≤0.05 时建立上下文关系。TF／激酶另以已核实的分子身份检查主干成员关系，不对单个调控蛋白使用多基因富集门槛。'),
      ('蛋白列表聚类','保留 complete-linkage 聚类及成员；同组任意两条定义的 Jaccard≥0.5。相似组用于比较组成，不自动产生一个新通路身份。'),
      ('组成相似度','Jaccard=共同蛋白数/蛋白并集数。source_coverage、target_coverage 分别表示共同蛋白占两端成员的比例。'),
      ('Pathway 分类','TF 调控程序、激酶底物程序和 PROGENy 信号通路读数均归 Pathway；它们提供机制信息，分类本身不产生额外训练损失。'),
      ('State',f'{state_count} 个 State 生物学身份；评分定义与分子组成单列。PROGENy 缺氧与 Buffa 缺氧保留来源读数的各自类别。'),
      ('条件性机制',f"{kinds['Reference']} 个 Reference 机制；{len(prim)-kinds['Reference']} 个突变、病原体或暴露机制。解释后者需要匹配的条件证据，宿主蛋白读数本身不能证明该条件。"),
      ('关系 Y' ,f'Y 描述 lncRNA 与精确机制读数的队列内关联证据；不同平台的患者分数保留来源尺度。{len(prim)} 是主干定义数，不是每个癌种均有可观测标签的数量。'),
      ('患者读数','各行列明数据平台、测量层、实际算法、单位及来源；蛋白丰度、磷酸化与 RNA 分别保留。'),
      ('组成与功能方向','共同蛋白或蛋白丰度增加不自动表示通路激活；激酶底物、功能位点和原始反应方向按来源保留。'),
    ]
    sheet('阅读说明',[{'条目':k,'说明':v} for k,v in notes])
    def bb_rows():
        for cid,b in prim.items():
            d=decisions[b['network_id']]
            yield {'NETWORK_ID':b['network_id'],'生物学身份ID':cid,'通路名称':b['network_name'],'机制类型':b['network_type'],
              '机制定义':d['mechanism_definition'],'展开反应':d['expanded_definition'],'生物学过程':b['biological_family'],
              '癌症证据类型':{'official_cancer_class':'KEGG 癌症分类','official_malignant_disease':'KEGG 恶性肿瘤疾病注释','curated_mechanism_evidence':'癌症机制文献'}[b['cancer_evidence_basis']],
              '癌症相关依据':b['cancer_rationale'],'证据':b['cancer_evidence'],'疾病注释':d['disease_annotations'],
              '适用条件':{'Reference':'一般生物学过程','Variant':'匹配的遗传改变','Pathogen':'匹配的病原体或病原体蛋白','Env factor':'匹配的环境暴露'}[b['network_type']],
              '原始基因数':int(b['original_gene_count']),'原始基因集':b['source_target_id'],'来源版本':b['metadata_archive_version'],'原始条目':'https://www.kegg.jp/entry/'+b['network_id']}
    sheet('KEGG癌症主干',bb_rows())
    if validation is not None:
        report=json.loads(Path(validation).read_text())
        sheet('Relation label coverage',(
            {'cancer':r['cancer_id'],'context':r['context'],
             'primary_definitions':r['primary_definitions'],
             'reference_definitions':r['unconditional_primary_definitions'],
             'conditional_definitions':r['conditional_definitions'],
             'primary_targets_with_observed_relation':r['primary_observed_targets'],
             'state_targets_with_observed_relation':r['state_observed_targets'],
             'observed_relations':r['observed_relations']}
            for r in report['results']))
    def catalog_rows(role):
        for cid,r in master.items():
            if cid not in active or (role=='State')!=(policy[cid]['analysis_role']=='state_auxiliary'):continue
            yield {k:r[k] for k in ['canonical_id','display_name','roles','entity_type','authority_id','source_database','merged_source_databases','integrated_protein_union_count','shared_protein_core_count']}|{'主干NETWORK_ID':prim.get(cid,{}).get('network_id','')}
    sheet('Pathway与机制',catalog_rows('Pathway'));sheet('State',catalog_rows('State'))
    def defs():
        for r in definitions:
            if r['node_id'] not in node_ids:continue
            cid=r['integrated_canonical_id'];p=policy.get(cid)
            if p:r=dict(r,roles=p['roles'],entity_type=p['biological_type'])
            source=source_roles.get(r['authority_id'])
            if source:r=dict(r,roles=source['display_category'])
            yield r
    sheet('来源定义与身份',defs())
    sheet('主干组成关联',(r for r in rows(C/'BACKBONE_ALIGNMENT.tsv') if r['admitted_context_link']=='True' and r['source_canonical_id'] in active),
      ['source_canonical_id','source_name','source_database','target_canonical_id','network_id','target_name','target_network_type','source_protein_count','target_protein_count','shared_proteins','jaccard','source_coverage','target_coverage','overlap_p','overlap_q','relation','alignment_evidence_basis','direct_regulator_proteins'])
    sheet('代表性主干关联',(r for r in rows(C/'BACKBONE_ALIGNMENT_TOP3.tsv') if r['admitted_context_link']=='True' and r['source_canonical_id'] in active),
      ['source_canonical_id','source_name','source_database','network_id','target_name','shared_proteins','jaccard','source_coverage','target_coverage','overlap_q','relation','direct_regulator_proteins'])
    sheet('蛋白相似组',(r for r in rows(C/'GROUP_MASTER.tsv') if r['protein_cluster_id'] in groups))
    sheet('相似组成员',(r for r in memberships if r['protein_cluster_id'] in groups))
    proteins=set()
    def compositions():
        for r in rows(C/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz'):
            if r['node_id'] in node_ids:
                proteins.add(r['uniprot_id']);yield r
    sheet('来源蛋白组成',compositions())
    sheet('蛋白注释',(r for r in rows(S/'PROTEIN_ANNOTATION.tsv') if r['uniprot_id'] in proteins))
    sheet('组核心与扩展蛋白',(r for r in rows(C/'GROUP_PROTEINS.tsv.gz') if r['protein_cluster_id'] in groups))
    sheet('正式跨库关系',(r for r in rows(C/'CURATED_EXTERNAL_RELATIONS.tsv') if r['source_canonical_id'] in active or r['target_canonical_id'] in active))
    sheet('功能范围关系',(r for r in rows(C/'FUNCTIONAL_SCOPE_DECISIONS.tsv') if r.get('node_a') in node_ids or r.get('node_b') in node_ids))
    sheet('跨库组成关系',(r for r in rows(S/'CROSS_DATABASE_RELATIONS.tsv.gz') if r.get('node_a') in node_ids and r.get('node_b') in node_ids))
    sheet('父子通路层级',(r for r in rows(S/'IDENTITY_HIERARCHY.tsv') if r['child_id'] in active or r['parent_id'] in active))
    sheet('来源ID绑定',({k:r.get(k,'') for k in ['source_target_id','source_column','canonical_id','source_name','source_database','relation','identity_evidence','display_category']} for r in source_roles.values() if r['canonical_id'] in active))
    def readouts():
        for r in rows(S/'SCORE_DEFINITIONS.tsv'):
            source=source_roles.get(r['source_readout_id'],{})
            cid=source.get('canonical_id',r['canonical_id'])
            if cid not in active or r['cancer_id'] not in {'BRCA','LUAD','COAD'}:continue
            b=prim.get(cid,{})
            yield dict(r,canonical_id=cid,display_category=source.get('display_category',policy[cid]['roles']),
                       measurement_scope='Host-component readout' if b.get('network_type','Reference')!='Reference' else 'Source-defined biological readout')
    sheet('患者读数定义',readouts())
    sheet('评分定义与测量层',(r for r in rows(C/'STATE_SCORE_DEFINITIONS.tsv') if r['canonical_id'] in active))
    sheet('来源反应组成',(r for r in rows(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz') if r['canonical_id'] in active))
    sheet('分子签名成员',(r for r in rows(S/'STATE_SIGNATURE_COMPONENTS.tsv.gz') if r['canonical_id'] in active))
    sheet('来源生物学范围',rows(C/'SOURCE_SCOPE_DESCRIPTIONS.tsv'))
    sheet('阈值敏感性',rows(S/'CLUSTER_THRESHOLD_SENSITIVITY.tsv'))
    used={p for f in json.loads((B/'CURATION.json').read_text())['families'] for p in f['pmids']}
    sheet('癌症机制文献',({'PMID':p,'Title':refs[p]['title'],'Abstract':refs[p]['abstract'],'URL':refs[p]['source_url']} for p in sorted(used)))
    wb.close()
    receipt={'target_host':'149','catalog':str(C),'backbone':str(B),'output':str(O),'sheets':counts,
       'sha256':hashlib.sha256(O.read_bytes()).hexdigest(),'bytes':O.stat().st_size,'sealed_test_arrays_read':False}
    (O.parent/'DICTIONARY_EXPORT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['catalog','backbone','source_tables','output']:p.add_argument('--'+key.replace('_','-'),required=True)
    p.add_argument('--validation')
    a=p.parse_args();print(json.dumps(export(a.catalog,a.backbone,a.source_tables,a.output,a.validation),ensure_ascii=False,indent=2))
