I=O/'source_inventory'; A=O/'event_conditions'; C=O/'y_contract'
inventory=json.loads((I/'SUMMARY.json').read_text())
composition=rd(I/'COMPOSITION_RESOURCE_INVENTORY.tsv')
resources=rd(I/'STATE_AND_REGULATOR_RESOURCE_INVENTORY.tsv')
events=rd(A/'EVENT_CATEGORY_COUNTS.tsv')
wnt=rd(C/'WNT_OBSERVATION_SUMMARY.tsv')
new={}

s=new['sources']=slide('收集了哪些资源：定义、评分与患者数据分开计数','来源盘点 · 项目冻结库存')
card(s,.65,1.95,3.85,3.45,'4 套组成资源','KEGG / Reactome\nWikiPathways / Hallmark\n共 2,740 份通路定义',TEAL,27,22)
card(s,4.75,1.95,3.85,3.45,'评分定义资源','DoRothEA + PROGENy：285\nPTMsigDB：624\nState：18 来源家族 / 200 读数',BLUE,25,20)
card(s,8.85,1.95,3.85,3.45,'2 个患者项目','TCGA 与 CPTAC\nTCPA 为 TCGA 提供 RPPA\nCPTAC 使用质谱',ORANGE,27,22)
tx(s,.85,5.9,11.7,.9,'通路定义库、算法／发表来源和患者项目各有用途。\n同一资源可服务多个组学，计数不直接相加。',23)

fig,ax=plt.subplots(figsize=(11.8,4.8));names=['KEGG MEDICUS','Reactome','WikiPathways','MSigDB Hallmark'];nums=[658,1839,193,50]
bars=ax.barh(names,nums,color=['#'+TEAL,'#'+BLUE,'#'+ORANGE,'#73968B']);ax.invert_yaxis();ax.set_xlim(0,2150);ax.set_xlabel('本项目收集的来源定义数')
for bar,n in zip(bars,nums):ax.text(n+25,bar.get_y()+bar.get_height()/2,f'{n:,}',va='center',fontsize=23,fontweight='bold')
fig.tight_layout();p=savefig(fig,'每库组成定义数')
s=new['library_counts']=slide('每个通路库收集多少条，最后如何连接主干','4 套组成资源 · 2,740 份定义')
pic(s,p,.5,1.85,7.0,4.7)
tx(s,7.65,1.95,5.1,4.8,'KEGG：658 → 477 份来源 → 476 个主干\n\nReactome：9 份等价定义；1,301 份组成关联覆盖 469 条主干\n\nWikiPathways：193 份组成，3 份等价；未生成独立患者评分\n\nHallmark：46 份组成关联覆盖 342 条主干',18)
tx(s,.75,6.6,11.9,.35,'数量为本项目冻结库存；等价映射与组成关联分别记录，关联条目不能当作新增主干相加。',15)

s=new['assay_counts']=slide('每层组学收集了多少评分来源','专属评分资源与共享组成定义')
rows=[('RNA','14 来源家族 / 411 定义','126 项 State + 285 项 TF／信号定义'),('蛋白','2 个实测平台','TCPA RPPA500 / CPTAC MS；复用共同组成定义'),('磷酸化','PTMsigDB 1 库 / 624 签名','KINASE-PSP 262；iKiP 313；其余通路签名 49'),('甲基化','2 来源 / 5 项 State','Malta 干性 4；甲基化白细胞比例 1'),('CNV','2 来源 / 7 项 State','ABSOLUTE 3；HRD 4'),('突变 / MSI','3 来源 / 62 项 State','SBS 60；TMB 1；MSI 1')]
for i,(layer,count,detail) in enumerate(rows):
    y=1.85+i*.73;tx(s,.8,y,1.65,.55,layer,23,TEAL,True);tx(s,2.55,y,4.05,.55,count,22,bold=True);tx(s,6.65,y,5.95,.62,detail,17)
tx(s,.8,6.5,11.8,.5,'4 套组成库可跨组学复用；RNA 的 14 来源已包含 Hallmark。此处列原始定义数，实测数量另列。',16)

s=new['regulators']=slide('调控与磷酸化来源：收集、绑定和实测有区别','TF／激酶作为成员状态读数，归入 Pathway')
for x,title,body,color in [(.7,'DoRothEA','271 份 TF 定义\n80 份绑定当前 Y\n两队列均有实测',TEAL),(4.85,'PROGENy','14 份信号定义\n1 份绑定当前 Y\n两队列均有实测',BLUE),(9,'PTMsigDB','624 份方向性签名\n276 份绑定当前 Y\nTCGA 实测 10；CPTAC 196',ORANGE)]:card(s,x,2,3.65,3.5,title,body,color,25,21)
tx(s,.85,5.95,11.6,.9,'PTMsigDB 绑定：137 份 PSP + 139 份 iKiP 激酶签名。\n“实测”指三个 demo 癌种中至少一名患者有值；不表示所有患者均覆盖。',20)

state=resources[~resources.resource.isin(['DoRothEA','PROGENy'])].copy().sort_values('input_readout_definitions',ascending=False)
labelmap={'ABSOLUTE / PanCanAtlas':'ABSOLUTE / PanCanAtlas','CIBERSORT / LM22':'CIBERSORT / LM22','COSMIC SBS / MutationalPatterns':'COSMIC SBS / MutationalPatterns','CYT / Rooney':'CYT / Rooney','HRD / PanCanAtlas':'HRD / PanCanAtlas','IPS / Immunophenogram':'IPS / Immunophenogram','MSIsensor2 / GDC':'MSIsensor2 / GDC','MSigDB Hallmark (State readout)':'Hallmark State','Malta stemness / PanCanAtlas':'Malta 干性','Methylation leukocyte fraction / PanCanAtlas':'甲基化白细胞比例','Mutation burden / PanCanAtlas':'TMB / PanCanAtlas','Panimmune 160 signatures':'Panimmune 来源','SenMayo':'SenMayo','signifinder':'signifinder','xCell':'xCell'}
fig,axs=plt.subplots(1,2,figsize=(12.8,5.1))
for ax,sub in zip(axs,[state.iloc[:9],state.iloc[9:]]):
    names=[labelmap.get(v,v) for v in sub.resource];ns=sub.input_readout_definitions.astype(int).tolist();bars=ax.barh(names,ns,color='#'+TEAL);ax.invert_yaxis();ax.set_xlim(0,max(ns)*1.35);ax.tick_params(axis='y',labelsize=12);ax.set_xlabel('原始读数定义数',fontsize=13)
    for bar,n in zip(bars,ns):ax.text(n+max(ns)*.025,bar.get_y()+bar.get_height()/2,str(n),va='center',fontsize=15,fontweight='bold')
fig.tight_layout(w_pad=2.5);p=savefig(fig,'State来源逐项数量')
s=new['state_sources']=slide('State 的 200 项原始读数来自哪些资源','18 个方法／发表来源家族 · 与 48 个主入口不同')
pic(s,p,.55,1.85,12.2,4.8);tx(s,.8,6.6,11.8,.4,'计数按来源定义；细胞组成与 SBS 子项保留。来源名不替代计算方法，具体算法随观测列记录。',15)

s=new['wnt']=slide('WNT 的层级如何影响 Y：先固定精确主干','真实例子 · KEGG N00056 · 46 个成员')
card(s,.65,1.95,4,3.65,'精确主干','WNT signaling pathway\n46 个蛋白固定\n成员实测进入 Y\n等价来源读数保留',TEAL,25,22)
card(s,4.9,1.95,3.85,3.65,'成员状态','LEF1 / MYC / TCF7 / TCF7L2\n4 个 TF 读数\nGSK3B 的 PSP / iKiP 读数\n分别保留方法与缺失',BLUE,25,18)
card(s,9,1.95,3.65,3.65,'其他层级','WNT 配体生成与运输\nTCF 依赖的下游反应\n保留组成和结构连接\n整体评分不混入本主干',ORANGE,25,18)
tx(s,.8,6.05,11.8,.8,'训练使用精确主干组织的多输出 Y。\n相似子通路影响图的结构与解释范围，不自动扩大主干成员。',23,bold=True)

s=new['wnt_scope']=slide('一个具体例子：26 个来源蛋白如何处理','Reactome WNT ligand biogenesis and trafficking')
card(s,.7,1.95,5.8,2.3,'18 个共享蛋白','属于主干 46 个成员\n对应测量按原主干归属进入 Y',TEAL,29,23)
card(s,6.9,1.95,5.7,2.3,'8 个额外蛋白','PORCN、SNX3、TMED5、VPS26A\nVPS29、VPS35、WLS、WNT11',ORANGE,29,20)
tx(s,.85,4.65,11.7,.9,'额外蛋白留在来源组成与结构连接中；该 Reactome 整体评分不属于主干 Y。',24,bold=True)
tx(s,.85,5.85,11.7,.95,'WNT · BRCA 实际维度：\nTCGA：147 个成员测量 + 8 个来源读数 = 155；CPTAC：251 + 9 = 260。',22)

s=new['m0']=slide('线性 M0：对同一主干的各个观测维度分别预测','后续模型设计 · 相同 Y、掩码与患者分折')
tx(s,.85,1.85,11.7,.7,'ŷᵢⱼ = aⱼ + Xᵢ βⱼ + Cᵢ γⱼ',31,TEAL,True)
card(s,.7,2.9,4,2.9,'X：三组比较','lncRNA；mRNA；lncRNA + mRNA\n同一癌种、同批有效患者\nLasso 仅惩罚 RNA 系数 β',TEAL,24,19)
card(s,4.9,2.9,3.85,2.9,'C：协变量','临床、独立纯度等\n系数 γ 不受 Lasso 惩罚\n若协变量来自 Y，则排除',BLUE,24,20)
card(s,8.95,2.9,3.7,2.9,'j：观测维度','成员蛋白／位点等\n每个来源、平台单独输出\n仅有实测标签的患者拟合',ORANGE,24,20)
tx(s,.8,6.3,11.8,.55,'标准化与正则强度只用训练数据确定；蛋白／磷酸化层的输出头分别保留 TCPA 与 CPTAC 的尺度。',17)

s=new['hhgt']=slide('HHGT：图编码器变化，预测的 Y 保持相同','后续模型设计 · 患者条件下的分子状态预测')
card(s,.65,2.05,3.55,3.6,'患者输入','与线性对照相同的可用特征\nlncRNA / mRNA / 其他组学\n非 RNA 目标：屏蔽目标层',TEAL,25,20)
arrow(s,4.35,3.55,.4)
card(s,4.9,2.05,3.55,3.6,'异质图编码','lncRNA → 蛋白／位点 → 主干\nRPI、PPI、反应／层级边\n患者特征进入节点表示',BLUE,25,19)
arrow(s,8.6,3.55,.4)
card(s,9.15,2.05,3.55,3.6,'分子／来源解码','预测与 M0 相同的观测列\nTCPA / CPTAC 独立输出头\n使用相同实测与条件掩码',ORANGE,25,20)
tx(s,.8,6.1,11.8,.75,'图的收益：固定输入层与 Y，比较线性模型、HHGT、打乱边的 HHGT。\n图边提供结构；患者实测值提供监督。',21,bold=True)

s=new['loss']=slide('两个模型如何使用这些 Y：同一个监督口径','后续模型设计 · 层级决定归属，不生成平均活性')
tx(s,.8,1.85,11.8,.7,'L = LPathway + 0.5 LState + 模型正则项',30,TEAL,True)
card(s,.7,2.95,3.85,2.9,'先确定有效标签','患者有实测值\n读数允许用于监督\n机制适用条件为 1',TEAL,24,22)
card(s,4.75,2.95,3.85,2.9,'再计算预测误差','来源／组学内标准化\n分层平均有效观测误差\n共享列按唯一 ID 计一次',BLUE,24,21)
card(s,8.8,2.95,3.85,2.9,'最后按主干汇报','各层各自的预测表现\n主干汇总保留分层结果\n不平均成跨平台活性值',ORANGE,24,21)
tx(s,.8,6.22,11.8,.7,'非 RNA 预测屏蔽目标层及其衍生特征。RNA 读数按重建解释，隐藏标签自身及计算它所需的 mRNA；lncRNA 仍可作输入。',17)

s=new['relation']=slide('如何得到“癌种 × lncRNA × 精确通路”关系','从预测分子状态到通路关联 · 后续分析设计')
for x,title,body,color in [(.65,'拟合实测 Y','线性与 HHGT 共用标签\n预测通路成员及来源读数',TEAL),(4.9,'检验 lncRNA 信息','在留出患者上比较预测\n移除 lncRNA 后重新拟合\n检验误差变化与稳定性',BLUE),(9.15,'汇总到精确主干','按既定成员归属汇总\n展示蛋白／磷酸化等分层证据\n输出主干名称及相关 lncRNA',ORANGE)]:card(s,x,2,3.55,3.8,title,body,color,23,20)
tx(s,.85,6.15,11.6,.7,'Y 的定义不使用 lncRNA 相关系数。单条 lncRNA 的模型贡献和总体 lncRNA 组增益分别报告。',22,bold=True)

fig,ax=plt.subplots(figsize=(12,4.8));ee=events.sort_values('pathway_count',ascending=True);bars=ax.barh(ee.event_family_zh,ee.pathway_count,color='#'+ORANGE);ax.set_xlim(0,55);ax.set_xlabel('需要该类患者事件证据的主干条目数')
for bar,n in zip(bars,ee.pathway_count):ax.text(n+.6,bar.get_y()+bar.get_height()/2,str(n),va='center',fontsize=20,fontweight='bold')
fig.tight_layout();p=savefig(fig,'141条机制条件分类')
s=new['event_categories']=slide('141 条究竟依赖什么事件？','8 类患者事件 · 不等于 141 种突变')
pic(s,p,.6,1.85,7.3,4.9)
tx(s,8.1,2,4.7,4.9,'扩增：MYCN、FGFR、CCND1\n\n突变：RET / PI3K 激活、FLT3 重复\n\n融合：BCR–ABL、IGH–MMSET\n\n病原：EBV LMP1、HTLV-1 p12\n\n暴露：NNK / NNN、金属等',20)

s=new['event_evidence']=slide('这些条件是真的没有，还是没有接入？','实测文件与事件判定的区别')
card(s,.7,1.95,3.85,3.9,'32 条有连续数据基础','17 条扩增：已有 CNV\n15 条表达变化：已有 RNA\n\n需要匹配患者并明确\n扩增／过表达判定规则',TEAL,23,21)
card(s,4.75,1.95,3.85,3.9,'44 条需要事件明细','变异 20 + 功能丧失 4\n融合／重排 19 + ER 阳性 1\n\n现有目录含变异／融合资源\n基因突变标志不足以代替',BLUE,23,20)
card(s,8.8,1.95,3.85,3.9,'65 条缺匹配检测','病原／效应物 46\n配体／化学暴露 19\n\n宿主表达或吸烟史\n不能替代具体事件证据',ORANGE,23,21)
tx(s,.8,6.2,11.8,.7,'141 条目前没有接入患者事件表；不能据此判定所有癌种均没有条件。现有官方目录还包含十癌事件资源。',20,bold=True)

s=new['current_scope']=slide('本轮监督范围：335 条主干，141 条保留事件注释','实际数据结果 · 不把未知条件当作阳性标签')
card(s,.7,2.05,5.8,3.5,'335 条 Pathway','存在有效成员／来源观测\n当前条件允许进入监督候选\n仍按每列实测与训练折有效样本拟合',TEAL,31,23)
card(s,6.9,2.05,5.7,3.5,'141 条事件机制','退出本轮可监督目标清单\n保留原机制、成员与数据来源\n具体事件得到确认后逐项恢复',ORANGE,31,23)
tx(s,.85,5.95,11.7,.9,'State：TCGA 有读数 45 项，CPTAC 33 项；3 个概念尚无独立读数。\n永久删除依据是计划癌种确实无法观测；目前证据不支持删除全部 141 条。',21)

s=new['closing']=slide('主干、来源、Y 与模型如何衔接','最终结构与本轮交付')
card(s,.7,2,3.85,3.45,'目录与来源','476 个主干身份\n48 个 State 主入口\n来源原始成员与读数可展开',TEAL,27,22)
card(s,4.75,2,3.85,3.45,'固定的患者 Y','6 份原始观测矩阵\n本轮可监督主干 335 条\n缺失与事件条件逐项保留',BLUE,27,22)
card(s,8.8,2,3.85,3.45,'模型设计','M0：逐观测线性预测\nHHGT：图编码 + 观测解码\n相同 Y 检验 lncRNA 的增量',ORANGE,27,22)
tx(s,.85,5.9,11.7,.95,'主干界定“研究哪个生物学过程”；各组学观测说明“预测该过程的哪些分子状态”。\n最终按癌种、lncRNA 与精确主干组织预测关联结果。',22,bold=True)

# Original figures are kept; replace ambiguous WNT, scope, model and final-summary pages.
CONTENT_ORDER=[original[0],original[1],new['sources'],new['library_counts'],new['assay_counts'],new['regulators'],new['state_sources'],original[2],original[3],original[4],new['wnt'],new['wnt_scope'],original[6],original[7],original[8],original[9],original[10],original[11],original[12],original[13],new['event_categories'],new['event_evidence'],new['current_scope'],new['m0'],new['hhgt'],new['loss'],new['relation'],new['closing']]
