import json,socket,sys,subprocess,shutil
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';REG=BASE/'observation_registry_r1';DATA=BASE/'member_measurements_r1';REPO=ROOT/'source_integration/repo'
sys.path.insert(0,str(REPO))
from cla_engine.common import sha
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
save(REG/'FINALIZE_PREFLIGHT.json',{'target_host':'149','operation':'commit verified observation registry and M0 specification','training':False,'lncrna_association':False})
d=json.loads((REG/'DICTIONARY_UPDATE.json').read_text());assert sha(d['dictionary'])==d['sha256']
tests=json.loads((REG/'TESTS.json').read_text());assert tests['returncode']==0
receipt=json.loads((REG/'COMPLETE.json').read_text());assert len(receipt['cases'])==3
original=(DATA/'MEASUREMENT_PRODUCER.py.txt').read_text();current=(REPO/'cla_engine/measured_members.py').read_text()
assert current.split('\ndef condition_qualified_mask')[0].rstrip()==original.rstrip()
save(REG/'MEMBER_PRODUCER_LINEAGE.json',{'target_host':'149','measurement_producer_sha256':sha(DATA/'MEASUREMENT_PRODUCER.py.txt'),'current_module_sha256':sha(REPO/'cla_engine/measured_members.py'),'only_postmaterialization_addition':'condition_qualified_mask','measurement_code_unchanged':True})

p=REPO/'docs/M0_RELATION_BASELINE.rst';t=p.read_text()
old='通路 k 的读数 r 对应::\n\n    predicted_Y[i,k,r] = intercept[k,r] + C[i] gamma[k,r]\n                        + X_lnc[i] beta_lnc[k,r]\n                        + X_mRNA[i] beta_mRNA[k,r]'
new='一个有平台与方法定义的唯一观测项 a 对应::\n\n    predicted_Y[i,a] = intercept[a] + C[i] gamma[a]\n                      + X_lnc[i] beta_lnc[a]\n                      + X_mRNA[i] beta_mRNA[a]\n\n+同一抗体、位点或来源评分即使属于多条通路，也只拟合一个输出 a。\n+通路 k 通过固定归属表 A[k] 连接这些输出；不得按归属次数重复训练。'.replace('\n+','\n')
assert t.count(old)==1;t=t.replace(old,new)
old='若最终 Y 为多维测量，各读数使用自己的输出和掩码；若采用已确定的标量\n融合，则使用冻结规则产生的标量。两者均按同一通路身份组织结果，\n不能先计算 lncRNA 相关性再作为 Y。'
new='Y 按精确通路组织为多维患者观测，保留来源、单位及实测掩码。\n+条件性机制的专属解释另要求匹配的患者条件已证实存在；未知不作为阴性。\n+lncRNA 相关性不参与 Y 构建。见 MEASURED_Y_REPRESENTATION.rst。'.replace('\n+','\n')
assert t.count(old)==1;t=t.replace(old,new)
t=t.replace('平台特异读数使用不同的 r，','平台特异观测使用不同的 a，')
p.write_text(t)
p=REPO/'docs/MEASURED_Y_REPRESENTATION.rst';t=p.read_text()
t=t.replace('每个癌种中，M0 以患者为行，预测固定的观测项。','每个癌种中，M0 以患者为行，预测固定的唯一观测项。同一观测跨通路\n复用时只拟合一次，再通过固定归属表解释其通路关系。')
t+='''\n数据接口\n--------\n\ncla_engine.measured_y_registry.load_target 根据精确通路 ID 与队列返回原始\n患者观测、实测掩码、单独的条件适用性和来源定义。返回值不进行均值汇总、\n相关分析、尺度融合或预测拟合。OBSERVATIONS 定义唯一测量，\nPATHWAY_OBSERVATIONS 保存多对多归属，MATRICES 固定数值文件及校验值。\n\n来源分类从 SOURCE_TASK_ROLES 读取，和共享生物学身份的显示角色分开。\nRPPA 原始评分以内部通路 ID 命名时，通过已验证的抗体组评分定义及\n正式身份重定向解析，不能用通路名猜测来源。\n\nESTIMATE 推算纯度与 PanImmune TLS12 公开读数保留为外部参照，原始\nRNA 计算背景单独追踪；默认读取的监督观测只包含测量依赖已经核实的\n项目。缺少这些项目的计算背景不改变对应生物学身份或其他已核实观测。\n'''
p.write_text(t)
p=REPO/'README.md';t=p.read_text()
old='The old association-RMS materialization entry is disabled. Measured source\ncomponents are not a completed scalar-fusion model.'
new='The old association-RMS materialization entry is disabled. Y groups patient\nobservations under an exact pathway identity, preserving assay and readout\ndimensions. The typed registry retains unique measurements and separate\npathway membership; it does not calculate lncRNA associations. See\n[the Y representation](docs/MEASURED_Y_REPRESENTATION.rst).'
assert t.count(old)==1;p.write_text(t.replace(old,new))
p=REPO/'docs/CANCER_RELATION_TASK.rst';t=p.read_text()
t=t.replace('These include\n334 Reference, 78 Variant, 47 Pathogen and 18 environmental mechanisms.','The source classifications include\n334 Reference, 78 Variant, 47 Pathogen and 18 environmental mechanisms.\nN01392 describes arsenic exposure despite its cached Pathogen category;\nthe effective condition requirements are 46 pathogen and 19 exposure mechanisms.')
old='The exporter preserves measured inputs to Y; it does not choose\na scalar fusion method or calculate an lncRNA-by-pathway correlation matrix.'
new='The registry groups these readouts and original member measurements\nunder exact pathway identities, with typed roles, platform-specific observation\nIDs and separate condition masks. No lncRNA-by-pathway correlation matrix\nis calculated. See MEASURED_Y_REPRESENTATION.rst.'
assert t.count(old)==1;p.write_text(t.replace(old,new))
p=REPO/'docs/MEMBER_STATE_EVIDENCE.rst';t=p.read_text();needle='pathway_graph_sources.materialize_graph_sources 保存完整来源组成与绑定表。';assert t.count(needle)==1
p.write_text(t.replace(needle,needle+'\nmeasured_members 保留抗体识别组、质谱位点与逐患者值；measured_y_registry\n把这些数值和来源读数登记到固定通路，分别返回测量掩码和条件适用性。'))

prov=REPO/'provenance/measured_y_registry_20261112';prov.mkdir(exist_ok=True)
for name in ['COMPLETE.json','DICTIONARY_UPDATE.json','TESTS.json','MEMBER_PRODUCER_LINEAGE.json','READER_OBSERVATION_SUMMARY.tsv']:
    shutil.copyfile(REG/name,prov/name)
for name in ['READER_MEMBER_SUMMARY.tsv','FIGURE_RECEIPT.json','READOUT_ROLE_AUDIT.json','TLS_METHOD_RECHECK.json']:
    shutil.copyfile(DATA/name,prov/name)
for name in ['materialize_measured_members_149.py','materialize_y_registry_149.py','publish_y_registry_149.py','finalize_y_registry_149.py','plot_measured_member_coverage_149.py']:
    shutil.copyfile(WORK/name,prov/name)
for cancer in ['BRCA','LUAD','COAD']:
    shutil.copyfile(REG/cancer/'ACCESSOR_VALIDATION.json',prov/(cancer+'_ACCESSOR_VALIDATION.json'))
save(prov/'SCOPE.json',{'target_host':'149','previous_goal_turn':'progress: inspected actual job state and retracted association-based Y coverage','current_goal_turn':'progress: complete typed observation registry for three cancers, overwrite dictionary, document unique-output M0','model_training':False,'lncrna_association':False,'goal_complete':False,'remaining':['external RNA lineage','condition evidence review','reader report and completion audit']})
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip();assert not staged,staged
subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
selected=['cla_engine/measured_members.py','cla_engine/measured_y_registry.py','tests/test_measured_members.py','tests/test_measured_y_registry.py','docs/MEASURED_Y_REPRESENTATION.rst','docs/M0_RELATION_BASELINE.rst','docs/CANCER_RELATION_TASK.rst','docs/MEMBER_STATE_EVIDENCE.rst','README.md',str(prov.relative_to(REPO))]
subprocess.run(['git','add','--',*selected],cwd=REPO,check=True)
subprocess.run(['git','commit','-m','Register lncRNA-independent pathway observations and matched M0 outputs'],cwd=REPO,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
bundle=WORK/'measured_y_registry_r1.bundle'
subprocess.run(['git','bundle','create',str(bundle),'32f28a5b4819080bee98ca59bee531214dbcaf7f..HEAD'],cwd=REPO,check=True)
subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
handoff={'target_host':'149','code_commit':head,'bundle':str(bundle),'bundle_sha256':sha(bundle),'dictionary':d['dictionary'],'dictionary_sha256':d['sha256'],'registry':str(REG),'registry_receipt_sha256':sha(REG/'COMPLETE.json'),'tests_passed':10,'training':False,'lncrna_association':False,'accepted_Y':None,'goal_complete':False,'remaining':['External RNA background lineage: ESTIMATE-derived purity and PanImmune TLS12','Review event-specific patient evidence for conditional mechanisms','Reader report and full completion audit']}
save(REG/'HANDOFF.json',handoff);print(json.dumps(handoff,ensure_ascii=False,indent=2),flush=True)
