# CancerLncAtlas V3.2 变更说明

> 阅读提示（2026-09-28）：以下 `CODE_ONLY` 和授权描述是 2026-08-25 的历史快照。当前设计复核及运行分支识别见文末「2026-09-28 — 历史设计复核与本轮决策」，不能用旧状态覆盖后续授权与实际运行记录。

版本：`3.2.0`  
状态：`CODE_ONLY / NOT TRAINED / NOT RELEASED`  
日期：2026-08-25

V3.2 是新的代码 lineage，不是 V3.1 cap1000/cap2000 的继续训练。旧 V3.1 目录、checkpoint、概率、校准器、任务完成状态和网站物化结果保持不可变，但都不能导入 V3.2。V2.9 现有网站主线也不会在本阶段被替换。

## 核心任务修正

- 最终目标固定为 `cancer × lncRNA × exact pathway` 的关联成员概率和排名。
- `pathway_family` 降为层级上下文与解释字段，禁止作为最终标签或向 exact pathway 广播概率。
- 主模型从“把 LOCO 输出解释为目标癌种特异 Gene Set”改为 pooled 33-cancer、patient-fold-local、cancer-conditioned 学习；LOCO只保留为以后可能的迁移消融。
- 患者连续 pathway activity、Gene Set membership、regulatory evidence 和临床结局继续作为不同 estimand，禁止混排。

## 数据和泄漏修正

- lncRNA 来源恢复到完整 GENCODE v26 稳定 ID 宇宙 16,889 条。
- 共享集合按 `logCPM>0`、癌内检出率 ≥10%、至少 3 个正式癌种重算，期望为 4,712 条。
- 仅在 1–2 个癌种合格的 lncRNA 进入相应 cancer-local 分支，不再因未达到泛癌门槛而从网站候选中全局删除。
- 旧 7,066/2,751 及 0.10/5-cancer、CPM≥1/20% 规则均标记为 superseded diagnostic，不能续训或发布。
- 每个患者外层折的 activity 参数、association、candidate、图上下文和预处理只能由训练患者建立。
- pair-level 文献、药物、interaction、perturbation 和单细胞证据在主模型 train/validation/test 中 100% 屏蔽；证据置信度独立输出。

## 模型和比较修正

- 新增同候选、同标签、同患者折、同评价行的纯 L1 logistic LASSO 和 L2 logistic Ridge。
- 历史 `${PRIVATE_WORK_ROOT}/code` 的 `alpha=0.5` 连续 Elastic Net 不再冒充本任务的 LASSO 基线。
- CC-HHGT decoder 显式表达 lncRNA×pathway、lncRNA×cancer 和 pathway×cancer 交互。
- 自由 Evidence MoE 不进入主头；最终 logit 使用有界残差：`z_final = z_L1 + lambda_c,p * tanh(delta_CC-HHGT)`。
- `lambda=0` 必须精确回退 LASSO；回退比例和残差贡献必须可见，不能只报告融合概率。
- 低成本 pilot 只保留 exact-pathway membership head，不训练 V2.9 state 辅助头。

## Gene Set 驱动的亚型

- 亚型分类发生在 Gene Set 生成之后，而不是模型输入之前。
- 对每条 exact pathway 独立比较不同癌种的 lncRNA 组成和排名，允许不同 pathway 产生不同癌种分组。
- 相似度固定为 50% RBO 与 50% probability-weighted Jaccard；主分析使用共同 4,712 条，local-only 只做敏感性分析。
- `K=1..4` 由 silhouette、bootstrap ARI 和最小类大小门禁选择；证据不足时输出 `UNAVAILABLE` 或连续异质性，不强制 K=3。
- 稳定的通路亚型再按 pathway family 等权汇总；癌症程序层搜索 K=2–6，未通过稳定性门禁时退回 K=1/连续相似图。
- 本轮亚型不得回流同一轮模型，避免以自身输出构造输入的循环自证。

## 运行与费用保护

- 默认配置为 `execution_mode=CODE_ONLY`、`training_authorized=false`、`paid_enabled=false`、付费时长和费用上限均为 0。
- 训练入口必须同时校验 `--allow-training`、授权文件及代码/配置/输入/任务哈希；当前阶段不创建授权文件。
- 计划任务仅为本地五个患者折、一个 seed `20260726`、一个 CC-HHGT 架构，全部保持 `BLOCKED`。
- 149 只定义为未来 CPU 资产/基线/报告端，本阶段不连接或提交；本地 CUDA 不初始化；付费端无任务、费用为 ¥0。
- task key 和 owner 唯一，禁止多端重复运行；跨硬件不得直接 resume optimizer/checkpoint。

## 当前阶段能够与不能说明的结论

代码和合成测试通过后只能说明：接口、泄漏门禁、输出 schema、残差恒等式、subtype 逻辑和 dry-run 编排符合 V3.2 合同。

当前不能说明：

- CC-HHGT 优于 LASSO、Ridge、V2.9 或任何历史 GNN；
- 某条 pathway 跨癌保守或存在稳定亚型；
- 五折、三 seed 或 33 癌种正式结果已经完成；
- V3.2 可以替换网站；
- 历史 cap checkpoint 可以继续训练。

真实训练和模型效果评价必须等待用户审查 pipeline 后另行授权。

## 2026-09-28 — 历史设计复核与本轮决策

记录类型：历史证据复核、解释纠正和后续评价决策；不是新架构实现或新实验结果。本次只更新文档，不修改训练代码、输入、checkpoint、标签、超参数或实验范围。用户明确正在运行的 G2 由另一个 agent 管理，本次不接管其运行、实例和收尾工作。

### 1. 已核对的历史修改与结果

| 日期 / 分支 | 原始记录 | 本轮可用的结论 |
|---|---|---|
| 2026-08-12，三癌 RNAss patient-fold 分解 | `abs(train partial rho)` median AUPRC 0.710；旧 patient-native 0.640；MoE 0.597；P0–P3 图方案约 0.417–0.424。[H1] | 部分图方案弱于表达信号的失败确实存在；任务是 RNAss，不能充当当前 exact-pathway G2 的性能结果。 |
| 2026-08-22，三癌连续 activity OOF | EvidenceLayered CC-HHGT macro R² 0.654841；匹配 LASSO 0.643418；Ridge 0.654145；旧 compact HHGT 0.657464。[H2] | 曾优于该 LASSO，但未证明优于 Ridge/旧 compact HHGT；不能说连续头普遍更差，也不能把 R² 与关联分类 AUPRC 混排。 |
| 2026-08-22，10-PC 残差与独立 pair 审计 | Residual HHGT R² 0.240295，Ridge 0.282334；E4 perturbation/rescue 每癌仅 4–6 阳性，top-100 均零命中。[H3] | 没有支持复杂图方案稳定占优或已验证真实功能发现的证据；这是 PC-only 调整，不能声称已完整控制 purity/cell composition。 |
| 2026-08-25，V3.2 任务和模型修改 | 从 LOCO/其他任务转为 patient-fold-local、cancer-conditioned exact-pathway 关联；自由 MoE 改为零初始化的有界 L1 残差，显式输出 gate、残差和回退。[H4] | 有界残差是早已记录的设计，不是本次临时补救。理由是稳定训练和可审计的基线回退，不是数学上禁止取得明显增益。 |
| 2026-08-25，旧 V3.2 一 seed 五折 | 旧报告 AUPRC：CC-HHGT 0.448534，L1 0.301776，Ridge 0.300805；5/5 折超过两条基线。[H5] | 历史存在正结果记录；但该记录后来被限定为历史诊断，不能继承为当前修正后的正式成绩。 |
| 2026-08-29—09-06，fresh G012 / R7 | 文档记录旧 prepared 对应代码树无法从不可变快照重建、偏相关自由度实现不一致、held-out direction 错沿用训练 effect、test 在架构锁定前已暴露；要求新准备链和明确评估边界。[H6] | 重做的原因包含统计正确性、来源可追溯性和评估有效性；不能直接断言旧正增益全部由泄漏造成，也不能再将旧 test 称为 untouched test。 |
| 2026-09-04，单细胞范围冻结 | 23 癌种独立单细胞 overlay，10 癌种 typed unavailable；zero fusion weight，不改变主 HHGT 分数。[H7] | 主头不融合单细胞符合已冻结方案。单细胞独立结果的保留/发布按其自身版本记录核验，不自动要求随 G2 重训。 |
| 2026-09-08 / 09-11，R6.2 Full PyG 诊断 | 9/11 的 G2/Fold 0、216 steps/24 epochs：pathway macro-AP 0.5450282 vs LASSO 0.5449443，NO_GO_WITHIN_DIAGNOSTIC_BUDGET；此前 9/8 诊断也几乎持平。[H8] | 另一实现分支确有近零增益，不能以增加复杂度保证性能；其任务、候选、基线及预算不同，不能直接替代当前 G2 五折结果。 |
| 2026-09-21—09-28，RBP / 图 C 修改 | Git `244f556`/`522f413` 引入实验类型和 typed binding；`cbd6ba2` 修正 ENCODE reproducible peaks/exon/context；`4e1ff50` 构建 v2。9/28 授权代码使用全局 physical binding、图 C overlay 与 G2-only 五折。[H9] | 本次主要修正图证据语义和实现，未更换 exact-pathway 关联主头。图 C 的 A/B/C 命名与 G0/G1/G2 图变体是不同维度。 |

### 2. 当前模型的准确含义与需要撤回的推断

- 当前配置标识为 `CancerLncAtlas_V3.2_C_GLOBAL_BINDING_G2`，运行 ID 为 `v32-g012-g2-c-global-binding-20260928-r1`；主模型是 `cc_hhgt_bounded_l1_residual`，`external_router`、`modalities: []`。这是 R7/G012 代码路线上的图 C 修正，不能因目录中的 `r1/r2` 或历史账本而改称 R6.2 Full PyG。[C1]
- 标签是 lncRNA 表达与连续通路活性的癌内、协变量调整关联经效应量/FDR规则得到的 proxy label；不是把患者的通路活性直接二分为开/关。该主任务衡量表达关联的重现，不能单凭它宣称因果调控。[C2]
- `z_final = z_L1 + graph_gate * tanh(raw_residual)` 为实际公式。`tanh` 限幅和门控确实限制贡献，但 `1e-4 * mean(raw_residual²)` 是软正则。不能据此推出“模型被锁死”“不可能明显超过 LASSO”或“必然只能获得微小 AUPRC 增益”。例如基线概率 0.5 对应 logit 0，正负 1 的 logit 改变量对应约 0.731/0.269；这是数学尺度说明，不是本轮测得的贡献。[C3]
- 代码具备不可用/不获准时精确回退功能，不等于当前有大量行实际触发回退。回退比例、gate 分布、raw residual 饱和程度和最终残差大小需要从结果核验后报告，不能用接口能力替代实测。[C3]
- 当前 `validation_loss` 是 membership nnPU risk、加权 direction BCE 与残差正则的组合；其下降不是 AUPRC 提高、更不是战胜 LASSO 的证据。[C4]
- 本任务 LASSO 是 L1 logistic 关联分类基线，包含训练来源的 effect/FDR、检出率、样本数、跨癌统计以及身份/交互特征；不是只有一列表达量，也不是患者连续 activity 回归的 Lasso。[C5]
- 基线实现有历史分叉：当前源码的 `_fast_l1` 使用 4096 维 hashed design、固定 C=0.1；R6.2 修复矩阵则记录了训练范围内的无碰撞词表和独立数值缩放。当前图 C 文档承诺直接复用 frozen batch 的 `base_logit`。正式结果报告必须核对实际 prepared 来源及基线实现，分别标注“相对冻结 L1 基线的增益”和“相对独立公平基线的增益”，不能自动认为 R6.2 的基线修正已经进入本分支。此项不是本次已完成的基线重算，也不据此中断训练。[C5/H10]
- 先前“G2 没有单细胞，所以最终需要接入融合/再训”的推断撤回。现有冻结方案允许单细胞作为独立证据层、融合权重为零。若以后改变该选择，应另行明确研究问题和评价合同；本次未作该变更。[H7]

### 3. 本轮决策及后续判断条件

**当前建议：保留正在使用的关联主头和有界残差设计，完成既定五折后评价；现在没有足够证据要求切换纯图、连续头或端到端单细胞融合。**本建议不是对模型有效性的预判。运行管理继续由用户指定的另一 agent 负责。

1. 评价同一 fold、候选键、患者划分、可用性掩码与标签上的 G2 和冻结 L1；报告逐折及分癌/通路的 AUPRC、AUROC、Precision@K、校准和差值。核对实际基线特征合同，不能借一个较弱或不同任务的 LASSO 宣称普遍胜出。
2. 同时核验残差、门控、回退、方向头与各项损失。若仅组合损失下降但关联排序不改善，应记录为未证实主任务增益；只有具体诊断才能支持后续针对性修改。
3. 用于早停/选择的 validation 仅支持开发结论。遵循既有 sealed-test / 外部验证合同；不能在查看 test 后反复选架构并继续称它为独立确认。当前审查没有访问新的 sealed test。
4. 用户已要求本次只训练最终 G2，故不自动补开 G0/G1、B/C-shuffle 等作业。只有 G2 对照 L1 无法将增益特异归因于 ENCODE 或某类 binding；归因需要相应匹配对照，这是结论边界，不是本次新增运行命令。
5. 若目标升级为“证明 lncRNA 因果调控通路”，优先缺口是独立 lncRNA 扰动等功能真值与验证设计。更换预测头不会自行产生该真值，RBP binding/KD 也不能自动替代 lncRNA 因果验证。

### 4. 证据索引

历史文档中的状态均按其记录日期解释，不代表当前 live 状态。

- [H1] `D:/model/exports/v3_1_3cancer_pilot_decomposition_20260812/V31_DECOMPOSITION_SUMMARY.md`（RNAss 分解表）。
- [H2] `D:/model/EvidenceLayered_CCHHGT_Regression_20260822/reports/FINAL_MODEL_REPORT_ZH.md`（最终结论、模型排名及配对比较）。
- [H3] `D:/model/EvidenceLayered_CCHHGT_Regression_20260822/reports/decisive_audit/DECISIVE_AUDIT_REPORT_ZH.md`（PC-only residual、pair-disjoint evidence）。
- [H4] `D:/model/CC_HHGT_v3_2_ranked_subtypes_dev/MODEL_CHANGELOG.jsonl`，`V32-BOUNDED-RESIDUAL` 事件；本文件 2026-08-25 原始内容。
- [H5] 同 `MODEL_CHANGELOG.jsonl` 的 `FORMAL_ONESEED_RELEASE_CANDIDATE_COMPLETED` 事件及 `artifacts/formal_release_report_1seed/MODEL_REPORT.md`。
- [H6] [fresh G012 设计](docs/v32_fresh_g012_preparation_design_20260829.md) §2.1；`D:/model/CancerLncAtlas_version_debug_logs_20260906/versions/r7/VERSION_CHANGE_LOG.jsonl` 的历史结果边界与同目录 `audits/R7_RETAINED_BLOCKERS.md`。R7 允许共享泛癌消息＋癌种条件 decoder，不能称为每癌独立消息图。
- [H7] [单细胞范围冻结](docs/V32_SINGLE_CELL_SCOPE_CLOSED_20260904.md) 的 Frozen decision；[多任务分工配置](config/model_v3_2_full_multitask.yaml)。该历史文档里的细胞计数不用于覆盖更新后的服务器结果计数。
- [H8] `D:/model/V32_R62_TRAINING_LOG_20260908.md`，Step 34 / Step 35；`D:/model/v32_pkg_r62_extracted/CancerLncAtlas_v32_full_pyg_r6_2/README.md`。
- [H9] 本仓库 Git 记录及 [全局 binding 授权源码](scripts/rbp_encode_v33/authorize_global_c_fold_149.py)。
- [H10] `D:/model/v32_pkg_r62_extracted/CancerLncAtlas_v32_full_pyg_r6_2/FIX_MATRIX_R62.md`；[冻结批次与 L1 复用说明](docs/rbp_encode_v33/RBP_AB_BASELINE_COMPARISON.md)。
- [C1] 149 既有 `.../rbp_encode_v33_20260921_r1/runtime/c_graph_global_binding_20260928_r1/auth_postpay_final_g2_fold0/config.yaml`（本次仅只读查看配置）；同 [授权源码](scripts/rbp_encode_v33/authorize_global_c_fold_149.py)。
- [C2] [associations.py](cc_hhgt/v32/associations.py)：`compute_fold_associations` / `label_associations`。
- [C3] [model.py](cc_hhgt/v32/model.py)：`build_v32_cc_hhgt_residual`。
- [C4] [training.py](cc_hhgt/v32/training.py)：`_global_loss_from_outputs`、`residual_shrinkage` 和 `direction_loss_weight` 的实际训练用法。
- [C5] [baselines.py](cc_hhgt/v32/baselines.py)：`SAFE_NUMERIC_FEATURES` / `_safe_records`；[prepare_v32_formal.py](scripts/prepare_v32_formal.py)：`_fast_l1`。

## 2026-09-28 — Level 0 图贡献诊断：产物前提与判读

- 用户提出不训练、直接读取 C 预测表，逐折计算 L1/final 的 Spearman、绝对 graph residual 分布及 graph gate 分布。本项可作为结果回传后的首项 CPU 诊断，不需要新训练。
- 本次在 host=149 只读查看当前 `c_graph_global_binding_20260928_r1`：运行目录存在，但其 `results` 目录尚不存在，未找到回传的预测 parquet。本次没有读取云端训练实例、没有调用推理或访问 sealed test，也没有产生数值诊断结果。
- `contracts.py:PREDICTION_COLUMNS` 是发布 schema，不是产物存在证明。当前 `validation_prediction_inference.py` 写出的 `G012_VALIDATION_PREDICTIONS.PRIVATE.parquet` 列为候选键、fold/selection 字段、`outer_test_queried`、`fusion_target` 和 `primary_probability`，未写 L1/residual/gate。不能直接承诺当前 G012 导出已包含旧发布 schema 的全部字段。已有发布脚本 `scripts/finalize_v32_release_report.py` 会统计 residual 的均值/P95/近零率及 gate 均值，但不统计 Spearman，并依赖旧发布产物合同，不能原样用于未完成的当前结果。
- 待实际预测表到位，先确认 run/fold/checkpoint、唯一候选键、相同评价行与概率校准口径，再计算逐折 Spearman；补充按 cancer×pathway 的排名/Top-K 改变，避免全折相关系数掩盖少数通路的有效重排。报告绝对 residual 的均值、中位数、P90/P95/P99、近零比例，以及 gate 的分位数、近零比例；常量分数的 Spearman 记为未定义，不能强写成 1。
- 先核对残差恒等式的口径：原始 logit 上才有 `z_final=z_l1+graph_residual`；独立温度校准后的概率不一定满足反 logit 恒等式。分块推理应使用注册的 logit 加权规则；gate 的加权平均不能冒充满足 `mean(gate)*tanh(mean(raw))` 的等效门控。
- 正确判读：残差接近零且排序几乎不变，仅说明图对该表的最终分数贡献很小；不能将 C−B 的差异归因于 nnPU。若残差精确为零且校准/评价行相同，final 与 L1 的概率及 AUPRC 应一致。非零残差也可能只改善校准、不改变排名；Spearman≈1 不能单独证明图未起作用。
- 排序改变而同口径 AUPRC 不改善，只能说未改善该指标，不能直接称为随机扰动；排序改变且 AUPRC改善，支持该评价集上有预测增益，不单独证明生物机制、跨数据泛化或 ENCODE 特异贡献。nnPU 的因果归因仍需同架构/同数据的损失对照，本次不新增该训练。
- 如实际产物缺少字段，训练管理方可在其既定验证/预测导出时一并保留这些小列，避免为诊断另做一遍整图推理。此项是交接建议，本次未修改正在运行的代码，也未实现/宣称已完成导出。
