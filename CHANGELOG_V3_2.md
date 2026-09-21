# CancerLncAtlas V3.2 变更说明

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
