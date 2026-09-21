# CancerLncAtlas V3.2 代码流程逐步说明

状态：`CODE_ONLY / IMPLEMENTATION_READY_NOT_TRAINED`  
版本目标：`3.2.0`  
适用任务：癌种 × lncRNA × exact pathway 的关联成员资格排序，以及基于排序结果的下游癌种程序亚型分析。

本文件描述代码应该执行的语义，而不是报告模型效果。当前阶段不允许真实患者拟合、优化器更新、CUDA 训练、149 任务提交或付费实例启动。V3.1、cap1000、cap2000 和 297-task 的 checkpoint、概率、校准器、网站物化结果不得导入或继承到 V3.2。

## 1. 首先分开四个不能混排的任务

| 任务 | 预测单位 | 输出 | 本轮角色 |
|---|---|---|---|
| Gene Set membership | cancer × lncRNA × exact pathway | 关联成员概率、方向和排名 | V3.2 主任务 |
| Pathway-context subtype | cancer × exact pathway | 由 Gene Set 组成和排名推导的癌种程序亚型 | 主模型完成后的下游任务 |
| Patient pathway activity | patient × exact pathway | 连续 ssGSEA/AUCell 类分数 | 独立模块，本轮不训练 |
| Regulatory evidence | cancer × lncRNA × exact pathway | 文献、实验或调控证据置信度 | 独立展示字段，不改变主概率 |

`pathway_family_id` 只能是 exact pathway 的层级上下文、分层汇总和解释字段，不能作为主标签，也不能把 family 概率广播给其成员 pathway。这里的 subtype 指“通路特异癌种程序亚型”，不是患者临床亚型。

## 2. 全流程不可变约束

1. 所有主预测均以 `pathway_id` 的 exact pathway 为目标。
2. 任何 fold 的候选、标签、关联、预处理和安全图只能读取该 fold 的训练患者。
3. 验证患者只可用于选择、早停和校准；测试患者只可用于一次性评价。
4. 文献、药物、interaction、perturbation、单细胞验证和其他 pair-level evidence 在主模型训练、验证和测试中 100% 屏蔽。
5. LASSO、Ridge 和 CC-HHGT 必须使用相同候选行、相同标签、相同患者折和相同评价行。
6. 泛癌共享 lncRNA 宇宙必须从完整输入重算；期望计数 4,712 是验收断言，不是绕过计算的硬编码名单。
7. 只在 1–2 个癌种满足表达门槛的 lncRNA 保留在相应癌种的 local-only 分支；未评估不得填成 0 或负例。
8. Gene Set 完成后才能计算 subtype；本轮 subtype 不回流到模型特征或标签。
9. 默认执行模式为 `CODE_ONLY`。缺少显式授权时，训练入口必须在 CUDA 初始化、数据拟合和远程连接之前拒绝执行。

## 3. 00–13 阶段说明

### 00 — 版本与任务合同

输入：V3.2 配置、schema、代码版本和执行模式。  
代码行为：解析配置、规范化 cancer/pathway/lncRNA 稳定 ID，验证 `target_unit=cancer_lncrna_exact_pathway`、`target_field=pathway_id`、禁止特征集合和执行预算；计算不可变合同哈希。  
输出：解析后的只读合同、配置哈希和运行身份。  
失败条件：主目标使用 family、`training_authorized=true` 出现在默认配置、付费预算非零，或必要 schema 缺失。

### 01 — 输入只读审计

输入：GDC 表达路径、样本/患者映射、GENCODE 注释、exact-pathway registry、pathway-family 映射及安全图来源清单。  
代码行为：只检查存在性、可读性、列类型、ID 唯一性、癌种覆盖、样本归属和内容哈希；不复制、重命名或改写源数据。  
输出：`INPUT_AUDIT.json` 的结构化审计内容。  
泄漏边界：此步不计算 association、label、evidence score 或训练统计量。

### 02 — lncRNA 宇宙

输入：完整 33 癌种 GDC STAR 表达和 16,889 个 GENCODE v26 稳定 lncRNA ID。  
代码行为：每癌种选择一个肿瘤 aliquot/患者，按生产预处理计算 `logCPM`；若 `logCPM>0` 则记为检出；计算每个 lncRNA 的癌内检出率和达到 10% 门槛的癌种数。  
输出：共享资格表和癌种局部资格表，字段至少包含稳定 ID、癌种、检出率、满足癌种数、`shared_or_local_scope` 和未评估原因。  
验收断言：完整集合为 16,889；至少 3 个正式癌种满足癌内检出率 ≥10% 的共享集合为 4,712。旧 2,751 或旧 7,066 名单不能作为输入名单复用。

### 03 — 患者外层折

输入：33 癌种的 canonical tumour patient IDs。  
代码行为：在每个癌种内部用固定 seed `20260726` 稳定分配五折；外层 fold 的验证/测试角色按冻结轮换规则产生，所有转换由 patient ID 而非行号驱动。  
输出：基础 `PATIENT_FOLD_MANIFEST.tsv` 记录 patient、cancer 和固定 `patient_fold_id`；展开的 outer-split 视图再为每个 outer fold 增加 `train/validation/test` 角色。  
泄漏边界：一个患者在同一 outer fold 中只能有一个角色；同一患者的多个 aliquot 不得跨 split。

### 04 — exact-pathway 活性

输入：蛋白编码基因表达、exact-pathway gene membership、fold manifest 和训练内协变量。  
代码行为：仅以蛋白编码基因计算患者 × exact-pathway 活性；标准化、缺失处理和协变量拟合参数只从训练患者估计，再原样应用到验证/测试患者。  
输出：fold-local pathway activity 及其训练参数摘要。  
泄漏边界：目标 lncRNA 不能参与其 pathway activity 的计算；pathway family 不能替代 exact pathway。

### 05 — 癌内关联和候选

输入：训练患者的 lncRNA 表达、exact-pathway 活性、协变量及步骤 02 的表达资格。  
代码行为：逐 cancer × lncRNA × exact pathway 计算方向、效应、显著性/FDR 和检出率；根据冻结阈值生成强阳性、弱阳性和未标注样本及样本权重。  
输出：fold-local association/candidate 表。  
泄漏边界：改变验证或测试患者的表达和结局不得改变训练候选、训练关联或安全图哈希。未标注不等于确认阴性。

### 06 — 匹配线性基线

输入：步骤 05 的训练候选和允许特征。  
代码行为：实现纯 L1 logistic LASSO 主基线和 L2 logistic Ridge 敏感性基线；只允许训练患者产生的连续关联统计、检出率、癌种/lncRNA/pathway 主效应及允许的二阶交互；保存 logits 和校准接口。  
输出：与 CC-HHGT 行键完全相同的 `l1_probability`、`ridge_probability` 和 baseline logits。  
泄漏边界：禁止外部证据、标签化 availability 和完整 `cancer×lncRNA×pathway` 三阶记忆特征。历史 `${PRIVATE_WORK_ROOT}/code` 中 `alpha=0.5` 的连续 Elastic Net 不进入本排名。

### 07 — 无 pair-evidence 的安全异构图

输入：稳定 ID registry、允许的 gene/protein/pathway/family/cancer 关系、训练患者产生的安全癌种上下文。  
代码行为：构建 typed nodes/edges，固定 canonical ordering，记录每类关系的来源、行数和 SHA-256；为每个 outer fold 只注入训练患者可产生的癌种上下文。  
输出：fold-safe graph manifest 和 tensorization 接口。  
泄漏边界：lncRNA–pathway label 边、文献/药物/扰动/单细胞验证边及验证/测试患者统计不得进入主图。

### 08 — Cancer-conditioned CC-HHGT 残差

输入：步骤 06 的 L1 logits、步骤 07 的安全图以及 cancer/lncRNA/exact-pathway 候选索引。  
代码行为：异构编码器学习节点表示；decoder 显式组合 lncRNA×pathway、lncRNA×cancer 和 pathway×cancer 交互；有界 gate 只允许图分支在 LASSO 上增加收缩残差：

\[
z_{final}=z_{L1}+\lambda_{c,p}\tanh(\Delta_{CC-HHGT})
\]

输出：final logits/probability、原始图残差和 `graph_gate`。  
强制性质：残差头零初始化；`lambda=0` 时最终 logit 必须在 `1e-7` 内等于 L1 logit。外部证据只作为分离的 `regulatory_evidence_confidence` 输出，不能进入公式。

### 09 — 多端任务编排

输入：运行身份、五折 manifest、模型/seed、owner 和预算策略。  
代码行为：生成唯一任务主键 `run_id|patient_fold|model|seed` 和原子 owner；检查重复任务、端点能力和费用上限。  
输出：`TASK_MANIFEST.tsv`。  
当前合同：恰好五个 `CC-HHGT × patient_fold_0..4 × seed_20260726` 本地任务；全部状态为 `BLOCKED`；付费任务为 0。

### 10 — 受保护训练入口

输入：任务 manifest、配置、代码/输入哈希以及可选授权文件。  
代码行为：`training_guard.py` 先验证显式 `--allow-training`、TRAINING 模式、approval、四类产物哈希、task/endpoint/硬件和费用范围；只有全部通过，CLI 才动态导入 `training.py`。`training.py` 此时才导入 Torch、读取已授权的 fold bundle、构建 CC-HHGT 残差模型并允许 CUDA。其完整训练循环实现梯度累积、逐 coverage-cycle 验证、连续 6 个 cycle 无改善早停、best/last 原子 checkpoint，以及绑定授权产物哈希和 runtime fingerprint 的同硬件精确 resume；跨硬件或输入漂移立即失败。  
当前行为：由于不创建 `TRAINING_APPROVAL.json`，入口必须 fail closed；不得产生真实 checkpoint。跨硬件恢复不继承 optimizer/checkpoint，只能保留旧产物为 diagnostic 并从 epoch 0 开始新任务。

### 11 — exact-pathway Gene Set 物化

输入：五折校准后的预测行；当前代码阶段只能用合成/契约数据验证接口。  
代码行为：按 cancer、exact pathway 和方向聚合五折概率、rank percentile 和选择频率；稳定排序并导出 Parquet/GMT。  
输出：`ranked_geneset_members.parquet` 及 exact-pathway GMT。  
泄漏边界：不得将 family score 复制到 exact pathways；证据置信度不得改变 association probability 或排序；local-only 与 shared 必须显式区分。

最低输出字段：

- `cancer_id`, `lncrna_id`, `pathway_id`, `pathway_family_id`
- `association_membership_probability`, `association_direction`
- `l1_probability`, `ridge_probability`, `graph_residual`, `graph_gate`
- `fold_rank_percentile`, `fold_selection_frequency`
- `shared_or_local_scope`, `regulatory_evidence_confidence`

### 12 — Gene Set 驱动的下游亚型

输入：步骤 11 的已完成 Gene Set 排名；主分析只用共同的 4,712 条共享 lncRNA，local-only 只进入敏感性分析。  
代码行为：对每个 exact pathway/方向取前 200 名，用 `0.5 × RBO(p=0.98) + 0.5 × probability-weighted Jaccard` 计算癌种相似度；候选 `K=1..4`，用 silhouette、200 次 bootstrap ARI 和最小类大小判定是否接受离散亚型。  
输出：`pathway_context_subtypes.parquet`、`pathway_conservation.parquet`；再按 pathway family 等权汇总为 `cancer_program_subtypes.parquet` 或连续全局相似图。通路层候选 K 为 1–4；全局癌症程序层候选 K 为 2–6，未通过相同稳定性门禁时退回 K=1/连续相似图。  
强制行为：少于 6 个可评价癌种或每癌种少于 10 个稳定成员时返回 `UNAVAILABLE`；只有 silhouette ≥0.25、ARI ≥0.75、每类至少 3 个癌种才接受 `K>1`；否则输出 K=1 或连续异质性，不强制分群。本轮 subtype 不回流步骤 00–11。

### 13 — 终态代码审计

输入：代码、配置、schema、单元测试、dry-run manifest 和变更日志。  
代码行为：汇总合同、测试、代码哈希、任务/费用状态和禁止声明；检查没有真实 checkpoint、训练成功标记、远程提交或付费 owner。  
输出：由最终验收流程生成 `DRY_RUN_REPORT.json`、`TEST_REPORT.md` 和 `IMPLEMENTATION_READY_NOT_TRAINED.json`。  
通过含义：代码和合成测试就绪，但没有任何模型效果、稳定性或网站替换结论。

## 4. 三端职责和数据流

| 端点 | 将来获授权后的职责 | 当前允许 | 当前禁止 |
|---|---|---|---|
| 149 CPU | 权威只读输入审计、fold-local 资产、L1/Ridge、subtype 与报告 | 路径/manifest 的本地静态验证 | SSH、提交、拟合、写远端 |
| 本地 RTX 4070 Ti SUPER 16GB | 顺序运行五个 CC-HHGT patient-fold 任务 | import、CPU 合成 forward、dry-run | CUDA 初始化、optimizer step、真实 checkpoint |
| 付费 32GB GPU | 仅在未来独立预算授权后接收显式 shard | 生成零任务计划 | 启动实例、连接、自动 failover、任何费用 |

密码、token 和云密钥不得出现在配置、任务清单、日志或 checkpoint 中。多端只能交换带 SHA-256 的不可变 manifest/结果，不能共享一个可写任务目录。

## 5. 检查 pipeline 时应逐项确认

- 任务目标和输出键始终是 exact pathway，而不是 family。
- 33 癌种都参与 pooled cancer-conditioned 学习；LOCO 只保留为以后可选的迁移消融。
- 同一 pathway 可因 cancer 条件产生不同的 lncRNA 组成和排名。
- subtype 从 Gene Set 排名推导，不是预先指定的全局 K=3，也不回喂同一轮模型。
- 主概率中没有 pair-level evidence；Evidence 层只有独立置信度。
- 4,712 是从 16,889 重算得到；local-only 没有被删除。
- L1/Ridge/CC-HHGT 的 candidate key 完全相同。
- 默认运行只生成 `BLOCKED` 任务，付费成本恒为 ¥0。
- 任何历史 V3.1/cap checkpoint、概率或网站结果均没有继承入口。

只有以上合同和全部测试通过，并由用户另行创建哈希匹配的授权文件后，才可讨论训练。代码就绪本身不代表模型优于 LASSO、Ridge、V2.9 或历史 HHGT。

## 6. 实际代码位置和调用关系

| 阶段 | 实现位置 | 可审查的主要接口 |
|---|---|---|
| 00 | `config/model_v3_2_code_only.yaml`, `cc_hhgt/v32/contracts.py` | `default_contract`, `validate_contract`, `validate_feature_role_manifest`, `assert_matched_candidate_universe` |
| 01 | `cc_hhgt/v32/input_audit.py` | `audit_inputs` |
| 02 | `cc_hhgt/v32/lncrna_scope.py` | `build_lncrna_scope` |
| 03 | `cc_hhgt/v32/patient_folds.py` | `build_patient_fold_manifest`, `assign_outer_split`, `build_all_outer_splits` |
| 04 | `cc_hhgt/v32/pathway_activity.py` | `validate_protein_coding_pathway_membership`, `compute_rank_mean_activity`, `ActivityScaler` |
| 05 | `cc_hhgt/v32/associations.py` | `compute_fold_associations`, `label_associations`, `benjamini_hochberg` |
| 06 | `cc_hhgt/v32/baselines.py` | `build_safe_hashed_design`, `fit_matched_logistic`, `attach_replication_labels` |
| 07 | `cc_hhgt/v32/safe_graph.py` | `build_safe_graph`, `graph_is_invariant_to_external_evidence` |
| 08 | `cc_hhgt/v32/model.py` | `build_v32_cc_hhgt_residual`, `compose_bounded_residual_numpy`, `assert_exact_l1_fallback`, `nnpu_membership_loss` |
| 09 | `cc_hhgt/v32/orchestration.py` | `build_task_manifest`, `validate_task_manifest`, `build_dry_run_report`, virtual checkpoint/resume 验证 |
| 10 | `cc_hhgt/v32/training_guard.py`, `cc_hhgt/v32/cli.py`, `cc_hhgt/v32/training.py` | `guard_training_entry`; guard 后动态导入的 `run_authorized_task`; 原子 checkpoint、同硬件 resume、coverage-cycle early-stop |
| 11 | `cc_hhgt/v32/genesets.py` | `aggregate_fold_predictions`, `materialize_ranked_genesets`, `genesets_to_gmt_lines` |
| 12 | `cc_hhgt/v32/subtypes.py` | `infer_ranked_subtypes`, `classify_pathway_context_subtypes`, `classify_cancer_program_subtypes` |
| 13 | `cc_hhgt/v32/audit.py`（终验审计模块）及顶层静态交付文件 | 汇总测试、哈希、0 checkpoint、0远程、0付费事实；不调用训练接口 |

命令入口是 `scripts/v32_pipeline.py`，其控制台名为 `cchhgt-v32`。当前安全检查只应调用：

```powershell
python scripts/v32_pipeline.py dry-run `
  --config config/model_v3_2_code_only.yaml `
  --output-dir <独立审计输出目录>
```

`run-shard` 虽有完整参数接口，但当前必须返回 `TRAINING_BLOCKED`。CLI 在 `guard_training_entry` 通过之前不会导入 `training.py`、trainer、PyTorch 或模型训练模块，因此缺少授权时不会触发 CUDA 初始化、读取 prepared fold 或创建 checkpoint。
