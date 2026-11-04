M0：相关性筛选后进行 Lasso 的三癌种 demo
=========================================

本次扩展将全部共同 lncRNA 输入转换为每个癌种、每项评分的固定非零系数名单。
Demo 范围为 BRCA、LUAD、COAD，TCGA 与 CPTAC 仍分别拟合；
本实现不宣称完成跨队列 Y 校准或得到跨队列共同系数。

输入与原 M0 的区别
-----------------

TCGA 从原始 GDC RangedSummarizedExperiment 的 tpm_unstrand assay
恢复未经过历史 10% logCPM 检测筛选的基因表达。样本必须与现有患者权威表精确匹配，
不选择其他样本替代、不补造 RNA。按稳定 Ensembl ID 与 CPTAC GENCODE v36 lncRNA
取共同可测集合；三个癌种都恢复到约 16,870 条共同 lncRNA。
两边均采用 log2(TPM+1)。原始定量流程仍然不同，不宣称批次效应已经消除。

现有 RNA pathway、TF、State、CPTAC 蛋白／磷酸化／甲基化标签逐值保留。
缺失标签不填补；无标签或样本数不足的目标明确记为不可用。
本流程不更改 HHGT 的 Pathway=1.0、State=0.5，不启用 MOGSA。

筛选与训练
----------

* 每个目标在当前训练患者中计算 lncRNA–Y 的双侧 Spearman 相关。
  默认 abs(rho)>=0.20、BH-FDR<=0.05、至少 10 对实测数据。
* 相关分析只用实测成对值；常数、缺测不足等不可检验基因的 p 值设为 1。
  BH 检验族是该目标、该癌种、该训练分组的全部输入 lncRNA。
  p 值采用常见 t 近似，小样本筛选属于探索性分析，不能将筛选 q 值
  当作整条建模流程或最终基因名单的错误发现率保证。
* 沿用患者五折。第 k 折为测试，下一折为验证，其他三折为训练。
  初次相关筛选、X 缺失值填补、均值和标准差均只在训练集拟合。
* 只在通过相关筛选的基因中拟合 Lasso；不预设最大基因数，不追加任意 Top20。
  标准化 X 和 Y 后，alpha_max=max(abs(X'Y))/n，
  alpha/alpha_max 在 1 到 0.001 的 25 个对数等距值中选择，
  按验证 MSE 最小、并列优先更强惩罚确定。
* 选好参数后，训练与验证患者合并，重新筛选、重新拟合预处理并训练，
  然后预测测试患者。每个目标只使用自身有观测 Y 的训练患者。
* 相关筛选没有合格基因时输出截距模型；验证选择 alpha_max 时也明确使用截距模型。
  不降低阈值凑出非空 gene set。
* Celer 0.7.4 加速求解相同 L1 目标；未安装时支持 sklearn 坐标下降。
  每条路径及最终拟合检查对偶间隙，未收敛不能标为完成。
  求解器写入运行收据；一致性测试独立对照 sklearn。

最终固定名单与验证结果的区别
----------------------------

评估完成后，每个目标采用五次内层验证所选 alpha 比例的中位数，
在全部有标签患者中重新筛选和拟合，导出该模型全部非零系数。
至少需要三个有效外层拟合才能生成最终模型。
折外指标评价的是完整训练流程，不能理解为最终全数据系数已在独立新队列验证。

同时导出五折入选频率与方向一致性。这些稳定性结果是注释，
不会在看过测试指标后另行筛选基因。
Lasso 系数是条件预测权重，不是因果效应；与单变量相关的符号可能不同。

输出
----

FINAL_GENE_SETS.tsv.gz / parquet
    最终全部非零基因、符号、原始表达单位系数、标准化系数、填补中位数、
    筛选 rho/FDR、观测对数、五折入选频率。
FINAL_MODELS.tsv
    每目标的截距、alpha、alpha 比例、观测数、筛选数和非零数，以及不可用原因。
FINAL_CORRELATION_PASS.parquet
    全部达到相关性标准的基因，包括最终被 Lasso 收缩为零者。
FOLD_SELECTED_GENES.parquet / FOLD_MODELS.tsv
    外层各折的稀疏模型，支持逐折重放预测。
GENE_SELECTION_STABILITY.tsv.gz
    各基因跨有效折的选择频率和方向一致性。
METRICS.tsv / OOF_PREDICTIONS.parquet
    保留负 R2 的折外指标和缺失标签掩码。
AUDIT_COMPLETE.json / MODEL_REPLAY_CHECKS.tsv
    输入输出哈希、阈值、对偶间隙和实际模型重放审计。

运行位置与范围
--------------

所有准备、训练、测试、绘图和审计只能在 hostname=149 执行。
先运行 scripts/export_m0_full_tcga_tpm.R，再运行
scripts/prepare_m0_lasso_inputs.py --cancer-list BRCA LUAD COAD。
训练入口为 scripts/run_m0_screened_lasso.py，最多允许三个癌种。
绘图和审计分别为 scripts/plot_m0_screened_lasso.py、
scripts/audit_m0_screened_lasso.py。依赖见 requirements/m0_lasso_20261104.txt。
Windows 仅接收完成的图表和交接记录，不运行分析。

模型与交付表的读法
------------------

对每个评分目标单独拟合标准化后的 Lasso：

    min_beta (1 / (2n)) * sum_i (y_i - x_i @ beta)^2 + alpha * sum_j abs(beta_j)

部署时使用导出的原始 log2(TPM+1) 单位系数与截距：

    predicted_score = intercept + sum_g coefficient_g * log2(TPM_g + 1)

缺失 X 用该模型保存的训练中位数填补。M0 各评分目标独立拟合，
不把 Pathway 和 State 相加成一个 Y；1.0 与 0.5 是原 HHGT 联合损失权重。

figures/ 中的每队列 FINAL_GENE_SETS.tsv.gz 和 ALL_FINAL_GENE_SETS.tsv.gz
额外附带目标的折外 R2、Spearman、相对训练均值预测的 skill、样本数和状态。
它们用于判断这份基因清单对应的预测是否有用，不代表已完成独立队列验证。
DEMO_SUMMARY.tsv 汇总各数据层的表现，负 R2 不删除。
TCGA_CPTAC_LASSO_GENE_SET_OVERLAP.tsv 按同名目标列出最终名单的交集、
Jaccard 和两侧覆盖率；汇总表只统计两边均非空的名单。
这些是描述性重叠，不代表两队列原生 Y 已经校准，也不产生合并的系数模型。

本次沿用的 Y 中，部分评分包含队列级标准化。这里的折外结果是在这些冻结标签条件下，
对 M0 的筛选、预处理与拟合做患者分割评估；不等于上游全部评分也已按训练折重新计算。
TCGA 恢复的 X 来自原始 TPM，输入收据中的历史 upstream_exposure 字段仍说明原标签来源。

字段兼容说明：已完成的 r2 输入及运行收据保留原始字节，其中 features_before_annotation
是旧 M0 沿用的历史数量，不能解释为本次恢复完整 TPM 后的输入数量。
实际输入以 n_features 和 FEATURES.tsv 为准；INPUTS.json 另有 previous_tcga_feature_count
与 restored_tcga_features 记录恢复过程。后续准备入口将旧字段明确命名为
legacy_features_before_annotation，并单列两队列交集前的注释 lncRNA 数量。
公开完成汇总同样采用明确的历史字段名，原运行文件不改写。
