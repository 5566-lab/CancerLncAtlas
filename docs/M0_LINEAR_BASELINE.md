# M0：仅用 lncRNA 表达预测现有 Pathway / State 分数

M0 在现有 `cc_hhgt/v31_continuous_baselines.py` 的训练集填补、标准化和嵌套 Ridge 连续回归方案上扩展了缺失标签、多输出批处理和 State 支持。核心在 `cc_hhgt/m0_linear.py`，入口为 `scripts/run_m0_linear.py`。没有 PCA、图网络、临床协变量、CNV、甲基化、蛋白或其他评分作为预测特征。

## 输入与目标

- 全部当前可用且经 GENCODE v36 核实为 lncRNA 的表达特征共同预测；每个训练折去掉全缺失和无变化列。并非逐条 lncRNA 单变量回归。
- TCGA 读取当前正式 lncRNA logCPM 矩阵，其基因轴已经经过历史队列筛选，**并非完整的 16,889 条原始 lncRNA**。M0 不再按相关性或测试表现筛选基因。
- CPTAC 读取当前整合包的肿瘤 RNA log2(TPM+1)，从完整基因矩阵提取 GENCODE v36 lncRNA。患者正常组织不会进入模型。
- 按癌种、队列分别训练：TCGA 33 癌种；CPTAC 10 队列。队列名称与 TCGA 癌种别名沿用既有映射。
- TCGA 标签：当前 ssGSEA、PROGENy、DoRothEA 和 485 项 State 目录。CPTAC 标签：RNA 通路、蛋白通路、启动子相关甲基化通路均值、磷酸化位点 PTM-SEA 和同一 State 目录。缺失标签始终为 NA，未计算的 RPPA500 通路分数不冒充现成标签。
- 甲基化均值是甲基化程度，蛋白 ssGSEA 是丰度相关分数，均不能称为“实测通路激活”。某些目录有相互重叠的评分，目标数也不代表独立的生物学程序数。

## 线性模型与验证

每个队列、癌种、目标单独拟合：

```
Y_hat_i,t = b_t + sum_j beta_j,t * Z_i,j
Z_i,j = (impute(X_i,j; training_median_j) - training_mean_j) / training_sd_j

min_(b,beta) sum_(i with measured Y_t) (Y_i,t - b - Z_i beta)^2
             + p * lambda_t * ||beta||_2^2
```

p 为该折保留的表达特征数，截距不受惩罚。lambda 在 10^4 到 10^-4 的 17 个对数等距值之间由验证集 MSE 选择，同时加入“仅截距/训练均值”候选；相同误差保留较简单模型。通过患者核矩阵求解，与相同 alpha=p*lambda 的 sklearn Ridge 等价。

TCGA 沿用现有患者五折权威表；CPTAC 按固定种子和完整患者 ID 的 SHA256 顺序均衡分为五折。外层第 k 折测试，第 (k+1) mod 5 折验证，其余三折训练；选好参数后以训练加验证患者重新拟合，最后只预测测试患者。每个患者至多贡献一次 OOF（out-of-fold，折外）预测。

X 的中位数、均值和标准差只由相应训练患者拟合，可以使用训练集中没有该项 Y 的患者的 X；标签缺失模式相同的目标共享矩阵分解，但独立选择正则化参数。各目标只使用自身有实测标签的训练行，重新中心化保证其独立截距。无 Y 的患者不会补成标签，也不进入指标计算。

最低样本数为训练 10、验证 3、测试 3，逐目标逐折判断。这是回归评估的最小数量限制，不是组学基因覆盖率阈值。跳过原因、部分 OOF、常数目标均明确记录；没有 95% 覆盖规则。默认运行全部可用目标，无诊断模式目标截断。

M0 的回归相互独立，不定义一个共享多任务损失；现有 HHGT 的 Pathway=1.0、State=0.5 不变，M0 不覆盖其模型或 Y 入口。对单个完全独立的任务整体乘一个权重不会改变最优解；各目标的 Ridge 正则化由自身验证集选择。

## 结果解释边界

M0 预测的是**现有冻结版本的原生评分**。TCGA 历史 TMM、表达入选和部分评分、CPTAC 部分 State/TF/PTM 预处理使用过整队列。因此模型层面严格留出患者，不等于整条原始数据到标签的流程已经完全归纳化。不得将这些内部交叉验证结果宣称为未来新患者的独立验证、因果调控或绝对活性预测。

TCGA 的 native logCPM 与 CPTAC 的 log2(TPM+1) 不能直接拼接；同名标签也不保证相同基因背景、方法和标度。共享 lncRNA 是**共同特征轴**，不是把两个队列变成一个 gene set。当前提供共同 ID 数量和单位对照，不声称已完成批次校正或跨队列验证。两者未来可以在统一表达处理及逐项核实标签后共同建模，但应保留队列分组验证。不得用所有患者做批次校正后随机混分，再声称完全独立验证。

参考官方处理定义：[GDC RNA-seq](https://docs.gdc.cancer.gov/Encyclopedia/pages/RNA-Seq/)；
[LinkedOmicsKB CPTAC 数据下载](https://kb.linkedomics.org/download)。

## 运行

所有准备、训练、评估和绘图只能在 hostname=149 执行，入口会检查主机并写入 LAUNCH_PREFLIGHT.json。不需要 GPU。

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python scripts/run_m0_linear.py \
  --config /dell_2/DSC/CancerLncAtlas/runtime/cptac_20261104_r1/CURRENT_149_CONFIG.json \
  --output /dell_2/DSC/CancerLncAtlas/results/model/M0_20261104_r2 \
  --cancer-type all --workers 4

# 小范围 demo，输出到另一目录：
python scripts/run_m0_linear.py --config /path/to/current_149_config.json \
  --output /path/to/new_demo --datasets TCGA CPTAC \
  --cancer-type list --cancer-list BRCA LUAD --workers 2

python scripts/plot_m0_linear.py \
  --results /dell_2/DSC/CancerLncAtlas/results/model/M0_20261104_r2 \
  --output /dell_2/DSC/CancerLncAtlas/results/model/M0_20261104_r2/figures
```

完整实验自动期限由启动命令的 `timeout 14400` 实施。完成的癌种可在代码、配置和输出哈希都一致时续跑；变更实验需新目录。CPTAC 缺 RNA 的患者没有 M0 输入，会在实际样本数量中体现。

每个癌种输出：

- INPUTS.json、TARGETS.tsv、FEATURES.tsv、PATIENT_FOLDS.tsv：输入哈希、评分方法、特征和患者划分。
- OBSERVED_SCORES.parquet、OOF_PREDICTIONS.parquet、OOF_TRAIN_MEAN_BASELINE.parquet：实测标签、折外预测、各折训练均值基线。
- METRICS.tsv：R²、RMSE、Spearman、相对训练均值的误差改进、标签与 OOF 覆盖率。负 R² 不截断，常数目标的 R² 为 NA。
- FOLD_AUDIT.parquet：每个目标每折样本数、所选 lambda、是否仅截距、未训练原因。
- models/：共享的训练变换及逐缺失模式的 Ridge 检查点。可用 `predict_checkpoint(path, x)` 对原始特征顺序一致的矩阵重放预测。
- TOP_LNCRNA_COEFFICIENTS.parquet：每目标每折绝对系数最大的 20 个 lncRNA；其余系数可由检查点恢复。系数是预测权重，不代表因果效应。

绘图输出 300 dpi PNG 和矢量 PDF，包括各癌种总体分布、均值基线比较、State 家族热图及预先指定的缺氧、EMT、氧化磷酸化、CYT、干性例图。例图不按测试集表现挑选。部分图展示 R² 的 [-1,1] 区间，会明确标出低于 -1 的目标数，完整数值保留在表中。

用户指定的 Windows 图片目标是 `D:\\public8\\DSC\\CancerLncAtlas\\results\\model\\图片\\M0_20261104_r2`。图在 149 生成后复制到 Windows，并核对 SHA256；Windows 不运行分析。

## 验证

`tests/test_m0_linear.py` 验证 dual Ridge 与 sklearn 等价、缺失标签处理、测试数据不参与拟合、训练缺失/常数特征处理、检查点预测重放，以及更改测试标签不会改变该折预测。`tests/test_v31_continuous_baselines.py` 同时检查原有 LASSO/Ridge 入口。

TARGETS.tsv / METRICS.tsv 中 method 是当前目录声明；图表目录的 M0_TARGET_METHODS.tsv.gz 和 M0_ALL_TARGET_METRICS.tsv.gz 另外提供 native_method，明确区分 CPTAC 实际重算方案与 TCGA 原始评分方法。

绘图附加依赖固定在 `requirements/m0_plot_20261104.txt`。本次独立安装目录为 `/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps`，绘图时加入 PYTHONPATH；不覆盖原训练环境。
