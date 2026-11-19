生物学范围关联与评分的模型接口
================================

目的
----
保留 KEGG 癌症主干的 476 个身份，并让已有来源评分按生物学范围与主干关联。
不会新增基因集、重算患者评分，或将上层评分复制成多个机制标签。
原固定 Y 仍保存患者的成员实测值、来源读数、缺失掩码与事件适用条件。

对应表
------
``SEMANTIC_SCOPE_RELATIONS.tsv`` 保存源生物学范围、目标主干、关系类型、依据和理由。
来源为既有正式 ID 对照、ComPath 范围关系和逐条核实的 KEGG NETWORK 层级。
``PROGENY_REVIEWED_ROUTES.tsv`` 逐条列出已有 PROGENy 评分对应的 KEGG 机制 ID、
反应链与依据。PROGENy 到 KEGG 的对应是本项目核实的背景关系，不是官方等价表。
``SCORE_BINDINGS.tsv`` 保存原始评分的唯一 ID、组学依赖、方法与来源。
``SCORE_PATHWAY_CONTEXT.tsv`` 连接评分 ID 与主干；值只保存在一个评分列中。

关系分别为 same_scope（同范围）、broader_scope_context（上层过程背景）、
narrower_scope_context（子过程背景）、shared_process_context（共享过程）及
regulator_context（明确调控者）。不将 shared_process 做传递性等价合并。
成员 Jaccard、共享蛋白数及富集显著性均不能创建这些关系，也不作边权。
旧 mechanism_context / rev_mechanism_context 的交集驱动边从新模型视图中移除；
蛋白/基因的真实组成边保留。

例如，PROGENy:WNT 保持一个原始 RNA 评分，关联到 13 个 WNT 机制身份。
APC 突变相关机制可以接收 WNT 背景信息，但事件条件仍须由患者事件证据提供。
Estrogen 仅连接核实的 ESR 机制，不把整个核受体大类中的 AR、PGR、AHR 当作雌激素。
TGFb 不因 TGFA 名称相近而连接到 EGFR 机制。

评分作为 Y
----------
M0 对明确选定的每个原始评分保留一组线性系数：

  prediction(i,r) = intercept(r) + RNA(i) beta(r) + clinical(i) gamma(r)

临床/纯度协变量不受 Lasso 惩罚。r 为原始观测 ID，不是其关联的每条机制。
同一个 WNT 评分即使有 13 条关联，也只有一个输出和一次观测误差。
M0 不执行图传播；归属对照表用于将原始读数的结果放到相应通路下展示，
不能把它的线性系数直接解释为所有下属精确机制的系数。

HHGT 的 score_target_graph 构造“通路节点 -> 原始评分节点”的有类型边，
评分节点不装入患者 Y 数值。ScoreObservationHead 从原有 CC-HHGT 编码器得到的
评分节点表示预测该读数。不同原始读数有各自输出参数，保留单位和方向差别。
预测误差沿已有边反传至通路表示；这能提供共同过程的监督，但不能单独辨认上游事件。

selected_score_loss 仅处理调用方显式选定的观测，空选择不产生损失。
启用的评分应占用原 Pathway 监督预算，而不是在已有误差外重复叠加。
Pathway=1.0、State=0.5 保持不变；字典与关系表不会自行登记训练损失。

评分作为其他任务的输入
----------------------
这是显式选择的另一种用途。例如预测蛋白层目标时，可允许 RNA 来源读数提供背景。
M0 使用与目标关联的 score / observed 两列；HHGT 使用
“source_score 节点 -> 精确通路节点”的有类型边。
两种模型通过同一个 ContextTask 排除目标组学及目标派生信息，且使用同一输入范围。
纯 lncRNA 基线须将 allowed_input_layers 设为空，不能暗中加入 mRNA 来源评分。
加入 RNA 来源评分后，应称为增加 RNA 背景的模型，不再称纯 lncRNA 模型。

输入标准化只使用固定 reference 患者；query 患者不参与参数估计。
缺失保留掩码。未确认组学依赖的评分保留原始数据，但不启用为输入。
评分作为输入和评分作为目标使用独立图视图，禁止将目标值装进图节点。
TCGA 和 CPTAC 仍分别读取，未声称已统一两平台的生物学活性尺度。

调用入口
--------
``fixed_y_bundle.load_target_with_context`` 读取原 Y 与允许的上下文，并核验文件摘要。
``measured_context_models.measured_m0_design`` / ``measured_m0_forward`` 使用线性输入。
``measured_context_models.measured_hhgt_graph`` 接入原始 JointHGT.encode。
``measured_context_models.score_target_graph`` 与 ``ScoreObservationHead`` 用于预测原始读数。
这些接口不调用优化器，也不替换旧的关系强度训练入口。

范围
----
BRCA、LUAD、COAD 的 TCGA 和 CPTAC 六个视图均有 14 项已有 PROGENy 读数。
其中 12 项通过 158 条生物学范围关联覆盖 133 个主干身份。
Androgen 和 VEGF 在当前固定主干内没有通过本轮核实的对应，保留评分但不强接。
其余已有来源读数的同身份对应单独保留；158 不是全部观测边数，也不是新 Y 的数量。

验证依据见项目内部日志 SEMANTIC_SCORE_ROUTING_20261119.json。
