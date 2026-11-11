M0：癌种内部的精确通路关系线性对照
=======================================

预测单位与标签
--------------

本入口的每一行是 ``癌种 × lncRNA × 精确通路``。它与已有的患者通路分数回归
M0 是两个不同端点；原有患者评分回归入口保持不变。

关系 M0 直接使用经过完整校验的 CCHHGT prepared 中的 ``label``、``available``、
``direction_label``、``direction_available``，不重新计算阈值，不从患者标签的
rho、p、q 构造训练特征。prepared 的具体标签定义复制到运行 receipt 中。

当前 fused FunctionalRelationContext 将未支持的关联记为 unlabelled，没有已确认
阴性。该目录不能直接当作二分类数据输入此入口。需先确定正例—未标记学习等目标，
完成相应 prepared 适配，再让两种模型使用同一端点。本次没有执行这一转换。

M0 的形式
---------

在每个癌种内部拟合稀疏线性逻辑回归：

.. math::

   \operatorname{logit}\widehat P(Y_{c,l,k}=1)
      = b_c + \beta_c^\top u_{c,l,k}^{\rm source}
        + \gamma_c^\top I(c,l,k).

``u`` 是 CCHHGT prepared 中已有的 source 患者数值特征块，包括发现集关联统计及
已声明的表达、CNV、突变背景等。``I`` 复用现有 exact_baseline 的稀疏身份项
``c, l, t, c_l, c_t, l_t``，词表仅在可用训练行中拟合。默认 L1；``--penalty l2``
提供 Ridge 对照。方向采用独立线性头，严格使用原方向可用性掩码。

因此这是多种已准备数值信息的关系对照，不是“只用 lncRNA 表达的患者预测”，也
不产出该患者模型的 Lasso RNA signature。临床信息和纯度沿用患者阶段既有的调整
设计，不将患者协变量伪造成一条 lncRNA—通路关系的协变量。

范围与分折
----------

* 默认仅 BRCA、LUAD、COAD，各癌种单独拟合；不自动运行 33 癌种。
* 仅 ``TARGET_POLICY.tsv`` 中 active 且 ``analysis_role=primary_pathway`` 的条目
  进入此主任务。TF、激酶即使显示分类为 Pathway，也不会自动获得主任务权重。
* 先过滤目标，再拟合 scaler、身份词表、模型及验证集选参。排除条目以及仅有交集
  但未获准的候选不会进入训练。
* 完整 prepared 校验核对患者分折、来源/标签患者不交叉、数值数组与来源校验和。
  M0 不访问 sealed test；训练、验证患者角色完全沿用 prepared。
* C 使用原 baseline 配置网格；主关系以验证 AP 选参，方向以验证 BCE 选参。
  ``EXACT_PATHWAY_METRICS.tsv`` 按癌种、精确通路、输出头分别报告指标。
* 原数据库 ID 需要显式一对一映射到 canonical ID。多个旧读数指向同一 canonical ID
  时拒绝直接改名，必须先重建对应标签；名称归并不等于标签已融合。

与 CCHHGT 的比较范围
--------------------

M0 对齐的是 prepared 的数值特征块、候选、患者分折和标签掩码。CCHHGT 还消费
基因、蛋白、通路节点上下文及可选节点 CNV/甲基化，这些信息并未全部展平到 M0
数值块。因此当前比较回答“线性数值/身份基线与完整图模型的差异”，不能把差值
全部解释成图拓扑贡献。若需单独检验拓扑，需要另设相同节点输入但取消消息传递
的对照；该模型不在本入口中。

运行三癌 demo
-------------

先准备通过当前标签契约与所选通路范围校验的 CCHHGT capsule。新目录的 KEGG 筛选
阈值和 fused 标签适配未完成时，不启动这一命令。所有执行均在服务器 149。

.. code-block:: bash

   /dsk2/user/dengsc/miniconda3/envs/cc_hhgt/bin/python scripts/run_m0_relation.py \
     --prepared /path/to/verified_scoped_prepared \
     --policy /path/to/TARGET_POLICY.tsv \
     --out /path/to/new_m0_relation_demo_directory \
     --cancers BRCA LUAD COAD --penalty l1

旧 ID 与 canonical ID 不同但严格一一对应时，额外传入 ``--identity-map``；TSV 列名
为 ``source_target_id`` 和 ``canonical_id``。每个 prepared capsule 对应一个既有
outer fold，五折分别传入五个 capsule，不能把不同 outer fold 的训练数组混合。

输出包括每癌种两头的系数与训练词表、验证预测、逐精确通路指标和
``M0_RELATION_RECEIPT.json``。receipt 固定 policy、prepared、输入数组的 SHA256，
记录分折及所用标签定义。本次仅完成代码与 fixture 验证，没有开始真实三癌训练。

验证
----

``tests/test_m0_relation.py`` 验证：排除目标与辅助因子的数值/标签改变不影响拟合；
Pathway 显示分类不改变主任务准入；多对一标签改名被拒绝；缺失标签不会变为阴性；
各癌种不会混合；修改 sealed test 标签不改变系数或 C；额外 held-out rho/p/q
列不进入特征；原始 fused 目录不会被隐式转换成二分类。
