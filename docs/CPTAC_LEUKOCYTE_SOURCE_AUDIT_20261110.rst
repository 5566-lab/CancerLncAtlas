CPTAC 甲基化白细胞比例：来源核实与方案修订
================================================

核实结论（2026-11-10）
----------------------

旧接入表中的 ``Leukocyte_fraction`` 在 1,043 人中全空，只能说明该表没有接入原 TCGA 定义的终点。旧代码直接将它写成 endpoint not available，未区分未下载、未计算和资源不存在。

本次从研究作者 ProTransDeconv 仓库下载 ``example/data/Cell_Type_Proportion.txt``。100 行患者 ID 全部一对一匹配现有 CPTAC CCRCC 患者，覆盖 100/103（97.1%）。三类细胞比例总和为 1；仅将约 10^-18 量级的负数浮点误差截到 0，保留原始数值。论文将此类比例定义为甲基化 EDec stage 1 估计。来源表只提供患者 ID，尚不能证明与项目蛋白数据属于同一组织 aliquot。

数据以 ``methylation_immune_fraction_EDec`` 保存，并记录原算法和来源。它没有被改名为原 TCGA ``leukocyte_fraction``，也没有被填入现有冻结 bundle。

GDC 当前文件清单与项目患者 ID 的匹配结果为：725 人关联甲基化 beta 文件，其中 720 人至少有一个仅关联 Primary Tumor 的文件。后者分布为 CCRCC 103、GBM 97、HNSCC 108、LSCC 107、LUAD 109、PDAC 101、UCEC 95。文件关联证明原始数据可获得；不等于白细胞比例已经算出，也不等于样本 aliquot 已完成匹配。BRCA、COAD、OV 在本次 GDC 与项目患者 ID 的匹配中为 0；这是本次检索范围的结果，不能扩大成全球不存在。

已有七癌种的基因启动子甲基化聚合矩阵，不能直接当作 CpG 探针矩阵执行原始细胞去卷积。新核查的 Cancer Cell 甲基化研究中，细胞组成分析采用 RNA xCell，因此该文的细胞富集分数也不能替代甲基化比例。

后续执行方案
------------

* 主比较：在每个癌种中，两平台采用相同的临床、纯度和既有细胞组成调整，均不强制加入甲基化白细胞比例。保留目标自身及同源派生指标的排除规则。
* 敏感性分析：只在有已核实甲基化比例的患者中，对同一组患者分别拟合不加、加入该比例的模型，比较关联方向、效应和支持稳定性。算法不同的比例保留方法标签。
* BRCA、LUAD、COAD demo 使用上述共同调整口径。LUAD 可继续从探针 beta 文件补算；BRCA、COAD 不凭缺失值构造比例，也不以 RNA 免疫分数冒充甲基化比例。
* 新方案涉及关系估计口径变化，TCGA/TCPA 和 CPTAC 都需按新版本重算，再合并关系证据。原 r6 的 Y 与历史结果保持原有口径。
* GBM 的 stage 全空应按癌种临床定义单独处理；不能通过填零获得虚假的有效样本。

当前已完成来源核实、100 人比例文件的匹配与保存、GDC 下载清单、缺失原因修正及本方案。新口径关系 Y 的重算、全量甲基化去卷积和模型训练尚未执行。

来源
----

* Song et al., doi:10.1021/acs.jproteome.5c00980: https://pmc.ncbi.nlm.nih.gov/articles/PMC12772123/
* 作者发布文件: https://github.com/HuangLabAtUAB/ProTransDeconv/blob/main/example/data/Cell_Type_Proportion.txt
* GDC CPTAC: https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac
* 本次原始查询与数据 SHA256、逐癌种计数: ``provenance/cptac_leukocyte_20261110/``。
