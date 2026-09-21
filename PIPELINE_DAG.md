# CancerLncAtlas V3.2 Pipeline DAG

状态：`CODE_ONLY`。下图是代码依赖和未来获授权后的执行责任，不表示任何任务已经训练。

```mermaid
flowchart TD
    C00["00 合同/版本/哈希"] --> C01["01 输入只读审计"]
    C01 --> C02["02 16,889 lncRNA<br/>共享 4,712 + cancer-local"]
    C01 --> C03["03 33癌种 × 5患者折"]
    C03 --> C04["04 exact-pathway activity<br/>仅蛋白编码基因"]
    C02 --> C05["05 训练患者内关联/候选"]
    C04 --> C05
    C05 --> C06["06 同候选 L1/Ridge"]
    C01 --> C07["07 无 pair-evidence 安全图"]
    C03 --> C07
    C06 --> C08["08 cancer-conditioned CC-HHGT<br/>L1 + bounded graph residual"]
    C07 --> C08
    C00 --> C09["09 多端任务 manifest"]
    C08 -. "训练代码" .-> C10G["10a training_guard/CLI<br/>先验权限与四类哈希"]
    C09 --> C10G
    LOCK["默认无 TRAINING_APPROVAL.json"] -->|"fail closed"| C10G
    C10G -->|"仅授权全部通过后动态导入"| C10T["10b training.py<br/>Torch/CUDA、coverage cycles、原子resume"]
    C10T -. "当前不可达，不产生真实预测" .-> C11["11 exact-pathway Gene Set"]
    C11 --> C12A["12a pathway-specific cancer subtype"]
    C12A --> C12B["12b cancer program subtype/相似图"]
    C00 --> C13["13 代码/测试/费用终态审计"]
    C09 --> C13
    C12B --> C13

    EVID["文献/实验/药物/单细胞 evidence"] --> ECONF["独立 regulatory evidence confidence"]
    ECONF -. "只并列展示，不改概率/排名" .-> C11
    EVID -.- BLOCKED["禁止进入 05/06/07/08"]

    style LOCK fill:#ffe5e5,stroke:#b00020
    style BLOCKED fill:#ffe5e5,stroke:#b00020
    style C10G fill:#fff4cc,stroke:#9a6700
    style C10T fill:#fff4cc,stroke:#9a6700
    style ECONF fill:#e8f1ff,stroke:#2457a6
```

## Fold-local 泄漏边界

```mermaid
flowchart LR
    TRAIN["外层训练患者"] --> FIT["activity参数 / association / candidates / graph context"]
    FIT --> BASE["L1 + Ridge"]
    FIT --> GNN["CC-HHGT"]
    VAL["外层验证患者"] --> SELECT["选择 / 早停 / 校准"]
    TEST["外层测试患者"] --> EVAL["一次性评价"]
    VAL -. "禁止" .-> FIT
    TEST -. "禁止" .-> FIT
    PAIR["pair-level evidence"] -. "100%屏蔽" .-> FIT
    PAIR -. "100%屏蔽" .-> GNN
```

改变验证或测试患者时，`FIT` 侧的候选、关联、安全图和训练资产哈希必须保持不变。

## 端点所有权

```mermaid
flowchart LR
    S149["149 CPU<br/>资产/基线/subtype/报告"]
    LOCAL["本地 4070 16GB<br/>未来顺序执行 5 folds"]
    PAID["付费 GPU<br/>禁用，0 tasks，¥0"]
    MAN["不可变 RUN/TASK manifest + SHA-256"]

    S149 <--> MAN
    LOCAL <--> MAN
    PAID -. "paid_enabled=false" .-> MAN
```

- 当前不连接 149；这里只定义将来获授权后的 CPU owner。
- 当前不初始化本地 CUDA；五个任务在 manifest 中均为 `BLOCKED`。
- 付费端必须同时具备独立 shard、非零预算和哈希匹配授权；当前三个条件均不成立。
- 多端不得同时拥有同一 task key；跨硬件不得续接 optimizer/checkpoint。
- `training.py` 只有在 guard 通过后才被动态导入；届时 checkpoint 同时绑定代码/配置/输入/任务哈希和 runtime fingerprint，任何漂移均拒绝 resume。

## 下游 subtype 方向

```mermaid
flowchart TD
    SCORE["每个 cancer × exact pathway 的 lncRNA 概率/排名"] --> TOP["每方向 Top-200，共同 4,712 宇宙"]
    TOP --> SIM["50% RBO + 50% weighted Jaccard"]
    SIM --> PCLUST["逐 pathway: K=1..4 + stability gates"]
    PCLUST --> CONS["pathway conservation / heterogeneity"]
    PCLUST --> GLOBAL["pathway-family 等权汇总"]
    GLOBAL --> CCLUST["cancer program subtype 或连续相似图"]
    PCLUST -. "本轮禁止回流" .-> SCORE
```

因此，DNA repair、EMT 或 immune pathway 可以各自产生不同的癌种分组；不存在强制所有 pathway 共享同一个癌症聚类。
