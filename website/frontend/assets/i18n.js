// ============================================================
// CancerLncAtlas v0.3 — 中英双语系统
// ============================================================
const I18N = {
  "zh-CN": { langName: "中文", langCode: "zh-CN", htmlLang: "zh-CN",
    siteTitle: "CancerLncAtlas v0.3",
    siteSubtitle: "泛癌 lncRNA–通路关联与多层级证据数据库",
    searchPlaceholder: "搜索 lncRNA、基因、癌种或通路",
    searchExample: "例如：NEAT1、TP53、LUAD、p53 signaling",
    toggleLang: "English",
    nav: {
      explore: "探索",
      cancer: "癌种",
      lncrna: "lncRNA",
      pathway: "通路",
      analyze: "分析",
      mixedGeneSet: "混合基因集分析",
      proteinToLncrna: "蛋白基因集关联 lncRNA",
      proteinInteracting: "蛋白互作 lncRNA",
      evidence: "证据",
      singleCell: "单细胞证据",
      clinical: "临床关联",
      mutation: "突变证据",
      datasets: "数据集",
      resources: "资源",
      downloads: "下载",
      api: "API",
      documentation: "使用文档",
      releases: "版本记录"
    },
    footer: {
      version: "CancerLncAtlas v0.3",
      dataRelease: "数据版本：2026.08",
      modelArch: "模型架构：CC-HHGT",
      disclaimer: "仅供科研使用"
    },
    home: {
      heroTitle: "解析癌症中的 lncRNA 功能关系",
      heroLede: "汇集 bulk、单细胞、物理互作、临床结局与患者级状态证据，展示 CC-HHGT 的癌种、通路与 lncRNA 关系。",
      searchBtn: "探索图谱 →",
      stats: {
        cancers: "癌种",
        scoredCancers: "scored",
        referenceOnly: "reference-only",
        lncrnas: "收录 lncRNA",
        families: "冻结静态 pathway families",
        relations: "三概率正式关系"
      },
      exploreTitle: "从生物问题出发",
      sections: {
        cancer: { title: "按癌种浏览", desc: "分别查看 lncRNA、通路、细胞类型、状态、药物与 Gene Set。" },
        lncrna: { title: "lncRNA 单基因", desc: "跨癌种景观、三概率关系、冷启动状态与实验物理互作。" },
        singleCell: { title: "单细胞证据", desc: "按数据集、细胞类型、恶性状态和通路查看实际关联证据。" },
        geneset: { title: "Gene Set", desc: "查询冻结 pathway family 与癌种、细胞类型及泛癌集合。" },
        network: { title: "关系网络", desc: "明确区分 family 预测、exact pathway 成员与实验物理互作。" },
        smart: { title: "智能查询", desc: "混合 ID、自定义基因集和蛋白 Set 的 CPU 轻量推理。" }
      },
      probabilityNotes: [
        { name: "Cross-cancer", desc: "严格 LOCO 跨癌种迁移概率" },
        { name: "Cancer-specific", desc: "患者级交叉拟合 adapter 概率" },
        { name: "Evidence-integrated", desc: "多证据综合可信度；不作为独立验证" }
      ]
    },
    loading: "正在载入…",
    error: {
      title: "数据加载失败",
      desc: "API 暂时无法访问，或请求已超时。",
      retry: "重新加载",
      status: "查看服务状态",
      static: "下载静态结果"
    },
    empty: {
      title: "当前筛选条件下没有结果。",
      clear: "清除筛选"
    },
    partial: "候选排名已经加载，但单细胞证据暂不可用。",
    cancer: {
      title: "Cancer 模块",
      desc: "33 个癌种逐一浏览",
      scoredCount: "scored",
      relations: "relations",
      tabs: ["概览","lncRNA 排名","通路","状态与临床","单细胞","突变","证据","下载"],
      overview: {
        patients: "患者数",
        samples: "样本数",
        candidateLncrnas: "候选 lncRNA",
        strictOofCandidates: "严格 OOF 候选",
        singleCellCohorts: "单细胞队列",
        evidenceEvents: "证据事件"
      },
      lncTable: {
        rank: "排名",
        lncrna: "lncRNA",
        crossCancer: "跨癌概率",
        cancerSpecific: "癌种特异概率",
        evidenceIntegrated: "证据整合概率",
        combinedPriority: "综合优先级",
        percentile: "癌种内百分位",
        confidence: "置信等级",
        evidenceCoverage: "证据覆盖",
        patients: "患者数",
        details: "详情",
        noData: "未发现证据",
        notTested: "尚未分析",
        notApplicable: "不适用",
        unavailable: "数据不可用",
        filters: {
          search: "搜索 lncRNA…",
          minPriority: "最低综合优先级",
          confidence: "置信等级",
          singleCell: "单细胞证据",
          clinical: "临床证据",
          mutation: "突变证据",
          directLiterature: "直接文献证据",
          export: "导出",
          exportAll: "导出全部",
          exportPage: "导出当前页"
        }
      },
      detail: {
        title: "候选详情",
        basicInfo: "基本信息",
        rankInCancer: "当前癌种内排名",
        percentileInCancer: "当前癌种内百分位",
        crossCancer: "Cross-cancer probability",
        cancerSpecific: "Cancer-specific probability",
        evidenceIntegrated: "Evidence-integrated probability",
        combinedPriority: "Combined priority",
        combMethod: "综合评分实际计算方法",
        modelApplicable: "模型是否适用",
        evidenceMatrix: "证据覆盖矩阵",
        singleCellEvidence: "单细胞证据",
        clinicalAssociation: "临床关联",
        mutationResults: "突变结果",
        pmidEvents: "PMID 级文献事件",
        dataSource: "数据来源",
        download: "下载入口"
      },
      changeCancer: "切换癌种",
      searchInCancer: "在 {{cancer}} 中搜索",
      downloadResults: "下载 {{cancer}} 结果"
    },
    lncrna: {
      title: "lncRNA 单基因",
      tabs: ["概览","癌种分布","通路","单细胞","临床关联","突变","证据","下载"],
      landscape: "跨癌种景观",
      coldStart: "冷启动状态",
      physicalInteractions: "实验物理互作"
    },
    pathway: {
      title: "通路",
      definition: "通路定义",
      sourceDb: "来源数据库",
      memberGenes: "成员基因",
      cancerDist: "癌种分布",
      rankedLncrnas: "排名 lncRNAs",
      crossCancerConsistency: "跨癌种一致性"
    },
    geneset: {
      title: "Gene Set",
      desc: "Pathway family 与 Gene Set",
      staticFamily: "静态 family 仅由无测试癌种信息的外部结构构建",
      matched: "matched families",
      shown: "exact pathway members shown"
    },
    singleCell: {
      title: "单细胞证据",
      dataset: "数据集",
      cancer: "癌种",
      patients: "患者数",
      samples: "样本数",
      cells: "细胞数",
      cellType: "细胞类型",
      scoringMethod: "评分方法",
      effectSize: "效应量",
      pValue: "P 值",
      fdr: "FDR",
      replication: "复制状态"
    },
    downloads: {
      title: "下载",
      filename: "文件名",
      description: "说明",
      version: "数据版本",
      rows: "行数",
      size: "大小",
      sha256: "SHA256",
      date: "生成日期",
      format: "格式"
    },
    datasets: {
      title: "数据集",
      training: "训练",
      validation: "验证",
      referenceOnly: "仅参考",
      externalEvidence: "外部证据"
    },
    docs: {
      title: "使用文档",
      model: "模型说明",
      data: "数据说明",
      scores: "分数说明",
      statistics: "统计方法",
      limitations: "已知限制",
      notClinical: "本数据库仅供科研使用。页面中的模型预测和统计关联不用于临床诊断、预后判断或治疗决策。"
    },
    smart: {
      mixed: { title: "混合基因集分析", desc: "同时识别 lncRNA、蛋白编码基因和蛋白 ID", placeholder: "每行一个基因 ID 或 lncRNA", queryBtn: "分析" },
      custom: { title: "自定义蛋白基因集 → lncRNA", desc: "将自定义编码基因集编码为 virtual pathway", placeholder: "每行一个蛋白编码基因", queryBtn: "分析" },
      protein: { title: "蛋白 Set → 互作 lncRNA", desc: "物理互作概率与功能关联概率严格分列", placeholder: "每行一个蛋白 ID", queryBtn: "分析" }
    },
    fieldNames: {
      crossCancerProbability: "跨癌概率",
      cancerSpecificProbability: "癌种特异概率",
      evidenceIntegratedProbability: "证据整合概率",
      combinedPriority: "综合优先级"
    }
  },

  "en-US": { langName: "English", langCode: "en-US", htmlLang: "en",
    siteTitle: "CancerLncAtlas v0.3",
    siteSubtitle: "A pan-cancer lncRNA–pathway atlas with multi-level evidence",
    searchPlaceholder: "Search lncRNA, gene, cancer or pathway",
    searchExample: "Examples: NEAT1, TP53, LUAD, p53 signaling",
    toggleLang: "中文",
    nav: {
      explore: "Explore",
      cancer: "Cancer",
      lncrna: "lncRNA",
      pathway: "Pathway",
      analyze: "Analyze",
      mixedGeneSet: "Mixed Gene Set",
      proteinToLncrna: "Protein Set to lncRNA",
      proteinInteracting: "Protein-interacting lncRNA",
      evidence: "Evidence",
      singleCell: "Single-cell Evidence",
      clinical: "Clinical Associations",
      mutation: "Mutation Evidence",
      datasets: "Datasets",
      resources: "Resources",
      downloads: "Downloads",
      api: "API",
      documentation: "Documentation",
      releases: "Releases"
    },
    footer: {
      version: "CancerLncAtlas v0.3",
      dataRelease: "Data release: 2026.08",
      modelArch: "Model architecture: CC-HHGT",
      disclaimer: "For research use only"
    },
    home: {
      heroTitle: "Decoding lncRNA functional relationships across cancers",
      heroLede: "Integrating bulk, single-cell, physical interactions, clinical outcomes and patient-level state evidence from the CC-HHGT model.",
      searchBtn: "Explore →",
      stats: {
        cancers: "Cancers",
        scoredCancers: "scored",
        referenceOnly: "reference-only",
        lncrnas: "Curated lncRNAs",
        families: "Static pathway families",
        relations: "Three-probability relations"
      },
      exploreTitle: "Explore by biological question",
      sections: {
        cancer: { title: "Browse by Cancer", desc: "lncRNAs, pathways, cell types, states, drugs and gene sets per cancer." },
        lncrna: { title: "lncRNA Single Gene", desc: "Cross-cancer landscape, three probabilities, cold-start status and physical interactions." },
        singleCell: { title: "Single-cell Evidence", desc: "Actual association evidence by dataset, cell type, malignancy and pathway." },
        geneset: { title: "Gene Set", desc: "Query frozen pathway families with cancer, cell-type and pan-cancer membership." },
        network: { title: "Network", desc: "Family predictions, exact pathway membership and experimental physical interactions." },
        smart: { title: "Smart Query", desc: "Mixed IDs, custom gene sets and protein set CPU-light inference." }
      },
      probabilityNotes: [
        { name: "Cross-cancer", desc: "Strict LOCO cross-cancer transfer probability" },
        { name: "Cancer-specific", desc: "Patient-level cross-fit adapter probability" },
        { name: "Evidence-integrated", desc: "Multi-evidence confidence; not independent validation" }
      ]
    },
    loading: "Loading…",
    error: {
      title: "Failed to load data",
      desc: "The API is temporarily unavailable or the request timed out.",
      retry: "Retry",
      status: "View service status",
      static: "Download static results"
    },
    empty: {
      title: "No results match the current filters.",
      clear: "Clear filters"
    },
    partial: "Candidate rankings are available, but single-cell evidence could not be loaded.",
    cancer: {
      title: "Cancer Module",
      desc: "Browse 33 cancer types",
      scoredCount: "scored",
      relations: "relations",
      tabs: ["Overview","Ranked lncRNAs","Pathways","State & Clinical","Single-cell","Mutation","Evidence","Downloads"],
      overview: {
        patients: "Patients",
        samples: "Samples",
        candidateLncrnas: "Candidate lncRNAs",
        strictOofCandidates: "Strict OOF candidates",
        singleCellCohorts: "Single-cell cohorts",
        evidenceEvents: "Evidence events"
      },
      lncTable: {
        rank: "Rank",
        lncrna: "lncRNA",
        crossCancer: "Cross-cancer",
        cancerSpecific: "Cancer-specific",
        evidenceIntegrated: "Evidence-integrated",
        combinedPriority: "Combined priority",
        percentile: "Cancer percentile",
        confidence: "Confidence",
        evidenceCoverage: "Evidence coverage",
        patients: "Patients",
        details: "Details",
        noData: "No evidence",
        notTested: "Not tested",
        notApplicable: "Not applicable",
        unavailable: "Unavailable",
        filters: {
          search: "Search lncRNA…",
          minPriority: "Min priority",
          confidence: "Confidence",
          singleCell: "SC evidence",
          clinical: "Clinical",
          mutation: "Mutation",
          directLiterature: "Direct literature",
          export: "Export",
          exportAll: "Export all",
          exportPage: "Export page"
        }
      },
      detail: {
        title: "Candidate Details",
        basicInfo: "Basic information",
        rankInCancer: "Rank in cancer",
        percentileInCancer: "Percentile in cancer",
        crossCancer: "Cross-cancer probability",
        cancerSpecific: "Cancer-specific probability",
        evidenceIntegrated: "Evidence-integrated probability",
        combinedPriority: "Combined priority",
        combMethod: "Combination method",
        modelApplicable: "Model applicable",
        evidenceMatrix: "Evidence coverage matrix",
        singleCellEvidence: "Single-cell evidence",
        clinicalAssociation: "Clinical association",
        mutationResults: "Mutation results",
        pmidEvents: "PMID-level literature events",
        dataSource: "Data source",
        download: "Download"
      },
      changeCancer: "Change cancer",
      searchInCancer: "Search in {{cancer}}",
      downloadResults: "Download {{cancer}} results"
    },
    lncrna: {
      title: "lncRNA Single Gene",
      tabs: ["Overview","Cancer landscape","Pathways","Single-cell","Clinical","Mutation","Evidence","Downloads"],
      landscape: "Cross-cancer landscape",
      coldStart: "Cold-start status",
      physicalInteractions: "Physical interactions"
    },
    pathway: {
      title: "Pathway",
      definition: "Pathway definition",
      sourceDb: "Source database",
      memberGenes: "Member genes",
      cancerDist: "Cancer distribution",
      rankedLncrnas: "Ranked lncRNAs",
      crossCancerConsistency: "Cross-cancer consistency"
    },
    geneset: {
      title: "Gene Set",
      desc: "Pathway families & Gene Sets",
      staticFamily: "Static families are built from external structure without test-cancer information",
      matched: "matched families",
      shown: "exact pathway members shown"
    },
    singleCell: {
      title: "Single-cell Evidence",
      dataset: "Dataset",
      cancer: "Cancer",
      patients: "Patients",
      samples: "Samples",
      cells: "Cells",
      cellType: "Cell type",
      scoringMethod: "Scoring method",
      effectSize: "Effect size",
      pValue: "P value",
      fdr: "FDR",
      replication: "Replication"
    },
    downloads: {
      title: "Downloads",
      filename: "File name",
      description: "Description",
      version: "Data version",
      rows: "Rows",
      size: "Size",
      sha256: "SHA256",
      date: "Generated",
      format: "Format"
    },
    datasets: {
      title: "Datasets",
      training: "Training",
      validation: "Validation",
      referenceOnly: "Reference-only",
      externalEvidence: "External evidence"
    },
    docs: {
      title: "Documentation",
      model: "Model",
      data: "Data",
      scores: "Scores",
      statistics: "Statistics",
      limitations: "Limitations",
      notClinical: "For research use only. Predictions and associations provided by CancerLncAtlas are not intended for clinical diagnosis, prognosis, or treatment decisions."
    },
    smart: {
      mixed: { title: "Mixed Gene Set Analysis", desc: "Simultaneously identify lncRNAs, protein-coding genes and protein IDs", placeholder: "One gene ID or lncRNA per line", queryBtn: "Analyze" },
      custom: { title: "Custom Protein Set to lncRNA", desc: "Encode a custom gene set as a virtual pathway", placeholder: "One protein-coding gene per line", queryBtn: "Analyze" },
      protein: { title: "Protein Set to lncRNA", desc: "Physical interaction and functional association probabilities strictly separated", placeholder: "One protein ID per line", queryBtn: "Analyze" }
    },
    fieldNames: {
      crossCancerProbability: "Cross-cancer probability",
      cancerSpecificProbability: "Cancer-specific probability",
      evidenceIntegratedProbability: "Evidence-integrated probability",
      combinedPriority: "Combined priority"
    }
  }
};

// Current language
let currentLang = localStorage.getItem("cancerlncatlas_lang") || "auto";
(function initLang() {
  if (currentLang === "auto") {
    const browserLang = (navigator.language || "en").toLowerCase();
    currentLang = browserLang.startsWith("zh") ? "zh-CN" : "en-US";
  }
  applyLanguage();
})();

function t(key) {
  const keys = key.split(".");
  let value = I18N[currentLang];
  for (const k of keys) {
    if (value && typeof value === "object") value = value[k];
    else return key;
  }
  return value || key;
}

function appText(obj) {
  if (typeof obj === "string") return obj;
  return obj[currentLang] || obj["en-US"] || Object.values(obj)[0] || "";
}

function setLanguage(lang) {
  currentLang = lang;
  localStorage.setItem("cancerlncatlas_lang", lang);
  applyLanguage();
  render();
}

function applyLanguage() {
  document.documentElement.lang = I18N[currentLang].htmlLang;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    if (el.tagName === "INPUT" && el.type === "text") {
      el.placeholder = t(key);
    } else {
      el.textContent = t(key);
    }
  });
}

function toggleLanguage() {
  setLanguage(currentLang === "zh-CN" ? "en-US" : "zh-CN");
}
