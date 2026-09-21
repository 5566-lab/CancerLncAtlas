// ============================================================
// CancerLncAtlas v0.3 — Production SPA
// Model: CC-HHGT | Data: 2026.08 | Platform: v0.3
// ============================================================"use strict";

const $p = document.getElementById("page");
const $main = document.getElementById("main");
const $sidebar = document.getElementById("sidebar");
const $searchInput = document.getElementById("global-search");
const $searchResults = document.getElementById("search-results");
const $menuBtn = document.getElementById("menu-button");

// ── Utilities ──────────────────────────────────────────────
const fmt = (v, d = 0) => (v == null || isNaN(v)) ? "—" : Number(v).toLocaleString("en-US", { maximumFractionDigits: d });
const esc = (v) => (v == null ? "" : String(v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"));
const num = (v) => (v == null ? "0" : Number(v) >= 1e6 ? (Number(v) / 1e6).toFixed(1) + "M" : Number(v) >= 1e3 ? (Number(v) / 1e3).toFixed(1) + "K" : Number(v).toLocaleString());

function display(value, key) {
  if (value == null) return '<span class="na">N/A</span>';
  if (key && (key.endsWith("_probability") || key.startsWith("mean_"))) return typeof value === "number" ? value.toFixed(3) : esc(value);
  if (typeof value === "number") return Math.abs(value) < 0.001 && value !== 0 ? value.toExponential(2) : value.toFixed(4);
  return esc(value);
}

async function api(url, opts = {}) {
  const res = await fetch(url, { signal: AbortSignal.timeout(30000), ...opts });
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}

function params() { return new URLSearchParams(location.search); }
function route() { const p = location.pathname; if (p.startsWith("/gene-set")) return "geneset"; if (p.startsWith("/cancer")) return "cancer"; if (p.startsWith("/lncrna")) return "lncrna"; if (p.startsWith("/single-cell")) return "sc"; if (p.startsWith("/network")) return "network"; if (p.startsWith("/smart/")) return "smart"; if (p.startsWith("/smart")) return "smart"; if (p.startsWith("/documentation")) return "docs"; if (p.startsWith("/releases")) return "releases"; if (p.startsWith("/datasets")) return "datasets"; if (p.startsWith("/downloads")) return "downloads"; return p === "/" ? "home" : "home"; }

// ── Lang switch in topbar ──────────────────────────────────
document.querySelector(".topbar")?.insertAdjacentHTML("beforeend", '<button id="lang-switch" class="lang-switch button" onclick="toggleLanguage()">' + (I18N[currentLang].toggleLang) + '</button>');

// ════════════════════════════════════════════════════════════
//  SKELETON / LOADING
// ════════════════════════════════════════════════════════════
function skeleton(type = "page") {
  return `<div class="skeleton-${type}"><div class="skel-bar w80"></div><div class="skel-bar w60"></div><div class="skel-bar w90"></div><div class="skel-bar w50"></div></div>`;
}

function errorView(msg, retryFn) {
  return `<div class="error-block"><h2>${t("error.title")}</h2><p>${t("error.desc")}</p><p>${esc(msg)}</p><div class="btn-row"><button class="button" onclick="${retryFn ? 'pages.'+retryFn+'()' : 'render()'}">${t("error.retry")}</button></div></div>`;
}

function emptyView() { return `<div class="empty-block"><p>${t("empty.title")}</p><button class="button" onclick="render()">${t("empty.clear")}</button></div>`; }

function heading(eyebrow, title, lede) {
  return `<div class="page-heading"><div><p class="eyebrow">${esc(eyebrow)}</p><h1>${title}</h1>${lede ? '<p class="lede">'+lede+'</p>' : ''}</div></div>`;
}

function table(rows, columns = null, limit = 200) {
  if (!rows || !rows.length) return '<p class="empty-note">' + t("empty.title") + '</p>';
  const keys = columns || Object.keys(rows[0]);
  const data = rows.slice(0, limit);
  return `<div class="table-wrap"><table><thead><tr>${keys.map(k => `<th>${esc(t("cancer.lncTable."+k) || t("cancer.lncTable.filters."+k) || t("cancer.overview."+k) || k)}</th>`).join("")}</tr></thead><tbody>${data.map(r => `<tr>${keys.map(k => `<td>${display(r[k], k)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

// ════════════════════════════════════════════════════════════
//  NAVIGATION RENDER
// ════════════════════════════════════════════════════════════
function buildNav() {
  const links = document.querySelectorAll("#sidebar [data-route]");
  const rt = route();
  links.forEach(a => { a.classList.toggle("active", a.getAttribute("data-route") === rt); });
}

function updateLangSwitch() { const btn = document.getElementById("lang-switch"); if (btn) btn.textContent = I18N[currentLang].toggleLang; }

// ════════════════════════════════════════════════════════════
//  PAGES
// ════════════════════════════════════════════════════════════
const pages = {

  async home() {
    try {
      const stats = await api("/api/site/stats");
      const T = I18N[currentLang].home;
      $p.innerHTML = `<section class="hero"><div class="hero-copy"><p class="eyebrow">CancerLncAtlas · 2026</p><h1>${T.heroTitle}</h1><p class="lede">${T.heroLede}</p></div>
        <form class="hero-search" id="hero-search"><input name="query" required placeholder="${t("searchPlaceholder")}"><span class="hint">${t("searchExample")}</span><button class="button">${T.searchBtn}</button></form></section>
        <div class="stats-grid">
          <div class="stat-card"><strong>${stats.cancers}</strong><span>${T.stats.cancers} (${stats.scored_cancers} ${T.stats.scoredCancers} + ${stats.reference_only_cancers} ${T.stats.referenceOnly})</span></div>
          <div class="stat-card"><strong>${num(stats.lncrnas)}</strong><span>${T.stats.lncrnas}</span></div>
          <div class="stat-card"><strong>${stats.pathway_families}</strong><span>${T.stats.families}</span></div>
          <div class="stat-card"><strong>${num(stats.predicted_relations)}</strong><span>${T.stats.relations}</span></div>
        </div>
        <div class="section-head"><h2>${T.exploreTitle}</h2></div>
        <div class="feature-grid">
          ${["cancer","lncrna","singleCell","geneset","network","smart"].map(k => `<a class="feature-card" href="/${k==='smart'?'smart/mixed':k==='sc'?'single-cell':k}" data-link><h3>${T.sections[k]?.title||k}</h3><p>${T.sections[k]?.desc||""}</p></a>`).join("")}
        </div>
        <div class="probability-strip">${T.probabilityNotes.map(n => `<div><strong>${n.name}</strong><span>${n.desc}</span></div>`).join("")}</div>`;
      document.querySelector("#hero-search").addEventListener("submit", e => { e.preventDefault(); const q = new FormData(e.currentTarget).get("query"); if (q) location.href = "/lncrna?q=" + encodeURIComponent(q); });
    } catch (e) { $p.innerHTML = errorView(e.message, "home"); }
  },

  async cancer() {
    const id = params().get("id");
    if (id) return pages._cancerDetail(id);
    try {
      const data = await api("/api/site/cancers");
      const T = I18N[currentLang].cancer;
      const items = (data.cancers || []).map(item => `<a class="cancer-tile ${item.reference_only?"reference":""}" href="/cancer?id=${esc(item.cancer_id)}" data-link><strong>${esc(item.cancer_id)}</strong><small>${esc(item.english_name||item.cancer_id)}</small><small>${item.reference_only ? "Reference-only" : num(item.scored_relationship_count||0)+" "+T.relations}</small></a>`).join("");
      $p.innerHTML = `${heading(T.title, T.desc, "")}<div class="cancer-grid">${items}</div>`;
    } catch (e) { $p.innerHTML = errorView(e.message, "cancer"); }
  },

  async _cancerDetail(id) {
    try {
      const data = await api("/api/site/cancer/" + id);
      const info = data.overview || {};
      const ref = info.reference_only;
      const T = I18N[currentLang].cancer;
      const L = T.lncTable;
      // Overview cards
      const cards = [
        [T.overview.patients, fmt(info.scored_relationship_count)],
        [T.overview.candidateLncrnas, fmt(info.detectable_lncRNAs)],
        [T.overview.strictOofCandidates, ref ? T.overview.unavailable : fmt(info.significant_lncRNA_pathway_relations)],
      ];
      $p.innerHTML = `${heading("Cancer", `<span class="cancer-header">${esc(id)}</span><span class="tag ${ref?"warn":""}">${ref?"reference-only":"scored"}</span>`, esc(info.english_name||""))}
        <div class="cancer-toolbar"><button class="button" onclick="location.href='/cancer'">${T.changeCancer}</button></div>
        <div class="stats-grid">${cards.map(([l,v]) => `<div class="stat-card"><strong>${v}</strong><span>${l}</span></div>`).join("")}</div>
        <div class="tab-row" id="cancer-tabs">${T.tabs.map((name,i) => `<button class="tab ${i===0?"active":""}" data-tab="${i}">${name}</button>`).join("")}</div>
        <section class="cancer-section" data-section="0">
          <div class="panel"><h3>${T.tabs[1]}</h3>
            <div class="filter-row">
              <input id="lnc-search" class="control" placeholder="${L.filters.search}" oninput="pages._filterLncTable()">
              <select id="lnc-conf" class="control" onchange="pages._filterLncTable()"><option value="">${L.filters.confidence}</option><option value="observed_core">Observed core</option><option value="model_supported">Model supported</option><option value="predicted_candidate">Predicted</option></select>
              <button class="button" onclick="pages._exportLncTable('${id}')">${L.filters.exportAll}</button>
            </div>
            <div id="lnc-table-wrap">${table(data.lncrna_ranking, ["rank_within_cancer","gene_symbol","lncrna_id","pathway_name","pathway_id","final_direction","calibrated_probability","seed_probability_std","observed_evidence_score","relationship_class","member_weight"], 100)}</div>
          </div>
        </section>
        <section class="cancer-section" data-section="1" hidden>${table(data.pathway_ranking, ["rank_within_cancer","pathway_name","pathway_id","pathway_family_id","lncRNA_count","max_calibrated_probability","mean_calibrated_probability","max_member_weight"], 50)}</section>
        <section class="cancer-section" data-section="2" hidden>${table(data.state_summary||[])}</section>
        <section class="cancer-section" data-section="3" hidden>${table(data.celltype_summary||[])}</section>
        <section class="cancer-section" data-section="4" hidden><p>${L.notTested}</p></section>
        <section class="cancer-section" data-section="5" hidden>${table(data.drug_summary||[])}</section>
        <section class="cancer-section" data-section="6" hidden><p>${T.tabs[6]} — ${L.notTested}</p></section>
        <section class="cancer-section" data-section="7" hidden>
          <div class="panel"><h3>${T.tabs[7]}</h3><p>${L.notTested}</p>
            <div class="btn-row"><button class="button" onclick="pages._exportLncTable('${id}')">${L.filters.exportAll}</button></div>
          </div>
        </section>
        <div id="candidate-drawer" class="drawer" hidden></div>`;
      document.querySelector("#cancer-tabs")?.addEventListener("click", e => {
        const tgt = e.target.closest("[data-tab]"); if (!tgt) return;
        document.querySelectorAll("#cancer-tabs .tab").forEach(b => b.classList.toggle("active", b===tgt));
        document.querySelectorAll(".cancer-section").forEach(s => { s.hidden = s.dataset.section !== tgt.dataset.tab; });
      });
      window._allLncrnaRows = data.lncrna_ranking || [];
    } catch (e) { $p.innerHTML = errorView(e.message, "cancer"); }
  },

  _filterLncTable() {
    const q = (document.getElementById("lnc-search")?.value||"").toLowerCase();
    const conf = document.getElementById("lnc-conf")?.value||"";
    let rows = window._allLncrnaRows || [];
    if (q) rows = rows.filter(r => (r.lncrna_id||"").toLowerCase().includes(q) || (r.gene_symbol||"").toLowerCase().includes(q));
    if (conf) rows = rows.filter(r => (r.confidence_level||r.relationship_class||"") === conf);
    document.getElementById("lnc-table-wrap").innerHTML = table(rows, ["rank_within_cancer","gene_symbol","lncrna_id","pathway_name","pathway_id","final_direction","calibrated_probability","seed_probability_std","observed_evidence_score","relationship_class","member_weight"], 200);
  },

  _exportLncTable(id) {
    const rows = window._allLncrnaRows || [];
    const csv = ["rank,lncrna_id,pathway_id,direction,calibrated_probability,seed_std,evidence_score,relationship_class,member_weight"].concat(rows.map(r => [r.rank_within_cancer,r.lncrna_id,r.pathway_id,r.final_direction,r.calibrated_probability,r.seed_probability_std,r.observed_evidence_score,r.relationship_class,r.member_weight].join(","))).join("\n");
    const blob = new Blob([csv], {type:"text/csv"});
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `cancerlncatlas_${id}_rankings.csv`; a.click();
  },

  async lncrna() {
    const q = params().get("q") || "";
    const id = params().get("id");
    try {
      if (id) {
        const data = await api("/api/site/lncrna/" + id);
        const T = I18N[currentLang].lncrna;
        $p.innerHTML = `${heading("lncRNA", esc(id), "")}<div class="tab-row">${T.tabs.map((n,i)=>`<button class="tab ${i===0?"active":""}" data-tab="${i}">${n}</button>`).join("")}</div>
          <section class="cancer-section" data-section="0"><div class="panel"><h3>${T.tabs[0]}</h3>${table(data.top_relationships||[], ["cancer_id","pathway_name","pathway_id","pathway_family_id","final_direction","calibrated_probability","seed_probability_std","observed_evidence_score","relationship_class","member_weight"], 100)}</div></section>
          <section class="cancer-section" data-section="1" hidden><div class="panel"><h3>${T.landscape}</h3>${table(data.cancer_landscape||[])}</div></section>
          <section class="cancer-section" data-section="2" hidden><div class="panel"><h3>${T.tabs[2]}</h3>${table(data.top_relationships||[], null, 50)}</div></section>
          <section class="cancer-section" data-section="3" hidden><div class="panel"><h3>${T.tabs[3]}</h3><p>${t("cancer.lncTable.notTested")}</p></div></section>
          <section class="cancer-section" data-section="4" hidden><p>${t("cancer.lncTable.notTested")}</p></section>
          <section class="cancer-section" data-section="5" hidden><p>${t("cancer.lncTable.notTested")}</p></section>
          <section class="cancer-section" data-section="6" hidden><div class="panel"><h3>${T.tabs[6]}</h3>${table(data.physical_interactions||[])}</div></section>
          <section class="cancer-section" data-section="7" hidden><p>${t("cancer.lncTable.notTested")}</p></section>`;
        document.querySelectorAll(".tab-row .tab").forEach(b => b.addEventListener("click", () => {
          document.querySelectorAll(".cancer-section").forEach(s => s.hidden = s.dataset.tab !== b.dataset.tab);
          document.querySelectorAll(".tab-row .tab").forEach(x => x.classList.toggle("active", x===b));
        }));
      } else {
        $p.innerHTML = heading("lncRNA", I18N[currentLang].lncrna.title, "") + `<form class="hero-search" onsubmit="event.preventDefault();location.href='/lncrna?q='+encodeURIComponent(this.querySelector('input').value)"><input name="query" placeholder="${t("searchPlaceholder")}" value="${esc(q)}"><button class="button">${I18N[currentLang].home.searchBtn}</button></form>`;
      }
    } catch (e) { $p.innerHTML = errorView(e.message, "lncrna"); }
  },

  async geneset() {
    const pathway = params().get("pathway") || params().get("family") || "";
    try {
      const data = await api("/api/site/genesets?limit=100" + (pathway ? `&search=${encodeURIComponent(pathway)}` : ""));
      const T = I18N[currentLang].geneset;
      $p.innerHTML = `${heading("Gene Set", "Cancer-specific exact-pathway lncRNA gene sets", "Pathway family is hierarchy metadata only")}
        <form class="filter-row" onsubmit="event.preventDefault();const v=new FormData(this).get('search');history.replaceState({},'','/gene-set'+(v?'?pathway='+encodeURIComponent(v):''));pages.geneset();"><input class="control" name="search" value="${esc(pathway)}" placeholder="exact pathway ID or name"><button class="button">Filter</button></form>
        <div class="result-summary"><div class="metric"><strong>${data.total_genesets||0}</strong><span>${T.matched}</span></div><div class="metric"><strong>${(data.members||[]).length}</strong><span>${T.shown}</span></div></div>
        <div class="panel"><h3>Exact-pathway gene sets</h3>${table(data.genesets)}</div>
        <div class="panel"><h3>lncRNA members</h3>${table(data.members)}</div>`;
    } catch (e) { $p.innerHTML = errorView(e.message, "geneset"); }
  },

  async sc() {
    try {
      const data = await api("/api/site/single-cell");
      const T = I18N[currentLang].singleCell;
      $p.innerHTML = `${heading("Single-cell", T.title, "")}${table(data||[])}`;
    } catch (e) { $p.innerHTML = errorView(e.message, "sc"); }
  },

  async network() {
    const lnc = params().get("lnc") || "";
    try {
      const data = await api("/api/site/network" + (lnc ? `?lnc=${encodeURIComponent(lnc)}` : ""));
      $p.innerHTML = heading("Network", "Network", "") + (data ? table(data) : `<p>Enter an lncRNA to view its network.</p>`);
    } catch (e) { $p.innerHTML = errorView(e.message, "network"); }
  },

  async smart() {
    const mode = (location.pathname.split("/").pop() || "mixed").replace("smart-","");
    const cfg = I18N[currentLang].smart[mode] || I18N[currentLang].smart.mixed;
    const endpoint = {mixed:"/v2.6/predict/mixed-gene-set",custom:"/v2.6/predict/custom-gene-set-lncrna",protein:"/v2.6/predict/protein-set-lncrna"}[mode] || "/v2.6/predict/mixed-gene-set";
    $p.innerHTML = `${heading("Analyze", I18N[currentLang].nav.analyze, "")}
      <form class="hero-search" id="smart-form"><textarea name="genes" placeholder="${esc(cfg.placeholder)}" rows="6"></textarea><button class="button">${cfg.queryBtn}</button></form><div id="smart-results"></div>`;
    document.querySelector("#smart-form").addEventListener("submit", async e => {
      e.preventDefault();
      const genes = new FormData(e.currentTarget).get("genes").split(/[\n,]+/).map(s=>s.trim()).filter(Boolean);
      if (!genes.length) return;
      document.getElementById("smart-results").innerHTML = skeleton("table");
      try {
        const res = await fetch(endpoint, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({gene_set:genes})});
        const data = await res.json();
        document.getElementById("smart-results").innerHTML = table(data.results||data);
      } catch (err) { document.getElementById("smart-results").innerHTML = errorView(err.message); }
    });
  },

  async datasets() {
    try {
      const data = await api("/api/site/datasets");
      $p.innerHTML = heading("Datasets", I18N[currentLang].datasets.title, "") + table(data);
    } catch(e) { $p.innerHTML = errorView(e.message, "datasets"); }
  },

  async downloads() {
    const rows = [
      {name:"V3.1 exact-pathway lncRNA gene sets (symbols)", href:"/api/site/download/v31-genesets"},
      {name:"V3.1 exact-pathway lncRNA gene sets (stable IDs)", href:"/api/site/download/v31-genesets-ids"},
      {name:"V3.1 exact-pathway gene-set master", href:"/api/site/download/v31-geneset-master"},
      {name:"V3.1 exact-pathway gene-set members", href:"/api/site/download/v31-geneset-members"},
      {name:"V2.9 state-graph model card", href:"/api/site/download/model-card"},
    ];
    $p.innerHTML = heading("Downloads", I18N[currentLang].downloads.title, "") + `<div class="panel"><ul>${rows.map(r=>`<li><a href="${r.href}">${esc(r.name)}</a></li>`).join("")}</ul></div>`;
  },

  async docs() {
    const T = I18N[currentLang].docs;
    $p.innerHTML = `${heading("Docs", T.title, "")}
      <div class="doc-section">
        <h3>${T.model}</h3><p>V3.1 selects R-GCN, HGT, or CC-HHGT using validation data only, then averages three calibrated seeds for cancer × lncRNA × exact-pathway scores. Logistic L1 LASSO is the matched primary article baseline. Pathway family is hierarchy metadata, not the prediction target.</p>
        <h3>${T.data}</h3><p>Data release: 2026.08. 33 TCGA cancers. Reference genome: GRCh38.</p>
        <h3>${T.scores}</h3><ul><li>Calibrated probability: three-seed exact-pathway model mean</li><li>Seed standard deviation: ensemble uncertainty</li><li>Observed evidence score: joined after inference and never changes model probability</li><li>Member weight: deterministic exact-pathway gene-set ranking</li></ul>
        <h3>${T.limitations}</h3><p>${T.notClinical}</p>
      </div>`;
  },

  async releases() {
    $p.innerHTML = heading("Releases", I18N[currentLang].nav.releases, "") + `<div class="panel"><table><tr><td>2026.08</td><td>CancerLncAtlas V3.1</td><td>33-cancer exact-pathway model; HNSC/LGG formal; pathway family auxiliary only</td></tr><tr><td>2026.08</td><td>V2.9/V3.0 add-ons</td><td>State graph and clinical interpretation retained as separate layers</td></tr></table></div>`;
  }
};

// ════════════════════════════════════════════════════════════
//  ROUTER
// ════════════════════════════════════════════════════════════
async function render() {
  $p.innerHTML = '<div class="skeleton-page"><div class="skel-bar w80"></div><div class="skel-bar w60"></div></div>';
  const rt = route();
  document.title = I18N[currentLang].siteTitle + " · " + ({home:"Home",cancer:I18N[currentLang].nav.cancer,lncrna:I18N[currentLang].nav.lncrna,geneset:"Gene Set",sc:I18N[currentLang].nav.singleCell,network:"Network",smart:I18N[currentLang].nav.analyze,datasets:I18N[currentLang].nav.datasets,downloads:I18N[currentLang].nav.downloads,docs:I18N[currentLang].nav.documentation,releases:I18N[currentLang].nav.releases}[rt]||rt);
  buildNav();
  updateLangSwitch();
  const map = { home: "home", cancer: "cancer", lncrna: "lncrna", geneset: "geneset", sc: "sc", network: "network", smart: "smart", datasets: "datasets", downloads: "downloads", docs: "docs", releases: "releases" };
  const fn = map[rt];
  if (fn) await pages[fn]();
  $main?.focus();
}

// ════════════════════════════════════════════════════════════
//  GLOBAL SEARCH
// ════════════════════════════════════════════════════════════
$searchInput?.addEventListener("input", async function () {
  const q = this.value.trim();
  if (q.length < 2) { $searchResults.hidden = true; return; }
  try {
    const data = await api("/api/site/search?q=" + encodeURIComponent(q) + "&limit=8");
    $searchResults.innerHTML = (data.results||[]).map(r => `<a href="${esc(r.href||'/')}" class="search-result-item" data-link><strong>${esc(r.label||r.id)}</strong><small>${esc(r.subtitle||r.type)}</small></a>`).join("");
    $searchResults.hidden = !data.results?.length;
  } catch { $searchResults.hidden = true; }
});

// Global click-away for search results
document.addEventListener("click", e => { if (!$searchInput?.contains(e.target)) $searchResults.hidden = true; });

// Menu toggle
$menuBtn?.addEventListener("click", () => $sidebar.classList.toggle("open"));

// ════════════════════════════════════════════════════════════
//  LINK HANDLING (SPA navigation)
// ════════════════════════════════════════════════════════════
document.addEventListener("click", e => {
  const a = e.target.closest("[data-link]"); if (!a) return;
  e.preventDefault();
  const href = a.getAttribute("href");
  if (href && href !== location.pathname + location.search) {
    history.pushState({}, "", href);
    render();
  }
});

window.addEventListener("popstate", render);

// Initial render
render();
