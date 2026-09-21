"use strict";

const API = "/v3.2-staging";
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;")
  .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const value = id => byId(id)?.value.trim() || "";

function queryString(entries) {
  const search = new URLSearchParams();
  Object.entries(entries).forEach(([key, item]) => {
    if (item !== "" && item !== null && item !== undefined) search.set(key, item);
  });
  const text = search.toString();
  return text ? `?${text}` : "";
}

async function request(url, options = {}) {
  const response = await fetch(url, { signal: AbortSignal.timeout(60000), ...options });
  const body = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
  if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
  return body;
}

function candidateRows(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.rows)) return data.rows;
  if (Array.isArray(data?.results)) return data.results;
  if (Array.isArray(data?.pathways)) return data.pathways;
  if (Array.isArray(data?.associations)) return data.associations;
  if (Array.isArray(data?.entity_results?.results)) return data.entity_results.results;
  if (Array.isArray(data?.patient_results?.results)) return data.patient_results.results;
  return [];
}

function renderResult(data) {
  const rows = candidateRows(data);
  if (!rows.length) {
    byId("query-result").innerHTML = `<pre>${esc(JSON.stringify(data, null, 2))}</pre>`;
    return;
  }
  const columns = [...new Set(rows.slice(0, 20).flatMap(row => Object.keys(row)))].slice(0, 16);
  const renderCell = (row, column) => {
    if (row[column] == null) return '<span class="na">N/A</span>';
    if (column === "figure_id" && data?.query_kind === "formal_figure_availability" && data?.cancer_id) {
      const href = `${API}/single-cell/figure/${encodeURIComponent(data.cancer_id)}/${encodeURIComponent(row[column])}`;
      return `<a href="${esc(href)}" target="_blank" rel="noreferrer">${esc(row[column])}</a>`;
    }
    return esc(row[column]);
  };
  const table = `<div class="table-wrap"><table><thead><tr>${columns.map(column => `<th>${esc(column)}</th>`).join("")}</tr></thead><tbody>${rows.slice(0, 100).map(row => `<tr>${columns.map(column => `<td>${renderCell(row, column)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const summary = `<p><strong>${rows.length.toLocaleString()}</strong> rows in response; showing at most 100. Null is never converted to zero.</p>`;
  byId("query-result").innerHTML = summary + table;
}

function required(input, label) {
  if (!input) throw new Error(`${label} is required for this query`);
  return input;
}

function common() {
  return {
    cancer: value("v32-cancer").toUpperCase(),
    lnc: value("v32-lncrna"),
    pathway: value("v32-pathway"),
    geneSet: value("v32-gene-set"),
    drug: value("v32-drug"),
    endpoint: value("v32-endpoint"),
    state: value("v32-state"),
    relationship: value("v32-relationship"),
    cellType: value("v32-cell-type"),
    compartment: value("v32-compartment"),
    scAvailability: value("v32-sc-availability") || "ALL",
  };
}

const QUERIES = {
  exact: v => request(`${API}/exact-pathway/associations${queryString({ cancer_id: v.cancer, lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
  "gene-set": v => v.geneSet
    ? request(`${API}/gene-sets/${encodeURIComponent(v.geneSet)}${queryString({ member_limit: 100 })}`)
    : request(`${API}/gene-sets${queryString({ cancer_id: v.cancer, pathway_id: v.pathway, limit: 100 })}`),
  subtype: v => request(`${API}/ranked-subtypes/detail${queryString({ cancer_id: v.cancer, pathway_id: v.pathway, limit: 100 })}`),
  async mixed(v) {
    const members = value("v32-members").split(/[\n,;]+/).map(item => item.trim()).filter(Boolean);
    if (!members.length) throw new Error("Provide at least one lncRNA or protein/gene identifier");
    return request(`${API}/enrichment/mixed-exact-pathway`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ members, cancer_id: v.cancer || null, top_k: 50 }),
    });
  },
  state: v => request(`${API}/state${queryString({ cancer_id: v.cancer, lncrna_id: v.lnc, state_id: v.state, limit: 100 })}`),
  "state-gene-set": v => request(`${API}/state-gene-sets${queryString({ cancer_id: v.cancer, state_id: v.state, limit: 100 })}`),
  clinical: v => request(`${API}/clinical${queryString({ clinical_endpoint: v.endpoint, cancer_id: v.cancer, subject_id: v.lnc, limit: 100 })}`),
  "clinical-priority": v => request(`${API}/clinical/translational-priority${queryString({ cancer_id: required(v.cancer, "Cancer ID"), endpoint: v.endpoint, limit: 100 })}`),
  mutation: v => request(`${API}/genomic${queryString({ modality: "mutation", cancer_id: v.cancer, lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
  "mutation-subgroup": v => request(`${API}/mutation/subgroups${queryString({ cancer_id: required(v.cancer, "Cancer ID"), lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
  cnv: v => request(`${API}/genomic${queryString({ modality: "cnv", cancer_id: v.cancer, lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
  "cnv-coverage": v => request(`${API}/cnv/coverage${queryString({ cancer_id: v.cancer })}`),
  drug: v => request(`${API}/drug/response-actionability${queryString({ cancer_id: required(v.cancer, "Cancer ID"), lncrna_id: required(v.lnc, "lncRNA ID"), drug_id: required(v.drug, "Drug ID") })}`),
  "drug-mechanism": v => request(`${API}/drug/structural-mechanisms${queryString({ lncrna_id: required(v.lnc, "lncRNA ID"), cancer_id: v.cancer, drug_id: v.drug, pathway_id: v.pathway, limit: 100 })}`),
  sc: v => request(`${API}/single-cell/exact-pathway-associations${queryString({ cancer_id: v.cancer, lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
  "sc-expression": v => request(`${API}/single-cell/lncrna-expression-summary${queryString({ cancer_id: v.cancer, limit: 100 })}`),
  "sc-audit": v => request(`${API}/single-cell/audit/coverage${queryString({ cancer_id: v.cancer, limit: 100 })}`),
  "sc-gaps": () => request(`${API}/single-cell/audit/gaps`),
  "sc-context-capability": () => request(`${API}/single-cell/context/capability`),
  "sc-context-lncrna": v => request(`${API}/single-cell/context/lncrna-celltype${queryString({ cancer_id: required(v.cancer, "Cancer ID"), lncrna_id: v.lnc, cell_type: v.cellType, compartment: v.compartment, availability: v.scAvailability, limit: 100 })}`),
  "sc-context-pathway": v => request(`${API}/single-cell/context/pathway-activity${queryString({ cancer_id: required(v.cancer, "Cancer ID"), pathway_id: v.pathway, cell_type: v.cellType, compartment: v.compartment, availability: v.scAvailability, limit: 100 })}`),
  "sc-context-predictions": v => request(`${API}/single-cell/context/celltype-predictions${queryString({ cancer_id: required(v.cancer, "Cancer ID"), lncrna_id: v.lnc, pathway_id: v.pathway, cell_type: v.cellType, compartment: v.compartment, availability: v.scAvailability, limit: 100 })}`),
  ucell: v => request(`${API}/single-cell/ucell/${encodeURIComponent(required(v.cancer, "Cancer ID"))}/donor-celltype-scores${queryString({ pathway_id: v.pathway, availability: v.scAvailability === "UNAVAILABLE" ? "TYPED_UNAVAILABLE" : v.scAvailability, limit: 100 })}`),
  "sc-activity": v => request(`${API}/single-cell/activity/${encodeURIComponent(required(v.cancer, "Cancer ID"))}${queryString({ pathway_id: v.pathway, limit: 100 })}`),
  pseudotime: v => request(`${API}/single-cell/trajectory/${encodeURIComponent(required(v.cancer, "Cancer ID"))}${queryString({ level: "PATHWAY", pathway_id: v.pathway, limit: 100 })}`),
  "sc-figures": v => request(`${API}/single-cell/figures/${encodeURIComponent(required(v.cancer, "Cancer ID"))}`),
  expression: v => request(`${API}/lncrna/${encodeURIComponent(required(v.lnc, "lncRNA ID"))}/expression${queryString({ cancer_id: v.cancer, limit: 100 })}`),
  survival: v => request(`${API}/lncrna/${encodeURIComponent(required(v.lnc, "lncRNA ID"))}/survival${queryString({ cancer_id: v.cancer, clinical_endpoint: v.endpoint, limit: 100 })}`),
  coexpression: v => request(`${API}/lncrna/${encodeURIComponent(required(v.lnc, "lncRNA ID"))}/coexpression${queryString({ cancer_id: v.cancer, limit: 100 })}`),
  validation: v => v.lnc
    ? request(`${API}/lncrna/${encodeURIComponent(v.lnc)}/external-validation${queryString({ cancer_id: v.cancer, limit: 100 })}`)
    : request(`${API}/validation/external${queryString({ limit: 100 })}`),
  activity: v => request(`${API}/activity/continuous/${encodeURIComponent(required(v.cancer, "Cancer ID"))}${queryString({ pathway_id: v.pathway, limit: 100 })}`),
  physical: v => request(`${API}/interaction/relationships${queryString({ lncrna_id: required(v.lnc, "lncRNA ID"), cancer_scope: v.cancer, limit: 100 })}`),
  network: v => request(`${API}/network/lncrna/${encodeURIComponent(required(v.lnc, "lncRNA ID"))}${queryString({ cancer_id: v.cancer, pathway_id: v.pathway, limit: 100 })}`),
  "interaction-evidence": v => request(`${API}/interaction/evidence${queryString({ relationship_id: required(v.relationship, "Relationship ID"), limit: 100 })}`),
  evidence: v => request(`${API}/evidence/confidence${queryString({ lncrna_id: required(v.lnc, "lncRNA ID"), cancer_id: v.cancer, pathway_id: v.pathway, limit: 100 })}`),
  "evidence-direction": v => request(`${API}/evidence/direction/probabilities${queryString({ lncrna_id: v.lnc, cancer_id: v.cancer, pathway_id: v.pathway, limit: 100 })}`),
  experiment: v => request(`${API}/experiment/perturbation/evidence-bridge${queryString({ cancer_id: v.cancer, lncrna_id: v.lnc, pathway_id: v.pathway, limit: 100 })}`),
};

async function runQuery(name) {
  const button = document.querySelector(`[data-query="${CSS.escape(name)}"]`);
  const status = byId("query-status");
  // Keep this template literal closed.  A stale mojibake sequence here
  // swallowed the following statements into a string at runtime.
  status.textContent = `Loading ${name}...`;
  byId("query-result").innerHTML = '<div class="skeleton-table"><div class="skel-bar w80"></div><div class="skel-bar w60"></div></div>';
  button?.setAttribute("disabled", "disabled");
  try {
    const data = await QUERIES[name](common());
    renderResult(data);
    status.textContent = `${name}: PASS`;
  } catch (error) {
    byId("query-result").innerHTML = `<div class="error-block"><strong>Query failed closed</strong><p>${esc(error.message)}</p></div>`;
    status.textContent = `${name}: unavailable or invalid input`;
  } finally {
    button?.removeAttribute("disabled");
  }
}

function badge(status) {
  if (status === "QUERYABLE_STAGING") return ["ready", "QUERYABLE"];
  if (status === "PARTIAL_STAGING") return ["partial", "PARTIAL"];
  return ["pending", "PENDING"];
}

function renderCapabilities(catalog, filter = "ALL") {
  const rows = catalog.capabilities.filter(item => filter === "ALL" || item.staging_status === filter);
  byId("capability-grid").innerHTML = rows.map(item => {
    const [klass, label] = badge(item.staging_status);
    const coverage = Array.isArray(item.coverage_scope)
      ? `<div class="v32-meta">${item.coverage_scope.map(scope => `<span><strong>${esc(scope.label)}:</strong> ${esc(scope.available)}/${esc(scope.total)}</span>`).join("")}</div>`
      : "";
    return `<article class="feature-card v32-capability-card" data-capability-id="${esc(item.capability_id)}">
      <span class="v32-badge ${klass}">${label}</span>
      <h3>${esc(item.capability_id)}</h3>
      <p>${esc(item.description)}</p>
      ${coverage}
      ${item.known_gap ? `<p class="v32-gap"><strong>Known gap:</strong> ${esc(item.known_gap)}</p>` : ""}
      <div class="v32-meta"><span><strong>Target:</strong> ${esc(item.target_level)}</span>
      <span><strong>Staging endpoints:</strong> ${item.staging_endpoints.length ? item.staging_endpoints.map(esc).join(" · ") : "not mounted"}</span></div>
      <details><summary>UI surface contract (${item.ui_surface_ids.length})</summary><ul class="v32-surface-list">${item.ui_surface_ids.map(id => `<li data-surface-id="${esc(id)}"><code>${esc(id)}</code></li>`).join("")}</ul></details>
      <details><summary>Download IDs (${item.download_ids.length})</summary><ul class="v32-surface-list">${item.download_ids.map(id => `<li><button type="button" class="v32-download-link" data-download-id="${esc(id)}"><code>${esc(id)}</code></button></li>`).join("")}</ul></details>
    </article>`;
  }).join("");
}

let downloadCatalog = null;

function renderDownloadEntry(entry) {
  const gap = entry.known_gap || entry.unavailable_reason || "";
  let actions = "";
  if (entry.status === "READY_FILE" && entry.download_url) {
    actions = `<p><a class="button" href="${esc(entry.download_url)}">Download ${esc(entry.file?.relative_name || entry.download_id)}</a></p>`;
  } else if (["READY_PARTS", "PARTIAL_READY_PARTS"].includes(entry.status)) {
    actions = `<details open><summary>${entry.parts?.length || 0} verified parts</summary><ul class="v32-surface-list">${(entry.parts || []).map(part => `<li><a href="${esc(part.download_url)}">${esc(part.relative_name)}</a> · <code>${esc(part.sha256)}</code></li>`).join("")}</ul></details>`;
  } else if (entry.status === "SERVER_HASH_PINNED_DOWNLOAD") {
    const cancer = value("v32-cancer").toUpperCase() || "ACC";
    const route = spec => String(spec || "").replace(/^GET\s+/, "").replace("{cancer_id}", encodeURIComponent(cancer));
    const manifest = route(entry.manifest_endpoint);
    const download = route(entry.download_endpoint);
    const direct = download.includes("{") ? "" : `<a class="button" href="${esc(download)}">Download server artifact</a>`;
    actions = `<p><a class="button" href="${esc(manifest)}">Open verified manifest</a> ${direct}</p><p>Server-side payload · request-time SHA256 recheck · scope: <code>${esc(entry.scope)}</code></p>`;
  } else if (entry.status === "DYNAMIC_QUERY_EXPORT") {
    actions = `<p>Dynamic export: <code>${esc(entry.export_endpoint)}</code></p>`;
  }
  byId("download-result").innerHTML = `<article><span class="v32-badge ${entry.status.startsWith("READY") || entry.status === "SERVER_HASH_PINNED_DOWNLOAD" ? "ready" : entry.status.startsWith("PARTIAL") ? "partial" : "pending"}">${esc(entry.status)}</span><h3>${esc(entry.download_id)}</h3><p>V3.2 · data_present=${esc(entry.data_present)} · download_implemented=${esc(entry.download_implemented)} · contract_complete=${esc(entry.contract_complete)}</p>${gap ? `<p class="v32-gap">${esc(gap)}</p>` : ""}${actions}</article>`;
}

async function loadDownloads() {
  downloadCatalog = await request(`${API}/downloads`);
  const select = byId("download-id");
  select.innerHTML = [...downloadCatalog.downloads]
    .sort((a, b) => a.download_id.localeCompare(b.download_id))
    .map(item => `<option value="${esc(item.download_id)}">${esc(item.download_id)} [${esc(item.status)}]</option>`)
    .join("");
}

async function inspectDownload(downloadId = value("download-id")) {
  const id = required(downloadId, "Download ID");
  byId("download-id").value = id;
  const entry = await request(`${API}/downloads/${encodeURIComponent(id)}`);
  renderDownloadEntry(entry);
  byId("download-result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  return entry;
}

async function startDownload() {
  const entry = await inspectDownload();
  if (entry.status === "SERVER_HASH_PINNED_DOWNLOAD") {
    const cancer = value("v32-cancer").toUpperCase() || "ACC";
    const spec = String(entry.download_endpoint || "").replace(/^GET\s+/, "").replace("{cancer_id}", encodeURIComponent(cancer));
    const target = spec.includes("{") ? String(entry.manifest_endpoint || "").replace(/^GET\s+/, "").replace("{cancer_id}", encodeURIComponent(cancer)) : spec;
    window.location.assign(target);
    return;
  }
  if (entry.status !== "READY_FILE" || !entry.download_url) {
    throw new Error("This ID is not a complete single-file download; use its part/server links or inspect the typed gap.");
  }
  window.location.assign(entry.download_url);
}

async function loadHealth() {
  const health = await request(`${API}/health`);
  // Formal-23 deliberately removes the superseded diagnostic key from the
  // top-level health payload. Prefer the canonical formal-23 status and
  // retain a fallback for older staging servers that still expose the legacy
  // key.
  const diagnosticStatus =
    health.single_cell_formal23_diagnostic ||
    health.single_cell_diagnostic_query ||
    health.legacy_aliases?.single_cell_diagnostic_query ||
    "NOT_MOUNTED";
  const metrics = [
    [health.status, "API"],
    [health.exact_pathway_browse_query, "Exact"],
    [health.gene_set_query, "Gene Set"],
    [health.ranked_subtype_query, "Subtype"],
    [health.single_cell_exact_pathway_query, "Single-cell"],
    [health.single_cell_gap_audit_query, "SC audit"],
    [health.single_cell_formal_context_query, "SC context 23+10"],
    [health.single_cell_formal_context_independent_audit, "SC context audit"],
    [diagnosticStatus, "SC trajectory/figures"],
    [health.evidence_direction_probability_query || health.evidence_query, "Evidence"],
    [health.download_catalog_query, "Downloads"],
    [health.drug_response_actionability_query, "Drug"],
  ];
  byId("health-metrics").innerHTML = metrics.map(([number, label]) => `<div class="stat-card"><strong>${esc(number)}</strong><span>${esc(label)}</span></div>`).join("");
}

async function loadSingleCellContext() {
  const [capability, formalUCell, diagnostic] = await Promise.all([
    request(`${API}/single-cell/context/capability`),
    request(`${API}/single-cell/ucell/capability`).catch(() => null),
    request(`${API}/single-cell/diagnostic/capability`).catch(() => null),
  ]);
  // The current formal-23 sidecar reports eligible and typed-unavailable
  // counts, while the superseded context sidecar reports raw/formal counts.
  // Normalize both contracts here without turning a typed gap into a zero
  // data value.
  const formalEligible = Number(
    capability.formal_eligible_cancer_count ??
      capability.formal_context_cancer_count ??
      0,
  );
  const typedUnavailable = Number(
    capability.typed_unavailable_cancer_count ?? 0,
  );
  const rawReported = Number(capability.raw_h5_cancer_count ?? 0);
  const authorityTotal =
    rawReported || formalEligible + typedUnavailable || 33;
  const formal = formalEligible;
  const hasFormal23 =
    capability.module === "single_cell_formal23" ||
    (formalEligible === 23 && typedUnavailable === 10);
  const ucell = capability.cell_level_ucell || {};
  const legacyUCellCount = Number(
    formalUCell?.cancer_count ||
      (Array.isArray(ucell.covered_cancers)
        ? ucell.covered_cancers.length
        : 0),
  );
  const covered = hasFormal23 ? formalEligible : legacyUCellCount;
  const formalTotal = hasFormal23
    ? formalEligible
    : Number(
        formalUCell?.cancer_count ||
          ucell.formal_cancers_total ||
          formal ||
          authorityTotal,
      );
  const diagnosticCount = hasFormal23
    ? null
    : Number(diagnostic?.cancer_count || 0);
  byId("sc-context-coverage").innerHTML = [
    [
      rawReported || authorityTotal,
      authorityTotal,
      rawReported ? "Raw H5 audited" : "33-cancer authority",
    ],
    [formal, authorityTotal, "Formal datasets"],
    [
      covered,
      formalTotal,
      hasFormal23 ? "Donor-level activity" : "Cell-level UCell",
    ],
    [
      diagnosticCount,
      hasFormal23 ? null : formalTotal,
      hasFormal23
        ? "Pseudotime + figures (typed unavailable)"
        : "Pseudotime + figures",
    ],
  ].map(([available, total, label]) => `<div class="stat-card"><strong>${available == null ? "N/A" : `${esc(available)}/${esc(total)}`}</strong><span>${esc(label)}</span></div>`).join("");
  const audit = capability.independent_audit || {};
  const auditText = hasFormal23
    ? "PASS (hash-pinned formal-23 authority)"
    : `${audit.check_count || 0}/${audit.total_checks || 25} PASS`;
  const ucellNote = hasFormal23
    ? `Donor-level activity: ${formalEligible}/${formalEligible} formal-eligible cancers, hash-pinned and independently audited.`
    : formalUCell
      ? `Legacy cell-level UCell: ${legacyUCellCount} covered cancers, hash-pinned and independently audited.`
      : "Cell-level UCell legacy binding is not mounted.";
  const diagnosticNote = hasFormal23
    ? "Pseudotime/figures are typed-unavailable in the formal-23 release; no numeric values are exposed."
    : diagnostic
      ? `Diagnostic pseudotime/figures: ${diagnostic.cancer_count} cancers; inferred root, primary/secondary weights both ${diagnostic.primary_score_weight}/${diagnostic.secondary_score_weight}.`
      : "Diagnostic pseudotime/figure publication is not mounted.";
  byId("sc-context-note").textContent = `Formal scope audit: ${auditText}; current secondary-score change: ${capability.single_cell_currently_changes_secondary_score === true}. ${ucellNote} ${diagnosticNote}`;
  return capability;
}

async function initialize() {
  document.querySelectorAll("[data-query]").forEach(button => {
    button.addEventListener("click", () => runQuery(button.dataset.query));
  });
  byId("download-refresh")?.addEventListener("click", () => loadDownloads().catch(showDownloadError));
  byId("download-inspect")?.addEventListener("click", () => inspectDownload().catch(showDownloadError));
  byId("download-start")?.addEventListener("click", () => startDownload().catch(showDownloadError));
  byId("capability-grid")?.addEventListener("click", event => {
    const button = event.target.closest("[data-download-id]");
    if (button) inspectDownload(button.dataset.downloadId).catch(showDownloadError);
  });
  try {
    const [catalog] = await Promise.all([
      request("/v32-capability-catalog.json"),
      loadHealth(),
      loadDownloads(),
      loadSingleCellContext().catch(error => {
        byId("sc-context-coverage").innerHTML = `<div class="error-block"><strong>Formal context unavailable</strong><p>${esc(error.message)}</p></div>`;
        byId("sc-context-note").textContent = "No coverage level is promoted when its audited capability cannot be loaded.";
      }),
    ]);
    renderCapabilities(catalog);
    byId("capability-filter").addEventListener("change", event => renderCapabilities(catalog, event.target.value));
  } catch (error) {
    byId("health-metrics").innerHTML = `<div class="error-block"><strong>Staging unavailable</strong><p>${esc(error.message)}</p></div>`;
  }
}

function showDownloadError(error) {
  byId("download-result").innerHTML = `<div class="error-block"><strong>Download failed closed</strong><p>${esc(error.message)}</p></div>`;
}

initialize();
