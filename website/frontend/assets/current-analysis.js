"use strict";
(() => {
  const el = id => document.getElementById(id);
  const labels = {
    geneset_master:"基因集目录",geneset_members:"基因集成员",geneset_coverage:"基因集覆盖",gmt:"GMT 基因集",
    candidates:"完整候选评分",pathway_context_subtype:"通路跨癌种分型",pathway_conservation:"通路保守性",
    pathway_pairwise_similarity:"癌种间通路相似性",cancer_program_subtype:"癌种程序分型",cancer_program_similarity:"癌种程序相似性",
    selected_geneset_state_profiles:"基因集状态概况",selected_member_states:"成员状态关联",selected_state_coverage:"状态覆盖",
    native_state_aggregate:"状态关联汇总",state_geneset_catalog:"状态基因集",state_geneset_members:"状态基因集成员",
    selected_geneset_clinical_support:"基因集临床支持",selected_member_clinical:"成员临床关联",native_entity_clinical:"实体临床关联",
    patient_risk_fold_coverage:"患者风险覆盖",selected_drug_prediction_profiles:"药物关联强度概况",selected_drug_predictions:"成员药物关联预测",
    selected_associations:"实测药敏关联",selected_associations_direction_profiles:"实测药敏方向概况",
    selected_dose_associations:"逐剂量实测关联",selected_dose_contrasts:"同模型剂量变化",
    selected_dose_associations_direction_profiles:"逐剂量方向概况",selected_dose_contrasts_direction_profiles:"剂量变化方向概况",
    selected_baseline:"模型消融基线",selected_attribution:"靶点与通路消融",selected_pathway_attribution_profiles:"通路消融概况",
    drug_pharmacology:"药物结构与药理",drug_target_annotations:"药物靶点注释",cancer_subtype_coverage:"癌种和亚型覆盖",
    dose_context_coverage:"剂量分析覆盖",summary:"规则证据汇总",source_events:"来源事件",source_coverage:"来源覆盖",
    events:"候选证据事件",contexts:"实验背景",metrics:"评价指标",coverage:"评价覆盖",details:"逐项评价",isolation:"来源隔离",model_use:"已登记模型使用来源",
    cancer_id:"癌种",lncrna_id:"lncRNA",pathway_id:"精确通路",pathway_family_id:"通路家族",geneset_id:"基因集 ID",
    geneset_name:"基因集名称",geneset_rank:"成员排序",selected_score:"选定评分",member_count:"成员数",
    association_direction:"关联方向",drug_id:"药物稳定 ID",gene_id:"靶基因",subtype_id:"亚型",subtype_system:"亚型体系",
    clinical_endpoint:"临床终点",state_id:"状态",subject_id:"实体 ID",subject_type:"实体类型",source_id:"来源",
    positive_available_weight_fraction:"正向成员权重占比",negative_available_weight_fraction:"负向成员权重占比",
    mean_association_strength_probability:"平均关联强度概率",drug_response_association_probability:"关联强度概率",
    signed_expression_sensitivity_rho:"实测药敏相关系数",observed_direction:"实测方向",unique_cell_models:"独立细胞模型数",
    mean_observed_oriented_response:"实测平均变化/反应",measured_response_available:"有实测反应",association_available:"相关性可用",
    availability:"可用",available:"可用",unavailable_reason:"不可用原因",failure_reason:"缺失原因",
    member_coverage:"成员覆盖率",selected_weight_coverage:"评分权重覆盖率",summary_available:"汇总可用",
    dose_molar_key:"剂量（M）",low_dose_molar_key:"低剂量（M）",high_dose_molar_key:"高剂量（M）",
    exposure_hours_key:"暴露时间（小时）",response_metric:"反应指标",response_unit:"反应单位",
    ablation_kind:"消融类型",ablation_entity_id:"消融靶点/通路",delta_probability:"消融概率差",cell_line_fold:"细胞模型折",
    clinical_effect_direction:"临床关联方向",meta_hazard_ratio:"合并风险比",state_effect:"状态相关性",
    state_membership_probability:"状态成员概率",mean_state_membership_probability:"平均状态成员概率",
    mean_held_out_state_effect:"平均留出状态相关性",descriptive_effect_direction:"描述性关联方向",availability_reason:"可用性说明",
    supported_risk_weight_fraction:"风险关联支持权重占比",supported_protective_weight_fraction:"保护关联支持权重占比",
    metric:"指标",value:"数值",estimate:"估计值",model_id:"模型",endpoint:"评价终点",isolation_scope:"隔离范围"
  };
  const preferred = ["geneset_id","geneset_name","cancer_id","pathway_id","lncrna_id","geneset_rank","selected_score","association_direction","member_count",
    "drug_id","subtype_system","subtype_id","state_id","clinical_endpoint","subject_type","subject_id","mean_association_strength_probability",
    "drug_response_association_probability","signed_expression_sensitivity_rho","observed_direction","unique_cell_models","dose_molar_key","low_dose_molar_key",
    "high_dose_molar_key","exposure_hours_key","response_metric","response_unit","mean_observed_oriented_response","measured_response_available","association_available",
    "positive_available_weight_fraction","negative_available_weight_fraction","ablation_kind","ablation_entity_id","cell_line_fold","delta_probability",
    "state_membership_probability","mean_state_membership_probability","state_effect","mean_held_out_state_effect","descriptive_effect_direction",
    "clinical_effect_direction","meta_hazard_ratio","supported_risk_weight_fraction","supported_protective_weight_fraction",
    "member_coverage","selected_weight_coverage","summary_available","availability","available","availability_reason","unavailable_reason","failure_reason","metric","value","estimate"];
  const statusLabels = {AVAILABLE:"可查询",NOT_PRODUCED:"尚未生成",SOURCE_UNAVAILABLE:"来源不可用",NOT_APPLICABLE:"不适用"};
  let catalog, active, selectedTable, offset = 0, rows = [], requestId = 0, controller;
  let drugOffset = 0, drugRequest = 0;
  const limit = 50;
  function node(tag,text,cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
  function message(text,error=false) { el("message").textContent=text; el("message").className="message"+(error?" error":""); el("message").hidden=!text; }
  async function fetchJSON(url,options={}) {
    const response=await fetch(url,options);
    if(!response.ok) { const body=await response.json().catch(()=>({})); throw new Error(response.status===409?"分析版本已变化，请重新打开页面选择版本。":typeof body.detail==="string"?body.detail:`查询失败（${response.status}）`); }
    return response.json();
  }
  function downloadURL(table,part) { return `/api/current/downloads/${encodeURIComponent(table.id)}/${part}?version=${catalog.version}`; }
  function display(value) {
    if(value===null||value===undefined)return "不可用";
    if(typeof value==="boolean")return value?"是":"否";
    if(typeof value==="number")return value===0?"0":Number.isInteger(value)?value.toLocaleString():Math.abs(value)<.0001?value.toExponential(3):Number(value.toPrecision(6)).toString();
    return typeof value==="object"?JSON.stringify(value):String(value);
  }
  function renderTable(target,data,all=false) {
    target.replaceChildren();
    if(!data.length){target.append(node("p","当前条件下没有结果。未出现的记录不代表数值为零。","empty"));return;}
    const available=Object.keys(data[0]);
    const columns=all?available:preferred.filter(k=>available.includes(k));
    if(!columns.length)columns.push(...available.slice(0,12));
    const table=node("table"),head=node("thead"),tr=node("tr"),body=node("tbody");
    columns.forEach(key=>{const th=node("th",labels[key]||key);th.scope="col";th.title=key;tr.append(th);});head.append(tr);table.append(head,body);
    for(const row of data){const r=node("tr");for(const key of columns){const cell=node("td");
      if(key==="geneset_id"&&row[key]){const link=node("button",row[key],"member-link");link.title=row[key];link.type="button";link.addEventListener("click",()=>selectComponent("selected","selected.geneset_members",{geneset_id:row[key]}));cell.append(link);}
      else{cell.textContent=display(row[key]);if(row[key]===null)cell.className="missing";if(cell.textContent.length>100)cell.classList.add("long");}r.append(cell);}body.append(r);}
    target.append(table);
  }
  function values(){return Object.fromEntries([...new FormData(el("filters-form"))].filter(([,v])=>String(v).length));}
  async function query() {
    const id=++requestId;controller?.abort();controller=new AbortController();message("");
    el("row-status").textContent="正在查询…";el("previous").disabled=el("next").disabled=true;
    const filters=values(),url=new URL(location.href);url.searchParams.set("table",selectedTable.id);url.searchParams.set("version",catalog.version);
    url.searchParams.set("filters",JSON.stringify(filters));history.replaceState(null,"",url);
    try{const result=await fetchJSON(`/api/current/tables/${encodeURIComponent(selectedTable.id)}/query`,{method:"POST",headers:{"Content-Type":"application/json"},signal:controller.signal,
      body:JSON.stringify({version:catalog.version,filters,limit,offset})});
      if(id!==requestId)return;rows=result.rows;renderTable(el("table-container"),rows,el("all-columns").checked);
      el("row-status").textContent=rows.length?`第 ${offset+1}–${offset+rows.length} 条 · 本表共 ${selectedTable.rows.toLocaleString()} 条` : "没有匹配记录";
      el("page-number").textContent=`第 ${offset/limit+1} 页`;el("previous").disabled=offset===0;el("next").disabled=!result.has_more;
    }catch(error){if(error.name!=="AbortError"&&id===requestId){message(error.message,true);el("row-status").textContent="查询失败";rows=[];el("table-container").replaceChildren();}}
  }
  function selectTable(identity,filters={}) {
    selectedTable=catalog.tables.find(t=>t.id===identity);offset=0;el("table-select").value=identity;
    el("primary-filters").replaceChildren();el("secondary-filters").replaceChildren();
    for(const key of selectedTable.filters){const label=node("label",labels[key]||key),input=node("input");input.name=key;input.autocomplete="off";input.value=filters[key]||"";label.append(input);
      el(["cancer_id","lncrna_id","pathway_id","geneset_id"].includes(key)?"primary-filters":"secondary-filters").append(label);}
    el("extra-filters").hidden=!el("secondary-filters").children.length;
    el("table-download").hidden=selectedTable.parts.length!==1;
    el("table-download").href=downloadURL(selectedTable,0);el("table-download").textContent="下载此表";
    query();
  }
  function selectComponent(component,identity,filters={}) {
    ++requestId;controller?.abort();active=component;
    for(const button of el("components").children)button.setAttribute("aria-pressed",String(button.dataset.component===component));
    const info=catalog.components[component];el("component-title").textContent=info.label;el("component-description").textContent=info.description;
    el("component-missing").hidden=info.status==="AVAILABLE";el("controls").hidden=info.status!=="AVAILABLE";
    if(info.status!=="AVAILABLE"){el("component-missing").textContent=`${statusLabels[info.status]}：${info.reason}`;return;}
    const tables=catalog.tables.filter(t=>t.component===component&&t.kind==="parquet");el("table-select").replaceChildren();
    for(const t of tables){const option=node("option",labels[t.name]||t.name);option.value=t.id;el("table-select").append(option);}
    const defaults={selected:"geneset_master",scores:"candidates",state:"selected_geneset_state_profiles",clinical:"selected_geneset_clinical_support",drug:"selected_drug_prediction_profiles",evidence:"summary",external:"metrics"};
    const chosen=tables.find(t=>t.id===identity)||tables.find(t=>t.name===defaults[component])||tables[0];
    el("drug-lookup").hidden=component!=="drug";
    if(!chosen){el("controls").hidden=true;el("component-missing").hidden=false;el("component-missing").textContent="本模块仅提供下载文件。";return;}
    selectTable(chosen.id,filters);
  }
  function renderCatalog(){
    el("release-name").textContent=catalog.release_id;
    const details={"选定评分版本":catalog.selected_score_run_id,"当前分析校验值":catalog.selected_analysis_sha256,
      "评分科学状态":catalog.score_scientific_status,"评分评价结果":catalog.score_performance_outcome,
      "独立外部验证":"尚未证明完整模型使用来源清单；请查阅外部评估的具体隔离范围。"};
    for(const [key,value] of Object.entries(details))el("version-details").append(node("dt",key),node("dd",value));
    for(const [key,info] of Object.entries(catalog.components)){const button=node("button");button.type="button";button.dataset.component=key;button.setAttribute("aria-pressed","false");
      button.append(node("span",info.label),node("small",statusLabels[info.status]));button.addEventListener("click",()=>selectComponent(key));el("components").append(button);}
    for(const [key,info] of Object.entries(catalog.components)){const group=node("details"),summary=node("summary",info.label);group.append(summary);
      if(info.status!=="AVAILABLE")group.append(node("p",`${statusLabels[info.status]}：${info.reason}`));
      else{const list=node("ul");for(const t of catalog.tables.filter(t=>t.component===key))for(const part of t.parts){const li=node("li"),a=node("a",(labels[t.name]||t.name)+(t.parts.length>1?` · 分区 ${part.part+1}`:""));
        a.href=downloadURL(t,part.part);li.append(a,node("span",`${(part.bytes/1024).toFixed(1)} KB`),node("code",`SHA256 ${part.sha256}`));list.append(li);}group.append(list);}el("download-list").append(group);}
  }
  async function drugQuery(){
    const id=++drugRequest,form=new FormData(el("drug-form"));el("drug-previous").disabled=el("drug-next").disabled=true;
    try{const p=new URLSearchParams({geneset_id:form.get("geneset_id"),drug_id:form.get("drug_id"),version:catalog.version,limit:String(limit),offset:String(drugOffset)});
      const result=await fetchJSON(`/api/current/drug-members?${p}`);if(id!==drugRequest)return;
      renderTable(el("drug-results"),result.rows);el("drug-previous").disabled=drugOffset===0;el("drug-next").disabled=result.rows.length<limit;
      el("drug-page").textContent=`第 ${drugOffset/limit+1} 页`;
    }catch(error){if(id===drugRequest){el("drug-results").replaceChildren();message(error.message,true);}}
  }
  el("filters-form").addEventListener("submit",event=>{event.preventDefault();offset=0;query();});
  el("table-select").addEventListener("change",event=>selectTable(event.target.value));
  el("clear-filters").addEventListener("click",()=>{for(const input of el("filters-form").querySelectorAll("input[name]"))input.value="";offset=0;query();});
  el("all-columns").addEventListener("change",()=>renderTable(el("table-container"),rows,el("all-columns").checked));
  el("previous").addEventListener("click",()=>{offset=Math.max(0,offset-limit);query();});el("next").addEventListener("click",()=>{offset+=limit;query();});
  el("downloads-toggle").addEventListener("click",()=>{el("downloads").hidden=!el("downloads").hidden;el("downloads-toggle").setAttribute("aria-expanded",String(!el("downloads").hidden));if(!el("downloads").hidden)el("downloads").scrollIntoView({behavior:"smooth"});});
  el("drug-form").addEventListener("submit",event=>{event.preventDefault();drugOffset=0;drugQuery();});
  el("drug-previous").addEventListener("click",()=>{drugOffset=Math.max(0,drugOffset-limit);drugQuery();});el("drug-next").addEventListener("click",()=>{drugOffset+=limit;drugQuery();});
  (async()=>{try{catalog=await fetchJSON("/api/current/catalog");const p=new URLSearchParams(location.search);
    if(p.has("version")&&p.get("version")!==catalog.version)throw new Error("此链接属于另一分析版本。请从网站首页打开当前版本；原链接的结果不会自动替换。");
    renderCatalog();const table=catalog.tables.find(t=>t.id===p.get("table"));let filters={};
    if(p.has("filters")){
      filters=JSON.parse(p.get("filters"));
      if(!filters||typeof filters!=="object"||Array.isArray(filters)||!Object.values(filters).every(v=>typeof v==="string"))throw new Error("链接中的筛选条件无效，请检查后重试。");
    }
    selectComponent(table?.component||"selected",table?.id,filters);
  }catch(error){message(error.message,true);el("component-title").textContent="当前分析无法载入";}})();
})();
