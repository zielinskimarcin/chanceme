"use strict";
const $ = function (id) { return document.getElementById(id); };
const form = $("profile-form");
const finite = function (n) { return typeof n === "number" && Number.isFinite(n); };
let ready = false, activeRequest = null, generation = 0, hasInteracted = false, lastType = "SAT";
let gpaEstimated = false;
const typeScores = { SAT: "", Bocconi: "" };
let ringAnimation = null, resultAnimation = null, redrawEvidenceChart = null;
const evidenceViewport = window.matchMedia("(max-width: 600px)");
if (typeof evidenceViewport.addEventListener === "function") evidenceViewport.addEventListener("change", function () { if (typeof ChanceReport !== "undefined") ChanceReport.redraw(); else if (redrawEvidenceChart) redrawEvidenceChart(); });
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function node(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = String(text);
  if (className) item.className = className;
  return item;
}
function readNumber(value) {
  const clean = String(value).trim().replace(",", ".");
  return clean === "" || !/^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(clean) ? NaN : Number(clean);
}
function announce(text) { $("result-announcement").textContent = text; }
function buttonLabel(text) { $("calculate-label").textContent = text; }
function error(id, text) {
  const message = $(id + "-error");
  if (message) { message.textContent = text; message.hidden = !text; }
  const field = $(id);
  if (field) { if (text) field.setAttribute("aria-invalid", "true"); else field.removeAttribute("aria-invalid"); }
}
function clearErrors() {
  ["score", "gpa", "course", "session"].forEach(function (id) { error(id, ""); });
  $("form-error").hidden = true;
}
function stopResultAnimation() {
  if (ringAnimation) ringAnimation.cancel();
  if (resultAnimation) resultAnimation.cancel();
  ringAnimation = null; resultAnimation = null;
}
function animateResult(probability, formal) {
  stopResultAnimation();
  if (reduceMotion.matches) return;
  const ring = $("ring-value"), result = $("calculated-state");
  if (!formal && typeof ring.animate === "function") {
    ringAnimation = ring.animate([
      { strokeDasharray: "0 100" },
      { strokeDasharray: String(probability * 100) + " 100" }
    ], { duration: 1100, delay: 100, easing: "cubic-bezier(.22,.65,.25,1)", fill: "backwards" });
  }
  if (typeof result.animate === "function") {
    resultAnimation = result.animate([
      { opacity: 0, transform: "translateY(6px)" },
      { opacity: 1, transform: "translateY(0)" }
    ], { duration: 420, easing: "ease-out" });
  }
}
function emptyState(title, text, state) {
  if (typeof ChanceReport !== "undefined") ChanceReport.reset();
  stopResultAnimation();
  redrawEvidenceChart = null;
  const panel = $("result-panel");
  panel.classList.add("is-empty");
  panel.classList.remove("is-formal");
  panel.classList.toggle("is-loading", state === "loading");
  panel.dataset.state = state || "empty";
  delete panel.dataset.tone;
  panel.setAttribute("aria-busy", state === "loading" ? "true" : "false");
  $("empty-state").hidden = false;
  $("calculated-state").hidden = true;
  $("empty-headline").textContent = title;
  $("empty-description").textContent = text;
  $("panel-independent").hidden = false;
  $("evidence").hidden = true;
  $("evidence").open = false;
  $("evidence-body").replaceChildren();
  $("percentage").replaceChildren();
}
function invalidate() {
  generation++;
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  $("calculate").disabled = !ready;
  buttonLabel("Calculate my chances");
  clearErrors();
  if (hasInteracted) {
    emptyState("Your profile has changed.", "Calculate again to see the estimate for these scores.", "stale");
    announce("Profile changed. Calculate again for an updated estimate.");
  }
}
function updateTestType() {
  const type = form.elements.test_type.value;
  if (type !== lastType) {
    typeScores[lastType] = $("score").value;
    $("score").value = typeScores[type];
    lastType = type;
  }
  const sat = type === "SAT";
  $("score-label").textContent = sat ? "SAT score" : "Bocconi Test score";
  $("score-max").textContent = sat ? "/ 1600" : "/ 50";
  $("score").placeholder = sat ? "e.g. 1450" : "e.g. 35";
  $("score").inputMode = sat ? "numeric" : "decimal";
}
function readInput() {
  const payload = {
    test_type: form.elements.test_type.value,
    score: readNumber($("score").value), gpa: readNumber($("gpa").value),
    course: $("course").value, session: $("session").value,
    cycle: "27-28", applicant_category: "international",
    // Legacy API flag: submitting a valid GPA no longer needs a separate checkbox.
    gpa_confirmed: true,
    gpa_estimated: gpaEstimated
  };
  return payload;
}
function validate(input) {
  clearErrors();
  let first = null;
  function fail(id, text) { error(id, text); if (!first) first = $(id); }
  if (input.test_type === "SAT") {
    if (!finite(input.score) || input.score < 400 || input.score > 1600 || input.score % 10 !== 0) fail("score", "Enter a SAT score from 400 to 1600, in steps of 10.");
  } else if (!finite(input.score) || input.score < -11.17 || input.score > 50) fail("score", "Enter a Bocconi Test score from −11.17 to 50.");
  if (!finite(input.gpa) || input.gpa < 6 || input.gpa > 10) fail("gpa", "Enter a Bocconi GPA between 6 and 10.");
  if (!input.course) fail("course", "Choose a programme.");
  if (!["Early", "Winter"].includes(input.session)) fail("session", "Choose an application round.");
  if (first) { first.focus(); announce("Check the highlighted fields before calculating."); return false; }
  return true;
}
function resultCopy(p, input) {
  let copy;
  if (p >= .85) copy = ["Your profile scores highly in this model.", "Your scores sit towards the stronger end of the model’s range. Admission still depends on the applicant pool and a complete, eligible application.", "Check the historical context and your programme choices."];
  else if (p >= .65) copy = ["A stronger profile in this model.", "Your scores compare favourably with the historical profiles used by the model. The estimate does not predict the exact competition in your round.", "Review the similar reports before deciding your next step."];
  else if (p >= .35) copy = ["An uncertain outcome.", "The model puts your profile in a competitive range, with room for either outcome. Similar past reports can help put the estimate in context.", "Try a higher test score or review programmes you would genuinely want to study."];
  else if (p >= .1) copy = ["A lower estimate, with room to improve.", "Your profile sits lower in the model’s range. A stronger test score may change the estimate; this is not an admissions decision.", "Compare a higher score and keep alternative options open."];
  else copy = ["A low estimate in this model.", "Your current scores sit near the lower end of the model’s range. This does not establish that admission is impossible.", "Review the requirements and consider another test attempt."];
  if (input.score >= (input.test_type === "SAT" ? 1590 : 49.5)) copy[2] = "Your test score is already near its maximum. Review your GPA, programme preferences and application requirements.";
  return copy;
}
function year(value) {
  const match = String(value).match(/^(\d{2}|\d{4})[-/](\d{2}|\d{4})$/);
  return match ? (match[1].length === 2 ? "20" : "") + match[1] + "/" + match[2].slice(-2) : String(value);
}
function format(value) { return Number(value).toLocaleString("en-GB", { maximumFractionDigits: 3 }); }
function table(parent, headings, rows) {
  const wrapper = node("div", undefined, "table-scroll");
  wrapper.tabIndex = 0; wrapper.setAttribute("role", "region"); wrapper.setAttribute("aria-label", headings.join(", ") + " table");
  const el = node("table"), head = node("thead"), tr = node("tr"), body = node("tbody");
  headings.forEach(function (title) { const th = node("th", title); th.scope = "col"; tr.append(th); });
  head.append(tr);
  rows.forEach(function (values) {
    const row = node("tr");
    values.forEach(function (value, index) { const td = node("td"); td.setAttribute("data-label", headings[index]); if (value instanceof Node) td.append(value); else td.textContent = String(value); row.append(td); });
    body.append(row);
  });
  el.append(head, body); wrapper.append(el); parent.append(wrapper);
}
function evidenceCounts(evidence, cycle) {
  if (!cycle) return evidence.nearby || null;
  const row = (evidence.by_year || []).find(function (r) { return r.cycle === cycle; });
  if (!row || ![row.nearby_total, row.nearby_admitted, row.nearby_rejected].every(finite)) return null;
  return { total: row.nearby_total, admitted: row.nearby_admitted, rejected: row.nearby_rejected };
}
function evidenceRecords(evidence, input, cycle) {
  const testWindow = finite(evidence.nearby && evidence.nearby.test_window) && evidence.nearby.test_window > 0 ? evidence.nearby.test_window : input.test_type === "SAT" ? 50 : 4;
  const gpaWindow = finite(evidence.nearby && evidence.nearby.gpa_window) && evidence.nearby.gpa_window > 0 ? evidence.nearby.gpa_window : .5;
  function distance(r) { return Math.abs(r.score - input.score) / testWindow + Math.abs(r.gpa - input.gpa) / gpaWindow; }
  return (Array.isArray(evidence.records) ? evidence.records : []).filter(function (r) {
    return finite(r.score) && finite(r.gpa) && ["admitted", "rejected"].includes(r.outcome) && (!cycle || r.cycle === cycle);
  }).slice().sort(function (a, b) { return distance(a) - distance(b) || String(b.cycle).localeCompare(String(a.cycle)); });
}
function svgNode(tag, attributes, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes || {}).forEach(function (pair) { el.setAttribute(pair[0], String(pair[1])); });
  if (text !== undefined) el.textContent = String(text);
  return el;
}
function drawEvidenceChart(host, reports, input) {
  host.replaceChildren();
  const compact = evidenceViewport.matches;
  const width = compact ? 360 : 700, height = compact ? 290 : 325, left = compact ? 45 : 57, right = compact ? 15 : 23, top = 34, bottom = compact ? 45 : 53;
  const scores = reports.map(function (r) { return r.score; }).concat(input.score);
  const gpas = reports.map(function (r) { return r.gpa; }).concat(input.gpa);
  const sat = input.test_type === "SAT", lowLimit = sat ? 400 : -11.17, highLimit = sat ? 1600 : 50;
  const spread = Math.max(...scores) - Math.min(...scores);
  const step = sat ? (spread > 600 ? 200 : spread > 250 ? 100 : 50) : (spread > 25 ? 10 : spread > 12 ? 5 : 2);
  let xMin = Math.max(lowLimit, Math.floor((Math.min(...scores) - step * .6) / step) * step);
  let xMax = Math.min(highLimit, Math.ceil((Math.max(...scores) + step * .6) / step) * step);
  if (xMax <= xMin) xMin = xMax - step;
  const yMin = Math.max(6, Math.floor((Math.min(...gpas) - .2 + 1e-9) * 2) / 2);
  const yMax = Math.min(10, Math.ceil((Math.max(...gpas) + .2) * 2) / 2);
  const x = function (n) { return left + (n - xMin) / (xMax - xMin) * (width - left - right); };
  const y = function (n) { return top + (yMax - n) / (yMax - yMin) * (height - top - bottom); };
  const svg = svgNode("svg", {viewBox: "0 0 " + width + " " + height, role: "img", "aria-label": reports.length + " selected reports and your profile. Horizontal axis: " + (sat ? "SAT" : "Bocconi Test") + " score. Vertical axis: Bocconi GPA."});
  svg.append(svgNode("desc", {}, "Green circles: reported admitted. Red crosses: reported not admitted. Navy diamond: your profile. The same reports are listed in the table below."));
  const tickStart = Math.ceil(xMin / step) * step;
  for (let tick = tickStart; tick <= xMax + 1e-7; tick += step) {
    svg.append(svgNode("line", {x1:x(tick),x2:x(tick),y1:top,y2:height-bottom,class:"chart-grid"}));
    svg.append(svgNode("text", {x:x(tick),y:height-bottom+23,"text-anchor":"middle",class:"chart-tick"}, Number(tick.toFixed(2))));
  }
  const yStep = yMax - yMin > 3 ? 1 : .5;
  for (let tick = yMin; tick <= yMax + 1e-7; tick += yStep) {
    svg.append(svgNode("line", {x1:left,x2:width-right,y1:y(tick),y2:y(tick),class:"chart-grid"}));
    svg.append(svgNode("text", {x:left-12,y:y(tick)+4,"text-anchor":"end",class:"chart-tick"}, tick.toFixed(1)));
  }
  svg.append(svgNode("text", {x:(left+width-right)/2,y:height-8,"text-anchor":"middle",class:"chart-axis"}, sat ? "SAT score" : "Bocconi Test score"));
  svg.append(svgNode("text", {x:16,y:(top+height-bottom)/2,transform:"rotate(-90 16 " + ((top+height-bottom)/2) + ")","text-anchor":"middle",class:"chart-axis"}, "Bocconi GPA"));
  // All points use the returned scores. No synthetic jitter or invented reports.
  reports.slice().sort(function(a,b){return (a.outcome === "rejected") - (b.outcome === "rejected");}).forEach(function (r) {
    const px=x(r.score), py=y(r.gpa), admitted=r.outcome === "admitted";
    const group=svgNode("g", {class:"chart-report " + (admitted ? "chart-admitted" : "chart-rejected"),tabindex:"0",role:"img","aria-label":(admitted ? "Admitted" : "Not admitted") + ": " + r.score + ", GPA " + r.gpa + ", " + year(r.cycle)});
    group.append(svgNode("title", {}, (admitted ? "Admitted" : "Not admitted") + " · " + r.score + " · GPA " + r.gpa + " · " + year(r.cycle) + (r.nearby === false ? " · Outside the similar-score range" : "")));
    group.append(svgNode("circle", {cx:px,cy:py,r:10,class:"chart-hit"}));
    const overlapsOtherOutcome = reports.some(function(other){return other.score === r.score && other.gpa === r.gpa && other.outcome !== r.outcome;});
    if (admitted) group.append(svgNode("circle", {cx:px,cy:py,r:overlapsOtherOutcome ? 7 : 5}));
    else group.append(svgNode("path", {d:"M"+(px-4)+","+(py-4)+" L"+(px+4)+","+(py+4)+" M"+(px+4)+","+(py-4)+" L"+(px-4)+","+(py+4)}));
    svg.append(group);
  });
  const ux=x(input.score), uy=y(input.gpa);
  svg.append(svgNode("line", {x1:ux,x2:ux,y1:uy+10,y2:height-bottom,class:"chart-you-guide"}));
  svg.append(svgNode("path", {d:"M"+ux+","+(uy-9)+" L"+(ux+9)+","+uy+" L"+ux+","+(uy+9)+" L"+(ux-9)+","+uy+" Z",class:"chart-you"}));
  svg.append(svgNode("text", {x:ux,y:uy-17,"text-anchor":"middle",class:"chart-you-label"}, "You"));
  host.append(svg);
}
function renderEvidence(data, input) {
  if (typeof ChanceReport !== "undefined") { ChanceReport.render(data, input); return; }
  const body = $("evidence-body"); body.replaceChildren();
  const e = data.evidence || {};
  const profile = node("div", undefined, "context-profile");
  [[input.test_type === "SAT" ? "SAT score" : "Bocconi Test", format(input.score)], ["Bocconi GPA", format(input.gpa)], ["Programme", input.course], ["Round", input.session]].forEach(function (pair) {
    const item=node("div",undefined,"context-profile-item"); item.append(node("span",pair[0]),node("strong",pair[1]));profile.append(item);
  });
  const edit=node("button","Edit scores ↗","text-button context-edit"); edit.type="button";
  edit.addEventListener("click",function(){form.scrollIntoView({behavior:reduceMotion.matches?"instant":"smooth",block:"center"});$("score").focus({preventScroll:true});});
  profile.append(edit); body.append(profile);
  const toolbar=node("div",undefined,"context-toolbar");toolbar.append(node("h3","Past applicants"));
  const filter=node("label",undefined,"cycle-filter"); filter.append(node("span","Admission cycle"));
  const select=node("select"); select.id="evidence-cycle"; select.setAttribute("aria-label","Admission cycle");
  const all=node("option","All cycles");all.value="";select.append(all);
  const cycles=Array.from(new Set((e.by_year || []).map(function(r){return r.cycle;}).filter(Boolean))).sort().reverse();
  cycles.forEach(function(cycle){const option=node("option",year(cycle));option.value=cycle;select.append(option);});
  select.value="";filter.append(select);toolbar.append(filter);body.append(toolbar);
  const grid=node("div",undefined,"context-grid");
  const chartCard=node("section",undefined,"context-chart-card");chartCard.append(node("h4","Scores and outcomes"));
  const chart=node("div",undefined,"context-chart");chartCard.append(chart);
  const legend=node("div",undefined,"chart-legend");
  [["Admitted","admitted"],["Not admitted","rejected"],["Your profile","you"]].forEach(function(pair){const label=node("span");label.append(node("i",undefined,"legend-mark "+pair[1]),node("span",pair[0]));legend.append(label);});
  chartCard.append(legend);const caption=node("p",undefined,"chart-caption");chartCard.append(caption);
  const summary=node("section",undefined,"context-summary"); summary.setAttribute("aria-label","Similar profile counts");
  grid.append(chartCard,summary);body.append(grid);
  const reportsSection=node("section",undefined,"context-reports");
  const reportsHeading=node("div",undefined,"reports-heading");reportsHeading.append(node("h3","Individual reports"));const shown=node("span");reportsHeading.append(shown);reportsSection.append(reportsHeading);
  const tableHost=node("div");reportsSection.append(tableHost);
  const more=node("button",undefined,"text-button reports-more");more.type="button";reportsSection.append(more);body.append(reportsSection);
  const about=node("details",undefined,"context-about-data");about.append(node("summary","How these reports are selected"));
  const aboutBody=node("div");
  const tw=e.nearby && e.nearby.test_window, gw=e.nearby && e.nearby.gpa_window;
  aboutBody.append(node("p", finite(tw)&&finite(gw) ? "Similar profiles: same historical programme and round, within ±"+format(tw)+" "+(input.test_type === "SAT"?"SAT":"Bocconi Test")+" points and ±"+format(gw)+" GPA." : "Similarity counts are unavailable for this result."));
  aboutBody.append(node("p","The chart and table show selected reports from each cycle, including scores outside that range. The counts include all similar reports in the selected cycle."));
  if(e.proxy_notice) aboutBody.append(node("p",e.proxy_notice));
  if((e.records||[]).some(function(r){return r.used_for_bt_fit===false;})) aboutBody.append(node("p","Reports marked ‘Comparison only’ were not used to fit the Bocconi Test model."));
  const methodology=node("a","Methodology & data limitations ↗");methodology.href="#methodology";methodology.addEventListener("click",function(){$("methodology").open=true;});aboutBody.append(methodology);about.append(aboutBody);body.append(about);
  let selected="", expanded=false;
  function renderRows(reports){
    tableHost.replaceChildren();const limit=evidenceViewport.matches?4:6;const visible=expanded?reports:reports.slice(0,limit);
    shown.textContent=reports.length ? visible.length+" of "+reports.length+" selected reports" : "0 selected reports";
    if(visible.length) table(tableHost,[input.test_type === "SAT"?"SAT":"Bocconi Test","GPA","Cycle","Reported outcome","Match"],visible.map(function(r){
      const outcome=node("span",undefined,"outcome "+(r.outcome==="rejected"?"rejected":"admitted"));outcome.append(node("span",r.outcome==="admitted"?"✓":"×","outcome-symbol"),node("span",r.outcome==="admitted"?"Admitted":"Not admitted"));
      const match=node("span",r.nearby===true?"Similar":"Outside range","report-match"+(r.nearby===true?" is-similar":""));
      if(r.used_for_bt_fit===false) match.append(node("small","Comparison only","comparison-only"));
      return [format(r.score),format(r.gpa),year(r.cycle),outcome,match];
    }));else tableHost.append(node("p","No individual reports available for this cycle.","reports-empty"));
    more.hidden=reports.length<=limit;more.textContent=expanded?"Show fewer reports ↑":"Show all "+reports.length+" reports ↓";
  }
  function update(){
    const counts=evidenceCounts(e,selected),reports=evidenceRecords(e,input,selected);
    drawEvidenceChart(chart,reports,input);caption.textContent=reports.length+" selected reports plotted. Counts include all similar profiles.";
    summary.replaceChildren();summary.append(node("h4","Similar profiles"));
    if(counts && [counts.total,counts.admitted,counts.rejected].every(finite)){
      const number=node("p",undefined,"context-big-count");
      number.append(node("strong",counts.total ? counts.admitted+" of "+counts.total : "0"));summary.append(number,node("p",counts.total?"reported admission":"similar reports in this range","context-count-label"));
      const list=node("dl",undefined,"context-outcomes");
      [["Admitted",counts.admitted,"admitted"],["Not admitted",counts.rejected,"rejected"]].forEach(function(pair){const row=node("div");const term=node("dt");term.append(node("i",undefined,"legend-mark "+pair[2]),node("span",pair[0]));row.append(term,node("dd",pair[1]));list.append(row);});summary.append(list);
      if(!counts.total)summary.append(node("p","The reports shown are outside your score range.","context-summary-note"));
    }else summary.append(node("p","Counts unavailable for this cycle.","context-summary-note"));
    summary.append(node("p",counts && counts.total > 0 ? "Self-reported results. This share is not your admission probability." : "Self-reported results.","context-summary-note"));
    renderRows(reports);
  }
  select.addEventListener("change",function(){selected=select.value;expanded=false;update();});
  more.addEventListener("click",function(){expanded=!expanded;renderRows(evidenceRecords(e,input,selected));});
  redrawEvidenceChart=function(){const reports=evidenceRecords(e,input,selected);drawEvidenceChart(chart,reports,input);renderRows(reports);};
  update();$("evidence").hidden=false;
}
function renderResult(data, input, options = {}) {
  const panel = $("result-panel");
  if (data.status === "invalid" || data.status === "unsupported") {
    emptyState(data.status === "invalid" ? "Check your profile." : "No estimate for this profile.", data.status === "invalid" ? "Check the score, GPA, programme and section scores, then try again." : "This model supports international applicants in the 2027/28 cycle and requires a confirmed GPA source.", "unavailable");
    $("form-error").textContent = $("empty-description").textContent; $("form-error").hidden = false; announce($("empty-description").textContent);
    return false;
  }
  const formal = data.status === "ineligible";
  if (!finite(data.probability) || data.probability < 0 || data.probability > 1 || typeof data.display_percent !== "string" || !/^(?:<1|>99|100|\d{1,2})%$/.test(data.display_percent)) throw new Error("Invalid estimate response");
  panel.classList.remove("is-empty", "is-loading");
  panel.classList.toggle("is-formal", formal);
  panel.dataset.state = formal ? "ineligible" : "result";
  panel.dataset.tone = formal || data.probability < .35 ? "low" : data.probability < .65 ? "middle" : "high";
  panel.setAttribute("aria-busy", "false");
  $("empty-state").hidden = true; $("calculated-state").hidden = false; $("panel-independent").hidden = true;
  $("result-kicker").textContent = formal ? "Formal test requirements" : input.gpa_estimated ? "GPA simulation" : "Estimated chance";
  $("result-context").textContent = input.course + " · " + input.session;
  const percentage = $("percentage"); percentage.replaceChildren();
  const value = data.display_percent.slice(0, -1);
  if (formal) percentage.append(node("span", "Not eligible"));
  else {
    if (/^[<>]/.test(value)) percentage.append(node("span", value[0], "bound-sign"));
    percentage.append(node("span", value.replace(/^[<>]/, "")), node("span", "%", "percent-sign"));
  }
  $("ring-value").setAttribute("stroke-dasharray", String(data.probability * 100) + " 100");
  const copy = resultCopy(data.probability, input);
  $("result-headline").textContent = formal ? "Test requirements not met." : copy[0];
  $("result-explanation").textContent = formal ? "This score does not meet one or more test requirements for the selected programme. Check the total and section scores before interpreting an admission estimate." : copy[1];
  $("endpoint-note").hidden = formal || !data.endpoint_note;
  $("endpoint-note").textContent = "This is an endpoint of the model’s assumed scale, rather than a certain admission or rejection.";
  const near = data.evidence && data.evidence.nearby;
  $("result-nearby").hidden = !near || !finite(near.total);
  $("result-nearby").replaceChildren();
  if (near && finite(near.total)) {
    if (near.total > 0) $("result-nearby").append(node("strong", near.admitted + " of " + near.total), node("span", "similar reports: admitted"));
    else $("result-nearby").textContent = "No similar historical reports in this score window.";
  }
  const notes = [];
  if (input.gpa_estimated) notes.push("Based on unofficial GPA. Confirm your GPA with Bocconi.");
  if (near && near.total < 10) notes.push("Small comparison sample.");
  if (String(data.evidence && data.evidence.support).includes("ekstrapolacja")) notes.push("Outside the historical context range.");
  if (data.eligibility && data.eligibility.status === "unverified") notes.push("Section minimums have not been checked.");
  $("result-support").textContent = notes.join(" "); $("result-support").hidden = !notes.length;
  $("explore-link").hidden = formal;
  if (!formal && !options.skipEvidence) renderEvidence(data, input);
  else if (formal) {
    if (!options.skipEvidence) $("evidence").hidden = true;
    $("result-support").textContent = "Review Bocconi’s official requirements in the methodology below.";
    $("result-support").hidden = false;
  }
  animateResult(data.probability, formal);
  announce(formal ? "Test requirements not met. Review your total and section scores." : (input.gpa_estimated ? "GPA simulation: " : "Model estimate: ") + data.display_percent + ". " + copy[0]);
  return true;
}
async function calculate() {
  if (!ready) return;
  const input = readInput();
  if (!validate(input)) return;
  window.ChanceAnalytics?.track('calculator_submitted', {test_type:input.test_type,program:input.course,round:input.session});
  hasInteracted = true;
  if (activeRequest) activeRequest.abort();
  const controller = new AbortController(), current = ++generation;
  activeRequest = controller;
  emptyState("Checking your profile…", "Calculating your estimate and finding the historical context.", "loading");
  $("calculate").disabled = true; buttonLabel("Calculating…"); announce("Calculating your estimate.");
  const timeout = setTimeout(function () { controller.abort("timeout"); }, 20000);
  try {
    const response = await fetch("/api/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input), signal: controller.signal });
    const data = await response.json();
    if (current !== generation) return;
    if (!response.ok && !(response.status === 400 && data.status === "invalid")) throw new Error("Unavailable service");
    if (!data || !["estimated", "ineligible", "unsupported", "invalid"].includes(data.status)) throw new Error("Unexpected response");
    renderResult(data, input);
    window.ChanceAnalytics?.track('calculator_completed', {test_type:input.test_type,program:input.course,round:input.session,status:data.status,result_band:data.probability<.35?'lower':data.probability<.65?'balanced':'stronger'});
  } catch (exception) {
    if (current !== generation) return;
    window.ChanceAnalytics?.track('calculator_failed',{reason:'unavailable'});
    emptyState("The estimate is unavailable.", "Please try again in a moment. Your scores are still in the form.", "error");
    $("form-error").textContent = "We couldn’t get a result from the model. Please try again."; $("form-error").hidden = false;
    announce("The estimate is unavailable. Your scores have been kept.");
  } finally {
    clearTimeout(timeout);
    if (current === generation) { activeRequest = null; $("calculate").disabled = !ready; buttonLabel("Calculate my chances"); }
  }
}
function panelScroll() { $("result-panel").scrollIntoView({ behavior: reduceMotion.matches ? "instant" : "smooth", block: "start" }); }
form.addEventListener("submit", function (event) { event.preventDefault(); calculate(); });
form.addEventListener("input", function (event) { if (event.target.tagName === "INPUT" && !["radio", "checkbox"].includes(event.target.type)) invalidate(); });
form.addEventListener("change", function (event) {
  if (event.target.name === "test_type") updateTestType();
  if (event.target.id === "course") {
    $("selected-programme").textContent = $("course").value;
    $("selected-programme").hidden = $("course").value.length < 30;
  }
  invalidate();
});
$("explore-link").addEventListener("click", function (event) {
  event.preventDefault(); if (typeof ChanceReport !== "undefined") { ChanceReport.open({scroll:true}); return; } $("evidence").open = true;
  $("evidence").scrollIntoView({ behavior: reduceMotion.matches ? "instant" : "smooth", block: "start" });
  $("evidence").querySelector("summary").focus({ preventScroll: true });
});
const menu = $("menu-toggle"), nav = $("main-nav"), guidesToggle = $("guides-toggle"), guidesPanel = $("guides-panel");
function setGuidesOpen(open) {
  guidesPanel.hidden = !open;
  guidesToggle.setAttribute("aria-expanded", String(open));
}
function closeMenu() {
  nav.classList.remove("is-open"); menu.setAttribute("aria-expanded", "false"); menu.setAttribute("aria-label", "Open navigation");
  setGuidesOpen(false);
}
guidesToggle.addEventListener("click", function () { setGuidesOpen(guidesToggle.getAttribute("aria-expanded") !== "true"); });
guidesToggle.addEventListener("keydown", function (event) {
  if (event.key === "ArrowDown") { event.preventDefault(); setGuidesOpen(true); guidesPanel.querySelector("a").focus(); }
});
menu.addEventListener("click", function () {
  const open = menu.getAttribute("aria-expanded") !== "true";
  menu.setAttribute("aria-expanded", String(open)); menu.setAttribute("aria-label", open ? "Close navigation" : "Open navigation"); nav.classList.toggle("is-open", open);
});
nav.addEventListener("click", function (event) { if (event.target.closest("a")) closeMenu(); });
document.addEventListener("click", function (event) { if (!$("guides-dropdown").contains(event.target)) setGuidesOpen(false); });
document.addEventListener("keydown", function (event) {
  if (event.key !== "Escape") return;
  if (guidesToggle.getAttribute("aria-expanded") === "true") { setGuidesOpen(false); guidesToggle.focus(); }
  else if (menu.getAttribute("aria-expanded") === "true") { closeMenu(); menu.focus(); }
});
setGuidesOpen(false);
function isFullReportPage() { return Boolean(document.body && document.body.dataset.page === "report"); }
function persistProfileDraft() {
  if (isFullReportPage()) return;
  try {
    sessionStorage.setItem("chanceme.profile-draft.v1", JSON.stringify({
      version: 1, created_at: Date.now(), test_type: form.elements.test_type.value,
      score: $("score").value, gpa: $("gpa").value, course: $("course").value,
      session: $("session").value, gpa_estimated: gpaEstimated,
      typeScores: { ...typeScores, [form.elements.test_type.value]: $("score").value }
    }));
  } catch (_) { /* The calculators remain usable without session storage. */ }
}
function restoreProfileDraft(programs) {
  if (isFullReportPage()) return false;
  try {
    const raw = sessionStorage.getItem("chanceme.profile-draft.v1");
    if (!raw) return false;
    const value = JSON.parse(raw), age = Date.now() - value.created_at;
    if (value.version !== 1 || !finite(value.created_at) || age < 0 || age >= 1800000 || !["SAT", "Bocconi"].includes(value.test_type)) return false;
    const text = v => typeof v === "string" ? v.slice(0, 40) : "";
    form.elements.test_type.value = value.test_type; lastType = value.test_type;
    typeScores.SAT = text(value.typeScores && value.typeScores.SAT); typeScores.Bocconi = text(value.typeScores && value.typeScores.Bocconi);
    $("score").value = text(value.score); $("gpa").value = text(value.gpa);
    $("course").value = programs.some(p => p.id === value.course) ? value.course : "";
    $("session").value = ["Early", "Winter"].includes(value.session) ? value.session : "";
    gpaEstimated = value.gpa_estimated === true;
    if (gpaEstimated) { $("gpa-transfer-note").textContent = "Estimated GPA restored. Your result will be labelled as a GPA simulation."; $("gpa-transfer-note").hidden = false; }
    $("selected-programme").textContent = $("course").value; $("selected-programme").hidden = $("course").value.length < 30;
    updateTestType(); return true;
  } catch (_) { return false; }
}
function consumeGpaTransfer() {
  try {
    sessionStorage.removeItem("chanceme.gpa-transfer.v1");
    const raw = sessionStorage.getItem("chanceme.gpa-transfer.v2"); sessionStorage.removeItem("chanceme.gpa-transfer.v2");
    if (!raw) return;
    const value = JSON.parse(raw), years = value.years, age = Date.now() - value.created_at;
    if (value.version !== 2 || value.category !== "international" || !finite(value.created_at) || age < 0 || age >= 1800000 || !Array.isArray(years) || years.length !== 2 || !years.every(function (y) { return finite(y.gpa) && y.gpa >= 6 && y.gpa <= 10; }) || !finite(value.model_gpa) || Math.abs(value.model_gpa - (years[0].gpa + years[1].gpa) / 2) > 1e-7) return;
    const official = value.official === true && years.every(function (y) { return y.source === "official_portal"; });
    $("gpa").value = String(Number(value.model_gpa.toFixed(10)));
    gpaEstimated = !official;
    $("gpa-transfer-note").textContent = official ? "Your Bocconi GPA was transferred." : "Estimated GPA transferred. Your result will be labelled as a GPA simulation.";
    $("gpa-transfer-note").hidden = false;
    return true;
  } catch (_) { /* Storage may be unavailable; the form remains usable. */ }
}
async function initialize() {
  updateTestType();
  try {
    const response = await fetch("/api/options");
    if (!response.ok) throw new Error("Unavailable options");
    const options = await response.json();
    if (!Array.isArray(options.programs) || !options.programs.length || !options.programs.every(function (p) { return typeof p.id === "string" && typeof p.name === "string"; })) throw new Error("Invalid options");
    const placeholder = node("option", "Select"); placeholder.value = "";
    $("course").replaceChildren(placeholder, ...options.programs.map(function (p) { const option = node("option", p.name); option.value = p.id; return option; }));
    $("course").disabled = false;
    if (typeof ChanceReport !== "undefined") ChanceReport.setPrograms(options.programs);
    ready = true; $("calculate").disabled = false; buttonLabel("Calculate my chances");
    if (!isFullReportPage()) {
      restoreProfileDraft(options.programs);
      if (consumeGpaTransfer()) persistProfileDraft();
    }
  } catch (_) {
    $("course").replaceChildren(node("option", "Programmes unavailable"));
    $("calculate").disabled = true; buttonLabel("Calculator unavailable");
    emptyState("The calculator is unavailable.", "Please refresh the page in a moment. No estimate has been calculated.", "error");
    $("form-error").textContent = "The model configuration could not be loaded. Refresh to try again."; $("form-error").hidden = false;
    announce("Calculator unavailable.");
  }
}
const campusReady = initialize();
form.addEventListener("input", persistProfileDraft);
form.addEventListener("change", persistProfileDraft);
document.querySelectorAll('a[href="/bocconi-gpa-calculator/"]').forEach(link => link.addEventListener("click", persistProfileDraft));
if (typeof window.addEventListener === "function") {
  window.addEventListener("pagehide", persistProfileDraft);
  window.addEventListener("pageshow", event => {
    if (event.persisted && !isFullReportPage() && consumeGpaTransfer()) { invalidate(); persistProfileDraft(); }
  });
}

// Calculator links return from the report without discarding the entered scores.
document.querySelectorAll('a[href="#calculator"]').forEach(function(link){link.addEventListener("click",function(){if(typeof ChanceReport!=="undefined")ChanceReport.close();});});
