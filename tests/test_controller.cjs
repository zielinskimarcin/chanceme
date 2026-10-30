// Test the actual controller with a minimal DOM double; no browser automation.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.value = ""; this.checked = false; this.hidden = false;
    this.children = []; this.attributes = {}; this.dataset = {}; this.listeners = {}; this.open = false;
    this.classes = new Set();
    this.classList = { add: (...names) => names.forEach(n => this.classes.add(n)), remove: (...names) => names.forEach(n => this.classes.delete(n)), toggle: (n, value) => value ? this.classes.add(n) : this.classes.delete(n) };
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text || "") + this.children.map(c => c.textContent).join(""); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.text = ""; this.children = nodes; }
  get lastChild() { return this.children[this.children.length - 1]; }
  addEventListener(type, handler) { const before = this.listeners[type]; this.listeners[type] = before ? (...args) => { before(...args); handler(...args); } : handler; }
  animate(frames, options) {
    const animation = { frames, options, cancelled: false, cancel() { this.cancelled = true; } };
    (this.animations ||= []).push(animation); return animation;
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  removeAttribute(key) { delete this.attributes[key]; }
  focus() { this.focused = true; }
  scrollIntoView() { this.scrolled = (this.scrolled || 0) + 1; }
  querySelector() { return new Element("summary"); }
}
const tick = () => new Promise(resolve => setImmediate(resolve));
function estimate(percent = "59%", p = .5935634541610315) {
  return { status: "estimated", probability: p, display_percent: percent, eligibility: { status: "score_minimums_met" }, evidence: { nearby: { total: 16, admitted: 7, rejected: 9, test_window: 50, gpa_window: .5 }, records: [], by_year: [] } };
}
async function setup(options = {}) {
  const html = fs.readFileSync(path.join(__dirname, "../dist/index.html"), "utf8");
  const elements = {};
  for (const match of html.matchAll(/<([a-z]+)[^>]*\bid="([^"]+)"[^>]*>/g)) elements[match[2]] = new Element(match[1]);
  elements["profile-form"].elements = { test_type: { value: "SAT" } };
  const storage = new Map(Object.entries(options.storage || {}));
  const calls = [];
  const scope = {
    console, Node: Element, AbortController, setTimeout, clearTimeout, URL,
    window: { matchMedia: query => ({ matches: Boolean((options.reducedMotion && query.includes("prefers-reduced-motion")) || (options.compact && query.includes("max-width:"))) }) },
    sessionStorage: { setItem: (k,v) => storage.set(k,v), getItem: k => storage.get(k) || null, removeItem: k => storage.delete(k) },
    document: { getElementById: id => elements[id], createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag), querySelectorAll: () => [], addEventListener() {} },
    fetch: async (url, init) => {
      calls.push({ url, init });
      if (url === "/api/options") {
        if (options.failOptions) throw new Error("offline");
        return { ok: true, json: async () => ({ programs: [{ id: "Management", name: "Management" }] }) };
      }
      return options.fetch ? options.fetch(url, init) : { ok: true, json: async () => estimate() };
    }
  };
  const context = vm.createContext(scope);
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../dist/campus.js"), "utf8"), context);
  await tick();
  const api = vm.runInContext("({calculate, invalidate, readInput, validate, consumeGpaTransfer, evidenceCounts, evidenceRecords, renderResult, persistProfileDraft, restoreProfileDraft})", context);
  function valid() {
    elements.score.value = "1450"; elements.gpa.value = "9"; elements.course.value = "Management";
    elements.session.value = "Winter";
  }
  return { elements, calls, api, valid, context, storage };
}
test("empty first load requests options only; no fabricated prediction", async () => {
  const x = await setup();
  assert.deepEqual(x.calls.map(c => c.url), ["/api/options"]);
  assert.equal(x.elements.score.value, "");
  assert.equal(x.elements.gpa.value, "");
  assert.equal(x.elements["gpa-confirmed"], undefined);
  assert.equal(x.elements.calculate.disabled, false);
});
test("blank form cannot send a prediction and focuses the first error", async () => {
  const x = await setup(); await x.api.calculate();
  assert.equal(x.calls.length, 1);
  assert.equal(x.elements.score.focused, true);
  assert.match(x.elements["gpa-error"].textContent, /between 6 and 10/);
});
test("real response drives percentage and exact ring; editing removes both result and evidence", async () => {
  const x = await setup(); x.valid(); await x.api.calculate();
  assert.equal(x.elements.percentage.textContent, "59%");
  assert.equal(x.elements["ring-value"].getAttribute("stroke-dasharray"), "59.35634541610315 100");
  assert.equal(x.elements.evidence.hidden, false);
  x.api.invalidate();
  assert.equal(x.elements.percentage.textContent, "");
  assert.equal(x.elements.evidence.hidden, true);
  assert.equal(x.elements["result-panel"].dataset.state, "stale");
});
test("a slow old response cannot replace a newer request even if fetch ignores abort", async () => {
  const pending = [];
  const x = await setup({ fetch: () => new Promise(resolve => pending.push(resolve)) });
  x.valid(); const old = x.api.calculate();
  x.elements.score.value = "1510"; x.api.invalidate();
  const newer = x.api.calculate();
  pending[1]({ ok: true, json: async () => estimate("67%", .67) }); await newer;
  pending[0]({ ok: true, json: async () => estimate("59%", .59) }); await old;
  assert.equal(x.elements.percentage.textContent, "67%");
  assert.equal(x.elements["result-panel"].dataset.state, "result");
});
test("unavailable service clears the previous result while preserving form input", async () => {
  let fail = false;
  const x = await setup({ fetch: async () => { if (fail) throw new Error("offline"); return { ok: true, json: async () => estimate() }; } });
  x.valid(); await x.api.calculate(); fail = true; await x.api.calculate();
  assert.equal(x.elements.percentage.textContent, "");
  assert.equal(x.elements["result-panel"].dataset.state, "error");
  assert.equal(x.elements.evidence.hidden, true);
  assert.equal(x.elements.score.value, "1450");
  assert.equal(x.elements.calculate.disabled, false);
});
test("valid form calculates without confirmation or optional section fields", async () => {
  const x = await setup(); x.valid(); await x.api.calculate();
  const payload = JSON.parse(x.calls.at(-1).init.body);
  assert.equal(payload.gpa_confirmed, true);
  assert.equal(payload.gpa_estimated, false);
  for (const id of ["optional-inputs", "gpa-source", "gpa-confirmed", "sat-reading", "sat-math", "bocconi-math"]) assert.equal(x.elements[id], undefined);
  for (const key of ["sat_reading", "sat_math", "bocconi_math"]) assert.equal(Object.hasOwn(payload, key), false);
  assert.equal(x.elements["result-kicker"].textContent, "Estimated chance");
});
test("mixed official/estimated transfer remains a simulation without extra confirmation", async () => {
  const transfer = { version: 2, category: "international", created_at: Date.now(), years: [{gpa: 8.8, source: "official_portal"}, {gpa: 9.6, source: "estimated"}], model_gpa: 9.2, official: true };
  const x = await setup({ storage: { "chanceme.gpa-transfer.v2": JSON.stringify(transfer) } });
  assert.equal(x.elements.gpa.value, "9.2");
  assert.equal(x.api.readInput().gpa_estimated, true);
  assert.equal(x.elements["gpa-confirmed"], undefined);
  assert.equal(x.calls.length, 1);
  x.valid(); await x.api.calculate();
  assert.equal(JSON.parse(x.calls.at(-1).init.body).gpa_estimated, true);
  assert.equal(x.elements["result-kicker"].textContent, "GPA simulation");
});
test("both test types submit totals without removed section inputs", async () => {
  const x = await setup(); x.valid();
  x.elements["profile-form"].elements.test_type.value = "Bocconi"; x.elements.score.value = "40";
  await x.api.calculate();
  const payload = JSON.parse(x.calls.at(-1).init.body);
  assert.equal(payload.test_type, "Bocconi"); assert.equal(payload.score, 40);
  assert.equal(Object.hasOwn(payload, "bocconi_math"), false);
});
test("formal minimum failure is labelled separately and never shown as model probability", async () => {
  const x = await setup({ fetch: async () => ({ ok: true, json: async () => ({status: "ineligible", probability: 0, display_percent: "0%"}) }) });
  x.valid(); await x.api.calculate();
  assert.equal(x.elements["result-kicker"].textContent, "Formal test requirements");
  assert.equal(x.elements.percentage.textContent, "Not eligible");
  assert.equal(x.elements.evidence.hidden, true);
});
test("failed configuration disables submit and exposes a usable error", async () => {
  const x = await setup({failOptions: true});
  assert.equal(x.elements.calculate.disabled, true);
  assert.equal(x.elements["result-panel"].dataset.state, "error");
});

test("ring animates to the exact probability and editing cancels the reveal", async () => {
  const x = await setup(); x.valid(); await x.api.calculate();
  const ring = x.elements["ring-value"].animations.at(-1);
  assert.equal(ring.frames[0].strokeDasharray, "0 100");
  assert.equal(ring.frames[1].strokeDasharray, "59.35634541610315 100");
  assert.equal(ring.options.duration, 1100);
  x.api.invalidate(); assert.equal(ring.cancelled, true);
  assert.equal(x.elements["calculated-state"].animations.at(-1).cancelled, true);
});
test("reduced motion shows the final accurate result without reveal animation", async () => {
  const x = await setup({reducedMotion: true}); x.valid(); await x.api.calculate();
  assert.equal(x.elements.percentage.textContent, "59%");
  assert.equal(x.elements["ring-value"].animations, undefined);
  assert.equal(x.elements["calculated-state"].animations, undefined);
});

function descendants(element, predicate) {
  return [element, ...element.children.flatMap(child => descendants(child, predicate))].filter(predicate);
}
function evidenceFixture() {
  return {
    nearby: {total:16, admitted:7, rejected:9, test_window:50, gpa_window:.5},
    by_year: [
      {cycle:"25-26",total:13,admitted:10,rejected:3,nearby_total:4,nearby_admitted:3,nearby_rejected:1},
      {cycle:"26-27",total:31,admitted:16,rejected:15,nearby_total:11,nearby_admitted:3,nearby_rejected:8}
    ],
    records: Array.from({length:8},(_,i)=>({id:"r"+i,score:1430+i*10,gpa:8.8+i*.05,cycle:i<4?"25-26":"26-27",outcome:i%2?"admitted":"rejected",nearby:i!==7}))
  };
}
test("cycle filter uses nearby counts rather than all applicants in that year", async()=>{
  const x=await setup(); const e=evidenceFixture();
  assert.equal(x.api.evidenceCounts(e,"26-27").total,11);
  assert.equal(x.api.evidenceCounts(e,"26-27").admitted,3);
  assert.equal(x.api.evidenceCounts(e,"").total,16);
  assert.equal(x.api.evidenceCounts(e,"unknown"),null);
  const before=JSON.stringify(e.records);
  const filtered=x.api.evidenceRecords(e,{test_type:"SAT",score:1450,gpa:9},"26-27");
  assert.equal(filtered.length,4); assert.ok(filtered.every(r=>r.cycle==="26-27"));
  assert.equal(JSON.stringify(e.records),before);
});
test("chart, counts and table respond together without changing the model estimate", async()=>{
  const data=estimate(); data.evidence=evidenceFixture();
  const x=await setup({fetch:async()=>({ok:true,json:async()=>data})});x.valid();await x.api.calculate();
  const body=x.elements["evidence-body"];
  const more=descendants(body,e=>e.tagName==="BUTTON"&&e.textContent.includes("Show all"))[0];
  assert.equal(descendants(body,e=>e.tagName==="TBODY")[0].children.length,6);
  more.listeners.click();
  assert.equal(descendants(body,e=>e.tagName==="TBODY")[0].children.length,8);
  const select=descendants(body,e=>e.tagName==="SELECT")[0];select.value="26-27";select.listeners.change();
  assert.equal(descendants(body,e=>e.tagName==="TBODY")[0].children.length,4);
  assert.match(body.textContent,/3 of 11/);
  assert.equal(descendants(body,e=>e.tagName==="SVG")[0].getAttribute("aria-label").startsWith("4 selected reports"),true);
  assert.equal(x.calls.length,2);assert.equal(x.elements.percentage.textContent,"59%");
  assert.equal(more.hidden,true);
  x.api.invalidate();assert.equal(body.children.length,0);assert.equal(x.elements.evidence.hidden,true);
});
test("zero close matches still show real outside-range records, without inventing a rate", async()=>{
  const data=estimate("18%",.18);data.evidence=evidenceFixture();data.evidence.nearby={total:0,admitted:0,rejected:0,test_window:4,gpa_window:.5};
  data.evidence.records.forEach(r=>{r.nearby=false;r.used_for_bt_fit=false;});
  const x=await setup({fetch:async()=>({ok:true,json:async()=>data})});x.valid();await x.api.calculate();
  const text=x.elements["evidence-body"].textContent;
  assert.match(text,/similar reports in this range/);assert.match(text,/outside your score range/);
  assert.doesNotMatch(text,/0 of 0/);assert.match(text,/Comparison only/);
  assert.equal(descendants(x.elements["evidence-body"],e=>e.tagName==="G").length,8);
});

test("compact evidence uses a readable chart and four initial report cards", async()=>{
  const data=estimate();data.evidence=evidenceFixture();
  const x=await setup({compact:true,fetch:async()=>({ok:true,json:async()=>data})});x.valid();await x.api.calculate();
  const body=x.elements["evidence-body"];
  assert.equal(descendants(body,e=>e.tagName==="TBODY")[0].children.length,4);
  assert.equal(descendants(body,e=>e.tagName==="SVG")[0].getAttribute("viewBox"),"0 0 360 290");
  assert.equal(descendants(body,e=>e.tagName==="TD")[0].getAttribute("data-label"),"SAT");
});

test("Calculate never scrolls the viewport or focuses the result on mobile", async () => {
  const x = await setup({compact:true}); x.valid(); await x.api.calculate();
  assert.equal(x.elements['result-panel'].scrolled, undefined);
  assert.equal(x.elements.evidence.scrolled, undefined);
  assert.equal(x.elements.percentage.focused, undefined);
});
test("filtered estimates update the hero without replacing inline comparison content", async () => {
  const x = await setup(); x.valid(); await x.api.calculate();
  const content = x.elements['evidence-body'].children;
  x.api.renderResult(estimate('54%', .54), {...x.api.readInput(), course:'Finance', session:'Early'}, {skipEvidence:true});
  assert.equal(x.elements.percentage.textContent, '54%');
  assert.equal(x.elements['result-context'].textContent, 'Finance · Early');
  assert.equal(x.elements['evidence-body'].children, content);
  assert.equal(x.elements.evidence.hidden, false);
});

test("switching calculators restores SAT/BT, GPA, programme and round without an automatic estimate",async()=>{
  const draft={version:1,created_at:Date.now(),test_type:'Bocconi',score:'32,5',gpa:'9.2',course:'Management',session:'Early',gpa_estimated:true,typeScores:{SAT:'1510',Bocconi:'32,5'}};
  const x=await setup({storage:{'chanceme.profile-draft.v1':JSON.stringify(draft)}});
  assert.equal(x.api.readInput().test_type,'Bocconi');assert.equal(x.api.readInput().score,32.5);assert.equal(x.api.readInput().gpa,9.2);assert.equal(x.api.readInput().course,'Management');assert.equal(x.api.readInput().session,'Early');assert.equal(x.api.readInput().gpa_estimated,true);assert.equal(x.calls.length,1);
  x.api.persistProfileDraft();const saved=JSON.parse(x.storage.get('chanceme.profile-draft.v1'));assert.equal(saved.typeScores.SAT,'1510');assert.equal(saved.score,'32,5');
});
test("fresh GPA transfer overrides only GPA after restoring the admissions draft",async()=>{
  const draft={version:1,created_at:Date.now(),test_type:'SAT',score:'1510',gpa:'8',course:'Management',session:'Winter',gpa_estimated:false};
  const transfer={version:2,created_at:Date.now(),category:'international',official:false,years:[{gpa:8.64,source:'linear_estimate'},{gpa:9.2,source:'linear_estimate'}],model_gpa:8.92};
  const x=await setup({storage:{'chanceme.profile-draft.v1':JSON.stringify(draft),'chanceme.gpa-transfer.v2':JSON.stringify(transfer)}});
  assert.equal(x.api.readInput().score,1510);assert.equal(x.api.readInput().gpa,8.92);assert.equal(x.api.readInput().course,'Management');assert.equal(x.api.readInput().session,'Winter');assert.equal(x.api.readInput().gpa_estimated,true);assert.equal(x.calls.length,1);assert.equal(x.storage.has('chanceme.gpa-transfer.v2'),false);
});
test("expired or malformed drafts never prefill a new profile",async()=>{
  for(const raw of ['{oops',JSON.stringify({version:1,created_at:Date.now()-1800001,test_type:'SAT',score:'1600'})]){
    const x=await setup({storage:{'chanceme.profile-draft.v1':raw}});assert.equal(x.elements.score.value,'');assert.equal(x.calls.length,1);
  }
});

test("typing still invalidates a result while saving the cross-calculator draft",async()=>{
  const x=await setup();x.valid();await x.api.calculate();x.elements.score.value='1500';x.elements['profile-form'].listeners.input({target:x.elements.score});
  assert.equal(x.elements.percentage.textContent,'');assert.equal(JSON.parse(x.storage.get('chanceme.profile-draft.v1')).score,'1500');
});

test('only explicit successful calculation emits submitted values and coarse bands',async()=>{
 const x=await setup(),events=[],helpers=require('../src/static/analytics.js');
 x.context.window.ChanceAnalytics={scoreBand:helpers.scoreBand,gpaBand:helpers.gpaBand,track:(...args)=>events.push(args)};
 x.valid();await x.api.calculate();assert.equal(events.length,1);assert.equal(events[0][0],'calculator_completed');
 assert.equal(JSON.stringify(events[0][1]),JSON.stringify({test_type:'SAT',program:'Management',round:'Winter',status:'estimated',score:1450,gpa:9,probability:.5935634541610315*100,result_band:'balanced',score_band:'1450-1499',gpa_band:'9.00-9.24'}));
 x.api.renderResult(estimate(),x.api.readInput(),{skipEvidence:true});assert.equal(events.length,1);
});
test('failed explicit calculation emits only fixed failure reason and preserves form',async()=>{
 const x=await setup({fetch:async()=>{throw new Error('offline');}}),events=[];
 x.context.window.ChanceAnalytics={track:(...args)=>events.push(args)};x.valid();await x.api.calculate();
 assert.equal(JSON.stringify(events),JSON.stringify([['calculator_failed',{reason:'unavailable'}]]));assert.equal(x.elements.score.value,'1450');
});
