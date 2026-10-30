/** Owner-only local report: never served by public/ or an unauthenticated API. */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { summarize, freshState } from '../lib/product-metrics.mjs';
import { redisStore, configured } from '../lib/product-metrics-store.mjs';

const names = {
 calculator_completed:'Calculations completed',calculator_failed:'Calculation errors',
 gpa_calculated:'GPA calculations',gpa_transferred:'GPA transfers',
 report_rendered:'Full reports displayed',report_failed:'Report errors',
 pdf_generated:'PDFs generated',pdf_failed:'PDF generation errors',pdf_download_initiated:'PDF downloads initiated',
 cta_clicked:'Navigation / contact clicks',booking_confirmed:'Booking confirmations observed',
};
const escape = value => String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function renderReport(summary,{demo=false,records=[],now=new Date()}={}) {
 const cards=Object.entries(names).map(([event,label])=>`<article><span>${escape(label)}</span><strong>${summary.totals[event]||0}</strong></article>`).join('');
 const histograms=Object.entries(summary.distributions).map(([dimension,bins])=>`<section class="histogram"><h2>${escape(dimension.replaceAll('_',' ').replace(':', ' · '))}</h2>${Object.entries(bins).sort((a,b)=>String(a[0]).localeCompare(String(b[0]),undefined,{numeric:true})).map(([label,count])=>`<div class="bar"><span>${escape(label)}</span><meter min="0" max="${Math.max(1,...Object.values(bins).filter(Number.isFinite))}" value="${count||0}"></meter><b>${count===null?'Hidden':count}</b></div>`).join('')}</section>`).join('');
 const rows=summary.days.map(day=>`<tr><td>${escape(day.date)}</td><td>${day.events.calculator_completed||0}</td><td>${day.events.report_rendered||0}</td><td>${day.events.pdf_download_initiated||0}</td><td>${day.events.cta_clicked||0}</td><td>${day.limited?'Limit reached':'Within ingestion limit'}</td></tr>`).join('');
 return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Chance Me · Private usage report</title><style>
 *{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f3f7f6;color:#17332e;margin:0}main{max-width:1150px;margin:40px auto;padding:0 24px}header{padding-bottom:20px;border-bottom:1px solid #cfded9}h1{font-size:34px;margin:8px 0}h2{font-size:17px}p{max-width:850px;color:#47635a}.badge{display:inline-block;padding:5px 10px;border-radius:20px;background:#dcebe4;font-size:13px;font-weight:700}.warning{padding:14px;background:#fff0ce;color:#58441d;border-radius:10px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}article,.histogram{background:white;border:1px solid #dce6e1;border-radius:14px;padding:18px}article span{display:block;font-size:14px}strong{display:block;font-size:32px}.histograms{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.bar{display:grid;grid-template-columns:minmax(100px,1fr) 110px 50px;align-items:center;gap:12px;margin:10px 0;font-size:14px}.bar span{overflow-wrap:anywhere}meter{width:100%;height:15px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #e1e9e5}.table{overflow:auto;margin:28px 0}footer{margin:30px 0;font-size:13px;color:#47635a}@media(max-width:500px){main{padding:0 14px}.histograms{grid-template-columns:1fr}.bar{grid-template-columns:1fr 80px 42px}}
 </style></head><body><main><header><span class="badge">${demo?'DEMO · SYNTHETIC DATA':'PRIVATE · OWNER REPORT'}</span><h1>How people use Chance Me</h1><p>${escape(summary.from)} to ${escape(summary.to)} · UTC days · Action counts, not unique people. ${summary.days.length} day(s) with recorded observations.</p></header>
 ${!summary.days.length?'<p class="warning">No observations are available. Collection may not be activated, or events may not have been received. This does not show zero website activity.</p>':''}
 <p>Only observed events are shown. A zero means no events were recorded, not that no activity occurred. Missing days can predate activation or reflect unobserved activity; collection health is unknown.</p>
 ${demo?'<p class="warning">This is a layout preview with invented test counts. It contains no real visitor data.</p>':''}
 ${summary.limited?'<p class="warning">An ingestion limit was reached. History or counts are incomplete. Detailed history is capped at 1,000 actions per UTC day; batch ingestion is capped separately.</p>':''}
 ${historyView(records,now)}<div class="cards">${cards}</div><p>Events can be blocked or lost. Repeat calculations count separately. A displayed report is not proof it was read; a download initiation is not proof the PDF was saved. Booking confirmations only reflect callbacks observed by the page.</p>
 <div class="table"><table><thead><tr><th>UTC date</th><th>Calculations</th><th>Reports</th><th>PDF downloads initiated</th><th>Clicks</th><th>Collection</th></tr></thead><tbody>${rows||'<tr><td colspan="6">No recorded actions in this period.</td></tr>'}</tbody></table></div>
 <div class="histograms">${histograms}</div><footer>Distributions are separate totals, not linked applicant profiles. Bins with fewer than five observations are hidden; this threshold alone is not an anonymity guarantee. Detailed event records above show observed visit activity where available; older aggregates cannot reconstruct past visits. Retention: daily aggregates expire after at most 90 days; transient random event hashes expire within 48 hours. No historical backfill.</footer></main></body></html>`;
}
function historyView(records,now) {
 // Escape script delimiters even though records are rendered with textContent below.
 const json=JSON.stringify(Array.isArray(records)?records:[]).replace(/[<>&\u2028\u2029]/g,c=>'\\u'+c.charCodeAt(0).toString(16).padStart(4,'0'));
 return `<section aria-labelledby="history-title"><h2 id="history-title">Visit and calculation history</h2><p>Private snapshot generated ${escape(new Date(now).toISOString())}. Times below use your browser's local timezone. Refreshing this file does not fetch new data; regenerate the private report to update it. Location is an approximate network location, not an exact address.</p>
 <style>.filters{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}.filters label{display:grid;font-size:13px;gap:4px}.filters input,.filters select{padding:9px;border:1px solid #bccfc5;border-radius:7px;background:white;color:#17332e}.history-table{min-width:1300px}.history-table td{max-width:260px;overflow-wrap:anywhere;vertical-align:top}.history-table small{display:block;color:#47635a}details.visit{background:white;border:1px solid #dce6e1;border-radius:10px;margin:10px 0;padding:12px}summary{cursor:pointer;font-weight:600}.timeline{padding-left:22px}.timeline li{margin:10px 0}.timeline p{margin:3px 0;font-size:14px;white-space:pre-wrap}.visit-note{font-size:13px}</style>
 <div class="filters"><label>From (local date)<input id="history-from" type="date"></label><label>Through (local date)<input id="history-to" type="date"></label><label>Country<select id="history-country"><option value="">All countries</option></select></label><label>Action<select id="history-event"><option value="">All actions</option></select></label><label>Sort<select id="history-sort"><option value="desc">Newest first</option><option value="asc">Oldest first</option></select></label><label>Visit ID<input id="history-visit" placeholder="Filter visit ID"></label></div>
 <p id="history-count" role="status"></p><div class="table"><table class="history-table"><thead><tr><th>Time / action</th><th>Visit</th><th>Country / region / city</th><th>Exact calculation</th><th>Page / source</th><th>Device</th></tr></thead><tbody id="history-rows"></tbody></table></div><h2>Visit timelines</h2><p class="visit-note">A visit ID links observed activity within a visit; it does not identify a person. Blocked or missing events may leave gaps. Historical visits from before this collection began are unavailable.</p><div id="history-visits"></div>
 <script type="application/json" id="history-data">${json}</script><script>
 (()=>{'use strict';const records=JSON.parse(document.getElementById('history-data').textContent);const byId=id=>document.getElementById(id);const filterNames=['from','to','country','event','sort','visit'];const filters=filterNames.map(name=>byId('history-'+name));
 const text=value=>value===undefined||value===null||value===''?'—':String(value);const details=r=>r.details||{};const date=r=>new Date(details(r).occurred_at||r.received_at);const validDate=r=>Number.isFinite(date(r).getTime());const localDay=r=>{const d=date(r);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0')};
 const node=(tag,value,className)=>{const n=document.createElement(tag);if(value!==undefined)n.textContent=text(value);if(className)n.className=className;return n};
 const label=event=>String(event||'unknown').replaceAll('_',' ');const fullTime=r=>validDate(r)?date(r).toLocaleString():text(r.received_at);const location=r=>[r.country,r.region,r.city].filter(Boolean).join(' / ')||'Unknown';
 const profile=r=>{const d=details(r),a=r.dimensions||{};return [['Test',a.test_type],['Score',d.score],['GPA',d.gpa],['Program',a.program],['Round',a.round],['Probability',d.probability],['Status',a.status],['Destination',a.destination]].filter(x=>x[1]!==undefined&&x[1]!==null).map(x=>x[0]+': '+x[1]).join(' · ')||'—'};
 const attribution=r=>{const d=details(r);return [['Page',d.page],['Referrer',d.referrer],['Source',d.source],['Medium',d.medium],['Campaign',d.campaign]].filter(x=>x[1]).map(x=>x[0]+': '+x[1]).join('\\n')||'—'};
 for(const [field,key]of [['country','country'],['event','event']])for(const value of [...new Set(records.map(r=>r[key]).filter(Boolean))].sort()){const o=node('option',value);o.value=value;byId('history-'+field).append(o)}
 const hashAvailable=typeof globalThis.location!=='undefined'&&typeof URLSearchParams!=='undefined';
 if(hashAvailable){const saved=new URLSearchParams(globalThis.location.hash.slice(1));for(let i=0;i<filterNames.length;i++){const value=saved.get(filterNames[i]);if(value!==null)filters[i].value=value}}
 function saveFilters(){if(!hashAvailable||typeof history==='undefined')return;const saved=new URLSearchParams();for(let i=0;i<filterNames.length;i++)if(filters[i].value)saved.set(filterNames[i],filters[i].value);history.replaceState(null,'','#'+saved.toString())}
 function render(){const [from,to,country,event,sort,visit]=filters.map(x=>x.value);const selected=records.filter(r=>(!from||validDate(r)&&localDay(r)>=from)&&(!to||validDate(r)&&localDay(r)<=to)&&(!country||r.country===country)&&(!event||r.event===event)&&(!visit||String(details(r).visit_id||'').includes(visit))).sort((a,b)=>(sort==='asc'?1:-1)*((date(a).getTime()||0)-(date(b).getTime()||0)));
 byId('history-count').textContent=selected.length+' of '+records.length+' recorded actions match the filters.';const rows=byId('history-rows');rows.replaceChildren();
 if(!selected.length){const tr=node('tr'),td=node('td','No detailed records match. Detailed history starts only after its release; old aggregate counts have no visit-level backfill.');td.colSpan=6;tr.append(td);rows.append(tr)}
 for(const r of selected){const tr=node('tr'),first=node('td',fullTime(r));first.append(node('small',label(r.event)));tr.append(first,node('td',details(r).visit_id),node('td',location(r)),node('td',profile(r)),node('td',attribution(r)),node('td',details(r).device));rows.append(tr)}
 const groups=new Map();for(const r of selected){const id=details(r).visit_id||'Unlinked event '+(r.dedup||records.indexOf(r));if(!groups.has(id))groups.set(id,[]);groups.get(id).push(r)}const visits=byId('history-visits');visits.replaceChildren();
 for(const [id,events]of groups){const box=node('details',undefined,'visit');const title=node('summary',id+' · '+events.length+' action(s) · '+location(events[0]));box.append(title);const list=node('ol',undefined,'timeline');for(const r of [...events].sort((a,b)=>date(a)-date(b))){const li=node('li');li.append(node('b',fullTime(r)+' · '+label(r.event)),node('p',profile(r)),node('p',attribution(r)),node('p',location(r)+' · '+text(details(r).device)));list.append(li)}box.append(list);visits.append(box)}
 }for(const field of filters)field.addEventListener('input',()=>{saveFilters();render()});render();})();
 </script></section>`;
}
function demoState(now) {
 const state=freshState(),date=now.toISOString().slice(0,10);
 state.days[date]={batches:18,events:{calculator_completed:42,gpa_calculated:18,gpa_transferred:12,report_rendered:14,pdf_generated:8,pdf_download_initiated:8,cta_clicked:11,calculator_failed:2},distributions:{
 'calculator_completed:test_type':{SAT:31,Bocconi:11},'calculator_completed:score_band':{'SAT:1350-1399':6,'SAT:1400-1449':10,'SAT:1450-1499':9,'SAT:1500-1549':6,'Bocconi:32-33.99':6,'Bocconi:34-35.99':5},
 'calculator_completed:gpa_band':{'8.50-8.74':8,'8.75-8.99':11,'9.00-9.24':15,'9.25-9.49':8},'calculator_completed:program':{Management:25,Finance:12,'Management and Computer Science':5},'calculator_completed:round':{Winter:30,Early:12},'cta_clicked:destination':{lessons:6,whatsapp:5}}};
 return state;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
 try {
  const args=process.argv.slice(2),demo=args.includes('--demo'),days=Number(args[args.indexOf('--days')+1])||7;
  const index=args.indexOf('--out'),out=path.resolve(index>=0?args[index+1]:'qa/product-metrics-report.html');
  const root=path.resolve(fileURLToPath(new URL('..',import.meta.url)));
  if(out===path.join(root,'public')||out.startsWith(path.join(root,'public')+path.sep))throw new Error('Report must remain outside public/');
  if(!demo&&!configured())throw new Error('Storage is not configured. Use the approved project-scoped environment; never paste credentials into a command.');
  const now=new Date(),store=demo?null:redisStore();
  const [state,records]=demo?[demoState(now),[]]:await Promise.all([store.history(days,now),store.detailedHistory(days,now)]);
  fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,renderReport(summarize(state,{days,now}),{demo,records,now}),{mode:0o600});fs.chmodSync(out,0o600);console.log(`${demo?'Demo':'Private report'} saved: ${out}`);
 } catch(error) {console.error(error.message.startsWith('Storage is not configured')?error.message:'Report unavailable. Check approved storage configuration and connectivity.');process.exitCode=1;}
}
