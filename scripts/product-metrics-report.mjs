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
export function renderReport(summary,{demo=false}={}) {
 const cards=Object.entries(names).map(([event,label])=>`<article><span>${escape(label)}</span><strong>${summary.totals[event]||0}</strong></article>`).join('');
 const histograms=Object.entries(summary.distributions).map(([dimension,bins])=>`<section class="histogram"><h2>${escape(dimension.replaceAll('_',' ').replace(':', ' · '))}</h2>${Object.entries(bins).sort((a,b)=>String(a[0]).localeCompare(String(b[0]),undefined,{numeric:true})).map(([label,count])=>`<div class="bar"><span>${escape(label)}</span><meter min="0" max="${Math.max(1,...Object.values(bins).filter(Number.isFinite))}" value="${count||0}"></meter><b>${count===null?'Hidden':count}</b></div>`).join('')}</section>`).join('');
 const rows=summary.days.map(day=>`<tr><td>${escape(day.date)}</td><td>${day.events.calculator_completed||0}</td><td>${day.events.report_rendered||0}</td><td>${day.events.pdf_download_initiated||0}</td><td>${day.events.cta_clicked||0}</td><td>${day.limited?'Limit reached':'Within ingestion limit'}</td></tr>`).join('');
 return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>Chance Me · Private usage report</title><style>
 *{box-sizing:border-box}body{font:16px/1.5 system-ui,sans-serif;background:#f3f7f6;color:#17332e;margin:0}main{max-width:1150px;margin:40px auto;padding:0 24px}header{padding-bottom:20px;border-bottom:1px solid #cfded9}h1{font-size:34px;margin:8px 0}h2{font-size:17px}p{max-width:850px;color:#47635a}.badge{display:inline-block;padding:5px 10px;border-radius:20px;background:#dcebe4;font-size:13px;font-weight:700}.warning{padding:14px;background:#fff0ce;color:#58441d;border-radius:10px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}article,.histogram{background:white;border:1px solid #dce6e1;border-radius:14px;padding:18px}article span{display:block;font-size:14px}strong{display:block;font-size:32px}.histograms{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}.bar{display:grid;grid-template-columns:minmax(100px,1fr) 110px 50px;align-items:center;gap:12px;margin:10px 0;font-size:14px}.bar span{overflow-wrap:anywhere}meter{width:100%;height:15px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #e1e9e5}.table{overflow:auto;margin:28px 0}footer{margin:30px 0;font-size:13px;color:#47635a}@media(max-width:500px){main{padding:0 14px}.histograms{grid-template-columns:1fr}.bar{grid-template-columns:1fr 80px 42px}}
 </style></head><body><main><header><span class="badge">${demo?'DEMO · SYNTHETIC DATA':'PRIVATE · OWNER REPORT'}</span><h1>How people use Chance Me</h1><p>${escape(summary.from)} to ${escape(summary.to)} · UTC days · Action counts, not unique people.</p></header>
 ${demo?'<p class="warning">This is a layout preview with invented test counts. It contains no real visitor data.</p>':''}
 ${summary.limited?'<p class="warning">An ingestion limit was reached. These counts are incomplete.</p>':''}
 <div class="cards">${cards}</div><p>Events can be blocked or lost. Repeat calculations count separately. A displayed report is not proof it was read; a download initiation is not proof the PDF was saved. Booking confirmations only reflect callbacks observed by the page.</p>
 <div class="table"><table><thead><tr><th>UTC date</th><th>Calculations</th><th>Reports</th><th>PDF downloads initiated</th><th>Clicks</th><th>Collection</th></tr></thead><tbody>${rows||'<tr><td colspan="6">No recorded actions in this period.</td></tr>'}</tbody></table></div>
 <div class="histograms">${histograms}</div><footer>Distributions are separate totals, not linked applicant profiles. Bins with fewer than five observations are hidden; this threshold alone is not an anonymity guarantee. No visitor journeys or source-to-conversion attribution are inferred. Retention: daily aggregates expire after at most 90 days; transient random event hashes expire within 48 hours. No historical backfill.</footer></main></body></html>`;
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
  const now=new Date(),state=demo?demoState(now):await redisStore().history(days,now);
  fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,renderReport(summarize(state,{days,now}),{demo}),{mode:0o600});console.log(`${demo?'Demo':'Private report'} saved: ${out}`);
 } catch(error) {console.error(error.message.startsWith('Storage is not configured')?error.message:'Report unavailable. Check approved storage configuration and connectivity.');process.exitCode=1;}
}
