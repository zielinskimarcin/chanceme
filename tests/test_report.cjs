const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
 constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.attributes={};this.dataset={};this.listeners={};this.hidden=false;this.value='';this.style={setProperty(){}};this.classes=new Set();this.classList={add:(...n)=>n.forEach(v=>this.classes.add(v)),remove:(...n)=>n.forEach(v=>this.classes.delete(v)),toggle:(n,v)=>v?this.classes.add(n):this.classes.delete(n)};}
 set textContent(v){this.text=String(v);this.children=[];}get textContent(){return(this.text||'')+this.children.map(e=>e.textContent).join('');}
 append(...n){n.forEach(e=>{e.parent=this;this.children.push(e);});}replaceChildren(...n){this.text='';this.children=[];this.append(...n);}get lastChild(){return this.children.at(-1);}
 insertBefore(e,target){e.parent=this;this.children.splice(this.children.indexOf(target),0,e);}remove(){if(this.parent)this.parent.children=this.parent.children.filter(e=>e!==this);}
 removeAttribute(k){delete this.attributes[k];}setAttribute(k,v){this.attributes[k]=String(v);}getAttribute(k){return this.attributes[k];}addEventListener(k,v){this.listeners[k]=v;}focus(){this.focused=true;}scrollIntoView(){this.scrolled=(this.scrolled||0)+1;}click(){}
 setPointerCapture(id){this.capture=id;}hasPointerCapture(id){return this.capture===id;}releasePointerCapture(){this.capture=null;}
 closest(selector){return selector.split(',').some(s=>{const c=s.trim().slice(1);return (this.className||this.attributes.class||'').split(' ').includes(c);})?this:this.parent?this.parent.closest(selector):null;}
 querySelectorAll(tag){return descend(this,e=>e!==this&&e.tagName===tag.toUpperCase());}getBoundingClientRect(){return{left:0,top:0,width:700,height:300};}
}
function descend(e,p){return [e,...e.children.flatMap(c=>descend(c,()=>true))].filter(p);}
const tick=()=>new Promise(r=>setImmediate(r));
const profile={test_type:'SAT',score:1450,gpa:9,course:'Management',session:'Winter',cycle:'27-28',applicant_category:'international',gpa_confirmed:true};
function comparison(count=8){return{status:'ok',complete:true,records:Array.from({length:count},(_,i)=>({id:'s'+i,score:1400+i*10,gpa:8.8,cycle:'26-27',session:i%2?'Early':'Winter',outcome:i%2?'admitted':'rejected',nearby:true})),nearby:{total:count,admitted:count/2,rejected:count/2,test_window:50,gpa_window:.5},by_year:[{cycle:'26-27',nearby_total:count,nearby_admitted:count/2,nearby_rejected:count/2}]};}
function setup(fetcher,fullPage=false){const ids={calculator:new Element(),evidence:new Element(),'evidence-body':new Element(),score:new Element('input'),course:new Element('select'),session:new Element('select')};ids.score.value='1450';const calls=[],results=[],events={};const scope={Node:Element,Blob,URLSearchParams,window:{history:{replaceState(...args){scope.lastURL=args[2];}},location:{assign(){}}},URL:{createObjectURL:()=>'',revokeObjectURL(){}},AbortController,Map,WeakMap,console,setTimeout,clearTimeout,requestAnimationFrame:f=>setTimeout(f,0),finite:n=>typeof n==='number'&&Number.isFinite(n),reduceMotion:{matches:true},evidenceViewport:{matches:false},renderResult:(data,input,options)=>results.push({data,input,options}),form:ids.calculator,document:{addEventListener:(name,handler)=>events[name]=handler,body:new Element('body'),getElementById:k=>ids[k],createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)},fetch:async(url,init)=>{calls.push({url,init});return fetcher?fetcher(url,init):{ok:true,json:async()=>comparison()};}};if(fullPage)scope.document.body.dataset.page="report";const ctx=vm.createContext(scope);const source=fs.readFileSync('dist/campus.js','utf8');vm.runInContext(source.slice(source.indexOf('function node('),source.indexOf('function readNumber(')),ctx);vm.runInContext(source.slice(source.indexOf('function year('),source.indexOf('function drawEvidenceChart(')),ctx);vm.runInContext(fs.readFileSync('dist/report.js','utf8'),ctx);const api=vm.runInContext('ChanceReport',ctx);api.setPrograms([{id:'Management',name:'Management'},{id:'Finance',name:'Finance'}]);return{api,ids,calls,scope,results,events};}

const estimate=(percent=59)=>({status:'estimated',probability:percent/100,display_percent:percent+'%'});
const ok=value=>({ok:true,json:async()=>value});
const find=(x,p)=>descend(x.ids['evidence-body'],p)[0];
const byId=(x,id)=>find(x,e=>e.id===id);
const mini=x=>find(x,e=>e.className==='report-mini-ring');
const selected=(x,id,value)=>{const el=byId(x,id);el.value=value;return el.listeners.change();};
test('Calculate opens inline details without hiding the calculator, scrolling or focusing the report',async()=>{
 const x=setup();x.api.render(estimate(),profile);await tick();
 assert.equal(x.ids.calculator.hidden,false);assert.equal(x.ids.evidence.open,true);assert.equal(x.ids.evidence.hidden,false);
 assert.equal(x.scope.document.body.classes.has('report-mode'),false);assert.equal(x.ids.evidence.scrolled,undefined);
 assert.equal(find(x,e=>e.className==='report-title').focused,undefined);assert.equal(find(x,e=>e.className==='report-profile-bar'),undefined);
 assert.equal(x.calls.length,1);assert.equal(x.calls[0].url,'/api/reports');assert.match(x.ids['evidence-body'].textContent,/8 reports/);
 x.api.close();assert.equal(x.ids.calculator.hidden,false);assert.equal(x.ids.score.value,'1450');
});
test('programme and round filters recalculate the hero and form without recreating the report',async()=>{
 const x=setup(async(url,init)=>ok(url==='/api/estimate'?estimate(JSON.parse(init.body).session==='Early'?54:48):comparison(2)));
 x.api.render(estimate(),profile);await tick();const course=byId(x,'report-course');
 await selected(x,'report-course','Finance');assert.equal(x.results.at(-1).data.display_percent,'48%');assert.equal(x.ids.course.value,'Finance');
 await selected(x,'report-round','Early');assert.equal(x.results.at(-1).data.display_percent,'54%');assert.equal(x.ids.session.value,'Early');
 assert.equal(x.results.length,2);assert.equal(x.results[1].data.probability,.54);assert.equal(x.results[1].options.skipEvidence,true);
 assert.equal(x.results.at(-1).input.course,'Finance');assert.equal(byId(x,'report-course'),course);
 const sent=x.calls.filter(c=>c.url==='/api/estimate').map(c=>JSON.parse(c.init.body));assert.equal(sent[1].score,1450);assert.equal(sent[1].gpa,9);assert.equal(sent[1].course,'Finance');assert.equal(sent[1].session,'Early');
});
test('late estimate and comparison responses cannot overwrite a newer selection',async()=>{
 const pending=[];const x=setup(()=>new Promise(resolve=>pending.push(resolve)));x.api.render(estimate(),profile);
 const old=selected(x,'report-round','Early');const latest=selected(x,'report-course','Finance');
 assert.equal(find(x,e=>e.className==='report-download report-full-link').getAttribute('aria-disabled'),'true');pending[3](ok(comparison(2)));pending[4](ok(estimate(54)));await latest;
 pending[1](ok(comparison(4)));pending[2](ok(estimate(85)));await old;pending[0](ok(comparison(8)));await tick();
 assert.equal(x.results.at(-1).data.display_percent,'54%');assert.match(x.ids['evidence-body'].textContent,/2 reports/);assert.equal(x.results.length,1);assert.equal(x.ids.course.value,'Finance');
});
test('failed recalculation restores the last labelled result and offers a retry',async()=>{
 let fail=false;const x=setup(async(url)=>{if(fail&&url==='/api/estimate')throw new Error('offline');return ok(comparison());});
 x.api.render(estimate(),profile);await tick();fail=true;await selected(x,'report-course','Finance');
 assert.equal(byId(x,'report-course').value,'Management');assert.equal(find(x,e=>e.className==='report-profile-bar'),undefined);assert.equal(x.results.length,0);
 assert.match(x.ids['evidence-body'].textContent,/Couldn’t update the estimate/);assert.match(x.ids['evidence-body'].textContent,/Try again/);
});
test('comparison failure keeps the real estimate and offers retry without invented counts',async()=>{
 const x=setup(async()=>{throw new Error('offline');});x.api.render(estimate(),profile);await tick();
 assert.equal(x.results.length,0);assert.match(x.ids['evidence-body'].textContent,/Couldn’t load/);assert.match(x.ids['evidence-body'].textContent,/Try again/);assert.doesNotMatch(x.ids['evidence-body'].textContent,/0 of 0/);
});
test('a new estimate remains available even when its comparison request fails',async()=>{
 let fail=false;const x=setup(async(url)=>{if(fail&&url==='/api/reports')throw new Error('offline');return ok(url==='/api/estimate'?estimate(54):comparison());});
 x.api.render(estimate(),profile);await tick();fail=true;await selected(x,'report-round','Early');
 assert.equal(x.results.at(-1).data.display_percent,'54%');assert.equal(x.results[0].input.session,'Early');assert.match(x.ids['evidence-body'].textContent,/Reports unavailable/);
});
test('year filters historical points and counts without claiming a new admission-cycle estimate',async()=>{
 const d=comparison();d.records[0].cycle='25-26';d.by_year.push({cycle:'25-26',nearby_total:1,nearby_admitted:0,nearby_rejected:1});const x=setup(async()=>ok(d));
 x.api.render(estimate(),profile);await tick();await selected(x,'evidence-cycle','25-26');assert.equal(x.calls.length,1);assert.equal(find(x,e=>e.className==='report-profile-bar'),undefined);
 const host=find(x,e=>e.className==='report-plot');assert.match(host.children[0].getAttribute('aria-label'),/^1 past applicant/);assert.match(find(x,e=>e.className==='context-summary').textContent,/0 of 1/);
});
test('homepage chart has no pan, zoom or expand interactions and only individual rounds',async()=>{
 const x=setup();x.api.render(estimate(),profile);await tick();const host=find(x,e=>e.className==='report-plot');
 for(const name of ['wheel','pointerdown','pointermove','pointerup'])assert.equal(host.listeners[name],undefined);
 assert.doesNotMatch(x.ids['evidence-body'].textContent,/Expand|Collapse|Near my scores|All scores|Both rounds/);
 assert.equal(byId(x,'report-round').children.length,2);
});
test('click opens point details; blank chart or outside click dismisses them; hover does not open them',async()=>{
 const x=setup();x.api.render(estimate(),profile);await tick();const point=find(x,e=>e.getAttribute('class')==='chart-report'),tip=byId(x,'chart-report-tooltip'),host=find(x,e=>e.className==='report-plot');
 assert.equal(point.listeners.pointerenter,undefined);assert.equal(point.listeners.focus,undefined);
 point.listeners.click({clientX:250,clientY:140});x.events.click({target:point.children[0]});assert.equal(tip.hidden,false);
 assert.match(tip.textContent,/SAT/);assert.match(tip.textContent,/Bocconi GPA/);assert.match(tip.textContent,/2026\/27/);assert.match(tip.textContent,/Admitted|Not admitted/);
 x.events.click({target:host});assert.equal(tip.hidden,true);point.listeners.click({});assert.equal(tip.hidden,false);x.events.click({target:x.ids.calculator});assert.equal(tip.hidden,true);
});
test('keyboard Enter opens a point and Escape closes it from anywhere',async()=>{
 const x=setup();x.api.render(estimate(),profile);await tick();const point=find(x,e=>e.getAttribute('class')==='chart-report'),tip=byId(x,'chart-report-tooltip');
 let prevented=false;point.listeners.keydown({key:'Enter',preventDefault(){prevented=true;}});assert.equal(prevented,true);assert.equal(tip.hidden,false);x.events.keydown({key:'Escape'});assert.equal(tip.hidden,true);
});
test('reset makes pending filter responses unable to repaint a stale result',async()=>{
 const pending=[];const x=setup(()=>new Promise(resolve=>pending.push(resolve)));x.api.render(estimate(),profile);const next=selected(x,'report-round','Early');x.api.reset();
 pending[0](ok(comparison()));pending[1](ok(comparison()));pending[2](ok(estimate(54)));await next;await tick();assert.equal(x.results.length,0);assert.equal(x.ids.evidence.hidden,true);assert.equal(x.ids.calculator.hidden,false);
});
test('PDF export sends the current recalculated profile and comparison filters',async()=>{
 const x=setup(async(url)=>url==='/api/report.pdf'?{ok:true,blob:async()=>new Blob(['%PDF-1.7'],{type:'application/pdf'})}:ok(url==='/api/estimate'?estimate(54):comparison()));
 x.api.render(estimate(),profile);await tick();await selected(x,'report-course','Finance');await selected(x,'report-round','Early');await selected(x,'evidence-cycle','26-27');
 await find(x,e=>e.className==='text-button report-pdf').listeners.click();const body=JSON.parse(x.calls.at(-1).init.body);assert.equal(body.profile.course,'Finance');assert.equal(body.profile.session,'Early');assert.equal(body.filters.cycle,'26-27');
});
test('CSV quotes cells and prevents spreadsheet formula injection',()=>{const x=setup();assert.equal(x.api.csvCell('=HYPERLINK("bad")'),'"\'=HYPERLINK(""bad"")"');assert.equal(x.api.csvCell(1450),'"1450"');assert.equal(x.api.csvCell('26-27'),'"26-27"');});

test('only an explicit Explore action scrolls to inline details',async()=>{
 const x=setup();x.api.render(estimate(),profile);await tick();assert.equal(x.ids.evidence.scrolled,undefined);
 x.api.open({scroll:true});assert.equal(x.ids.evidence.scrolled,1);assert.equal(x.ids.calculator.hidden,false);
});

test('homepage chart and table share 20 nearest reports while counts remain complete',async()=>{
 const d=comparison(40);d.records.forEach((r,i)=>{r.score=1200+i*10;r.gpa=9;r.cycle=i<5?'25-26':'26-27';});d.records.reverse();
 d.by_year=[{cycle:'25-26',nearby_total:5,nearby_admitted:2,nearby_rejected:3},{cycle:'26-27',nearby_total:35,nearby_admitted:18,nearby_rejected:17}];
 const x=setup(async()=>ok(d));x.api.render(estimate(),profile);await tick();
 const chart=()=>find(x,e=>e.className==='report-plot');
 const markers=()=>descend(chart(),e=>e.getAttribute('class')==='chart-report');
 assert.equal(markers().length,20);assert.match(chart().children[0].getAttribute('aria-label'),/^20 past applicant/);
 const labels=markers().map(p=>p.getAttribute('aria-label')).join(';');assert.match(labels,/, 1450,/);assert.match(labels,/, 1360,/);assert.match(labels,/, 1540,/);assert.doesNotMatch(labels,/, 1200,|, 1590,/);
 assert.match(find(x,e=>e.className==='context-summary').textContent,/20 of 40/);
 assert.match(find(x,e=>e.className==='reports-heading').textContent,/6 of 20/);
 const tableHost=find(x,e=>e.className==='context-reports'),more=find(x,e=>e.className==='text-button reports-more'),svg=chart().children[0];
 assert.equal(descend(tableHost,e=>e.tagName==='TR').length,7);assert.equal(more.textContent,'Show more ↓');assert.equal(more.getAttribute('aria-expanded'),'false');
 more.listeners.click();assert.equal(descend(tableHost,e=>e.tagName==='TR').length,21);assert.equal(chart().children[0],svg);assert.equal(more.getAttribute('aria-expanded'),'true');
 more.listeners.click();assert.equal(descend(tableHost,e=>e.tagName==='TR').length,7);assert.equal(chart().children[0],svg);
 await selected(x,'evidence-cycle','25-26');assert.equal(markers().length,5);assert.match(chart().children[0].getAttribute('aria-label'),/^5 past applicant/);
 assert.ok(markers().every(p=>p.getAttribute('aria-label').includes('2025/26')));assert.match(chart().textContent,/1200/);
});
test('nearest displayed reports are selected again for each programme and round response',async()=>{
 const x=setup(async(url,init)=>{const p=JSON.parse(init.body);if(url==='/api/estimate')return ok(estimate(54));const d=comparison(30);d.records.forEach((r,i)=>{r.score=1300+i*10;r.gpa=p.course==='Finance'?9.1:9;r.session=p.session;});return ok(d);});
 x.api.render(estimate(),profile);await tick();const labels=()=>descend(find(x,e=>e.className==='report-plot'),e=>e.getAttribute('class')==='chart-report').map(p=>p.getAttribute('aria-label'));
 assert.equal(labels().length,20);assert.ok(labels().every(s=>s.includes('GPA 9,')));
 await selected(x,'report-course','Finance');assert.equal(labels().length,20);assert.ok(labels().every(s=>s.includes('GPA 9.1,')));
 await selected(x,'report-round','Early');assert.equal(labels().length,20);assert.ok(labels().every(s=>s.endsWith('Early')));
});

test('homepage action links replace CSV with a secondary PDF and a new-tab full report',async()=>{
 const x=setup();x.api.render(estimate(),{...profile,gpa_estimated:true});await tick();await selected(x,'evidence-cycle','26-27');
 const link=find(x,e=>e.className==='report-download report-full-link');assert.equal(link.target,'_blank');assert.equal(link.rel,'noopener noreferrer');
 const params=new URLSearchParams(link.href.split('#')[1]);assert.equal(params.get('score'),'1450');assert.equal(params.get('gpa'),'9');assert.equal(params.get('gpa_estimated'),'true');assert.equal(params.get('year'),'26-27');assert.ok(link.href.startsWith('/report/#'));assert.ok(!link.href.includes('?'));
 assert.ok(find(x,e=>e.className==='text-button report-pdf'));assert.doesNotMatch(x.ids['evidence-body'].textContent,/Export CSV/);assert.equal(find(x,e=>e.className==='report-profile-bar'),undefined);
});
test('table rows exactly match displayed reports after year filtering',async()=>{
 const d=comparison(40);d.records.forEach((r,i)=>{r.score=1200+i*10;r.gpa=9;r.cycle=i<6?'25-26':'26-27';});d.by_year.push({cycle:'25-26',nearby_total:6,nearby_admitted:3,nearby_rejected:3});
 const x=setup(async()=>ok(d));x.api.render(estimate(),profile);await tick();
 function check(expected){const points=descend(find(x,e=>e.className==='report-plot'),e=>e.getAttribute('class')==='chart-report');const tr=descend(find(x,e=>e.className==='context-reports'),e=>e.tagName==='TR').slice(1);assert.equal(points.length,expected);assert.equal(tr.length,expected);assert.deepEqual(tr.map(r=>Number(r.children[0].textContent.replaceAll(',',''))),points.map(p=>Number(p.getAttribute('aria-label').split(', ')[1])));}
 find(x,e=>e.className==='text-button reports-more').listeners.click();check(20);await selected(x,'evidence-cycle','25-26');check(6);assert.equal(find(x,e=>e.className==='text-button reports-more').hidden,true);
});
test('separate full report restores the full dataset, ring, zoom, pan and expand',async()=>{
 const d=comparison(40);d.records.forEach((r,i)=>{r.score=1200+i*10;r.gpa=9;});const x=setup(async()=>ok(d),true);x.api.render(estimate(),profile);await tick();
 assert.equal(x.ids.calculator.hidden,true);assert.equal(x.scope.document.body.classes.has('report-mode'),true);assert.match(mini(x).textContent,/59%/);
 const host=find(x,e=>e.className==='report-plot');assert.match(host.children[0].getAttribute('aria-label'),/^40 past applicant/);assert.equal(descend(host,e=>e.getAttribute('class')==='chart-report').length,40);
 const before=host.textContent;find(x,e=>e.getAttribute('aria-label')==='Zoom in').listeners.click();assert.notEqual(host.textContent,before);
 host.listeners.keydown({key:'ArrowLeft',target:host,preventDefault(){}});assert.equal(typeof host.listeners.pointermove,'function');assert.equal(typeof host.listeners.wheel,'function');
 host.listeners.pointerdown({button:0,target:host,clientX:100,clientY:100,pointerId:1});host.listeners.pointermove({clientX:200,clientY:120});await new Promise(r=>setTimeout(r,5));host.listeners.pointerup();
 find(x,e=>e.textContent==='⛶  Expand').listeners.click();assert.match(host.children[0].getAttribute('viewBox'),/1000 470/);assert.match(mini(x).textContent,/59%/);
});
test('full-report filters recalculate its ring and update the reloadable URL',async()=>{
 const x=setup(async(url)=>ok(url==='/api/estimate'?estimate(54):comparison()),true);x.api.render(estimate(),profile,{cycle:'26-27'});await tick();await selected(x,'report-course','Finance');await selected(x,'report-round','Early');
 assert.match(mini(x).textContent,/54%/);assert.match(find(x,e=>e.className==='report-profile-terms').textContent,/FinanceRoundEarly/);const params=new URLSearchParams(x.scope.lastURL.split('#')[1]);assert.equal(params.get('course'),'Finance');assert.equal(params.get('session'),'Early');assert.equal(params.get('year'),'26-27');
});

test('full report counts only populated comparison once and PDF only after response blob exists',async()=>{
 const events=[],x=setup(async url=>url==='/api/report.pdf'?{ok:true,blob:async()=>new Blob(['%PDF-1.7'],{type:'application/pdf'})}:ok(url==='/api/estimate'?estimate():comparison()),true);
 x.scope.window.ChanceAnalytics={track:(...args)=>events.push(args)};
 x.api.render(estimate(),profile);assert.equal(events.length,0);await tick();assert.equal(JSON.stringify(events),JSON.stringify([['report_rendered']]));
 await selected(x,'report-course','Finance');assert.equal(events.filter(e=>e[0]==='report_rendered').length,1);
 await find(x,e=>e.className==='report-download').listeners.click();assert.equal(events.at(-2)[0],'pdf_generated');assert.equal(events.at(-1)[0],'pdf_download_initiated');assert.equal(JSON.stringify(events.at(-1)[1]),JSON.stringify({destination:'pdf'}));
});
test('failed report comparison or PDF never emits successful render or download',async()=>{
 const events=[],x=setup(async()=>{throw new Error('offline');},true);x.scope.window.ChanceAnalytics={track:(...args)=>events.push(args)};
 x.api.render(estimate(),profile);await tick();assert.equal(JSON.stringify(events),JSON.stringify([['report_failed',{reason:'unavailable'}]]));
 await find(x,e=>e.className==='report-download').listeners.click();assert.equal(events.some(e=>e[0]==='pdf_download_initiated'),false);
 const y=setup();y.scope.window.ChanceAnalytics={track:(...args)=>events.push(args)};y.api.render(estimate(),profile);await tick();assert.equal(events.some(e=>e[0]==='report_rendered'),false);
});

test('PDF events require a nonempty PDF response and ignore request cancellation',async()=>{
 for(const blob of [new Blob([],{type:'application/pdf'}),new Blob(['error'],{type:'text/html'}),new Blob(['pdf'])]){
  const events=[],x=setup(async url=>url==='/api/report.pdf'?{ok:true,blob:async()=>blob}:ok(comparison()),true);
  x.scope.window.ChanceAnalytics={track:(...args)=>events.push(args)};x.api.render(estimate(),profile);await tick();events.length=0;
  await find(x,e=>e.className==='report-download').listeners.click();assert.equal(JSON.stringify(events),JSON.stringify([['pdf_failed',{reason:'unavailable'}]]));
 }
 const events=[],x=setup(async url=>{if(url==='/api/report.pdf'){const error=new Error('cancelled');error.name='AbortError';throw error;}return ok(comparison());},true);
 x.scope.window.ChanceAnalytics={track:(...args)=>events.push(args)};x.api.render(estimate(),profile);await tick();events.length=0;
 await find(x,e=>e.className==='report-download').listeners.click();assert.equal(events.length,0);
});

test('full-report link carries allowed visit context only in the fragment',async()=>{
 const visit='12345678-1234-4123-8123-123456789abc',x=setup();x.scope.window.ChanceAnalytics={visitId:()=>visit};
 const url=x.api.reportURL(profile,'26-27');assert.equal(url.split('#')[0],'/report/');assert.equal(new URLSearchParams(url.split('#')[1]).get('visit_id'),visit);
 x.scope.window.ChanceAnalytics.visitId=()=>null;assert.equal(new URLSearchParams(x.api.reportURL(profile,'').split('#')[1]).has('visit_id'),false);
 x.scope.window.ChanceAnalytics.visitId=()=> 'invalid';assert.equal(x.api.reportURL(profile,'').includes('visit_id'),false);
});
