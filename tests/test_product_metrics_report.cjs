const test=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const now=new Date('2026-10-30T01:00:00Z');
test('private history preserves exact values and safely embeds untrusted record text',async()=>{
 const {renderReport}=await import('../scripts/product-metrics-report.mjs');
 const {summarize,freshState}=await import('../lib/product-metrics.mjs');
 const records=[{event:'calculator_completed',dimensions:{test_type:'SAT',program:'Finance',round:'Winter'},details:{visit_id:'visit-123',score:1520,gpa:9.32,page:'</script><img src=x onerror=alert(1)>',referrer:'https://example.com/?x=<x>',occurred_at:now.toISOString()},country:'PL',region:'14',city:'Warsaw',received_at:now.toISOString()}];
 const html=renderReport(summarize(freshState(),{now}),{records,now});
 assert.ok(html.includes('history-country'));assert.ok(html.includes('history-sort'));assert.ok(html.includes('Visit timelines'));assert.ok(html.includes('2026-10-30T01:00:00.000Z'));
 assert.ok(!html.includes('</script><img'));assert.ok(!html.includes('<img src=x'));assert.ok(!html.includes('innerHTML'));
 const json=html.match(/<script type="application\/json" id="history-data">([\s\S]*?)<\/script>/)[1];
 assert.deepEqual(JSON.parse(json),records);
 const script=html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
 assert.doesNotThrow(()=>new vm.Script(script));
});
test('empty history is an explicit snapshot with no fabricated rows',async()=>{
 const {renderReport}=await import('../scripts/product-metrics-report.mjs');
 const {summarize,freshState}=await import('../lib/product-metrics.mjs');
 const html=renderReport(summarize(freshState(),{now}),{now});
 assert.ok(html.includes('id="history-data">[]</script>'));assert.ok(html.includes('regenerate the private report'));assert.ok(html.includes('old aggregate counts have no visit-level backfill'));
 assert.ok(!html.includes('DEMO · SYNTHETIC DATA'));
});
test('history date/country/action filters and order drive the actual visible rows',async()=>{
 const {renderReport}=await import('../scripts/product-metrics-report.mjs');
 const {summarize,freshState}=await import('../lib/product-metrics.mjs');
 const records=[
  {event:'calculator_completed',details:{visit_id:'a',score:1520,gpa:9.32,occurred_at:'2026-10-29T12:00:00Z'},country:'PL'},
  {event:'report_rendered',details:{visit_id:'a',occurred_at:'2026-10-29T12:01:00Z'},country:'PL'},
  {event:'pdf_download_initiated',details:{visit_id:'b',occurred_at:'2026-10-30T12:00:00Z'},country:'US'},
 ];
 const html=renderReport(summarize(freshState(),{now}),{records,now});
 class Element{constructor(){this.value='';this.children=[];this.listeners={};this.textContent=''}append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes}addEventListener(name,fn){this.listeners[name]=fn}}
 const ids=new Map();const byId=id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id)};
 byId('history-data').textContent=JSON.stringify(records);byId('history-sort').value='desc';
 const script=html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1];
 vm.runInNewContext(script,{document:{getElementById:byId,createElement:()=>new Element()}});
 assert.equal(byId('history-rows').children.length,3);assert.equal(byId('history-visits').children.length,2);
 assert.equal(byId('history-rows').children[0].children[1].textContent,'b');
 byId('history-country').value='PL';byId('history-country').listeners.input();
 assert.equal(byId('history-rows').children.length,2);assert.equal(byId('history-visits').children.length,1);
 byId('history-event').value='calculator_completed';byId('history-event').listeners.input();
 assert.equal(byId('history-rows').children.length,1);assert.ok(byId('history-rows').children[0].children[3].textContent.includes('Score: 1520'));
 byId('history-country').value='';byId('history-event').value='';byId('history-from').value='2026-10-30';byId('history-from').listeners.input();
 assert.equal(byId('history-rows').children.length,1);assert.equal(byId('history-rows').children[0].children[1].textContent,'b');
 byId('history-from').value='';byId('history-sort').value='asc';byId('history-sort').listeners.input();
 assert.equal(byId('history-rows').children[0].children[1].textContent,'a');
 const restoredIds=new Map();const restored=id=>{if(!restoredIds.has(id))restoredIds.set(id,new Element());return restoredIds.get(id)};restored('history-data').textContent=JSON.stringify(records);let savedHash='';
 vm.runInNewContext(script,{document:{getElementById:restored,createElement:()=>new Element()},URLSearchParams,location:{hash:'#country=PL&event=calculator_completed&sort=asc'},history:{replaceState:(state,title,url)=>{savedHash=url}}});
 assert.equal(restored('history-country').value,'PL');assert.equal(restored('history-event').value,'calculator_completed');assert.equal(restored('history-rows').children.length,1);
 restored('history-visit').value='a';restored('history-visit').listeners.input();const hashParams=new URLSearchParams(savedHash.slice(1));assert.equal(hashParams.get('visit'),'a');assert.equal(hashParams.get('country'),'PL');assert.ok(!savedHash.includes('1520'));assert.ok(!savedHash.includes('9.32'));
});
test('QA separation, inferred visit source, full inputs and filtered formula-safe CSV',async()=>{
 const {renderReport}=await import('../scripts/product-metrics-report.mjs');
 const {summarize,freshState}=await import('../lib/product-metrics.mjs');
 const records=[
 {event:'page_view',traffic:'user',details:{visit_id:'visitor',source:'google',page:'/calculator',occurred_at:'2026-10-29T12:00:00Z'}},
 {event:'calculator_completed',traffic:'user',dimensions:{program:'=HYPERLINK("evil")',test_type:'SAT',round:'Winter'},details:{visit_id:'visitor',score:1510,gpa:9.4,cycle:'2027',applicant_category:'International',gpa_confirmed:true,gpa_estimated:false,inputs:{grades:[9,10]},result:{probability:0.72,status:'ok'},browser:'Firefox',os:'Windows',timezone:'Europe/Warsaw',occurred_at:'2026-10-29T12:01:00Z'}},
 {event:'calculator_completed',traffic:'test',details:{visit_id:'qa_1',source:'owner-qa',score:1600,occurred_at:'2026-10-29T13:00:00Z'}},
 ];
 const html=renderReport(summarize(freshState(),{now}),{records,now});
 class Element{constructor(){this.value='';this.children=[];this.listeners={};this.textContent=''}append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes}addEventListener(n,fn){this.listeners[n]=fn}click(){}}
 const ids=new Map();const get=id=>{if(!ids.has(id))ids.set(id,new Element());return ids.get(id)};get('history-data').textContent=JSON.stringify(records);get('history-sort').value='desc';let exported;
 class CsvURL extends URL{static createObjectURL(blob){exported=blob;return 'blob:test'}static revokeObjectURL(){}}
 vm.runInNewContext(html.match(/<\/script><script>([\s\S]*?)<\/script>/)[1],{document:{getElementById:get,createElement:()=>new Element()},Blob,URL:CsvURL,setTimeout:()=>{}});
 assert.equal(get('history-traffic').value,'user');assert.equal(get('history-rows').children.length,2);assert.ok(get('history-traffic-note').textContent.includes('1 known QA/test'));
 get('history-source').value='google';get('history-source').listeners.input();assert.equal(get('history-rows').children.length,2);
 const row=get('history-rows').children[0];assert.ok(row.children[3].textContent.includes('1510'));assert.ok(row.children[3].textContent.includes('grades'));assert.ok(row.children[3].textContent.includes('International'));assert.ok(row.children[5].textContent.includes('Firefox'));assert.ok(row.children[5].textContent.includes('Europe/Warsaw'));
 assert.equal(get('history-visits').children[0].open,true);assert.ok(get('history-visits').children[0].children.some(n=>n.className==='visit-preview'&&n.textContent.includes('1510')));
 get('history-export').listeners.click();const csv=await exported.text();assert.ok(csv.includes("'=HYPERLINK"));assert.ok(csv.includes('1510'));assert.ok(!csv.includes('1600'));assert.ok(csv.includes('google'));assert.ok(csv.includes('gpa_confirmed'));
 get('history-source').value='';get('history-traffic').value='test';get('history-traffic').listeners.input();assert.equal(get('history-rows').children.length,1);assert.ok(get('history-rows').children[0].children[3].textContent.includes('1600'));
});
