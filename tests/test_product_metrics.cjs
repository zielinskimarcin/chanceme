const test = require('node:test'), assert = require('node:assert/strict');
const {randomUUID} = require('node:crypto');
const now = new Date('2026-10-29T12:00:00Z');
const api = import('../lib/product-metrics.mjs');
const batch = (event='calculator_completed', extra={}) => ({version:1,events:[{id:randomUUID(),event,...(event==='calculator_completed'?{test_type:'SAT',program:'Management',round:'Winter',status:'estimated',score_band:'1450-1499',gpa_band:'9.00-9.24'}:{}),...extra}]});
function memory() {
 let state=null, etag=0;
 return {read:async()=>state?{state:structuredClone(state),etag:String(etag)}:null,
 write:async(next,expected)=>{if((state?String(etag):undefined)!==expected){const e=new Error();e.code='CONFLICT';throw e;}state=structuredClone(next);etag++;},
 isConflict:e=>e.code==='CONFLICT',snapshot:()=>state};
}
test('rejects exact scores, identifiers, payloads, URLs, unknown dimensions and invalid bins',async()=>{
 const {validateBatch}=await api;
 for(const extra of [{score:1450},{gpa:9.12},{email:'private@example.com'},{url:'/#score=1450'},{visitor_id:'x'},{score_band:'1450'},{gpa_band:'9.12'},{test_type:'SAT',score_band:'32-33.99'},{program:'__proto__'}]) assert.throws(()=>validateBatch(batch('calculator_completed',extra)));
 assert.throws(()=>validateBatch({...batch(),private:'no'}));
 assert.throws(()=>validateBatch(batch('__proto__')));
 assert.throws(()=>validateBatch({version:1,events:Array.from({length:17},()=>batch().events[0])}));
});
test('never retains raw event IDs, combined profiles, time stamps or event rows',async()=>{
 const {validateBatch,aggregate,freshState}=await api;const input=batch();const out=aggregate(freshState(),validateBatch(input),now).state;
 assert.equal(out.days['2026-10-29'].events.calculator_completed,1);
 const raw=JSON.stringify(out);assert.ok(!raw.includes(input.events[0].id));assert.ok(!raw.includes('profile'));assert.ok(!raw.includes('12:00'));
 assert.deepEqual(out.days['2026-10-29'].distributions['calculator_completed:score_band'],{'SAT:1450-1499':1});
 assert.deepEqual(out.days['2026-10-29'].distributions['calculator_completed:gpa_band'],{'9.00-9.24':1});
});
test('duplicate event IDs count once, distinct calculations count separately',async()=>{
 const {validateBatch,collect}=await api;const store=memory();const input=validateBatch(batch());await collect(store,input,{now});await collect(store,input,{now});await collect(store,validateBatch(batch()),{now});
 assert.equal(store.snapshot().days['2026-10-29'].events.calculator_completed,2);
});
test('concurrent updates retry without overwriting counts',async()=>{
 const {validateBatch,collect}=await api;const store=memory();await Promise.all([collect(store,validateBatch(batch()),{now}),collect(store,validateBatch(batch()),{now})]);assert.equal(store.snapshot().days['2026-10-29'].events.calculator_completed,2);
});
test('limits cap ingestion and report explicitly marks incomplete days',async()=>{
 const {validateBatch,aggregate,freshState,MAX_DAILY_WRITES,summarize}=await api;const state=freshState();state.days['2026-10-29']={batches:MAX_DAILY_WRITES,events:{calculator_completed:12},distributions:{}};
 const out=aggregate(state,validateBatch(batch()),now);assert.equal(out.capped,true);assert.equal(out.changed,false);assert.equal(summarize(state,{now}).limited,true);
});
test('report hides small bins and all dedup data, excludes expired days',async()=>{
 const {summarize,freshState}=await api;const state=freshState();state.dedup=['private-hash'];state.days={'2026-10-29':{batches:1,events:{calculator_completed:10},distributions:{'calculator_completed:gpa_band':{'9.00-9.24':6,'9.25-9.49':4}}},'2026-01-01':{batches:1,events:{calculator_completed:50},distributions:{}}};
 const result=summarize(state,{now,days:90});assert.equal(result.totals.calculator_completed,10);assert.equal(result.distributions['calculator_completed:gpa_band']['9.00-9.24'],6);assert.equal(result.distributions['calculator_completed:gpa_band']['9.25-9.49'],undefined);assert.ok(!JSON.stringify(result).includes('private-hash'));
});
test('storage failures propagate to collector, without modifying the business API',async()=>{
 const {collect,validateBatch}=await api;await assert.rejects(collect({read:async()=>{throw new Error('offline');}},validateBatch(batch()),{now}));
 const fs=require('node:fs');assert.ok(!fs.readFileSync('api/index.py','utf8').includes('product-metrics'));
});
test('Redis adapter uses atomic expiry and short-lived independent dedup storage',async()=>{
 const {redisStore}=await import('../lib/product-metrics-store.mjs');const {aggregate,validateBatch,freshState}=await api;const calls=[];
 const env={UPSTASH_REDIS_REST_URL:'https://test.upstash.io',UPSTASH_REDIS_REST_TOKEN:'test-secret'};
 const store=redisStore({env,request:async(url,options)=>{calls.push(JSON.parse(options.body));return new Response(JSON.stringify({result:1}));}});
 const state=aggregate(freshState(),validateBatch(batch()),now).state;await store.write(state,undefined,now);
 const command=calls[0];assert.equal(command[0],'EVAL');assert.ok(command[1].includes("'EXAT'"));assert.equal(command[2],2);
 const persistent=JSON.parse(command[6]);assert.equal(persistent.dedup,undefined);assert.ok(Array.isArray(JSON.parse(command[7])));
 assert.equal(Number(command[8]),Date.parse('2027-01-27T00:00:00Z')/1000);assert.equal(Number(command[9]),Date.parse('2026-10-31T00:00:00Z')/1000);
 assert.ok(!JSON.stringify(persistent).includes('test-secret'));
});
test('storage configuration rejects redirected/non-HTTPS/unexpected destinations',async()=>{
 const {configured}=await import('../lib/product-metrics-store.mjs');for(const url of ['http://test.upstash.io','https://evil.test','https://test.upstash.io/path','https://test.upstash.io?token=x'])assert.equal(configured({UPSTASH_REDIS_REST_URL:url,UPSTASH_REDIS_REST_TOKEN:'x'}),false);
});
test('collector guards methods, origin, payload size, privacy signals and missing storage',async()=>{
 const {makeHandler}=await import('../api/product-events.mjs');let writes=0;
 const handler=makeHandler({enabled:()=>true,store:()=>({read:async()=>{writes++;return null;},write:async()=>{},isConflict:()=>false}),clock:()=>now.getTime()});
 async function invoke(changes={}){const req={method:'POST',headers:{origin:'https://chance-me.com','sec-fetch-site':'same-origin','content-type':'application/json'},body:batch(),...changes};const res={statusCode:200,setHeader(){},end(){}};await handler(req,res);return res.statusCode;}
 assert.equal(await invoke({method:'GET'}),405);assert.equal(await invoke({headers:{origin:'https://evil.test'}}),403);
 assert.equal(await invoke({headers:{origin:'https://chance-me.com','sec-fetch-site':'same-origin',dnt:'1'}}),204);assert.equal(writes,0);
 assert.equal(await invoke({body:'x'.repeat(9000)}),413);assert.equal(await invoke({body:batch('calculator_completed',{gpa:9.2})}),400);assert.equal(await invoke(),204);assert.equal(writes,1);
});
test('collector returns generic error without sensitive data when storage fails',async()=>{
 const {makeHandler}=await import('../api/product-events.mjs');const handler=makeHandler({enabled:()=>true,store:()=>({read:async()=>{throw new Error('secret payload');}}),clock:()=>now.getTime()});const res={setHeader(){},end(body){this.body=body;}};await handler({method:'POST',headers:{origin:'https://chance-me.com','sec-fetch-site':'same-origin','content-type':'application/json'},body:batch()},res);assert.equal(res.statusCode,503);assert.equal(res.body,undefined);
});
test('private report labels absent observations as unknown rather than verified zero traffic',async()=>{
 const {renderReport}=await import('../scripts/product-metrics-report.mjs');const {summarize,freshState}=await api;const html=renderReport(summarize(freshState(),{now}));assert.ok(html.includes('No observations are available'));assert.ok(html.includes('collection health is unknown'));assert.ok(html.includes('0 day(s) with recorded observations'));assert.ok(!html.includes('private-hash'));
});
test('provider-managed Vercel KV variable names work without duplicating secrets',async()=>{
 const {configured}=await import('../lib/product-metrics-store.mjs');assert.equal(configured({KV_REST_API_URL:'https://test.upstash.io',KV_REST_API_TOKEN:'secret'}),true);
});

test('collector supports valid-origin browsers without Fetch Metadata but still rejects cross-origin or missing Origin',async()=>{
 const {makeHandler}=await import('../api/product-events.mjs');let writes=0;const handler=makeHandler({enabled:()=>true,clock:()=>now.getTime(),store:()=>({read:async()=>null,write:async()=>{writes++;},isConflict:()=>false})});
 const invoke=async headers=>{const res={setHeader(){},end(){}};await handler({method:'POST',headers:{'content-type':'application/json',...headers},body:batch()},res);return res.statusCode;};
 assert.equal(await invoke({origin:'https://chance-me.com'}),204);assert.equal(writes,1);
 assert.equal(await invoke({origin:'https://evil.example'}),403);assert.equal(await invoke({}),403);assert.equal(await invoke({origin:'https://chance-me.com','sec-fetch-site':'cross-site'}),403);assert.equal(writes,1);
});
