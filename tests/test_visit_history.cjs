const test=require('node:test'),assert=require('node:assert/strict');
const {randomUUID}=require('node:crypto');
const now=new Date('2026-10-30T00:30:00Z');
const payload=()=>({version:2,events:[{id:randomUUID(),event:'calculator_completed',visit_id:randomUUID(),occurred_at:'2026-10-30T00:29:00Z',page:'/',referrer:'https://google.com/',device:'mobile',source:'google',test_type:'SAT',program:'Management',round:'Winter',status:'estimated',score:1490,gpa:9.12,probability:72.34}]});
test('v2 retains submitted exact input, linked visit time and trusted approximate geography; deduplicates retries',async()=>{
 const {validateBatch,aggregate,freshState,requestGeography}=await import('../lib/product-metrics.mjs');
 const body=payload(),events=validateBatch(body),geo=requestGeography({'x-vercel-ip-country':'PL','x-vercel-ip-country-region':'MZ','x-vercel-ip-city':'Warszawa'});
 const first=aggregate(freshState(),events,now,geo);const second=aggregate(first.state,events,now,geo);
 assert.equal(second.changed,false);assert.equal(first.state.records.length,1);
 const row=first.state.records[0];assert.equal(row.dedup,undefined);assert.equal(row.details.score,1490);assert.equal(row.details.gpa,9.12);assert.equal(row.details.visit_id,body.events[0].visit_id);assert.equal(row.country,'PL');assert.equal(row.received_at,now.toISOString());assert.equal(row.details.occurred_at,'2026-10-30T00:29:00.000Z');
 for(const extra of [{gpa:11},{score:1700},{page:'/?email=x'},{referrer:'https://google.com/?secret=x'},{email:'x@y.com'},{visit_id:'bad'},{probability:Infinity},{source:'<script>'}]) assert.throws(()=>validateBatch({version:2,events:[{...body.events[0],...extra}]}));
});
test('v2 atomic journal has 30 day expiry separate from 90 day aggregates',async()=>{
 const {redisStore,DETAIL_WRITE_LUA}=await import('../lib/product-metrics-store.mjs');const {aggregate,validateBatch,freshState}=await import('../lib/product-metrics.mjs');let command;
 const store=redisStore({env:{KV_REST_API_URL:'https://test.upstash.io',KV_REST_API_TOKEN:'test'},request:async(u,options)=>{command=JSON.parse(options.body);return new Response(JSON.stringify({result:1}));}});
 await store.write(aggregate(freshState(),validateBatch(payload()),now,{country:'PL'}).state,undefined,now);
 assert.equal(command[1],DETAIL_WRITE_LUA);assert.equal(command[2],3);assert.equal(command[5],'chanceme:product:v1:2026-10-30:records');
 assert.equal(JSON.parse(command[11])[0].details.score,1490);assert.equal(command[12],Date.parse('2026-11-29T00:00:00Z')/1000);
 assert.equal(JSON.parse(command[7]).records,undefined);
});
test('history bounds daily detail volume and ignores forged client time for sorting',async()=>{
 const {validateBatch,aggregate,freshState}=await import('../lib/product-metrics.mjs');let state=freshState();state.records=Array.from({length:1000},()=>({}));
 let out=aggregate(state,validateBatch(payload()),now);assert.equal(out.state.records.length,1000);assert.equal(out.state.days['2026-10-30'].detailCapped,true);
 const body=payload();body.events[0].occurred_at='2000-01-01T00:00:00Z';out=aggregate(freshState(),validateBatch(body),now);assert.equal(out.state.records[0].details.occurred_at,now.toISOString());
});
