/** Synthetic verification in an isolated, self-expiring QA namespace. No secrets printed. */
import assert from 'node:assert/strict';
import {randomUUID} from 'node:crypto';
import {redisStore,configured} from '../lib/product-metrics-store.mjs';
import {collect,validateBatch,summarize} from '../lib/product-metrics.mjs';
if(!configured())throw new Error('Approved storage environment unavailable');
const env=process.env,url=env.UPSTASH_REDIS_REST_URL||env.KV_REST_API_URL,token=env.UPSTASH_REDIS_REST_TOKEN||env.KV_REST_API_TOKEN;
const prefix='chanceme:qa:'+randomUUID()+':',now=new Date(),key=prefix+now.toISOString().slice(0,10);
const command=async args=>{
 const response=await fetch(url,{method:'POST',headers:{Authorization:`Bearer ${token}`,'Content-Type':'application/json'},body:JSON.stringify(args),signal:AbortSignal.timeout(5000)});
 if(!response.ok)throw new Error('QA storage request failed');const result=await response.json();if(result.error)throw new Error('QA storage command failed');return result.result;
};
const visit=randomUUID();
const event=()=>validateBatch({version:2,events:[{id:randomUUID(),visit_id:visit,occurred_at:now.toISOString(),page:'/',device:'desktop',score:1470,gpa:9.15,event:'calculator_completed',test_type:'SAT',program:'Management',round:'Winter',status:'estimated',score_band:'1450-1499',gpa_band:'9.00-9.24'}]});
const store=redisStore({prefix}),duplicate=event();
try{
 await collect(store,duplicate,{now,attempts:4});await collect(store,duplicate,{now,attempts:4});
 await Promise.all([collect(store,event(),{now,attempts:4}),collect(store,event(),{now,attempts:4})]);
 const state=await store.history(1,now);assert.equal(state.days[now.toISOString().slice(0,10)].events.calculator_completed,3);
 const raw=await command(['GET',key]);assert.equal(JSON.parse(raw).dedup,undefined);assert.ok(!raw.includes(duplicate[0].dedup));
 const ttl=await command(['TTL',key]),dedupTtl=await command(['TTL',key+':dedup']);assert.ok(ttl>89*86400&&ttl<=90*86400);assert.ok(dedupTtl>0&&dedupTtl<=2*86400);
 const journal=await store.detailedHistory(1,now);assert.equal(journal.length,3);assert.equal(journal[0].details.score,1470);assert.equal(journal[0].details.gpa,9.15);assert.equal(journal[0].details.visit_id,visit);assert.equal(journal[0].dedup,undefined);const journalTtl=await command(['TTL',key+':records']);assert.ok(journalTtl>29*86400&&journalTtl<=30*86400);
 const report=summarize(state,{now,days:1});assert.equal(report.totals.calculator_completed,3);assert.equal(report.distributions['calculator_completed:gpa_band']['9.00-9.24'],undefined);
 console.log(JSON.stringify({status:'PASS',isolated_qa_namespace:true,unique_actions:3,replay_counted_once:true,concurrent_updates_preserved:true,aggregate_ttl_seconds:ttl,dedup_ttl_seconds:dedupTtl,small_bins_hidden:true,exact_input_journal_verified:true,journal_ttl_seconds:journalTtl,production_counts_untouched:true}));
}finally{
 // Only our newly created synthetic QA keys; no production metrics or user data.
 await command(['EXPIRE',key,60]);await command(['EXPIRE',key+':dedup',60]);await command(['EXPIRE',key+':records',60]);
}
