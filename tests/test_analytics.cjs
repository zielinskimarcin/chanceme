const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/static/analytics.js','utf8');
const helpers=require('../src/static/analytics.js');
function boot({host='chance-me.com',path='/',gpc=false,dnt=false,legacy=null,legacyRaw=null,off=null,referrer='',cookies='',session=null,noSession=false,userAgent='',search='?score=1450',hash='#gpa=9',fetcher=()=>Promise.resolve({ok:true})}={}){
 const scripts=[],queue=[],writes=[],cookieWrites=[],requests=[],timers=new Map(),listeners={},documentListeners={},status={textContent:''};let timerId=0,eventId=0;
 const button={hidden:true,disabled:false,textContent:'',addEventListener:(name,fn)=>{button.click=fn;}};
 const sessionStorage=new Map(session||[]);
 const storage=new Map([['__va_attribution','{"userId":"old-private-id"}']]);if(legacyRaw!==null)storage.set('chance-consent-v1',legacyRaw);if(legacy!==null)storage.set('chance-consent-v1',JSON.stringify({analytics:legacy,expires:Date.now()+60000}));if(off!==null)storage.set('chance-anonymous-stats-off-v1',off);
 const document={referrer,addEventListener:(name,fn)=>documentListeners[name]=fn,head:{appendChild:s=>scripts.push(s.src)},createElement:type=>{assert.equal(type,'script','measurement must not create a banner');return{};},querySelectorAll:selector=>selector==='[data-analytics-toggle]'?[button]:selector==='[data-analytics-status]'?[status]:[]};
 Object.defineProperty(document,'cookie',{get:()=>cookies,set:v=>cookieWrites.push(v)});
 const window={document,setTimeout:(fn,ms)=>{const id=++timerId;timers.set(id,fn);window.lastTimerDelay=ms;return id;},clearTimeout:id=>timers.delete(id),addEventListener:(name,fn)=>listeners[name]=fn,crypto:{randomUUID:()=>('00000000-0000-4000-8000-'+String(++eventId).padStart(12,'0'))},fetch:(url,init)=>{requests.push({url,init});return fetcher(url,init);},sessionStorage:{getItem:key=>{if(noSession)throw new Error('storage blocked');return sessionStorage.get(key)??null;},setItem:(key,value)=>{if(noSession)throw new Error('storage blocked');sessionStorage.set(key,value);},removeItem:key=>sessionStorage.delete(key)},navigator:{userAgent,globalPrivacyControl:gpc,doNotTrack:dnt?'1':'0'},location:{hostname:host,origin:'https://'+host,pathname:path,search,hash,reload:()=>{window.reloaded=true;},assign:()=>{}},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>{writes.push([key,value]);storage.set(key,value);},removeItem:key=>storage.delete(key)},va:function(){queue.push([...arguments]);}};
 vm.runInNewContext(source,{window,Date,Set,JSON,Object,URL,URLSearchParams});
 return{scripts,queue,writes,cookieWrites,requests,timers,listeners,documentListeners,window,button,status,storage,sessionStorage,flush:()=>listeners.pagehide(),beforeSend:queue.find(e=>e[0]==='beforeSend')?.[1]};
}
test('sanitizers drop scores, grades, report fragments and arbitrary values',()=>{
 assert.deepEqual(helpers.cleanDimensions({score:1450,gpa:9,email:'private@example.com',round:'Winter',program:'Management',destination:'/report/#gpa=9'}),{program:'Management',round:'Winter'});
 assert.equal(helpers.safeLocation({origin:'https://chance-me.com',pathname:'/report/',search:'?score=1450',hash:'#gpa=9'}),'https://chance-me.com/report/');
});
test('a new production visit starts cookieless measurement without clicking or writing persistent local identifiers',()=>{
 const b=boot();assert.deepEqual(b.scripts,['/_vercel/insights/script.js']);assert.deepEqual(b.writes,[]);assert.deepEqual(b.cookieWrites,[]);assert.deepEqual(b.queue.map(e=>e[0]),['beforeSend']);assert.equal(b.window.ChanceAnalytics.allowed(),true);
});
test('GPC, DNT, explicit opt-out and preview do not load measurement',()=>{
 for(const options of [{gpc:true},{dnt:true},{off:'1'},{host:'preview.vercel.app'}]){const b=boot(options);assert.deepEqual(b.scripts,[]);assert.equal(b.window.ChanceAnalytics.allowed(),false);}
});
test('an old refusal is preserved and can be changed explicitly on the privacy page',()=>{
 const b=boot({legacy:false});assert.deepEqual(b.scripts,[]);assert.equal(b.storage.get('chance-anonymous-stats-off-v1'),'1');assert.equal(b.storage.has('chance-consent-v1'),false);b.button.click();assert.deepEqual(b.scripts,['/_vercel/insights/script.js']);assert.equal(b.storage.get('chance-anonymous-stats-off-v1'),'0');
});
test('an old acceptance never enables Google Analytics or session recording on any page',()=>{
 for(const path of ['/','/sat-tutoring/','/bocconi-gpa-calculator/','/report/','/bocconi-sat-score/'])assert.deepEqual(boot({path,legacy:true}).scripts,['/_vercel/insights/script.js']);
});
test('beforeSend strips query, hash, identity, flags and payload and rejects custom events',()=>{
 const b=boot({path:'/report/'});const result=b.beforeSend({type:'pageview',url:'https://chance-me.com/report/?email=private#score=1450',payload:{gpa:9},__cdp:{userId:'private'}});
 assert.equal(JSON.stringify(result),JSON.stringify({type:'pageview',url:'https://chance-me.com/report/'}));assert.equal(b.beforeSend({type:'event',payload:{name:'private',data:{score:1450}}}),null);
});
test('unknown page paths collapse to404 and unsafe referrer queries are not transmitted',()=>{
 assert.equal(boot({path:'/private/1450'}).beforeSend({type:'pageview'}).url,'https://chance-me.com/404/');
 assert.equal(boot({referrer:'https://other.example/?email=private'}).beforeSend({type:'pageview'}),null);
 assert.equal(boot({referrer:'https://other.example/'}).beforeSend({type:'pageview'}).url,'https://chance-me.com/');
});
test('turning statistics off suppresses subsequent events, stores only a persistent opt-out flag and stops the library',()=>{
 const b=boot();b.button.click();assert.equal(b.beforeSend({type:'pageview'}),null);assert.equal(b.window.reloaded,true);assert.equal(b.storage.get('chance-anonymous-stats-off-v1'),'1');assert.deepEqual(b.writes,[['chance-anonymous-stats-off-v1','1']]);assert.match(b.status.textContent,/are off/);
});
test('live privacy signals suppress already-loaded measurement',()=>{
 const b=boot();b.window.navigator.globalPrivacyControl=true;assert.equal(b.beforeSend({type:'pageview'}),null);
});
test('legacy GA and Clarity cookies are expired, while functional cookies are left intact',()=>{
 const b=boot({cookies:'_ga=old; _ga_TEST=old; _clck=old; _clsk=old; necessary=keep'});assert.equal(b.cookieWrites.length,12);assert.ok(b.cookieWrites.every(v=>v.includes('Max-Age=0')));assert.ok(b.cookieWrites.every(v=>!v.startsWith('necessary=')));
});
test('incomplete calculation hooks are rejected and paid Vercel events are never emitted',()=>{
 const b=boot();const count=b.queue.length;assert.equal(b.window.ChanceAnalytics.track('calculator_completed',{score:1450,gpa:9}),false);assert.equal(b.queue.length,count);assert.equal(b.scripts.length,1);
});

test('malformed legacy preferences cannot preserve optional identity data',()=>{const b=boot({legacyRaw:'{broken'});assert.equal(b.storage.has('__va_attribution'),false);assert.equal(b.storage.has('chance-consent-v1'),false);});

const profile={test_type:'SAT',program:'Finance',round:'Early',status:'estimated',result_band:'balanced',score_band:'1450-1499',gpa_band:'9.00-9.24'};
test('bands are coarse at boundaries and invalid inputs cannot become dimensions',()=>{
 assert.equal(helpers.scoreBand('SAT',1499),'1450-1499');assert.equal(helpers.scoreBand('SAT',1600),'1550-1600');assert.equal(helpers.scoreBand('SAT',399),undefined);
 assert.equal(helpers.scoreBand('Bocconi',33.999),'32-33.99');assert.equal(helpers.scoreBand('Bocconi',50),'48-50');assert.equal(helpers.gpaBand(9.249),'9.00-9.24');assert.equal(helpers.gpaBand(10),'9.75-10.00');assert.equal(helpers.gpaBand('9'),undefined);
 for(const extra of [{score:Infinity},{gpa:11},{email:'x@y.com'},{url:'/report/#score=1450'},{score_band:'1450'},{gpa_band:'9'},{test_type:'Bocconi',score_band:'1450-1499'}])assert.equal(helpers.cleanEvent('calculator_completed',{...profile,...extra}),null);
 assert.equal(helpers.cleanEvent('arbitrary',{}),null);assert.equal(helpers.cleanEvent('cta_clicked',{score_band:'1450-1499'}),null);
});
test('events are batched with independent short lived UUIDs and never paid Vercel events',async()=>{
 const b=boot();assert.equal(b.window.ChanceAnalytics.track('calculator_completed',profile),true);assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),true);
 assert.equal(b.requests.length,0);assert.equal(b.timers.size,1);b.flush();await Promise.resolve();
 assert.equal(b.requests.length,1);const request=b.requests[0],body=JSON.parse(request.init.body);assert.equal(request.url,'/api/product-events');assert.equal(request.init.credentials,'omit');assert.equal(request.init.referrerPolicy,'no-referrer');assert.equal(request.init.keepalive,true);
 assert.equal(body.version,2);assert.equal(body.events.length,3);assert.notEqual(body.events[0].id,body.events[1].id);assert.equal(body.events[1].score_band,'1450-1499');assert.equal(b.queue.length,1);assert.deepEqual(b.writes,[]);assert.deepEqual(b.cookieWrites,[]);
});
test('a full batch flushes once, repeated lifecycle flush does not replay IDs',async()=>{
 const b=boot();for(let i=0;i<15;i++)assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),true);assert.equal(b.requests.length,1);b.flush();b.flush();assert.equal(b.requests.length,1);assert.equal(JSON.parse(b.requests[0].init.body).events.length,16);assert.ok(b.requests[0].init.body.length<=8192);
});
test('failed requests are discarded without retries or blocking user actions',async()=>{
 const b=boot({fetcher:()=>Promise.reject(new Error('offline'))});assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),true);b.flush();await new Promise(setImmediate);assert.equal(b.requests.length,1);assert.equal(b.timers.size,0);b.flush();assert.equal(b.requests.length,1);
 const c=boot({fetcher:()=>{throw new Error('sync failure');}});assert.doesNotThrow(()=>{c.window.ChanceAnalytics.track('gpa_calculated');c.flush();});
});
test('slow requests leave a bounded queue without concurrent sends or retries',async()=>{
 const b=boot({fetcher:()=>new Promise(()=>{})});b.window.ChanceAnalytics.track('gpa_calculated');b.flush();for(let i=0;i<16;i++)assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),true);assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),false);assert.equal(b.requests.length,1);
});
test('privacy withdrawal clears queued events and timers, including live browser signals',()=>{
 const b=boot();b.window.ChanceAnalytics.track('gpa_calculated');b.button.click();assert.equal(b.timers.size,0);b.flush();assert.equal(b.requests.length,0);
 const c=boot();c.window.ChanceAnalytics.track('gpa_calculated');c.window.navigator.globalPrivacyControl=true;c.flush();assert.equal(c.requests.length,0);assert.equal(c.timers.size,0);
 for(const options of [{off:'1'},{host:'preview.vercel.app'},{dnt:true},{gpc:true}]){const x=boot(options);assert.equal(x.window.ChanceAnalytics.track('gpa_calculated'),false);x.flush();assert.equal(x.requests.length,0);}
});
test('hidden pages flush and raw query/hash never enter an event payload',()=>{
 const b=boot();b.window.ChanceAnalytics.track('calculator_completed',profile);b.window.document.visibilityState='hidden';b.documentListeners.visibilitychange();assert.equal(b.requests.length,1);assert.equal(b.requests[0].init.body.includes('#gpa'),false);assert.equal(b.requests[0].init.body.includes('?score'),false);
});

test('PDF generated and failed events accept only their fixed coarse schema',()=>{
 assert.deepEqual(helpers.cleanEvent('pdf_generated',{}),{event:'pdf_generated'});assert.deepEqual(helpers.cleanEvent('pdf_failed',{reason:'unavailable'}),{event:'pdf_failed',reason:'unavailable'});
 assert.equal(helpers.cleanEvent('pdf_generated',{filename:'private.pdf'}),null);assert.equal(helpers.cleanEvent('pdf_failed',{reason:'raw server exception'}),null);
});

test('maximum scores share a broad band with nearby values and singleton bands are rejected',()=>{
 assert.equal(helpers.scoreBand('SAT',1550),helpers.scoreBand('SAT',1600));assert.equal(helpers.scoreBand('Bocconi',48),helpers.scoreBand('Bocconi',50));assert.equal(helpers.gpaBand(9.75),helpers.gpaBand(10));
 for(const extra of [{score_band:'1600'},{test_type:'Bocconi',score_band:'50'},{gpa_band:'10'}])assert.equal(helpers.cleanEvent('calculator_completed',{...profile,...extra}),null);
});
test('frontend events and band boundaries conform to the actual server validator',async()=>{
 const {validateBatch,EVENT_DIMENSIONS}=await import('../lib/product-metrics.mjs');
 const samples={page_view:{},calculator_completed:{...profile,score:1450,gpa:9.123,probability:54.5,gpa_estimated:true,gpa_confirmed:true,cycle:'27-28',applicant_category:'international',result:{probability:.545,display_percent:'55%',status:'estimated'}},calculator_failed:{reason:'unavailable'},gpa_calculated:{},gpa_transferred:{destination:'calculator'},report_rendered:{},report_failed:{reason:'unavailable'},pdf_generated:{},pdf_failed:{reason:'unavailable'},pdf_download_initiated:{destination:'pdf'},cta_clicked:{destination:'booking',position:'content'},booking_confirmed:{destination:'booking'}};
 assert.deepEqual(Object.keys({...samples,page_view:{}}).sort(),Object.keys(EVENT_DIMENSIONS).sort());
 const check=event=>validateBatch({version:2,events:[{id:'00000000-0000-4000-8000-000000000001',visit_id:'00000000-0000-4000-8000-000000000002',occurred_at:new Date().toISOString(),page:'/',referrer:'',device:'desktop',browser:'Chrome',os:'macOS',timezone:'America/Denver',utc_offset_minutes:360,content:'button-a',term:'bocconi',utm_id:'campaign-1',...event}]});
 for(const [name,dims] of Object.entries(samples)){const event=helpers.cleanEvent(name,dims);assert.ok(event);assert.doesNotThrow(()=>check(event));}
 for(const kind of ['SAT','Bocconi'])for(const score of kind==='SAT'?[400,449,450,1549,1550,1599,1600]:[0,1.999,2,47.999,48,49.999,50])for(const gpa of [0,.249,.25,9.749,9.75,9.999,10]){
  const event=helpers.cleanEvent('calculator_completed',{...profile,test_type:kind,score_band:helpers.scoreBand(kind,score),gpa_band:helpers.gpaBand(gpa)});assert.ok(event);assert.doesNotThrow(()=>check(event));
 }
});

test('visit metadata is tab-scoped, sanitized, and includes one initial page view',()=>{
 const b=boot({referrer:'https://example.com/path?email=secret#private',search:'?utm_source=google&utm_medium=cpc&utm_campaign=SAT-2026&email=private',userAgent:'iPhone Mobile'});b.flush();
 const event=JSON.parse(b.requests[0].init.body).events[0];assert.equal(event.event,'page_view');assert.equal(event.page,'/');assert.equal(event.referrer,'https://example.com/path');assert.equal(event.device,'mobile');assert.equal(event.source,'google');assert.equal(event.campaign,'SAT-2026');assert.ok(Number.isFinite(Date.parse(event.occurred_at)));assert.equal(event.visit_id,b.sessionStorage.get('chanceme.product-visit.v2'));assert.ok(!b.requests[0].init.body.includes('secret'));
 const c=boot({session:[...b.sessionStorage]});c.flush();assert.equal(JSON.parse(c.requests[0].init.body).events[0].visit_id,event.visit_id);
 b.button.click();assert.equal(b.sessionStorage.has('chanceme.product-visit.v2'),false);
 assert.equal(helpers.campaignValue('email@example.com'),undefined);assert.equal(helpers.campaignValue('x'.repeat(81)),undefined);assert.equal(helpers.cleanReferrer('javascript:secret'),'');
});
test('exact submitted inputs are permitted only on completed calculations',()=>{
 assert.ok(helpers.cleanEvent('calculator_completed',{...profile,score:1450,gpa:9.123,probability:54.5}));
 for(const dims of [{score:Infinity},{gpa:-1},{probability:101}])assert.equal(helpers.cleanEvent('calculator_completed',{...profile,...dims}),null);
 assert.equal(helpers.cleanEvent('report_rendered',{score:1450}),null);assert.equal(helpers.cleanEvent('page_view',{email:'private@example.com'}),null);
});

test('blocked session storage falls back to page memory without interrupting usage',()=>{
 const b=boot({noSession:true});assert.equal(b.window.ChanceAnalytics.track('gpa_calculated'),true);b.flush();
 const events=JSON.parse(b.requests[0].init.body).events;assert.equal(events.length,2);assert.equal(events[0].visit_id,events[1].visit_id);assert.notEqual(events[0].id,events[1].id);assert.equal(b.sessionStorage.size,0);
});

test('new-tab report continuity uses only an allowed UUID fragment before its first page view',()=>{
 const carried='12345678-1234-4123-8123-123456789abc';
 const b=boot({path:'/report/',hash:'#score=1450&gpa=9&visit_id='+carried});assert.equal(b.window.ChanceAnalytics.visitId(),carried);b.flush();
 const payload=JSON.parse(b.requests[0].init.body);assert.equal(payload.events[0].event,'page_view');assert.equal(payload.events[0].visit_id,carried);assert.equal(payload.events[0].page,'/report/');assert.equal(b.sessionStorage.get('chanceme.product-visit.v2'),carried);
 assert.equal(b.beforeSend({type:'pageview'}).url,'https://chance-me.com/report/');assert.equal(b.beforeSend({type:'pageview'}).url.includes(carried),false);
 const ordinary=boot({path:'/',hash:'#visit_id='+carried});assert.notEqual(ordinary.window.ChanceAnalytics.visitId(),carried);
 const invalid=boot({path:'/report/',hash:'#visit_id=private@example.com'});assert.notEqual(invalid.window.ChanceAnalytics.visitId(),'private@example.com');
 for(const options of [{off:'1'},{gpc:true},{dnt:true},{host:'preview.vercel.app'}]){const x=boot({...options,path:'/report/',hash:'#visit_id='+carried});assert.equal(x.window.ChanceAnalytics.visitId(),null);assert.equal(x.sessionStorage.size,0);}
});

test('browser and operating system families use fixed enums without sending full UA',()=>{
 for(const [ua,browser,os] of [['Mozilla Chrome/130 Safari/537 Edg/130 Windows','Edge','Windows'],['Mozilla Chrome/130 Safari/537 OPR/100 Linux','Opera','Linux'],['Android Chrome/130 SamsungBrowser/24','Samsung Internet','Android'],['iPhone FxiOS/120 Safari/605','Firefox','iOS'],['Macintosh Version/18 Safari/605','Safari','macOS'],['CrOS Chrome/130','Chrome','ChromeOS'],['unknown','Other','Other']])assert.deepEqual(helpers.browserInfo(ua),{browser,os});
 const b=boot({userAgent:'Windows Chrome/130 private-agent',search:'?utm_source=owner-qa&utm_content=button-a&utm_term=bocconi&utm_id=campaign-1'});assert.equal(b.window.lastTimerDelay,5000);b.flush();
 const event=JSON.parse(b.requests[0].init.body).events[0];assert.equal(event.browser,'Chrome');assert.equal(event.os,'Windows');assert.equal(event.source,'owner-qa');assert.equal(event.content,'button-a');assert.equal(event.term,'bocconi');assert.equal(event.utm_id,'campaign-1');assert.ok(event.timezone.length<=80);assert.ok(Number.isInteger(event.utc_offset_minutes));assert.equal(b.requests[0].init.body.includes('private-agent'),false);
});
test('completed calculation detail flags and model result have a strict schema',()=>{
 const detail={...profile,gpa_estimated:true,gpa_confirmed:true,cycle:'27-28',applicant_category:'international',result:{probability:.54,display_percent:'54%',status:'estimated'}};assert.ok(helpers.cleanEvent('calculator_completed',detail));
 for(const extra of [{gpa_estimated:'true'},{cycle:'26-27'},{applicant_category:'private'},{result:{probability:1.1,display_percent:'54%',status:'estimated'}},{result:{probability:.54,display_percent:'private',status:'estimated'}},{result:{probability:.54,display_percent:'54%',status:'estimated',email:'private'}}])assert.equal(helpers.cleanEvent('calculator_completed',{...detail,...extra}),null);
});
