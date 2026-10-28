const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('src/static/analytics.js','utf8');
const helpers=require('../src/static/analytics.js');
function boot({host='chance-me.com',path='/',gpc=false,dnt=false,legacy=null,legacyRaw=null,off=null,referrer='',cookies=''}={}){
 const scripts=[],queue=[],writes=[],cookieWrites=[],status={textContent:''};
 const button={hidden:true,disabled:false,textContent:'',addEventListener:(name,fn)=>{button.click=fn;}};
 const storage=new Map([['__va_attribution','{"userId":"old-private-id"}']]);if(legacyRaw!==null)storage.set('chance-consent-v1',legacyRaw);if(legacy!==null)storage.set('chance-consent-v1',JSON.stringify({analytics:legacy,expires:Date.now()+60000}));if(off!==null)storage.set('chance-anonymous-stats-off-v1',off);
 const document={referrer,head:{appendChild:s=>scripts.push(s.src)},createElement:type=>{assert.equal(type,'script','measurement must not create a banner');return{};},querySelectorAll:selector=>selector==='[data-analytics-toggle]'?[button]:selector==='[data-analytics-status]'?[status]:[]};
 Object.defineProperty(document,'cookie',{get:()=>cookies,set:v=>cookieWrites.push(v)});
 const window={document,navigator:{globalPrivacyControl:gpc,doNotTrack:dnt?'1':'0'},location:{hostname:host,origin:'https://'+host,pathname:path,search:'?score=1450',hash:'#gpa=9',reload:()=>{window.reloaded=true;},assign:()=>{}},localStorage:{getItem:key=>storage.get(key)??null,setItem:(key,value)=>{writes.push([key,value]);storage.set(key,value);},removeItem:key=>storage.delete(key)},va:function(){queue.push([...arguments]);}};
 vm.runInNewContext(source,{window,Date,Set,JSON,Object,URL});
 return{scripts,queue,writes,cookieWrites,window,button,status,storage,beforeSend:queue.find(e=>e[0]==='beforeSend')?.[1]};
}
test('sanitizers drop scores, grades, report fragments and arbitrary values',()=>{
 assert.deepEqual(helpers.cleanDimensions({score:1450,gpa:9,email:'private@example.com',round:'Winter',program:'Management',destination:'/report/#gpa=9'}),{program:'Management',round:'Winter'});
 assert.equal(helpers.safeLocation({origin:'https://chance-me.com',pathname:'/report/',search:'?score=1450',hash:'#gpa=9'}),'https://chance-me.com/report/');
});
test('a new production visit starts only cookieless Vercel without clicking or writing an identifier',()=>{
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
test('turning statistics off suppresses subsequent events, stores only an opt-out flag and stops the library',()=>{
 const b=boot();b.button.click();assert.equal(b.beforeSend({type:'pageview'}),null);assert.equal(b.window.reloaded,true);assert.equal(b.storage.get('chance-anonymous-stats-off-v1'),'1');assert.deepEqual(b.writes,[['chance-anonymous-stats-off-v1','1']]);assert.match(b.status.textContent,/are off/);
});
test('live privacy signals suppress already-loaded measurement',()=>{
 const b=boot();b.window.navigator.globalPrivacyControl=true;assert.equal(b.beforeSend({type:'pageview'}),null);
});
test('legacy GA and Clarity cookies are expired, while functional cookies are left intact',()=>{
 const b=boot({cookies:'_ga=old; _ga_TEST=old; _clck=old; _clsk=old; necessary=keep'});assert.equal(b.cookieWrites.length,12);assert.ok(b.cookieWrites.every(v=>v.includes('Max-Age=0')));assert.ok(b.cookieWrites.every(v=>!v.startsWith('necessary=')));
});
test('compatibility hooks do not send paid custom events or raw calculator inputs',()=>{
 const b=boot();const count=b.queue.length;assert.equal(b.window.ChanceAnalytics.track('calculator_completed',{score:1450,gpa:9}),false);assert.equal(b.queue.length,count);assert.equal(b.scripts.length,1);
});

test('malformed legacy preferences cannot preserve optional identity data',()=>{const b=boot({legacyRaw:'{broken'});assert.equal(b.storage.has('__va_attribution'),false);assert.equal(b.storage.has('chance-consent-v1'),false);});
