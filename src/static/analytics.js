/* Cookieless page views and bounded first-party aggregate product events. */
(function(root){
 'use strict';
 const paths=new Set(['/','/bocconi-gpa-calculator/','/sat-tutoring/','/bocconi-sat-score/','/bocconi-test-score/','/report/','/privacy/','/gpa-school-sources/']);
 const dimensions={test_type:['SAT','Bocconi'],program:['Management','Finance','Management and Computer Science','Management for the Arts and Cultures','International Politics and Government','Mathematical and Computing Sciences for Artificial Intelligence'],round:['Early','Winter'],status:['estimated','ineligible','invalid','unsupported'],result_band:['lower','balanced','stronger'],position:['header','hero','content','footer'],destination:['calculator','gpa','lessons','booking','whatsapp','sat_guide','bocconi_test_guide','report','pdf'],reason:['unavailable'],guide:['bocconi_sat_score','bocconi_test_score']};
 function cleanPath(path){return paths.has(path)?path:'/404/';}
 function cleanDimensions(input){const out={};for(const [key,allowed] of Object.entries(dimensions))if(allowed.includes(input?.[key]))out[key]=input[key];return out;}
 function safeLocation(location){return location.origin+cleanPath(location.pathname);}
 const eventDimensions={calculator_completed:['test_type','program','round','status','result_band','score_band','gpa_band'],calculator_failed:['reason'],gpa_calculated:[],gpa_transferred:['destination'],report_rendered:[],report_failed:['reason'],pdf_generated:[],pdf_failed:['reason'],pdf_download_initiated:['destination'],cta_clicked:['destination','position'],booking_confirmed:['destination']};
 function scoreBand(kind,value){
  if(typeof value!=='number'||!Number.isFinite(value))return undefined;
  if(kind==='SAT'){if(value<400||value>1600)return undefined;return value>=1550?'1550-1600':Math.floor(value/50)*50+'-'+(Math.floor(value/50)*50+49);}
  if(kind==='Bocconi'){if(value<0||value>50)return undefined;return value>=48?'48-50':Math.floor(value/2)*2+'-'+(Math.floor(value/2)*2+1)+'.99';}
 }
 function gpaBand(value){if(typeof value!=='number'||!Number.isFinite(value)||value<0||value>10)return undefined;if(value>=9.75)return '9.75-10.00';const lo=Math.floor(value*4)/4;return lo.toFixed(2)+'-'+(lo+.24).toFixed(2);}
 const satBands=new Set(Array.from({length:23},(_,i)=>(400+i*50)+'-'+(449+i*50)).concat('1550-1600'));
 const btBands=new Set(Array.from({length:24},(_,i)=>(i*2)+'-'+(i*2+1)+'.99').concat('48-50'));
 const gpaBands=new Set(Array.from({length:39},(_,i)=>(i/4).toFixed(2)+'-'+(i/4+.24).toFixed(2)).concat('9.75-10.00'));
 function cleanEvent(name,input={}){
  if(!Object.prototype.hasOwnProperty.call(eventDimensions,name))return null;
  const keys=eventDimensions[name];if(!input||typeof input!=='object'||Array.isArray(input))return null;
  // Unknown fields reject the whole event, rather than risk accidental raw-profile collection.
  if(Object.keys(input).some(key=>!keys.includes(key)))return null;
  const output={};
  for(const key of Object.keys(input)){
   const value=input[key];
   const valid=key==='score_band'?(input.test_type==='SAT'?satBands:input.test_type==='Bocconi'?btBands:new Set()).has(value):key==='gpa_band'?gpaBands.has(value):dimensions[key]?.includes(value);
   if(!valid)return null;output[key]=value;
  }
  if(name==='calculator_completed'&&!['test_type','program','round','status'].every(key=>key in output))return null;
  return {event:name,...output};
 }
 const helpers={cleanDimensions,cleanPath,safeLocation,scoreBand,gpaBand,cleanEvent};
 if(typeof module==='object'&&module.exports){module.exports=helpers;return;}
 if(!root.document)return;
 const doc=root.document,KEY='chance-anonymous-stats-off-v1',LEGACY='chance-consent-v1';
 const production=['chance-me.com','www.chance-me.com'].includes(root.location.hostname);
 let optedOut=false,loaded=false,events=[],timer=null,inflight=false;
 const blocked=()=>root.navigator.globalPrivacyControl===true||root.navigator.doNotTrack==='1'||root.doNotTrack==='1';
 const allowed=()=>production&&!optedOut&&!blocked();
 try{
  const saved=root.localStorage.getItem(KEY);
  optedOut=saved==='1';
  // Preserve a previous explicit refusal instead of treating the migration as acceptance.
  if(saved===null){
   const previous=JSON.parse(root.localStorage.getItem(LEGACY));
   if(previous?.expires>Date.now()&&previous.analytics===false){optedOut=true;root.localStorage.setItem(KEY,'1');}
  }
 }catch{}
 // Cleanup is independent of parsing an old or malformed preference.
 try{root.localStorage.removeItem(LEGACY);root.localStorage.removeItem('__va_attribution');}catch{}
 function clearLegacyCookies(){
  if(!production)return;
  try{
   for(const cookie of doc.cookie.split(';')){
    const name=cookie.split('=')[0].trim();
    if(!/^(_ga(?:_|$)|_gid$|_gat(?:_|$)|_clck$|_clsk$)/.test(name))continue;
    for(const domain of ['',root.location.hostname,'.chance-me.com'])doc.cookie=name+'=; Max-Age=0; Path=/'+(domain?'; Domain='+domain:'');
   }
  }catch{}
 }
 function beforeSend(event){
  if(!allowed()||event.type!=='pageview')return null;
  // Drop unexpected external referrer query data rather than forwarding it.
  try{if(doc.referrer){const ref=new URL(doc.referrer);if(ref.search||ref.hash||ref.username||ref.password)return null;}}catch{return null;}
  // Return only the safe page URL: no profile fragment, query, payload, identity or flags.
  return {type:'pageview',url:safeLocation(root.location)};
 }
 function start(){
  if(!allowed()||loaded)return;
  loaded=true;
  root.va=root.va||function(){(root.vaq=root.vaq||[]).push(arguments);};
  root.va('beforeSend',beforeSend);
  const script=doc.createElement('script');script.src='/_vercel/insights/script.js';script.async=true;
  doc.head.appendChild(script);
 }
 function renderSettings(){
  const message=!production?'Statistics are disabled in this preview.':blocked()?'Your browser’s privacy signal disables statistics.':optedOut?'Anonymous visit statistics are off in this browser.':'Anonymous visit statistics are on in this browser.';
  doc.querySelectorAll('[data-analytics-status]').forEach(node=>{node.textContent=message;});
  doc.querySelectorAll('[data-analytics-toggle]').forEach(button=>{
   button.hidden=false;button.disabled=!production||blocked();
   button.textContent=optedOut?'Enable anonymous statistics':'Turn off anonymous statistics';
  });
 }
 function setOptOut(value){
  optedOut=value;
  if(value)clearEvents();
  try{root.localStorage.setItem(KEY,value?'1':'0');}catch{}
  renderSettings();
  if(!value)start();
  // Stop the already-loaded library after withdrawal; the calculator draft stays intact.
  else if(loaded)root.location.reload();
 }
 function clearEvents(){events=[];if(timer!==null){root.clearTimeout(timer);timer=null;}}
 function schedule(){if(timer===null&&events.length)timer=root.setTimeout(flush,30000);}
 function flush(){
  if(timer!==null){root.clearTimeout(timer);timer=null;}
  if(!allowed()){clearEvents();return;}
  if(inflight||!events.length)return;
  const batch=events;events=[];
  const body=JSON.stringify({version:1,events:batch});
  if(body.length>8192)return;
  inflight=true;
  // Single attempt: no replay queue, identifiers on disk, or background retry storm.
  try{Promise.resolve(root.fetch('/api/product-events',{method:'POST',headers:{'Content-Type':'application/json'},body,credentials:'omit',referrerPolicy:'no-referrer',keepalive:true})).catch(()=>{}).finally(()=>{inflight=false;if(allowed())schedule();else clearEvents();});}catch{inflight=false;}
 }
 function track(name,input){
  try{
   if(!allowed()){clearEvents();return false;}
   const event=cleanEvent(name,input);if(!event||events.length>=16||!root.crypto?.randomUUID)return false;
   const item={id:root.crypto.randomUUID(),...event};
   if(JSON.stringify({version:1,events:events.concat(item)}).length>8192)return false;
   events.push(item);schedule();if(events.length===16)flush();return true;
  }catch{return false;}
 }
 root.ChanceAnalytics={track,allowed,scoreBand,gpaBand,showSettings:()=>root.location.assign('/privacy/#analytics')};
 root.addEventListener?.('pagehide',flush);
 doc.addEventListener?.('visibilitychange',()=>{if(doc.visibilityState==='hidden')flush();});
 doc.addEventListener?.('click',event=>{
  const link=event.target.closest?.('a[href]');if(!link)return;
  let url;try{url=new URL(link.href,root.location.origin);}catch{return;}
  const local=url.origin===root.location.origin;
  const destination=local?({'/':'calculator','/bocconi-gpa-calculator/':'gpa','/sat-tutoring/':'lessons','/bocconi-sat-score/':'sat_guide','/bocconi-test-score/':'bocconi_test_guide','/report/':'report'}[url.pathname]):url.hostname==='wa.me'?'whatsapp':url.hostname==='cal.com'?'booking':null;
  // An in-page booking CTA is also meaningful; unrelated anchor navigation is not.
  const resolved=local&&['#booking','#book-lesson'].includes(url.hash)?'booking':destination;
  if(!resolved||(local&&url.pathname===root.location.pathname&&!['#booking','#book-lesson'].includes(url.hash)))return;
  const position=link.closest('header')?'header':link.closest('footer')?'footer':link.closest('.hero')?'hero':'content';
  track('cta_clicked',{destination:resolved,position});
 });
 doc.querySelectorAll('[data-analytics-toggle]').forEach(button=>button.addEventListener('click',()=>setOptOut(!optedOut)));
 clearLegacyCookies();renderSettings();start();
})(typeof window!=='undefined'?window:globalThis);
