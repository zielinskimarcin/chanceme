/* Cookieless, aggregate page views through existing Vercel Web Analytics only. */
(function(root){
 'use strict';
 const paths=new Set(['/','/bocconi-gpa-calculator/','/sat-tutoring/','/bocconi-sat-score/','/bocconi-test-score/','/report/','/privacy/','/gpa-school-sources/']);
 const dimensions={test_type:['SAT','Bocconi'],program:['Management','Finance','Management and Computer Science','Management for the Arts and Cultures','International Politics and Government','Mathematical and Computing Sciences for Artificial Intelligence'],round:['Early','Winter'],status:['estimated','ineligible','invalid','unsupported'],result_band:['lower','balanced','stronger'],position:['header','hero','content','footer'],destination:['calculator','gpa','lessons','booking','whatsapp','sat_guide','bocconi_test_guide','report','pdf'],reason:['unavailable'],guide:['bocconi_sat_score','bocconi_test_score']};
 function cleanPath(path){return paths.has(path)?path:'/404/';}
 function cleanDimensions(input){const out={};for(const [key,allowed] of Object.entries(dimensions))if(allowed.includes(input?.[key]))out[key]=input[key];return out;}
 function safeLocation(location){return location.origin+cleanPath(location.pathname);}
 const helpers={cleanDimensions,cleanPath,safeLocation};
 if(typeof module==='object'&&module.exports){module.exports=helpers;return;}
 if(!root.document)return;
 const doc=root.document,KEY='chance-anonymous-stats-off-v1',LEGACY='chance-consent-v1';
 const production=['chance-me.com','www.chance-me.com'].includes(root.location.hostname);
 let optedOut=false,loaded=false;
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
  try{root.localStorage.setItem(KEY,value?'1':'0');}catch{}
  renderSettings();
  if(!value)start();
  // Stop the already-loaded library after withdrawal; the calculator draft stays intact.
  else if(loaded)root.location.reload();
 }
 // Existing calculator/booking hooks remain compatible. The current Hobby plan has
 // page views only: do not emit paid custom events or send them to a cookie-based service.
 root.ChanceAnalytics={track:()=>false,allowed,showSettings:()=>root.location.assign('/privacy/#analytics')};
 doc.querySelectorAll('[data-analytics-toggle]').forEach(button=>button.addEventListener('click',()=>setOptOut(!optedOut)));
 clearLegacyCookies();renderSettings();start();
})(typeof window!=='undefined'?window:globalThis);
