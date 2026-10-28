/* Existing measurement destinations, explicit consent, and coarse events only. */
(function(root){
 'use strict';
 const GA='G-TQX27XKE8K',CLARITY='vx5oj467o6',KEY='chance-consent-v1';
 const paths=new Set(['/','/bocconi-gpa-calculator/','/sat-tutoring/','/bocconi-sat-score/','/bocconi-test-score/','/report/','/privacy/','/gpa-school-sources/']);
 const events=new Set(['page_view','calculator_started','calculator_submitted','calculator_completed','calculator_failed','gpa_started','gpa_calculated','gpa_transferred','cta_clicked','guide_opened','guide_cta_clicked','booking_cta_click','booking_submitted','whatsapp_click','pdf_download','full_report_opened']);
 const dimensions={test_type:['SAT','Bocconi'],program:['Management','Finance','Management and Computer Science','Management for the Arts and Cultures','International Politics and Government','Mathematical and Computing Sciences for Artificial Intelligence'],round:['Early','Winter'],status:['estimated','ineligible','invalid','unsupported'],result_band:['lower','balanced','stronger'],position:['header','hero','content','footer'],destination:['calculator','gpa','lessons','booking','whatsapp','sat_guide','bocconi_test_guide','report','pdf'],reason:['unavailable'],guide:['bocconi_sat_score','bocconi_test_score']};
 function cleanPath(path){return paths.has(path)?path:'/404/';}
 function cleanDimensions(input){const out={};for(const [key,allowed] of Object.entries(dimensions))if(allowed.includes(input?.[key]))out[key]=input[key];return out;}
 function safeLocation(location){return location.origin+cleanPath(location.pathname);}
 const helpers={cleanDimensions,cleanPath,safeLocation};
 if(typeof module==='object'&&module.exports){module.exports=helpers;return;}
 if(!root.document)return;
 const doc=root.document,production=['chance-me.com','www.chance-me.com'].includes(root.location.hostname);
 let choice=null,loaded=false,pageSent=false,notice=null,started=false,gpaStarted=false;
 const blocked=()=>root.navigator.globalPrivacyControl===true||root.navigator.doNotTrack==='1'||root.doNotTrack==='1';
 const allowed=()=>choice===true&&!blocked();
 try{const saved=JSON.parse(root.localStorage.getItem(KEY));if(saved&&saved.expires>Date.now()&&typeof saved.analytics==='boolean')choice=saved.analytics;}catch{}
 function script(src){const s=doc.createElement('script');s.src=src;s.async=true;doc.head.appendChild(s);}
 function consent(state){return {analytics_storage:state?'granted':'denied',ad_storage:'denied',ad_user_data:'denied',ad_personalization:'denied'};}
 function track(name,parameters={}){
  if(!allowed()||!production||!events.has(name))return;
  const params=cleanDimensions(parameters);
  root.gtag('event',name,{...params,page_location:safeLocation(root.location),page_path:cleanPath(root.location.pathname),page_referrer:'',send_to:GA});
 }
 function start(){
  if(!allowed()||!production)return;
  if(!loaded){
   loaded=true;root.dataLayer=root.dataLayer||[];root.gtag=root.gtag||function(){root.dataLayer.push(arguments);};
   root.gtag('consent','default',consent(false));root.gtag('js',new Date());
   root.gtag('set',{page_location:safeLocation(root.location),page_referrer:'',allow_google_signals:false,allow_ad_personalization_signals:false});
   root.gtag('consent','update',consent(true));
   root.gtag('config',GA,{send_page_view:false,page_location:safeLocation(root.location),page_referrer:'',allow_google_signals:false,allow_ad_personalization_signals:false});
   script('https://www.googletagmanager.com/gtag/js?id='+GA);
   root.va=root.va||function(){(root.vaq=root.vaq||[]).push(arguments);};
   root.va('beforeSend',event=>allowed()?{...event,url:safeLocation(root.location)}:null);
   script('/_vercel/insights/script.js');
   // Personalised tools and links with query/fragment data are never session-recorded.
   if(['/sat-tutoring/','/bocconi-sat-score/','/bocconi-test-score/'].includes(root.location.pathname)&&!root.location.search&&!root.location.hash){
    root.clarity=root.clarity||function(){(root.clarity.q=root.clarity.q||[]).push(arguments);};
    root.clarity('consentv2',{ad_Storage:'denied',analytics_Storage:'granted'});script('https://www.clarity.ms/tag/'+CLARITY);
   }
  }
  root['ga-disable-'+GA]=false;
  root.gtag('consent','update',consent(true));
  if(!pageSent){pageSent=true;track('page_view');const guide=guideName();if(guide)track('guide_opened',{guide});}
 }
 function guideName(){return root.location.pathname==='/bocconi-sat-score/'?'bocconi_sat_score':root.location.pathname==='/bocconi-test-score/'?'bocconi_test_score':null;}
 function setChoice(value){
  const wasAllowed=allowed();choice=value;
  try{root.localStorage.setItem(KEY,JSON.stringify({analytics:value,expires:Date.now()+180*86400000}));}catch{}
  if(notice){notice.remove();notice=null;}
  if(allowed())start();
  else{
   root['ga-disable-'+GA]=true;
   if(root.gtag)root.gtag('consent','update',consent(false));
   if(root.clarity)root.clarity('consentv2',{ad_Storage:'denied',analytics_Storage:'denied'});
   for(const cookie of doc.cookie.split(';')){const name=cookie.split('=')[0].trim();if(!/^(_ga|_gid|_gat|_clck|_clsk)/.test(name))continue;for(const domain of ['',root.location.hostname,'.chance-me.com'])doc.cookie=name+'=; Max-Age=0; Path=/'+(domain?'; Domain='+domain:'');}
   // Stop already-loaded third-party libraries after withdrawal. The local draft survives reload.
   if(wasAllowed&&loaded)root.location.reload();
  }
 }
 function showSettings(){
  if(notice)return;
  notice=doc.createElement('section');notice.className='cookie-notice';notice.setAttribute('data-nosnippet','');notice.setAttribute('aria-label','Analytics preferences');
  const title=doc.createElement('h2');title.textContent='Analytics cookies';
  const text=doc.createElement('p');text.append('We use optional analytics to improve Chance Me. Your scores and grades aren’t included. ');
  const link=doc.createElement('a');link.href='/privacy/';link.textContent='Privacy';text.append(link);
  const actions=doc.createElement('div');actions.className='cookie-actions';
  for(const [label,value] of [['Accept',true],['Reject',false]]){const button=doc.createElement('button');button.type='button';button.textContent=label;button.addEventListener('click',()=>setChoice(value));actions.append(button);}
  notice.append(title,text);
  if(blocked()){const status=doc.createElement('p');status.className='cookie-status';status.textContent='Your browser’s privacy signal disables analytics.';notice.append(status);}
  notice.append(actions);doc.body.append(notice);
 }
 root.ChanceAnalytics={track,showSettings,allowed};
 doc.querySelectorAll('[data-cookie-settings]').forEach(button=>button.addEventListener('click',showSettings));
 const form=doc.getElementById('profile-form');
 if(form)form.addEventListener('input',()=>{if(!started){started=true;track('calculator_started');}},{passive:true});
 const gpa=doc.getElementById('gpa-form');
 if(gpa)gpa.addEventListener('input',()=>{if(!gpaStarted){gpaStarted=true;track('gpa_started');}},{passive:true});
 doc.addEventListener('click',event=>{
  const a=event.target.closest('a,button');if(!a)return;
  const href=a.getAttribute('href')||'';
  let destination=href==='#book-lesson'?'booking':href.startsWith('https://wa.me/')?'whatsapp':href.startsWith('/sat-tutoring/')?'lessons':href.startsWith('/bocconi-gpa-calculator/')?'gpa':href.startsWith('/bocconi-sat-score/')?'sat_guide':href.startsWith('/bocconi-test-score/')?'bocconi_test_guide':href.startsWith('/report/')?'report':null;
  if(!destination)return;
  const position=a.closest('header')?'header':a.closest('footer')?'footer':a.closest('.sat-hero,.hero')?'hero':'content';
  track('cta_clicked',{position,destination});
  if(destination==='booking')track('booking_cta_click',{position,destination});
  if(destination==='whatsapp')track('whatsapp_click',{position,destination});
  if(destination==='report')track('full_report_opened',{position,destination});
  const guide=guideName();if(guide)track('guide_cta_clicked',{guide,position,destination});
 });
 start();if(choice===null&&!blocked())showSettings();
})(typeof window!=='undefined'?window:globalThis);
