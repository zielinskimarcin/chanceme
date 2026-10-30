(function(){
  'use strict';
  const $=id=>document.getElementById(id),menu=$('menu-toggle'),nav=$('main-nav'),guides=$('guides-toggle'),panel=$('guides-panel');
  function closeGuides(){panel.hidden=true;guides.setAttribute('aria-expanded','false');}
  function closeMenu(){nav.classList.remove('is-open');menu.setAttribute('aria-expanded','false');menu.setAttribute('aria-label','Open navigation');closeGuides();}
  menu.addEventListener('click',()=>{const open=menu.getAttribute('aria-expanded')!=='true';nav.classList.toggle('is-open',open);menu.setAttribute('aria-expanded',String(open));menu.setAttribute('aria-label',open?'Close navigation':'Open navigation');});
  guides.addEventListener('click',()=>{const open=guides.getAttribute('aria-expanded')!=='true';guides.setAttribute('aria-expanded',String(open));panel.hidden=!open;});
  guides.addEventListener('keydown',e=>{if(e.key==='ArrowDown'){e.preventDefault();guides.setAttribute('aria-expanded','true');panel.hidden=false;panel.querySelector('a').focus();}});
  nav.addEventListener('click',e=>{if(e.target.closest('a'))closeMenu();});
  document.addEventListener('click',e=>{if(!$('guides-dropdown').contains(e.target))closeGuides();});
  document.addEventListener('keydown',e=>{if(e.key!=='Escape')return;if(guides.getAttribute('aria-expanded')==='true'){closeGuides();guides.focus();}else if(menu.getAttribute('aria-expanded')==='true'){closeMenu();menu.focus();}});
  const dialog=document.querySelector('.sat-screen-dialog'),image=dialog.querySelector('img');
  document.querySelectorAll('[data-screen]').forEach(button=>button.addEventListener('click',()=>{image.src=button.dataset.screen;image.alt=button.querySelector('img').alt;$('screen-title').textContent=button.dataset.title;dialog.showModal();}));
  dialog.querySelector('button').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('click',e=>{if(e.target!==dialog)return;const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();});
  // Reveal once without moving layout; content stays visible without JS or with reduced motion.
  if('IntersectionObserver' in window && !window.matchMedia('(prefers-reduced-motion: reduce)').matches){
    const reveal=new IntersectionObserver(entries=>entries.forEach(entry=>{
      if(entry.isIntersecting){entry.target.classList.add('is-visible');reveal.unobserve(entry.target);}
    }),{threshold:0.08});
    document.querySelectorAll('.sat-reveal').forEach(section=>{
      reveal.observe(section);section.classList.add('is-ready');
      section.addEventListener('focusin',()=>{section.classList.add('is-visible');reveal.unobserve(section);},{once:true});
    });
  }
  let mounted=false;
  function mountCalendar(){
    if(mounted)return;mounted=true;
    const mount=$('sat-calendar'),status=$('calendar-status');
    function fail(){mount.setAttribute('aria-busy','false');status.hidden=false;status.textContent='The calendar could not load. Open it below, or message me to arrange a lesson.';}
    const enqueue=(api,args)=>api.q.push(args);
    window.Cal=window.Cal||function(){const args=arguments,api=window.Cal;if(!api.loaded){api.ns={};api.q=api.q||[];const script=document.createElement('script');script.src='https://app.cal.com/embed/embed.js';script.async=true;script.onerror=fail;document.head.appendChild(script);api.loaded=true;}if(args[0]==='init'){const namespace=args[1],namespaced=function(){enqueue(namespaced,arguments);};namespaced.q=[];api.ns[namespace]=api.ns[namespace]||namespaced;enqueue(api.ns[namespace],args);enqueue(api,['initNamespace',namespace]);}else enqueue(api,args);};
    window.Cal('init','chance-me-sat',{origin:'https://app.cal.com'});
    const cal=window.Cal.ns['chance-me-sat'],timeout=setTimeout(fail,20000);
    cal('on',{action:'linkReady',callback:()=>{clearTimeout(timeout);status.hidden=true;mount.setAttribute('aria-busy','false');}});
    cal('on',{action:'bookingSuccessfulV2',callback:()=>{window.ChanceAnalytics?.track('booking_submitted',{destination:'booking'});}});
    cal('on',{action:'linkFailed',callback:()=>{clearTimeout(timeout);fail();}});
    cal('inline',{elementOrSelector:'#sat-calendar',calLink:'marcinz/sat-lessons',config:{layout:'month_view',theme:'light',locale:'en'}});
    cal('ui',{theme:'light',layout:'month_view',hideEventTypeDetails:true,showTimezoneWhenEventDetailsHidden:true,cssVarsPerTheme:{light:{'cal-brand':'#1d655c','cal-brand-emphasis':'#124c43','cal-brand-text':'#ffffff','cal-brand-subtle':'#edf7f1','cal-bg':'#ffffff','cal-bg-subtle':'#f8fafc','cal-text':'#17324d','cal-text-emphasis':'#17324d','cal-text-subtle':'#58718c','cal-border':'#dbe6ef','cal-border-booker-width':'0px','radius':'12px'}}});
  }
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){observer.disconnect();mountCalendar();}},{rootMargin:'500px 0px'});observer.observe($('book-lesson'));}else mountCalendar();
})();
