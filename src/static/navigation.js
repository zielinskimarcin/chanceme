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
})();
