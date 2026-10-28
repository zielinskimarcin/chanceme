/* Searchable curriculum/country dropdown and sourced school scales; no external requests. */
(function(root,factory){const catalogue=typeof module==='object'&&module.exports?require('./gpa-systems.js'):root.GpaSystems;const api=factory(catalogue);if(typeof module==='object'&&module.exports)module.exports=api;else root.GpaPicker=api;})(typeof globalThis!=='undefined'?globalThis:this,function(catalogue){
  'use strict';
  const normalize=text=>String(text||'').normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().replace(/ł/g,'l');
  const countries=catalogue.countries,byId=Object.fromEntries(countries.map(c=>[c.id,c]));
  const regionOrder=['International','Europe','Asia','Americas','Africa','Oceania'];
  function findCountries(query){const q=normalize(query).trim();return countries.filter(c=>normalize([c.name,c.id,c.aliases,...c.systems.map(s=>catalogue.systems[s].label)].join(' ')).includes(q)).sort((a,b)=>regionOrder.indexOf(a.region)-regionOrder.indexOf(b.region)||a.name.localeCompare(b.name,'en'));}
  function systemsFor(country){const c=byId[country];if(!c)return[];return[...new Set([...c.systems,...(country==='INT'?[]:catalogue.international),'int-custom','int-portal'])];}
  const programmes=catalogue.programmes,programmeById=Object.fromEntries(programmes.map(p=>[p.id,p]));
  function programmeFromDraft(d){
    if(Object.prototype.hasOwnProperty.call(programmeById,d.programme))return d.programme;
    // A legacy national selection stays national, even if it uses an international scale.
    if(byId[d.country]&&d.country!=='INT')return 'national';
    const direct=programmes.find(p=>p.systems.includes(d.scale));
    if(direct)return direct.id;
    return byId[d.country]||d.scale?'national':'';
  }
  function findChoices(query){
    const q=normalize(query).replace(/[-–]/g,' ').trim();
    const curricula=programmes.filter(p=>p.id!=='national'&&normalize([p.label,p.short,p.detail,...p.systems.map(id=>catalogue.systems[id].label)].join(' ')).replace(/[-–]/g,' ').includes(q)).map(p=>({...p,name:p.label,kind:'programme',region:'International curricula'}));
    const found=findCountries(query),national=found.filter(c=>c.id!=='INT'),other=found.filter(c=>c.id==='INT').map(c=>({...c,region:'Other'}));
    return [...curricula,...national,...other];
  }
  function create(host,prefix,onChange){
    const doc=host.ownerDocument,make=(tag,text,cls)=>{const n=doc.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
    const field=(label,id)=>{const wrap=make('div',undefined,'field'),caption=make('label',label),input=make('input'),error=make('p',undefined,'field-error');caption.htmlFor=id;input.id=id;input.inputMode='decimal';error.id=id+'-error';error.hidden=true;input.setAttribute('aria-describedby',error.id);wrap.append(caption,input,error);return{wrap,input,error};};
    let country='',programme='',active=-1,visible=[],open=false,manualMode=false;
    host.classList.add('gpa-picker');
    const grid=make('div',undefined,'gpa-country-grid'),countryField=make('div',undefined,'field gpa-country-field'),caption=make('label','Country or curriculum'),trigger=make('button',undefined,'gpa-country-trigger');
    trigger.id=prefix+'-country';trigger.type='button';caption.htmlFor=trigger.id;trigger.setAttribute('aria-haspopup','dialog');trigger.setAttribute('aria-expanded','false');trigger.setAttribute('aria-controls',prefix+'-country-menu');
    const selected=make('span',undefined,'gpa-country-selected'),arrow=make('span',undefined,'chevron');arrow.setAttribute('aria-hidden','true');trigger.append(selected,arrow);countryField.append(caption,trigger);
    const popup=make('div',undefined,'gpa-country-menu');popup.id=prefix+'-country-menu';popup.hidden=true;popup.setAttribute('role','dialog');popup.setAttribute('aria-label','Choose country or curriculum');
    const searchLabel=make('label','Search countries or curricula','sr-only'),search=make('input',undefined,'gpa-country-search');search.id=prefix+'-country-search';searchLabel.htmlFor=search.id;search.type='search';search.autocomplete='off';search.placeholder='Search country or curriculum…';search.setAttribute('role','combobox');search.setAttribute('aria-autocomplete','list');search.setAttribute('aria-expanded','false');search.setAttribute('aria-controls',prefix+'-country-list');
    const list=make('div',undefined,'gpa-country-list');list.id=prefix+'-country-list';list.setAttribute('role','listbox');list.setAttribute('aria-label','Countries and curricula');const empty=make('p','No matches. Try “International” for another scale.','gpa-country-no-results');empty.hidden=true;
    popup.append(searchLabel,search,list,empty);countryField.append(popup);
    const scaleField=make('div',undefined,'field gpa-system-field'),scaleLabel=make('label','Education system'),scale=make('select');scale.id=prefix+'-scale';scaleLabel.htmlFor=scale.id;scale.disabled=true;const scaleError=make('p',undefined,'field-error');scaleError.id=prefix+'-scale-error';scaleError.hidden=true;scale.setAttribute('aria-describedby',prefix+'-scale-info '+scaleError.id);scaleField.append(scaleLabel,scale);grid.append(countryField,scaleField);
    const info=make('div',undefined,'gpa-scale-info'),summary=make('p'),note=make('p',undefined,'gpa-scale-detail'),links=make('div',undefined,'gpa-scale-links'),source=make('a','Scale source ↗'),edit=make('button','Use a different scale','text-button gpa-manual-toggle');
    info.id=prefix+'-scale-info';source.target='_blank';source.rel='noopener noreferrer';edit.type='button';links.append(source,edit);info.append(summary,note,links);
    const bounds=make('div',undefined,'gpa-custom-bounds'),pass=field('Passing grade',prefix+'-pass'),best=field('Best possible grade',prefix+'-best');bounds.hidden=true;pass.input.placeholder='From your transcript';best.input.placeholder='From your transcript';bounds.append(pass.wrap,best.wrap);host.append(grid,scaleError,info,bounds);
    function image(c){const img=make('img');img.src='/assets/flags/'+c.flag+'.svg';img.width=24;img.height=18;img.alt='';img.setAttribute('aria-hidden','true');img.loading='lazy';return img;}
    function programmeBadge(p){const img=image({flag:p.symbol||'globe'});if(p.id==='ib')img.className='gpa-ib-symbol';return img;}
    function paintSelected(){
      const p=programme!=='national'&&programmeById[programme],c=byId[country];selected.replaceChildren();
      const name=p?p.label:c?c.name:'';trigger.setAttribute('aria-label','Country or curriculum'+(name?': '+name:''));
      if(p)selected.append(programmeBadge(p),make('span',p.label));
      else if(c)selected.append(image(c),make('span',c.name));
      else selected.append(make('span','Choose country or curriculum','gpa-country-placeholder'));
    }
    function activeAt(index){if(!visible.length){active=-1;search.removeAttribute('aria-activedescendant');return;}active=(index+visible.length)%visible.length;visible.forEach((n,i)=>n.classList.toggle('is-active',i===active));search.setAttribute('aria-activedescendant',visible[active].id);visible[active].scrollIntoView({block:'nearest'});}
    function paintList(){list.replaceChildren();visible=[];active=-1;search.removeAttribute('aria-activedescendant');const found=findChoices(search.value);let region='';found.forEach(c=>{
      if(c.region!==region){region=c.region;const heading=make('div',region,'gpa-country-region');heading.setAttribute('role','presentation');list.append(heading);}
      const isProgramme=c.kind==='programme',isSelected=isProgramme?c.id===programme:programme==='national'&&c.id===country;
      const row=make('div',undefined,'gpa-country-option');row.id=prefix+'-option-'+(isProgramme?'programme-':'')+c.id;row.setAttribute('role','option');row.setAttribute('aria-selected',String(isSelected));
      row.append(isProgramme?programmeBadge(c):image(c),make('span',c.name));if(isSelected){const check=make('span','✓','gpa-country-check');check.setAttribute('aria-hidden','true');row.append(check);}
      row.addEventListener('pointerdown',event=>event.preventDefault());row.addEventListener('click',()=>{if(isProgramme)setProgramme(c.id);else setCountry(c.id);close(true);onChange();});list.append(row);visible.push(row);
    });empty.hidden=found.length!==0;}
    function show(){open=true;popup.hidden=false;trigger.setAttribute('aria-expanded','true');search.setAttribute('aria-expanded','true');search.value='';paintList();search.focus();}
    function close(focus){if(!open)return;open=false;popup.hidden=true;trigger.setAttribute('aria-expanded','false');search.setAttribute('aria-expanded','false');search.removeAttribute('aria-activedescendant');if(focus)trigger.focus();}
    function paintProgramme(){
      info.hidden=!scale.value||GpaMath.requiresPortal(scale.value);
      scaleLabel.textContent=programme==='alevel'?'Qualification':programme==='gcse'?'Grades on your transcript':programme==='us'?'School grading scale':'Grading scale';
    }
    function renderInfo(){paintProgramme();const s=catalogue.systems[scale.value],custom=scale.value==='custom'||s&&s.kind==='custom',preset=GpaMath.scales[scale.value];
      bounds.hidden=!custom;source.hidden=!s;edit.hidden=!s||manualMode||GpaMath.requiresPortal(scale.value);edit.setAttribute('aria-expanded',String(manualMode));
      if(s){source.href=catalogue.sources[s.source].url;source.title=catalogue.sources[s.source].title;note.textContent=s.note+(programme==='us'?' AP students: use unweighted school grades, not AP exam scores.':'');note.hidden=!s.note;}
      else {note.hidden=true;source.removeAttribute('href');}
      summary.textContent=GpaMath.requiresPortal(scale.value)?'Grade conversion unavailable here · use Bocconi’s GPA.':custom?'Enter your school’s passing and best possible marks.':preset&&preset.pass!==undefined?'Pass: '+preset.pass+' · Best: '+preset.best+(preset.best<preset.pass?' · Lower is better':''):'Choose your education system to set the scale.';
      if(!scale.value){scaleError.hidden=true;scale.removeAttribute('aria-invalid');}
    }
    function addOption(id){const s=catalogue.systems[id],base=GpaMath.baseScales[id];const option=make('option',s?s.label:base.label);option.value=id;scale.append(option);}
    function fillOptions(value){scale.replaceChildren();const first=make('option',programme==='national'&&!country?'Choose a country first':'Choose your grading scale');first.value='';scale.append(first);
      const p=programmeById[programme],ids=manualMode?Object.keys(GpaMath.baseScales):programme==='national'?systemsFor(country):p?[...p.systems]:[];
      if(value&&!ids.includes(value)&&Object.prototype.hasOwnProperty.call(GpaMath.scales,value))ids.unshift(value);ids.forEach(addOption);scale.disabled=(!programme||programme==='national'&&!country)&&!manualMode;scale.value=value||'';
    }
    function setProgramme(id){
      programme=id;country='';manualMode=false;const p=programmeById[programme];
      const ids=programme==='national'&&country?byId[country].systems:p.systems;
      const value=programme==='national'?(ids.length===1?ids[0]:''):ids[0]||'';
      pass.input.value='';best.input.value='';fillOptions(value);paintSelected();renderInfo();
    }
    function setCountry(id){programme='national';country=byId[id]?id:'';manualMode=false;const ids=country?byId[country].systems:[];const value=ids.length===1?ids[0]:'';fillOptions(value);const s=catalogue.systems[value];pass.input.value='';best.input.value=s&&s.kind==='custom'&&s.best!==undefined?String(s.best):'';paintSelected();renderInfo();}
    trigger.addEventListener('click',()=>open?close(false):show());trigger.addEventListener('keydown',e=>{if(['ArrowDown','ArrowUp'].includes(e.key)){e.preventDefault();show();activeAt(e.key==='ArrowUp'?-1:0);}});
    search.addEventListener('input',paintList);search.addEventListener('keydown',e=>{
      if(['ArrowDown','ArrowUp','Home','End','Enter','Escape'].includes(e.key)){e.preventDefault();e.stopPropagation();if(e.key==='Escape'){close(true);return;}if(e.key==='Enter'){if(active>=0)visible[active].click();else if(visible.length===1)visible[0].click();return;}activeAt(e.key==='Home'?0:e.key==='End'?visible.length-1:e.key==='ArrowDown'?active+1:active<0?visible.length-1:active-1);}
    });
    doc.addEventListener('click',e=>{if(!countryField.contains(e.target))close(false);});doc.addEventListener('keydown',e=>{if(e.key==='Escape'&&open){e.preventDefault();close(true);}});countryField.addEventListener('focusout',e=>{if(e.relatedTarget&&!countryField.contains(e.relatedTarget))close(false);});
    scale.addEventListener('change',()=>{const s=catalogue.systems[scale.value];pass.input.value='';best.input.value=s&&s.kind==='custom'&&s.best!==undefined?String(s.best):'';renderInfo();onChange();});
    edit.addEventListener('click',()=>{manualMode=true;fillOptions('custom');pass.input.value='';best.input.value='';renderInfo();onChange();pass.input.focus();});
    paintSelected();fillOptions('');renderInfo();
    return{scale,pass,best,trigger,refresh:renderInfo,record:()=>({scale:scale.value,pass:pass.input.value,best:best.input.value}),snapshot:()=>({programme,country,scale:scale.value,pass:pass.input.value,best:best.input.value,manual:manualMode}),restore:d=>{country=byId[d.country]?d.country:'';programme=programmeFromDraft(d);const value=Object.prototype.hasOwnProperty.call(GpaMath.scales,d.scale)?d.scale:'';manualMode=d.manual===true;if(programme==='national'&&!country&&value){country='INT';manualMode=true;}fillOptions(value);pass.input.value=typeof d.pass==='string'?d.pass.slice(0,20):'';best.input.value=typeof d.best==='string'?d.best.slice(0,20):'';paintSelected();renderInfo();},errorFor:field=>field==='scale'?{input:scale.disabled?trigger:scale,error:scaleError}:field==='country'?{input:trigger,error:scaleError}:field==='pass'?pass:best};
  }
  return{create,findCountries,findChoices,systemsFor,programmes,programmeFromDraft};
});
