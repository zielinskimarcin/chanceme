(function(){
  'use strict';
  const $=id=>document.getElementById(id),math=GpaMath,form=$('gpa-form'),entries=[],draftKey='chanceme.gpa-draft.v1';
  let result=null,restoring=false;
  const reduced=window.matchMedia('(prefers-reduced-motion: reduce)');
  function el(tag,text,className){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(className)n.className=className;return n;}
  function field(label,id,tag='input'){
    const wrap=el('div',undefined,'field'),caption=el('label',label),input=el(tag),error=el('p',undefined,'field-error');
    caption.htmlFor=id;input.id=id;error.id=id+'-error';error.hidden=true;input.setAttribute('aria-describedby',error.id);if(tag==='input')input.inputMode='decimal';wrap.append(caption,input,error);return{wrap,caption,input,error};
  }
  function changed(){invalidate();refresh();saveDraft();}
  const mainPicker=GpaPicker.create($('main-school-picker'),'gpa',changed);
  [0,1].forEach(i=>{
    const card=el('section',undefined,'gpa-year'),heading=el('div',undefined,'gpa-year-heading'),number=el('span',i+1,'gpa-year-number'),headingText=el('div'),title=el('h2',i===0?'Third-last year':'Second-last year'),dates=el('p',undefined,'gpa-year-dates');
    title.id='year-'+i+'-title';card.setAttribute('aria-labelledby',title.id);headingText.append(title,dates);heading.append(number,headingText);card.append(heading);
    const mode=field('What are you entering?','year-'+i+'-mode','select');[['average','School average'],['subjects','Subject grades'],['official','Bocconi GPA']].forEach(([value,label])=>{const option=el('option',label);option.value=value;mode.input.append(option);});mode.input.value='average';
    const value=field('Annual average','year-'+i+'-value'),grades=field('Final subject grades','year-'+i+'-grades','textarea');grades.input.rows=4;grades.input.maxLength=10000;grades.input.placeholder='5\n4\n6';grades.wrap.hidden=true;
    const hint=el('p',undefined,'gpa-entry-hint'),subjectNote=el('p','Omit PE, religion, ethics, conduct and military training; for IB, also TOK, CAS and EE.','gpa-subject-note');subjectNote.hidden=true;
    card.append(mode.wrap,value.wrap,grades.wrap,hint,subjectNote);
    const entry={card,title,dates,mode,value,grades,hint,subjectNote,lastMode:'average',values:{average:'',official:''},override:false};
    if(i===1){
      const toggle=el('button','Different grading scale?','text-button gpa-scale-toggle');toggle.type='button';toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-controls','year-1-override');
      const override=el('div',undefined,'gpa-year-override');override.id='year-1-override';override.hidden=true;const pickerHost=el('div');override.append(pickerHost);card.append(toggle,override);const picker=GpaPicker.create(pickerHost,'year-1',changed);
      Object.assign(entry,{toggle,overrideWrap:override,picker});
      toggle.addEventListener('click',()=>{entry.override=!entry.override;refresh();invalidate();saveDraft();});
    }
    mode.input.addEventListener('change',()=>{if(entry.lastMode!=='subjects')entry.values[entry.lastMode]=entry.value.input.value;entry.lastMode=entry.mode.input.value;if(entry.lastMode!=='subjects')entry.value.input.value=entry.values[entry.lastMode]||'';refresh();});
    entries.push(entry);$('gpa-years').append(card);
  });
  function currentScale(entry,i){return i===1&&entry.override?entry.picker.record():mainPicker.record();}
  function needsPortal(entry,i){return entry.mode.input.value!=='official'&&math.requiresPortal(currentScale(entry,i).scale);}
  function refresh(){
    const bothOfficial=entries.every(e=>e.mode.input.value==='official');
    $('common-scale-field').hidden=false;mainPicker.refresh();
    const blocked=entries.filter(needsPortal);$('letter-notice').hidden=!blocked.length;
    $('use-portal-values').textContent=blocked.length===1?'I have the GPA for this year →':'I already have my GPA →';
    $('gpa-year-setup').hidden=blocked.length===entries.length;
    $('calculate-gpa').hidden=blocked.length>0;$('gpa-form-note').hidden=blocked.length>0;
    $('calculate-gpa').disabled=blocked.length>0;
    if(blocked.length)$('calculate-gpa').setAttribute('aria-describedby','portal-instructions');else $('calculate-gpa').removeAttribute('aria-describedby');
    $('gpa-form-note').textContent=bothOfficial?'We average these two values for the Chance Me admission model.':'School-grade conversions are estimates. Your application uses Bocconi’s own calculation.';
    let ends;try{ends=math.yearEnds($('graduation').value,$('eleven-year').checked?11:12);}catch(_){ends=null;}
    entries.forEach((entry,i)=>{
      const mode=entry.mode.input.value,official=mode==='official',subjects=mode==='subjects',scale=currentScale(entry,i),preset=math.scales[scale.scale],portal=math.requiresPortal(scale.scale),locked=needsPortal(entry,i);
      entry.title.textContent=$('eleven-year').checked?'Grade '+(i===0?'10':'11'):i===0?'Third-last year':'Second-last year';
      entry.dates.textContent=ends?(ends[i]-1)+'/'+String(ends[i]).slice(-2):'Enter your graduation year';
      entry.card.hidden=locked&&!entry.override;entry.mode.wrap.hidden=locked;entry.mode.input.disabled=locked;entry.hint.hidden=locked;
      Array.from(entry.mode.input.options).forEach(option=>{option.disabled=portal&&option.value!=='official';});
      entry.value.input.disabled=locked;entry.grades.input.disabled=locked;
      entry.value.wrap.hidden=subjects||locked;entry.grades.wrap.hidden=!subjects||locked;entry.subjectNote.hidden=!subjects||locked;
      entry.value.caption.textContent=official?'Bocconi GPA':'Annual average';entry.value.input.placeholder=official?'e.g. 9.20':preset&&preset.pass!==undefined?'e.g. '+(preset.pass+.625*(preset.best-preset.pass)).toFixed(2):'Enter your average';
      entry.hint.textContent=locked?'Grade entry is unavailable for this scale. Enter your Bocconi GPA to continue.':official?'Enter the 6–10 result from Bocconi’s application.':subjects?'One final grade per line, or separate with semicolons.':'Use the unweighted average printed on your transcript.';
      if(i===1){entry.toggle.hidden=false;entry.toggle.textContent=entry.override?'Use the main school system':'Different school or system?';entry.toggle.setAttribute('aria-expanded',String(entry.override));entry.overrideWrap.hidden=!entry.override;entry.picker.refresh();}
    });
    $('gpa-years').hidden=entries.every(e=>e.card.hidden);
  }
  function clearErrors(){form.querySelectorAll('[aria-invalid]').forEach(n=>n.removeAttribute('aria-invalid'));form.querySelectorAll('.field-error').forEach(n=>{n.hidden=true;n.textContent='';});$('gpa-error').hidden=true;}
  function invalidate(){result=null;$('gpa-result').dataset.state='empty';$('gpa-empty').hidden=false;$('gpa-calculated').hidden=true;$('gpa-transfer-error').hidden=true;$('gpa-announcement').textContent='';clearErrors();}
  function data(){return{graduation:$('graduation').value,schoolYears:$('eleven-year').checked?11:12,years:entries.map((e,i)=>({...currentScale(e,i),mode:e.mode.input.value,value:e.value.input.value,grades:e.grades.input.value}))};}
  function showError(error){
    const entry=entries[error.year],fieldName=error.field;let input,errorNode;
    if(entry&&['value','grades','mode'].includes(fieldName)){input=entry[fieldName].input;errorNode=entry[fieldName].error;}
    else if(entry&&['scale','pass','best','country'].includes(fieldName)){const field=(error.year===1&&entry.override?entry.picker:mainPicker).errorFor(fieldName);input=field.input;errorNode=field.error;}
    else if(fieldName==='graduation'){input=$('graduation');errorNode=$('graduation-error');}
    const message=(entry?(error.year===0?'First year: ':'Second year: '):'')+error.message;
    $('gpa-error').textContent=message;$('gpa-error').hidden=false;$('gpa-announcement').textContent=message;
    if(input){input.setAttribute('aria-invalid','true');if(errorNode){errorNode.textContent=error.message;errorNode.hidden=false;}input.focus();}
  }
  function calculate(scroll){
    invalidate();
    try{
      result=math.calculate(data());$('gpa-result').dataset.state='result';$('gpa-empty').hidden=true;$('gpa-calculated').hidden=false;
      $('gpa-result-kicker').textContent=result.official?'Average Bocconi GPA':'Estimated GPA';$('gpa-result-value').textContent=math.format(result.model_gpa);
      $('gpa-breakdown').replaceChildren(...result.years.map((year,i)=>{const item=el('div');item.append(el('span',entries[i].dates.textContent),el('strong',math.format(year.gpa)),el('small',year.source==='official_portal'?'From Bocconi':'Estimate'));return item;}));
      $('gpa-result-note').textContent=result.official?'Average of the two Bocconi GPA values you entered.':'This is an estimate. Confirm your GPA in Bocconi’s application.';
      const clamped=result.years.reduce((sum,y)=>sum+y.clamped,0);$('gpa-clamped-note').hidden=!clamped;$('gpa-clamped-note').textContent=clamped+' below-passing '+(clamped===1?'grade was':'grades were')+' treated as passing.';
      $('gpa-announcement').textContent=(result.official?'Average Bocconi GPA: ':'Estimated GPA: ')+math.format(result.model_gpa)+' out of 10.';
      if(!reduced.matches&&typeof $('gpa-calculated').animate==='function')$('gpa-calculated').animate([{opacity:0,transform:'translateY(5px)'},{opacity:1,transform:'translateY(0)'}],{duration:350,easing:'ease-out'});
      if(scroll&&window.matchMedia('(max-width: 800px)').matches)$('gpa-result').scrollIntoView({behavior:reduced.matches?'instant':'smooth',block:'nearest'});
      if(scroll)window.ChanceAnalytics?.track('gpa_calculated');saveDraft();return true;
    }catch(error){showError(error);return false;}
  }
  function snapshot(){
    entries.forEach(e=>{if(e.mode.input.value!=='subjects')e.values[e.mode.input.value]=e.value.input.value;});
    return{version:2,created_at:Date.now(),graduation:$('graduation').value,eleven:$('eleven-year').checked,school:mainPicker.snapshot(),calculated:!!result,years:entries.map((e,i)=>({mode:e.mode.input.value,values:e.values,grades:e.grades.input.value,override:i===1&&e.override,...(i===1?{school:e.picker.snapshot()}:{})}))};
  }
  function saveDraft(){if(restoring)return;try{sessionStorage.setItem(draftKey,JSON.stringify(snapshot()));}catch(_){/* Calculation remains available without browser storage. */}}
  function restoreDraft(){
    try{
      const raw=sessionStorage.getItem(draftKey);if(!raw)return;const d=JSON.parse(raw),age=Date.now()-d.created_at;
      if(![1,2].includes(d.version)||!Number.isFinite(d.created_at)||age<0||age>=1800000||!Array.isArray(d.years)||d.years.length!==2)return;
      restoring=true;const text=(v,max=10000)=>typeof v==='string'?v.slice(0,max):'';
      $('graduation').value=text(d.graduation,4);$('eleven-year').checked=d.eleven===true;mainPicker.restore(d.school||{scale:text(d.scale,80),pass:text(d.pass,20),best:text(d.best,20),manual:true});
      entries.forEach((e,i)=>{const y=d.years[i];if(!y||!['average','subjects','official'].includes(y.mode))return;e.mode.input.value=y.mode;e.lastMode=y.mode;e.values={average:text(y.values&&y.values.average,20),official:text(y.values&&y.values.official,20)};e.value.input.value=e.values[y.mode]||'';e.grades.input.value=text(y.grades);if(i===1){e.override=y.override===true;e.picker.restore(y.school||{scale:text(y.scale,80),pass:text(y.pass,20),best:text(y.best,20),manual:true});}});
      refresh();if(d.calculated===true)calculate(false);
    }catch(_){/* Ignore malformed drafts. */}finally{restoring=false;}
  }
  form.addEventListener('submit',event=>{event.preventDefault();calculate(true);});
  form.addEventListener('input',event=>{if(event.target.classList.contains('gpa-country-search'))return;changed();});form.addEventListener('change',event=>{if(event.target.classList.contains('gpa-country-search'))return;changed();});
  $('eleven-year').addEventListener('change',()=>{invalidate();refresh();saveDraft();});
  $('use-portal-values').addEventListener('click',()=>{const blocked=entries.filter(needsPortal);blocked.forEach(e=>{if(e.lastMode!=='subjects')e.values[e.lastMode]=e.value.input.value;e.mode.input.value=e.lastMode='official';e.value.input.value=e.values.official;});invalidate();refresh();saveDraft();if(blocked.length)blocked[0].value.input.focus();});
  $('use-gpa').addEventListener('click',()=>{
    if(!result)return;saveDraft();
    try{sessionStorage.setItem('chanceme.gpa-transfer.v2',JSON.stringify({...result,created_at:Date.now()}));window.ChanceAnalytics?.track('gpa_transferred',{destination:'calculator'});window.location.assign('/#calculator');}
    catch(_){$('gpa-transfer-error').textContent='Copy '+math.format(result.model_gpa)+' into the admission calculator. Your browser could not transfer it automatically.';$('gpa-transfer-error').hidden=false;}
  });
  window.addEventListener('pagehide',saveDraft);
  // Same header interactions as the admissions calculator, without loading its form controller.
  const toggle=$('menu-toggle'),nav=$('main-nav'),guides=$('guides-toggle'),panel=$('guides-panel');
  function closeGuides(){panel.hidden=true;guides.setAttribute('aria-expanded','false');}
  function closeMenu(){nav.classList.remove('is-open');toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-label','Open navigation');closeGuides();}
  toggle.addEventListener('click',()=>{const open=toggle.getAttribute('aria-expanded')!=='true';nav.classList.toggle('is-open',open);toggle.setAttribute('aria-expanded',String(open));toggle.setAttribute('aria-label',open?'Close navigation':'Open navigation');});
  guides.addEventListener('click',()=>{const open=guides.getAttribute('aria-expanded')!=='true';guides.setAttribute('aria-expanded',String(open));panel.hidden=!open;});
  guides.addEventListener('keydown',event=>{if(event.key==='ArrowDown'){event.preventDefault();guides.setAttribute('aria-expanded','true');panel.hidden=false;panel.querySelector('a').focus();}});
  nav.addEventListener('click',event=>{if(event.target.closest('a')){saveDraft();closeMenu();}});
  document.addEventListener('click',event=>{if(!$('guides-dropdown').contains(event.target))closeGuides();});
  document.addEventListener('keydown',event=>{if(event.key==='Escape'){if(guides.getAttribute('aria-expanded')==='true'){closeGuides();guides.focus();}else if(toggle.getAttribute('aria-expanded')==='true'){closeMenu();toggle.focus();}}});
  closeGuides();refresh();restoreDraft();
})();
