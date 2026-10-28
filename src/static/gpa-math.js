/* Independent estimate, not Bocconi's unpublished conversion tables. */
(function(root,factory){const catalogue=typeof module==='object'&&module.exports?require('./gpa-systems.js'):root.GpaSystems;const api=factory(catalogue);if(typeof module==='object'&&module.exports)module.exports=api;else root.GpaMath=api;})(typeof globalThis!=='undefined'?globalThis:this,function(catalogue){
  'use strict';
  const baseScales=Object.freeze({
    numeric6:{label:'Numeric · 2–6',pass:2,best:6,min:1,max:6},
    ib:{label:'IB · 2–7',pass:2,best:7,min:1,max:7},
    french:{label:'French · 10–20',pass:10,best:20,min:0,max:20},
    reversed:{label:'Numeric · 4–1 (reversed)',pass:4,best:1,min:1,max:6},
    numeric10:{label:'Numeric · 5–10',pass:5,best:10,min:0,max:10},
    numericSwiss:{label:'Numeric · 4–6',pass:4,best:6,min:1,max:6},
    percentage:{label:'Numeric · 50–100',pass:50,best:100,min:0,max:100},
    custom:{label:'Other numeric scale'},
    letters:{label:'Letter grades / mixed scales'}
  });
  const scales=Object.freeze({...baseScales,...(catalogue?catalogue.systems:{})});
  function requiresPortal(scale){return scale==='letters'||Object.prototype.hasOwnProperty.call(scales,scale)&&scales[scale].kind==='portal';}
  class InputError extends Error{constructor(message,field,year){super(message);this.name='InputError';this.field=field;this.year=year;}}
  function number(value,field,year){
    const text=String(value==null?'':value).trim().replace(',','.');
    if(!/^-?(?:\d+(?:\.\d*)?|\.\d+)$/.test(text)||!Number.isFinite(Number(text)))throw new InputError('Enter a valid number.',field,year);
    return Number(text);
  }
  function yearEnds(graduation,schoolYears=12){
    const n=number(graduation,'graduation');
    if(!Number.isInteger(n)||n<2000||n>2100)throw new InputError('Enter a graduation year from 2000 to 2100.','graduation');
    if(![11,12].includes(schoolYears))throw new InputError('Choose the correct school programme.','schoolYears');
    return schoolYears===11?[n-1,n]:[n-2,n-1];
  }
  function bounds(record,year){
    const preset=Object.prototype.hasOwnProperty.call(scales,record.scale)?scales[record.scale]:null;
    if(!preset)throw new InputError('Choose the grading scale used on your transcript.','scale',year);
    if(requiresPortal(record.scale))throw new InputError('This scale requires the GPA from Bocconi’s application.','scale',year);
    const custom=record.scale==='custom'||preset.kind==='custom';
    const pass=custom?number(record.pass,'pass',year):preset.pass,best=custom?number(record.best,'best',year):preset.best;
    if(pass===best)throw new InputError('The passing and best grades must be different.','best',year);
    return{pass,best,label:preset.label,min:custom?undefined:preset.min,max:custom?undefined:preset.max};
  }
  function convert(grade,scale,allowFailing,year,field){
    const value=number(grade,field,year),position=(value-scale.pass)/(scale.best-scale.pass);
    if(position>1+1e-10)throw new InputError('A grade is better than the maximum for this scale. Check the scale and your entry.',field,year);
    if((Number.isFinite(scale.min)&&value<scale.min-1e-10)||(Number.isFinite(scale.max)&&value>scale.max+1e-10))throw new InputError('A grade is outside this school scale. Check your entry or choose a different scale.',field,year);
    if(position<0&&!allowFailing)throw new InputError('This average is below the passing grade. Check the average and the scale.',field,year);
    return{gpa:6+4*Math.min(1,Math.max(0,position)),clamped:position<0};
  }
  function annual(record,year){
    if(record.mode==='official'){
      const gpa=number(record.value,'value',year);
      if(gpa<6||gpa>10)throw new InputError('Enter a Bocconi GPA from 6 to 10.','value',year);
      return{gpa,source:'official_portal',count:null,clamped:0};
    }
    if(!['average','subjects'].includes(record.mode))throw new InputError('Choose what you are entering for this year.','mode',year);
    const scale=bounds(record,year),values=record.mode==='subjects'?String(record.grades||'').split(/[;\n\t]+/).map(s=>s.trim()).filter(Boolean):[record.value];
    if(!values.length)throw new InputError('Enter at least one final subject grade.','grades',year);
    const results=values.map(value=>convert(value,scale,record.mode==='subjects',year,record.mode==='subjects'?'grades':'value'));
    return{gpa:results.reduce((sum,r)=>sum+r.gpa,0)/results.length,source:'linear_estimate',count:record.mode==='subjects'?results.length:null,clamped:results.filter(r=>r.clamped).length,scale:record.scale,pass:scale.pass,best:scale.best,...(catalogue&&catalogue.systems[record.scale]?{education_system:record.scale,school_country:catalogue.systems[record.scale].country,scale_source:catalogue.sources[catalogue.systems[record.scale].source].url}:{})};
  }
  function calculate(data){
    const ends=yearEnds(data.graduation,data.schoolYears===undefined?12:data.schoolYears);
    if(!Array.isArray(data.years)||data.years.length!==2)throw new InputError('Enter both school years.','years');
    const years=data.years.map((record,i)=>({...annual(record,i),year:ends[i]}));
    return{version:2,category:'international',years,model_gpa:(years[0].gpa+years[1].gpa)/2,official:years.every(y=>y.source==='official_portal'),created_at:Date.now()};
  }
  const format=value=>Number(value).toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:3});
  return{scales,baseScales,requiresPortal,InputError,number,yearEnds,annual,calculate,format};
});
