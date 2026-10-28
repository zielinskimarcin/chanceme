const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const c=require('../dist/gpa-systems.js'),g=require('../dist/gpa-math.js'),p=require('../dist/gpa-picker.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('broad secondary-school coverage has unique countries, systems, locally served SVG flags and cited sources',()=>{
 assert.equal(c.countries.length,71);assert.ok(c.countries.filter(x=>x.region==='Europe').length>=40);assert.ok(c.countries.filter(x=>x.region==='Asia').length>=14);
 assert.equal(new Set(c.countries.map(x=>x.id)).size,c.countries.length);
 for(const country of c.countries){assert.ok(country.systems.length);const svg=fs.readFileSync(path.join(__dirname,'../dist/assets/flags',country.flag+'.svg'),'utf8');assert.match(svg,/<svg/);assert.doesNotMatch(svg,/<script|javascript:|<!ENTITY/i);for(const id of country.systems){const s=c.systems[id];assert.equal(s.country,country.id);assert.equal(s.id,id);assert.ok(c.sources[s.source]);assert.match(c.sources[s.source].url,/^https:\/\//);assert.equal(c.sources[s.source].checked,'2026-10-25');}}
});
test('Poland is discoverable by local name and automatically has the report-card scale, not Matura percentages',()=>{
 assert.equal(p.findCountries('Polska')[0].id,'PL');assert.equal(p.findCountries('ŚWIADECTWO')[0].id,'PL');
 assert.equal(c.systems['pl-school'].pass,2);assert.equal(c.systems['pl-school'].best,6);assert.equal(c.systems['pl-school'].min,1);close(g.annual({scale:'pl-school',mode:'average',value:5},0).gpa,9);
});
test('each numeric preset maps its passing and maximum grades exactly and rejects impossible grades',()=>{
 for(const s of Object.values(c.systems)){if(s.kind!=='numeric')continue;assert.ok(Number.isFinite(s.pass)&&Number.isFinite(s.best)&&s.pass!==s.best);assert.ok(s.min<=Math.min(s.best,s.pass));assert.ok(s.max>=Math.max(s.best,s.pass));close(g.annual({scale:s.id,mode:'average',value:s.pass},0).gpa,6);close(g.annual({scale:s.id,mode:'average',value:s.best},0).gpa,10);assert.throws(()=>g.annual({scale:s.id,mode:'subjects',grades:String(s.min-1)},0));assert.throws(()=>g.annual({scale:s.id,mode:'subjects',grades:String(s.max+1)},0));}
});
test('different regional and school scales are not flattened to a country-wide threshold',()=>{
 assert.equal(c.systems['ca-ontario'].pass,50);assert.equal(c.systems['ca-bc'].pass,50);assert.equal(c.systems['ca-quebec'].pass,60);assert.equal(c.systems['in-cbse'].pass,33);assert.equal(c.systems['in-isc'].pass,35);assert.equal(c.systems['pk-old'].pass,33);assert.equal(c.systems['pk-new'].pass,40);
 for(const id of ['br-school','ar-school','co-school','id-school','be-flemish','be-french']){assert.equal(c.systems[id].kind,'custom');assert.equal(c.systems[id].pass,undefined);assert.throws(()=>g.annual({scale:id,mode:'average',value:8,best:10,pass:''},0));}
});
test('German reversed marks and Oberstufe points preserve different scales and provenance',()=>{
 close(g.annual({scale:'de-marks',mode:'average',value:2.5},0).gpa,8);close(g.annual({scale:'de-points',mode:'average',value:10},0).gpa,8);
 const y=g.annual({scale:'pl-school',mode:'average',value:5},0);assert.equal(y.school_country,'PL');assert.equal(y.education_system,'pl-school');assert.match(y.scale_source,/nuffic/);assert.equal(y.source,'linear_estimate');
});
test('letter, rank and unresolved scales cannot silently produce a numeric estimate',()=>{
 for(const s of Object.values(c.systems)){if(s.kind!=='portal')continue;assert.throws(()=>g.annual({scale:s.id,mode:'average',value:4},0),/Bocconi/);}
 assert.equal(c.systems['ua-school'].kind,'portal');assert.equal(c.systems['kr-school'].kind,'portal');
 const official=g.calculate({graduation:2027,years:[{scale:'kr-school',mode:'official',value:8.8},{scale:'ua-school',mode:'official',value:9.2}]});close(official.model_gpa,9);assert.equal(official.official,true);
});
test('global curricula remain available for every country, independent of school location',()=>{
 for(const country of c.countries){const ids=p.systemsFor(country.id);assert.ok(ids.includes('int-ib'));assert.ok(ids.includes('int-custom'));assert.ok(ids.includes('int-portal'));assert.equal(new Set(ids).size,ids.length);}
 assert.deepEqual(p.systemsFor('invalid'),[]);assert.equal(p.findCountries('Việt Nam')[0].id,'VN');
});
test('numeric scale bounds reject nonsensical failed-subject inputs instead of clamping them',()=>{
 assert.throws(()=>g.annual({scale:'pl-school',mode:'subjects',grades:'-1;5'},0),/outside/);assert.throws(()=>g.annual({scale:'de-marks',mode:'subjects',grades:'7;1'},0),/outside/);
 const y=g.annual({scale:'pl-school',mode:'subjects',grades:'1;6'},0);close(y.gpa,8);assert.equal(y.clamped,1);
});
test('research ledger and public source page retain conflicts instead of presenting guessed conversions',()=>{
 const research=fs.readFileSync('./docs/GPA-COUNTRY-SYSTEMS-RESEARCH.md','utf8');assert.match(research,/Ukraine/);assert.match(research,/2026–2028/);assert.match(research,/unpublished/);assert.match(research,/Brazil and Argentina|Brazil/);
 const html=fs.readFileSync('./dist/gpa-school-sources.html','utf8');assert.match(html,/noindex,nofollow/);for(const country of c.countries)assert.ok(html.includes(country.name.replace(/&/g,'&amp;')));
});

test('unsupported or inherited property names never yield a NaN GPA',()=>{
 for(const scale of ['constructor','__proto__','toString','not-a-school'])assert.throws(()=>g.annual({scale,mode:'average',value:5},0),/Choose/);
});

test('curricula appear before countries in the shared dropdown, with an explicit route for AS and GCSE letters',()=>{
 const choices=p.findChoices('');assert.deepEqual(choices.slice(0,5).map(x=>x.id),['ib','alevel','gcse','us','eb']);
 assert.ok(choices.slice(0,5).every(x=>x.kind==='programme'));assert.equal(choices.filter(x=>x.kind!=='programme'&&x.id!=='INT').length,70);
 assert.equal(choices.at(-1).id,'INT');assert.ok(p.programmes.find(x=>x.id==='alevel').systems.includes('int-as'));assert.ok(p.programmes.find(x=>x.id==='gcse').systems.includes('int-igcse-letters'));
});
test('the dropdown searches programmes and local country names without requiring a country first',()=>{
 assert.equal(p.findChoices('IB')[0].id,'ib');assert.equal(p.findChoices('International Baccalaureate')[0].id,'ib');assert.equal(p.findChoices('A-level')[0].id,'alevel');assert.equal(p.findChoices('AS levels')[0].id,'alevel');assert.equal(p.findChoices('IGCSE')[0].id,'gcse');assert.equal(p.findChoices('AP')[0].id,'us');assert.equal(p.findChoices('Polska')[0].id,'PL');assert.deepEqual(p.findChoices('no such curriculum xyz'),[]);
});
test('saved and older drafts keep their chosen route rather than silently changing country or scale',()=>{
 assert.equal(p.programmeFromDraft({programme:'ib',country:'',scale:'int-ib'}),'ib');assert.equal(p.programmeFromDraft({country:'INT',scale:'int-cambridge'}),'alevel');assert.equal(p.programmeFromDraft({country:'INT',scale:'int-as'}),'alevel');assert.equal(p.programmeFromDraft({country:'PL',scale:'int-ib'}),'national');assert.equal(p.programmeFromDraft({scale:'custom',manual:true}),'national');assert.equal(p.programmeFromDraft({}),'');assert.notEqual(p.programmeFromDraft({programme:'constructor',scale:'int-ib'}),'constructor');
});
test('direct international routes use the same sourced scales and preserve estimate versus portal provenance',()=>{
 const annual=(scale,value)=>({scale,mode:'average',value});close(g.calculate({graduation:2027,years:[annual('int-ib',6),annual('int-ib',7)]}).model_gpa,9.6);
 close(g.calculate({graduation:2027,years:[annual('int-eb',8),annual('int-eb',9)]}).model_gpa,8.8);
 close(g.annual(annual('us-gpa',3.5),0).gpa,6+4*2.5/3);
 for(const scale of ['int-cambridge','int-as','int-igcse-letters'])assert.throws(()=>g.annual({scale,mode:'subjects',grades:'A;B'},0),/Bocconi/);
});

test('unsupported input policy agrees with the calculation guard for every portal route',()=>{
 for(const s of Object.values(c.systems)){
  assert.equal(g.requiresPortal(s.id),s.kind==='portal',s.id);
  if(s.kind==='portal'){
   assert.throws(()=>g.annual({scale:s.id,mode:'subjects',grades:'A*;A;B'},0),/requires.*Bocconi/);
   close(g.annual({scale:s.id,mode:'official',value:'9.2'},0).gpa,9.2);
  }
 }
 assert.equal(g.requiresPortal('letters'),true);for(const scale of ['int-ib','gb-gcse','custom','constructor','__proto__',''])assert.equal(g.requiresPortal(scale),false);
});
test('one portal year can be combined with a numeric year without reinterpreting its grades',()=>{
 const years=[{scale:'int-ib',mode:'average',value:6},{scale:'int-igcse-letters',mode:'official',value:9.2}];
 const result=g.calculate({graduation:2027,years});close(result.model_gpa,9.2);assert.equal(result.official,false);assert.deepEqual(result.years.map(x=>x.source),['linear_estimate','official_portal']);
 assert.throws(()=>g.calculate({graduation:2027,years:[years[0],{...years[1],value:'A*'}]}),/number/);
 assert.throws(()=>g.calculate({graduation:2027,years:[years[0],{...years[1],mode:'subjects',grades:'A;B'}]}),/Bocconi/);
});
