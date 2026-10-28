const test=require('node:test'),assert=require('node:assert/strict'),g=require('../dist/gpa-math.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} differs from ${b}`);
const average=(scale,value)=>({mode:'average',scale,value});
test('every numeric template maps passing, midpoint and best to 6, 8 and 10',()=>{
  for(const [id,s] of Object.entries(g.scales)){if(s.pass===undefined)continue;close(g.annual(average(id,s.pass),0).gpa,6);close(g.annual(average(id,(s.pass+s.best)/2),0).gpa,8);close(g.annual(average(id,s.best),0).gpa,10);}
});
test('two different school systems reproduce the existing 8.92 example without premature rounding',()=>{
  const result=g.calculate({graduation:2027,years:[average('french','16,6'),average('numeric10',9)]});
  close(result.years[0].gpa,8.64);close(result.years[1].gpa,9.2);close(result.model_gpa,8.92);assert.equal(result.version,2);assert.equal(result.official,false);assert.deepEqual(result.years.map(y=>y.year),[2025,2026]);
});
test('reversed and custom numeric scales work in either direction',()=>{
  close(g.annual(average('reversed',2.5),0).gpa,8);close(g.annual({mode:'average',scale:'custom',pass:5,best:10,value:9},0).gpa,9.2);
  assert.throws(()=>g.annual({mode:'average',scale:'custom',pass:5,best:5,value:5},0),/different/);
});
test('official entries bypass conversion but a mixed input remains an estimate',()=>{
  const official={mode:'official',value:9.2},result=g.calculate({graduation:2027,years:[{...official,value:8.8},official]});
  close(result.model_gpa,9);assert.equal(result.official,true);assert.ok(result.years.every(y=>y.source==='official_portal'));
  assert.equal(g.calculate({graduation:2027,years:[official,average('numeric6',5)]}).official,false);
  for(const value of [5.9,10.1,''])assert.throws(()=>g.annual({mode:'official',value},0));
});
test('failing subject grades are flagged and treated as passing; an annual average below passing is rejected',()=>{
  const result=g.annual({mode:'subjects',scale:'french',grades:'8;20'},0);close(result.gpa,8);assert.equal(result.clamped,1);assert.equal(result.count,2);
  assert.throws(()=>g.annual(average('french',8),0),/below the passing/);
});
test('grades better than the maximum never silently become 10',()=>{
  assert.throws(()=>g.annual(average('french',21),0),/maximum/);assert.throws(()=>g.annual(average('reversed',.5),0),/maximum/);
});
test('decimal commas and newline, tab or semicolon grade lists stay unambiguous',()=>{
  close(g.annual({mode:'subjects',scale:'french',grades:'16,6\n17,6;15\t20'},0).gpa,(8.64+9.04+8+10)/4);
  for(const value of ['0x10','1e2','Infinity','NaN','4,5,6',''])assert.throws(()=>g.number(value,'value',0));
});
test('missing values and scales identify the year and field for inline validation',()=>{
  assert.throws(()=>g.annual({mode:'average',scale:'',value:5},1),e=>e.year===1&&e.field==='scale');
  assert.throws(()=>g.annual({mode:'subjects',scale:'ib',grades:''},0),e=>e.year===0&&e.field==='grades');
  assert.throws(()=>g.annual({mode:'average',scale:'custom',pass:'',best:15,value:10},0),e=>e.field==='pass');
});
test('letter grades and mixed scales do not receive an invented official mapping',()=>{
  assert.throws(()=>g.annual({mode:'subjects',scale:'letters',grades:'A;B'},0),/Bocconi/);
});
test('11-year diplomas use the years of grades 10 and 11 instead of the standard date labels',()=>{
  assert.deepEqual(g.yearEnds(2027),[2025,2026]);assert.deepEqual(g.yearEnds(2027,11),[2026,2027]);
  for(const value of ['',2027.5,1900])assert.throws(()=>g.yearEnds(value));
  assert.throws(()=>g.calculate({graduation:2027,years:[]}));
});
