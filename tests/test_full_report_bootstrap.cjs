const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
class Element {
  constructor(tag,text=''){this.tag=tag;this.text=text;this.children=[];this.listeners={};}
  set textContent(value){this.text=value;this.children=[];} get textContent(){return this.text+this.children.map(c=>c.textContent).join('');}
  append(...children){this.children.push(...children);} replaceChildren(...children){this.text='';this.children=[];this.append(...children);}
  addEventListener(name,handler){this.listeners[name]=handler;}
}
async function setup(hash,fetcher){
  const status=new Element('div'),requests=[],renders=[];
  const scope={URLSearchParams,Number,AbortController,setTimeout,clearTimeout,campusReady:Promise.resolve(),ready:true,finite:Number.isFinite,
    window:{location:{hash}},document:{getElementById:()=>status},node:(tag,text)=>new Element(tag,text),
    ChanceReport:{render:(...args)=>renders.push(args)},fetch:async(url,init)=>{requests.push({url,body:JSON.parse(init.body)});return fetcher();}};
  const completion=vm.runInNewContext(fs.readFileSync('dist/full-report.js','utf8'),scope);await completion;
  return {status,requests,renders};
}
const fragment='#test_type=SAT&score=1450&gpa=9&course=Finance&session=Early&year=26-27&gpa_estimated=true&probability=.99';
test('opening a report without a profile gives a calculator link and makes no estimate request',async()=>{
  const x=await setup('',()=>{throw new Error('must not fetch');});assert.equal(x.requests.length,0);assert.equal(x.renders.length,0);assert.match(x.status.textContent,/Calculate your result first/);assert.equal(x.status.children.at(-1).href,'/#calculator');
});
test('report fragment carries the profile and historical year while estimate comes from the server',async()=>{
  const x=await setup(fragment,async()=>({ok:true,json:async()=>({status:'estimated',probability:.54,display_percent:'54%'})}));
  assert.equal(x.requests[0].url,'/api/estimate');assert.equal(x.requests[0].body.score,1450);assert.equal(x.requests[0].body.course,'Finance');assert.equal(x.requests[0].body.gpa_estimated,true);assert.equal(x.requests[0].body.cycle,'27-28');
  assert.equal(x.renders[0][0].probability,.54);assert.equal(x.renders[0][2].cycle,'26-27');assert.equal(x.status.hidden,true);
});
test('failed report loads show a working retry and never render a guessed percentage',async()=>{
  let offline=true;const x=await setup(fragment,async()=>{if(offline)throw new Error('offline');return{ok:true,json:async()=>({status:'estimated',probability:.54,display_percent:'54%'})};});
  assert.equal(x.renders.length,0);assert.match(x.status.textContent,/couldn’t be loaded/);offline=false;await x.status.children.find(c=>c.tag==='button').listeners.click();assert.equal(x.renders.length,1);assert.equal(x.status.hidden,true);
});
