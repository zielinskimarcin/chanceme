const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs');
class El {
 constructor(){this.children=[];this.listeners={};this.dataset={};this.value='';this.text='';this.disabled=false;}
 set textContent(v){this.text=v;this.children=[];}get textContent(){return this.text+this.children.map(e=>typeof e==='string'?e:e.textContent).join('');}
 append(...els){this.children.push(...els);}after(el){this.sibling=el;}addEventListener(n,f){this.listeners[n]=f;}focus(){this.focused=true;}
 reportValidity(){return true;}
}
const tick=()=>new Promise(r=>setImmediate(r));
function setup(fetcher){const calls=[],button=new El(),status=new El();button.disabled=true;
 const context={window:{},document:{createElement:()=>new El()},AbortController,setTimeout,clearTimeout,
 fetch:async(url,options)=>{calls.push({url,options});return fetcher(url,options);}};
 vm.runInNewContext(fs.readFileSync('src/static/report-email.js','utf8'),context);
 let active=true;const mount=context.window.ChanceReportEmail.mount(button,status,()=>({profile:{score:1450},filters:{course:'Finance'}}),()=>active);
 return {calls,button,status,form:status.sibling,mount,deactivate:()=>active=false};}
const ok=data=>({ok:true,json:async()=>data});
test('unconfigured service never enables capture or sends an address',async()=>{
 const x=setup(async()=>ok({enabled:false}));await tick();assert.equal(x.button.disabled,true);assert.equal(x.form.hidden,true);
 assert.deepEqual(x.calls.map(c=>c.url),['/api/report-email/status']);
});
test('email request then code verification use POST bodies and confirm provider acceptance',async()=>{
 const x=setup(async(url)=>ok(url.endsWith('status')?{enabled:true}:url.endsWith('verify')?{status:'accepted',message:'Submitted for delivery'}:{status:'confirmation_required',request_id:'a'.repeat(43),message:'Enter your code'}));
 await tick();assert.equal(x.button.disabled,false);x.button.listeners.click();assert.equal(x.form.hidden,false);
 const email=x.form.children[0].children[0];email.value='student@example.com';
 await x.form.listeners.submit({preventDefault(){}});assert.equal(x.form.children[1].hidden,false);
 assert.equal(email.readOnly,true);assert.equal(x.status.textContent,'Enter your code');
 const sent=JSON.parse(x.calls[1].options.body);assert.equal(sent.email,'student@example.com');assert.equal(sent.filters.course,'Finance');
 x.form.children[1].children[0].value='123456';await x.form.listeners.submit({preventDefault(){}});
 assert.deepEqual(JSON.parse(x.calls[2].options.body),{request_id:'a'.repeat(43),code:'123456'});
 assert.equal(x.status.textContent,'Submitted for delivery');assert.equal(x.form.hidden,true);assert.equal(email.value,'');
 assert.ok(x.calls.every(c=>!c.url.includes('student')&&!c.url.includes('123456')));
});
test('failure leaves code retryable and reset clears personal fields',async()=>{
 const x=setup(async(url)=>url.endsWith('status')?ok({enabled:true}):{ok:false,json:async()=>({message:'Please wait'})});await tick();
 const email=x.form.children[0].children[0];email.value='student@example.com';await x.form.listeners.submit({preventDefault(){}});
 assert.equal(x.status.textContent,'Please wait');assert.equal(x.form.children[2].disabled,false);x.mount.reset();
 assert.equal(email.value,'');assert.equal(x.form.hidden,true);
});
test('a stale report cannot enable an email form after switching away',async()=>{
 let resolve;const x=setup(()=>new Promise(r=>resolve=r));x.deactivate();resolve(ok({enabled:true}));await tick();
 assert.equal(x.button.disabled,true);
});
