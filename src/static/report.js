/* Inline homepage result and historical comparison. Advanced report: qa/advanced-report-for-later/. */
const ChanceReport = (function () {
  let current = null, programs = [], sequence = 0;
  const fullPage = document.body.dataset.page === "report";
  const byId = id => document.getElementById(id);
  function close() {
    document.body.classList.remove("report-mode");
    byId("calculator").hidden = false; byId("evidence").hidden = true;
    if (current) current.visible = false;
  }
  function reset() {
    if (current) { if (current.request) current.request.abort(); if (current.exportRequest) current.exportRequest.abort(); if (current.animation) current.animation.cancel(); }
    current = null; sequence++; close();
  }
  function open(options = {}) {
    if (!current) return;
    current.visible = true; document.body.classList.toggle("report-mode",fullPage);
    byId("calculator").hidden = fullPage; byId("evidence").hidden = false; byId("evidence").open = true;
    byId("evidence").classList.toggle("inline-report",!fullPage);
    if(options.scroll)byId("evidence").scrollIntoView({behavior:reduceMotion.matches?"instant":"smooth",block:"start"});
  }
  function button(text, className, handler) {
    const b=node("button",text,className); b.type="button";if(handler)b.addEventListener("click",handler);return b;
  }
  function reportURL(input, cycle) {
    const params=new URLSearchParams();
    ["test_type","score","gpa","course","session","cycle","gpa_estimated"].forEach(key=>{if(input[key]!==undefined)params.set(key,String(input[key]));});
    if(cycle)params.set("year",cycle);
    return "/report/#"+params.toString();
  }
  function csvCell(value) {
    let s=String(value == null ? "" : value);
    if (typeof value === "string" && /^[=+\-@]/.test(s)) s="'"+s;
    return '"'+s.replace(/"/g,'""')+'"';
  }
  function download(blob, filename) {
    const url=URL.createObjectURL(blob),link=node("a");link.href=url;link.download=filename;
    document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function viewBounds(reports, input) {
    const sat=input.test_type==="SAT",xs=reports.map(r=>r.score).concat(input.score),ys=reports.map(r=>r.gpa).concat(input.gpa);
    const span=Math.max(...xs)-Math.min(...xs),step=sat?(span>400?100:50):(span>20?5:2);
    return {xmin:Math.max(sat?400:-11.17,Math.floor((Math.min(...xs)-step*.4)/step)*step),xmax:Math.min(sat?1600:50,Math.ceil((Math.max(...xs)+step*.4)/step)*step),ymin:Math.max(6,Math.floor((Math.min(...ys)-.2+1e-9)*2)/2),ymax:Math.min(10,Math.ceil((Math.max(...ys)+.2)*2)/2)};
  }
  function clampView(v, input) {
    const sat=input.test_type==="SAT",lo=sat?400:-11.17,hi=sat?1600:50;
    const xspan=Math.min(hi-lo,Math.max(sat?40:2,v.xmax-v.xmin)),yspan=Math.min(4,Math.max(.4,v.ymax-v.ymin));
    const xmin=Math.max(lo,Math.min(hi-xspan,v.xmin)),ymin=Math.max(6,Math.min(10-yspan,v.ymin));
    return {xmin,xmax:xmin+xspan,ymin,ymax:ymin+yspan};
  }
  function zoomView(v,input,factor) {
    const cx=(v.xmin+v.xmax)/2,cy=(v.ymin+v.ymax)/2,dx=(v.xmax-v.xmin)*factor/2,dy=(v.ymax-v.ymin)*factor/2;
    return clampView({xmin:cx-dx,xmax:cx+dx,ymin:cy-dy,ymax:cy+dy},input);
  }
  function render(data, input, options = {}) {
    reset(); const own={input:{...input},data,course:input.course,session:input.session,cycle:options.cycle||"",dataset:null,view:null,expanded:false,showAll:false,visible:true,request:null,requestId:0};current=own;
    const body=byId("evidence-body");body.replaceChildren();
    const headingRow=node("div",undefined,"report-heading-row");const heading=node(fullPage?"h1":"h2","Your result, in context","report-title");heading.tabIndex=-1;own.heading=heading;headingRow.append(heading);
    const exports=node("div",undefined,"report-exports"),exportStatus=node("p",undefined,"report-export-status");exportStatus.setAttribute("role","status");
    const pdf=button("Download PDF ↓",fullPage?"report-download":"text-button report-pdf"),full=node("a","See full report ↗","report-download report-full-link");
    pdf.disabled=true;full.target="_blank";full.rel="noopener noreferrer";full.setAttribute("aria-label","See full report (opens in a new tab)");
    if(fullPage)exports.append(pdf);else exports.append(pdf,full);headingRow.append(exports);
    const profile=node("section",undefined,"report-profile-bar");profile.hidden=!fullPage;profile.setAttribute("aria-label","Your calculated result");
    const terms=node("div",undefined,"report-profile-terms"),termValues=[];
    [[input.test_type==="SAT"?"SAT score":"Bocconi Test",format(input.score)],["Bocconi GPA",Number(input.gpa).toLocaleString("en-GB",{minimumFractionDigits:1,maximumFractionDigits:3})],["Programme",input.course],["Round",input.session]].forEach(pair=>{const item=node("div");const value=node("strong",pair[1]);termValues.push(value);item.append(node("span",pair[0]),value);terms.append(item);});profile.append(terms);
    const chance=node("div",undefined,"report-chance");const mini=node("div",undefined,"report-mini-ring");
    const ring=svgNode("svg",{viewBox:"0 0 110 110","aria-hidden":"true"});ring.append(svgNode("circle",{cx:55,cy:55,r:45,class:"report-ring-track"}));
    const arc=svgNode("circle",{cx:55,cy:55,r:45,pathLength:100,"stroke-dasharray":data.probability*100+" 100",class:"report-ring-value"});ring.append(arc);
    mini.style.setProperty("--report-ring",data.probability<.35?"#cc8392":data.probability<.65?"#dbb66a":"#8fc7b4");const miniValue=node("strong",data.display_percent),chanceLabel=node("span",input.gpa_estimated?"GPA simulation":"Estimated chance");mini.append(ring,miniValue);chance.append(mini,chanceLabel);profile.append(chance);
    const feedback=node("p",undefined,"report-profile-feedback");profile.append(feedback);
    function showEstimate(estimate) {
      profile.setAttribute("aria-busy","false");
      termValues[2].textContent=own.input.course;termValues[3].textContent=own.input.session;
      const formal=estimate.status==="ineligible",p=estimate.probability;
      miniValue.textContent=formal?"—":estimate.display_percent;
      chanceLabel.textContent=formal?"Not eligible":own.input.gpa_estimated?"GPA simulation":"Estimated chance";
      feedback.textContent=formal?"Test requirements not met.":p>=.65?"Your scores compare well with past applicants.":p>=.35?"Admission could go either way with these scores.":"A higher test score could improve your position.";
      mini.style.setProperty("--report-ring",p<.35?"#cc8392":p<.65?"#dbb66a":"#8fc7b4");
      arc.setAttribute("stroke-dasharray",p*100+" 100");
      if(own.animation)own.animation.cancel();
      if(!reduceMotion.matches&&typeof arc.animate==="function")own.animation=arc.animate([{strokeDasharray:"0 100"},{strokeDasharray:p*100+" 100"}],{duration:1100,easing:"cubic-bezier(.22,.65,.25,1)"});
    }
    showEstimate(data);
    profile.append(button("✎  Edit scores","report-edit",()=>{if(fullPage){window.location.assign("/#calculator");return;}close();form.scrollIntoView({behavior:reduceMotion.matches?"instant":"smooth",block:"center"});byId("score").focus({preventScroll:true});}));
    const explorer=node("section",undefined,"report-explorer");
    const explorerHeading=node("div",undefined,"report-explorer-heading");explorerHeading.append(node("h3","Explore the data"));explorerHeading.append(node("span",undefined,"report-dataset-count"));explorer.append(explorerHeading);
    const filters=node("div",undefined,"report-filters");
    function filter(label,id,options,value){const wrap=node("label",undefined,"report-filter");wrap.append(node("span",label));const select=node("select");select.id=id;select.setAttribute("aria-label",label);options.forEach(pair=>{const option=node("option",pair[1]);option.value=pair[0];select.append(option);});select.value=value;wrap.append(select);filters.append(wrap);return select;}
    const course=filter("Programme","report-course",programs.map(p=>[p.id,p.name]),own.course);
    const round=filter("Round","report-round",[["Early","Early"],["Winter","Winter"]],own.session);
    const cycle=filter("Admission cycle","evidence-cycle",[["","All cycles"]],"");
    const help=button("How similarity is defined ↗","text-button report-similarity");filters.append(help);explorer.append(filters);
    const error=node("div",undefined,"report-data-error");error.hidden=true;error.setAttribute("role","status");explorer.append(error);
    const grid=node("div",undefined,"context-grid report-grid");const card=node("section",undefined,"context-chart-card report-chart-card");
    const chartHeader=node("div",undefined,"report-chart-heading");chartHeader.append(node("h3","Where your profile sits"));
    let all,near;
    function zoomBy(factor){if(!own.dataset)return;own.view=zoomView(own.view||viewBounds(rows(),input),input,factor);draw();}
    if(fullPage){
      const expand=button("⛶  Expand","text-button",()=>{own.expanded=!own.expanded;grid.classList.toggle("chart-expanded",own.expanded);expand.textContent=own.expanded?"↙  Collapse":"⛶  Expand";expand.setAttribute("aria-pressed",String(own.expanded));draw();});expand.setAttribute("aria-pressed","false");chartHeader.append(expand);
    }
    card.append(chartHeader);
    if(fullPage){
      const controls=node("div",undefined,"report-chart-controls"),ranges=node("div",undefined,"report-chart-ranges");
      all=button("All scores","is-active",()=>{own.view=null;draw();});near=button("Near my scores","",()=>{const dx=input.test_type==="SAT"?100:6;own.view=clampView({xmin:input.score-dx,xmax:input.score+dx,ymin:input.gpa-.6,ymax:input.gpa+.6},input);draw();});ranges.append(all,near);controls.append(ranges);
      const zoom=node("div",undefined,"report-chart-zoom"),minus=button("−","",()=>zoomBy(1.35)),plus=button("+","",()=>zoomBy(.74)),restore=button("Reset","",()=>{own.view=null;draw();});minus.setAttribute("aria-label","Zoom out");plus.setAttribute("aria-label","Zoom in");zoom.append(minus,plus,restore);controls.append(zoom);card.append(controls);
    }
    const host=node("div",undefined,"report-plot");host.setAttribute("role","group");host.setAttribute("aria-label","Past applicants’ scores and outcomes. Select a point for details.");if(fullPage){host.tabIndex=0;host.setAttribute("aria-label","Score chart. Drag to pan; use arrow keys to move or plus and minus to zoom.");}card.append(host);
    const legend=node("div",undefined,"chart-legend");[["Admitted","admitted"],["Not admitted","rejected"],["Your profile","you"]].forEach(pair=>{const item=node("span");item.append(node("i",undefined,"legend-mark "+pair[1]),node("span",pair[0]));legend.append(item);});card.append(legend);
    const summary=node("section",undefined,"context-summary");summary.setAttribute("aria-label","Similar profile counts");grid.append(card,summary);explorer.append(grid);
    const reports=node("section",undefined,"context-reports");const reportsHeading=node("div",undefined,"reports-heading");reportsHeading.append(node("h3","Individual reports"));const shown=node("span");reportsHeading.append(shown);reports.append(reportsHeading);const tableHost=node("div");tableHost.id="report-table";reports.append(tableHost);const more=button("","text-button reports-more");more.setAttribute("aria-controls","report-table");reports.append(more);explorer.append(reports);
    const about=node("details",undefined,"context-about-data");about.append(node("summary","How these reports are selected"));const aboutBody=node("div");about.append(aboutBody);explorer.append(about);if(fullPage)body.append(headingRow,exportStatus,profile,explorer);else body.append(explorer,headingRow,exportStatus);
    help.addEventListener("click",()=>{about.open=true;about.scrollIntoView({behavior:reduceMotion.matches?"instant":"smooth",block:"center"});});
    const tooltip=node("div",undefined,"chart-tooltip");tooltip.id="chart-report-tooltip";tooltip.setAttribute("role","tooltip");tooltip.hidden=true;host.append(tooltip);
    const pointRecords=new WeakMap();let drag=null,frame=null;
    function rows(){const records=own.dataset?evidenceRecords(own.dataset,own.input,own.cycle):[];return fullPage?records:records.slice(0,20);}
    function syncFullLink(){full.href=reportURL(own.input,own.cycle);if(fullPage)window.history.replaceState(null,"",full.href);}
    function hideTip(){tooltip.hidden=true;}
    own.hideTip=hideTip;
    function tip(reports,event,anchor){
      tooltip.replaceChildren();const first=reports[0];const title=node("div",undefined,"chart-tooltip-heading");title.append(node("strong",reports.length>1?reports.length+" reports":first.outcome==="admitted"?"Admitted":"Not admitted"));title.append(button("×","chart-tooltip-close",hideTip));tooltip.append(title);
      const kv=node("div",undefined,"chart-tooltip-scores");kv.append(node("span",input.test_type==="SAT"?"SAT":"Bocconi Test"),node("strong",format(first.score)),node("span","Bocconi GPA"),node("strong",format(first.gpa)));tooltip.append(kv);
      reports.forEach(r=>{const line=node("div",undefined,"chart-tooltip-outcome "+r.outcome);line.append(node("span",year(r.cycle)+" · "+r.session),node("strong",r.outcome==="admitted"?"✓ Admitted":"× Not admitted"));tooltip.append(line);});
      tooltip.hidden=false;const box=host.getBoundingClientRect(),target=anchor.getBoundingClientRect();const px=event&&finite(event.clientX)?event.clientX-box.left:target.left-box.left+target.width/2,py=event&&finite(event.clientY)?event.clientY-box.top:target.top-box.top;
      const w=tooltip.offsetWidth||232,h=tooltip.offsetHeight||150;tooltip.style.left=Math.max(5,Math.min(box.width-w-5,px+14))+"px";tooltip.style.top=Math.max(5,Math.min(box.height-h-5,py-h-12))+"px";
    }
    function draw(){
      hideTip();host.querySelectorAll("svg").forEach(el=>el.remove());if(!own.dataset)return;
      // The homepage table and chart share exactly the same nearest reports.
      const records=rows(),bounds=own.view||viewBounds(records,input);const compact=evidenceViewport.matches,W=compact?360:own.expanded?1000:700,H=compact?310:own.expanded?470:335,L=compact?47:57,R=20,T=34,B=48;
      const X=n=>L+(n-bounds.xmin)/(bounds.xmax-bounds.xmin)*(W-L-R),Y=n=>T+(bounds.ymax-n)/(bounds.ymax-bounds.ymin)*(H-T-B);
      const svg=svgNode("svg",{viewBox:"0 0 "+W+" "+H,role:"group","aria-label":records.length+" past applicant reports. "+(input.test_type==="SAT"?"SAT":"Bocconi Test")+" score versus Bocconi GPA."});
      const defs=svgNode("defs"),clip=svgNode("clipPath",{id:"report-plot-clip"});clip.append(svgNode("rect",{x:L-9,y:T-9,width:W-L-R+18,height:H-T-B+18}));defs.append(clip);svg.append(defs);
      function niceStep(raw){const base=Math.pow(10,Math.floor(Math.log10(raw))),fraction=raw/base;return [1,2,2.5,5,10].find(n=>n>=fraction)*base;}
      const xstep=Math.max(input.test_type==="SAT"?10:.1,niceStep((bounds.xmax-bounds.xmin)/6)),ystep=Math.max(.1,niceStep((bounds.ymax-bounds.ymin)/5));
      for(let n=Math.ceil(bounds.xmin/xstep)*xstep;n<=bounds.xmax+1e-7;n+=xstep){const xx=X(n);svg.append(svgNode("line",{x1:xx,x2:xx,y1:T,y2:H-B,class:"chart-grid"}),svgNode("text",{x:xx,y:H-B+23,"text-anchor":"middle",class:"chart-tick"},input.test_type==="SAT"?Math.round(n):Number(n.toFixed(1))));}
      for(let n=Math.ceil(bounds.ymin/ystep)*ystep;n<=bounds.ymax+1e-7;n+=ystep){const yy=Y(n);svg.append(svgNode("line",{x1:L,x2:W-R,y1:yy,y2:yy,class:"chart-grid"}),svgNode("text",{x:L-12,y:yy+4,"text-anchor":"end",class:"chart-tick"},n.toFixed(1)));}
      svg.append(svgNode("text",{x:(L+W-R)/2,y:H-6,"text-anchor":"middle",class:"chart-axis"},input.test_type==="SAT"?"SAT score":"Bocconi Test score"),svgNode("text",{x:15,y:(T+H-B)/2,transform:"rotate(-90 15 "+((T+H-B)/2)+")","text-anchor":"middle",class:"chart-axis"},"Bocconi GPA"));
      const plot=svgNode("g",{"clip-path":"url(#report-plot-clip)"});const buckets=new Map();records.forEach(r=>{const key=r.score+":"+r.gpa;if(!buckets.has(key))buckets.set(key,[]);buckets.get(key).push(r);});
      buckets.forEach(group=>{const r=group[0],px=X(r.score),py=Y(r.gpa);if(px<L-8||px>W-R+8||py<T-8||py>H-B+8)return;const admitted=group.some(r=>r.outcome==="admitted"),rejected=group.some(r=>r.outcome==="rejected");
        const point=svgNode("g",{class:"chart-report",tabindex:"0",role:"button","aria-describedby":tooltip.id,"aria-label":group.map(r=>(r.outcome==="admitted"?"Admitted":"Not admitted")+", "+r.score+", GPA "+r.gpa+", "+year(r.cycle)+", "+r.session).join("; ")});point.append(svgNode("circle",{cx:px,cy:py,r:12,class:"chart-hit"}));if(admitted)point.append(svgNode("circle",{cx:px,cy:py,r:rejected?7:5,class:"report-dot-admitted"}));if(rejected)point.append(svgNode("path",{d:"M"+(px-4)+","+(py-4)+" L"+(px+4)+","+(py+4)+" M"+(px+4)+","+(py-4)+" L"+(px-4)+","+(py+4),class:"report-dot-rejected"}));
        point.addEventListener("click",e=>{if(!own.wasDrag)tip(group,e,point);});pointRecords.set(point,group);
        point.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();tip(group,null,point);}});plot.append(point);
      });
      const ux=X(input.score),uy=Y(input.gpa);if(ux>=L&&ux<=W-R&&uy>=T&&uy<=H-B){plot.append(svgNode("line",{x1:ux,x2:ux,y1:uy+10,y2:H-B,class:"chart-you-guide"}),svgNode("path",{d:"M"+ux+","+(uy-8)+" L"+(ux+8)+","+uy+" L"+ux+","+(uy+8)+" L"+(ux-8)+","+uy+" Z",class:"chart-you"}),svgNode("text",{x:ux,y:uy-T<24?uy+24:uy-16,"text-anchor":"middle",class:"chart-you-label"},"You"));}svg.append(plot);host.insertBefore(svg,tooltip);if(fullPage){all.classList.toggle("is-active",!own.view);near.classList.toggle("is-active",Boolean(own.view));}
    }
    function renderRows(){tableHost.replaceChildren();const records=rows(),limit=fullPage?(evidenceViewport.matches?4:8):6,visible=own.showAll?records:records.slice(0,limit);shown.textContent=visible.length+" of "+records.length;
      if(visible.length)table(tableHost,[input.test_type==="SAT"?"SAT":"Bocconi Test","GPA","Cycle","Reported outcome","Round"],visible.map(r=>{const outcome=node("span",undefined,"outcome "+(r.outcome==="rejected"?"rejected":"admitted"));outcome.append(node("span",r.outcome==="admitted"?"✓":"×"),node("span",r.outcome==="admitted"?"Admitted":"Not admitted"));const wrap=node("div");wrap.append(outcome);if(r.used_for_bt_fit===false){const note=node("small","Comparison only","comparison-only");note.title="This report was not used to fit the Bocconi Test model.";wrap.append(note);}return[format(r.score),format(r.gpa),year(r.cycle),wrap,r.session];}));else tableHost.append(node("p","No reports for these filters.","reports-empty"));more.hidden=records.length<=limit;more.textContent=fullPage?(own.showAll?"Show fewer ↑":"Show all "+records.length+" reports ↓"):(own.showAll?"Show less ↑":"Show more ↓");more.setAttribute("aria-expanded",String(own.showAll));
    }
    function update(){if(!own.dataset)return;const counts=evidenceCounts(own.dataset,own.cycle),records=rows();explorerHeading.lastChild.textContent=records.length+" reports";summary.replaceChildren();summary.append(node("h4","Similar profiles"));if(counts){const number=node("p",undefined,"context-big-count");number.append(node("strong",counts.total?counts.admitted+" of "+counts.total:"0"));summary.append(number,node("p",counts.total?"reported admission":"similar reports","context-count-label"));const list=node("dl",undefined,"context-outcomes");[["Admitted",counts.admitted,"admitted"],["Not admitted",counts.rejected,"rejected"]].forEach(pair=>{const row=node("div"),dt=node("dt");dt.append(node("i",undefined,"legend-mark "+pair[2]),node("span",pair[0]));row.append(dt,node("dd",pair[1]));list.append(row);});summary.append(list);}summary.append(node("p","Self-reported results. This share is not your admission probability.","context-summary-note"));
      aboutBody.replaceChildren(node("p",fullPage?"The chart includes all reports for the selected filters.":"The chart and table show up to 20 reports closest to your test score and GPA for the selected filters."),node("p","Similar = within ±"+own.dataset.nearby.test_window+" "+(input.test_type==="SAT"?"SAT":"Bocconi Test")+" points and ±0.5 GPA, for the programme, round and cycle selected above."));if(own.dataset.proxy_notice)aboutBody.append(node("p",own.dataset.proxy_notice));aboutBody.append(node("p","Reports are self-reported; their GPA and applicant category are not independently verified."));if(records.some(r=>r.used_for_bt_fit===false))aboutBody.append(node("p","‘Comparison only’ means a report was not used to fit the Bocconi Test model."));draw();renderRows();pdf.disabled=own.data.status==="ineligible";full.removeAttribute("aria-disabled");syncFullLink();
    }
    async function load(recalculate=false) {
      const id=++own.requestId,target={...own.input,course:course.value,session:round.value},previousDataset=own.dataset;
      if(own.request)own.request.abort();if(own.exportRequest)own.exportRequest.abort();const controller=new AbortController();own.request=controller;
      const timeout=setTimeout(()=>controller.abort(),20000);
      error.hidden=false;error.replaceChildren(node("span",recalculate?"Updating result…":"Loading reports…"));pdf.disabled=true;full.removeAttribute("href");full.setAttribute("aria-disabled","true");
      own.dataset=null;hideTip();host.querySelectorAll("svg").forEach(el=>el.remove());tableHost.replaceChildren();more.hidden=true;
      summary.replaceChildren(node("h4","Similar profiles"),node("p","…","context-big-count"));explorerHeading.lastChild.textContent="";
      if(recalculate){miniValue.textContent="…";arc.setAttribute("stroke-dasharray","0 100");profile.setAttribute("aria-busy","true");feedback.textContent="Updating estimate…";}
      async function request(url){const res=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(target),signal:controller.signal});const result=await res.json();if(!res.ok)throw new Error("Unavailable service");return result;}
      const [comparison,estimate]=await Promise.allSettled([request("/api/reports"),recalculate?request("/api/estimate"):Promise.resolve(own.data)]);
      clearTimeout(timeout);if(current!==own||id!==own.requestId)return;
      const d=estimate.status==="fulfilled"?estimate.value:null;
      if(!d||!["estimated","ineligible"].includes(d.status||"estimated")||!finite(d.probability)||d.probability<0||d.probability>1||typeof d.display_percent!=="string"||!/^(?:<1|>99|100|\d{1,2})%$/.test(d.display_percent)){
        course.value=own.input.course;round.value=own.input.session;own.dataset=previousDataset;showEstimate(own.data);if(own.dataset)update();
        error.replaceChildren(node("span","Couldn’t update the estimate."),button("Try again","text-button",()=>{course.value=target.course;round.value=target.session;load(true);}));return;
      }
      own.input=target;own.course=target.course;own.session=target.session;own.data=d;if(recalculate)showEstimate(d);
      if(recalculate){if(byId("course"))byId("course").value=target.course;if(byId("session"))byId("session").value=target.session;renderResult(d,target,{skipEvidence:true});}
      const reports=comparison.status==="fulfilled"?comparison.value:null;
      if(!reports||reports.status!=="ok"||!Array.isArray(reports.records)){
        summary.replaceChildren(node("p","Reports unavailable."));error.replaceChildren(node("span","Couldn’t load the comparison data."),button("Try again","text-button",()=>load()));return;
      }
      own.dataset=reports;const previous=own.cycle;cycle.replaceChildren();const allOption=node("option","All cycles");allOption.value="";cycle.append(allOption);
      reports.by_year.slice().reverse().forEach(r=>{const o=node("option",year(r.cycle));o.value=r.cycle;cycle.append(o);});
      own.cycle=reports.by_year.some(r=>r.cycle===previous)?previous:"";cycle.value=own.cycle;own.view=null;own.showAll=false;error.hidden=true;update();
    }
    course.addEventListener("change",()=>load(true));round.addEventListener("change",()=>load(true));
    cycle.addEventListener("change",()=>{own.cycle=cycle.value;own.view=null;own.showAll=false;update();});
    more.addEventListener("click",()=>{own.showAll=!own.showAll;renderRows();});
    host.addEventListener("keydown",e=>{if(e.key==="Escape")hideTip();});
    if(fullPage){
      host.addEventListener("keydown",e=>{
        if(e.key==="Escape"){hideTip();return;}if(e.target!==host)return;
        if(["+","="].includes(e.key)){e.preventDefault();zoomBy(.74);}else if(e.key==="-"){e.preventDefault();zoomBy(1.35);}
        else if(["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)&&own.dataset){e.preventDefault();const v=own.view||viewBounds(rows(),input),dx=(v.xmax-v.xmin)*.1,dy=(v.ymax-v.ymin)*.1,sx=e.key==="ArrowRight"?dx:e.key==="ArrowLeft"?-dx:0,sy=e.key==="ArrowUp"?dy:e.key==="ArrowDown"?-dy:0;own.view=clampView({xmin:v.xmin+sx,xmax:v.xmax+sx,ymin:v.ymin+sy,ymax:v.ymax+sy},input);draw();}
      });
      host.addEventListener("wheel",e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();zoomBy(e.deltaY>0?1.15:.87);}},{passive:false});
      host.addEventListener("pointerdown",e=>{if(!own.dataset||e.button!==0||e.target.closest(".chart-tooltip"))return;own.wasDrag=false;drag={x:e.clientX,y:e.clientY,view:{...(own.view||viewBounds(rows(),input))},box:host.getBoundingClientRect(),id:e.pointerId};});
      host.addEventListener("pointermove",e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)<5&&!own.wasDrag)return;if(!own.wasDrag)host.setPointerCapture(drag.id);own.wasDrag=true;const v=drag.view;own.view=clampView({xmin:v.xmin-dx/drag.box.width*(v.xmax-v.xmin),xmax:v.xmax-dx/drag.box.width*(v.xmax-v.xmin),ymin:v.ymin+dy/drag.box.height*(v.ymax-v.ymin),ymax:v.ymax+dy/drag.box.height*(v.ymax-v.ymin)},input);if(!frame)frame=requestAnimationFrame(()=>{frame=null;draw();});});
      function endDrag(){if(drag&&host.hasPointerCapture(drag.id))host.releasePointerCapture(drag.id);drag=null;}host.addEventListener("pointerup",endDrag);host.addEventListener("pointercancel",endDrag);host.addEventListener("pointerleave",()=>{if(drag&&!own.wasDrag)drag=null;});
    }
    pdf.addEventListener("click",async()=>{pdf.disabled=true;pdf.textContent="Creating PDF…";exportStatus.textContent="";own.exportRequest=new AbortController();try{const res=await fetch("/api/report.pdf",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({profile:own.input,filters:{course:own.course,session:own.session,cycle:own.cycle}}),signal:own.exportRequest.signal});if(!res.ok)throw new Error("PDF unavailable");const blob=await res.blob();if(!blob.size||blob.type.split(";")[0].toLowerCase()!=="application/pdf")throw new Error("Invalid PDF payload");if(current===own&&own.visible){download(blob,"chance-me-report.pdf");window.ChanceAnalytics?.track('pdf_download',{destination:'pdf'});}}catch(e){if(e.name!=="AbortError")exportStatus.textContent="The PDF couldn’t be created. Please try again.";}finally{pdf.disabled=!own.dataset;pdf.textContent="Download PDF ↓";}});
    own.redraw=()=>{draw();renderRows();};open();load();
  }
  document.addEventListener("click",function(e){if(current&&current.hideTip&&!e.target.closest(".chart-report, .chart-tooltip"))current.hideTip();});
  document.addEventListener("keydown",function(e){if(e.key==="Escape"&&current&&current.hideTip)current.hideTip();});
  return {render,open,close,reset,setPrograms:list=>{programs=list.slice();},redraw:()=>{if(current&&current.dataset)current.redraw();},csvCell,reportURL,viewBounds,clampView,zoomView};
})();
