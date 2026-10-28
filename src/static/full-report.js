/* Profile travels in the fragment, so it is not sent in the page URL or referrer. */
(async function () {
  const status=document.getElementById("full-report-status");
  function message(text,retry) {
    status.hidden=false;status.replaceChildren(node("p",text));
    if(retry){const button=node("button","Try again","text-button");button.type="button";button.addEventListener("click",load);status.append(button);}
    const back=node("a","Back to calculator");back.href="/#calculator";status.append(back);
  }
  async function load() {
    status.hidden=false;status.textContent="Loading your report…";
    await campusReady;
    if(!ready){message("The calculator is unavailable. Please refresh to try again.",false);return;}
    const params=new URLSearchParams(window.location.hash.slice(1));
    const score=params.get("score"),gpa=params.get("gpa"),course=params.get("course"),session=params.get("session"),kind=params.get("test_type");
    if(!score||!gpa||!course||!["Early","Winter"].includes(session)||!["SAT","Bocconi"].includes(kind)){
      message("Calculate your result first to open a full report.",false);return;
    }
    const input={test_type:kind,score:Number(score),gpa:Number(gpa),course,session,cycle:"27-28",applicant_category:"international",gpa_confirmed:true,gpa_estimated:params.get("gpa_estimated")==="true"};
    if(!Number.isFinite(input.score)||!Number.isFinite(input.gpa)){message("Check your scores in the calculator.",false);return;}
    const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),20000);
    try {
      const response=await fetch("/api/estimate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(input),signal:controller.signal});
      const data=await response.json();
      if(!response.ok||!["estimated","ineligible"].includes(data.status)||!finite(data.probability)||data.probability<0||data.probability>1||typeof data.display_percent!=="string")throw new Error("No estimate");
      ChanceReport.render(data,input,{cycle:params.get("year")||""});status.hidden=true;
    } catch (_) { message("The report couldn’t be loaded. Your original result is still open in the calculator tab.",true); }
    finally {clearTimeout(timeout);}
  }
  await load();
})();
