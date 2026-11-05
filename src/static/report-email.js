/* Transactional email only: addresses and codes stay out of URLs, storage and analytics. */
window.ChanceReportEmail = (function () {
  let availability;
  function available() {
    if (!availability) availability = fetch('/api/report-email/status', {cache:'no-store'})
      .then(async r => r.ok && (await r.json()).enabled === true).catch(() => false);
    return availability;
  }
  function mount(button, status, payload, active) {
    let controller, ident = '', busy = false;
    const form = document.createElement('form'); form.className = 'report-email-form'; form.hidden = true;
    const label = document.createElement('label'); label.textContent = 'Email address';
    const email = document.createElement('input'); email.type = 'email'; email.required = true; email.maxLength = 254;
    email.autocomplete = 'email'; label.append(email);
    const codeLabel = document.createElement('label'); codeLabel.textContent = 'Confirmation code'; codeLabel.hidden = true;
    const code = document.createElement('input'); code.inputMode = 'numeric'; code.autocomplete = 'one-time-code';
    code.pattern = '[0-9]{6}'; code.maxLength = 6; codeLabel.append(code);
    const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Send confirmation code';
    const restart = document.createElement('button'); restart.type = 'button'; restart.textContent = 'Use another email'; restart.hidden = true;
    const privacy = document.createElement('p'); privacy.append('Only for this requested report. No newsletter. Confirmation expires after 15 minutes. ');
    const link = document.createElement('a'); link.href = '/privacy/'; link.textContent = 'Privacy'; privacy.append(link);
    form.append(label,codeLabel,submit,restart,privacy); status.after(form);
    function reset() {
      if(controller)controller.abort(); controller = null; busy = false; ident = '';
      if(button.dataset.emailEnabled==='true')status.textContent='Receive your report by email after confirming your address. Includes at most 10 close profiles.';
      form.hidden = true; email.value = ''; email.readOnly = false; code.value = ''; code.required = false;
      codeLabel.hidden = true; restart.hidden = true; submit.disabled = false; submit.textContent = 'Send confirmation code';
    }
    restart.addEventListener('click', () => {reset(); form.hidden = false; email.focus();});
    button.addEventListener('click', () => {form.hidden = !form.hidden;if(!form.hidden)email.focus();});
    form.addEventListener('submit', async event => {
      event.preventDefault(); if(busy || !active() || !form.reportValidity())return;
      busy = true; submit.disabled = true; controller = new AbortController();
      const verifying = Boolean(ident), currentController = controller;
      const timeout = setTimeout(() => currentController.abort(), 25000);
      try {
        const response = await fetch(verifying ? '/api/report-email/verify' : '/api/report-email', {
          method:'POST',headers:{'Content-Type':'application/json'},signal:currentController.signal,
          body:JSON.stringify(verifying ? {request_id:ident,code:code.value} : {...payload(),email:email.value})
        });
        const result = await response.json();
        if(!active() || controller!==currentController)return;
        if(!response.ok)throw new Error(result.message || 'Email is temporarily unavailable. Please try again.');
        if(verifying && result.status==='accepted') {
          reset(); status.textContent = result.message; button.disabled = true; return;
        }
        if(result.status!=='confirmation_required' || !/^[a-zA-Z0-9_-]{43}$/.test(result.request_id))throw new Error('Please try again.');
        ident = result.request_id; email.readOnly = true; code.required = true; codeLabel.hidden = false;
        restart.hidden = false; submit.textContent = 'Confirm and email my PDF'; status.textContent = result.message; code.focus();
      } catch(error) {
        if(active() && controller===currentController)status.textContent = error.name==='AbortError' ? 'The request timed out. Wait one minute before retrying.' : error.message;
      } finally {
        clearTimeout(timeout);if(controller===currentController){busy = false;submit.disabled = false;}
      }
    });
    available().then(enabled => {
      if(!enabled || !active())return;
      button.dataset.emailEnabled = 'true'; button.disabled = false; button.textContent = 'Email my PDF';
      status.textContent = 'Receive your report by email after confirming your address. Includes at most 10 close profiles.';
    });
    return {reset};
  }
  return {mount};
})();
