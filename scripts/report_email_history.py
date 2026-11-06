"""Owner-only HTML snapshot. Uses existing Vercel credentials; never a public route."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lib.report_email import Store
from lib.report_journal import Journal, display_status

LABELS={'requested':'Zgłoszenie zapisane','waiting_for_code':'Czeka na kod',
        'email_verified':'Adres potwierdzony','report_submitted':'PDF przekazany do wysyłki',
        'code_send_error':'Błąd próby wysyłki kodu','report_send_error':'Błąd próby wysyłki PDF',
        'code_locked':'Zablokowane po 5 błędnych kodach','expired':'Kod wygasł'}
DELIVERY={'sent':'Wysłano','delivered':'Doręczono','delivery_delayed':'Opóźnienie',
          'bounced':'Odrzucono','failed':'Błąd doręczenia','opened':'Otwarto',
          'clicked':'Kliknięto','complained':'Zgłoszono spam','queued':'W kolejce',
          'scheduled':'Zaplanowano','canceled':'Anulowano','suppressed':'Wstrzymano'}


def stamp(value):
    return datetime.fromtimestamp(value,timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if value else '—'


def delivery(record):
    ident=record.get('report_provider_id')
    if not ident:return '—'
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,100}',ident):return 'Nieprawidłowy identyfikator'
    req=Request('https://api.resend.com/emails/'+ident,headers={
        'Authorization':'Bearer '+os.environ['RESEND_API_KEY'],
        'User-Agent':'ChanceMe/2.0 (+https://chance-me.com)'})
    try:
        with urlopen(req,timeout=8) as response:data=json.load(response)
        # Do not associate provider data with a different mailbox/message.
        if data.get('to')!=[record['email']] or data.get('subject')!='Your Chance Me report':
            return 'Nie można potwierdzić'
        return DELIVERY.get(data.get('last_event'),'Nieznany status')
    except Exception:
        return 'Status chwilowo niedostępny'


def render(rows, include_tests=False):
    real=[r for r in rows if not r.get('test_request')]
    verified=sum(bool(r.get('email_verified_at')) for r in real)
    submitted=sum(bool(r.get('report_submitted_at')) for r in real)
    waiting=sum(display_status(r)=='waiting_for_code' for r in real)
    counts=''.join(f'<div><strong>{v}</strong><span>{escape(k)}</span></div>' for k,v in
                   [('Zgłoszenia',len(real)),('Potwierdzone adresy',verified),('PDF-y przekazane',submitted),('Czekają na kod',waiting)])
    table=[]
    for r in rows:
        status=display_status(r)
        table.append(f'''<tr data-test="{'true' if r.get('test_request') else 'false'}"><td>{escape(r['email'])}{'<small>Test</small>' if r.get('test_request') else ''}</td>
<td>{escape(LABELS.get(status,status))}{'<small>W toku przed uruchomieniem rejestru</small>' if r.get('source')=='inflight_at_release' else ''}</td>
<td>{stamp(r['created_at']) if r.get('source')!='inflight_at_release' else '—'}</td><td>{stamp(r.get('code_submitted_at'))}</td>
<td>{stamp(r.get('email_verified_at'))}</td><td>{stamp(r.get('report_submitted_at'))}</td>
<td>{escape(r.get('delivery_status','Nie sprawdzono'))}</td><td>{r.get('attempts',0)}</td></tr>''')
    return f'''<!doctype html><html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zgłoszenia raportów · Chance Me</title>
<style>*{{box-sizing:border-box}}body{{margin:0;padding:32px;color:#17324d;background:#f8fafc;font:14px/1.6 system-ui,sans-serif}}main{{max-width:1500px;margin:auto}}h1{{font-size:28px;margin:0 0 8px}}p{{color:#58718c;margin:8px 0}}.counts{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}.counts div{{background:white;border:1px solid #dbe6ef;border-radius:10px;padding:14px 20px;min-width:160px}}strong{{display:block;font-size:28px}}.counts span{{color:#58718c}}.tools{{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:20px 0}}input[type=search]{{padding:10px 14px;border:1px solid #bccfd4;border-radius:8px;font:inherit;min-width:260px}}.table{{overflow:auto;background:white;border:1px solid #dbe6ef;border-radius:10px}}table{{width:100%;border-collapse:collapse;text-align:left;white-space:nowrap}}th,td{{padding:13px 16px;border-bottom:1px solid #e7eef2;font-size:12px}}th{{font-weight:600;background:#edf7f2}}small{{display:block;color:#58718c}}#empty{{padding:18px}}[hidden]{{display:none!important}}@media(max-width:600px){{body{{padding:20px}}.counts{{gap:8px}}.counts div{{min-width:140px;flex:1}}}}</style></head>
<body><main><h1>Zgłoszenia raportów Chance Me</h1><p>Prywatny podgląd z {stamp(datetime.now(timezone.utc).timestamp())} UTC. To zapis stanu w chwili odczytu.</p>
<div class="counts">{counts}</div><p>Liczby powyżej pomijają testy. Przekazanie PDF-a do Resend i doręczenie to osobne etapy.</p>
<div class="tools"><input id="search" type="search" placeholder="Szukaj adresu e-mail" aria-label="Szukaj adresu e-mail"><label><input type="checkbox" id="tests" {'checked' if include_tests else ''}> Pokaż testy ({len(rows)-len(real)})</label></div>
<div class="table"><table><thead><tr><th>E-mail</th><th>Etap</th><th>Zgłoszenie (UTC)</th><th>Kod przekazany (UTC)</th><th>Adres potwierdzony (UTC)</th><th>PDF przekazany (UTC)</th><th>Doręczenie PDF</th><th>Błędne kody</th></tr></thead><tbody>{''.join(table)}</tbody></table><p id="empty">Brak zgłoszeń dla wybranych filtrów.</p></div>
<p>Adresy zostają w prywatnym rejestrze bez automatycznego wygaśnięcia. Rejestr nie zawiera kodów, PDF-ów, ocen ani IP. Wcześniejsze zgłoszenia nie są odtwarzane z domysłów.</p></main>
<script>const rows=[...document.querySelectorAll('tbody tr')],search=document.getElementById('search'),tests=document.getElementById('tests');function filter(){{let count=0;for(const r of rows){{r.hidden=(!tests.checked&&r.dataset.test==='true')||!r.cells[0].textContent.toLowerCase().includes(search.value.toLowerCase());if(!r.hidden)count++;}}document.getElementById('empty').hidden=count>0;}}search.addEventListener('input',filter);tests.addEventListener('change',filter);filter();</script></body></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'qa/report-requests.html')
    parser.add_argument('--include-tests',action='store_true')
    parser.add_argument('--no-delivery',action='store_true',help='Skip read-only Resend delivery lookups')
    args=parser.parse_args()
    target=args.output.resolve()
    # Private snapshots must never land in source, a build tree or Git history.
    if target.suffix!='.html' or (target.is_relative_to(ROOT) and not target.is_relative_to(ROOT/'qa')):
        parser.error('Write the private HTML into qa/ or outside the repository.')
    rows=Journal(Store().command).all()
    if not args.no_delivery:
        for r in rows:r['delivery_status']=delivery(r)
    target.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as out:out.write(render(rows,args.include_tests))
    os.chmod(target,0o600)
    print(json.dumps({'requests':sum(not r.get('test_request') for r in rows),
                      'tests':sum(bool(r.get('test_request')) for r in rows),'snapshot':str(target)},ensure_ascii=False))

if __name__=='__main__':main()
