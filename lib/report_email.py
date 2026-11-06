"""Email-only report delivery. No provider credentials, addresses or PDFs in logs."""
from __future__ import annotations
import base64
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import time
from urllib.request import Request, urlopen
from lib.report_email_template import confirmation_html, report_html
from lib.report_journal import Journal, keys as journal_keys, new_record

PREFIX = 'chanceme:report-email:v1:'
TTL = 15 * 60
QA_RECIPIENTS = {'marcinpzdr@gmail.com', 'marzielek200@gmail.com', 'marcin@smaklo.com'}

class Unavailable(Exception):
    pass

class Limited(Exception):
    pass

class Invalid(Exception):
    pass

def configured():
    return (os.environ.get('REPORT_EMAIL_ENABLED') == 'true'
            and all(os.environ.get(k) for k in ('RESEND_API_KEY', 'REPORT_EMAIL_FROM',
                                                'KV_REST_API_URL', 'KV_REST_API_TOKEN')))

def digest(value):
    return hmac.new(os.environ['RESEND_API_KEY'].encode(), value.encode(), hashlib.sha256).hexdigest()

def address(value):
    if not isinstance(value, str) or len(value) > 254:
        raise Invalid()
    value = value.strip().lower()
    # Single ASCII mailbox only; never accept display names, multiple recipients or headers.
    if not re.fullmatch(r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+", value):
        raise Invalid()
    if os.environ.get('VERCEL_ENV') != 'production' and value not in QA_RECIPIENTS:
        raise Invalid()
    return value

def post_json(url, payload, headers):
    try:
        req = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', 'User-Agent': 'ChanceMe/2.0 (+https://chance-me.com)', **headers})
        with urlopen(req, timeout=8) as res:
            return json.load(res)
    except Exception:
        # Provider error bodies may contain personal information. Never reflect or log them.
        raise Unavailable() from None

class Store:
    def command(self, *args):
        url = os.environ['KV_REST_API_URL']
        if not url.startswith('https://'):
            raise Unavailable()
        result = post_json(url, list(args), {'Authorization': 'Bearer ' + os.environ['KV_REST_API_TOKEN']})
        if 'error' in result or 'result' not in result:
            raise Unavailable()
        return result['result']

    def limit(self, email, ip):
        # All quotas are checked and consumed atomically. Hash IPs/mailboxes; no raw IP retention.
        day = int(time.time() // 86400)
        keys = [PREFIX + 'limit:' + str(day) + ':' + v for v in ('global', digest('email:' + email), digest('ip:' + ip))]
        lua = """
        for i,k in ipairs(KEYS) do
          if tonumber(redis.call('GET',k) or '0') >= tonumber(ARGV[i]) then return 0 end
        end
        for _,k in ipairs(KEYS) do redis.call('INCR',k); redis.call('EXPIRE',k,86400) end
        return 1
        """
        if self.command('EVAL', lua, 3, *keys, 40, 3, 8) != 1:
            raise Limited()

    def create(self, ident, pending):
        record = new_record(ident, pending['email'], pending['email'] in QA_RECIPIENTS)
        record['code_expires_at'] = record['created_at'] + TTL
        lua = """
        if not redis.call('SET',KEYS[1],ARGV[1],'EX',ARGV[2],'NX') then return 0 end
        redis.call('SET',KEYS[2],ARGV[3],'NX')
        redis.call('ZADD',KEYS[3],ARGV[4],ARGV[5])
        return 1
        """
        if self.command('EVAL',lua,3,PREFIX+ident,*journal_keys(ident),json.dumps(pending),TTL,
                        json.dumps(record),record['created_at'],record['id']) != 1:
            raise Unavailable()

    def claim(self, ident, code):
        lua = """
        local raw=redis.call('GET',KEYS[1]); if not raw then return 'expired' end
        local p=cjson.decode(raw)
        if (p.attempts or 0)>=6 then return 'expired' end
        if p.code~=ARGV[1] then
          p.attempts=(p.attempts or 0)+1
          redis.call('SET',KEYS[1],cjson.encode(p),'KEEPTTL'); return 'invalid:'..p.attempts
        end
        if p.sent then return 'sent' end
        if not redis.call('SET',KEYS[2],'1','NX','EX',60) then return 'busy' end
        return raw
        """
        result = self.command('EVAL', lua, 2, PREFIX + ident, PREFIX + ident + ':lock', digest(ident + ':' + code))
        if isinstance(result,str) and result.startswith('invalid:'):
            attempts=int(result.split(':')[1])
            self.event(ident,'code_locked' if attempts>=5 else 'code_rejected',attempts=attempts)
            raise Invalid()
        if result == 'expired':
            raise Invalid()
        if result == 'busy':
            raise Limited()
        if result == 'sent':
            return None
        pending=json.loads(result)
        self.event(ident,'email_verified',email=pending['email'])
        return pending

    def event(self, ident, kind, provider_id=None, email=None, attempts=None):
        return Journal(self.command).event(ident,kind,provider_id,email,
                                           bool(email in QA_RECIPIENTS) if email else False,attempts)

    def complete(self, ident, provider_id=None):
        # Persist provider acceptance first; if this fails, idempotent retry can repair it.
        self.event(ident,'report_submitted',provider_id=provider_id)
        lua = """
        local raw=redis.call('GET',KEYS[1]); if not raw then return 0 end
        local p=cjson.decode(raw); p.sent=true; p.pdf=nil
        redis.call('SET',KEYS[1],cjson.encode(p),'KEEPTTL'); return 1
        """
        self.command('EVAL', lua, 1, PREFIX + ident)


def send(recipient, subject, text, ident, attachment=None):
    payload = {'from': os.environ['REPORT_EMAIL_FROM'], 'to': [recipient], 'subject': subject, 'text': text}
    if subject == 'Confirm your Chance Me report request':
        match = re.search(r'code is ([0-9]{6})', text)
        if not match:
            raise Unavailable()
        payload['html'] = confirmation_html(match[1])
    elif subject == 'Your Chance Me report':
        payload['html'] = report_html()
    if attachment:
        payload['attachments'] = [{'filename': 'chance-me-report.pdf', 'content': attachment}]
    result = post_json('https://api.resend.com/emails', payload,
                       {'Authorization': 'Bearer ' + os.environ['RESEND_API_KEY'], 'Idempotency-Key': ident})
    if not isinstance(result.get('id'), str) or not result['id']:
        raise Unavailable()
    return result['id']


def request_report(email, ip, pdf_factory, store=None):
    if not configured():
        raise Unavailable()
    email = address(email)
    try:
        ipaddress.ip_address(ip)
    except ValueError:
        # Missing or invalid trusted edge IP: fail closed; never trust a client-supplied forwarded IP.
        raise Unavailable() from None
    store = store or Store()
    store.limit(email, ip)
    ident = secrets.token_urlsafe(32)
    code = f'{secrets.randbelow(1000000):06d}'
    pdf = pdf_factory()
    if not pdf.startswith(b'%PDF-') or len(pdf) > 1024 * 1024:
        raise Unavailable()
    pending = {'email': email, 'code': digest(ident + ':' + code), 'attempts': 0,
               'pdf': base64.b64encode(pdf).decode('ascii')}
    store.create(ident, pending)
    try:
        provider_id = send(email, 'Confirm your Chance Me report request',
         f'Your confirmation code is {code}. It expires in 15 minutes.\n\n'
         'Enter it on Chance Me to receive your report. If you did not request this, ignore this email.\n\n'
         'Chance Me\nNot affiliated with Bocconi University.\nPrivacy: https://chance-me.com/privacy/',
             'report-code/' + ident)
    except Unavailable:
        store.event(ident,'code_send_error')
        raise
    store.event(ident,'code_submitted',provider_id=provider_id)
    return ident


def verify_report(ident, code, store=None):
    if not configured():
        raise Unavailable()
    if not isinstance(ident, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{43}', ident):
        raise Invalid()
    if not isinstance(code, str) or not re.fullmatch(r'[0-9]{6}', code):
        raise Invalid()
    store = store or Store()
    pending = store.claim(ident, code)
    if pending is None:
        return  # Already accepted; do not send another attachment.
    try:
        provider_id = send(pending['email'], 'Your Chance Me report',
         'Your PDF report is attached.\n\n'
         'Results are experimental model estimates from self-reported data, not admission guarantees.\n\n'
         'Chance Me\nNot affiliated with Bocconi University.\nPrivacy: https://chance-me.com/privacy/',
             'report-pdf/' + ident, pending['pdf'])
    except Unavailable:
        store.event(ident,'report_send_error')
        raise
    store.complete(ident,provider_id)
