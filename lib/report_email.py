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
        req = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json', **headers})
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
        if self.command('SET', PREFIX + ident, json.dumps(pending), 'EX', TTL, 'NX') != 'OK':
            raise Unavailable()

    def claim(self, ident, code):
        lua = """
        local raw=redis.call('GET',KEYS[1]); if not raw then return 'expired' end
        local p=cjson.decode(raw)
        if (p.attempts or 0)>=5 then return 'expired' end
        if p.code~=ARGV[1] then
          p.attempts=(p.attempts or 0)+1
          redis.call('SET',KEYS[1],cjson.encode(p),'KEEPTTL'); return 'invalid'
        end
        if p.sent then return 'sent' end
        if not redis.call('SET',KEYS[2],'1','NX','EX',60) then return 'busy' end
        return raw
        """
        result = self.command('EVAL', lua, 2, PREFIX + ident, PREFIX + ident + ':lock', digest(ident + ':' + code))
        if result in ('invalid', 'expired'):
            raise Invalid()
        if result == 'busy':
            raise Limited()
        return None if result == 'sent' else json.loads(result)

    def complete(self, ident):
        lua = """
        local raw=redis.call('GET',KEYS[1]); if not raw then return 0 end
        local p=cjson.decode(raw); p.sent=true; p.pdf=nil
        redis.call('SET',KEYS[1],cjson.encode(p),'KEEPTTL'); return 1
        """
        self.command('EVAL', lua, 1, PREFIX + ident)


def send(recipient, subject, text, ident, attachment=None):
    payload = {'from': os.environ['REPORT_EMAIL_FROM'], 'to': [recipient], 'subject': subject, 'text': text}
    if attachment:
        payload['attachments'] = [{'filename': 'chance-me-report.pdf', 'content': attachment}]
    result = post_json('https://api.resend.com/emails', payload,
                       {'Authorization': 'Bearer ' + os.environ['RESEND_API_KEY'], 'Idempotency-Key': ident})
    if not isinstance(result.get('id'), str) or not result['id']:
        raise Unavailable()


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
    send(email, 'Confirm your Chance Me report request',
         f'Your confirmation code is {code}. It expires in 15 minutes.\n\n'
         'Enter it in the Chance Me form to receive your report. If you did not request a report, ignore this message.\n\n'
         'Chance Me — independent Bocconi admissions tool. https://chance-me.com/privacy/',
         'report-code/' + ident)
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
    send(pending['email'], 'Your Chance Me report',
         'Your requested report is attached. It includes at most 10 close historical profiles.\n\n'
         'Results are experimental model estimates from self-reported data, not admission guarantees.\n\n'
         'Chance Me — independent Bocconi admissions tool. https://chance-me.com/privacy/',
         'report-pdf/' + ident, pending['pdf'])
    store.complete(ident)
