"""Transactional email boundary: recipient ownership, limits, private data and PDF scope."""
import io
import base64
import json
import os
import re
import unittest
from unittest.mock import patch
from api.index import app, report_pdf
from lib import report_email as mail
from test_api import PROFILE

ENV = {'VERCEL_ENV':'production','REPORT_EMAIL_ENABLED':'true','RESEND_API_KEY':'test-key-not-a-secret',
       'REPORT_EMAIL_FROM':'Chance Me <reports@example.invalid>',
       'KV_REST_API_URL':'https://redis.example.invalid','KV_REST_API_TOKEN':'test-token'}

class FakeStore:
    def __init__(self):
        self.pending = {}; self.attempts = {}; self.count = 0
    def limit(self, email, ip):
        self.count += 1
        if self.count > 3: raise mail.Limited()
    def create(self, ident, pending): self.pending[ident] = dict(pending)
    def claim(self, ident, code):
        p=self.pending.get(ident)
        if not p or self.attempts.get(ident,0)>=5:raise mail.Invalid()
        if p['code'] != mail.digest(ident+':'+code):
            self.attempts[ident]=self.attempts.get(ident,0)+1; raise mail.Invalid()
        return None if p.get('sent') else dict(p)
    def complete(self, ident): self.pending[ident].update(sent=True); self.pending[ident].pop('pdf',None)

class MailTest(unittest.TestCase):
    def setUp(self):
        self.env=patch.dict(os.environ,ENV,clear=True);self.env.start();self.addCleanup(self.env.stop)
        self.store=FakeStore();self.sent=[]
        self.sender=patch.object(mail,'send',side_effect=lambda *args:self.sent.append(args));self.sender.start();self.addCleanup(self.sender.stop)
    def test_http_client_identifies_chance_me_without_browser_impersonation(self):
        with patch.object(mail, 'urlopen', return_value=io.BytesIO(b'{"id":"safe-id"}')) as opened:
            self.assertEqual(mail.post_json('https://api.resend.com/emails', {}, {}), {'id':'safe-id'})
            request = opened.call_args[0][0]
            self.assertEqual(request.get_header('User-agent'), 'ChanceMe/2.0 (+https://chance-me.com)')

    def test_email_ownership_precedes_pdf_and_retries_do_not_duplicate(self):
        ident=mail.request_report('Student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        self.assertEqual(len(self.sent),1);self.assertEqual(self.sent[0][0],'student@example.com')
        self.assertEqual(len(self.sent[0]),4) # Confirmation has no report attachment.
        code=re.search(r'code is ([0-9]{6})',self.sent[0][2])[1]
        self.assertNotIn(code,json.dumps(self.store.pending))
        mail.verify_report(ident,code,self.store);mail.verify_report(ident,code,self.store)
        self.assertEqual(len(self.sent),2);self.assertEqual(base64.b64decode(self.sent[1][4]),b'%PDF-test')
        self.assertNotIn('pdf',self.store.pending[ident])
    def test_bad_codes_lock_out_after_five_attempts_and_do_not_send_report(self):
        ident=mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        code=re.search(r'code is ([0-9]{6})',self.sent[0][2])[1]
        bad='000000' if code!='000000' else '111111'
        for _ in range(5):
            with self.assertRaises(mail.Invalid):mail.verify_report(ident,bad,self.store)
        with self.assertRaises(mail.Invalid):mail.verify_report(ident,code,self.store)
        self.assertEqual(len(self.sent),1)
    def test_invalid_and_multiple_recipients_never_send(self):
        for email in ['a@example.com,b@example.com','Name <a@example.com>','a@example.com\r\nBcc:x@example.com',None,'a@localhost','a..b@example.com']:
            with self.subTest(email=email),self.assertRaises(mail.Invalid):mail.request_report(email,'127.0.0.1',lambda:b'%PDF-test',self.store)
        self.assertEqual(self.sent,[])
    def test_limits_and_preview_allowlist_are_enforced(self):
        for _ in range(3):mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        with self.assertRaises(mail.Limited):mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        with patch.dict(os.environ,{'VERCEL_ENV':'preview'}):
            with self.assertRaises(mail.Invalid):mail.address('student@example.com')
            self.assertEqual(mail.address('marcinpzdr@gmail.com'),'marcinpzdr@gmail.com')
    def test_missing_config_fails_closed_without_storing_an_address(self):
        with patch.dict(os.environ,{'REPORT_EMAIL_ENABLED':'false'}):
            with self.assertRaises(mail.Unavailable):mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        self.assertEqual(self.store.pending,{})
    def test_api_preserves_origin_and_configuration_gates(self):
        client=app.test_client()
        self.assertEqual(client.post('/api/report-email',json={}).status_code,403)
        self.assertEqual(client.post('/api/report-email',json={},headers={'Origin':'https://evil.example'}).status_code,403)
        with patch.dict(os.environ,{'REPORT_EMAIL_ENABLED':'false'}):
            self.assertFalse(client.get('/api/report-email/status').get_json()['enabled'])
            response=client.post('/api/report-email',json={'email':'private@example.com'},headers={'Origin':'https://chance-me.com'})
            self.assertEqual(response.status_code,503);self.assertNotIn(b'private@example.com',response.data)
    def test_pdf_uses_only_server_selected_close_profiles(self):
        from lib.report_pdf import render
        with patch('lib.report_pdf.render',wraps=render) as renderer:
            pdf=report_pdf(PROFILE,{'course':'Management','session':'Winter','cycle':''})
            self.assertTrue(pdf.startswith(b'%PDF-'));self.assertTrue(pdf.rstrip().endswith(b'%%EOF'))
            payload=renderer.call_args[0][0];rows=payload['comparison']['records']
            self.assertGreaterEqual(len(rows),15);self.assertLessEqual(len(rows),20);self.assertTrue(all(r['nearby'] for r in rows))
        with self.assertRaises(ValueError):report_pdf(PROFILE,{'cycle':'invented'})
    def test_provider_error_is_generic_and_never_marks_a_failed_send_complete(self):
        ident=mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',self.store)
        code=re.search(r'code is ([0-9]{6})',self.sent[0][2])[1]
        with patch.object(mail,'send',side_effect=mail.Unavailable):
            with self.assertRaises(mail.Unavailable):mail.verify_report(ident,code,self.store)
        self.assertNotIn('sent',self.store.pending[ident])


class EmailDesignTest(unittest.TestCase):
    def test_provider_payload_contains_html_text_and_no_attachment_for_code(self):
        with patch.dict(os.environ,ENV,clear=True), patch.object(mail,'post_json',return_value={'id':'test-id'}) as provider:
            mail.send('student@example.com','Confirm your Chance Me report request','Your confirmation code is 123456.','code/test')
            payload=provider.call_args[0][1]
            self.assertIn('123456',payload['html']);self.assertIn('123456',payload['text'])
            self.assertNotIn('attachments',payload)
            self.assertIn('chance me</a>',payload['html'])
            self.assertNotIn('letter-spacing:1.',payload['html'])
            self.assertNotIn('CONFIRMATION CODE',payload['html'])
            self.assertNotIn('in context',payload['html'])
            self.assertIn('https://chance-me.com/privacy/',payload['html'])
            self.assertNotIn('123456',payload['html'].split('<table role="presentation"')[0])
            mail.send('student@example.com','Your Chance Me report','Your report is attached.','pdf/test','cGRm')
            payload=provider.call_args[0][1]
            self.assertIn('Your PDF is attached.',payload['html'])
            self.assertEqual(payload['attachments'][0]['content'],'cGRm')
    def test_template_rejects_html_instead_of_a_code(self):
        from lib.report_email_template import confirmation_html
        with self.assertRaises(ValueError):confirmation_html('<script>')

if __name__=='__main__':unittest.main()
