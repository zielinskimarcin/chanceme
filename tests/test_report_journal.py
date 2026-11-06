"""Journal privacy, failure handling and private owner snapshot boundaries."""
import io
import contextlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from lib import report_email as mail
from lib.report_journal import new_record, record_id, display_status, Journal
from scripts.report_email_history import render, main, delivery
from test_report_email import ENV, FakeStore
from api.index import app

class JournalTest(unittest.TestCase):
    def test_record_never_keeps_authentication_or_profile_data(self):
        with patch.dict(os.environ,ENV,clear=True):record=new_record('private-request-token','student@example.com',False)
        self.assertNotIn('private-request-token',json.dumps(record))
        self.assertEqual(record['id'],record_id('private-request-token'))
        for key in ['code','pdf','score','gpa','ip','request_id']:
            self.assertNotIn(key,record)
        self.assertEqual(record['status'],'requested')
    def test_expiry_is_derived_without_losing_history_or_overwriting_success(self):
        r=new_record('request','student@example.com',False)
        self.assertEqual(display_status(r,r['code_expires_at']+1),'expired')
        r['status']='report_submitted'
        self.assertEqual(display_status(r,r['code_expires_at']+1),'report_submitted')
    def test_provider_reference_cannot_inject_or_store_arbitrary_error_content(self):
        journal=Journal(lambda *a:None)
        for value in ['<html>','secret\nheader',{},'a'*101]:
            with self.assertRaises(ValueError):journal.event('request','report_submitted',provider_id=value)
        with self.assertRaises(ValueError):journal.event('request','arbitrary-visitor-data')
    def test_failed_confirmation_is_recorded_before_raising(self):
        store=FakeStore()
        with patch.dict(os.environ,ENV,clear=True),patch.object(mail,'send',side_effect=mail.Unavailable):
            with self.assertRaises(mail.Unavailable):mail.request_report('student@example.com','127.0.0.1',lambda:b'%PDF-test',store)
        self.assertEqual(store.events[-1][1],'code_send_error')
    def test_failed_journal_commit_does_not_mark_pending_complete(self):
        store=mail.Store()
        with patch.object(store,'event',side_effect=mail.Unavailable),patch.object(store,'command') as cmd:
            with self.assertRaises(mail.Unavailable):store.complete('request','provider-id')
            cmd.assert_not_called()
    def test_owner_snapshot_escapes_mailboxes_and_hides_tests_by_default(self):
        r=new_record('id','<script>alert(1)</script>@example.com',True)
        page=render([r])
        self.assertNotIn('<script>alert(1)</script>',page)
        self.assertIn('&lt;script&gt;',page)
        self.assertIn('Pokaż testy (1)',page)
        self.assertNotIn('id="tests" checked',page)
        self.assertIn('id="tests" checked',render([r],True))
        self.assertNotIn('https://',page) # No external analytics, fonts or network calls.
    def test_snapshot_cannot_be_written_into_public_build_or_source(self):
        for path in ['public/requests.html','src/requests.html','dist/requests.html']:
            with patch.object(sys,'argv',['history','--output',path]),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()
    def test_journal_has_no_public_api_or_file_route(self):
        client=app.test_client()
        for path in ['/api/report-journal','/lib/report_journal.py','/scripts/report_email_history.py','/qa/report-requests.html']:
            with client.get(path) as response:self.assertEqual(response.status_code,404)
    def test_provider_delivery_is_not_assigned_to_another_address(self):
        from io import BytesIO
        r=new_record('id','student@example.com',False);r['report_provider_id']='safe-id'
        with patch.dict(os.environ,ENV,clear=True),patch('scripts.report_email_history.urlopen',return_value=BytesIO(json.dumps({'to':['other@example.com'],'subject':'Your Chance Me report','last_event':'delivered'}).encode())):
            self.assertEqual(delivery(r),'Nie można potwierdzić')

if __name__=='__main__':unittest.main()
