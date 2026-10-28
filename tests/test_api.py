"""Public API boundary, portable artifact and report regressions."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from api.index import app, engine
from release_loader import load_release_snapshot

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'lib/engine'
PROFILE = dict(test_type='SAT', score=1450, gpa=9.0, course='Management',
               session='Winter', cycle='27-28', applicant_category='international',
               gpa_confirmed=True)


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def post(self, route='/api/estimate', **changes):
        return self.client.post(route, json=dict(PROFILE, **changes))

    def test_health_is_available_without_local_paths(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['status'], 'ok')
        manifest = json.loads((ENGINE / 'release/manifest.json').read_text())
        self.assertEqual(body['version'], manifest['version'])
        self.assertEqual(body['reports'], manifest['rows'])
        self.assertNotIn('engine_root', body)
        self.assertNotIn('/Users/', response.get_data(as_text=True))

    def test_options_use_supported_programmes(self):
        response = self.client.get('/api/options')
        self.assertEqual(response.status_code, 200)
        programs = response.get_json()['programs']
        self.assertEqual({p['id'] for p in programs}, set(engine.PROGRAM_CONTRACTS))
        self.assertEqual(len(programs), len({p['id'] for p in programs}))

    def test_sat_and_bocconi_estimates_have_bounded_numeric_probabilities(self):
        for kind, score in [('SAT', 1450), ('Bocconi', 38), ('SAT', 1600), ('Bocconi', 50)]:
            with self.subTest(kind=kind, score=score):
                response = self.post(test_type=kind, score=score)
                self.assertEqual(response.status_code, 200)
                body = response.get_json()
                self.assertEqual(body['status'], 'estimated')
                self.assertTrue(math.isfinite(body['probability']))
                self.assertGreaterEqual(body['probability'], 0)
                self.assertLessEqual(body['probability'], 1)
                self.assertRegex(body['display_percent'], r'^(<1|>99|100|\d{1,2})%$')
                self.assertEqual(body['diagnostics']['model_version'], engine.VERSION)

    def test_below_formal_minimum_is_ineligible_not_an_estimate(self):
        for kind, score in [('SAT', 1030), ('Bocconi', 16.9)]:
            with self.subTest(kind=kind):
                response = self.post(test_type=kind, score=score)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()['status'], 'ineligible')
                self.assertEqual(response.get_json()['probability'], 0)

    def test_invalid_input_returns_controlled_400(self):
        invalid = [dict(score=True), dict(score='NaN'), dict(score=1700), dict(gpa=5.99),
                   dict(gpa=10.01), dict(gpa=False), dict(gpa='Infinity'),
                   dict(course='unknown'), dict(course=[]), dict(session='Spring'),
                   dict(test_type='ACT')]
        for change in invalid:
            with self.subTest(change=change):
                response = self.post(**change)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()['status'], 'invalid')
                self.assertIsNone(response.get_json()['probability'])
        for payload in [None, [], {}, 'not a profile']:
            with self.subTest(payload=payload):
                response = self.client.post('/api/estimate', data=json.dumps(payload), content_type='application/json')
                self.assertEqual(response.status_code, 400)
        response = self.client.post('/api/estimate', data='{broken', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_out_of_scope_profiles_do_not_get_a_probability(self):
        for change in [dict(cycle='28-29'), dict(applicant_category='italian'), dict(gpa_confirmed=False)]:
            response = self.post(**change)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['status'], 'unsupported')
            self.assertIsNone(response.get_json()['probability'])

    def test_request_size_is_bounded_on_all_post_endpoints(self):
        for route in ['/api/estimate', '/api/reports', '/api/report.pdf']:
            response = self.client.post(route, data='x' * (16 * 1024 + 1), content_type='application/json')
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.get_json()['status'], 'invalid')

    def test_comparison_counts_and_round_filters_match_records(self):
        for kind, score in [('SAT', 1450), ('Bocconi', 38)]:
            combined = self.post('/api/reports', test_type=kind, score=score, session='All')
            self.assertEqual(combined.status_code, 200)
            body = combined.get_json()
            records = body['records']
            self.assertTrue(body['complete'])
            self.assertEqual(body['context']['total'], len(records))
            self.assertEqual(len(records), len({r['id'] for r in records}))
            self.assertEqual(body['nearby']['total'], sum(r['nearby'] for r in records))
            self.assertEqual(body['nearby']['admitted'], sum(r['nearby'] and r['outcome'] == 'admitted' for r in records))
            self.assertEqual(sum(r['total'] for r in body['by_year']), len(records))
            for session in ['Early', 'Winter']:
                selected = self.post('/api/reports', test_type=kind, score=score, session=session).get_json()
                self.assertEqual({r['id'] for r in selected['records']}, {r['id'] for r in records if r['session'] == session})
            allowed = {'id', 'score', 'gpa', 'course', 'session', 'cycle', 'outcome', 'nearby', 'used_for_bt_fit', 'source_status'}
            self.assertTrue(all(set(r) <= allowed for r in records))
        self.assertEqual(self.post('/api/reports', session='Spring').status_code, 400)

    def test_pdf_is_a_real_portable_document_with_validated_filters(self):
        comparison = self.post('/api/reports', course='Finance', session='Early').get_json()
        filters = dict(course='Finance', session='Early', cycle=comparison['by_year'][0]['cycle'])
        response = self.client.post('/api/report.pdf', json=dict(profile=PROFILE, filters=filters))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, 'application/pdf')
        self.assertTrue(response.data.startswith(b'%PDF-'))
        self.assertTrue(response.data.rstrip().endswith(b'%%EOF'))
        self.assertGreater(len(response.data), 1000)
        self.assertIn('attachment;', response.headers['Content-Disposition'])
        for payload in [{}, dict(profile=PROFILE, filters=[]), dict(profile=PROFILE, filters=dict(cycle='unknown')),
                        dict(profile=dict(PROFILE, score=1030))]:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post('/api/report.pdf', json=payload).status_code, 400)

    def test_private_api_responses_are_not_cached_or_indexed(self):
        with patch.dict('os.environ', {'VERCEL_ENV': 'production'}):
            for response in [self.client.get('/api/health'), self.post(), self.post(score=1700), self.post('/api/reports')]:
                self.assertEqual(response.headers['Cache-Control'], 'no-store')
                self.assertIn('noindex', response.headers['X-Robots-Tag'])
                self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')

    def test_private_engine_and_source_paths_are_not_served(self):
        for route in ['/lib/engine/release/snapshot.json', '/lib/engine/server.py', '/api/validation',
                      '/src/pages/index.html', '/.git/config', '/requirements.txt', '/api/index.py']:
            with self.subTest(route=route):
                with self.client.get(route) as response:
                    self.assertEqual(response.status_code, 404)

    def test_arbitrary_contact_fields_are_not_reflected_into_result(self):
        response = self.post(email='private-canary@example.invalid', free_text='PRIVATE-CANARY-ONLY')
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertNotIn('private-canary@example.invalid', text)
        self.assertNotIn('PRIVATE-CANARY-ONLY', text)

    def test_model_unavailability_fails_closed(self):
        with patch.object(engine, 'ENGINE_ERROR', 'integrity mismatch'):
            self.assertEqual(self.client.get('/api/health').status_code, 503)
            response = self.post()
            self.assertEqual(response.status_code, 503)
            self.assertIsNone(response.get_json()['probability'])


class FrozenPackageTest(unittest.TestCase):
    def test_manifest_hashes_and_exported_record_schema(self):
        manifest = json.loads((ENGINE / 'release/manifest.json').read_text())
        for name, digest in manifest['files'].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ENGINE / name).read_bytes()).hexdigest(), digest)
        data = load_release_snapshot(ENGINE)
        allowed = {'id', 'score', 'GPA', 'Course', 'Session', 'Academic_Year', 'outcome'}
        for kind, (frame, _) in data['frames'].items():
            self.assertEqual(set(frame.columns), allowed)
            self.assertEqual(len(frame), manifest['rows'][kind])
        self.assertEqual(len(data['bundles']['BUNDLE']['bt_fit_ids']), manifest['bt_fit_rows'])

    def test_changed_model_snapshot_fails_integrity_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'engine'
            shutil.copytree(ENGINE, copy)
            with (copy / 'release/snapshot.json').open('ab') as file:
                file.write(b' ')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_release_snapshot(copy)

    def test_manifest_cannot_resolve_files_outside_its_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / 'engine'
            shutil.copytree(ENGINE, copy)
            file = copy / 'release/manifest.json'
            manifest = json.loads(file.read_text())
            manifest['files']['../private'] = '0' * 64
            file.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest path'):
                load_release_snapshot(copy)


if __name__ == '__main__':
    unittest.main()
