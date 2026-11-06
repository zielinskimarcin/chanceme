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

    def test_comparisons_only_share_a_bounded_close_selection(self):
        from api.index import comparison_data, closest_records
        for kind, score in [('SAT', 1450), ('Bocconi', 38)]:
            for session in ['All', 'Early', 'Winter']:
                with self.subTest(kind=kind, session=session):
                    profile = dict(PROFILE, test_type=kind, score=score, session=session)
                    body = self.client.post('/api/reports', json=profile).get_json()
                    records = body['records']
                    self.assertFalse(body['complete'])
                    self.assertEqual(body['record_limit'], 20)
                    self.assertLessEqual(len(records), 20)
                    self.assertTrue(all(r['nearby'] for r in records))
                    self.assertEqual(len(records), len({r['id'] for r in records}))
                    self.assertGreaterEqual(body['nearby']['total'], len(records))
                    self.assertEqual(sum(r['total'] for r in body['by_year']), body['context']['total'])
                    self.assertTrue(all(session == 'All' or r['session'] == session for r in records))
                    self.assertEqual(records, closest_records(records, profile))
        # A client cannot request a larger page or an export of distant profiles.
        limited = self.post('/api/reports', limit=9999, offset=0, all=True).get_json()
        self.assertLessEqual(len(limited['records']), 20)
        self.assertEqual(self.post('/api/reports', session='Spring').status_code, 400)
        distant = self.post('/api/reports', score=1100, gpa=6).get_json()
        self.assertEqual(distant['records'], [])

    def test_closest_selection_is_stable_and_never_backfills_distant_profiles(self):
        from api.index import closest_records, comparison_size
        records = [dict(id=str(i), score=1450+i, gpa=9) for i in range(30)]
        records += [dict(id='far', score=1600, gpa=10)]
        self.assertEqual([r['id'] for r in closest_records(list(reversed(records)), PROFILE)],
                         [str(i) for i in range(comparison_size(PROFILE))])
        for kind, score in [('SAT', 1450), ('Bocconi', 38)]:
            body = self.post(test_type=kind, score=score).get_json()
            records = body['evidence']['records']
            self.assertLessEqual(len(records), 20)
            self.assertTrue(all(r['nearby'] for r in records))
            original = engine.evaluate(dict(PROFILE, test_type=kind, score=score))
            self.assertEqual(body['probability'], original['probability'])

    def test_selection_size_varies_stably_without_padding_sparse_data(self):
        from api.index import comparison_size, closest_records
        sizes = {comparison_size(dict(PROFILE, score=score)) for score in range(1400, 1501)}
        self.assertEqual(sizes, set(range(15, 21)))
        self.assertEqual(comparison_size(PROFILE), comparison_size(dict(PROFILE, score='1450', gpa='9')))
        sparse = [dict(id='only', score=1450, gpa=9)]
        self.assertEqual(closest_records(sparse, PROFILE), sparse)
        self.assertEqual(closest_records([], PROFILE), [])

    def test_direct_pdf_download_is_disabled_even_with_an_email(self):
        for payload in [{}, dict(profile=PROFILE), dict(profile=PROFILE, email='owner@example.invalid')]:
            response = self.client.post('/api/report.pdf', json=payload)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.get_json()['delivery'], 'email')
            self.assertNotIn(b'%PDF-', response.data)
            self.assertNotIn('Content-Disposition', response.headers)

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
