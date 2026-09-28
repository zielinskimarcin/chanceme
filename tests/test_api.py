import unittest

from api.index import app, df_bt, df_sat


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'ok')
        self.assertGreater(len(df_sat) + len(df_bt), 600)

    def test_sat_calculation(self):
        response = self.client.post('/api/calculate', json={
            'sat': 1450,
            'gpa': 9.0,
            'course': 'BIEM',
            'session': 'Winter',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()['Status']['Zone'], {
            'Safe', 'Competitive', 'High Risk', 'Unknown'
        })

    def test_bocconi_test_calculation(self):
        response = self.client.post('/api/calculate', json={
            'sat': 38,
            'gpa': 9.0,
            'course': 'BIEF',
            'session': 'Winter',
        })
        self.assertEqual(response.status_code, 200)

    def test_invalid_payload_returns_400(self):
        response = self.client.post('/api/calculate', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

    def test_out_of_range_values_return_400(self):
        response = self.client.post('/api/calculate', json={
            'sat': 1700,
            'gpa': 12,
            'course': 'BIEM',
            'session': 'Winter',
        })
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main()
