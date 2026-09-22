"""Regression tests for the interactive logistic regression lab."""

import unittest

import numpy as np
from sklearn.datasets import make_classification

from app import app, session_data
from backend.algorithms.logistic_regression import LogisticRegressionAlgorithm


class LogisticRegressionAlgorithmTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=240,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.8,
            random_state=7,
        )

    def test_training_converges_and_returns_replayable_checkpoints(self):
        result = LogisticRegressionAlgorithm().train(
            self.X, self.y, epochs=300, learning_rate=0.12, regularization=0.01
        )
        self.assertEqual(result['checkpoints'][0]['epoch'], 0)
        self.assertEqual(result['checkpoints'][-1]['epoch'], 300)
        self.assertLess(result['history'][-1]['train_loss'], result['history'][0]['train_loss'])
        self.assertGreater(result['metrics']['accuracy'], 0.9)
        self.assertTrue(np.isfinite(result['model']['coefficients']).all())
        self.assertEqual(sum(map(sum, result['metrics']['confusion_matrix'])), len(result['data']['test_indices']))

    def test_api_uses_existing_dataset_and_rejects_missing_sessions(self):
        client = app.test_client()
        missing = client.post('/api/logistic_regression/train', json={'session_key': 'missing'})
        self.assertEqual(missing.status_code, 400)

        session_data['datasets']['logistic_test'] = {'X': self.X.tolist(), 'y': self.y.tolist()}
        response = client.post('/api/logistic_regression/train', json={
            'session_key': 'logistic_test',
            'params': {'epochs': 200, 'learning_rate': 0.12},
        })
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['status'], 'success')
        self.assertEqual(payload['results']['params']['epochs'], 200)
        self.assertTrue(payload['results']['checkpoints'])


if __name__ == '__main__':
    unittest.main()
