"""Optional CPU integration tests, run unconditionally in the neural CI job."""
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import unittest

import numpy as np
from numpy.testing import assert_allclose

from ecg_afib.config import Experiment, Model, Paths, Training
from ecg_afib.data import save_feature_store
from ecg_afib.evaluation import aggregate_records
from ecg_afib.experiments import run_experiment
from ecg_afib.splits import create_splits

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH_AVAILABLE, 'PyTorch is not installed; install the neural extra')
class NeuralTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)

    def test_nt_xent_rewards_correct_pairing_and_backpropagates(self):
        import torch
        from ecg_afib.models import nt_xent
        first = torch.eye(4, requires_grad=True)
        second = torch.eye(4, requires_grad=True)
        good = nt_xent(first, second)
        bad = nt_xent(first, second.roll(1, 0))
        self.assertLess(float(good), float(bad))
        good.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        with self.assertRaises(ValueError):
            nt_xent(first[:1], second[:1])

    def test_same_cnn_accepts_50_and_100_frequency_inputs(self):
        import torch
        from ecg_afib.models import WaveletEncoder
        for frequencies in (50, 100):
            model = WaveletEncoder(frequencies=frequencies, samples=128, embedding_dim=8)
            output = model(torch.randn(2, frequencies, 128))
            self.assertEqual(tuple(output.shape), (2, 8))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_anomaly_score_increases_when_pair_similarity_falls(self):
        import torch
        from ecg_afib.models import pair_anomaly_scores
        first = torch.tensor([[1., 0.]])
        assert_allclose(pair_anomaly_scores(first, first).numpy(), [0])
        assert_allclose(pair_anomaly_scores(first, -first).numpy(), [2])

    def test_all_rr_neural_paths_train_and_reload(self):
        from ecg_afib.training import NeuralPredictor
        from ecg_afib.baselines import ClassicalPredictor
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(8)
            records = [(f'r_{label}_{index}', label, rng.normal(5 + label, .3, (1, 2, 5)))
                       for label in (0, 1) for index in range(10)]
            store = save_feature_store(root / 'features', records, 'rr')
            splits = create_splits(store, root / 'splits.json')
            for method in ('autoencoder', 'contrastive', 'one_class', 'kpca_classifier'):
                config = Experiment(name=method, method=method, root=root,
                                    paths=Paths('features', 'splits.json', method),
                                    training=Training(epochs=1, classifier_epochs=1, batch_size=4),
                                    model=Model(embedding_dim=2, hidden_dims=[4], autoencoder_hidden=[3],
                                                classifier_hidden=[3], kpca_components=3))
                result = run_experiment(config)
                predictor = (ClassicalPredictor if method == 'kpca_classifier' else NeuralPredictor).load(root / method)
                scores = aggregate_records(predictor.scores(store, splits['test']))
                self.assertEqual(len(scores), result['test']['n_records'])
                self.assertTrue(all(np.isfinite(row['score']) for row in scores))

    def test_wavelet_contrastive_training_roundtrip(self):
        from ecg_afib.training import NeuralPredictor
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(5)
            records = [(f'w_{label}_{index}', label, rng.random((1, 2, 24, 32)))
                       for label in (0, 1) for index in range(10)]
            store = save_feature_store(root / 'features', records, 'wavelet')
            splits = create_splits(store, root / 'splits.json')
            config = Experiment(name='wavelet_smoke', method='contrastive', root=root,
                                paths=Paths('features', 'splits.json', 'run'),
                                training=Training(epochs=1, classifier_epochs=1, batch_size=4),
                                model=Model(embedding_dim=4, classifier_hidden=[]))
            run_experiment(config)
            predictor = NeuralPredictor.load(root / 'run')
            self.assertEqual(len(aggregate_records(predictor.scores(store, splits['test']))), len(splits['test']))


if __name__ == '__main__':
    unittest.main()
