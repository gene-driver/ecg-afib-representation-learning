"""Regression tests for data lineage, preprocessing, and evaluation boundaries."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from ecg_afib.baselines import ClassicalPredictor
from ecg_afib.config import Experiment, Model, Paths
from ecg_afib.data import FeatureStore, read_references, save_feature_store
from ecg_afib.evaluation import aggregate_records, compare_runs, compute_metrics, select_threshold
from ecg_afib.experiments import run_experiment
from ecg_afib.features import augment_rr, frequency_statistics, intervals_from_peaks, paired_windows, rr_pairs
from ecg_afib.splits import create_splits, validate_splits
from ecg_afib.transforms import InputTransform


def rr_fixture(folder, per_class=20):
    rng = np.random.default_rng(7)
    rows = []
    for label in (0, 1):
        for index in range(per_class):
            # Multiple pairs per recording exercise grouping after flattening.
            values = rng.normal(250 + 30 * label, 3 + 30 * label, (2, 2, 5)).astype(np.float32)
            rows.append((f"record_{label}_{index}", label, values))
    return save_feature_store(folder, rows, "rr", {"rr_length": 5}, {"synthetic": True})


class FeatureTests(unittest.TestCase):
    def test_pairing_has_exact_boundaries_and_rejects_short_records(self):
        assert_allclose(paired_windows(np.arange(15), samples=5), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with self.assertRaises(ValueError):
            paired_windows(np.arange(9), samples=5)

    def test_rr_differences_are_samples_and_padding_does_not_bias_moments(self):
        assert_allclose(intervals_from_peaks([4, 14, 34]), [10, 20])
        assert_allclose(augment_rr([[10, 20, 0, 0]]), [[5, 15, 10, 20, 0, 0]])
        with self.assertRaises(ValueError):
            intervals_from_peaks([10, 10])

    def test_rr_window_and_chunk_modes_have_correct_pairing(self):
        with patch('ecg_afib.features.detect_peaks', return_value=np.array([0, 2, 5])):
            result = rr_pairs(np.arange(20), length=4, samples=10)
        assert_allclose(result, [[[2, 3, 0, 0], [2, 3, 0, 0]]])
        with patch('ecg_afib.features.detect_peaks', return_value=np.arange(11) * 2):
            result = rr_pairs(np.arange(30), length=2, mode='chunks')
        self.assertEqual(result.shape, (2, 2, 2))
        assert_allclose(result, 2)

    def test_frequency_statistics_use_true_minimum_and_pooled_variance(self):
        first = np.array([[[2, 4], [10, 20]], [[6, 8], [30, 40]]])
        second = first + 2
        result = frequency_statistics([first, second])
        flat = np.moveaxis(np.stack([first, second]), -2, 0).reshape(2, -1)
        assert_allclose(result['minimum'], flat.min(axis=1))
        assert_allclose(result['mean'], flat.mean(axis=1))
        assert_allclose(result['std'], flat.std(axis=1))

    def test_transform_is_fitted_only_on_supplied_training_values(self):
        transform = InputTransform(standardize=True).fit([[1, 2], [3, 4]])
        original = transform.to_dict()
        transform.apply([[1000, 1000]])
        self.assertEqual(original, transform.to_dict())
        assert_allclose(transform.apply([[1, 2], [3, 4]]).mean(axis=0), [0, 0])

    def test_reference_reader_excludes_other_classes_and_rejects_duplicate_ids(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'REFERENCE.csv'
            path.write_text('A01,N\nA02,A\nA03,O\nA04,~\n')
            self.assertEqual([r['label'] for r in read_references(temporary)], [0, 1])
            path.write_text('A01,N\nA01,A\n')
            with self.assertRaises(ValueError):
                read_references(temporary)


class EvaluationTests(unittest.TestCase):
    def test_record_aggregation_precedes_threshold_selection(self):
        observations = [dict(record_id='a', label=0, score=0.1), dict(record_id='a', label=0, score=0.3),
                        dict(record_id='b', label=1, score=0.8), dict(record_id='b', label=1, score=1.0)]
        records = aggregate_records(observations)
        self.assertEqual([r['observations'] for r in records], [2, 2])
        threshold = select_threshold(records)
        metrics = compute_metrics(records, threshold)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0], [0, 1]])
        self.assertEqual(metrics['n_records'], 2)
        self.assertEqual(metrics['auroc'], 1.0)

    def test_tied_scores_have_deterministic_finite_threshold(self):
        rows = [dict(record_id=str(i), label=i % 2, score=0.5) for i in range(8)]
        threshold = select_threshold(rows)
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(compute_metrics(rows, threshold)['balanced_accuracy'], 0.5)

    def test_conflicting_record_labels_fail(self):
        with self.assertRaises(ValueError):
            aggregate_records([dict(record_id='same', label=0, score=0), dict(record_id='same', label=1, score=1)])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = rr_fixture(self.root / 'features')
        self.splits = create_splits(self.store, self.root / 'splits.json')

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_windows_remain_with_their_recording(self):
        names = [set(self.store.flat_views(self.splits[part])[2]) for part in ('train', 'validation', 'test')]
        for i in range(3):
            for j in range(i):
                self.assertFalse(names[i] & names[j])
        corrupted = json.loads(json.dumps(self.splits))
        corrupted['test'].append(corrupted['train'][0])
        with self.assertRaises(ValueError):
            validate_splits(self.store, corrupted)

    def test_cohort_hash_is_independent_of_feature_representation(self):
        values = [(r['record_id'], r['label'], np.ones((1, 2, 20, 20))) for r in self.store.records]
        other = save_feature_store(self.root / 'wavelets', values, 'wavelet')
        self.assertEqual(other.cohort_hash, self.store.cohort_hash)
        validate_splits(other, self.splits)

    def test_feature_store_rejects_path_escape(self):
        manifest = self.root / 'features' / 'manifest.json'
        values = json.loads(manifest.read_text())
        values['records'][0]['file'] = '../outside.npy'
        manifest.write_text(json.dumps(values))
        with self.assertRaises(ValueError):
            FeatureStore(self.root / 'features')

    def test_classical_experiments_save_reload_and_compare(self):
        outputs = []
        for method in ('kpca', 'ocsvm'):
            config = Experiment(name=method, method=method, root=self.root,
                                paths=Paths('features', 'splits.json', method),
                                model=Model(rr_statistics=True, kpca_components=3, standardize=True))
            metrics = run_experiment(config)
            output = self.root / method
            outputs.append(output)
            self.assertEqual(metrics['test']['n_records'], len(self.splits['test']))
            self.assertEqual(metrics['test']['threshold'], metrics['validation']['threshold'])
            predictor = ClassicalPredictor.load(output)
            records = aggregate_records(predictor.scores(self.store, self.splits['test']))
            self.assertEqual(compute_metrics(records, metrics['test']['threshold']), metrics['test'])
            self.assertEqual(json.loads((output / 'run.json').read_text())['status'], 'complete')
            with self.assertRaises(FileExistsError):
                run_experiment(config)
        self.assertEqual(len(compare_runs(outputs)), 2)
        run_file = outputs[1] / 'run.json'
        metadata = json.loads(run_file.read_text())
        metadata['split_signature'] = 'different'
        run_file.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            compare_runs(outputs)


if __name__ == '__main__':
    unittest.main()
