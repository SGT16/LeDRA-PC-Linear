import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from ledra_experiments.cli import main
from ledra_experiments.factory import BENCHMARK, MATCHED, make_model, cache_name
from ledra_experiments.inputs import validate_split
from lesion_cross_dataset.metrics import empirical_operating_points, fixed_threshold_metrics
from lesion_exchange.train import select_checkpoint


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_prefix_tokens = 5
        self.num_features = 384
        self.global_pool = "avg"
        self.blocks = nn.ModuleList([nn.LayerNorm(384) for _ in range(12)])
        self.norm = nn.LayerNorm(384)

    def forward_head(self, tokens):
        return tokens[:, 5:].mean(dim=1)


class PublicPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_model_constructors_and_cache_choices(self):
        import timm
        encoder = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=False, img_size=(392, 392), num_classes=0)
        encoder.global_pool = "avg"
        for phase, variants in (("benchmark", BENCHMARK), ("matched", MATCHED)):
            for name in variants:
                model = make_model(phase, name, 823, encoder, "cpu")
                self.assertGreater(model.parameter_counts()["trainable"], 0)
                self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
                self.assertTrue(cache_name(phase, name).endswith(".npy"))
                if name == "BASE_MATCHED":
                    self.assertEqual(model.parameter_counts()["trainable"], 25377)
                if phase == "matched" and name != "BASE_MATCHED":
                    self.assertEqual(model.parameter_counts()["trainable"], 36229)

    def test_complete_distinct_training_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            for phase, count in (("benchmark", 1575), ("matched", 630)):
                path = Path(temporary) / (phase + ".csv")
                main(["plan", "--phase", phase, "--output", str(path)])
                with path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), count)
                self.assertEqual(len({tuple(r.values()) for r in rows}), count)
                self.assertEqual({int(r["seed"]) for r in rows}, {823, 289, 592, 605, 329})

    def test_roc95_uses_real_tied_points(self):
        y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
        scores = np.r_[np.full(19, .1), .8, np.full(19, .9), .2]
        result = empirical_operating_points(y, scores)
        self.assertAlmostEqual(result["Se_at_Sp95"], 1.)
        self.assertAlmostEqual(result["Sp_at_Se95"], 1.)
        self.assertGreaterEqual(result["se_sp95_specificity"], .95)
        self.assertGreaterEqual(result["sp_se95_sensitivity"], .95)
        tied = empirical_operating_points(y, np.full(40, .5))
        self.assertEqual(tied["Se_at_Sp95"], 0.)
        self.assertEqual(tied["Sp_at_Se95"], 0.)

    def test_source_threshold_is_applied_without_target_tuning(self):
        result = fixed_threshold_metrics(np.array([0, 0, 1, 1]), np.array([.1, .6, .6, .9]), .6)
        self.assertEqual(result["TP"], 2)
        self.assertEqual(result["FP"], 1)
        self.assertAlmostEqual(result["F1"], .8)
        self.assertAlmostEqual(result["accuracy"], .75)

    def test_checkpoint_tie_break_does_not_reset_ap_patience(self):
        best = dict(AP=.7, logloss=.5)
        self.assertEqual(select_checkpoint(dict(AP=.7, logloss=.4), best, 2), (True, False))
        self.assertEqual(select_checkpoint(dict(AP=.71, logloss=.6), best, 2), (True, True))
        self.assertEqual(select_checkpoint(dict(AP=.69, logloss=.3), best, 2), (False, False))

    def test_dependency_leakage_is_rejected(self):
        rows = [dict(dependency_group_id=str(i)) for i in range(6)]
        split = dict(fit=[0, 1], validation=[2, 3], holdout=[4, 5])
        validate_split(rows, split)
        rows[5]["dependency_group_id"] = rows[0]["dependency_group_id"]
        with self.assertRaises(ValueError):
            validate_split(rows, split)

    def test_historical_known_image_loss_excludes_unknowns(self):
        from lesion_exchange.train import image_mean_loss
        logits = torch.tensor([.2, -.3, .8, -.7], requires_grad=True)
        labels = torch.tensor([-1., 0., 1., -1.])
        actual = image_mean_loss(logits, labels)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[1:3], labels[1:3])
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertEqual(float(logits.grad[0]), 0.)
        self.assertEqual(float(logits.grad[3]), 0.)

    def test_benchmark_external_uses_four_image_batches_and_no_teacher(self):
        from ledra_experiments.run import predict_benchmark_external
        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(()))
                self.sizes = []
            def forward(self, batch):
                value = batch if torch.is_tensor(batch) else batch.tokens
                self.sizes.append(len(value))
                return value[:, 0, 0] + self.weight
        values = np.zeros((9, 789, 384), np.float32)
        values[:, 0, 0] = np.arange(9)
        calls = []
        def batch(indexes, device, teacher=True):
            calls.append(teacher)
            return SimpleNamespace(tokens=torch.as_tensor(values[indexes], device=device))
        inputs = SimpleNamespace(rows=[{}] * 9, tokens=values, features=False, batch=batch)
        for name in BENCHMARK:
            model = RecordingModel()
            actual = predict_benchmark_external(name, model, inputs)
            np.testing.assert_array_equal(actual, np.arange(9))
            self.assertEqual(model.sizes, [4, 4, 1])
        self.assertTrue(calls)
        self.assertFalse(any(calls))


if __name__ == "__main__":
    unittest.main()
