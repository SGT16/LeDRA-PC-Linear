from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from lesion_direction.model import DirectionBatch
from lesion_oof.model import DIM, PREFIX, PATCHES, TOKENS, state_hash
from lesion_pc_linear_ablation_v1.model import LinearAblationModel
from lesion_base_matched_v1 import model as base_model
from lesion_base_matched_v1 import worker
from lesion_mamba.train import weighted_chunk_loss

class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_prefix_tokens, self.num_features, self.global_pool = (PREFIX, DIM, 'avg')
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [nn.LayerNorm(DIM), nn.LayerNorm(DIM)])
        self.norm = nn.LayerNorm(DIM)

    def forward_head(self, tokens):
        return tokens[:, PREFIX:].mean(dim=1)

class BaseMatchedTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError('CPU tests started with CUDA initialized')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        assert not torch.cuda.is_initialized(), 'CPU tests initialized CUDA'

    def synthetic(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(712)
            return (FakeEncoder(), torch.randn(2, TOKENS, DIM))

    def test_all_five_shared_initializers_bridge_to_saved_reference_hash_and_preserve_rng(self):
        encoder, tokens = self.synthetic()
        before = torch.get_rng_state().clone()
        for seed in (823, 289, 592, 605, 329):
            reference = LinearAblationModel(copy.deepcopy(encoder), 'LINEAR_REF', seed)
            model = base_model.make_model(seed=seed, encoder=copy.deepcopy(encoder), device='cpu')
            context = dict(source_seed=seed, model={'control_id': 'LINEAR_REF'}, initial_trainable_state_hash=state_hash(reference.export_trainable_state()))
            proof = base_model.verify_initialization_against_saved_reference(model, seed, model.encoder, context, device='cpu')
            self.assertTrue(proof['exact_initialization_match'])
            self.assertEqual(proof['fresh_base_initial_state_hash'], state_hash(model.export_trainable_state()))
            self.assertEqual(proof['reference_initial_base_subset_hash'], proof['fresh_base_initial_state_hash'])
            self.assertEqual(model.parameter_counts()['trainable'], 25377)
            self.assertEqual(set(model.export_trainable_state()), set(base_model.BASE_STATE_KEYS))
            self.assertFalse(any((hasattr(model, x) for x in ('semantic_reader', 'projection', 'lesion_head'))))
            with torch.no_grad():
                expected = reference.forward_components(DirectionBatch(tokens, None, torch.ones(2, PATCHES, dtype=torch.bool)))
                actual = model.forward_components(tokens)
            torch.testing.assert_close(actual['z'], expected['base'], atol=0, rtol=0)
            torch.testing.assert_close(actual['z'], actual['base'], atol=0, rtol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)

    def test_reference_recorded_hash_is_required_and_not_replaced_by_two_fresh_initializers(self):
        encoder, _tokens = self.synthetic()
        model = base_model.make_model(encoder=encoder, device='cpu')
        for context in ({'source_seed': 823, 'model': {'control_id': 'LINEAR_REF'}}, {'source_seed': 823, 'model': {'control_id': 'LINEAR_REF'}, 'initial_trainable_state_hash': '0' * 64}):
            with self.assertRaises(ValueError):
                base_model.verify_initialization_against_saved_reference(model, 823, encoder, context, device='cpu')

    def test_saved_reference_initializer_mismatch_rejected_before_training(self):
        encoder, _tokens = self.synthetic()
        reference = LinearAblationModel(copy.deepcopy(encoder), 'LINEAR_REF', 823)
        model = base_model.make_model(encoder=encoder, device='cpu')
        with torch.no_grad():
            model.head.bias.add_(0.001)
        context = dict(source_seed=823, model={'control_id': 'LINEAR_REF'}, initial_trainable_state_hash=state_hash(reference.export_trainable_state()))
        with self.assertRaisesRegex(ValueError, 'Fresh Base initial'):
            base_model.verify_initialization_against_saved_reference(model, 823, encoder, context, device='cpu')

    def test_activated_base_matches_independent_patch_adapter_suffix_formula_fp32_bf16(self):
        encoder, tokens = self.synthetic()
        model = base_model.make_model(encoder=encoder, device='cpu')
        with torch.no_grad():
            model.adapter[2].weight.fill_(0.0017)
            model.adapter[2].bias.copy_(torch.linspace(-0.02, 0.03, DIM))
        for bf16 in (False, True):
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                hidden = F.gelu(F.linear(tokens[:, PREFIX:], model.adapter[0].weight, model.adapter[0].bias))
                delta = F.linear(hidden, model.adapter[2].weight, model.adapter[2].bias)
                adapted = torch.cat((tokens[:, :PREFIX], tokens[:, PREFIX:] + delta), dim=1)
                for block in model.encoder.blocks[10:]:
                    adapted = block(adapted)
                pooled = model.encoder.norm(adapted)[:, PREFIX:].mean(dim=1)
                expected = F.linear(pooled, model.head.weight, model.head.bias).squeeze(-1).float()
                actual = model.forward_tokens(DirectionBatch(tokens, torch.full((2, PATCHES, 4), float('nan')), torch.zeros(2, PATCHES, dtype=torch.bool)))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertIsNone(model.forward_training(tokens)[1])

    def test_only_base_parameters_optimize_and_frozen_encoder_stays_eval(self):
        encoder, tokens = self.synthetic()
        model = base_model.make_model(encoder=encoder, device='cpu').train()
        self.assertFalse(model.encoder.training)
        frozen_before = state_hash(model.encoder.state_dict())
        optimizer = base_model.optimizer_for(model)
        self.assertEqual(sum((p.numel() for g in optimizer.param_groups for p in g['params'])), 25377)
        self.assertEqual(optimizer.defaults['lr'], 0.001)
        self.assertEqual(optimizer.defaults['weight_decay'], 0.0001)
        loss = F.binary_cross_entropy_with_logits(model.forward_tokens(tokens), torch.tensor([0.0, 1.0]))
        loss.backward()
        optimizer.step()
        self.assertTrue(all((p.grad is None and (not p.requires_grad) for p in model.encoder.parameters())))
        self.assertGreater(model.adapter[2].weight.grad.abs().sum().item(), 0)
        self.assertEqual(state_hash(model.encoder.state_dict()), frozen_before)

    def test_known_image_full_batch_bce_matches_uneven_microchunk_gradient_without_aux(self):
        features = torch.tensor([[0.3, 0.7], [-0.1, 0.4], [0.9, -0.2], [0.8, 0.1], [-0.5, 0.6]], dtype=torch.float64)
        labels = torch.tensor([-1.0, -1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        initial = torch.tensor([0.21, -0.34], dtype=torch.float64)
        reference = initial.clone().requires_grad_()
        expected = F.binary_cross_entropy_with_logits((features @ reference)[2:], labels[2:])
        expected.backward()
        actual = initial.clone().requires_grad_()
        contributions = []
        for lo, hi in ((0, 2), (2, 4), (4, 5)):
            logits = features[lo:hi] @ actual
            loss, _cl, aux = weighted_chunk_loss(logits, labels[lo:hi], None, 0.0, 3, 5)
            self.assertEqual(float(aux), 0.0)
            contributions.append(loss.detach())
            loss.backward()
        torch.testing.assert_close(sum(contributions), expected.detach(), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-12, atol=1e-12)

    def test_real_trainer_one_complete_step_unknown_microchunk_and_resume_binding(self):

        class ToyBase(nn.Module):
            auxiliary_coefficient = 0.0
            run_seed = 823
            control_id = 'BASE_MATCHED'

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([0.21, -0.34]))

            def parameter_counts(self):
                return {'trainable': 25377}

            def forward_training(self, batch):
                return (batch.features @ self.weight, None)

            def export_trainable_state(self):
                return {'weight': self.weight.detach().cpu().clone()}

            def load_trainable_state(self, state):
                with torch.no_grad():
                    self.weight.copy_(state['weight'])

        class CountedSGD(torch.optim.SGD):

            def __init__(self, params):
                super().__init__(params, lr=0.03)
                self.steps = self.zeros = 0

            def step(self, closure=None):
                self.steps += 1
                return super().step(closure)

            def zero_grad(self, set_to_none=True):
                self.zeros += 1
                return super().zero_grad(set_to_none=set_to_none)
        features = torch.tensor([[0.1 + i * 0.07, 0.3 - i * 0.04] for i in range(13)])
        labels = np.array([-1] * 8 + [0, 1, 0, 0, 1])
        rows = [dict(dataset='deepdrid', image_id=f'i{i}', source_image_key=f'key{i}', patient_id=f'p{i}', dependency_group_id=f'p{i}', image_label=int(labels[i]), patient_label=int(labels[i]) if labels[i] >= 0 else 1, eligible_bilateral=True) for i in range(13)]
        requested = []

        def batch(indexes, device, teacher=True):
            requested.append(teacher)
            return SimpleNamespace(features=features[torch.as_tensor(indexes)].to(device))
        inputs = SimpleNamespace(dataset='deepdrid', rows=rows, labels=labels, batch=batch)
        model = ToyBase()
        opt = CountedSGD(model.parameters())
        reference = model.weight.detach().clone().requires_grad_()
        expected = F.binary_cross_entropy_with_logits(features[8:11] @ reference, torch.tensor([0.0, 1.0, 0.0]))
        expected.backward()
        expected_weight = reference.detach() - 0.03 * reference.grad
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            with mock.patch.object(base_model, 'optimizer_for', return_value=opt), mock.patch('lesion_pc_linear_ablation_v1.worker.batches', return_value=[list(range(11))]), mock.patch.object(worker, 'predict_base', return_value=(np.array([-0.5, 0.5]), np.array([-0.5, 0.5]))):
                receipt = worker.train_base(model, inputs, {'fit': list(range(11)), 'validation': [11, 12], 'fold': 1}, folder, 'fixed-test-context', lambda **x: None, lambda: True)
            self.assertIsNone(receipt)
            self.assertEqual(opt.steps, 1)
            self.assertEqual(opt.zeros, 1)
            self.assertEqual(requested, [False, False])
            torch.testing.assert_close(model.weight, expected_weight, rtol=2e-06, atol=2e-07)
            history = worker.read(folder / 'history.json')
            self.assertEqual(history[0]['images'], 3)
            self.assertEqual(history[0]['all_batch_images'], 11)
            self.assertEqual(history[0]['auxiliary_loss'], 0.0)
            self.assertAlmostEqual(history[0]['fit_loss'], float(expected.detach()), places=6)
            with self.assertRaisesRegex(ValueError, 'resume context changed'):
                worker.train_base(model, inputs, {'fit': list(range(11)), 'validation': [11, 12], 'fold': 1}, folder, 'changed-test-context', lambda **x: None, lambda: False)
if __name__ == '__main__':
    unittest.main()
