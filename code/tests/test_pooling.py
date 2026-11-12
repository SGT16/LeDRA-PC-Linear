import copy
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from lesion_direction.data import auxiliary_loss_per_image
from lesion_direction.model import DirectionBatch
from lesion_oof.model import DIM, PATCHES, PREFIX, TOKENS, state_hash
from lesion_pc_linear_ablation_v1.model import LinearAblationModel
from lesion_pc_linear_pooling_v1.model import LesionPoolingModel, lesion_weighted_pool_features

class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_prefix_tokens, self.num_features, self.global_pool = (PREFIX, DIM, 'avg')
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [nn.LayerNorm(DIM), nn.LayerNorm(DIM)])
        self.norm = nn.LayerNorm(DIM)

    def forward_head(self, tokens):
        return tokens[:, PREFIX:].mean(dim=1)

def scalar_pool_oracle(values, probabilities, fov):
    p = probabilities.detach().float().cpu().tolist()
    valid = fov.cpu().tolist()
    images = []
    for image in range(len(values)):
        types = []
        for lesion in range(4):
            indexes = [i for i, included in enumerate(valid[image]) if included]
            mass = sum((p[image][i][lesion] for i in indexes))
            result = values[image].sum(dim=0) * 0
            if mass > 0:
                for i in indexes:
                    result = result + values[image, i] * (p[image][i][lesion] / mass)
            types.append(result)
        images.append(torch.stack(types))
    return torch.stack(images)

class PoolingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():
            raise AssertionError('These tests must begin with CUDA uninitialized')

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)
        assert not torch.cuda.is_initialized(), 'CPU tests initialized CUDA'

    def synthetic(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(593)
            encoder = FakeEncoder()
            tokens = torch.randn(1, TOKENS, DIM)
            fov = torch.zeros(1, PATCHES, dtype=torch.bool)
            fov[:, [0, 9, 101, 211, 701, 783]] = True
            targets = torch.sigmoid(0.4 * tokens[:, PREFIX:, :4] + 0.3)
            return (encoder, DirectionBatch(tokens, targets, fov))

    def test_values_and_gradients_match_independent_scalar_oracle(self):
        generator = torch.Generator().manual_seed(447)
        values = torch.randn(2, 7, 3, generator=generator, dtype=torch.float64)
        p = torch.rand(2, 7, 4, generator=generator).requires_grad_()
        fov = torch.tensor([[True, False, True, True, False, True, True], [False, True, False, False, False, False, False]])
        with torch.no_grad():
            p[0, :, 0] = 0
            p[0, :, 3] = 1
        source = values.clone().requires_grad_()
        actual = lesion_weighted_pool_features(source, p, fov, grid=(1, 7))
        oracle_source = values.clone().requires_grad_()
        expected = scalar_pool_oracle(oracle_source, p, fov)
        coefficient = torch.linspace(-0.6, 0.7, 24, dtype=torch.float64).reshape(2, 4, 3)
        gradient, p_gradient = torch.autograd.grad((actual * coefficient).sum(), (source, p), allow_unused=True)
        oracle_gradient = torch.autograd.grad((expected * coefficient).sum(), oracle_source)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-06, atol=2e-07)
        torch.testing.assert_close(gradient, oracle_gradient, rtol=2e-06, atol=2e-07)
        self.assertIsNone(p_gradient)
        self.assertEqual(int(torch.count_nonzero(gradient[~fov])), 0)
        torch.testing.assert_close(actual[1], values[1, 1].expand(4, -1), rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 0], torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(actual[0, 3], values[0, fov[0]].mean(dim=0), rtol=2e-06, atol=2e-07)

    def test_no_reference_eligibility_self_exclusion_or_background_subtraction(self):
        values = torch.tensor([[[2.0, -3.0]]], dtype=torch.float64)
        probabilities = torch.tensor([[[0.3, 1.0, 0.01, 0.0]]])
        fov = torch.ones(1, 1, dtype=torch.bool)
        actual = lesion_weighted_pool_features(values, probabilities, fov, grid=(1, 1))
        torch.testing.assert_close(actual[0, :3], values[0, 0].expand(3, -1), atol=0, rtol=0)
        torch.testing.assert_close(actual[0, 3], torch.zeros(2, dtype=torch.float64), atol=0, rtol=0)
        translated = lesion_weighted_pool_features(values + 7, probabilities, fov, grid=(1, 1))
        torch.testing.assert_close(translated[0, :3], actual[0, :3] + 7, atol=0, rtol=0)

    def test_outside_FOV_and_other_images_cannot_change_an_image_result(self):
        values = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3)
        p = torch.tensor([[[0.2] * 4, [0.8] * 4, [0.4] * 4, [0.6] * 4]]).expand(2, -1, -1).clone()
        fov = torch.tensor([[True, False, True, False], [False, True, True, False]])
        expected = lesion_weighted_pool_features(values, p, fov, grid=(1, 4))
        changed, changed_p = (values.clone(), p.clone())
        changed[~fov] = 10000
        changed_p[~fov] = 1
        actual = lesion_weighted_pool_features(changed, changed_p, fov, grid=(1, 4))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        single = lesion_weighted_pool_features(values[:1], p[:1], fov[:1], grid=(1, 4))
        torch.testing.assert_close(single[0], expected[0], rtol=0, atol=0)

    def test_five_seeds_preserve_initial_state_layout_and_caller_rng(self):
        encoder, batch = self.synthetic()
        before = torch.get_rng_state().clone()
        for seed in (823, 289, 592, 605, 329):
            reference = LinearAblationModel(copy.deepcopy(encoder), 'LINEAR_REF', seed)
            model = LesionPoolingModel(copy.deepcopy(encoder), 'LINEAR_POOL_ONLY', seed)
            self.assertEqual(model.parameter_counts()['trainable'], 36229)
            self.assertEqual(set(model.export_trainable_state()), set(reference.export_trainable_state()))
            self.assertEqual(state_hash(model.export_trainable_state()), state_hash(reference.export_trainable_state()))
            self.assertEqual(model.auxiliary_coefficient, 0.1)
            self.assertEqual(model.variant, 'GLOBAL_LINEAR')
            self.assertEqual(model.control_id, 'LINEAR_POOL_ONLY')
            self.assertEqual(model.scientific_metadata()['objective_connected_parameters'], 36229)
            with torch.no_grad():
                result = model.forward_components(batch)
                ref = reference.forward_components(batch)
            torch.testing.assert_close(result['z'], ref['z'], atol=0, rtol=0)
            torch.testing.assert_close(result['q'], ref['q'], atol=0, rtol=0)
            self.assertTrue(all((not p.requires_grad for p in model.encoder.parameters())))
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)

    def test_activated_forward_equals_explicit_pool_and_preserves_gradient_policy(self):
        encoder, batch = self.synthetic()
        model = LesionPoolingModel(encoder, seed=823)
        with torch.no_grad():
            model.lesion_head.weight.copy_(torch.linspace(-0.05, 0.09, 96).reshape(4, 24))
        result = model.forward_components(batch)
        _global, patch = model._features(batch)
        projected = F.linear(patch.float(), model.projection.weight.float())
        p = torch.sigmoid(model.semantic_logits(batch)).detach()
        beta = p * batch.fov[..., None]
        beta = beta / beta.sum(dim=1, keepdim=True)
        expected = torch.einsum('bil,bir->blr', beta, projected)
        torch.testing.assert_close(result['lesion_pooled_evidence'], expected, atol=0, rtol=0)
        torch.testing.assert_close(result['z'], result['base'] + (expected * model.lesion_head.weight[None]).sum((-1, -2)), atol=0, rtol=0)
        F.binary_cross_entropy_with_logits(result['z'], torch.ones(1)).backward()
        self.assertIsNone(model.semantic_reader.weight.grad)
        self.assertIsNone(model.semantic_reader.bias.grad)
        self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.lesion_head.weight.grad.abs().sum()), 0)
        self.assertTrue(all((p.grad is None for p in model.encoder.parameters())))
        model.zero_grad(set_to_none=True)
        q = model.semantic_logits(batch)
        auxiliary_loss_per_image(q, batch.targets, batch.fov, torch.full((4,), 0.3)).mean().backward()
        self.assertGreater(float(model.semantic_reader.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.projection.weight.grad)
        self.assertIsNone(model.lesion_head.weight.grad)
        with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            bf16 = model.forward_components(batch)
        self.assertTrue(bool(torch.isfinite(bf16['z']).all()))
        self.assertEqual(bf16['pair_evidence'].dtype, torch.float32)
if __name__ == '__main__':
    unittest.main(verbosity=2)
