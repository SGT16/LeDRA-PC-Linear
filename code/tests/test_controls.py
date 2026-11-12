import copy
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from lesion_direction.data import auxiliary_loss_per_image
from lesion_direction.model import DirectionBatch
from lesion_global.model import GlobalPairContrastModel
from lesion_oof.model import DIM, PATCHES, PREFIX, TOKENS, state_hash
from lesion_pc_linear_ablation_v1.model import AlignmentBatch, CONTROL_IDS, LinearAblationModel, deterministic_permutations, optimizer_for, permute_probabilities, uniform_reference_pair_features, uniform_reference_weights

class FakeEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_prefix_tokens, self.num_features, self.global_pool = (PREFIX, DIM, 'avg')
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(10)] + [nn.LayerNorm(DIM), nn.LayerNorm(DIM)])
        self.norm = nn.LayerNorm(DIM)

    def forward_head(self, value):
        return value[:, PREFIX:].mean(dim=1)

def uniform_loop_oracle(values, probabilities, fov):
    p = probabilities.detach().float().cpu().tolist()
    valid = fov.cpu().tolist()
    images = []
    for image in range(len(values)):
        inside = [i for i, included in enumerate(valid[image]) if included]
        types = []
        for lesion in range(4):
            eligible = inside if len(inside) > 1 else []
            mass = sum((p[image][i][lesion] for i in eligible))
            result = values[image].sum(dim=0) * 0
            if mass > 0:
                for query in eligible:
                    references = [j for j in inside if j != query]
                    contribution = sum((values[image, query] - values[image, j] for j in references)) / len(references)
                    result = result + p[image][query][lesion] / mass * contribution
            types.append(result)
        images.append(torch.stack(types))
    return torch.stack(images)

class PCLinearControlTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        assert not torch.cuda.is_initialized(), 'CPU-only checks initialized CUDA'

    def synthetic(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(719)
            encoder = FakeEncoder()
            tokens = torch.randn(1, TOKENS, DIM)
            fov = torch.zeros(1, PATCHES, dtype=torch.bool)
            fov[:, [0, 7, 38, 211, 700, 783]] = True
            targets = torch.sigmoid(tokens[:, PREFIX:, :4] * 0.3 + 0.5)
            return (encoder, DirectionBatch(tokens, targets, fov))

    @staticmethod
    def activate(model):
        with torch.no_grad():
            model.lesion_head.weight.copy_(torch.linspace(-0.08, 0.09, 96).reshape(4, 24))
            model.adapter[2].weight.copy_(torch.linspace(-0.01, 0.01, DIM * 32).reshape(DIM, 32))

    def test_uniform_values_and_gradients_match_independent_oracle(self):
        generator = torch.Generator().manual_seed(334)
        original = torch.randn(2, 7, 3, generator=generator, dtype=torch.float64)
        probabilities = torch.rand(2, 7, 4, generator=generator).requires_grad_()
        with torch.no_grad():
            probabilities[0, :, 0] = 0
            probabilities[1, :, 3] = 1
        fov = torch.tensor([[True, False, True, True, False, True, True], [False, True, True, False, True, False, True]])
        actual_source = original.clone().requires_grad_()
        actual = uniform_reference_pair_features(actual_source, probabilities, fov, grid=(1, 7), query_chunk=3)
        oracle_source = original.clone().requires_grad_()
        expected = uniform_loop_oracle(oracle_source, probabilities, fov)
        coefficient = torch.linspace(-0.5, 0.8, 24, dtype=torch.float64).reshape(2, 4, 3)
        actual_grad, p_grad = torch.autograd.grad((actual * coefficient).sum(), (actual_source, probabilities), allow_unused=True)
        oracle_grad = torch.autograd.grad((expected * coefficient).sum(), oracle_source)[0]
        torch.testing.assert_close(actual, expected, rtol=2e-06, atol=2e-07)
        torch.testing.assert_close(actual_grad, oracle_grad, rtol=2e-06, atol=2e-07)
        self.assertIsNone(p_grad)
        self.assertEqual(int(torch.count_nonzero(actual_grad[~fov])), 0)
        self.assertEqual(int(torch.count_nonzero(actual[0, 0])), 0)
        weights, eligible = uniform_reference_weights(probabilities, fov, grid=(1, 7))
        self.assertTrue(bool(eligible[1, fov[1], 3].all()))
        torch.testing.assert_close(weights.sum(dim=2)[eligible], torch.ones_like(weights.sum(dim=2)[eligible]), rtol=1e-06, atol=1e-07)

    def test_uniform_closed_form_and_zero_mass_edges(self):
        values = torch.tensor([[[-1.0, 2.0], [3.0, -4.0], [2.0, 1.0]]], dtype=torch.float64)
        p = torch.tensor([[[0.2, 0.0, 1.0, 0.8], [0.8, 0.0, 1.0, 0.1], [0.4, 0.0, 1.0, 0.7]]])
        fov = torch.ones(1, 3, dtype=torch.bool)
        actual = uniform_reference_pair_features(values, p, fov, grid=(1, 3), query_chunk=1)
        for lesion in (0, 2, 3):
            b = p[0, :, lesion].double() / p[0, :, lesion].double().sum()
            expected = 1.5 * ((values[0] * b[:, None]).sum(dim=0) - values[0].mean(dim=0))
            torch.testing.assert_close(actual[0, lesion], expected, rtol=2e-06, atol=3e-07)
        for included in ([True, False, False], [False, True, False]):
            singleton = torch.tensor([included])
            result = uniform_reference_pair_features(values, p, singleton, grid=(1, 3))
            torch.testing.assert_close(result, torch.zeros_like(result), atol=0, rtol=0)
        two = uniform_reference_pair_features(values[:, :2], torch.tensor([[[1.0] * 4, [0.0] * 4]]), fov[:, :2], grid=(1, 2))
        torch.testing.assert_close(two[0], (values[0, 0] - values[0, 1]).expand(4, -1), atol=0, rtol=0)
        constant = uniform_reference_pair_features(values, torch.full_like(p, 0.5), fov, grid=(1, 3))
        torch.testing.assert_close(constant, torch.zeros_like(constant), atol=1e-15, rtol=0)

    def test_all_controls_preserve_initial_state_rng_and_checkpoint_layout(self):
        encoder, _batch = self.synthetic()
        before = torch.get_rng_state().clone()
        for seed in (823, 289, 592, 605, 329):
            historical = GlobalPairContrastModel(copy.deepcopy(encoder), 'GLOBAL_LINEAR', seed)
            expected_hash = state_hash(historical.export_trainable_state())
            for control_id in CONTROL_IDS:
                model = LinearAblationModel(copy.deepcopy(encoder), control_id, seed)
                self.assertEqual(model.variant, 'GLOBAL_LINEAR')
                self.assertEqual(model.parameter_counts()['trainable'], 36229)
                self.assertEqual(state_hash(model.export_trainable_state()), expected_hash)
                self.assertEqual(set(model.export_trainable_state()), set(historical.export_trainable_state()))
                metadata = model.scientific_metadata()
                self.assertEqual(metadata['objective_connected_parameters'], 34689 if control_id == 'LINEAR_NOAUX' else 36229)
                self.assertEqual(metadata['auxiliary_coefficient'], 0.0 if control_id == 'LINEAR_NOAUX' else 0.1)
                self.assertTrue(all((not parameter.requires_grad for parameter in model.encoder.parameters())))
        torch.testing.assert_close(torch.get_rng_state(), before, atol=0, rtol=0)

    def test_reference_forward_and_gradients_are_bit_exact_with_historical(self):
        encoder, batch = self.synthetic()
        historical = GlobalPairContrastModel(copy.deepcopy(encoder), 'GLOBAL_LINEAR', 823)
        reference = LinearAblationModel(copy.deepcopy(encoder), 'LINEAR_REF', 823)
        self.activate(historical)
        reference.load_trainable_state(historical.export_trainable_state())
        old = historical.forward_components(batch)
        new = reference.forward_components(batch)
        for name in ('z', 'q', 'base', 'residual', 'pair_evidence', 'probabilities'):
            torch.testing.assert_close(new[name], old[name], atol=0, rtol=0)
        for model, components in ((historical, old), (reference, new)):
            aux = auxiliary_loss_per_image(components['q'], batch.targets, batch.fov, torch.full((4,), 0.3)).mean()
            loss = F.binary_cross_entropy_with_logits(components['z'], torch.ones(1)) + 0.1 * aux
            loss.backward()
        for (old_name, old_parameter), (new_name, new_parameter) in zip(historical.named_parameters(), reference.named_parameters()):
            self.assertEqual(old_name, new_name)
            if old_parameter.grad is None:
                self.assertIsNone(new_parameter.grad)
            else:
                torch.testing.assert_close(new_parameter.grad, old_parameter.grad, atol=0, rtol=0)
        with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            old_bf16 = historical.forward_components(batch)
            new_bf16 = reference.forward_components(batch)
        for name in ('z', 'q', 'base', 'residual', 'pair_evidence', 'probabilities'):
            torch.testing.assert_close(new_bf16[name], old_bf16[name], atol=0, rtol=0)

    def test_shuffle_uses_stable_joint_map_and_leaves_teacher_logits_unchanged(self):
        encoder, batch = self.synthetic()
        rows = [dict(dataset='brset', source_image_key='immutable-test-image')]
        mapping = deterministic_permutations(rows, batch.fov, 'brset')
        torch.manual_seed(900)
        replay = deterministic_permutations(rows, batch.fov, 'brset')
        torch.testing.assert_close(mapping, replay, atol=0, rtol=0)
        model = LinearAblationModel(encoder, 'LINEAR_SHUFFLED', 823)
        self.activate(model)
        carrier = AlignmentBatch(batch.tokens, batch.targets, batch.fov, mapping)
        components = model.forward_components(carrier)
        raw_q = model.semantic_logits(batch)
        torch.testing.assert_close(components['q'], raw_q, atol=0, rtol=0)
        raw_p = torch.sigmoid(raw_q).detach()
        torch.testing.assert_close(components['probabilities'], raw_p.gather(1, mapping[..., None].expand(-1, -1, 4)), atol=0, rtol=0)
        torch.testing.assert_close(components['probabilities'][~batch.fov], raw_p[~batch.fov], atol=0, rtol=0)
        components['z'].square().sum().backward()
        self.assertIsNone(model.semantic_reader.weight.grad)
        self.assertIsNone(model.semantic_reader.bias.grad)
        with self.assertRaises(ValueError):
            model.forward_components(batch)

    def test_noaux_reader_has_no_gradient_or_weight_decay_while_classifier_learns(self):
        encoder, batch = self.synthetic()
        model = LinearAblationModel(encoder, 'LINEAR_NOAUX', 823)
        self.activate(model)
        before_reader = copy.deepcopy(model.semantic_reader.state_dict())
        before_head = model.head.weight.detach().clone()
        optimizer = optimizer_for(model)
        optimizer.zero_grad(set_to_none=True)
        z, q = model.forward_training(batch)
        loss = F.binary_cross_entropy_with_logits(z, torch.ones_like(z))
        loss.backward()
        self.assertIsNone(model.semantic_reader.weight.grad)
        self.assertIsNone(model.semantic_reader.bias.grad)
        optimizer.step()
        for name, parameter in model.semantic_reader.state_dict().items():
            torch.testing.assert_close(parameter, before_reader[name], atol=0, rtol=0)
        self.assertFalse(torch.equal(model.head.weight, before_head))
        self.assertEqual(q.shape, (1, PATCHES, 4))
        self.assertFalse(torch.cuda.is_initialized())
if __name__ == '__main__':
    unittest.main(verbosity=2)
