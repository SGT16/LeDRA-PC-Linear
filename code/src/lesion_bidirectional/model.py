from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from lesion_direction.model import DirectionModel
from lesion_global.model import GlobalPairContrastModel, global_pair_features
from lesion_oof.model import DIM, build_encoder
from lesion_paircontrast.model import GRID, BRANCH_DIM, QUERY_CHUNK_SIZE, _floating_finite, optimizer_for as historical_optimizer_for
VARIANTS = ('BIDIR_PAIR', 'GENERIC_R46')
TRAINER_ARMS = {'BIDIR_PAIR': 'DIR_REAL', 'GENERIC_R46': 'B0'}
TRAINABLE_COUNTS = {'BIDIR_PAIR': 36325, 'GENERIC_R46': 36143}
MODEL_MODE = 'BIDIRECTIONAL_GLOBAL_PAIR'
BACKEND = 'torch_bidirectional_global_pair_fp32'
GENERIC_MODE = 'GENERIC_CAPACITY_CONTROL'
GENERIC_BACKEND = 'torch_generic_rank46_bf16'
MODES = {'BIDIR_PAIR': MODEL_MODE, 'GENERIC_R46': GENERIC_MODE}
BACKENDS = {'BIDIR_PAIR': BACKEND, 'GENERIC_R46': GENERIC_BACKEND}
EXTRA_SEED_OFFSET = 196613
NEGATIVE_READOUT_SEED_OFFSET = EXTRA_SEED_OFFSET
GENERIC_EXPANSION_SEED_OFFSET = EXTRA_SEED_OFFSET
GENERIC_RANK = 46
PROJECTION_DIM = BRANCH_DIM
QUERY_CHUNK = QUERY_CHUNK_SIZE
BRANCH_FINGERPRINT = dict(mode=MODEL_MODE, backend=BACKEND, branch_dim=BRANCH_DIM, query_chunk=QUERY_CHUNK_SIZE, reference_geometry='same_image_global_nonself', reference_prior='one_minus_detached_student_probability', query_prior='detached_student_probability_over_type_eligible_queries', reference_weight_dtype='float32', model_pair_dtype='float32', classifier_reader_gradient=False, teacher_auxiliary=True, auxiliary_lambda=0.1, positive_features='immutable_GLOBAL_RECT(projected)', negative_features='immutable_GLOBAL_RECT(-projected)', readout='separate_unconstrained_positive_and_negative_four_type_vectors', initialization='historical_GLOBAL_RECT_state_plus_zero_negative_readout', extra_seed_offset=EXTRA_SEED_OFFSET, generic_rank=32, trainable_parameters=TRAINABLE_COUNTS['BIDIR_PAIR'], final_rounding='sum_base_and_residual_then_original_base_dtype_then_float32')
GENERIC_FINGERPRINT = dict(mode=GENERIC_MODE, backend=GENERIC_BACKEND, generic_rank=GENERIC_RANK, teacher_auxiliary=False, auxiliary_lambda=0.0, extra_seed_offset=EXTRA_SEED_OFFSET, initialization='historical_B0_head_and_first32_down_rows; extra14_rows_isolated_Linear384to14; all_up_zero', trainable_parameters=TRAINABLE_COUNTS['GENERIC_R46'], parameter_match='nearby_generic_budget; 182_fewer_than_BIDIR_PAIR')

def variant_fingerprint(variant):
    if variant not in VARIANTS:
        raise ValueError('Known bidirectional experiment variant required')
    return dict(BRANCH_FINGERPRINT if variant == 'BIDIR_PAIR' else GENERIC_FINGERPRINT)

def bidirectional_pair_features(projected, probabilities, fov, *, grid=GRID, query_chunk=QUERY_CHUNK_SIZE):
    positive = global_pair_features(projected, probabilities, fov, grid=grid, variant='GLOBAL_RECT', query_chunk=query_chunk)
    negative = global_pair_features(-projected, probabilities, fov, grid=grid, variant='GLOBAL_RECT', query_chunk=query_chunk)
    return (positive, negative)

def bidirectional_readout(positive, negative, positive_weight, negative_weight):
    for value, name in ((positive, 'Positive evidence'), (negative, 'Negative evidence'), (positive_weight, 'Positive readout'), (negative_weight, 'Negative readout')):
        _floating_finite(value, name)
    if positive.ndim != 3 or positive.shape[1] != 4 or positive.shape[-1] == 0 or (negative.shape != positive.shape) or (positive_weight.shape != positive.shape[1:]) or (negative_weight.shape != positive_weight.shape) or any((value.device != positive.device for value in (negative, positive_weight, negative_weight))):
        raise ValueError('Aligned [B,4,R] evidence and two [4,R] readouts required')
    return (positive * positive_weight[None, :, :]).sum(dim=(-1, -2)) + (negative * negative_weight[None, :, :]).sum(dim=(-1, -2))

class BidirectionalPairModel(GlobalPairContrastModel):

    def __init__(self, encoder, seed=823):
        super().__init__(encoder, variant='GLOBAL_RECT', seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + EXTRA_SEED_OFFSET)
            self.lesion_head_negative = nn.Linear(BRANCH_DIM, 4, bias=False)
        nn.init.zeros_(self.lesion_head_negative.weight)
        self.variant, self.mode, self.arm = ('BIDIR_PAIR', MODEL_MODE, 'DIR_REAL')
        self.backend = BACKEND
        self.gradient_policy = 'separate_teacher_only_reader_probability_stop_gradient_two_signed_readouts'
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[self.variant]:
            raise ValueError('Bidirectional parameter budget differs')

    def parameter_counts(self):
        result = super().parameter_counts()
        if hasattr(self, 'lesion_head_negative'):
            result['lesion_head_negative'] = sum((p.numel() for p in self.lesion_head_negative.parameters() if p.requires_grad))
        return result

    def forward_components(self, batch):
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            probabilities = torch.sigmoid(q.float()).detach()
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            positive, negative = bidirectional_pair_features(projected, probabilities, batch.fov, grid=GRID, query_chunk=QUERY_CHUNK_SIZE)
            residual = bidirectional_readout(positive, negative, self.lesion_head.weight.float(), self.lesion_head_negative.weight.float())
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=positive, positive_evidence=positive, negative_evidence=negative, probabilities=probabilities)

    def load_rect_subfamily_state(self, state):
        from collections.abc import Mapping
        if not isinstance(state, Mapping) or 'lesion_head_negative.weight' in state:
            raise ValueError('An exact historical RECT trainable state is required')
        expanded = dict(state)
        expanded['lesion_head_negative.weight'] = torch.zeros_like(self.lesion_head_negative.weight)
        return self.load_trainable_state(expanded)

    @contextmanager
    def zero_readout(self):
        before = [module.weight.detach().clone() for module in (self.lesion_head, self.lesion_head_negative)]
        with torch.no_grad():
            self.lesion_head.weight.zero_()
            self.lesion_head_negative.weight.zero_()
        try:
            yield self
        finally:
            with torch.no_grad():
                self.lesion_head.weight.copy_(before[0])
                self.lesion_head_negative.weight.copy_(before[1])

class GenericRank46Model(DirectionModel):

    def __init__(self, encoder, seed=823):
        super().__init__(encoder, arm='B0', seed=seed)
        old_down, old_up = (self.adapter[0], self.adapter[2])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + EXTRA_SEED_OFFSET)
            extra_down = nn.Linear(DIM, GENERIC_RANK - 32, bias=True)
            new_down = nn.Linear(DIM, GENERIC_RANK, bias=True)
            new_up = nn.Linear(GENERIC_RANK, DIM, bias=True)
        with torch.no_grad():
            new_down.weight[:32].copy_(old_down.weight)
            new_down.bias[:32].copy_(old_down.bias)
            new_down.weight[32:].copy_(extra_down.weight)
            new_down.bias[32:].copy_(extra_down.bias)
            new_up.weight.zero_()
            new_up.weight[:, :32].copy_(old_up.weight)
            new_up.bias.copy_(old_up.bias)
        self.adapter[0], self.adapter[2] = (new_down, new_up)
        self.variant, self.mode, self.arm = ('GENERIC_R46', GENERIC_MODE, 'B0')
        self.backend, self.generic_rank = (GENERIC_BACKEND, GENERIC_RANK)
        self.gradient_policy = 'generic_classification_only_no_teacher_objective'
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[self.variant]:
            raise ValueError('Generic rank46 parameter budget differs')

    def forward_components(self, batch):
        base = self.head(self.pooled_features(batch)).squeeze(-1)
        return dict(z=base.float(), q=None, base=base.float(), residual=base.float() * 0.0)

    def forward_training(self, batch):
        return (self.forward_tokens(batch), None)

def optimizer_for(model):
    return historical_optimizer_for(model)

def make_model(variant, seed=823, encoder=None, device='cuda', weights_path=None):
    if variant not in VARIANTS:
        raise ValueError('Unknown bidirectional experiment variant')
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights or an explicit frozen encoder required')
        encoder = build_encoder(weights_path)
    model = BidirectionalPairModel(encoder, seed=seed) if variant == 'BIDIR_PAIR' else GenericRank46Model(encoder, seed=seed)
    return model.to(device)
Model = BidirectionalPairModel
