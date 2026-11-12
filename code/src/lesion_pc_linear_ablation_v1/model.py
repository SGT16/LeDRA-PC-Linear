from __future__ import annotations
import torch
from torch.nn import functional as F
from lesion_alignment.model import AlignmentBatch, AlignmentInput, CONTROL_SEED, PERMUTATION_RECIPE, deterministic_permutations, permutation_digest, permute_probabilities
from lesion_global.model import GlobalPairContrastModel, global_pair_features
from lesion_oof.model import build_encoder
from lesion_paircontrast.model import GRID, BRANCH_DIM, QUERY_CHUNK_SIZE, _common_contract, _floating_finite, _query_indices, optimizer_for as historical_optimizer_for
CONTROL_IDS = ('LINEAR_REF', 'LINEAR_SHUFFLED', 'LINEAR_UNIFORM_REFERENCE', 'LINEAR_NOAUX')
AUXILIARY_COEFFICIENTS = {key: 0.0 if key == 'LINEAR_NOAUX' else 0.1 for key in CONTROL_IDS}
NOMINAL_TRAINABLE_PARAMETERS = 36229
READER_PARAMETERS = 1540
OBJECTIVE_CONNECTED_PARAMETERS = {key: NOMINAL_TRAINABLE_PARAMETERS - (READER_PARAMETERS if key == 'LINEAR_NOAUX' else 0) for key in CONTROL_IDS}
MODEL_MODE = 'PC_LINEAR_MECHANISM_ABLATIONS_V1'

def _uniform_reference_weights_checked(probabilities, valid, indexes):
    p = probabilities
    with torch.no_grad(), torch.autocast(device_type=p.device.type, enabled=False):
        references = torch.arange(p.shape[1], dtype=torch.int64, device=p.device)
        allowed = valid.index_select(1, indexes)[:, :, None] & valid[:, None, :] & (indexes[:, None] != references[None, :])
        raw = allowed.float()[..., None].expand(-1, -1, -1, 4)
        mass = raw.sum(dim=2, keepdim=True)
        weights = raw / torch.where(mass > 0, mass, torch.ones_like(mass))
        return (weights.detach(), (mass > 0).squeeze(2).detach())

def uniform_reference_weights(probabilities, fov, query_indices=None, *, grid=GRID):
    p, valid, _context, _checked_grid = _common_contract(probabilities, fov, None, grid, 'GLOBAL_RECT')
    indexes = _query_indices(query_indices, p.shape[1], p.device)
    return _uniform_reference_weights_checked(p, valid, indexes)

def uniform_reference_pair_features(projected, probabilities, fov, *, grid=GRID, query_chunk=QUERY_CHUNK_SIZE):
    _floating_finite(projected, 'Projected uniform-reference features')
    p, valid, _context, checked_grid = _common_contract(probabilities, fov, None, grid, 'GLOBAL_RECT')
    if projected.ndim != 3 or projected.shape[:2] != p.shape[:2] or projected.shape[-1] == 0 or (projected.device != p.device) or isinstance(query_chunk, bool) or (not isinstance(query_chunk, int)) or (query_chunk <= 0):
        raise ValueError('Aligned projected features and a positive integer chunk required')
    with torch.autocast(device_type=projected.device.type, enabled=False):
        value = projected if projected.dtype == torch.float64 else projected.float()
        eligibility = []
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            _weights, eligible = _uniform_reference_weights_checked(p, valid, indexes)
            eligibility.append(eligible)
        eligible = torch.cat(eligibility, dim=1)
        query_raw = p * eligible
        query_mass = query_raw.sum(dim=1, keepdim=True)
        query_weights = query_raw / torch.where(query_mass > 0, query_mass, torch.ones_like(query_mass))
        result = value.new_zeros((len(value), 4, value.shape[-1]))
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            weights, _eligible = _uniform_reference_weights_checked(p, valid, indexes)
            differences = value.index_select(1, indexes)[:, :, None, :] - value[:, None, :, :]
            reference_pool = torch.einsum('bqnk,bqnr->bqkr', weights.to(value.dtype), differences)
            result = result + torch.einsum('bqk,bqkr->bkr', query_weights.index_select(1, indexes).to(value.dtype), reference_pool)
        return result

class LinearAblationModel(GlobalPairContrastModel):

    def __init__(self, encoder, control_id='LINEAR_REF', seed=823):
        if control_id not in CONTROL_IDS:
            raise ValueError('Unknown PC-Linear mechanism control')
        super().__init__(encoder, variant='GLOBAL_LINEAR', seed=seed)
        self.control_id = control_id
        self.auxiliary_coefficient = AUXILIARY_COEFFICIENTS[control_id]
        self.control_mode = MODEL_MODE
        if self.parameter_counts()['trainable'] != NOMINAL_TRAINABLE_PARAMETERS:
            raise ValueError('PC-Linear control nominal parameter count changed')

    def scientific_metadata(self):
        return dict(control_id=self.control_id, source_variant=self.variant, control_mode=MODEL_MODE, auxiliary_coefficient=self.auxiliary_coefficient, nominal_trainable_parameters=NOMINAL_TRAINABLE_PARAMETERS, objective_connected_parameters=OBJECTIVE_CONNECTED_PARAMETERS[self.control_id], reader_parameters=READER_PARAMETERS, reader_policy='seeded_initialization_no_gradient_no_decay' if self.control_id == 'LINEAR_NOAUX' else 'teacher_only', classification_reader_gradient=False, pair_rectification=False, branch_dim=BRANCH_DIM, reference_rule='uniform_valid_nonself' if self.control_id == 'LINEAR_UNIFORM_REFERENCE' else 'one_minus_detached_student_probability_valid_nonself', permutation_recipe=dict(PERMUTATION_RECIPE) if self.control_id == 'LINEAR_SHUFFLED' else None, decision_readout_initialization='all_four_vectors_zero', same_architecture_state_keys=True)

    def forward_components(self, batch):
        if self.control_id in ('LINEAR_REF', 'LINEAR_NOAUX'):
            return super().forward_components(batch)
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            raw_probabilities = torch.sigmoid(q.float()).detach()
            probabilities = raw_probabilities
            if self.control_id == 'LINEAR_SHUFFLED':
                permutation = getattr(batch, 'permutation', None)
                if permutation is None:
                    raise ValueError('LINEAR_SHUFFLED requires the fixed image-keyed FOV permutation')
                probabilities = permute_probabilities(raw_probabilities, batch.fov, permutation)
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            if self.control_id == 'LINEAR_UNIFORM_REFERENCE':
                evidence = uniform_reference_pair_features(projected, probabilities, batch.fov, grid=GRID, query_chunk=QUERY_CHUNK_SIZE)
            else:
                evidence = global_pair_features(projected, probabilities, batch.fov, grid=GRID, variant='GLOBAL_LINEAR', query_chunk=QUERY_CHUNK_SIZE)
            residual = (evidence * self.lesion_head.weight.float()[None, :, :]).sum(dim=(-1, -2))
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=evidence, probabilities=probabilities, raw_probabilities=raw_probabilities)

def optimizer_for(model):
    return historical_optimizer_for(model)

def make_model(control_id, seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('Explicit frozen encoder or general pretrained weights required')
        encoder = build_encoder(weights_path)
    return LinearAblationModel(encoder, control_id=control_id, seed=seed).to(device)
Model = LinearAblationModel
