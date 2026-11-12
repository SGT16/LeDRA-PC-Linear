from __future__ import annotations
import torch
from torch.nn import functional as F
from lesion_pc_linear_ablation_v1.model import LinearAblationModel, optimizer_for as reference_optimizer_for
from lesion_paircontrast.model import GRID, BRANCH_DIM, _common_contract, _floating_finite
from lesion_oof.model import build_encoder
CONTROL_IDS = ('LINEAR_POOL_ONLY',)
AUXILIARY_COEFFICIENTS = {'LINEAR_POOL_ONLY': 0.1}
NOMINAL_TRAINABLE_PARAMETERS = 36229
OBJECTIVE_CONNECTED_PARAMETERS = {'LINEAR_POOL_ONLY': 36229}
MODEL_MODE = 'PC_LINEAR_PURE_LESION_POOLING_CONTROL_V1'

def lesion_weighted_pool_features(projected, probabilities, fov, *, grid=GRID):
    _floating_finite(projected, 'Projected pure-pooling features')
    p, valid, _unused_context, _checked_grid = _common_contract(probabilities, fov, None, grid, 'GLOBAL_RECT')
    if projected.ndim != 3 or projected.shape[:2] != p.shape[:2] or projected.shape[-1] == 0 or (projected.device != p.device):
        raise ValueError('Projected pooling features must align with lesion probabilities')
    with torch.autocast(device_type=projected.device.type, enabled=False):
        value = projected if projected.dtype == torch.float64 else projected.float()
        raw = p * valid[..., None]
        mass = raw.sum(dim=1, keepdim=True)
        beta = raw / torch.where(mass > 0, mass, torch.ones_like(mass))
        return torch.einsum('bil,bir->blr', beta.to(value.dtype), value)

class LesionPoolingModel(LinearAblationModel):

    def __init__(self, encoder, control_id='LINEAR_POOL_ONLY', seed=823):
        if control_id not in CONTROL_IDS:
            raise ValueError('Unknown lesion-weighted pooling control')
        super().__init__(encoder, control_id='LINEAR_REF', seed=seed)
        self.control_id = control_id
        self.control_mode = MODEL_MODE
        self.auxiliary_coefficient = AUXILIARY_COEFFICIENTS[control_id]
        if self.parameter_counts()['trainable'] != NOMINAL_TRAINABLE_PARAMETERS:
            raise ValueError('Pure-pooling control parameter count changed')

    def scientific_metadata(self):
        return dict(control_id=self.control_id, source_variant=self.variant, reference_control_id='LINEAR_REF', control_mode=MODEL_MODE, auxiliary_coefficient=self.auxiliary_coefficient, nominal_trainable_parameters=NOMINAL_TRAINABLE_PARAMETERS, objective_connected_parameters=OBJECTIVE_CONNECTED_PARAMETERS[self.control_id], reader_parameters=1540, reader_policy='teacher_only', classification_reader_gradient=False, evidence_rule='sum_i beta_il V_i; beta proportional to detached p_il within RGB FOV', query_prior='detached_student_probability_over_all_valid_FOV_queries', reference_rule='not_used', reference_subtraction=False, complementary_reference_eligibility=False, pair_rectification=False, branch_dim=BRANCH_DIM, zero_mass_rule='zero_evidence_no_floor_no_fallback', singleton_FOV_rule='positive_mass_pools_its_single_projected_feature', permutation_recipe=None, decision_readout_initialization='all_four_vectors_zero', same_architecture_state_keys=True)

    def forward_components(self, batch):
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            probabilities = torch.sigmoid(q.float()).detach()
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            evidence = lesion_weighted_pool_features(projected, probabilities, batch.fov, grid=GRID)
            residual = (evidence * self.lesion_head.weight.float()[None, :, :]).sum(dim=(-1, -2))
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=evidence, lesion_pooled_evidence=evidence, probabilities=probabilities)

def optimizer_for(model):
    return reference_optimizer_for(model)

def make_model(control_id='LINEAR_POOL_ONLY', seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('Explicit frozen encoder or general pretrained weights required')
        encoder = build_encoder(weights_path)
    return LesionPoolingModel(encoder, control_id=control_id, seed=seed).to(device)
Model = LesionPoolingModel
