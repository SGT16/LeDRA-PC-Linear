import torch
from torch.nn import functional as F
from lesion_oof.model import build_encoder
from lesion_paircontrast.model import PairContrastModel as HistoricalPairContrastModel, GRID, BRANCH_DIM, QUERY_CHUNK_SIZE, _common_contract, _floating_finite, _reference_weights_checked, pair_features as historical_pair_features, reference_weights as historical_reference_weights, optimizer_for as historical_optimizer_for
VARIANTS = ('GLOBAL_RECT', 'GLOBAL_LINEAR')
TRAINER_ARMS = {variant: 'DIR_REAL' for variant in VARIANTS}
TRAINABLE_COUNTS = {variant: 36229 for variant in VARIANTS}
MODEL_MODE = 'GLOBAL_PAIR_CONTRAST'
BACKEND = 'torch_global_pair_contrast_fp32'
PROJECTION_DIM = BRANCH_DIM
QUERY_CHUNK = QUERY_CHUNK_SIZE
BRANCH_FINGERPRINT = dict(mode=MODEL_MODE, backend=BACKEND, branch_dim=BRANCH_DIM, query_chunk=QUERY_CHUNK_SIZE, reference_geometry='same_image_global_nonself', reference_prior='one_minus_detached_student_probability', query_prior='detached_student_probability_over_type_eligible_queries', reference_weight_dtype='float32', model_pair_dtype='float32', classifier_reader_gradient=False, teacher_auxiliary_all_variants=True, rectified_implementation='immutable_lesion_paircontrast_GLOBAL_RECT')

def global_reference_weights(probabilities, fov, query_indices=None, *, grid=GRID):
    return historical_reference_weights(probabilities, fov, None, query_indices, grid=grid, variant='GLOBAL_RECT')

def global_pair_features(projected, probabilities, fov, *, grid=GRID, variant='GLOBAL_RECT', query_chunk=QUERY_CHUNK_SIZE):
    if variant not in VARIANTS:
        raise ValueError('Known global pair-contrast variant required')
    if variant == 'GLOBAL_RECT':
        return historical_pair_features(projected, probabilities, fov, None, grid=grid, variant='GLOBAL_RECT', query_chunk=query_chunk)
    _floating_finite(projected, 'Projected global pair features')
    p, valid, _context, checked_grid = _common_contract(probabilities, fov, None, grid, 'GLOBAL_RECT')
    if projected.ndim != 3 or projected.shape[:2] != p.shape[:2] or projected.shape[-1] == 0 or (projected.device != p.device) or isinstance(query_chunk, bool) or (not isinstance(query_chunk, int)) or (query_chunk <= 0):
        raise ValueError('Projected features must align and query_chunk must be a positive integer')
    with torch.autocast(device_type=projected.device.type, enabled=False):
        value = projected if projected.dtype == torch.float64 else projected.float()
        eligibility = []
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            _weights, eligible = _reference_weights_checked(p, valid, None, indexes, checked_grid, 'GLOBAL_RECT')
            eligibility.append(eligible)
        eligible = torch.cat(eligibility, dim=1)
        query_raw = p * eligible
        query_mass = query_raw.sum(dim=1, keepdim=True)
        query_weights = query_raw / torch.where(query_mass > 0, query_mass, torch.ones_like(query_mass))
        result = value.new_zeros((len(value), 4, value.shape[-1]))
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            weights, _eligible = _reference_weights_checked(p, valid, None, indexes, checked_grid, 'GLOBAL_RECT')
            differences = value.index_select(1, indexes)[:, :, None, :] - value[:, None, :, :]
            reference_pool = torch.einsum('bqnk,bqnr->bqkr', weights.to(value.dtype), differences)
            result = result + torch.einsum('bqk,bqkr->bkr', query_weights.index_select(1, indexes).to(value.dtype), reference_pool)
        return result

class GlobalPairContrastModel(HistoricalPairContrastModel):

    def __init__(self, encoder, variant='GLOBAL_RECT', seed=823):
        if variant not in VARIANTS:
            raise ValueError('Unknown global pair-contrast variant')
        super().__init__(encoder, variant='GLOBAL_RECT', seed=seed)
        self.variant, self.mode, self.arm = (variant, MODEL_MODE, TRAINER_ARMS[variant])
        self.backend = BACKEND
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[variant]:
            raise ValueError('Global pair-contrast active parameter budget differs')

    def forward_components(self, batch):
        if self.variant == 'GLOBAL_RECT':
            return super().forward_components(batch)
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            probabilities = torch.sigmoid(q.float()).detach()
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            evidence = global_pair_features(projected, probabilities, batch.fov, grid=GRID, variant=self.variant, query_chunk=QUERY_CHUNK_SIZE)
            residual = (evidence * self.lesion_head.weight.float()[None, :, :]).sum(dim=(-1, -2))
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=evidence, probabilities=probabilities)

def optimizer_for(model):
    return historical_optimizer_for(model)

def make_model(variant, seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights or an explicit frozen encoder required')
        encoder = build_encoder(weights_path)
    return GlobalPairContrastModel(encoder, variant=variant, seed=seed).to(device)
Model = GlobalPairContrastModel
