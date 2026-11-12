from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from lesion_direction.model import DirectionBatch, DirectionModel, _tokens
from lesion_oof.model import DIM, PATCHES, PREFIX, SPLIT_BLOCK, build_encoder
VARIANTS = ('MATCH_RECT', 'MATCH_LINEAR', 'GLOBAL_RECT', 'LOCAL_UNIFORM_RECT')
TRAINER_ARMS = {variant: 'DIR_REAL' for variant in VARIANTS}
TRAINABLE_COUNTS = {variant: 36229 for variant in VARIANTS}
GRID = (28, 28)
BRANCH_DIM = 24
QUERY_CHUNK_SIZE = 32
MATCH_TEMPERATURE = 0.2
MIN_REFERENCE_DISTANCE = 2.0
MAX_REFERENCE_DISTANCE = 8.0
READER_SEED_OFFSET = 104729
PROJECTION_SEED_OFFSET = 130363
BACKEND = 'torch_reader_null_pair_contrast_fp32'
MODEL_MODE = 'LOCAL_PAIR_CONTRAST'
PROJECTION_DIM = BRANCH_DIM
QUERY_CHUNK = QUERY_CHUNK_SIZE
TEMPERATURE = MATCH_TEMPERATURE
MIN_DISTANCE = MIN_REFERENCE_DISTANCE
MAX_DISTANCE = MAX_REFERENCE_DISTANCE
MATCHED_VARIANTS = ('MATCH_RECT', 'MATCH_LINEAR')

def _grid_contract(grid, length):
    if not isinstance(grid, (tuple, list)) or len(grid) != 2 or any((isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in grid)) or (grid[0] * grid[1] != length):
        raise ValueError('Positive integer grid must contain exactly the patch count')
    return tuple(grid)

def _floating_finite(value, name):
    if not torch.is_tensor(value) or not torch.is_floating_point(value) or (not bool(torch.isfinite(value).all())):
        raise ValueError(name + ' must be a finite floating-point tensor')

def _common_contract(probabilities, fov, context, grid, variant):
    _floating_finite(probabilities, 'Student probabilities')
    if variant not in VARIANTS or probabilities.ndim != 3 or probabilities.shape[-1] != 4 or (probabilities.shape[0] == 0) or (probabilities.shape[1] == 0) or (not bool(((probabilities >= 0) & (probabilities <= 1)).all())):
        raise ValueError('Known variant and four probabilities in [0,1] per patch required')
    if not torch.is_tensor(fov) or fov.dtype != torch.bool or fov.shape != probabilities.shape[:2] or (fov.device != probabilities.device) or (not bool(fov.any(dim=1).all())):
        raise ValueError('Aligned nonempty explicit RGB FOV required')
    checked_grid = _grid_contract(grid, probabilities.shape[1])
    normalized = None
    if context is None:
        if variant in MATCHED_VARIANTS:
            raise ValueError('Matched variants require a reader-complement context')
    else:
        _floating_finite(context, 'Matching context')
        if context.ndim != 3 or context.shape[:2] != probabilities.shape[:2] or context.shape[-1] == 0 or (context.device != probabilities.device):
            raise ValueError('Matching context must align with probability patches/device')
        with torch.no_grad(), torch.autocast(device_type=context.device.type, enabled=False):
            normalized = F.normalize(context.detach().float(), p=2, dim=-1, eps=1e-12)
    return (probabilities.detach().float(), fov.detach(), normalized, checked_grid)

def _query_indices(query_indices, length, device):
    if query_indices is None:
        return torch.arange(length, dtype=torch.int64, device=device)
    indexes = torch.as_tensor(query_indices, device=device)
    if indexes.ndim != 1 or indexes.dtype != torch.int64 or indexes.numel() == 0 or (not bool(((indexes >= 0) & (indexes < length)).all())):
        raise ValueError('Nonempty one-dimensional int64 query indexes within the grid required')
    return indexes.detach()

def reader_null_context(tokens, reader_weight):
    _floating_finite(tokens, 'Context tokens')
    _floating_finite(reader_weight, 'Reader weights')
    if tokens.ndim != 3 or tokens.shape[0] == 0 or tokens.shape[1] == 0 or (tokens.shape[2] == 0) or (reader_weight.ndim != 2) or (reader_weight.shape != (4, tokens.shape[2])) or (reader_weight.device != tokens.device):
        raise ValueError('Context tokens [B,N,D] and four reader rows [4,D] must align')
    with torch.no_grad(), torch.autocast(device_type=tokens.device.type, enabled=False):
        value, readers = (tokens.detach().float(), reader_weight.detach().float())
        _left, singular, right = torch.linalg.svd(readers, full_matrices=False)
        tolerance = torch.finfo(torch.float32).eps * max(readers.shape) * singular.amax()
        basis = right[singular > tolerance]
        if basis.shape[0] == value.shape[-1]:
            complement = torch.zeros_like(value)
        else:
            complement = value - value @ basis.t() @ basis
        return F.normalize(complement, p=2, dim=-1, eps=1e-12).detach()

def _reference_weights_checked(probabilities, fov, context, indexes, grid, variant):
    with torch.no_grad(), torch.autocast(device_type=probabilities.device.type, enabled=False):
        length, width = (probabilities.shape[1], grid[1])
        references = torch.arange(length, dtype=torch.int64, device=probabilities.device)
        allowed = fov.index_select(1, indexes)[:, :, None] & fov[:, None, :] & (indexes[:, None] != references[None, :])
        if variant != 'GLOBAL_RECT':
            row_delta = torch.div(indexes, width, rounding_mode='floor')[:, None] - torch.div(references, width, rounding_mode='floor')[None, :]
            column_delta = indexes.remainder(width)[:, None] - references.remainder(width)[None, :]
            squared_distance = row_delta.square() + column_delta.square()
            annulus = (squared_distance >= MIN_REFERENCE_DISTANCE ** 2) & (squared_distance <= MAX_REFERENCE_DISTANCE ** 2)
            allowed = allowed & annulus[None, :, :]
        if variant in MATCHED_VARIANTS:
            cosine = torch.bmm(context.index_select(1, indexes), context.transpose(1, 2))
            kernel = torch.exp(cosine / MATCH_TEMPERATURE)
        else:
            kernel = torch.ones(allowed.shape, device=probabilities.device, dtype=torch.float32)
        raw = kernel[..., None] * (1.0 - probabilities[:, None, :, :]) * allowed[..., None]
        mass = raw.sum(dim=2, keepdim=True)
        positive = mass > 0
        weights = raw / torch.where(positive, mass, torch.ones_like(mass))
        eligible = positive.squeeze(2)
        return (weights.detach(), eligible.detach())

def reference_weights(probabilities, fov, context=None, query_indices=None, *, grid=GRID, variant='MATCH_RECT'):
    p, valid, normalized, checked_grid = _common_contract(probabilities, fov, context, grid, variant)
    indexes = _query_indices(query_indices, p.shape[1], p.device)
    return _reference_weights_checked(p, valid, normalized, indexes, checked_grid, variant)

def pair_features(projected, probabilities, fov, context=None, *, grid=GRID, variant='MATCH_RECT', query_chunk=QUERY_CHUNK_SIZE):
    _floating_finite(projected, 'Projected pair features')
    p, valid, normalized, checked_grid = _common_contract(probabilities, fov, context, grid, variant)
    if projected.ndim != 3 or projected.shape[:2] != p.shape[:2] or projected.shape[-1] == 0 or (projected.device != p.device) or isinstance(query_chunk, bool) or (not isinstance(query_chunk, int)) or (query_chunk <= 0):
        raise ValueError('Projected features must align and query_chunk must be a positive integer')
    with torch.autocast(device_type=projected.device.type, enabled=False):
        value = projected if projected.dtype == torch.float64 else projected.float()
        eligibility = []
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            _weights, eligible = _reference_weights_checked(p, valid, normalized, indexes, checked_grid, variant)
            eligibility.append(eligible)
        eligible = torch.cat(eligibility, dim=1)
        query_raw = p * eligible
        query_mass = query_raw.sum(dim=1, keepdim=True)
        query_weights = query_raw / torch.where(query_mass > 0, query_mass, torch.ones_like(query_mass))
        result = value.new_zeros((len(value), 4, value.shape[-1]))
        for lo in range(0, p.shape[1], query_chunk):
            indexes = torch.arange(lo, min(lo + query_chunk, p.shape[1]), dtype=torch.int64, device=p.device)
            weights, _eligible = _reference_weights_checked(p, valid, normalized, indexes, checked_grid, variant)
            differences = value.index_select(1, indexes)[:, :, None, :] - value[:, None, :, :]
            pairs = differences if variant == 'MATCH_LINEAR' else F.relu(differences)
            reference_pool = torch.einsum('bqnk,bqnr->bqkr', weights.to(value.dtype), pairs)
            result = result + torch.einsum('bqk,bqkr->bkr', query_weights.index_select(1, indexes).to(value.dtype), reference_pool)
        return result

class PairContrastModel(DirectionModel):

    def __init__(self, encoder, variant='MATCH_RECT', seed=823):
        if variant not in VARIANTS:
            raise ValueError('Unknown pair-contrast variant')
        super().__init__(encoder, arm='B0', seed=seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + READER_SEED_OFFSET)
            self.semantic_reader = nn.Linear(DIM, 4, bias=True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed) + PROJECTION_SEED_OFFSET)
            self.projection = nn.Linear(DIM, BRANCH_DIM, bias=False)
            self.lesion_head = nn.Linear(BRANCH_DIM, 4, bias=False)
        nn.init.zeros_(self.lesion_head.weight)
        self.variant, self.mode, self.arm = (variant, MODEL_MODE, TRAINER_ARMS[variant])
        self.generic_rank, self.reader_start = (32, None)
        self.gradient_policy = 'separate_teacher_only_reader_probability_and_matching_stop_gradient'
        self.backend = BACKEND
        if hasattr(self.encoder, 'fc_norm') and (not isinstance(self.encoder.fc_norm, nn.Identity)):
            raise ValueError('Current feature contract requires original identity fc_norm')
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[variant]:
            raise ValueError('Pair-contrast parameter budget differs')

    def parameter_counts(self):
        result = super().parameter_counts()
        result['frozen'] = sum((p.numel() for p in self.parameters() if not p.requires_grad))
        result['frozen_encoder'] = sum((p.numel() for p in self.encoder.parameters()))
        result['generic_adapter'] = sum((p.numel() for p in self.adapter.parameters()))
        for name in ('semantic_reader', 'projection', 'lesion_head'):
            if hasattr(self, name):
                result[name] = sum((p.numel() for p in getattr(self, name).parameters() if p.requires_grad))
        return result

    def adapted_tokens(self, batch, *, semantic=None):
        value = _tokens(batch).to(dtype=self.head.weight.dtype)
        patch = value[:, PREFIX:]
        return torch.cat((value[:, :PREFIX], patch + self.adapter(patch)), dim=1)

    def _features(self, batch):
        if not isinstance(batch, DirectionBatch) or batch.fov is None:
            raise ValueError('Inference requires token grid and deterministic RGB FOV')
        value = self.adapted_tokens(batch)
        if batch.fov.shape != (len(value), PATCHES) or batch.fov.dtype != torch.bool or batch.fov.device != value.device or (not bool(batch.fov.any(dim=1).all())):
            raise ValueError('Aligned nonempty explicit RGB FOV required')
        for block in self.encoder.blocks[SPLIT_BLOCK:]:
            value = block(value)
        normalized = self.encoder.norm(value)
        pooled = self.encoder.forward_head(normalized)
        if pooled.shape != (len(value), DIM):
            raise ValueError('Expected 384-dimensional patch-average features')
        return (pooled, normalized[:, PREFIX:])

    def pooled_features(self, batch):
        return self._features(batch)[0]

    def semantic_logits(self, batch):
        value = _tokens(batch)
        with torch.autocast(device_type=value.device.type, enabled=False):
            return F.linear(value[:, PREFIX:].float(), self.semantic_reader.weight.float(), self.semantic_reader.bias.float())

    def forward_components(self, batch):
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            probabilities = torch.sigmoid(q.float()).detach()
            context = reader_null_context(_tokens(batch)[:, PREFIX:], self.semantic_reader.weight) if self.variant in MATCHED_VARIANTS else None
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            evidence = pair_features(projected, probabilities, batch.fov, context, grid=GRID, variant=self.variant, query_chunk=QUERY_CHUNK_SIZE)
            residual = (evidence * self.lesion_head.weight.float()[None, :, :]).sum(dim=(-1, -2))
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=evidence, probabilities=probabilities)

    def forward_training(self, batch):
        components = self.forward_components(batch)
        return (components['z'], components['q'])

    def forward_tokens(self, batch):
        return self.forward_training(batch)[0]

    @contextmanager
    def zero_readout(self):
        before = self.lesion_head.weight.detach().clone()
        with torch.no_grad():
            self.lesion_head.weight.zero_()
        try:
            yield self
        finally:
            with torch.no_grad():
                self.lesion_head.weight.copy_(before)

def optimizer_for(model):
    return torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

def make_model(variant, seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights or an explicit frozen encoder required')
        encoder = build_encoder(weights_path)
    return PairContrastModel(encoder, variant=variant, seed=seed).to(device)
Model = PairContrastModel
