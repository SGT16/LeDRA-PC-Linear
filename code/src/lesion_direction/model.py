from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from lesion_oof.model import AdaptationModel, DIM, PREFIX, PATCHES, SPLIT_BLOCK, _tokens_check, build_encoder
ARMS = ('B0', 'DIR_NOAUX', 'DIR_REAL', 'DIR_SPATIAL', 'AUX_UNTIED')
DIRECTION_ARMS = ('DIR_NOAUX', 'DIR_REAL', 'DIR_SPATIAL')
TRAINABLE_COUNTS = {arm: 23845 if arm in DIRECTION_ARMS else 25377 for arm in ARMS}

@dataclass(frozen=True)
class DirectionBatch:
    tokens: Tensor
    targets: Tensor | None = None
    fov: Tensor | None = None

    def __len__(self):
        return len(self.tokens)

    def to(self, device, *, dtype=None, non_blocking=False):
        return DirectionBatch(self.tokens.to(device=device, dtype=dtype or self.tokens.dtype, non_blocking=non_blocking), None if self.targets is None else self.targets.to(device=device, non_blocking=non_blocking), None if self.fov is None else self.fov.to(device=device, non_blocking=non_blocking))

    def index_select(self, indexes):
        indexes = torch.as_tensor(indexes, dtype=torch.int64, device=self.tokens.device)
        if indexes.ndim != 1:
            raise ValueError('Batch indexes must be one-dimensional')
        return DirectionBatch(self.tokens.index_select(0, indexes), None if self.targets is None else self.targets.index_select(0, indexes.to(self.targets.device)), None if self.fov is None else self.fov.index_select(0, indexes.to(self.fov.device)))

def _tokens(batch):
    value = batch.tokens if isinstance(batch, DirectionBatch) else batch
    if not torch.is_tensor(value):
        raise TypeError('Expected a token tensor or DirectionBatch')
    _tokens_check(value)
    return value

class DirectionModel(AdaptationModel):

    def __init__(self, encoder, arm='B0', seed=823):
        if arm not in ARMS:
            raise ValueError('Unknown direction arm')
        super().__init__(encoder, arm='B0', seed=seed)
        self.arm = arm
        if arm in DIRECTION_ARMS:
            old_up = self.adapter[2]
            with torch.random.fork_rng(devices=[]):
                new_up = nn.Linear(28, DIM, bias=True)
            with torch.no_grad():
                new_up.weight.copy_(old_up.weight[:, :28])
                new_up.bias.copy_(old_up.bias)
            self.adapter[2] = new_up
            self.alpha = nn.Parameter(torch.zeros(4, dtype=old_up.weight.dtype))
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[arm]:
            raise ValueError('Direction parameter budget differs')

    def semantic_logits(self, batch):
        tokens = _tokens(batch)
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            return F.linear(tokens[:, PREFIX:].float(), self.adapter[0].weight[28:32].float(), self.adapter[0].bias[28:32].float())

    def semantic_residual(self, q):
        if self.arm not in DIRECTION_ARMS:
            raise ValueError('This arm has no tied semantic writer')
        if q.ndim != 3 or tuple(q.shape[1:]) != (PATCHES, 4):
            raise ValueError('Semantic logits must have shape [B,784,4]')
        with torch.autocast(device_type=q.device.type, enabled=False):
            C = self.adapter[0].weight[28:32].float()
            directions = C / C.norm(dim=1, keepdim=True).clamp_min(1e-08)
            return F.gelu(q.float()) * self.alpha.float() @ directions

    def adapted_tokens(self, batch, *, semantic=None):
        tokens = _tokens(batch)
        value = tokens.to(dtype=self.head.weight.dtype)
        patch = value[:, PREFIX:]
        if self.arm in DIRECTION_ARMS:
            hidden = self.adapter[1](self.adapter[0](patch)[..., :28])
            generic = self.adapter[2](hidden)
            q = self.semantic_logits(tokens) if semantic is None else semantic
            if not torch.is_tensor(q) or q.shape != (len(value), PATCHES, 4) or q.device != value.device or (not torch.is_floating_point(q)):
                raise ValueError('Semantic logits must match the token batch, device and [B,784,4] shape')
            residual = generic + self.semantic_residual(q)
        else:
            residual = self.adapter(patch)
        return torch.cat((value[:, :PREFIX], patch + residual), dim=1)

    def _pooled(self, batch, semantic=None):
        value = self.adapted_tokens(batch, semantic=semantic)
        for block in self.encoder.blocks[SPLIT_BLOCK:]:
            value = block(value)
        features = self.encoder.forward_head(self.encoder.norm(value))
        if tuple(features.shape) != (len(value), DIM):
            raise ValueError('Expected 384-dimensional patch-average features')
        return features

    def pooled_features(self, batch):
        return self._pooled(batch)

    def forward_tokens(self, batch):
        return self.head(self._pooled(batch)).squeeze(-1)

    def forward_training(self, batch):
        q = self.semantic_logits(batch)
        return (self.head(self._pooled(batch, semantic=q)).squeeze(-1), q)

    def forward(self, batch):
        return self.forward_tokens(batch)

    @contextmanager
    def zero_semantic_alpha(self):
        if self.arm not in DIRECTION_ARMS:
            raise ValueError('Only direction arms have semantic alpha')
        before = self.alpha.detach().clone()
        with torch.no_grad():
            self.alpha.zero_()
        try:
            yield self
        finally:
            with torch.no_grad():
                self.alpha.copy_(before)

def make_model(arm, seed=823, weights_path=None, encoder=None, device='cuda'):
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights or an explicit frozen encoder are required')
        encoder = build_encoder(weights_path)
    return DirectionModel(encoder, arm=arm, seed=seed).to(device)
Model = DirectionModel
