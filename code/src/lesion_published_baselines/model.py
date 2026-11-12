from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from lesion_oof.model import DIM, PREFIX, TOKENS, build_encoder
ARMS = ('LP', 'HOULSBY', 'ADAPTFORMER')
DEPTH = 12
BOTTLENECK = 64
TRAINABLE_COUNTS = {'LP': 385, 'HOULSBY': 1209985, 'ADAPTFORMER': 595585}
MODEL_LABELS = {'LP': 'RETFound-Green + linear probing', 'HOULSBY': 'RETFound-Green + Houlsby-type adapter (r64, 12 blocks)', 'ADAPTFORMER': 'RETFound-Green + AdaptFormer (r64, 12 blocks)'}
PRIMARY_SOURCES = {'HOULSBY': ('https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf', 'https://github.com/google-research/adapter-bert/blob/master/modeling.py'), 'ADAPTFORMER': ('https://arxiv.org/abs/2205.13535', 'https://github.com/ShoufaChen/AdaptFormer/blob/main/models/custom_modules.py', 'https://github.com/ShoufaChen/AdaptFormer/blob/main/models/adapter.py', 'https://github.com/ShoufaChen/AdaptFormer/blob/main/main_image.py')}

def _encoder_check(encoder: nn.Module) -> None:
    if len(encoder.blocks) != DEPTH or encoder.num_prefix_tokens != PREFIX or encoder.num_features != DIM or (encoder.global_pool != 'avg'):
        raise ValueError('Expected 12-block RETFound-Green with five prefix tokens and patch-average pooling')
    required = ('norm1', 'attn', 'ls1', 'drop_path1', 'norm2', 'mlp', 'ls2', 'drop_path2')
    if any((not all((hasattr(block, name) for name in required)) for block in encoder.blocks)):
        raise ValueError('Unsupported encoder block topology; no silent architecture substitutions')
    norms = [encoder.norm] + [norm for block in encoder.blocks for norm in (block.norm1, block.norm2)]
    if any((not isinstance(norm, nn.LayerNorm) or norm.normalized_shape != (DIM,) or (not norm.elementwise_affine) or (norm.bias is None) for norm in norms)):
        raise ValueError('Expected affine 384-dimensional block and final layer normalization')

def _tokens_check(tokens: Tensor) -> None:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (TOKENS, DIM):
        raise ValueError('Expected pre-block-0 tokens [N,789,384], five prefix tokens first')
    if not torch.is_floating_point(tokens):
        raise TypeError('Input tokens must be floating point')

def encode_input_tokens(encoder: nn.Module, images: Tensor) -> Tensor:
    _encoder_check(encoder)
    value = encoder.patch_embed(images)
    value = encoder._pos_embed(value)
    value = encoder.patch_drop(value)
    value = encoder.norm_pre(value)
    _tokens_check(value)
    return value

class SerialBottleneck(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(DIM, BOTTLENECK, bias=True)
        self.activation = nn.GELU(approximate='tanh')
        self.up_proj = nn.Linear(BOTTLENECK, DIM, bias=True)
        for projection in (self.down_proj, self.up_proj):
            nn.init.trunc_normal_(projection.weight, std=0.001, a=-0.002, b=0.002)
            nn.init.zeros_(projection.bias)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.up_proj(self.activation(self.down_proj(value)))

class ParallelBottleneck(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(DIM, BOTTLENECK, bias=True)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.up_proj = nn.Linear(BOTTLENECK, DIM, bias=True)
        self.scale = 0.1
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.scale * self.up_proj(self.dropout(self.activation(self.down_proj(value))))

class PublishedBaseline(nn.Module):

    def __init__(self, encoder: nn.Module, arm: str='LP', seed: int=823, gradient_checkpointing: bool=False):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('arm must be LP, HOULSBY, or ADAPTFORMER')
        _encoder_check(encoder)
        self.encoder, self.arm, self.seed = (encoder, arm, int(seed))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.head = nn.Linear(DIM, 1, bias=True)
            if arm == 'HOULSBY':
                self.attention_adapters = nn.ModuleList([SerialBottleneck() for _ in range(DEPTH)])
                self.ffn_adapters = nn.ModuleList([SerialBottleneck() for _ in range(DEPTH)])
            elif arm == 'ADAPTFORMER':
                self.adapters = nn.ModuleList([ParallelBottleneck() for _ in range(DEPTH)])
        if arm == 'HOULSBY':
            self.attention_norms = nn.ModuleList([deepcopy(block.norm1) for block in encoder.blocks])
            self.ffn_norms = nn.ModuleList([deepcopy(block.norm2) for block in encoder.blocks])
            self.output_norm = deepcopy(encoder.norm)
            self.attention_norms.requires_grad_(True)
            self.ffn_norms.requires_grad_(True)
            self.output_norm.requires_grad_(True)
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[arm]:
            raise ValueError('Unexpected trainable parameter budget for published comparator')

    def train(self, mode: bool=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _forward_block(self, value: Tensor, index: int) -> Tensor:
        block = self.encoder.blocks[index]
        if self.arm == 'HOULSBY':
            attention = block.attn(self.attention_norms[index](value))
            value = value + block.drop_path1(block.ls1(self.attention_adapters[index](attention)))
            ffn = block.mlp(self.ffn_norms[index](value))
            return value + block.drop_path2(block.ls2(self.ffn_adapters[index](ffn)))
        if self.arm == 'ADAPTFORMER':
            value = value + block.drop_path1(block.ls1(block.attn(block.norm1(value))))
            return value + block.drop_path2(block.ls2(block.mlp(block.norm2(value)))) + self.adapters[index](value)
        return block(value)

    def pooled_features(self, tokens: Tensor) -> Tensor:
        _tokens_check(tokens)
        value = tokens.to(dtype=self.head.weight.dtype)
        for index in range(DEPTH):
            if self.gradient_checkpointing and self.training and (self.arm != 'LP') and torch.is_grad_enabled():
                value = checkpoint(lambda x, block_index=index: self._forward_block(x, block_index), value, use_reentrant=False, preserve_rng_state=True)
            else:
                value = self._forward_block(value, index)
        norm = self.output_norm if self.arm == 'HOULSBY' else self.encoder.norm
        features = self.encoder.forward_head(norm(value))
        if tuple(features.shape) != (len(tokens), DIM):
            raise ValueError('Expected final normalized 384-dimensional patch-average features')
        return features

    def forward_features(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != DIM:
            raise ValueError('Expected pooled features [N,384]')
        if not torch.is_floating_point(features):
            raise TypeError('Pooled features must be floating point')
        return self.head(features.to(dtype=self.head.weight.dtype)).squeeze(-1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.forward_features(self.pooled_features(tokens))

    def export_trainable_state(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_trainable_state(self, state: Mapping[str, Tensor]):
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("Checkpoint must contain exactly this arm's task state, without frozen encoder weights")
        for name, parameter in expected.items():
            value = state[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape:
                raise ValueError('Trainable checkpoint tensor shape mismatch: ' + name)
            if not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise ValueError('Trainable checkpoint tensor must be finite floating point: ' + name)
        with torch.no_grad():
            for name, parameter in expected.items():
                parameter.copy_(state[name].to(parameter))
        return self

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum((parameter.numel() for parameter in self.parameters() if parameter.requires_grad))
        head = sum((parameter.numel() for parameter in self.head.parameters()))
        normalization = sum((parameter.numel() for name, parameter in self.named_parameters() if parameter.requires_grad and (name.startswith('attention_norms.') or name.startswith('ffn_norms.') or name.startswith('output_norm.'))))
        frozen = sum((parameter.numel() for parameter in self.parameters() if not parameter.requires_grad))
        return {'trainable': trainable, 'frozen': frozen, 'classifier': head, 'adapter': trainable - head - normalization, 'normalization': normalization, 'scaler': 0}

def make_model(arm: str, seed: int=823, weights_path: str | Path | None=None, encoder: nn.Module | None=None, device='cuda', gradient_checkpointing: bool=False) -> PublishedBaseline:
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights_path or explicit frozen encoder is required')
        encoder = build_encoder(weights_path)
    return PublishedBaseline(encoder, arm=arm, seed=seed, gradient_checkpointing=gradient_checkpointing).to(device)

def optimizer_for(model: PublishedBaseline) -> torch.optim.Optimizer:
    return torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
Model = PublishedBaseline
