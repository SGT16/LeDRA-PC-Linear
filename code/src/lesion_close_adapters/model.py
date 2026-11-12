from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from lesion_oof.model import DIM, PREFIX, TOKENS, build_encoder
from lesion_published_baselines.model import _encoder_check, encode_input_tokens
ARMS = ('ADAPTERPLUS', 'ADAPTERDROP', 'B0_MATCHED', 'B0_EARLY', 'B0_MIDDLE', 'B0_ALLTOKENS')
DEPTH = 12
ARM_SPLIT_BLOCK = {'ADAPTERPLUS': 0, 'ADAPTERDROP': 11, 'B0_MATCHED': 10, 'B0_EARLY': 2, 'B0_MIDDLE': 6, 'B0_ALLTOKENS': 10}
TRAINABLE_COUNTS = {'ADAPTERPLUS': 304897, 'ADAPTERDROP': 28065, 'B0_MATCHED': 25377, 'B0_EARLY': 25377, 'B0_MIDDLE': 25377, 'B0_ALLTOKENS': 25377}
MODEL_LABELS = {'ADAPTERPLUS': 'RETFound-Green + Adapter+ (r32, 12 post-block adapters)', 'ADAPTERDROP': 'RETFound-Green + final-block Houlsby-type AdapterDrop (r16)', 'B0_MATCHED': 'B0 matched-input reference: after block 10, patch tokens only', 'B0_EARLY': 'B0 placement ablation: after block 2, patch tokens only', 'B0_MIDDLE': 'B0 placement ablation: after block 6, patch tokens only', 'B0_ALLTOKENS': 'B0 token ablation: after block 10, all tokens'}
PRIMARY_SOURCES = {'ADAPTERPLUS': ('https://openaccess.thecvf.com/content/CVPR2024/html/Steitz_Adapters_Strike_Back_CVPR_2024_paper.html', 'https://github.com/visinf/adapter_plus/blob/main/adapter_plus/vit_adapter.py', 'https://github.com/visinf/adapter_plus/blob/main/conf/adapter/adapter_plus.yaml'), 'ADAPTERDROP': ('https://arxiv.org/pdf/2203.16329', 'https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf', 'https://github.com/google-research/adapter-bert/blob/master/modeling.py')}

def _tokens_check(tokens: Tensor) -> None:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (TOKENS, DIM):
        raise ValueError('Expected split-declared tokens [N,789,384], five prefix tokens first')
    if not torch.is_floating_point(tokens):
        raise TypeError('Input tokens must be floating point')

def _split_check(split_block: int, maximum: int=DEPTH) -> None:
    if isinstance(split_block, bool) or not isinstance(split_block, int) or (not 0 <= split_block <= maximum):
        raise ValueError('Cache split must be an integer between zero and the first trainable operation')

def encode_prefix_tokens(encoder: nn.Module, tokens: Tensor, split_block: int, input_split_block: int=0) -> Tensor:
    _encoder_check(encoder)
    _tokens_check(tokens)
    _split_check(input_split_block)
    _split_check(split_block)
    if input_split_block > split_block:
        raise ValueError('Cannot reverse a frozen prefix or infer its block boundary')
    if encoder.training or any((parameter.requires_grad for parameter in encoder.parameters())):
        raise ValueError('Prefix caching requires an eval-mode, fully frozen encoder')
    with torch.no_grad():
        value = tokens.detach()
        for block in encoder.blocks[input_split_block:split_block]:
            value = block(value)
    return value

def encode_prefix(encoder: nn.Module, images: Tensor, split_block: int=0) -> Tensor:
    _split_check(split_block)
    if encoder.training or any((parameter.requires_grad for parameter in encoder.parameters())):
        raise ValueError('Prefix caching requires an eval-mode, fully frozen encoder')
    with torch.no_grad():
        tokens = encode_input_tokens(encoder, images)
        return encode_prefix_tokens(encoder, tokens, split_block)

class AdapterStochasticDepth(nn.Module):

    def __init__(self, drop_prob: float=0.1):
        super().__init__()
        self.drop_prob = float(drop_prob)
        if not 0 <= self.drop_prob < 1:
            raise ValueError('drop_prob must be in [0,1)')

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0:
            return value
        keep_probability = 1 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep_probability)
        return value * mask.div_(keep_probability)

class AdapterPlus(nn.Module):

    def __init__(self, bottleneck: int=32):
        super().__init__()
        self.down_proj = nn.Linear(DIM, bottleneck, bias=True)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck, DIM, bias=True)
        self.drop_path = AdapterStochasticDepth(0.1)
        self.scaling = nn.Parameter(torch.ones(DIM))
        for projection in (self.down_proj, self.up_proj):
            nn.init.trunc_normal_(projection.weight, std=0.01, a=-0.02, b=0.02)
            nn.init.zeros_(projection.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.drop_path(self.up_proj(self.activation(self.down_proj(value)))) * self.scaling

class FinalSerialBottleneck(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(DIM, 16, bias=True)
        self.activation = nn.GELU(approximate='tanh')
        self.up_proj = nn.Linear(16, DIM, bias=True)
        for projection in (self.down_proj, self.up_proj):
            nn.init.trunc_normal_(projection.weight, std=0.001, a=-0.002, b=0.002)
            nn.init.zeros_(projection.bias)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.up_proj(self.activation(self.down_proj(value)))

class CloseAdapterModel(nn.Module):

    def __init__(self, encoder: nn.Module, arm: str='ADAPTERPLUS', seed: int=823, gradient_checkpointing: bool=False, input_split_block: int=0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown close adapter arm: ' + str(arm))
        _encoder_check(encoder)
        _split_check(input_split_block, ARM_SPLIT_BLOCK[arm])
        self.encoder, self.arm, self.seed = (encoder, arm, int(seed))
        self.input_split_block = input_split_block
        self.split_block = ARM_SPLIT_BLOCK[arm]
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.head = nn.Linear(DIM, 1, bias=True)
            if arm == 'ADAPTERPLUS':
                self.adapters = nn.ModuleList([AdapterPlus() for _ in range(DEPTH)])
            elif arm == 'ADAPTERDROP':
                self.attention_adapter = FinalSerialBottleneck()
                self.ffn_adapter = FinalSerialBottleneck()
            else:
                self.adapter = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, DIM))
                nn.init.zeros_(self.adapter[2].weight)
                nn.init.zeros_(self.adapter[2].bias)
        if arm == 'ADAPTERDROP':
            self.attention_norm = deepcopy(encoder.blocks[-1].norm1)
            self.ffn_norm = deepcopy(encoder.blocks[-1].norm2)
            self.output_norm = deepcopy(encoder.norm)
            self.attention_norm.requires_grad_(True)
            self.ffn_norm.requires_grad_(True)
            self.output_norm.requires_grad_(True)
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[arm]:
            raise ValueError('Unexpected trainable parameter budget for close adapter arm')

    def train(self, mode: bool=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _adapt_interblock_tokens(self, value: Tensor) -> Tensor:
        if self.arm == 'B0_ALLTOKENS':
            return value + self.adapter(value)
        if self.arm not in ('B0_MATCHED', 'B0_EARLY', 'B0_MIDDLE'):
            raise ValueError('Only B0 ablations have a single inter-block adapter')
        patches = value[:, PREFIX:]
        return torch.cat((value[:, :PREFIX], patches + self.adapter(patches)), dim=1)

    def _forward_block(self, value: Tensor, index: int) -> Tensor:
        block = self.encoder.blocks[index]
        if self.arm == 'ADAPTERPLUS':
            if index == 0:
                with torch.no_grad():
                    value = block(value)
            else:
                value = block(value)
            return value + self.adapters[index](value)
        if self.arm == 'ADAPTERDROP' and index == DEPTH - 1:
            attention = block.attn(self.attention_norm(value))
            value = value + block.drop_path1(block.ls1(self.attention_adapter(attention)))
            ffn = block.mlp(self.ffn_norm(value))
            return value + block.drop_path2(block.ls2(self.ffn_adapter(ffn)))
        return block(value)

    def pooled_features(self, tokens: Tensor) -> Tensor:
        _tokens_check(tokens)
        value = tokens.to(dtype=self.head.weight.dtype)
        value = encode_prefix_tokens(self.encoder, value, self.split_block, input_split_block=self.input_split_block)
        if self.arm.startswith('B0_'):
            value = self._adapt_interblock_tokens(value)
        for index in range(self.split_block, DEPTH):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                value = checkpoint(lambda x, block_index=index: self._forward_block(x, block_index), value, use_reentrant=False, preserve_rng_state=True)
            else:
                value = self._forward_block(value, index)
        norm = self.output_norm if self.arm == 'ADAPTERDROP' else self.encoder.norm
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

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        return self.forward_features(self.pooled_features(tokens))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.forward_tokens(tokens)

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
        normalization = sum((parameter.numel() for name, parameter in self.named_parameters() if parameter.requires_grad and name.startswith(('attention_norm.', 'ffn_norm.', 'output_norm.'))))
        scaler = sum((parameter.numel() for name, parameter in self.named_parameters() if parameter.requires_grad and name.endswith('.scaling')))
        frozen = sum((parameter.numel() for parameter in self.parameters() if not parameter.requires_grad))
        return {'trainable': trainable, 'frozen': frozen, 'classifier': head, 'adapter': trainable - head - normalization - scaler, 'normalization': normalization, 'scaler': scaler}

def make_model(arm: str, seed: int=823, weights_path: str | Path | None=None, encoder: nn.Module | None=None, device='cuda', gradient_checkpointing: bool=False, input_split_block: int=0) -> CloseAdapterModel:
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights_path or explicit frozen encoder is required')
        encoder = build_encoder(weights_path)
    return CloseAdapterModel(encoder, arm=arm, seed=seed, gradient_checkpointing=gradient_checkpointing, input_split_block=input_split_block).to(device)

def optimizer_for(model: CloseAdapterModel) -> torch.optim.Optimizer:
    return torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
Model = CloseAdapterModel
