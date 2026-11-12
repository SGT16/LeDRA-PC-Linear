from __future__ import annotations
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import torch
from torch import Tensor, nn
DIM, PREFIX, PATCHES, TOKENS, SPLIT_BLOCK = (384, 5, 784, 789, 10)
ARCHITECTURE = 'vit_small_patch14_reg4_dinov2'
ARMS = ('B0', 'AF')
TRAINABLE_COUNTS = {'B0': 25377, 'AF': 25761}
GENERAL_WEIGHTS_SHA256 = '431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1'

def state_hash(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def _encoder_check(encoder: nn.Module) -> None:
    if len(encoder.blocks) != 12 or encoder.num_prefix_tokens != PREFIX or encoder.num_features != DIM or (encoder.global_pool != 'avg'):
        raise ValueError('Expected 12-block RETFound-Green, five prefix tokens and patch-average pooling')

def build_encoder(weights_path: str | Path) -> nn.Module:
    import timm
    with torch.random.fork_rng(devices=[]):
        encoder = timm.create_model(ARCHITECTURE, pretrained=False, img_size=(392, 392), num_classes=0)
    encoder.global_pool = 'avg'
    state = torch.load(Path(weights_path), map_location='cpu', weights_only=True)
    if not isinstance(state, Mapping) or not state or any((not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())):
        raise ValueError('General checkpoint must be a nonempty plain tensor state dictionary')
    encoder.load_state_dict(state, strict=True)
    _encoder_check(encoder)
    encoder.requires_grad_(False)
    return encoder.eval()

def _tokens_check(tokens: Tensor) -> None:
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (TOKENS, DIM):
        raise ValueError('Expected cached prefix tokens [N,789,384], CLS/register tokens first')
    if not torch.is_floating_point(tokens):
        raise TypeError('Cached prefix tokens must be floating point')

def encode_prefix(encoder: nn.Module, images: Tensor) -> Tensor:
    _encoder_check(encoder)
    value = encoder.patch_embed(images)
    value = encoder._pos_embed(value)
    value = encoder.patch_drop(value)
    value = encoder.norm_pre(value)
    for block in encoder.blocks[:SPLIT_BLOCK]:
        value = block(value)
    _tokens_check(value)
    return value

class ParallelBottleneck(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(DIM, 16, bias=True)
        self.non_linear_func = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.up_proj = nn.Linear(16, DIM, bias=True)
        self.scale = 0.1
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, value: Tensor) -> Tensor:
        return self.scale * self.up_proj(self.dropout(self.non_linear_func(self.down_proj(value))))

class AdaptationModel(nn.Module):

    def __init__(self, encoder: nn.Module, arm: str='B0', seed: int=823):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('arm must be B0 or AF')
        _encoder_check(encoder)
        self.encoder, self.arm, self.seed = (encoder, arm, int(seed))
        self.encoder.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            self.head = nn.Linear(DIM, 1, bias=True)
            if arm == 'B0':
                self.adapter = nn.Sequential(nn.Linear(DIM, 32), nn.GELU(), nn.Linear(32, DIM))
                nn.init.zeros_(self.adapter[2].weight)
                nn.init.zeros_(self.adapter[2].bias)
            else:
                self.adapters = nn.ModuleList([ParallelBottleneck(), ParallelBottleneck()])
        if arm == 'AF':
            for block in encoder.blocks[SPLIT_BLOCK:]:
                attributes = ('norm1', 'attn', 'ls1', 'drop_path1', 'norm2', 'mlp', 'ls2', 'drop_path2')
                if type(block).__name__ != 'Block' or not all((hasattr(block, key) for key in attributes)):
                    raise ValueError('Unsupported AF backbone block; architecture substitution is forbidden')
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[arm]:
            raise ValueError('Trainable parameter budget differs from the recorded comparator')
        self.encoder.eval()

    def train(self, mode: bool=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def pooled_features(self, tokens: Tensor) -> Tensor:
        _tokens_check(tokens)
        value = tokens.to(dtype=self.head.weight.dtype)
        if self.arm == 'B0':
            patch = value[:, PREFIX:]
            value = torch.cat((value[:, :PREFIX], patch + self.adapter(patch)), dim=1)
            for block in self.encoder.blocks[SPLIT_BLOCK:]:
                value = block(value)
        else:
            for block, adapter in zip(self.encoder.blocks[SPLIT_BLOCK:], self.adapters):
                u = value + block.drop_path1(block.ls1(block.attn(block.norm1(value))))
                original_ffn = u + block.drop_path2(block.ls2(block.mlp(block.norm2(u))))
                value = original_ffn + adapter(u)
        features = self.encoder.forward_head(self.encoder.norm(value))
        if tuple(features.shape) != (len(tokens), DIM):
            raise ValueError('Expected classifier-free 384-dimensional patch-average features')
        return features

    def forward_tokens(self, tokens: Tensor) -> Tensor:
        return self.head(self.pooled_features(tokens)).squeeze(-1)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.forward_tokens(tokens)

    def export_trainable_state(self) -> dict[str, Tensor]:
        return {name: parameter.detach().cpu().clone() for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_trainable_state(self, state: Mapping[str, Tensor]):
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("Checkpoint must contain exactly this arm's adapter/head state, without encoder weights")
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
        trainable = sum((p.numel() for p in self.parameters() if p.requires_grad))
        head = sum((p.numel() for p in self.head.parameters()))
        return dict(trainable=trainable, frozen=sum((p.numel() for p in self.encoder.parameters())), classifier=head, adapter=trainable - head, scaler=0)

def make_model(arm: str, seed: int=823, weights_path=None, encoder=None, device='cuda') -> AdaptationModel:
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights_path or an explicit frozen encoder is required')
        encoder = build_encoder(weights_path)
    return AdaptationModel(encoder, arm=arm, seed=seed).to(device)

def optimizer_for(model: AdaptationModel) -> torch.optim.Optimizer:
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
Model = AdaptationModel
