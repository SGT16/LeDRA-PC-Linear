from pathlib import Path

import numpy as np
import torch

from lesion_close_adapters.model import CloseAdapterModel, encode_prefix_tokens
from lesion_published_baselines.model import PublishedBaseline, encode_input_tokens


MODELS = ("LP", "HOULSBY", "ADAPTFORMER", "ADAPTERPLUS", "ADAPTERDROP")


def load_model(encoder, model_name, seed, checkpoint, device="cuda"):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}")
    if model_name in ("ADAPTERPLUS", "ADAPTERDROP"):
        split = 11 if model_name == "ADAPTERDROP" else 0
        model = CloseAdapterModel(encoder, arm=model_name, seed=seed, input_split_block=split)
    else:
        model = PublishedBaseline(encoder, arm=model_name, seed=seed)
    state = torch.load(Path(checkpoint), weights_only=True, map_location="cpu")
    model.load_trainable_state(state)
    exported = model.export_trainable_state()
    if set(exported) != set(state) or not all(torch.equal(value.cpu(), state[key].cpu()) for key, value in exported.items()):
        raise ValueError("Checkpoint failed exact state comparison")
    return model.to(device).eval()


@torch.inference_mode()
def extract_prefixes(encoder, images):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = encode_input_tokens(encoder, images.to("cuda")).half()
        split11 = encode_prefix_tokens(encoder, tokens.float(), split_block=11).float()
    return tokens.cpu().numpy(), split11.cpu().numpy()


@torch.inference_mode()
def forward_batch(model, array, pool_only=False, pooled_input=False):
    device = next(model.parameters()).device
    tokens = torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if pool_only:
            result = model.pooled_features(tokens)
        elif pooled_input:
            result = model.forward_features(tokens)
        else:
            result = model(tokens)
    result = result.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite model output")
    return result

