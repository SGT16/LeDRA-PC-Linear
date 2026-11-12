from dataclasses import dataclass
import hashlib
import json
import numpy as np
import torch
from torch.nn import functional as F
from lesion_direction.model import DirectionBatch
from lesion_global.model import GlobalPairContrastModel, GRID, BRANCH_DIM, QUERY_CHUNK_SIZE, global_pair_features, optimizer_for as historical_optimizer_for
from lesion_oof.model import build_encoder, PREFIX
VARIANTS = ('ALIGN_SHUFFLE',)
TECHNICAL_VARIANTS = ('ALIGN_REAL',)
TRAINER_ARMS = {variant: 'DIR_REAL' for variant in VARIANTS + TECHNICAL_VARIANTS}
TRAINABLE_COUNTS = {variant: 36229 for variant in VARIANTS + TECHNICAL_VARIANTS}
MODEL_MODE = 'GLOBAL_PAIR_ALIGNMENT_CONTROL'
BACKEND = 'torch_global_pair_alignment_fp32'
PROJECTION_DIM = BRANCH_DIM
QUERY_CHUNK = QUERY_CHUNK_SIZE
CONTROL_SEED = 823
PERMUTATION_NAMESPACE = 'lesion-alignment-v1'
PERMUTATION_RECIPE_ID = 'sha256_dataset_source_image_key_control823_pcg64_within_fov_joint4_v1'
PERMUTATION_RECIPE = dict(recipe_id=PERMUTATION_RECIPE_ID, namespace=PERMUTATION_NAMESPACE, control_seed=CONTROL_SEED, key_fields=['dataset', 'source_image_key'], identity_encoding='canonical JSON [namespace,dataset,source_image_key,control_seed]; UTF-8; ensure_ascii=False; separators=comma,colon', hash='SHA256 full 32-byte digest interpreted as unsigned big-endian integer', generator='numpy.Generator(numpy.random.PCG64(hash_integer))', mapping='destination[inside] reads source[inside[rng.permutation(len(inside))]]; outside identity', four_types='one joint map for complete four-score vectors', scope='fixed per image across all epochs and model seeds', labels_used=False, targets_permuted=False, redraw_allowed=False, perfect_derangement_required=False, digest='SHA256 of C-contiguous little-endian int64 mapping bytes')
BRANCH_FINGERPRINT = dict(mode=MODEL_MODE, backend=BACKEND, branch_dim=BRANCH_DIM, query_chunk=QUERY_CHUNK_SIZE, reference_geometry='same_image_global_nonself', reference_prior='one_minus_detached_permuted_student_probability', query_prior='detached_permuted_student_probability_over_type_eligible_queries', reference_weight_dtype='float32', model_pair_dtype='float32', classifier_reader_gradient=False, teacher_auxiliary_all_variants=True, teacher_logits='original unpermuted semantic reader q', rectified_implementation='immutable_lesion_paircontrast_GLOBAL_RECT', permutation_recipe=PERMUTATION_RECIPE)

def deterministic_permutations(rows, fov, dataset, control_seed=CONTROL_SEED):
    rows = list(rows)
    if not isinstance(dataset, str) or not dataset or isinstance(control_seed, (bool, np.bool_)) or (not isinstance(control_seed, (int, np.integer))) or (int(control_seed) != CONTROL_SEED):
        raise ValueError('Dataset and fixed control seed 823 required')
    if torch.is_tensor(fov):
        if fov.device.type != 'cpu' or fov.dtype != torch.bool:
            raise ValueError('Permutation construction requires explicit CPU boolean FOV')
        valid = fov.detach().numpy()
    else:
        valid = np.asarray(fov)
    if valid.ndim != 2 or valid.shape[0] != len(rows) or len(rows) == 0 or (valid.shape[1] == 0) or (valid.dtype != np.bool_) or (not valid.any(axis=1).all()):
        raise ValueError('Aligned nonempty boolean FOV per image required')
    keys = []
    for row in rows:
        key = row.get('source_image_key')
        if row.get('dataset') != dataset or not isinstance(key, str) or (not key):
            raise ValueError('Immutable source_image_key and matching dataset required')
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError('Immutable image keys must be unique within a cohort')
    maps = np.broadcast_to(np.arange(valid.shape[1], dtype=np.int64), valid.shape).copy()
    for image, key in enumerate(keys):
        identity = json.dumps([PERMUTATION_NAMESPACE, dataset, key, int(control_seed)], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        seed = int.from_bytes(hashlib.sha256(identity).digest(), 'big')
        rng = np.random.Generator(np.random.PCG64(seed))
        inside = np.flatnonzero(valid[image])
        maps[image, inside] = inside[rng.permutation(len(inside))]
    return torch.from_numpy(maps)

def permutation_digest(permutation):
    if not torch.is_tensor(permutation) or permutation.dtype != torch.int64 or permutation.ndim != 2 or (permutation.numel() == 0):
        raise ValueError('Nonempty two-dimensional int64 permutation required')
    values = permutation.detach().cpu().numpy().astype('<i8', copy=False)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()

def validate_permutation(permutation, fov):
    if not torch.is_tensor(fov) or fov.dtype != torch.bool or fov.ndim != 2 or (fov.shape[0] == 0) or (fov.shape[1] == 0) or (not bool(fov.any(dim=1).all())) or (not torch.is_tensor(permutation)) or (permutation.dtype != torch.int64) or (permutation.shape != fov.shape) or (permutation.device != fov.device):
        raise ValueError('Aligned nonempty FOV and int64 permutation required')
    identity = torch.arange(fov.shape[1], dtype=torch.int64, device=fov.device).expand_as(permutation)
    if not bool(((permutation >= 0) & (permutation < fov.shape[1])).all()):
        raise ValueError('Permutation indexes must remain within each image')
    if not torch.equal(permutation.sort(dim=1).values, identity) or not torch.equal(fov.gather(1, permutation), fov) or (not torch.equal(permutation[~fov], identity[~fov])):
        raise ValueError('Permutation must be a FOV bijection with outside identity')
    return permutation.detach()

def permute_probabilities(probabilities, fov, permutation):
    indexes = validate_permutation(permutation, fov)
    if not torch.is_tensor(probabilities) or not torch.is_floating_point(probabilities) or probabilities.ndim != 3 or (probabilities.shape[-1] != 4) or (probabilities.shape[:2] != fov.shape) or (probabilities.device != fov.device) or (not bool(torch.isfinite(probabilities).all())) or (not bool(((probabilities >= 0) & (probabilities <= 1)).all())):
        raise ValueError('Finite aligned four-type probabilities in [0,1] required')
    with torch.no_grad(), torch.autocast(device_type=probabilities.device.type, enabled=False):
        return probabilities.detach().float().gather(1, indexes[..., None].expand(-1, -1, 4)).detach()

@dataclass(frozen=True)
class AlignmentBatch(DirectionBatch):
    permutation: torch.Tensor | None = None

    def to(self, device, *, dtype=None, non_blocking=False):
        return AlignmentBatch(self.tokens.to(device=device, dtype=dtype or self.tokens.dtype, non_blocking=non_blocking), None if self.targets is None else self.targets.to(device=device, non_blocking=non_blocking), None if self.fov is None else self.fov.to(device=device, non_blocking=non_blocking), None if self.permutation is None else self.permutation.to(device=device, non_blocking=non_blocking))

    def index_select(self, indexes):
        indexes = torch.as_tensor(indexes, dtype=torch.int64, device=self.tokens.device)
        if indexes.ndim != 1:
            raise ValueError('Batch indexes must be one-dimensional')
        return AlignmentBatch(self.tokens.index_select(0, indexes), None if self.targets is None else self.targets.index_select(0, indexes.to(self.targets.device)), None if self.fov is None else self.fov.index_select(0, indexes.to(self.fov.device)), None if self.permutation is None else self.permutation.index_select(0, indexes.to(self.permutation.device)))

class AlignmentInput:

    def __init__(self, tokens, targets=None, fov=None, permutation=None):
        if not torch.is_tensor(tokens) or tokens.ndim != 3 or len(tokens) == 0 or (fov is None) or (len(tokens) != len(fov)) or (fov.shape[1] != tokens.shape[1] - PREFIX) or (fov.device != tokens.device):
            raise ValueError('Tokens, FOV and mapping must align on one device')
        validate_permutation(permutation, fov)
        if targets is not None and (targets.shape != (len(tokens), fov.shape[1], 4) or targets.device != tokens.device):
            raise ValueError('Original teacher targets must align with tokens')
        self.tokens, self.targets, self.fov, self.permutation = (tokens, targets, fov, permutation)

    def __len__(self):
        return len(self.tokens)

    def batch(self, indexes, device):
        carrier = AlignmentBatch(self.tokens, self.targets, self.fov, self.permutation)
        return carrier.index_select(indexes).to(device, dtype=torch.float32)

class AlignmentModel(GlobalPairContrastModel):

    def __init__(self, encoder, variant='ALIGN_SHUFFLE', seed=823):
        if variant not in VARIANTS + TECHNICAL_VARIANTS:
            raise ValueError('Unknown alignment control variant')
        super().__init__(encoder, variant='GLOBAL_RECT', seed=seed)
        self.variant, self.mode, self.arm = (variant, MODEL_MODE, TRAINER_ARMS[variant])
        self.backend = BACKEND
        if self.parameter_counts()['trainable'] != TRAINABLE_COUNTS[variant]:
            raise ValueError('Alignment control active parameter budget differs')

    def forward_components(self, batch):
        if self.variant == 'ALIGN_SHUFFLE' and (not isinstance(batch, AlignmentBatch) or batch.permutation is None):
            raise ValueError('Classification alignment control requires an explicit fixed image map')
        global_features, patch_features = self._features(batch)
        q = self.semantic_logits(batch)
        with torch.autocast(device_type=q.device.type, enabled=False):
            raw_probabilities = torch.sigmoid(q.float()).detach()
            probabilities = permute_probabilities(raw_probabilities, batch.fov, batch.permutation) if self.variant == 'ALIGN_SHUFFLE' else raw_probabilities
            projected = F.linear(patch_features.float(), self.projection.weight.float())
            evidence = global_pair_features(projected, probabilities, batch.fov, grid=GRID, variant='GLOBAL_RECT', query_chunk=QUERY_CHUNK_SIZE)
            residual = (evidence * self.lesion_head.weight.float()[None, :, :]).sum(dim=(-1, -2))
        base = self.head(global_features).squeeze(-1)
        logit = (base.float() + residual).to(dtype=base.dtype).float()
        return dict(z=logit, q=q, base=base.float(), residual=residual, pair_evidence=evidence, probabilities=probabilities, raw_probabilities=raw_probabilities)

def optimizer_for(model):
    return historical_optimizer_for(model)

def make_model(variant='ALIGN_SHUFFLE', seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('General pretrained weights or explicit frozen encoder required')
        encoder = build_encoder(weights_path)
    return AlignmentModel(encoder, variant=variant, seed=seed).to(device)
Model = AlignmentModel
