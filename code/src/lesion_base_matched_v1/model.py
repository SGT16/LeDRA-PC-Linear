from __future__ import annotations
import torch
from lesion_direction.model import DirectionBatch
from lesion_oof.model import AdaptationModel, build_encoder, state_hash
from lesion_oof.model import optimizer_for as historical_optimizer_for
CONTROL_IDS = ('BASE_MATCHED',)
CONTROL = CONTROL_IDS[0]
DISPLAY_LABEL = 'LeDRA-Base (exact matched)'
MODEL_MODE = 'LEDRA_BASE_EXACT_MATCHED_V1'
NOMINAL_TRAINABLE_PARAMETERS = 25377
OBJECTIVE_CONNECTED_PARAMETERS = {CONTROL: NOMINAL_TRAINABLE_PARAMETERS}
AUXILIARY_COEFFICIENTS = {CONTROL: 0.0}
BASE_STATE_KEYS = ('adapter.0.bias', 'adapter.0.weight', 'adapter.2.bias', 'adapter.2.weight', 'head.bias', 'head.weight')

def base_subset(state):
    if not set(BASE_STATE_KEYS).issubset(state):
        raise ValueError('Reference initialization lacks original Base tensors')
    return {key: state[key] for key in BASE_STATE_KEYS}

class BaseMatchedModel(AdaptationModel):

    def __init__(self, encoder, control_id=CONTROL, seed=823):
        if control_id not in CONTROL_IDS:
            raise ValueError('Unknown exact-matched Base control')
        super().__init__(encoder, arm='B0', seed=seed)
        self.control_id, self.control_mode = (control_id, MODEL_MODE)
        self.auxiliary_coefficient = 0.0
        self.generic_rank = 32
        if self.parameter_counts()['trainable'] != NOMINAL_TRAINABLE_PARAMETERS or set(self.export_trainable_state()) != set(BASE_STATE_KEYS):
            raise ValueError('Exact-matched Base architecture changed')

    @staticmethod
    def _raw_tokens(batch):
        return batch.tokens if isinstance(batch, DirectionBatch) else batch

    def forward_components(self, batch):
        logit = AdaptationModel.forward_tokens(self, self._raw_tokens(batch)).float()
        return dict(z=logit, base=logit)

    def forward_training(self, batch):
        return (self.forward_components(batch)['z'], None)

    def forward_tokens(self, batch):
        return self.forward_components(batch)['z']

    def scientific_metadata(self):
        return dict(control_id=CONTROL, display_label=DISPLAY_LABEL, control_mode=MODEL_MODE, source_variant='B0', generic_rank=32, nominal_trainable_parameters=NOMINAL_TRAINABLE_PARAMETERS, objective_connected_parameters=NOMINAL_TRAINABLE_PARAMETERS, auxiliary_coefficient=0.0, reader_parameters=0, reader_policy='absent', lesion_branch=False, teacher_targets_used=False, reference_subtraction=False, pair_rectification=False, independent_base_training=True, post_training_branch_zeroing=False, initial_shared_state_rule='original LINEAR_REF head+adapter subset', decision='head(patch-average(frozen_suffix(patch+rank32_adapter)))')

def verify_initialization_against_saved_reference(model, seed, encoder, reference_context, device='cuda'):
    from lesion_pc_linear_ablation_v1.model import make_model as reference_model
    if model.control_id != CONTROL or int(seed) != int(model.seed) or reference_context.get('source_seed') != int(seed) or (reference_context.get('model', {}).get('control_id') != 'LINEAR_REF'):
        raise ValueError('Original per-condition LINEAR_REF identity differs')
    expected_full = reference_context.get('initial_trainable_state_hash')
    if not isinstance(expected_full, str) or len(expected_full) != 64:
        raise ValueError('Original LINEAR_REF recorded initial state hash is required')
    reference = reference_model('LINEAR_REF', int(seed), encoder=encoder, device=device)
    state = reference.export_trainable_state()
    actual_full = state_hash(state)
    if actual_full != expected_full:
        raise ValueError('Recreated LINEAR_REF initial state differs from saved original hash')
    shared = base_subset(state)
    own = model.export_trainable_state()
    if set(own) != set(shared) or any((not torch.equal(own[k], shared[k]) for k in shared)):
        raise ValueError('Fresh Base initial head/adapter differs from original LINEAR_REF')
    subset_hash = state_hash(shared)
    if state_hash(own) != subset_hash:
        raise ValueError('Initial Base state hash differs')
    return dict(reference_control_id='LINEAR_REF', reference_seed=int(seed), recorded_reference_full_initial_state_hash=expected_full, recreated_reference_full_initial_state_hash=actual_full, reference_initial_base_subset_hash=subset_hash, fresh_base_initial_state_hash=state_hash(own), shared_initial_tensors=len(BASE_STATE_KEYS), shared_initial_parameters=25377, exact_initialization_match=True, learned_reference_weights_loaded=False, independently_trained_base=True, post_training_branch_zeroing=False)

def optimizer_for(model):
    return historical_optimizer_for(model)

def make_model(control_id=CONTROL, seed=823, encoder=None, device='cuda', weights_path=None):
    if encoder is None:
        if weights_path is None:
            raise ValueError('Frozen general encoder or general pretrained weights required')
        encoder = build_encoder(weights_path)
    return BaseMatchedModel(encoder, control_id=control_id, seed=seed).to(device)
Model = BaseMatchedModel
