from __future__ import annotations
import torch
from torch.nn import functional as F
AUXILIARY_ARMS = ('DIR_REAL', 'DIR_SPATIAL', 'AUX_UNTIED')

def weighted_chunk_loss(logits, labels, auxiliary, coefficient, known_count, image_count):
    if logits.ndim != 1 or labels.shape != logits.shape or known_count <= 0 or (image_count <= 0) or (known_count > image_count) or (len(logits) > image_count) or (coefficient not in (0.0, 0.1)):
        raise ValueError('Invalid full-batch denominators or chunk objective')
    known = (labels == 0) | (labels == 1)
    if int(known.sum()) > known_count:
        raise ValueError('Known chunk labels exceed full-batch count')
    classification = F.binary_cross_entropy_with_logits(logits[known], labels[known].to(logits.dtype), reduction='sum') / known_count if bool(known.any()) else logits.sum() * 0.0
    aux = classification.new_zeros(())
    if coefficient:
        if auxiliary is None or auxiliary.shape != logits.shape or (not bool(torch.isfinite(auxiliary).all())):
            raise ValueError('Expected finite auxiliary loss per image')
        aux = auxiliary.sum() / image_count
    return (classification + coefficient * aux, classification, aux)
