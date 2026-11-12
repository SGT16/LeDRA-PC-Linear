from importlib import import_module
from lesion_select.metrics import aggregate

def metric_records(dataset, rows, logits):
    if dataset in ('sustech', 'paraguay', 'aptos'):
        return import_module('lesion_direction_' + dataset + '.metrics').image_records(rows, logits)
    return aggregate(rows, logits)
