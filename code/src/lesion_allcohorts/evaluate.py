from __future__ import annotations
import numpy as np
from lesion_direction_multiseed.evaluate import metric_records
DATASETS = ('brset', 'deepdrid', 'mbrset', 'messidor2', 'sustech', 'paraguay', 'aptos')
SCORING_UNITS = dict(brset='patient', deepdrid='patient', mbrset='patient', messidor2='exam', sustech='image', paraguay='image', aptos='image')

def _require(condition, message):
    if not condition:
        raise ValueError(message)

def scoring_unit(dataset):
    _require(dataset in DATASETS, 'Unknown seven-cohort dataset')
    return SCORING_UNITS[dataset]

def records_for(dataset, rows, logits):
    scoring_unit(dataset)
    rows, values = (list(rows), np.asarray(logits, dtype=np.float64))
    _require(bool(rows) and values.shape == (len(rows),) and np.isfinite(values).all(), 'Finite aligned one-dimensional predictions required')
    _require(all((r.get('dataset') == dataset for r in rows)), 'Prediction cohort differs')
    return metric_records(dataset, rows, values)
