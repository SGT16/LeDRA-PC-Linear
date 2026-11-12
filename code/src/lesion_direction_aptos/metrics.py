from __future__ import annotations
import numpy as np
from scipy.special import expit
from lesion_select.metrics import summarize

def image_records(rows, logits):
    rows = list(rows)
    logits = np.asarray(logits, dtype=np.float64)
    if not rows or logits.shape != (len(rows),) or (not np.isfinite(logits).all()):
        raise ValueError('Finite one-dimensional image predictions required')
    keys, output = (set(), [])
    for index, (row, z) in enumerate(zip(rows, logits)):
        if row.get('dataset') != 'aptos' or row.get('scoring_unit') != 'image' or 'patient_id' in row or ('eligible_bilateral' in row):
            raise ValueError('Canonical APTOS image records must not claim patient/bilateral identity')
        key = row['source_image_key']
        if not key or key in keys or (not row['dependency_group_id']):
            raise ValueError('Missing/repeated image or dependency identity')
        keys.add(key)
        if str(row['image_label']) not in ('0', '1') or str(row['label']) != str(row['image_label']):
            raise ValueError('Known, unchanged image labels required')
        if str(row['metric_eligible']).lower() not in ('true', '1'):
            raise ValueError('Image metric eligibility must be explicit')
        output.append(dict(patient_id=key, sample_id=row['sample_id'], source_image_key=key, dependency_group_id=row['dependency_group_id'], label=int(row['image_label']), eligible=True, logit=float(z), probability=float(expit(z)), winner_row=index, scoring_unit='image', identity_unit='image_technical_alias_not_patient'))
    return sorted(output, key=lambda r: r['source_image_key'])

def image_summary(rows, logits):
    result = summarize(image_records(rows, logits))
    result.update(accuracy_05=(result['TP_05'] + result['TN_05']) / result['n'], scoring_unit='image')
    return result
