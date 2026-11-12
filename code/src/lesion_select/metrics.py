from __future__ import annotations
import numpy as np
from scipy.special import expit

def boolean(v):
    if str(v).lower() in ('true', '1'):
        return True
    if str(v).lower() in ('false', '0'):
        return False
    raise ValueError('Explicit boolean required')

def aggregate(rows, logits):
    logits = np.asarray(logits, dtype=np.float64)
    if len(rows) != len(logits) or not np.isfinite(logits).all():
        raise ValueError('Invalid predictions')
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row['patient_id']), []).append(i)
    output = []
    for pid in sorted(groups):
        ix = groups[pid]
        labels = {int(rows[i]['patient_label']) for i in ix}
        deps = {rows[i]['dependency_group_id'] for i in ix}
        eligible = {boolean(rows[i]['eligible_bilateral']) for i in ix}
        if len(labels) != 1 or len(deps) != 1 or len(eligible) != 1:
            raise ValueError('Patient metadata conflict')
        winner = ix[int(np.argmax(logits[ix]))]
        z = float(logits[winner])
        output.append(dict(patient_id=pid, dependency_group_id=deps.pop(), label=labels.pop(), eligible=eligible.pop(), logit=z, probability=float(expit(z)), winner_row=winner))
    return output

def arrays(patients):
    return (np.array([r['label'] for r in patients], dtype=np.int64), np.array([r['probability'] for r in patients], dtype=np.float64), np.array([r['eligible'] for r in patients], dtype=bool))

def ranking(y, p, eligible):
    y, p, eligible = (np.asarray(y), np.asarray(p), np.asarray(eligible, bool))
    P, N = (int(y.sum()), int(len(y) - y.sum()))
    if not P or not N:
        return dict(S5=None, A005=None, AP=None, AUROC=None)
    order = np.argsort(-p, kind='stable')
    ys, ps = (y[order], p[order])
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    fp = ends + 1 - tp
    ap = np.sum(np.diff(np.r_[0, tp]) / P * tp / (tp + fp))
    roc = np.sum(np.diff(np.r_[0, fp]) / N * (tp + np.r_[0, tp[:-1]]) / (2 * P))
    ix = np.flatnonzero(eligible)
    ix = ix[np.argsort(p[ix], kind='stable')]
    if not len(ix):
        return dict(S5=0.0, A005=0.0, AP=float(ap), AUROC=float(roc))
    ends = np.r_[np.flatnonzero(p[ix][:-1] != p[ix][1:]), len(ix) - 1]
    fn = np.r_[0, np.cumsum(y[ix])[ends]]
    tn = np.r_[0, ends + 1 - np.cumsum(y[ix])[ends]]
    s5 = float(tn[20 * fn <= P].max() / N)
    levels, starts = np.unique(fn, return_index=True)
    envelope = np.maximum.reduceat(tn, starts)
    left = levels / P
    right = np.r_[left[1:], 1.0]
    widths = np.maximum(0.0, np.minimum(0.05, right) - np.minimum(0.05, left))
    area = np.sum(widths * envelope / N) / 0.05
    return dict(S5=s5, A005=float(area), AP=float(ap), AUROC=float(roc))

def binary(y, p, threshold):
    positive = np.asarray(p) >= threshold
    y = np.asarray(y)
    tp = int(np.sum(positive & (y == 1)))
    fp = int(np.sum(positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1)))
    tn = int(np.sum(~positive & (y == 0)))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, recall=tp / (tp + fn) if tp + fn else None, specificity=tn / (tn + fp) if tn + fp else None, FNR=fn / (tp + fn) if tp + fn else None)

def calibrate(patients):
    y, p, eligible = arrays(patients)
    P = int(y.sum())
    if not P or P == len(y):
        raise ValueError('CAL must have both classes')
    positive = p[(y == 1) & eligible]
    candidates = sorted({0.0, 0.5, *positive[positive <= 0.5].tolist()})
    neg = max((t for t in candidates if 20 * np.sum(positive < t) <= P))
    choices = []
    for t in sorted({0.0, 0.5, 1.0, *p.tolist()}):
        b = binary(y, p, t)
        choices.append(((b['F1'], b['recall'], -abs(t - 0.5), -t), t))
    f1 = max(choices)[1]
    return dict(t_neg=float(neg), t_pos=0.5, t_f1=float(f1), positive_n=P, calibration_FN=int(np.sum(positive < neg)), fitted_role='CALIBRATION')

def policy(y, p, eligible, threshold):
    y, p, eligible = (np.asarray(y), np.asarray(p), np.asarray(eligible, bool))
    neg = eligible & (p < threshold['t_neg'])
    pos = eligible & (p >= 0.5)
    P = int(y.sum())
    N = len(y) - P
    fn = int(np.sum(neg & (y == 1)))
    tn = int(np.sum(neg & (y == 0)))
    return dict(auto_FN=fn, auto_TN=tn, auto_TP=int(np.sum(pos & (y == 1))), auto_FP=int(np.sum(pos & (y == 0))), deferred=int(np.sum(~(neg | pos))), FNR_CAL=fn / P if P else None, NA_CAL=tn / N if N else None, coverage=float(np.mean(neg | pos)))

def summarize(patients, threshold=None):
    y, p, eligible = arrays(patients)
    z = np.array([r['logit'] for r in patients])
    out = dict(n=len(y), positive_n=int(y.sum()), negative_n=int(len(y) - y.sum()), ineligible_n=int(np.sum(~eligible)), **ranking(y, p, eligible), logloss=float(np.mean(np.logaddexp(0, z) - y * z)))
    out.update({k + '_05': v for k, v in binary(y, p, 0.5).items()})
    if threshold:
        out.update(policy(y, p, eligible, threshold))
        out.update({k + '_CALF1': v for k, v in binary(y, p, threshold['t_f1']).items()})
    return out

def transitions(candidate, anchor):
    if [r['patient_id'] for r in candidate] != [r['patient_id'] for r in anchor]:
        raise ValueError('Alignment')
    y, p, _ = arrays(candidate)
    _, q, _ = arrays(anchor)
    old_ok = (q >= 0.5) == y
    new_ok = (p >= 0.5) == y
    z = np.array([r['logit'] for r in candidate])
    b = np.array([r['logit'] for r in anchor])
    delta = np.logaddexp(0, z) - y * z - (np.logaddexp(0, b) - y * b)
    return dict(corrected=int(np.sum(~old_ok & new_ok)), harmed=int(np.sum(old_ok & ~new_ok)), kept_correct=int(np.sum(old_ok & new_ok)), kept_wrong=int(np.sum(~old_ok & ~new_ok)), worsened_BCE_n=int(np.sum(delta > 1e-12)), improved_BCE_n=int(np.sum(delta < -1e-12)), mean_BCE_change=float(delta.mean()), mean_positive_regret=float(np.maximum(delta, 0).mean()))
