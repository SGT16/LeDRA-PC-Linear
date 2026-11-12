from __future__ import annotations
import math
import numpy as np
POLICIES = ('source_INNER_F1', 'source_INNER_Se95', 'source_INNER_Sp95')
UNITS = dict(brset='patient', deepdrid='patient', mbrset='patient', messidor2='exam', sustech='image', paraguay='image', aptos='image')

def _arrays(labels, probabilities):
    raw_y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if raw_y.ndim != 1 or p.shape != raw_y.shape or (not len(raw_y)) or (not np.isfinite(raw_y).all()) or (set(raw_y.tolist()) != {0.0, 1.0}) or (not np.isfinite(p).all()) or (not ((p >= 0) & (p <= 1)).all()):
        raise ValueError('Aligned finite binary outcomes/probabilities with both classes required')
    return (raw_y.astype(np.int64), p)

def empirical_operating_points(labels, probabilities):
    y, p = _arrays(labels, probabilities)
    positive, negative = (int(y.sum()), int(len(y) - y.sum()))
    order = np.argsort(-p, kind='stable')
    ys, ps = (y[order], p[order])
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    cumulative = np.cumsum(ys, dtype=np.int64)
    tp = np.r_[0, cumulative[ends]].astype(np.int64)
    fp = np.r_[0, ends + 1 - cumulative[ends]].astype(np.int64)
    thresholds = np.r_[np.inf, ps[ends]]
    fn, tn = (positive - tp, negative - fp)
    se_index = max(np.flatnonzero(20 * fp <= negative).tolist(), key=lambda i: (int(tp[i]), -int(fp[i]), float(thresholds[i])))
    sp_index = max(np.flatnonzero(20 * fn <= positive).tolist(), key=lambda i: (int(tn[i]), int(tp[i]), float(thresholds[i])))

    def point(index, prefix):
        cut = float(thresholds[index])
        return {prefix + '_threshold': '+Infinity' if math.isinf(cut) else cut, prefix + '_threshold_kind': 'positive_infinity' if math.isinf(cut) else 'finite', prefix + '_TP': int(tp[index]), prefix + '_FP': int(fp[index]), prefix + '_FN': int(fn[index]), prefix + '_TN': int(tn[index]), prefix + '_sensitivity': float(tp[index] / positive), prefix + '_specificity': float(tn[index] / negative)}
    return dict(Se_at_Sp95=float(tp[se_index] / positive), Sp_at_Se95=float(tn[sp_index] / negative), roc_attainable_points=int(len(tp)), **point(se_index, 'se_sp95'), **point(sp_index, 'sp_se95'))

def decode_threshold(value):
    if value == '+Infinity':
        return math.inf
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError('A saved probability threshold must be in [0,1] or +Infinity')
    return number

def source_thresholds(records, existing_F1):
    records = list(records)
    points = empirical_operating_points([r['label'] for r in records], [r['probability'] for r in records])
    decode_threshold(existing_F1)
    return dict(source_INNER_F1=existing_F1, source_INNER_Se95=points['sp_se95_threshold'], source_INNER_Sp95=points['se_sp95_threshold'])

def fixed_threshold_metrics(labels, probabilities, threshold):
    y, p = _arrays(labels, probabilities)
    prediction = p >= decode_threshold(threshold)
    tp = int(((y == 1) & prediction).sum())
    fp = int(((y == 0) & prediction).sum())
    fn = int(((y == 1) & ~prediction).sum())
    tn = int(((y == 0) & ~prediction).sum())
    P, N = (int(y.sum()), int(len(y) - y.sum()))
    precision = tp / (tp + fp) if tp + fp else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, F1=2 * tp / (2 * tp + fp + fn), accuracy=(tp + tn) / len(y), precision=precision, sensitivity=tp / P, specificity=tn / N, FNR=fn / P, TN_over_negative_n=tn / N, TN_over_all_n=tn / len(y))

def scoring_records(dataset, rows, logits):
    from lesion_allcohorts.evaluate import records_for
    if dataset not in UNITS:
        raise ValueError('Unknown target cohort')
    return records_for(dataset, rows, logits)

def measure_records(records, thresholds):
    from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
    records = list(records)
    y, p = _arrays([r['label'] for r in records], [r['probability'] for r in records])
    if set(thresholds) != set(POLICIES):
        raise ValueError('All three source-locked policies are required')
    result = dict(n=len(y), positive_n=int(y.sum()), negative_n=int(len(y) - y.sum()), prevalence=float(y.mean()), AUROC=float(roc_auc_score(y, p)), AP=float(average_precision_score(y, p)), logloss=float(log_loss(y, p)), former_S5_ineligible_units_included=sum((not bool(r.get('eligible', True)) for r in records)))
    result.update(empirical_operating_points(y, p))
    for policy in POLICIES:
        result['threshold_' + policy] = thresholds[policy]
        result.update({metric + '_' + policy: value for metric, value in fixed_threshold_metrics(y, p, thresholds[policy]).items()})
    return result

def measure(dataset, rows, logits, thresholds):
    return measure_records(scoring_records(dataset, rows, logits), thresholds)
SUMMARY_METRICS = ('AUROC', 'AP', 'logloss', 'Se_at_Sp95', 'Sp_at_Se95', 'se_sp95_specificity', 'sp_se95_sensitivity') + tuple((metric + '_' + policy for policy in POLICIES for metric in ('F1', 'accuracy', 'precision', 'sensitivity', 'specificity', 'FNR', 'TN_over_negative_n', 'TN_over_all_n')))

def aggregate_metrics(records, datasets, variants, seeds, folds):
    records = list(records)
    expected = {(a, b, v, s, f) for a in datasets for b in datasets if a != b for v in variants for s in seeds for f in folds}
    identity = lambda r: (r['source_dataset'], r['target_dataset'], r['variant'], int(r['seed']), int(r['source_fold']))
    index = {identity(r): r for r in records}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError('A complete transfer matrix without duplicates is required')
    seed_rows, summaries = ([], [])
    for source in datasets:
        for target in datasets:
            if source == target:
                continue
            group = [r for r in records if r['source_dataset'] == source and r['target_dataset'] == target]
            for name in ('n', 'positive_n', 'negative_n', 'target_population_sha256'):
                if len({r[name] for r in group}) != 1:
                    raise ValueError('Paired models/checkpoints use different target populations: ' + name)
            for variant in variants:
                built = []
                for seed in seeds:
                    rows = [index[source, target, variant, seed, f] for f in folds]
                    if any((not all((math.isfinite(float(r[m])) for m in SUMMARY_METRICS)) for r in rows)):
                        raise ValueError('Missing or nonfinite transfer metric')
                    row = dict(source_dataset=source, target_dataset=target, variant=variant, seed=seed, source_folds=len(folds), scoring_unit=UNITS[target], target_population_sha256=rows[0]['target_population_sha256'], n=rows[0]['n'], positive_n=rows[0]['positive_n'], negative_n=rows[0]['negative_n'], **{m: float(np.mean([r[m] for r in rows])) for m in SUMMARY_METRICS})
                    seed_rows.append(row)
                    built.append(row)
                summary = dict(source_dataset=source, target_dataset=target, variant=variant, seeds=len(seeds), folds_per_seed=len(folds), scoring_unit=UNITS[target], n=built[0]['n'], positive_n=built[0]['positive_n'], negative_n=built[0]['negative_n'], target_population_sha256=built[0]['target_population_sha256'], seed_SD_ddof=1)
                for metric in SUMMARY_METRICS:
                    values = [r[metric] for r in built]
                    summary[metric + '_mean'] = float(np.mean(values))
                    summary[metric + '_seed_SD'] = float(np.std(values, ddof=1))
                summaries.append(summary)
    return (seed_rows, summaries)
