from __future__ import annotations
import hashlib
import json
import time
from typing import Any
import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit
LAMBDA = 0.001
GTOL = 1e-06
SOLVER_OPTIONS = {'gtol': GTOL, 'ftol': 0.0, 'maxiter': 1000, 'maxls': 50, 'maxcor': 20}
FEATURE_DIM = 384
PATCHES = 784

def pos384() -> dict[str, Any]:
    frequencies = sorted(((u, v) for u in range(28) for v in range(28) if (u, v) != (0, 0)), key=lambda uv: (uv[0] ** 2 + uv[1] ** 2, uv[0], uv[1]))[:384]
    r, c = np.meshgrid(np.arange(28, dtype=np.float64), np.arange(28, dtype=np.float64), indexing='ij')
    values = np.empty((PATCHES, FEATURE_DIM), dtype=np.float64)
    for column, (u, v) in enumerate(frequencies):
        ku = 1 / np.sqrt(28.0) if u == 0 else np.sqrt(2.0 / 28.0)
        kv = 1 / np.sqrt(28.0) if v == 0 else np.sqrt(2.0 / 28.0)
        values[:, column] = (ku * kv * np.cos(np.pi * (2 * r + 1) * u / 56.0) * np.cos(np.pi * (2 * c + 1) * v / 56.0)).reshape(-1)
    return {'features': values, 'frequency_pairs': frequencies, 'dtype': 'float64', 'shape': [784, 384], 'patch_order': 'row-major r*28+c', 'feature_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(), 'frequency_order': '(u*u+v*v,u,v), exclude(0,0), first384', 'uses_image_content': False, 'uses_mask_content': False, 'learned_coefficients': 384, 'learned_biases': 1}

def image_equal_weights(sample_image_index) -> tuple[np.ndarray, dict[str, Any]]:
    index = np.asarray(sample_image_index)
    if index.ndim != 1 or not len(index):
        raise ValueError('Nonempty sample-to-image membership is required.')
    if index.dtype.kind == 'f' and (not np.isfinite(index).all()):
        raise ValueError('Image memberships must be finite.')
    identities, inverse, counts = np.unique(index, return_inverse=True, return_counts=True)
    weights = 1.0 / (len(identities) * counts[inverse].astype(np.float64))
    return (weights, {'n_images': len(identities), 'n_samples': len(index), 'samples_per_image': counts.tolist(), 'image_order': identities.tolist(), 'image_weight': 1.0 / len(identities), 'weight_sum': float(weights.sum())})

def _flatten(X, y=None, *, sample_image_index=None, image_ids=None, expected_dim=FEATURE_DIM):
    values = np.asarray(X)
    if values.ndim not in (2, 3) or not values.size or values.shape[-1] != expected_dim:
        raise ValueError(f'Expected nonempty [images,patches,{expected_dim}] or [samples,{expected_dim}] features.')
    if not np.isfinite(values).all():
        raise ValueError('Features contain nonfinite values.')
    original_shape = values.shape[:-1]
    if values.ndim == 3:
        n, m, d = values.shape
        if sample_image_index is not None:
            raise ValueError('3D features already define whole-image membership.')
        if image_ids is not None and (len(image_ids) != n or len(set(image_ids)) != n):
            raise ValueError('Image IDs must be unique and match the image dimension.')
        index = np.repeat(np.arange(n), m)
    else:
        if sample_image_index is None or len(sample_image_index) != len(values):
            raise ValueError('Flat features require explicit sample_image_index; patches are not independent images.')
        index = np.asarray(sample_image_index)
    flat = np.asarray(values.reshape(-1, expected_dim), dtype=np.float64)
    labels = None
    if y is not None:
        labels = np.asarray(y)
        if labels.shape != original_shape or not np.all((labels == 0) | (labels == 1)):
            raise ValueError('Binary known labels must exactly align with feature samples; unknown labels are not zero.')
        labels = labels.astype(np.float64, copy=False).reshape(-1)
    weights, membership = image_equal_weights(index)
    return (flat, labels, weights, membership, original_shape)

def fit_scaler(X, *, role='PROBE_TRAIN', sample_image_index=None, image_ids=None, expected_dim=FEATURE_DIM):
    if role != 'PROBE_TRAIN':
        raise ValueError('Scaler fitting is allowed only on PROBE_TRAIN.')
    start = time.perf_counter()
    flat, _, weights, membership, _ = _flatten(X, sample_image_index=sample_image_index, image_ids=image_ids, expected_dim=expected_dim)
    mean = np.sum(flat * weights[:, None], axis=0, dtype=np.float64)
    zero = np.ptp(flat, axis=0) == 0
    mean[zero] = flat[0, zero]
    centered = flat - mean
    variance = np.einsum('ij,i,ij->j', centered, weights, centered, optimize=True)
    variance[zero] = 0.0
    scale = np.sqrt(variance)
    scale[zero] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ArithmeticError('Invalid population standard deviation; no epsilon fallback is permitted.')
    return {'mean': mean, 'scale': scale, 'zero_variance_mask': zero, 'population_variance': variance, 'zero_variance_count': int(zero.sum()), 'fit_role': role, 'membership': membership, 'training_image_ids': list(image_ids) if image_ids is not None else None, 'dtype': 'float64', 'elapsed_s': time.perf_counter() - start, 'weighting': 'equal image; equal known patch within each image', 'variance_definition': 'weighted population variance (ddof=0)'}

def transform_features(X, scaler):
    if scaler.get('fit_role') != 'PROBE_TRAIN':
        raise ValueError('A PROBE_TRAIN-fitted scaler is required.')
    values = np.asarray(X, dtype=np.float64)
    mean, scale = (np.asarray(scaler['mean'], dtype=np.float64), np.asarray(scaler['scale'], dtype=np.float64))
    zero = np.asarray(scaler['zero_variance_mask'], dtype=bool)
    if values.ndim not in (2, 3) or values.shape[-1] != len(mean) or (not np.isfinite(values).all()):
        raise ValueError('Prediction feature shape/dtype or values are invalid.')
    result = (values - mean) / scale
    result[..., zero] = 0.0
    return result

def logistic_objective(theta, X, y, weights):
    theta = np.asarray(theta, dtype=np.float64)
    X, y, weights = (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(weights, dtype=np.float64))
    if X.ndim != 2 or theta.shape != (X.shape[1] + 1,) or y.shape != (len(X),) or (weights.shape != y.shape):
        raise ValueError('Objective dimensions differ.')
    z = X @ theta[:-1] + theta[-1]
    losses = np.logaddexp(0.0, z) - y * z
    objective = float(np.dot(weights, losses) + 0.5 * LAMBDA * np.dot(theta[:-1], theta[:-1]))
    residual = weights * (expit(z) - y)
    gradient = np.empty_like(theta)
    gradient[:-1] = X.T @ residual + LAMBDA * theta[:-1]
    gradient[-1] = residual.sum(dtype=np.float64)
    return (objective, gradient)

def fit_probe(X, y, *, role='PROBE_TRAIN', image_ids=None, sample_image_index=None, expected_dim=FEATURE_DIM, progress=None) -> dict[str, Any]:
    if role != 'PROBE_TRAIN':
        raise ValueError('Probe fitting is allowed only on PROBE_TRAIN.')
    flat, labels, weights, membership, _ = _flatten(X, y, sample_image_index=sample_image_index, image_ids=image_ids, expected_dim=expected_dim)
    scaler = fit_scaler(X, role=role, sample_image_index=sample_image_index, image_ids=image_ids, expected_dim=expected_dim)
    standardized = transform_features(flat, scaler)
    start = time.perf_counter()
    logs = []
    objective_computations = 0
    solver_requests = 0
    callback_extra_computations = 0
    cache_theta, cache_value = (None, None)

    def evaluate(theta):
        nonlocal cache_theta, cache_value, objective_computations
        if cache_theta is None or not np.array_equal(theta, cache_theta):
            cache_value = logistic_objective(theta, standardized, labels, weights)
            cache_theta = np.asarray(theta, dtype=np.float64).copy()
            objective_computations += 1
        return cache_value

    def solver(theta):
        nonlocal solver_requests
        solver_requests += 1
        return evaluate(theta)

    def record(theta, iteration):
        value, gradient = evaluate(theta)
        row = {'iteration': iteration, 'objective': float(value), 'gradient_inf_norm': float(np.max(np.abs(gradient))), 'elapsed_s': time.perf_counter() - start, 'actual_objective_evaluations_so_far': objective_computations, 'optimizer_function_requests_so_far': solver_requests}
        logs.append(row)
        if progress is not None:
            progress(**row)

    def callback(theta):
        nonlocal callback_extra_computations
        previous = objective_computations
        record(theta, len(logs))
        callback_extra_computations += objective_computations - previous
    initial = np.zeros(expected_dim + 1, dtype=np.float64)
    record(initial, 0)
    result = minimize(solver, initial, jac=True, method='L-BFGS-B', callback=callback, options=dict(SOLVER_OPTIONS))
    final_objective, final_gradient = evaluate(result.x)
    gradient_inf_norm = float(np.max(np.abs(final_gradient)))
    finite = bool(np.isfinite(result.x).all() and np.isfinite(final_objective) and np.isfinite(final_gradient).all())
    converged = finite and gradient_inf_norm <= GTOL
    if int(result.nit) > 1000:
        raise ArithmeticError('Solver exceeded the fixed iteration budget.')
    training = {'state': 'completed_converged' if converged else 'completed_not_converged', 'converged': converged, 'technical_valid': converged, 'finite_solution': finite, 'scipy_success': bool(result.success), 'scipy_status': int(result.status), 'scipy_message': str(result.message), 'iterations': int(result.nit), 'optimizer_nfev': int(result.nfev), 'optimizer_njev': int(result.njev), 'optimizer_function_requests': solver_requests, 'actual_objective_evaluations': objective_computations, 'callback_extra_objective_evaluations': callback_extra_computations, 'final_objective': float(final_objective), 'gradient_inf_norm': gradient_inf_norm, 'scaler_elapsed_s': scaler['elapsed_s'], 'fit_elapsed_s': time.perf_counter() - start, 'n_images': membership['n_images'], 'n_patches': membership['n_samples'], 'positive_patches': int(labels.sum()), 'zero_patches': int(len(labels) - labels.sum()), 'learned_coefficients': expected_dim, 'learned_biases': 1, 'zero_variance_features': scaler['zero_variance_count'], 'lambda': LAMBDA, 'bias_regularized': False, 'initialization': 'all zeros', 'loss': 'mean over images of mean full BCE over known patches plus .001/2 * coefficient L2 squared', 'solver': 'SciPy L-BFGS-B analytic gradient', 'solver_options': dict(SOLVER_OPTIONS), 'numpy_version': np.__version__, 'scipy_version': scipy.__version__, 'dtype': 'float64', 'fit_role': role, 'class_weighting': False, 'oversampling': False, 'epoch_or_patience_rule': None, 'decision': 'logit>=0 (equivalent sigmoid>=.5)', 'new_probe_fits': 1, 'neural_backbone_optimizer_steps': 0}
    return {'coefficients': np.asarray(result.x[:-1], dtype=np.float64), 'bias': float(result.x[-1]), 'scaler': scaler, 'training_result': training, 'iteration_log': logs}

def predict_probe(X, fitted):
    weights = np.asarray(fitted['coefficients'], dtype=np.float64)
    standardized = transform_features(X, fitted['scaler'])
    if weights.shape != (standardized.shape[-1],) or not np.isfinite(weights).all() or (not np.isfinite(fitted['bias'])):
        raise ValueError('Invalid frozen probe coefficients.')
    logits = standardized @ weights + float(fitted['bias'])
    return {'logits': logits, 'probabilities': expit(logits)}
