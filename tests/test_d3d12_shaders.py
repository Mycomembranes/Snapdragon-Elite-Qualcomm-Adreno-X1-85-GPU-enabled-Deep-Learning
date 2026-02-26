"""Per-shader correctness tests for D3D12 native compute backend.

Tests each shader's GPU output against numpy reference.
Run: python3 test_d3d12_shaders.py

Usage:
    python3 test_d3d12_shaders.py           # Run all tests
    python3 test_d3d12_shaders.py matmul     # Run specific test
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d3d12_tensor import (
    d3d12_init, D3D12Tensor,
    d3d12_add, d3d12_sub, d3d12_mul, d3d12_neg, d3d12_scalar_mul,
    d3d12_relu, d3d12_gelu, d3d12_sigmoid, d3d12_tanh,
    d3d12_matmul, d3d12_transpose,
    d3d12_matmul_add_relu, d3d12_matmul_add,
    d3d12_relu_backward, d3d12_gelu_backward,
    d3d12_softmax_backward, d3d12_layernorm_backward,
    d3d12_layer_norm, d3d12_softmax,
    d3d12_sum, d3d12_mean, d3d12_max_reduce,
    d3d12_cross_entropy, d3d12_focal_bce,
    d3d12_embedding_lookup,
)


def _check(name, gpu_result, expected, atol=1e-5):
    """Verify GPU result matches expected numpy output."""
    if isinstance(gpu_result, D3D12Tensor):
        gpu_result = gpu_result.numpy()
    if not np.allclose(expected, gpu_result, atol=atol, rtol=1e-4):
        max_diff = np.max(np.abs(expected - gpu_result))
        print(f"  FAIL: {name} — max diff={max_diff:.2e} (atol={atol})")
        # Show first few mismatches
        mask = ~np.isclose(expected, gpu_result, atol=atol, rtol=1e-4)
        if mask.any():
            idxs = np.argwhere(mask)[:5]
            for idx in idxs:
                idx_t = tuple(idx)
                print(f"    [{idx_t}] expected={expected[idx_t]:.6f} got={gpu_result[idx_t]:.6f}")
        return False
    print(f"  PASS: {name}")
    return True


# ========================================================================
# Elementwise operations
# ========================================================================

def test_add():
    for shape in [(4,), (1000,), (64, 64), (100, 50)]:
        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)
        expected = a + b
        result = d3d12_add(D3D12Tensor.from_numpy(a), D3D12Tensor.from_numpy(b))
        if not _check(f"add {shape}", result, expected):
            return False
    return True


def test_sub():
    for shape in [(4,), (1000,), (64, 64)]:
        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)
        expected = a - b
        result = d3d12_sub(D3D12Tensor.from_numpy(a), D3D12Tensor.from_numpy(b))
        if not _check(f"sub {shape}", result, expected):
            return False
    return True


def test_mul():
    for shape in [(4,), (1000,), (64, 64)]:
        a = np.random.randn(*shape).astype(np.float32)
        b = np.random.randn(*shape).astype(np.float32)
        expected = a * b
        result = d3d12_mul(D3D12Tensor.from_numpy(a), D3D12Tensor.from_numpy(b))
        if not _check(f"mul {shape}", result, expected):
            return False
    return True


def test_neg():
    for shape in [(4,), (1000,), (64, 64)]:
        a = np.random.randn(*shape).astype(np.float32)
        expected = -a
        result = d3d12_neg(D3D12Tensor.from_numpy(a))
        if not _check(f"neg {shape}", result, expected):
            return False
    return True


def test_scalar_mul():
    a = np.random.randn(1000).astype(np.float32)
    for s in [0.0, 1.0, -2.5, 3.14]:
        expected = a * s
        result = d3d12_scalar_mul(D3D12Tensor.from_numpy(a), s)
        if not _check(f"scalar_mul s={s}", result, expected):
            return False
    return True


# ========================================================================
# Activation functions
# ========================================================================

def test_relu():
    for shape in [(4,), (1000,), (64, 64)]:
        x = np.random.randn(*shape).astype(np.float32)
        expected = np.maximum(0, x)
        result = d3d12_relu(D3D12Tensor.from_numpy(x))
        if not _check(f"relu {shape}", result, expected):
            return False
    return True


def test_gelu():
    for shape in [(4,), (1000,), (64, 64)]:
        x = np.random.randn(*shape).astype(np.float32)
        # GELU approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
        expected = x * cdf
        result = d3d12_gelu(D3D12Tensor.from_numpy(x))
        if not _check(f"gelu {shape}", result, expected, atol=1e-4):
            return False
    return True


def test_sigmoid():
    for shape in [(4,), (1000,), (64, 64)]:
        x = np.random.randn(*shape).astype(np.float32)
        expected = 1.0 / (1.0 + np.exp(-x))
        result = d3d12_sigmoid(D3D12Tensor.from_numpy(x))
        if not _check(f"sigmoid {shape}", result, expected, atol=1e-5):
            return False
    return True


def test_tanh():
    for shape in [(4,), (1000,), (64, 64)]:
        x = np.random.randn(*shape).astype(np.float32)
        expected = np.tanh(x)
        result = d3d12_tanh(D3D12Tensor.from_numpy(x))
        if not _check(f"tanh {shape}", result, expected, atol=1e-5):
            return False
    return True


# ========================================================================
# Matrix operations
# ========================================================================

def test_matmul():
    sizes = [(4, 4, 4), (64, 32, 16), (100, 50, 200), (256, 256, 256)]
    for m, k, n in sizes:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        expected = a @ b
        result = d3d12_matmul(D3D12Tensor.from_numpy(a), D3D12Tensor.from_numpy(b))
        if not _check(f"matmul ({m},{k})@({k},{n})", result, expected, atol=1e-3):
            return False
    return True


def test_transpose():
    for shape in [(4, 8), (64, 32), (100, 200)]:
        x = np.random.randn(*shape).astype(np.float32)
        expected = x.T
        result = d3d12_transpose(D3D12Tensor.from_numpy(x))
        if not _check(f"transpose {shape}", result, expected):
            return False
    return True


def test_matmul_add_relu():
    for m, k, n in [(16, 8, 4), (64, 32, 16)]:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        bias = np.random.randn(m, n).astype(np.float32)
        pre_relu_expected = a @ b + bias
        expected = np.maximum(0, pre_relu_expected)
        out, pre_relu = d3d12_matmul_add_relu(
            D3D12Tensor.from_numpy(a),
            D3D12Tensor.from_numpy(b),
            D3D12Tensor.from_numpy(bias))
        if not _check(f"matmul_add_relu out ({m},{k},{n})", out, expected, atol=1e-3):
            return False
        if not _check(f"matmul_add_relu pre ({m},{k},{n})", pre_relu, pre_relu_expected, atol=1e-3):
            return False
    return True


def test_matmul_add():
    for m, k, n in [(16, 8, 4), (64, 32, 16)]:
        a = np.random.randn(m, k).astype(np.float32)
        b = np.random.randn(k, n).astype(np.float32)
        bias = np.random.randn(m, n).astype(np.float32)
        expected = a @ b + bias
        result = d3d12_matmul_add(
            D3D12Tensor.from_numpy(a),
            D3D12Tensor.from_numpy(b),
            D3D12Tensor.from_numpy(bias))
        if not _check(f"matmul_add ({m},{k},{n})", result, expected, atol=1e-3):
            return False
    return True


# ========================================================================
# Backward operations
# ========================================================================

def test_relu_backward():
    x = np.random.randn(1000).astype(np.float32)
    grad = np.random.randn(1000).astype(np.float32)
    expected = grad * (x > 0).astype(np.float32)
    result = d3d12_relu_backward(D3D12Tensor.from_numpy(grad), D3D12Tensor.from_numpy(x))
    return _check("relu_backward", result, expected)


def test_gelu_backward():
    x = np.random.randn(1000).astype(np.float32)
    grad = np.random.randn(1000).astype(np.float32)
    # Sigmoid approximation: gelu_deriv ≈ sigmoid(1.702 * x)
    gelu_deriv = 1.0 / (1.0 + np.exp(-1.702 * x))
    expected = grad * gelu_deriv
    result = d3d12_gelu_backward(D3D12Tensor.from_numpy(grad), D3D12Tensor.from_numpy(x))
    return _check("gelu_backward", result, expected, atol=1e-4)


def test_softmax_backward():
    rows, width = 8, 32
    probs = np.random.dirichlet(np.ones(width), size=rows).astype(np.float32)
    grad = np.random.randn(rows, width).astype(np.float32)
    # softmax_backward: P * (grad - sum(grad * P))
    dot = np.sum(grad * probs, axis=-1, keepdims=True)
    expected = probs * (grad - dot)
    result = d3d12_softmax_backward(D3D12Tensor.from_numpy(grad), D3D12Tensor.from_numpy(probs))
    return _check("softmax_backward", result, expected, atol=1e-4)


def test_layernorm_backward():
    rows, width = 8, 64
    x = np.random.randn(rows, width).astype(np.float32)
    gamma = np.random.randn(width).astype(np.float32)
    grad = np.random.randn(rows, width).astype(np.float32)
    eps = 1e-5

    # Reference computation
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std
    dx_hat = grad * gamma
    sum_dxhat = dx_hat.sum(axis=-1, keepdims=True)
    sum_dxhat_xhat = (dx_hat * x_hat).sum(axis=-1, keepdims=True)
    n = float(width)
    expected = inv_std / n * (n * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat)

    result = d3d12_layernorm_backward(
        D3D12Tensor.from_numpy(grad),
        D3D12Tensor.from_numpy(x),
        D3D12Tensor.from_numpy(gamma), eps)
    return _check("layernorm_backward", result, expected, atol=1e-3)


# ========================================================================
# Normalization / Softmax
# ========================================================================

def test_layer_norm():
    for rows, width in [(4, 16), (8, 64), (16, 256)]:
        x = np.random.randn(rows, width).astype(np.float32)
        gamma = np.ones(width, dtype=np.float32)
        beta = np.zeros(width, dtype=np.float32)
        eps = 1e-5

        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        expected = (x - mean) / np.sqrt(var + eps) * gamma + beta

        result = d3d12_layer_norm(
            D3D12Tensor.from_numpy(x),
            D3D12Tensor.from_numpy(gamma),
            D3D12Tensor.from_numpy(beta), eps)
        if not _check(f"layer_norm ({rows},{width})", result, expected, atol=1e-4):
            return False
    return True


def test_softmax():
    for rows, width in [(4, 16), (8, 64), (2, 256)]:
        x = np.random.randn(rows, width).astype(np.float32)
        # Stable softmax
        x_max = x.max(axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        expected = exp_x / exp_x.sum(axis=-1, keepdims=True)

        result = d3d12_softmax(D3D12Tensor.from_numpy(x))
        if not _check(f"softmax ({rows},{width})", result, expected, atol=1e-5):
            return False
    return True


# ========================================================================
# Reductions
# ========================================================================

def test_sum():
    x = np.random.randn(1000).astype(np.float32)
    expected_sum = x.sum()
    result = d3d12_sum(D3D12Tensor.from_numpy(x))
    result_val = result.numpy().item()
    if not np.isclose(expected_sum, result_val, atol=1e-1, rtol=1e-3):
        print(f"  FAIL: sum — expected={expected_sum:.4f} got={result_val:.4f}")
        return False
    print(f"  PASS: sum (expected={expected_sum:.4f}, got={result_val:.4f})")
    return True


def test_mean():
    x = np.random.randn(1000).astype(np.float32)
    expected_mean = x.mean()
    result = d3d12_mean(D3D12Tensor.from_numpy(x))
    result_val = result.numpy().item()
    if not np.isclose(expected_mean, result_val, atol=1e-3, rtol=1e-3):
        print(f"  FAIL: mean — expected={expected_mean:.6f} got={result_val:.6f}")
        return False
    print(f"  PASS: mean (expected={expected_mean:.6f}, got={result_val:.6f})")
    return True


def test_max():
    x = np.random.randn(1000).astype(np.float32)
    expected_max = x.max()
    result = d3d12_max_reduce(D3D12Tensor.from_numpy(x))
    result_val = result.numpy().item()
    if not np.isclose(expected_max, result_val, atol=1e-5):
        print(f"  FAIL: max — expected={expected_max:.6f} got={result_val:.6f}")
        return False
    print(f"  PASS: max (expected={expected_max:.6f}, got={result_val:.6f})")
    return True


# ========================================================================
# Loss functions
# ========================================================================

def test_cross_entropy():
    n = 100
    logits = np.random.randn(n).astype(np.float32) * 0.5  # small values
    targets = np.random.randint(0, 2, n).astype(np.float32)

    log_p = logits
    expected = -targets * log_p - (1.0 - targets) * np.log(1.0 - np.exp(log_p) + 1e-6)
    result = d3d12_cross_entropy(D3D12Tensor.from_numpy(logits), D3D12Tensor.from_numpy(targets))
    return _check("cross_entropy", result, expected, atol=1e-3)


def test_focal_bce():
    n = 100
    logits = np.random.randn(n).astype(np.float32)
    targets = np.random.randint(0, 2, n).astype(np.float32)
    gamma, alpha = 2.0, 0.25

    p = 1.0 / (1.0 + np.exp(-logits))
    ce = -targets * np.log(p + 1e-6) - (1.0 - targets) * np.log(1.0 - p + 1e-6)
    p_t = np.where(targets > 0.5, p, 1.0 - p)
    focal_weight = (1.0 - p_t) ** gamma
    expected = alpha * (1.0 - alpha) * focal_weight * ce

    result = d3d12_focal_bce(D3D12Tensor.from_numpy(logits), D3D12Tensor.from_numpy(targets), gamma, alpha)
    return _check("focal_bce", result, expected, atol=1e-3)


# ========================================================================
# Embedding
# ========================================================================

def test_embedding():
    vocab_size, dim = 100, 32
    weight = np.random.randn(vocab_size, dim).astype(np.float32)
    indices = np.array([0, 5, 99, 42, 10], dtype=np.uint32)
    expected = weight[indices]

    result = d3d12_embedding_lookup(
        D3D12Tensor.from_numpy(weight),
        D3D12Tensor(D3D12Tensor.from_numpy(indices.view(np.float32)).handle, indices.shape, "uint32"))
    return _check("embedding", result, expected)


# ========================================================================
# Main
# ========================================================================

ALL_TESTS = {
    "add": test_add,
    "sub": test_sub,
    "mul": test_mul,
    "neg": test_neg,
    "scalar_mul": test_scalar_mul,
    "relu": test_relu,
    "gelu": test_gelu,
    "sigmoid": test_sigmoid,
    "tanh": test_tanh,
    "matmul": test_matmul,
    "transpose": test_transpose,
    "matmul_add_relu": test_matmul_add_relu,
    "matmul_add": test_matmul_add,
    "relu_backward": test_relu_backward,
    "gelu_backward": test_gelu_backward,
    "softmax_backward": test_softmax_backward,
    "layernorm_backward": test_layernorm_backward,
    "layer_norm": test_layer_norm,
    "softmax": test_softmax,
    "sum": test_sum,
    "mean": test_mean,
    "max": test_max,
    "cross_entropy": test_cross_entropy,
    "focal_bce": test_focal_bce,
    "embedding": test_embedding,
}


if __name__ == "__main__":
    d3d12_init()

    # Filter tests if argument given
    filter_name = sys.argv[1] if len(sys.argv) > 1 else None
    tests_to_run = {}
    if filter_name:
        for name, fn in ALL_TESTS.items():
            if filter_name in name:
                tests_to_run[name] = fn
    else:
        tests_to_run = ALL_TESTS

    print("=" * 60)
    print("D3D12 Shader Correctness Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests_to_run.items():
        print(f"\n[{name}]")
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                errors.append(name)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            errors.append(f"{name} (exception)")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print(f"Failed: {', '.join(errors)}")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
