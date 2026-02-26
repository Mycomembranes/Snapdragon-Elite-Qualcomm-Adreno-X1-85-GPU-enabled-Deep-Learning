#!/usr/bin/env python3
"""Integration test for d3d12_tensor.py high-level API."""
import sys, os
sys.path.insert(0, "/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")
os.chdir("/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")

import numpy as np
from d3d12_tensor import (
    d3d12_init, D3D12Tensor,
    d3d12_add, d3d12_sub, d3d12_mul, d3d12_neg,
    d3d12_relu, d3d12_gelu, d3d12_sigmoid, d3d12_tanh,
    d3d12_scalar_mul, d3d12_matmul, d3d12_transpose,
    d3d12_layer_norm, d3d12_softmax,
    d3d12_relu_backward, d3d12_gelu_backward,
    d3d12_cross_entropy, d3d12_sum, d3d12_mean,
    d3d12_embedding_lookup, d3d12_max_reduce,
)

d3d12_init()  # raises on failure

N = 256
passed = 0
failed = 0

def test(name, gpu_fn, ref_fn, atol=1e-5):
    global passed, failed
    try:
        result = gpu_fn()
        expected = ref_fn()
        if isinstance(result, D3D12Tensor):
            result = result.numpy()
        ok = np.allclose(result, expected, atol=atol)
        diff = np.max(np.abs(result - expected))
        status = "OK" if ok else "FAIL"
        print(f"  {name}: {status} (max_diff={diff:.2e})")
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"    Expected: {expected.flatten()[:5]}")
            print(f"    Got:      {result.flatten()[:5]}")
    except Exception as e:
        failed += 1
        print(f"  {name}: ERROR - {e}")

a_np = np.random.randn(N).astype(np.float32)
b_np = np.random.randn(N).astype(np.float32)
a = D3D12Tensor.from_numpy(a_np)
b = D3D12Tensor.from_numpy(b_np)

print("=== Element-wise ops ===")
test("add", lambda: d3d12_add(a, b), lambda: a_np + b_np)
test("sub", lambda: d3d12_sub(a, b), lambda: a_np - b_np)
test("mul", lambda: d3d12_mul(a, b), lambda: a_np * b_np)
test("neg", lambda: d3d12_neg(a), lambda: -a_np)
test("scalar_mul", lambda: d3d12_scalar_mul(a, 2.5), lambda: a_np * 2.5)

print("\n=== Activations ===")
x_np = np.random.randn(N).astype(np.float32)
x = D3D12Tensor.from_numpy(x_np)
test("relu", lambda: d3d12_relu(x), lambda: np.maximum(x_np, 0))
test("sigmoid", lambda: d3d12_sigmoid(x), lambda: 1.0 / (1.0 + np.exp(-x_np)))
test("tanh", lambda: d3d12_tanh(x), lambda: np.tanh(x_np))
# GELU uses tanh approx: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
def gelu_ref(v):
    return v * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (v + 0.044715 * v**3)))
test("gelu", lambda: d3d12_gelu(x), lambda: gelu_ref(x_np), atol=1e-5)

print("\n=== MatMul ===")
M, K, Nn = 32, 16, 8
ma_np = np.random.randn(M, K).astype(np.float32)
mb_np = np.random.randn(K, Nn).astype(np.float32)
ma = D3D12Tensor.from_numpy(ma_np)
mb = D3D12Tensor.from_numpy(mb_np)
test("matmul", lambda: d3d12_matmul(ma, mb), lambda: ma_np @ mb_np, atol=1e-4)

print("\n=== Transpose ===")
t_np = np.random.randn(16, 8).astype(np.float32)
t = D3D12Tensor.from_numpy(t_np)
test("transpose", lambda: d3d12_transpose(t), lambda: t_np.T)

print("\n=== LayerNorm ===")
ln_np = np.random.randn(4, 64).astype(np.float32)
g_np = np.ones(64, dtype=np.float32)
b_ln = np.zeros(64, dtype=np.float32)
ln = D3D12Tensor.from_numpy(ln_np)
gamma = D3D12Tensor.from_numpy(g_np)
beta = D3D12Tensor.from_numpy(b_ln)
def ln_ref():
    mu = ln_np.mean(axis=-1, keepdims=True)
    var = ln_np.var(axis=-1, keepdims=True)
    return (ln_np - mu) / np.sqrt(var + 1e-5) * g_np + b_ln
test("layer_norm", lambda: d3d12_layer_norm(ln, gamma, beta), ln_ref, atol=1e-4)

print("\n=== Softmax ===")
sm_np = np.random.randn(4, 64).astype(np.float32)
sm = D3D12Tensor.from_numpy(sm_np)
def sm_ref():
    e = np.exp(sm_np - sm_np.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
test("softmax", lambda: d3d12_softmax(sm), sm_ref, atol=1e-5)

print("\n=== Backward ops ===")
go_np = np.random.randn(N).astype(np.float32)
go = D3D12Tensor.from_numpy(go_np)
test("relu_backward", lambda: d3d12_relu_backward(go, x),
     lambda: go_np * (x_np > 0).astype(np.float32))
test("gelu_backward", lambda: d3d12_gelu_backward(go, x),
     lambda: go_np * (1.0 / (1.0 + np.exp(-1.702 * x_np))), atol=1e-5)

print("\n=== Reductions ===")
test("sum", lambda: d3d12_sum(x), lambda: np.array([x_np.sum()], dtype=np.float32), atol=1e-2)
test("mean", lambda: d3d12_mean(x), lambda: np.array([x_np.mean()], dtype=np.float32), atol=1e-4)
test("max_reduce", lambda: d3d12_max_reduce(x),
     lambda: np.array([x_np.max()], dtype=np.float32), atol=1e-5)

print("\n=== Cross Entropy ===")
log_p = -np.abs(np.random.randn(N).astype(np.float32)) - 0.1
tgt = np.random.randint(0, 2, N).astype(np.float32)
lp_t = D3D12Tensor.from_numpy(log_p)
tgt_t = D3D12Tensor.from_numpy(tgt)
test("cross_entropy", lambda: d3d12_cross_entropy(lp_t, tgt_t),
     lambda: -tgt * log_p - (1.0 - tgt) * np.log(1.0 - np.exp(log_p) + 1e-6), atol=1e-4)

print("\n=== Embedding ===")
vocab, dim = 32, 16
w_np = np.random.randn(vocab, dim).astype(np.float32)
idx_np = np.array([0, 5, 10, 31], dtype=np.int32)
w_t = D3D12Tensor.from_numpy(w_np)
# Embedding expects float32 indices (cast in shader via bitcast)
idx_t = D3D12Tensor.from_numpy(idx_np.view(np.float32))
result_emb = d3d12_embedding_lookup(w_t, idx_t)
result_emb_np = result_emb.numpy()
expected_emb = w_np[idx_np]
emb_ok = np.allclose(result_emb_np, expected_emb, atol=1e-6)
emb_diff = np.max(np.abs(result_emb_np - expected_emb))
print(f"  embedding: {'OK' if emb_ok else 'FAIL'} (max_diff={emb_diff:.2e})")
if emb_ok:
    passed += 1
else:
    failed += 1
    # Debug: show shapes and where they differ
    print(f"    result shape: {result_emb_np.shape}, expected shape: {expected_emb.shape}")
    diff_mask = ~np.isclose(result_emb_np.flatten(), expected_emb.flatten(), atol=1e-6)
    diff_idx = np.where(diff_mask)[0]
    if len(diff_idx) > 0:
        print(f"    First diff at flat idx {diff_idx[0]}: got={result_emb_np.flatten()[diff_idx[0]]}, exp={expected_emb.flatten()[diff_idx[0]]}")

print(f"\n=== Results: {passed} passed, {failed} failed ===")
sys.exit(1 if failed > 0 else 0)
