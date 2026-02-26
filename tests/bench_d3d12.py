#!/usr/bin/env python3
"""Quick benchmark of D3D12 native dispatch times."""
import sys, os, time
sys.path.insert(0, "/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")
os.chdir("/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")

import numpy as np
from d3d12_tensor import (
    d3d12_init, D3D12Tensor,
    d3d12_add, d3d12_matmul, d3d12_relu, d3d12_gelu,
    d3d12_layer_norm, d3d12_softmax, d3d12_sigmoid,
    d3d12_transpose, d3d12_cross_entropy,
)

d3d12_init()

def bench(name, fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    print(f"  {name:30s} {elapsed:8.3f} ms")

N = 4096
x_np = np.random.randn(N).astype(np.float32)
x = D3D12Tensor.from_numpy(x_np)
y = D3D12Tensor.from_numpy(x_np)

print("=== Element-wise (N=4096) ===")
bench("add", lambda: d3d12_add(x, y))
bench("relu", lambda: d3d12_relu(x))
bench("gelu", lambda: d3d12_gelu(x))
bench("sigmoid", lambda: d3d12_sigmoid(x))

print("\n=== MatMul ===")
for M, K, Nn in [(64, 64, 64), (256, 256, 256), (58, 256, 128)]:
    a = D3D12Tensor.from_numpy(np.random.randn(M, K).astype(np.float32))
    b = D3D12Tensor.from_numpy(np.random.randn(K, Nn).astype(np.float32))
    bench(f"matmul {M}x{K} @ {K}x{Nn}", lambda: d3d12_matmul(a, b))

print("\n=== LayerNorm / Softmax (4x256) ===")
ln = D3D12Tensor.from_numpy(np.random.randn(4, 256).astype(np.float32))
g = D3D12Tensor.from_numpy(np.ones(256, dtype=np.float32))
b = D3D12Tensor.from_numpy(np.zeros(256, dtype=np.float32))
bench("layer_norm 4x256", lambda: d3d12_layer_norm(ln, g, b))
bench("softmax 4x256", lambda: d3d12_softmax(ln))

print("\n=== Transpose (64x32) ===")
t = D3D12Tensor.from_numpy(np.random.randn(64, 32).astype(np.float32))
bench("transpose 64x32", lambda: d3d12_transpose(t))

print("\n=== MLP Forward (58->256->128->1) ===")
# Simulate contact head MLP
batch = 300
x_mlp = D3D12Tensor.from_numpy(np.random.randn(batch, 58).astype(np.float32))
w1 = D3D12Tensor.from_numpy(np.random.randn(58, 256).astype(np.float32))
w2 = D3D12Tensor.from_numpy(np.random.randn(256, 128).astype(np.float32))
w3 = D3D12Tensor.from_numpy(np.random.randn(128, 1).astype(np.float32))

def mlp_forward():
    h1 = d3d12_matmul(x_mlp, w1)
    h1 = d3d12_relu(h1)
    h2 = d3d12_matmul(h1, w2)
    h2 = d3d12_relu(h2)
    h3 = d3d12_matmul(h2, w3)
    return h3

bench("MLP 300x58->256->128->1", mlp_forward)

# Numpy comparison
print("\n=== Numpy comparison ===")
x_np_mlp = np.random.randn(batch, 58).astype(np.float32)
w1_np = np.random.randn(58, 256).astype(np.float32)
w2_np = np.random.randn(256, 128).astype(np.float32)
w3_np = np.random.randn(128, 1).astype(np.float32)

def mlp_numpy():
    h1 = np.maximum(x_np_mlp @ w1_np, 0)
    h2 = np.maximum(h1 @ w2_np, 0)
    return h2 @ w3_np

bench("MLP numpy 300x58->256->128->1", mlp_numpy)
