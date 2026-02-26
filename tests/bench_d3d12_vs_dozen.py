"""Benchmark: D3D12 native vs numpy for all critical operations.

Measures per-operation dispatch time across multiple sizes.
Run: python3 bench_d3d12_vs_dozen.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from d3d12_tensor import (
    d3d12_init, D3D12Tensor, d3d12_get_adapter_name,
    d3d12_add, d3d12_sub, d3d12_mul, d3d12_neg, d3d12_scalar_mul,
    d3d12_relu, d3d12_gelu, d3d12_sigmoid,
    d3d12_matmul, d3d12_transpose,
    d3d12_layer_norm, d3d12_softmax,
    d3d12_relu_backward, d3d12_gelu_backward,
    d3d12_softmax_backward, d3d12_layernorm_backward,
)


def _bench(name, fn, warmup=3, repeats=10):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    med = np.median(times)
    return med


def bench_elementwise():
    """Benchmark elementwise operations at various sizes."""
    print("\n## Elementwise Operations\n")
    print(f"| {'Op':<12} | {'Size':>10} | {'D3D12 (ms)':>12} | {'NumPy (ms)':>12} | {'Speedup':>8} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*14}|{'-'*14}|{'-'*10}|")

    ops = [
        ("add", lambda a, b: d3d12_add(a, b), lambda a, b: a + b),
        ("mul", lambda a, b: d3d12_mul(a, b), lambda a, b: a * b),
        ("relu", lambda a, _: d3d12_relu(a), lambda a, _: np.maximum(0, a)),
        ("gelu", lambda a, _: d3d12_gelu(a), lambda a, _: a * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (a + 0.044715 * a**3)))),
        ("sigmoid", lambda a, _: d3d12_sigmoid(a), lambda a, _: 1/(1+np.exp(-a))),
    ]

    for n in [10_000, 100_000, 1_000_000]:
        a_np = np.random.randn(n).astype(np.float32)
        b_np = np.random.randn(n).astype(np.float32)
        a_gpu = D3D12Tensor.from_numpy(a_np)
        b_gpu = D3D12Tensor.from_numpy(b_np)

        for op_name, gpu_fn, np_fn in ops:
            t_gpu = _bench(f"{op_name}_{n}", lambda: gpu_fn(a_gpu, b_gpu))
            t_np = _bench(f"{op_name}_{n}_np", lambda: np_fn(a_np, b_np))
            speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
            print(f"| {op_name:<12} | {n:>10,} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms | {speedup:>6.1f}x |")


def bench_matmul():
    """Benchmark matrix multiplication."""
    print("\n## Matrix Multiplication\n")
    print(f"| {'Size':>20} | {'D3D12 (ms)':>12} | {'NumPy (ms)':>12} | {'Speedup':>8} |")
    print(f"|{'-'*22}|{'-'*14}|{'-'*14}|{'-'*10}|")

    sizes = [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]
    for m, k, n in sizes:
        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)
        a_gpu = D3D12Tensor.from_numpy(a_np)
        b_gpu = D3D12Tensor.from_numpy(b_np)

        t_gpu = _bench(f"matmul_{m}", lambda: d3d12_matmul(a_gpu, b_gpu))
        t_np = _bench(f"matmul_{m}_np", lambda: a_np @ b_np)
        speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
        size_str = f"({m},{k})@({k},{n})"
        print(f"| {size_str:>20} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms | {speedup:>6.1f}x |")


def bench_normalization():
    """Benchmark layer_norm and softmax."""
    print("\n## Normalization / Softmax\n")
    print(f"| {'Op':<15} | {'Shape':>15} | {'D3D12 (ms)':>12} | {'NumPy (ms)':>12} | {'Speedup':>8} |")
    print(f"|{'-'*17}|{'-'*17}|{'-'*14}|{'-'*14}|{'-'*10}|")

    for rows, width in [(16, 256), (32, 512), (64, 1024)]:
        x_np = np.random.randn(rows, width).astype(np.float32)
        gamma_np = np.ones(width, dtype=np.float32)
        beta_np = np.zeros(width, dtype=np.float32)
        x_gpu = D3D12Tensor.from_numpy(x_np)
        gamma_gpu = D3D12Tensor.from_numpy(gamma_np)
        beta_gpu = D3D12Tensor.from_numpy(beta_np)

        def np_ln():
            m = x_np.mean(-1, keepdims=True)
            v = x_np.var(-1, keepdims=True)
            return (x_np - m) / np.sqrt(v + 1e-5) * gamma_np + beta_np

        t_gpu = _bench("ln", lambda: d3d12_layer_norm(x_gpu, gamma_gpu, beta_gpu))
        t_np = _bench("ln_np", np_ln)
        speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
        print(f"| {'layer_norm':<15} | {f'({rows},{width})':>15} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms | {speedup:>6.1f}x |")

        def np_sm():
            mx = x_np.max(-1, keepdims=True)
            e = np.exp(x_np - mx)
            return e / e.sum(-1, keepdims=True)

        t_gpu = _bench("sm", lambda: d3d12_softmax(x_gpu))
        t_np = _bench("sm_np", np_sm)
        speedup = t_np / t_gpu if t_gpu > 0 else float('inf')
        print(f"| {'softmax':<15} | {f'({rows},{width})':>15} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms | {speedup:>6.1f}x |")


def bench_backward():
    """Benchmark backward operations."""
    print("\n## Backward Operations\n")
    print(f"| {'Op':<20} | {'Shape':>10} | {'D3D12 (ms)':>12} | {'NumPy (ms)':>12} |")
    print(f"|{'-'*22}|{'-'*12}|{'-'*14}|{'-'*14}|")

    n = 100_000
    x_np = np.random.randn(n).astype(np.float32)
    g_np = np.random.randn(n).astype(np.float32)
    x_gpu = D3D12Tensor.from_numpy(x_np)
    g_gpu = D3D12Tensor.from_numpy(g_np)

    backs = [
        ("relu_backward", lambda: d3d12_relu_backward(g_gpu, x_gpu),
         lambda: g_np * (x_np > 0).astype(np.float32)),
        ("gelu_backward", lambda: d3d12_gelu_backward(g_gpu, x_gpu),
         lambda: g_np * (1.0 / (1.0 + np.exp(-1.702 * x_np)))),
    ]
    for name, gpu_fn, np_fn in backs:
        t_gpu = _bench(name, gpu_fn)
        t_np = _bench(f"{name}_np", np_fn)
        print(f"| {name:<20} | {n:>10,} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms |")

    # Softmax backward (2D)
    rows, width = 32, 256
    p_np = np.random.dirichlet(np.ones(width), rows).astype(np.float32)
    g2_np = np.random.randn(rows, width).astype(np.float32)
    p_gpu = D3D12Tensor.from_numpy(p_np)
    g2_gpu = D3D12Tensor.from_numpy(g2_np)

    t_gpu = _bench("softmax_bwd", lambda: d3d12_softmax_backward(g2_gpu, p_gpu))
    def np_sm_bwd():
        dot = (g2_np * p_np).sum(-1, keepdims=True)
        return p_np * (g2_np - dot)
    t_np = _bench("softmax_bwd_np", np_sm_bwd)
    print(f"| {'softmax_backward':<20} | {f'{rows}x{width}':>10} | {t_gpu:>10.3f}ms | {t_np:>10.3f}ms |")


def bench_mlp_forward():
    """Benchmark MLP forward pass (58→256→128→1) typical for OperonFold contact head."""
    print("\n## MLP Forward (58→256→128→1, N=300)\n")

    N = 300
    feat = np.random.randn(N, 58).astype(np.float32)
    W0 = np.random.randn(58, 256).astype(np.float32)
    b0 = np.random.randn(N, 256).astype(np.float32)
    W2 = np.random.randn(256, 128).astype(np.float32)
    b2 = np.random.randn(N, 128).astype(np.float32)
    W4 = np.random.randn(128, 1).astype(np.float32)
    b4 = np.random.randn(N, 1).astype(np.float32)

    # GPU tensors
    feat_g = D3D12Tensor.from_numpy(feat)
    W0_g = D3D12Tensor.from_numpy(W0)
    b0_g = D3D12Tensor.from_numpy(b0)
    W2_g = D3D12Tensor.from_numpy(W2)
    b2_g = D3D12Tensor.from_numpy(b2)
    W4_g = D3D12Tensor.from_numpy(W4)
    b4_g = D3D12Tensor.from_numpy(b4)

    from d3d12_tensor import d3d12_matmul_add_relu, d3d12_matmul_add

    def gpu_mlp():
        h0, _ = d3d12_matmul_add_relu(feat_g, W0_g, b0_g)
        h1, _ = d3d12_matmul_add_relu(h0, W2_g, b2_g)
        out = d3d12_matmul_add(h1, W4_g, b4_g)
        return out

    def np_mlp():
        h0 = np.maximum(0, feat @ W0 + b0)
        h1 = np.maximum(0, h0 @ W2 + b2)
        return h1 @ W4 + b4

    t_gpu = _bench("mlp_gpu", gpu_mlp, warmup=5, repeats=20)
    t_np = _bench("mlp_np", np_mlp, warmup=5, repeats=20)

    print(f"| {'Metric':<25} | {'Value':>12} |")
    print(f"|{'-'*27}|{'-'*14}|")
    print(f"| {'D3D12 native':.<25} | {t_gpu:>10.3f}ms |")
    print(f"| {'NumPy (CPU)':.<25} | {t_np:>10.3f}ms |")
    print(f"| {'Speedup vs NumPy':.<25} | {t_np/t_gpu if t_gpu > 0 else 0:>10.1f}x |")
    print(f"| {'Target (was ~90ms Dozen)':.<25} | {'< 5ms':>12} |")


if __name__ == "__main__":
    d3d12_init()

    print("=" * 70)
    print(f"D3D12 Compute Benchmark — {d3d12_get_adapter_name()}")
    print("=" * 70)

    bench_elementwise()
    bench_matmul()
    bench_normalization()
    bench_backward()
    bench_mlp_forward()

    print(f"\n{'=' * 70}")
    print("Benchmark complete")
    print("=" * 70)
