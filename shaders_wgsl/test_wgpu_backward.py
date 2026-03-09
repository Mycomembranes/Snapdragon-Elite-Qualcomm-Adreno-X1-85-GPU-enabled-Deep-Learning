"""Tests for wgpu_backward.py: chain rule backward passes and numerical integration.

Verifies each backward function against numpy finite-difference gradients
and integration functions against numpy/scipy reference implementations.
"""

import numpy as np
import sys

# ============================================================================
# Test Infrastructure
# ============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0


def check_close(name, gpu_result, expected, atol=1e-3, rtol=1e-3):
    """Check that GPU result matches expected within tolerance."""
    global PASS_COUNT, FAIL_COUNT
    gpu_np = gpu_result.numpy() if hasattr(gpu_result, 'numpy') else np.asarray(gpu_result)
    expected = np.asarray(expected, dtype=np.float32)

    if gpu_np.shape != expected.shape:
        print(f"  FAIL {name}: shape mismatch {gpu_np.shape} vs {expected.shape}")
        FAIL_COUNT += 1
        return False

    max_diff = np.max(np.abs(gpu_np - expected))
    if np.allclose(gpu_np, expected, atol=atol, rtol=rtol):
        print(f"  PASS {name} (max_diff={max_diff:.2e})")
        PASS_COUNT += 1
        return True
    else:
        print(f"  FAIL {name} (max_diff={max_diff:.2e})")
        print(f"    GPU:      {gpu_np.ravel()[:8]}")
        print(f"    Expected: {expected.ravel()[:8]}")
        FAIL_COUNT += 1
        return False


def numerical_gradient(f, x, eps=1e-4):
    """Compute numerical gradient via central finite differences.

    Args:
        f: function that takes numpy array and returns scalar numpy value
        x: numpy array, point at which to compute gradient
        eps: finite difference step size

    Returns:
        grad: numpy array, same shape as x
    """
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        grad[idx] = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


# ============================================================================
# Import GPU functions
# ============================================================================

try:
    from wgpu_tensor import (
        WgpuTensor, add, sub, mul, neg, scalar_mul, matmul,
        transpose_2d, relu, gelu, sigmoid, tanh_act,
        softmax, layer_norm, cross_entropy, focal_bce,
        sum_reduce, mean_reduce, embedding_lookup,
        matmul_add_relu, matmul_add,
        relu_backward as wt_relu_backward,
        gelu_backward as wt_gelu_backward,
    )
    from wgpu_backward import (
        sigmoid_backward, tanh_backward,
        matmul_backward, add_backward, sub_backward,
        mul_backward, scalar_mul_backward, neg_backward,
        transpose_backward, matmul_add_backward,
        matmul_add_relu_backward, cross_entropy_backward,
        focal_bce_backward, sum_reduce_backward,
        mean_reduce_backward, layer_norm_backward_full,
        embedding_backward,
        # Integration
        trapezoid, simpson, cumulative_trapezoid, trapezoid_2d,
        # Re-exported
        relu_backward, gelu_backward, softmax_backward,
        layernorm_backward,
    )
    GPU_AVAILABLE = True
except Exception as e:
    print(f"GPU not available: {e}")
    GPU_AVAILABLE = False


def run_tests():
    if not GPU_AVAILABLE:
        print("Skipping tests -- GPU not available")
        return

    np.random.seed(42)

    print("=" * 60)
    print("Testing Chain Rule Backward Passes")
    print("=" * 60)

    # ------------------------------------------------------------------
    print("\n--- sigmoid_backward ---")
    x_np = np.random.randn(4, 8).astype(np.float32)
    sig_np = 1.0 / (1.0 + np.exp(-x_np))
    grad_np = np.random.randn(4, 8).astype(np.float32)
    expected = grad_np * sig_np * (1.0 - sig_np)

    grad_gpu = WgpuTensor.from_numpy(grad_np)
    sig_gpu = WgpuTensor.from_numpy(sig_np)
    result = sigmoid_backward(grad_gpu, sig_gpu)
    check_close("sigmoid_backward", result, expected)

    # Verify with numerical gradient
    def sigmoid_scalar(x):
        return np.sum(1.0 / (1.0 + np.exp(-x)))
    num_grad = numerical_gradient(sigmoid_scalar, x_np[:1, :2], eps=1e-4)
    ones_grad = WgpuTensor.from_numpy(np.ones_like(x_np[:1, :2]).astype(np.float32))
    sig_sub = WgpuTensor.from_numpy(sig_np[:1, :2])
    analytic = sigmoid_backward(ones_grad, sig_sub)
    check_close("sigmoid_backward (vs finite diff)", analytic, num_grad, atol=1e-2)

    # ------------------------------------------------------------------
    print("\n--- tanh_backward ---")
    tanh_np = np.tanh(x_np)
    expected_tanh = grad_np * (1.0 - tanh_np ** 2)

    tanh_gpu = WgpuTensor.from_numpy(tanh_np)
    result = tanh_backward(grad_gpu, tanh_gpu)
    check_close("tanh_backward", result, expected_tanh)

    # ------------------------------------------------------------------
    print("\n--- add_backward ---")
    ga, gb = add_backward(grad_gpu)
    check_close("add_backward grad_a", ga, grad_np)
    check_close("add_backward grad_b", gb, grad_np)

    # ------------------------------------------------------------------
    print("\n--- sub_backward ---")
    ga, gb = sub_backward(grad_gpu)
    check_close("sub_backward grad_a", ga, grad_np)
    check_close("sub_backward grad_b", gb, -grad_np)

    # ------------------------------------------------------------------
    print("\n--- mul_backward ---")
    a_np = np.random.randn(4, 8).astype(np.float32)
    b_np = np.random.randn(4, 8).astype(np.float32)
    a_gpu = WgpuTensor.from_numpy(a_np)
    b_gpu = WgpuTensor.from_numpy(b_np)
    ga, gb = mul_backward(grad_gpu, a_gpu, b_gpu)
    check_close("mul_backward grad_a", ga, grad_np * b_np)
    check_close("mul_backward grad_b", gb, grad_np * a_np)

    # ------------------------------------------------------------------
    print("\n--- scalar_mul_backward ---")
    s = 3.14
    result = scalar_mul_backward(grad_gpu, s)
    check_close("scalar_mul_backward", result, grad_np * s)

    # ------------------------------------------------------------------
    print("\n--- neg_backward ---")
    result = neg_backward(grad_gpu)
    check_close("neg_backward", result, -grad_np)

    # ------------------------------------------------------------------
    print("\n--- transpose_backward ---")
    grad_2d = np.random.randn(3, 5).astype(np.float32)
    grad_2d_gpu = WgpuTensor.from_numpy(grad_2d)
    result = transpose_backward(grad_2d_gpu)
    check_close("transpose_backward", result, grad_2d.T)

    # ------------------------------------------------------------------
    print("\n--- matmul_backward ---")
    M, K, N = 4, 6, 8
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)
    grad_out_np = np.random.randn(M, N).astype(np.float32)

    A_gpu = WgpuTensor.from_numpy(A_np)
    B_gpu = WgpuTensor.from_numpy(B_np)
    grad_out_mm = WgpuTensor.from_numpy(grad_out_np)

    ga, gb = matmul_backward(grad_out_mm, A_gpu, B_gpu)
    expected_ga = grad_out_np @ B_np.T
    expected_gb = A_np.T @ grad_out_np
    check_close("matmul_backward grad_a", ga, expected_ga)
    check_close("matmul_backward grad_b", gb, expected_gb)

    # ------------------------------------------------------------------
    print("\n--- matmul_add_backward ---")
    gi, gw, gbias = matmul_add_backward(grad_out_mm, A_gpu, B_gpu)
    check_close("matmul_add_backward grad_input", gi, expected_ga)
    check_close("matmul_add_backward grad_weight", gw, expected_gb)
    check_close("matmul_add_backward grad_bias", gbias, grad_out_np)

    # ------------------------------------------------------------------
    print("\n--- matmul_add_relu_backward ---")
    bias_np = np.random.randn(M, N).astype(np.float32)
    pre_relu_np = (A_np @ B_np + bias_np).astype(np.float32)
    relu_mask = (pre_relu_np > 0).astype(np.float32)
    grad_after_relu = grad_out_np * relu_mask

    pre_relu_gpu = WgpuTensor.from_numpy(pre_relu_np)
    gi, gw, gb = matmul_add_relu_backward(grad_out_mm, pre_relu_gpu, A_gpu, B_gpu)

    expected_gi = grad_after_relu @ B_np.T
    expected_gw = A_np.T @ grad_after_relu
    check_close("matmul_add_relu_backward grad_input", gi, expected_gi)
    check_close("matmul_add_relu_backward grad_weight", gw, expected_gw)
    check_close("matmul_add_relu_backward grad_bias", gb, grad_after_relu)

    # ------------------------------------------------------------------
    print("\n--- cross_entropy_backward ---")
    logits_np = np.random.randn(16).astype(np.float32)
    targets_np = np.random.randint(0, 2, 16).astype(np.float32)
    logits_gpu = WgpuTensor.from_numpy(logits_np)
    targets_gpu = WgpuTensor.from_numpy(targets_np)

    result = cross_entropy_backward(logits_gpu, targets_gpu)
    p = 1.0 / (1.0 + np.exp(-logits_np))
    expected_ce = (p - targets_np).astype(np.float32)
    check_close("cross_entropy_backward", result, expected_ce)

    # ------------------------------------------------------------------
    print("\n--- focal_bce_backward ---")
    gamma, alpha = 2.0, 0.25
    result = focal_bce_backward(logits_gpu, targets_gpu, gamma, alpha)
    # Compute expected on CPU
    p = 1.0 / (1.0 + np.exp(-logits_np))
    ce = -targets_np * np.log(p + 1e-6) - (1 - targets_np) * np.log(1 - p + 1e-6)
    p_t = np.where(targets_np > 0.5, p, 1 - p)
    dp_t_dz = np.where(targets_np > 0.5, p * (1 - p), -p * (1 - p))
    one_minus_pt = 1 - p_t
    focal_weight = one_minus_pt ** gamma
    focal_weight_deriv = -gamma * one_minus_pt ** (gamma - 1) * dp_t_dz
    dce_dz = p - targets_np
    expected_focal = alpha * (1 - alpha) * (focal_weight_deriv * ce + focal_weight * dce_dz)
    check_close("focal_bce_backward", result, expected_focal.astype(np.float32), atol=1e-2)

    # ------------------------------------------------------------------
    print("\n--- sum_reduce_backward ---")
    grad_scalar = WgpuTensor.from_numpy(np.array([2.5], dtype=np.float32))
    result = sum_reduce_backward(grad_scalar, (3, 4))
    expected_sum = np.full((3, 4), 2.5, dtype=np.float32)
    check_close("sum_reduce_backward", result, expected_sum)

    # ------------------------------------------------------------------
    print("\n--- mean_reduce_backward ---")
    result = mean_reduce_backward(grad_scalar, (3, 4))
    expected_mean = np.full((3, 4), 2.5 / 12.0, dtype=np.float32)
    check_close("mean_reduce_backward", result, expected_mean)

    # ------------------------------------------------------------------
    print("\n--- layer_norm_backward_full ---")
    D = 32
    x_ln = np.random.randn(2, D).astype(np.float32)
    gamma_ln = np.random.randn(D).astype(np.float32) * 0.5 + 1.0
    beta_ln = np.random.randn(D).astype(np.float32) * 0.1
    grad_ln = np.random.randn(2, D).astype(np.float32)

    x_gpu = WgpuTensor.from_numpy(x_ln)
    gamma_gpu = WgpuTensor.from_numpy(gamma_ln.astype(np.float32))
    beta_gpu = WgpuTensor.from_numpy(beta_ln.astype(np.float32))
    grad_ln_gpu = WgpuTensor.from_numpy(grad_ln)

    gi, gg, gb = layer_norm_backward_full(grad_ln_gpu, x_gpu, gamma_gpu)

    # CPU reference for grad_gamma and grad_beta
    mean_ln = x_ln.mean(axis=-1, keepdims=True)
    var_ln = x_ln.var(axis=-1, keepdims=True)
    x_hat = (x_ln - mean_ln) / np.sqrt(var_ln + 1e-5)
    expected_gg = (grad_ln * x_hat).sum(axis=0)
    expected_gb = grad_ln.sum(axis=0)
    check_close("layer_norm_backward_full grad_gamma", gg, expected_gg, atol=1e-2)
    check_close("layer_norm_backward_full grad_beta", gb, expected_gb, atol=1e-2)

    # ------------------------------------------------------------------
    print("\n--- embedding_backward ---")
    vocab_size, emb_dim = 10, 4
    indices_np = np.array([2, 5, 2, 8], dtype=np.uint32)  # note: index 2 appears twice
    grad_emb_np = np.random.randn(4, emb_dim).astype(np.float32)
    indices_gpu = WgpuTensor.from_numpy(indices_np)
    grad_emb_gpu = WgpuTensor.from_numpy(grad_emb_np)

    result = embedding_backward(grad_emb_gpu, indices_gpu, vocab_size, emb_dim)
    expected_gw = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    np.add.at(expected_gw, indices_np.astype(np.int64), grad_emb_np)
    check_close("embedding_backward", result, expected_gw)

    # ------------------------------------------------------------------
    print("\n--- relu_backward (re-exported) ---")
    x_relu = np.random.randn(8).astype(np.float32)
    grad_relu_np = np.random.randn(8).astype(np.float32)
    x_relu_gpu = WgpuTensor.from_numpy(x_relu)
    grad_relu_gpu = WgpuTensor.from_numpy(grad_relu_np)
    result = relu_backward(grad_relu_gpu, x_relu_gpu)
    expected_relu = grad_relu_np * (x_relu > 0)
    check_close("relu_backward (re-export)", result, expected_relu)

    # ------------------------------------------------------------------
    print("\n--- gelu_backward (re-exported) ---")
    result = gelu_backward(grad_relu_gpu, x_relu_gpu)
    # Approximate: gelu_deriv(x) ~ sigmoid(1.702 * x)
    gelu_d = 1.0 / (1.0 + np.exp(-1.702 * x_relu))
    expected_gelu = grad_relu_np * gelu_d
    check_close("gelu_backward (re-export)", result, expected_gelu, atol=1e-2)

    # ============================================================
    print("\n" + "=" * 60)
    print("Testing Chain Rule Composition")
    print("=" * 60)

    print("\n--- Verify: relu(x @ W + b) fused vs composed backward ---")
    M, K, N = 4, 6, 8
    x_comp = np.random.randn(M, K).astype(np.float32)
    w_comp = np.random.randn(K, N).astype(np.float32)
    b_comp = np.random.randn(M, N).astype(np.float32)
    grad_comp = np.random.randn(M, N).astype(np.float32)

    # Compute pre_relu on CPU for reference
    pre_relu_comp = (x_comp @ w_comp + b_comp).astype(np.float32)

    x_gpu = WgpuTensor.from_numpy(x_comp)
    w_gpu = WgpuTensor.from_numpy(w_comp)
    pre_gpu = WgpuTensor.from_numpy(pre_relu_comp)
    grad_comp_gpu = WgpuTensor.from_numpy(grad_comp)

    # Fused backward
    gi_fused, gw_fused, gb_fused = matmul_add_relu_backward(
        grad_comp_gpu, pre_gpu, x_gpu, w_gpu
    )

    # Manual composition: relu_backward -> matmul_add_backward
    grad_relu_manual = relu_backward(grad_comp_gpu, pre_gpu)
    gi_manual, gw_manual, gb_manual = matmul_add_backward(
        grad_relu_manual, x_gpu, w_gpu
    )

    check_close("composed vs fused: grad_input", gi_fused, gi_manual.numpy())
    check_close("composed vs fused: grad_weight", gw_fused, gw_manual.numpy())
    check_close("composed vs fused: grad_bias", gb_fused, gb_manual.numpy())

    # ============================================================
    print("\n" + "=" * 60)
    print("Testing Numerical Integration")
    print("=" * 60)

    # ------------------------------------------------------------------
    print("\n--- trapezoid ---")
    # Integrate sin(x) from 0 to pi -> expected = 2.0
    n_pts = 1001
    x_trap = np.linspace(0, np.pi, n_pts, dtype=np.float32)
    y_trap = np.sin(x_trap).astype(np.float32)
    dx_trap = float(x_trap[1] - x_trap[0])

    y_gpu = WgpuTensor.from_numpy(y_trap)
    result = trapezoid(y_gpu, dx=dx_trap)
    _np_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    np_result = _np_trapz(y_trap, dx=dx_trap)
    check_close("trapezoid sin(x) [0,pi]", result, np.array([np_result], dtype=np.float32), atol=1e-3)

    # Test with x^2 from 0 to 1 -> expected = 1/3
    x_sq = np.linspace(0, 1, 101, dtype=np.float32)
    y_sq = (x_sq ** 2).astype(np.float32)
    dx_sq = float(x_sq[1] - x_sq[0])
    y_sq_gpu = WgpuTensor.from_numpy(y_sq)
    result = trapezoid(y_sq_gpu, dx=dx_sq)
    np_result = _np_trapz(y_sq, dx=dx_sq)
    check_close("trapezoid x^2 [0,1]", result, np.array([np_result], dtype=np.float32), atol=1e-3)

    # ------------------------------------------------------------------
    print("\n--- simpson ---")
    # Integrate sin(x) from 0 to pi -> expected = 2.0
    y_simp_gpu = WgpuTensor.from_numpy(y_trap)
    result = simpson(y_simp_gpu, dx=dx_trap)
    # scipy reference
    try:
        from scipy.integrate import simpson as scipy_simpson
        scipy_result = scipy_simpson(y_trap, dx=dx_trap)
        check_close("simpson sin(x) [0,pi]", result,
                     np.array([scipy_result], dtype=np.float32), atol=1e-3)
    except ImportError:
        # Fallback: exact answer is 2.0
        check_close("simpson sin(x) [0,pi]", result,
                     np.array([2.0], dtype=np.float32), atol=1e-3)

    # ------------------------------------------------------------------
    print("\n--- cumulative_trapezoid ---")
    # Cumulative integral of x from 0 to 4 -> expected = x^2/2
    x_cum = np.linspace(0, 4, 5, dtype=np.float32)  # [0, 1, 2, 3, 4]
    y_cum = x_cum.copy()  # f(x) = x
    dx_cum = 1.0
    y_cum_gpu = WgpuTensor.from_numpy(y_cum)
    result = cumulative_trapezoid(y_cum_gpu, dx=dx_cum)
    # Expected: [0.5, 2.0, 4.5, 8.0] (cumulative sum of trapezoid panels)
    expected_cum = np.array([0.5, 2.0, 4.5, 8.0], dtype=np.float32)
    check_close("cumulative_trapezoid f(x)=x", result, expected_cum)

    # 2D test
    y_2d = np.stack([y_cum, y_cum * 2], axis=0).astype(np.float32)  # (2, 5)
    y_2d_gpu = WgpuTensor.from_numpy(y_2d)
    result_2d = cumulative_trapezoid(y_2d_gpu, dx=dx_cum)
    expected_2d = np.stack([expected_cum, expected_cum * 2], axis=0)
    check_close("cumulative_trapezoid 2D", result_2d, expected_2d)

    # ------------------------------------------------------------------
    print("\n--- trapezoid_2d ---")
    # 2D: integrate each row of sin values
    n_rows = 3
    y_2d_trap = np.tile(y_trap, (n_rows, 1)).astype(np.float32)
    y_2d_gpu = WgpuTensor.from_numpy(y_2d_trap)
    result = trapezoid_2d(y_2d_gpu, dx=dx_trap)
    np_row_result = _np_trapz(y_trap, dx=dx_trap)
    expected_2d = np.full(n_rows, np_row_result, dtype=np.float32)
    check_close("trapezoid_2d", result, expected_2d, atol=1e-3)

    # ============================================================
    print("\n" + "=" * 60)
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 60)
    return FAIL_COUNT == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
