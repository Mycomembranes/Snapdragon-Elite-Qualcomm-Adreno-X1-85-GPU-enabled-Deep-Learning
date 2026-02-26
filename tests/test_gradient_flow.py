#!/usr/bin/env python3
"""Gradient flow verification test for OperonFold.

Tests that the backward pass propagates gradients to ALL 1.23M parameters,
not just the MLM head (~36K). Uses CPU-only mock of wgpu_tensor.

Usage:
  python test_gradient_flow.py
"""

import sys
import os
import numpy as np

# ============================================================================
# Mock wgpu_tensor module (CPU fallback for testing without GPU)
# ============================================================================

class MockWgpuTensor:
    """CPU-based mock of WgpuTensor for gradient flow testing."""

    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self._data = data.astype(np.float32)
        else:
            self._data = np.array(data, dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr):
        return cls(arr)

    def numpy(self):
        return self._data.copy()

    @property
    def shape(self):
        return self._data.shape

    def reshape(self, *args):
        return MockWgpuTensor(self._data.reshape(*args))

    def expand(self, *shape):
        return MockWgpuTensor(np.broadcast_to(self._data, shape).copy())


def mock_matmul(a, b):
    return MockWgpuTensor(a.numpy() @ b.numpy())

def mock_softmax(x):
    d = x.numpy()
    e = np.exp(d - np.max(d, axis=-1, keepdims=True))
    return MockWgpuTensor(e / np.sum(e, axis=-1, keepdims=True))

def mock_layer_norm(x, gamma, beta, eps):
    d = x.numpy()
    g = gamma.numpy()
    b = beta.numpy()
    mean = d.mean(axis=-1, keepdims=True)
    var = d.var(axis=-1, keepdims=True)
    xn = (d - mean) / np.sqrt(var + eps)
    return MockWgpuTensor(xn * g + b)

def mock_gelu(x):
    d = x.numpy()
    return MockWgpuTensor(0.5 * d * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (d + 0.044715 * d**3))))

def mock_sigmoid(x):
    d = x.numpy()
    return MockWgpuTensor(1.0 / (1.0 + np.exp(-d)))

def mock_embedding_lookup(weight, indices):
    w = weight.numpy()
    idx = indices.numpy().astype(int)
    return MockWgpuTensor(w[idx])

def mock_add(a, b):
    return MockWgpuTensor(a.numpy() + b.numpy())

def mock_scalar_mul(x, s):
    return MockWgpuTensor(x.numpy() * s)

def mock_sub(a, b):
    return MockWgpuTensor(a.numpy() - b.numpy())

def mock_neg(x):
    return MockWgpuTensor(-x.numpy())

def mock_transpose_2d(x):
    return MockWgpuTensor(x.numpy().T)

def mock_gelu_backward(grad_out, x):
    """GELU backward: d/dx GELU(x) ≈ sigmoid(1.702 * x)."""
    d = x.numpy()
    g = grad_out.numpy()
    deriv = 1.0 / (1.0 + np.exp(-1.702 * d))
    return MockWgpuTensor(g * deriv)

def mock_softmax_backward(grad_out, softmax_out):
    """Softmax backward: grad_input = softmax * (grad - sum(grad * softmax))."""
    s = softmax_out.numpy()
    g = grad_out.numpy()
    dot = np.sum(g * s, axis=-1, keepdims=True)
    return MockWgpuTensor(s * (g - dot))

def mock_layernorm_backward(grad_out, x, gamma, eps):
    """LayerNorm backward: returns gradient w.r.t. input."""
    d = x.numpy()
    g = grad_out.numpy()
    gam = gamma.numpy()
    D = d.shape[-1]
    mean = d.mean(axis=-1, keepdims=True)
    var = d.var(axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (d - mean) * std_inv
    dx_hat = g * gam
    dvar = (dx_hat * (d - mean) * (-0.5) * std_inv**3).sum(axis=-1, keepdims=True)
    dmean = (-dx_hat * std_inv).sum(axis=-1, keepdims=True) + dvar * (-2.0 / D) * (d - mean).sum(axis=-1, keepdims=True)
    dx = dx_hat * std_inv + dvar * 2.0 * (d - mean) / D + dmean / D
    return MockWgpuTensor(dx)


# Inject mock module before importing the model
import types
mock_module = types.ModuleType('wgpu_tensor')
mock_module.WgpuTensor = MockWgpuTensor
mock_module.matmul = mock_matmul
mock_module.softmax = mock_softmax
mock_module.layer_norm = mock_layer_norm
mock_module.gelu = mock_gelu
mock_module.sigmoid = mock_sigmoid
mock_module.embedding_lookup = mock_embedding_lookup
mock_module.add = mock_add
mock_module.scalar_mul = mock_scalar_mul
mock_module.sub = mock_sub
mock_module.neg = mock_neg
mock_module.transpose_2d = mock_transpose_2d
mock_module.gelu_backward = mock_gelu_backward
mock_module.softmax_backward = mock_softmax_backward
mock_module.layernorm_backward = mock_layernorm_backward
sys.modules['wgpu_tensor'] = mock_module

# Now import the model
sys.path.insert(0, os.path.dirname(__file__))
from operonfold_wgpu_model import CoevolutionTransformerWGPU, softmax_numpy


def test_phase1_gradient_flow():
    """Test that Phase 1 MLM backward pass updates ALL parameters."""
    print("=" * 70)
    print("TEST 1: Phase 1 MLM Gradient Flow")
    print("=" * 70)

    np.random.seed(42)

    # Small model for fast testing
    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    total_params = model.count_parameters()
    print(f"Model parameters: {total_params:,}")

    # Create small batch
    B, L = 2, 16
    token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
    token_ids[:, 0] = 23   # CLS
    token_ids[:, 7] = 24   # SEP
    token_ids[:, -1] = 24  # SEP
    segment_ids = np.zeros((B, L), dtype=np.int64)
    segment_ids[:, 8:] = 1

    # Apply masking
    masked_ids = token_ids.copy()
    mask_pos = np.zeros((B, L), dtype=bool)
    mask_pos[:, 2:5] = True  # Mask positions 2-4
    masked_ids[mask_pos] = 21  # MASK token
    labels = token_ids.copy()

    # Zero all gradients
    for key in model.grads:
        model.grads[key] = np.zeros_like(model.grads[key])

    # Forward
    print("\nRunning forward pass...")
    logits, cache = model.forward_mlm(masked_ids, segment_ids)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Cache keys: {sorted(cache.keys())}")

    # Compute loss
    V = 25
    probs = softmax_numpy(logits)
    targets_onehot = np.eye(V, dtype=np.float32)[labels]
    grad_logits = (probs - targets_onehot)
    grad_logits *= mask_pos[:, :, np.newaxis].astype(np.float32)
    n_masked = mask_pos.sum()
    if n_masked > 0:
        grad_logits /= n_masked

    # Loss value
    log_probs = np.log(probs + 1e-10)
    loss = -np.sum(log_probs * targets_onehot * mask_pos[:, :, np.newaxis]) / max(n_masked, 1)
    print(f"  MLM Loss: {loss:.4f}")

    # Backward through MLM head
    print("\nRunning backward pass through MLM head...")
    final_out = cache["final_output"]
    D = model.d_model

    # MLM head chain: final_out → dense → GELU → LayerNorm(norm) → proj → logits
    # Step 1: proj backward
    model.grads["mlm_head.proj.bias"] = grad_logits.sum(axis=(0, 1))
    dense_pre = final_out @ model.params["mlm_head.dense.weight"].T + model.params["mlm_head.dense.bias"]
    gelu_out = model._gelu_np(dense_pre)
    normed = model._layernorm_np(gelu_out, "mlm_head.norm")
    gl_flat = grad_logits.reshape(-1, V)
    normed_flat = normed.reshape(-1, D)
    fo_flat = final_out.reshape(-1, D)
    model.grads["mlm_head.proj.weight"] = gl_flat.T @ normed_flat

    # Step 2: grad through proj → LayerNorm backward
    proj_w = model.params["mlm_head.proj.weight"]
    grad_normed = grad_logits @ proj_w  # (B, L, D)
    grad_gelu_out = model._layernorm_backward_np(grad_normed, gelu_out, "mlm_head.norm")

    # Step 3: GELU backward
    gelu_deriv = model._gelu_deriv_np(dense_pre)
    grad_dense_pre = grad_gelu_out * gelu_deriv

    # Step 4: dense backward
    gd_flat = grad_dense_pre.reshape(-1, D)
    model.grads["mlm_head.dense.weight"] = gd_flat.T @ fo_flat
    model.grads["mlm_head.dense.bias"] = grad_dense_pre.sum(axis=(0, 1))
    grad_dense = (gd_flat @ model.params["mlm_head.dense.weight"]).reshape(B, L, D)

    # Backward through final_norm
    pre_fn = cache.get("pre_final_norm", final_out)
    grad_pre_norm = model._layernorm_backward_np(grad_dense, pre_fn, "final_norm")

    # CRITICAL: Backward through ALL transformer layers
    print("Running backward pass through transformer layers...")
    model.backward_transformer(grad_pre_norm, cache)

    # Check gradient flow
    print("\n" + "-" * 70)
    print("GRADIENT FLOW CHECK")
    print("-" * 70)

    zero_params = []
    nonzero_params = []
    total_grad_sum = 0.0

    for key in sorted(model.grads.keys()):
        grad = model.grads[key]
        grad_abs_sum = np.abs(grad).sum()
        total_grad_sum += grad_abs_sum
        is_zero = grad_abs_sum == 0.0

        if is_zero:
            zero_params.append(key)
        else:
            nonzero_params.append(key)

    # Report by layer
    for layer_idx in range(6):
        layer_keys = [k for k in sorted(model.grads.keys()) if f"layers.{layer_idx}." in k]
        layer_grads = [np.abs(model.grads[k]).sum() for k in layer_keys]
        all_nonzero = all(g > 0 for g in layer_grads)
        status = "OK" if all_nonzero else "FAIL"
        print(f"  Layer {layer_idx}: {status} (grad_sum={sum(layer_grads):.6f})")
        if not all_nonzero:
            for k, g in zip(layer_keys, layer_grads):
                if g == 0:
                    print(f"    ZERO: {k}")

    # MLM head
    mlm_keys = [k for k in sorted(model.grads.keys()) if "mlm_head" in k]
    mlm_grads = [np.abs(model.grads[k]).sum() for k in mlm_keys]
    print(f"  MLM Head: {'OK' if all(g > 0 for g in mlm_grads) else 'FAIL'} (grad_sum={sum(mlm_grads):.6f})")

    # Final norm
    fn_keys = [k for k in sorted(model.grads.keys()) if "final_norm" in k]
    fn_grads = [np.abs(model.grads[k]).sum() for k in fn_keys]
    print(f"  Final Norm: {'OK' if all(g > 0 for g in fn_grads) else 'FAIL'} (grad_sum={sum(fn_grads):.6f})")

    # Embed norm
    en_keys = [k for k in sorted(model.grads.keys()) if "embed_norm" in k]
    en_grads = [np.abs(model.grads[k]).sum() for k in en_keys]
    print(f"  Embed Norm: {'OK' if all(g > 0 for g in en_grads) else 'FAIL'} (grad_sum={sum(en_grads):.6f})")

    # Embeddings
    emb_keys = [k for k in sorted(model.grads.keys()) if "embedding" in k]
    emb_grads = [np.abs(model.grads[k]).sum() for k in emb_keys]
    print(f"  Embeddings: {'OK' if all(g > 0 for g in emb_grads) else 'FAIL'} (grad_sum={sum(emb_grads):.6f})")

    # Contact head (not used in Phase 1)
    contact_keys = [k for k in sorted(model.grads.keys()) if "contact_head" in k]

    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Parameters with gradients: {len(nonzero_params)}")
    print(f"  Parameters with zero grad: {len(zero_params)}")

    # Count updated params (excluding embeddings and contact head)
    trainable_nonzero = [k for k in nonzero_params if "embedding" not in k and "contact_head" not in k]
    trainable_zero = [k for k in zero_params if "embedding" not in k and "contact_head" not in k]

    print(f"\n  Trainable (excl embeddings+contact) with grad: {len(trainable_nonzero)}")
    print(f"  Trainable (excl embeddings+contact) with zero: {len(trainable_zero)}")

    if trainable_zero:
        print(f"\n  ZERO GRADIENT params:")
        for k in trainable_zero:
            print(f"    {k}")

    # NaN/Inf check
    has_nan = False
    for key in model.grads:
        if not np.isfinite(model.grads[key]).all():
            print(f"  WARNING: NaN/Inf in gradient for {key}")
            has_nan = True

    print("\n" + "=" * 70)
    if len(trainable_zero) == 0 and not has_nan:
        print("PHASE 1 GRADIENT FLOW: PASS")
        print(f"  All transformer layers receive gradients!")
        return True
    else:
        print("PHASE 1 GRADIENT FLOW: FAIL")
        return False


def test_phase2_gradient_flow():
    """Test that Phase 2 contact backward pass updates all parameters."""
    print("\n" + "=" * 70)
    print("TEST 2: Phase 2 Contact Gradient Flow")
    print("=" * 70)

    np.random.seed(123)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    # Create paired sequence input
    B, L = 1, 20  # Single pair, short
    l_a, l_b = 8, 8

    token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
    token_ids[:, 0] = 23   # CLS
    token_ids[:, l_a] = 24  # SEP
    token_ids[:, l_a + l_b + 1] = 24  # SEP
    segment_ids = np.zeros((B, L), dtype=np.int64)
    segment_ids[:, l_a + 1:] = 1

    # Zero all gradients
    for key in model.grads:
        model.grads[key] = np.zeros_like(model.grads[key])

    # Forward contact prediction
    print("\nRunning contact forward pass...")
    contact_logits, cache = model.forward_contact(token_ids, segment_ids, l_a, l_b)
    print(f"  Contact logits shape: {contact_logits.shape}")

    # Create random target contact map
    target_contacts = (np.random.rand(l_a, l_b) > 0.8).astype(np.float32)

    # BCE loss
    logits_flat = contact_logits.reshape(-1)
    sigmoid_pred = 1.0 / (1.0 + np.exp(-np.clip(logits_flat, -20, 20)))
    targets_flat = target_contacts.reshape(-1)
    bce_loss = -np.mean(
        targets_flat * np.log(sigmoid_pred + 1e-7) +
        (1 - targets_flat) * np.log(1 - sigmoid_pred + 1e-7)
    )
    print(f"  BCE Loss: {bce_loss:.4f}")

    # Backward
    print("\nRunning contact backward pass...")
    model.backward_contact(contact_logits, target_contacts, cache)

    # Check gradient flow
    print("\n" + "-" * 70)
    print("CONTACT GRADIENT FLOW CHECK")
    print("-" * 70)

    zero_params = []
    nonzero_params = []

    for key in sorted(model.grads.keys()):
        grad = model.grads[key]
        grad_abs_sum = np.abs(grad).sum()
        if grad_abs_sum == 0.0:
            zero_params.append(key)
        else:
            nonzero_params.append(key)

    # Check contact head specifically
    contact_keys = [k for k in sorted(model.grads.keys()) if "contact_head" in k]
    for k in contact_keys:
        g = np.abs(model.grads[k]).sum()
        status = "OK" if g > 0 else "ZERO"
        print(f"  {k}: {status} (grad_sum={g:.6f})")

    # Check transformer layers
    for layer_idx in range(6):
        layer_keys = [k for k in sorted(model.grads.keys()) if f"layers.{layer_idx}." in k]
        layer_grads = [np.abs(model.grads[k]).sum() for k in layer_keys]
        all_nonzero = all(g > 0 for g in layer_grads)
        print(f"  Layer {layer_idx}: {'OK' if all_nonzero else 'FAIL'} (grad_sum={sum(layer_grads):.6f})")

    # NaN/Inf check
    has_nan = any(not np.isfinite(model.grads[k]).all() for k in model.grads)

    # In Phase 2, MLM head params are expected to have zero grad (unused)
    trainable_zero = [k for k in zero_params
                      if "embedding" not in k and "mlm_head" not in k]
    print(f"\n  Non-embedding/non-MLM params with zero grad: {len(trainable_zero)}")
    if trainable_zero:
        for k in trainable_zero:
            print(f"    {k}")

    print("\n" + "=" * 70)
    if len(trainable_zero) == 0 and not has_nan:
        print("PHASE 2 GRADIENT FLOW: PASS")
        return True
    else:
        print("PHASE 2 GRADIENT FLOW: FAIL")
        return False


def test_gradient_accumulation():
    """Test that gradient accumulation works correctly."""
    print("\n" + "=" * 70)
    print("TEST 3: Gradient Accumulation")
    print("=" * 70)

    np.random.seed(99)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    B, L, V = 2, 12, 25
    accum_steps = 4

    # Zero grads
    for key in model.grads:
        model.grads[key] = np.zeros_like(model.grads[key])

    # Accumulate over multiple mini-batches
    for step in range(accum_steps):
        token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
        token_ids[:, 0] = 23
        token_ids[:, -1] = 24
        segment_ids = np.zeros((B, L), dtype=np.int64)

        masked_ids = token_ids.copy()
        mask_pos = np.zeros((B, L), dtype=bool)
        mask_pos[:, 2:4] = True
        masked_ids[mask_pos] = 21
        labels = token_ids.copy()

        logits, cache = model.forward_mlm(masked_ids, segment_ids)

        probs = softmax_numpy(logits)
        targets_onehot = np.eye(V, dtype=np.float32)[labels]
        grad_logits = (probs - targets_onehot) * mask_pos[:, :, np.newaxis].astype(np.float32)
        n_masked = mask_pos.sum()
        if n_masked > 0:
            grad_logits /= n_masked

        D = model.d_model
        final_out = cache["final_output"]
        model.grads["mlm_head.proj.bias"] += grad_logits.sum(axis=(0, 1))
        gl_flat = grad_logits.reshape(-1, V)
        fo_flat = final_out.reshape(-1, D)
        model.grads["mlm_head.proj.weight"] += gl_flat.T @ fo_flat

        dense_w = model.params["mlm_head.dense.weight"]
        dense_b = model.params["mlm_head.dense.bias"]
        dense_pre = final_out @ dense_w.T + dense_b
        gelu_deriv = 1.0 / (1.0 + np.exp(-1.702 * dense_pre))
        proj_w = model.params["mlm_head.proj.weight"]
        grad_dense = grad_logits @ proj_w
        grad_dense *= gelu_deriv
        gd_flat = grad_dense.reshape(-1, D)
        model.grads["mlm_head.dense.weight"] += gd_flat.T @ fo_flat
        model.grads["mlm_head.dense.bias"] += grad_dense.sum(axis=(0, 1))

        pre_fn = cache.get("pre_final_norm", final_out)
        model._layernorm_backward_np(grad_dense, pre_fn, "final_norm")
        model.backward_transformer(grad_pre_norm := model._layernorm_backward_np(
            grad_dense, pre_fn, "final_norm"
        ) if False else grad_dense, cache)

    # Average
    for key in model.grads:
        if model.grads[key] is not None:
            model.grads[key] /= accum_steps

    # Check grads are reasonable (not exploded)
    max_grad = max(np.abs(model.grads[k]).max() for k in model.grads if model.grads[k] is not None)
    mean_grad = np.mean([np.abs(model.grads[k]).mean() for k in model.grads if model.grads[k] is not None])

    print(f"  After {accum_steps} accumulation steps:")
    print(f"  Max gradient magnitude: {max_grad:.6f}")
    print(f"  Mean gradient magnitude: {mean_grad:.8f}")
    print(f"  Gradients finite: {all(np.isfinite(model.grads[k]).all() for k in model.grads)}")

    passed = max_grad < 100.0 and mean_grad > 0 and not any(
        not np.isfinite(model.grads[k]).all() for k in model.grads
    )

    print(f"\n{'=' * 70}")
    print(f"GRADIENT ACCUMULATION: {'PASS' if passed else 'FAIL'}")
    return passed


def test_loss_decrease():
    """Test that loss decreases over multiple steps."""
    print("\n" + "=" * 70)
    print("TEST 4: Loss Decrease (50 steps)")
    print("=" * 70)

    np.random.seed(7)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    # Simple AdamW
    lr = 5e-4
    beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.01
    m = {k: np.zeros_like(v) for k, v in model.params.items()}
    v = {k: np.zeros_like(v) for k, v in model.params.items()}

    B, L, V = 2, 16, 25
    losses = []

    # Fixed dataset (same batch every step) so model can overfit
    token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
    token_ids[:, 0] = 23
    token_ids[:, 7] = 24
    token_ids[:, -1] = 24
    segment_ids = np.zeros((B, L), dtype=np.int64)
    segment_ids[:, 8:] = 1
    masked_ids = token_ids.copy()
    mask_pos = np.zeros((B, L), dtype=bool)
    mask_pos[:, 2:5] = True
    masked_ids[mask_pos] = 21
    labels = token_ids.copy()

    for step in range(50):
        # Zero grads
        for key in model.grads:
            model.grads[key] = np.zeros_like(model.grads[key])

        # Forward
        logits, cache = model.forward_mlm(masked_ids, segment_ids)

        # Loss
        probs = softmax_numpy(logits)
        n_masked = mask_pos.sum()
        log_probs = np.log(probs + 1e-10)
        targets_onehot = np.eye(V, dtype=np.float32)[labels]
        loss = -np.sum(log_probs * targets_onehot * mask_pos[:, :, np.newaxis]) / max(n_masked, 1)
        losses.append(loss)

        # Backward
        grad_logits = (probs - targets_onehot) * mask_pos[:, :, np.newaxis].astype(np.float32)
        if n_masked > 0:
            grad_logits /= n_masked

        D = model.d_model
        final_out = cache["final_output"]

        # Correct MLM head backward: final_out → dense → GELU → LN → proj → logits
        dense_pre = final_out @ model.params["mlm_head.dense.weight"].T + model.params["mlm_head.dense.bias"]
        gelu_out = model._gelu_np(dense_pre)
        normed = model._layernorm_np(gelu_out, "mlm_head.norm")

        model.grads["mlm_head.proj.bias"] = grad_logits.sum(axis=(0, 1))
        gl_flat = grad_logits.reshape(-1, V)
        normed_flat = normed.reshape(-1, D)
        fo_flat = final_out.reshape(-1, D)
        model.grads["mlm_head.proj.weight"] = gl_flat.T @ normed_flat

        proj_w = model.params["mlm_head.proj.weight"]
        grad_normed = grad_logits @ proj_w
        grad_gelu_out = model._layernorm_backward_np(grad_normed, gelu_out, "mlm_head.norm")
        gelu_deriv = model._gelu_deriv_np(dense_pre)
        grad_dense_pre = grad_gelu_out * gelu_deriv
        gd_flat = grad_dense_pre.reshape(-1, D)
        model.grads["mlm_head.dense.weight"] = gd_flat.T @ fo_flat
        model.grads["mlm_head.dense.bias"] = grad_dense_pre.sum(axis=(0, 1))
        grad_dense = (gd_flat @ model.params["mlm_head.dense.weight"]).reshape(B, L, D)

        pre_fn = cache.get("pre_final_norm", final_out)
        grad_pre_norm = model._layernorm_backward_np(grad_dense, pre_fn, "final_norm")
        model.backward_transformer(grad_pre_norm, cache)

        # Clip
        total_norm = 0.0
        for key in model.grads:
            if model.grads[key] is not None:
                total_norm += np.sum(model.grads[key] ** 2)
        total_norm = np.sqrt(total_norm)
        if total_norm > 1.0:
            for key in model.grads:
                if model.grads[key] is not None:
                    model.grads[key] *= 1.0 / total_norm

        # AdamW update
        t = step + 1
        for name, param in model.params.items():
            g = model.grads.get(name)
            if g is None:
                continue
            m[name] = beta1 * m[name] + (1 - beta1) * g
            v[name] = beta2 * v[name] + (1 - beta2) * (g ** 2)
            m_hat = m[name] / (1 - beta1 ** t)
            v_hat = v[name] / (1 - beta2 ** t)
            model.params[name] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * param)

        if (step + 1) % 10 == 0:
            avg_recent = np.mean(losses[-10:])
            print(f"  Step {step + 1}: loss={loss:.4f}, avg_last_10={avg_recent:.4f}")

    # Check trend
    first_10_avg = np.mean(losses[:10])
    last_10_avg = np.mean(losses[-10:])
    decreased = last_10_avg < first_10_avg

    print(f"\n  First 10 avg loss: {first_10_avg:.4f}")
    print(f"  Last 10 avg loss:  {last_10_avg:.4f}")
    print(f"  Decrease: {first_10_avg - last_10_avg:.4f}")

    print(f"\n{'=' * 70}")
    print(f"LOSS DECREASE: {'PASS' if decreased else 'FAIL'}")
    return decreased


def test_contact_head_dimensions():
    """Test that contact head produces correct dimensions (57D input to MLP)."""
    print("\n" + "=" * 70)
    print("TEST 5: Contact Head Dimensions")
    print("=" * 70)

    np.random.seed(55)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    B, L = 1, 20
    l_a, l_b = 8, 8

    token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
    token_ids[:, 0] = 23
    token_ids[:, l_a] = 24
    token_ids[:, -1] = 24
    segment_ids = np.zeros((B, L), dtype=np.int64)
    segment_ids[:, l_a + 1:] = 1

    contact_logits, cache = model.forward_contact(token_ids, segment_ids, l_a, l_b)

    print(f"  Contact logits shape: {contact_logits.shape}")
    print(f"  Expected: ({l_a}, {l_b}, 1)")

    cc = cache["contact_cache"]
    feat_shape = cc["contact_features"].shape
    print(f"  Contact features shape: {feat_shape}")
    print(f"  Expected: ({l_a}, {l_b}, 57)")

    correct_output = contact_logits.shape == (l_a, l_b, 1)
    correct_features = feat_shape == (l_a, l_b, 57)
    no_nan = np.isfinite(contact_logits).all()

    print(f"  Output shape correct: {correct_output}")
    print(f"  Features shape correct: {correct_features}")
    print(f"  No NaN/Inf: {no_nan}")

    passed = correct_output and correct_features and no_nan
    print(f"\n{'=' * 70}")
    print(f"CONTACT HEAD DIMENSIONS: {'PASS' if passed else 'FAIL'}")
    return passed


def test_embed_norm_consistency():
    """Test that embed_norm params exist and get gradients in wgpu model."""
    print("\n" + "=" * 70)
    print("TEST 6: Embed Norm Consistency")
    print("=" * 70)

    np.random.seed(77)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    # Check embed_norm params exist
    has_gamma = "embed_norm.gamma" in model.params
    has_beta = "embed_norm.beta" in model.params
    print(f"  embed_norm.gamma exists: {has_gamma}")
    print(f"  embed_norm.beta exists: {has_beta}")

    if not (has_gamma and has_beta):
        print(f"\n{'=' * 70}")
        print("EMBED NORM CONSISTENCY: FAIL (params missing)")
        return False

    # Verify shapes
    print(f"  embed_norm.gamma shape: {model.params['embed_norm.gamma'].shape} (expected ({model.d_model},))")
    print(f"  embed_norm.beta shape: {model.params['embed_norm.beta'].shape} (expected ({model.d_model},))")

    # Forward pass — check that embed_norm is applied (output should be normalized)
    B, L = 2, 12
    token_ids = np.random.randint(0, 20, (B, L)).astype(np.int64)
    token_ids[:, 0] = 23
    token_ids[:, -1] = 24
    segment_ids = np.zeros((B, L), dtype=np.int64)

    logits, cache = model.forward_mlm(token_ids, segment_ids)

    # Check pre_embed_norm is in cache
    has_cache = "pre_embed_norm" in cache
    print(f"  pre_embed_norm in cache: {has_cache}")

    if has_cache:
        pre_en = cache["pre_embed_norm"]
        print(f"  pre_embed_norm shape: {pre_en.shape}")
        print(f"  pre_embed_norm mean: {pre_en.mean():.4f}, std: {pre_en.std():.4f}")

    # Run backward and check embed_norm gets gradients
    for key in model.grads:
        model.grads[key] = np.zeros_like(model.grads[key])

    V = 25
    masked_ids = token_ids.copy()
    mask_pos = np.zeros((B, L), dtype=bool)
    mask_pos[:, 2:4] = True
    masked_ids[mask_pos] = 21
    labels = token_ids.copy()

    logits, cache = model.forward_mlm(masked_ids, segment_ids)
    probs = softmax_numpy(logits)
    targets_onehot = np.eye(V, dtype=np.float32)[labels]
    grad_logits = (probs - targets_onehot) * mask_pos[:, :, np.newaxis].astype(np.float32)
    n_masked = mask_pos.sum()
    if n_masked > 0:
        grad_logits /= n_masked

    D = model.d_model
    final_out = cache["final_output"]
    dense_pre = final_out @ model.params["mlm_head.dense.weight"].T + model.params["mlm_head.dense.bias"]
    gelu_out = model._gelu_np(dense_pre)
    normed = model._layernorm_np(gelu_out, "mlm_head.norm")

    model.grads["mlm_head.proj.bias"] = grad_logits.sum(axis=(0, 1))
    gl_flat = grad_logits.reshape(-1, V)
    normed_flat = normed.reshape(-1, D)
    fo_flat = final_out.reshape(-1, D)
    model.grads["mlm_head.proj.weight"] = gl_flat.T @ normed_flat

    proj_w = model.params["mlm_head.proj.weight"]
    grad_normed = grad_logits @ proj_w
    grad_gelu_out = model._layernorm_backward_np(grad_normed, gelu_out, "mlm_head.norm")
    gelu_deriv = model._gelu_deriv_np(dense_pre)
    grad_dense_pre = grad_gelu_out * gelu_deriv
    gd_flat = grad_dense_pre.reshape(-1, D)
    model.grads["mlm_head.dense.weight"] = gd_flat.T @ fo_flat
    model.grads["mlm_head.dense.bias"] = grad_dense_pre.sum(axis=(0, 1))
    grad_dense = (gd_flat @ model.params["mlm_head.dense.weight"]).reshape(B, L, D)

    pre_fn = cache.get("pre_final_norm", final_out)
    grad_pre_norm = model._layernorm_backward_np(grad_dense, pre_fn, "final_norm")
    model.backward_transformer(grad_pre_norm, cache)

    # Check embed_norm gradients
    gamma_grad = np.abs(model.grads["embed_norm.gamma"]).sum()
    beta_grad = np.abs(model.grads["embed_norm.beta"]).sum()
    print(f"  embed_norm.gamma grad magnitude: {gamma_grad:.6f}")
    print(f"  embed_norm.beta grad magnitude: {beta_grad:.6f}")

    # Check embedding gradients too
    tok_emb_grad = np.abs(model.grads["token_embedding.weight"]).sum()
    seg_emb_grad = np.abs(model.grads["segment_embedding.weight"]).sum()
    print(f"  token_embedding.weight grad magnitude: {tok_emb_grad:.6f}")
    print(f"  segment_embedding.weight grad magnitude: {seg_emb_grad:.6f}")

    passed = (has_gamma and has_beta and has_cache and
              gamma_grad > 0 and beta_grad > 0 and
              tok_emb_grad > 0 and seg_emb_grad > 0)

    print(f"\n{'=' * 70}")
    print(f"EMBED NORM CONSISTENCY: {'PASS' if passed else 'FAIL'}")
    return passed


def test_wgpu_pytorch_key_mapping():
    """Test that embed_norm keys map correctly between wgpu and PyTorch."""
    print("\n" + "=" * 70)
    print("TEST 7: wgpu ↔ PyTorch Key Mapping for embed_norm")
    print("=" * 70)

    model = CoevolutionTransformerWGPU(
        d_model=128, n_layers=6, n_heads=4, d_ff=512,
        vocab_size=25, max_len=64
    )

    # Test key mapping
    gamma_torch = model._wgpu_key_to_torch("embed_norm.gamma")
    beta_torch = model._wgpu_key_to_torch("embed_norm.beta")
    print(f"  embed_norm.gamma → {gamma_torch} (expected: embed_norm.weight)")
    print(f"  embed_norm.beta → {beta_torch} (expected: embed_norm.bias)")

    gamma_back = model._torch_key_to_wgpu("embed_norm.weight")
    beta_back = model._torch_key_to_wgpu("embed_norm.bias")
    print(f"  embed_norm.weight → {gamma_back} (expected: embed_norm.gamma)")
    print(f"  embed_norm.bias → {beta_back} (expected: embed_norm.beta)")

    passed = (gamma_torch == "embed_norm.weight" and
              beta_torch == "embed_norm.bias" and
              gamma_back == "embed_norm.gamma" and
              beta_back == "embed_norm.beta")

    print(f"\n{'=' * 70}")
    print(f"KEY MAPPING: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    results = {}

    results["Phase 1 Gradient Flow"] = test_phase1_gradient_flow()
    results["Phase 2 Gradient Flow"] = test_phase2_gradient_flow()
    results["Gradient Accumulation"] = test_gradient_accumulation()
    results["Loss Decrease"] = test_loss_decrease()
    results["Contact Head Dimensions"] = test_contact_head_dimensions()
    results["Embed Norm Consistency"] = test_embed_norm_consistency()
    results["Key Mapping"] = test_wgpu_pytorch_key_mapping()

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)
