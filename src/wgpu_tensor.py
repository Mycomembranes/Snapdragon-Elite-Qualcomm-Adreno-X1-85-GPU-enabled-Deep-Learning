"""wgpu-native GPU tensor library for Adreno X1-85 training via Vulkan/D3D12.

This module provides a complete GPU tensor implementation with compute shaders
for deep learning operations on mobile Adreno GPUs.

Backend auto-detection: if libd3d12_compute.so is available, the D3D12 native
backend is used (bypassing Vulkan/Dozen overhead). Set WGPU_FORCE_BACKEND=wgpu
to disable auto-detection.
"""

import os
import atexit
import struct
import numpy as np
import wgpu
import wgpu.backends.wgpu_native  # noqa: F401

# ============================================================================
# Device Singleton & Pipeline Cache
# ============================================================================

_device = None
_pipeline_cache = {}


def _get_device():
    """Get or create the wgpu device singleton."""
    global _device
    if _device is None:
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No wgpu adapter found")
        # Request device — use adapter's supported limits
        supported = adapter.limits
        _device = adapter.request_device_sync(
            required_limits={
                "max-buffer-size": supported.get("max-buffer-size", 268435456),
                "max-storage-buffer-binding-size": supported.get(
                    "max-storage-buffer-binding-size", 134217728
                ),
            }
        )
    return _device


def _cleanup():
    """Cleanup wgpu resources on exit."""
    global _device
    if _device is not None:
        try:
            _device.destroy()
        except Exception:
            pass
        _device = None


atexit.register(_cleanup)


def get_device_info():
    """Return device info string for diagnostics."""
    dev = _get_device()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = adapter.info
    return {
        "vendor": info.get("vendor", "unknown"),
        "device": info.get("device", "unknown"),
        "backend": info.get("backend_type", "unknown"),
        "adapter_type": info.get("adapter_type", "unknown"),
    }


# ============================================================================
# WGSL Compute Shader Sources
# ============================================================================

WGSL_ADD = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] + b[idx];
    }
}
"""

WGSL_MUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] * b[idx];
    }
}
"""

WGSL_SUB = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] - b[idx];
    }
}
"""

WGSL_SCALAR_MUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> scalar: f32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = a[idx] * scalar;
    }
}
"""

WGSL_NEG = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = -a[idx];
    }
}
"""

WGSL_GELU = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let val = x[idx];
        let cdf = 0.5 * (1.0 + tanh(
            sqrt(2.0 / 3.14159265359) * (val + 0.044715 * val * val * val)
        ));
        out[idx] = val * cdf;
    }
}
"""

WGSL_RELU = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = max(0.0, x[idx]);
    }
}
"""

WGSL_SIGMOID = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = 1.0 / (1.0 + exp(-x[idx]));
    }
}
"""

WGSL_TANH = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        out[idx] = tanh(x[idx]);
    }
}
"""

WGSL_RELU_BACKWARD = """
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> x: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        grad_in[idx] = select(0.0, grad_out[idx], x[idx] > 0.0);
    }
}
"""

WGSL_GELU_BACKWARD = """
// GELU backward: grad_in = grad_out * gelu_deriv(x)
// Uses sigmoid approximation: gelu_deriv(x) ≈ sigmoid(1.702 * x)
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> x: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        let val = x[idx];
        let gelu_deriv = 1.0 / (1.0 + exp(-1.702 * val));
        grad_in[idx] = grad_out[idx] * gelu_deriv;
    }
}
"""

WGSL_SOFTMAX_BACKWARD = """
// Softmax backward: grad_input = P * (grad_output - sum(grad_output * P))
// One workgroup per row. params.x = width
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> probs: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = params.x;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Compute sum(grad_out * probs) for this row
    var local_sum = 0.0;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + grad_out[row_offset + col] * probs[row_offset + col];
        col = col + 256u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let dot_sum = shared_val[0];
    workgroupBarrier();

    // Step 2: grad_in = probs * (grad_out - dot_sum)
    col = tid;
    loop {
        if (col >= width) { break; }
        grad_in[row_offset + col] = probs[row_offset + col] * (grad_out[row_offset + col] - dot_sum);
        col = col + 256u;
    }
}
"""

WGSL_LAYERNORM_BACKWARD = """
// LayerNorm backward pass. One workgroup per row.
// Computes grad_input from grad_output, input x, gamma.
// Also accumulates grad_gamma and grad_beta into atomic buffers.
// params: x=width, y=eps (as f32 reinterpreted)
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> x: array<f32>;
@group(0) @binding(2)
var<storage, read> gamma: array<f32>;
@group(0) @binding(3)
var<storage, read_write> grad_in: array<f32>;
@group(0) @binding(4)
var<uniform> params: vec4<f32>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = u32(params.x);
    let eps = params.y;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Compute mean
    var local_sum = 0.0;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + x[row_offset + col];
        col = col + 256u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mean_val = shared_val[0] / f32(width);
    workgroupBarrier();

    // Step 2: Compute variance
    var local_var = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        let diff = x[row_offset + col] - mean_val;
        local_var = local_var + diff * diff;
        col = col + 256u;
    }
    shared_val[tid] = local_var;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let var_val = shared_val[0] / f32(width);
    let inv_std = 1.0 / sqrt(var_val + eps);
    workgroupBarrier();

    // Step 3: Compute dx_hat * gamma and two sums needed for gradient
    // dx_hat = grad_out * gamma
    // Need: sum(dx_hat) and sum(dx_hat * x_hat) where x_hat = (x - mean) * inv_std
    var local_sum_dxhat = 0.0;
    var local_sum_dxhat_xhat = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        let x_hat = (x[row_offset + col] - mean_val) * inv_std;
        let dx_hat = grad_out[row_offset + col] * gamma[col];
        local_sum_dxhat = local_sum_dxhat + dx_hat;
        local_sum_dxhat_xhat = local_sum_dxhat_xhat + dx_hat * x_hat;
        col = col + 256u;
    }

    // Reduce sum_dxhat
    shared_val[tid] = local_sum_dxhat;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let sum_dxhat = shared_val[0];
    workgroupBarrier();

    // Reduce sum_dxhat_xhat
    shared_val[tid] = local_sum_dxhat_xhat;
    workgroupBarrier();
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let sum_dxhat_xhat = shared_val[0];
    workgroupBarrier();

    // Step 4: Compute grad_input
    // dx = inv_std / N * (N * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat)
    let n_f = f32(width);
    col = tid;
    loop {
        if (col >= width) { break; }
        let x_hat = (x[row_offset + col] - mean_val) * inv_std;
        let dx_hat = grad_out[row_offset + col] * gamma[col];
        grad_in[row_offset + col] = inv_std / n_f * (n_f * dx_hat - sum_dxhat - x_hat * sum_dxhat_xhat);
        col = col + 256u;
    }
}
"""

WGSL_SUM_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = params.x;
    let stride = params.y;
    let idx = gid.x;
    let lid_x = lid.x;

    var sum_val = 0.0;
    if (idx < numel) {
        sum_val = data[idx];
    }
    shared[lid_x] = sum_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = shared[lid_x] + shared[lid_x + stride_val];
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0];
        }
    }
}
"""

WGSL_MAX_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = params.x;
    let idx = gid.x;
    let lid_x = lid.x;

    var max_val = -3.4e38;
    if (idx < numel) {
        max_val = data[idx];
    }
    shared[lid_x] = max_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = max(shared[lid_x], shared[lid_x + stride_val]);
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0];
        }
    }
}
"""

WGSL_MEAN_REDUCE = """
@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let numel = u32(params.x);
    let idx = gid.x;
    let lid_x = lid.x;

    var sum_val = 0.0;
    if (idx < numel) {
        sum_val = data[idx];
    }
    shared[lid_x] = sum_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            shared[lid_x] = shared[lid_x] + shared[lid_x + stride_val];
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = shared[0] / f32(numel);
        }
    }
}
"""

WGSL_MATMUL = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    let lx = lid.x;
    let ly = lid.y;

    let row = wid.y * 16u + ly;
    let col = wid.x * 16u + lx;

    var result = 0.0;

    var tile_idx = 0u;
    loop {
        if (tile_idx >= k) { break; }

        let a_col = tile_idx + lx;
        let a_idx = row * k + a_col;
        tile_a[ly * 16u + lx] = select(0.0, a[a_idx], row < m && a_col < k);

        let b_row = tile_idx + ly;
        let b_idx = b_row * n + col;
        tile_b[ly * 16u + lx] = select(0.0, b[b_idx], b_row < k && col < n);

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            result = result + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
        tile_idx = tile_idx + 16u;
    }

    if (row < m && col < n) {
        out[row * n + col] = result;
    }
}
"""

# ============================================================================
# Fused Shaders — reduce GPU dispatch overhead by combining operations
# ============================================================================

# --------------------------------------------------------------------------
# Single-kernel MLP forward: 58 → 256 (relu) → 128 (relu) → 1
# One workgroup per sample, all layers in shared memory, 1 dispatch total.
# --------------------------------------------------------------------------
WGSL_MLP_FORWARD_58_256_128_1 = """
// Fused 3-layer MLP: (N, 58) -> relu(256) -> relu(128) -> (N, 1)
// Uses chain rule: each sample computed end-to-end in one workgroup.
// Eliminates ALL intermediate buffer allocations and GPU syncs.
//
// Bindings:
//   0: input       (N, 58)       read
//   1: weights     packed        read   [W0(58*256) + W2(256*128) + W4(128)]
//   2: biases      packed        read   [b0(256) + b2(128) + b4(1)]
//   3: output      (N,)          rw     logits
//   4: caches      packed        rw     [pre0(N*256) + h0(N*256) + pre1(N*128) + h1(N*128)]
//   5: masks       packed        read   [mask0(N*256) + mask1(N*128)]  (1.0 if no dropout)
//   6: params      uniform       [N, 0, 0, 0]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> caches: array<f32>;
@group(0) @binding(5) var<storage, read> masks: array<f32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;

// Hardcoded architecture constants
const DIM_IN: u32 = 58u;
const DIM_H0: u32 = 256u;
const DIM_H1: u32 = 128u;
// Weight offsets in packed buffer
const W0_OFF: u32 = 0u;           // 58 * 256 = 14848
const W2_OFF: u32 = 14848u;       // 256 * 128 = 32768
const W4_OFF: u32 = 47616u;       // 128
// Bias offsets in packed buffer
const B0_OFF: u32 = 0u;
const B2_OFF: u32 = 256u;
const B4_OFF: u32 = 384u;

var<workgroup> act_a: array<f32, 256>;  // layer 0 output (ping)
var<workgroup> act_b: array<f32, 256>;  // layer 2 output (pong)

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let sample = wid.x;
    let tid = lid.x;
    let N = params.x;
    if (sample >= N) { return; }

    // Cache layout offsets (depend on N):
    //   pre_relu0: 0                     size N*256
    //   hidden0:   N*256                 size N*256
    //   pre_relu1: N*512                 size N*128
    //   hidden1:   N*512 + N*128 = N*640 size N*128
    let pre0_off = sample * DIM_H0;
    let h0_off   = N * DIM_H0 + sample * DIM_H0;
    let pre1_off = N * DIM_H0 * 2u + sample * DIM_H1;
    let h1_off   = N * DIM_H0 * 2u + N * DIM_H1 + sample * DIM_H1;
    // Mask offsets: mask0 at 0 (N*256), mask1 at N*256 (N*128)
    let m0_off = sample * DIM_H0;
    let m1_off = N * DIM_H0 + sample * DIM_H1;

    // === Layer 0: input(58) → hidden0(256) with relu + dropout ===
    // All 256 threads active, each computes one hidden unit
    {
        var sum = biases[B0_OFF + tid];
        let in_base = sample * DIM_IN;
        for (var k = 0u; k < DIM_IN; k = k + 1u) {
            sum = sum + input[in_base + k] * weights[W0_OFF + k * DIM_H0 + tid];
        }
        // Store pre-relu cache
        caches[pre0_off + tid] = sum;
        // ReLU + dropout mask
        let activated = max(0.0, sum) * masks[m0_off + tid];
        caches[h0_off + tid] = activated;
        act_a[tid] = activated;
    }
    workgroupBarrier();

    // === Layer 2: hidden0(256) → hidden1(128) with relu + dropout ===
    // 128 threads active
    if (tid < DIM_H1) {
        var sum = biases[B2_OFF + tid];
        for (var k = 0u; k < DIM_H0; k = k + 1u) {
            sum = sum + act_a[k] * weights[W2_OFF + k * DIM_H1 + tid];
        }
        caches[pre1_off + tid] = sum;
        let activated = max(0.0, sum) * masks[m1_off + tid];
        caches[h1_off + tid] = activated;
        act_b[tid] = activated;
    }
    workgroupBarrier();

    // === Layer 4: hidden1(128) → output(1) ===
    // Thread 0 computes the final dot product
    if (tid == 0u) {
        var sum = biases[B4_OFF];
        for (var k = 0u; k < DIM_H1; k = k + 1u) {
            sum = sum + act_b[k] * weights[W4_OFF + k];
        }
        output[sample] = sum;
    }
}
"""

WGSL_MATMUL_ADD_RELU = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read> bias: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<storage, read_write> pre_relu: array<f32>;
@group(0) @binding(5)
var<uniform> params: vec4<u32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    let lx = lid.x;
    let ly = lid.y;

    let row = wid.y * 16u + ly;
    let col = wid.x * 16u + lx;

    var result = 0.0;

    var tile_idx = 0u;
    loop {
        if (tile_idx >= k) { break; }

        let a_col = tile_idx + lx;
        let a_idx = row * k + a_col;
        tile_a[ly * 16u + lx] = select(0.0, a[a_idx], row < m && a_col < k);

        let b_row = tile_idx + ly;
        let b_idx = b_row * n + col;
        tile_b[ly * 16u + lx] = select(0.0, b[b_idx], b_row < k && col < n);

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            result = result + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
        tile_idx = tile_idx + 16u;
    }

    if (row < m && col < n) {
        let idx = row * n + col;
        let with_bias = result + bias[idx];
        pre_relu[idx] = with_bias;
        out[idx] = max(0.0, with_bias);
    }
}
"""

WGSL_MATMUL_ADD = """
@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read> bias: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<uniform> params: vec4<u32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    let lx = lid.x;
    let ly = lid.y;

    let row = wid.y * 16u + ly;
    let col = wid.x * 16u + lx;

    var result = 0.0;

    var tile_idx = 0u;
    loop {
        if (tile_idx >= k) { break; }

        let a_col = tile_idx + lx;
        let a_idx = row * k + a_col;
        tile_a[ly * 16u + lx] = select(0.0, a[a_idx], row < m && a_col < k);

        let b_row = tile_idx + ly;
        let b_idx = b_row * n + col;
        tile_b[ly * 16u + lx] = select(0.0, b[b_idx], b_row < k && col < n);

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            result = result + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
        tile_idx = tile_idx + 16u;
    }

    if (row < m && col < n) {
        out[row * n + col] = result + bias[row * n + col];
    }
}
"""

WGSL_TRANSPOSE_2D = """
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = params.x;
    let cols = params.y;
    let i = gid.y;
    let j = gid.x;

    if (i < rows && j < cols) {
        out[j * rows + i] = x[i * cols + j];
    }
}
"""

WGSL_LAYER_NORM = """
// One workgroup per row. params.x = width, params.y = eps (as f32)
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read> gamma: array<f32>;
@group(0) @binding(2)
var<storage, read> beta: array<f32>;
@group(0) @binding(3)
var<storage, read_write> out: array<f32>;
@group(0) @binding(4)
var<uniform> params: vec4<f32>;

var<workgroup> shared_val: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = u32(params.x);
    let eps = params.y;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Compute mean
    var local_sum = 0.0;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + x[row_offset + col];
        col = col + 256u;
    }
    shared_val[tid] = local_sum;
    workgroupBarrier();

    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let mean_val = shared_val[0] / f32(width);
    workgroupBarrier();

    // Step 2: Compute variance
    var local_var = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        let diff = x[row_offset + col] - mean_val;
        local_var = local_var + diff * diff;
        col = col + 256u;
    }
    shared_val[tid] = local_var;
    workgroupBarrier();

    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) { shared_val[tid] = shared_val[tid] + shared_val[tid + s]; }
        workgroupBarrier();
        s = s >> 1u;
    }
    let var_val = shared_val[0] / f32(width);
    let inv_std = 1.0 / sqrt(var_val + eps);
    workgroupBarrier();

    // Step 3: Normalize and apply gamma/beta
    col = tid;
    loop {
        if (col >= width) { break; }
        let norm = (x[row_offset + col] - mean_val) * inv_std;
        out[row_offset + col] = norm * gamma[col] + beta[col];
        col = col + 256u;
    }
}
"""

WGSL_SOFTMAX = """
// One workgroup per row. Each thread handles multiple elements if width > 256.
// params.x = width (last dim), params.y = num_rows
@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = params.x;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Step 1: Find max in this row (each thread covers multiple elements)
    var local_max = -3.4e38;
    var col = tid;
    loop {
        if (col >= width) { break; }
        local_max = max(local_max, x[row_offset + col]);
        col = col + 256u;
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_max = shared_max[0];
    workgroupBarrier();

    // Step 2: Compute exp(x - max) and sum
    var local_sum = 0.0;
    col = tid;
    loop {
        if (col >= width) { break; }
        local_sum = local_sum + exp(x[row_offset + col] - row_max);
        col = col + 256u;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let row_sum = shared_sum[0];
    workgroupBarrier();

    // Step 3: Write normalized values
    col = tid;
    loop {
        if (col >= width) { break; }
        out[row_offset + col] = exp(x[row_offset + col] - row_max) / row_sum;
        col = col + 256u;
    }
}
"""

WGSL_CROSS_ENTROPY = """
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let log_p = logits[idx];
        let target = targets[idx];
        out[idx] = -target * log_p - (1.0 - target) * log(1.0 - exp(log_p) + 1e-6);
    }
}
"""

WGSL_FOCAL_BCE = """
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let logit = logits[idx];
        let target = targets[idx];
        let gamma = params.x;
        let alpha = params.y;

        let p = 1.0 / (1.0 + exp(-logit));
        let ce = -target * log(p + 1e-6) - (1.0 - target) * log(1.0 - p + 1e-6);
        let p_t = select(1.0 - p, p, target > 0.5);
        let focal_weight = pow(1.0 - p_t, gamma);
        let focal_loss = alpha * (1.0 - alpha) * focal_weight * ce;

        out[idx] = focal_loss;
    }
}
"""

WGSL_EMBEDDING = """
@group(0) @binding(0)
var<storage, read> weight: array<f32>;
@group(0) @binding(1)
var<storage, read> indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_indices = params.x;
    let embedding_dim = params.y;

    if (idx < num_indices * embedding_dim) {
        let i = idx / embedding_dim;
        let j = idx % embedding_dim;
        let word_idx = indices[i];
        let weight_idx = word_idx * embedding_dim + j;
        out[idx] = weight[weight_idx];
    }
}
"""


# ============================================================================
# Dispatch Helper
# ============================================================================

def _dispatch_shader(device, wgsl_code, buffers, workgroups):
    """Execute a compute shader on GPU.

    Args:
        device: wgpu device
        wgsl_code: WGSL source code string
        buffers: list of (wgpu.GPUBuffer, access_mode) tuples
            access_mode: "read" or "read_write"
        workgroups: tuple (x, y=1, z=1) for dispatch
    """
    # Check cache
    cache_key = wgsl_code
    if cache_key in _pipeline_cache:
        pipeline = _pipeline_cache[cache_key]
    else:
        # Create shader module
        shader_module = device.create_shader_module(code=wgsl_code)

        # Build bind group layout
        entries = []
        for i, (buf, access) in enumerate(buffers):
            visibility = wgpu.ShaderStage.COMPUTE
            if access == "read":
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "read-only-storage",
                        "has_dynamic_offset": False,
                    },
                }
            elif access == "uniform":
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": False,
                    },
                }
            else:  # "read_write"
                entry = {
                    "binding": i,
                    "visibility": visibility,
                    "buffer": {
                        "type": "storage",
                        "has_dynamic_offset": False,
                    },
                }
            entries.append(entry)

        bind_group_layout = device.create_bind_group_layout(entries=entries)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )

        # Create compute pipeline
        pipeline = device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": shader_module, "entry_point": "main"},
        )
        _pipeline_cache[cache_key] = pipeline

    # Create bind group
    resources = []
    for i, (buf, _) in enumerate(buffers):
        resources.append({
            "binding": i,
            "resource": {"buffer": buf, "offset": 0, "size": buf.size},
        })

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=resources,
    )

    # Submit compute pass
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    if len(workgroups) == 1:
        compute_pass.dispatch_workgroups(workgroups[0])
    elif len(workgroups) == 2:
        compute_pass.dispatch_workgroups(workgroups[0], workgroups[1])
    else:
        compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2])

    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    device.queue.on_submitted_work_done_sync()


def _get_or_create_pipeline(device, wgsl_code, buffers):
    """Get cached pipeline or create new one."""
    cache_key = wgsl_code
    if cache_key in _pipeline_cache:
        return _pipeline_cache[cache_key]

    shader_module = device.create_shader_module(code=wgsl_code)
    entries = []
    for i, (buf, access) in enumerate(buffers):
        visibility = wgpu.ShaderStage.COMPUTE
        if access == "read":
            buf_type = "read-only-storage"
        elif access == "uniform":
            buf_type = "uniform"
        else:
            buf_type = "storage"
        entries.append({
            "binding": i,
            "visibility": visibility,
            "buffer": {"type": buf_type, "has_dynamic_offset": False},
        })

    bind_group_layout = device.create_bind_group_layout(entries=entries)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )
    _pipeline_cache[cache_key] = pipeline
    return pipeline


def _dispatch_batch(device, dispatches):
    """Execute multiple compute dispatches in a single command buffer submission.

    Eliminates per-dispatch sync overhead by batching all dispatches into one
    command encoder with one submit + one sync at the end.

    Args:
        device: wgpu device
        dispatches: list of (wgsl_code, buffers, workgroups) tuples
            Each follows the same format as _dispatch_shader arguments.
    """
    command_encoder = device.create_command_encoder()

    for wgsl_code, buffers, workgroups in dispatches:
        pipeline = _get_or_create_pipeline(device, wgsl_code, buffers)

        resources = []
        for i, (buf, _) in enumerate(buffers):
            resources.append({
                "binding": i,
                "resource": {"buffer": buf, "offset": 0, "size": buf.size},
            })

        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=resources,
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(pipeline)
        compute_pass.set_bind_group(0, bind_group)

        if len(workgroups) == 1:
            compute_pass.dispatch_workgroups(workgroups[0])
        elif len(workgroups) == 2:
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1])
        else:
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2])

        compute_pass.end()

    device.queue.submit([command_encoder.finish()])
    device.queue.on_submitted_work_done_sync()


def mlp_forward_fused(feat, weights_biases):
    """Execute a full MLP forward pass with minimal GPU dispatches.

    Chains matmul+add+relu layers and a final matmul+add in a single
    command buffer, with only one GPU sync at the end.

    Args:
        feat: WgpuTensor (N, in_dim) — input features
        weights_biases: list of (W_gpu, b_gpu, activation) tuples
            W_gpu: WgpuTensor (in_dim, out_dim)
            b_gpu: WgpuTensor (N, out_dim) — pre-broadcast bias
            activation: "relu" or None

    Returns:
        (output, pre_relu_caches): output WgpuTensor and list of pre-relu WgpuTensors
    """
    device = _get_device()
    dispatches = []
    pre_relu_caches = []
    current = feat

    for W_gpu, b_gpu, activation in weights_biases:
        m = current.shape[-2]
        k = current.shape[-1]
        n = W_gpu.shape[-1]

        params = struct.pack("4I", m, n, k, 0)
        params_buffer = device.create_buffer_with_data(
            data=params,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        if activation == "relu":
            out = WgpuTensor.zeros((m, n), current.dtype)
            pre_relu = WgpuTensor.zeros((m, n), current.dtype)
            dispatches.append((
                WGSL_MATMUL_ADD_RELU,
                [
                    (current.buffer, "read"),
                    (W_gpu.buffer, "read"),
                    (b_gpu.buffer, "read"),
                    (out.buffer, "read_write"),
                    (pre_relu.buffer, "read_write"),
                    (params_buffer, "uniform"),
                ],
                ((n + 15) // 16, (m + 15) // 16),
            ))
            pre_relu_caches.append(pre_relu)
            current = out
        else:
            out = WgpuTensor.zeros((m, n), current.dtype)
            dispatches.append((
                WGSL_MATMUL_ADD,
                [
                    (current.buffer, "read"),
                    (W_gpu.buffer, "read"),
                    (b_gpu.buffer, "read"),
                    (out.buffer, "read_write"),
                    (params_buffer, "uniform"),
                ],
                ((n + 15) // 16, (m + 15) // 16),
            ))
            pre_relu_caches.append(None)
            current = out

    # Single submission for all layers
    _dispatch_batch(device, dispatches)
    return current, pre_relu_caches


def contact_mlp_forward_fused(feat_flat, W0, b0, W2, b2, W4, b4,
                               dropout_mask0=None, dropout_mask1=None,
                               dropout_scale=1.0):
    """Single-kernel fused MLP forward for 58→256→128→1 contact head.

    Computes the entire 3-layer MLP in ONE GPU dispatch using the chain rule:
        logits = W4 @ relu(W2 @ relu(W0 @ x + b0) + b2) + b4

    Each sample is computed end-to-end in one workgroup with shared memory.
    Eliminates all intermediate buffer allocations and inter-layer GPU syncs.

    Args:
        feat_flat: numpy (N, 58) input features
        W0: numpy (256, 58) — layer 0 weight (will be transposed internally)
        b0: numpy (256,) — layer 0 bias
        W2: numpy (128, 256) — layer 2 weight (will be transposed internally)
        b2: numpy (128,) — layer 2 bias
        W4: numpy (1, 128) — layer 4 weight (will be transposed internally)
        b4: numpy (1,) — layer 4 bias
        dropout_mask0: numpy (N, 256) or None — scaled dropout mask for layer 0
        dropout_mask1: numpy (N, 128) or None — scaled dropout mask for layer 2
        dropout_scale: float — 1/(1-p) scale if using dropout

    Returns:
        logits: numpy (N,) — output logits
        mlp_cache: dict with keys:
            mlp_pre_relu0: numpy (N, 256)
            mlp_hidden0:   numpy (N, 256) — post-relu, post-dropout
            mlp_pre_relu1: numpy (N, 128)
            mlp_hidden1:   numpy (N, 128) — post-relu, post-dropout
    """
    device = _get_device()
    N = feat_flat.shape[0]

    # Pack weights: W0.T(58,256) + W2.T(256,128) + W4.T(128,1) contiguously
    W0_t = np.ascontiguousarray(W0.T, dtype=np.float32)  # (58, 256)
    W2_t = np.ascontiguousarray(W2.T, dtype=np.float32)  # (256, 128)
    W4_t = np.ascontiguousarray(W4.T, dtype=np.float32)  # (128, 1)
    weights_packed = np.concatenate([W0_t.ravel(), W2_t.ravel(), W4_t.ravel()])
    biases_packed = np.concatenate([
        b0.astype(np.float32),
        b2.astype(np.float32),
        b4.astype(np.float32),
    ])

    # Pack dropout masks (all 1.0 if no dropout)
    if dropout_mask0 is not None:
        mask0 = (dropout_mask0 * dropout_scale).astype(np.float32)
    else:
        mask0 = np.ones((N, 256), dtype=np.float32)
    if dropout_mask1 is not None:
        mask1 = (dropout_mask1 * dropout_scale).astype(np.float32)
    else:
        mask1 = np.ones((N, 128), dtype=np.float32)
    masks_packed = np.concatenate([mask0.ravel(), mask1.ravel()])

    # Allocate output and cache buffers
    output_size = N
    cache_size = N * 256 + N * 256 + N * 128 + N * 128  # pre0 + h0 + pre1 + h1

    # Create GPU buffers
    input_buf = device.create_buffer_with_data(
        data=np.ascontiguousarray(feat_flat, dtype=np.float32).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    weights_buf = device.create_buffer_with_data(
        data=weights_packed.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    biases_buf = device.create_buffer_with_data(
        data=biases_packed.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    output_buf = device.create_buffer(
        size=output_size * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    cache_buf = device.create_buffer(
        size=cache_size * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    masks_buf = device.create_buffer_with_data(
        data=masks_packed.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    params = struct.pack("4I", N, 0, 0, 0)
    params_buf = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # Single dispatch: N workgroups, 256 threads each
    _dispatch_shader(
        device,
        WGSL_MLP_FORWARD_58_256_128_1,
        [
            (input_buf, "read"),
            (weights_buf, "read"),
            (biases_buf, "read"),
            (output_buf, "read_write"),
            (cache_buf, "read_write"),
            (masks_buf, "read"),
            (params_buf, "uniform"),
        ],
        (N,),
    )

    # Read back results
    logits = np.frombuffer(device.queue.read_buffer(output_buf), dtype=np.float32).copy()
    cache_data = np.frombuffer(device.queue.read_buffer(cache_buf), dtype=np.float32).copy()

    # Unpack caches
    off = 0
    pre_relu0 = cache_data[off:off + N * 256].reshape(N, 256)
    off += N * 256
    hidden0 = cache_data[off:off + N * 256].reshape(N, 256)
    off += N * 256
    pre_relu1 = cache_data[off:off + N * 128].reshape(N, 128)
    off += N * 128
    hidden1 = cache_data[off:off + N * 128].reshape(N, 128)

    mlp_cache = {
        "mlp_pre_relu0": pre_relu0,
        "mlp_hidden0": hidden0,
        "mlp_pre_relu1": pre_relu1,
        "mlp_hidden1": hidden1,
    }
    return logits, mlp_cache


# ============================================================================
# WgpuTensor Class
# ============================================================================

class WgpuTensor:
    """GPU tensor wrapper around wgpu storage buffers."""

    def __init__(self, buffer, shape, dtype="float32", strides=None):
        """Initialize a GPU tensor.

        Args:
            buffer: wgpu.GPUBuffer storage buffer
            shape: tuple of dimensions
            dtype: data type ("float32", "int32", "uint32")
            strides: tuple of strides for non-contiguous views (optional)
        """
        self.buffer = buffer
        self._shape = tuple(shape)
        self.dtype = dtype
        self._strides = strides

        # Type info
        self._dtype_map = {
            "float32": np.float32,
            "int32": np.int32,
            "uint32": np.uint32,
        }
        self._dtype_bytes = {
            "float32": 4,
            "int32": 4,
            "uint32": 4,
        }

    # ---- Properties ----
    @property
    def shape(self):
        """Shape of the tensor."""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self._shape)

    def numel(self):
        """Total number of elements."""
        result = 1
        for s in self._shape:
            result *= s
        return result

    # ---- Factory Methods ----
    @staticmethod
    def zeros(shape, dtype="float32"):
        """Create a tensor filled with zeros."""
        if _BACKEND == "d3d12":
            return D3D12Tensor.empty(shape)
        numel = 1
        for s in shape:
            numel *= s
        device = _get_device()
        buffer = device.create_buffer(
            size=numel * (4 if dtype == "float32" else 4),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            mapped_at_creation=True,
        )
        buffer.unmap()
        return WgpuTensor(buffer, shape, dtype)

    @staticmethod
    def ones(shape, dtype="float32"):
        """Create a tensor filled with ones."""
        cpu_array = np.ones(shape, dtype=np.float32 if dtype == "float32" else np.int32)
        return WgpuTensor.from_numpy(cpu_array)

    @staticmethod
    def from_numpy(arr):
        """Create a tensor from a numpy array."""
        # Auto-convert float64 to float32 (GPU only supports f32)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            arr = arr.astype(np.int32)

        if _BACKEND == "d3d12":
            return D3D12Tensor.from_numpy(arr)

        device = _get_device()
        dtype_map = {np.float32: "float32", np.int32: "int32", np.uint32: "uint32"}
        dtype_name = dtype_map.get(arr.dtype.type, "float32")

        arr_c = np.ascontiguousarray(arr)
        buffer_data = arr_c.tobytes()

        buffer = device.create_buffer_with_data(
            data=buffer_data,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        return WgpuTensor(buffer, arr.shape, dtype_name)

    @staticmethod
    def randn(shape):
        """Create a tensor with random normal values."""
        cpu_array = np.random.randn(*shape).astype(np.float32)
        return WgpuTensor.from_numpy(cpu_array)

    @staticmethod
    def arange(start, end=None, step=1, dtype="float32"):
        """Create a 1D tensor with values in a range."""
        if end is None:
            end = start
            start = 0
        cpu_array = np.arange(start, end, step, dtype=np.float32)
        return WgpuTensor.from_numpy(cpu_array)

    # ---- Data Transfer ----
    def numpy(self):
        """Read tensor data back to CPU as numpy array."""
        device = _get_device()
        np_dtype = self._dtype_map.get(self.dtype, np.float32)
        data = device.queue.read_buffer(self.buffer)
        arr = np.frombuffer(data, dtype=np_dtype).copy()
        return arr.reshape(self.shape)

    # ---- Shape Manipulation ----
    def reshape(self, *shape):
        """Return a view with new shape (no copy)."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        numel_old = self.numel()
        numel_new = 1
        for s in new_shape:
            numel_new *= s
        if numel_old != numel_new:
            raise ValueError(f"Cannot reshape {numel_old} elements to {new_shape}")
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    def transpose(self, dim0, dim1):
        """Transpose two dimensions (metadata only, no copy)."""
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    def expand(self, *shape):
        """Expand shape (metadata only, no copy)."""
        new_shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return WgpuTensor(self.buffer, new_shape, self.dtype)

    # ---- Operators ----
    def __add__(self, other):
        """Element-wise addition."""
        return add(self, other)

    def __mul__(self, other):
        """Element-wise multiplication."""
        if isinstance(other, (int, float)):
            return scalar_mul(self, other)
        return mul(self, other)

    def __rmul__(self, scalar):
        """Right multiplication by scalar."""
        return scalar_mul(self, scalar)

    def __sub__(self, other):
        """Element-wise subtraction."""
        return sub(self, other)

    def __neg__(self):
        """Negation."""
        return neg(self)

    def __matmul__(self, other):
        """Matrix multiplication via @ operator."""
        return matmul(self, other)

    @property
    def T(self):
        """Transpose last two dimensions."""
        return transpose_2d(self)

    # ---- Named Operations ----
    def matmul(self, other):
        """Matrix multiplication."""
        return matmul(self, other)

    def sum(self, axis=None):
        """Sum reduction."""
        return sum_reduce(self, axis)

    def mean(self, axis=None):
        """Mean reduction."""
        return mean_reduce(self, axis)

    def max(self, axis=None):
        """Max reduction."""
        return max_reduce(self, axis)


# ============================================================================
# Functional API - Elementwise
# ============================================================================

def add(a, b):
    """Element-wise addition: a + b."""
    if _BACKEND == "d3d12":
        return d3d12_add(a, b)
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_ADD,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def mul(a, b):
    """Element-wise multiplication: a * b."""
    if _BACKEND == "d3d12":
        return d3d12_mul(a, b)
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_MUL,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def sub(a, b):
    """Element-wise subtraction: a - b."""
    if _BACKEND == "d3d12":
        return d3d12_sub(a, b)
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SUB,
        [(a.buffer, "read"), (b.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def scalar_mul(a, scalar):
    """Element-wise multiplication by scalar: a * s."""
    if _BACKEND == "d3d12":
        return d3d12_scalar_mul(a, scalar)
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)

    scalar_bytes = struct.pack("f", scalar)
    scalar_buffer = device.create_buffer_with_data(
        data=scalar_bytes,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SCALAR_MUL,
        [(a.buffer, "read"), (out.buffer, "read_write"), (scalar_buffer, "uniform")],
        (workgroups_x,),
    )
    return out


def neg(a):
    """Negation: -a."""
    if _BACKEND == "d3d12":
        return d3d12_neg(a)
    device = _get_device()
    out = WgpuTensor.zeros(a.shape, a.dtype)
    numel = a.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_NEG,
        [(a.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Activations
# ============================================================================

def gelu(x):
    """GELU activation function."""
    if _BACKEND == "d3d12":
        return d3d12_gelu(x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_GELU,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def relu(x):
    """ReLU activation function."""
    if _BACKEND == "d3d12":
        return d3d12_relu(x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_RELU,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def gelu_backward(grad_out, x):
    """GELU backward: grad_in = grad_out * gelu_deriv(x)."""
    if _BACKEND == "d3d12":
        return d3d12_gelu_backward(grad_out, x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_GELU_BACKWARD,
        [(grad_out.buffer, "read"), (x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def relu_backward(grad_out, x):
    """ReLU backward: grad_in = grad_out * (x > 0)."""
    if _BACKEND == "d3d12":
        return d3d12_relu_backward(grad_out, x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_RELU_BACKWARD,
        [(grad_out.buffer, "read"), (x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def softmax_backward(grad_out, probs):
    """Softmax backward: grad_in = P * (grad_out - sum(grad_out * P)).

    One workgroup per row along last axis.
    """
    if _BACKEND == "d3d12":
        return d3d12_softmax_backward(grad_out, probs)
    device = _get_device()
    out = WgpuTensor.zeros(grad_out.shape, grad_out.dtype)
    width = grad_out.shape[-1]
    num_rows = grad_out.numel() // width

    params = struct.pack("4I", width, num_rows, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_SOFTMAX_BACKWARD,
        [
            (grad_out.buffer, "read"),
            (probs.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return out


def layernorm_backward(grad_out, x, gamma, eps=1e-5):
    """LayerNorm backward: compute grad_input.

    Note: grad_gamma and grad_beta must be computed separately on CPU
    (atomic float add not available in WGSL). This function only returns grad_input.

    One workgroup per row.
    """
    if _BACKEND == "d3d12":
        return d3d12_layernorm_backward(grad_out, x, gamma, eps)
    device = _get_device()
    grad_in = WgpuTensor.zeros(x.shape, x.dtype)
    width = x.shape[-1]
    num_rows = x.numel() // width

    params = struct.pack("4f", float(width), eps, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_LAYERNORM_BACKWARD,
        [
            (grad_out.buffer, "read"),
            (x.buffer, "read"),
            (gamma.buffer, "read"),
            (grad_in.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return grad_in


def sigmoid(x):
    """Sigmoid activation function."""
    if _BACKEND == "d3d12":
        return d3d12_sigmoid(x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SIGMOID,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def tanh_act(x):
    """Tanh activation function."""
    if _BACKEND == "d3d12":
        return d3d12_tanh(x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_TANH,
        [(x.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Reductions
# ============================================================================

def sum_reduce(x, axis=None):
    """Sum reduction along axis."""
    if _BACKEND == "d3d12" and axis is None:
        return d3d12_sum(x)
    device = _get_device()

    if axis is None:
        # Full reduction
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4I", numel, 1, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_SUM_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


def max_reduce(x, axis=None):
    """Max reduction along axis."""
    if _BACKEND == "d3d12" and axis is None:
        return d3d12_max_reduce(x)
    device = _get_device()

    if axis is None:
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4I", numel, 1, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MAX_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


def mean_reduce(x, axis=None):
    """Mean reduction along axis."""
    if _BACKEND == "d3d12" and axis is None:
        return d3d12_mean(x)
    device = _get_device()

    if axis is None:
        out_shape = (1,)
    else:
        out_shape = list(x.shape)
        out_shape.pop(axis)
        out_shape = tuple(out_shape)

    out = WgpuTensor.zeros(out_shape, x.dtype)
    numel = x.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", numel, 1.0, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MEAN_REDUCE,
        [
            (x.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Matrix Operations
# ============================================================================

def matmul(a, b):
    """Matrix multiplication with tiled shared memory.

    Handles:
    - (M, K) @ (K, N) -> (M, N)
    - Batched: (..., M, K) @ (..., K, N) -> (..., M, N)
    """
    if _BACKEND == "d3d12":
        return d3d12_matmul(a, b)
    device = _get_device()

    # Extract dimensions
    m = a.shape[-2]
    k = a.shape[-1]
    n = b.shape[-1]

    # Output shape
    batch_shape = a.shape[:-2]
    out_shape = batch_shape + (m, n)
    out = WgpuTensor.zeros(out_shape, a.dtype)

    # Dispatch with 16x16 tiles
    workgroups_x = (n + 15) // 16
    workgroups_y = (m + 15) // 16

    params = struct.pack("4I", m, n, k, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MATMUL,
        [
            (a.buffer, "read"),
            (b.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x, workgroups_y),
    )
    return out


def matmul_add_relu(a, b, bias):
    """Fused matmul + add + relu in a single GPU dispatch.

    Computes: relu(a @ b + bias) and also returns pre_relu values for backward.
    Reduces 3 GPU dispatches to 1.

    Args:
        a: WgpuTensor (M, K)
        b: WgpuTensor (K, N)
        bias: WgpuTensor (M, N) — pre-broadcast bias

    Returns:
        (out, pre_relu): both WgpuTensor (M, N)
    """
    if _BACKEND == "d3d12":
        return d3d12_matmul_add_relu(a, b, bias)
    device = _get_device()
    m = a.shape[-2]
    k = a.shape[-1]
    n = b.shape[-1]

    out = WgpuTensor.zeros((m, n), a.dtype)
    pre_relu = WgpuTensor.zeros((m, n), a.dtype)

    workgroups_x = (n + 15) // 16
    workgroups_y = (m + 15) // 16

    params = struct.pack("4I", m, n, k, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MATMUL_ADD_RELU,
        [
            (a.buffer, "read"),
            (b.buffer, "read"),
            (bias.buffer, "read"),
            (out.buffer, "read_write"),
            (pre_relu.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x, workgroups_y),
    )
    return out, pre_relu


def matmul_add(a, b, bias):
    """Fused matmul + add in a single GPU dispatch.

    Computes: a @ b + bias
    Reduces 2 GPU dispatches to 1.

    Args:
        a: WgpuTensor (M, K)
        b: WgpuTensor (K, N)
        bias: WgpuTensor (M, N) — pre-broadcast bias

    Returns:
        WgpuTensor (M, N)
    """
    if _BACKEND == "d3d12":
        return d3d12_matmul_add(a, b, bias)
    device = _get_device()
    m = a.shape[-2]
    k = a.shape[-1]
    n = b.shape[-1]

    out = WgpuTensor.zeros((m, n), a.dtype)

    workgroups_x = (n + 15) // 16
    workgroups_y = (m + 15) // 16

    params = struct.pack("4I", m, n, k, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_MATMUL_ADD,
        [
            (a.buffer, "read"),
            (b.buffer, "read"),
            (bias.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x, workgroups_y),
    )
    return out


def transpose_2d(x):
    """Transpose last two dimensions."""
    if x.ndim < 2:
        raise ValueError("transpose_2d requires at least 2D tensor")
    if _BACKEND == "d3d12":
        return d3d12_transpose(x)

    device = _get_device()
    rows = x.shape[-2]
    cols = x.shape[-1]

    new_shape = x.shape[:-2] + (cols, rows)
    out = WgpuTensor.zeros(new_shape, x.dtype)

    workgroups_x = (cols + 15) // 16
    workgroups_y = (rows + 15) // 16

    params = struct.pack("4I", rows, cols, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_TRANSPOSE_2D,
        [(x.buffer, "read"), (out.buffer, "read_write"), (params_buffer, "uniform")],
        (workgroups_x, workgroups_y),
    )
    return out


# ============================================================================
# Functional API - Normalization & Softmax
# ============================================================================

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta.

    One workgroup per row (sequence position).
    """
    if _BACKEND == "d3d12":
        return d3d12_layer_norm(x, gamma, beta, eps)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)

    width = x.shape[-1]
    num_rows = x.numel() // width

    params = struct.pack("4f", width, eps, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    # One workgroup per row
    _dispatch_shader(
        device,
        WGSL_LAYER_NORM,
        [
            (x.buffer, "read"),
            (gamma.buffer, "read"),
            (beta.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return out


def softmax(x, axis=-1):
    """Stable softmax along last axis. One workgroup per row."""
    if _BACKEND == "d3d12":
        return d3d12_softmax(x)
    device = _get_device()
    out = WgpuTensor.zeros(x.shape, x.dtype)

    if axis == -1 or axis == x.ndim - 1:
        width = x.shape[-1]
        num_rows = x.numel() // width

        params = struct.pack("4I", width, num_rows, 0, 0)
        params_buffer = device.create_buffer_with_data(
            data=params,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # One workgroup per row
        _dispatch_shader(
            device,
            WGSL_SOFTMAX,
            [(x.buffer, "read"), (out.buffer, "read_write"), (params_buffer, "uniform")],
            (num_rows,),
        )
    else:
        raise NotImplementedError("softmax only supports axis=-1 for now")

    return out


# ============================================================================
# Functional API - Loss Functions
# ============================================================================

def cross_entropy(logits, targets):
    """Cross entropy loss: -target * log(p) - (1 - target) * log(1 - p)."""
    if _BACKEND == "d3d12":
        return d3d12_cross_entropy(logits, targets)
    device = _get_device()
    out = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256

    _dispatch_shader(
        device,
        WGSL_CROSS_ENTROPY,
        [(logits.buffer, "read"), (targets.buffer, "read"), (out.buffer, "read_write")],
        (workgroups_x,),
    )
    return out


def focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    """Focal binary cross entropy loss."""
    if _BACKEND == "d3d12":
        return d3d12_focal_bce(logits, targets, gamma, alpha)
    device = _get_device()
    out = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", gamma, alpha, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_FOCAL_BCE,
        [
            (logits.buffer, "read"),
            (targets.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Functional API - Embedding
# ============================================================================

def embedding_lookup(weight, indices):
    """Gather rows from weight matrix by indices."""
    if _BACKEND == "d3d12":
        return d3d12_embedding_lookup(weight, indices)
    device = _get_device()

    num_indices = indices.numel()
    embedding_dim = weight.shape[-1]
    out_shape = indices.shape + (embedding_dim,)
    out = WgpuTensor.zeros(out_shape, weight.dtype)

    total_elements = num_indices * embedding_dim
    workgroups_x = (total_elements + 255) // 256

    params = struct.pack("4I", num_indices, embedding_dim, 0, 0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_EMBEDDING,
        [
            (weight.buffer, "read"),
            (indices.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return out


# ============================================================================
# Backend Auto-Detection: D3D12 Native vs wgpu/Dozen
# ============================================================================

_BACKEND = "wgpu"

if os.environ.get("WGPU_FORCE_BACKEND") != "wgpu":
    try:
        from d3d12_tensor import (
            is_available as _d3d12_available,
            d3d12_init as _d3d12_init_fn,
            d3d12_get_adapter_name,
            D3D12Tensor,
            d3d12_matmul, d3d12_add, d3d12_sub, d3d12_mul,
            d3d12_relu, d3d12_gelu, d3d12_sigmoid, d3d12_tanh,
            d3d12_transpose, d3d12_scalar_mul, d3d12_neg,
            d3d12_matmul_add_relu, d3d12_matmul_add,
            d3d12_relu_backward, d3d12_gelu_backward,
            d3d12_softmax_backward, d3d12_layernorm_backward,
            d3d12_layer_norm, d3d12_softmax,
            d3d12_sum, d3d12_mean, d3d12_max_reduce,
            d3d12_cross_entropy, d3d12_focal_bce,
            d3d12_embedding_lookup,
            get_device_info as _d3d12_device_info,
        )
        if _d3d12_available():
            _d3d12_init_fn()
            _BACKEND = "d3d12"
    except (ImportError, RuntimeError):
        pass


def get_backend():
    """Return active backend name: 'd3d12' or 'wgpu'."""
    return _BACKEND
