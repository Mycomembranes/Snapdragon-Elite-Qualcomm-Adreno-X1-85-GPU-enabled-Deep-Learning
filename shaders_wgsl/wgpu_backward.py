"""Chain rule backward passes and numerical integration for WGSL GPU tensors.

This module extends wgpu_tensor.py with:

1. **Chain Rule Backward Passes** -- For every differentiable forward operation,
   a backward function that computes dL/dx given dL/dy (upstream gradient) using
   the chain rule: dL/dx = dL/dy * dy/dx.

   Some backward passes require new WGSL compute shaders (sigmoid, tanh, cross
   entropy, focal BCE, broadcast fill). Others compose existing forward shaders
   from wgpu_tensor.py (matmul backward uses matmul + transpose, mul backward
   uses mul, etc.).

2. **Numerical Integration** -- GPU-accelerated trapezoidal rule, Simpson's rule,
   and cumulative integration for computing areas, CDFs, and integrals on GPU.

Formulas:
    sigmoid_backward:  grad * sigma * (1 - sigma)        where sigma = sigmoid(x)
    tanh_backward:     grad * (1 - tanh(x)^2)
    relu_backward:     grad * (x > 0)                    [existing shader]
    gelu_backward:     grad * gelu'(x)                   [existing shader]
    matmul_backward:   grad_A = grad @ B^T,  grad_B = A^T @ grad
    add_backward:      grad_a = grad,  grad_b = grad
    sub_backward:      grad_a = grad,  grad_b = -grad
    mul_backward:      grad_a = grad * b,  grad_b = grad * a
    scalar_mul_backward: grad = grad * scalar
    neg_backward:      grad = -grad
    cross_entropy_backward: sigmoid(logit) - target
    trapezoid:         sum( (f[i] + f[i+1]) / 2 * dx )
    simpson:           dx/3 * (f[0] + 4*f[1] + 2*f[2] + ... + f[n])
    cumulative_trapezoid: out[i] = out[i-1] + (f[i-1] + f[i]) / 2 * dx
"""

import struct
import numpy as np
import wgpu

from wgpu_tensor import (
    _get_device,
    _dispatch_shader,
    WgpuTensor,
    # Forward ops used for composition in backward passes
    add,
    sub,
    mul,
    neg,
    scalar_mul,
    matmul,
    transpose_2d,
    sum_reduce,
    # Existing backward passes -- re-exported
    relu_backward,
    gelu_backward,
    softmax_backward,
    layernorm_backward,
)

# Re-export existing backward functions for convenience
__all__ = [
    # Re-exported from wgpu_tensor
    "relu_backward",
    "gelu_backward",
    "softmax_backward",
    "layernorm_backward",
    # New backward passes
    "sigmoid_backward",
    "tanh_backward",
    "matmul_backward",
    "add_backward",
    "sub_backward",
    "mul_backward",
    "scalar_mul_backward",
    "neg_backward",
    "transpose_backward",
    "matmul_add_backward",
    "matmul_add_relu_backward",
    "cross_entropy_backward",
    "focal_bce_backward",
    "sum_reduce_backward",
    "mean_reduce_backward",
    "layer_norm_backward_full",
    "embedding_backward",
    # Integration
    "trapezoid",
    "simpson",
    "cumulative_trapezoid",
    "trapezoid_2d",
]


# ============================================================================
# New WGSL Backward Shaders
# ============================================================================

WGSL_SIGMOID_BACKWARD = """
// Sigmoid backward: grad_in = grad_out * sigma * (1 - sigma)
// Uses saved sigmoid output (more efficient than recomputing from x).
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> sigmoid_out: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        let s = sigmoid_out[idx];
        grad_in[idx] = grad_out[idx] * s * (1.0 - s);
    }
}
"""

WGSL_TANH_BACKWARD = """
// Tanh backward: grad_in = grad_out * (1 - tanh_out^2)
// Uses saved tanh output.
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> tanh_out: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        let t = tanh_out[idx];
        grad_in[idx] = grad_out[idx] * (1.0 - t * t);
    }
}
"""

WGSL_CROSS_ENTROPY_BACKWARD = """
// BCE with logits backward: dL/d(logit) = sigmoid(logit) - target
// This is the derivative of -t * log(sigma(z)) - (1-t) * log(1 - sigma(z))
// which simplifies to sigma(z) - t.
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        let p = 1.0 / (1.0 + exp(-logits[idx]));
        grad_in[idx] = p - targets[idx];
    }
}
"""

WGSL_FOCAL_BCE_BACKWARD = """
// Focal BCE backward with gamma and alpha parameters.
// L = alpha * (1-alpha) * (1 - p_t)^gamma * CE
// where p_t = p if t=1, (1-p) if t=0, p = sigmoid(logit)
//
// Full derivative via product rule:
// dL/dz = alpha*(1-alpha) * [
//     -gamma * (1-p_t)^(gamma-1) * dp_t/dz * CE
//     + (1-p_t)^gamma * dCE/dz
// ]
// where dCE/dz = p - t, dp_t/dz = p*(1-p) if t=1, -p*(1-p) if t=0
@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_in: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_in)) {
        let logit = logits[idx];
        let t = targets[idx];
        let gamma = params.x;
        let alpha = params.y;

        let p = 1.0 / (1.0 + exp(-logit));
        let ce = -t * log(p + 1e-6) - (1.0 - t) * log(1.0 - p + 1e-6);
        let p_t = select(1.0 - p, p, t > 0.5);
        let dp_t_dz = select(-p * (1.0 - p), p * (1.0 - p), t > 0.5);
        let one_minus_pt = 1.0 - p_t;

        // Product rule: d/dz[ (1-p_t)^gamma * CE ]
        let focal_weight = pow(one_minus_pt, gamma);
        let focal_weight_deriv = -gamma * pow(one_minus_pt, gamma - 1.0) * dp_t_dz;
        let dce_dz = p - t;

        grad_in[idx] = alpha * (1.0 - alpha) * (focal_weight_deriv * ce + focal_weight * dce_dz);
    }
}
"""

WGSL_BROADCAST_FILL = """
// Fill output array with scalar value * scale.
// Used for sum_reduce_backward (scale=1.0) and mean_reduce_backward (scale=1/n).
// params.x = scale factor
@group(0) @binding(0)
var<storage, read> grad_scalar: array<f32>;
@group(0) @binding(1)
var<storage, read_write> grad_out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_out)) {
        grad_out[idx] = grad_scalar[0] * params.x;
    }
}
"""

WGSL_MATMUL_ADD_RELU_BACKWARD = """
// Fused backward for relu(A @ W + bias):
// Step 1: Apply ReLU mask to grad_out: grad_relu = grad_out * (pre_relu > 0)
// This shader only does the ReLU masking step. The matmul backward
// (grad_A = grad_relu @ W^T, grad_W = A^T @ grad_relu) is done with
// existing matmul + transpose shaders, and grad_bias = grad_relu.
@group(0) @binding(0)
var<storage, read> grad_out: array<f32>;
@group(0) @binding(1)
var<storage, read> pre_relu: array<f32>;
@group(0) @binding(2)
var<storage, read_write> grad_relu: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&grad_relu)) {
        grad_relu[idx] = select(0.0, grad_out[idx], pre_relu[idx] > 0.0);
    }
}
"""


# ============================================================================
# New WGSL Integration Shaders
# ============================================================================

WGSL_TRAPEZOID_INTEGRATE = """
// Trapezoidal rule: integral = sum( (f[i] + f[i+1]) / 2 * dx )
// Each workgroup handles a contiguous block of 256 panels.
// params.x = n (number of points), params.y = dx.
// For n points, there are n-1 panels.
@group(0) @binding(0)
var<storage, read> f: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = u32(params.x);
    let dx = params.y;
    let tid = lid.x;
    let num_panels = n - 1u;

    // Global panel index for this thread
    let panel_idx = wid.x * 256u + tid;

    var local_sum = 0.0;
    if (panel_idx < num_panels) {
        local_sum = (f[panel_idx] + f[panel_idx + 1u]) * 0.5 * dx;
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out[wid.x] = sdata[0];
    }
}
"""

WGSL_SIMPSON_INTEGRATE = """
// Simpson's 1/3 rule: integral = dx/3 * (f[0] + 4*f[1] + 2*f[2] + 4*f[3] + ... + f[n-1])
// Requires odd number of points (even number of panels).
// params.x = n (number of points), params.y = dx
@group(0) @binding(0)
var<storage, read> f: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = u32(params.x);
    let dx = params.y;
    let tid = lid.x;

    // Each thread computes weighted sum for a stride of points
    var local_sum = 0.0;
    var i = tid;
    loop {
        if (i >= n) { break; }
        var weight = 1.0;
        if (i == 0u || i == n - 1u) {
            weight = 1.0;
        } else if (i % 2u == 1u) {
            weight = 4.0;
        } else {
            weight = 2.0;
        }
        local_sum = local_sum + weight * f[i];
        i = i + 256u;
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out[wid.x] = sdata[0] * dx / 3.0;
    }
}
"""

WGSL_CUMULATIVE_TRAPEZOID = """
// Cumulative trapezoidal integration (sequential prefix sum).
// out[i] = sum_{j=0}^{i} (f[j] + f[j+1]) / 2 * dx  for i = 0..n-2
// This is a sequential scan (single thread) -- suitable for moderate n.
// For large n, a parallel prefix sum would be needed.
// params.x = n (number of input points), params.y = dx
@group(0) @binding(0)
var<storage, read> f: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let n = u32(params.x);
    let dx = params.y;
    let in_offset = row * n;
    let out_offset = row * (n - 1u);

    var running_sum = 0.0;
    for (var i = 0u; i < n - 1u; i = i + 1u) {
        running_sum = running_sum + (f[in_offset + i] + f[in_offset + i + 1u]) * 0.5 * dx;
        out[out_offset + i] = running_sum;
    }
}
"""

WGSL_ROMBERG_STEP = """
// Romberg extrapolation refinement step.
// Given two estimates T[k-1] (coarser) and T[k] (finer) with step ratio 4^k:
// R[k] = (4^k * T[k] - T[k-1]) / (4^k - 1)
// params.x = 4^k (the extrapolation factor)
@group(0) @binding(0)
var<storage, read> t_coarse: array<f32>;
@group(0) @binding(1)
var<storage, read> t_fine: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let factor = params.x;
        out[idx] = (factor * t_fine[idx] - t_coarse[idx]) / (factor - 1.0);
    }
}
"""

WGSL_TRAPEZOID_2D = """
// Trapezoidal integration along the last axis for a 2D tensor.
// One workgroup per row. params.x = width (number of points per row),
// params.y = dx
@group(0) @binding(0)
var<storage, read> f: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<f32>;

var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let width = u32(params.x);
    let dx = params.y;
    let row = wid.x;
    let tid = lid.x;
    let row_offset = row * width;

    // Each thread sums a stride of panels in this row
    var local_sum = 0.0;
    var i = tid;
    loop {
        if (i >= width - 1u) { break; }
        local_sum = local_sum + (f[row_offset + i] + f[row_offset + i + 1u]) * 0.5 * dx;
        i = i + 256u;
    }
    sdata[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction
    var s = 128u;
    loop {
        if (s == 0u) { break; }
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        out[row] = sdata[0];
    }
}
"""


# ============================================================================
# Python Backward Wrapper Functions -- New WGSL Shaders
# ============================================================================

def sigmoid_backward(grad_out, sigmoid_output):
    """Sigmoid backward: grad_in = grad_out * sigma * (1 - sigma).

    Args:
        grad_out: WgpuTensor, upstream gradient
        sigmoid_output: WgpuTensor, saved output from sigmoid forward pass

    Returns:
        grad_in: WgpuTensor, gradient w.r.t. sigmoid input
    """
    device = _get_device()
    grad_in = WgpuTensor.zeros(grad_out.shape, grad_out.dtype)
    numel = grad_out.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_SIGMOID_BACKWARD,
        [
            (grad_out.buffer, "read"),
            (sigmoid_output.buffer, "read"),
            (grad_in.buffer, "read_write"),
        ],
        (workgroups_x,),
    )
    return grad_in


def tanh_backward(grad_out, tanh_output):
    """Tanh backward: grad_in = grad_out * (1 - tanh_out^2).

    Args:
        grad_out: WgpuTensor, upstream gradient
        tanh_output: WgpuTensor, saved output from tanh forward pass

    Returns:
        grad_in: WgpuTensor, gradient w.r.t. tanh input
    """
    device = _get_device()
    grad_in = WgpuTensor.zeros(grad_out.shape, grad_out.dtype)
    numel = grad_out.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_TANH_BACKWARD,
        [
            (grad_out.buffer, "read"),
            (tanh_output.buffer, "read"),
            (grad_in.buffer, "read_write"),
        ],
        (workgroups_x,),
    )
    return grad_in


def cross_entropy_backward(logits, targets):
    """BCE with logits backward: grad = sigmoid(logit) - target.

    This is the gradient of binary cross entropy with logits:
    L = -t * log(sigma(z)) - (1-t) * log(1 - sigma(z))
    dL/dz = sigma(z) - t

    Args:
        logits: WgpuTensor, raw logits (not probabilities)
        targets: WgpuTensor, binary targets (0 or 1)

    Returns:
        grad_logits: WgpuTensor, gradient w.r.t. logits
    """
    device = _get_device()
    grad_in = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_CROSS_ENTROPY_BACKWARD,
        [
            (logits.buffer, "read"),
            (targets.buffer, "read"),
            (grad_in.buffer, "read_write"),
        ],
        (workgroups_x,),
    )
    return grad_in


def focal_bce_backward(logits, targets, gamma=2.0, alpha=0.25):
    """Focal BCE backward with gamma and alpha parameters.

    Computes the full derivative of focal binary cross entropy loss:
    L = alpha * (1-alpha) * (1-p_t)^gamma * CE(p, t)

    Args:
        logits: WgpuTensor, raw logits
        targets: WgpuTensor, binary targets
        gamma: float, focusing parameter (default 2.0)
        alpha: float, class balance weight (default 0.25)

    Returns:
        grad_logits: WgpuTensor, gradient w.r.t. logits
    """
    device = _get_device()
    grad_in = WgpuTensor.zeros(logits.shape, logits.dtype)
    numel = logits.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", gamma, alpha, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_FOCAL_BCE_BACKWARD,
        [
            (logits.buffer, "read"),
            (targets.buffer, "read"),
            (grad_in.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return grad_in


def _broadcast_fill(grad_scalar, output_shape, scale=1.0):
    """Broadcast a scalar gradient to a full tensor, scaled by `scale`.

    Used internally by sum_reduce_backward (scale=1.0) and
    mean_reduce_backward (scale=1/n).

    Args:
        grad_scalar: WgpuTensor with shape (1,), the scalar gradient
        output_shape: tuple, shape of the output gradient tensor
        scale: float, multiplicative scale factor

    Returns:
        grad_out: WgpuTensor of shape output_shape
    """
    device = _get_device()
    grad_out = WgpuTensor.zeros(output_shape, "float32")
    numel = grad_out.numel()
    workgroups_x = (numel + 255) // 256

    params = struct.pack("4f", scale, 0.0, 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_BROADCAST_FILL,
        [
            (grad_scalar.buffer, "read"),
            (grad_out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (workgroups_x,),
    )
    return grad_out


def _relu_mask_grad(grad_out, pre_relu):
    """Apply ReLU mask to gradient: grad_relu = grad_out * (pre_relu > 0).

    Args:
        grad_out: WgpuTensor, upstream gradient
        pre_relu: WgpuTensor, pre-activation values from forward pass

    Returns:
        grad_relu: WgpuTensor, masked gradient
    """
    device = _get_device()
    grad_relu = WgpuTensor.zeros(grad_out.shape, grad_out.dtype)
    numel = grad_out.numel()
    workgroups_x = (numel + 255) // 256
    _dispatch_shader(
        device,
        WGSL_MATMUL_ADD_RELU_BACKWARD,
        [
            (grad_out.buffer, "read"),
            (pre_relu.buffer, "read"),
            (grad_relu.buffer, "read_write"),
        ],
        (workgroups_x,),
    )
    return grad_relu


# ============================================================================
# Python Backward Wrapper Functions -- Composed from Existing Forward Ops
# ============================================================================

def matmul_backward(grad_out, a, b):
    """Matmul backward: grad_A = grad @ B^T, grad_B = A^T @ grad.

    For y = A @ B, the chain rule gives:
        dL/dA = dL/dy @ B^T
        dL/dB = A^T @ dL/dy

    Args:
        grad_out: WgpuTensor (M, N), upstream gradient
        a: WgpuTensor (M, K), left input saved from forward
        b: WgpuTensor (K, N), right input saved from forward

    Returns:
        (grad_a, grad_b): tuple of WgpuTensors
    """
    b_t = transpose_2d(b)      # (N, K)
    a_t = transpose_2d(a)      # (K, M)
    grad_a = matmul(grad_out, b_t)   # (M, N) @ (N, K) = (M, K)
    grad_b = matmul(a_t, grad_out)   # (K, M) @ (M, N) = (K, N)
    return grad_a, grad_b


def add_backward(grad_out):
    """Add backward: gradient passes through identically to both inputs.

    For y = a + b: dL/da = dL/dy, dL/db = dL/dy

    Args:
        grad_out: WgpuTensor, upstream gradient

    Returns:
        (grad_a, grad_b): both are grad_out (same reference, no copy needed
        since downstream ops create new buffers)
    """
    return grad_out, grad_out


def sub_backward(grad_out):
    """Sub backward: identity for a, negation for b.

    For y = a - b: dL/da = dL/dy, dL/db = -dL/dy

    Args:
        grad_out: WgpuTensor, upstream gradient

    Returns:
        (grad_a, grad_b): grad_a = grad_out, grad_b = -grad_out
    """
    return grad_out, neg(grad_out)


def mul_backward(grad_out, a, b):
    """Mul backward: grad_a = grad * b, grad_b = grad * a.

    For y = a * b (element-wise): dL/da = dL/dy * b, dL/db = dL/dy * a

    Args:
        grad_out: WgpuTensor, upstream gradient
        a: WgpuTensor, left input saved from forward
        b: WgpuTensor, right input saved from forward

    Returns:
        (grad_a, grad_b): tuple of WgpuTensors
    """
    grad_a = mul(grad_out, b)
    grad_b = mul(grad_out, a)
    return grad_a, grad_b


def scalar_mul_backward(grad_out, s):
    """Scalar mul backward: grad_in = grad_out * scalar.

    For y = a * s: dL/da = dL/dy * s

    Args:
        grad_out: WgpuTensor, upstream gradient
        s: float, scalar multiplier saved from forward

    Returns:
        grad_in: WgpuTensor
    """
    return scalar_mul(grad_out, s)


def neg_backward(grad_out):
    """Neg backward: grad_in = -grad_out.

    For y = -a: dL/da = -dL/dy

    Args:
        grad_out: WgpuTensor, upstream gradient

    Returns:
        grad_in: WgpuTensor
    """
    return neg(grad_out)


def transpose_backward(grad_out):
    """Transpose backward: transpose the gradient.

    For y = A^T: dL/dA = (dL/dy)^T

    Args:
        grad_out: WgpuTensor, upstream gradient

    Returns:
        grad_in: WgpuTensor
    """
    return transpose_2d(grad_out)


def matmul_add_backward(grad_out, input_tensor, weight):
    """Backward for fused matmul+add: y = input @ weight + bias.

    Args:
        grad_out: WgpuTensor (M, N), upstream gradient
        input_tensor: WgpuTensor (M, K), saved from forward
        weight: WgpuTensor (K, N), saved from forward

    Returns:
        (grad_input, grad_weight, grad_bias):
            grad_input: WgpuTensor (M, K)
            grad_weight: WgpuTensor (K, N)
            grad_bias: WgpuTensor (M, N) -- same as grad_out since d(x+b)/db = 1
    """
    weight_t = transpose_2d(weight)       # (N, K)
    input_t = transpose_2d(input_tensor)  # (K, M)
    grad_input = matmul(grad_out, weight_t)    # (M, N) @ (N, K) = (M, K)
    grad_weight = matmul(input_t, grad_out)    # (K, M) @ (M, N) = (K, N)
    grad_bias = grad_out                        # (M, N)
    return grad_input, grad_weight, grad_bias


def matmul_add_relu_backward(grad_out, pre_relu, input_tensor, weight):
    """Backward for fused matmul+add+relu: y = relu(input @ weight + bias).

    Step 1: Apply ReLU mask to upstream gradient.
    Step 2: Compute matmul backward with the masked gradient.

    Args:
        grad_out: WgpuTensor (M, N), upstream gradient
        pre_relu: WgpuTensor (M, N), pre-activation values saved from forward
        input_tensor: WgpuTensor (M, K), saved from forward
        weight: WgpuTensor (K, N), saved from forward

    Returns:
        (grad_input, grad_weight, grad_bias):
            grad_input: WgpuTensor (M, K)
            grad_weight: WgpuTensor (K, N)
            grad_bias: WgpuTensor (M, N)
    """
    # Step 1: ReLU backward -- mask out gradients where pre_relu <= 0
    grad_relu = _relu_mask_grad(grad_out, pre_relu)

    # Step 2: Matmul+add backward with the masked gradient
    grad_input, grad_weight, grad_bias = matmul_add_backward(
        grad_relu, input_tensor, weight
    )
    return grad_input, grad_weight, grad_bias


def sum_reduce_backward(grad_out, input_shape):
    """Sum reduce backward: broadcast scalar gradient to input shape.

    For y = sum(x): dL/dx_i = dL/dy for all i

    Args:
        grad_out: WgpuTensor with shape (1,), upstream gradient
        input_shape: tuple, shape of the original input tensor

    Returns:
        grad_in: WgpuTensor of shape input_shape, filled with grad_out value
    """
    return _broadcast_fill(grad_out, input_shape, scale=1.0)


def mean_reduce_backward(grad_out, input_shape):
    """Mean reduce backward: broadcast scalar gradient / n to input shape.

    For y = mean(x) = sum(x) / n: dL/dx_i = dL/dy / n for all i

    Args:
        grad_out: WgpuTensor with shape (1,), upstream gradient
        input_shape: tuple, shape of the original input tensor

    Returns:
        grad_in: WgpuTensor of shape input_shape
    """
    n = 1
    for s in input_shape:
        n *= s
    return _broadcast_fill(grad_out, input_shape, scale=1.0 / n)


def layer_norm_backward_full(grad_out, x, gamma, eps=1e-5):
    """Full LayerNorm backward: computes grad_input, grad_gamma, grad_beta.

    The existing layernorm_backward() only computes grad_input on GPU.
    This function additionally computes grad_gamma and grad_beta on CPU
    (since WGSL lacks atomic f32 add for cross-row accumulation).

    grad_gamma[j] = sum_rows( grad_out[i,j] * x_hat[i,j] )
    grad_beta[j]  = sum_rows( grad_out[i,j] )
    where x_hat = (x - mean) / sqrt(var + eps)

    Args:
        grad_out: WgpuTensor (..., D), upstream gradient
        x: WgpuTensor (..., D), input saved from forward
        gamma: WgpuTensor (D,), scale parameter
        eps: float, epsilon for numerical stability

    Returns:
        (grad_input, grad_gamma, grad_beta):
            grad_input: WgpuTensor (..., D) -- computed on GPU
            grad_gamma: WgpuTensor (D,) -- computed on CPU
            grad_beta: WgpuTensor (D,) -- computed on CPU
    """
    # grad_input on GPU using existing shader
    grad_input = layernorm_backward(grad_out, x, gamma, eps)

    # grad_gamma and grad_beta on CPU (need x_hat which requires mean/var)
    x_np = x.numpy()
    grad_out_np = grad_out.numpy()

    width = x_np.shape[-1]
    x_flat = x_np.reshape(-1, width)
    grad_flat = grad_out_np.reshape(-1, width)

    mean = x_flat.mean(axis=-1, keepdims=True)
    var = x_flat.var(axis=-1, keepdims=True)
    x_hat = (x_flat - mean) / np.sqrt(var + eps)

    grad_gamma_np = (grad_flat * x_hat).sum(axis=0).astype(np.float32)
    grad_beta_np = grad_flat.sum(axis=0).astype(np.float32)

    grad_gamma = WgpuTensor.from_numpy(grad_gamma_np)
    grad_beta = WgpuTensor.from_numpy(grad_beta_np)

    return grad_input, grad_gamma, grad_beta


def embedding_backward(grad_out, indices, num_embeddings, dim):
    """Embedding backward: scatter-add gradients back to weight matrix.

    For y = weight[indices], the gradient w.r.t. weight is a scatter-add:
    grad_weight[idx] += grad_out[i] for each i where indices[i] == idx.

    Uses CPU fallback with np.add.at for correctness with duplicate indices,
    since WGSL lacks atomic f32 add.

    Args:
        grad_out: WgpuTensor (num_indices, dim), upstream gradient
        indices: WgpuTensor (num_indices,) with dtype uint32, token indices
        num_embeddings: int, vocabulary size (number of rows in weight matrix)
        dim: int, embedding dimension

    Returns:
        grad_weight: WgpuTensor (num_embeddings, dim)
    """
    grad_out_np = grad_out.numpy()
    indices_np = indices.numpy().astype(np.int64).ravel()

    grad_weight_np = np.zeros((num_embeddings, dim), dtype=np.float32)
    np.add.at(grad_weight_np, indices_np, grad_out_np.reshape(-1, dim))

    return WgpuTensor.from_numpy(grad_weight_np)


# ============================================================================
# Numerical Integration Wrappers
# ============================================================================

def trapezoid(y, dx=1.0):
    """Trapezoidal rule integration on GPU.

    Computes the definite integral of y using the trapezoidal rule:
        integral = sum( (y[i] + y[i+1]) / 2 * dx ) for i = 0..n-2

    For 1D tensors, returns a scalar. Uses one trapezoid pass per workgroup,
    then reduces partial sums with sum_reduce.

    Args:
        y: WgpuTensor (n,), function values at equally-spaced points
        dx: float, spacing between points (default 1.0)

    Returns:
        result: WgpuTensor (1,), the computed integral
    """
    device = _get_device()
    n = y.shape[0]
    if n < 2:
        return WgpuTensor.zeros((1,), "float32")

    num_panels = n - 1
    num_wg = (num_panels + 255) // 256

    out = WgpuTensor.zeros((max(num_wg, 1),), "float32")

    params = struct.pack("4f", float(n), float(dx), 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_TRAPEZOID_INTEGRATE,
        [
            (y.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_wg,),
    )

    if num_wg == 1:
        return out.reshape((1,))

    # Multiple workgroups produced partial sums -- sum on CPU (tiny array)
    partial_sums = out.numpy()
    total = np.array([partial_sums.sum()], dtype=np.float32)
    return WgpuTensor.from_numpy(total)


def simpson(y, dx=1.0):
    """Simpson's 1/3 rule integration on GPU.

    Computes the definite integral using Simpson's rule:
        integral = dx/3 * (y[0] + 4*y[1] + 2*y[2] + 4*y[3] + ... + y[n-1])

    Requires an odd number of points (even number of panels). If n is even,
    the last panel uses the trapezoidal rule as a fallback.

    Args:
        y: WgpuTensor (n,), function values at equally-spaced points
        dx: float, spacing between points (default 1.0)

    Returns:
        result: WgpuTensor (1,), the computed integral
    """
    device = _get_device()
    n = y.shape[0]
    if n < 2:
        return WgpuTensor.zeros((1,), "float32")

    # Simpson's needs odd number of points
    effective_n = n if n % 2 == 1 else n - 1

    out = WgpuTensor.zeros((1,), "float32")

    params = struct.pack("4f", float(effective_n), float(dx), 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_SIMPSON_INTEGRATE,
        [
            (y.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (1,),
    )

    # If n was even, add the last panel using trapezoidal rule on CPU
    if n % 2 == 0 and n >= 2:
        y_np = y.numpy()
        last_panel = (y_np[-2] + y_np[-1]) * 0.5 * dx
        result_np = out.numpy().ravel()
        result_np[0] += last_panel
        out = WgpuTensor.from_numpy(result_np.reshape(1))

    return out


def cumulative_trapezoid(y, dx=1.0):
    """Cumulative trapezoidal integration on GPU.

    Computes running integral: out[i] = sum_{j=0}^{i} (y[j] + y[j+1]) / 2 * dx

    For 1D input of length n, returns tensor of length n-1.
    For 2D input (batch, n), returns (batch, n-1) with integration along last axis.

    Args:
        y: WgpuTensor (n,) or (batch, n), function values
        dx: float, spacing between points (default 1.0)

    Returns:
        result: WgpuTensor (n-1,) or (batch, n-1)
    """
    device = _get_device()

    if y.ndim == 1:
        n = y.shape[0]
        if n < 2:
            return WgpuTensor.zeros((0,), "float32")
        out = WgpuTensor.zeros((n - 1,), "float32")
        num_rows = 1
    elif y.ndim == 2:
        batch, n = y.shape
        if n < 2:
            return WgpuTensor.zeros((batch, 0), "float32")
        out = WgpuTensor.zeros((batch, n - 1), "float32")
        num_rows = batch
    else:
        raise ValueError("cumulative_trapezoid supports 1D and 2D tensors")

    params = struct.pack("4f", float(n), float(dx), 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_CUMULATIVE_TRAPEZOID,
        [
            (y.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return out


def trapezoid_2d(y, dx=1.0, axis=-1):
    """Trapezoidal integration along the last axis of a 2D tensor.

    One workgroup per row with shared memory tree reduction.

    Args:
        y: WgpuTensor (num_rows, width), function values
        dx: float, spacing between points (default 1.0)
        axis: int, must be -1 (last axis)

    Returns:
        result: WgpuTensor (num_rows,), integral for each row
    """
    if axis not in (-1, y.ndim - 1):
        raise NotImplementedError("trapezoid_2d only supports axis=-1")
    if y.ndim != 2:
        raise ValueError("trapezoid_2d requires 2D input")

    device = _get_device()
    num_rows, width = y.shape
    out = WgpuTensor.zeros((num_rows,), "float32")

    params = struct.pack("4f", float(width), float(dx), 0.0, 0.0)
    params_buffer = device.create_buffer_with_data(
        data=params,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
    )

    _dispatch_shader(
        device,
        WGSL_TRAPEZOID_2D,
        [
            (y.buffer, "read"),
            (out.buffer, "read_write"),
            (params_buffer, "uniform"),
        ],
        (num_rows,),
    )
    return out
