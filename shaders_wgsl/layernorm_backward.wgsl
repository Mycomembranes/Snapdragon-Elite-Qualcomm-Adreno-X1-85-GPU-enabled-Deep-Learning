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
