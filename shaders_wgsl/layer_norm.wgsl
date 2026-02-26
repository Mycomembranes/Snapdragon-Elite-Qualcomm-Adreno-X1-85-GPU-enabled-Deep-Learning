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
