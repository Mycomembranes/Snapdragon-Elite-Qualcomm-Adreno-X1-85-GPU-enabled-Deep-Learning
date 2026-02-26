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
