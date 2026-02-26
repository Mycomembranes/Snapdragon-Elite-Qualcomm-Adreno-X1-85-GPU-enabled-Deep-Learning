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
