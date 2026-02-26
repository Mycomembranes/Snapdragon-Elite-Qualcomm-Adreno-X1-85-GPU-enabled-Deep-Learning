@group(0) @binding(0)
var<storage, read> data: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;
@group(0) @binding(2)
var<uniform> params: vec4<u32>;

var<workgroup> wg_data: array<f32, 256>;

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
    wg_data[lid_x] = sum_val;
    workgroupBarrier();

    var stride_val = 128u;
    loop {
        if (stride_val == 0u) { break; }
        if (lid_x < stride_val) {
            wg_data[lid_x] = wg_data[lid_x] + wg_data[lid_x + stride_val];
        }
        workgroupBarrier();
        stride_val = stride_val >> 1u;
    }

    if (lid_x == 0u) {
        let out_idx = gid.x / 256u;
        if (out_idx < arrayLength(&out)) {
            out[out_idx] = wg_data[0];
        }
    }
}
