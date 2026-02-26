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
