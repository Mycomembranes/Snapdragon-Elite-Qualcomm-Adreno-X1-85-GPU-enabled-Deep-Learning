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
