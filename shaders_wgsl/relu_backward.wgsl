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
