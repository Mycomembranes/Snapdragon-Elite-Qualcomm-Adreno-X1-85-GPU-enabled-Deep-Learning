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
