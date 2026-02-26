@group(0) @binding(0)
var<storage, read> x: array<f32>;
@group(0) @binding(1)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let val = x[idx];
        let cdf = 0.5 * (1.0 + tanh(
            sqrt(2.0 / 3.14159265359) * (val + 0.044715 * val * val * val)
        ));
        out[idx] = val * cdf;
    }
}
