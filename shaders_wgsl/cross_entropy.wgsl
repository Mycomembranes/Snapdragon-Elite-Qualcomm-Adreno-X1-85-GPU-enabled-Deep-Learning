@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let log_p = logits[idx];
        let tgt = targets[idx];
        out[idx] = -tgt * log_p - (1.0 - tgt) * log(1.0 - exp(log_p) + 1e-6);
    }
}
