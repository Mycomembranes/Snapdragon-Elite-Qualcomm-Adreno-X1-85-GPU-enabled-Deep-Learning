@group(0) @binding(0)
var<storage, read> logits: array<f32>;
@group(0) @binding(1)
var<storage, read> targets: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx < arrayLength(&out)) {
        let logit = logits[idx];
        let tgt = targets[idx];
        let gamma = params.x;
        let alpha = params.y;

        let p = 1.0 / (1.0 + exp(-logit));
        let ce = -tgt * log(p + 1e-6) - (1.0 - tgt) * log(1.0 - p + 1e-6);
        let p_t = select(1.0 - p, p, tgt > 0.5);
        let focal_weight = pow(1.0 - p_t, gamma);
        let focal_loss = alpha * (1.0 - alpha) * focal_weight * ce;

        out[idx] = focal_loss;
    }
}
