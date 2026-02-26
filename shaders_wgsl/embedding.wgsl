@group(0) @binding(0)
var<storage, read> weight: array<f32>;
@group(0) @binding(1)
var<storage, read> indices: array<u32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_indices = params.x;
    let embedding_dim = params.y;

    if (idx < num_indices * embedding_dim) {
        let i = idx / embedding_dim;
        let j = idx % embedding_dim;
        let word_idx = indices[i];
        let weight_idx = word_idx * embedding_dim + j;
        out[idx] = weight[weight_idx];
    }
}
