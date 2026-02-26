@group(0) @binding(0)
var<storage, read> a: array<f32>;
@group(0) @binding(1)
var<storage, read> b: array<f32>;
@group(0) @binding(2)
var<storage, read_write> out: array<f32>;
@group(0) @binding(3)
var<uniform> params: vec4<u32>;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let m = params.x;
    let n = params.y;
    let k = params.z;
    let lx = lid.x;
    let ly = lid.y;

    let row = wid.y * 16u + ly;
    let col = wid.x * 16u + lx;

    var result = 0.0;

    var tile_idx = 0u;
    loop {
        if (tile_idx >= k) { break; }

        let a_col = tile_idx + lx;
        let a_idx = row * k + a_col;
        tile_a[ly * 16u + lx] = select(0.0, a[a_idx], row < m && a_col < k);

        let b_row = tile_idx + ly;
        let b_idx = b_row * n + col;
        tile_b[ly * 16u + lx] = select(0.0, b[b_idx], b_row < k && col < n);

        workgroupBarrier();

        for (var i = 0u; i < 16u; i = i + 1u) {
            result = result + tile_a[ly * 16u + i] * tile_b[i * 16u + lx];
        }

        workgroupBarrier();
        tile_idx = tile_idx + 16u;
    }

    if (row < m && col < n) {
        out[row * n + col] = result;
    }
}
