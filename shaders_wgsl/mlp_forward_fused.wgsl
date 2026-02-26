// Fused 3-layer MLP: (N, 58) -> relu(256) -> relu(128) -> (N, 1)
// Uses chain rule: each sample computed end-to-end in one workgroup.
// Eliminates ALL intermediate buffer allocations and GPU syncs.
//
// Bindings:
//   0: input       (N, 58)       read
//   1: weights     packed        read   [W0(58*256) + W2(256*128) + W4(128)]
//   2: biases      packed        read   [b0(256) + b2(128) + b4(1)]
//   3: output      (N,)          rw     logits
//   4: caches      packed        rw     [pre0(N*256) + h0(N*256) + pre1(N*128) + h1(N*128)]
//   5: masks       packed        read   [mask0(N*256) + mask1(N*128)]  (1.0 if no dropout)
//   6: params      uniform       [N, 0, 0, 0]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> caches: array<f32>;
@group(0) @binding(5) var<storage, read> masks: array<f32>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;

// Hardcoded architecture constants
const DIM_IN: u32 = 58u;
const DIM_H0: u32 = 256u;
const DIM_H1: u32 = 128u;
// Weight offsets in packed buffer
const W0_OFF: u32 = 0u;           // 58 * 256 = 14848
const W2_OFF: u32 = 14848u;       // 256 * 128 = 32768
const W4_OFF: u32 = 47616u;       // 128
// Bias offsets in packed buffer
const B0_OFF: u32 = 0u;
const B2_OFF: u32 = 256u;
const B4_OFF: u32 = 384u;

var<workgroup> act_a: array<f32, 256>;  // layer 0 output (ping)
var<workgroup> act_b: array<f32, 256>;  // layer 2 output (pong)

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let sample = wid.x;
    let tid = lid.x;
    let N = params.x;
    if (sample >= N) { return; }

    // Cache layout offsets (depend on N):
    //   pre_relu0: 0                     size N*256
    //   hidden0:   N*256                 size N*256
    //   pre_relu1: N*512                 size N*128
    //   hidden1:   N*512 + N*128 = N*640 size N*128
    let pre0_off = sample * DIM_H0;
    let h0_off   = N * DIM_H0 + sample * DIM_H0;
    let pre1_off = N * DIM_H0 * 2u + sample * DIM_H1;
    let h1_off   = N * DIM_H0 * 2u + N * DIM_H1 + sample * DIM_H1;
    // Mask offsets: mask0 at 0 (N*256), mask1 at N*256 (N*128)
    let m0_off = sample * DIM_H0;
    let m1_off = N * DIM_H0 + sample * DIM_H1;

    // === Layer 0: input(58) → hidden0(256) with relu + dropout ===
    // All 256 threads active, each computes one hidden unit
    {
        var sum = biases[B0_OFF + tid];
        let in_base = sample * DIM_IN;
        for (var k = 0u; k < DIM_IN; k = k + 1u) {
            sum = sum + input[in_base + k] * weights[W0_OFF + k * DIM_H0 + tid];
        }
        // Store pre-relu cache
        caches[pre0_off + tid] = sum;
        // ReLU + dropout mask
        let activated = max(0.0, sum) * masks[m0_off + tid];
        caches[h0_off + tid] = activated;
        act_a[tid] = activated;
    }
    workgroupBarrier();

    // === Layer 2: hidden0(256) → hidden1(128) with relu + dropout ===
    // 128 threads active
    if (tid < DIM_H1) {
        var sum = biases[B2_OFF + tid];
        for (var k = 0u; k < DIM_H0; k = k + 1u) {
            sum = sum + act_a[k] * weights[W2_OFF + k * DIM_H1 + tid];
        }
        caches[pre1_off + tid] = sum;
        let activated = max(0.0, sum) * masks[m1_off + tid];
        caches[h1_off + tid] = activated;
        act_b[tid] = activated;
    }
    workgroupBarrier();

    // === Layer 4: hidden1(128) → output(1) ===
    // Thread 0 computes the final dot product
    if (tid == 0u) {
        var sum = biases[B4_OFF];
        for (var k = 0u; k < DIM_H1; k = k + 1u) {
            sum = sum + act_b[k] * weights[W4_OFF + k];
        }
        output[sample] = sum;
    }
}
