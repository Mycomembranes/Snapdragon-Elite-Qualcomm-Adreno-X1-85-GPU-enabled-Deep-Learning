# Benchmark Results

All benchmarks on:
- **GPU**: Qualcomm Adreno X1-85 (TileBasedRenderer=1, UMA=1)
- **Platform**: WSL2 Ubuntu on Surface Pro ARM64
- **Shader Model**: 6.2
- **D3D12 Feature Level**: 12_1

---

## Dispatch Overhead (Empty Submit)

The most dramatic improvement. Measures the time for a complete GPU submit cycle with no actual compute work (empty command list or noop shader).

| Metric | Dozen (Vulkan->D3D12) | D3D12 Native | Speedup |
|--------|----------------------|--------------|---------|
| Empty submit (mean) | 10-18 ms | 0.090 ms | **111-200x** |
| Minimum dispatch | ~10 ms | 0.025 ms | **400x** |

**Methodology**: 1000 iterations of begin_commands -> end_commands_and_wait (D3D12 native) vs equivalent wgpu submit cycle (Dozen). First 10 iterations discarded as warmup.

**Why such a large difference**: Dozen translates every Vulkan API call to D3D12 equivalents. A single `vkQueueSubmit` triggers command buffer translation, descriptor set remapping, pipeline state re-validation, and multiple D3D12 command list operations. The native path does one `ExecuteCommandLists` + fence signal.

---

## Per-Operation Latency

Time for a single complete dispatch: buffer alloc + upload + command record + execute + fence wait + readback.

| Operation | N | D3D12 Native (ms) | Dozen Estimated (ms) | Speedup |
|-----------|---|-------------------|---------------------|---------|
| add | 4,096 | 0.93 | ~15 | **~16x** |
| relu | 4,096 | 0.89 | ~15 | **~17x** |
| sigmoid | 4,096 | 0.91 | ~15 | **~16x** |
| matmul | 64x64 | 2.2 | ~15 | **~7x** |
| layer_norm | 4,096 | 1.8 | ~15 | **~8x** |
| softmax | 4,096 | 1.5 | ~15 | **~10x** |

**Note**: Per-op times include buffer allocation and CPU-GPU synchronization. The actual GPU compute time is a small fraction of the total.

---

## Dispatch Time Breakdown

Where time is spent in a single D3D12 native dispatch (add, N=4096):

| Phase | Time (ms) | Percentage |
|-------|-----------|-----------|
| Buffer allocation + upload | ~0.30 | 32% |
| Command recording | ~0.10 | 11% |
| GPU execution | ~0.30 | 32% |
| Fence wait + readback | ~0.23 | 25% |
| **Total** | **~0.93** | **100%** |

**Key insight**: With native D3D12, actual GPU compute (0.3ms) is comparable to the plumbing overhead. With Dozen, plumbing was 93% of total time.

---

## End-to-End MLP Forward Pass

OperonFold's contact prediction MLP: 300 samples x 58 features -> 256 -> 128 -> 1.

Operations per forward pass:
- 3x matmul_add (fused matrix multiply + bias)
- 2x relu activation
- Buffer allocations for intermediates

| Backend | Time (ms) | Notes |
|---------|-----------|-------|
| Numpy CPU | 0.124 | Single-threaded, contiguous memory |
| Dozen (Vulkan) | ~90 | 6 dispatches x ~15ms each |
| D3D12 Native | 10.035 | 6 dispatches x ~1.7ms each |

**Speedup**: D3D12 Native is **~9x faster than Dozen**.

**CPU vs GPU**: Numpy CPU is still faster at this scale because:
1. Per-dispatch overhead (0.9ms) x 6 ops = 5.4ms just in overhead
2. The matrices (300x58, 58x256, etc.) are small enough that CPU cache locality wins
3. numpy uses optimized BLAS (OpenBLAS/MKL) with vectorized instructions

**Break-even point**: GPU becomes faster than CPU at batch sizes > ~1,000 samples, where the O(N) compute dominates the O(1) dispatch overhead.

---

## Comparison Summary

```
Dispatch Overhead:     Dozen 10-18ms  ->  D3D12 0.090ms  (111-200x faster)
Per-Op (element-wise): Dozen ~15ms    ->  D3D12 ~0.9ms   (16x faster)
Per-Op (matmul 64x64): Dozen ~15ms    ->  D3D12 ~2.2ms   (7x faster)
MLP Forward Pass:      Dozen ~90ms    ->  D3D12 ~10ms    (9x faster)
```

---

## Theoretical Maximum Speedup

The theoretical maximum speedup is bounded by:
- **Dispatch overhead**: 200x improvement
- **Actual compute**: Identical (same DXIL executed on same GPU)
- **Buffer management**: ~2x improvement (no Vulkan memory type translation)

For compute-bound workloads (large matrices, many elements), the speedup approaches 1x because the Dozen overhead is amortized. For dispatch-bound workloads (many small operations), the speedup approaches 200x.

The MLP benchmark (9x speedup) is in between: each operation is small enough that overhead matters but large enough that compute isn't negligible.

---

## Future Optimization Opportunities

| Optimization | Expected Impact | Complexity |
|-------------|----------------|------------|
| Batch dispatches (single command list) | 3-5x on multi-op pipelines | Medium |
| Persistent buffer pool | 30% per-op reduction | Low |
| Descriptor heap reuse | 10% per-op reduction | Low |
| Async compute (overlap CPU/GPU) | 2x on pipelined workloads | High |
| Shared memory tiling (matmul) | 5-10x on large matmul | High |
