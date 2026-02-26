# File Index

Complete index of every file in `gpu_working/` with original location, size, and description.

---

## Root Documentation

| File | Size | Lines | Description |
|------|------|-------|-------------|
| `README.md` | ~25K | ~500 | Master documentation: problem, solution, architecture, journey, API reference, benchmarks |
| `INDEX.md` | ~10K | ~200 | This file. File-by-file index with descriptions |
| `RECREATION_GUIDE.md` | ~15K | ~300 | Step-by-step guide to recreate from fresh WSL2 Ubuntu |

---

## src/ - Source Code

| File | Size | Lines | Original Location | Description |
|------|------|-------|--------------------|-------------|
| `d3d12_compute.c` | 55K | 1,465 | `operonfold/d3d12_compute.c` | C wrapper for D3D12 COM API. Core of the native backend. Handles DXCore adapter enumeration, device creation with experimental shader models, SPIR-V to DXIL compilation via `libspirv_to_dxil.so`, PSV0 parsing for resource binding auto-detection, root signature construction, descriptor heap management (SRV/UAV/CBV), command list recording and fence-based synchronization. Exports `d3d12c_*` functions via ctypes. |
| `d3d12_tensor.py` | 28K | 805 | `operonfold/d3d12_tensor.py` | Python bindings for `libd3d12_compute.so`. Defines `D3D12Tensor` class with numpy interop (`from_numpy()`, `.numpy()`), operator overloads, and 25 GPU operation functions (`d3d12_add`, `d3d12_matmul`, `d3d12_layer_norm`, etc.). Each function loads SPIR-V, creates/caches pipeline, dispatches with correct SRV/UAV/CBV bindings. |
| `wgpu_tensor.py` | 70K | 2,459 | `operonfold/wgpu_tensor.py` | Modified wgpu tensor API with D3D12 backend auto-detection. Contains inline WGSL shader sources (from which the 26 .wgsl files were extracted). Falls back to Vulkan/Dozen if D3D12 init fails. Key modification: `_get_backend()` checks for D3D12 availability first. |
| `build_d3d12.sh` | 1.1K | ~30 | `operonfold/build_d3d12.sh` | Shell script to compile `d3d12_compute.c` into `libd3d12_compute.so`. Uses gcc with `-shared -fPIC`, links against DirectX headers, sets include paths for DXCore/D3D12 headers. |
| `compile_shaders.sh` | 1.4K | ~40 | `operonfold/compile_shaders.sh` | Shell script to compile all WGSL shaders to SPIR-V using `naga-cli`. Iterates over `shaders_wgsl/*.wgsl`, outputs to `shaders_spv/`. Validates each output with `spirv-val`. |
| `libd3d12_compute.so` | 76K | (binary) | `operonfold/libd3d12_compute.so` | Pre-built shared library (aarch64 Linux). Compiled from `d3d12_compute.c`. Can be loaded directly via ctypes on matching platform. |

---

## shaders_wgsl/ - WGSL Shader Sources (26 files)

All shaders originally embedded in `wgpu_tensor.py`, extracted to individual files for `naga-cli` compilation.

| File | Size | Description |
|------|------|-------------|
| `add.wgsl` | 364B | Element-wise addition: out[i] = a[i] + b[i] |
| `sub.wgsl` | 364B | Element-wise subtraction: out[i] = a[i] - b[i] |
| `mul.wgsl` | 364B | Element-wise multiplication: out[i] = a[i] * b[i] |
| `neg.wgsl` | 300B | Negation: out[i] = -x[i] |
| `scalar_mul.wgsl` | 356B | Scalar multiply: out[i] = x[i] * scalar (scalar in CBV) |
| `relu.wgsl` | 309B | ReLU activation: out[i] = max(0, x[i]) |
| `gelu.wgsl` | 452B | GELU activation (sigmoid approximation) |
| `sigmoid.wgsl` | 319B | Sigmoid: out[i] = 1/(1+exp(-x[i])) |
| `tanh_act.wgsl` | 305B | Tanh activation: out[i] = tanh(x[i]) |
| `relu_backward.wgsl` | 408B | ReLU gradient: out[i] = grad[i] * (x[i] > 0) |
| `gelu_backward.wgsl` | 599B | GELU gradient: sigmoid(1.702*x) approximation |
| `matmul.wgsl` | 1.4K | Matrix multiplication with dimensions in CBV |
| `matmul_add.wgsl` | 1.5K | Matmul + bias addition |
| `matmul_add_relu.wgsl` | 1.6K | Matmul + bias + ReLU fused |
| `layer_norm.wgsl` | 2.1K | Layer normalization: (x-mean)/sqrt(var+eps)*gamma+beta |
| `layernorm_backward.wgsl` | 3.7K | Layer norm backward pass (3 gradient computations) |
| `softmax.wgsl` | 2.1K | Softmax: exp(x-max) / sum(exp(x-max)) |
| `softmax_backward.wgsl` | 1.5K | Softmax backward (Jacobian-vector product) |
| `cross_entropy.wgsl` | 482B | Cross-entropy loss (expects log-probabilities) |
| `focal_bce.wgsl` | 836B | Focal binary cross-entropy with gamma/alpha params |
| `embedding.wgsl` | 673B | Embedding lookup: out[i] = weight[indices[i]] |
| `sum_reduce.wgsl` | 1.1K | Sum reduction over array |
| `mean_reduce.wgsl` | 1.0K | Mean reduction over array |
| `max_reduce.wgsl` | 1.0K | Max reduction over array |
| `mlp_forward_fused.wgsl` | 3.9K | Multi-layer fused MLP forward pass |
| `transpose_2d.wgsl` | 439B | 2D matrix transpose |

---

## shaders_spv/ - Compiled SPIR-V Binaries (28 files)

Each `.spv` is compiled from the corresponding `.wgsl` via `naga-cli v22.0.0`. Two extra files: `noop.spv` (test shader compiled from GLSL) and `add_glsl.spv` (GLSL variant of add).

| File | Size | Source |
|------|------|--------|
| `add.spv` | 1.1K | add.wgsl |
| `add_glsl.spv` | 1.4K | GLSL variant (glslangValidator) |
| `cross_entropy.spv` | 1.3K | cross_entropy.wgsl |
| `embedding.spv` | 1.5K | embedding.wgsl |
| `focal_bce.spv` | 1.8K | focal_bce.wgsl |
| `gelu.spv` | 1.1K | gelu.wgsl |
| `gelu_backward.spv` | 1.2K | gelu_backward.wgsl |
| `layer_norm.spv` | 4.4K | layer_norm.wgsl |
| `layernorm_backward.spv` | 6.4K | layernorm_backward.wgsl |
| `matmul.spv` | 3.3K | matmul.wgsl |
| `matmul_add.spv` | 3.6K | matmul_add.wgsl |
| `matmul_add_relu.spv` | 3.7K | matmul_add_relu.wgsl |
| `max_reduce.spv` | 2.4K | max_reduce.wgsl |
| `mean_reduce.spv` | 2.4K | mean_reduce.wgsl |
| `mlp_forward_fused.spv` | 5.0K | mlp_forward_fused.wgsl |
| `mul.spv` | 1.1K | mul.wgsl |
| `neg.spv` | 892B | neg.wgsl |
| `noop.spv` | 280B | noop.comp (GLSL test shader) |
| `relu.spv` | 920B | relu.wgsl |
| `relu_backward.spv` | 1.1K | relu_backward.wgsl |
| `scalar_mul.spv` | 1.1K | scalar_mul.wgsl |
| `sigmoid.spv` | 972B | sigmoid.wgsl |
| `softmax.spv` | 3.9K | softmax.wgsl |
| `softmax_backward.spv` | 3.1K | softmax_backward.wgsl |
| `sub.spv` | 1.1K | sub.wgsl |
| `sum_reduce.spv` | 2.4K | sum_reduce.wgsl |
| `tanh_act.spv` | 900B | tanh_act.wgsl |
| `transpose_2d.spv` | 1.3K | transpose_2d.wgsl |

---

## tests/ - Test Files (14 files)

| File | Size | Original Location | Description |
|------|------|-------------------|-------------|
| `test_buffer_dispatch.py` | 5.5K | `/tmp/` | Low-level buffer roundtrip test. Creates GPU buffers, uploads data, dispatches noop shader, reads back. Verifies D3D12 buffer management works correctly. |
| `test_dispatch_srv.py` | 8.4K | `/tmp/` | SRV vs UAV descriptor binding correctness. Tests that NonWritable SSBOs get SRV descriptors (CreateShaderResourceView) and writable SSBOs get UAV descriptors. Critical test for the dispatch fix. |
| `test_layernorm_fix.py` | 3.4K | `/tmp/` | LayerNorm 5-binding fix verification. Tests that layer_norm correctly uses 3 SRVs (x, gamma, beta) + 1 UAV (output) + 1 CBV (params). Previously failed when only 2 SRVs were passed. |
| `test_all_shaders.py` | 21K | `/tmp/` | Comprehensive 17-shader correctness suite. Tests every shader against numpy reference implementation with tolerance 1e-4. Includes corrected formulas for gelu_backward (sigmoid approx) and cross_entropy (log-prob input). |
| `test_fix2.py` | 3.9K | `/tmp/` | Targeted fixes for gelu_backward and cross_entropy. gelu_backward: corrected numpy reference to use sigmoid(1.702x). cross_entropy: corrected test to pass log-probabilities instead of raw logits. |
| `test_d3d12_api.py` | 5.9K | `/tmp/` | High-level API integration test. 20 test cases covering all d3d12_* functions through the Python API. Tests tensor creation, arithmetic, matmul, activations, normalization, reductions, backward passes. All 20/20 pass. |
| `test_d3d12_compute.py` | 7.3K | `operonfold/` | Original correctness + benchmark test. Tests basic buffer operations, pipeline creation, dispatch. Includes timing benchmarks comparing D3D12 native vs Dozen. |
| `test_d3d12_shaders.py` | 17K | `operonfold/` | Per-shader GPU vs numpy comparison. Tests each shader individually with random inputs, compares output against numpy reference. Reports max absolute difference. |
| `bench_d3d12.py` | 2.9K | `/tmp/` | Dispatch time benchmarks. Measures per-dispatch latency for various operations (add, matmul, relu) over 100-500 iterations. Reports mean, min, max times. |
| `bench_d3d12_vs_dozen.py` | 8.3K | `operonfold/` | Comparative benchmarks: D3D12 native vs Vulkan/Dozen vs numpy CPU. Side-by-side timing for identical operations. Produces the headline speedup numbers. |
| `test_vk_compute.py` | 2.8K | `/tmp/` | Vulkan baseline comparison. Runs the same dispatch through wgpu/Vulkan/Dozen stack to measure baseline overhead. Used to establish the 10-18ms Dozen overhead number. |
| `test_gpu_contact_head.py` | 5.3K | `operonfold/` | Contact prediction MLP test. Tests the actual OperonFold use case: batched residue-pair processing through multi-layer MLP with GPU acceleration. |
| `test_gpu_setup.py` | 8.1K | `operonfold/` | GPU environment verification. Checks all prerequisites: WSL2 D3D12 libraries present, environment variables set, adapter enumerable, device creatable, experimental features available. |
| `test_gradient_flow.py` | 31K | `operonfold/` | Full gradient flow tests. End-to-end forward + backward pass through MLP layers, verifying gradient correctness for all operations (matmul, relu, gelu, layer_norm, softmax, cross_entropy). |

---

## docs/ - Supporting Documentation

| File | Size | Description |
|------|------|-------------|
| `GPU_SETUP.md` | 3.0K | WSL2 GPU environment setup guide. Package installation, Mesa Dozen build, environment variables, verification steps. |
| `ARCHITECTURE.md` | ~12K | Technical architecture deep-dive: COM vtables in C, DXCore enumeration, dlopen vs linking, SPIR-V to DXIL pipeline, PSV0 parsing, root signature construction, descriptor creation. |
| `BLOCKERS_AND_FIXES.md` | ~10K | Every blocker encountered with root cause analysis and fix. 8 major blockers documented with error messages, diagnosis, and resolution. |
| `BENCHMARK_RESULTS.md` | ~5K | Detailed performance numbers: dispatch overhead, per-op latency, MLP benchmarks, overhead breakdown. |

---

## reference/ - Older Versions

| File | Size | Original Location | Description |
|------|------|-------------------|-------------|
| `gpu_acceleration_snapshot/d3d12_compute.c` | 38K | `gpu_acceleration/` | Earlier version of the C wrapper (~900 lines vs current 1,465). Missing PSV0 parsing, SRV support, experimental features fix. Useful for understanding the evolution. |
| `gpu_acceleration_snapshot/d3d12_tensor.py` | 22K | `gpu_acceleration/` | Earlier Python bindings. Missing many operation functions, no cached pipelines, simpler dispatch (UAV-only). |
| `gpu_acceleration_snapshot/README.md` | 6.2K | `gpu_acceleration/` | Original documentation from the first implementation attempt. |

---

## File Counts

| Directory | Files | Total Size |
|-----------|-------|------------|
| `src/` | 6 | ~231K |
| `shaders_wgsl/` | 26 | ~24K |
| `shaders_spv/` | 28 | ~58K |
| `tests/` | 14 | ~131K |
| `docs/` | 4 | ~30K |
| `reference/` | 3 | ~66K |
| Root docs | 3 | ~50K |
| **Total** | **84** | **~590K** |
