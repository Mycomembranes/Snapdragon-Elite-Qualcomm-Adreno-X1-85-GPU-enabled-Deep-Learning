#  Native D3D12 Compute Backend for ultimate use in Protein Folding

## Overview

This directory contains the complete implementation of a native D3D12 compute backend for OperonFold, bypassing the Vulkan-to-Dozen-to-D3D12 translation stack on WSL2 (Surface Pro ARM64, Qualcomm Adreno X1-85). The result: **111-200x speedup** on GPU dispatch overhead.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Architecture](#3-architecture)
4. [The Full Journey](#4-the-full-journey)
5. [C API Reference](#5-c-api-reference)
6. [Python API Reference](#6-python-api-reference)
7. [Shader Binding Reference](#7-shader-binding-reference)
8. [Performance Results](#8-performance-results)
9. [System Dependencies](#9-system-dependencies)
10. [Environment Variables](#10-environment-variables)
11. [File Index](#11-file-index)

---

## 1. Problem Statement

OperonFold (unreleased by me as of 3/8/26) uses wgpu (via wgpu-py) for GPU-accelerated tensor operations. On WSL2 with Qualcomm Adreno X1-85, the only available GPU path is:

```
Python wgpu-py -> wgpu-native (Rust) -> Vulkan API -> Mesa Dozen VK driver -> D3D12 API -> Windows D3D12 -> Adreno GPU
```

This 7-layer translation stack added **10-18ms of overhead per GPU submit**, regardless of workload size. Profiling showed **93% of GPU time** was spent in the Vulkan-to-D3D12 translation layer (Dozen), not actual computation.

For a simple `add` of two 4096-element vectors (which should take microseconds), the overhead dominated: 15ms total, of which ~14ms was translation overhead.

For an MLP forward pass (300x58 -> 256 -> 128 -> 1), the same calculation took **0.12ms on numpy CPU** but **~90ms via wgpu/Dozen**. The GPU path was **750x slower than CPU**.

### Why Not Just Use CPU?

CPU is faster for small/medium tensors. But OperonFold's contact prediction head processes batches of thousands of residue pairs through multi-layer MLPs. At batch sizes > 10K, GPU parallelism wins -- but only if dispatch overhead is negligible. The Dozen overhead made GPU acceleration impossible for any realistic workload.

---

## 2. Solution Overview

Bypass the entire Vulkan/wgpu/Dozen stack. Call D3D12 COM interfaces directly from C, expose via Python ctypes:

```
Python ctypes -> C wrapper (d3d12_compute.c) -> D3D12 COM API -> Windows D3D12 -> Adreno GPU
```

This 4-layer stack eliminates 3 translation layers. The C wrapper handles:
- DXCore adapter enumeration (no DXGI on Linux)
- D3D12 device creation with experimental shader models
- SPIR-V to DXIL compilation via Mesa's `libspirv_to_dxil.so`
- PSV0 parsing to auto-detect resource bindings (SRV/UAV/CBV)
- Root signature construction from DXIL metadata
- Descriptor heap management (SRV + UAV + CBV)
- Command list recording, execution, and fence-based synchronization
- Buffer management (default heap, upload heap, readback heap)

**Result**: Empty submit dropped from 10-18ms to 0.090ms (111-200x speedup). Minimum dispatch dropped from ~10ms to 0.025ms (400x speedup).

---

## 3. Architecture

### Before: Vulkan/Dozen Stack (7 layers)

```
                    +------------------+
                    |  Python (wgpu-py) |
                    +--------+---------+
                             |
                    +--------v---------+
                    | wgpu-native (Rust)|
                    +--------+---------+
                             |
                    +--------v---------+
                    |    Vulkan API     |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Mesa Dozen VK ICD |  <-- 93% overhead here
                    | (Vulkan -> D3D12) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    D3D12 API      |
                    +--------+---------+
                             |
                    +--------v---------+
                    | Windows D3D12 RT  |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Adreno GPU HW   |
                    +------------------+
```

### After: Native D3D12 Stack (4 layers)

```
                    +------------------+
                    | Python (ctypes)   |
                    +--------+---------+
                             |
                    +--------v---------+
                    | d3d12_compute.c   |
                    | (COM vtable calls)|
                    +--------+---------+
                             |
                    +--------v---------+
                    |    D3D12 API      |
                    | + Windows D3D12   |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   Adreno GPU HW   |
                    +------------------+
```

### Shader Pipeline

```
WGSL source (26 shaders)
    |
    v  naga-cli v22.0.0
SPIR-V bytecode (28 .spv files)
    |
    v  libspirv_to_dxil.so (Mesa)
DXIL bytecode (at runtime)
    |
    v  D3D12 PSO creation
GPU pipeline state object
```

The SPIR-V to DXIL compilation happens at runtime inside `d3d12_compute.c` using the same Mesa library that Dozen uses internally. This means our compiled shaders are identical to what Dozen would produce -- we just skip the Vulkan translation overhead.

---

## 4. The Full Journey

### Session 1: Proof of Concept (d3d12_compute.c + basic dispatch)

**Goal**: Prove that direct D3D12 calls are faster than Dozen.

- Built `d3d12_compute.c` (initial ~800 lines) with DXCore adapter enumeration, device creation, command queue, buffer management
- Created `D3D12Tensor` Python class with `from_numpy()`, `numpy()`, basic arithmetic
- Compiled a trivial compute shader (noop) via GLSL -> SPIR-V
- **Blocker**: `CreateComputePipelineState` returned `E_INVALIDARG` (0x80070057) for all shaders
- **Root cause**: WSL2 D3D12 runtime rejects unsigned DXIL. Mesa's Dozen driver gets a special bypass; standalone apps do not
- **Fix**: Call `D3D12EnableExperimentalFeatures()` with `D3D12ExperimentalShaderModels` GUID **before** device creation
- **Result**: Empty submit benchmark: **0.090ms vs 10-18ms** = 111-200x speedup confirmed

### Session 2: Shader Extraction and Compilation

**Goal**: Get all 26 OperonFold compute shaders compiled to SPIR-V.

- Extracted 26 WGSL shader sources from `wgpu_tensor.py` into individual `.wgsl` files
- Installed `naga-cli` v22.0.0 via Cargo
- Compiled all 26 WGSL -> SPIR-V with `compile_shaders.sh`
- Fixed WGSL reserved keyword issues (variable naming conflicts)
- Validated all SPIR-V with `spirv-val`
- **Result**: 26 WGSL sources -> 28 SPIR-V binaries (includes `noop.spv` and `add_glsl.spv` test variants)

### Session 3: PSO Creation and Root Signature

**Goal**: Create pipeline state objects for real shaders (not just noop).

- **Blocker**: PSO creation failed for add shader even with experimental features enabled
- **Root cause**: Root signature mismatch. `spirv_to_dxil` converts `NonWritable` SSBO storage buffers to SRVs (t-registers), but we were building root signatures with only UAV entries
- **Fix**: Parse the PSV0 (Pipeline State Validation) section from the DXBC container output by `spirv_to_dxil`. The PSV0 contains exact resource type information (SRV/UAV/CBV) with register ranges
- Implemented `create_root_sig_from_dxil()` which:
  1. Parses DXBC container to find PSV0 chunk
  2. Reads resource binding table (type, register, space)
  3. Counts SRVs, UAVs, CBVs
  4. Builds root signature with separate descriptor tables in order: SRV -> UAV -> CBV
- **Result**: All 26 shaders create PSOs successfully

### Session 4: Dispatch Correctness (SRV vs UAV)

**Goal**: Verify shader outputs match numpy reference implementations.

- **Blocker**: `add` shader returned all zeros despite PSO creation success
- **Root cause**: `d3d12c_dispatch()` created UAV descriptors for ALL buffer handles. But the `add` shader has 2 SRVs (t0, t1) + 1 UAV (u2). SRVs need `CreateShaderResourceView` with `DXGI_FORMAT_R32_TYPELESS` and `D3D12_BUFFER_SRV_FLAG_RAW`, not `CreateUnorderedAccessView`
- **Fix**: Rewrote dispatch to accept separate SRV/UAV/CBV handle arrays with correct descriptor creation for each type
- Ran 17-shader correctness suite (GPU vs numpy, tolerance 1e-4):
  - 15/17 passed immediately
  - `gelu_backward`: max_diff=0.567 -- numpy used exact GELU derivative, shader uses sigmoid approximation `sigmoid(1.702*x)`. Fixed numpy reference
  - `cross_entropy`: max_diff=NaN -- test passed raw logits, shader expects log-probabilities. Fixed test formula
- **Result**: 17/17 shaders correct, all max_diff < 4e-07

### Session 5: Python API Integration

**Goal**: Full integration with `wgpu_tensor.py` so OperonFold can auto-detect and use D3D12 backend.

- Updated `d3d12_tensor.py` with all 25 operation functions matching shader bindings
- Fixed `d3d12_init()` not being called during auto-detection in `wgpu_tensor.py`
- Fixed `WgpuTensor.from_numpy()` returning wgpu buffers instead of `D3D12Tensor` when backend=d3d12
- Ran comprehensive API test suite: **20/20 tests passed**
- MLP forward pass benchmark: 10.035ms (vs ~90ms Dozen, vs 0.124ms numpy)
- **Result**: Drop-in D3D12 backend working end-to-end

---

## 5. C API Reference

All functions exported from `libd3d12_compute.so` (1,465 lines):

### Initialization

```c
int d3d12c_init(void);
// Initialize D3D12: enumerate adapters via DXCore, create device with
// experimental shader models, create command queue/allocator/list,
// descriptor heaps, fence. Returns 0 on success.

void d3d12c_shutdown(void);
// Release all D3D12 objects, close handles.

const char* d3d12c_get_adapter_name(void);
// Returns GPU adapter name (e.g., "Qualcomm(R) Adreno(TM) X1-85 GPU").

const char* d3d12c_get_last_error(void);
// Returns last error message string.
```

### Buffer Management

```c
uint64_t d3d12c_create_buffer(uint64_t size_bytes);
// Create GPU-local buffer (D3D12_HEAP_TYPE_DEFAULT). Returns handle.

uint64_t d3d12c_create_upload_buffer(uint64_t size_bytes);
// Create CPU-writable staging buffer. Returns handle.

uint64_t d3d12c_create_readback_buffer(uint64_t size_bytes);
// Create CPU-readable staging buffer. Returns handle.

void d3d12c_release_buffer(uint64_t handle);
// Release buffer and its ID3D12Resource.

int d3d12c_upload(uint64_t dst_handle, const void *data, uint64_t size_bytes);
// Upload CPU data to GPU buffer via staging. Returns 0 on success.

int d3d12c_readback(uint64_t src_handle, void *data, uint64_t size_bytes);
// Readback GPU buffer to CPU via staging. Returns 0 on success.

uint64_t d3d12c_get_buffer_gpu_address(uint64_t handle);
// Get GPU virtual address for a buffer.
```

### Pipeline Creation

```c
int d3d12c_compile_spirv_to_dxil(
    const void *spirv_data, uint32_t spirv_size,
    void **out_dxil, uint32_t *out_dxil_size);
// Compile SPIR-V bytecode to DXIL via libspirv_to_dxil.so.
// Outputs heap-allocated DXIL blob. Returns 0 on success.

uint64_t d3d12c_create_compute_pipeline(
    const void* spirv_data, uint32_t spirv_size_bytes);
// Full pipeline creation: SPIR-V -> DXIL -> parse PSV0 -> build root sig -> create PSO.
// Auto-detects SRV/UAV/CBV counts from DXIL metadata. Returns pipeline handle.

void d3d12c_release_pipeline(uint64_t handle);
// Release pipeline state object and root signature.
```

### Dispatch

```c
int d3d12c_dispatch(
    uint64_t pipeline_handle,
    const uint64_t *srv_handles, uint32_t num_srvs,
    const uint64_t *uav_handles, uint32_t num_uavs,
    const uint64_t *cbv_handles, uint32_t num_cbvs,
    uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
// Record dispatch into current command list. Must be called between
// begin_commands() and end_commands_and_wait(). Creates appropriate
// SRV/UAV/CBV descriptors and sets root descriptor tables.

int d3d12c_dispatch_sync(
    uint64_t pipeline_handle,
    const uint64_t *srv_handles, uint32_t num_srvs,
    const uint64_t *uav_handles, uint32_t num_uavs,
    const uint64_t *cbv_handles, uint32_t num_cbvs,
    uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
// Convenience: begin_commands + dispatch + end_commands_and_wait in one call.

int d3d12c_begin_commands(void);
// Reset command allocator and command list for recording.

int d3d12c_end_commands_and_wait(void);
// Close command list, execute on queue, wait via fence.
```

---

## 6. Python API Reference

`d3d12_tensor.py` (805 lines) provides the Python interface:

### D3D12Tensor Class

```python
class D3D12Tensor:
    """GPU tensor backed by D3D12 buffer."""

    @staticmethod
    def empty(shape: tuple) -> D3D12Tensor
    # Allocate uninitialized GPU buffer.

    @staticmethod
    def from_numpy(arr: np.ndarray) -> D3D12Tensor
    # Upload numpy array to GPU. Converts to float32.

    @staticmethod
    def zeros(shape: tuple) -> D3D12Tensor

    @staticmethod
    def ones(shape: tuple) -> D3D12Tensor

    @staticmethod
    def arange(n: int) -> D3D12Tensor

    def numpy(self) -> np.ndarray
    # Readback GPU buffer to numpy array.

    def numel(self) -> int
    def ndim(self) -> int
    def expand(self, shape) -> D3D12Tensor
    def transpose(self) -> D3D12Tensor

    # Operator overloads
    def __add__(self, other) -> D3D12Tensor      # d3d12_add
    def __sub__(self, other) -> D3D12Tensor      # d3d12_sub
    def __mul__(self, other) -> D3D12Tensor      # d3d12_mul/scalar_mul
    def __neg__(self) -> D3D12Tensor             # d3d12_neg
    def __matmul__(self, other) -> D3D12Tensor   # d3d12_matmul
```

### Operation Functions (25 total)

Each function loads its SPIR-V shader, creates/caches the pipeline, dispatches, and returns a `D3D12Tensor`.

```python
# Element-wise (2 SRV + 1 UAV)
d3d12_add(a, b)
d3d12_sub(a, b)
d3d12_mul(a, b)
d3d12_cross_entropy(logits, targets)

# Unary (1 SRV + 1 UAV)
d3d12_neg(x)
d3d12_relu(x)
d3d12_gelu(x)
d3d12_sigmoid(x)
d3d12_tanh(x)

# With params (1 SRV + 1 UAV + 1 CBV)
d3d12_scalar_mul(a, scalar)

# Matrix ops (2-3 SRV + 1 UAV + 1 CBV)
d3d12_matmul(a, b)                    # 2 SRV + 1 UAV + 1 CBV
d3d12_matmul_add(a, b, bias)          # 3 SRV + 1 UAV + 1 CBV
d3d12_matmul_add_relu(a, b, bias)     # 3 SRV + 1 UAV + 1 CBV
d3d12_transpose(x)                     # 1 SRV + 1 UAV + 1 CBV

# Normalization (3 SRV + 1 UAV + 1 CBV)
d3d12_layer_norm(x, gamma, beta, eps=1e-5)
d3d12_softmax(x)                       # 1 SRV + 1 UAV + 1 CBV

# Backward passes
d3d12_relu_backward(grad_out, x)       # 2 SRV + 1 UAV
d3d12_gelu_backward(grad_out, x)       # 2 SRV + 1 UAV
d3d12_softmax_backward(grad_out, probs)# 2 SRV + 1 UAV + 1 CBV
d3d12_layernorm_backward(grad_out, x, gamma, eps=1e-5)  # 3 SRV + 1 UAV + 1 CBV

# Reductions (1 SRV + 1 UAV + 1 CBV)
d3d12_sum(x)
d3d12_mean(x)
d3d12_max_reduce(x)

# Special
d3d12_focal_bce(logits, targets, gamma=2.0, alpha=0.25)
d3d12_embedding_lookup(weight, indices)  # 2 SRV + 1 UAV + 1 CBV
```

### Utility Functions

```python
def is_available() -> bool
# Check if D3D12 backend is usable (init + test dispatch).

def get_device_info() -> str
# Return adapter name string.
```

---

## 7. Shader Binding Reference

Every shader's resource bindings as determined by PSV0 parsing of compiled DXIL. `spirv_to_dxil` maps SPIR-V `NonWritable` SSBOs to SRVs (t-registers) and writable SSBOs to UAVs (u-registers).

| Shader | SRVs | UAVs | CBVs | Register Layout | Notes |
|--------|------|------|------|-----------------|-------|
| add | 2 | 1 | 0 | t0,t1 / u2 | Element-wise a+b |
| sub | 2 | 1 | 0 | t0,t1 / u2 | Element-wise a-b |
| mul | 2 | 1 | 0 | t0,t1 / u2 | Element-wise a*b |
| neg | 1 | 1 | 0 | t0 / u1 | Negate |
| relu | 1 | 1 | 0 | t0 / u1 | max(0, x) |
| gelu | 1 | 1 | 0 | t0 / u1 | GELU approximation |
| sigmoid | 1 | 1 | 0 | t0 / u1 | 1/(1+exp(-x)) |
| tanh_act | 1 | 1 | 0 | t0 / u1 | tanh(x) |
| scalar_mul | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {scalar, 0, 0, 0} |
| relu_backward | 2 | 1 | 0 | t0,t1 / u2 | grad * (x > 0) |
| gelu_backward | 2 | 1 | 0 | t0,t1 / u2 | sigmoid(1.702x) approx |
| cross_entropy | 2 | 1 | 0 | t0,t1 / u2 | Expects log-probs |
| matmul | 2 | 1 | 1 | t0,t1 / u2 / b3 | CBV: {M, N, K, 0} |
| matmul_add | 3 | 1 | 1 | t0-t2 / u3 / b4 | + bias |
| matmul_add_relu | 3 | 1 | 1 | t0-t2 / u3 / b4 | + bias + relu |
| softmax | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {N, 0, 0, 0} |
| softmax_backward | 2 | 1 | 1 | t0,t1 / u2 / b3 | |
| layer_norm | 3 | 1 | 1 | t0-t2 / u3 / b4 | SRVs: x,gamma,beta; CBV: {N,eps,0,0} |
| layernorm_backward | 3 | 1 | 1 | t0-t2 / u3 / b4 | |
| sum_reduce | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {N, 0, 0, 0} |
| mean_reduce | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {N, 0, 0, 0} |
| max_reduce | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {N, 0, 0, 0} |
| embedding | 2 | 1 | 1 | t0,t1 / u2 / b3 | SRVs: weight,indices |
| focal_bce | 2 | 1 | 1 | t0,t1 / u2 / b3 | CBV: {N,gamma,alpha,0} |
| mlp_forward_fused | 3+ | 1+ | 1 | varies | Multi-layer fused MLP |
| transpose_2d | 1 | 1 | 1 | t0 / u1 / b2 | CBV: {rows, cols, 0, 0} |

**Root signature layout** (always in this order):
1. Descriptor Table 0: SRV range (t0..tN)
2. Descriptor Table 1: UAV range (u0..uN)
3. Descriptor Table 2: CBV range (b0..bN)

---

## 8. Performance Results

All benchmarks on Qualcomm Adreno X1-85, WSL2 Ubuntu, Surface Pro ARM64.

### Dispatch Overhead

| Metric | Dozen (Vulkan) | D3D12 Native | Speedup |
|--------|----------------|--------------|---------|
| Empty submit (no shader) | 10-18 ms | 0.090 ms | **111-200x** |
| Minimum dispatch | ~10 ms | 0.025 ms | **400x** |

### Per-Operation Latency

| Operation | N | D3D12 (ms) | Dozen (ms) | Speedup |
|-----------|---|------------|------------|---------|
| add | 4096 | 0.93 | ~15 | ~16x |
| matmul | 64x64 | 2.2 | ~15 | ~7x |
| relu | 4096 | 0.89 | ~15 | ~17x |

### End-to-End MLP Forward Pass

| Configuration | Numpy (CPU) | Dozen (GPU) | D3D12 (GPU) | D3D12 Speedup |
|---------------|-------------|-------------|-------------|---------------|
| 300x58 -> 256 -> 128 -> 1 | 0.124 ms | ~90 ms | 10.035 ms | **9x vs Dozen** |

**Note**: Numpy CPU is still faster than D3D12 GPU for this small MLP because the per-dispatch overhead (~0.9ms) multiplied by the number of operations (matmul+bias+relu per layer) exceeds the parallelism benefit. The GPU advantage appears at larger batch sizes (>1000 samples) where compute dominates overhead.

### Overhead Breakdown (per dispatch)

| Phase | Time (ms) | % |
|-------|-----------|---|
| Buffer alloc + upload | ~0.3 | 33% |
| Command record | ~0.1 | 11% |
| GPU execution | ~0.3 | 33% |
| Fence wait + readback | ~0.2 | 22% |

---

## 9. System Dependencies

### Required System Packages

```bash
sudo apt install directx-headers-dev build-essential spirv-tools
```

### Mesa Dozen Driver (custom build)

Required for `libspirv_to_dxil.so` and Vulkan baseline testing.

```
Install path: /home/mukshud/mesa-dozen-install/
Library: /home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu/libspirv_to_dxil.so
ICD: /home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
```

### Rust / naga-cli

```bash
cargo install naga-cli@22.0.0
# Provides: ~/.cargo/bin/naga
# Used for: WGSL -> SPIR-V compilation
```

### WSL2 D3D12 Runtime

Provided by Windows via `/usr/lib/wsl/lib/`:
- `libdxcore.so` - DXCore adapter enumeration
- `libd3d12.so` - D3D12 API
- `libdxcompiler.so` - DXIL compiler (not used directly)

### Python Dependencies

```
numpy >= 1.24
# ctypes (stdlib) for libd3d12_compute.so binding
```

---

## 10. Environment Variables

### Required

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}
export VK_ICD_FILENAMES=/home/mukshud/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
```

### Optional / Debug

```bash
MESA_VK_VERSION_OVERRIDE=1.0    # Force Vulkan 1.0 version reporting
D3D12C_DUMP_NIR=1                # Dump NIR output from spirv_to_dxil
DZN_DEBUG=dxil                   # Debug DXIL in Dozen driver
WGPU_FORCE_BACKEND=wgpu          # Disable D3D12 auto-detect, force wgpu/Vulkan
```

---

## 11. File Index

See [INDEX.md](INDEX.md) for a complete file-by-file index with descriptions.

See [RECREATION_GUIDE.md](RECREATION_GUIDE.md) for step-by-step instructions to recreate from scratch.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for technical deep-dive on design decisions.

See [docs/BLOCKERS_AND_FIXES.md](docs/BLOCKERS_AND_FIXES.md) for every blocker and how it was resolved.

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for detailed performance data.

---

## Hardware Context

- **GPU**: Qualcomm Adreno X1-85 (TileBasedRenderer=1, UMA=1)
- **Shader Model**: 6.2
- **Platform**: WSL2 aarch64 on Windows 11 ARM64
- **Device**: Microsoft Surface Pro (ARM)
- **D3D12 Feature Level**: 12_1
- **Max Shared Memory**: UMA (unified memory architecture)

## Key Insight

The fundamental insight is that Mesa's `spirv_to_dxil` compiler -- the same code path Dozen uses internally -- is available as a standalone library. By calling it directly and feeding the DXIL output into D3D12's `CreateComputePipelineState`, we get identical GPU code execution without the Vulkan translation overhead. The challenge was purely in the plumbing: experimental features for unsigned DXIL, PSV0 parsing for resource bindings, and correct SRV/UAV descriptor creation.
