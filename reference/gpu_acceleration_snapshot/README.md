# Native D3D12 Compute Backend — Bypass Vulkan/Dozen Overhead

## Problem

The existing GPU compute path goes through 4 translation layers:

```
Python → wgpu-py → wgpu-native → Vulkan → Mesa Dozen → D3D12 → /dev/dxg → GPU
```

Profiling showed **10-18ms per GPU submit** — 93% of which was dispatch+sync overhead from the Dozen translation layer. This made every GPU operation 3-150x slower than CPU numpy at all matrix sizes. Fused shaders and single-kernel MLPs couldn't overcome this fixed overhead.

## Solution

Bypass the entire Vulkan/Dozen stack by calling `libd3d12.so` and `libdxcore.so` directly:

```
Python (d3d12_tensor.py)
  → ctypes → libd3d12_compute.so (thin C wrapper, ~600 LOC)
    → D3D12 COM interfaces (official directx-headers-dev)
      → /dev/dxg (WSL2 DirectX passthrough)
        → Qualcomm Adreno X1-85 GPU
```

## Results

| Metric | Vulkan/Dozen | D3D12 Native | Speedup |
|--------|-------------|--------------|---------|
| Empty submit overhead | 10-18ms | 0.090ms | **111-200x** |
| Buffer roundtrip (1M floats) | ~25ms | ~12ms | ~2x |
| Min dispatch latency | ~10ms | 0.025ms | **400x** |

## How It Works

### Step 1: C Wrapper (`d3d12_compute.c`)

A ~600 LOC C shared library that wraps D3D12 COM vtable calls into a flat C API.

**Why C instead of pure ctypes?** COM vtable calls require precise pointer arithmetic and struct layouts. The C compiler handles vtable dispatch correctly using the official `directx-headers-dev` headers. Python ctypes just calls the flat C functions.

**Key discovery:** WSL2's `directx-headers-dev` package provides:
- `CINTERFACE` mode for D3D12 headers — enables pure C COM vtable access
- `COBJMACROS` — convenience macros like `ID3D12Device_CreateCommandQueue()`
- WSL stubs at `/usr/include/wsl/stubs/` — `IUnknown`, `HRESULT`, etc.

**DXCore adapter enumeration** (C++ headers, so we manually define vtable structures):
1. `dlopen("libdxcore.so")` → `DXCoreCreateAdapterFactory()`
2. `CreateAdapterList(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)`
3. Iterate adapters, pick hardware GPU
4. `D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, ...)`

**SPIR-V → DXIL compilation** (optional, for shader support):
Uses Mesa's `libspirv_to_dxil.so` — the same library that Mesa Dozen uses internally.

### Step 2: Build Script (`build_d3d12.sh`)

```bash
gcc -shared -fPIC -O2 -Wall \
    -I/usr/include/wsl/stubs -I/usr/include/directx \
    d3d12_compute.c \
    -ldl -lpthread \
    -o libd3d12_compute.so
```

All D3D12/DXCore libraries are loaded via `dlopen()` at runtime (not linked), so the build only needs standard C libraries.

### Step 3: Python Bindings (`d3d12_tensor.py`)

`D3D12Tensor` class is a drop-in replacement for `WgpuTensor`:

```python
from d3d12_tensor import D3D12Tensor, d3d12_init

d3d12_init()
t = D3D12Tensor.from_numpy(np.random.randn(100, 50).astype(np.float32))
result = t.numpy()  # bit-perfect roundtrip
```

### Step 4: Auto-Detection (`wgpu_tensor.py`)

Backend auto-detection added at module load:
- If `libd3d12_compute.so` exists → use D3D12 native
- Otherwise → fall back to wgpu/Dozen
- Set `WGPU_FORCE_BACKEND=wgpu` to override

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `d3d12_compute.c` | C wrapper: D3D12 COM → flat C API | ~600 |
| `build_d3d12.sh` | Build script: gcc → libd3d12_compute.so | ~20 |
| `d3d12_tensor.py` | Python ctypes bindings + D3D12Tensor class | ~450 |
| `test_d3d12_compute.py` | Correctness tests + benchmarks | ~220 |
| `wgpu_tensor.py` | Modified: backend auto-detection at end | +30 lines |
| `libd3d12_compute.so` | Pre-built shared library (aarch64) | 76KB |

## Dependencies (all pre-installed on this system)

| Component | Path |
|-----------|------|
| D3D12 headers | `/usr/include/directx/d3d12.h` (directx-headers-dev 1.614.1) |
| WSL stubs | `/usr/include/wsl/stubs/` |
| libd3d12.so | `/usr/lib/wsl/lib/libd3d12.so` |
| libdxcore.so | `/usr/lib/wsl/lib/libdxcore.so` |
| libspirv_to_dxil.so | `/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu/` |

## C API Reference

```c
// Lifecycle
int  d3d12c_init(void);
void d3d12c_shutdown(void);
const char* d3d12c_get_adapter_name(void);
const char* d3d12c_get_last_error(void);

// Buffers
uint64_t d3d12c_create_buffer(uint64_t size_bytes);
uint64_t d3d12c_create_upload_buffer(uint64_t size);
uint64_t d3d12c_create_readback_buffer(uint64_t size);
void     d3d12c_release_buffer(uint64_t handle);
int      d3d12c_upload(uint64_t dst, const void* data, uint64_t size);
int      d3d12c_readback(uint64_t src, void* data, uint64_t size);

// Pipelines
uint64_t d3d12c_create_compute_pipeline(const uint32_t* spirv, uint32_t words, uint32_t num_uavs, uint32_t num_cbvs);
uint64_t d3d12c_create_pipeline_from_dxil(const void* dxil, uint32_t size, uint32_t num_uavs, uint32_t num_cbvs);
void     d3d12c_release_pipeline(uint64_t handle);

// Dispatch
int  d3d12c_begin_commands(void);
int  d3d12c_dispatch(uint64_t pipeline, const uint64_t* uavs, uint32_t n_uavs, const uint64_t* cbvs, uint32_t n_cbvs, uint32_t gx, uint32_t gy, uint32_t gz);
int  d3d12c_end_commands_and_wait(void);
int  d3d12c_dispatch_sync(uint64_t pipeline, ...);  // begin + dispatch + end
```

## Next Steps

1. **Compile WGSL shaders to SPIR-V** using `naga-cli` (`cargo install naga-cli`)
2. **Wire up compute operations** (matmul, relu, etc.) through the D3D12 pipeline
3. **Benchmark full MLP** forward/backward through D3D12 vs numpy
4. **Resume training** with D3D12 backend

## Architecture Decision Records

**Why not pure ctypes for COM?** COM vtable pointer arithmetic is fragile in ctypes. One wrong offset crashes with no error. The C compiler uses the official headers and gets it right.

**Why DXCore over DXGI?** WSL2 doesn't have DXGI. DXCore is the Linux-native adapter enumeration API for D3D12.

**Why dlopen instead of linking?** The .so is portable — works on any WSL2 system with D3D12 support. Libraries at `/usr/lib/wsl/lib/` are injected by the WSL2 kernel and may not be on the linker path.

**Why SPIR-V → DXIL?** D3D12 requires DXIL bytecode, not SPIR-V. Mesa's `libspirv_to_dxil.so` handles the translation — the same code path that Dozen uses, proven to work with Adreno.
