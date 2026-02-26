# Architecture Deep-Dive

Technical design decisions and implementation details for the native D3D12 compute backend.

---

## Why COM Vtables in C, Not Python ctypes

D3D12 is a COM (Component Object Model) API. COM objects use vtable-based virtual dispatch: each interface pointer points to a vtable of function pointers. Calling a method like `ID3D12Device::CreateComputePipelineState` requires:

```c
device->lpVtbl->CreateComputePipelineState(device, &desc, &IID_ID3D12PipelineState, &pso);
```

The vtable is a contiguous array of function pointers at known offsets. In C, this is natural -- the DirectX headers define the vtable structs. In Python ctypes, you'd need to:

1. Cast the interface pointer to `POINTER(c_void_p)`
2. Read the vtable pointer at offset 0
3. Index into the vtable at the correct offset (e.g., offset 10 for method 10)
4. Cast to the correct function pointer type with `CFUNCTYPE`
5. Call with correct argument types

This is fragile: vtable offsets change between D3D12 SDK versions, each method has different parameter types, and pointer arithmetic errors cause silent corruption or crashes. A single off-by-one in the vtable index reads the wrong function pointer and crashes with no useful error.

**Decision**: Write a C wrapper that uses the official D3D12 headers for correct vtable access, and export simple C functions (`int d3d12c_init()`, `uint64_t d3d12c_create_buffer(uint64_t size)`) that Python calls via ctypes with trivial type signatures.

---

## DXCore Adapter Enumeration (No DXGI on Linux)

On Windows, GPU enumeration uses DXGI (`CreateDXGIFactory` -> `EnumAdapters`). On WSL2/Linux, DXGI is not available. Instead, Microsoft provides DXCore:

```c
#include <dxcore.h>

DXCoreCreateAdapterFactory(&IID_IDXCoreAdapterFactory, &factory);
factory->lpVtbl->CreateAdapterList(factory, 1, &DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS, &IID_IDXCoreAdapterList, &list);
list->lpVtbl->GetAdapter(list, 0, &IID_IDXCoreAdapter, &adapter);
```

DXCore enumerates the same GPU adapters but through a Linux-compatible interface. The adapter handle can then be passed to `D3D12CreateDevice`:

```c
D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_12_0, &IID_ID3D12Device, &device);
```

**Key difference from DXGI**: DXCore adapters implement `IDXCoreAdapter`, not `IDXGIAdapter`. The `D3D12CreateDevice` function accepts `IUnknown*`, so both work, but you must use the correct enumeration API for your platform.

---

## dlopen vs Linking

We do NOT link against `libd3d12.so`, `libdxcore.so`, or `libspirv_to_dxil.so` at compile time. Instead, `d3d12_compute.c` uses `dlopen()` to load them at runtime:

```c
void* h_dxcore = dlopen("libdxcore.so", RTLD_NOW);
void* h_d3d12 = dlopen("libd3d12.so", RTLD_NOW);
void* h_spirv = dlopen("libspirv_to_dxil.so", RTLD_NOW);

// Resolve function pointers
pfn_DXCoreCreateAdapterFactory = dlsym(h_dxcore, "DXCoreCreateAdapterFactory");
pfn_D3D12CreateDevice = dlsym(h_d3d12, "D3D12CreateDevice");
pfn_spirv_to_dxil = dlsym(h_spirv, "spirv_to_dxil");
```

**Why**:
- `libd3d12.so` and `libdxcore.so` are in `/usr/lib/wsl/lib/`, which is not in the default library search path. Linking at compile time requires `-L/usr/lib/wsl/lib` which makes the binary non-portable.
- `libspirv_to_dxil.so` is from our custom Mesa build at `~/mesa-dozen-install/lib/`. Path varies per user.
- With `dlopen()`, the binary works on any WSL2 system as long as `LD_LIBRARY_PATH` is set correctly at runtime. It also provides clean error messages if a library is missing.

---

## SPIR-V to DXIL via libspirv_to_dxil.so

Mesa's `spirv_to_dxil` is the same compiler that the Dozen Vulkan driver uses internally. It takes SPIR-V bytecode and produces DXIL (DirectX Intermediate Language), which is what D3D12 actually executes.

The pipeline is:

```
WGSL  --[naga-cli]--> SPIR-V  --[spirv_to_dxil]--> DXIL  --[D3D12]--> GPU
```

The `spirv_to_dxil` function signature:

```c
bool spirv_to_dxil(
    const uint32_t *words,          // SPIR-V bytecode
    size_t word_count,               // Number of 32-bit words
    struct dxil_spirv_metadata *meta,// Input metadata (entry point, etc.)
    const struct dxil_spirv_debug_options *dbg,
    const struct dxil_spirv_runtime_conf *conf,
    struct dxil_spirv_object *out    // Output DXIL blob
);
```

The output is a DXBC container (not raw DXIL). The DXBC container includes:
- DXIL bytecode (the actual shader)
- PSV0 (Pipeline State Validation) section with resource binding metadata
- RDAT (Runtime Data) section
- Hash and signature sections

---

## D3D12EnableExperimentalFeatures for Unsigned DXIL

D3D12 normally requires DXIL to be signed by the DXIL validator (`dxil.dll` on Windows, not available on Linux). Mesa's `spirv_to_dxil` produces unsigned DXIL.

The Dozen Vulkan driver gets a special bypass from the D3D12 runtime on WSL2. Standalone apps do not get this bypass by default.

**Solution**: Call `D3D12EnableExperimentalFeatures` with the `D3D12ExperimentalShaderModels` GUID before creating the device:

```c
static const GUID D3D12ExperimentalShaderModels = {
    0x76f5573e, 0xf13a, 0x40f5,
    {0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f}
};

HRESULT hr = D3D12EnableExperimentalFeatures(
    1, &D3D12ExperimentalShaderModels, NULL, NULL);
```

This MUST be called before `D3D12CreateDevice`. It tells the runtime to accept unsigned DXIL shader bytecode. Without this, `CreateComputePipelineState` returns `E_INVALIDARG` (0x80070057) for every shader.

**Critical ordering**:
1. `D3D12EnableExperimentalFeatures()` -- first
2. `DXCoreCreateAdapterFactory()` -- enumerate adapters
3. `D3D12CreateDevice()` -- create device with experimental features active

If you create the device first and then enable experimental features, it does NOT apply to the existing device.

---

## PSV0 Section Parsing

The PSV0 (Pipeline State Validation v0) section in the DXBC container describes all resource bindings. We parse it to auto-detect SRV/UAV/CBV counts instead of requiring manual specification.

### DXBC Container Format

```
DXBC Header (32 bytes)
  magic: "DXBC"
  hash: 16 bytes
  version: 1
  total_size: uint32
  chunk_count: uint32

Chunk Offset Table
  offset[0]: uint32  -- points to first chunk
  offset[1]: uint32  -- points to second chunk
  ...

Chunks:
  Each chunk:
    fourcc: 4 bytes (e.g., "PSV0", "DXIL", "RDAT")
    size: uint32
    data: [size bytes]
```

### PSV0 Resource Table

The PSV0 chunk contains a resource binding table after a fixed header:

```c
struct PSV0_Header {
    uint32_t unknown[6];  // Version-dependent header fields
    uint32_t resource_count;
    uint32_t resource_stride;  // Bytes per resource entry (typically 24)
};

struct PSV0_Resource {
    uint32_t resource_type;  // See constants below
    uint32_t space;
    uint32_t lower_bound;    // Register start
    uint32_t upper_bound;    // Register end
    // ... additional fields
};
```

### Resource Type Constants

```c
#define PSV_INVALID       0
#define PSV_SAMPLER       1
#define PSV_CBV           2
#define PSV_SRV_TYPED     3
#define PSV_SRV_RAW       4  // NonWritable SSBOs become this
#define PSV_SRV_STRUCT    5
#define PSV_UAV_TYPED     6
#define PSV_UAV_RAW       7  // Writable SSBOs become this
#define PSV_UAV_STRUCT    8
#define PSV_UAV_STRUCT_CTR 9
```

`spirv_to_dxil` maps SPIR-V `NonWritable` decorated storage buffers to SRV_RAW (type 4), and writable storage buffers to UAV_RAW (type 7). Uniform buffers become CBV (type 2).

Our `create_root_sig_from_dxil()` function:
1. Finds the PSV0 chunk in the DXBC container
2. Reads the resource table
3. Counts SRVs (types 3-5), UAVs (types 6-9), CBVs (type 2)
4. Records register ranges for each type
5. Builds the root signature accordingly

---

## Root Signature Construction

D3D12 root signatures define how shader resources are bound to the pipeline. We use descriptor tables (not root descriptors) for maximum flexibility.

The root signature layout (always in this order):

```
Root Parameter 0: Descriptor Table [SRV range: t0..tN]
Root Parameter 1: Descriptor Table [UAV range: u0..uN]
Root Parameter 2: Descriptor Table [CBV range: b0..bN]
```

Only non-empty types get root parameters. If a shader has 0 CBVs, the root signature has only 2 parameters (SRV table + UAV table).

### Serialization

```c
D3D12_ROOT_PARAMETER params[3];
int param_count = 0;

if (num_srvs > 0) {
    D3D12_DESCRIPTOR_RANGE srv_range = {
        .RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        .NumDescriptors = num_srvs,
        .BaseShaderRegister = 0,
        .RegisterSpace = 0,
        .OffsetInDescriptorsFromTableStart = 0
    };
    params[param_count++] = (D3D12_ROOT_PARAMETER){
        .ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        .DescriptorTable = { 1, &srv_range },
        .ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL
    };
}
// Similarly for UAV and CBV...

D3D12_ROOT_SIGNATURE_DESC desc = {
    .NumParameters = param_count,
    .pParameters = params,
    .Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE
};

D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error);
device->lpVtbl->CreateRootSignature(device, 0, blob->GetBufferPointer(),
    blob->GetBufferSize(), &IID_ID3D12RootSignature, &root_sig);
```

---

## SRV vs UAV Descriptors

This was the most subtle blocker. Both SRVs and UAVs point to the same type of buffer (ID3D12Resource), but they are DIFFERENT descriptor types that the GPU treats differently.

### SRV (Shader Resource View) -- Read-Only

```c
D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc = {
    .Format = DXGI_FORMAT_R32_TYPELESS,
    .ViewDimension = D3D12_SRV_DIMENSION_BUFFER,
    .Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
    .Buffer = {
        .FirstElement = 0,
        .NumElements = buffer_size / 4,  // Number of 32-bit elements
        .StructureByteStride = 0,
        .Flags = D3D12_BUFFER_SRV_FLAG_RAW  // Raw byte access
    }
};
device->lpVtbl->CreateShaderResourceView(device, resource, &srv_desc, cpu_handle);
```

### UAV (Unordered Access View) -- Read/Write

```c
D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc = {
    .Format = DXGI_FORMAT_R32_TYPELESS,
    .ViewDimension = D3D12_UAV_DIMENSION_BUFFER,
    .Buffer = {
        .FirstElement = 0,
        .NumElements = buffer_size / 4,
        .StructureByteStride = 0,
        .Flags = D3D12_BUFFER_UAV_FLAG_RAW
    }
};
device->lpVtbl->CreateUnorderedAccessView(device, resource, NULL, &uav_desc, cpu_handle);
```

### Key Differences

| Aspect | SRV | UAV |
|--------|-----|-----|
| Access | Read-only | Read/Write |
| DXIL register | t-register (t0, t1, ...) | u-register (u0, u1, ...) |
| Descriptor creation | CreateShaderResourceView | CreateUnorderedAccessView |
| SPIR-V mapping | `NonWritable` storage buffer | Writable storage buffer |
| Root param type | D3D12_DESCRIPTOR_RANGE_TYPE_SRV | D3D12_DESCRIPTOR_RANGE_TYPE_UAV |

If you create a UAV descriptor for a buffer that the shader expects as SRV, the GPU reads garbage or zeros. This was the root cause of the "dispatch outputs all zeros" bug.

---

## Per-Dispatch Overhead Breakdown

A single dispatch operation involves:

1. **Allocate output buffer** (~0.1ms): `CreateCommittedResource` for the result buffer
2. **Create CBV if needed** (~0.05ms): Upload uniform parameters to constant buffer
3. **Create descriptors** (~0.05ms): SRV + UAV + CBV descriptors in CPU descriptor heap, then copy to GPU-visible heap
4. **Record commands** (~0.05ms):
   - Reset command allocator
   - Reset command list
   - Set pipeline state + root signature
   - Set descriptor heaps
   - Set compute root descriptor tables
   - Dispatch(groupsX, groupsY, groupsZ)
   - Close command list
5. **Execute** (~0.3ms): `ExecuteCommandLists` + signal fence
6. **Wait** (~0.2ms): CPU waits for GPU fence to reach target value
7. **Readback** (~0.2ms): Copy to readback heap + Map + memcpy + Unmap

**Total**: ~0.9ms per dispatch (vs 10-18ms with Dozen)

**Optimization opportunities** (not yet implemented):
- Batch multiple dispatches into one command list (eliminate per-dispatch fence wait)
- Reuse command allocators (avoid per-dispatch reset overhead)
- Pre-allocate descriptor heaps (eliminate per-dispatch descriptor creation)
- Persistent buffer pool (eliminate per-dispatch buffer allocation)
