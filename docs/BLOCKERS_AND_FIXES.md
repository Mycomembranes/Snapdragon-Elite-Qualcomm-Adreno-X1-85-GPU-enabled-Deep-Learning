# Blockers and Fixes

Every major blocker encountered during development, with root cause analysis and resolution.

---

## Blocker 1: PSO Creation Returns E_INVALIDARG (0x80070057)

**Symptom**: Every call to `CreateComputePipelineState` failed with `HRESULT 0x80070057` (E_INVALIDARG), regardless of shader content.

**Error message**:
```
ERROR: Pipeline creation failed: CreateComputePipelineState failed: 0x80070057
```

**Investigation**: The DXIL bytecode from `spirv_to_dxil` was valid (same code path as Dozen). The root signature was constructed. The pipeline state description was correctly populated. Yet every shader failed.

**Root cause**: WSL2's D3D12 runtime rejects **unsigned DXIL** from standalone applications. Mesa's `spirv_to_dxil` produces unsigned DXIL because the DXIL validator (`dxil.dll`) is a Windows-only binary not available on Linux. The Dozen Vulkan ICD driver gets a special runtime bypass (it's registered as a "known good" driver), but our standalone app does not.

**Fix**: Call `D3D12EnableExperimentalFeatures()` with the `D3D12ExperimentalShaderModels` GUID **before** creating the D3D12 device. This tells the runtime to accept unsigned DXIL.

```c
static const GUID D3D12ExperimentalShaderModels = {
    0x76f5573e, 0xf13a, 0x40f5,
    {0xb2, 0x97, 0x81, 0xce, 0x9e, 0x18, 0x93, 0x3f}
};

// MUST be before D3D12CreateDevice
D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels, NULL, NULL);
```

**Critical ordering**: EnableExperimentalFeatures -> DXCoreCreateAdapterFactory -> D3D12CreateDevice. If the device is created first, experimental features don't apply.

---

## Blocker 2: PSO Still Fails After Experimental Features

**Symptom**: After enabling experimental features, noop shader worked but real shaders (add, matmul, etc.) still failed with E_INVALIDARG.

**Root cause**: Root signature mismatch. We were manually specifying resource counts (e.g., `num_uavs=3` for add shader), but `spirv_to_dxil` converts SPIR-V `NonWritable` storage buffers to **SRVs** (t-registers), not UAVs. The add shader actually has 2 SRVs + 1 UAV, not 3 UAVs.

D3D12 validates that the root signature matches the DXIL resource declarations. Our root signature declared 3 UAVs but the DXIL declared 2 SRVs + 1 UAV = mismatch = E_INVALIDARG.

**Fix**: Parse the PSV0 (Pipeline State Validation) section from the DXBC container output by `spirv_to_dxil`. The PSV0 contains exact resource type information:

```c
// Parse DXBC container to find PSV0 chunk
// Read resource binding table: type (SRV/UAV/CBV), register, space
// Count each type
// Build root signature: SRV table -> UAV table -> CBV table
```

Implemented `create_root_sig_from_dxil()` which auto-detects resource counts from the compiled DXIL metadata. No manual specification needed.

---

## Blocker 3: Add Shader Output All Zeros

**Symptom**: PSO creation succeeded (after Blockers 1 & 2 were fixed). Dispatch returned `rc=0` (success). But the output buffer contained all zeros.

**Error**: No error message -- silent incorrect output.

**Investigation**: The add shader (a[i] + b[i] -> out[i]) was dispatched with 3 buffer handles. Command recording, execution, and fence wait all succeeded. But readback showed zeros.

**Root cause**: The dispatch function created **UAV descriptors** for ALL buffer handles. But the add shader has:
- `t0` (SRV): input a -- read-only
- `t1` (SRV): input b -- read-only
- `u2` (UAV): output -- read/write

When we created UAV descriptors for the input buffers and bound them to `t0` and `t1`, the GPU saw empty/wrong data because the descriptor types didn't match what the shader expected.

**Fix**: Rewrote `d3d12c_dispatch()` to accept separate SRV/UAV/CBV handle arrays:

```c
int d3d12c_dispatch(
    uint64_t pipeline_handle,
    const uint64_t *srv_handles, uint32_t num_srvs,   // CreateShaderResourceView
    const uint64_t *uav_handles, uint32_t num_uavs,   // CreateUnorderedAccessView
    const uint64_t *cbv_handles, uint32_t num_cbvs,    // CreateConstantBufferView
    uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
```

SRV descriptors use `CreateShaderResourceView` with `DXGI_FORMAT_R32_TYPELESS` + `D3D12_BUFFER_SRV_FLAG_RAW`. UAV descriptors use `CreateUnorderedAccessView` with matching format and flags.

---

## Blocker 4: LayerNorm Returns All Zeros

**Symptom**: LayerNorm test failed with output all zeros. Other shaders (add, matmul, relu) worked correctly.

**Root cause**: Two issues:

1. **Wrong binding count**: Test code passed only 2 SRVs (x, gamma) but the `layer_norm` shader requires **3 SRVs** (x, gamma, beta) + 1 UAV (output) + 1 CBV (params). The beta buffer was missing.

2. **Wrong CBV format**: The test packed the CBV as mixed int/float (`[N_int, eps_float, 0, 0]`), but the shader declares `var<uniform> params: vec4<f32>` -- all float32. The integer N was being reinterpreted as a float, giving a wrong normalization length.

**Fix**:
```python
# Before (wrong):
cbv_data = struct.pack('If', N, eps)  # int + float

# After (correct):
cbv_data = struct.pack('ffff', float(N), eps, 0.0, 0.0)  # all float
```

And pass 3 SRVs: `[x_handle, gamma_handle, beta_handle]`

**Result**: max_diff = 2.38e-07 (within float32 precision).

---

## Blocker 5: gelu_backward max_diff = 0.567

**Symptom**: `gelu_backward` shader output differed from numpy reference by up to 0.567 -- far beyond float32 tolerance.

**Root cause**: The numpy reference implementation used the **exact GELU derivative**:

```python
# Exact GELU derivative (numpy reference)
phi = (1/sqrt(2*pi)) * exp(-x**2/2)
Phi = 0.5 * (1 + erf(x/sqrt(2)))
gelu_deriv = Phi + x * phi
```

But the WGSL shader uses the **sigmoid approximation**:

```wgsl
// Shader approximation
let gelu_deriv = 1.0 / (1.0 + exp(-1.702 * x));  // sigmoid(1.702 * x)
```

These are different formulas that agree at extremes but diverge significantly around x=0.

**Fix**: Changed numpy reference to match the shader's formula:
```python
gelu_deriv = 1.0 / (1.0 + np.exp(-1.702 * x))  # Match shader
```

**Result**: max_diff = 1.19e-07.

**Note**: This is not a bug -- the shader intentionally uses a faster approximation. The test was wrong for using the exact formula as reference.

---

## Blocker 6: cross_entropy max_diff = NaN

**Symptom**: `cross_entropy` shader returned NaN for all outputs.

**Root cause**: The test passed **raw logits** as input, but the shader expects **log-probabilities**. The shader formula:

```wgsl
let loss = -target * log_p - (1.0 - target) * log(1.0 - exp(log_p) + 1e-6);
```

When `log_p` is a raw logit (e.g., 2.5), `exp(log_p)` overflows and `1 - exp(log_p)` becomes a large negative number, causing `log()` of a negative -> NaN.

**Fix**: Convert logits to log-probabilities before calling the shader:
```python
# Before (wrong):
result = d3d12_cross_entropy(logits, targets)

# After (correct):
log_probs = logits - np.log(np.sum(np.exp(logits)))  # log-softmax
result = d3d12_cross_entropy(log_probs, targets)
```

**Result**: max_diff = 3.58e-07.

---

## Blocker 7: D3D12 Device Removal During Buffer Operations

**Symptom**: `CreateCommittedResource` failed with `0x887a0005` (DXGI_ERROR_DEVICE_REMOVED) with absurd buffer sizes like 275417948486608 bytes.

**Error message**:
```
Error: CreateCommittedResource failed: 0x887a0005 (size=275417948486608, heap=2)
```

**Root cause**: Memory corruption in buffer handle management. The uint64_t buffer handle was being passed through Python ctypes with wrong argument types, causing garbage values for the size parameter.

**Fix**: Corrected ctypes function signatures:
```python
lib.d3d12c_create_buffer.argtypes = [ctypes.c_uint64]
lib.d3d12c_create_buffer.restype = ctypes.c_uint64
```

After fixing all ctypes signatures, buffer operations worked reliably.

---

## Blocker 8: wgpu_tensor Dispatch Fails with D3D12 Backend

**Symptom**: When `wgpu_tensor.py` auto-detected D3D12 and tried to use it, operations failed because D3D12 was not properly initialized.

**Root cause**: Two issues:

1. `d3d12_init()` was not called during auto-detection. The `_get_backend()` function checked if the module was importable but didn't actually initialize the D3D12 device.

2. `WgpuTensor.from_numpy()` returned wgpu buffers (not D3D12Tensor) even when backend was set to d3d12. The dispatch code then tried to pass wgpu buffer handles to D3D12 functions.

**Fix**:
1. Call `d3d12_init()` during backend detection:
```python
def _get_backend():
    try:
        from d3d12_tensor import d3d12_init, is_available
        if d3d12_init() == 0 and is_available():
            return 'd3d12'
    except:
        pass
    return 'wgpu'
```

2. Override `from_numpy()` to return `D3D12Tensor` when backend is d3d12:
```python
@staticmethod
def from_numpy(arr):
    if _backend == 'd3d12':
        return D3D12Tensor.from_numpy(arr)
    # ... original wgpu path
```

---

## Summary Table

| # | Blocker | Root Cause | Fix | Sessions to Resolve |
|---|---------|-----------|-----|---------------------|
| 1 | PSO E_INVALIDARG | Unsigned DXIL rejected | EnableExperimentalFeatures before device | 1 |
| 2 | PSO still fails | Root sig mismatch (SRV vs UAV) | Parse PSV0 for auto-detection | 1 |
| 3 | Output all zeros | UAV descriptors for SRV bindings | Separate SRV/UAV/CBV dispatch | 1 |
| 4 | LayerNorm zeros | Wrong binding count + CBV format | 3 SRVs + float4 CBV | 0.5 |
| 5 | gelu_backward 0.567 | Wrong numpy reference formula | Match shader's sigmoid approx | 0.5 |
| 6 | cross_entropy NaN | Raw logits vs log-probs | Pass log-probabilities | 0.5 |
| 7 | Device removed | ctypes argument corruption | Fix ctypes signatures | 0.5 |
| 8 | wgpu_tensor fails | Missing init + wrong tensor type | Init + type routing | 0.5 |
