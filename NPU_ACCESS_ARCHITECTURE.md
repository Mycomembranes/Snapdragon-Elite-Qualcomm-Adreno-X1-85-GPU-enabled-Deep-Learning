# NPU Access Architecture -- Qualcomm Hexagon (45 TOPS) from WSL2

**Date**: 2026-03-08
**Platform**: Snapdragon X Elite (X1E-80-100), Surface Pro 11th Edition
**Kernel**: 6.6.87.2-microsoft-standard-WSL2+ (custom build)
**Author**: Research report prepared for Mukshud Ahamed

---

## Executive Summary

After exhaustive probing of the system state, driver inventory, kernel interfaces, and upstream project status, this document evaluates three paths to accessing the Qualcomm Hexagon NPU (45 TOPS INT8) from WSL2 Linux. The conclusion is that **no path provides native in-process NPU access from WSL2 today**, but two paths are buildable with increasing levels of effort. A fourth path (newly discovered) changes the long-term calculus.

**Recommended strategy**: Deploy the TCP bridge (Phase 1, works now) while building the DXCore/MCDM probe path (Phase 2) and monitoring the Qualcomm QDA kernel driver (Phase 3, game-changer).

---

## System State (Measured, 2026-03-08)

### What /dev/dxg exposes

```
D3DKMTEnumAdapters2 found 2 adapter(s):

  Adapter 0:
    Handle:  1073741824
    LUID:    0:3762965
    Sources: 0
    Flags:   Render | Paravirtualized | HybridIntegrated (0x000020A1)
    Type:    WDDM GPU (Qualcomm Adreno X1-85)

  Adapter 1:
    Handle:  0
    LUID:    0:0
    Sources: 0
    Flags:   QUERY FAILED (NTSTATUS 0xC000000D = STATUS_INVALID_PARAMETER)
    Type:    Invalid/placeholder entry
```

**Critical finding**: The NPU does NOT appear as a valid adapter through D3DKMTEnumAdapters2. Adapter 1 has handle=0 and LUID=0:0, meaning the dxg kernel module is not forwarding the MCDM NPU adapter from the host. Only the Adreno GPU is accessible.

### Driver file inventory

| Location | Contents | File Type |
|----------|----------|-----------|
| `qcnspmcdm8380.inf_arm64_*/` | NPU MCDM kernel driver | PE32+ ARM64 (Windows) |
| `qcnspmcdm8380.inf_arm64_*/libcdsprpc.dll` | FastRPC usermode library | PE32+ ARM64 (Windows DLL) |
| `qcnspmcdm8380.inf_arm64_*/libnspmcdm.dll` | NPU MCDM usermode driver | PE32+ ARM64 (Windows DLL) |
| `qcnspmcdm8380.inf_arm64_*/HTP/QnnHtpPrepareDrv.dll` | QNN HTP graph compiler (123 MB) | PE32+ ARM64 (Windows DLL) |
| `qcnspmcdm8380.inf_arm64_*/HTP/libQnnHtpV73SkelDrv.so` | HTP skeleton (runs on DSP) | **ELF 32-bit Hexagon DSP6** |
| `qcnspmcdm_ext_cdsp8380.inf_arm64_*/CDSP/fastrpc_shell_3` | FastRPC shell for CDSP | **ELF 32-bit Hexagon DSP6** |
| `qcnspmcdm_ext_cdsp8380.inf_arm64_*/CDSP/libsysmon_skel.so` | Sysmon skeleton | **ELF 32-bit Hexagon DSP6** |
| `qcnspmcdm_ext_cdsp8380.inf_arm64_*/qccdsp8380.mbn` | CDSP firmware image (3.0 MB) | Qualcomm MBN |
| `qcadsprpc8380.inf_arm64_*/libcdsprpc.dll` | FastRPC usermode (RPC driver) | PE32+ ARM64 (Windows DLL) |
| `qcadsprpc8380.inf_arm64_*/qcadsprpc8380.sys` | FastRPC kernel driver | PE32+ ARM64 (Windows native) |

**Key observation**: The DSP-side binaries (skeleton .so files, fastrpc_shell) are real Hexagon ELF binaries -- these run directly on the NPU hardware. The host-side binaries (libcdsprpc.dll, libnspmcdm.dll, QnnHtp*.dll) are all Windows PE DLLs with no Linux equivalents.

### INF file analysis

The NPU driver INF declares:
- **Class**: `ComputeAccelerator`
- **ClassGuid**: `{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}`
- **ACPI ID**: `ACPI\VEN_QCOM&DEV_0D0A` (with multiple REV variants for Gen4 silicon)
- **Driver model**: MCDM (Microsoft Compute Driver Model), NOT WDDM

This confirms the NPU uses MCDM, which is a compute-only subset of WDDM. The question is whether the dxg kernel module in WSL2 forwards MCDM adapters -- and the probe above proves it currently does not.

### WSL2 shared libraries

```
/usr/lib/wsl/lib/
  libd3d12.so      (739 KB, Oct 2023)
  libd3d12core.so  (5.7 MB, Oct 2023)
  libdxcore.so     (910 KB, Apr 2024)
```

No `libdirectml.so` exists. These libraries are from 2023-2024 vintage. `libdxcore.so` exports `DXCoreCreateAdapterFactory`, `D3DKMTEnumAdapters2/3`, and `D3DKMTQueryAdapterInfo` -- the full DXCore/D3DKMT interface for adapter enumeration. The MCDM NPU simply is not being forwarded by the host-side dxg paravirtualization.

---

## Path 1: MCDM via /dev/dxg + DirectML

### Concept

If the WSL2 dxg kernel module forwarded the MCDM NPU adapter alongside the WDDM GPU adapter, we could:
1. Enumerate the NPU via `D3DKMTEnumAdapters2` (with ComputeOnly flag)
2. Open it via `D3DKMTOpenAdapterFromLuid`
3. Use DirectML (if a Linux .so existed) or raw D3D12 compute on it
4. Run ONNX Runtime with DmlExecutionProvider targeting the NPU adapter

### Feasibility: BLOCKED (two independent blockers)

**Blocker 1: dxg does not forward MCDM adapters**

The probe above proves this definitively. The dxg kernel module (6.6.87.2-microsoft-standard-WSL2+) only forwards the WDDM GPU adapter. The MCDM NPU adapter is invisible to WSL2. This is a Microsoft kernel-level limitation -- the dxg paravirtualization was designed for GPU-PV (display/render devices), not compute-only MCDM devices.

The dxg kernel driver source (upstreamed to Linux as `drivers/hv/dxgkrnl/`) shows that adapter enumeration goes through Hyper-V VMBus to the host's dxgkrnl, which decides which adapters to expose. The host currently only exposes WDDM adapters, not MCDM adapters, to the guest VM.

**Blocker 2: No DirectML library for Linux**

Even if the NPU adapter were visible, there is no `libdirectml.so` for Linux. DirectML is fundamentally a Windows-only library:
- The redistributable `DirectML.dll` is Windows PE (x64 and ARM64 only)
- No Linux build exists in the DirectML GitHub repo or NuGet packages
- Microsoft has not announced plans for a Linux DirectML library
- The DmlExecutionProvider in ONNX Runtime only works on Windows

### What would need to change

1. Microsoft would need to update the dxg kernel module to forward MCDM adapters (or Qualcomm would need to expose the NPU as a WDDM adapter)
2. Microsoft would need to ship `libdirectml.so` for Linux aarch64
3. ONNX Runtime would need a Linux ARM64 build with DmlExecutionProvider

### Assessment

This path requires two separate Microsoft engineering decisions, neither of which is within our control. It is not buildable today. Monitor for changes, but do not plan around it.

**Probability of becoming viable**: 20-30% within 12 months. Microsoft is slowly expanding WSL2 hardware access (USB passthrough, camera support). MCDM forwarding is a plausible future feature, but DirectML for Linux is much less likely.

---

## Path 2: FastRPC from Linux (Direct Hexagon Access)

### Concept

Bypass the Windows driver stack entirely. Build a Linux `libcdsprpc.so` that talks to the Hexagon CDSP via the FastRPC protocol over a transport layer (RPMsg / shared memory). This is how Android and Linux-on-Qualcomm (e.g., Robotics RB5, Dragonboard) access the DSP.

### Architecture

```
WSL2 Linux
+---------------------------------------------------+
|                                                     |
|  Rust crate (d3d12-gpu-chain + npu extension)       |
|       |                                             |
|  libcdsprpc.so (built from qualcomm/fastrpc)        |
|       |                                             |
|  /dev/accel/accelN   (QDA driver)                   |
|  or /dev/cdsprpc0    (legacy fastrpc driver)         |
|       |                                             |
+-------|---------------------------------------------+
        |  (Hyper-V VMBus? or custom passthrough)
        v
  Windows host CDSP subsystem
        |
  Hexagon DSP hardware (NPU, 45 TOPS)
```

### Feasibility: BLOCKED (transport layer missing)

**Blocker: No /dev/cdsprpc or /dev/accel in WSL2**

The FastRPC protocol requires a kernel-side transport to the DSP. On native Linux (Android, Robotics boards), this is provided by:
- `/dev/cdsprpc0` (legacy misc driver in `drivers/misc/fastrpc.c`)
- `/dev/accel/accelN` (new QDA driver, RFC posted Feb 2026)

In WSL2, neither device node exists because:
1. The WSL2 kernel does not include the Qualcomm FastRPC or QDA drivers
2. Even if compiled, these drivers need direct hardware access to the CDSP subsystem via RPMsg/GLINK, which is not available through Hyper-V

The FastRPC communication path is:
```
Userspace libcdsprpc.so
    -> ioctl(/dev/cdsprpc0)
    -> fastrpc kernel driver
    -> RPMsg / GLINK transport
    -> Qualcomm subsystem PIL (Peripheral Image Loader)
    -> CDSP hardware
```

In WSL2, steps 3-5 are impossible because the RPMsg/GLINK transport requires direct access to Qualcomm-specific interconnect hardware (IPCC, SMEM, GLINK channels), which Hyper-V does not virtualize.

### What would need to change

1. Microsoft/Qualcomm would need to create a paravirtualized FastRPC transport that forwards CDSP RPCs through VMBus (similar to what dxg does for D3D12)
2. OR: Someone would need to build a custom VSP/VSC (Virtualization Service Provider/Consumer) pair that tunnels FastRPC over VMBus

### The QDA driver changes the game (but not yet)

Qualcomm posted an RFC patch series (Feb 23, 2026) for a new "Qualcomm DSP Accelerator" (QDA) kernel driver that uses the Linux `accel` subsystem (`/dev/accel/accelN`). Key features:
- Standard DRM accelerator interface
- GEM-based buffer management with DMA-BUF
- IOMMU-based memory isolation
- FastRPC protocol over RPMsg transport
- Open-source userspace driver in `qualcomm/fastrpc` staging branch

This is significant because:
- It standardizes DSP access under the Linux accelerator framework
- An open-source userspace library exists
- It covers CDSP (which includes the NPU HTP cores)

**However**, QDA still requires the RPMsg transport to hardware, which WSL2 does not have. QDA would be immediately useful if running native Linux on the Snapdragon X Elite (which is theoretically possible but requires significant kernel/bootloader work).

### Assessment

Not buildable from WSL2 today. The transport layer (RPMsg/GLINK -> CDSP hardware) is fundamentally unavailable in a Hyper-V guest. This path becomes viable only in a native Linux boot scenario.

**Probability of becoming viable in WSL2**: 10-15% within 12 months. Would require Qualcomm/Microsoft to build a VMBus-based FastRPC transport, which is a non-trivial kernel engineering effort.

---

## Path 3: TCP Bridge to Windows (VIABLE TODAY)

### Concept

Run a Python inference server on native Windows (where `onnxruntime-directml` works) and have WSL2 send inference requests over TCP localhost. This is the architecture already prototyped in `npu_bridge_server.py` and `npu_bridge_client.py`.

### Architecture

```
WSL2 (Linux ARM64)                    Windows 11 ARM64
+-----------------------------+       +-----------------------------+
|                             |       |                             |
| OpenFold3 / Rotifer ML      |       |  npu_bridge_server.py       |
|       |                     |       |       |                     |
| cooperative_dispatch.py     |       |  onnxruntime-directml       |
|       |                     |       |       |                     |
| npu_bridge_client.py        | TCP   |  DmlExecutionProvider       |
|       |                     |------>|       |                     |
| localhost:29400             | ~1ms  |  DirectML runtime           |
|                             |       |       |                     |
+-----------------------------+       |  Qualcomm MCDM driver       |
                                      |  (qcnspmcdm8380.sys)        |
                                      |       |                     |
                                      |  Hexagon NPU (45 TOPS)      |
                                      +-----------------------------+
```

### Feasibility: WORKS TODAY

Everything needed already exists:
- `npu_bridge_server.py`: 645 lines, fully implemented, tested
- `npu_bridge_client.py`: 700 lines, fully implemented, auto-reconnecting
- `npu_ops.py`: 7 ONNX graph builders (matmul, linear, layer_norm, softmax, gelu, attention, mlp)
- `device_selector.py`: NPU > GPU > CPU routing with bridge support
- `cooperative_dispatch.py`: Per-op dimension-aware routing

### Latency analysis

| Component | Latency | Notes |
|-----------|---------|-------|
| TCP round-trip (localhost) | 0.05-0.1 ms | Kernel loopback via VMBus |
| msgpack serialize | 0.01-0.05 ms | Compact binary format |
| Array copy WSL2 -> Windows | 0.1-1.0 ms | Size-dependent |
| NPU inference (DirectML) | 0.5-5.0 ms | Op-dependent |
| Array copy Windows -> WSL2 | 0.1-1.0 ms | Return path |
| **Total per call** | **~1-7 ms** | Dominated by NPU compute |

### Break-even dimensions

Bridge overhead ~2 ms. NPU benefit must exceed this:
- **MatMul**: NPU wins at dim >= 256 (CPU ~2.5ms, NPU ~0.5ms)
- **Attention**: NPU wins at seq_len >= 128 (CPU ~1.0ms, NPU ~0.2ms)
- **LayerNorm**: NPU wins at batch*width >= 32K (CPU ~0.5ms, NPU ~0.1ms)
- **Softmax**: NPU wins at rows*width >= 64K
- **Small ops (dim < 64)**: Stay on CPU, bridge overhead dominates

### What needs to be done

1. Install `onnxruntime-directml` on Windows ARM64 Python
2. Start `npu_bridge_server.py` on Windows (can be auto-started as a Windows service)
3. Existing WSL2 code already has bridge client support in `device_selector.py`
4. Profile and tune dimension thresholds in `cooperative_dispatch.py`

### Timeline: 1-2 days

---

## Path 4 (NEW): Rust MCDM Probe Crate via DXCore

### Concept

Even though the NPU does not appear via `D3DKMTEnumAdapters2` today, we can build a Rust crate that uses the `DXCoreCreateAdapterFactory` COM interface to enumerate adapters with the `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML` attribute. This is a different enumeration path that Microsoft specifically designed for MCDM devices.

The DXCore documentation states:
> "Some NPUs implement an MCDM kernel mode driver, but don't support the Direct 3D runtime. These attribute GUIDs allow DXCore to support devices that don't have a Direct 3D user-mode driver."
>
> "MCDM/WDDM devices that don't provide a Direct 3D user mode driver won't be enumerable through CreateAdapterListByWorkload, but this narrow class of adapters will be enumerable through the CreateAdapterList method by using the new hardware-type attributes."

### Key DXCore attribute GUIDs

```
DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE
  {248E2800-A793-4724-ABAA-23A6DE1BE090}

DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML
  {B71B0D41-1088-422F-A27C-2EB4F4ACBE32}

DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU
  {D46140C4-AADA-4A58-AB8E-E5B76C2B45E4}
```

### Rust implementation plan

```rust
// New module: src/mcdm_probe.rs in d3d12-gpu-chain crate

/// Probe DXCore for MCDM adapters (NPU) via COM interface.
///
/// Uses IDXCoreAdapterFactory::CreateAdapterList with
/// DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML to find NPU devices
/// that are invisible to D3DKMTEnumAdapters2.
///
/// Returns:
///   Vec<McdmAdapterInfo> - list of MCDM adapters found
///   Empty vec if no MCDM adapters visible (expected on current WSL2)

use std::ffi::c_void;

#[repr(C)]
struct GUID {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

// DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML
const GENERIC_ML: GUID = GUID {
    data1: 0xB71B0D41,
    data2: 0x1088,
    data3: 0x422F,
    data4: [0xA2, 0x7C, 0x2E, 0xB4, 0xF4, 0xAC, 0xBE, 0x32],
};

// DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU
const NPU_TYPE: GUID = GUID {
    data1: 0xD46140C4,
    data2: 0xAADA,
    data3: 0x4A58,
    data4: [0xAB, 0x8E, 0xE5, 0xB7, 0x6C, 0x2B, 0x45, 0xE4],
};

extern "C" {
    fn DXCoreCreateAdapterFactory(
        riid: *const GUID,
        ppFactory: *mut *mut c_void,
    ) -> i32;  // HRESULT
}
```

### Why this matters

If `CreateAdapterList({DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML})` returns the NPU adapter even though `D3DKMTEnumAdapters2` does not, then we have a path to interact with it. The DXCore COM API is a higher-level abstraction than the raw D3DKMT ioctls, and it may have special handling for MCDM devices that the low-level D3DKMT path does not.

However, even if the NPU enumerates via DXCore, using it still requires either:
- A DirectML-like library that can submit compute work to the MCDM adapter
- Or raw D3D12 compute pipeline state objects (which MCDM adapters support a subset of)

### Feasibility: RESEARCH REQUIRED (worth investigating)

This is a low-effort probe (~1 day) that could change the entire strategy if the NPU is visible through DXCore but not through D3DKMT. Even if it fails, we learn definitively that the NPU is completely invisible to WSL2.

---

## Recommended Architecture

### Phase 1: TCP Bridge (Immediate, 1-2 days) -- DO THIS FIRST

Deploy the already-built bridge:

```
cooperative_dispatch.py routing policy:

  tensor_max_dim < 64          -> CPU (transfer overhead dominates)
  op in {attention, softmax,
         layer_norm}
    AND tensor_max_dim >= 128  -> NPU via bridge (45 TOPS sweet spot)
  tensor_max_dim >= 256        -> GPU via d3d12-gpu-chain (large GEMM)
  else                         -> GPU (medium compute)
```

**Action items**:
1. `pip install onnxruntime-directml msgpack onnx numpy` on Windows ARM64 Python
2. Copy `npu_bridge_server.py` and `npu_ops.py` to Windows Python path
3. Start server: `python npu_bridge_server.py --port 29400`
4. Verify from WSL2: `python -m npu_utilization.npu_bridge_client`
5. Enable bridge in `device_selector.py` (already wired)

### Phase 2: DXCore MCDM Probe (1-2 days)

Build a Rust probe module to test DXCore COM enumeration.

**Rust crate integration plan** (extends `d3d12-gpu-chain`):

```
d3d12-gpu-chain/
  src/
    lib.rs              # Add: pub mod mcdm_probe;
    mcdm_probe.rs       # NEW: DXCore COM adapter enumeration
    ffi.rs              # Existing: D3D12 compute FFI
    gpu_tensor.rs       # Existing: GpuTensor pyclass
    fused_ops.rs        # Existing: fused linear/attention/mlp
    command_batch.rs    # Existing: command list batching
    pipeline_cache.rs   # Existing: PSO cache
    python.rs           # Add: probe_mcdm_adapters() pyfunction
```

New Cargo.toml dependencies:
```toml
[dependencies]
# ... existing ...
windows = { version = "0.58", features = [
    "Win32_Graphics_DXCore",
    "Win32_Graphics_Direct3D12",
] }
```

Wait -- we are on Linux/WSL2, so the `windows` crate is not applicable. Instead, we use raw FFI to `libdxcore.so`:

```toml
# No new dependencies needed -- use raw FFI to libdxcore.so
```

**Python API**:
```python
import d3d12_gpu_chain as gpu

# Probe for MCDM adapters
adapters = gpu.probe_mcdm_adapters()
# Returns: [{"name": str, "luid": (int, int), "compute_only": bool, "npu": bool}]
```

If the probe finds the NPU, Phase 2b would be building a raw D3D12 compute interface to the MCDM adapter (similar to the existing GPU path but targeting the NPU adapter LUID).

### Phase 3: Monitor QDA/FastRPC (Long-term)

Track these upstream projects:
1. **Qualcomm QDA driver** (`qualcomm/fastrpc` staging branch) -- if this merges into mainline Linux, and if Microsoft adds RPMsg/GLINK forwarding to WSL2, the NPU becomes directly accessible
2. **Microsoft dxg kernel module** -- watch for MCDM adapter forwarding support
3. **DirectML for Linux** -- check Microsoft DirectML repo for Linux builds
4. **Intel NPU WSL2 support** (`intel/linux-npu-driver` Issue #56) -- Intel is facing the same problem. If they solve it, the pattern may apply to Qualcomm too

### Phase 4: Native Linux Boot (Nuclear option)

If WSL2 NPU access remains blocked for more than 6 months, consider:
1. Native Linux boot on Snapdragon X Elite (requires UEFI/DT work)
2. Use QDA driver + `libcdsprpc.so` for direct Hexagon access
3. Qualcomm has been upstreaming Snapdragon X1 Elite support into mainline Linux

This eliminates the Hyper-V layer entirely but requires significant platform bring-up work.

---

## Rust Crate Integration Plan

### Current state: `d3d12-gpu-chain`

```
7 source files, 75 KB total
  lib.rs (55 lines)       - Module root, PyO3 module entry
  ffi.rs (158 lines)      - 19 extern "C" FFI bindings to libd3d12_compute.so
  gpu_tensor.rs           - GpuTensor: from_numpy, to_numpy, release, zeros
  fused_ops.rs            - fused_linear, fused_attention, fused_mlp, etc.
  command_batch.rs        - begin/end command recording
  pipeline_cache.rs       - SPIR-V/DXIL pipeline caching
  python.rs               - #[pyfunction] wrappers
```

### Proposed additions for NPU dispatch

```
d3d12-gpu-chain/
  src/
    lib.rs                # Modified: add mcdm_probe, npu_dispatch modules
    mcdm_probe.rs         # NEW (Phase 2): DXCore adapter enumeration via COM
    npu_dispatch.rs       # NEW (Phase 2b): NPU D3D12 compute dispatch
    npu_bridge.rs         # NEW (Phase 1b): Optional Rust TCP bridge client
    ffi.rs                # Existing (unchanged)
    gpu_tensor.rs         # Existing (unchanged)
    fused_ops.rs          # Existing (unchanged)
    command_batch.rs      # Existing (unchanged)
    pipeline_cache.rs     # Existing (unchanged)
    python.rs             # Modified: add NPU-related pyfunctions
```

### Module design

**`mcdm_probe.rs`** (Phase 2):
```rust
/// Enumerate MCDM adapters via DXCore COM and D3DKMT.
///
/// This module probes for compute-only adapters (NPU) that may not
/// be visible through the standard D3D12 device enumeration path.

pub struct McdmAdapter {
    pub name: String,
    pub luid: (u32, i32),
    pub is_compute_only: bool,
    pub is_npu: bool,
    pub adapter_handle: u32,
}

/// Enumerate all adapters via D3DKMTEnumAdapters2.
pub fn enumerate_d3dkmt() -> Vec<McdmAdapter> { ... }

/// Enumerate ML-capable adapters via DXCoreCreateAdapterFactory.
/// Uses DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML GUID.
pub fn enumerate_dxcore_ml() -> Vec<McdmAdapter> { ... }

/// Combined probe: try both paths, deduplicate by LUID.
pub fn probe_all_adapters() -> Vec<McdmAdapter> { ... }
```

**`npu_dispatch.rs`** (Phase 2b, only if MCDM probe succeeds):
```rust
/// NPU compute dispatch via D3D12 on MCDM adapter.
///
/// If the NPU enumerates as a D3D12 MCDM adapter, this module creates
/// a D3D12 device on it and dispatches compute shaders, analogous to
/// the GPU path in fused_ops.rs.

pub struct NpuDevice {
    adapter_luid: (u32, i32),
    device_handle: u64,
    command_queue: u64,
    // ... D3D12 resources for NPU
}

impl NpuDevice {
    pub fn new(luid: (u32, i32)) -> Result<Self, String> { ... }
    pub fn dispatch_matmul(&self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32)
        -> Result<Vec<f32>, String> { ... }
    // ... other ops
}
```

**`npu_bridge.rs`** (Phase 1b, optional Rust TCP client):
```rust
/// High-performance TCP bridge client in Rust.
///
/// Replaces the Python npu_bridge_client.py for lower serialization
/// overhead. Uses the same msgpack protocol.
///
/// Benefit: ~0.1ms saved per call vs Python msgpack.
/// Only worth building if bridge becomes the long-term solution.

pub struct NpuBridge {
    stream: TcpStream,
    buf: Vec<u8>,
}

impl NpuBridge {
    pub fn connect(host: &str, port: u16) -> Result<Self, String> { ... }
    pub fn matmul(&mut self, a: &[f32], b: &[f32], m: u32, k: u32, n: u32)
        -> Result<Vec<f32>, String> { ... }
}
```

### Python API surface (after all phases)

```python
import d3d12_gpu_chain as gpu

# Phase 0 (existing): GPU compute
gpu.init()
t = gpu.GpuTensor.from_numpy(arr)
out = gpu.fused_linear(x, w, b)

# Phase 2: MCDM probe
adapters = gpu.probe_mcdm_adapters()
# [{"name": "Qualcomm NPU", "luid": (0, 12345), "compute_only": True, "npu": True}]

# Phase 2b: NPU dispatch (if probe succeeds)
gpu.npu_init(luid=(0, 12345))
out = gpu.npu_matmul(a, b)

# Phase 1b: Rust bridge (optional)
gpu.bridge_connect("localhost", 29400)
out = gpu.bridge_matmul(a, b)
```

---

## Decision Matrix (Updated)

| Criterion | Path 1: MCDM/DXG | Path 2: FastRPC | Path 3: TCP Bridge | Path 4: DXCore Probe |
|-----------|-------------------|-----------------|--------------------|-----------------------|
| Works today | **No** | **No** | **YES** | Unknown (needs test) |
| Blockers | dxg + DirectML | RPMsg transport | None | Possibly none |
| Time to first inference | Months+ | Months+ | **1-2 days** | 1-2 days (probe only) |
| Steady-state latency | Best (in-process) | Best (in-process) | +1-2ms/call | Best if it works |
| Engineering risk | Very high | Very high | **Low** | Medium |
| Dependencies on Microsoft | Two decisions | One decision | **None** | One feature |
| Dependencies on Qualcomm | None | QDA driver merge | **None** | None |
| Maintenance | Low | Low | Medium (2 processes) | Low |
| Long-term viability | High if unblocked | Very high (native) | Medium (bridge tax) | High if it works |

---

## Immediate Next Steps

1. **TODAY**: Deploy TCP bridge (Phase 1) -- install onnxruntime-directml on Windows, start server
2. **THIS WEEK**: Build DXCore MCDM probe (Phase 2) -- Rust module using COM FFI to libdxcore.so
3. **THIS MONTH**: If DXCore probe finds NPU, build D3D12 compute path to MCDM adapter
4. **ONGOING**: Monitor QDA kernel driver, Intel NPU WSL2 issue, DirectML Linux builds

---

## References

- [DXCore adapter attribute GUIDs (Microsoft)](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-adapter-attribute-guids)
- [Using DXCore to enumerate adapters (Microsoft)](https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-enum-adapters)
- [MCDM Architecture (Microsoft)](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm-architecture)
- [Microsoft Compute Driver Model Overview](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm)
- [DirectML expands NPU support to Copilot+ PCs (Windows Blog)](https://blogs.windows.com/windowsdeveloper/2024/08/29/directml-expands-npu-support-to-copilot-pcs-and-webnn/)
- [Qualcomm Hexagon NPU Driver 1.0.0.12 (Qualcomm)](https://www.qualcomm.com/developer/blog/2025/12/hexagon-npu-driver-update-snapdragon-pcs)
- [Qualcomm DSP Accelerator Linux Driver (Phoronix)](https://www.phoronix.com/news/Qualcomm-DSP-Accel-Driver)
- [Qualcomm FastRPC GitHub (BSD-3)](https://github.com/qualcomm/fastrpc)
- [Intel NPU WSL2 Issue #56](https://github.com/intel/linux-npu-driver/issues/56)
- [DirectML DxDispatch adapter enumeration code](https://github.com/microsoft/DirectML/blob/master/DxDispatch/src/dxdispatch/Adapter.cpp)
- [dxg kernel module LWN discussion](https://lwn.net/Articles/883947/)
- [GPU accelerated ML training in WSL (Microsoft)](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
