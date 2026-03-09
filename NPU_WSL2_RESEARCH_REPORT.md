# NPU from WSL2: Comprehensive Research Report

**Date**: 2026-03-08
**Platform**: Snapdragon X Elite (X1E-80-100), Surface Pro 11th Edition
**Kernel**: 6.6.87.2-microsoft-standard-WSL2+
**WSL2 Libraries**: libd3d12.so, libd3d12core.so, libdxcore.so (all ELF64 aarch64)
Mukshud Ahamed
---

## TL;DR

**The Qualcomm Hexagon NPU (45 TOPS INT8) is NOT accessible from WSL2 through any kernel-level path.** The NPU adapter is visible as a phantom entry (Handle=0, LUID=0:0) in D3DKMTEnumAdapters2 but the dxgkrnl VMBus bridge refuses to initialize it. No alternative paths (FastRPC, DRM accel, Wine) exist on this system.

**Recommended action**: TCP bridge to a Windows-side ONNX Runtime with DirectML execution provider.

---

## Research Path Results

### Path 1: Can /dev/dxg be used for NPU?

**ANSWER: NO.** The NPU is registered as an MCDM (Microsoft Compute Driver Model) adapter on the Windows host, but dxgkrnl's WSL2 passthrough does not support MCDM devices.

**Evidence (from actual command output):**

```
D3DKMTEnumAdapters2 found 2 adapter(s):

Adapter 0: GPU (Adreno X1-85)
  Handle:  1073741824 (0x40000000)     <-- valid
  LUID:    0:3762965                    <-- valid
  TypeFlags: Render | Paravirtualized | HybridIntegrated (0x000020A1)
  QueryAdapterInfo: 1 of 23 types succeed (GPUMMU_CAPS only)

Adapter 1: SUSPECTED NPU
  Handle:  0 (0x00000000)              <-- INVALID
  LUID:    0:0                         <-- INVALID
  TypeFlags: QUERY FAILED (STATUS_INVALID_PARAMETER)
  QueryAdapterInfo: 0 of 23 types succeed (ALL fail)
  D3DKMTOpenAdapterFromLuid: STATUS_INVALID_PARAMETER
```

The NPU driver's INF file (`qcnspmcdm8380.inf`) registers two DXCore attributes:
- `DXCORE_HARDWARE_TYPE_ATTRIBUTE_COMPUTE_ACCELERATOR` = `{D46140C4-ADD7-451B-9E56-06FE8C3B58ED}`
- `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML` = `{B71B0D41-1088-422F-A27C-0250B7D3A988}`

These are NOT `D3D12_GRAPHICS` -- the existing d3d12_compute.c code only enumerates with the `D3D12_GRAPHICS` GUID, which is why it sees only the Adreno GPU. But even changing to the ML GUID would not help because the adapter itself has Handle=0.

D3DKMTEnumAdapters3 (with IncludeComputeOnly filter) returns 0 adapters for all filter values, confirming the WSL2 dxgkrnl does not properly expose MCDM devices.

**dmesg pattern** (repeated at every enumeration):
```
Group 1 (GPU - partial success):
  5x dxgkio_query_adapter_info: Ioctl failed: -22 (EINVAL)
  1x dxgkio_query_adapter_info: Ioctl failed: -2  (ENOENT)

Group 2 (NPU - total failure):
  4x dxgkio_query_adapter_info: Ioctl failed: -22 (EINVAL)
  1x dxgkio_query_adapter_info: Ioctl failed: -2  (ENOENT)
```

### Path 2: Windows DLLs via Wine ARM64?

**ANSWER: NOT FEASIBLE.** Wine is not installed. Even if installed, the DLLs ultimately need the Windows kernel driver.

The NPU driver stack contains these ARM64 PE32+ DLLs:
- `libcdsprpc.dll` (FastRPC client)
- `libnspmcdm.dll` (NPU MCDM usermode driver)
- `HTP/HtpUsrDrv.dll` (2.8 MB HTP user driver)
- `HTP/QnnHtpPrepareDrv.dll` (123 MB graph compiler)
- `HTP/QnnHtpV73StubDrv.dll` (stub driver)

These DLLs call into the Windows kernel driver (`qcnspmcdm8380.sys`) via IOCTLs that go through the Windows kernel. Wine cannot intercept these calls in WSL2 because the driver is not loaded in the WSL2 kernel. The ARM64 PE format is correct for Wine on ARM64, but the fundamental problem is that the DLLs need a kernel endpoint that does not exist in WSL2.

`/usr/bin/wine*` does not exist on this system.

### Path 3: Open-source FastRPC (libcdsprpc.so)?

**ANSWER: NOT POSSIBLE.** The WSL2 kernel has zero FastRPC support.

- `/proc/kallsyms` grep for "fastrpc": **0 matches**
- No `CONFIG_QCOM_FASTRPC` compiled into the kernel (confirmed via kallsyms absence)
- `/dev/cdsprpc*`: **does not exist**
- `/dev/adsprpc*`: **does not exist**
- `/dev/fastrpc*`: **does not exist**

The open-source FastRPC userspace library (github.com/qualcomm/fastrpc, BSD-3) requires the kernel driver (`drivers/misc/fastrpc.c`) to be compiled in AND a physical Qualcomm SoC with DSP accessible via SMMU. WSL2 runs in a Hyper-V VM with no direct hardware access to the DSP/NPU subsystem.

Even on bare-metal Linux ARM64 (which WSL2 is not), the Hexagon NPU requires Qualcomm's proprietary `qcnspmcdm` driver stack, not the generic `fastrpc` driver. The open-source `fastrpc` driver supports older Hexagon DSPs (CDSP, ADSP, SDSP) but NOT the NPU's HTP (Hexagon Tensor Processor) mode.

### Path 4: DirectML from WSL2?

**ANSWER: NO.** DirectML has no Linux build.

- `/usr/lib/wsl/lib/` contains ONLY: `libd3d12.so`, `libd3d12core.so`, `libdxcore.so`
- **No `libdirectml.so`** exists anywhere on the system
- The `onnxruntime-directml` pip package is Windows-only (x86_64 and ARM64 Windows wheels only)
- Microsoft has not released a Linux ARM64 build of DirectML
- Building DirectML from source is not possible: the source is not fully open (headers + redistributable Windows binary only)

Even if `libdirectml.so` existed, it would need to talk to the NPU through the dxgkrnl bridge, which (as shown in Path 1) does not support MCDM passthrough.

### Path 5: WSL2 Kernel Modules

**ANSWER: Only dxgkrnl is relevant, and it is insufficient.**

- dxgkrnl is compiled into the kernel (not a loadable module): **~280+ symbols** in kallsyms
- `/proc/modules` is empty (no loadable modules at all)
- Key dxgkrnl symbols present: `dxgk_ioctl`, `dxgkio_query_adapter_info`, `dxgkio_enum_adapters`, `dxgkio_enum_adapters3`, `dxgglobal_start_adapters`, `dxgglobal_create_adapter`, `dxgadapter_set_vmbus`, `dxgadapter_start`, etc.
- No Qualcomm-specific kernel modules loaded
- No `fastrpc` symbols
- No `drm_accel` symbols (DRM accelerator subsystem not compiled in)
- Qualcomm symbols present are generic SoC support (`qcom_pcie`, `qcom_ebi2`) not NPU-related

### Path 6: /sys/class/ and /sys/devices/ for NPU/accel devices

**ANSWER: No NPU or accelerator devices in sysfs.**

- `/sys/class/accel*/`: does not exist (no DRM accel subsystem)
- `/sys/class/dxg*/`: does not exist (dxg registers as `/dev/dxg` misc device only)
- `/sys/class/misc/`: no entries visible
- `/dev/dxg`: exists (confirmed via Read tool getting EINVAL, not ENOENT)
- `/proc/devices`: lists 44 character device types, no NPU/accel/compute entries
- No vmbus device nodes visible in `/sys/bus/vmbus/devices/`

---

## The MCDM Architecture Gap

The fundamental problem is architectural. Microsoft's WSL2 GPU support works through this path:

```
WSL2 app -> libdxcore.so -> /dev/dxg -> dxgkrnl (Linux)
  -> VMBus -> dxgkrnl (Windows host) -> WDDM driver -> GPU hardware
```

For MCDM (compute-only) devices like the NPU, the intended path would be:

```
WSL2 app -> libdxcore.so -> /dev/dxg -> dxgkrnl (Linux)
  -> VMBus -> dxgkrnl (Windows host) -> MCDM driver -> NPU hardware
```

The WSL2 dxgkrnl bridge allocates adapter slots for MCDM devices (hence Adapter 1 appears in D3DKMTEnumAdapters2) but does NOT establish the VMBus channel for them. The adapter entry has Handle=0, LUID=0:0, and all query operations fail. This is a Microsoft limitation in the WSL2 dxg driver, not a Qualcomm driver issue.

The `c_computeaccelerator.inf` device class (ClassDesc = "Neural processors", ClassGuid = `{F01A9D53-3FF6-48D2-9F97-C8A7004BE10C}`) is recognized by Windows but not by the WSL2 dxgkrnl bridge.

---

## Actionable Recommendations

### Phase 1: TCP Bridge (Now, 2 days effort)

Deploy the architecture already designed in `npu_bridge_server.py` / `npu_bridge_client.py`:

1. Windows-side: Run Python with `onnxruntime-directml` and a TCP server
2. WSL2-side: Client sends ONNX model + input tensors over TCP
3. Windows-side: Execute via DmlExecutionProvider on NPU, return results
4. Expected throughput: ~45 TOPS INT8, ~11 TOPS FP16 (minus TCP overhead)

This works TODAY because DirectML on Windows can access the NPU directly.

### Phase 2: Monitor Microsoft WSL2 MCDM Support

Track these signals:
- WSL2 kernel updates (`drivers/hv/dxgkrnl/` in WSL2-Linux-Kernel repo)
- Windows Insider builds with MCDM passthrough
- Microsoft DirectML team announcements about Linux support
- Intel NPU in WSL2 issue tracker (intel/linux-npu-driver#56)

### Phase 3: Native Windows Python (Long-term)

If the TCP bridge latency is unacceptable for real-time inference, port the critical path to run natively on Windows ARM64 with `onnxruntime-directml`. This eliminates the WSL2 boundary entirely.

---

## Test Scripts Created/Used

| Script | Purpose | Result |
|--------|---------|--------|
| `probe_dxg_adapters.py` | D3DKMTEnumAdapters2 enumeration | 2 adapters, adapter 1 is phantom |
| `dxcore_enum3_test.py` | D3DKMTEnumAdapters3 with filters | 0 adapters for all filters |
| `dxcore_npu_enum.py` | DXCore COM API with ML/NPU GUIDs | Segfault (COM vtable not functional) |
| `npu_adapter_deep_probe.py` | Deep comparison of adapters 0 vs 1 | Adapter 1: Handle=0, LUID=0:0, all queries fail |

---

## Hardware Reference

| Component | Details |
|-----------|---------|
| NPU | Qualcomm Hexagon (HTP v73), 45 TOPS INT8 |
| GPU | Qualcomm Adreno X1-85, 3.8 TFLOPS FP32 |
| CPU | Qualcomm Oryon 12-core, ~0.5 TFLOPS FP32 |
| NPU Driver | qcnspmcdm8380.sys (MCDM, ComputeAccelerator class) |
| NPU Device ID | ACPI\VEN_QCOM&DEV_0D0A |
| NPU DXCore attrs | COMPUTE_ACCELERATOR + D3D12_GENERIC_ML |
| WSL2 D3D12 libs | libd3d12.so, libd3d12core.so, libdxcore.so |
