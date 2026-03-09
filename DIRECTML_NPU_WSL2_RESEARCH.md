# DirectML NPU Access from WSL2 on Snapdragon X Elite -- Research Findings

**Date**: 2026-03-08
**System**: Surface Pro 11 (Snapdragon X Elite X1E-80100), Windows 11 24H2, WSL2 6.6.87.2

## Executive Summary

**Can DirectML route to the NPU from WSL2?** No -- not currently. The NPU (MCDM
compute-only adapter) is NOT exposed through the WSL2 /dev/dxg bridge. Only the
Adreno GPU is visible.

## Findings

### 1. WSL2 GPU Libraries Present

```
/usr/lib/wsl/lib/libd3d12.so       (739 KB)
/usr/lib/wsl/lib/libd3d12core.so   (5.7 MB)
/usr/lib/wsl/lib/libdxcore.so      (910 KB)
```

No `libdirectml.so` exists. DirectML has no Linux native library.

### 2. Windows Side DirectML

```
/mnt/c/Windows/System32/directml.dll          (11 KB, stub/forwarder)
/mnt/c/Windows/System32/directml_arm64.dll    (9.8 MB, real ARM64 implementation)
/mnt/c/Windows/System32/directml_x64.dll      (9.7 MB, x64 emulation layer)
```

DirectML exists on the Windows side but is a Windows-only DLL.

### 3. onnxruntime-directml

```
$ pip install onnxruntime-directml
ERROR: No matching distribution found for onnxruntime-directml
```

The `onnxruntime-directml` package only ships Windows AMD64 wheels. No Linux
ARM64 (aarch64) builds exist. Standard `onnxruntime` (1.23.2) is installed
but only provides `CPUExecutionProvider` and `AzureExecutionProvider`.

### 4. Qualcomm NPU MCDM Driver Present in WSL2 Driver Store

The Qualcomm MCDM (Microsoft Compute Driver Model) driver IS present in the
WSL2 driver store:

```
/usr/lib/wsl/drivers/qcnspmcdm8380.inf_arm64_c41374b662899730/
    qcnspmcdm8380.inf    -- INF file (Class=ComputeAccelerator)
    qcnspmcdm8380.sys    -- Kernel driver
    libnspmcdm.dll       -- User-mode DLL
```

The INF confirms: `NSPMCDM.DeviceDesc = "Snapdragon(R) X Elite - X1E80100 -
Qualcomm(R) Hexagon(TM) NPU"`, driver version 30.0.0145.1000 (2025-07-14).

### 5. D3DKMTEnumAdapters2 -- Only GPU Visible

```
Found 1 adapter(s):
  Adapter 0:
    Handle:  0x40000000
    LUID:    0:3762965
    TypeFlags: 0x000020A1
      RenderSupported:  True
      DisplaySupported: False
      Paravirtualized:  True
      ComputeOnly:      False
      HybridIntegrated: True
```

Only the Adreno GPU appears. The ComputeOnly flag is False. No NPU adapter.

### 6. D3DKMTEnumAdapters3 -- Broken in WSL2

D3DKMTEnumAdapters3 (which has filter flags for IncludeComputeOnly) returns 0
adapters for ALL filter values, including 0xFFFFFFFF. This API appears
non-functional in the WSL2 libdxcore.so.

### 7. DXCore COM API -- Factory Works, GetAdapterCount Crashes

```
DXCoreCreateAdapterFactory: SUCCESS (0x00000000)
CreateAdapterList(CORE_COMPUTE): SUCCESS (0x00000000)
GetAdapterCount: SEGFAULT (signal 11)
```

The factory and adapter list creation succeed, but calling GetAdapterCount on
the resulting list causes a segfault inside libdxcore.so. The crash occurs
when the library makes an ioctl to /dev/dxg that fails:

```
dmesg: misc dxg: dxgk: dxgkio_open_adapter_from_luid: Ioctl failed: -22
```

This means the DXCore COM layer can create objects but cannot actually query
adapter state through the dxg kernel bridge for compute-only adapters.

### 8. Kernel dxg Module Limitations

The WSL2 dxg kernel module (dxgkrnl) repeatedly shows:
```
dxgkio_query_adapter_info: Ioctl failed: -22 (EINVAL)
dxgkio_query_adapter_info: Ioctl failed: -2  (ENOENT)
dxgkio_is_feature_enabled: Ioctl failed: -22 (EINVAL)
```

These errors indicate that many WDDM/MCDM query types are not implemented
in the WSL2 kernel's dxg virtual device.

### 9. DirectML Status

DirectML is officially in **maintenance mode**. Microsoft recommends Windows ML
(WinML) for new development on Windows 11 24H2+. Neither has Linux support.

## Architecture Analysis

```
Current Working Path (GPU only):
  WSL2 Python
    -> libd3d12.so / libdxcore.so
    -> /dev/dxg (kernel)
    -> Hyper-V VMBus
    -> Windows dxgkrnl
    -> WDDM GPU driver (qcdx12xx8380.sys)
    -> Adreno X1-85 GPU

What Would Need to Work for NPU (does NOT work):
  WSL2 Python
    -> libdxcore.so (DXCore COM API)
    -> /dev/dxg (kernel) [FAILS HERE - MCDM not supported]
    -> Hyper-V VMBus
    -> Windows dxgkrnl
    -> MCDM driver (qcnspmcdm8380.sys)
    -> Hexagon NPU
```

The blocker is at the **dxg kernel module level**: it does not virtualize MCDM
(compute-only) adapters. It only virtualizes WDDM (GPU) adapters.

## What Works Today (Bridge Architecture)

The existing `npu_bridge_server.py` / `npu_bridge_client.py` architecture
remains the only viable path:

```
WSL2 Python (npu_bridge_client.py)
    --[TCP localhost:29400]-->
Native Windows Python (npu_bridge_server.py)
    --[onnxruntime-directml]-->
DirectML
    --[D3D12 MCDM]-->
Qualcomm Hexagon NPU (45 TOPS)
```

## Potential Future Paths

1. **Microsoft adds MCDM to dxg**: If the WSL2 dxg kernel module gets updated
   to expose compute-only adapters, the DXCore COM path could work. This would
   require changes to both the kernel module and Hyper-V VMBus protocol.

2. **DirectML for Linux**: Microsoft would need to port DirectML to native Linux.
   Currently zero evidence this is planned.

3. **QNN SDK for Linux ARM64**: Qualcomm's QNN (Qualcomm Neural Network) SDK may
   provide direct NPU access on Linux ARM64, bypassing DirectML entirely. This
   would need Qualcomm to release Linux ARM64 QNN drivers.

4. **Windows ML WSL2 support**: WinML (the DirectML replacement) could potentially
   be exposed through WSL2 in the future, but no announcement exists.

## Recommendations

1. **Keep the TCP bridge** as the production NPU path. It works and provides
   access to the full 45 TOPS NPU.

2. **Optimize the bridge** for batch operations to amortize TCP overhead.

3. **Monitor WSL2 releases** for MCDM/compute-only adapter support in dxg.

4. **Watch QNN SDK** for native Linux ARM64 support.

## Files Created During This Research

- `dxcore_enum_test.py` -- D3DKMTEnumAdapters2 enumeration
- `dxcore_enum3_test.py` -- D3DKMTEnumAdapters3 with filter flags
- `dxcore_factory_test.py` -- DXCore COM API (factory + CreateAdapterList)
- `dxcore_factory_iid.py` -- IID probing for factory creation
- `dxcore_minimal.py` -- Minimal vtable probe (factory creation works)
- `dxcore_probe_list.py` -- List vtable probe (GetAdapterCount crashes)
- `dxcore_full_enum.py` -- Full enumeration attempt (crashes at GetAdapterCount)
- `dxcore_npu_enum.py` -- Comprehensive attribute GUID enumeration
- `dxcore_list_adapters.py` -- Safe adapter list enumeration
- `dxcore_test.c` -- C-based DXCore test (native ABI, not compiled)
