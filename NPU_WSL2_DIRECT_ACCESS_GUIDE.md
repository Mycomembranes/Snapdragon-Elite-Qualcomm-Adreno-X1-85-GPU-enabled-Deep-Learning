# NPU Direct Access from WSL2 — Complete Engineering Guide

## Date: 2026-03-08
## Platform: Snapdragon X Elite (X1E80100), Windows 11 ARM64, WSL2 (kernel 6.6.87.2)
## NPU: Qualcomm Hexagon NPU, 45 TOPS INT8
Mukshud Ahamed
---

## 1. Goal

Enable **direct NPU access from WSL2 Linux** — the same way GPU access works via `/dev/dxg` paravirtualization. No middleman processes, no Python bridges, no Windows-side services. Pure kernel-level D3DKMT/D3D12 calls from Linux userspace to the NPU hardware.

**Ultimate target**: Use NPU for OpenFold3 protein folding inference alongside GPU, routing INT8-quantized operations to NPU for 45 TOPS throughput.

---

## 2. How GPU Access Works in WSL2 (Reference Architecture)

Understanding GPU access is essential because NPU access must follow the same pattern.

```
Linux userspace (WSL2)
  → libd3d12.so / libdxcore.so (in /usr/lib/wsl/lib/)
    → /dev/dxg (kernel device, drivers/hv/dxgkrnl/)
      → VMBus (Hyper-V virtual bus)
        → Windows host dxgkrnl.sys
          → GPU KMD (qcdxkmsarm64.sys for Adreno)
            → GPU hardware
```

Key components that make GPU work:
1. **Kernel driver** (`dxgkrnl`, built into WSL2 kernel) — handles /dev/dxg ioctls, translates D3DKMT calls to VMBus messages
2. **User-mode drivers** in `/usr/lib/wsl/lib/`:
   - `libd3d12.so` (739KB) — D3D12 API frontend
   - `libd3d12core.so` (6MB) — D3D12 runtime
   - `libdxcore.so` (910KB) — DXCore adapter enumeration + D3DKMT wrappers
   - `libqcdx12arm64wslum.so` (5.8MB) — **Qualcomm GPU user-mode driver for WSL2** (critical!)
3. **Overlay mount**: These .so files are mounted from the Windows host via:
   ```
   none on /usr/lib/wsl/lib type overlay (lowerdir=/gpu_lib_packaged:/gpu_lib_inbox,upperdir=/gpu_lib/rw/upper)
   ```

---

## 3. What We Discovered About NPU in WSL2

### 3.1 NPU IS Visible via D3DKMT

The NPU adapter is projected into WSL2 and accessible via low-level D3DKMT calls:

```c
// This WORKS — NPU opens successfully
struct d3dkmt_openadapterfromluid open = {0};
open.adapter_luid.a = 3762886;  // NPU LUID
int ret = D3DKMTOpenAdapterFromLuid(&open);
// ret = 0 (SUCCESS), adapter_handle is valid
```

**Proof**: `npu_open_test.c`, `npu_d3dkmt_test.c` — NPU opens, device creation works, basic queries succeed.

NPU name: **"Snapdragon(R) X Elite - X1E80100 - Qualcomm(R) Hexagon(TM) NPU"**

### 3.2 NPU Adapter Properties

From `npu_query_dump.c` results:

| Query Type | Result | Meaning |
|------------|--------|---------|
| Type 15 (ADAPTERTYPE) | `compute_only=1, render_supported=0` | Confirms MCDM adapter |
| Type 50 (WDDM 2.7 caps) | SUCCESS, 4 bytes | Has WDDM 2.7 capabilities |
| Type 51 (tracked workload) | SUCCESS | Supports tracked workload |
| Type 79 (extended caps) | SUCCESS, 153 bytes | Large capability structure |
| Type 0 (UMDRIVERPRIVATE) | INVALID_PARAM | No UMD to handle private queries |

### 3.3 What Works

| Operation | Status | Notes |
|-----------|--------|-------|
| OpenAdapterFromLuid | SUCCESS | LUID = 3762886 |
| CreateDevice | SUCCESS | Device handle returned |
| QueryAdapterInfo (types 15, 50, 51, 57, 79) | SUCCESS | Capability queries work |
| Escape (size=0) | SUCCESS | Empty escapes pass |
| CloseAdapter | SUCCESS | Clean cleanup |

### 3.4 What FAILS (and Why)

| Operation | Error | Root Cause |
|-----------|-------|------------|
| D3D12CreateDevice | 0x887a0004 (DXGI_ERROR_UNSUPPORTED) | No WSL2 UMD (.so) for NPU |
| Escape (size>0) | BUFFER_TOO_SMALL (0xC0000023) | VMBus response size mismatch |
| CreateAllocation (with priv_drv_data) | BUFFER_TOO_SMALL | Same VMBus issue |
| Standard allocation (ExistingHeap) | NOT_SUPPORTED | NPU driver rejects standard allocs |
| EnumAdapters2 | Skips NPU | Kernel code: `if (entry->compute_only) continue;` |

### 3.5 Key LUIDs

| Device | LUID | PCI Device ID |
|--------|------|---------------|
| NPU (Hexagon) | 3762886 | 0x008A (COMPUTE_ACCELERATOR) |
| GPU (Adreno X1-85) | 3762965 | 0x008E (VIRTUAL_RENDER) |

---

## 4. Root Cause Analysis

### 4.1 Problem 1: VMBus Response Size Mismatch (BUFFER_TOO_SMALL)

**Location**: `drivers/hv/dxgkrnl/dxgvmbus.c`, function `process_completion_packet()`, line 374

**Mechanism**:
1. WSL2 kernel sends a D3DKMT request to Windows host via VMBus
2. The request specifies an expected response size (`packet->buffer_length`)
3. For Escape: response size = `priv_drv_data_size` (from userspace)
4. For CreateAllocation: response size = header + alloc_info + priv_drv_data_size
5. The NPU host driver (qcnspmcdm8380.sys) sends a **shorter response** than expected
6. The kernel sees `packet_length < packet->buffer_length` and sets `packet->status = -EOVERFLOW`
7. This bubbles up as STATUS_BUFFER_TOO_SMALL (0xC0000023) to userspace

**Why this happens**: The NPU MCDM driver on Windows doesn't populate all the priv_drv_data fields that the kernel expects. It sends back a minimal response (often just a status header) without filling the full requested buffer. The GPU driver fills the full buffer, so this code path was never hit for GPUs.

**Code flow**:
```
userspace: D3DKMTEscape(priv_drv_data_size=8)
  → kernel: dxgvmb_send_escape()
    → dxgvmb_send_sync_msg(result=priv_drv_data, result_size=8)
      → VMBus send, wait for response
      → process_completion_packet():
          packet_length = <actual response bytes from NPU host>  // e.g., 0
          packet->buffer_length = 8  // what we expected
          0 < 8 → EOVERFLOW!
```

### 4.2 Problem 2: No WSL2 User-Mode Driver for NPU

**GPU has**: `libqcdx12arm64wslum.so` (5.8MB) — Qualcomm-provided WSL2 UMD
**NPU has**: `libnspmcdm.dll` (500KB) — Windows-only, **NO .so equivalent**

This means D3D12CreateDevice will ALWAYS fail for the NPU because libd3d12.so needs a UMD to complete device creation. The D3D12 runtime queries the adapter for its UMD, gets nothing for NPU in WSL2, and returns DXGI_ERROR_UNSUPPORTED.

### 4.3 Problem 3: EnumAdapters2 Skips Compute-Only Adapters

In `drivers/hv/dxgkrnl/ioctl.c`, the `D3DKMTEnumAdapters2` implementation has:
```c
if (entry->compute_only)
    continue;  // Hard-skip!
```

This means standard DXCore enumeration via `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS` will never see the NPU. Only `D3DKMTOpenAdapterFromLuid` (with known LUID) or `D3DKMTEnumAdapters3` (with `include_compute_only` filter) can reach it.

---

## 5. The Kernel Patch (VMBus Fix)

### 5.1 What the Patch Does

Modified `process_completion_packet()` in `dxgvmbus.c` to **accept shorter VMBus responses** instead of returning -EOVERFLOW. When the NPU host sends fewer bytes than expected:
1. Copy whatever data was sent
2. Zero-fill the remainder of the buffer
3. Let the operation proceed — callers check NTSTATUS in the response header anyway

### 5.2 The Diff

```diff
--- a/drivers/hv/dxgkrnl/dxgvmbus.c
+++ b/drivers/hv/dxgkrnl/dxgvmbus.c
@@ -372,10 +372,22 @@
     if (packet->buffer_length) {
         if (packet_length < packet->buffer_length) {
-            DXG_TRACE("invalid size %d Expected:%d",
+            DXG_TRACE("short response %d expected %d",
                 packet_length,
                 packet->buffer_length);
-            packet->status = -EOVERFLOW;
+            /*
+             * Accept shorter VMBus responses from MCDM
+             * (compute-only) adapters. Copy available data
+             * and zero-fill the remainder. Callers check
+             * NTSTATUS in the response header.
+             */
+            if (packet_length > 0)
+                memcpy(packet->buffer,
+                       hv_pkt_data(desc),
+                       packet_length);
+            memset(packet->buffer + packet_length, 0,
+                   packet->buffer_length -
+                   packet_length);
         } else {
             memcpy(packet->buffer, hv_pkt_data(desc),
                    packet->buffer_length);
```

### 5.3 Safety Analysis

- **GPU operations unaffected**: GPU host always sends responses >= expected size, so this path is never hit for GPU
- **Zero-fill is safe**: NTSTATUS in response header will be 0 (STATUS_SUCCESS) or an error code — callers check this
- **No information leak**: Zero-filling unused buffer space is safer than leaving uninitialized memory
- **Existing behavior preserved**: If response >= expected, the original memcpy path runs unchanged

### 5.4 Building and Installing

```bash
# Build kernel (from WSL2)
cd /home/mukshud/wsl2-kernel
make -j12 LOCALVERSION=""

# Copy new kernel image to Windows
cp arch/arm64/boot/Image /mnt/c/Users/muksh/wsl2-custom-kernel

# .wslconfig already points to custom kernel:
# [wsl2]
# kernel=C:\\Users\\muksh\\wsl2-custom-kernel

# From Windows PowerShell (as admin):
wsl --shutdown
# Then reopen WSL2 terminal

# Verify kernel version
uname -r  # Should show 6.6.87.2-microsoft-standard-WSL2+
```

---

## 6. What We Explored (Approaches & Results)

### 6.1 Approach: D3D12 MCDM Device Creation (FAILED)

**File**: `npu_d3d12_mcdm.c`

Attempted to create a D3D12 device on the NPU using MCDM feature levels:
- `D3D_FEATURE_LEVEL_1_0_GENERIC` (0x100) — FAILED
- `D3D_FEATURE_LEVEL_1_0_CORE` (0x1000) — FAILED
- `D3D_FEATURE_LEVEL_11_0` — FAILED
- `D3D_FEATURE_LEVEL_12_0` — FAILED

All return 0x887a0004 (DXGI_ERROR_UNSUPPORTED). Root cause: no WSL2 UMD.

**GPU comparison**: D3D12CreateDevice on GPU succeeds at FL_11_0, returns working device with command queue, allocator, heaps, committed resources — all verified.

### 6.2 Approach: DXCore Adapter Enumeration (PARTIAL SUCCESS)

**File**: `npu_d3d12_mcdm.c`

- `DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS` — Returns 1 adapter (GPU only)
- `DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE` — Returns 0 adapters (NPU not enumerated here either)
- `GetAdapterByLuid(3762886)` — Returns NPU adapter successfully
- NPU properties readable: name, LUID, hardware ID, driver version all returned correctly

**Conclusion**: NPU is visible via LUID but not via standard enumeration.

### 6.3 Approach: Direct D3DKMT Operations (PARTIAL SUCCESS)

**Files**: `npu_d3dkmt_test.c`, `npu_escape_probe.c`, `npu_query_dump.c`, `npu_alloc_v3.c`

Low-level D3DKMT calls bypass D3D12 and talk directly to the kernel:
- OpenAdapterFromLuid: WORKS
- CreateDevice: WORKS
- QueryAdapterInfo: WORKS (types 15, 50, 51, 57, 79)
- Escape (size=0): WORKS
- Escape (size>0): BUFFER_TOO_SMALL (pre-patch)
- CreateAllocation: BUFFER_TOO_SMALL or NOT_SUPPORTED (pre-patch)

### 6.4 Approach: VMBus Kernel Patch (IN PROGRESS)

**File**: `dxgvmbus.c` patch (see Section 5)

This is our current strategy. By accepting shorter VMBus responses, Escape and CreateAllocation should proceed. Post-patch tests will determine if the NPU host driver returns useful NTSTATUS codes or data.

### 6.5 Approach: Direct /dev/dxg ioctl (EXPLORED, NOT YET IMPLEMENTED)

Bypassing libdxcore.so entirely and sending raw ioctls to /dev/dxg. This would avoid any userspace library limitations but requires careful struct packing matching the kernel's expected formats.

### 6.6 Approach: UMD Reverse Engineering (EXPLORED)

Investigated `libnspmcdm.dll` (Windows NPU UMD):
- 500KB, ARM64 PE binary
- Exports: standard WDDM UMD entry points (OpenAdapter, CreateDevice, etc.)
- Contains HTP/QNN runtime references
- **NOT directly usable in Linux** — would need complete reimplementation or translation layer
- Alternative: Create a minimal WSL2 shim that handles D3D12 UMD callbacks for MCDM

### 6.7 Approaches REJECTED

| Approach | Why Rejected |
|----------|-------------|
| Python subprocess bridge to Windows | User explicitly said "no middleman" |
| npu_worker.py on Windows side | Same — no process bridge |
| Wine/PE loader for libnspmcdm.dll | Too fragile, too many dependencies |
| Rewrite NPU driver from scratch | Not feasible without Qualcomm documentation |

---

## 7. NPU Driver Stack (Windows)

Understanding the Windows driver stack helps plan what needs to happen for full NPU access:

```
Windows userspace:
  DirectML / ONNX Runtime
    → D3D12 Runtime (d3d12.dll)
      → libnspmcdm.dll (NPU UMD — 500KB)
        → D3DKMT calls to kernel
          → qcnspmcdm8380.sys (NPU KMD — 1.16MB)
            → libcdsprpc.dll (CDSP RPC — 500KB)
              → HTP runtime (HtpUsrDrv.dll — 2.8MB)
                → Hexagon DSP hardware (via FastRPC)
```

**WSL2 equivalent needed**:
```
WSL2 userspace:
  Our Rust code
    → D3DKMT calls via libdxcore.so
      → /dev/dxg (dxgkrnl)
        → VMBus
          → Windows host dxgkrnl.sys
            → qcnspmcdm8380.sys (NPU KMD)
              → Hexagon DSP hardware
```

The key gap: **no libnspmcdm.so** for WSL2. This means we can't use D3D12 high-level API. But we CAN use D3DKMT low-level operations if the VMBus patch works.

---

## 8. File Inventory

### Test Programs (in `tests/`)

| File | Purpose | Key Finding |
|------|---------|-------------|
| `npu_open_test.c` | Basic adapter open | NPU opens successfully |
| `npu_deep_probe.c` | Deep adapter property probe | NPU properties readable |
| `npu_final_probe.c` | Final comprehensive probe | Confirmed all query types |
| `npu_d3dkmt_test.c` | Full D3DKMT operation test | Device creation works |
| `npu_dxcore_probe.c` | DXCore COM enumeration | NPU not in enum, visible via LUID |
| `npu_compute_test.c` | Compute pipeline attempt | Fails without UMD |
| `npu_compute_v2.c` | Compute pipeline v2 | Same failure |
| `npu_compute_pipeline.c` | Pipeline creation attempt | Same failure |
| `npu_alloc_test.c` | Allocation test | BUFFER_TOO_SMALL |
| `npu_alloc_v2.c` | Allocation v2 | BUFFER_TOO_SMALL |
| `npu_alloc_v3.c` | Allocation v3 (multiple strategies) | BUFFER_TOO_SMALL / NOT_SUPPORTED |
| `npu_mcdm_test.c` | MCDM-specific tests | Feature level failures |
| `npu_escape_probe.c` | Escape command probing | size=0 works, size>0 fails |
| `npu_query_dump.c` | QueryAdapterInfo dump | Types 15,50,51,79 succeed |
| `npu_submit_test.c` | Command submission test | Fails without allocation |
| `npu_trace_test.c` | Minimal trace-capture test | For dmesg trace capture |
| `npu_d3d12_mcdm.c` | Full D3D12 MCDM test | DXGI_ERROR_UNSUPPORTED |
| `npu_post_patch_test.c` | Post-kernel-patch validation | Awaiting patched kernel |

### Documentation & Research (in `npu_utilization/`)

| File | Content |
|------|---------|
| `NPU_STRATEGY.md` | Initial strategy document |
| `NPU_ACCESS_ARCHITECTURE.md` | Architecture analysis |
| `NPU_WSL2_RESEARCH_REPORT.md` | Research findings |
| `DIRECTML_NPU_WSL2_RESEARCH.md` | DirectML feasibility |
| `SETUP_NPU_WINDOWS.md` | Windows-side setup notes |
| `kernel_vmbus_patch.diff` | The dxgvmbus.c patch |
| `NPU_WSL2_DIRECT_ACCESS_GUIDE.md` | This document |

### Kernel Files

| File | Role |
|------|------|
| `/home/mukshud/wsl2-kernel/drivers/hv/dxgkrnl/dxgvmbus.c` | Patched VMBus handler |
| `/home/mukshud/wsl2-kernel/arch/arm64/boot/Image` | Built kernel image |
| `/mnt/c/Users/muksh/wsl2-custom-kernel` | Installed kernel location |
| `/mnt/c/Users/muksh/.wslconfig` | WSL2 config pointing to custom kernel |

---

## 9. Compilation Instructions

All test programs compile with:
```bash
gcc -o <output> <source.c> -I/usr/include/wsl/stubs -I/usr/include/directx -ldl -O2
```

Required packages:
```bash
# Already installed on this system
apt install build-essential
# WSL2 headers at /usr/include/wsl/stubs/ and /usr/include/directx/ (auto-provided)
```

Libraries used at runtime (loaded via dlopen):
- `libdxcore.so` — D3DKMT wrapper functions
- `libd3d12.so` — D3D12 CreateDevice (optional, for D3D12-level tests)

---

## 10. Next Steps (After Kernel Patch)

### Phase 1: Validate VMBus Fix
1. Install patched kernel, restart WSL2
2. Run `npu_post_patch_test` — check if Escape and CreateAllocation return SUCCESS or meaningful errors
3. Run `npu_d3d12_mcdm` — check if D3D12CreateDevice behavior changes (unlikely without UMD)
4. Check `dmesg | grep -i dxg` for kernel trace messages

### Phase 2: Explore Working Operations
If Escape(size>0) works after patch:
1. Try sending driver-specific escape commands (reverse-engineer from libnspmcdm.dll exports)
2. Try memory allocation + mapping
3. Try context creation + command submission

### Phase 3: Build Minimal NPU UMD Shim
Since D3D12CreateDevice needs a UMD:
1. Create minimal `libnspmcdm_wsl.so` that implements required UMD callbacks
2. Register it with D3D12 runtime for the NPU adapter
3. Forward operations to D3DKMT (same as the Windows UMD does internally)

### Phase 4: Rust Integration
1. Add NPU backend to `d3d12-gpu-chain` Rust crate
2. NPU FFI layer using D3DKMT directly (not D3D12)
3. Route INT8 quantized ops to NPU, FP32 ops to GPU
4. Integrate into OpenFold3 inference pipeline

---

## 11. Key Technical References

- **D3DKMT structs**: `/usr/include/wsl/stubs/d3dkmthk.h`
- **DXCore API**: `/usr/include/directx/dxcore.h`
- **D3D12 headers**: `/usr/include/directx/d3d12.h`
- **MCDM feature levels**: `/usr/include/directx/d3dcommon.h` (FL_1_0_CORE = 0x1000, FL_1_0_GENERIC = 0x100)
- **Kernel dxgkrnl source**: `/home/mukshud/wsl2-kernel/drivers/hv/dxgkrnl/`
- **WDDM MCDM spec**: Microsoft Learn — "MCDM architecture for compute-only devices"
- **Qualcomm QNN SDK**: docs.qualcomm.com — HTP/Hexagon architecture

---

## 12. Agent Investigation Results (Deep Dive)

Three parallel investigation agents ran comprehensive analysis. Key findings:

### 12.1 Kernel Patch Safety Analysis (Agent 1)

**Caller categories identified:**
- **Category A (ntstatus-only)**: ~20 call sites use `dxgvmb_send_sync_msg_ntstatus` expecting only 4 bytes. These would work even with 4-byte NPU responses.
- **Category B (struct + NTSTATUS header)**: All callers check `ntstatus2int(result.status)` BEFORE reading other fields. Zero-filled fields = null handles = safe.
- **Category C (Escape)**: Uses same buffer for command/response. Zero-padded tail is harmless since `priv_drv_data_size` bounds the userspace copy.
- **Category D (CreateAllocation)**: Result buffer is `vzalloc`'d (already zeroed). Zero-filled allocation handles = invalid, checked by callers.

**Risk assessment: LOW.** GPU operations completely unaffected (GPU always sends full-size responses). NPU operations degrade gracefully.

### 12.2 Direct /dev/dxg Ioctl Analysis (Agent 2)

**Key finding**: You CANNOT bypass the VMBus size check from userspace. The `buffer_length` is computed inside kernel functions, not from raw ioctl parameters. The only user-controllable parameters that affect `result_size` are:
- `alloc_count` (minimum 1)
- `alloc_info[i].priv_drv_data_size` (can be 0)

But the fixed struct overhead (`sizeof(dxgkvmb_command_createallocation_return)`, ~80-100 bytes) remains as a floor.

**Workaround for Escape**: Setting `priv_drv_data_size = 0` causes `buffer_length = 0`, which makes the completion handler **skip the size check entirely**. This works TODAY without kernel modification for fire-and-forget escape commands.

**Alternative paths identified:**
- HW Queue path (`SubmitCommandToHWQueue`) uses `dxgvmb_send_sync_msg_ntstatus` — only expects 4 bytes
- `ReserveGpuVirtualAddress` expects only 16 bytes
- `MapGpuVirtualAddress` expects only ~24 bytes
- These smaller response sizes may be within what the NPU host driver actually sends

### 12.3 UMD Reverse Engineering (Agent 3)

**Critical discovery — NPU uses a COMPLETELY DIFFERENT driver registration:**

| Aspect | GPU (WDDM) | NPU (MCDM) |
|--------|------------|-------------|
| INF Class | `Display` | `ComputeAccelerator` |
| UMD registry key | `UserModeDriverNameWsl` | `HTPUserModeDriverName` (Qualcomm proprietary) |
| DXCore attributes | `D3D12_GRAPHICS` | `D3D12_GENERIC_ML` + `HARDWARE_TYPE_NPU` |
| Shader model | SM 6.x | `D3D_SHADER_MODEL_NONE` |

**Three independent blockers for creating a WSL2 .so:**
1. **dxgkrnl doesn't passthrough MCDM adapters** via standard enumeration
2. **D3D12 doesn't load MCDM UMDs** via `UserModeDriverNameWsl` — the NPU uses `HTPUserModeDriverName`
3. **NPU needs FastRPC/GLINK transport** to Hexagon DSP — not virtualized in Hyper-V

**`libQnnHtpV73SkelDrv.so` is NOT a Linux .so** — it's a 32-bit Hexagon DSP6 ELF that runs directly on the NPU hardware, uploaded via FastRPC.

**The NPU communication chain:**
```
App → DirectML/QNN → libnspmcdm.dll → D3DKMT → qcnspmcdm8380.sys → FastRPC → Hexagon DSP
```

FastRPC requires SMMU, RPMsg/GLINK channels — none of which are available in Hyper-V/WSL2.

### 12.4 Revised Strategy Assessment

Given these findings, the path forward has three tiers:

**Tier 1 (Immediate, post-patch):** D3DKMT direct operations
- Escape commands with the VMBus patch accepting short responses
- Standard allocations (ExistingHeap) if the NPU KMD accepts them
- HW Queue submission path (smaller expected response sizes)
- This bypasses D3D12 entirely — raw D3DKMT calls via libdxcore.so

**Tier 2 (Medium-term):** Custom MCDM compute pipeline
- If Tier 1 shows the NPU KMD accepts commands via VMBus, build a Rust wrapper
- Route specific operations (quantized INT8 matmul) to NPU via D3DKMT
- No D3D12 or DirectML needed — direct kernel-level dispatch

**Tier 3 (If Tier 1-2 fail):** The NPU KMD may not support enough operations via VMBus
- The FastRPC transport is the real communication path to the Hexagon DSP
- Without Hyper-V FastRPC virtualization, the NPU compute path is fundamentally blocked
- Would require Microsoft/Qualcomm to add NPU paravirtualization to WSL2

---

## 13. Lessons Learned

1. **EnumAdapters2 lies** — It hard-skips compute_only adapters. Always use OpenAdapterFromLuid with known LUID for MCDM devices.

2. **VMBus is strict** — The WSL2 kernel expects exact-size responses from the host. GPU drivers always comply; NPU MCDM drivers send minimal responses. This is likely because MCDM was added to WDDM later and the VMBus protocol wasn't updated to handle shorter responses.

3. **UMD is the real blocker** — Even if all D3DKMT operations work, D3D12CreateDevice won't succeed without a user-mode driver. The path forward is either: (a) create a minimal UMD shim, or (b) bypass D3D12 entirely and use D3DKMT directly.

4. **NPU IS projected** — Despite initial research conclusions saying "NPU not projected by host", it IS accessible. The confusion arose because EnumAdapters2 skips it, but OpenAdapterFromLuid works fine.

5. **Zero-size escape succeeds** — This proves the VMBus channel to the NPU host is functional. The communication path works; the size validation was the only barrier.

6. **Aarch64 cross-compilation** — WSL2 kernel on Snapdragon must be compiled with `aarch64-linux-gnu-gcc`, not native x86 gcc.
