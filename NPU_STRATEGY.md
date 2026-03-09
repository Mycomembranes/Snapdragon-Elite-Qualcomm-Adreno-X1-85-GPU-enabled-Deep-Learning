# NPU Access Strategy -- Snapdragon X Elite on WSL2

**Date**: 2026-03-08
**Platform**: Snapdragon X Elite (Surface Pro 11th Edition) running WSL2 on Windows 11 ARM64
**Author**: Mukshud Ahamed

---

## Hardware Inventory

| Accelerator | Silicon | Peak Throughput | WSL2 Status |
|-------------|---------|-----------------|-------------|
| **NPU** | Qualcomm Hexagon | 45 TOPS (INT8) | NOT ACCESSIBLE |
| **GPU** | Qualcomm Adreno X1-85 | 3.8 TFLOPS (FP32) | FULLY WORKING |
| **CPU** | Qualcomm Oryon (12-core) | ~0.5 TFLOPS (FP32) | Always available |

The GPU path is production-ready: 38 SPIR-V compute shaders, a Rust `d3d12-gpu-chain` crate
(7 source files, PyO3 bindings), and a Python `cooperative_dispatch.py` that routes operations
across NPU > GPU > CPU with per-op timing and fallback. The NPU path is the gap.

---

## Architecture Diagram

```
 WINDOWS 11 ARM64 (host)
 +------------------------------------------------------------------+
 |                                                                    |
 |   Qualcomm NPU Driver          Qualcomm GPU Driver (D3D12)        |
 |   (Hexagon DSP runtime)        (Adreno X1-85)                     |
 |         |                              |                           |
 |   DirectML Runtime             D3D12 Runtime                      |
 |   (NPU execution provider)    (compute pipeline)                  |
 |         |                              |                           |
 |         v                              v                           |
 |   onnxruntime-directml         /dev/dxg (WSL2 passthrough)        |
 |   [Windows-only today]                 |                           |
 |         :                              |                           |
 +---------|------------------------------|---------------------------+
           :                              |
           : (no Linux ARM64 build)       | (works today)
           :                              |
 WSL2 (Ubuntu ARM64)                      |
 +---------|------------------------------|---------------------------+
 |         v                              v                           |
 |   Option A: Build          d3d12-gpu-chain (Rust/PyO3)             |
 |   onnxruntime-directml     libd3d12_compute.so                    |
 |   for Linux aarch64        38 SPIR-V shaders                      |
 |         |                       |                                  |
 |   Option B: TCP Bridge     d3d12_tensor.py (36 ops)               |
 |   to Windows Python             |                                  |
 |         |                       v                                  |
 |         +-------> cooperative_dispatch.py <--------+               |
 |                   (NPU > GPU > CPU routing)        |               |
 |                          |                         |               |
 |                          v                         |               |
 |                   device_selector.py          npu_ops.py           |
 |                   (priority singleton)        (7 ONNX graphs)      |
 |                          |                                         |
 |                          v                                         |
 |                   OpenFold3 / Rotifer ML Aligner                   |
 +--------------------------------------------------------------------+
```

---

## Option A: Build onnxruntime-directml for Linux aarch64

### Concept

DirectML is built on top of D3D12, which already works in WSL2 via the `/dev/dxg` kernel
driver. If we can compile `onnxruntime` with the `DmlExecutionProvider` targeting Linux ARM64,
the NPU becomes directly accessible from WSL2 Python -- no bridge, no Windows process.

### Key Repositories

- ONNX Runtime: https://github.com/microsoft/onnxruntime
- DirectML: https://github.com/microsoft/DirectML
- DirectML NuGet (prebuilt Windows binaries): https://www.nuget.org/packages/Microsoft.AI.DirectML

### Build Steps (estimated)

1. **Cross-compile DirectML shared library for Linux aarch64**
   - DirectML source is partially open (headers + redistributable binary).
   - The redistributable `DirectML.dll` is Windows-only. We would need Microsoft to release
     a Linux `.so` or build from the open headers against the Linux D3D12 translation layer.
   - Alternative: use `dxcore` + `d3d12` headers from `mesa` / `WSL2 libdxcore.so` to
     create a shim that maps DirectML API calls to the existing `/dev/dxg` D3D12 path.

2. **Build ONNX Runtime with DML provider on Linux aarch64**
   ```bash
   git clone --recursive https://github.com/microsoft/onnxruntime
   cd onnxruntime
   ./build.sh --config Release \
     --build_shared_lib \
     --use_dml \
     --dml_path /path/to/cross-compiled-directml \
     --cmake_extra_defines CMAKE_SYSTEM_PROCESSOR=aarch64
   ```

3. **Install into conda environment**
   ```bash
   pip install build/Linux/Release/dist/onnxruntime-*.whl
   ```

4. **Validate with existing `npu_ops.py`**
   - The 7 ONNX graphs (matmul, linear, layer_norm, softmax, gelu, attention, mlp) are
     already built and tested. If `DmlExecutionProvider` appears in
     `ort.get_available_providers()`, everything lights up automatically.

### Feasibility: MEDIUM

- **Blocker**: DirectML redistributable is currently Windows-only (PE DLL, not ELF .so).
  Microsoft would need to ship a Linux ARM64 DirectML library, or we would need to build
  one from the open DirectML headers + D3D12 mapping layer.
- **Positive signal**: `/dev/dxg` already exposes D3D12 to WSL2, and `libdxcore.so` exists.
  The plumbing is partially there.
- **Risk**: Even if the library compiles, the Hexagon NPU device may not enumerate through
  the WSL2 D3D12 adapter path (it may be exposed as a separate DXCore adapter type that
  Linux `dxcore` does not forward).

### Benefit

Full native NPU access from WSL2. Zero bridge overhead. The existing `cooperative_dispatch.py`
and `device_selector.py` would work unmodified -- they already check for `DmlExecutionProvider`.

### Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Research: DirectML Linux feasibility | 1-2 days | Check if `libdirectml.so` exists or can be built |
| Cross-compile attempt | 3-5 days | May require Microsoft engagement |
| ONNX Runtime integration | 2-3 days | Standard build if DirectML .so is available |
| Testing + validation | 1-2 days | Run all 7 ONNX graphs on NPU |
| **Total** | **1-2 weeks** | Depends heavily on DirectML availability |

---

## Option B: WSL2-to-Windows NPU Bridge

### Concept

Run a Python inference server on native Windows (where `onnxruntime-directml` works today)
and have WSL2 Python send inference requests over TCP. `localhost` is shared between WSL2 and
Windows, so the bridge is just two Python scripts connected by a socket.

### Architecture

```
WSL2 (Linux ARM64)                    Windows 11 ARM64
+---------------------------+         +---------------------------+
|                           |         |                           |
|  rotifer / openfold3      |         |  npu_bridge_server.py     |
|       |                   |         |       |                   |
|  npu_bridge_client.py     |  TCP    |  onnxruntime-directml     |
|       |                   | ------> |       |                   |
|  localhost:9742           |  ~0.1ms |  DmlExecutionProvider     |
|                           |         |       |                   |
+---------------------------+         |  Hexagon NPU (45 TOPS)    |
                                      +---------------------------+
```

### Protocol Design

```
Request  (client -> server):
  4 bytes: message length (uint32 LE)
  N bytes: msgpack payload
    {
      "op": "matmul" | "linear" | "attention" | ... ,
      "arrays": { "A": bytes, "B": bytes, ... },
      "shapes": { "A": [m, k], "B": [k, n], ... },
      "params": { "scale": 0.125, "eps": 1e-5, ... }
    }

Response (server -> client):
  4 bytes: message length (uint32 LE)
  N bytes: msgpack payload
    {
      "status": "ok" | "error",
      "result": bytes,          # numpy array bytes
      "shape": [m, n],
      "elapsed_ms": 0.42,
      "device": "npu"
    }
```

### Implementation Sketch

**Server (Windows side)** -- `npu_bridge_server.py`:
```python
import socket, msgpack, numpy as np, onnxruntime as ort
from npu_ops import _build_matmul_graph, _build_attention_graph, ...

def handle_request(data):
    msg = msgpack.unpackb(data)
    op = msg["op"]
    arrays = {k: np.frombuffer(v, dtype=np.float32).reshape(msg["shapes"][k])
              for k, v in msg["arrays"].items()}
    # Build ONNX graph, run on DML provider
    graph_bytes = GRAPH_BUILDERS[op](**arrays, **msg.get("params", {}))
    sess = ort.InferenceSession(graph_bytes,
                                providers=["DmlExecutionProvider"])
    result = sess.run(None, arrays)[0]
    return msgpack.packb({"status": "ok",
                          "result": result.tobytes(),
                          "shape": list(result.shape)})

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 9742))
server.listen(1)
# ... accept loop ...
```

**Client (WSL2 side)** -- `npu_bridge_client.py`:
```python
import socket, msgpack, numpy as np

class NPUBridgeClient:
    def __init__(self, host="localhost", port=9742):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def matmul(self, a, b):
        return self._call("matmul",
                          {"A": a.tobytes(), "B": b.tobytes()},
                          {"A": list(a.shape), "B": list(b.shape)})

    def _call(self, op, arrays, shapes, params=None):
        msg = msgpack.packb({"op": op, "arrays": arrays,
                             "shapes": shapes, "params": params or {}})
        self.sock.sendall(len(msg).to_bytes(4, "little") + msg)
        resp_len = int.from_bytes(self.sock.recv(4), "little")
        resp = msgpack.unpackb(self._recv_exact(resp_len))
        return np.frombuffer(resp["result"],
                             dtype=np.float32).reshape(resp["shape"])
```

### Latency Budget

| Component | Latency | Notes |
|-----------|---------|-------|
| TCP round-trip (localhost) | 0.05-0.1 ms | Kernel loopback, no NIC |
| msgpack serialize + deserialize | 0.01-0.05 ms | Small overhead |
| Array copy (WSL2 -> Windows) | 0.1-1.0 ms | Depends on tensor size |
| NPU inference (DirectML) | 0.5-5.0 ms | Depends on op and dimensions |
| Array copy (Windows -> WSL2) | 0.1-1.0 ms | Return path |
| **Total per call** | **~1-7 ms** | Dominated by NPU compute |

For comparison, a 384x384 matmul on CPU takes ~0.3 ms. The bridge is worth it only for
larger operations or batched workloads where NPU's 45 TOPS throughput amortizes transfer cost.

**Break-even point**: Operations where NPU compute time savings exceed ~2 ms of bridge overhead.
This typically means matrix dimensions >= 512 or batched attention with sequence length >= 128.

### Feasibility: EASY

- Both Windows and WSL2 have Python and numpy.
- `onnxruntime-directml` installs trivially on Windows ARM64 via pip.
- `localhost` TCP between WSL2 and Windows works out of the box.
- No custom builds, no driver hacking.

### Benefit

Works today with stock tools. Can be prototyped in an afternoon. Provides real NPU access
for operations large enough to justify the bridge overhead.

### Limitations

- Per-call TCP overhead makes it unsuitable for small/frequent operations.
- Requires a Windows Python process to be running.
- Debugging spans two OS environments.

### Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Server + client implementation | 1 day | Straightforward socket code |
| Integration with `device_selector.py` | 0.5 days | Add bridge as NPU backend |
| Testing + latency profiling | 0.5 days | Identify break-even dimensions |
| **Total** | **2 days** | |

---

## Option C: Port Rotifer to Native Windows

### Concept

Run the entire rotifer ecosystem directly on Windows ARM64 -- no WSL2, no bridge. This gives
direct access to all accelerators (NPU, GPU, CPU) through their native Windows drivers with
zero abstraction overhead.

### Required Changes

| Component | Current (WSL2/Linux) | Windows Port | Effort |
|-----------|---------------------|--------------|--------|
| Shell scripts (~230) | bash/zsh | PowerShell or batch | HIGH |
| File paths | `/home/mukshud/...` | `C:\Users\mukshud\...` | MEDIUM |
| Conda environment | Linux ARM64 | Windows ARM64 | LOW |
| Rust crates (d3d12-gpu-chain, rotifer-bio-native) | Linux ELF | Windows PE | MEDIUM |
| D3D12 compute shaders | Via /dev/dxg | Native D3D12 API | LOW (simpler) |
| ONNX Runtime | CPU-only (Linux) | onnxruntime-directml (Windows) | LOW |
| Perl scripts | Linux perl | Strawberry Perl or WSL fallback | MEDIUM |
| Process management | tmux, bash jobs | Windows Terminal, PowerShell jobs | MEDIUM |

### Key Challenges

1. **Shell script ecosystem**: 230+ bash scripts with Linux-specific paths, pipes, process
   substitution, and GNU coreutils assumptions. Converting to PowerShell is a major effort.

2. **Perl dependencies**: `setup_perl_env.sh` and several bioinformatics tools assume Linux
   perl with CPAN modules. Windows perl (Strawberry) has compatibility gaps.

3. **Build toolchain**: Rust cross-compilation to `aarch64-pc-windows-msvc` requires the
   Windows ARM64 SDK and MSVC linker. Maturin supports this but it needs setup.

4. **Testing surface**: 286+ Rust tests, all Python integration tests, and shell-based
   pipeline tests need to pass on Windows.

### Feasibility: HIGH EFFORT

This is a full platform port. While individual components (Python, Rust, D3D12) work fine on
Windows ARM64, the glue (shell scripts, path conventions, process management) is deeply
Linux-specific.

### Benefit

Best possible performance. Direct NPU + GPU access without any bridge or translation layer.
Eliminates WSL2 overhead entirely (~2-5% for compute, more for I/O).

### Timeline Estimate

| Phase | Duration | Notes |
|-------|----------|-------|
| Environment setup (conda, rust, msvc) | 2-3 days | Windows ARM64 toolchain |
| Rust crate ports | 3-5 days | d3d12-gpu-chain + rotifer-bio-native |
| Python package port | 2-3 days | Path fixes, import adjustments |
| Shell script triage | 1-2 days | Identify critical-path scripts |
| Critical script conversion | 5-10 days | PowerShell equivalents for top 20 scripts |
| Testing + validation | 3-5 days | All pipelines on Windows |
| **Total** | **3-5 weeks** | Conservative estimate |

---

## Current State Summary

### What Works (GPU path -- production ready)

```
d3d12-gpu-chain/                        # Rust crate, 7 source files
  src/lib.rs                            # Module root
  src/gpu_tensor.rs                     # GpuTensor: from_numpy, to_numpy, release
  src/fused_ops.rs                      # fused_linear, fused_attention, fused_mlp, ...
  src/command_batch.rs                  # D3D12 command list batching
  src/pipeline_cache.rs                 # Pipeline state object cache
  src/ffi.rs                            # C FFI to libd3d12_compute.so
  src/python.rs                         # PyO3 bindings

openfold-3-mlx/.../d3d12/
  d3d12_tensor.py                       # 36 Python tensor operations
  shaders_spv/                          # 38 SPIR-V compute shaders
  d3d12_gpu_only.py                     # Monkey-patch nn.Linear/LayerNorm

npu_utilization/
  device_selector.py                    # NPU > GPU > CPU priority routing
  cooperative_dispatch.py               # Per-op device routing with stats
  npu_ops.py                            # 7 ONNX graph builders (ready for DML)
  npu_detect.py                         # DXCore adapter enumeration
  setup_npu.sh                          # onnxruntime-directml installer
```

### Benchmark Context (GPU vs CPU)

From `bench_gpu_chain.py` at realistic OpenFold3 tensor sizes:

| Operation | Tensor Shape | CPU (ms) | GPU (ms) | Speedup |
|-----------|-------------|----------|----------|---------|
| fused_linear | (1,384) x (384,384) | ~0.3 | ~0.15 | ~2x |
| fused_linear | (128,256) x (256,256) | ~2.5 | ~0.8 | ~3x |
| fused_attention | q(64,64) k(64,64) v(64,64) | ~0.5 | ~0.3 | ~1.7x |
| fused_mlp | (1,384) w1(384,1536) w2(1536,384) | ~1.2 | ~0.4 | ~3x |
| fused_swiglu | (1,1536) gate + value | ~0.02 | ~0.05 | 0.4x* |

*SwiGLU is element-wise and too small for GPU transfer overhead to pay off.

**NPU expected performance** (45 TOPS INT8, ~11 TOPS FP16 equivalent):

| Operation | NPU Expected (ms) | vs GPU | vs CPU |
|-----------|--------------------|--------|--------|
| Attention (seq=256, dim=64) | ~0.1 | ~3x faster | ~5x faster |
| Softmax (256x256) | ~0.05 | ~2x faster | ~4x faster |
| LayerNorm (batch=128, dim=384) | ~0.08 | ~2x faster | ~6x faster |
| Large MatMul (512x512) | ~0.3 | comparable | ~4x faster |

NPU excels at regular memory-access patterns (attention, softmax, normalization).
GPU excels at large irregular compute (big GEMMs with non-power-of-2 dimensions).

---

## Recommendation

### Phase 1: Option B -- TCP Bridge (immediate, ~2 days)

Start here. This provides real NPU access with minimal engineering risk.

**Action items**:
1. Install `onnxruntime-directml` on Windows Python (`pip install onnxruntime-directml`).
2. Implement `npu_bridge_server.py` (Windows) and `npu_bridge_client.py` (WSL2).
3. Wire the bridge client into `device_selector.py` as the NPU backend.
4. Profile break-even dimensions; configure `cooperative_dispatch.py` thresholds.
5. For operations below break-even, continue using GPU (`d3d12-gpu-chain`) or CPU.

**Routing policy after Phase 1**:
```
if tensor_max_dim < 64:         -> CPU (transfer overhead dominates)
elif op in {attention, softmax, layer_norm}
     and tensor_max_dim >= 128: -> NPU via bridge (45 TOPS sweet spot)
elif tensor_max_dim >= 256:     -> GPU via d3d12-gpu-chain (large GEMM)
else:                           -> GPU (medium compute)
```

### Phase 2: Option A -- Native DirectML (1-2 weeks, when Microsoft ships Linux ARM64 DirectML)

Monitor the DirectML and ONNX Runtime repos for Linux ARM64 support. If Microsoft releases
`libdirectml.so` for Linux (or if we can build one against `/dev/dxg`):

1. Build `onnxruntime` with `--use_dml` for Linux aarch64.
2. Replace the TCP bridge with direct in-process NPU access.
3. Remove bridge latency; all 7 ONNX graphs run natively on NPU.

This eliminates the TCP overhead and the need for a Windows-side Python process.

### Phase 3: Option C -- Windows Native (only if needed, 3-5 weeks)

Pursue only if:
- WSL2 overhead becomes a measured bottleneck for production workloads.
- A full Windows ARM64 deployment is required for other reasons.
- Microsoft announces end-of-life for WSL2 D3D12 passthrough (unlikely).

For our current use case (OpenFold3 protein folding + rotifer ML alignment), the WSL2
environment with GPU acceleration and NPU bridge is sufficient.

---

## Decision Matrix

| Criterion | Option A (Build DML) | Option B (Bridge) | Option C (Windows Port) |
|-----------|---------------------|-------------------|------------------------|
| Time to first NPU inference | 1-2 weeks | 2 days | 3-5 weeks |
| Steady-state latency | Best (in-process) | +1-2 ms per call | Best (in-process) |
| Engineering risk | Medium (depends on MS) | Low | Low (but high effort) |
| Maintenance burden | Low (pip install) | Medium (two processes) | High (two platforms) |
| Works today | No | **Yes** | No |
| Existing code changes | None | Add bridge client | Major refactor |

---

## File Inventory (existing NPU infrastructure)

```
/home/mukshud/windows_built/gpu_acceleration/
  npu_utilization/
    __init__.py                 # Package init
    npu_detect.py               # Accelerator enumeration (NPU, GPU, CPU)
    npu_ops.py                  # 7 ONNX graph builders + session cache (LRU-64)
    device_selector.py          # NPU > GPU > CPU singleton router
    cooperative_dispatch.py     # Per-op routing with timing/stats/fallback
    setup_npu.sh                # onnxruntime-directml installer script
    NPU_STRATEGY.md             # This document
  d3d12-gpu-chain/
    Cargo.toml                  # Rust crate (PyO3 0.21, numpy 0.21, ndarray 0.15)
    src/                        # 7 Rust source files
    bench_gpu_chain.py          # GPU vs CPU benchmark suite
```
