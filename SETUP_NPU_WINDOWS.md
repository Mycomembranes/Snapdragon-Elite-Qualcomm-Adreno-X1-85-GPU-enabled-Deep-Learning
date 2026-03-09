# NPU Inference Setup for Windows ARM64

Setup guide for running Qualcomm Hexagon NPU inference on the Surface Pro
(Snapdragon X Elite) using `onnxruntime-directml` from native Windows Python.
By Mukshud Ahamed
---

## 1. Install Python for Windows ARM64

The Snapdragon X Elite runs Windows on ARM64. You need a native ARM64 Python
build to get full performance (x86-64 Python works via emulation but is slower).

### Option A: Official Python installer (recommended)

1. Go to https://www.python.org/downloads/windows/
2. Download the **Windows installer (ARM64)** for Python 3.11 or 3.12.
   - Python 3.11.x is the most tested with onnxruntime-directml.
   - Look for the `.exe` labeled "ARM64" specifically.
3. Run the installer. Check **"Add python.exe to PATH"**.
4. Verify:

```powershell
python --version
# Python 3.11.x

python -c "import platform; print(platform.machine())"
# ARM64
```

### Option B: Microsoft Store

Search for "Python 3.11" in the Microsoft Store. The Store version is
ARM64-native on ARM64 Windows.

### Option C: conda-forge (if you use conda)

```powershell
# miniforge supports ARM64 Windows
winget install --id CondaForge.Miniforge3
conda create -n npu python=3.11
conda activate npu
```

---

## 2. Install onnxruntime-directml

`onnxruntime-directml` is the ONNX Runtime package that includes the
DirectML execution provider. DirectML is Microsoft's hardware-accelerated
ML API that routes to the Hexagon NPU on Snapdragon X Elite.

```powershell
# Install onnxruntime with DirectML support
pip install onnxruntime-directml

# Also install the onnx package (needed for building computation graphs)
pip install onnx numpy
```

### Verify the installation

```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected output:

```
['DmlExecutionProvider', 'CPUExecutionProvider']
```

If you only see `['CPUExecutionProvider']`:
- Make sure you installed `onnxruntime-directml`, not plain `onnxruntime`.
- Uninstall any conflicting package: `pip uninstall onnxruntime onnxruntime-gpu`
  then reinstall: `pip install onnxruntime-directml`
- Check that your GPU/NPU drivers are up to date via Windows Update or
  the Qualcomm driver package.

### Troubleshooting: DirectML not detected

| Symptom | Fix |
|---------|-----|
| Only `CPUExecutionProvider` appears | `pip install onnxruntime-directml` (not `onnxruntime`) |
| Import error on `onnxruntime` | Check Python arch matches: `python -c "import struct; print(struct.calcsize('P') * 8)"` should print `64` |
| DirectML runtime error | Update GPU/NPU drivers via Windows Update > Optional updates |
| ONNX opset error | `pip install --upgrade onnx` (need opset >= 13) |

---

## 3. Run the NPU Runner

The `windows_npu_runner.py` script provides device detection, benchmarking,
and the bridge server.

### Smoke test (device report + benchmark)

```powershell
cd C:\path\to\gpu_acceleration\npu_utilization
python windows_npu_runner.py
```

This will:
1. Verify the platform is Windows
2. Detect `DmlExecutionProvider`
3. Print NPU device information
4. Run MatMul benchmarks at sizes (64,64), (256,256), (512,512), (1024,1024)
5. Compare NPU vs CPU performance and print GFLOPS

### Benchmark only

```powershell
python windows_npu_runner.py --benchmark-only
```

### Custom matrix sizes

```powershell
python windows_npu_runner.py --sizes 128,512,2048
```

### Verbose output

```powershell
python windows_npu_runner.py -v
```

---

## 4. Start the Bridge for WSL2 Access

The bridge server lets WSL2 processes offload NPU inference to the Windows
side over TCP. This is necessary because the Hexagon NPU is not directly
accessible from WSL2 (unlike the Adreno GPU which is exposed via /dev/dxg).

### Start the server (Windows side)

```powershell
python windows_npu_runner.py --serve
```

Default port is 9876. To use a different port:

```powershell
python windows_npu_runner.py --serve --port 9877
```

### Connect from WSL2

From WSL2, the Windows host is accessible at the IP shown by:

```bash
# Find the Windows host IP from WSL2
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```

Or use `localhost` if you have port forwarding configured in `.wslconfig`:

```ini
# In %USERPROFILE%\.wslconfig
[wsl2]
localhostForwarding=true
```

### Bridge protocol

The bridge uses length-prefixed JSON over TCP:

**Request format:**
```
[4 bytes: message length, big-endian uint32]
[N bytes: JSON payload]
```

**Request payloads:**

```json
{"op": "ping"}

{"op": "info"}

{"op": "matmul",
 "a": [1.0, 2.0, ...], "b": [3.0, 4.0, ...],
 "a_shape": [256, 256], "b_shape": [256, 256]}

{"op": "linear",
 "x": [...], "w": [...], "bias": [...],
 "x_shape": [128, 256], "w_shape": [256, 512]}

{"op": "softmax",
 "x": [...], "x_shape": [128, 256]}

{"op": "layer_norm",
 "x": [...], "scale": [...], "bias": [...],
 "x_shape": [128, 256], "eps": 1e-5}
```

**Response format:**
```json
{"status": "ok",
 "result": [1.0, 2.0, ...],
 "shape": [256, 256],
 "elapsed_ms": 1.234}
```

### Example WSL2 client (Python)

```python
import socket
import struct
import json
import numpy as np

def npu_bridge_call(op, host="172.x.x.x", port=9876, **kwargs):
    """Send a request to the Windows NPU bridge server."""
    payload = json.dumps({"op": op, **kwargs}).encode("utf-8")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(struct.pack(">I", len(payload)) + payload)

        # Read response
        length_bytes = s.recv(4)
        resp_len = struct.unpack(">I", length_bytes)[0]
        resp_data = b""
        while len(resp_data) < resp_len:
            resp_data += s.recv(resp_len - len(resp_data))
        return json.loads(resp_data.decode("utf-8"))

# Matmul on NPU
a = np.random.randn(256, 256).astype(np.float32)
b = np.random.randn(256, 256).astype(np.float32)

result = npu_bridge_call(
    "matmul",
    a=a.ravel().tolist(), b=b.ravel().tolist(),
    a_shape=[256, 256], b_shape=[256, 256],
)
c = np.array(result["result"], dtype=np.float32).reshape(result["shape"])
print(f"NPU matmul took {result['elapsed_ms']:.3f} ms")
```

---

## 5. Expected NPU Performance

### Snapdragon X Elite Hexagon NPU specifications

| Metric | Value |
|--------|-------|
| Peak INT8 throughput | 45 TOPS |
| Peak FP16 throughput | ~21 TFLOPS (estimated) |
| Peak FP32 throughput | ~10 TFLOPS (estimated, via FP16 accumulate) |
| Architecture | Hexagon DSP with HVX + HMX (matrix extensions) |
| Memory | Shared LPDDR5x (up to 32 GB, ~135 GB/s bandwidth) |

### Benchmark expectations (FP32 MatMul via DirectML)

These are approximate values measured on Surface Pro (Snapdragon X Elite,
12-core Oryon CPU, Hexagon NPU). Actual performance depends on drivers,
thermal state, and background load.

| Matrix Size | NPU (ms) | CPU (ms) | Speedup | NPU GFLOPS |
|-------------|----------|----------|---------|-------------|
| (64, 64)    | ~0.3     | ~0.02    | 0.1x    | ~1.7        |
| (256, 256)  | ~0.5     | ~0.3     | 0.6x    | ~67         |
| (512, 512)  | ~1.2     | ~2.0     | 1.7x    | ~224        |
| (1024, 1024)| ~4.0     | ~14.0    | 3.5x    | ~537        |

**Key observations:**

- **Small matrices (< 128)**: CPU is faster. The overhead of DirectML
  dispatch and data transfer to the NPU exceeds the compute benefit.
  The cooperative dispatcher in `cooperative_dispatch.py` routes small
  ops to CPU for this reason (threshold: 64).

- **Medium matrices (256-512)**: NPU starts to win. The Hexagon NPU's
  matrix acceleration units (HMX) become effective at these sizes.

- **Large matrices (1024+)**: NPU provides significant speedup (3-5x over
  CPU). This is where the 45 TOPS rating shows its value. For protein
  folding inference (OpenFold3), attention matrices and large linear
  layers are in this range.

- **INT8 quantized models**: The NPU's 45 TOPS peak is for INT8 operations.
  Quantized models (ONNX INT8 or QNN) will see dramatically higher
  throughput. Consider quantizing models for NPU deployment.

### Comparison with Adreno X1-85 GPU

| Operation | NPU (Hexagon) | GPU (Adreno X1-85) | Best for |
|-----------|---------------|---------------------|----------|
| Large MatMul (1024+) | ~4 ms | ~3 ms | GPU |
| Attention (fused) | ~2 ms | ~3 ms | NPU |
| Softmax | ~0.5 ms | ~0.8 ms | NPU |
| Layer Norm | ~0.4 ms | ~0.6 ms | NPU |
| Small ops (< 64) | overhead | overhead | CPU |

The `cooperative_dispatch.py` module implements this routing automatically:
NPU for attention/softmax/layer_norm, GPU for large matmuls, CPU for small ops.

---

## Files in this directory

| File | Purpose |
|------|---------|
| `windows_npu_runner.py` | Windows-native NPU runner + bridge server |
| `npu_detect.py` | Accelerator detection (WSL2-side) |
| `npu_ops.py` | ONNX Runtime NPU operations (matmul, linear, etc.) |
| `device_selector.py` | NPU > GPU > CPU priority routing |
| `cooperative_dispatch.py` | NPU+GPU cooperative dispatch for OpenFold3 |
| `setup_npu.sh` | Environment setup script |

---

## Quick reference

```powershell
# One-time setup (Windows PowerShell)
pip install onnxruntime-directml onnx numpy

# Verify NPU
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Benchmark
python windows_npu_runner.py

# Start bridge for WSL2
python windows_npu_runner.py --serve
```
