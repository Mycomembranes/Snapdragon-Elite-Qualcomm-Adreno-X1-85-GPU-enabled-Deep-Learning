# Recreation Guide: Native D3D12 Compute Backend from Scratch

Step-by-step instructions to recreate the entire D3D12 native compute backend on a fresh WSL2 Ubuntu installation with a Qualcomm Adreno GPU (Surface Pro ARM64).

---

## Prerequisites

- Windows 11 ARM64 with WSL2 enabled
- A GPU visible to WSL2 (verify: `ls /usr/lib/wsl/lib/libdxcore.so` exists)
- Ubuntu 22.04+ in WSL2
- ~2GB disk space for Mesa build

---

## Step 1: Install System Packages

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    directx-headers-dev \
    spirv-tools \
    meson \
    ninja-build \
    python3-mako \
    python3-pip \
    python3-numpy \
    pkg-config \
    libdrm-dev \
    llvm-dev \
    libclang-dev \
    glslang-tools \
    cmake \
    git
```

**What each does**:
- `directx-headers-dev`: D3D12 and DXCore C headers for compilation
- `spirv-tools`: SPIR-V validator (`spirv-val`) and optimizer
- `meson/ninja`: Build system for Mesa
- `python3-mako`: Template engine required by Mesa build
- `llvm-dev/libclang-dev`: Required by Mesa for DXIL support
- `glslang-tools`: GLSL to SPIR-V compiler (optional, for GLSL shaders)

---

## Step 2: Build Mesa Dozen Driver from Source

Mesa provides `libspirv_to_dxil.so`, the critical library that converts SPIR-V shaders to DXIL (DirectX Intermediate Language). We also get the Dozen Vulkan ICD for baseline comparison.

```bash
cd ~
git clone https://gitlab.freedesktop.org/mesa/mesa.git
cd mesa

# Configure for Dozen (D3D12 Vulkan) + SPIR-V to DXIL
meson setup build \
    --prefix=/home/$(whoami)/mesa-dozen-install \
    -Dvulkan-drivers=microsoft-experimental \
    -Dgallium-drivers= \
    -Dllvm=enabled \
    -Dmicrosoft-clc=enabled \
    -Dspirv-to-dxil=true \
    -Dbuildtype=release

# Build and install
ninja -C build
ninja -C build install
```

**Verify installation**:
```bash
ls ~/mesa-dozen-install/lib/aarch64-linux-gnu/libspirv_to_dxil.so
# Should exist

ls ~/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
# Vulkan ICD manifest
```

**Note**: The exact Mesa version matters. We used the main branch as of Feb 2026. The `spirv_to_dxil` API is relatively stable but the `microsoft-experimental` Vulkan driver may change.

---

## Step 3: Set Environment Variables

Add to `~/.bashrc` or `~/.profile`:

```bash
# D3D12 native backend
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/home/$(whoami)/mesa-dozen-install/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH}

# Dozen Vulkan ICD (for wgpu/Vulkan baseline)
export VK_ICD_FILENAMES=/home/$(whoami)/mesa-dozen-install/share/vulkan/icd.d/dzn_icd.aarch64.json
```

Source it:
```bash
source ~/.bashrc
```

**Verify**:
```bash
# D3D12 libraries accessible
python3 -c "import ctypes; ctypes.CDLL('libdxcore.so'); print('DXCore OK')"
python3 -c "import ctypes; ctypes.CDLL('libd3d12.so'); print('D3D12 OK')"
python3 -c "import ctypes; ctypes.CDLL('libspirv_to_dxil.so'); print('spirv_to_dxil OK')"
```

---

## Step 4: Install Rust and naga-cli

naga-cli compiles WGSL (WebGPU Shading Language) to SPIR-V.

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install naga-cli (specific version for compatibility)
cargo install naga-cli@22.0.0
```

**Verify**:
```bash
naga --version
# Should print naga-cli 22.0.0 or similar
```

---

## Step 5: Extract WGSL Shaders

The shader sources are embedded in `wgpu_tensor.py`. We've already extracted them to `shaders_wgsl/`. If starting from scratch:

```bash
mkdir -p shaders_wgsl shaders_spv
```

Each shader is a standalone WGSL file with:
- `@group(0) @binding(N)` annotations for resource bindings
- `@compute @workgroup_size(256)` entry point
- `var<storage, read>` for input buffers (become SRVs)
- `var<storage, read_write>` for output buffers (become UAVs)
- `var<uniform>` for constant parameters (become CBVs)

The 26 shader files are in `shaders_wgsl/` in this archive. Copy them to your working directory.

---

## Step 6: Compile WGSL to SPIR-V

```bash
chmod +x compile_shaders.sh
bash compile_shaders.sh
```

Or manually:
```bash
for wgsl in shaders_wgsl/*.wgsl; do
    name=$(basename "$wgsl" .wgsl)
    naga "$wgsl" "shaders_spv/${name}.spv"
    spirv-val "shaders_spv/${name}.spv"
    echo "OK: ${name}"
done
```

**Expected output**: 26 `.spv` files in `shaders_spv/`. All should pass `spirv-val` validation.

**Common issues**:
- naga version mismatch: Use v22.0.0. Newer versions may change WGSL syntax requirements
- Reserved keywords: Some variable names may conflict with WGSL reserved words

---

## Step 7: Build libd3d12_compute.so

```bash
chmod +x build_d3d12.sh
bash build_d3d12.sh
```

Or manually:
```bash
gcc -shared -fPIC -o libd3d12_compute.so d3d12_compute.c \
    -I/usr/include/directx \
    -ldl -lpthread \
    -O2
```

**Key compilation notes**:
- We do NOT link against `libd3d12.so` or `libdxcore.so` at compile time
- Instead, `d3d12_compute.c` uses `dlopen()` to load them at runtime
- This makes the binary portable across WSL2 systems (libraries may be in different paths)
- The `-I/usr/include/directx` flag finds D3D12 and DXCore headers from `directx-headers-dev`

**Verify**:
```bash
ls -la libd3d12_compute.so
# Should be ~76KB

# Quick test
python3 -c "
import ctypes
lib = ctypes.CDLL('./libd3d12_compute.so')
print('Library loaded OK')
rc = lib.d3d12c_init()
print(f'Init: {\"OK\" if rc == 0 else \"FAILED\"} (rc={rc})')
name = ctypes.c_char_p(lib.d3d12c_get_adapter_name()).value
print(f'Adapter: {name.decode()}')
lib.d3d12c_shutdown()
"
```

Expected: `Adapter: Qualcomm(R) Adreno(TM) X1-85 GPU`

---

## Step 8: Run Correctness Tests

```bash
# Copy d3d12_tensor.py and shaders_spv/ to same directory
# (d3d12_tensor.py expects shaders_spv/ relative to its location)

# Basic buffer roundtrip
python3 tests/test_buffer_dispatch.py

# SRV/UAV descriptor correctness
python3 tests/test_dispatch_srv.py

# Full shader correctness suite (17 shaders)
python3 tests/test_all_shaders.py

# High-level API test (20 tests)
python3 tests/test_d3d12_api.py
```

**Expected results**:
- `test_buffer_dispatch.py`: All buffer operations pass
- `test_dispatch_srv.py`: SRV and UAV descriptors created correctly
- `test_all_shaders.py`: 17/17 PASS, all max_diff < 4e-07
- `test_d3d12_api.py`: 20/20 PASS

---

## Step 9: Enable D3D12 Backend in wgpu_tensor.py

The modified `wgpu_tensor.py` includes auto-detection. It checks for D3D12 availability at import time:

```python
# In wgpu_tensor.py, the _get_backend() function:
# 1. Tries to import d3d12_tensor
# 2. Calls d3d12_init()
# 3. If successful, routes all tensor ops through D3D12
# 4. Falls back to wgpu/Vulkan/Dozen if D3D12 fails
```

To use:
```python
import sys
sys.path.insert(0, '/path/to/gpu_working/src')

from wgpu_tensor import WgpuTensor

# Auto-detects D3D12 if available
x = WgpuTensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
y = WgpuTensor.from_numpy(np.random.randn(64, 64).astype(np.float32))
z = x @ y  # Uses D3D12 matmul if available
```

---

## Step 10: Verify End-to-End MLP Forward Pass

```bash
python3 -c "
import numpy as np
import sys
sys.path.insert(0, 'src')
from d3d12_tensor import *
import time

# Initialize
d3d12_init()
print(f'GPU: {get_device_info()}')

# Create test data (300 samples, 58 features)
X = D3D12Tensor.from_numpy(np.random.randn(300, 58).astype(np.float32))

# Layer 1: 58 -> 256
W1 = D3D12Tensor.from_numpy(np.random.randn(58, 256).astype(np.float32) * 0.1)
b1 = D3D12Tensor.from_numpy(np.zeros(256, dtype=np.float32))

# Layer 2: 256 -> 128
W2 = D3D12Tensor.from_numpy(np.random.randn(256, 128).astype(np.float32) * 0.1)
b2 = D3D12Tensor.from_numpy(np.zeros(128, dtype=np.float32))

# Layer 3: 128 -> 1
W3 = D3D12Tensor.from_numpy(np.random.randn(128, 1).astype(np.float32) * 0.1)
b3 = D3D12Tensor.from_numpy(np.zeros(1, dtype=np.float32))

# Warmup
h1 = d3d12_relu(d3d12_matmul_add(X, W1, b1))
h2 = d3d12_relu(d3d12_matmul_add(h1, W2, b2))
out = d3d12_matmul_add(h2, W3, b3)
_ = out.numpy()

# Benchmark
times = []
for _ in range(10):
    t0 = time.perf_counter()
    h1 = d3d12_relu(d3d12_matmul_add(X, W1, b1))
    h2 = d3d12_relu(d3d12_matmul_add(h1, W2, b2))
    out = d3d12_matmul_add(h2, W3, b3)
    _ = out.numpy()  # force sync
    times.append(time.perf_counter() - t0)

print(f'MLP forward: {np.mean(times)*1000:.2f} ms (mean of 10)')
print(f'Output shape: {out.numpy().shape}')
print('SUCCESS')
"
```

**Expected**: ~10ms per forward pass, output shape (300, 1).

---

## Troubleshooting

### "libdxcore.so: cannot open shared object file"
- Check LD_LIBRARY_PATH includes `/usr/lib/wsl/lib`
- Verify running inside WSL2 (not WSL1)

### "D3D12CreateDevice failed"
- GPU might not be visible: `ls /dev/dxg` should exist
- Try: `wsl --update` from Windows PowerShell

### "D3D12EnableExperimentalFeatures failed"
- This is required for unsigned DXIL
- Must be called BEFORE D3D12CreateDevice
- If it fails, the Windows D3D12 runtime may not support experimental features

### "CreateComputePipelineState returns E_INVALIDARG"
- Root signature likely wrong. Enable PSV0 debug output:
  ```bash
  D3D12C_DUMP_NIR=1 python3 test_buffer_dispatch.py
  ```
- Check that SRV/UAV/CBV counts match PSV0 resource table

### "Dispatch outputs all zeros"
- Check SRV vs UAV descriptor creation
- SRVs need `CreateShaderResourceView` with `DXGI_FORMAT_R32_TYPELESS` + `D3D12_BUFFER_SRV_FLAG_RAW`
- UAVs need `CreateUnorderedAccessView` with `DXGI_FORMAT_R32_TYPELESS` + `D3D12_BUFFER_UAV_FLAG_RAW`
- Never use UAV descriptors for SRV bindings (the GPU reads garbage)

### "spirv_to_dxil fails / returns NULL"
- Check `libspirv_to_dxil.so` is loadable
- Validate SPIR-V input: `spirv-val shader.spv`
- Some SPIR-V features may not be supported by the Mesa converter

### Shader correctness failures
- `gelu_backward`: Numpy reference must use `sigmoid(1.702*x)`, not exact GELU derivative
- `cross_entropy`: Input must be log-probabilities, not raw logits
- `layer_norm`: Requires exactly 3 SRVs (x, gamma, beta), 1 UAV, 1 CBV with vec4 format

---

## Build Dependency Chain

```
                     directx-headers-dev
                            |
                     d3d12_compute.c  -----> libd3d12_compute.so
                            |                      |
                      (at runtime)           d3d12_tensor.py
                            |                      |
              libspirv_to_dxil.so (Mesa)    wgpu_tensor.py
                            |                      |
                     shaders_spv/*.spv       (Python app)
                            |
                      naga-cli v22
                            |
                     shaders_wgsl/*.wgsl
```

Everything flows from WGSL sources through naga to SPIR-V, then at runtime through Mesa's spirv_to_dxil to DXIL, then into D3D12 pipeline state objects. The C wrapper handles all the D3D12 COM plumbing. Python sees a simple `d3d12_add(a, b)` API.
