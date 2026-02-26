#!/usr/bin/env python3
"""Test D3D12 dispatch with proper SRV/UAV binding for add shader."""
import ctypes, struct, sys, os, numpy as np

os.chdir("/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")
lib = ctypes.CDLL("./libd3d12_compute.so")

# Function signatures
lib.d3d12c_init.restype = ctypes.c_int
lib.d3d12c_get_last_error.restype = ctypes.c_char_p
lib.d3d12c_get_adapter_name.restype = ctypes.c_char_p
lib.d3d12c_create_buffer.restype = ctypes.c_uint64
lib.d3d12c_create_buffer.argtypes = [ctypes.c_uint64]
lib.d3d12c_upload.restype = ctypes.c_int
lib.d3d12c_upload.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
lib.d3d12c_readback.restype = ctypes.c_int
lib.d3d12c_readback.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
lib.d3d12c_release_buffer.argtypes = [ctypes.c_uint64]
lib.d3d12c_create_compute_pipeline.restype = ctypes.c_uint64
lib.d3d12c_create_compute_pipeline.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
lib.d3d12c_dispatch_sync.restype = ctypes.c_int
lib.d3d12c_dispatch_sync.argtypes = [
    ctypes.c_uint64,                                        # pipeline
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # srvs
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # uavs
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # cbvs
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32      # groups
]
lib.d3d12c_shutdown.restype = None

rc = lib.d3d12c_init()
print(f"Init: {rc}")
if rc != 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    sys.exit(1)

def load_pipeline(name):
    path = f"shaders_spv/{name}.spv"
    with open(path, 'rb') as f:
        data = f.read()
    buf = (ctypes.c_ubyte * len(data))(*data)
    h = lib.d3d12c_create_compute_pipeline(buf, len(data))
    if h == 0:
        print(f"Failed to create pipeline {name}: {lib.d3d12c_get_last_error()}")
    return h

def upload(handle, arr):
    data = arr.astype(np.float32).tobytes()
    src = (ctypes.c_ubyte * len(data))(*data)
    rc = lib.d3d12c_upload(handle, src, len(data))
    if rc != 0:
        print(f"Upload failed: {lib.d3d12c_get_last_error()}")
    return rc

def readback(handle, n):
    nbytes = n * 4
    dst = (ctypes.c_ubyte * nbytes)()
    rc = lib.d3d12c_readback(handle, dst, nbytes)
    if rc != 0:
        print(f"Readback failed: {lib.d3d12c_get_last_error()}")
        return None
    return np.frombuffer(bytes(dst), dtype=np.float32)

def test_shader(name, inputs, expected, srvs_idx, uavs_idx, cbvs_idx=None,
                groups=(1,1,1), cbv_data=None):
    """Test a shader with proper SRV/UAV/CBV binding.
    inputs: list of numpy arrays to upload
    expected: expected output numpy array
    srvs_idx: indices into inputs list for SRV-bound buffers
    uavs_idx: indices into inputs list for UAV-bound buffers (last one is output)
    """
    print(f"\n--- {name} ---")
    pipeline = load_pipeline(name)
    if pipeline == 0:
        return False

    # Create buffers and upload
    handles = []
    for arr in inputs:
        nbytes = len(arr.astype(np.float32).tobytes())
        h = lib.d3d12c_create_buffer(nbytes)
        handles.append(h)
        upload(h, arr)

    # Create CBV buffers if needed
    cbv_handles_list = []
    if cbvs_idx and cbv_data:
        for data in cbv_data:
            bdata = data.astype(np.float32).tobytes()
            # CBVs need 256-byte alignment
            padded = bdata + b'\x00' * (256 - len(bdata) % 256) if len(bdata) % 256 != 0 else bdata
            h = lib.d3d12c_create_buffer(len(padded))
            cbv_handles_list.append(h)
            src = (ctypes.c_ubyte * len(padded))(*padded)
            lib.d3d12c_upload(h, src, len(padded))

    # Build SRV/UAV/CBV handle arrays
    srv_arr = (ctypes.c_uint64 * len(srvs_idx))(*[handles[i] for i in srvs_idx]) if srvs_idx else None
    uav_arr = (ctypes.c_uint64 * len(uavs_idx))(*[handles[i] for i in uavs_idx]) if uavs_idx else None
    cbv_arr = (ctypes.c_uint64 * len(cbv_handles_list))(*cbv_handles_list) if cbv_handles_list else None

    rc = lib.d3d12c_dispatch_sync(
        pipeline,
        srv_arr, len(srvs_idx) if srvs_idx else 0,
        uav_arr, len(uavs_idx) if uavs_idx else 0,
        cbv_arr, len(cbv_handles_list),
        groups[0], groups[1], groups[2]
    )

    if rc != 0:
        print(f"Dispatch failed: {lib.d3d12c_get_last_error()}")
        for h in handles + cbv_handles_list:
            lib.d3d12c_release_buffer(h)
        return False

    # Readback last UAV (output)
    out_handle = handles[uavs_idx[-1]]
    result = readback(out_handle, len(expected))

    for h in handles + cbv_handles_list:
        lib.d3d12c_release_buffer(h)

    if result is None:
        return False

    close = np.allclose(result, expected, atol=1e-5)
    if close:
        print(f"OK (max_diff={np.max(np.abs(result - expected)):.2e})")
    else:
        print(f"FAIL")
        print(f"  Expected: {expected[:8]}")
        print(f"  Got:      {result[:8]}")
    return close

# ============================================================================
# Test each shader
# ============================================================================
results = {}

# ADD: c[i] = a[i] + b[i]
# PSV0: 2 SRVs t0-1, 1 UAV u2
N = 256
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
c = np.zeros(N, dtype=np.float32)
results['add'] = test_shader('add', [a, b, c], a + b,
    srvs_idx=[0, 1], uavs_idx=[2], groups=(N, 1, 1))

# RELU: out[i] = max(0, in[i])
# PSV0: 1 SRV t0, 1 UAV u1
x = np.random.randn(N).astype(np.float32)
out = np.zeros(N, dtype=np.float32)
results['relu'] = test_shader('relu', [x, out], np.maximum(0, x),
    srvs_idx=[0], uavs_idx=[1], groups=(N, 1, 1))

# GELU: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# PSV0: 1 SRV t0, 1 UAV u1
def gelu_ref(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))
results['gelu'] = test_shader('gelu', [x, out], gelu_ref(x),
    srvs_idx=[0], uavs_idx=[1], groups=(N, 1, 1))

# SIGMOID: out[i] = 1 / (1 + exp(-x[i]))
# PSV0: 1 SRV t0, 1 UAV u1
results['sigmoid'] = test_shader('sigmoid', [x, out], 1.0 / (1.0 + np.exp(-x)),
    srvs_idx=[0], uavs_idx=[1], groups=(N, 1, 1))

# SOFTMAX: need to check PSV0
# PSV0: 3 resources (1 SRVs t0-0, 1 UAVs u1-1, 1 CBVs b2-2)
# CBV contains [N, 0, 0, 0] (row_size as uint padded to 256 bytes)
row_size = 64
sm_in = np.random.randn(row_size).astype(np.float32)
sm_out = np.zeros(row_size, dtype=np.float32)
sm_expected = np.exp(sm_in - np.max(sm_in)) / np.sum(np.exp(sm_in - np.max(sm_in)))
# CBV: row_size as uint32
cbv = np.array([row_size], dtype=np.uint32).view(np.float32)
results['softmax'] = test_shader('softmax', [sm_in, sm_out], sm_expected,
    srvs_idx=[0], uavs_idx=[1], cbvs_idx=[0], cbv_data=[cbv],
    groups=(1, 1, 1))

# LAYER_NORM: need PSV0 info
# PSV0: 4 resources (2 SRVs t0-1, 1 UAV u2, 1 CBV b3)
# input, gamma -> SRVs; output -> UAV; params (N, eps) -> CBV
ln_size = 64
ln_in = np.random.randn(ln_size).astype(np.float32)
gamma = np.ones(ln_size, dtype=np.float32)
ln_out = np.zeros(ln_size, dtype=np.float32)
mean = np.mean(ln_in)
var = np.var(ln_in)
eps = 1e-5
ln_expected = (ln_in - mean) / np.sqrt(var + eps) * gamma
# CBV: [N_as_uint, eps_as_float, 0, 0] padded
import struct as st
cbv_bytes = st.pack('If', ln_size, eps) + b'\x00' * 248  # pad to 256
cbv_arr = np.frombuffer(cbv_bytes[:256], dtype=np.float32)
results['layer_norm'] = test_shader('layer_norm', [ln_in, gamma, ln_out], ln_expected,
    srvs_idx=[0, 1], uavs_idx=[2], cbvs_idx=[0], cbv_data=[cbv_arr],
    groups=(1, 1, 1))

# EMBEDDING: PSV0: 3 resources (2 SRVs t0-1, 1 UAV u2)
# Actually need to check the shader source
# Skip for now

# TRANSPOSE_2D: PSV0: 3 resources (1 SRV t0, 1 UAV u1, 1 CBV b2)
rows, cols = 4, 8
t_in = np.arange(rows * cols, dtype=np.float32)
t_out = np.zeros(rows * cols, dtype=np.float32)
t_expected = t_in.reshape(rows, cols).T.flatten()
# CBV: [rows, cols, 0, 0]
t_cbv = np.array([rows, cols], dtype=np.uint32).view(np.float32)
results['transpose_2d'] = test_shader('transpose_2d', [t_in, t_out], t_expected,
    srvs_idx=[0], uavs_idx=[1], cbvs_idx=[0], cbv_data=[t_cbv],
    groups=((cols+15)//16, (rows+15)//16, 1))

# Print summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
passed = sum(1 for v in results.values() if v)
total = len(results)
for name, ok in results.items():
    print(f"  {name:20s}: {'OK' if ok else 'FAIL'}")
print(f"\n{passed}/{total} passed")

lib.d3d12c_shutdown()
