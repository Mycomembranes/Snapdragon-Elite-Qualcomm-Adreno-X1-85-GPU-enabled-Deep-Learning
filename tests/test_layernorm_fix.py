#!/usr/bin/env python3
"""Test layer_norm with correct bindings."""
import ctypes, struct, sys, os, numpy as np

os.chdir("/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")
lib = ctypes.CDLL("./libd3d12_compute.so")

lib.d3d12c_init.restype = ctypes.c_int
lib.d3d12c_get_last_error.restype = ctypes.c_char_p
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
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
]
lib.d3d12c_shutdown.restype = None

rc = lib.d3d12c_init()
assert rc == 0

# Load pipeline
with open("shaders_spv/layer_norm.spv", 'rb') as f:
    spv = f.read()
spv_buf = (ctypes.c_ubyte * len(spv))(*spv)
pipeline = lib.d3d12c_create_compute_pipeline(spv_buf, len(spv))
assert pipeline != 0, lib.d3d12c_get_last_error()

# PSV0: 3 SRVs t0-2 (x, gamma, beta), 1 UAV u3 (out), 1 CBV b4 (params)
# Shader: params = vec4<f32>; params.x = width(as f32), params.y = eps
width = 64
num_rows = 2
N = num_rows * width
eps = 1e-5

x = np.random.randn(N).astype(np.float32)
gamma = np.ones(width, dtype=np.float32)
beta = np.zeros(width, dtype=np.float32)
out = np.zeros(N, dtype=np.float32)

# CBV: vec4<f32> = [width_f32, eps_f32, 0, 0]
params = struct.pack('4f', float(width), eps, 0.0, 0.0)
# Pad to 256 bytes for CBV
params_padded = params + b'\x00' * (256 - len(params))

def make_buf(data_bytes):
    h = lib.d3d12c_create_buffer(len(data_bytes))
    src = (ctypes.c_ubyte * len(data_bytes))(*data_bytes)
    lib.d3d12c_upload(h, src, len(data_bytes))
    return h

h_x = make_buf(x.tobytes())
h_gamma = make_buf(gamma.tobytes())
h_beta = make_buf(beta.tobytes())
h_out = make_buf(out.tobytes())
h_params = make_buf(params_padded)

srvs = (ctypes.c_uint64 * 3)(h_x, h_gamma, h_beta)
uavs = (ctypes.c_uint64 * 1)(h_out)
cbvs = (ctypes.c_uint64 * 1)(h_params)

# One workgroup per row
rc = lib.d3d12c_dispatch_sync(pipeline,
    srvs, 3, uavs, 1, cbvs, 1,
    num_rows, 1, 1)
print(f"Dispatch: {rc}")
if rc != 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")

# Readback
nbytes = N * 4
dst = (ctypes.c_ubyte * nbytes)()
rc = lib.d3d12c_readback(h_out, dst, nbytes)
result = np.frombuffer(bytes(dst), dtype=np.float32)

# Numpy reference
x_2d = x.reshape(num_rows, width)
mean = x_2d.mean(axis=1, keepdims=True)
var = x_2d.var(axis=1, keepdims=True)
expected = ((x_2d - mean) / np.sqrt(var + eps) * gamma + beta).flatten()

print(f"Expected: {expected[:8]}")
print(f"Got:      {result[:8]}")
print(f"Max diff: {np.max(np.abs(result - expected)):.2e}")
print(f"Match:    {np.allclose(result, expected, atol=1e-4)}")

for h in [h_x, h_gamma, h_beta, h_out, h_params]:
    lib.d3d12c_release_buffer(h)
lib.d3d12c_shutdown()
