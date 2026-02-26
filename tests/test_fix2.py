#!/usr/bin/env python3
"""Fix gelu_backward and cross_entropy reference formulas."""
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

def make_buf(data_bytes):
    h = lib.d3d12c_create_buffer(len(data_bytes))
    src = (ctypes.c_ubyte * len(data_bytes))(*data_bytes)
    lib.d3d12c_upload(h, src, len(data_bytes))
    return h

def make_arr_buf(arr):
    return make_buf(arr.astype(np.float32).tobytes())

def readback_arr(handle, n):
    nbytes = n * 4
    dst = (ctypes.c_ubyte * nbytes)()
    lib.d3d12c_readback(handle, dst, nbytes)
    return np.frombuffer(bytes(dst), dtype=np.float32).copy()

rc = lib.d3d12c_init()
assert rc == 0

N = 256

# 1. GELU_BACKWARD: uses sigmoid approximation gelu_deriv = sigmoid(1.702 * x)
print("gelu_backward...", end=" ")
with open("shaders_spv/gelu_backward.spv", 'rb') as f:
    spv = f.read()
buf = (ctypes.c_ubyte * len(spv))(*spv)
pipeline = lib.d3d12c_create_compute_pipeline(buf, len(spv))

x = np.random.randn(N).astype(np.float32)
grad_out = np.random.randn(N).astype(np.float32)
hgo = make_arr_buf(grad_out)
hx = make_arr_buf(x)
hgi = make_arr_buf(np.zeros(N, dtype=np.float32))

srvs = (ctypes.c_uint64 * 2)(hgo, hx)
uavs = (ctypes.c_uint64 * 1)(hgi)
rc = lib.d3d12c_dispatch_sync(pipeline, srvs, 2, uavs, 1, None, 0, N, 1, 1)
assert rc == 0

result = readback_arr(hgi, N)
# Correct reference: sigmoid(1.702 * x)
gelu_deriv = 1.0 / (1.0 + np.exp(-1.702 * x))
expected = grad_out * gelu_deriv
ok = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if ok else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hgo, hx, hgi]: lib.d3d12c_release_buffer(h)

# 2. CROSS_ENTROPY: input is log-probabilities, not raw logits
# Formula: -target * log_p - (1 - target) * log(1 - exp(log_p) + 1e-6)
print("cross_entropy...", end=" ")
with open("shaders_spv/cross_entropy.spv", 'rb') as f:
    spv = f.read()
buf = (ctypes.c_ubyte * len(spv))(*spv)
pipeline2 = lib.d3d12c_create_compute_pipeline(buf, len(spv))

# Generate log-probabilities (negative values, since log(p) < 0 for p < 1)
log_probs = -np.abs(np.random.randn(N).astype(np.float32)) - 0.1  # ensure < 0
targets = np.random.randint(0, 2, N).astype(np.float32)
hl = make_arr_buf(log_probs)
ht = make_arr_buf(targets)
ho = make_arr_buf(np.zeros(N, dtype=np.float32))

srvs = (ctypes.c_uint64 * 2)(hl, ht)
uavs = (ctypes.c_uint64 * 1)(ho)
rc = lib.d3d12c_dispatch_sync(pipeline2, srvs, 2, uavs, 1, None, 0, N, 1, 1)
assert rc == 0

result = readback_arr(ho, N)
expected = -targets * log_probs - (1.0 - targets) * np.log(1.0 - np.exp(log_probs) + 1e-6)
ok2 = np.allclose(result, expected, atol=1e-4)
print(f"{'OK' if ok2 else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
if not ok2:
    print(f"  Expected: {expected[:5]}")
    print(f"  Got:      {result[:5]}")
for h in [hl, ht, ho]: lib.d3d12c_release_buffer(h)

lib.d3d12c_shutdown()
