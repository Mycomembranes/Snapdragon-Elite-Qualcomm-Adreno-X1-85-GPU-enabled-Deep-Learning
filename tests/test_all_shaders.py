#!/usr/bin/env python3
"""Comprehensive correctness test for all D3D12 compute shaders vs numpy."""
import ctypes, struct, sys, os, time
import numpy as np

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
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
]
lib.d3d12c_shutdown.restype = None

# ============================================================================
# Helpers
# ============================================================================

def make_buf(data_bytes):
    """Create GPU buffer and upload data."""
    h = lib.d3d12c_create_buffer(len(data_bytes))
    assert h != 0, f"create_buffer failed: {lib.d3d12c_get_last_error()}"
    src = (ctypes.c_ubyte * len(data_bytes))(*data_bytes)
    rc = lib.d3d12c_upload(h, src, len(data_bytes))
    assert rc == 0, f"upload failed: {lib.d3d12c_get_last_error()}"
    return h

def make_arr_buf(arr):
    """Create buffer from numpy array."""
    return make_buf(arr.astype(np.float32).tobytes())

def make_cbv(data_bytes):
    """Create CBV buffer (256-byte aligned)."""
    padded = data_bytes + b'\x00' * (256 - len(data_bytes) % 256) if len(data_bytes) % 256 != 0 else data_bytes
    return make_buf(padded)

def readback_arr(handle, n):
    """Read back n float32 values from GPU buffer."""
    nbytes = n * 4
    dst = (ctypes.c_ubyte * nbytes)()
    rc = lib.d3d12c_readback(handle, dst, nbytes)
    assert rc == 0, f"readback failed: {lib.d3d12c_get_last_error()}"
    return np.frombuffer(bytes(dst), dtype=np.float32).copy()

pipelines = {}
def get_pipeline(name):
    if name not in pipelines:
        with open(f"shaders_spv/{name}.spv", 'rb') as f:
            spv = f.read()
        buf = (ctypes.c_ubyte * len(spv))(*spv)
        h = lib.d3d12c_create_compute_pipeline(buf, len(spv))
        assert h != 0, f"pipeline {name} failed: {lib.d3d12c_get_last_error()}"
        pipelines[name] = h
    return pipelines[name]

def dispatch(pipeline, srvs=None, uavs=None, cbvs=None, groups=(1,1,1)):
    srv_arr = (ctypes.c_uint64 * len(srvs))(*srvs) if srvs else None
    uav_arr = (ctypes.c_uint64 * len(uavs))(*uavs) if uavs else None
    cbv_arr = (ctypes.c_uint64 * len(cbvs))(*cbvs) if cbvs else None
    rc = lib.d3d12c_dispatch_sync(
        pipeline,
        srv_arr, len(srvs) if srvs else 0,
        uav_arr, len(uavs) if uavs else 0,
        cbv_arr, len(cbvs) if cbvs else 0,
        groups[0], groups[1], groups[2]
    )
    return rc

def ceildiv(a, b):
    return (a + b - 1) // b

# ============================================================================
# Init
# ============================================================================
rc = lib.d3d12c_init()
assert rc == 0, f"init failed: {lib.d3d12c_get_last_error()}"

results = {}
N = 256  # element count for most tests

# ============================================================================
# 1. ADD: c = a + b (2 SRVs, 1 UAV)
# ============================================================================
print("Testing add...", end=" ", flush=True)
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
ha, hb, hc = make_arr_buf(a), make_arr_buf(b), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('add'), srvs=[ha, hb], uavs=[hc], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(hc, N)
expected = a + b
results['add'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['add'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, hb, hc]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 2. MUL: c = a * b (2 SRVs, 1 UAV)
# ============================================================================
print("Testing mul...", end=" ", flush=True)
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
ha, hb, hc = make_arr_buf(a), make_arr_buf(b), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('mul'), srvs=[ha, hb], uavs=[hc], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(hc, N)
expected = a * b
results['mul'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['mul'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, hb, hc]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 3. SUB: c = a - b (2 SRVs, 1 UAV)
# ============================================================================
print("Testing sub...", end=" ", flush=True)
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
ha, hb, hc = make_arr_buf(a), make_arr_buf(b), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('sub'), srvs=[ha, hb], uavs=[hc], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(hc, N)
expected = a - b
results['sub'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['sub'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, hb, hc]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 4. NEG: out = -a (1 SRV, 1 UAV)
# ============================================================================
print("Testing neg...", end=" ", flush=True)
a = np.random.randn(N).astype(np.float32)
ha, ho = make_arr_buf(a), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('neg'), srvs=[ha], uavs=[ho], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
expected = -a
results['neg'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['neg'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, ho]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 5. RELU: out = max(0, x) (1 SRV, 1 UAV)
# ============================================================================
print("Testing relu...", end=" ", flush=True)
x = np.random.randn(N).astype(np.float32)
hx, ho = make_arr_buf(x), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('relu'), srvs=[hx], uavs=[ho], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
expected = np.maximum(0, x)
results['relu'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['relu'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, ho]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 6. GELU: out = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) (1 SRV, 1 UAV)
# ============================================================================
print("Testing gelu...", end=" ", flush=True)
x = np.random.randn(N).astype(np.float32)
hx, ho = make_arr_buf(x), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('gelu'), srvs=[hx], uavs=[ho], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
expected = 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)))
results['gelu'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['gelu'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, ho]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 7. SIGMOID: out = 1/(1+exp(-x)) (1 SRV, 1 UAV)
# ============================================================================
print("Testing sigmoid...", end=" ", flush=True)
x = np.random.randn(N).astype(np.float32)
hx, ho = make_arr_buf(x), make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('sigmoid'), srvs=[hx], uavs=[ho], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
expected = 1.0 / (1.0 + np.exp(-x))
results['sigmoid'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['sigmoid'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, ho]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 8. SOFTMAX: (1 SRV, 1 UAV, 1 CBV) - params: vec4u [width, num_rows, 0, 0]
# ============================================================================
print("Testing softmax...", end=" ", flush=True)
width = 64
num_rows = 4
sm_N = num_rows * width
x = np.random.randn(sm_N).astype(np.float32)
hx = make_arr_buf(x)
ho = make_arr_buf(np.zeros(sm_N, dtype=np.float32))
# params: vec4<u32> - [width, num_rows, 0, 0]
cbv = make_cbv(struct.pack('4I', width, num_rows, 0, 0))
rc = dispatch(get_pipeline('softmax'), srvs=[hx], uavs=[ho], cbvs=[cbv], groups=(num_rows, 1, 1))
assert rc == 0
result = readback_arr(ho, sm_N)
x2d = x.reshape(num_rows, width)
expected = (np.exp(x2d - x2d.max(axis=1, keepdims=True)) /
            np.exp(x2d - x2d.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)).flatten()
results['softmax'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['softmax'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 9. LAYER_NORM: (3 SRVs: x,gamma,beta; 1 UAV: out; 1 CBV: params)
#    params = vec4<f32> [width, eps, 0, 0]
# ============================================================================
print("Testing layer_norm...", end=" ", flush=True)
width = 64
num_rows = 4
ln_N = num_rows * width
eps = 1e-5
x = np.random.randn(ln_N).astype(np.float32)
gamma = np.random.randn(width).astype(np.float32) * 0.1 + 1.0
beta = np.random.randn(width).astype(np.float32) * 0.1
hx = make_arr_buf(x)
hg = make_arr_buf(gamma)
hb = make_arr_buf(beta)
ho = make_arr_buf(np.zeros(ln_N, dtype=np.float32))
cbv = make_cbv(struct.pack('4f', float(width), eps, 0.0, 0.0))
rc = dispatch(get_pipeline('layer_norm'), srvs=[hx, hg, hb], uavs=[ho], cbvs=[cbv], groups=(num_rows, 1, 1))
assert rc == 0
result = readback_arr(ho, ln_N)
x2d = x.reshape(num_rows, width)
mean = x2d.mean(axis=1, keepdims=True)
var = x2d.var(axis=1, keepdims=True)
expected = ((x2d - mean) / np.sqrt(var + eps) * gamma + beta).flatten()
results['layer_norm'] = np.allclose(result, expected, atol=1e-4)
print(f"{'OK' if results['layer_norm'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, hg, hb, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 10. TRANSPOSE_2D: (1 SRV, 1 UAV, 1 CBV) - params: vec4u [rows, cols, 0, 0]
# ============================================================================
print("Testing transpose_2d...", end=" ", flush=True)
rows, cols = 16, 32
t_N = rows * cols
x = np.arange(t_N, dtype=np.float32)
hx = make_arr_buf(x)
ho = make_arr_buf(np.zeros(t_N, dtype=np.float32))
cbv = make_cbv(struct.pack('4I', rows, cols, 0, 0))
rc = dispatch(get_pipeline('transpose_2d'), srvs=[hx], uavs=[ho], cbvs=[cbv],
              groups=(ceildiv(cols, 16), ceildiv(rows, 16), 1))
assert rc == 0
result = readback_arr(ho, t_N)
expected = x.reshape(rows, cols).T.flatten()
results['transpose_2d'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['transpose_2d'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hx, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 11. MATMUL: (2 SRVs: a,b; 1 UAV: out; 1 CBV: params)
#     params = vec4<u32> [M, N, K, 0]
# ============================================================================
print("Testing matmul...", end=" ", flush=True)
M, K, Nm = 32, 16, 24
a = np.random.randn(M, K).astype(np.float32)
b = np.random.randn(K, Nm).astype(np.float32)
ha = make_arr_buf(a.flatten())
hb = make_arr_buf(b.flatten())
ho = make_arr_buf(np.zeros(M * Nm, dtype=np.float32))
cbv = make_cbv(struct.pack('4I', M, Nm, K, 0))
rc = dispatch(get_pipeline('matmul'), srvs=[ha, hb], uavs=[ho], cbvs=[cbv],
              groups=(ceildiv(Nm, 16), ceildiv(M, 16), 1))
assert rc == 0
result = readback_arr(ho, M * Nm).reshape(M, Nm)
expected = a @ b
results['matmul'] = np.allclose(result, expected, atol=1e-4)
print(f"{'OK' if results['matmul'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, hb, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 12. RELU_BACKWARD: grad_in = grad_out * (x > 0) (2 SRVs, 1 UAV)
# ============================================================================
print("Testing relu_backward...", end=" ", flush=True)
x = np.random.randn(N).astype(np.float32)
grad_out = np.random.randn(N).astype(np.float32)
hgo = make_arr_buf(grad_out)
hx = make_arr_buf(x)
hgi = make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('relu_backward'), srvs=[hgo, hx], uavs=[hgi], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(hgi, N)
expected = grad_out * (x > 0).astype(np.float32)
results['relu_backward'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['relu_backward'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hgo, hx, hgi]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 13. GELU_BACKWARD: (2 SRVs, 1 UAV)
# ============================================================================
print("Testing gelu_backward...", end=" ", flush=True)
x = np.random.randn(N).astype(np.float32)
grad_out = np.random.randn(N).astype(np.float32)
hgo = make_arr_buf(grad_out)
hx = make_arr_buf(x)
hgi = make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('gelu_backward'), srvs=[hgo, hx], uavs=[hgi], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(hgi, N)
# GELU backward: d/dx[GELU(x)] = 0.5*(1+tanh(a)) + 0.5*x*sech^2(a)*(sqrt(2/pi)*(1+3*0.044715*x^2))
# where a = sqrt(2/pi)*(x + 0.044715*x^3)
a_val = np.sqrt(2.0/np.pi) * (x + 0.044715 * x**3)
tanh_a = np.tanh(a_val)
sech2_a = 1.0 - tanh_a**2
da_dx = np.sqrt(2.0/np.pi) * (1.0 + 3.0 * 0.044715 * x**2)
dgelu_dx = 0.5 * (1.0 + tanh_a) + 0.5 * x * sech2_a * da_dx
expected = grad_out * dgelu_dx
results['gelu_backward'] = np.allclose(result, expected, atol=1e-4)
print(f"{'OK' if results['gelu_backward'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hgo, hx, hgi]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 14. EMBEDDING: (2 SRVs: weight, indices; 1 UAV: out; 1 CBV: params)
#     params = vec4<u32> [vocab_size, embed_dim, num_indices, 0]
# ============================================================================
print("Testing embedding...", end=" ", flush=True)
vocab_size = 32
embed_dim = 16
num_indices = 8
weight = np.random.randn(vocab_size, embed_dim).astype(np.float32)
indices = np.random.randint(0, vocab_size, num_indices).astype(np.uint32)
hw = make_arr_buf(weight.flatten())
hi = make_buf(indices.tobytes())
ho = make_arr_buf(np.zeros(num_indices * embed_dim, dtype=np.float32))
cbv = make_cbv(struct.pack('4I', vocab_size, embed_dim, num_indices, 0))
rc = dispatch(get_pipeline('embedding'), srvs=[hw, hi], uavs=[ho], cbvs=[cbv],
              groups=(num_indices, 1, 1))
assert rc == 0
result = readback_arr(ho, num_indices * embed_dim).reshape(num_indices, embed_dim)
expected = weight[indices]
results['embedding'] = np.allclose(result, expected, atol=1e-5)
print(f"{'OK' if results['embedding'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hw, hi, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 15. SCALAR_MUL: out = a * scalar (1 SRV, 1 UAV, 1 CBV)
# ============================================================================
print("Testing scalar_mul...", end=" ", flush=True)
a = np.random.randn(N).astype(np.float32)
scalar = 3.14
ha = make_arr_buf(a)
ho = make_arr_buf(np.zeros(N, dtype=np.float32))
cbv = make_cbv(struct.pack('f', scalar) + b'\x00' * 252)
rc = dispatch(get_pipeline('scalar_mul'), srvs=[ha], uavs=[ho], cbvs=[cbv], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
expected = a * scalar
results['scalar_mul'] = np.allclose(result, expected, atol=1e-4)
print(f"{'OK' if results['scalar_mul'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [ha, ho, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 16. CROSS_ENTROPY: out = -target*log(sigmoid(logit)) - (1-target)*log(1-sigmoid(logit))
#     (2 SRVs, 1 UAV)
# ============================================================================
print("Testing cross_entropy...", end=" ", flush=True)
logits = np.random.randn(N).astype(np.float32) * 2
targets = np.random.randint(0, 2, N).astype(np.float32)
hl = make_arr_buf(logits)
ht = make_arr_buf(targets)
ho = make_arr_buf(np.zeros(N, dtype=np.float32))
rc = dispatch(get_pipeline('cross_entropy'), srvs=[hl, ht], uavs=[ho], groups=(N, 1, 1))
assert rc == 0
result = readback_arr(ho, N)
# Binary cross entropy with logits
sig = 1.0 / (1.0 + np.exp(-logits))
expected = -(targets * np.log(sig + 1e-7) + (1.0 - targets) * np.log(1.0 - sig + 1e-7))
results['cross_entropy'] = np.allclose(result, expected, atol=1e-3)
print(f"{'OK' if results['cross_entropy'] else 'FAIL'} (max_diff={np.max(np.abs(result-expected)):.2e})")
for h in [hl, ht, ho]: lib.d3d12c_release_buffer(h)

# ============================================================================
# 17. SOFTMAX_BACKWARD: (2 SRVs: grad_out, probs; 1 UAV: grad_in; 1 CBV: params)
# ============================================================================
print("Testing softmax_backward...", end=" ", flush=True)
width = 64
num_rows = 4
sm_N = num_rows * width
probs_2d = np.random.dirichlet(np.ones(width), num_rows).astype(np.float32)
grad_out = np.random.randn(sm_N).astype(np.float32)
probs = probs_2d.flatten()
hgo = make_arr_buf(grad_out)
hp = make_arr_buf(probs)
hgi = make_arr_buf(np.zeros(sm_N, dtype=np.float32))
cbv = make_cbv(struct.pack('4I', width, num_rows, 0, 0))
rc = dispatch(get_pipeline('softmax_backward'), srvs=[hgo, hp], uavs=[hgi], cbvs=[cbv],
              groups=(num_rows, 1, 1))
assert rc == 0
result = readback_arr(hgi, sm_N).reshape(num_rows, width)
go_2d = grad_out.reshape(num_rows, width)
# softmax backward: grad_in = probs * (grad_out - sum(grad_out * probs))
dot = (go_2d * probs_2d).sum(axis=1, keepdims=True)
expected = (probs_2d * (go_2d - dot)).flatten()
results['softmax_backward'] = np.allclose(result.flatten(), expected, atol=1e-4)
print(f"{'OK' if results['softmax_backward'] else 'FAIL'} (max_diff={np.max(np.abs(result.flatten()-expected)):.2e})")
for h in [hgo, hp, hgi, cbv]: lib.d3d12c_release_buffer(h)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print(f"{'SHADER':<25s} {'RESULT':>6s}")
print("=" * 60)
passed = 0
for name, ok in results.items():
    status = "OK" if ok else "FAIL"
    print(f"  {name:<23s} {status:>6s}")
    if ok: passed += 1
print("=" * 60)
print(f"  {passed}/{len(results)} passed")

if passed == len(results):
    print("\n  ALL SHADERS CORRECT!")
else:
    print(f"\n  {len(results)-passed} FAILURES")

lib.d3d12c_shutdown()
sys.exit(0 if passed == len(results) else 1)
