#!/usr/bin/env python3
"""Test D3D12 buffer upload/readback and compute dispatch end-to-end."""
import ctypes, struct, sys, os

os.chdir("/home/mukshud/claude_wsl/new/claude_rotifer/operonfold")
lib = ctypes.CDLL("./libd3d12_compute.so")

# Set up function signatures
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
lib.d3d12c_shutdown.restype = None

print("=== Test 1: Init ===")
rc = lib.d3d12c_init()
print(f"Init: {rc}")
if rc != 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    sys.exit(1)
print(f"GPU: {lib.d3d12c_get_adapter_name().decode()}")

print("\n=== Test 2: Buffer roundtrip ===")
# Create float32 data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
data = struct.pack('8f', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
size = len(data)  # 32 bytes
print(f"Data size: {size} bytes")

buf = lib.d3d12c_create_buffer(size)
print(f"Buffer handle: {buf}")
if buf == 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    sys.exit(1)

# Upload
src = (ctypes.c_ubyte * size)(*data)
rc = lib.d3d12c_upload(buf, src, size)
print(f"Upload rc: {rc}")
if rc != 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    sys.exit(1)

# Readback
dst = (ctypes.c_ubyte * size)()
rc = lib.d3d12c_readback(buf, dst, size)
print(f"Readback rc: {rc}")
if rc != 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    sys.exit(1)

# Compare
result = struct.unpack('8f', bytes(dst))
expected = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
print(f"Expected: {expected}")
print(f"Got:      {result}")
match = all(abs(a-b) < 1e-6 for a,b in zip(expected, result))
print(f"Match: {match}")

lib.d3d12c_release_buffer(buf)

if not match:
    print("FAIL: buffer roundtrip mismatch")
    sys.exit(1)

print("\n=== Test 3: Compute dispatch (add shader) ===")
# Load add shader SPIR-V
lib.d3d12c_create_compute_pipeline.restype = ctypes.c_uint64
lib.d3d12c_create_compute_pipeline.argtypes = [
    ctypes.c_void_p, ctypes.c_uint32
]

spv_path = "shaders_spv/add.spv"
if not os.path.exists(spv_path):
    # Try GLSL-compiled version
    spv_path = "shaders_spv/noop.spv"
    print(f"Warning: add.spv not found, using {spv_path}")

with open(spv_path, 'rb') as f:
    spv_data = f.read()

spv_buf = (ctypes.c_ubyte * len(spv_data))(*spv_data)
pipeline = lib.d3d12c_create_compute_pipeline(spv_buf, len(spv_data))
print(f"Pipeline handle: {pipeline}")
if pipeline == 0:
    print(f"Error: {lib.d3d12c_get_last_error()}")
    print("Skipping dispatch test")
else:
    # For add shader: a[i] = b[i] + c[i]
    # Bindings from PSV0: t0 (SRV), t1 (SRV), u2 (UAV)
    # We need 3 buffers
    N = 8
    nbytes = N * 4  # float32

    buf_a = lib.d3d12c_create_buffer(nbytes)  # input A
    buf_b = lib.d3d12c_create_buffer(nbytes)  # input B
    buf_c = lib.d3d12c_create_buffer(nbytes)  # output C
    print(f"Buffers: A={buf_a}, B={buf_b}, C={buf_c}")

    # Upload input data
    a_data = struct.pack(f'{N}f', *[float(i) for i in range(N)])
    b_data = struct.pack(f'{N}f', *[float(i*10) for i in range(N)])

    src_a = (ctypes.c_ubyte * nbytes)(*a_data)
    src_b = (ctypes.c_ubyte * nbytes)(*b_data)

    rc1 = lib.d3d12c_upload(buf_a, src_a, nbytes)
    rc2 = lib.d3d12c_upload(buf_b, src_b, nbytes)
    print(f"Upload A: {rc1}, Upload B: {rc2}")

    # Dispatch - need to figure out the right binding order
    # The dispatch function expects UAV handles, but our shader has SRVs + UAV
    # This is where the mismatch is!

    # dispatch_sync(pipeline, uav_handles, num_uavs, cbv_handles, num_cbvs, gx, gy, gz)
    lib.d3d12c_dispatch_sync.restype = ctypes.c_int
    lib.d3d12c_dispatch_sync.argtypes = [
        ctypes.c_uint64,                           # pipeline
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,  # uavs
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,  # cbvs
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32  # groups
    ]

    uavs = (ctypes.c_uint64 * 3)(buf_a, buf_b, buf_c)
    rc = lib.d3d12c_dispatch_sync(pipeline, uavs, 3, None, 0, N, 1, 1)
    print(f"Dispatch rc: {rc}")
    if rc != 0:
        print(f"Error: {lib.d3d12c_get_last_error()}")
    else:
        # Readback output
        out = (ctypes.c_ubyte * nbytes)()
        rc = lib.d3d12c_readback(buf_c, out, nbytes)
        print(f"Readback rc: {rc}")
        if rc == 0:
            result = struct.unpack(f'{N}f', bytes(out))
            expected = tuple(float(i + i*10) for i in range(N))
            print(f"Expected: {expected}")
            print(f"Got:      {result}")

    lib.d3d12c_release_buffer(buf_a)
    lib.d3d12c_release_buffer(buf_b)
    lib.d3d12c_release_buffer(buf_c)

print("\n=== Test 4: Noop dispatch ===")
with open("shaders_spv/noop.spv", 'rb') as f:
    spv_data = f.read()
spv_buf = (ctypes.c_ubyte * len(spv_data))(*spv_data)
noop_pipeline = lib.d3d12c_create_compute_pipeline(spv_buf, len(spv_data))
print(f"Noop pipeline: {noop_pipeline}")
if noop_pipeline != 0:
    rc = lib.d3d12c_dispatch_sync(noop_pipeline, None, 0, None, 0, 1, 1, 1)
    print(f"Noop dispatch rc: {rc}")
    if rc != 0:
        print(f"Error: {lib.d3d12c_get_last_error()}")

lib.d3d12c_shutdown()
print("\nDone.")
