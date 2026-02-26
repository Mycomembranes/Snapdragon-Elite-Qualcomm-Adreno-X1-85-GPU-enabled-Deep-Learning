"""Native D3D12 GPU tensor library — bypasses Vulkan/Dozen overhead.

Drop-in replacement for WgpuTensor operations. Calls libd3d12_compute.so
directly via ctypes, eliminating the Vulkan→Dozen→D3D12 translation that
adds 10-18ms per GPU submit.

Usage:
    from d3d12_tensor import D3D12Tensor, d3d12_init, d3d12_get_adapter_name
    d3d12_init()
    t = D3D12Tensor.from_numpy(np.array([1,2,3], dtype=np.float32))
    result = t.numpy()
"""

import os
import sys
import struct
import ctypes
import atexit
import numpy as np

# ============================================================================
# Library Loading
# ============================================================================

_lib = None
_initialized = False

def _load_library():
    """Load libd3d12_compute.so from the same directory as this module."""
    global _lib
    if _lib is not None:
        return _lib

    lib_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(lib_dir, "libd3d12_compute.so")

    if not os.path.exists(lib_path):
        raise RuntimeError(
            f"libd3d12_compute.so not found at {lib_path}. "
            "Run 'bash build_d3d12.sh' to build it."
        )

    _lib = ctypes.CDLL(lib_path)

    # Declare function signatures
    _lib.d3d12c_init.restype = ctypes.c_int
    _lib.d3d12c_shutdown.restype = None
    _lib.d3d12c_get_adapter_name.restype = ctypes.c_char_p
    _lib.d3d12c_get_last_error.restype = ctypes.c_char_p

    _lib.d3d12c_create_buffer.restype = ctypes.c_uint64
    _lib.d3d12c_create_buffer.argtypes = [ctypes.c_uint64]
    _lib.d3d12c_create_upload_buffer.restype = ctypes.c_uint64
    _lib.d3d12c_create_upload_buffer.argtypes = [ctypes.c_uint64]
    _lib.d3d12c_create_readback_buffer.restype = ctypes.c_uint64
    _lib.d3d12c_create_readback_buffer.argtypes = [ctypes.c_uint64]
    _lib.d3d12c_release_buffer.restype = None
    _lib.d3d12c_release_buffer.argtypes = [ctypes.c_uint64]

    _lib.d3d12c_upload.restype = ctypes.c_int
    _lib.d3d12c_upload.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
    _lib.d3d12c_readback.restype = ctypes.c_int
    _lib.d3d12c_readback.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]

    _lib.d3d12c_create_compute_pipeline.restype = ctypes.c_uint64
    _lib.d3d12c_create_compute_pipeline.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32  # spirv_data, spirv_size_bytes
    ]
    _lib.d3d12c_create_pipeline_from_dxil.restype = ctypes.c_uint64
    _lib.d3d12c_create_pipeline_from_dxil.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32  # dxil_data, dxil_size_bytes
    ]
    _lib.d3d12c_release_pipeline.restype = None
    _lib.d3d12c_release_pipeline.argtypes = [ctypes.c_uint64]

    _lib.d3d12c_begin_commands.restype = ctypes.c_int
    _lib.d3d12c_dispatch.restype = ctypes.c_int
    _lib.d3d12c_dispatch.argtypes = [
        ctypes.c_uint64,                                        # pipeline
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # srvs
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # uavs
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # cbvs
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32      # groups
    ]
    _lib.d3d12c_end_commands_and_wait.restype = ctypes.c_int
    _lib.d3d12c_dispatch_sync.restype = ctypes.c_int
    _lib.d3d12c_dispatch_sync.argtypes = [
        ctypes.c_uint64,                                        # pipeline
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # srvs
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # uavs
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32,      # cbvs
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32      # groups
    ]

    _lib.d3d12c_compile_spirv_to_dxil.restype = ctypes.c_int
    _lib.d3d12c_compile_spirv_to_dxil.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32)
    ]

    return _lib


# ============================================================================
# Initialization
# ============================================================================

def d3d12_init():
    """Initialize D3D12 backend. Call once before any GPU operations."""
    global _initialized
    if _initialized:
        return
    lib = _load_library()
    ret = lib.d3d12c_init()
    if ret != 0:
        err = lib.d3d12c_get_last_error()
        raise RuntimeError(f"D3D12 init failed: {err.decode() if err else 'unknown'}")
    _initialized = True
    atexit.register(_shutdown)


def _shutdown():
    global _initialized
    if _initialized and _lib is not None:
        _lib.d3d12c_shutdown()
        _initialized = False


def d3d12_get_adapter_name():
    """Return GPU adapter name string."""
    lib = _load_library()
    name = lib.d3d12c_get_adapter_name()
    return name.decode() if name else "unknown"


def get_device_info():
    """Return device info dict (compatible with wgpu_tensor API)."""
    d3d12_init()
    return {
        "vendor": "Qualcomm",
        "device": d3d12_get_adapter_name(),
        "backend": "d3d12-native",
        "adapter_type": "discrete",
    }


# ============================================================================
# Pipeline Cache
# ============================================================================

_pipeline_cache = {}  # key -> pipeline handle


def _get_or_create_pipeline(key, spirv_bytes):
    """Cache and return pipeline handle for given shader.

    The C layer auto-detects SRV/UAV/CBV counts from the DXBC PSV0 section
    after SPIR-V → DXIL compilation. No need to specify them here.
    """
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    lib = _load_library()

    spirv_arr = (ctypes.c_ubyte * len(spirv_bytes))(*spirv_bytes)

    handle = lib.d3d12c_create_compute_pipeline(
        spirv_arr, len(spirv_bytes)
    )
    if handle == 0:
        err = lib.d3d12c_get_last_error()
        raise RuntimeError(f"Pipeline creation failed: {err.decode() if err else 'unknown'}")

    _pipeline_cache[key] = handle
    return handle


# ============================================================================
# D3D12Tensor
# ============================================================================

class D3D12Tensor:
    """GPU tensor backed by a D3D12 default-heap buffer.

    Drop-in replacement for WgpuTensor with the same API surface.
    """

    def __init__(self, handle, shape, dtype="float32"):
        self.handle = handle
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._size_bytes = int(np.prod(shape)) * self.dtype.itemsize

    @staticmethod
    def from_numpy(arr):
        """Upload numpy array to GPU. Returns D3D12Tensor."""
        d3d12_init()
        lib = _load_library()

        arr = np.ascontiguousarray(arr, dtype=np.float32)
        size = arr.nbytes

        handle = lib.d3d12c_create_buffer(size)
        if handle == 0:
            raise RuntimeError("Failed to create GPU buffer")

        ret = lib.d3d12c_upload(handle, arr.ctypes.data, size)
        if ret != 0:
            lib.d3d12c_release_buffer(handle)
            err = lib.d3d12c_get_last_error()
            raise RuntimeError(f"Upload failed: {err.decode() if err else 'unknown'}")

        return D3D12Tensor(handle, arr.shape, arr.dtype)

    def numpy(self):
        """Download tensor from GPU to numpy array."""
        lib = _load_library()
        out = np.empty(self.shape, dtype=self.dtype)
        ret = lib.d3d12c_readback(self.handle, out.ctypes.data, self._size_bytes)
        if ret != 0:
            err = lib.d3d12c_get_last_error()
            raise RuntimeError(f"Readback failed: {err.decode() if err else 'unknown'}")
        return out

    @staticmethod
    def empty(shape, dtype="float32"):
        """Create uninitialized GPU tensor."""
        d3d12_init()
        lib = _load_library()
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        handle = lib.d3d12c_create_buffer(size)
        if handle == 0:
            raise RuntimeError("Failed to create GPU buffer")
        return D3D12Tensor(handle, shape, dtype)

    @staticmethod
    def zeros(shape, dtype="float32"):
        """Create zero-filled GPU tensor."""
        arr = np.zeros(shape, dtype=dtype)
        return D3D12Tensor.from_numpy(arr)

    def release(self):
        """Explicitly release GPU buffer."""
        if self.handle and _lib is not None:
            _lib.d3d12c_release_buffer(self.handle)
            self.handle = 0

    def __del__(self):
        self.release()

    def __repr__(self):
        return f"D3D12Tensor(shape={self.shape}, dtype={self.dtype}, handle={self.handle})"


# ============================================================================
# Dispatch helpers
# ============================================================================

def _dispatch_compute(pipeline_handle, srv_handles, uav_handles, cbv_handles,
                      groups_x, groups_y=1, groups_z=1):
    """Execute a compute dispatch with separate SRV/UAV/CBV bindings.

    srv_handles: list of buffer handles for read-only bindings (SRV/t-registers)
    uav_handles: list of buffer handles for read-write bindings (UAV/u-registers)
    cbv_handles: list of buffer handles for uniform bindings (CBV/b-registers)
    """
    lib = _load_library()

    num_srvs = len(srv_handles)
    num_uavs = len(uav_handles)
    num_cbvs = len(cbv_handles)

    srv_arr = (ctypes.c_uint64 * num_srvs)(*srv_handles) if num_srvs > 0 else None
    uav_arr = (ctypes.c_uint64 * num_uavs)(*uav_handles) if num_uavs > 0 else None
    cbv_arr = (ctypes.c_uint64 * num_cbvs)(*cbv_handles) if num_cbvs > 0 else None

    ret = lib.d3d12c_dispatch_sync(
        pipeline_handle,
        srv_arr, num_srvs,
        uav_arr, num_uavs,
        cbv_arr, num_cbvs,
        groups_x, groups_y, groups_z
    )
    if ret != 0:
        err = lib.d3d12c_get_last_error()
        raise RuntimeError(f"Dispatch failed: {err.decode() if err else 'unknown'}")


def _dispatch_batch(dispatches):
    """Execute multiple dispatches in a single command list submission.

    dispatches: list of (pipeline_handle, srv_handles, uav_handles, cbv_handles, gx, gy, gz)
    """
    lib = _load_library()

    ret = lib.d3d12c_begin_commands()
    if ret != 0:
        raise RuntimeError("begin_commands failed")

    for pipeline_handle, srv_handles, uav_handles, cbv_handles, gx, gy, gz in dispatches:
        num_srvs = len(srv_handles)
        num_uavs = len(uav_handles)
        num_cbvs = len(cbv_handles)
        srv_arr = (ctypes.c_uint64 * num_srvs)(*srv_handles) if num_srvs else None
        uav_arr = (ctypes.c_uint64 * num_uavs)(*uav_handles) if num_uavs else None
        cbv_arr = (ctypes.c_uint64 * num_cbvs)(*cbv_handles) if num_cbvs else None

        ret = lib.d3d12c_dispatch(
            pipeline_handle,
            srv_arr, num_srvs,
            uav_arr, num_uavs,
            cbv_arr, num_cbvs,
            gx, gy, gz
        )
        if ret != 0:
            err = lib.d3d12c_get_last_error()
            raise RuntimeError(f"Dispatch in batch failed: {err.decode() if err else 'unknown'}")

    ret = lib.d3d12c_end_commands_and_wait()
    if ret != 0:
        err = lib.d3d12c_get_last_error()
        raise RuntimeError(f"end_commands_and_wait failed: {err.decode() if err else 'unknown'}")


# ============================================================================
# SPIR-V Shader Loading
# ============================================================================

_spv_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders_spv")


def _load_spirv(name):
    """Load pre-compiled SPIR-V binary from shaders_spv/ directory.

    NonWritable decorations are preserved — the C layer auto-detects
    SRV vs UAV bindings from the DXBC PSV0 section after SPIR-V→DXIL
    compilation and builds a matching root signature.
    """
    path = os.path.join(_spv_cache_dir, f"{name}.spv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"SPIR-V shader not found: {path}. "
            f"Pre-compile with: naga {name}.wgsl {name}.spv"
        )
    with open(path, "rb") as f:
        return f.read()


# ============================================================================
# High-level GPU operations (matching wgpu_tensor API)
# ============================================================================

def _make_params_buffer(values):
    """Create a GPU buffer from uint32 or float32 parameter values."""
    # Pack as vec4<u32> (16 bytes, 256-byte aligned by the C layer)
    while len(values) < 4:
        values = list(values) + [0]
    data = struct.pack("4I", *[int(v) for v in values[:4]])
    buf = D3D12Tensor.from_numpy(np.frombuffer(data, dtype=np.float32))
    return buf


def _make_params_buffer_f32(values):
    """Create a GPU buffer from float32 parameter values."""
    while len(values) < 4:
        values = list(values) + [0.0]
    arr = np.array(values[:4], dtype=np.float32)
    return D3D12Tensor.from_numpy(arr)


def _ensure_pipeline(name, spirv_bytes):
    """Get or create a cached pipeline from SPIR-V bytes."""
    return _get_or_create_pipeline(name, spirv_bytes)


# ============================================================================
# Elementwise operations
# ============================================================================

def d3d12_add(a, b):
    """Element-wise addition: out = a + b"""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    spv = _load_spirv("add")
    pipeline = _ensure_pipeline("add", spv)
    out = D3D12Tensor.empty(a.shape)
    n = int(np.prod(a.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [a.handle, b.handle], [out.handle], [], groups)
    return out


def d3d12_sub(a, b):
    """Element-wise subtraction: out = a - b"""
    assert a.shape == b.shape
    spv = _load_spirv("sub")
    pipeline = _ensure_pipeline("sub", spv)
    out = D3D12Tensor.empty(a.shape)
    n = int(np.prod(a.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [a.handle, b.handle], [out.handle], [], groups)
    return out


def d3d12_mul(a, b):
    """Element-wise multiplication: out = a * b"""
    assert a.shape == b.shape
    spv = _load_spirv("mul")
    pipeline = _ensure_pipeline("mul", spv)
    out = D3D12Tensor.empty(a.shape)
    n = int(np.prod(a.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [a.handle, b.handle], [out.handle], [], groups)
    return out


def d3d12_scalar_mul(a, scalar):
    """Scalar multiplication: out = a * scalar"""
    spv = _load_spirv("scalar_mul")
    pipeline = _ensure_pipeline("scalar_mul", spv)
    out = D3D12Tensor.empty(a.shape)
    params = D3D12Tensor.from_numpy(np.array([scalar], dtype=np.float32))
    n = int(np.prod(a.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [a.handle], [out.handle], [params.handle], groups)
    return out


# ============================================================================
# Activation functions
# ============================================================================

def d3d12_relu(x):
    """ReLU activation: out = max(0, x)"""
    spv = _load_spirv("relu")
    pipeline = _ensure_pipeline("relu", spv)
    out = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [x.handle], [out.handle], [], groups)
    return out


def d3d12_gelu(x):
    """GELU activation"""
    spv = _load_spirv("gelu")
    pipeline = _ensure_pipeline("gelu", spv)
    out = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [x.handle], [out.handle], [], groups)
    return out


def d3d12_sigmoid(x):
    """Sigmoid activation"""
    spv = _load_spirv("sigmoid")
    pipeline = _ensure_pipeline("sigmoid", spv)
    out = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [x.handle], [out.handle], [], groups)
    return out


def d3d12_tanh(x):
    """Tanh activation"""
    spv = _load_spirv("tanh_act")
    pipeline = _ensure_pipeline("tanh_act", spv)
    out = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [x.handle], [out.handle], [], groups)
    return out


# ============================================================================
# Matrix operations
# ============================================================================

def d3d12_matmul(a, b):
    """Matrix multiplication: out = a @ b
    a: (M, K), b: (K, N) -> out: (M, N)
    """
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[0], f"Shape mismatch: {a.shape} @ {b.shape}"

    m, k = a.shape
    n = b.shape[1]

    spv = _load_spirv("matmul")
    pipeline = _ensure_pipeline("matmul", spv)

    out = D3D12Tensor.empty((m, n))
    params = _make_params_buffer([m, n, k, 0])

    groups_x = (n + 15) // 16
    groups_y = (m + 15) // 16

    _dispatch_compute(pipeline,
                      [a.handle, b.handle], [out.handle],
                      [params.handle],
                      groups_x, groups_y, 1)
    return out


def d3d12_transpose(x):
    """2D transpose: (M, N) -> (N, M)"""
    assert len(x.shape) == 2
    rows, cols = x.shape
    spv = _load_spirv("transpose_2d")
    pipeline = _ensure_pipeline("transpose_2d", spv)
    out = D3D12Tensor.empty((cols, rows))
    params = _make_params_buffer([rows, cols, 0, 0])
    groups_x = (cols + 15) // 16
    groups_y = (rows + 15) // 16
    _dispatch_compute(pipeline, [x.handle], [out.handle], [params.handle],
                      groups_x, groups_y, 1)
    return out


# ============================================================================
# Fused operations
# ============================================================================

def d3d12_matmul_add_relu(a, b, bias):
    """Fused matmul + bias + relu: out = relu(a @ b + bias)"""
    assert len(a.shape) == 2 and len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]

    spv = _load_spirv("matmul_add_relu")
    pipeline = _ensure_pipeline("matmul_add_relu", spv)

    out = D3D12Tensor.empty((m, n))
    pre_relu = D3D12Tensor.empty((m, n))
    params = _make_params_buffer([m, n, k, 0])

    groups_x = (n + 15) // 16
    groups_y = (m + 15) // 16

    _dispatch_compute(pipeline,
                      [a.handle, b.handle, bias.handle],
                      [out.handle, pre_relu.handle],
                      [params.handle],
                      groups_x, groups_y, 1)
    return out, pre_relu


def d3d12_matmul_add(a, b, bias):
    """Fused matmul + bias: out = a @ b + bias"""
    assert len(a.shape) == 2 and len(b.shape) == 2
    m, k = a.shape
    n = b.shape[1]

    spv = _load_spirv("matmul_add")
    pipeline = _ensure_pipeline("matmul_add", spv)

    out = D3D12Tensor.empty((m, n))
    params = _make_params_buffer([m, n, k, 0])

    groups_x = (n + 15) // 16
    groups_y = (m + 15) // 16

    _dispatch_compute(pipeline,
                      [a.handle, b.handle, bias.handle],
                      [out.handle],
                      [params.handle],
                      groups_x, groups_y, 1)
    return out


# ============================================================================
# Backward / gradient operations
# ============================================================================

def d3d12_relu_backward(grad_out, x):
    """ReLU backward: grad_in = grad_out * (x > 0)"""
    spv = _load_spirv("relu_backward")
    pipeline = _ensure_pipeline("relu_backward", spv)
    grad_in = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [grad_out.handle, x.handle], [grad_in.handle], [], groups)
    return grad_in


def d3d12_gelu_backward(grad_out, x):
    """GELU backward"""
    spv = _load_spirv("gelu_backward")
    pipeline = _ensure_pipeline("gelu_backward", spv)
    grad_in = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [grad_out.handle, x.handle], [grad_in.handle], [], groups)
    return grad_in


# ============================================================================
# Reduction operations
# ============================================================================

def d3d12_sum(x):
    """Sum all elements."""
    spv = _load_spirv("sum_reduce")
    n = int(np.prod(x.shape))

    current = x
    current_n = n

    while current_n > 1:
        out_n = (current_n + 255) // 256
        out = D3D12Tensor.empty((out_n,))
        params = _make_params_buffer([current_n, 256, 0, 0])

        pipeline = _ensure_pipeline("sum_reduce", spv)
        groups = out_n
        _dispatch_compute(pipeline, [current.handle], [out.handle], [params.handle], groups)

        current = out
        current_n = out_n

    return current


def d3d12_mean(x):
    """Mean of all elements."""
    spv = _load_spirv("mean_reduce")
    pipeline = _ensure_pipeline("mean_reduce", spv)
    n = int(np.prod(x.shape))
    out_n = (n + 255) // 256
    out = D3D12Tensor.empty((out_n,))
    params = _make_params_buffer_f32([float(n), 0.0, 0.0, 0.0])
    _dispatch_compute(pipeline, [x.handle], [out.handle], [params.handle], out_n)

    # Reduce further if needed
    if out_n > 1:
        return d3d12_sum(out)
    return out


# ============================================================================
# Normalization
# ============================================================================

def d3d12_layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization. x: (rows, width), gamma/beta: (width,)"""
    assert len(x.shape) == 2
    rows, width = x.shape
    spv = _load_spirv("layer_norm")
    pipeline = _ensure_pipeline("layer_norm", spv)
    out = D3D12Tensor.empty(x.shape)
    params = _make_params_buffer_f32([float(width), eps, 0.0, 0.0])
    _dispatch_compute(pipeline,
                      [x.handle, gamma.handle, beta.handle], [out.handle],
                      [params.handle],
                      rows, 1, 1)
    return out


def d3d12_softmax(x):
    """Softmax over last dimension. x: (rows, width)"""
    assert len(x.shape) == 2
    rows, width = x.shape
    spv = _load_spirv("softmax")
    pipeline = _ensure_pipeline("softmax", spv)
    out = D3D12Tensor.empty(x.shape)
    params = _make_params_buffer([width, 0, 0, 0])
    _dispatch_compute(pipeline, [x.handle], [out.handle], [params.handle], rows, 1, 1)
    return out


# ============================================================================
# Additional elementwise operations
# ============================================================================

def d3d12_neg(x):
    """Negation: out = -x"""
    spv = _load_spirv("neg")
    pipeline = _ensure_pipeline("neg", spv)
    out = D3D12Tensor.empty(x.shape)
    n = int(np.prod(x.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [x.handle], [out.handle], [], groups)
    return out


# ============================================================================
# Backward / gradient operations (additional)
# ============================================================================

def d3d12_softmax_backward(grad_out, probs):
    """Softmax backward: grad_in = P * (grad_out - sum(grad_out * P))"""
    assert len(grad_out.shape) == 2
    rows, width = grad_out.shape
    spv = _load_spirv("softmax_backward")
    pipeline = _ensure_pipeline("softmax_backward", spv)
    grad_in = D3D12Tensor.empty(grad_out.shape)
    params = _make_params_buffer([width, rows, 0, 0])
    _dispatch_compute(pipeline,
                      [grad_out.handle, probs.handle], [grad_in.handle],
                      [params.handle],
                      rows, 1, 1)
    return grad_in


def d3d12_layernorm_backward(grad_out, x, gamma, eps=1e-5):
    """LayerNorm backward: compute grad_input."""
    assert len(x.shape) == 2
    rows, width = x.shape
    spv = _load_spirv("layernorm_backward")
    pipeline = _ensure_pipeline("layernorm_backward", spv)
    grad_in = D3D12Tensor.empty(x.shape)
    params = _make_params_buffer_f32([float(width), eps, 0.0, 0.0])
    _dispatch_compute(pipeline,
                      [grad_out.handle, x.handle, gamma.handle], [grad_in.handle],
                      [params.handle],
                      rows, 1, 1)
    return grad_in


# ============================================================================
# Loss functions
# ============================================================================

def d3d12_cross_entropy(logits, targets):
    """Cross entropy loss."""
    spv = _load_spirv("cross_entropy")
    pipeline = _ensure_pipeline("cross_entropy", spv)
    out = D3D12Tensor.empty(logits.shape)
    n = int(np.prod(logits.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [logits.handle, targets.handle], [out.handle], [], groups)
    return out


def d3d12_focal_bce(logits, targets, gamma=2.0, alpha=0.25):
    """Focal binary cross entropy loss."""
    spv = _load_spirv("focal_bce")
    pipeline = _ensure_pipeline("focal_bce", spv)
    out = D3D12Tensor.empty(logits.shape)
    params = _make_params_buffer_f32([gamma, alpha, 0.0, 0.0])
    n = int(np.prod(logits.shape))
    groups = (n + 255) // 256
    _dispatch_compute(pipeline, [logits.handle, targets.handle], [out.handle],
                      [params.handle], groups)
    return out


# ============================================================================
# Embedding
# ============================================================================

def d3d12_embedding_lookup(weight, indices):
    """Gather rows from weight matrix by indices."""
    num_indices = int(np.prod(indices.shape))
    embedding_dim = weight.shape[-1]
    out_shape = indices.shape + (embedding_dim,)
    spv = _load_spirv("embedding")
    pipeline = _ensure_pipeline("embedding", spv)
    out = D3D12Tensor.empty(out_shape)
    total_elements = num_indices * embedding_dim
    groups = (total_elements + 255) // 256
    params = _make_params_buffer([num_indices, embedding_dim, 0, 0])
    _dispatch_compute(pipeline,
                      [weight.handle, indices.handle], [out.handle],
                      [params.handle], groups)
    return out


# ============================================================================
# Additional reductions
# ============================================================================

def d3d12_max_reduce(x):
    """Max of all elements."""
    spv = _load_spirv("max_reduce")
    n = int(np.prod(x.shape))

    current = x
    current_n = n

    while current_n > 1:
        out_n = (current_n + 255) // 256
        out = D3D12Tensor.empty((out_n,))
        params = _make_params_buffer([current_n, 0, 0, 0])
        pipeline = _ensure_pipeline("max_reduce", spv)
        _dispatch_compute(pipeline, [current.handle], [out.handle], [params.handle], out_n)
        current = out
        current_n = out_n

    return current


# ============================================================================
# Convenience: check availability
# ============================================================================

def is_available():
    """Check if D3D12 native backend is available."""
    try:
        _load_library()
        return True
    except (RuntimeError, OSError):
        return False
