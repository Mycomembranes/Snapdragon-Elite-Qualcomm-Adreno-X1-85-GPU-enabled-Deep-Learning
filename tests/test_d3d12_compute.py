"""Tests and benchmarks for native D3D12 compute backend.

Tests:
  1. Init + adapter detection
  2. Buffer upload/readback correctness (various sizes)
  3. Data roundtrip integrity
  4. D3D12Tensor API (from_numpy, numpy, empty, zeros)
  5. Backend auto-detection in wgpu_tensor.py
  6. Benchmark: upload/readback latency vs size
  7. Benchmark: dispatch overhead (once shaders are compiled)

Usage:
    python3 test_d3d12_compute.py
"""

import time
import sys
import os
import numpy as np

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_init():
    """Test D3D12 initialization and adapter detection."""
    from d3d12_tensor import d3d12_init, d3d12_get_adapter_name, get_device_info

    d3d12_init()
    name = d3d12_get_adapter_name()
    info = get_device_info()

    assert len(name) > 0, "Adapter name is empty"
    assert "Adreno" in name or "Qualcomm" in name, f"Unexpected adapter: {name}"
    assert info["backend"] == "d3d12-native"
    print(f"  PASS: init — {name}")
    return True


def test_buffer_roundtrip():
    """Test data integrity across upload/readback for various sizes and patterns."""
    from d3d12_tensor import D3D12Tensor

    test_cases = [
        ("tiny", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)),
        ("zeros", np.zeros(1000, dtype=np.float32)),
        ("ones", np.ones(1000, dtype=np.float32)),
        ("sequential", np.arange(10000, dtype=np.float32)),
        ("random_1d", np.random.randn(50000).astype(np.float32)),
        ("random_2d", np.random.randn(100, 200).astype(np.float32)),
        ("random_3d", np.random.randn(10, 20, 30).astype(np.float32)),
        ("large", np.random.randn(500000).astype(np.float32)),
        ("negative", np.array([-1e6, -0.001, 0.0, 0.001, 1e6], dtype=np.float32)),
        ("subnormal", np.array([1e-38, 1e-30, 1e-20], dtype=np.float32)),
    ]

    for name, arr in test_cases:
        t = D3D12Tensor.from_numpy(arr)
        out = t.numpy()
        assert out.shape == arr.shape, f"{name}: shape mismatch {out.shape} vs {arr.shape}"
        assert np.array_equal(arr, out), f"{name}: data mismatch, max diff={np.max(np.abs(arr - out))}"
        t.release()
        print(f"  PASS: roundtrip — {name} (shape={arr.shape})")

    return True


def test_tensor_api():
    """Test D3D12Tensor class methods."""
    from d3d12_tensor import D3D12Tensor

    # from_numpy
    arr = np.random.randn(50, 30).astype(np.float32)
    t = D3D12Tensor.from_numpy(arr)
    assert t.shape == (50, 30)
    assert t.dtype == np.float32
    assert t.handle > 0

    # numpy()
    out = t.numpy()
    assert np.array_equal(arr, out)

    # empty
    e = D3D12Tensor.empty((10, 20))
    assert e.shape == (10, 20)
    assert e.handle > 0

    # zeros
    z = D3D12Tensor.zeros((5, 5))
    zout = z.numpy()
    assert np.all(zout == 0.0)

    # release
    t.release()
    assert t.handle == 0

    # repr
    t2 = D3D12Tensor.from_numpy(np.zeros(10, dtype=np.float32))
    r = repr(t2)
    assert "D3D12Tensor" in r
    assert "shape=(10,)" in r

    print("  PASS: D3D12Tensor API")
    return True


def test_backend_detection():
    """Test that wgpu_tensor auto-detects D3D12 backend."""
    # Remove cached module if loaded
    for mod in list(sys.modules.keys()):
        if "wgpu_tensor" in mod:
            del sys.modules[mod]

    # Force reimport
    os.environ.pop("WGPU_FORCE_BACKEND", None)
    import wgpu_tensor
    backend = wgpu_tensor.get_backend()
    print(f"  Backend detected: {backend}")
    assert backend in ("d3d12", "wgpu"), f"Unknown backend: {backend}"
    # If library exists, should detect d3d12
    if os.path.exists(os.path.join(os.path.dirname(__file__), "libd3d12_compute.so")):
        assert backend == "d3d12", "D3D12 library exists but not detected"
        print("  PASS: backend auto-detection (d3d12)")
    else:
        print(f"  PASS: backend auto-detection ({backend})")
    return True


def bench_upload_readback():
    """Benchmark upload and readback latency across sizes."""
    from d3d12_tensor import D3D12Tensor

    print("\n  Upload/Readback Benchmark:")
    print(f"  {'Size':>12s}  {'Upload':>10s}  {'Readback':>10s}  {'Total':>10s}  {'BW(GB/s)':>10s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for n in [100, 1000, 10000, 100000, 1000000, 4000000]:
        arr = np.random.randn(n).astype(np.float32)
        nbytes = arr.nbytes

        # Warmup
        t = D3D12Tensor.from_numpy(arr)
        _ = t.numpy()
        t.release()

        # Measure (average of 5 runs)
        upload_times = []
        readback_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            t = D3D12Tensor.from_numpy(arr)
            t1 = time.perf_counter()
            _ = t.numpy()
            t2 = time.perf_counter()
            upload_times.append(t1 - t0)
            readback_times.append(t2 - t1)
            t.release()

        up = np.median(upload_times) * 1000
        rb = np.median(readback_times) * 1000
        total = up + rb
        bw = (2 * nbytes) / (total / 1000) / 1e9  # round-trip bandwidth

        print(f"  {n:>12,d}  {up:>8.2f}ms  {rb:>8.2f}ms  {total:>8.2f}ms  {bw:>8.2f}")

    return True


def bench_dispatch_overhead():
    """Benchmark raw dispatch overhead (empty dispatches to measure submit cost).

    This measures the D3D12 command list overhead without shader execution.
    """
    import ctypes

    from d3d12_tensor import _load_library, d3d12_init, D3D12Tensor

    d3d12_init()
    lib = _load_library()

    # Create a small buffer
    buf = D3D12Tensor.from_numpy(np.zeros(256, dtype=np.float32))

    # Measure begin + end (no dispatch) overhead
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        lib.d3d12c_begin_commands()
        lib.d3d12c_end_commands_and_wait()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = times[10:]  # drop first 10 warmup
    med = np.median(times)
    p95 = np.percentile(times, 95)
    print(f"\n  Empty Submit Overhead (D3D12 native):")
    print(f"    Median: {med:.3f}ms")
    print(f"    P95:    {p95:.3f}ms")
    print(f"    Min:    {min(times):.3f}ms")

    buf.release()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("D3D12 Native Compute Backend — Tests & Benchmarks")
    print("=" * 60)

    tests = [
        ("Initialization", test_init),
        ("Buffer Roundtrip", test_buffer_roundtrip),
        ("D3D12Tensor API", test_tensor_api),
        ("Backend Detection", test_backend_detection),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Benchmarks
    print(f"\n{'=' * 60}")
    print("Benchmarks")
    print("=" * 60)

    try:
        bench_upload_readback()
    except Exception as e:
        print(f"  Benchmark error: {e}")

    try:
        bench_dispatch_overhead()
    except Exception as e:
        print(f"  Benchmark error: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
