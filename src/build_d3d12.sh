#!/bin/bash
# Build native D3D12 compute backend for WSL2
# Produces: libd3d12_compute.so
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# WSL2 DirectX headers + stubs
INCLUDES="-I/usr/include/wsl/stubs -I/usr/include/directx"

# Mesa spirv_to_dxil header path
SPIRV_INCLUDES="-I/home/mukshud/mesa-25.2.8/src/microsoft/spirv_to_dxil \
-I/home/mukshud/mesa-25.2.8/src/microsoft/compiler"

# Library paths
D3D12_LIBS="-L/usr/lib/wsl/lib"
SPIRV_LIBS="-L/home/mukshud/mesa-dozen-install/lib/aarch64-linux-gnu"

echo "Building libd3d12_compute.so..."

gcc -shared -fPIC -O2 -Wall -Wno-unused-function \
    -I/usr/include/wsl/stubs -I/usr/include/directx \
    d3d12_compute.c \
    -ldl -lpthread \
    -o libd3d12_compute.so

echo "Build successful: $(ls -lh libd3d12_compute.so | awk '{print $5}')"
echo ""
echo "Library dependencies:"
ldd libd3d12_compute.so 2>/dev/null | grep -E "d3d12|dxcore|spirv" || echo "  (all loaded via dlopen at runtime)"
echo ""
echo "Exported symbols:"
nm -D libd3d12_compute.so | grep " T d3d12c_" | awk '{print "  " $3}'
