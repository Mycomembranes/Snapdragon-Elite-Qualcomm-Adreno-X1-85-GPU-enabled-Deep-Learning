#!/bin/bash
# Compile all WGSL shaders to SPIR-V using naga-cli.
# Validates each output with spirv-val.
# Idempotent: safe to re-run after shader changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WGSL_DIR="$SCRIPT_DIR/shaders_wgsl"
SPV_DIR="$SCRIPT_DIR/shaders_spv"
NAGA="${NAGA:-$(command -v naga 2>/dev/null || echo "$HOME/.cargo/bin/naga")}"
SPIRV_VAL="${SPIRV_VAL:-$(command -v spirv-val 2>/dev/null || echo "")}"

if [ ! -x "$NAGA" ]; then
    echo "ERROR: naga not found. Install with: cargo install naga-cli"
    exit 1
fi

mkdir -p "$SPV_DIR"

passed=0
failed=0

for f in "$WGSL_DIR"/*.wgsl; do
    name=$(basename "$f" .wgsl)
    out="$SPV_DIR/${name}.spv"
    if "$NAGA" "$f" "$out" 2>&1; then
        if [ -n "$SPIRV_VAL" ] && [ -x "$SPIRV_VAL" ]; then
            if "$SPIRV_VAL" "$out" 2>&1; then
                echo "  OK: $name"
                passed=$((passed + 1))
            else
                echo "  FAIL (spirv-val): $name"
                failed=$((failed + 1))
            fi
        else
            echo "  OK: $name (spirv-val not available)"
            passed=$((passed + 1))
        fi
    else
        echo "  FAIL (naga): $name"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Results: $passed passed, $failed failed out of $((passed + failed)) shaders"
[ $failed -eq 0 ] || exit 1
