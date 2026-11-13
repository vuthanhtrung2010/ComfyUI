#!/usr/bin/env bash
set -euo pipefail

KERNEL_SLUG="devtrung/comfyui-tpu-8-core-benchmark"
KERNEL_DIR="kaggle/tpu_comfyui_benchmark"
OUTPUT_DIR="out/kaggle_tpu"

echo "=========================================================="
echo "  Google Cloud TPU v3-8 (8-Core) ComfyUI Kaggle Benchmark "
echo "=========================================================="

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI is not installed or not in PATH."
    exit 1
fi

echo "1. Pushing benchmark kernel to Kaggle..."
kaggle kernels push -p "$KERNEL_DIR"

echo "2. Monitoring Kaggle kernel execution..."
STATUS="queued"
while [[ "$STATUS" == "queued" || "$STATUS" == "running" ]]; do
    sleep 30
    STATUS_LINE=$(kaggle kernels status "$KERNEL_SLUG" || echo "running")
    echo "Kernel status: $STATUS_LINE"
    if echo "$STATUS_LINE" | grep -qi "complete"; then
        STATUS="complete"
        break
    elif echo "$STATUS_LINE" | grep -qi "error"; then
        echo "Error: Kernel execution failed on Kaggle."
        STATUS="error"
        break
    fi
done

if [[ "$STATUS" == "complete" ]]; then
    echo "3. Downloading benchmark outputs and generated images..."
    mkdir -p "$OUTPUT_DIR"
    kaggle kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR"

    echo "=========================================================="
    echo "  Benchmark Results Retrieved Successfully! "
    echo "=========================================================="
    ls -lh "$OUTPUT_DIR"
    if [[ -f "$OUTPUT_DIR/benchmark_report.json" ]]; then
        cat "$OUTPUT_DIR/benchmark_report.json"
    fi
else
    echo "Kernel did not complete successfully. Status: $STATUS"
    exit 1
fi
