#!/usr/bin/env python3
"""Automated end-to-end runner for Batch 2: provision TPU, run 3 images @ 25 steps, download, stop VM."""
import os
import sys
import time
import subprocess
import shutil

COLAB_CLI = "/home/devtrung/.local/bin/colab"
SESSION = "comfy-batch2"
OUT_DIR = "/home/devtrung/orca/ComfyUI/output/neptune_batch2"
ARTIFACT_DIR = "/home/devtrung/.gemini/antigravity-cli/brain/55517722-56f8-4d5b-a54c-8f67dac28dc1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def provision_session():
    print("=== Step 1: Provisioning Colab TPU v5e Session ===")
    res = subprocess.run([COLAB_CLI, "--auth=oauth2", "new", "-s", SESSION, "--tpu", "v5e1"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Error provisioning session:", res.stderr)
        raise RuntimeError("Failed to provision Colab session")
    print("TPU Session is READY.")

def execute_batch():
    print("=== Step 2: Executing Optimized 25-Step Benchmark ===")
    cmd = [COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION, "-f", "tools/run_neptune_optimized_batch2.py", "--timeout", "2400"]
    print(f"Running command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Batch stdout:")
    print(res.stdout)
    if res.returncode != 0:
        print("Batch stderr:", res.stderr)

def download_artifacts():
    print("=== Step 3: Downloading Output Images & Performance Metrics ===")
    files = [
        "neptune25_batch2_01_neptune_viking_ocean_voyage_00001_.png",
        "neptune25_batch2_02_neptune_scandinavian_forest_00001_.png",
        "neptune25_batch2_03_neptune_mammoth_bone_hut_00001_.png"
    ]
    for f in files:
        remote = f"/content/ComfyUI/output/{f}"
        local = os.path.join(OUT_DIR, f)
        artifact = os.path.join(ARTIFACT_DIR, f)
        print(f"Downloading {f}...")
        subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote, local])
        if os.path.exists(local):
            print(f"Downloaded: {local} ({os.path.getsize(local)} bytes)")
            shutil.copy(local, artifact)

    # Download results json
    remote_json = "/content/neptune3_batch2_results.json"
    local_json = os.path.join(OUT_DIR, "neptune3_batch2_results.json")
    subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote_json, local_json])

def stop_vm():
    print("=== Step 4: Stopping Session to Prevent Compute Waste ===")
    res = subprocess.run([COLAB_CLI, "--auth=oauth2", "stop", "-s", SESSION], capture_output=True, text=True)
    print(res.stdout)
    print("Colab session stopped successfully.")

if __name__ == "__main__":
    try:
        provision_session()
        execute_batch()
        download_artifacts()
    finally:
        stop_vm()
