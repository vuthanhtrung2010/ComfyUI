#!/usr/bin/env python3
"""Automated end-to-end runner for Batch 3: provision TPU, run 3 images @ 25 steps with CLIP on TPU, download, stop VM."""
import os
import sys
import time
import subprocess
import shutil

COLAB_CLI = "/home/devtrung/.local/bin/colab"
SESSION = "comfy-batch3"
OUT_DIR = "/home/devtrung/orca/ComfyUI/output/neptune_batch3"
ARTIFACT_DIR = "/home/devtrung/.gemini/antigravity-cli/brain/55517722-56f8-4d5b-a54c-8f67dac28dc1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def provision_session():
    print("=== Step 1: Provisioning Colab TPU v5e Session ===")
    res = subprocess.run([COLAB_CLI, "--auth=oauth2", "new", "-s", SESSION, "--tpu", "v5e1"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Error provisioning session:", res.stderr)
        raise RuntimeError("Failed to provision Colab session")
    print("TPU Session is READY.")

def execute_batch():
    print("=== Step 2: Executing Optimized 25-Step Benchmark (CLIP ON TPU) ===")
    cmd = [COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION, "-f", "tools/run_neptune_optimized_batch3.py", "--timeout", "2400"]
    print(f"Running command: {' '.join(cmd)}")
    
    # Stream stdout/stderr in real-time
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"Batch script exited with code {proc.returncode}")

def download_artifacts():
    print("=== Step 3: Downloading Output Images & Performance Metrics ===")
    files = [
        "neptune25_batch3_01_neptune_predator_defense_00001_.png",
        "neptune25_batch3_02_neptune_skeletal_truss_00001_.png",
        "neptune25_batch3_03_neptune_argentinosaurus_fossil_00001_.png"
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
    remote_json = "/content/neptune3_batch3_results.json"
    local_json = os.path.join(OUT_DIR, "neptune3_batch3_results.json")
    subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote_json, local_json])
    if os.path.exists(local_json):
        shutil.copy(local_json, os.path.join(ARTIFACT_DIR, "neptune3_batch3_results.json"))

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
