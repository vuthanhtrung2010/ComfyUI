#!/usr/bin/env python3
"""Orchestrate the 25-step wait, 8-step run, artifact download, and clean shutdown of Colab VM."""
import os
import sys
import time
import subprocess
import json

COLAB_CLI = "/home/devtrung/.local/bin/colab"
SESSION = "comfy-batch"
OUT_DIR = "/home/devtrung/orca/ComfyUI/output/neptune_comparison"
ARTIFACT_DIR = "/home/devtrung/.gemini/antigravity-cli/brain/55517722-56f8-4d5b-a54c-8f67dac28dc1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def run_cmd(cmd):
    print(f"[LOCAL RUN] {' '.join(cmd)}")
    return subprocess.run(cmd, capture_output=True, text=True)

def wait_for_25step_run():
    print("=== Step 1: Waiting for 25-Step Run to Finish ===")
    for _ in range(300):
        time.sleep(10)
        # Check if 3rd image exists
        res = run_cmd([COLAB_CLI, "--auth=oauth2", "ls", "-s", SESSION, "/content/ComfyUI/output/"])
        if "neptune25_opt_03_neptune_bone_needles_00001_.png" in res.stdout:
            print("Detected neptune25_opt_03! 25-step run is complete.")
            time.sleep(5)
            return True
        print("Waiting for 25-step images... still in progress...")
    raise TimeoutError("25-step run did not complete in time.")

def run_8step_comparison():
    print("=== Step 2: Executing 8-Step Run on Same 3 Prompts ===")
    cmd = [COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION, "-f", "tools/run_neptune_optimized_3prod_8steps.py", "--timeout", "1200"]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("8-step run stdout:")
    print(res.stdout)
    if res.returncode != 0:
        print("8-step run stderr:", res.stderr)

def download_and_compare():
    print("=== Step 3: Downloading Comparison Images and Reports ===")
    files_to_download = [
        "neptune25_opt_01_neptune_glacier_wall_00001_.png",
        "neptune25_opt_02_neptune_bone_fires_00001_.png",
        "neptune25_opt_03_neptune_bone_needles_00001_.png",
        "neptune8_opt_01_neptune_glacier_wall_00001_.png",
        "neptune8_opt_02_neptune_bone_fires_00001_.png",
        "neptune8_opt_03_neptune_bone_needles_00001_.png"
    ]
    for f in files_to_download:
        remote_path = f"/content/ComfyUI/output/{f}"
        local_path = os.path.join(OUT_DIR, f)
        artifact_path = os.path.join(ARTIFACT_DIR, f)
        print(f"Downloading {f}...")
        res = run_cmd([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote_path, local_path])
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            print(f"Downloaded: {local_path} ({size} bytes)")
            # Copy to artifact dir for markdown embedding
            import shutil
            shutil.copy(local_path, artifact_path)

    # Download reports
    for rep in ["neptune3_opt_results.json", "neptune3_8step_opt_results.json"]:
        remote = f"/content/{rep}"
        local = os.path.join(OUT_DIR, rep)
        run_cmd([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote, local])

def stop_session():
    print("=== Step 4: Stopping Colab Session to Prevent Compute Waste ===")
    cmd = [COLAB_CLI, "--auth=oauth2", "stop", "-s", SESSION]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    print("Colab VM stopped successfully!")

if __name__ == "__main__":
    try:
        wait_for_25step_run()
        run_8step_comparison()
        download_and_compare()
    finally:
        stop_session()
