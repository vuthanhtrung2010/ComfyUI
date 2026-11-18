#!/usr/bin/env python3
"""Automated end-to-end runner for Batch 4: provision TPU, upload local optimizations, microbench, run 3 prod images + warm check, download, stop VM."""
import os
import sys
import time
import tempfile
import subprocess
import shutil

COLAB_CLI = "/home/devtrung/.local/bin/colab"
SESSION = "comfy-batch4"
REPO_DIR = "/home/devtrung/orca/ComfyUI"
OUT_DIR = "/home/devtrung/orca/ComfyUI/output/neptune_batch4"
ARTIFACT_DIR = "/home/devtrung/.gemini/antigravity-cli/brain/55517722-56f8-4d5b-a54c-8f67dac28dc1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def stream_cmd(cmd):
    safe_cmd = []
    mask_next = False
    for part in cmd:
        if mask_next:
            safe_cmd.append("HF_TOKEN=***" if part.startswith("HF_TOKEN=") else part)
            mask_next = False
        else:
            safe_cmd.append(part)
            if part == "--env":
                mask_next = True
    print(f"[RUN] {' '.join(safe_cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def provision_session(max_retries=2):
    print("=== Step 1: Provisioning Colab TPU v5e Session ===", flush=True)
    for attempt in range(1, max_retries + 2):
        print(f"Attempt {attempt}/{max_retries + 1}...", flush=True)
        res = subprocess.run([COLAB_CLI, "--auth=oauth2", "new", "-s", SESSION, "--tpu", "v5e1"], capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.returncode == 0:
            print("TPU Session is READY.", flush=True)
            return
        print(f"Attempt {attempt} failed: {res.stderr}", flush=True)
        subprocess.run([COLAB_CLI, "--auth=oauth2", "stop", "-s", SESSION], capture_output=True, text=True)
        if attempt <= max_retries:
            time.sleep(10)
    raise RuntimeError("Failed to provision Colab session after retries")


def clone_and_upload_patches():
    print("=== Step 2a: Cloning Base Repo & Uploading Local Optimizations ===", flush=True)
    bootstrap_code = """
import os, subprocess
COMFY_DIR = "/content/ComfyUI"
if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
                    "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR], check=True)
print("Base repo ready at /content/ComfyUI", flush=True)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(bootstrap_code)
        tf_path = tf.name
    try:
        rc, _ = stream_cmd([COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION, "-f", tf_path, "--timeout", "300"])
        if rc != 0:
            raise RuntimeError("Failed to bootstrap ComfyUI repo on Colab")
    finally:
        os.unlink(tf_path)

    marker_path = None
    with tempfile.NamedTemporaryFile("w", delete=False) as mf:
        mf.write("1\n")
        marker_path = mf.name

    files_to_upload = [
        ("comfy/model_management.py", "/content/ComfyUI/comfy/model_management.py"),
        ("comfy/quant_ops.py", "/content/ComfyUI/comfy/quant_ops.py"),
        ("comfy/ops.py", "/content/ComfyUI/comfy/ops.py"),
        ("comfy/text_encoders/qwen_image21.py", "/content/ComfyUI/comfy/text_encoders/qwen_image21.py"),
        ("tools/run_neptune_optimized_batch4.py", "/content/ComfyUI/tools/run_neptune_optimized_batch4.py"),
    ]
    try:
        for rel_local, remote_path in files_to_upload:
            local_path = os.path.join(REPO_DIR, rel_local)
            res = subprocess.run([COLAB_CLI, "--auth=oauth2", "upload", "-s", SESSION, local_path, remote_path], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Failed uploading {rel_local}: {res.stderr}")
            print(f"Uploaded {rel_local} -> {remote_path}", flush=True)

        res = subprocess.run([COLAB_CLI, "--auth=oauth2", "upload", "-s", SESSION, marker_path, "/content/ComfyUI/.local_patched"], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed uploading .local_patched: {res.stderr}")
    finally:
        if marker_path and os.path.exists(marker_path):
            os.unlink(marker_path)


def run_microbench():
    print("=== Step 2b: Running TPU INT8 ConvRot Micro-Benchmark ===", flush=True)
    cmd = [
        COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION,
        "--env", "COMFY_NO_GIT_PULL=1",
        "--env", "COMFY_MICROBENCH=1",
        "-f", "tools/run_neptune_optimized_batch4.py",
        "--timeout", "300"
    ]
    rc, output = stream_cmd(cmd)
    subprocess.run([
        COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION,
        "/content/microbench_result.json", os.path.join(OUT_DIR, "microbench_result.json")
    ], capture_output=True, text=True)
    if "REPLAN_TRIGGER" in output or rc != 0:
        raise RuntimeError(f"Microbench halted execution (rc={rc}). Check output/microbench_result.json")


def execute_batch():
    print("=== Step 3: Executing Batch 4 (Staged 3-Image Prod + Warm Single Check) ===", flush=True)
    cmd = [
        COLAB_CLI, "--auth=oauth2", "exec", "-s", SESSION,
        "--env", "COMFY_NO_GIT_PULL=1",
        "--env", "COMFY_MICROBENCH=0",
    ]
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        cmd += ["--env", f"HF_TOKEN={hf_token}"]
    cmd += ["-f", "tools/run_neptune_optimized_batch4.py", "--timeout", "2400"]
    rc, _ = stream_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"Batch script exited with code {rc}")


def download_artifacts():
    print("=== Step 4: Downloading Output Images, Results JSON & Server Log ===", flush=True)
    prod_files = [
        "neptune25_batch4_01_neptune_brachiosaurus_rearing_00001_.png",
        "neptune25_batch4_02_neptune_source_redundancy_architecture_00001_.png",
        "neptune25_batch4_03_neptune_entropy_deployment_00001_.png",
    ]
    extra_files = [
        "neptune25_batch4_04_neptune_sauropod_herd_warm_check_00001_.png",
    ]
    for f in prod_files + extra_files:
        remote = f"/content/ComfyUI/output/{f}"
        local = os.path.join(OUT_DIR, f)
        print(f"Downloading {f}...", flush=True)
        subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote, local])
        if os.path.exists(local):
            print(f"Downloaded: {local} ({os.path.getsize(local)} bytes)", flush=True)
            shutil.copy(local, os.path.join(ARTIFACT_DIR, f))

    for remote, name, copy_art in [
        ("/content/neptune3_batch4_results.json", "neptune3_batch4_results.json", True),
        ("/content/comfyui.log", "comfyui.log", False),
    ]:
        local = os.path.join(OUT_DIR, name)
        subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", SESSION, remote, local])
        if os.path.exists(local):
            print(f"Downloaded: {local} ({os.path.getsize(local)} bytes)", flush=True)
            if copy_art:
                shutil.copy(local, os.path.join(ARTIFACT_DIR, name))


def stop_vm():
    print("=== Step 5: Stopping Session to Prevent Compute Waste ===", flush=True)
    res = subprocess.run([COLAB_CLI, "--auth=oauth2", "stop", "-s", SESSION], capture_output=True, text=True)
    print(res.stdout, flush=True)
    chk = subprocess.run([COLAB_CLI, "--auth=oauth2", "sessions"], capture_output=True, text=True)
    print("Active sessions check:", chk.stdout or chk.stderr, flush=True)


if __name__ == "__main__":
    try:
        provision_session(max_retries=2)
        clone_and_upload_patches()
        execute_batch()
        download_artifacts()
    finally:
        stop_vm()
