#!/usr/bin/env python3
"""Run full Qwen-Image 2.1 text-to-image pipeline on Colab TPU v5e."""
import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess

COMFY_DIR = "/content/ComfyUI"

def run_cmd(cmd, check=True):
    print(f"[RUN] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res

def setup_environment():
    print("=== Step 1: Setting up ComfyUI Repository and Dependencies ===")
    os.makedirs(COMFY_DIR, exist_ok=True)
    os.chdir(COMFY_DIR)

    # 1. Clone or update branch
    if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
        run_cmd(["git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
                 "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR])
    else:
        run_cmd(["git", "fetch", "origin", "feat/tpu-multi-core-support"])
        run_cmd(["git", "checkout", "feat/tpu-multi-core-support"])
        run_cmd(["git", "pull"])

    # 2. Install requirements & GGUF
    print("Installing requirements & gguf...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "gguf"])

    # 3. Setup ComfyUI-GGUF custom node
    gguf_node_dir = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-GGUF")
    if not os.path.exists(gguf_node_dir):
        print("Cloning ComfyUI-GGUF...")
        run_cmd(["git", "clone", "https://github.com/city96/ComfyUI-GGUF.git", gguf_node_dir])
    
    # Apply both Neptune Motion Qwen-Image 2.1 loader patches and TPU weight caching optimizations
    patch_script = os.path.join(COMFY_DIR, "tools", "patch_comfyui_gguf.py")
    if os.path.exists(patch_script):
        run_cmd([sys.executable, patch_script, gguf_node_dir])

def download_models():
    print("=== Step 2: Downloading Qwen-Image 2.1 Model Checkpoints ===")
    models = [
        (
            "models/unet/qwen-image-2.1-UC-Q4_K_M.gguf",
            "https://huggingface.co/abenzerps/Qwen-Image-2.1-Turbo-Quantized/resolve/main/qwen-image-2.1-turbo-Q4_K_M.gguf"
        ),
        (
            "models/clip/qwen3vl_8b_int8_convrot.safetensors",
            "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
        ),
        (
            "models/vae/qwen_image_2.1_vae_bf16.safetensors",
            "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"
        )
    ]
    for rel_path, url in models:
        dest = os.path.join(COMFY_DIR, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[SKIP] Model already present: {rel_path} ({size_mb:.1f} MB)")
            continue
        print(f"[DOWNLOAD] Downloading {rel_path} from {url}...")
        # Use aria2c or curl for resilient download
        subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", dest, url], check=True)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[DOWNLOAD COMPLETE] {rel_path}: {size_mb:.1f} MB")

def start_server():
    print("=== Step 3: Launching ComfyUI Server on TPU Core 0 ===")
    os.chdir(COMFY_DIR)
    # Stop existing instance if any
    subprocess.run(["pkill", "-f", "main.py"])
    time.sleep(1)

    log_file = open("/content/comfyui.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--tpu-device", "0", "--port", "8188", "--listen", "127.0.0.1"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=COMFY_DIR
    )
    print(f"ComfyUI server started (PID: {proc.pid})")

    # Wait for server to come online
    for _ in range(45):
        time.sleep(1)
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            if req.status == 200:
                print("Server online and responding!")
                stats = json.loads(req.read().decode())
                print("Detected Devices:", [d["name"] for d in stats.get("devices", [])])
                return proc
        except Exception:
            pass
    raise RuntimeError("Server failed to respond within 45s. Check /content/comfyui.log")

def run_workflow():
    print("=== Step 4: Dispatching Qwen-Image 2.1 Workflow ===")
    workflow = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-2.1-UC-Q4_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0],
            "prompt": "An artistic educational diagram explaining Shannon information theory and entropy, detailed whiteboard drawing",
            "negative_prompt": "blurry, low quality, distorted, watermark",
            "resolution": 1024
        }},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 640, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "seed": 42,
            "steps": 8,  # Turbo model reaches convergence in 4-8 steps
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "explainer"}}
    }

    data = json.dumps({"prompt": workflow, "client_id": "qwen_tpu_test"}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    prompt_res = json.loads(resp.read().decode())
    prompt_id = prompt_res["prompt_id"]
    print(f"Workflow dispatched! Prompt ID: {prompt_id}")

    # Monitor execution
    start_time = time.time()
    last_log_print = start_time
    while True:
        time.sleep(3)
        elapsed = time.time() - start_time
        try:
            h_req = urllib.request.urlopen("http://127.0.0.1:8188/history")
            history = json.loads(h_req.read().decode())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                completed = status.get("completed", False)
                if completed:
                    print(f"\n[SUCCESS] Image Generation Completed in {elapsed:.1f}s!")
                    outputs = entry.get("outputs", {})
                    print("Generated Outputs:", json.dumps(outputs, indent=2))
                    return outputs
                elif status.get("status_str") == "error":
                    print(f"\n[FAILURE] Workflow execution failed:\n{json.dumps(status, indent=2)}")
                    raise RuntimeError("Workflow execution error")
        except urllib.error.URLError:
            pass

        # Print progress heartbeat
        if time.time() - last_log_print > 15:
            last_log_print = time.time()
            print(f"Still running... elapsed: {elapsed:.0f}s")
            if os.path.exists("/content/comfyui.log"):
                with open("/content/comfyui.log", "r") as f:
                    lines = f.readlines()
                    print("Recent log:", lines[-1].strip() if lines else "")

        if elapsed > 600:
            raise TimeoutError("Workflow execution exceeded 10 minutes timeout.")

if __name__ == "__main__":
    setup_environment()
    download_models()
    start_server()
    outputs = run_workflow()
    print("ALL DONE! Visual image confirmed.")
