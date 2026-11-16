#!/usr/bin/env python3
"""Run 3 new Neptune Motion production images at 25 steps with all single-core optimizations."""
import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess

COMFY_DIR = "/content/ComfyUI"

PROD_PROMPTS = [
    {
        "id": 1,
        "name": "neptune_viking_ocean_voyage",
        "video": "why-vikings-are-strong",
        "scene": "ocean_voyage",
        "prompt": "The image is a wide cinematic realistic photograph of an authentic wooden Viking longship navigating through choppy Arctic waters at sunrise, dark oak hull, striped woolen sail billowing, sea spray rising against dramatic distant snowy mountains, 8k resolution",
        "negative": "blurry, low quality, distorted, cartoon, modern ships, watermark",
        "seed": 5101,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "neptune_scandinavian_forest",
        "video": "why-vikings-are-strong",
        "scene": "scandinavian_forest",
        "prompt": "The image is a wide cinematic realistic photograph of an ancient Scandinavian pine forest at misty dawn, dense tall Scots pines rising into thick morning fog, golden light rays filtering through pine needles, mossy boulders on the forest floor, hyper-realistic",
        "negative": "blurry, low quality, distorted, cartoon, modern roads, watermark",
        "seed": 5102,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "neptune_mammoth_bone_hut",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "mammoth_hut_scene",
        "prompt": "The image is a wide cinematic realistic photograph of a prehistoric mammoth bone dwelling standing on a snow-swept Ukrainian steppe under twilight stars, intricate interlocking curved tusks forming the dome entrance, wind-carved snow drifts, atmospheric cinematic lighting",
        "negative": "blurry, low quality, distorted, cartoon, modern building, watermark",
        "seed": 5103,
        "steps": 25,
        "width": 1024,
        "height": 640
    }
]

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

    if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
        run_cmd(["git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
                 "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR])
    else:
        run_cmd(["git", "fetch", "origin", "feat/tpu-multi-core-support"])
        run_cmd(["git", "checkout", "feat/tpu-multi-core-support"])
        run_cmd(["git", "pull"])

    print("Installing requirements & gguf...")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "gguf"])

    gguf_node_dir = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-GGUF")
    if not os.path.exists(gguf_node_dir):
        print("Cloning ComfyUI-GGUF...")
        run_cmd(["git", "clone", "https://github.com/city96/ComfyUI-GGUF.git", gguf_node_dir])
    
    # Apply TPU optimizations and Qwen-Image 2.1 support
    patch_script = os.path.join(COMFY_DIR, "tools", "patch_comfyui_gguf.py")
    if os.path.exists(patch_script):
        run_cmd([sys.executable, patch_script, gguf_node_dir])

def download_models():
    print("=== Step 2: Verifying Qwen-Image 2.1 Model Checkpoints ===")
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
        if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"[CACHE HIT] {rel_path} ({size_mb:.1f} MB)")
            continue
        print(f"[DOWNLOAD] Downloading {rel_path}...")
        subprocess.run(["curl", "-L", "-C", "-", "--retry", "5", "-o", dest, url], check=True)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[DOWNLOAD COMPLETE] {rel_path}: {size_mb:.1f} MB")

def start_server():
    print("=== Step 3: Launching ComfyUI Server on TPU Core 0 ===")
    os.chdir(COMFY_DIR)
    subprocess.run(["pkill", "-f", "main.py"])
    time.sleep(1)

    log_file = open("/content/comfyui.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--tpu-device", "0", "--port", "8188", "--listen", "127.0.0.1"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=COMFY_DIR
    )
    print(f"ComfyUI server started with PID {proc.pid}")

    for _ in range(45):
        time.sleep(1)
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            if req.status == 200:
                stats = json.loads(req.read().decode())
                devices = [d["name"] for d in stats.get("devices", [])]
                print(f"SUCCESS: Server is ONLINE! Detected devices: {devices}")
                return proc
        except Exception:
            pass
    raise RuntimeError("Server failed to respond within 45s.")

def build_workflow(p):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-2.1-UC-Q4_K_M.gguf"}},
        # CLIP loaded to TPU (default) with automatic BF16 cast & dynamic offload
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0],
            "prompt": p["prompt"],
            "negative_prompt": p["negative"],
            "resolution": 1024
        }},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0],
            "seed": p["seed"],
            "steps": p["steps"],  # 25 steps per user request
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"neptune25_batch2_{p['id']:02d}_{p['name']}"}}
    }

def run_benchmark():
    print(f"\n=========================================================================")
    print(f"=== Running 3 NEW Neptune Motion Production Images (25 Steps, Optimized) ===")
    print(f"=========================================================================\n")
    results = []
    overall_start = time.time()

    for idx, p in enumerate(PROD_PROMPTS, 1):
        print(f"\n--- [PROD IMAGE {idx}/3: {p['name']}] (Video: {p['video']}, Scene: {p['scene']}, Steps: {p['steps']}) ---")
        print(f"Prompt: {p['prompt'][:100]}...")
        workflow = build_workflow(p)

        data = json.dumps({"prompt": workflow, "client_id": f"batch2_{p['id']}"}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        prompt_res = json.loads(resp.read().decode())
        prompt_id = prompt_res["prompt_id"]

        t_start = time.time()
        completed = False
        output_files = []

        while time.time() - t_start < 600:
            time.sleep(3)
            try:
                h_req = urllib.request.urlopen("http://127.0.0.1:8188/history")
                history = json.loads(h_req.read().decode())
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        duration = time.time() - t_start
                        outputs = entry.get("outputs", {})
                        if "8" in outputs and "images" in outputs["8"]:
                            for img in outputs["8"]["images"]:
                                output_files.append(img["filename"])
                        print(f"--> [SUCCESS] Completed in {duration:.1f}s (25 steps). Generated: {output_files}")
                        results.append({
                            "id": p["id"],
                            "name": p["name"],
                            "video": p["video"],
                            "scene": p["scene"],
                            "steps": p["steps"],
                            "prompt": p["prompt"],
                            "duration_seconds": round(duration, 2),
                            "output_files": output_files,
                            "status": "success"
                        })
                        completed = True
                        break
                    elif status.get("status_str") == "error":
                        duration = time.time() - t_start
                        print(f"--> [ERROR] Failed after {duration:.1f}s: {status}")
                        results.append({
                            "id": p["id"],
                            "name": p["name"],
                            "steps": p["steps"],
                            "duration_seconds": round(duration, 2),
                            "error": str(status),
                            "status": "error"
                        })
                        completed = True
                        break
            except Exception:
                pass

        if not completed:
            duration = time.time() - t_start
            print(f"--> [TIMEOUT] Timed out after {duration:.1f}s")
            results.append({
                "id": p["id"],
                "name": p["name"],
                "steps": p["steps"],
                "duration_seconds": round(duration, 2),
                "status": "timeout"
            })

    total_time = time.time() - overall_start
    avg_time = total_time / len(PROD_PROMPTS) if PROD_PROMPTS else 0
    print(f"\n=========================================================================")
    print(f"3 New Images 25-Step Benchmark Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"Average Time Per Image (25 steps): {avg_time:.1f}s")
    print(f"=========================================================================")

    report = {
        "benchmark_name": "Neptune Motion Batch 2 (3 new scenes, 25 steps, fully optimized)",
        "total_images": len(PROD_PROMPTS),
        "steps_per_image": 25,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(avg_time, 2),
        "results": results
    }
    with open("/content/neptune3_batch2_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune3_batch2_results.json")
    return report

if __name__ == "__main__":
    setup_environment()
    download_models()
    start_server()
    run_benchmark()
