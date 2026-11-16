#!/usr/bin/env python3
"""Run a batch of 10 different images using Qwen-Image 2.1 on Colab TPU v5e."""
import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess

COMFY_DIR = "/content/ComfyUI"

PROMPTS = [
    {
        "id": 1,
        "name": "shannon_entropy",
        "prompt": "An artistic educational diagram explaining Shannon information theory and entropy formula, detailed whiteboard illustration, sharp clean lines",
        "negative": "blurry, low quality, distorted, watermark",
        "seed": 1001,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "tokyo_rain_cafe",
        "prompt": "A cozy coffee shop on a rainy afternoon in Tokyo, warm amber interior lighting, rain streaks on large glass window, photorealistic cinematic lighting",
        "negative": "blurry, dark, overexposed, low quality, cartoon",
        "seed": 1002,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "steampunk_clockwork",
        "prompt": "An intricate steampunk clockwork mechanism with polished brass gears, glowing vacuum tubes, and copper pressure gauges, macro photography, 8k resolution",
        "negative": "blurry, flat, plastic, low resolution, watermark",
        "seed": 1003,
        "width": 1024,
        "height": 640
    },
    {
        "id": 4,
        "name": "himalayan_snow_leopard",
        "prompt": "A majestic snow leopard resting on a snowy rocky cliff in the Himalayas, golden hour rim lighting, hyper-realistic fur detail, national geographic photography",
        "negative": "deformed, blurry, cartoon, low quality, bad anatomy",
        "seed": 1004,
        "width": 1024,
        "height": 640
    },
    {
        "id": 5,
        "name": "solarpunk_city",
        "prompt": "Futuristic solarpunk metropolis with lush vertical gardens on skyscrapers, sleek elevated maglev trains, transparent sky bridges, sunny clear blue sky, architectural visualization",
        "negative": "dystopian, smog, dark, gloomy, blurry, low quality",
        "seed": 1005,
        "width": 1024,
        "height": 640
    },
    {
        "id": 6,
        "name": "retro_scifi_mars",
        "prompt": "A vintage 1970s science fiction paperback book cover illustration of a retro-futuristic astronaut discovering a glowing black monolith on the red dunes of Mars, pulp scifi art style",
        "negative": "modern photo, 3d render, blurry, distorted text, low resolution",
        "seed": 1006,
        "width": 1024,
        "height": 640
    },
    {
        "id": 7,
        "name": "cherry_blossom_watercolor",
        "prompt": "Delicate watercolor painting of blooming pink cherry blossoms over a tranquil koi pond with ripples, soft pastel tones, fluid expressive brush strokes, traditional Japanese sumi-e aesthetic",
        "negative": "photorealistic, sharp digital render, high contrast, oversaturated, muddy",
        "seed": 1007,
        "width": 1024,
        "height": 640
    },
    {
        "id": 8,
        "name": "violin_craftsman",
        "prompt": "A cinematic close-up studio portrait of an elderly artisan violin maker carving spruce wood in a rustic workshop, sunbeam with floating dust motes, dramatic Rembrandt lighting",
        "negative": "blurry, plastic skin, distorted hands, bad eyes, low quality",
        "seed": 1008,
        "width": 1024,
        "height": 640
    },
    {
        "id": 9,
        "name": "bioluminescent_deep_sea",
        "prompt": "An alien deep underwater coral trench teeming with translucent bioluminescent jellyfish emitting cyan and violet neon light, deep marine caustics, cinematic wide shot",
        "negative": "blurry, noisy, grainy, murky, overexposed, low detail",
        "seed": 1009,
        "width": 1024,
        "height": 640
    },
    {
        "id": 10,
        "name": "isometric_potion_shop",
        "prompt": "A stylized isometric 3D miniature diorama of a whimsical wizard potion shop, colorful glowing potion flasks on wooden shelves, bubbling magical cauldron, warm cozy ambient glow, octane render 3D art",
        "negative": "flat 2D, real photo, dark, murky, blurry, low resolution",
        "seed": 1010,
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
    
    loader_path = os.path.join(gguf_node_dir, "loader.py")
    if os.path.exists(loader_path):
        with open(loader_path, "r") as f:
            code = f.read()
        code = code.replace('"qwen_image"', '"qwen_image", "qwen_image21"')
        code = code.replace('raise ValueError(f"This model is not currently supported - ({e})")', 'arch_str = "qwen_image"')
        with open(loader_path, "w") as f:
            f.write(code)
        print("ComfyUI-GGUF patches applied successfully.")

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
            "steps": 8,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"batch_{p['id']:02d}_{p['name']}"}}
    }

def run_batch():
    print(f"\n=== Step 4: Executing Batch of {len(PROMPTS)} Images ===")
    results = []
    overall_start = time.time()

    for idx, p in enumerate(PROMPTS, 1):
        print(f"\n--- [IMAGE {idx}/{len(PROMPTS)}] {p['name']} ---")
        print(f"Prompt: {p['prompt']}")
        workflow = build_workflow(p)

        data = json.dumps({"prompt": workflow, "client_id": f"batch_{p['id']}"}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req)
        prompt_res = json.loads(resp.read().decode())
        prompt_id = prompt_res["prompt_id"]

        t_start = time.time()
        completed = False
        output_files = []

        while time.time() - t_start < 400:
            time.sleep(2)
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
                        print(f"--> Completed in {duration:.1f}s! Generated: {output_files}")
                        results.append({
                            "id": p["id"],
                            "name": p["name"],
                            "prompt": p["prompt"],
                            "duration_seconds": round(duration, 2),
                            "output_files": output_files,
                            "status": "success"
                        })
                        completed = True
                        break
                    elif status.get("status_str") == "error":
                        duration = time.time() - t_start
                        print(f"--> ERROR after {duration:.1f}s: {status}")
                        results.append({
                            "id": p["id"],
                            "name": p["name"],
                            "prompt": p["prompt"],
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
            print(f"--> TIMEOUT after {duration:.1f}s")
            results.append({
                "id": p["id"],
                "name": p["name"],
                "duration_seconds": round(duration, 2),
                "status": "timeout"
            })

    total_time = time.time() - overall_start
    print(f"\n==========================================")
    print(f"Batch Execution Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"==========================================")
    
    report = {
        "total_images": len(PROMPTS),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(total_time / len(PROMPTS), 2) if PROMPTS else 0,
        "results": results
    }
    with open("/content/batch_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/batch_results.json")
    return report

if __name__ == "__main__":
    setup_environment()
    download_models()
    start_server()
    report = run_batch()
