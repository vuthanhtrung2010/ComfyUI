#!/usr/bin/env python3
"""Run 3 new Neptune Motion production images at 25 steps with native INT8 ConvRot DiT, CLIP on TPU, and dequant caching."""
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
        "name": "neptune_brachiosaurus_rearing",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "chewing_cost",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a mounted sauropod skeleton rearing toward the ceiling of an exhibition hall. In the centre of the frame, a towering sauropod skeleton rises vertically with its forelimbs much taller than its hind limbs, drawn with precise pencil outlines showing the ribcage, shoulder blades, and an upward-curving neck. On the left side of the frame, the colossal front limbs stand planted on the museum floor like massive upright pillars, surrounded by empty space. On the right side of the frame, the long tail slopes gently downward to balance the elevated neck, extending toward the right edge. Across the upper third, the small delicate skull tilts forward at the apex of the arched neck near the top margin. Across the lower third, a clean horizontal floor line anchors the upright legs with faint reflection marks. The lines are crisp dark grey pencil lines with subtle terracotta touches along the spine and no shading or background clutter. The overall composition is monumental, balanced, and serene, surrounded by generous margins of blank warm paper.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, modern background, watermark",
        "seed": 5301,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "neptune_source_redundancy_architecture",
        "video": "ly-thuyet-thong-tin",
        "scene": "scene-04-architecture",
        "prompt": "A detailed educational whiteboard sketch illustrating Source Redundancy on a clean light ivory background. Hand-drawn dark charcoal line work depicts computational blocks, vector coordinates, and transformation arrows. Each block shows mathematical symbols clearly rendered with technical accuracy and clean hatching. Flow lines connect the initial stage to intermediate transformations and output variables. Subtle watercolor accents in muted navy and amber highlight active gradients and data paths, creating an elegant and readable visual reference.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, 3d render, watermark",
        "seed": 5302,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "neptune_entropy_deployment",
        "video": "ly-thuyet-thong-tin",
        "scene": "scene-09-deployment-sketch",
        "prompt": "A precise architectural drawing of a production deployment showing Information Entropy embedded within an engineering workflow. Crisp technical line art shows input sensors, preprocessing units, computation engines, and output monitors on a warm cream canvas. Labeled annotations highlight operational latency and throughput metrics. Delicate cross-hatching renders mechanical casings and circuit components with draftsmanship precision and balanced visual composition across the layout.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, 3d render, watermark",
        "seed": 5303,
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

def download_models():
    print("=== Step 2: Verifying Qwen-Image 2.1 Native INT8 Checkpoints ===")
    models = [
        (
            "models/unet/qwen-image-2.1-turbo-int8_convrot.safetensors",
            "https://huggingface.co/abenzerps/Qwen-Image-2.1-Turbo-Quantized/resolve/main/qwen-image-2.1-turbo-int8_convrot.safetensors"
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
        # Native UNETLoader for INT8 ConvRot DiT
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen-image-2.1-turbo-int8_convrot.safetensors", "weight_dtype": "default"}},
        # Native CLIP on TPU in BF16
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"neptune25_batch4_{p['id']:02d}_{p['name']}"}}
    }

def run_benchmark():
    print("\n=========================================================================")
    print("=== Running 3 NEW Neptune Motion Production Images (25 Steps, NATIVE INT8 DiT) ===")
    print("=========================================================================\n")

    results = []
    overall_start = time.time()

    for idx, p in enumerate(PROD_PROMPTS):
        print(f"\n--- [PROD IMAGE {idx+1}/{len(PROD_PROMPTS)}: {p['name']}] (Video: {p['video']}, Scene: {p['scene']}, Steps: {p['steps']}) ---")
        print(f"Prompt: {p['prompt'][:90]}...")
        workflow = build_workflow(p)
        payload = json.dumps({"prompt": workflow}).encode("utf-8")

        req = urllib.request.Request(
            "http://127.0.0.1:8188/prompt",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        t_start = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                prompt_id = data.get("prompt_id")
                print(f"Queued prompt_id: {prompt_id}")
        except Exception as e:
            print(f"[ERROR] Failed to queue prompt: {e}")
            results.append({"id": p["id"], "name": p["name"], "error": str(e), "status": "failed"})
            continue

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
                        print(f"--> [SUCCESS] Completed in {duration:.1f}s (25 steps, NATIVE INT8 DiT). Generated: {output_files}")
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
    print(f"3 New Images (NATIVE INT8 DiT) Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"Average Time Per Image (25 steps): {avg_time:.1f}s")
    print(f"=========================================================================")

    report = {
        "benchmark_name": "Neptune Motion Batch 4 (3 new scenes, 25 steps, NATIVE INT8 DiT + CLIP ON TPU + BF16)",
        "total_images": len(PROD_PROMPTS),
        "steps_per_image": 25,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(avg_time, 2),
        "results": results
    }
    with open("/content/neptune3_batch4_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune3_batch4_results.json")
    return report

if __name__ == "__main__":
    setup_environment()
    download_models()
    start_server()
    run_benchmark()
