#!/usr/bin/env python3
"""Run 3 new Neptune Motion production images at 25 steps with CLIP on TPU and all single-core optimizations."""
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
        "name": "neptune_predator_defense",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "predator_defense",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a predatory theropod skeleton standing in defensive posture beside the colossal leg of a sauropod. On the left side of the frame, the complete skeleton of an Allosaurus crouches with arched neck, gaping tooth-lined jaws, and curved hand claws raised in warning. In the centre of the frame, wide empty beige paper creates a vast physical gulf between the predator and the herbivore. On the right side of the frame, only the lower portion of a single sauropod hind leg appears, drawn as a monumental column of thick bone that dwarfs the entire carnivorous dinosaur. Across the lower third, a simple horizontal ground plane with a few scattered fossil stones grounds both skeletons. The lines are sharp dark grey pencil lines with subtle crimson accents along the predator claw tips and amber touches on the giant leg bone, with no shading. The overall composition is dramatic, stark, and balanced, surrounded by generous empty space.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, modern background, watermark",
        "seed": 5201,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "neptune_skeletal_truss",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "skeletal_truss",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a fossilized sauropod neck vertebra showing hollow internal pneumatic cavities. In the centre of the frame, a single massive cervical vertebra is drawn in side profile, displaying a sturdy central centrum flanked by expansive hollow chambers. On the left side of the frame, deep smooth depressions known as pleurocoels indent the bone wall, drawn with elegant curved contour lines that reveal internal air chambers. On the right side of the frame, delicate bony struts and thin buttresses form an internal truss network that supports the neural arch without solid weight. Across the upper third, the tall neural spine rises with winged transverse processes extending into open paper. The lines are clean dark grey pencil lines with gentle warm amber highlights tracing the internal bony support struts, free of texture or heavy shading. Across the lower third, the smooth articular surface rests quietly above the empty margin. The overall composition is balanced, precise, and sculptural, surrounded by vast calm beige paper.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, modern roads, watermark",
        "seed": 5202,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "neptune_argentinosaurus_fossil",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "fossil_legacy",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a full life-sized reconstruction of an Argentinosaurus skeleton spanning an expansive hall. In the centre of the frame, the colossal ribcage and pelvis bridge two pairs of vertical pillar legs, drawn with elegant continuous pencil contours. On the left side of the frame, the immense tail stretches horizontally into the distance, tapering gracefully into twenty small vertebrae near the left margin. On the right side of the frame, the long neck angles upward at forty degrees, elevating the diminutive skull toward the upper right corner. Across the lower third, three tiny human silhouettes stand beneath the belly to establish the gigantic scale of the fossil skeleton. The lines are clean dark grey pencil lines with subtle golden-brown touches highlighting the massive dorsal spines, completely free of shading, cross-hatching, or background walls. The overall composition is majestic, grand, and uncluttered, floating effortlessly across a wide field of warm beige paper.",
        "negative": "blurry, low quality, distorted, cartoon, photorealistic, modern building, watermark",
        "seed": 5203,
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
        # CLIP on TPU (default device) with BF16 resolution and dynamic HBM management
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"neptune25_batch3_{p['id']:02d}_{p['name']}"}}
    }

def run_benchmark():
    print("\n=========================================================================")
    print("=== Running 3 NEW Neptune Motion Production Images (25 Steps, CLIP ON TPU) ===")
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
                        print(f"--> [SUCCESS] Completed in {duration:.1f}s (25 steps, CLIP on TPU). Generated: {output_files}")
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
    print(f"3 New Images (CLIP ON TPU) Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"Average Time Per Image (25 steps): {avg_time:.1f}s")
    print(f"=========================================================================")

    report = {
        "benchmark_name": "Neptune Motion Batch 3 (3 new scenes, 25 steps, CLIP ON TPU + BF16 + GGUF cached)",
        "total_images": len(PROD_PROMPTS),
        "steps_per_image": 25,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(avg_time, 2),
        "results": results
    }
    with open("/content/neptune3_batch3_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune3_batch3_results.json")
    return report

if __name__ == "__main__":
    setup_environment()
    download_models()
    start_server()
    run_benchmark()
