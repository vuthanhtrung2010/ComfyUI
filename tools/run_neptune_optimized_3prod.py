#!/usr/bin/env python3
"""Run 3 production Neptune Motion images at 25 steps with CPU CLIP optimization on Colab TPU v5e."""
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
        "name": "neptune_glacier_wall",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "glacier_visual",
        "prompt": "The image is a wide cinematic realistic photograph of a massive blue glacial ice wall towering over a frozen gravel coastline, dramatic overcast sky, mist rising from cracking ice, photorealistic, 8k resolution",
        "negative": "blurry, low quality, distorted, cartoon, watermark",
        "seed": 4201,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "neptune_bone_fires",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "bone_fires",
        "prompt": "The image is a wide realistic photograph of an ancient stone-lined circular hearth burning brightly inside a dark earthen shelter, glowing embers, wisps of grey smoke drifting upward into shadows, dramatic firelight",
        "negative": "blurry, low quality, distorted, cartoon, watermark",
        "seed": 4202,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "neptune_bone_needles",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "needle_image",
        "prompt": "The image is a wide realistic photograph of three polished prehistoric bone sewing needles resting on a weathered piece of reindeer hide, macro photography, delicate drilled eyelets, sharp tapered points",
        "negative": "blurry, low quality, distorted, modern tools, watermark",
        "seed": 4203,
        "steps": 25,
        "width": 1024,
        "height": 640
    }
]

def build_workflow(p):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-2.1-UC-Q4_K_M.gguf"}},
        # OPTIMIZATION: device="cpu" keeps CLIP on 50GB host RAM, freeing TPU HBM completely for DiT + VAE!
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "cpu"}},
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"neptune25_opt_{p['id']:02d}_{p['name']}"}}
    }

def run_benchmark():
    print(f"\n=================================================================")
    print(f"=== Running Neptune Motion 3-Image Benchmark (25 Steps, Optimized) ===")
    print(f"=================================================================\n")
    results = []
    overall_start = time.time()

    for idx, p in enumerate(PROD_PROMPTS, 1):
        print(f"\n--- [PROD IMAGE {idx}/3: {p['name']}] (Video: {p['video']}, Scene: {p['scene']}) ---")
        print(f"Prompt: {p['prompt'][:100]}...")
        workflow = build_workflow(p)

        data = json.dumps({"prompt": workflow, "client_id": f"neptune3_{p['id']}"}).encode("utf-8")
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
    print(f"\n=================================================================")
    print(f"Neptune Motion 3-Image 25-Step Benchmark Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"Average Time Per Image (25 steps): {avg_time:.1f}s")
    print(f"=================================================================")

    report = {
        "benchmark_name": "Neptune Motion 3 Prod Images (25 steps, CPU CLIP optimization)",
        "total_images": len(PROD_PROMPTS),
        "steps_per_image": 25,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(avg_time, 2),
        "results": results
    }
    with open("/content/neptune3_opt_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune3_opt_results.json")
    return report

if __name__ == "__main__":
    run_benchmark()
