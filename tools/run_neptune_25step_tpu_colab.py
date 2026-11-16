#!/usr/bin/env python3
"""Run 10 images with Neptune Motion prompts at default 25 diffusion steps on Colab TPU v5e."""
import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess

COMFY_DIR = "/content/ComfyUI"

NEPTUNE_PROMPTS = [
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
    },
    {
        "id": 4,
        "name": "neptune_mammoth_hut",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "mammoth_hut_scene",
        "prompt": "The image is a wide cinematic realistic photograph of a prehistoric mammoth bone dwelling standing on a snow-swept Ukrainian steppe under twilight stars, intricate interlocking curved tusks forming the dome entrance",
        "negative": "blurry, low quality, distorted, cartoon, modern building",
        "seed": 4204,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 5,
        "name": "neptune_hunting_steppe",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "hunting_steppe",
        "prompt": "The image is a wide realistic photograph of a sunlit glacial grassland steppe in late autumn, dotted with grazing prehistoric wild horses and reindeer herds in the distant haze, golden hour rim lighting",
        "negative": "blurry, low quality, fences, roads, modern cars, low resolution",
        "seed": 4205,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 6,
        "name": "neptune_cave_art",
        "video": "15-minutes-length-about-why-human-surviv",
        "scene": "cave_art_visual",
        "prompt": "The image is a wide cinematic realistic photograph of an ancient Upper Paleolithic cave wall covered in charcoal and red ochre paintings of running bison and galloping horses, flickering torchlight creating deep shadows on limestone",
        "negative": "blurry, modern museum frame, bright electrical light, low resolution",
        "seed": 4206,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 7,
        "name": "neptune_shannon_data_transmission",
        "video": "lecture-01-ly-thuyet-thong-tin-infor",
        "scene": "intro",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a data transmission system spanning across open space. On the left side of the frame, a simple geometric transmitter box rests neatly on a horizontal baseline, with a slender vertical antenna projecting upwards from its top corner. In the centre of the frame, three parallel horizontal dashed lines depict a communication pathway stretching steadily across the middle, accompanied by four small square packet outlines floating sequentially between the stations. In the upper-right area, faint curved signal arcs fan gently outwards to indicate wave propagation into the surrounding area. On the right side of the frame, an identical receiver terminal stands quietly, aligned with the incoming pathway to capture the arriving signals without distortion. The lines are clean dark grey pencil lines with subtle blue accents on the moving square packets; no shading, no texture. The overall composition is balanced, calm, and simple, leaving generous margins of empty paper around all objects.",
        "negative": "photorealistic, 3d render, glossy, dark background, blurry",
        "seed": 4207,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 8,
        "name": "neptune_entropy_funnel",
        "video": "lecture-01-ly-thuyet-thong-tin-infor",
        "scene": "entropy_definition",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a wide landscape diagram showing data compression limits. Across the left side of the frame, a large diffuse cloud of diverse geometric shapes gathers loosely together, consisting of eight assorted squares, diamonds, and pentagons scattered in natural groups. In the centre of the frame, two long horizontal funnel lines taper gradually from left to right, channeling the scattered cluster into a compact, orderly stream. A single sharp horizontal dashed boundary line runs beneath the funnel, marking the theoretical baseline that cannot be crossed. On the right side of the frame, a tightly packed column of four uniform slender blocks stands firmly on the foundation line, illustrating the maximum possible reduction of the original content. The lines are clean dark grey pencil lines with subtle blue accents on the lower boundary line; no shading, no texture. The overall composition is balanced, calm, and simple, leaving generous margins of empty paper.",
        "negative": "photorealistic, 3d render, glossy, dark background, blurry",
        "seed": 4208,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 9,
        "name": "neptune_viking_longship",
        "video": "why-vikings-are-strong",
        "scene": "scene_01",
        "prompt": "The image is a cinematic historical photograph of a wooden Norse Viking longship cutting through icy fjord waters at dawn, dragon figurehead carved from dark oak, mist clinging to jagged mountain slopes, dramatic volumetric morning sunlight",
        "negative": "blurry, low quality, cartoon, modern boats, watermark",
        "seed": 4209,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 10,
        "name": "neptune_2050_smart_home",
        "video": "bot-30-un-dia-en-2050",
        "scene": "scene_01",
        "prompt": "The image is a futuristic realistic architectural scene of a sunlit living room in the year 2050, sleek transparent holographic interfaces floating seamlessly near floor-to-ceiling smart glass windows overlooking a green sustainable mega-city",
        "negative": "blurry, low quality, dark, cluttered, low resolution",
        "seed": 4210,
        "steps": 25,
        "width": 1024,
        "height": 640
    }
]

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
            "steps": p["steps"],  # Exactly 25 steps per user request
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["4", 0],
            "negative": ["4", 1],
            "latent_image": ["5", 0],
            "denoise": 1.0
        }},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"neptune25_{p['id']:02d}_{p['name']}"}}
    }

def run_neptune_25step_batch():
    print(f"\n=======================================================")
    print(f"=== Running Neptune Motion Batch: 10 Images @ 25 Steps ===")
    print(f"=======================================================\n")
    results = []
    overall_start = time.time()

    for idx, p in enumerate(NEPTUNE_PROMPTS, 1):
        print(f"\n--- [IMAGE {idx}/10: {p['name']}] (Video: {p['video']}, Scene: {p['scene']}, Steps: {p['steps']}) ---")
        print(f"Prompt: {p['prompt'][:100]}...")
        workflow = build_workflow(p)

        data = json.dumps({"prompt": workflow, "client_id": f"neptune25_{p['id']}"}).encode("utf-8")
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
                        print(f"--> [SUCCESS] Completed in {duration:.1f}s ({p['steps']} steps). Generated: {output_files}")
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
    avg_time = total_time / len(NEPTUNE_PROMPTS) if NEPTUNE_PROMPTS else 0
    print(f"\n=======================================================")
    print(f"Neptune Motion 25-Step Batch Completed in {total_time:.1f}s ({total_time / 60:.2f} mins)!")
    print(f"Average Time Per Image (25 steps): {avg_time:.1f}s")
    print(f"=======================================================")

    report = {
        "benchmark_name": "Neptune Motion Production Prompts (25 steps)",
        "total_images": len(NEPTUNE_PROMPTS),
        "steps_per_image": 25,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "total_duration_seconds": round(total_time, 2),
        "average_duration_seconds": round(avg_time, 2),
        "results": results
    }
    with open("/content/neptune25_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune25_results.json")
    return report

if __name__ == "__main__":
    run_neptune_25step_batch()
