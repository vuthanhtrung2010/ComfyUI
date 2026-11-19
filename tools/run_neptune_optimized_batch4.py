#!/usr/bin/env python3
"""Run 3 new Neptune Motion production images at 25 steps with native Uncensored INT8 ConvRot DiT, CLIP on TPU, W8A16 fused ConvRot linear, FlashAttention memory accounting, 64-token bucket padding, and staged batch DAG."""
import argparse
import asyncio
import dataclasses
import os
import re
import shutil
import sys
import threading
import time
import json
import uuid
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

WARM_SINGLE_PROMPT = {
    "id": 4,
    "name": "neptune_sauropod_herd_warm_check",
    "video": "why-did-sauropod-dinosaurs-grow-so-huge",
    "scene": "herd_scale",
    "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a towering sauropod skeleton herd striding across an expansive museum gallery floor. In the centre of the frame, three interconnected fossilized skeletons of varying scale advance from left to right, drawn with sharp dark grey pencil outlines showing elongated cervical vertebrae, columnar limbs, and sweeping tails. On the left side of the frame, a subadult skeleton lowers its neck toward ground level beside scattered fossil stones. On the right side of the frame, the largest adult skeleton raises its head gracefully toward the upper right margin. Across the lower third, a single thin horizontal ground line anchors every pillar-like foot without shadows or heavy texture. The lines are crisp dark grey pencil lines with subtle terracotta touches along the dorsal spines and pale amber accents on the ribcages. The overall composition is clean, serene, and balanced, surrounded by vast open beige paper.",
    "seed": 5304,
    "steps": 25,
    "width": 1024,
    "height": 640
}


def run_cmd(cmd, check=True):
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr}", flush=True)
        raise RuntimeError(f"Command failed: {cmd}")
    return res


def setup_environment(no_git_pull=False):
    print("=== Step 1: Setting up ComfyUI Repository and Dependencies ===", flush=True)
    os.makedirs(COMFY_DIR, exist_ok=True)
    os.chdir(COMFY_DIR)

    if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
        run_cmd(["git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
                 "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR])
    elif not no_git_pull and not os.path.exists(os.path.join(COMFY_DIR, ".local_patched")):
        run_cmd(["git", "fetch", "origin", "feat/tpu-multi-core-support"])
        run_cmd(["git", "checkout", "feat/tpu-multi-core-support"])
        run_cmd(["git", "pull"])
    else:
        print("[SKIP GIT PULL] Keeping local uploaded optimizations intact.", flush=True)

    print("Installing requirements & gguf (preserving preinstalled TPU torch)...", flush=True)
    with open(os.path.join(COMFY_DIR, "requirements.txt"), "r") as f:
        reqs = [
            l.strip() for l in f
            if l.strip() and not l.strip().startswith("#")
            and l.strip().lower().split("==")[0].split(">=")[0].split("<=")[0].strip()
            not in ("torch", "torchvision", "torchaudio")
        ]
    req_tmp = "/tmp/comfy_reqs_nobase_torch.txt"
    with open(req_tmp, "w") as f:
        f.write("\n".join(reqs) + "\n")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-r", req_tmp, "gguf"])


def run_tpu_microbench():
    """Benchmark [1, 2560, 4096] -> [24576, 4096] int8_convrot linear paths on TPU v5e."""
    print("=== TPU Micro-Benchmark: INT8 ConvRot Linear Paths ===", flush=True)
    sys.path.insert(0, COMFY_DIR)
    os.chdir(COMFY_DIR)
    import torch
    import comfy.model_management as mm
    import comfy.quant_ops
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    dev = mm.get_torch_device()
    print(f"Device: {dev}", flush=True)

    torch.manual_seed(0)
    M, K, N = 2560, 4096, 24576
    w_cpu = torch.randn(N, K, dtype=torch.bfloat16)
    qd_cpu, prm_cpu = TensorWiseINT8Layout.quantize(
        w_cpu, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256
    )
    prm_tpu = dataclasses.replace(prm_cpu, scale=prm_cpu.scale.to(dev))
    qt_tpu = QuantizedTensor(qd_cpu.to(dev), "TensorWiseINT8Layout", prm_tpu)
    x_tpu = torch.randn(1, M, K, device=dev, dtype=torch.bfloat16)
    mm.synchronize(dev)

    results = {}

    def bench_fn(name, fn, warmup=2, iters=5):
        out_sample = None
        try:
            for _ in range(warmup):
                out_sample = fn()
                mm.synchronize(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                out_sample = fn()
                mm.synchronize(dev)
            dt_ms = (time.perf_counter() - t0) * 1000.0 / iters
            results[name] = {"ms_per_call": round(dt_ms, 2), "status": "ok"}
            print(f"  [{name}] {dt_ms:.2f} ms/call", flush=True)
            return out_sample
        except Exception as e:
            results[name] = {"status": "error", "error": repr(e)}
            print(f"  [{name}] ERROR: {e}", flush=True)
            return None

    def fp32_scale_w8a16():
        params = qt_tpu._params
        qdata = qt_tpu._qdata
        gs = params.convrot_groupsize
        h = comfy.quant_ops._build_hadamard(gs, device=x_tpu.device, dtype=x_tpu.dtype)
        xr = comfy.quant_ops._rotate_activation(x_tpu, h, gs)
        out = torch.nn.functional.linear(xr, qdata.to(dtype=x_tpu.dtype))
        sc = params.scale.to(device=out.device, dtype=torch.float32).reshape(-1)
        return (out.float() * sc).to(dtype=x_tpu.dtype)

    out_w8a16 = bench_fn("w8a16_bf16_scale", lambda: comfy.quant_ops.int8_tensorwise_linear(x_tpu, qt_tpu))
    out_w8a16_fp32 = bench_fn("w8a16_fp32_scale", fp32_scale_w8a16)
    out_deq = bench_fn("dequantize_plus_linear", lambda: torch.nn.functional.linear(x_tpu, qt_tpu.dequantize()))

    if out_deq is not None and out_w8a16 is not None:
        cos_w8a16 = torch.nn.functional.cosine_similarity(
            out_w8a16.flatten().float(), out_deq.flatten().float(), dim=0
        ).item()
        results["cos_w8a16_bf16_vs_deq"] = round(cos_w8a16, 6)
        print(f"  Cosine sim (W8A16 bf16-scale vs deq+linear): {cos_w8a16:.6f}", flush=True)

    if out_w8a16_fp32 is not None and out_w8a16 is not None:
        cos_bf16_fp32 = torch.nn.functional.cosine_similarity(
            out_w8a16.flatten().float(), out_w8a16_fp32.flatten().float(), dim=0
        ).item()
        results["cos_bf16_vs_fp32_scale"] = round(cos_bf16_fp32, 6)
        print(f"  Cosine sim (bf16-scale vs fp32-scale): {cos_bf16_fp32:.6f}", flush=True)

    # Benchmark SDPA mask variants on TPU (Q=2560, K=2688, H=32, D=128)
    print("=== TPU Micro-Benchmark: SDPA Mask Variants ===", flush=True)
    q = torch.randn(1, 32, 2560, 128, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 32, 2688, 128, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, 32, 2688, 128, device=dev, dtype=torch.bfloat16)
    bool_mask = torch.ones((1, 1, 1, 2688), dtype=torch.bool, device=dev)
    bool_mask[..., 100:128] = False
    bf16_mask = torch.zeros((1, 1, 1, 2688), dtype=torch.bfloat16, device=dev)
    bf16_mask[..., 100:128] = torch.finfo(torch.bfloat16).min / 4
    mm.synchronize(dev)

    bench_fn("sdpa_no_mask", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    out_bool = bench_fn("sdpa_bool_mask", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask))
    out_bf16m = bench_fn("sdpa_bf16_mask", lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bf16_mask))
    if out_bool is not None and out_bf16m is not None:
        cos_mask = torch.nn.functional.cosine_similarity(out_bool.flatten().float(), out_bf16m.flatten().float(), dim=0).item()
        results["cos_sdpa_bool_vs_bf16_mask"] = round(cos_mask, 6)
        print(f"  Cosine sim (bool mask vs bf16 mask): {cos_mask:.6f}", flush=True)

    # Probe torch.compile backends on TPU
    print("=== TPU Micro-Benchmark: torch.compile Linear Probe ===", flush=True)
    results["dynamo_backends"] = list(torch._dynamo.list_backends())
    print(f"  Available dynamo backends: {results['dynamo_backends']}", flush=True)
    for backend in ("openxla", "inductor"):
        if backend in results["dynamo_backends"]:
            try:
                fn_c = torch.compile(lambda a, b: torch.nn.functional.linear(a, b.to(dtype=a.dtype)), backend=backend)
                bench_fn(f"compile_{backend}_w8a16_mm", lambda: fn_c(x_tpu, qt_tpu._qdata), warmup=2, iters=5)
            except Exception as e:
                results[f"compile_{backend}_w8a16_mm"] = {"status": "error", "error": repr(e)}
                print(f"  [compile_{backend}_w8a16_mm] ERROR: {e}", flush=True)

    print("MICROBENCH_RESULT:", json.dumps(results), flush=True)
    with open("/content/microbench_result.json", "w") as f:
        json.dump(results, f, indent=2)
    return 0


def _resolve_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    return tok.strip() if isinstance(tok, str) and tok.strip() else None


def download_models():
    print("=== Step 2: Verifying Qwen-Image 2.1 Uncensored Native INT8 Checkpoints ===", flush=True)
    hf_token = _resolve_hf_token()
    use_aria2 = shutil.which("aria2c") is not None
    print(f"Download config: tool={'aria2c' if use_aria2 else 'curl'}, auth={'HF_TOKEN active' if hf_token else 'anonymous'}", flush=True)

    models = [
        (
            "models/unet/qwen-image-2.1-turbo-int8_convrot.safetensors",
            "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/qwen-image-2.1-turbo-int8_convrot.safetensors"
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
            print(f"[CACHE HIT] {rel_path} ({size_mb:.1f} MB)", flush=True)
            continue
        print(f"[DOWNLOAD] Downloading {rel_path} from {url}...", flush=True)
        t0 = time.time()
        if use_aria2:
            cmd = [
                "aria2c", "-x", "8", "-s", "8", "-k", "1M", "--continue=true",
                "-d", os.path.dirname(dest), "-o", os.path.basename(dest),
            ]
            if hf_token:
                cmd += ["--header", f"Authorization: Bearer {hf_token}"]
            cmd.append(url)
        else:
            cmd = ["curl", "-L", "-C", "-", "--retry", "5", "-o", dest]
            if hf_token:
                cmd += ["-H", f"Authorization: Bearer {hf_token}"]
            cmd.append(url)
        subprocess.run(cmd, check=True)
        dt = max(time.time() - t0, 1e-3)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[DOWNLOAD COMPLETE] {rel_path}: {size_mb:.1f} MB in {dt:.1f}s ({size_mb / dt:.1f} MB/s)", flush=True)


def start_server():
    print("=== Step 3: Launching ComfyUI Server on TPU Core 0 ===", flush=True)
    os.chdir(COMFY_DIR)
    subprocess.run(["pkill", "-f", "main.py"])
    time.sleep(1)

    log_file = open("/content/comfyui.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--tpu-device", "0", "--port", "8188", "--listen", "127.0.0.1", "--verbose", "INFO"],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=COMFY_DIR
    )
    log_file.close()
    print(f"ComfyUI server started with PID {proc.pid}", flush=True)

    for _ in range(90):
        if proc.poll() is not None:
            print(f"Server exited early with code {proc.returncode}. Log tail:", flush=True)
            with open("/content/comfyui.log", "r", errors="replace") as f:
                print("".join(f.readlines()[-40:]), flush=True)
            raise RuntimeError("Server exited early")
        time.sleep(1)
        try:
            req = urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
            if req.status == 200:
                stats = json.loads(req.read().decode())
                devices = [d["name"] for d in stats.get("devices", [])]
                print(f"SUCCESS: Server is ONLINE! Detected devices: {devices}", flush=True)
                return proc
        except Exception:
            pass
    with open("/content/comfyui.log", "r", errors="replace") as f:
        print("".join(f.readlines()[-40:]), flush=True)
    raise RuntimeError("Server failed to respond within 90s.")


def build_staged_batch_workflow(prompts):
    """Build a single DAG forcing: all CLIP encodes -> all KSamplers -> all VAEDecodes -> SaveImages."""
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen-image-2.1-turbo-int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 640, "batch_size": 1}},
    }

    for p in prompts:
        i = p["id"]
        wf[f"enc{i}"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["clip", 0]}}
        wf[f"zero{i}"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": [f"enc{i}", 0]}}

    # Barrier across all zero-out conds so every CLIP encode finishes before KSampler begins
    ids = [p["id"] for p in prompts]
    gate_node = f"zero{ids[0]}"
    for idx in range(1, len(ids)):
        next_gate = f"gate_{idx}"
        wf[next_gate] = {
            "class_type": "ConditioningAverage",
            "inputs": {
                "conditioning_to": [gate_node, 0],
                "conditioning_from": [f"zero{ids[idx]}", 0],
                "conditioning_to_strength": 1.0,
            },
        }
        gate_node = next_gate

    for p in prompts:
        i = p["id"]
        wf[f"ks{i}"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "seed": p["seed"],
                "steps": p["steps"],
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": [f"enc{i}", 0],
                "negative": [gate_node, 0],
                "latent_image": ["latent", 0],
                "denoise": 1.0,
            },
        }

    # Barrier across all sampled latents so every KSampler finishes before VAEDecode begins
    batch_node = f"ks{ids[0]}"
    for idx in range(1, len(ids)):
        next_batch = f"lbatch_{idx}"
        wf[next_batch] = {
            "class_type": "LatentBatch",
            "inputs": {"samples1": [batch_node, 0], "samples2": [f"ks{ids[idx]}", 0]},
        }
        batch_node = next_batch

    for idx, p in enumerate(prompts):
        i = p["id"]
        wf[f"lfb{i}"] = {
            "class_type": "LatentFromBatch",
            "inputs": {"samples": [batch_node, 0], "batch_index": idx, "length": 1},
        }
        wf[f"dec{i}"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": [f"lfb{i}", 0], "vae": ["vae", 0]},
        }
        wf[f"save{i}"] = {
            "class_type": "SaveImage",
            "inputs": {"images": [f"dec{i}", 0], "filename_prefix": f"neptune25_batch4_{i:02d}_{p['name']}"},
        }
    return wf


def build_single_workflow(p):
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen-image-2.1-turbo-int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "enc": {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": ["clip", 0]}},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["enc", 0]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "ks": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "seed": p["seed"],
                "steps": p["steps"],
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["enc", 0],
                "negative": ["zero", 0],
                "latent_image": ["latent", 0],
                "denoise": 1.0,
            },
        },
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": f"neptune25_batch4_{p['id']:02d}_{p['name']}"}},
    }


def _start_ws_node_profiler(client_id, stop_event, events):
    """Record ComfyUI WebSocket 'executing' timestamps for exact per-node profiling."""
    async def _listen():
        try:
            import aiohttp
        except ImportError:
            return
        url = f"http://127.0.0.1:8188/ws?clientId={client_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, heartbeat=30) as ws:
                    while not stop_event.is_set():
                        try:
                            msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                            except Exception:
                                continue
                            if payload.get("type") == "executing":
                                d = payload.get("data", {})
                                events.append((time.time(), d.get("prompt_id"), d.get("node")))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception:
            pass

    def _runner():
        asyncio.run(_listen())

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t


def _summarize_node_timings(events, prompt_id, workflow):
    filtered = [(ts, node) for ts, pid, node in events if pid == prompt_id]
    timings = []
    for i in range(len(filtered) - 1):
        ts, node = filtered[i]
        next_ts, _ = filtered[i + 1]
        if node is not None:
            cls = workflow.get(str(node), {}).get("class_type", "?")
            timings.append({"node": str(node), "class_type": cls, "seconds": round(next_ts - ts, 3)})
    return timings


def submit_and_wait(workflow, label, timeout=900):
    client_id = str(uuid.uuid4())
    stop_event = threading.Event()
    ws_events = []
    ws_thread = _start_ws_node_profiler(client_id, stop_event, ws_events)
    time.sleep(0.3)

    payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
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
            print(f"[{label}] Queued prompt_id: {prompt_id}", flush=True)
    except urllib.error.HTTPError as e:
        stop_event.set()
        body = e.read().decode(errors="replace")
        print(f"[{label}] HTTP {e.code} queueing prompt: {body}", flush=True)
        return {"status": "failed", "error": body, "duration_seconds": round(time.time() - t_start, 2)}

    last_print = 0
    try:
        while time.time() - t_start < timeout:
            time.sleep(2)
            elapsed = time.time() - t_start
            if elapsed - last_print >= 15:
                last_print = elapsed
                print(f"  [{label}] {elapsed:.0f}s elapsed...", flush=True)
            try:
                h_req = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=15)
                history = json.loads(h_req.read().decode())
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})
                    if status.get("completed", False):
                        duration = time.time() - t_start
                        time.sleep(0.3)
                        stop_event.set()
                        ws_thread.join(timeout=2)
                        node_timings = _summarize_node_timings(ws_events, prompt_id, workflow)
                        outputs = entry.get("outputs", {})
                        files = []
                        for node_out in outputs.values():
                            for img in node_out.get("images", []):
                                files.append(img["filename"])
                        print(f"--> [{label}] COMPLETED in {duration:.2f}s. Files: {files}", flush=True)
                        if node_timings:
                            print(f"    Per-node timings [{label}]:", flush=True)
                            for nt in node_timings:
                                print(f"      - {nt['node']} ({nt['class_type']}): {nt['seconds']:.2f}s", flush=True)
                        return {
                            "status": "success",
                            "prompt_id": prompt_id,
                            "duration_seconds": round(duration, 2),
                            "output_files": sorted(files),
                            "node_timings": node_timings,
                        }
                    elif status.get("status_str") == "error":
                        duration = time.time() - t_start
                        stop_event.set()
                        print(f"--> [{label}] ERROR after {duration:.2f}s: {status}", flush=True)
                        return {
                            "status": "error",
                            "prompt_id": prompt_id,
                            "duration_seconds": round(duration, 2),
                            "error": str(status),
                        }
            except Exception:
                pass
    finally:
        stop_event.set()

    return {"status": "timeout", "prompt_id": prompt_id, "duration_seconds": round(time.time() - t_start, 2)}


def parse_comfy_log(log_path="/content/comfyui.log"):
    metrics = {"step_speeds_s_per_it": [], "prompt_exec_times": [], "warnings": []}
    if not os.path.exists(log_path):
        return metrics
    with open(log_path, "r", errors="replace") as f:
        text = f.read()
    for m in re.finditer(r"25/25\s*\[([^,\]]+),\s*([0-9.]+)(s/it|it/s)\]", text):
        val = float(m.group(2))
        s_per_it = val if m.group(3) == "s/it" else (1.0 / val if val > 0 else 0.0)
        metrics["step_speeds_s_per_it"].append(round(s_per_it, 3))
    for m in re.finditer(r"Prompt executed in ([0-9.]+) seconds", text):
        metrics["prompt_exec_times"].append(float(m.group(1)))
    return metrics


def run_benchmark():
    print("\n=========================================================================", flush=True)
    print("=== Running Batch 4: Staged 3-Image Prod Batch + Warm Single Check ===", flush=True)
    print("=========================================================================\n", flush=True)

    staged_wf = build_staged_batch_workflow(PROD_PROMPTS)
    batch_res = submit_and_wait(staged_wf, "RUN_A_STAGED_3PROD", timeout=1200)

    warm_wf = build_single_workflow(WARM_SINGLE_PROMPT)
    warm_res = submit_and_wait(warm_wf, "RUN_B_WARM_SINGLE", timeout=600)

    if batch_res.get("status") != "success" or warm_res.get("status") != "success":
        print("\n=== ComfyUI Server Log Tail (last 60 lines) ===", flush=True)
        if os.path.exists("/content/comfyui.log"):
            with open("/content/comfyui.log", "r", errors="replace") as f:
                print("".join(f.readlines()[-60:]), flush=True)

    log_metrics = parse_comfy_log("/content/comfyui.log")

    per_image_avg = None
    if batch_res.get("status") == "success":
        per_image_avg = round(batch_res["duration_seconds"] / len(PROD_PROMPTS), 2)

    report = {
        "benchmark_name": "Neptune Motion Batch 4 (Uncensored INT8 ConvRot DiT + W8A16 + FlashAttn HBM + 64-tok Bucket Padding + Staged DAG)",
        "total_prod_images": len(PROD_PROMPTS),
        "steps_per_image": 25,
        "run_a_staged_batch": {
            **batch_res,
            "average_seconds_per_image": per_image_avg,
            "prompts": [{"id": p["id"], "name": p["name"], "video": p["video"], "scene": p["scene"]} for p in PROD_PROMPTS],
        },
        "run_b_warm_single": warm_res,
        "log_metrics": log_metrics,
    }
    with open("/content/neptune3_batch4_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Results saved to /content/neptune3_batch4_results.json", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-git-pull", action="store_true")
    parser.add_argument("--microbench", action="store_true")
    args, _ = parser.parse_known_args()

    no_git_pull = args.no_git_pull or os.environ.get("COMFY_NO_GIT_PULL") == "1"
    microbench_only = args.microbench or os.environ.get("COMFY_MICROBENCH") == "1"

    setup_environment(no_git_pull=no_git_pull)
    if microbench_only:
        rc = run_tpu_microbench()
        if rc != 0:
            raise RuntimeError("REPLAN_TRIGGER: ck.int8_linear outperformed W8A16 on TPU")
    else:
        download_models()
        proc = start_server()
        try:
            run_benchmark()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
