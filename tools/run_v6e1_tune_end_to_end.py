#!/usr/bin/env python3
"""Colab TPU v6e-1 (32 GB HBM) Profile A batch-size tuning runner (B=1, B=2, B=4) with pre-launch HBM check and auto VM stop."""
import argparse
import os
import sys
import time
import json
import shutil
import tempfile
import subprocess

COLAB_CLI = "/home/devtrung/.local/bin/colab"
SESSION = "comfy-v6e1-tune"
REPO_DIR = "/home/devtrung/orca/ComfyUI"
OUT_DIR = "/home/devtrung/orca/ComfyUI/output/neptune_v6e1_tune"
ARTIFACT_DIR = "/home/devtrung/.gemini/antigravity-cli/brain/55517722-56f8-4d5b-a54c-8f67dac28dc1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

REMOTE_RUNNER = r'''
import asyncio, json, os, re, shutil, subprocess, sys, threading, time, uuid, urllib.error, urllib.request

COMFY_DIR = "/content/ComfyUI"
sys.path.insert(0, COMFY_DIR)
os.chdir(COMFY_DIR)

from tools.tpu_profile import probe_tpu_hbm_subprocess, probe_tpu_hbm_from_server, select_tpu_profile

PROMPTS = [
    {
        "id": 1,
        "name": "neptune_brachiosaurus_rearing",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a mounted sauropod skeleton rearing toward the ceiling of an exhibition hall. In the centre of the frame, a towering sauropod skeleton rises vertically with its forelimbs much taller than its hind limbs, drawn with precise pencil outlines showing the ribcage, shoulder blades, and an upward-curving neck. On the left side of the frame, the colossal front limbs stand planted on the museum floor like massive upright pillars, surrounded by empty space. On the right side of the frame, the long tail slopes gently downward to balance the elevated neck, extending toward the right edge. Across the upper third, the small delicate skull tilts forward at the apex of the arched neck near the top margin. Across the lower third, a clean horizontal floor line anchors the upright legs with faint reflection marks. The lines are crisp dark grey pencil lines with subtle terracotta touches along the spine and no shading or background clutter. The overall composition is monumental, balanced, and serene, surrounded by generous margins of blank warm paper.",
        "seed": 6101,
    },
    {
        "id": 2,
        "name": "neptune_source_redundancy",
        "prompt": "A detailed educational whiteboard sketch illustrating Source Redundancy on a clean light ivory background. Hand-drawn dark charcoal line work depicts computational blocks, vector coordinates, and transformation arrows. Each block shows mathematical symbols clearly rendered with technical accuracy and clean hatching. Flow lines connect the initial stage to intermediate transformations and output variables. Subtle watercolor accents in muted navy and amber highlight active gradients and data paths, creating an elegant and readable visual reference.",
        "seed": 6102,
    },
    {
        "id": 3,
        "name": "neptune_entropy_deployment",
        "prompt": "A precise architectural drawing of a production deployment showing Information Entropy embedded within an engineering workflow. Crisp technical line art shows input sensors, preprocessing units, computation engines, and output monitors on a warm cream canvas. Labeled annotations highlight operational latency and throughput metrics. Delicate cross-hatching renders mechanical casings and circuit components with draftsmanship precision and balanced visual composition across the layout.",
        "seed": 6103,
    },
    {
        "id": 4,
        "name": "neptune_sauropod_herd",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a towering sauropod skeleton herd striding across an expansive museum gallery floor. In the centre of the frame, three interconnected fossilized skeletons of varying scale advance from left to right, drawn with sharp dark grey pencil outlines showing elongated cervical vertebrae, columnar limbs, and sweeping tails. On the left side of the frame, a subadult skeleton lowers its neck toward ground level beside scattered fossil stones. On the right side of the frame, the largest adult skeleton raises its head gracefully toward the upper right margin. Across the lower third, a single thin horizontal ground line anchors every pillar-like foot without shadows or heavy texture. The lines are crisp dark grey pencil lines with subtle terracotta touches along the dorsal spines and pale amber accents on the ribcages. The overall composition is clean, serene, and balanced, surrounded by vast open beige paper.",
        "seed": 6104,
    },
]

def install_deps():
    print("=== Step 1: Installing ComfyUI Dependencies (preserving TPU torch) ===", flush=True)
    with open(os.path.join(COMFY_DIR, "requirements.txt")) as f:
        reqs = [
            l.strip() for l in f
            if l.strip() and not l.strip().startswith("#")
            and l.strip().lower().split("==")[0].split(">=")[0].split("<=")[0].strip()
            not in ("torch", "torchvision", "torchaudio")
        ]
    with open("/tmp/comfy_reqs_nobase_torch.txt", "w") as f:
        f.write("\n".join(reqs) + "\n")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/tmp/comfy_reqs_nobase_torch.txt", "gguf"], check=True)

def detect_hbm():
    print("=== Step 2: Pre-Selection TPU HBM Probe ===", flush=True)
    hbm_bytes = probe_tpu_hbm_subprocess()
    profile = select_tpu_profile(hbm_bytes)
    print(f"Detected TPU HBM: {profile['hbm_gb']:.2f} GB -> tier={profile['tier']} resident_all={profile['resident_all_models']}", flush=True)
    return profile

def download_models(profile):
    print("=== Step 3: Downloading Profile A Uncensored INT8 Checkpoints ===", flush=True)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    use_aria2 = shutil.which("aria2c") is not None
    for key, info in profile["checkpoints"].items():
        dest = os.path.join(COMFY_DIR, info["rel_path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
            print(f"[CACHE HIT] {info['rel_path']}", flush=True)
            continue
        t0 = time.time()
        if use_aria2:
            cmd = ["aria2c", "-x", "8", "-s", "8", "-k", "1M", "--continue=true",
                   "-d", os.path.dirname(dest), "-o", os.path.basename(dest)]
            if hf_token:
                cmd += ["--header", f"Authorization: Bearer {hf_token}"]
            cmd.append(info["url"])
        else:
            cmd = ["curl", "-L", "-C", "-", "--retry", "5", "-o", dest]
            if hf_token:
                cmd += ["-H", f"Authorization: Bearer {hf_token}"]
            cmd.append(info["url"])
        subprocess.run(cmd, check=True)
        dt = max(time.time() - t0, 1e-3)
        mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[DOWNLOAD] {key}: {mb:.1f} MB in {dt:.1f}s ({mb/dt:.1f} MB/s)", flush=True)

def start_server(profile):
    print(f"=== Step 4: Launching ComfyUI Server (extra_args={profile['comfy_extra_args']}) ===", flush=True)
    subprocess.run(["pkill", "-f", "main.py"])
    time.sleep(1)
    log_file = open("/content/comfyui.log", "w")
    cmd = [sys.executable, "main.py", "--tpu-device", "0", "--port", "8188", "--listen", "127.0.0.1", "--verbose", "INFO"]
    cmd += profile["comfy_extra_args"]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True, cwd=COMFY_DIR)
    log_file.close()
    for _ in range(90):
        if proc.poll() is not None:
            with open("/content/comfyui.log", errors="replace") as f:
                print("".join(f.readlines()[-40:]), flush=True)
            raise RuntimeError(f"Server exited early rc={proc.returncode}")
        time.sleep(1)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2) as resp:
                if resp.status == 200:
                    stats = json.loads(resp.read().decode())
                    print("Server ONLINE:", json.dumps(stats.get("devices", [])), flush=True)
                    return proc
        except Exception:
            pass
    raise RuntimeError("Server did not start in 90s")

def _start_ws_profiler(client_id, stop_event, events):
    async def _listen():
        import aiohttp
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
                                p = json.loads(msg.data)
                            except Exception:
                                continue
                            if p.get("type") == "executing":
                                d = p.get("data", {})
                                events.append((time.time(), d.get("prompt_id"), d.get("node")))
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
        except Exception:
            pass
    t = threading.Thread(target=lambda: asyncio.run(_listen()), daemon=True)
    t.start()
    return t

def _summarize(events, prompt_id, wf):
    filt = [(ts, node) for ts, pid, node in events if pid == prompt_id]
    out = []
    for i in range(len(filt) - 1):
        ts, node = filt[i]
        nts, _ = filt[i + 1]
        if node is not None:
            out.append({"node": str(node), "class_type": wf.get(str(node), {}).get("class_type", "?"), "seconds": round(nts - ts, 3)})
    return out

def build_batched_wf(batch_prompts, prefix, steps=25, width=1024, height=640, seed_offset=0):
    B = len(batch_prompts)
    enc_inputs = {"clip": ["clip", 0]}
    for idx, p in enumerate(batch_prompts):
        enc_inputs[f"texts.text{idx}"] = p["prompt"]
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen-image-2.1-turbo-int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "enc": {"class_type": "CLIPTextEncodeBatch", "inputs": enc_inputs},
        "zero": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["enc", 0]}},
        "latent": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": B}},
        "ks": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["unet", 0],
                "seed": batch_prompts[0]["seed"] + seed_offset,
                "steps": steps,
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
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": prefix}},
    }

def submit_and_wait(wf, label, timeout=900):
    client_id = str(uuid.uuid4())
    stop_event = threading.Event()
    ws_events = []
    ws_t = _start_ws_profiler(client_id, stop_event, ws_events)
    time.sleep(0.25)
    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=json.dumps({"prompt": wf, "client_id": client_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            pid = json.loads(resp.read().decode())["prompt_id"]
            print(f"[{label}] queued prompt_id={pid}", flush=True)
    except urllib.error.HTTPError as e:
        stop_event.set()
        body = e.read().decode(errors="replace")
        print(f"[{label}] HTTP {e.code}: {body}", flush=True)
        return {"status": "failed", "error": body, "duration_seconds": round(time.time() - t0, 2)}

    try:
        while time.time() - t0 < timeout:
            time.sleep(1.5)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:8188/history/{pid}", timeout=15) as hresp:
                    hist = json.loads(hresp.read().decode())
                if pid in hist:
                    entry = hist[pid]
                    st = entry.get("status", {})
                    if st.get("completed", False):
                        dur = time.time() - t0
                        time.sleep(0.25)
                        stop_event.set()
                        ws_t.join(timeout=2)
                        nt = _summarize(ws_events, pid, wf)
                        files = sorted(img["filename"] for no in entry.get("outputs", {}).values() for img in no.get("images", []))
                        print(f"--> [{label}] DONE in {dur:.2f}s | nodes={[(x['node'], x['seconds']) for x in nt]} | files={files}", flush=True)
                        return {"status": "success", "duration_seconds": round(dur, 2), "node_timings": nt, "output_files": files}
                    if st.get("status_str") == "error":
                        stop_event.set()
                        print(f"--> [{label}] ERROR: {st}", flush=True)
                        return {"status": "error", "duration_seconds": round(time.time() - t0, 2), "error": str(st)}
            except Exception:
                pass
    finally:
        stop_event.set()
    return {"status": "timeout", "duration_seconds": round(time.time() - t0, 2)}

def main():
    install_deps()
    profile = detect_hbm()
    download_models(profile)
    proc = start_server(profile)
    results = {"profile": profile, "runs": {}}
    try:
        server_hbm = probe_tpu_hbm_from_server()
        results["server_hbm_gb"] = round(server_hbm / (1024 ** 3), 2)
        print(f"Server /system_stats verified TPU HBM: {results['server_hbm_gb']:.2f} GB", flush=True)

        # Sweep B=1 (cold + warm same-bucket), B=2 (cold + warm), B=4 (cold + warm), B=6 (cold + warm)
        sweep = [
            ("B1_cold", PROMPTS[:1], "v6e1_b1_cold", 0),
            ("B1_warm", PROMPTS[3:4], "v6e1_b1_warm", 0),
            ("B2_cold", PROMPTS[:2], "v6e1_b2_cold", 10),
            ("B2_warm", PROMPTS[2:4], "v6e1_b2_warm", 10),
            ("B4_cold", PROMPTS[:4], "v6e1_b4_cold", 20),
            ("B4_warm", PROMPTS[1:] + PROMPTS[:1], "v6e1_b4_warm", 30),
            ("B6_cold", PROMPTS + PROMPTS[:2], "v6e1_b6_cold", 40),
            ("B6_warm", PROMPTS[2:] + PROMPTS[:4], "v6e1_b6_warm", 50),
        ]
        for label, batch_p, prefix, soff in sweep:
            wf = build_batched_wf(batch_p, prefix=prefix, steps=25, width=1024, height=640, seed_offset=soff)
            r = submit_and_wait(wf, label, timeout=600)
            if r.get("status") == "success":
                r["batch_size"] = len(batch_p)
                r["seconds_per_image"] = round(r["duration_seconds"] / len(batch_p), 2)
            results["runs"][label] = r
            if r.get("status") != "success":
                print(f"Stopping sweep after {label} failed (likely HBM limit for B={len(batch_p)}).", flush=True)
                break

        with open("/content/v6e1_tune_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("SUMMARY:", json.dumps(results, indent=2), flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

main()
'''


def stream_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def run(session_name=SESSION, use_existing=False):
    try:
        if use_existing:
            print(f"=== Using Existing Colab Session '{session_name}' ===", flush=True)
        else:
            print(f"=== Provisioning Colab TPU v6e-1 Session '{session_name}' ===", flush=True)
            provisioned = False
            max_retries = 100
            for attempt in range(1, max_retries + 1):
                print(f"Attempt {attempt}/{max_retries}...", flush=True)
                res = subprocess.run([COLAB_CLI, "--auth=oauth2", "new", "-s", session_name, "--tpu", "v6e1"], capture_output=True, text=True)
                print(res.stdout, res.stderr, flush=True)
                if res.returncode == 0:
                    provisioned = True
                    break
                subprocess.run([COLAB_CLI, "--auth=oauth2", "stop", "-s", session_name], capture_output=True, text=True)
                if attempt < max_retries:
                    time.sleep(5)
            if not provisioned:
                raise RuntimeError("Failed to provision Colab v6e1 session after retries")

        bootstrap = """
import os, subprocess
COMFY_DIR = "/content/ComfyUI"
if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
    subprocess.run(["git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
                    "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR], check=True)
print("Repo cloned to /content/ComfyUI", flush=True)
"""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
            tf.write(bootstrap)
            tf_path = tf.name
        try:
            boot_ok = False
            for b_try in range(1, 4):
                if b_try > 1 or not use_existing:
                    time.sleep(5 * b_try)
                rc, _ = stream_cmd([COLAB_CLI, "--auth=oauth2", "exec", "-s", session_name, "-f", tf_path, "--timeout", "300"])
                if rc == 0:
                    boot_ok = True
                    break
                print(f"Bootstrap exec attempt {b_try}/3 failed (rc={rc}), restarting kernel & retrying...", flush=True)
                subprocess.run([COLAB_CLI, "--auth=oauth2", "restart-kernel", "-s", session_name], capture_output=True, text=True)
            if not boot_ok:
                raise RuntimeError("Bootstrap failed after 3 attempts")
        finally:
            os.unlink(tf_path)

        files_to_upload = [
            "comfy/model_management.py",
            "comfy/model_patcher.py",
            "comfy/quant_ops.py",
            "comfy/sd.py",
            "comfy/sd1_clip.py",
            "comfy/text_encoders/qwen_image21.py",
            "comfy/ldm/qwen_image21/model.py",
            "comfy_extras/nodes_cond.py",
            "tools/tpu_profile.py",
        ]
        for rel in files_to_upload:
            local = os.path.join(REPO_DIR, rel)
            remote = f"/content/ComfyUI/{rel}"
            r = subprocess.run([COLAB_CLI, "--auth=oauth2", "upload", "-s", session_name, local, remote], capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"Upload failed for {rel}: {r.stderr}")
            print(f"Uploaded {rel}", flush=True)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as rf:
            rf.write(REMOTE_RUNNER)
            rf_path = rf.name
        try:
            cmd = [COLAB_CLI, "--auth=oauth2", "exec", "-s", session_name]
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                cmd += ["--env", f"HF_TOKEN={hf_token}"]
            cmd += ["-f", rf_path, "--timeout", "2400"]
            rc, _ = stream_cmd(cmd)
            if rc != 0:
                raise RuntimeError(f"Remote v6e1 tuning script failed rc={rc}")
        finally:
            os.unlink(rf_path)

        for remote, name in [
            ("/content/v6e1_tune_results.json", "v6e1_tune_results.json"),
            ("/content/comfyui.log", "comfyui.log"),
        ]:
            local = os.path.join(OUT_DIR, name)
            subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", session_name, remote, local])

        res_json = os.path.join(OUT_DIR, "v6e1_tune_results.json")
        if os.path.exists(res_json):
            shutil.copy(res_json, os.path.join(ARTIFACT_DIR, "v6e1_tune_results.json"))
            data = json.load(open(res_json))
            for label, r in data.get("runs", {}).items():
                for fn in r.get("output_files", []):
                    local = os.path.join(OUT_DIR, fn)
                    subprocess.run([COLAB_CLI, "--auth=oauth2", "download", "-s", session_name, f"/content/ComfyUI/output/{fn}", local])
                    if os.path.exists(local):
                        print(f"Downloaded {fn} ({os.path.getsize(local)} bytes)", flush=True)
                        if label in ("B1_warm", "B2_warm", "B4_warm"):
                            shutil.copy(local, os.path.join(ARTIFACT_DIR, fn))
    finally:
        print(f"=== Stopping Colab Session '{session_name}' ===", flush=True)
        subprocess.run([COLAB_CLI, "--auth=oauth2", "stop", "-s", session_name], capture_output=True, text=True)
        chk = subprocess.run([COLAB_CLI, "--auth=oauth2", "sessions"], capture_output=True, text=True)
        print("Active sessions:", chk.stdout or chk.stderr, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--use-existing", action="store_true")
    cli_args = parser.parse_args()
    run(session_name=cli_args.session, use_existing=cli_args.use_existing)
