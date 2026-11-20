#!/usr/bin/env python3
"""Pre-selection TPU HBM detector and Profile A tier selector (v5e-1 16 GB vs v6e-1 32 GB)."""
import json
import subprocess
import sys
import urllib.request


PROFILE_A_CHECKPOINTS = {
    "unet": {
        "name": "qwen-image-2.1-turbo-int8_convrot.safetensors",
        "rel_path": "models/unet/qwen-image-2.1-turbo-int8_convrot.safetensors",
        "url": "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/qwen-image-2.1-turbo-int8_convrot.safetensors",
    },
    "clip": {
        "name": "qwen3vl_8b_int8_convrot.safetensors",
        "rel_path": "models/clip/qwen3vl_8b_int8_convrot.safetensors",
        "url": "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    },
    "vae": {
        "name": "qwen_image_2.1_vae_bf16.safetensors",
        "rel_path": "models/vae/qwen_image_2.1_vae_bf16.safetensors",
        "url": "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors",
    },
}


def select_tpu_profile(hbm_bytes: int) -> dict:
    """Choose execution profile from detected TPU HBM capacity (bytes)."""
    hbm_gb = hbm_bytes / (1024 ** 3)
    if hbm_gb >= 24.0:
        return {
            "tier": "tpu_32gb",
            "hbm_bytes": int(hbm_bytes),
            "hbm_gb": round(hbm_gb, 2),
            "resident_all_models": True,
            "comfy_extra_args": ["--highvram"],
            "clip_batch_size": 4,
            "dit_batch_size": 2,
            "vae_batch_size": 2,
            "staged_chunk_size": 8,
            "checkpoints": PROFILE_A_CHECKPOINTS,
        }
    return {
        "tier": "tpu_16gb",
        "hbm_bytes": int(hbm_bytes),
        "hbm_gb": round(hbm_gb, 2),
        "resident_all_models": False,
        "comfy_extra_args": [],
        "clip_batch_size": 1,
        "dit_batch_size": 1,
        "vae_batch_size": 1,
        "staged_chunk_size": 25,
        "checkpoints": PROFILE_A_CHECKPOINTS,
    }


def probe_tpu_hbm_subprocess(timeout: int = 60) -> int:
    """Probe TPU HBM bytes in a short-lived child process so the parent never holds /dev/accel0."""
    code = r"""
import json, torch
try:
    import torch_tpu  # noqa: F401
    try:
        torch.utils.backend_registration.rename_privateuse1_backend("tpu")
        torch.utils.backend_registration.generate_methods_for_privateuse1_backend()
    except Exception:
        try:
            torch.utils.rename_privateuse1_backend("tpu")
            torch.utils.generate_methods_for_privateuse1_backend()
        except Exception:
            pass
except Exception:
    pass
dev = torch.device("tpu")
try:
    free_b, total_b = torch.accelerator.get_memory_info(dev)
except Exception:
    free_b, total_b = torch.tpu.get_memory_info(dev)
print("TPU_HBM_PROBE:" + json.dumps({"free_bytes": int(free_b), "total_bytes": int(total_b)}))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    for line in res.stdout.splitlines():
        if line.startswith("TPU_HBM_PROBE:"):
            data = json.loads(line.split("TPU_HBM_PROBE:", 1)[1])
            return int(data["total_bytes"])
    raise RuntimeError(f"Failed to probe TPU HBM (rc={res.returncode}): {res.stderr[-500:]}")


def probe_tpu_hbm_from_server(base_url: str = "http://127.0.0.1:8188", timeout: int = 5) -> int:
    """Read TPU HBM bytes from a running ComfyUI /system_stats endpoint."""
    with urllib.request.urlopen(f"{base_url}/system_stats", timeout=timeout) as resp:
        stats = json.loads(resp.read().decode())
    for dev in stats.get("devices", []):
        if dev.get("type") == "tpu" or "TPU" in str(dev.get("name", "")):
            return int(dev["vram_total"])
    raise RuntimeError(f"No TPU device reported by {base_url}/system_stats")


if __name__ == "__main__":
    hbm = probe_tpu_hbm_subprocess()
    print(json.dumps(select_tpu_profile(hbm), indent=2))
