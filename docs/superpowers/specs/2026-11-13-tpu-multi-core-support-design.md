# Google Cloud TPU Multi-Core Support & Kaggle Parallel Inference Design

- **Date:** 2026-11-13
- **Topic:** Google Cloud TPU Multi-Core Hardware Abstraction, Standalone Cluster Orchestrator, and Kaggle Benchmark
- **Target Target:** ComfyUI Core & Multi-Core Deployment Tooling

---

## 1. Executive Summary & Goals

ComfyUI currently supports NVIDIA GPUs, AMD GPUs (ROCm), Intel XPUs, Ascend NPUs, MLUs, Apple MPS, and CPU backends. It lacks native support for Google Cloud TPUs (Tensor Processing Units). 

This design introduces:
1. **Native TorchTPU Device Abstraction**: First-class TPU support in `comfy/model_management.py` and `comfy/cli_args.py` using PyTorch's native `torch.accelerator` API and `torch.device("tpu")`.
2. **Multi-Core Process Architecture**: Because TPU High-Bandwidth Memory (HBM) is strictly partitioned per core (non-shared) and runtime device binding operates on a one-core-per-process model, ComfyUI instances are spawned per core using `TPU_VISIBLE_DEVICES`.
3. **Multi-Core Orchestrator (`tools/tpu_orchestrator.py`)**: A standalone cluster manager that runs and health-checks 8 ComfyUI instances (ports 8188–8195) and concurrently dispatches prompt batches across all cores.
4. **Qwen-Image Diffusion Model Support**: Full compatibility with the Neptune Motion Qwen-Image 2.1 pipeline (UNet, CLIP, VAE), automatically casting GGUF and safetensors weights to TPU-native `torch.bfloat16`.
5. **Kaggle CLI In-Kernel Benchmark Suite**: A complete Kaggle notebook kernel package (`kaggle/tpu_comfyui_benchmark/`) executing parallel 8-core inference benchmarks on Kaggle's free TPU v3-8 accelerators, saving outputs and metrics for retrieval via `kaggle kernels output`.

---

## 2. Hardware & Runtime Architectural Context

### 2.1 The TPU v3-8 Topology
A Google Cloud TPU v3-8 consists of 4 physical chips, each containing 2 Tensor Cores, yielding **8 logical cores**.
- **Memory**: Each core possesses 16 GB dedicated HBM (128 GB total across the system).
- **Interconnect**: Memory is **not shared** across cores; each core is an independent accelerator.
- **Process Model**: PyTorch's native TPU backend (`torch_tpu`) mandates a **one-chip-per-process** model. Attempting to switch or access multiple TPU cores within a single Python process produces a runtime error:
  `ValueError: Cannot set TPU device to index X, current process is bound to device index 0. Changing the active TPU device within a process is not supported.`

### 2.2 Device Selection via `TPU_VISIBLE_DEVICES`
Core selection must occur before accelerator runtime initialization via the environment variable `TPU_VISIBLE_DEVICES`:
```bash
# Core 0:
TPU_VISIBLE_DEVICES=0 python main.py --port 8188 --tpu-device 0
# Core 1:
TPU_VISIBLE_DEVICES=1 python main.py --port 8189 --tpu-device 1
```

---

## 3. Core ComfyUI Changes

All core modifications respect the architectural guidelines in [AGENTS.md](file:///home/devtrung/orca/ComfyUI/AGENTS.md): keeping `comfy/` focused on model inference and hardware management, with no outbound network requests or UI leakage.

### 3.1 CLI Arguments (`comfy/cli_args.py`)
Add `--tpu-device` argument:
```python
parser.add_argument(
    "--tpu-device",
    type=str,
    default=None,
    metavar="DEVICE_ID",
    help="Set the id of the TPU device/core this instance will use (sets TPU_VISIBLE_DEVICES)."
)
```

### 3.2 Process Environment Binding (`main.py`)
Early in `main.py`, before any Torch modules are imported:
```python
if args.tpu_device is not None:
    os.environ['TPU_VISIBLE_DEVICES'] = str(args.tpu_device)
    logging.info(f"Set TPU visible device to: {args.tpu_device}")
```

### 3.3 Hardware Abstraction (`comfy/model_management.py`)

#### A. Backend Detection
```python
try:
    import torch_tpu
except ImportError:
    pass

tpu_available = False
try:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        tpu_available = (getattr(acc, "type", None) == "tpu")
except Exception:
    tpu_available = False

def is_tpu():
    global cpu_state
    global tpu_available
    if cpu_state == CPUState.GPU and tpu_available:
        return True
    return False
```

#### B. Device Resolution
- `get_torch_device()`:
  ```python
  if is_tpu():
      return torch.device("tpu")
  ```
- `get_all_torch_devices()`:
  ```python
  elif is_tpu():
      devices.append(torch.device("tpu"))
  ```
- `get_torch_device_name(device)`:
  ```python
  if hasattr(device, 'type') and device.type == "tpu":
      return "Google Cloud TPU"
  ```

#### C. Memory Tracking & Synchronization
- **Total Memory (`get_total_memory`)**:
  ```python
  if is_tpu():
      try:
          mem_free, mem_total = torch.accelerator.get_memory_info(dev)
          mem_total_torch = torch.accelerator.memory_reserved(dev)
      except Exception:
          mem_total = 16 * 1024 * 1024 * 1024  # 16 GB default HBM per v3 core
          mem_total_torch = mem_total
  ```
- **Free Memory (`get_free_memory`)**:
  ```python
  if is_tpu():
      try:
          mem_free, _ = torch.accelerator.get_memory_info(dev)
          mem_active = torch.accelerator.memory_allocated(dev)
          mem_reserved = torch.accelerator.memory_reserved(dev)
          mem_free_torch = mem_reserved - mem_active
          mem_free_total = mem_free + mem_free_torch
      except Exception:
          mem_free_total = 14 * 1024 * 1024 * 1024
          mem_free_torch = mem_free_total
  ```
- **Synchronization (`synchronize`)**:
  ```python
  if is_tpu():
      torch.accelerator.synchronize(dev)
  ```
- **Cache Clearing (`soft_empty_cache`)**:
  ```python
  if is_tpu():
      torch.accelerator.empty_cache()
  ```

### 3.4 Compute Dtype and Attention
- **Dtype Selection**:
  TPU Matrix Multiply Units (MXUs) natively process `torch.bfloat16`. `should_use_bf16()` returns `True` when `is_tpu()` is active.
- **Attention Backend**:
  PyTorch native Scaled Dot Product Attention (SDPA) is enabled for TPU:
  ```python
  if is_tpu():
      ENABLE_PYTORCH_ATTENTION = True
  ```

---

## 4. Qwen-Image Model Execution & GGUF Support

In accordance with `/home/devtrung/Projects/neptune-motion/README.md`, the pipeline utilizes:
1. `models/unet/qwen-image-2.1-UC-Q4_K_M.gguf`
2. `models/clip/qwen3vl_8b_int8_convrot.safetensors`
3. `models/vae/qwen_image_2.1_vae_bf16.safetensors`

### 4.1 Weight Placement & Dequantization on TPU
- Custom CUDA/GGML kernels do not exist on TPUs.
- When weights from GGUF containers are moved to the TPU device, ComfyUI dequantizes and casts them to `torch.bfloat16`.
- SAFETENSORS text encoder and VAE weights load directly into TPU memory in bfloat16.

---

## 5. Multi-Core Cluster Orchestrator (`tools/tpu_orchestrator.py`)

A standalone management tool that coordinates multi-core execution:

```
tools/tpu_orchestrator.py
  ├── Subprocess Manager (launches/terminates 8 instances with ports 8188..8195)
  ├── Health-Check Poller (verifies /system_stats on all instances)
  └── Batch Prompt Dispatcher (submits parallel prompts and aggregates outputs)
```

### 5.1 Subprocess Management
- Computes core allocations: for `core_id in range(cores)`:
  - Port: `base_port + core_id` (e.g., 8188 + 0 = 8188)
  - Environment: `TPU_VISIBLE_DEVICES = str(core_id)`
  - Command: `python main.py --port {port} --tpu-device {core_id} --listen 127.0.0.1`
- Sets process group on each child for clean termination on `SIGINT`/`SIGTERM`.

### 5.2 Concurrent Prompt Dispatch
- Accepts a ComfyUI prompt workflow JSON.
- Uses `asyncio` and `aiohttp` (or `urllib` thread pool) to submit prompt requests simultaneously to all 8 ports.
- Listens to WebSocket events or polls `/history/{prompt_id}` until all 8 instances finish generation.
- Collects outputs into a unified destination directory and outputs performance summary (latency per core, total throughput in images/minute).

---

## 6. Kaggle TPU Deployment & In-Kernel Benchmark Suite

### 6.1 Architecture Overview
Kaggle kernels run in isolated environments without public ingress ports. The entire benchmark executes inside the kernel via localhost, saving outputs to `/kaggle/working/`:

```
kaggle/tpu_comfyui_benchmark/
  ├── kernel-metadata.json
  └── comfyui_tpu_benchmark.ipynb
```

### 6.2 `kernel-metadata.json`
```json
{
  "id": "devtrung/comfyui-tpu-8core-benchmark",
  "title": "ComfyUI TPU 8-Core Benchmark",
  "code_file": "comfyui_tpu_benchmark.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "true",
  "enable_internet": "true"
}
```

### 6.3 In-Kernel Notebook Steps
1. **Environment Initialization**:
   - Installs `torch_tpu`:
     ```bash
     pip install -q torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
     ```
   - Verifies `torch.accelerator.is_available() and torch.accelerator.current_accelerator().type == "tpu"`.
2. **Repository & Dependencies**:
   - Clones or packages current ComfyUI source code with the TPU changes.
   - Installs ComfyUI requirements.
3. **Model Preparation**:
   - Downloads Qwen-Image weights as specified in Neptune Motion:
     - UNet GGUF -> `models/unet/`
     - CLIP -> `models/clip/`
     - VAE -> `models/vae/`
4. **Cluster Execution**:
   - Launches `tools/tpu_orchestrator.py start --cores 8 --base-port 8188`.
   - Waits for all 8 instances to report ready.
5. **Parallel 8-Core Inference**:
   - Dispatches 8 parallel prompts across cores 0–7.
   - Measures individual latency and total parallel generation speed.
6. **Artifact Output**:
   - Copies generated images and `benchmark_report.json` to `/kaggle/working/output/`.
   - Shuts down the ComfyUI cluster cleanly.

### 6.4 Automation Script (`tools/run_kaggle_benchmark.sh`)
Automates the local Kaggle CLI cycle:
```bash
kaggle kernels push -p kaggle/tpu_comfyui_benchmark
kaggle kernels status devtrung/comfyui-tpu-8core-benchmark  # Poll until "complete"
kaggle kernels output devtrung/comfyui-tpu-8core-benchmark -p out/kaggle_tpu/
```

---

## 7. Error Handling & Edge Cases

| Scenario | Handling Strategy |
| :--- | :--- |
| `torch_tpu` not installed | Gracefully falls back to existing device detection (CPU / CUDA / etc.) without errors. |
| Incompatible device index requested | Fails early in `main.py` with an informative error if `--tpu-device` is out of bounds. |
| Port collision on multi-core startup | Orchestrator verifies port availability before spawning; logs clear message if a port is in use. |
| One TPU core errors out during batch | Orchestrator captures per-core HTTP error responses, marks failed prompt, and allows remaining 7 cores to complete. |
| Unclean shutdown leaves TPU locks | Orchestrator registers signal handlers (`SIGINT`, `SIGTERM`, `atexit`) to terminate child process groups. |

---

## 8. Verification & Test Plan

1. **Unit & Device Detection Verification**:
   - Verify `comfy/model_management.py` imports without regressions on non-TPU systems.
   - Verify `--tpu-device` argument correctly sets `TPU_VISIBLE_DEVICES`.
2. **Multi-Core Orchestrator Verification**:
   - Test `tools/tpu_orchestrator.py` dry-run and startup/shutdown lifecycle.
3. **Kaggle Kernel Push & Execution**:
   - Push benchmark kernel to Kaggle using `kaggle kernels push`.
   - Monitor run status until completion (`kaggle kernels status`).
   - Download results via `kaggle kernels output` and inspect generated images and throughput logs.
