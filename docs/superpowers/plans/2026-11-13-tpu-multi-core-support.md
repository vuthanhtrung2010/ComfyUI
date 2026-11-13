# Google Cloud TPU Multi-Core Support & Kaggle Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce native Google Cloud TPU device support (`torch.device("tpu")`), multi-core instance coordination (`TPU_VISIBLE_DEVICES`), Qwen-Image bfloat16 execution, and an automated Kaggle 8-core TPU parallel inference benchmark suite.

**Architecture:** ComfyUI core detects TPU availability through PyTorch's unified `torch.accelerator` API and `torch_tpu`, configuring `torch.device("tpu")`, bfloat16 compute, and native SDPA attention. Since TPU HBM is isolated per core and the runtime is strictly one-core-per-process, a standalone orchestrator (`tools/tpu_orchestrator.py`) spawns 8 ComfyUI instances on ports 8188–8195 and dispatches parallel prompt batches. A Kaggle benchmark package (`kaggle/tpu_comfyui_benchmark/`) runs the full workflow in Kaggle's free TPU v3-8 environment and exports results.

**Tech Stack:** Python 3.10+, PyTorch 2.5+, `torch_tpu` (`PrivateUse1`), `torch.accelerator`, ComfyUI, Kaggle CLI.

**Spec:** [docs/superpowers/specs/2026-11-13-tpu-multi-core-support-design.md](file:///home/devtrung/orca/ComfyUI/docs/superpowers/specs/2026-11-13-tpu-multi-core-support-design.md)

## Global Constraints
- Core ComfyUI changes must touch the narrowest path possible in `comfy/cli_args.py`, `main.py`, and `comfy/model_management.py`.
- No outbound network requests from core ComfyUI ([AGENTS.md](file:///home/devtrung/orca/ComfyUI/AGENTS.md)).
- Strictly follow the PyTorch `torch.accelerator` interface for TPU queries and operations.
- Preserve backward compatibility with CUDA, ROCm, XPU, NPU, MLU, MPS, and CPU.

---

### Task 1: Add `--tpu-device` CLI Flag and Environment Binding

**Files:**
- Modify: `comfy/cli_args.py:80-110`
- Modify: `main.py:90-115`
- Test: `tests/test_tpu_cli.py`

**Interfaces:**
- Consumes: `args.tpu_device` from `comfy.cli_args`
- Produces: `os.environ["TPU_VISIBLE_DEVICES"]` bound early in `main.py`

- [ ] **Step 1: Write unit test for `--tpu-device` CLI parsing and environment binding**

Create `tests/test_tpu_cli.py`:
```python
import os
import sys
import unittest
from unittest.mock import patch

class TestTPUCLI(unittest.TestCase):
    def test_tpu_device_argument(self):
        from comfy.cli_args import parser
        args = parser.parse_args(["--tpu-device", "3"])
        self.assertEqual(args.tpu_device, "3")

    def test_tpu_device_env_setting(self):
        from comfy.cli_args import parser
        args = parser.parse_args(["--tpu-device", "2"])
        with patch.dict(os.environ, {}, clear=False):
            if args.tpu_device is not None:
                os.environ['TPU_VISIBLE_DEVICES'] = str(args.tpu_device)
            self.assertEqual(os.environ.get("TPU_VISIBLE_DEVICES"), "2")

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_cli.py`
Expected: FAIL with `unrecognized arguments: --tpu-device`

- [ ] **Step 3: Implement `--tpu-device` in `comfy/cli_args.py` and `main.py`**

In `comfy/cli_args.py`, add:
```python
parser.add_argument("--tpu-device", type=str, default=None, metavar="DEVICE_ID", help="Set the id of the TPU device/core this instance will use (sets TPU_VISIBLE_DEVICES).")
```

In `main.py` right alongside `args.cuda_device` handling:
```python
    if args.tpu_device is not None:
        os.environ['TPU_VISIBLE_DEVICES'] = str(args.tpu_device)
        logging.info("Set TPU visible device to: {}".format(args.tpu_device))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_cli.py`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add comfy/cli_args.py main.py tests/test_tpu_cli.py
git commit -m "feat(cli): add --tpu-device argument and TPU_VISIBLE_DEVICES environment binding"
```

---

### Task 2: Implement TPU Hardware Detection and Device Resolution in `model_management.py`

**Files:**
- Modify: `comfy/model_management.py:120-250,578-610`
- Test: `tests/test_tpu_model_management.py`

**Interfaces:**
- Consumes: `torch.accelerator.is_available()`, `torch.accelerator.current_accelerator()`
- Produces: `is_tpu() -> bool`, `get_torch_device() -> torch.device("tpu")`, `get_all_torch_devices() -> [torch.device("tpu")]`, `get_torch_device_name() -> str`

- [ ] **Step 1: Write unit tests for TPU detection and device routing**

Create `tests/test_tpu_model_management.py`:
```python
import unittest
from unittest.mock import MagicMock, patch
import torch

class TestTPUModelManagement(unittest.TestCase):
    def test_is_tpu_when_available(self):
        import comfy.model_management as mm
        mock_acc = MagicMock()
        mock_acc.type = "tpu"
        with patch.object(torch.accelerator, "is_available", return_value=True), \
             patch.object(torch.accelerator, "current_accelerator", return_value=mock_acc), \
             patch.object(mm, "tpu_available", True), \
             patch.object(mm, "cpu_state", mm.CPUState.GPU):
            self.assertTrue(mm.is_tpu())
            dev = mm.get_torch_device()
            self.assertEqual(dev.type, "tpu")

    def test_is_tpu_when_not_available(self):
        import comfy.model_management as mm
        with patch.object(mm, "tpu_available", False):
            self.assertFalse(mm.is_tpu())

    def test_get_torch_device_name_tpu(self):
        import comfy.model_management as mm
        mock_device = MagicMock()
        mock_device.type = "tpu"
        name = mm.get_torch_device_name(mock_device)
        self.assertIn("TPU", name)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_model_management.py`
Expected: FAIL with `AttributeError: module 'comfy.model_management' has no attribute 'is_tpu'`

- [ ] **Step 3: Implement TPU detection and device getters in `comfy/model_management.py`**

In `comfy/model_management.py`:
1. Module initialization:
```python
try:
    import torch_tpu  # noqa: F401
except ImportError:
    pass

tpu_available = False
try:
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        acc = torch.accelerator.current_accelerator()
        tpu_available = (getattr(acc, "type", None) == "tpu")
    elif hasattr(torch, "tpu") and hasattr(torch.tpu, "is_available"):
        tpu_available = torch.tpu.is_available()
except Exception:
    tpu_available = False

def is_tpu():
    global cpu_state
    global tpu_available
    if cpu_state == CPUState.GPU and tpu_available:
        return True
    return False
```

2. In `get_torch_device()`:
```python
    if is_intel_xpu():
        return torch.device("xpu", torch.xpu.current_device())
    elif is_ascend_npu():
        return torch.device("npu", torch.npu.current_device())
    elif is_mlu():
        return torch.device("mlu", torch.mlu.current_device())
    elif is_tpu():
        return torch.device("tpu")
    else:
        return torch.device(torch.cuda.current_device())
```

3. In `get_all_torch_devices()`:
```python
    elif is_mlu():
        for i in range(torch.mlu.device_count()):
            devices.append(torch.device("mlu", i))
    elif is_tpu():
        devices.append(torch.device("tpu"))
```

4. In `get_torch_device_name()`:
```python
    elif hasattr(device, 'type') and device.type == "tpu":
        return "Google Cloud TPU"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_model_management.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add comfy/model_management.py tests/test_tpu_model_management.py
git commit -m "feat(devices): add TPU device detection and resolution in model_management"
```

---

### Task 3: Implement Memory Management, Synchronization, Dtype, and Attention for TPU

**Files:**
- Modify: `comfy/model_management.py:296-335,438-455,800-840,1841-1882,1930-1960`
- Test: `tests/test_tpu_memory_attention.py`

**Interfaces:**
- Consumes: `torch.accelerator.get_memory_info()`, `torch.accelerator.synchronize()`, `torch.accelerator.empty_cache()`
- Produces: `get_total_memory(dev)`, `get_free_memory(dev)`, `synchronize()`, `soft_empty_cache()`, `should_use_bf16() -> True`

- [ ] **Step 1: Write unit tests for TPU memory queries, synchronization, and dtype**

Create `tests/test_tpu_memory_attention.py`:
```python
import unittest
from unittest.mock import MagicMock, patch
import torch

class TestTPUMemoryAttention(unittest.TestCase):
    def test_tpu_total_and_free_memory(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"

        with patch.object(mm, "is_tpu", return_value=True), \
             patch.object(torch.accelerator, "get_memory_info", return_value=(8 * 1024**3, 16 * 1024**3)), \
             patch.object(torch.accelerator, "memory_allocated", return_value=2 * 1024**3), \
             patch.object(torch.accelerator, "memory_reserved", return_value=4 * 1024**3):
            total = mm.get_total_memory(mock_dev)
            self.assertEqual(total, 16 * 1024**3)

            free = mm.get_free_memory(mock_dev)
            self.assertGreater(free, 0)

    def test_tpu_bf16_and_attention(self):
        import comfy.model_management as mm
        with patch.object(mm, "is_tpu", return_value=True):
            self.assertTrue(mm.should_use_bf16())

    def test_tpu_synchronize_and_empty_cache(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        with patch.object(mm, "is_tpu", return_value=True), \
             patch.object(torch.accelerator, "synchronize") as mock_sync, \
             patch.object(torch.accelerator, "empty_cache") as mock_empty:
            mm.synchronize(mock_dev)
            mock_sync.assert_called()
            mm.soft_empty_cache()
            mock_empty.assert_called()

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_memory_attention.py`
Expected: FAIL

- [ ] **Step 3: Implement memory, synchronization, and dtype operations in `comfy/model_management.py`**

1. In `get_total_memory()`:
```python
        elif is_tpu():
            try:
                mem_free, mem_total = torch.accelerator.get_memory_info(dev)
                mem_total_torch = torch.accelerator.memory_reserved(dev)
            except Exception:
                mem_total = 16 * 1024 * 1024 * 1024
                mem_total_torch = mem_total
```

2. In `get_free_memory()`:
```python
        elif is_tpu():
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

3. In `synchronize()`:
```python
    if is_tpu():
        try:
            torch.accelerator.synchronize(dev)
        except Exception:
            pass
```

4. In `soft_empty_cache()`:
```python
    if is_tpu():
        try:
            torch.accelerator.empty_cache()
        except Exception:
            pass
```

5. In attention backend initialization:
```python
    if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca() or is_tpu():
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
```

6. In `should_use_bf16()`:
```python
    if is_tpu():
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_memory_attention.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add comfy/model_management.py tests/test_tpu_memory_attention.py
git commit -m "feat(tpu): implement memory stats, cache cleanup, synchronization, and bf16 for TPU"
```

---

### Task 4: Ensure Transparent Bfloat16 Casting for TPU Model Weights

**Files:**
- Modify: `comfy/model_patcher.py` / `comfy/model_management.py`
- Test: `tests/test_tpu_weight_cast.py`

**Interfaces:**
- Consumes: Model weights loaded in CPU/RAM
- Produces: `torch.bfloat16` tensors when placed on `torch.device("tpu")`

- [ ] **Step 1: Write test for casting weights to TPU bfloat16**

Create `tests/test_tpu_weight_cast.py`:
```python
import unittest
from unittest.mock import patch, MagicMock
import torch

class TestTPUWeightCast(unittest.TestCase):
    def test_supports_cast_tpu(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        self.assertTrue(mm.supports_cast(mock_dev, torch.bfloat16))
        self.assertTrue(mm.supports_cast(mock_dev, torch.float32))

    def test_cast_to_device_bf16(self):
        import comfy.model_management as mm
        tensor = torch.randn(10, 10, dtype=torch.float32)
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        with patch.object(mm, "is_tpu", return_value=True):
            # Casting should produce bfloat16 for TPU target
            out = mm.cast_to(tensor, mock_dev, dtype=torch.bfloat16)
            self.assertEqual(out.dtype, torch.bfloat16)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it passes / verify implementation**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_weight_cast.py`
Verify `supports_cast` and `cast_to` properly recognize TPU device type.

- [ ] **Step 3: Update `supports_cast` in `comfy/model_management.py` if needed**

Ensure `supports_cast` includes TPU for bfloat16 and float32.

- [ ] **Step 4: Commit**

```bash
git add comfy/model_management.py tests/test_tpu_weight_cast.py
git commit -m "feat(tpu): support bfloat16 casting for TPU device targets"
```

---

### Task 5: Build Multi-Core Cluster Orchestrator (`tools/tpu_orchestrator.py`)

**Files:**
- Create: `tools/tpu_orchestrator.py`
- Test: `tests/test_tpu_orchestrator.py`

**Interfaces:**
- CLI:
  - `start --cores 8 --base-port 8188`
  - `stop`
  - `benchmark --cores 8 --prompts 8 --workflow <path>`
- Internal: Launches ComfyUI subprocesses with `TPU_VISIBLE_DEVICES=i` and `--tpu-device i`

- [ ] **Step 1: Write unit tests for cluster configuration and port management**

Create `tests/test_tpu_orchestrator.py`:
```python
import unittest
from tools.tpu_orchestrator import TPUClusterManager

class TestTPUClusterManager(unittest.TestCase):
    def test_cluster_instance_specs(self):
        manager = TPUClusterManager(cores=8, base_port=8188)
        specs = manager.get_instance_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual(specs[0]["port"], 8188)
        self.assertEqual(specs[0]["core_id"], 0)
        self.assertEqual(specs[7]["port"], 8195)
        self.assertEqual(specs[7]["core_id"], 7)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_orchestrator.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.tpu_orchestrator'`

- [ ] **Step 3: Implement `tools/tpu_orchestrator.py`**

Create `tools/tpu_orchestrator.py`:
- `TPUClusterManager`:
  - `get_instance_specs()`: Computes port and `TPU_VISIBLE_DEVICES` per core.
  - `start()`: Spawns $N$ subprocesses: `python main.py --port {port} --tpu-device {core_id} --listen 127.0.0.1`.
  - `wait_ready(timeout=180)`: Polls `/system_stats` on each port until all respond HTTP 200.
  - `stop()`: Sends SIGTERM to all child processes and waits for clean exit.
  - `dispatch_parallel(prompts)`: Uses concurrent thread pool to POST `/prompt` to each port simultaneously, polls `/history/{prompt_id}` until completed, and aggregates generated images and timing statistics.
- Main CLI parser supporting `start`, `stop`, `status`, and `benchmark` commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest tests/test_tpu_orchestrator.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/tpu_orchestrator.py tests/test_tpu_orchestrator.py
git commit -m "feat(tools): add TPU cluster orchestrator for multi-core lifecycle and parallel prompt dispatch"
```

---

### Task 6: Create Kaggle In-Kernel Benchmark Suite & Automation Scripts

**Files:**
- Create: `kaggle/tpu_comfyui_benchmark/kernel-metadata.json`
- Create: `kaggle/tpu_comfyui_benchmark/comfyui_tpu_benchmark.ipynb`
- Create: `tools/run_kaggle_benchmark.sh`
- Create: `workflows/qwen_image_tpu_test.json`

**Interfaces:**
- Consumes: Kaggle CLI (`kaggle kernels push`, `kaggle kernels status`, `kaggle kernels output`)
- Produces: Executed 8-core benchmark on Kaggle TPU v3-8, saving results to `/kaggle/working/output/`

- [ ] **Step 1: Create sample Qwen-Image TPU workflow JSON**

Create `workflows/qwen_image_tpu_test.json` with standard text-to-image prompt graph using Qwen-Image UNet, CLIP, and VAE.

- [ ] **Step 2: Create Kaggle `kernel-metadata.json`**

Create `kaggle/tpu_comfyui_benchmark/kernel-metadata.json`:
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

- [ ] **Step 3: Create Kaggle notebook `comfyui_tpu_benchmark.ipynb`**

Notebook sections:
1. Verification of TPU v3-8 environment and `torch.accelerator`.
2. Installation of `torch_tpu`.
3. Cloning/copying ComfyUI with TPU support.
4. Model downloading (Qwen-Image UNet, CLIP, VAE per Neptune Motion README).
5. Launching the 8-core cluster (`python tools/tpu_orchestrator.py start --cores 8`).
6. Dispatching 8 parallel prompts across all cores.
7. Saving images and `benchmark_report.json` to `/kaggle/working/output/`.
8. Clean cluster shutdown.

- [ ] **Step 4: Create automation script `tools/run_kaggle_benchmark.sh`**

Create executable script `tools/run_kaggle_benchmark.sh` that wraps:
1. Kernel push: `kaggle kernels push -p kaggle/tpu_comfyui_benchmark`
2. Polling loop: checks `kaggle kernels status devtrung/comfyui-tpu-8core-benchmark` every 30 seconds.
3. Output download: `kaggle kernels output devtrung/comfyui-tpu-8core-benchmark -p out/kaggle_tpu/`
4. Summarizing results and confirming 8 generated images.

- [ ] **Step 5: Commit**

```bash
git add kaggle/ workflows/ tools/run_kaggle_benchmark.sh
git commit -m "feat(kaggle): create 8-core TPU benchmark notebook and CLI push/monitor runner"
```

---

### Task 7: End-to-End Verification & Kaggle Execution

**Files:**
- Output directory: `out/kaggle_tpu/`

**Interfaces:**
- Consumes: `tools/run_kaggle_benchmark.sh` and Kaggle API
- Produces: Verified 8-core parallel inference logs and generated images

- [ ] **Step 1: Run local test suite**

Run: `/home/devtrung/Projects/neptune-motion/.venv/bin/python -m unittest discover -s tests -p "test_tpu_*.py"`
Expected: All TPU unit tests pass.

- [ ] **Step 2: Trigger Kaggle benchmark execution via CLI**

Run: `bash tools/run_kaggle_benchmark.sh` (or `kaggle kernels push -p kaggle/tpu_comfyui_benchmark`)

- [ ] **Step 3: Monitor until kernel completion**

Run: `kaggle kernels status devtrung/comfyui-tpu-8core-benchmark` until status is `complete`.

- [ ] **Step 4: Download and verify artifacts**

Run: `kaggle kernels output devtrung/comfyui-tpu-8core-benchmark -p out/kaggle_tpu/`
Verify that `benchmark_report.json` and 8 image files exist in `out/kaggle_tpu/`.

- [ ] **Step 5: Final review and commit**

```bash
git status
git commit -am "chore: complete Google Cloud TPU multi-core verification and benchmark suite"
```
