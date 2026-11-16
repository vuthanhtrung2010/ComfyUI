#!/usr/bin/env python3
"""Patch ComfyUI-GGUF with Neptune Motion Qwen-Image 2.1 support and TPU weight caching."""
import os
import sys

def patch_gguf(gguf_dir):
    loader_path = os.path.join(gguf_dir, "loader.py")
    if os.path.exists(loader_path):
        with open(loader_path, "rb") as f:
            b = f.read()
        b = b.replace(b'"qwen_image"', b'"qwen_image", "qwen_image21"')
        b = b.replace(b'raise ValueError(f"This model is not currently supported - ({e})")', b'arch_str = "qwen_image"')
        with open(loader_path, "wb") as f:
            f.write(b)
        print(f"[PATCH] Patched {loader_path} for qwen_image21 support.")

    ops_path = os.path.join(gguf_dir, "ops.py")
    if os.path.exists(ops_path):
        with open(ops_path, "rb") as f:
            src = f.read().decode("utf-8")
        
        # Check if already patched
        if "_cached_weight" not in src:
            # 1. Add _apply hook to GGMLLayer
            target_apply = "class GGMLLayer(torch.nn.Module):\n"
            replacement_apply = (
                "class GGMLLayer(torch.nn.Module):\n"
                "    def _apply(self, fn, recurse=True):\n"
                "        self._cached_weight = None\n"
                "        return super()._apply(fn, recurse=recurse)\n"
            )
            src = src.replace(target_apply, replacement_apply, 1)

            # 2. Add caching to get_weight
            target_get_weight = "    def get_weight(self, tensor, dtype):\n        if tensor is None:\n            return\n"
            replacement_get_weight = (
                "    def get_weight(self, tensor, dtype):\n"
                "        if tensor is None:\n"
                "            return\n"
                "\n"
                "        device = tensor.device\n"
                "        cached = getattr(self, '_cached_weight', None)\n"
                "        if cached is not None and cached.device == device and cached.dtype == dtype:\n"
                "            return cached\n"
            )
            src = src.replace(target_get_weight, replacement_get_weight, 1)

            # 3. Store in _cached_weight before return
            target_return = "        return weight\n\n    @torch_compiler_disable()"
            replacement_return = "        self._cached_weight = weight\n        return weight\n\n    @torch_compiler_disable()"
            src = src.replace(target_return, replacement_return, 1)

            with open(ops_path, "wb") as f:
                f.write(src.encode("utf-8"))
            print(f"[PATCH] Patched {ops_path} with TPU weight caching optimization.")
        else:
            print(f"[PATCH] {ops_path} already has weight caching enabled.")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "custom_nodes/ComfyUI-GGUF"
    patch_gguf(target_dir)
