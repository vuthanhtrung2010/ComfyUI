import unittest
from unittest.mock import MagicMock, patch
import torch

try:
    torch.utils.rename_privateuse1_backend("tpu")
except Exception:
    pass

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

    def test_get_all_torch_devices_tpu(self):
        import comfy.model_management as mm
        with patch.object(mm, "is_tpu", return_value=True), \
             patch.object(mm, "is_nvidia", return_value=False), \
             patch.object(mm, "is_amd", return_value=False), \
             patch.object(mm, "is_intel_xpu", return_value=False), \
             patch.object(mm, "is_ascend_npu", return_value=False), \
             patch.object(mm, "is_mlu", return_value=False), \
             patch.object(mm, "cpu_state", mm.CPUState.GPU):
            devices = mm.get_all_torch_devices()
            self.assertTrue(any(d.type == "tpu" for d in devices))

    def test_get_torch_device_name_tpu(self):
        import comfy.model_management as mm
        mock_device = MagicMock()
        mock_device.type = "tpu"
        name = mm.get_torch_device_name(mock_device)
        self.assertIn("TPU", name)

    def test_text_encoder_dtype_tpu(self):
        import comfy.model_management as mm
        mock_device = MagicMock()
        mock_device.type = "tpu"
        with patch.object(mm, "should_use_bf16", return_value=True):
            dtype = mm.text_encoder_dtype(mock_device)
            self.assertEqual(dtype, torch.bfloat16)

    def test_pick_weight_dtype_tpu_fallback(self):
        import comfy.model_management as mm
        mock_device = MagicMock()
        mock_device.type = "tpu"
        with patch.object(mm, "supports_cast", return_value=False):
            dtype = mm.pick_weight_dtype(torch.float16, torch.bfloat16, mock_device)
            self.assertEqual(dtype, torch.bfloat16)

    def test_pytorch_attention_flash_attention_and_xformers_tpu(self):
        import comfy.model_management as mm
        with patch.object(mm, "is_tpu", return_value=True), \
             patch.object(mm, "ENABLE_PYTORCH_ATTENTION", True), \
             patch.object(mm, "cpu_state", mm.CPUState.GPU), \
             patch.object(mm, "is_nvidia", return_value=False), \
             patch.object(mm, "is_intel_xpu", return_value=False), \
             patch.object(mm, "is_ascend_npu", return_value=False), \
             patch.object(mm, "is_mlu", return_value=False), \
             patch.object(mm, "is_amd", return_value=False), \
             patch.object(mm, "is_ixuca", return_value=False):
            self.assertTrue(mm.pytorch_attention_flash_attention())
            self.assertFalse(mm.xformers_enabled())

    def test_tpu_dtype_and_capability_flags(self):
        import comfy.model_management as mm
        mock_device = MagicMock()
        mock_device.type = "tpu"
        self.assertTrue(mm.supports_dtype(mock_device, torch.float32))
        self.assertTrue(mm.supports_dtype(mock_device, torch.bfloat16))
        self.assertFalse(mm.supports_dtype(mock_device, torch.float16))
        self.assertFalse(mm.supports_fp64(mock_device))
        self.assertFalse(mm.supports_int8_compute(mock_device))

    def test_int8_tensorwise_linear_accuracy(self):
        import comfy.quant_ops
        from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
        torch.manual_seed(0)
        for per_ch, convrot in [(True, True), (True, False), (False, False)]:
            for x_shape in [(2, 17, 512), (17, 512)]:
                for use_bias in [False, True]:
                    w = torch.randn(384, 512, dtype=torch.bfloat16)
                    b = torch.randn(384, dtype=torch.bfloat16) if use_bias else None
                    qd, prm = TensorWiseINT8Layout.quantize(
                        w, is_weight=True, per_channel=per_ch, convrot=convrot, convrot_groupsize=256
                    )
                    qt = QuantizedTensor(qd, "TensorWiseINT8Layout", prm)
                    x = torch.randn(*x_shape, dtype=torch.bfloat16)
                    ref = torch.nn.functional.linear(x, qt.dequantize(), b)
                    got = comfy.quant_ops.int8_tensorwise_linear(x, qt, b)
                    self.assertEqual(got.shape, ref.shape)
                    self.assertEqual(got.dtype, ref.dtype)
                    cos = torch.nn.functional.cosine_similarity(
                        got.flatten().float(), ref.flatten().float(), dim=0
                    ).item()
                    self.assertGreater(cos, 0.9999)

    def test_qwen21_tpu_bucket_padding_and_strip(self):
        from comfy import sd1_clip
        from comfy.text_encoders import qwen3vl, qwen_image21

        clip_model = qwen_image21.QwenImage21Qwen3VLClipModel.__new__(qwen_image21.QwenImage21Qwen3VLClipModel)
        tpu_dev = MagicMock()
        tpu_dev.type = "tpu"
        raw_embeds = torch.randn(1, 70, 16)
        raw_mask = torch.ones(1, 70, dtype=torch.long)
        with patch.object(qwen3vl.Qwen3VLClipModel, "process_tokens", return_value=(raw_embeds, raw_mask, [70], [])):
            embeds, mask, num_tokens, _ = clip_model.process_tokens([[]], tpu_dev)
        self.assertEqual(tuple(embeds.shape), (1, 128, 16))
        self.assertEqual(tuple(mask.shape), (1, 128))
        self.assertEqual(int(mask[:, :70].sum()), 70)
        self.assertEqual(int(mask[:, 70:].sum()), 0)
        self.assertEqual(num_tokens, [70])

        te_model = qwen_image21.QwenImage21TEModel.__new__(qwen_image21.QwenImage21TEModel)
        te_model.clip = "qwen3vl_8b"
        te_model.qwen3vl_8b = MagicMock(image_spans=[])
        padded_out = torch.randn(1, 128, 16)
        padded_extra = {"attention_mask": mask}
        token_pairs = {"qwen3vl_8b": [[(151644, 1.0)] + [(100, 1.0)] * 10 + [(151644, 1.0)] + [(200, 1.0)] * 58]}
        with patch.object(sd1_clip.SD1ClipModel, "encode_token_weights", return_value=(padded_out, None, padded_extra)):
            out, _, extra = te_model.encode_token_weights(token_pairs)
        self.assertEqual(tuple(out.shape), (1, 59, 16))
        self.assertNotIn("attention_mask", extra)

    def test_tpu_cluster_manager_8_core_isolation(self):
        from tools.tpu_orchestrator import TPUClusterManager
        mgr = TPUClusterManager(cores=8, base_port=8188)
        specs = mgr.get_instance_specs()
        self.assertEqual(len(specs), 8)
        ports = [s["port"] for s in specs]
        self.assertEqual(ports, list(range(8188, 8196)))
        proc_ports = {s["env"]["TPU_PROCESS_PORT"] for s in specs}
        metric_ports = {p for s in specs for p in s["env"]["TPU_RUNTIME_METRICS_PORTS"].split(",")}
        self.assertEqual(len(proc_ports), 8)
        self.assertEqual(len(metric_ports), 32)
        for idx, s in enumerate(specs):
            self.assertEqual(s["env"]["TPU_VISIBLE_CHIPS"], str(idx))
            self.assertEqual(s["env"]["TPU_VISIBLE_DEVICES"], str(idx))
            self.assertEqual(s["env"]["TPU_CHIPS_PER_HOST_BOUNDS"], "1,1,1")

    def test_sdclip_compute_dtype_bf16_on_tpu(self):
        from comfy import sd1_clip
        m = sd1_clip.SDClipModel.__new__(sd1_clip.SDClipModel)
        m.dtype = torch.bfloat16
        tpu_dev = MagicMock()
        tpu_dev.type = "tpu"
        self.assertEqual(m._compute_dtype(tpu_dev), torch.bfloat16)
        self.assertEqual(m._compute_dtype(torch.device("cpu")), torch.float32)
        m.dtype = torch.float32
        self.assertEqual(m._compute_dtype(tpu_dev), torch.float32)

    def test_qwen21_dit_tpu_prefix_bucketing_equivalence(self):
        import comfy.ops
        import comfy.model_management as mm
        from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

        torch.manual_seed(42)
        model = QwenImage21Transformer2DModel(
            in_channels=16, out_channels=16, num_layers=2,
            attention_head_dim=32, num_attention_heads=4,
            context_in_dim=64, mlp_ratio=2, axes_dims_rope=(8, 12, 12),
            dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init,
        )
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        x = torch.randn(1, 16, 4, 4, dtype=torch.float32)
        t1 = torch.tensor([0.8], dtype=torch.float32)
        t2 = torch.tensor([0.4], dtype=torch.float32)
        ctx = torch.randn(1, 37, 64, dtype=torch.float32)

        fake_patcher = MagicMock()
        fake_patcher.is_dynamic.return_value = False
        fake_patcher.get_free_memory.return_value = 10**10
        model.current_patcher = fake_patcher

        with torch.no_grad():
            model.reset_prefix_cache(True)
            ref1 = model(x, t1, ctx)
            ref2 = model(x, t2, ctx)

            model.reset_prefix_cache(True)
            with patch.object(mm, "is_device_tpu", return_value=True):
                tpu1 = model(x, t1, ctx)
                tpu2 = model(x, t2, ctx)

        for a, b in [(ref1, tpu1), (ref2, tpu2)]:
            self.assertFalse(torch.isnan(b).any())
            cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
            self.assertGreater(cos, 0.999999)

    def test_get_free_memory_tpu_clamps_negative_reserved_delta(self):
        import comfy.model_management as mm
        tpu_dev = MagicMock()
        tpu_dev.type = "tpu"
        total = 16 * 1024 ** 3
        free = 6 * 1024 ** 3
        with patch.object(torch.accelerator, "get_memory_info", return_value=(free, total)), \
             patch.object(torch.accelerator, "memory_allocated", return_value=9 * 1024 ** 3), \
             patch.object(torch.accelerator, "memory_reserved", return_value=1 * 1024 ** 3):
            got_total, got_torch = mm.get_free_memory(tpu_dev, torch_free_too=True)
            self.assertEqual(got_torch, 0)
            self.assertEqual(got_total, free)

    def test_qwen21_te_batch_encode_and_pad(self):
        from comfy.text_encoders import qwen3vl, qwen_image21

        clip_model = qwen_image21.QwenImage21Qwen3VLClipModel.__new__(qwen_image21.QwenImage21Qwen3VLClipModel)
        tpu_dev = MagicMock()
        tpu_dev.type = "tpu"
        raw_embeds = torch.randn(2, 70, 16)
        raw_mask = torch.ones(2, 70, dtype=torch.long)
        raw_mask[0, 50:] = 0
        with patch.object(qwen3vl.Qwen3VLClipModel, "process_tokens", return_value=(raw_embeds, raw_mask, [50, 70], [])):
            embeds, mask, num_tokens, _ = clip_model.process_tokens([[], []], tpu_dev)
        self.assertEqual(tuple(embeds.shape), (2, 128, 16))
        self.assertEqual(tuple(mask.shape), (2, 128))
        self.assertEqual(int(mask[0].sum()), 50)
        self.assertEqual(int(mask[1].sum()), 70)

        te_model = qwen_image21.QwenImage21TEModel.__new__(qwen_image21.QwenImage21TEModel)
        te_model.clip = "qwen3vl_8b"
        te_model.clip_name = "qwen3vl_8b"
        fake_clip = MagicMock()
        fake_out = torch.randn(2, 128, 16)
        fake_clip.encode.return_value = (fake_out, None, {"attention_mask": mask})
        te_model.qwen3vl_8b = fake_clip
        twp0 = {"qwen3vl_8b": [[(151644, 1.0)] + [(10, 1.0)] * 9 + [(151644, 1.0)] + [(20, 1.0)] * 39]}
        twp1 = {"qwen3vl_8b": [[(151644, 1.0)] + [(10, 1.0)] * 9 + [(151644, 1.0)] + [(30, 1.0)] * 59]}
        batched, _, extra = te_model.encode_token_weights_batch([twp0, twp1])
        self.assertEqual(tuple(batched.shape), (2, 60, 16))
        self.assertIn("attention_mask", extra)
        self.assertEqual(int(extra["attention_mask"][0].sum()), 40)
        self.assertEqual(int(extra["attention_mask"][1].sum()), 60)
        self.assertTrue(torch.equal(batched[0, :40], fake_out[0, 10:50]))
        self.assertTrue(torch.equal(batched[1, :60], fake_out[1, 10:70]))

    def test_qwen21_dit_tpu_batched_heterogeneous_equivalence(self):
        import comfy.ops
        import comfy.model_management as mm
        from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

        torch.manual_seed(42)
        model = QwenImage21Transformer2DModel(
            in_channels=16, out_channels=16, num_layers=2,
            attention_head_dim=32, num_attention_heads=4,
            context_in_dim=64, mlp_ratio=2, axes_dims_rope=(8, 12, 12),
            dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init,
        )
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.02)

        x0, x1 = torch.randn(1, 16, 4, 4), torch.randn(1, 16, 4, 4)
        t1, t2 = torch.tensor([0.8]), torch.tensor([0.4])
        ctx0, ctx1 = torch.randn(1, 23, 64), torch.randn(1, 59, 64)
        ctx_b = torch.zeros(2, 59, 64)
        ctx_b[0, :23], ctx_b[1, :59] = ctx0[0], ctx1[0]
        mask_b = torch.zeros(2, 59, dtype=torch.long)
        mask_b[0, :23], mask_b[1, :59] = 1, 1

        fake_patcher = MagicMock()
        fake_patcher.is_dynamic.return_value = False
        fake_patcher.get_free_memory.return_value = 10**10
        model.current_patcher = fake_patcher

        with torch.no_grad():
            model.reset_prefix_cache(True)
            r0_1, r0_2 = model(x0, t1, ctx0), model(x0, t2, ctx0)
            model.reset_prefix_cache(True)
            r1_1, r1_2 = model(x1, t1, ctx1), model(x1, t2, ctx1)
            model.reset_prefix_cache(True)
            with patch.object(mm, "is_device_tpu", return_value=True):
                b_1 = model(torch.cat([x0, x1]), torch.cat([t1, t1]), ctx_b, attention_mask=mask_b)
                b_2 = model(torch.cat([x0, x1]), torch.cat([t2, t2]), ctx_b, attention_mask=mask_b)

        for ref, got in [(r0_1, b_1[:1]), (r1_1, b_1[1:]), (r0_2, b_2[:1]), (r1_2, b_2[1:])]:
            self.assertFalse(torch.isnan(got).any())
            cos = torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
            self.assertGreater(cos, 0.999999)

    def test_tpu_compiled_w8a16_per_shape_cache(self):
        import comfy.quant_ops as qo

        seen_names = []

        def fake_compile(fn, backend=None):
            self.assertEqual(backend, "tpu")
            seen_names.append(fn.__code__.co_name)
            return fn

        with patch.object(qo, "_tpu_backend_available", return_value=True), \
             patch.object(qo, "_TPU_COMPILED_W8A16_CACHE", {}), \
             patch.object(qo, "_TPU_COMPILED_MAX_SHAPES", 3), \
             patch.object(torch, "compile", side_effect=fake_compile):
            f1 = qo._get_tpu_compiled_w8a16_fn(True, (1, 128, 3072), (3072, 3072), 64)
            f1_again = qo._get_tpu_compiled_w8a16_fn(True, (1, 128, 3072), (3072, 3072), 64)
            f2 = qo._get_tpu_compiled_w8a16_fn(True, (2, 128, 3072), (3072, 3072), 64)
            f3 = qo._get_tpu_compiled_w8a16_fn(False, (4, 128, 3072), (3072, 3072), 0)
            f4 = qo._get_tpu_compiled_w8a16_fn(True, (6, 128, 3072), (3072, 3072), 64)

        self.assertIs(f1, f1_again)
        self.assertIsNot(f1, f2)
        self.assertIsNot(f2, f3)
        self.assertIsNone(f4)
        self.assertEqual(len(set(seen_names)), 3)

if __name__ == "__main__":
    unittest.main()
