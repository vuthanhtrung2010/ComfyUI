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

if __name__ == "__main__":
    unittest.main()
