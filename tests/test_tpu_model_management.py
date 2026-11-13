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

if __name__ == "__main__":
    unittest.main()
