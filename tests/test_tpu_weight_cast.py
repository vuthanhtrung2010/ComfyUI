import unittest
from unittest.mock import MagicMock, patch
import torch

try:
    torch.utils.rename_privateuse1_backend("tpu")
except Exception:
    pass

class TestTPUWeightCast(unittest.TestCase):
    def test_supports_cast_tpu(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        self.assertTrue(mm.supports_cast(mock_dev, torch.bfloat16))
        self.assertTrue(mm.supports_cast(mock_dev, torch.float32))

    def test_pick_weight_dtype_tpu(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        chosen = mm.pick_weight_dtype(torch.bfloat16, torch.bfloat16, device=mock_dev)
        self.assertEqual(chosen, torch.bfloat16)

    def test_cast_to_cpu_to_tpu_concept(self):
        import comfy.model_management as mm
        tensor = torch.randn(10, 10, dtype=torch.float32)
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        mock_tpu_tensor = torch.zeros(10, 10, dtype=torch.bfloat16)
        with patch.object(torch, "empty_like", return_value=mock_tpu_tensor):
            out = mm.cast_to(tensor, dtype=torch.bfloat16, device=mock_dev)
            self.assertEqual(out.dtype, torch.bfloat16)

if __name__ == "__main__":
    unittest.main()
