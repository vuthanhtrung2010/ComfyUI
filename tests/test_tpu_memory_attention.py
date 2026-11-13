import unittest
from unittest.mock import MagicMock, patch
import torch

try:
    torch.utils.rename_privateuse1_backend("tpu")
except Exception:
    pass

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

    def test_tpu_bf16(self):
        import comfy.model_management as mm
        mock_dev = MagicMock()
        mock_dev.type = "tpu"
        with patch.object(mm, "is_tpu", return_value=True):
            self.assertTrue(mm.should_use_bf16(device=mock_dev))

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
