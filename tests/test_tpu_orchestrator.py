import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.tpu_orchestrator import TPUClusterManager

class TestTPUClusterManager(unittest.TestCase):
    def test_cluster_instance_specs(self):
        manager = TPUClusterManager(cores=8, base_port=8188)
        specs = manager.get_instance_specs()
        self.assertEqual(len(specs), 8)
        self.assertEqual(specs[0]["port"], 8188)
        self.assertEqual(specs[0]["core_id"], 0)
        self.assertEqual(specs[0]["env"]["TPU_VISIBLE_DEVICES"], "0")
        self.assertEqual(specs[7]["port"], 8195)
        self.assertEqual(specs[7]["core_id"], 7)
        self.assertEqual(specs[7]["env"]["TPU_VISIBLE_DEVICES"], "7")

    def test_start_and_stop_lifecycle_mock(self):
        manager = TPUClusterManager(cores=2, base_port=9000)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("os.killpg") as mock_killpg, \
             patch("os.getpgid", return_value=12345):
            manager.start()
            self.assertEqual(len(manager.processes), 2)
            self.assertEqual(manager.processes[0]["pid"], 12345)

            manager.stop()
            self.assertEqual(mock_killpg.call_count, 2)
            self.assertEqual(len(manager.processes), 0)

    def test_dispatch_parallel_mock(self):
        manager = TPUClusterManager(cores=2, base_port=8188)
        dummy_prompts = [{"1": {"inputs": {}}}, {"2": {"inputs": {}}}]

        def mock_dispatch(port, prompt, *args, **kwargs):
            return {
                "port": port,
                "prompt_id": "test-prompt-id",
                "images": ["image_1.png"],
                "duration_seconds": 1.5,
                "status": "success"
            }

        with patch.object(manager, "dispatch_prompt", side_effect=mock_dispatch):
            result = manager.dispatch_parallel(dummy_prompts)
            self.assertEqual(result["total_prompts"], 2)
            self.assertEqual(result["successful_prompts"], 2)
            self.assertEqual(len(result["results"]), 2)

if __name__ == "__main__":
    unittest.main()
