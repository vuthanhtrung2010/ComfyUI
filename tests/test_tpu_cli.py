import os
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
