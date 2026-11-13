#!/usr/bin/env python3
"""
TPU Cluster Orchestrator for ComfyUI.

Coordinates multi-instance execution of ComfyUI across multiple Google Cloud TPU
cores (e.g. TPU v3-8 with 8 independent cores). Manages process lifecycles,
health checking, and parallel prompt batch dispatching.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class TPUClusterManager:
    """Manages ComfyUI instances bound to individual TPU cores."""

    def __init__(
        self,
        cores: int = 8,
        base_port: int = 8188,
        host: str = "127.0.0.1",
        python_executable: Optional[str] = None,
        comfy_root: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
    ):
        self.cores = cores
        self.base_port = base_port
        self.host = host
        self.python_executable = python_executable or sys.executable
        self.comfy_root = comfy_root or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.extra_args = extra_args or []
        self.processes: List[Dict[str, Any]] = []

    def get_instance_specs(self) -> List[Dict[str, Any]]:
        """Return configuration specifications for each TPU core instance."""
        specs = []
        for core_id in range(self.cores):
            port = self.base_port + core_id
            env_vars = {
                "TPU_VISIBLE_DEVICES": str(core_id),
            }
            specs.append({
                "core_id": core_id,
                "port": port,
                "env": env_vars,
            })
        return specs

    def start(self, log_dir: Optional[str] = None) -> None:
        """Launch ComfyUI instances for each configured TPU core."""
        if self.processes:
            logging.warning("Cluster already has running instances. Call stop() first.")
            return

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        for spec in self.get_instance_specs():
            core_id = spec["core_id"]
            port = spec["port"]

            cmd = [
                self.python_executable,
                os.path.join(self.comfy_root, "main.py"),
                "--port", str(port),
                "--listen", self.host,
                "--tpu-device", str(core_id),
            ] + self.extra_args

            instance_env = os.environ.copy()
            instance_env.update(spec["env"])

            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL
            if log_dir:
                log_file = open(os.path.join(log_dir, f"instance_core_{core_id}.log"), "a", encoding="utf-8")
                stdout_dest = log_file
                stderr_dest = log_file

            logging.info("Starting ComfyUI on TPU Core %d (Port %d)...", core_id, port)
            proc = subprocess.Popen(
                cmd,
                cwd=self.comfy_root,
                env=instance_env,
                stdout=stdout_dest,
                stderr=stderr_dest,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            self.processes.append({
                "core_id": core_id,
                "port": port,
                "pid": proc.pid,
                "proc": proc,
            })

    def wait_ready(self, timeout: float = 180.0) -> bool:
        """Poll /system_stats on each instance until all report healthy."""
        start_time = time.time()
        logging.info("Waiting for all %d TPU instances to become ready (timeout=%ds)...", self.cores, timeout)

        pending = {p["port"]: p["core_id"] for p in self.processes}
        while pending and (time.time() - start_time) < timeout:
            for port in list(pending.keys()):
                url = f"http://{self.host}:{port}/system_stats"
                try:
                    req = urllib.request.Request(url, method="GET")
                    with urllib.request.urlopen(req, timeout=2.0) as resp:
                        if resp.status == 200:
                            logging.info("Core %d (Port %d) is READY.", pending[port], port)
                            del pending[port]
                except Exception:
                    pass
            if pending:
                time.sleep(2.0)

        if pending:
            logging.error("Timed out waiting for TPU instances on ports: %s", list(pending.keys()))
            return False

        logging.info("All %d TPU instances are operational and ready.", self.cores)
        return True

    def is_healthy(self) -> Dict[int, bool]:
        """Check health status of all instances."""
        status = {}
        for p in self.processes:
            port = p["port"]
            core_id = p["core_id"]
            url = f"http://{self.host}:{port}/system_stats"
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    status[core_id] = (resp.status == 200)
            except Exception:
                status[core_id] = False
        return status

    def stop(self) -> None:
        """Terminate all running cluster instances cleanly."""
        logging.info("Stopping %d TPU ComfyUI instances...", len(self.processes))
        for p in self.processes:
            proc: subprocess.Popen = p["proc"]
            try:
                if proc.poll() is None:
                    try:
                        if os.name != "nt":
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        else:
                            proc.terminate()
                    except Exception:
                        proc.terminate()
            except Exception as e:
                logging.debug("Error stopping process %s: %s", p["pid"], e)

        # Wait for shutdown
        for p in self.processes:
            proc: subprocess.Popen = p["proc"]
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass

        self.processes.clear()
        logging.info("All cluster instances stopped.")

    def dispatch_prompt(self, port: int, prompt: Dict[str, Any], timeout: float = 300.0) -> Dict[str, Any]:
        """Submit a single prompt to an instance and wait for image generation completion."""
        start_time = time.time()
        prompt_payload = json.dumps({"prompt": prompt}).encode("utf-8")
        prompt_url = f"http://{self.host}:{port}/prompt"

        req = urllib.request.Request(prompt_url, data=prompt_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            prompt_id = data.get("prompt_id")

        if not prompt_id:
            raise RuntimeError(f"Failed to queue prompt on port {port}: {data}")

        # Poll history
        history_url = f"http://{self.host}:{port}/history/{prompt_id}"
        images: List[str] = []
        while (time.time() - start_time) < timeout:
            try:
                with urllib.request.urlopen(history_url, timeout=5.0) as resp:
                    history = json.loads(resp.read().decode("utf-8"))
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                for img in node_output["images"]:
                                    images.append(img.get("filename", ""))
                        break
            except Exception:
                pass
            time.sleep(1.0)

        duration = time.time() - start_time
        return {
            "port": port,
            "prompt_id": prompt_id,
            "images": images,
            "duration_seconds": round(duration, 3),
            "status": "success" if images else "timeout",
        }

    def dispatch_parallel(self, prompts: List[Dict[str, Any]], timeout: float = 600.0) -> Dict[str, Any]:
        """Concurrently dispatch a list of prompts across all cluster instances."""
        num_prompts = len(prompts)
        ports = [self.base_port + (i % self.cores) for i in range(num_prompts)]

        logging.info("Dispatching %d prompts across %d TPU cores in parallel...", num_prompts, self.cores)
        start_time = time.time()
        results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cores) as executor:
            futures = [
                executor.submit(self.dispatch_prompt, ports[i], prompts[i], timeout)
                for i in range(num_prompts)
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    results.append({"status": "failed", "error": str(e)})

        total_duration = time.time() - start_time
        successful = [r for r in results if r.get("status") == "success"]

        throughput_ipm = (len(successful) / total_duration) * 60.0 if total_duration > 0 else 0.0

        return {
            "total_prompts": num_prompts,
            "successful_prompts": len(successful),
            "total_duration_seconds": round(total_duration, 3),
            "throughput_images_per_min": round(throughput_ipm, 2),
            "results": results,
        }


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Multi-Core TPU Cluster Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # start command
    start_parser = subparsers.add_parser("start", help="Start ComfyUI instances on TPU cores")
    start_parser.add_argument("--cores", type=int, default=8, help="Number of TPU cores (default: 8)")
    start_parser.add_argument("--base-port", type=int, default=8188, help="Base HTTP port (default: 8188)")
    start_parser.add_argument("--log-dir", type=str, default="logs/tpu_cluster", help="Directory for instance logs")

    # stop command
    subparsers.add_parser("stop", help="Stop all running TPU cluster instances")

    # status command
    status_parser = subparsers.add_parser("status", help="Check cluster health")
    status_parser.add_argument("--cores", type=int, default=8)
    status_parser.add_argument("--base-port", type=int, default=8188)

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run parallel inference benchmark")
    bench_parser.add_argument("--cores", type=int, default=8, help="Number of TPU cores (default: 8)")
    bench_parser.add_argument("--base-port", type=int, default=8188, help="Base HTTP port (default: 8188)")
    bench_parser.add_argument("--workflow", type=str, required=True, help="Path to workflow prompt JSON")
    bench_parser.add_argument("--prompts", type=int, default=8, help="Number of prompts to dispatch in parallel")
    bench_parser.add_argument("--output-report", type=str, default="benchmark_report.json", help="Path to save benchmark report")

    args = parser.parse_args()

    if args.command == "start":
        manager = TPUClusterManager(cores=args.cores, base_port=args.base_port)
        manager.start(log_dir=args.log_dir)
        ready = manager.wait_ready()
        if not ready:
            logging.error("Failed to start all TPU instances.")
            manager.stop()
            sys.exit(1)
        logging.info("Cluster ready. Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop()

    elif args.command == "benchmark":
        manager = TPUClusterManager(cores=args.cores, base_port=args.base_port)
        with open(args.workflow, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        prompt_graph = workflow.get("prompt", workflow)

        prompts = [prompt_graph for _ in range(args.prompts)]
        results = manager.dispatch_parallel(prompts)
        logging.info("Benchmark Complete: %s", json.dumps(results, indent=2))
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
