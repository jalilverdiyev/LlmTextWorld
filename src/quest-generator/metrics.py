"""Performance metrics: background CPU / memory / GPU sampling and metrics logging.

`ResourceMonitor` samples resource usage while a request runs. `MetricsLogger` logs a
human-readable summary and appends the full metrics record as one JSON line to a file.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

import psutil

log = logging.getLogger("metrics")

DEFAULT_METRICS_LOG = "metrics.jsonl"
MB = 1024 * 1024


class GpuReader:
    """Reads GPU utilisation and memory via pynvml, falling back to nvidia-smi."""

    def __init__(self) -> None:
        self._nvml = None
        self._handles: list[Any] = []
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:
            self._nvml = None
        self._smi = shutil.which("nvidia-smi") if self._nvml is None else None
        self.available = bool(self._handles) or self._smi is not None

    def read(self) -> Optional[dict[str, float]]:
        """Returns {'util_percent': max across GPUs, 'mem_used_mb': sum across GPUs}."""
        try:
            if self._nvml is not None:
                utils, mems = [], []
                for h in self._handles:
                    utils.append(self._nvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    mems.append(self._nvml.nvmlDeviceGetMemoryInfo(h).used / MB)
                return {"util_percent": max(utils), "mem_used_mb": sum(mems)}
            if self._smi is not None:
                out = subprocess.run(
                    [self._smi, "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, check=True,
                ).stdout
                rows = [[float(x) for x in line.split(",")] for line in out.strip().splitlines()]
                return {"util_percent": max(r[0] for r in rows),
                        "mem_used_mb": sum(r[1] for r in rows)}
        except Exception:
            return None
        return None


class ResourceMonitor:
    """Samples CPU, memory and GPU usage in a background thread.

    Tracks this process plus every process named like `process_match` (the Ollama
    server does the actual inference, so its usage is what matters).
    """

    def __init__(self, interval: float = 0.5, process_match: str = "ollama") -> None:
        self.interval = interval
        self.process_match = process_match.lower()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._procs: dict[int, psutil.Process] = {}
        self._gpu = GpuReader()
        self.samples: list[dict[str, float]] = []

    def _tracked(self) -> list[psutil.Process]:
        alive: dict[int, psutil.Process] = {}
        me = os.getpid()
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if p.pid == me or self.process_match in (p.info["name"] or "").lower():
                    alive[p.pid] = self._procs.get(p.pid) or p
            except psutil.Error:
                continue
        self._procs = alive
        return list(alive.values())

    def _sample(self) -> None:
        cpu = rss = 0.0
        for p in self._tracked():
            try:
                cpu += p.cpu_percent(interval=None)
                rss += p.memory_info().rss
            except psutil.Error:
                continue
        sample = {
            "t": time.perf_counter(),
            "proc_cpu_percent": cpu,
            "proc_rss_mb": rss / MB,
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "system_mem_used_mb": psutil.virtual_memory().used / MB,
        }
        gpu = self._gpu.read() if self._gpu.available else None
        if gpu:
            sample["gpu_util_percent"] = gpu["util_percent"]
            sample["gpu_mem_used_mb"] = gpu["mem_used_mb"]
        self.samples.append(sample)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._tracked()
        for p in self._procs.values():  # prime cpu_percent baselines
            try:
                p.cpu_percent(interval=None)
            except psutil.Error:
                pass
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._sample()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        def stat(key: str) -> Optional[dict[str, float]]:
            vals = [s[key] for s in self.samples if key in s]
            if not vals:
                return None
            return {"avg": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2)}

        keys = ["proc_cpu_percent", "proc_rss_mb", "system_cpu_percent", "system_mem_used_mb",
                "gpu_util_percent", "gpu_mem_used_mb"]
        out: dict[str, Any] = {k: stat(k) for k in keys}
        out["samples"] = len(self.samples)
        out["gpu_available"] = self._gpu.available
        return out


class MetricsLogger:
    """Logs a one-line summary of a metrics record and appends it to a JSONL file."""

    def __init__(self, path: Optional[str] = DEFAULT_METRICS_LOG) -> None:
        self.path = path

    def record(self, metrics: dict[str, Any]) -> None:
        r = metrics["resources"]

        def fmt(key: str, unit: str) -> str:
            s = r.get(key)
            return f"avg {s['avg']}{unit} / peak {s['peak']}{unit}" if s else "n/a"

        log.info(
            "model=%s latency=%.2fs ttft=%s tokens/s=%s out_tokens=%s | cpu(proc) %s | rss %s | gpu %s | gpu mem %s",
            metrics["model"], metrics["latency_s"], metrics["ttft_s"], metrics["tokens_per_s"],
            metrics["output_tokens"], fmt("proc_cpu_percent", "%"), fmt("proc_rss_mb", "MB"),
            fmt("gpu_util_percent", "%"), fmt("gpu_mem_used_mb", "MB"),
        )
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")
