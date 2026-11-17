"""Local LLM inference through an Ollama server.

Each call measures latency, time to first token (TTFT), tokens/s and Ollama-reported
durations, and records resource usage via metrics.py.

Usage:
    python inferer.py --model llama3.2 --prompt "Say hi"
    python inferer.py --model llama3.2 --prompt-file prompt.txt --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import ConfigError, load_dotenv, resolve, to_bool
from metrics import DEFAULT_METRICS_LOG, MB, MetricsLogger, ResourceMonitor

log = logging.getLogger("inferer")

DEFAULT_HOST = "http://localhost:11434"

# CLI flag name -> (environment variable, type, Ollama `options` key)
OPTION_SPECS: dict[str, tuple[str, type, str]] = {
    "temperature": ("OLLAMA_TEMPERATURE", float, "temperature"),
    "top_p": ("OLLAMA_TOP_P", float, "top_p"),
    "top_k": ("OLLAMA_TOP_K", int, "top_k"),
    "repeat_penalty": ("OLLAMA_REPEAT_PENALTY", float, "repeat_penalty"),
    "seed": ("OLLAMA_SEED", int, "seed"),
    "num_ctx": ("OLLAMA_NUM_CTX", int, "num_ctx"),
    "max_tokens": ("OLLAMA_MAX_TOKENS", int, "num_predict"),
}


class InferenceError(RuntimeError):
    pass


@dataclass
class InferenceResult:
    text: str
    thinking: str
    metrics: dict[str, Any] = field(default_factory=dict)


class OllamaInferer:
    def __init__(self, model: str, host: str = DEFAULT_HOST,
                 options: Optional[dict[str, Any]] = None,
                 keep_alive: Optional[str] = None, think: Optional[bool] = None,
                 metrics_log: Optional[str] = DEFAULT_METRICS_LOG,
                 sample_interval: float = 0.5, timeout: float = 600.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.options = dict(options or {})
        self.keep_alive = keep_alive
        self.think = think
        self.metrics_logger = MetricsLogger(metrics_log)
        self.sample_interval = sample_interval
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]):
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            raise InferenceError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise InferenceError(
                f"Cannot reach Ollama at {self.host} ({e.reason}). Is `ollama serve` running?"
            ) from e

    def _model_vram_mb(self) -> Optional[float]:
        """VRAM used by the loaded model, as reported by Ollama (/api/ps)."""
        try:
            with urllib.request.urlopen(self.host + "/api/ps", timeout=5) as r:
                for m in json.load(r).get("models", []):
                    if self.model in (m.get("name"), m.get("model")) or \
                            m.get("name", "").split(":")[0] == self.model.split(":")[0]:
                        return round(m.get("size_vram", 0) / MB, 2)
        except Exception:
            pass
        return None

    def generate(self, prompt: str, system: Optional[str] = None, json_mode: bool = False,
                 options: Optional[dict[str, Any]] = None, label: str = "") -> InferenceResult:
        """`options` are Ollama options that override the instance defaults for this call."""
        merged = {**self.options, **(options or {})}
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": True}
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        if merged:
            payload["options"] = merged
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if self.think is not None:
            payload["think"] = self.think

        monitor = ResourceMonitor(self.sample_interval)
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        final: dict[str, Any] = {}
        ttft: Optional[float] = None
        chunks = 0

        monitor.start()
        start = time.perf_counter()
        try:
            with self._post("/api/generate", payload) as resp:
                for raw in resp:
                    if not raw.strip():
                        continue
                    chunk = json.loads(raw)
                    if "error" in chunk:
                        raise InferenceError(f"Ollama error: {chunk['error']}")
                    token = chunk.get("response") or ""
                    thought = chunk.get("thinking") or ""
                    if (token or thought) and ttft is None:
                        ttft = time.perf_counter() - start
                    if token:
                        text_parts.append(token)
                    if thought:
                        thinking_parts.append(thought)
                    chunks += bool(token or thought)
                    if chunk.get("done"):
                        final = chunk
        finally:
            total = time.perf_counter() - start
            resources = monitor.stop()

        ns = 1e-9
        eval_count = final.get("eval_count")
        eval_dur = (final.get("eval_duration") or 0) * ns
        metrics: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "model": self.model,
            "latency_s": round(total, 4),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "generation_s": round(total - ttft, 4) if ttft is not None else None,
            "load_s": round((final.get("load_duration") or 0) * ns, 4),
            "prompt_eval_s": round((final.get("prompt_eval_duration") or 0) * ns, 4),
            "prompt_tokens": final.get("prompt_eval_count"),
            "output_tokens": eval_count,
            "streamed_chunks": chunks,
            "tokens_per_s": round(eval_count / eval_dur, 2) if eval_count and eval_dur else None,
            "model_vram_mb": self._model_vram_mb(),
            "resources": resources,
        }
        self.metrics_logger.record(metrics)
        return InferenceResult("".join(text_parts), "".join(thinking_parts), metrics)


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Shared by inferer.py and generator.py so both expose the same options.

    Every default is None here; the effective value is resolved in `inferer_from_args`
    as CLI flag > environment variable / .env file > built-in default.
    """
    parser.add_argument("--env-file", help="Path to a .env file (default: ./.env or next to this script)")
    parser.add_argument("--model", help="Ollama model tag, e.g. llama3.2 [env: OLLAMA_MODEL]")
    parser.add_argument("--host", help=f"Ollama server URL [env: OLLAMA_HOST, default: {DEFAULT_HOST}]")
    for flag, (env_name, type_, _) in OPTION_SPECS.items():
        parser.add_argument("--" + flag.replace("_", "-"), type=type_, help=f"[env: {env_name}]")
    parser.add_argument("--keep-alive", help="How long Ollama keeps the model loaded, e.g. 5m [env: OLLAMA_KEEP_ALIVE]")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable thinking for models that support it [env: OLLAMA_THINK]")
    parser.add_argument("--timeout", type=float, help="Request timeout in seconds [env: OLLAMA_TIMEOUT, default: 600]")
    parser.add_argument("--metrics-log",
                        help=f"JSONL file metrics are appended to [env: METRICS_LOG, default: {DEFAULT_METRICS_LOG}]")
    parser.add_argument("--sample-interval", type=float,
                        help="Resource sampling interval in seconds [env: METRICS_SAMPLE_INTERVAL, default: 0.5]")


def inferer_from_args(args: argparse.Namespace) -> OllamaInferer:
    """Builds an inferer from CLI args, the environment and the .env file. Raises ConfigError."""
    load_dotenv(args.env_file)

    model = resolve(args.model, "OLLAMA_MODEL", str)
    if not model:
        raise ConfigError("No model given: use --model or set OLLAMA_MODEL")
    host = resolve(args.host, "OLLAMA_HOST", str) or DEFAULT_HOST
    if "://" not in host:
        host = f"http://{host}"

    options = {}
    for flag, (env_name, type_, option_key) in OPTION_SPECS.items():
        value = resolve(getattr(args, flag), env_name, type_)
        if value is not None:
            options[option_key] = value

    return OllamaInferer(
        model, host=host, options=options,
        keep_alive=resolve(args.keep_alive, "OLLAMA_KEEP_ALIVE", str),
        think=resolve(args.think, "OLLAMA_THINK", to_bool),
        metrics_log=resolve(args.metrics_log, "METRICS_LOG", str) or DEFAULT_METRICS_LOG,
        sample_interval=resolve(args.sample_interval, "METRICS_SAMPLE_INTERVAL", float) or 0.5,
        timeout=resolve(args.timeout, "OLLAMA_TIMEOUT", float) or 600.0,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_inference_args(parser)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt", help="Prompt text")
    src.add_argument("--prompt-file", help="File containing the prompt")
    parser.add_argument("--system", help="Optional system prompt")
    parser.add_argument("--json", action="store_true", help="Constrain output to valid JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()

    try:
        result = inferer_from_args(args).generate(prompt, system=args.system, json_mode=args.json)
    except (ConfigError, InferenceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
