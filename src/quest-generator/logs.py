"""Per-run logging.

Every run writes its own file `<LOG_DIR>/<timestamp>_<script>.log` with full DEBUG detail
(settings, prompt, the model's raw response before any processing, filtering decisions),
while the console only shows INFO and above.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import load_dotenv, resolve

DEFAULT_LOG_DIR = "logs"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

log = logging.getLogger("run")


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-dir",
                        help=f"Directory for per-run log files [env: LOG_DIR, default: {DEFAULT_LOG_DIR}]")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Console log level; the log file always records DEBUG [env: LOG_LEVEL, default: INFO]")


def block(title: str, body: str) -> str:
    """Formats multi-line content (prompt, model output) as a delimited block."""
    return f"{title}\n----- begin {title} -----\n{body}\n----- end {title} -----"


def _new_log_file(log_dir: Path, script: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1000):
        path = log_dir / f"{stamp}_{script}{'' if n == 0 else f'_{n}'}.log"
        try:
            path.touch(exist_ok=False)  # never share or overwrite another run's file
            return path
        except FileExistsError:
            continue
    raise OSError(f"Could not create a unique log file in {log_dir}")


def setup_logging(script: str, log_dir: Optional[str] = None, console_level: str = "INFO",
                  argv: Optional[list[str]] = None) -> Path:
    """Configures the root logger (console + a fresh per-run file) and returns the file path."""
    path = _new_log_file(Path(log_dir or DEFAULT_LOG_DIR), script)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter(FORMAT))
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(console)
    root.addHandler(file_handler)

    log.info("Logging to %s", path)
    log.debug("Run started: %s", " ".join([script] + list(sys.argv[1:] if argv is None else argv)))
    return path


def setup_from_args(script: str, args: argparse.Namespace, argv: Optional[list[str]] = None) -> Path:
    """Resolves LOG_DIR / LOG_LEVEL (flag > environment > .env > default) and sets up logging."""
    load_dotenv(args.env_file)
    return setup_logging(
        script,
        log_dir=resolve(args.log_dir, "LOG_DIR", str),
        console_level=resolve(args.log_level, "LOG_LEVEL", str) or "INFO",
        argv=argv,
    )
