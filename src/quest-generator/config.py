"""Minimal .env support (no third-party dependency).

Precedence used across the project: CLI flag > real environment variable > .env file > default.
Variables already present in the environment are never overridden by the .env file, and
empty values are treated as unset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

DEFAULT_ENV_FILE = ".env"
_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


class ConfigError(ValueError):
    pass


def _parse_value(raw: str) -> str:
    if raw[:1] in ("'", '"'):
        end = raw.find(raw[0], 1)
        if end != -1:
            return raw[1:end]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def load_dotenv(path: Optional[str] = None) -> Optional[Path]:
    """Loads KEY=VALUE pairs into os.environ without overriding existing variables.

    With no explicit path, `.env` is looked up in the working directory, then next to
    this file. An explicit path that doesn't exist is an error. Returns the file used.
    """
    if path:
        candidates = [Path(path)]
    else:
        candidates = [Path.cwd() / DEFAULT_ENV_FILE, Path(__file__).with_name(DEFAULT_ENV_FILE)]
    for candidate in candidates:
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                m = _LINE_RE.match(line)
                if m:
                    os.environ.setdefault(m.group(1), _parse_value(m.group(2)))
            return candidate
    if path:
        raise ConfigError(f"Env file not found: {path}")
    return None


def env_value(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value if value is not None and value.strip() != "" else None


def resolve(cli_value: Optional[T], env_name: str, cast: Callable[[str], T]) -> Optional[T]:
    """CLI value if given, else the (cast) environment value, else None."""
    if cli_value is not None:
        return cli_value
    raw = env_value(env_name)
    if raw is None:
        return None
    try:
        return cast(raw)
    except ValueError as e:
        raise ConfigError(f"Invalid value for {env_name}: {raw!r}") from e


def to_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(raw)
