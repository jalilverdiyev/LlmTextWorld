"""Compiles a TextWorld game JSON into a playable .z8 (and its Inform7 .ni source).

Quests, rewards and winning conditions are baked into the compiled game, so the .z8 must be
rebuilt whenever the JSON's quests change. The JSON is expected next to the .z8 with the same
file name, which is how TextWorld finds it when playing.

Usage:
    python game_compiler.py game.json [--output game.z8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import config
import logs

try:
    from textworld.generator import generate_inform7_source
    from textworld.generator.game import Game
    from textworld.generator.inform7.world2inform7 import compile_inform7_game
except ImportError as e:  # pragma: no cover - depends on the environment
    Game = None
    _IMPORT_ERROR: Optional[Exception] = e
else:
    _IMPORT_ERROR = None


class CompileError(RuntimeError):
    pass


def compile_game_file(json_path: str | Path, z8_path: Optional[str | Path] = None) -> Path:
    """Compiles `json_path` to a .z8 (default: same name next to the JSON). Returns the .z8 path."""
    if _IMPORT_ERROR is not None:
        raise CompileError(
            f"The 'textworld' package is required to compile games ({_IMPORT_ERROR}). "
            "Run inside the environment where TextWorld is installed.")
    json_path = Path(json_path)
    z8_path = Path(z8_path) if z8_path else json_path.with_suffix(".z8")
    if z8_path.suffix != ".z8":
        raise CompileError(f"Output must have a .z8 extension: {z8_path}")

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CompileError(f"Cannot read JSON file {json_path}: {e}") from e
    missing = [i for i, q in enumerate(data.get("quests", []), 1) if not q.get("win_events")]
    if missing:
        raise CompileError(
            f"Quest(s) {missing} have no win_events; TextWorld cannot load or compile them. "
            "Fill them first with: generator.py <game.json> --fill-win-events")
    try:
        source = generate_inform7_source(Game.deserialize(data))
        compile_inform7_game(source, str(z8_path))
    except Exception as e:
        raise CompileError(f"Compilation failed: {e}") from e
    if z8_path.stem != json_path.stem:
        # TextWorld looks for <name>.json next to <name>.z8 when playing.
        print(f"Note: TextWorld expects the JSON next to the game as {z8_path.with_suffix('.json').name}",
              file=sys.stderr)
    return z8_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", help="Path to the TextWorld game JSON file")
    parser.add_argument("--output", help="Output .z8 path (default: next to the JSON)")
    parser.add_argument("--env-file", help="Path to a .env file")
    logs.add_logging_args(parser)
    args = parser.parse_args(argv)
    try:
        logs.setup_from_args("game_compiler", args, argv)
        z8 = compile_game_file(args.game, args.output)
    except (CompileError, config.ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Compiled {z8} (and {z8.with_suffix('.ni').name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
