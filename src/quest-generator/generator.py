"""Builds an LLM prompt from a TextWorld game file and (optionally) generates new quests.

Usage:
    # Only print the prompt
    python generator.py game.json --prompt-only

    # Generate quests with a local Ollama model and write a new game file
    python generator.py game.json --model llama3.2 --num-quests 3 --output game.generated.json

Entity types and predicate wording live in entity_types.json (override with --types).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

import config
import inferer
import validator

log = logging.getLogger("generator")

DEFAULT_TYPES_PATH = Path(__file__).with_name("entity_types.json")
SPECIAL_NAMES = {"P": "the player", "I": "the inventory"}
COMMAND_RE = re.compile(r'^\s*([\w/]+)\s*::\s*"([^"]+)"\s*::\s*"[^"]*"\s*;', re.MULTILINE)
PARENT_RE = re.compile(r"^type\s+(\w+)\s*:\s*(\w+)\s*\{", re.MULTILINE)


class GameFormatError(ValueError):
    pass


def load_json(path: str | Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise GameFormatError(f"Cannot read JSON file {path}: {e}") from e


class TypeDefinitions:
    """Meaning of entity types and wording of predicates (editable in entity_types.json)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.types: dict[str, dict[str, str]] = data.get("types", {})
        self.predicates: dict[str, list[dict[str, Any]]] = data.get("predicates", {})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_TYPES_PATH) -> "TypeDefinitions":
        return cls(load_json(path))

    def label(self, type_: str) -> str:
        return self.types.get(type_, {}).get("label", type_)

    def description(self, type_: str) -> str:
        return self.types.get(type_, {}).get("description", "")

    def describe_fact(self, name: str, args: list[tuple[str, str]], display) -> str:
        """args are (entity_id, type) pairs; display maps an entity id to readable text."""
        types = [t for _, t in args]
        for rule in self.predicates.get(name, []):
            pattern = rule["args"]
            if len(pattern) == len(types) and all(p in ("*", t) for p, t in zip(pattern, types)):
                return rule["text"].format(*[display(eid) for eid, _ in args])
        return f"{name}({', '.join(display(eid) for eid, _ in args)})"


class Game:
    def __init__(self, data: dict[str, Any]) -> None:
        for key in ("world", "infos"):
            if key not in data:
                raise GameFormatError(f"Game file is missing required key '{key}'")
        self.data = data
        self.world: list[dict[str, Any]] = data["world"]
        self.entities: dict[str, dict[str, Any]] = {}
        for entry in data["infos"]:
            if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[1], dict)):
                raise GameFormatError(f"Malformed 'infos' entry: {entry!r}")
            self.entities[entry[0]] = entry[1]
        self.quests: list[dict[str, Any]] = data.get("quests", [])

    def display(self, entity_id: str) -> str:
        if entity_id in SPECIAL_NAMES:
            return SPECIAL_NAMES[entity_id]
        name = self.entities.get(entity_id, {}).get("name")
        return f"the {name}" if name else entity_id

    def entity_type(self, entity_id: str) -> str:
        return self.entities.get(entity_id, {}).get("type", entity_id.split("_")[0])

    def command_templates(self) -> list[str]:
        """Command templates (e.g. 'lock {c} with {k}') from the game's knowledge base that
        are applicable to the entity types present in this game."""
        logic = self.data.get("KB", {}).get("logic", "")
        parents = dict(PARENT_RE.findall(logic))

        present: set[str] = set()
        for eid in self.entities:
            t: Optional[str] = self.entity_type(eid)
            while t and t not in present:
                present.add(t)
                t = parents.get(t)

        rooms = sum(1 for eid in self.entities if self.entity_type(eid) == "r")
        seen: dict[str, None] = {}
        for _, template in COMMAND_RE.findall(logic):
            placeholders = set(re.findall(r"\{(\w+)\}", template))
            if not placeholders <= present or (template.startswith("go ") and rooms < 2):
                continue
            seen.setdefault(template)
        return list(seen)

    def world_facts(self) -> list[tuple[str, list[tuple[str, str]]]]:
        facts = []
        for fact in self.world:
            args = [(a["name"], a["type"]) for a in fact.get("arguments", [])]
            facts.append((fact["name"], args))
        return facts


def clean_description(desc: Optional[str]) -> str:
    """Drops Inform7 conditional text like '[if c_0 is open]...[end if]' and collapses whitespace."""
    if not desc:
        return ""
    desc = re.sub(r"\[if [^\]]*\].*?\[end if\]", "", desc, flags=re.DOTALL)
    desc = re.sub(r"\[[^\]]*\]", "", desc)
    return re.sub(r"\s+", " ", desc).strip()


def build_prompt(game: Game, types: TypeDefinitions, num_quests: int = 1) -> str:
    present_types = list(dict.fromkeys(game.entity_type(eid) for eid in game.entities))
    lines: list[str] = []

    lines += [
        "You design quests for a text-based adventure game (TextWorld).",
        f"Create {num_quests} new, distinct quest(s) that can be completed in the world described below.",
        "",
        "## Entity types",
    ]
    for t in present_types:
        lines.append(f"- {t}: {types.label(t)}" + (f" - {types.description(t)}" if types.description(t) else ""))

    lines += ["", "## Entities"]
    for eid, info in game.entities.items():
        if eid in SPECIAL_NAMES:
            continue
        line = f"- {eid} ({types.label(game.entity_type(eid))}): \"{info.get('name') or eid}\""
        desc = clean_description(info.get("desc"))
        if desc:
            line += f" - {desc}"
        lines.append(line)

    lines += ["", "## Initial world state"]
    for name, args in game.world_facts():
        lines.append("- " + types.describe_fact(name, args, game.display))

    templates = game.command_templates()
    if templates:
        lines += ["", "## Available commands",
                  "Use ONLY these command forms; replace {x} with the exact entity name (without 'the'):"]
        lines += [f"- {t}" for t in templates]

    if game.quests:
        lines += ["", "## Existing quests (new quests must differ from these)"]
        for q in game.quests:
            lines.append(f"- {q.get('desc', '')}")
            lines.append("  commands: " + json.dumps(q.get("commands", [])))

    lines += [
        "",
        "## Rules",
        "- `commands` is the ordered list of commands the player types, starting from the initial world state, to complete the quest.",
        "- Every command must be valid in the world state reached by the previous commands (be in the right room, hold required items, open closed containers, etc.).",
        "- Only use entities that exist above, written with their exact names.",
        "- `desc` is a short, player-facing description of the goal; do not reveal the commands.",
        "",
        "## Output format",
        "Respond with a single JSON object and nothing else:",
        '{"quests": [{"desc": "<goal description>", "commands": ["<command 1>", "<command 2>"]}]}',
    ]
    return "\n".join(lines)


def extract_json(text: str) -> Any:
    """Parses JSON from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", text):
        try:
            return decoder.raw_decode(text[m.start():])[0]
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON found in model output")


def parse_quests(text: str) -> list[dict[str, Any]]:
    data = extract_json(text)
    if isinstance(data, dict):
        data = data["quests"] if isinstance(data.get("quests"), list) else [data]
    if not isinstance(data, list):
        raise ValueError("Model output is not a list of quests")
    return [q for q in data if isinstance(q, dict)]


def make_quest(desc: str, commands: list[str]) -> dict[str, Any]:
    """Quest in the game-file schema. win_events are left empty for now."""
    return {
        "desc": desc,
        "reward": 1,
        "commands": commands,
        "win_events": [],
        "fail_events": [],
        "optional": False,
        "repeatable": False,
    }


def collect_valid_quests(raw_quests: list[dict[str, Any]], game: Game) -> list[dict[str, Any]]:
    quests = []
    for raw in raw_quests:
        candidate = make_quest(str(raw.get("desc", "")).strip(), raw.get("commands"))
        result = validator.validate_quest(game.data, candidate)
        if result.ok:
            quests.append(candidate)
        else:
            log.warning("Dropping invalid quest %r: %s", raw.get("desc"), "; ".join(result.errors))
    return quests


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", help="Path to the TextWorld game JSON file")
    parser.add_argument("--types", default=str(DEFAULT_TYPES_PATH),
                        help="Entity type / predicate definitions (default: entity_types.json)")
    parser.add_argument("--num-quests", type=int, default=1, help="Number of quests to ask for")
    parser.add_argument("--output", help="Output game file (default: <game>.generated.json)")
    parser.add_argument("--replace-quests", action="store_true",
                        help="Replace existing quests instead of appending")
    parser.add_argument("--prompt-only", action="store_true", help="Print the prompt and exit")
    parser.add_argument("--save-prompt", help="Also write the prompt to this file")
    inferer.add_inference_args(parser)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        game = Game(load_json(args.game))
        prompt = build_prompt(game, TypeDefinitions.load(args.types), args.num_quests)
    except GameFormatError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.save_prompt:
        Path(args.save_prompt).write_text(prompt, encoding="utf-8")
    if args.prompt_only:
        print(prompt)
        return 0
    try:
        result = inferer.inferer_from_args(args).generate(
            prompt, json_mode=True, label=Path(args.game).name)
    except (config.ConfigError, inferer.InferenceError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        quests = collect_valid_quests(parse_quests(result.text), game)
    except (ValueError, KeyError) as e:
        print(f"error: could not parse model output ({e}). Raw output:\n{result.text}", file=sys.stderr)
        return 1
    if not quests:
        print("error: the model produced no valid quests", file=sys.stderr)
        return 1

    output = copy.deepcopy(game.data)
    output["quests"] = quests if args.replace_quests else output.get("quests", []) + quests
    out_path = Path(args.output) if args.output else Path(args.game).with_suffix(".generated.json")
    out_path.write_text(json.dumps(output, indent=4), encoding="utf-8")
    log.info("Wrote %d new quest(s) to %s", len(quests), out_path)
    for q in quests:
        print(f"- {q['desc']}\n  {q['commands']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
