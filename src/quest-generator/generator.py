"""Builds an LLM prompt from a TextWorld game file and (optionally) generates new quests.

Usage:
    # Only print the prompt
    python generator.py game.json --prompt-only

    # Fill win_events of the quests already in the file from their commands (no LLM)
    python generator.py game.json --fill-win-events [--overwrite] [--output out.json]

    # Add --compile to either mode to rebuild the playable .z8 for the written JSON
    # (the compiled game contains the quests, so it must be rebuilt after they change)

    # Generate quests with a local Ollama model and write a new game file
    python generator.py game.json --model llama3.2 --num-quests 3 --output game.generated.json

Entity types and predicate wording live in entity_types.json (override with --types).
Each generated quest gets its `win_events` derived from its commands (see quest_builder.py),
which needs the `textworld` package, so run this inside the environment where it is installed.
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
import game_compiler
import inferer
import logs
from quest_builder import QuestBuilder, QuestBuildError, TextWorldUnavailableError, refresh_metadata
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
        f"Create exactly {num_quests} new quest(s) for the world described below.",
        "A quest is ONE complete multi-step objective. Its `commands` list holds the entire "
        "sequence of steps needed to finish it, usually 3 or more commands.",
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
        lines += ["", "## Existing quests",
                  "Each of these is ONE quest. Do not repeat them, and do not return them or "
                  "parts of them (single steps or sub-sequences) as new quests. "
                  "Your quests must pursue different goals."]
        for i, q in enumerate(game.quests, 1):
            lines.append(f"- Existing quest {i}: {q.get('desc', '')}")
            lines.append("  commands: " + json.dumps(q.get("commands", [])))

    lines += [
        "",
        "## Rules",
        f"- Return exactly {num_quests} quest(s) in the `quests` array, no more and no fewer.",
        "- Never split one quest into several quests: each individual step belongs inside the `commands` of the quest it is part of, not in a quest of its own.",
        "- `commands` is the ordered list of commands the player types, starting from the initial world state, to complete the quest.",
        "- Every command must be valid in the world state reached by the previous commands (be in the right room, hold required items, open closed containers, etc.).",
        "- Only use entities that exist above, written with their exact names.",
        "- `desc` is a short, player-facing description of the whole goal; do not reveal the commands.",
        "",
        "## Output format",
        f"Respond with a single JSON object and nothing else. The `quests` array must contain exactly {num_quests} object(s); each has one `desc` string and one `commands` array holding ALL steps of that quest:",
        '{"quests": [{"desc": "<goal of this quest>", "commands": ["<step 1>", "<step 2>", "<step 3>", "..."]}]}',
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


def make_quest(desc: str, commands: list[str], win_events: list[dict[str, Any]]) -> dict[str, Any]:
    """Quest in the game-file schema."""
    return {
        "desc": desc,
        "reward": 1,
        "commands": commands,
        "win_events": win_events,
        "fail_events": [],
        "optional": False,
        "repeatable": False,
    }


def _norm(commands: list[str]) -> list[str]:
    return [" ".join(c.lower().split()) for c in commands]


def _contains(whole: list[str], part: list[str]) -> bool:
    """True if `part` appears as a contiguous run inside `whole`."""
    n = len(part)
    return any(whole[i:i + n] == part for i in range(len(whole) - n + 1))


def collect_valid_quests(raw_quests: list[dict[str, Any]], game: Game, builder: QuestBuilder,
                         limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Turns raw model quests into complete game-file quests.

    Quests are kept only if they are well-formed and executable from the initial world state
    (the builder derives their `win_events`; commands are rewritten to their canonical form).
    Then candidates are dropped when their commands are a contiguous part of an existing quest
    (the model re-emitting or splitting a known quest) or of another candidate (one quest split
    into pieces; identical duplicates keep only the first). The count is capped at `limit`.
    """
    candidates = []
    for raw in raw_quests:
        desc = str(raw.get("desc", "")).strip()
        structural = validator.validate_quest(game.data, make_quest(desc, raw.get("commands"), []))
        if not structural.ok:
            log.warning("Dropping invalid quest %r: %s", raw.get("desc"), "; ".join(structural.errors))
            continue
        try:
            commands, win_event = builder.build_win_event(raw["commands"])
        except QuestBuildError as e:
            log.warning("Dropping quest %r: not executable from the initial state (%s)", desc, e)
            continue
        candidates.append(make_quest(desc, commands, [win_event]))

    existing = [_norm(q["commands"]) for q in game.quests if isinstance(q.get("commands"), list)]
    commands = [_norm(c["commands"]) for c in candidates]
    quests = []
    for i, candidate in enumerate(candidates):
        if any(_contains(e, commands[i]) for e in existing):
            log.warning("Dropping quest %r: repeats or splits an existing quest", candidate["desc"])
        elif any(j != i and _contains(commands[j], commands[i])
                 and (len(commands[j]) > len(commands[i]) or j < i)
                 for j in range(len(candidates))):
            log.warning("Dropping quest %r: duplicate or fragment of another generated quest",
                        candidate["desc"])
        else:
            quests.append(candidate)

    if limit is not None and len(quests) > limit:
        log.warning("Model returned %d usable quests but %d were requested; keeping the first %d",
                    len(quests), limit, limit)
        quests = quests[:limit]
    return quests


def fill_win_events(data: dict[str, Any], builder: QuestBuilder,
                    overwrite: bool = False) -> tuple[dict[str, Any], int, int, int]:
    """Fills `win_events` of existing quests from their `commands`, without any LLM.

    Quests that already have win_events are left alone unless `overwrite` is set. Quests whose
    commands cannot be executed are left unchanged and reported. Returns
    (new game data, filled, skipped, failed).
    """
    output = copy.deepcopy(data)
    filled = skipped = failed = 0
    for i, quest in enumerate(output.get("quests", []), 1):
        label = f"quest {i} ({str(quest.get('desc', ''))[:50]!r})"
        if quest.get("win_events") and not overwrite:
            skipped += 1
            log.info("Skipping %s: already has win_events", label)
            continue
        commands = quest.get("commands")
        if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
            failed += 1
            log.warning("Cannot fill %s: it has no valid 'commands' list", label)
            continue
        try:
            quest["commands"], win_event = builder.build_win_event(commands)
        except QuestBuildError as e:
            failed += 1
            log.warning("Cannot fill %s: %s", label, e)
            continue
        quest["win_events"] = [win_event]
        filled += 1
        log.info("Filled win_events of %s", label)
    return output, filled, skipped, failed


def refresh_derived_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Rebuilds `objective` and `metadata.walkthrough` from the quests; keeps the data on failure."""
    try:
        return refresh_metadata(data)
    except (QuestBuildError, TextWorldUnavailableError) as e:
        log.warning("objective/walkthrough not refreshed: %s", e)
        return data


def compile_output(path: Path) -> bool:
    """Compiles the game JSON at `path` to a .z8 next to it, reporting errors. True on success."""
    try:
        z8 = game_compiler.compile_game_file(path)
    except game_compiler.CompileError as e:
        log.error("Compilation failed: %s", e)
        print(f"error: {e}", file=sys.stderr)
        return False
    log.info("Compiled %s", z8)
    print(f"Compiled {z8}")
    return True


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
    parser.add_argument("--fill-win-events", action="store_true",
                        help="No LLM: fill the win_events of the file's existing quests from their commands")
    parser.add_argument("--overwrite", action="store_true",
                        help="With --fill-win-events: also rebuild quests that already have win_events")
    parser.add_argument("--keep-objective", action="store_true",
                        help="Do not rebuild the game's objective text and metadata.walkthrough from the quests")
    parser.add_argument("--compile", action="store_true",
                        help="After writing the output JSON, compile it to a playable .z8 next to it")
    parser.add_argument("--prompt-only", action="store_true", help="Print the prompt and exit")
    parser.add_argument("--save-prompt", help="Also write the prompt to this file")
    inferer.add_inference_args(parser)
    args = parser.parse_args(argv)

    try:
        config.load_dotenv(args.env_file)
        logs.setup_from_args("generator", args, argv)
        game = Game(load_json(args.game))
        if not args.fill_win_events:
            prompt = build_prompt(game, TypeDefinitions.load(args.types), args.num_quests)
    except (GameFormatError, config.ConfigError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    log.debug("Game %s: %d entities, %d existing quest(s), num_quests=%d",
              args.game, len(game.entities), len(game.quests), args.num_quests)
    out_path = Path(args.output) if args.output else Path(args.game).with_suffix(".generated.json")

    if args.fill_win_events:
        try:
            output, filled, skipped, failed = fill_win_events(
                game.data, QuestBuilder(game.data), args.overwrite)
        except (QuestBuildError, TextWorldUnavailableError) as e:
            log.error("Filling win_events failed: %s", e)
            print(f"error: {e}", file=sys.stderr)
            return 1
        refreshed = False
        if not args.keep_objective and not failed:
            updated = refresh_derived_fields(output)
            refreshed = (updated.get("objective"), updated.get("metadata", {}).get("walkthrough")) != (
                game.data.get("objective"), game.data.get("metadata", {}).get("walkthrough"))
            output = updated
        changed = bool(filled) or refreshed
        if changed:
            out_path.write_text(json.dumps(output, indent=4), encoding="utf-8")
        print(f"win_events: {filled} filled, {skipped} skipped, {failed} failed; "
              f"objective/walkthrough {'refreshed' if refreshed else 'unchanged'}"
              + (f" -> {out_path}" if changed else " (nothing written)"))
        compiled = True
        if args.compile and not failed:
            # With nothing to change, the game file itself is already the final result.
            compiled = compile_output(out_path if changed else Path(args.game))
        elif args.compile:
            print("error: not compiling because some quests could not be filled", file=sys.stderr)
            compiled = False
        return 0 if compiled and not failed else 1

    if args.save_prompt:
        Path(args.save_prompt).write_text(prompt, encoding="utf-8")
    if args.prompt_only:
        log.debug(logs.block("prompt", prompt))
        print(prompt)
        return 0
    try:
        # Created before inference so a missing TextWorld install fails fast, not after the LLM call.
        builder = QuestBuilder(game.data)
        result = inferer.inferer_from_args(args).generate(
            prompt, json_mode=True, label=Path(args.game).name)
    except (config.ConfigError, inferer.InferenceError, QuestBuildError, TextWorldUnavailableError) as e:
        log.error("Generation failed: %s", e)
        print(f"error: {e}", file=sys.stderr)
        return 1

    try:
        parsed = parse_quests(result.text)
        log.debug("Parsed %d quest object(s) from the model response", len(parsed))
        quests = collect_valid_quests(parsed, game, builder, args.num_quests)
    except (ValueError, KeyError) as e:
        log.error("Could not parse model output: %s", e)
        print(f"error: could not parse model output ({e}). Raw output:\n{result.text}", file=sys.stderr)
        return 1
    if not quests:
        log.error("The model produced no valid quests")
        print("error: the model produced no valid quests", file=sys.stderr)
        return 1

    output = copy.deepcopy(game.data)
    output["quests"] = quests if args.replace_quests else output.get("quests", []) + quests
    if not args.keep_objective:
        output = refresh_derived_fields(output)
    out_path.write_text(json.dumps(output, indent=4), encoding="utf-8")
    log.info("Wrote %d new quest(s) to %s", len(quests), out_path)
    log.debug(logs.block("accepted quests", json.dumps(quests, indent=2)))
    for q in quests:
        print(f"- {q['desc']}\n  {q['commands']}")
    if args.compile and not compile_output(out_path):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
