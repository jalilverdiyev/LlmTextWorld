"""Final validation of a quest-augmented TextWorld game file.

`validate()` is meant to run once all generation has finished and the final JSON is ready. It
does four things and returns a report holding a cleaned copy of the game:

1. Schema: the game JSON and every quest have the expected structure, reference existing
   entities and load in TextWorld.
2. Achievability: every quest can actually be played from the initial world state. Quests are
   replayed with TextWorld's logic engine; when that fails, `planner.py` tells *why*
   (provably unreachable goal, e.g. behind an unreachable room, vs. merely wrong commands).
   The game as a whole is also checked for a joint winning playthrough.
3. Soundness of `win_events`: their actions are valid instances of the knowledge-base rules,
   are executable in order, match the quest's commands, trigger the win condition (and no fail
   event on the way), and the condition is not already true at the start.
4. Duplicates: new quests repeating a pre-existing quest, another new quest, or being a mere
   fragment of one, or sharing another quest's goal, are removed.

Pre-existing quests (those the input file already had) are never removed, only reported.
Everything found is logged as a WARNING, so it ends up in the per-run log file.

Standalone usage:
    python validator.py game.json [--baseline original.json] [--output cleaned.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import config
import logs
import planner
from quest_builder import QuestBuilder, QuestBuildError, TextWorldUnavailableError

try:
    from textworld.generator.game import Event, Game, GameProgression
except ImportError:  # pragma: no cover - depends on the environment
    Event = Game = GameProgression = None

log = logging.getLogger("validator")

DEFAULT_PLAN_BUDGET = 5000
PLAN_TIMEOUT_S = 60.0
ENTITY_KEYS = ("id", "type", "name", "noun", "adj", "desc", "room_type", "definite", "indefinite", "synonyms")


# --------------------------------------------------------------------------- results

@dataclass
class Issue:
    severity: str  # "error": the quest is defective; "warning": worth knowing, not fatal
    quest: Optional[int]  # 1-based position in the validated file, None for game-level issues
    code: str
    message: str

    def __str__(self) -> str:
        where = f"quest {self.quest}" if self.quest else "game"
        return f"[{self.code}] {where}: {self.message}"


@dataclass
class Removal:
    quest: int
    desc: str
    reason: str


@dataclass
class ValidationReport:
    data: dict[str, Any]  # cleaned copy of the validated game
    issues: list[Issue] = field(default_factory=list)
    removed: list[Removal] = field(default_factory=list)
    checked: int = 0
    semantic: bool = True  # False when TextWorld was unavailable and only schema/text checks ran
    fatal: bool = False  # the game file itself is unusable

    @property
    def ok(self) -> bool:
        return not self.fatal

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def summary(self) -> str:
        head = (f"Validation: {self.checked} quest(s) checked, {len(self.removed)} removed, "
                f"{len(self.errors)} error(s), {len(self.issues) - len(self.errors)} warning(s)")
        lines = ["Validation FAILED: the game file is invalid" if self.fatal else head]
        if not self.semantic:
            lines.append("  (TextWorld unavailable: achievability and win_events checks were skipped)")
        lines += [f"  - {i}" for i in self.issues if i.quest is None]
        lines += [f"  - removed quest {r.quest} ({r.desc[:50]!r}): {r.reason}" for r in self.removed]
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers

def normalize_commands(commands: Sequence[str]) -> list[str]:
    return [" ".join(c.lower().split()) for c in commands]


def is_contiguous_part(whole: Sequence[str], part: Sequence[str]) -> bool:
    n = len(part)
    return n > 0 and any(list(whole[i:i + n]) == list(part) for i in range(len(whole) - n + 1))


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _desc(quest: Any) -> str:
    return str(quest.get("desc", "")) if isinstance(quest, dict) else ""


# --------------------------------------------------------------------------- 1. schema

def check_game_schema(data: Any) -> list[str]:
    """Structural problems of the game file as a whole (quests are checked separately)."""
    if not isinstance(data, dict):
        return ["the game file must be a JSON object"]
    errors: list[str] = []
    for key, typ in (("version", int), ("world", list), ("grammar", dict),
                     ("quests", list), ("infos", list), ("KB", dict)):
        if key not in data:
            errors.append(f"missing required key '{key}'")
        elif not isinstance(data[key], typ):
            errors.append(f"'{key}' must be of type {typ.__name__}")
    if errors:
        return errors

    for key in ("logic", "text_grammars_path"):
        if not isinstance(data["KB"].get(key), str):
            errors.append(f"KB.{key} must be a string")

    ids: set[str] = set()
    for i, entry in enumerate(data["infos"]):
        if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str)
                and isinstance(entry[1], dict)):
            errors.append(f"infos[{i}] must be [id, {{properties}}]")
            continue
        eid, info = entry
        for key in ("id", "type"):
            if not isinstance(info.get(key), str):
                errors.append(f"infos[{i}] ({eid}) must have a string '{key}'")
        if info.get("id") != eid:
            errors.append(f"infos[{i}]: key {eid!r} differs from its id {info.get('id')!r}")
        unknown = sorted(set(info) - set(ENTITY_KEYS))
        if unknown:
            errors.append(f"infos[{i}] ({eid}) has unknown properties {unknown}")
        ids.add(eid)

    for i, fact in enumerate(data["world"]):
        if not (isinstance(fact, dict) and isinstance(fact.get("name"), str)
                and isinstance(fact.get("arguments"), list)):
            errors.append(f"world[{i}] must be {{name, arguments}}")
            continue
        for arg in fact["arguments"]:
            if not (isinstance(arg, dict) and isinstance(arg.get("name"), str)
                    and isinstance(arg.get("type"), str)):
                errors.append(f"world[{i}] ({fact['name']}) has a malformed argument {arg!r}")
            elif arg["name"] not in ids:
                errors.append(f"world[{i}] ({fact['name']}) refers to unknown entity {arg['name']!r}")

    if "metadata" in data and not isinstance(data["metadata"], dict):
        errors.append("'metadata' must be an object")
    if data.get("objective") is not None and not isinstance(data["objective"], str):
        errors.append("'objective' must be a string")
    return errors


def check_quest_schema(quest: Any) -> list[str]:
    """Structural problems of one quest. `win_events` is mandatory and must be non-empty."""
    if not isinstance(quest, dict):
        return ["a quest must be a JSON object"]
    errors: list[str] = []
    if not (isinstance(quest.get("desc"), str) and quest["desc"].strip()):
        errors.append("'desc' must be a non-empty string")
    if not (_is_number(quest.get("reward")) and quest["reward"] >= 0):
        errors.append("'reward' must be a non-negative number")
    commands = quest.get("commands")
    if not (isinstance(commands, list) and commands
            and all(isinstance(c, str) and c.strip() for c in commands)):
        errors.append("'commands' must be a non-empty list of non-empty strings")
    for key in ("optional", "repeatable"):
        if key in quest and not isinstance(quest[key], bool):
            errors.append(f"'{key}' must be a boolean")
    for key in ("win_events", "fail_events"):
        events = quest.get(key)
        if not isinstance(events, list):
            errors.append(f"'{key}' must be a list")
            continue
        for i, ev in enumerate(events):
            if not (isinstance(ev, dict) and isinstance(ev.get("commands"), list)
                    and isinstance(ev.get("actions"), list) and isinstance(ev.get("condition"), dict)):
                errors.append(f"{key}[{i}] must have 'commands', 'actions' and 'condition'")
    if isinstance(quest.get("win_events"), list) and not quest["win_events"]:
        errors.append("'win_events' must not be empty (TextWorld cannot load a quest without them)")
    return errors


def _parse_events(quest: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """TextWorld `Event` objects of a quest; raises if TextWorld rejects any of them."""
    return ([Event.deserialize(e) for e in quest["win_events"]],
            [Event.deserialize(e) for e in quest.get("fail_events", [])])


# --------------------------------------------------------------------------- 2/3. semantics

class _Context:
    """The game's rules, entity names and initial state, shared by all semantic checks."""

    def __init__(self, builder: QuestBuilder, plan_budget: int) -> None:
        self.game = builder.game
        self.kb = self.game.kb
        self.rules = list(self.kb.rules.values())
        self.constants = self.kb.types.constants_mapping
        self.initial = builder.initial_state
        self.names = builder.names
        self.entities = set(self.game.infos)
        self.plan_budget = plan_budget

    def command(self, action: Any) -> str:
        """Canonical command of an action, derived from the knowledge base (not the JSON's template)."""
        mapping = self.kb.rules[action.name].match(action)
        template = self.kb.inform7_commands[action.name]
        return template.format(**{ph.name: self.names.get(v.name, v.name) for ph, v in mapping.items()})

    def fact(self, prop: Any) -> str:
        return f"{prop.name}({', '.join(self.names.get(a.name, a.name) for a in prop.arguments)})"


def _check_event(ctx: _Context, quest: dict[str, Any], event: Any, fail_events: list[Any],
                 label: str, first: bool) -> list[tuple[str, str, str]]:
    """Returns (severity, code, message) findings for one win event."""
    out: list[tuple[str, str, str]] = []

    def err(code: str, msg: str) -> None:
        out.append(("error", code, f"{label}: {msg}"))

    unknown = sorted({v.name for a in (*event.actions, event.condition) for v in a.variables} - ctx.entities)
    if unknown:
        err("unknown-entity", f"refers to entities that do not exist in the world: {unknown}")
        return out

    bad_rules = False
    for i, action in enumerate(event.actions, 1):
        rule = ctx.kb.rules.get(action.name)
        if rule is None:
            err("unknown-rule", f"step {i} uses unknown rule {action.name!r}")
            bad_rules = True
        elif rule.match(action) is None:
            err("invalid-action", f"step {i} is not a valid instance of rule {action.name!r} "
                                  "(its pre/postconditions do not match the game's logic)")
            bad_rules = True
    if bad_rules:
        return out

    canonical = [ctx.command(a) for a in event.actions]
    if normalize_commands(event.commands) != normalize_commands(canonical):
        err("commands-mismatch", f"commands {list(event.commands)} do not describe its actions {canonical}")
    if first and normalize_commands(quest.get("commands", [])) != normalize_commands(canonical):
        err("commands-mismatch", f"the quest's commands {quest.get('commands')} differ from what its "
                                 f"win_events perform {canonical}")

    if event.is_triggering(ctx.initial):
        err("trivial", "the win condition already holds in the initial world, so the quest is "
                       "completed without doing anything (irrelevant quest)")
        return out

    state = ctx.initial.copy()
    replayed = True
    for i, (action, command) in enumerate(zip(event.actions, canonical), 1):
        if not state.is_applicable(action):
            err("not-executable", f"step {i} ({command!r}) cannot be executed at that point of the playthrough")
            replayed = False
            break
        state.apply(action)
        if any(f.is_triggering(state) for f in fail_events):
            err("fail-event", f"step {i} ({command!r}) triggers one of the quest's fail events")
    if replayed and not event.is_triggering(state):
        err("not-triggered", "executing all actions does not satisfy the win condition")
        replayed = False

    if replayed:
        required = set(event.condition.preconditions)
        missing = [p for p in event.actions[-1].added if p not in required] if event.actions else []
        if missing:
            out.append(("warning", "weak-condition", f"{label}: the win condition does not require the effect "
                        f"of the last step ({', '.join(ctx.fact(p) for p in missing)})"))
        return out

    # The recorded actions fail: find out whether the *goal* is achievable at all.
    goal = list(event.condition.preconditions)
    unreachable = planner.unreachable_goal_facts(ctx.initial, goal, ctx.rules, ctx.constants)
    if unreachable:
        err("unreachable", "the goal can never be reached in this world; impossible: "
                           + ", ".join(ctx.fact(p) for p in unreachable))
        return out
    result = planner.find_plan(ctx.initial, goal, ctx.rules, ctx.constants,
                               max_expansions=ctx.plan_budget, timeout_s=PLAN_TIMEOUT_S)
    if result.found:
        plan = [ctx.command(a) for a in result.plan]
        err("wrong-actions", f"the goal is achievable ({len(plan)} steps: {plan}) but the recorded "
                             "actions are not executable")
    elif result.exhausted:
        err("unachievable", "the planner exhausted all reachable states: the goal cannot be reached")
    else:
        err("unproven", f"no plan found within the search budget ({result.expanded} states expanded); "
                        "the goal could not be shown to be achievable")
    return out


def _goal_key(win_events: list[Any]) -> frozenset:
    return frozenset(frozenset(str(p) for p in ev.condition.preconditions) for ev in win_events)


def _check_joint_winnability(ctx: _Context, data: dict[str, Any], quests: list[dict[str, Any]]) -> Optional[str]:
    """Returns a problem description if the quests cannot all be completed in one playthrough."""
    try:
        game = Game.deserialize({**data, "quests": copy.deepcopy(quests)})
        policy = GameProgression(game).winning_policy
    except Exception as e:  # noqa: BLE001 - TextWorld raises assorted errors on odd games
        return f"could not compute a joint winning policy ({e})"
    if policy is None:
        return "TextWorld finds no joint winning policy for the quests"
    state = ctx.initial.copy()
    done = [False] * len(game.quests)
    for step, action in enumerate(policy, 1):
        if not state.is_applicable(action):
            return f"the merged winning policy is not executable (step {step}: {ctx.command(action)!r})"
        state.apply(action)
        for i, quest in enumerate(game.quests):
            done[i] = done[i] or any(e.is_triggering(state) for e in quest.win_events)
    pending = [i + 1 for i, q in enumerate(game.quests) if not done[i] and not q.optional]
    if pending:
        return f"following the merged winning policy leaves quest(s) {pending} (of the checked ones) uncompleted"
    return None


# --------------------------------------------------------------------------- validate()

@dataclass
class _Entry:
    number: int  # 1-based position in the validated file
    quest: dict[str, Any]
    protected: bool
    removed: Optional[str] = None
    win: list[Any] = field(default_factory=list)
    fail: list[Any] = field(default_factory=list)
    schema_ok: bool = True
    semantic_ok: bool = True


def validate(data: dict[str, Any], existing: Optional[Sequence[dict[str, Any]]] = None,
             keep_existing: bool = True, plan_budget: int = DEFAULT_PLAN_BUDGET) -> ValidationReport:
    """Validates the final game file and returns a report whose `data` is the cleaned game.

    Args:
        data: the final game JSON (pre-existing plus newly generated quests).
        existing: quests the input file already had. New quests repeating them are removed.
        keep_existing: True if `data["quests"]` begins with `existing` (those are then
            protected: reported but never removed). False if the existing quests were
            replaced, so every quest in `data` is new.
        plan_budget: maximum states the planner expands per quest it has to investigate.
    """
    report = ValidationReport(data=copy.deepcopy(data))
    existing = list(existing or [])

    def issue(severity: str, quest: Optional[int], code: str, message: str) -> None:
        item = Issue(severity, quest, code, message)
        report.issues.append(item)
        log.warning("%s", item)

    schema_errors = check_game_schema(data)
    if schema_errors:
        report.fatal = True
        for e in schema_errors:
            issue("error", None, "schema", e)
        return report

    quests = data["quests"]
    n_protected = min(len(existing), len(quests)) if keep_existing else 0
    entries = [_Entry(i + 1, q, i < n_protected) for i, q in enumerate(quests)]
    report.checked = len(entries)

    def remove(entry: _Entry, reason: str) -> None:
        if entry.protected:
            issue("warning", entry.number, "kept", f"pre-existing quest kept despite: {reason}")
            return
        entry.removed = reason
        report.removed.append(Removal(entry.number, _desc(entry.quest), reason))
        log.warning("Removing quest %d (%r): %s", entry.number, _desc(entry.quest)[:60], reason)

    # TextWorld context: the game must load with its quests stripped.
    ctx: Optional[_Context] = None
    try:
        ctx = _Context(QuestBuilder(data), plan_budget)
    except TextWorldUnavailableError as e:
        report.semantic = False
        issue("warning", None, "no-textworld", f"semantic validation skipped: {e}")
    except QuestBuildError as e:
        report.fatal = True
        issue("error", None, "schema", str(e))
        return report

    # 1. per-quest schema
    for entry in entries:
        problems = check_quest_schema(entry.quest)
        if not problems and ctx is not None:
            try:
                entry.win, entry.fail = _parse_events(entry.quest)
            except Exception as e:  # noqa: BLE001 - TextWorld raises assorted errors on malformed events
                problems = [f"TextWorld cannot load its events: {e}"]
        if problems:
            entry.schema_ok = False
            remove(entry, "invalid schema: " + "; ".join(problems))

    # 4a. duplicates and fragments, judged on the commands
    baseline_refs: list[tuple[str, list[str], bool]] = []
    if not keep_existing:
        for i, q in enumerate(existing, 1):
            cmds = q.get("commands") if isinstance(q, dict) else None
            if isinstance(cmds, list) and all(isinstance(c, str) for c in cmds):
                baseline_refs.append((f"pre-existing quest {i}", normalize_commands(cmds), True))
    cmds_of = {e.number: normalize_commands(e.quest["commands"]) for e in entries if e.schema_ok}
    for entry in entries:
        if entry.removed is not None or not entry.schema_ok:
            continue
        mine = cmds_of[entry.number]
        others = list(baseline_refs)
        for other in entries:
            if other is entry or not other.schema_ok or other.removed is not None:
                continue
            label = f"{'pre-existing quest' if other.protected else 'quest'} {other.number}"
            others.append((label, cmds_of[other.number], other.protected or other.number < entry.number))
        for label, theirs, earlier in others:
            if entry.protected and not earlier:
                continue  # pre-existing quests are only compared with those before them
            if mine == theirs and earlier:
                remove(entry, f"already exists as {label}" if label.startswith("pre-existing")
                       else f"duplicate of {label}")
                break
            if mine != theirs and is_contiguous_part(theirs, mine):
                remove(entry, f"only a fragment of {label} (a split-up copy of a larger quest)")
                break

    if ctx is not None:
        # 2/3. achievability and win_events soundness
        for entry in entries:
            if entry.removed is not None or not entry.schema_ok:
                continue
            findings: list[tuple[str, str, str]] = []
            for i, event in enumerate(entry.win):
                findings += _check_event(ctx, entry.quest, event, entry.fail, f"win_events[{i}]", first=(i == 0))
            for severity, code, message in findings:
                issue(severity, entry.number, code, message)
            fatal = [f for f in findings if f[0] == "error"]
            if fatal:
                entry.semantic_ok = False
                remove(entry, "; ".join(f"[{code}] {message}" for _, code, message in fatal))

        # 4b. same goal as another quest, whatever the commands
        seen: dict[frozenset, str] = {}
        if not keep_existing:
            for i, q in enumerate(existing, 1):
                try:
                    seen.setdefault(_goal_key(_parse_events(q)[0]), f"pre-existing quest {i}")
                except Exception:  # noqa: BLE001 - unparseable baseline quests are simply not compared
                    continue
        for entry in entries:
            if entry.removed is not None or not entry.schema_ok or not entry.semantic_ok:
                continue
            key = _goal_key(entry.win)
            if key in seen:
                remove(entry, f"same goal as {seen[key]} (even though its commands differ)")
                if entry.removed is not None:
                    continue
            seen.setdefault(key, f"{'pre-existing quest' if entry.protected else 'quest'} {entry.number}")

        # game level: can everything be completed in one playthrough?
        playable = [e.quest for e in entries if e.removed is None and e.schema_ok and e.semantic_ok]
        if playable:
            problem = _check_joint_winnability(ctx, data, playable)
            if problem:
                issue("warning", None, "joint", problem)

    report.data["quests"] = [e.quest for e in entries if e.removed is None]
    log.info("%s", report.summary().splitlines()[0])
    return report


# --------------------------------------------------------------------------- legacy structural check

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_quest(game: dict[str, Any], quest: dict[str, Any]) -> ValidationResult:
    """Quick structural check of a freshly generated quest (before its win_events are built)."""
    errors: list[str] = []
    if not quest.get("desc"):
        errors.append("missing description")
    commands = quest.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    elif not all(isinstance(c, str) and c.strip() for c in commands):
        errors.append("every command must be a non-empty string")
    return ValidationResult(not errors, errors)


# --------------------------------------------------------------------------- CLI plumbing

def add_validation_args(parser: argparse.ArgumentParser, include_skip: bool = True) -> None:
    if include_skip:
        parser.add_argument("--skip-validations", action="store_true", default=None,
                            help="Skip the final validation (schema, achievability, win_events, "
                                 "duplicates); enabled by default [env: SKIP_VALIDATIONS]")
    parser.add_argument("--plan-budget", type=int,
                        help="Max states the planner expands per quest it has to investigate "
                             f"[env: VALIDATION_PLAN_BUDGET, default: {DEFAULT_PLAN_BUDGET}]")


def validation_settings(args: argparse.Namespace) -> tuple[bool, int]:
    """(enabled, plan_budget) resolved as flag > environment > .env > default."""
    skip = config.resolve(getattr(args, "skip_validations", None), "SKIP_VALIDATIONS", config.to_bool)
    budget = config.resolve(args.plan_budget, "VALIDATION_PLAN_BUDGET", int) or DEFAULT_PLAN_BUDGET
    if budget < 1:
        raise config.ConfigError("The plan budget must be at least 1")
    return not skip, budget


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game", help="Path to the game JSON file to validate")
    parser.add_argument("--baseline", help="The original game file: its quests count as pre-existing "
                                           "(never removed; new duplicates of them are)")
    parser.add_argument("--output", help="Write the cleaned game here (default: nothing is written)")
    parser.add_argument("--env-file", help="Path to a .env file")
    add_validation_args(parser, include_skip=False)
    logs.add_logging_args(parser)
    args = parser.parse_args(argv)

    try:
        config.load_dotenv(args.env_file)
        logs.setup_from_args("validator", args, argv)
        _, budget = validation_settings(args)
        data = json.loads(Path(args.game).read_text(encoding="utf-8"))
        existing: list[dict[str, Any]] = []
        keep = True
        if args.baseline:
            existing = json.loads(Path(args.baseline).read_text(encoding="utf-8")).get("quests", [])
            head = [normalize_commands(q.get("commands", [])) for q in data.get("quests", [])[:len(existing)]]
            keep = head == [normalize_commands(q.get("commands", [])) for q in existing]
    except (config.ConfigError, OSError, json.JSONDecodeError, AttributeError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    report = validate(data, existing, keep, budget)
    print(report.summary())
    if args.output and report.ok:
        Path(args.output).write_text(json.dumps(report.data, indent=4), encoding="utf-8")
        print(f"Cleaned game written to {args.output}")
    return 0 if report.ok and not report.errors else 1


if __name__ == "__main__":
    sys.exit(main())
