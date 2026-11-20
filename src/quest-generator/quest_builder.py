"""Builds TextWorld `win_events` for a list of text commands.

The commands are replayed from the game's initial state with TextWorld's logic engine and the
rules stored in the game file's knowledge base (no Inform7 compilation needed). For each
command the applicable grounded action producing that exact text is selected and applied.
The resulting actions are wrapped in a TextWorld `Event`, whose default trigger condition is
the set of postconditions of the last action, exactly as TextWorld does when it records a quest.

A command that no applicable action produces makes the quest impossible, so building also
acts as a first, semantic validation of the quest.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

try:
    import numpy as np
    from textworld.generator.game import Event, Game, GameProgression
    from textworld.generator.text_generation import describe_event
    from textworld.generator.text_grammar import Grammar
    from textworld.logic import Action
except ImportError as e:  # pragma: no cover - depends on the environment
    Event = Game = Action = None
    _IMPORT_ERROR: Exception | None = e
else:
    _IMPORT_ERROR = None


log = logging.getLogger("quest_builder")


class QuestBuildError(ValueError):
    """The commands cannot be executed from the initial world state."""


class TextWorldUnavailableError(RuntimeError):
    pass


def _normalize(command: str) -> str:
    return " ".join(command.lower().split())


class QuestBuilder:
    def __init__(self, game_data: dict[str, Any]) -> None:
        if _IMPORT_ERROR is not None:
            raise TextWorldUnavailableError(
                f"The 'textworld' package is required to build win_events ({_IMPORT_ERROR}). "
                "Run inside the environment where TextWorld is installed or `pip install textworld`."
            )
        try:
            # Quests are irrelevant here, and TextWorld refuses to load quests without win_events,
            # which is precisely the input this builder has to handle.
            self.game = Game.deserialize({**game_data, "quests": []})
        except Exception as e:
            raise QuestBuildError(f"Cannot load the game with TextWorld: {e}") from e
        self.rules = list(self.game.kb.rules.values())
        self.names = {eid: info.name or eid for eid, info in self.game.infos.items()}
        self.initial_state = self.game.world.state

    def _find_action(self, state, command: str) -> "Action":
        wanted = _normalize(command)
        for action in state.all_applicable_actions(self.rules):
            if _normalize(action.format_command(self.names)) == wanted:
                return action
        raise QuestBuildError(f"command {command!r} cannot be executed in the current world state")

    def build_actions(self, commands: list[str]) -> list["Action"]:
        state = self.initial_state.copy()
        actions = []
        for i, command in enumerate(commands, 1):
            try:
                action = self._find_action(state, command)
            except QuestBuildError as e:
                raise QuestBuildError(f"step {i}: {e}") from None
            state.apply(action)
            action.reverse_name = None  # TextWorld stores recorded event actions without it
            actions.append(action)
        return actions

    def build_win_event(self, commands: list[str]) -> tuple[list[str], dict[str, Any]]:
        """Returns (canonical commands, serialized win event) for the given commands."""
        if not commands:
            raise QuestBuildError("no commands")
        actions = self.build_actions(commands)
        canonical = [a.format_command(self.names) for a in actions]
        event = Event(actions=actions, commands=canonical)
        return canonical, json.loads(json.dumps(event.serialize()))


def refresh_metadata(game_data: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    """Returns a copy of the game with `objective` and `metadata.walkthrough` rebuilt from its quests.

    Both are derived data that TextWorld computes when quests are set: the walkthrough is the
    winning policy over all quests and the objective its description, generated from the game's
    text grammar. Entity names and descriptions are left untouched. The seed only picks the
    phrasing, and a fixed one keeps repeated runs identical.
    Every quest must have win_events, otherwise TextWorld cannot load the game.
    """
    if _IMPORT_ERROR is not None:
        raise TextWorldUnavailableError(str(_IMPORT_ERROR))
    try:
        game = Game.deserialize(game_data)
    except Exception as e:
        raise QuestBuildError(f"Cannot load the game with TextWorld: {e}") from e

    output = copy.deepcopy(game_data)
    metadata = output.setdefault("metadata", {})
    policy = GameProgression(game).winning_policy
    if not policy:
        metadata.pop("walkthrough", None)
        output["objective"] = ""
        return output

    names = {eid: info.name or eid for eid, info in game.infos.items()}
    metadata["walkthrough"] = [a.format_command(names) for a in policy]
    try:
        grammar = Grammar(game.grammar_options, rng=np.random.RandomState(seed), kb=game.kb)
        output["objective"] = describe_event(Event(policy), game, grammar)
    except Exception as e:
        # Same fallback TextWorld uses for games without a grammar.
        log.warning("Could not generate the objective text from the grammar (%s); "
                    "joining the quest descriptions instead", e)
        output["objective"] = "\nAND\n".join(q["desc"] for q in output["quests"] if q.get("desc"))
    return output
