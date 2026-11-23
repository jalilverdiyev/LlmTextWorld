"""Goal-directed planning over TextWorld's logic engine.

TextWorld itself only tracks progress along a recorded action list (`GameProgression`); it has
no search. This module adds the two pieces validation needs, working on the game's rules and
initial state:

* `relaxed_reachable_facts`: delete-relaxation reachability. Facts never become false, so the
  result over-approximates what can ever hold. A goal fact missing from it is *provably*
  unreachable (e.g. the player can never be in a room behind a locked door without any key).
* `find_plan`: bounded greedy best-first search for an action sequence reaching a goal. Finding
  one proves the goal achievable; running out of budget proves nothing.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

try:
    from textworld.logic import State
except ImportError:  # pragma: no cover - depends on the environment
    State = None


@dataclass
class PlanResult:
    found: bool
    plan: list[Any] = field(default_factory=list)
    expanded: int = 0
    exhausted: bool = False  # the whole reachable space was explored (no plan exists)
    out_of_budget: bool = False


def relaxed_reachable_facts(state: "State", rules: Iterable[Any], constants_mapping: Any = None,
                            max_rounds: int = 100) -> set[Any]:
    """Every fact that can hold at some point, ignoring that actions also delete facts."""
    rules = list(rules)
    relaxed = state.copy()
    known = set(relaxed.facts)
    for _ in range(max_rounds):
        new = set()
        for action in relaxed.all_applicable_actions(rules, constants_mapping):
            new.update(f for f in action.added if f not in known)
        if not new:
            break
        known |= new
        relaxed.add_facts(new)
    return known


def unreachable_goal_facts(state: "State", goal: Iterable[Any], rules: Iterable[Any],
                           constants_mapping: Any = None) -> list[Any]:
    """Goal facts that can never hold from `state` (empty if the goal may be reachable)."""
    reachable = relaxed_reachable_facts(state, rules, constants_mapping)
    return [g for g in goal if g not in reachable]


def find_plan(state: "State", goal: Iterable[Any], rules: Iterable[Any], constants_mapping: Any = None,
              max_expansions: int = 20000, timeout_s: float = 30.0) -> PlanResult:
    """Greedy best-first search (heuristic: number of unmet goal facts, ties broken by path length)."""
    goal = frozenset(goal)
    rules = list(rules)

    def key(s: "State") -> frozenset:
        return frozenset(s.facts)

    def h(facts: frozenset) -> int:
        return len(goal - facts)

    start_facts = key(state)
    counter = itertools.count()
    frontier: list[tuple[int, int, int, "State", tuple]] = [
        (h(start_facts), 0, next(counter), state.copy(), ())]
    seen = {start_facts}
    deadline = time.monotonic() + timeout_s
    expanded = 0

    while frontier:
        _, depth, _, current, path = heapq.heappop(frontier)
        if goal <= set(current.facts):
            return PlanResult(True, list(path), expanded)
        if expanded >= max_expansions or time.monotonic() > deadline:
            return PlanResult(False, expanded=expanded, out_of_budget=True)
        expanded += 1
        for action in current.all_applicable_actions(rules, constants_mapping):
            nxt = current.copy()
            nxt.apply(action)
            facts = key(nxt)
            if facts in seen:
                continue
            seen.add(facts)
            heapq.heappush(frontier, (h(facts), depth + 1, next(counter), nxt, path + (action,)))
    return PlanResult(False, expanded=expanded, exhausted=True)
