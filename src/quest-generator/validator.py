"""Validation of generated quests.

Only a structural check exists for now. Semantic validation (simulating the commands
with a planner / the TextWorld engine against the game's world state) is planned for
a later iteration and should be added to `validate_quest`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate_quest(game: dict[str, Any], quest: dict[str, Any]) -> ValidationResult:
    errors: list[str] = []
    if not quest.get("desc"):
        errors.append("missing description")
    commands = quest.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("commands must be a non-empty list")
    elif not all(isinstance(c, str) and c.strip() for c in commands):
        errors.append("every command must be a non-empty string")
    return ValidationResult(not errors, errors)
