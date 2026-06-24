from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoveCommand:
    """A minimal strategy command: move one logical unit group to a map point."""

    unit: str
    x: float
    y: float


class StrategyParseError(ValueError):
    """Raised when a user strategy command is not supported by the MVP parser."""


def parse_strategy(text: str, default_unit: str = "worker") -> MoveCommand:
    """Parse the first supported natural command for the real-game MVP.

    Supported forms:
    - move worker 35 42
    - move marine 35 42
    - move 35 42

    The parser is intentionally tiny because the current milestone is proving that
    a command can control a real StarCraft II unit on screen.
    """

    parts = text.strip().split()
    if not parts:
        raise StrategyParseError("strategy command is empty")

    verb = parts[0].lower()
    if verb != "move":
        raise StrategyParseError("only 'move' is supported in this MVP")

    if len(parts) == 3:
        unit = default_unit
        x_text, y_text = parts[1:]
    elif len(parts) == 4:
        unit = normalize_unit(parts[1])
        x_text, y_text = parts[2:]
    else:
        raise StrategyParseError("use: move worker 35 42 or move 35 42")

    try:
        x = float(x_text)
        y = float(y_text)
    except ValueError as exc:
        raise StrategyParseError("move coordinates must be numbers") from exc

    return MoveCommand(unit=unit, x=x, y=y)


def normalize_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    aliases = {
        "scv": "worker",
        "probe": "worker",
        "drone": "worker",
        "worker": "worker",
        "marine": "marine",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported unit for MVP: {unit}")
    return aliases[normalized]
