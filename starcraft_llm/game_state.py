from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SupplySummary:
    used: int
    cap: int
    left: int


@dataclass(frozen=True)
class GameStateSummary:
    """Small observation payload for future strategy/LLM planning."""

    minerals: int
    vespene: int
    supply: SupplySummary
    workers: int
    townhalls: int
    army: dict[str, int]
    known_enemy_units: int
    game_time_seconds: float
    structures: dict[str, int] = field(default_factory=dict)


def game_state_summary_to_dict(summary: GameStateSummary) -> dict[str, Any]:
    return {
        "minerals": summary.minerals,
        "vespene": summary.vespene,
        "supply": {
            "used": summary.supply.used,
            "cap": summary.supply.cap,
            "left": summary.supply.left,
        },
        "workers": summary.workers,
        "townhalls": summary.townhalls,
        "army": dict(sorted(summary.army.items())),
        "structures": dict(sorted(summary.structures.items())),
        "known_enemy_units": summary.known_enemy_units,
        "game_time_seconds": summary.game_time_seconds,
    }


def game_state_summary_to_json(summary: GameStateSummary) -> str:
    return json.dumps(game_state_summary_to_dict(summary), ensure_ascii=False, indent=2)
