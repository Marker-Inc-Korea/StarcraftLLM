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
class UnitObservation:
    """Ability-relevant observation for one known unit or structure.

    These fields are planner context only. In particular, energy, cargo, order,
    and form flags are not proof that an ability is currently available; the SC2
    executor must still query live available abilities immediately before
    issuing stateful commands.
    """

    unit: str
    alliance: str = "self"
    tag: int | str | None = None
    x: float | None = None
    y: float | None = None
    health: float | None = None
    health_max: float | None = None
    energy: float | None = None
    is_ready: bool | None = None
    is_flying: bool | None = None
    is_burrowed: bool | None = None
    is_loaded: bool | None = None
    is_idle: bool | None = None
    cargo_used: int | None = None
    cargo_max: int | None = None
    add_on_tag: int | str | None = None
    passenger_tags: tuple[int | str, ...] = ()
    passenger_units: tuple[str, ...] = ()
    is_biological: bool | None = None
    is_mechanical: bool | None = None
    is_psionic: bool | None = None
    is_massive: bool | None = None
    is_detector: bool | None = None
    weapon_cooldown: float | None = None
    orders: tuple[str, ...] = ()


UnitObservationSnapshot = UnitObservation


@dataclass(frozen=True)
class LocationSnapshot:
    """Executor-observed point for an allowlisted semantic location."""

    x: float
    y: float
    resolved: bool = True


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
    structures_ready: dict[str, int] = field(default_factory=dict)
    structures_pending: dict[str, int] = field(default_factory=dict)
    upgrades: tuple[str, ...] = ()
    unit_observations: tuple[UnitObservation, ...] = ()
    semantic_locations: dict[str, LocationSnapshot | None] = field(default_factory=dict)


def game_state_summary_to_dict(summary: GameStateSummary) -> dict[str, Any]:
    result: dict[str, Any] = {
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
        "structures_ready": dict(sorted(summary.structures_ready.items())),
        "structures_pending": dict(sorted(summary.structures_pending.items())),
        "upgrades": sorted(summary.upgrades),
        "known_enemy_units": summary.known_enemy_units,
        "game_time_seconds": summary.game_time_seconds,
    }
    if summary.unit_observations:
        result["unit_observations"] = [
            _unit_observation_to_dict(observation)
            for observation in summary.unit_observations
        ]
    if summary.semantic_locations:
        result["semantic_locations"] = {
            key: _location_snapshot_to_dict(value) if value is not None else None
            for key, value in sorted(summary.semantic_locations.items())
        }
    return result


def game_state_summary_to_json(summary: GameStateSummary) -> str:
    return json.dumps(game_state_summary_to_dict(summary), ensure_ascii=False, indent=2)


def _unit_observation_to_dict(observation: UnitObservation) -> dict[str, Any]:
    result: dict[str, Any] = {
        "unit": observation.unit,
        "alliance": observation.alliance,
    }
    optional_fields = (
        "tag",
        "x",
        "y",
        "health",
        "health_max",
        "energy",
        "is_ready",
        "is_flying",
        "is_burrowed",
        "is_loaded",
        "is_idle",
        "cargo_used",
        "cargo_max",
        "add_on_tag",
        "is_biological",
        "is_mechanical",
        "is_psionic",
        "is_massive",
        "is_detector",
        "weapon_cooldown",
    )
    for field_name in optional_fields:
        value = getattr(observation, field_name)
        if value is not None:
            result[field_name] = value
    if observation.orders:
        result["orders"] = list(observation.orders)
    if observation.passenger_tags:
        result["passenger_tags"] = list(observation.passenger_tags)
    if observation.passenger_units:
        result["passenger_units"] = list(observation.passenger_units)
    return result


def _location_snapshot_to_dict(location: LocationSnapshot) -> dict[str, Any]:
    return {"x": location.x, "y": location.y, "resolved": location.resolved}
