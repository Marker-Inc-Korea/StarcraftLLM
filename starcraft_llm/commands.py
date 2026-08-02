"""Typed command constructors and function-call adapter for LLM planners.

The canonical on-wire contract remains ``{"actions": [...]}``, but these named
constructors provide the same surface to function-calling model adapters without
letting model text invoke arbitrary Python or SC2 APIs.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from starcraft_llm.command_catalog import (
    ADDON_SPECS,
    MORPH_SPECS,
    REPAIRABLE_TARGET_KEYS,
    STRUCTURE_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
)
from starcraft_llm.strategy import StrategyAction, StrategyParseError, StrategyPlan, strategy_plan_from_dict


def _command(action_type: str, **arguments: Any) -> StrategyAction:
    payload = {"type": action_type}
    payload.update({key: value for key, value in arguments.items() if value is not None})
    return strategy_plan_from_dict([payload]).actions[0]


def move(unit: str, x: float, y: float) -> StrategyAction:
    return _command("move", unit=unit, x=x, y=y)


def attack_move(unit: str, x: float, y: float) -> StrategyAction:
    return _command("attack", unit=unit, x=x, y=y)


def attack_enemy(unit: str = "marine") -> StrategyAction:
    return _command("attack_enemy", unit=unit)


def patrol(unit: str, x: float, y: float) -> StrategyAction:
    return _command("patrol", unit=unit, x=x, y=y)


def hold_position(unit: str) -> StrategyAction:
    return _command("hold", unit=unit)


def stop(unit: str) -> StrategyAction:
    return _command("stop", unit=unit)


def rally(building: str, x: float, y: float) -> StrategyAction:
    return _command("rally", building=building, x=x, y=y)


def wait(seconds: float) -> StrategyAction:
    return _command("wait", seconds=seconds)


def wait_until(condition: str, at_least: float = 1, target: Optional[str] = None) -> StrategyAction:
    return _command("wait_until", condition=condition, at_least=at_least, target=target)


def gather(resource: str, workers: Optional[int] = None) -> StrategyAction:
    return _command("gather", unit="worker", resource=resource, workers=workers)


def distribute_workers(mineral_to_gas_ratio: float = 2) -> StrategyAction:
    return _command("distribute_workers", mineral_to_gas_ratio=mineral_to_gas_ratio)


def train(unit: str, count: int = 1) -> StrategyAction:
    return _command("train", unit=unit, count=count)


def build(building: str, count: int = 1) -> StrategyAction:
    return _command("build", building=building, worker="worker", count=count)


def expand(count: int = 1) -> StrategyAction:
    return _command("expand", count=count)


def build_addon(addon: str, count: int = 1, producer: Optional[str] = None) -> StrategyAction:
    return _command("build_addon", addon=addon, producer=producer, count=count)


def morph(building: str) -> StrategyAction:
    return _command("morph", building=building)


def research(upgrade: str) -> StrategyAction:
    return _command("research", upgrade=upgrade)


def repair(target: str, workers: int = 1) -> StrategyAction:
    return _command("repair", target=target, workers=workers)


def create_plan(actions: Iterable[StrategyAction]) -> StrategyPlan:
    values = tuple(actions)
    if not values:
        raise StrategyParseError("strategy plan must contain at least one command")
    return StrategyPlan(actions=values)


def strategy_plan_from_function_calls(calls: Iterable[Mapping[str, Any]]) -> StrategyPlan:
    """Convert common LLM function-call payloads into a canonical StrategyPlan.

    Accepted items are ``{"name": "build", "arguments": {...}}`` and the
    OpenAI-style nested form ``{"function": {"name": ..., "arguments": ...}}``.
    JSON-encoded argument strings are decoded locally before validation.
    """

    actions = []
    for index, call in enumerate(calls, start=1):
        function_payload = call.get("function", call)
        if not isinstance(function_payload, Mapping):
            raise StrategyParseError(f"function call {index} must be an object")
        name = str(function_payload.get("name", "")).strip()
        if name not in LLM_COMMAND_FUNCTIONS:
            raise StrategyParseError(f"unsupported command function: {name!r}")
        arguments = function_payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise StrategyParseError(f"function call {index} arguments are invalid JSON: {exc.msg}") from exc
        if not isinstance(arguments, Mapping):
            raise StrategyParseError(f"function call {index} arguments must be an object")
        try:
            actions.append(LLM_COMMAND_FUNCTIONS[name](**dict(arguments)))
        except TypeError as exc:
            raise StrategyParseError(f"invalid arguments for command function {name}: {exc}") from exc
    return create_plan(actions)


LLM_COMMAND_FUNCTIONS: Mapping[str, Callable[..., StrategyAction]] = MappingProxyType(
    {
        "move": move,
        "attack_move": attack_move,
        "attack_enemy": attack_enemy,
        "patrol": patrol,
        "hold_position": hold_position,
        "stop": stop,
        "rally": rally,
        "wait": wait,
        "wait_until": wait_until,
        "gather": gather,
        "distribute_workers": distribute_workers,
        "train": train,
        "build": build,
        "expand": expand,
        "build_addon": build_addon,
        "morph": morph,
        "research": research,
        "repair": repair,
    }
)


def _object_schema(properties: dict[str, dict[str, Any]], required: Tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _function(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters}


def llm_command_function_schemas() -> tuple[dict[str, Any], ...]:
    """Return provider-neutral JSON function declarations for every safe command."""

    unit_enum = ["worker", *[key for key in UNIT_SPECS if key != "scv"]]
    structure_enum = list(STRUCTURE_SPECS)
    repair_enum = list(REPAIRABLE_TARGET_KEYS)
    safe_coordinate = {"type": "number", "minimum": 0, "maximum": 256}
    positive_count = {"type": "integer", "minimum": 1, "maximum": 20}
    point = {"x": safe_coordinate, "y": safe_coordinate}
    unit = {"type": "string", "enum": unit_enum}

    return (
        _function("move", "Move a Terran unit group to a safe map coordinate.", _object_schema({"unit": unit, **point}, ("unit", "x", "y"))),
        _function("attack_move", "Attack-move a Terran unit group to a map coordinate.", _object_schema({"unit": unit, **point}, ("unit", "x", "y"))),
        _function("attack_enemy", "Attack the nearest visible enemy.", _object_schema({"unit": unit}, ("unit",))),
        _function("patrol", "Patrol a Terran unit group to a map coordinate.", _object_schema({"unit": unit, **point}, ("unit", "x", "y"))),
        _function("hold_position", "Hold a Terran unit group in place.", _object_schema({"unit": unit}, ("unit",))),
        _function("stop", "Stop a Terran unit group's current order.", _object_schema({"unit": unit}, ("unit",))),
        _function("rally", "Set a production structure rally point.", _object_schema({"building": {"type": "string", "enum": ["command_center", "orbital_command", "planetary_fortress", "barracks", "factory", "starport"]}, **point}, ("building", "x", "y"))),
        _function("wait", "Wait a bounded number of game-clock seconds.", _object_schema({"seconds": {"type": "number", "minimum": 0, "maximum": 30}}, ("seconds",))),
        _function("wait_until", "Wait for a resource, supply, unit, structure, base, upgrade, or game-time condition.", _object_schema({"condition": {"type": "string", "enum": ["minerals", "vespene", "supply_left", "supply_used", "supply_cap", "structure_count", "structure_ready", "structure_pending", "unit_count", "townhall_count", "upgrade_complete", "game_time"]}, "target": {"type": "string"}, "at_least": {"type": "number", "minimum": 0, "maximum": 10000}}, ("condition", "at_least"))),
        _function("gather", "Assign workers to minerals or vespene.", _object_schema({"resource": {"type": "string", "enum": ["minerals", "vespene"]}, "workers": positive_count}, ("resource",))),
        _function("distribute_workers", "Rebalance workers between mineral and gas income.", _object_schema({"mineral_to_gas_ratio": {"type": "number", "minimum": 0, "maximum": 20}})),
        _function("train", "Train any standard Terran unit.", _object_schema({"unit": {"type": "string", "enum": list(UNIT_SPECS)}, "count": positive_count}, ("unit",))),
        _function("build", "Build any standard Terran structure.", _object_schema({"building": {"type": "string", "enum": structure_enum}, "count": positive_count}, ("building",))),
        _function("expand", "Build command centers at the next available expansions.", _object_schema({"count": positive_count})),
        _function(
            "build_addon",
            "Build a Terran Tech Lab or Reactor on a compatible producer.",
            _object_schema(
                {
                    "addon": {"type": "string", "enum": list(ADDON_SPECS)},
                    "count": positive_count,
                },
                ("addon",),
            ),
        ),
        _function("morph", "Morph a command center into an Orbital Command or Planetary Fortress.", _object_schema({"building": {"type": "string", "enum": list(MORPH_SPECS)}}, ("building",))),
        _function("research", "Research a supported Terran upgrade.", _object_schema({"upgrade": {"type": "string", "enum": list(UPGRADE_SPECS)}}, ("upgrade",))),
        _function("repair", "Assign SCVs to repair a Terran unit or structure.", _object_schema({"target": {"type": "string", "enum": repair_enum}, "workers": positive_count}, ("target",))),
    )


__all__ = (
    "LLM_COMMAND_FUNCTIONS",
    "attack_enemy",
    "attack_move",
    "build",
    "build_addon",
    "create_plan",
    "distribute_workers",
    "expand",
    "gather",
    "hold_position",
    "llm_command_function_schemas",
    "morph",
    "move",
    "patrol",
    "rally",
    "repair",
    "research",
    "stop",
    "strategy_plan_from_function_calls",
    "train",
    "wait",
    "wait_until",
)
