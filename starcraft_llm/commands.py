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
    ABILITY_SPECS,
    ADDON_SPECS,
    ATTACK_CAPABLE_UNIT_KEYS,
    BUNKER_LOADABLE_UNIT_KEYS,
    FLYING_STRUCTURE_ACTOR_KEYS,
    LOCATION_SPECS,
    LIFTABLE_STRUCTURE_KEYS,
    MAX_SELECTION_COUNT,
    MAX_STRUCTURE_ACTION_COUNT,
    MAX_WORKER_ASSIGNMENT_COUNT,
    MEDIVAC_LOADABLE_UNIT_KEYS,
    MORPH_SPECS,
    REPAIRABLE_TARGET_KEYS,
    SALVAGEABLE_STRUCTURE_KEYS,
    SPECIAL_UNIT_SPECS,
    STRUCTURE_SPECS,
    TARGET_SELECTORS,
    TRANSFORM_ABILITY_KEYS,
    TRANSPORT_ACTOR_KEYS,
    UNIT_SPECS,
    UPGRADE_SPECS,
)
from starcraft_llm.strategy import (
    StrategyAction,
    StrategyParseError,
    StrategyPlan,
    strategy_plan_from_dict,
)


def _command(action_type: str, **arguments: Any) -> StrategyAction:
    payload = {"type": action_type}
    payload.update(
        {key: value for key, value in arguments.items() if value is not None}
    )
    return strategy_plan_from_dict([payload]).actions[0]


def move(
    unit: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    location: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "move",
        unit=unit,
        x=x,
        y=y,
        location=location,
        selection=selection,
        queued=queued,
    )


def attack_move(
    unit: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    location: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "attack",
        unit=unit,
        x=x,
        y=y,
        location=location,
        selection=selection,
        queued=queued,
    )


def attack_enemy(
    unit: str = "marine",
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("attack_enemy", unit=unit, selection=selection, queued=queued)


def patrol(
    unit: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    location: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "patrol",
        unit=unit,
        x=x,
        y=y,
        location=location,
        selection=selection,
        queued=queued,
    )


def hold_position(
    unit: str,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("hold", unit=unit, selection=selection, queued=queued)


def stop(
    unit: str,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("stop", unit=unit, selection=selection, queued=queued)


def rally(
    building: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    location: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "rally",
        building=building,
        x=x,
        y=y,
        location=location,
        selection=selection,
        queued=queued,
    )


def wait(seconds: float) -> StrategyAction:
    return _command("wait", seconds=seconds)


def wait_until(
    condition: str, at_least: float = 1, target: Optional[str] = None
) -> StrategyAction:
    return _command("wait_until", condition=condition, at_least=at_least, target=target)


def gather(resource: str, workers: Optional[int] = None) -> StrategyAction:
    return _command("gather", unit="worker", resource=resource, workers=workers)


def distribute_workers(mineral_to_gas_ratio: float = 2) -> StrategyAction:
    return _command("distribute_workers", mineral_to_gas_ratio=mineral_to_gas_ratio)


def train(unit: str, count: int = 1) -> StrategyAction:
    return _command("train", unit=unit, count=count)


def build(
    building: str,
    count: int = 1,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> StrategyAction:
    return _command(
        "build",
        building=building,
        worker="worker",
        count=count,
        location=location,
        x=x,
        y=y,
        selection=selection,
    )


def expand(count: int = 1) -> StrategyAction:
    return _command("expand", count=count)


def build_addon(
    addon: str, count: int = 1, producer: Optional[str] = None
) -> StrategyAction:
    return _command("build_addon", addon=addon, producer=producer, count=count)


def morph(building: str) -> StrategyAction:
    return _command("morph", building=building)


def research(upgrade: str) -> StrategyAction:
    return _command("research", upgrade=upgrade)


def repair(target: str, workers: int = 1) -> StrategyAction:
    return _command("repair", target=target, workers=workers)


def use_ability(
    ability: str,
    actor: Optional[str] = None,
    target_unit: Optional[str] = None,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "use_ability",
        ability=ability,
        actor=actor,
        target_unit=target_unit,
        location=location,
        x=x,
        y=y,
        selection=selection,
        queued=queued,
    )


def scan(
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("scan", location=location, x=x, y=y, queued=queued)


def call_down_mule(
    location: Optional[str] = "nearest_mineral",
    x: Optional[float] = None,
    y: Optional[float] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("call_down_mule", location=location, x=x, y=y, queued=queued)


def supply_drop(
    target_unit: str = "supply_depot", queued: bool = False
) -> StrategyAction:
    return _command("supply_drop", target_unit=target_unit, queued=queued)


def transform(
    ability: str, actor: Optional[str] = None, queued: bool = False
) -> StrategyAction:
    return _command("transform", ability=ability, actor=actor, queued=queued)


def lift(actor: str, queued: bool = False) -> StrategyAction:
    return _command("lift", actor=actor, queued=queued)


def land(
    actor: str,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("land", actor=actor, location=location, x=x, y=y, queued=queued)


def load(
    actor: str,
    target_unit: Optional[str] = None,
    count: Optional[int] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "load",
        actor=actor,
        target_unit=target_unit,
        count=count,
        selection=selection,
        queued=queued,
    )


def unload(
    actor: str,
    target_unit: Optional[str] = None,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "unload",
        actor=actor,
        target_unit=target_unit,
        location=location,
        x=x,
        y=y,
        queued=queued,
    )


def cancel(
    ability: str = "cancel_any",
    actor: Optional[str] = None,
    target: Optional[str] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "cancel",
        ability=ability if target is None else None,
        target=target,
        actor=actor,
        queued=queued,
    )


def salvage(actor: str, queued: bool = False) -> StrategyAction:
    return _command("salvage", actor=actor, queued=queued)


def build_nuke(queued: bool = False) -> StrategyAction:
    return _command("build_nuke", queued=queued)


def launch_nuke(
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("launch_nuke", location=location, x=x, y=y, queued=queued)


def replan(reason: str = "requested") -> StrategyAction:
    return _command("replan", reason=reason)


def create_plan(actions: Iterable[StrategyAction]) -> StrategyPlan:
    values = tuple(actions)
    if not values:
        raise StrategyParseError("strategy plan must contain at least one command")
    return StrategyPlan(actions=values)


def strategy_plan_from_function_calls(
    calls: Iterable[Mapping[str, Any]]
) -> StrategyPlan:
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
                raise StrategyParseError(
                    f"function call {index} arguments are invalid JSON: {exc.msg}"
                ) from exc
        if not isinstance(arguments, Mapping):
            raise StrategyParseError(
                f"function call {index} arguments must be an object"
            )
        try:
            actions.append(LLM_COMMAND_FUNCTIONS[name](**dict(arguments)))
        except TypeError as exc:
            raise StrategyParseError(
                f"invalid arguments for command function {name}: {exc}"
            ) from exc
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
        "use_ability": use_ability,
        "scan": scan,
        "call_down_mule": call_down_mule,
        "supply_drop": supply_drop,
        "transform": transform,
        "lift": lift,
        "land": land,
        "load": load,
        "unload": unload,
        "cancel": cancel,
        "salvage": salvage,
        "build_nuke": build_nuke,
        "launch_nuke": launch_nuke,
        "replan": replan,
    }
)


def _object_schema(
    properties: dict[str, dict[str, Any]], required: Tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def _point_target_schema(
    properties: dict[str, dict[str, Any]], required: Tuple[str, ...] = ()
) -> dict[str, Any]:
    schema = _object_schema(properties, required)
    schema["anyOf"] = [
        {"required": ["location"]},
        {"required": ["x", "y"]},
    ]
    return schema


def _function(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters}


def llm_command_function_schemas() -> tuple[dict[str, Any], ...]:
    """Return provider-neutral JSON function declarations for every safe command."""

    standard_unit_enum = ["worker", *[key for key in UNIT_SPECS if key != "scv"]]
    attack_unit_enum = [
        "worker" if key == "scv" else key for key in ATTACK_CAPABLE_UNIT_KEYS
    ]
    movable_unit_enum = [
        *standard_unit_enum,
        *SPECIAL_UNIT_SPECS,
        *FLYING_STRUCTURE_ACTOR_KEYS,
    ]
    structure_enum = list(STRUCTURE_SPECS)
    repair_enum = list(REPAIRABLE_TARGET_KEYS)
    safe_coordinate: dict[str, Any] = {"type": "number", "minimum": 0, "maximum": 256}
    structure_count: dict[str, Any] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_STRUCTURE_ACTION_COUNT,
    }
    worker_count: dict[str, Any] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_WORKER_ASSIGNMENT_COUNT,
    }
    selection_count: dict[str, Any] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_SELECTION_COUNT,
    }
    train_count = dict(selection_count)
    point: dict[str, dict[str, Any]] = {"x": safe_coordinate, "y": safe_coordinate}
    location: dict[str, Any] = {"type": "string", "enum": list(LOCATION_SPECS)}
    attack_unit: dict[str, Any] = {
        "type": "string",
        "enum": attack_unit_enum,
    }
    movable_unit: dict[str, Any] = {
        "type": "string",
        "enum": movable_unit_enum,
    }
    actor: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys(
                (
                    "any",
                    *movable_unit_enum,
                    *structure_enum,
                    *ADDON_SPECS,
                    *MORPH_SPECS,
                )
            )
        ),
    }
    ability: dict[str, Any] = {"type": "string", "enum": list(ABILITY_SPECS)}
    transform_ability: dict[str, Any] = {
        "type": "string",
        "enum": list(TRANSFORM_ABILITY_KEYS),
    }
    liftable_actor: dict[str, Any] = {
        "type": "string",
        "enum": list(LIFTABLE_STRUCTURE_KEYS),
    }
    transport_actor: dict[str, Any] = {
        "type": "string",
        "enum": list(TRANSPORT_ACTOR_KEYS),
    }
    cancel_abilities = [key for key in ABILITY_SPECS if key.startswith("cancel_")]
    cancel_targets = [key.removeprefix("cancel_") for key in cancel_abilities]
    selection: dict[str, Any] = _object_schema(
        {
            "mode": {
                "type": "string",
                "enum": ["all", "ready", "idle", "closest", "lowest_health"],
            },
            "count": selection_count,
        }
    )
    target_unit: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys(
                (
                    *TARGET_SELECTORS,
                    *movable_unit_enum,
                    *structure_enum,
                    *ADDON_SPECS,
                    *MORPH_SPECS,
                )
            )
        ),
    }
    loadable_unit: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys((*BUNKER_LOADABLE_UNIT_KEYS, *MEDIVAC_LOADABLE_UNIT_KEYS))
        ),
    }
    queued: dict[str, Any] = {"type": "boolean"}
    point_order: dict[str, dict[str, Any]] = {
        "location": location,
        **point,
        "selection": selection,
        "queued": queued,
    }

    return (
        _function(
            "move",
            "Move a bounded Terran unit group to coordinates or a semantic location.",
            _point_target_schema({"unit": movable_unit, **point_order}, ("unit",)),
        ),
        _function(
            "attack_move",
            "Attack-move a bounded Terran unit group to coordinates or a semantic location.",
            _point_target_schema({"unit": attack_unit, **point_order}, ("unit",)),
        ),
        _function(
            "attack_enemy",
            "Attack the nearest visible enemy.",
            _object_schema(
                {"unit": attack_unit, "selection": selection, "queued": queued},
                ("unit",),
            ),
        ),
        _function(
            "patrol",
            "Patrol a bounded Terran unit group to coordinates or a semantic location.",
            _point_target_schema({"unit": movable_unit, **point_order}, ("unit",)),
        ),
        _function(
            "hold_position",
            "Hold a bounded Terran unit group in place.",
            _object_schema(
                {"unit": movable_unit, "selection": selection, "queued": queued},
                ("unit",),
            ),
        ),
        _function(
            "stop",
            "Stop a bounded Terran unit group's current order.",
            _object_schema(
                {"unit": movable_unit, "selection": selection, "queued": queued},
                ("unit",),
            ),
        ),
        _function(
            "rally",
            "Set a production structure rally point.",
            _point_target_schema(
                {
                    "building": {
                        "type": "string",
                        "enum": [
                            "command_center",
                            "orbital_command",
                            "planetary_fortress",
                            "barracks",
                            "factory",
                            "starport",
                        ],
                    },
                    **point_order,
                },
                ("building",),
            ),
        ),
        _function(
            "wait",
            "Wait a bounded number of game-clock seconds.",
            _object_schema(
                {"seconds": {"type": "number", "minimum": 0, "maximum": 30}},
                ("seconds",),
            ),
        ),
        _function(
            "wait_until",
            "Wait for a resource, supply, unit, structure, base, upgrade, or game-time condition.",
            _object_schema(
                {
                    "condition": {
                        "type": "string",
                        "enum": [
                            "minerals",
                            "vespene",
                            "supply_left",
                            "supply_used",
                            "supply_cap",
                            "structure_count",
                            "structure_ready",
                            "structure_pending",
                            "unit_count",
                            "townhall_count",
                            "upgrade_complete",
                            "game_time",
                        ],
                    },
                    "target": {"type": "string"},
                    "at_least": {"type": "number", "minimum": 0, "maximum": 10000},
                },
                ("condition", "at_least"),
            ),
        ),
        _function(
            "gather",
            "Assign workers to minerals or vespene.",
            _object_schema(
                {
                    "resource": {"type": "string", "enum": ["minerals", "vespene"]},
                    "workers": worker_count,
                },
                ("resource",),
            ),
        ),
        _function(
            "distribute_workers",
            "Rebalance workers between mineral and gas income.",
            _object_schema(
                {
                    "mineral_to_gas_ratio": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 20,
                    }
                }
            ),
        ),
        _function(
            "train",
            "Train any standard Terran unit.",
            _object_schema(
                {
                    "unit": {"type": "string", "enum": list(UNIT_SPECS)},
                    "count": train_count,
                },
                ("unit",),
            ),
        ),
        _function(
            "build",
            "Build any standard Terran structure at an optional semantic location or coordinates.",
            _object_schema(
                {
                    "building": {"type": "string", "enum": structure_enum},
                    "count": structure_count,
                    "location": location,
                    **point,
                    "selection": selection,
                },
                ("building",),
            ),
        ),
        _function(
            "expand",
            "Build command centers at the next available expansions.",
            _object_schema({"count": structure_count}),
        ),
        _function(
            "build_addon",
            "Build a Terran Tech Lab or Reactor on a compatible producer.",
            _object_schema(
                {
                    "addon": {"type": "string", "enum": list(ADDON_SPECS)},
                    "count": structure_count,
                },
                ("addon",),
            ),
        ),
        _function(
            "morph",
            "Morph a command center into an Orbital Command or Planetary Fortress.",
            _object_schema(
                {"building": {"type": "string", "enum": list(MORPH_SPECS)}},
                ("building",),
            ),
        ),
        _function(
            "research",
            "Research a supported Terran upgrade.",
            _object_schema(
                {"upgrade": {"type": "string", "enum": list(UPGRADE_SPECS)}},
                ("upgrade",),
            ),
        ),
        _function(
            "repair",
            "Assign SCVs to repair a Terran unit or structure.",
            _object_schema(
                {
                    "target": {"type": "string", "enum": repair_enum},
                    "workers": worker_count,
                },
                ("target",),
            ),
        ),
        _function(
            "use_ability",
            "Use an allowlisted Terran ability; executor verifies live availability before issuing it.",
            _object_schema(
                {
                    "ability": ability,
                    "actor": actor,
                    "target_unit": target_unit,
                    "location": location,
                    **point,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("ability",),
            ),
        ),
        _function(
            "scan",
            "Scanner sweep a semantic location or coordinates.",
            _object_schema(
                {"location": location, **point, "queued": {"type": "boolean"}}
            ),
        ),
        _function(
            "call_down_mule",
            "Call down a MULE near minerals.",
            _object_schema(
                {"location": location, **point, "queued": {"type": "boolean"}}
            ),
        ),
        _function(
            "supply_drop",
            "Use Extra Supplies on a supply depot target.",
            _object_schema(
                {
                    "target_unit": {"type": "string", "enum": ["supply_depot"]},
                    "queued": {"type": "boolean"},
                }
            ),
        ),
        _function(
            "transform",
            "Transform a Terran unit/structure mode using a transform ability.",
            _object_schema(
                {
                    "ability": transform_ability,
                    "actor": actor,
                    "queued": {"type": "boolean"},
                },
                ("ability",),
            ),
        ),
        _function(
            "lift",
            "Lift a supported Terran structure.",
            _object_schema(
                {"actor": liftable_actor, "queued": {"type": "boolean"}},
                ("actor",),
            ),
        ),
        _function(
            "land",
            "Land a supported flying Terran structure.",
            _object_schema(
                {
                    "actor": liftable_actor,
                    "location": location,
                    **point,
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "load",
            "Load units into a supported transport/container; command centers load nearby workers when target_unit is omitted.",
            _object_schema(
                {
                    "actor": transport_actor,
                    "target_unit": loadable_unit,
                    "count": selection_count,
                    "selection": selection,
                    "queued": queued,
                },
                ("actor",),
            ),
        ),
        _function(
            "unload",
            "Unload all at a location or unload one unit from a supported transport/container.",
            _object_schema(
                {
                    "actor": transport_actor,
                    "target_unit": loadable_unit,
                    "location": location,
                    **point,
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "cancel",
            "Cancel a supported order, queue item, add-on, morph, nuke, or lock-on.",
            _object_schema(
                {
                    "ability": {"type": "string", "enum": cancel_abilities},
                    "target": {"type": "string", "enum": cancel_targets},
                    "actor": actor,
                    "queued": {"type": "boolean"},
                }
            ),
        ),
        _function(
            "salvage",
            "Salvage a bunker or sensor tower.",
            _object_schema(
                {
                    "actor": {
                        "type": "string",
                        "enum": list(SALVAGEABLE_STRUCTURE_KEYS),
                    },
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "build_nuke",
            "Build a tactical nuke at a Ghost Academy.",
            _object_schema({"queued": {"type": "boolean"}}),
        ),
        _function(
            "launch_nuke",
            "Launch a tactical nuke at a semantic location or coordinates.",
            _object_schema(
                {"location": location, **point, "queued": {"type": "boolean"}}
            ),
        ),
        _function(
            "replan",
            "Request bounded closed-loop replanning with a short reason.",
            _object_schema({"reason": {"type": "string"}}),
        ),
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
    "build_nuke",
    "call_down_mule",
    "cancel",
    "land",
    "launch_nuke",
    "lift",
    "load",
    "replan",
    "salvage",
    "scan",
    "supply_drop",
    "transform",
    "unload",
    "use_ability",
    "stop",
    "strategy_plan_from_function_calls",
    "train",
    "wait",
    "wait_until",
)
