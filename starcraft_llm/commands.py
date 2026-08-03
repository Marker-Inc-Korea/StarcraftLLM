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
    MAX_POLICY_SECONDS,
    MAX_SELECTION_COUNT,
    MAX_STRUCTURE_ACTION_COUNT,
    MAX_WORKER_ASSIGNMENT_COUNT,
    MEDIVAC_LOADABLE_UNIT_KEYS,
    MOBILE_ATTACK_CAPABLE_UNIT_KEYS,
    MORPH_SPECS,
    MOVABLE_SPECIAL_UNIT_KEYS,
    REPAIRABLE_TARGET_KEYS,
    SALVAGEABLE_STRUCTURE_KEYS,
    STRUCTURE_SPECS,
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


def move_target(
    unit: str,
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "move_target",
        unit=unit,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
    )


def move_and_wait(
    unit: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    location: Optional[str] = None,
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
    arrival_tolerance: float = 2.5,
    timeout_seconds: float = 90,
) -> StrategyAction:
    return _command(
        "move_and_wait",
        unit=unit,
        x=x,
        y=y,
        location=location,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
        arrival_tolerance=arrival_tolerance,
        timeout_seconds=timeout_seconds,
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
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
) -> StrategyAction:
    return _command(
        "attack_enemy",
        unit=unit,
        selection=selection,
        queued=queued,
        target_unit=target_unit,
        target_tag=target_tag,
    )


def attack_target(
    unit: str = "marine",
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "attack_target",
        unit=unit,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
    )


def focus_fire(
    unit: str = "marine",
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
    timeout_seconds: float = 60,
) -> StrategyAction:
    return _command(
        "focus_fire",
        unit=unit,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
        timeout_seconds=timeout_seconds,
    )


def kite(
    unit: str,
    target_unit: str = "nearest_enemy",
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    duration_seconds: float = 8,
    retreat_distance: float = 2,
) -> StrategyAction:
    return _command(
        "kite",
        unit=unit,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        duration_seconds=duration_seconds,
        retreat_distance=retreat_distance,
    )


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
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "rally",
        building=building,
        x=x,
        y=y,
        location=location,
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
    )


def wait(seconds: float) -> StrategyAction:
    return _command("wait", seconds=seconds)


def wait_until(
    condition: str,
    at_least: float = 1,
    target: Optional[str] = None,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    radius: float = 12,
    selection: Optional[Mapping[str, Any]] = None,
    timeout_seconds: float = 120,
    on_timeout: str = "replan",
) -> StrategyAction:
    return _command(
        "wait_until",
        condition=condition,
        at_least=at_least,
        target=target,
        location=location,
        x=x,
        y=y,
        radius=radius,
        selection=selection,
        timeout_seconds=timeout_seconds,
        on_timeout=on_timeout,
    )


def gather(
    resource: str,
    workers: Optional[int] = None,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "gather",
        unit="worker",
        resource=resource,
        workers=workers,
        location=location,
        x=x,
        y=y,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
    )


def return_cargo(
    unit: str = "worker",
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("return_cargo", unit=unit, selection=selection, queued=queued)


def distribute_workers(mineral_to_gas_ratio: float = 2) -> StrategyAction:
    return _command("distribute_workers", mineral_to_gas_ratio=mineral_to_gas_ratio)


def train(
    unit: str,
    count: int = 1,
    producer_selection: Optional[Mapping[str, Any]] = None,
) -> StrategyAction:
    return _command(
        "train", unit=unit, count=count, producer_selection=producer_selection
    )


def produce_until(
    unit: str,
    target_count: int,
    producer_selection: Optional[Mapping[str, Any]] = None,
    reserve_minerals: int = 0,
    reserve_vespene: int = 0,
    reserve_supply: int = 0,
    max_seconds: float = 300,
) -> StrategyAction:
    return _command(
        "produce_until",
        unit=unit,
        target_count=target_count,
        producer_selection=producer_selection,
        reserve_minerals=reserve_minerals,
        reserve_vespene=reserve_vespene,
        reserve_supply=reserve_supply,
        max_seconds=max_seconds,
    )


def maintain_production(
    unit: str,
    target_count: int,
    producer_selection: Optional[Mapping[str, Any]] = None,
    reserve_minerals: int = 0,
    reserve_vespene: int = 0,
    reserve_supply: int = 0,
    max_seconds: float = 300,
) -> StrategyAction:
    return _command(
        "maintain_production",
        unit=unit,
        target_count=target_count,
        producer_selection=producer_selection,
        reserve_minerals=reserve_minerals,
        reserve_vespene=reserve_vespene,
        reserve_supply=reserve_supply,
        max_seconds=max_seconds,
    )


def build(
    building: str,
    count: int = 1,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
    placement_mode: Optional[str] = None,
    max_distance: Optional[int] = None,
    reserve_addon_space: bool = False,
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
        placement_mode=placement_mode,
        max_distance=max_distance,
        reserve_addon_space=reserve_addon_space,
    )


def expand(count: int = 1) -> StrategyAction:
    return _command("expand", count=count)


def build_addon(
    addon: str,
    count: int = 1,
    producer: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> StrategyAction:
    return _command(
        "build_addon", addon=addon, producer=producer, count=count, selection=selection
    )


def morph(
    building: str, selection: Optional[Mapping[str, Any]] = None
) -> StrategyAction:
    return _command("morph", building=building, selection=selection)


def research(
    upgrade: str, researcher_selection: Optional[Mapping[str, Any]] = None
) -> StrategyAction:
    return _command(
        "research", upgrade=upgrade, researcher_selection=researcher_selection
    )


def repair(
    target: Optional[str] = None,
    workers: int = 1,
    target_tag: Optional[int | str] = None,
    target_selector: Optional[str] = None,
    target_selection: Optional[Mapping[str, Any]] = None,
    selection: Optional[Mapping[str, Any]] = None,
) -> StrategyAction:
    return _command(
        "repair",
        target=target,
        workers=workers,
        target_tag=target_tag,
        target_selector=target_selector,
        target_selection=target_selection,
        selection=selection,
    )


def use_ability(
    ability: str,
    actor: Optional[str] = None,
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
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
        target_tag=target_tag,
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
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "scan", location=location, x=x, y=y, selection=selection, queued=queued
    )


def call_down_mule(
    location: Optional[str] = "nearest_mineral",
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "call_down_mule",
        location=location,
        x=x,
        y=y,
        selection=selection,
        queued=queued,
    )


def supply_drop(
    target_unit: str = "supply_depot",
    target_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "supply_drop",
        target_unit=target_unit,
        target_tag=target_tag,
        selection=selection,
        queued=queued,
    )


def transform(
    ability: str,
    actor: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "transform", ability=ability, actor=actor, selection=selection, queued=queued
    )


def lift(
    actor: str,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("lift", actor=actor, selection=selection, queued=queued)


def land(
    actor: str,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    target_addon: Optional[str] = None,
    target_addon_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "land",
        actor=actor,
        location=location,
        x=x,
        y=y,
        target_addon=target_addon,
        target_addon_tag=target_addon_tag,
        selection=selection,
        queued=queued,
    )


def land_on_addon(
    actor: str,
    target_addon: Optional[str] = None,
    target_addon_tag: Optional[int | str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "land_on_addon",
        actor=actor,
        target_addon=target_addon,
        target_addon_tag=target_addon_tag,
        selection=selection,
        queued=queued,
    )


def load(
    actor: str,
    target_unit: Optional[str] = None,
    target_tag: Optional[int | str] = None,
    target_selection: Optional[Mapping[str, Any]] = None,
    count: Optional[int] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "load",
        actor=actor,
        target_unit=target_unit,
        target_tag=target_tag,
        target_selection=target_selection,
        count=count,
        selection=selection,
        queued=queued,
    )


def unload(
    actor: str,
    target_unit: Optional[str] = None,
    passenger_tag: Optional[int | str] = None,
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "unload",
        actor=actor,
        target_unit=target_unit,
        passenger_tag=passenger_tag,
        location=location,
        x=x,
        y=y,
        selection=selection,
        queued=queued,
    )


def cancel(
    ability: str = "cancel_any",
    actor: Optional[str] = None,
    target: Optional[str] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "cancel",
        ability=ability if target is None else None,
        target=target,
        actor=actor,
        selection=selection,
        queued=queued,
    )


def salvage(
    actor: str,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command("salvage", actor=actor, selection=selection, queued=queued)


def build_nuke(
    selection: Optional[Mapping[str, Any]] = None, queued: bool = False
) -> StrategyAction:
    return _command("build_nuke", selection=selection, queued=queued)


def launch_nuke(
    location: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    selection: Optional[Mapping[str, Any]] = None,
    queued: bool = False,
) -> StrategyAction:
    return _command(
        "launch_nuke", location=location, x=x, y=y, selection=selection, queued=queued
    )


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
        "move_target": move_target,
        "move_and_wait": move_and_wait,
        "attack_move": attack_move,
        "attack_enemy": attack_enemy,
        "attack_target": attack_target,
        "focus_fire": focus_fire,
        "kite": kite,
        "patrol": patrol,
        "hold_position": hold_position,
        "stop": stop,
        "rally": rally,
        "wait": wait,
        "wait_until": wait_until,
        "gather": gather,
        "return_cargo": return_cargo,
        "distribute_workers": distribute_workers,
        "train": train,
        "produce_until": produce_until,
        "maintain_production": maintain_production,
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
        "land_on_addon": land_on_addon,
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


def _one_target_schema(
    properties: dict[str, dict[str, Any]],
    target_fields: Tuple[str, ...],
    required: Tuple[str, ...] = (),
) -> dict[str, Any]:
    schema = _object_schema(properties, required)
    schema["anyOf"] = [{"required": [field]} for field in target_fields]
    return schema


def _target_choice_schema(
    properties: dict[str, dict[str, Any]],
    target_options: Tuple[Tuple[str, ...], ...],
    required: Tuple[str, ...] = (),
) -> dict[str, Any]:
    schema = _object_schema(properties, required)
    schema["anyOf"] = [
        {"required": list(target_fields)} for target_fields in target_options
    ]
    return schema


def _build_placement_schema(
    properties: dict[str, dict[str, Any]], required: Tuple[str, ...] = ()
) -> dict[str, Any]:
    schema = _object_schema(properties, required)
    exact_count = {"properties": {"count": {"maximum": 1}}}
    schema["anyOf"] = [
        {"properties": {"placement_mode": {"enum": ["near"]}}},
        {"required": ["location"], **exact_count},
        {"required": ["x", "y"], **exact_count},
    ]
    return schema


def _function(
    name: str, description: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    return {"name": name, "description": description, "parameters": parameters}


def llm_command_function_schemas() -> tuple[dict[str, Any], ...]:
    """Return provider-neutral JSON function declarations for every safe command."""

    standard_unit_enum = ["worker", *[key for key in UNIT_SPECS if key != "scv"]]
    attack_actor_enum = [
        "worker" if key == "scv" else key for key in ATTACK_CAPABLE_UNIT_KEYS
    ]
    mobile_attack_unit_enum = [
        "worker" if key == "scv" else key for key in MOBILE_ATTACK_CAPABLE_UNIT_KEYS
    ]
    movable_unit_enum = [
        *standard_unit_enum,
        *MOVABLE_SPECIAL_UNIT_KEYS,
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
    tag: dict[str, Any] = {
        "anyOf": [
            {"type": "integer", "minimum": 1},
            {"type": "string", "pattern": "^[1-9][0-9]*$"},
        ]
    }
    point: dict[str, dict[str, Any]] = {"x": safe_coordinate, "y": safe_coordinate}
    location: dict[str, Any] = {"type": "string", "enum": list(LOCATION_SPECS)}
    attack_unit: dict[str, Any] = {
        "type": "string",
        "enum": mobile_attack_unit_enum,
    }
    attack_actor: dict[str, Any] = {
        "type": "string",
        "enum": attack_actor_enum,
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
                "enum": [
                    "all",
                    "ready",
                    "idle",
                    "closest",
                    "lowest_health",
                    "highest_energy",
                ],
            },
            "count": selection_count,
            "tags": {
                "type": "array",
                "items": tag,
                "minItems": 1,
                "maxItems": MAX_SELECTION_COUNT,
                "uniqueItems": True,
            },
        }
    )
    ability_target_unit: dict[str, Any] = {
        "type": "string",
        "pattern": "^[a-z0-9_]{1,64}$",
        "description": (
            "A target selector, canonical friendly type, or normalized observed "
            "enemy unit/structure type such as zergling, carrier, or hatchery."
        ),
    }
    enemy_target: dict[str, Any] = {
        "type": "string",
        "pattern": "^[a-z0-9_]{1,64}$",
        "description": (
            "An enemy selector such as nearest_enemy_air/lowest_health_enemy, "
            "or an observed normalized enemy unit type such as zergling or carrier."
        ),
    }
    rally_target: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys(
                (
                    "nearest_mineral",
                    "nearest_friendly",
                    "damaged_friendly",
                    "lowest_health_friendly",
                    "highest_energy_friendly",
                    "any_friendly",
                    *movable_unit_enum,
                    *structure_enum,
                    *MORPH_SPECS,
                )
            )
        ),
    }
    friendly_move_target: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys(
                (
                    "nearest_friendly",
                    "damaged_friendly",
                    "lowest_health_friendly",
                    "highest_energy_friendly",
                    "any_friendly",
                    *movable_unit_enum,
                    *structure_enum,
                    *MORPH_SPECS,
                )
            )
        ),
    }
    friendly_target_selector: dict[str, Any] = {
        "type": "string",
        "enum": [
            "nearest_friendly",
            "damaged_friendly",
            "lowest_health_friendly",
            "any_friendly",
        ],
    }
    production_flying_actor: dict[str, Any] = {
        "type": "string",
        "enum": ["barracks", "factory", "starport"],
    }
    loadable_unit: dict[str, Any] = {
        "type": "string",
        "enum": list(
            dict.fromkeys((*BUNKER_LOADABLE_UNIT_KEYS, *MEDIVAC_LOADABLE_UNIT_KEYS))
        ),
    }
    addon_target: dict[str, Any] = {"type": "string", "enum": list(ADDON_SPECS)}
    placement_mode: dict[str, Any] = {"type": "string", "enum": ["near", "exact"]}
    max_distance: dict[str, Any] = {
        "type": "integer",
        "minimum": 0,
        "maximum": 20,
    }
    queued: dict[str, Any] = {"type": "boolean"}
    bounded_policy_seconds: dict[str, Any] = {
        "type": "number",
        "minimum": 1,
        "maximum": MAX_POLICY_SECONDS,
    }
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
            "move_target",
            "Move or follow a specific visible friendly unit by selector/type or runtime unit tag.",
            _one_target_schema(
                {
                    "unit": movable_unit,
                    "target_unit": friendly_move_target,
                    "target_tag": tag,
                    "selection": selection,
                    "queued": queued,
                },
                ("target_unit", "target_tag"),
                ("unit",),
            ),
        ),
        _function(
            "move_and_wait",
            "Move a bounded Terran unit group to a point or visible friendly and wait for arrival with a bounded timeout.",
            _target_choice_schema(
                {
                    "unit": movable_unit,
                    **point_order,
                    "target_unit": friendly_move_target,
                    "target_tag": tag,
                    "arrival_tolerance": {
                        "type": "number",
                        "minimum": 0.25,
                        "maximum": 20,
                    },
                    "timeout_seconds": bounded_policy_seconds,
                },
                (("location",), ("x", "y"), ("target_unit",), ("target_tag",)),
                ("unit",),
            ),
        ),
        _function(
            "attack_move",
            "Attack-move a bounded Terran unit group to coordinates or a semantic location.",
            _point_target_schema({"unit": attack_unit, **point_order}, ("unit",)),
        ),
        _function(
            "attack_enemy",
            "Attack the nearest visible enemy, or a specific enemy when target_unit/target_tag is provided.",
            _object_schema(
                {
                    "unit": attack_actor,
                    "target_unit": enemy_target,
                    "target_tag": tag,
                    "selection": selection,
                    "queued": queued,
                },
                ("unit",),
            ),
        ),
        _function(
            "attack_target",
            "Attack a specific visible enemy by selector/type or runtime unit tag.",
            _one_target_schema(
                {
                    "unit": attack_actor,
                    "target_unit": enemy_target,
                    "target_tag": tag,
                    "selection": selection,
                    "queued": queued,
                },
                ("target_unit", "target_tag"),
                ("unit",),
            ),
        ),
        _function(
            "focus_fire",
            "Keep selected attack-capable actors focused on one visible enemy until it dies or timeout.",
            _one_target_schema(
                {
                    "unit": attack_actor,
                    "target_unit": enemy_target,
                    "target_tag": tag,
                    "selection": selection,
                    "queued": queued,
                    "timeout_seconds": bounded_policy_seconds,
                },
                ("target_unit", "target_tag"),
                ("unit",),
            ),
        ),
        _function(
            "kite",
            "Run bounded weapon-cooldown-aware stutter-step micro against one visible enemy.",
            _object_schema(
                {
                    "unit": attack_unit,
                    "target_unit": enemy_target,
                    "target_tag": tag,
                    "selection": selection,
                    "duration_seconds": {
                        "type": "number",
                        "minimum": 0.25,
                        "maximum": 30,
                    },
                    "retreat_distance": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 10,
                    },
                },
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
                {
                    "unit": {
                        "type": "string",
                        "enum": list(
                            dict.fromkeys((*movable_unit_enum, *attack_actor_enum))
                        ),
                    },
                    "selection": selection,
                    "queued": queued,
                },
                ("unit",),
            ),
        ),
        _function(
            "rally",
            "Set a production structure rally point to a point or visible unit tag/selector.",
            _target_choice_schema(
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
                            "bunker",
                        ],
                    },
                    **point_order,
                    "target_unit": rally_target,
                    "target_tag": tag,
                },
                (("location",), ("x", "y"), ("target_unit",), ("target_tag",)),
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
            "Wait for a bounded observed resource, unit, enemy, proximity, cargo, production, tech, or time condition.",
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
                            "army_supply",
                            "enemy_unit_count",
                            "enemy_structure_count",
                            "idle_structure_count",
                            "producer_available",
                            "cargo_used",
                            "unit_near_location",
                            "enemy_near_location",
                            "under_attack",
                        ],
                    },
                    "target": {"type": "string"},
                    "at_least": {"type": "number", "minimum": 0, "maximum": 10000},
                    "location": location,
                    **point,
                    "radius": {"type": "number", "minimum": 0.5, "maximum": 64},
                    "selection": selection,
                    "timeout_seconds": bounded_policy_seconds,
                    "on_timeout": {"type": "string", "enum": ["replan", "fail"]},
                },
                ("condition", "at_least"),
            ),
        ),
        _function(
            "gather",
            "Assign selected workers to minerals or vespene by resource type, location, or target tag.",
            _object_schema(
                {
                    "resource": {"type": "string", "enum": ["minerals", "vespene"]},
                    "workers": worker_count,
                    "location": location,
                    **point,
                    "target_tag": tag,
                    "selection": selection,
                    "queued": queued,
                },
                ("resource",),
            ),
        ),
        _function(
            "return_cargo",
            "Return carried resources from selected workers or MULEs.",
            _object_schema(
                {
                    "unit": {"type": "string", "enum": ["worker", "mule"]},
                    "selection": selection,
                    "queued": queued,
                }
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
                    "producer_selection": selection,
                },
                ("unit",),
            ),
        ),
        _function(
            "produce_until",
            "Continuously train a Terran unit until an absolute owned-unit count is reached.",
            _object_schema(
                {
                    "unit": {"type": "string", "enum": list(UNIT_SPECS)},
                    "target_count": selection_count,
                    "producer_selection": selection,
                    "reserve_minerals": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                    },
                    "reserve_vespene": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                    },
                    "reserve_supply": {"type": "integer", "minimum": 0, "maximum": 200},
                    "max_seconds": bounded_policy_seconds,
                },
                ("unit", "target_count"),
            ),
        ),
        _function(
            "maintain_production",
            "Register bounded background production while later plan actions continue.",
            _object_schema(
                {
                    "unit": {"type": "string", "enum": list(UNIT_SPECS)},
                    "target_count": selection_count,
                    "producer_selection": selection,
                    "reserve_minerals": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                    },
                    "reserve_vespene": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10000,
                    },
                    "reserve_supply": {"type": "integer", "minimum": 0, "maximum": 200},
                    "max_seconds": bounded_policy_seconds,
                },
                ("unit", "target_count"),
            ),
        ),
        _function(
            "build",
            "Build any standard Terran structure at an optional semantic location or coordinates.",
            _build_placement_schema(
                {
                    "building": {"type": "string", "enum": structure_enum},
                    "count": structure_count,
                    "location": location,
                    **point,
                    "selection": selection,
                    "placement_mode": placement_mode,
                    "max_distance": max_distance,
                    "reserve_addon_space": {"type": "boolean"},
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
                    "selection": selection,
                },
                ("addon",),
            ),
        ),
        _function(
            "morph",
            "Morph a command center into an Orbital Command or Planetary Fortress.",
            _object_schema(
                {
                    "building": {"type": "string", "enum": list(MORPH_SPECS)},
                    "selection": selection,
                },
                ("building",),
            ),
        ),
        _function(
            "research",
            "Research a supported Terran upgrade.",
            _object_schema(
                {
                    "upgrade": {"type": "string", "enum": list(UPGRADE_SPECS)},
                    "researcher_selection": selection,
                },
                ("upgrade",),
            ),
        ),
        _function(
            "repair",
            "Assign SCVs to repair a Terran unit or structure by type, selector, or runtime tag.",
            _one_target_schema(
                {
                    "target": {"type": "string", "enum": repair_enum},
                    "target_tag": tag,
                    "target_selector": friendly_target_selector,
                    "target_selection": selection,
                    "workers": worker_count,
                    "selection": selection,
                },
                ("target", "target_tag", "target_selector"),
            ),
        ),
        _function(
            "use_ability",
            "Use an allowlisted Terran ability; executor verifies live availability before issuing it.",
            _object_schema(
                {
                    "ability": ability,
                    "actor": actor,
                    "target_unit": ability_target_unit,
                    "target_tag": tag,
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
            _point_target_schema(
                {
                    "location": location,
                    **point,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                }
            ),
        ),
        _function(
            "call_down_mule",
            "Call down a MULE near minerals.",
            _object_schema(
                {
                    "location": location,
                    **point,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                }
            ),
        ),
        _function(
            "supply_drop",
            "Use Extra Supplies on a supply depot target.",
            _object_schema(
                {
                    "target_unit": {"type": "string", "enum": ["supply_depot"]},
                    "target_tag": tag,
                    "selection": selection,
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
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("ability",),
            ),
        ),
        _function(
            "lift",
            "Lift a supported Terran structure.",
            _object_schema(
                {
                    "actor": liftable_actor,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "land",
            "Land a supported flying Terran structure.",
            _point_target_schema(
                {
                    "actor": liftable_actor,
                    "location": location,
                    **point,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "land_on_addon",
            "Land a supported flying production structure on a specific add-on by type or runtime tag.",
            _one_target_schema(
                {
                    "actor": production_flying_actor,
                    "target_addon": addon_target,
                    "target_addon_tag": tag,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("target_addon", "target_addon_tag"),
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
                    "target_tag": tag,
                    "target_selection": selection,
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
                    "passenger_tag": tag,
                    "location": location,
                    **point,
                    "selection": selection,
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
                    "selection": selection,
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
                    "selection": selection,
                    "queued": {"type": "boolean"},
                },
                ("actor",),
            ),
        ),
        _function(
            "build_nuke",
            "Build a tactical nuke at a Ghost Academy.",
            _object_schema({"selection": selection, "queued": {"type": "boolean"}}),
        ),
        _function(
            "launch_nuke",
            "Launch a tactical nuke at a semantic location or coordinates.",
            _point_target_schema(
                {
                    "location": location,
                    **point,
                    "selection": selection,
                    "queued": {"type": "boolean"},
                }
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
    "attack_target",
    "focus_fire",
    "build",
    "build_addon",
    "create_plan",
    "distribute_workers",
    "expand",
    "gather",
    "hold_position",
    "kite",
    "llm_command_function_schemas",
    "morph",
    "move",
    "move_and_wait",
    "move_target",
    "patrol",
    "rally",
    "repair",
    "research",
    "return_cargo",
    "maintain_production",
    "build_nuke",
    "call_down_mule",
    "cancel",
    "land",
    "land_on_addon",
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
    "produce_until",
    "wait",
    "wait_until",
)
