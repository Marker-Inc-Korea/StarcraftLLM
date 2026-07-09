from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeAlias


@dataclass(frozen=True)
class MoveCommand:
    """Move one logical unit group to a map point."""

    unit: str
    x: float
    y: float


@dataclass(frozen=True)
class AttackMoveCommand:
    """Attack-move one logical unit group toward a map point."""

    unit: str
    x: float
    y: float


@dataclass(frozen=True)
class AttackEnemyCommand:
    """Attack the nearest currently visible enemy with one logical unit group."""

    unit: str = "marine"


@dataclass(frozen=True)
class WaitCommand:
    """Pause strategy execution for a small amount of game-clock time."""

    seconds: float


@dataclass(frozen=True)
class WaitUntilCommand:
    """Pause execution until the observed game state satisfies a condition."""

    condition: str
    at_least: float
    target: str | None = None


@dataclass(frozen=True)
class GatherMineralsCommand:
    """Send a logical worker group to nearby mineral fields."""

    unit: str = "worker"


@dataclass(frozen=True)
class GatherGasCommand:
    """Send workers to ready refineries for vespene gathering."""

    unit: str = "worker"


@dataclass(frozen=True)
class TrainUnitCommand:
    """Train one or more units from available production structures."""

    unit: str
    count: int = 1


@dataclass(frozen=True)
class BuildStructureCommand:
    """Build one Terran structure at an executor-selected safe placement."""

    building: str
    worker: str = "worker"


StrategyAction: TypeAlias = (
    MoveCommand
    | AttackMoveCommand
    | AttackEnemyCommand
    | WaitCommand
    | WaitUntilCommand
    | GatherMineralsCommand
    | GatherGasCommand
    | TrainUnitCommand
    | BuildStructureCommand
)


@dataclass(frozen=True)
class StrategyPlan:
    """A small, deterministic action plan that an LLM can target.

    The SC2 executor consumes this plan instead of free-form natural text. That
    keeps realtime gameplay deterministic while allowing a later LLM layer to
    translate higher-level requests into these primitive steps.
    """

    actions: tuple[StrategyAction, ...]


class StrategyParseError(ValueError):
    """Raised when a user strategy command is not supported by the MVP parser."""


_COMMAND_SPLIT_RE = re.compile(r"\s*(?:;|\n+|\bthen\b)\s*", flags=re.IGNORECASE)

# Tiny deterministic routes for rule-based intent translation. These are not
# meant to be smart StarCraft strategy yet; they are stable target plans that an
# LLM can later learn to emit or refine.
_SCOUT_ROUTE = ((35.0, 42.0), (45.0, 42.0), (55.0, 45.0))
_RALLY_ROUTE = ((35.0, 42.0),)
_ATTACK_ROUTE = ((55.0, 45.0),)


def parse_strategy(text: str, default_unit: str = "worker") -> MoveCommand:
    """Parse one supported movement command for backward compatibility."""

    action = parse_strategy_action(text, default_unit=default_unit)
    if not isinstance(action, MoveCommand):
        raise StrategyParseError("single-command parser only supports 'move'")
    return action


def parse_strategy_request(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Parse any supported user-facing strategy input into a StrategyPlan.

    Accepted input forms, in priority order:
    1. JSON StrategyPlan, useful as the future LLM output contract.
    2. Deterministic DSL, e.g. ``move worker 35 42; wait 1``.
    3. Small rule-based natural-language intents, e.g. ``일꾼으로 정찰해``.
    """

    stripped = text.strip()
    if not stripped:
        raise StrategyParseError("strategy command is empty")

    if _looks_like_json(stripped):
        return parse_strategy_plan_json(stripped, default_unit=default_unit)

    try:
        return parse_strategy_plan(stripped, default_unit=default_unit)
    except StrategyParseError as dsl_error:
        try:
            return translate_strategy_intent(stripped, default_unit=default_unit)
        except StrategyParseError as intent_error:
            raise StrategyParseError(
                f"could not parse strategy as DSL, JSON plan, or known intent: {intent_error}"
            ) from dsl_error


def parse_strategy_plan(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Parse a small deterministic DSL strategy plan.

    Supported primitive actions:
    - move worker 35 42
    - attack marine 55 45
    - attack enemy
    - wait 2
    - wait until minerals 100
    - wait until structure supply depot ready
    - wait until unit marine 1
    - gather minerals
    - gather gas
    - train scv
    - train marine 2
    - build supply depot
    - build barracks
    - build refinery

    Multiple actions can be separated by semicolons, newlines, or the word
    "then", for example: "move worker 35 42; wait 1; move worker 42 42".
    """

    chunks = [chunk.strip() for chunk in _COMMAND_SPLIT_RE.split(text.strip()) if chunk.strip()]
    if not chunks:
        raise StrategyParseError("strategy command is empty")

    return StrategyPlan(
        actions=tuple(parse_strategy_action(chunk, default_unit=default_unit) for chunk in chunks)
    )


def parse_strategy_action(text: str, default_unit: str = "worker") -> StrategyAction:
    parts = text.strip().split()
    if not parts:
        raise StrategyParseError("strategy command is empty")

    verb = parts[0].lower()
    if verb == "move":
        return _parse_move(parts, default_unit=default_unit)
    if verb in {"attack", "attack_move", "attack-move"}:
        return _parse_attack(parts, default_unit=default_unit)
    if verb == "wait":
        return _parse_wait(parts)
    if verb == "gather":
        return _parse_gather(parts, default_unit=default_unit)
    if verb == "train":
        return _parse_train(parts)
    if verb == "build":
        return _parse_build(parts)

    raise StrategyParseError("only 'move', 'attack', 'wait', 'gather', 'train', and 'build' are supported")


def parse_strategy_plan_json(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Parse the canonical JSON StrategyPlan format.

    Expected object form:
    {
      "actions": [
        {"type": "move", "unit": "worker", "x": 35, "y": 42},
        {"type": "attack", "unit": "marine", "x": 55, "y": 45},
        {"type": "attack_enemy", "unit": "marine"},
        {"type": "wait", "seconds": 1},
        {"type": "wait_until", "condition": "minerals", "at_least": 100},
        {"type": "wait_until", "condition": "structure_ready", "target": "supply_depot", "at_least": 1},
        {"type": "gather", "unit": "worker", "resource": "minerals"},
        {"type": "gather", "unit": "worker", "resource": "vespene"},
        {"type": "train", "unit": "scv", "count": 2},
        {"type": "build", "building": "supply_depot"}
      ]
    }

    A bare JSON array of action objects is also accepted for convenience.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StrategyParseError(f"invalid strategy JSON: {exc.msg}") from exc

    return strategy_plan_from_dict(payload, default_unit=default_unit)


def strategy_plan_from_dict(payload: Any, default_unit: str = "worker") -> StrategyPlan:
    if isinstance(payload, list):
        actions_payload = payload
    elif isinstance(payload, dict):
        if "actions" not in payload:
            raise StrategyParseError("strategy JSON object must contain an 'actions' field")
        actions_payload = payload["actions"]
    else:
        raise StrategyParseError("strategy JSON must be an object or an actions array")

    if not isinstance(actions_payload, list) or not actions_payload:
        raise StrategyParseError("strategy JSON 'actions' must be a non-empty array")

    return StrategyPlan(
        actions=tuple(_action_from_dict(action, default_unit=default_unit) for action in actions_payload)
    )


def strategy_plan_to_dict(plan: StrategyPlan) -> dict[str, list[dict[str, object]]]:
    return {"actions": [_action_to_dict(action) for action in plan.actions]}


def strategy_plan_to_json(plan: StrategyPlan) -> str:
    return json.dumps(strategy_plan_to_dict(plan), ensure_ascii=False, indent=2)


def translate_strategy_intent(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Translate a tiny set of natural-language intents into StrategyPlan.

    This is deliberately rule-based, deterministic, and small. It creates the
    seam where an LLM will later plug in: the LLM should produce the same JSON
    StrategyPlan shape that this translator returns.
    """

    normalized = _normalize_intent(text)
    unit = _unit_from_intent(normalized, default_unit=default_unit)

    if any(keyword in normalized for keyword in ("정찰", "scout", "scouting")):
        return _route_plan(unit=unit, route=_SCOUT_ROUTE, wait_seconds=1)

    if any(keyword in normalized for keyword in ("공격", "attack", "rush")):
        return StrategyPlan(actions=tuple(AttackMoveCommand(unit=unit, x=x, y=y) for x, y in _ATTACK_ROUTE))

    if any(keyword in normalized for keyword in ("집결", "전진", "rally", "advance")):
        return _route_plan(unit=unit, route=_RALLY_ROUTE, wait_seconds=0)

    if any(keyword in normalized for keyword in ("서플", "보급고", "supply depot", "supply_depot")):
        return StrategyPlan(actions=(BuildStructureCommand(building="supply_depot"),))

    if any(keyword in normalized for keyword in ("병영", "배럭", "barracks")):
        return StrategyPlan(actions=(BuildStructureCommand(building="barracks"),))

    if any(keyword in normalized for keyword in ("정제소", "refinery", "gas")):
        return StrategyPlan(actions=(BuildStructureCommand(building="refinery"),))

    if any(keyword in normalized for keyword in ("자원", "미네랄", "mineral", "minerals", "gather")):
        return StrategyPlan(actions=(GatherMineralsCommand(unit="worker"),))

    if any(keyword in normalized for keyword in ("마린 생산", "해병 생산", "train marine", "make marine")):
        return StrategyPlan(actions=(TrainUnitCommand(unit="marine"),))

    if any(keyword in normalized for keyword in ("일꾼 생산", "scv 생산", "train scv", "make scv")):
        return StrategyPlan(actions=(TrainUnitCommand(unit="scv"),))

    raise StrategyParseError(f"unknown strategy intent: {text}")


def _parse_move(parts: list[str], default_unit: str) -> MoveCommand:
    if len(parts) == 3:
        unit = default_unit
        x_text, y_text = parts[1:]
    elif len(parts) == 4:
        unit = normalize_unit(parts[1])
        x_text, y_text = parts[2:]
    else:
        raise StrategyParseError("use: move worker 35 42 or move 35 42")

    x, y = _parse_coordinates(x_text, y_text)
    return MoveCommand(unit=unit, x=x, y=y)


def _parse_attack(parts: list[str], default_unit: str) -> AttackMoveCommand | AttackEnemyCommand:
    if len(parts) >= 2 and parts[1].lower() == "move":
        parts = [parts[0], *parts[2:]]

    if _is_enemy_attack(parts):
        unit = normalize_unit(parts[1]) if len(parts) >= 3 and parts[1].lower() != "nearest" else "marine"
        return AttackEnemyCommand(unit=unit)

    if len(parts) == 3:
        unit = default_unit
        x_text, y_text = parts[1:]
    elif len(parts) == 4:
        unit = normalize_unit(parts[1])
        x_text, y_text = parts[2:]
    else:
        raise StrategyParseError("use: attack marine 55 45, attack move marine 55 45, or attack 55 45")

    x, y = _parse_coordinates(x_text, y_text)
    return AttackMoveCommand(unit=unit, x=x, y=y)


def _is_enemy_attack(parts: list[str]) -> bool:
    lowered = [part.lower() for part in parts]
    return (
        lowered == ["attack", "enemy"]
        or lowered == ["attack", "nearest", "enemy"]
        or (len(lowered) == 3 and lowered[2] in {"enemy", "enemies"})
        or (len(lowered) == 4 and lowered[2:] == ["nearest", "enemy"])
    )


def _parse_wait(parts: list[str]) -> WaitCommand | WaitUntilCommand:
    if len(parts) >= 2 and parts[1].lower() == "until":
        return _parse_wait_until(parts[2:])

    if len(parts) != 2:
        raise StrategyParseError("use: wait 2 or wait until minerals 100")

    try:
        seconds = float(parts[1])
    except ValueError as exc:
        raise StrategyParseError("wait duration must be a number of seconds") from exc

    if seconds < 0:
        raise StrategyParseError("wait duration must not be negative")

    return WaitCommand(seconds=seconds)


def _parse_wait_until(parts: list[str]) -> WaitUntilCommand:
    if not parts:
        raise StrategyParseError(
            "use: wait until minerals 100, wait until structure supply depot ready, or wait until unit marine 1"
        )

    metric = parts[0].lower().replace("-", "_")
    if metric in {"mineral", "minerals", "vespene", "gas"}:
        if len(parts) != 2:
            raise StrategyParseError(f"use: wait until {metric} 100")
        condition = "vespene" if metric in {"vespene", "gas"} else "minerals"
        return WaitUntilCommand(condition=condition, at_least=_parse_at_least(parts[1]))

    if metric in {"supply_left", "supply"}:
        if metric == "supply" and len(parts) >= 2 and parts[1].lower() == "left":
            value_parts = parts[2:]
        else:
            value_parts = parts[1:]
        if len(value_parts) != 1:
            raise StrategyParseError("use: wait until supply left 1 or wait until supply_left 1")
        return WaitUntilCommand(condition="supply_left", at_least=_parse_at_least(value_parts[0]))

    if metric in {"structure", "building"}:
        return _parse_wait_until_structure(parts[1:])

    if metric in {"unit", "units"}:
        if len(parts) != 3:
            raise StrategyParseError("use: wait until unit marine 1")
        return WaitUntilCommand(
            condition="unit_count",
            target=normalize_wait_unit(parts[1]),
            at_least=_parse_at_least(parts[2]),
        )

    raise StrategyParseError(f"unsupported wait-until condition: {parts[0]}")


def _parse_wait_until_structure(parts: list[str]) -> WaitUntilCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: wait until structure supply depot ready")

    status_words = {"ready", "complete", "completed", "pending", "started", "count", "total"}
    status_index = next((index for index, part in enumerate(parts) if part.lower() in status_words), None)
    if status_index is None:
        raise StrategyParseError("structure wait must end with ready, pending, started, count, or total")

    building_text = " ".join(parts[:status_index])
    if not building_text:
        raise StrategyParseError("structure wait is missing the structure name")

    status = parts[status_index].lower()
    remaining = parts[status_index + 1 :]
    if len(remaining) > 1:
        raise StrategyParseError("structure wait count must be a single number")
    at_least = _parse_at_least(remaining[0]) if remaining else 1

    if status in {"ready", "complete", "completed"}:
        condition = "structure_ready"
    elif status in {"pending", "started"}:
        condition = "structure_pending"
    else:
        condition = "structure_count"

    return WaitUntilCommand(condition=condition, target=normalize_building(building_text), at_least=at_least)


def _parse_gather(parts: list[str], default_unit: str) -> GatherMineralsCommand | GatherGasCommand:
    if len(parts) == 2:
        unit = default_unit
        resource = parts[1]
    elif len(parts) == 3:
        unit = normalize_unit(parts[1])
        resource = parts[2]
    else:
        raise StrategyParseError("use: gather minerals or gather worker minerals")

    if unit != "worker":
        raise StrategyParseError("only workers can gather resources in this MVP")

    normalized_resource = resource.strip().lower()
    if normalized_resource in {"mineral", "minerals", "미네랄"}:
        return GatherMineralsCommand(unit=unit)
    if normalized_resource in {"gas", "vespene", "vespene_gas", "가스", "베스핀"}:
        return GatherGasCommand(unit=unit)

    raise StrategyParseError("only mineral and gas gathering are supported in this MVP")


def _parse_train(parts: list[str]) -> TrainUnitCommand:
    if len(parts) not in {2, 3}:
        raise StrategyParseError("use: train scv, train marine, or train marine 2")

    count = _parse_positive_int(parts[2], "train count") if len(parts) == 3 else 1
    return TrainUnitCommand(unit=normalize_train_unit(parts[1]), count=count)


def _parse_build(parts: list[str]) -> BuildStructureCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: build supply depot, build barracks, or build refinery")
    return BuildStructureCommand(building=normalize_building(" ".join(parts[1:])))


def _action_from_dict(payload: Any, default_unit: str) -> StrategyAction:
    if not isinstance(payload, dict):
        raise StrategyParseError("each strategy JSON action must be an object")

    action_type = str(payload.get("type", "")).strip().lower()
    if action_type == "move":
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        x = _required_number(payload, "x")
        y = _required_number(payload, "y")
        return MoveCommand(unit=unit, x=x, y=y)

    if action_type in {"attack", "attack_move", "attack-move"}:
        target = str(payload.get("target", "")).strip().lower()
        if target in {"enemy", "nearest_enemy", "nearest enemy"}:
            return AttackEnemyCommand(unit=normalize_unit(str(payload.get("unit", default_unit))))
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        x = _required_number(payload, "x")
        y = _required_number(payload, "y")
        return AttackMoveCommand(unit=unit, x=x, y=y)

    if action_type in {"attack_enemy", "attack-enemy"}:
        return AttackEnemyCommand(unit=normalize_unit(str(payload.get("unit", "marine"))))

    if action_type == "wait":
        seconds = _required_number(payload, "seconds")
        if seconds < 0:
            raise StrategyParseError("wait duration must not be negative")
        return WaitCommand(seconds=seconds)

    if action_type in {"wait_until", "wait-until"}:
        return _wait_until_from_dict(payload)

    if action_type in {"gather", "gather_minerals"}:
        resource = str(payload.get("resource", "minerals")).strip().lower()
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        if unit != "worker":
            raise StrategyParseError("only workers can gather resources in this MVP")
        if resource in {"mineral", "minerals", "미네랄"}:
            return GatherMineralsCommand(unit=unit)
        if resource in {"gas", "vespene", "vespene_gas", "가스", "베스핀"}:
            return GatherGasCommand(unit=unit)
        raise StrategyParseError("only mineral and gas gathering are supported in this MVP")

    if action_type in {"gather_gas", "gather-gas", "gather_vespene"}:
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        if unit != "worker":
            raise StrategyParseError("only workers can gather gas in this MVP")
        return GatherGasCommand(unit=unit)

    if action_type == "train":
        return TrainUnitCommand(
            unit=normalize_train_unit(str(payload.get("unit", ""))),
            count=_positive_int_from_payload(payload, "count", default=1),
        )

    if action_type == "build":
        return BuildStructureCommand(
            building=normalize_building(str(payload.get("building", ""))),
            worker=normalize_unit(str(payload.get("worker", "worker"))),
        )

    raise StrategyParseError(f"unsupported JSON action type: {action_type!r}")


def _action_to_dict(action: StrategyAction) -> dict[str, object]:
    if isinstance(action, MoveCommand):
        return {"type": "move", "unit": action.unit, "x": action.x, "y": action.y}
    if isinstance(action, AttackMoveCommand):
        return {"type": "attack", "unit": action.unit, "x": action.x, "y": action.y}
    if isinstance(action, AttackEnemyCommand):
        return {"type": "attack_enemy", "unit": action.unit}
    if isinstance(action, WaitCommand):
        return {"type": "wait", "seconds": action.seconds}
    if isinstance(action, WaitUntilCommand):
        payload: dict[str, object] = {
            "type": "wait_until",
            "condition": action.condition,
            "at_least": action.at_least,
        }
        if action.target is not None:
            payload["target"] = action.target
        return payload
    if isinstance(action, GatherMineralsCommand):
        return {"type": "gather", "unit": action.unit, "resource": "minerals"}
    if isinstance(action, GatherGasCommand):
        return {"type": "gather", "unit": action.unit, "resource": "vespene"}
    if isinstance(action, TrainUnitCommand):
        payload: dict[str, object] = {"type": "train", "unit": action.unit}
        if action.count != 1:
            payload["count"] = action.count
        return payload
    if isinstance(action, BuildStructureCommand):
        return {"type": "build", "building": action.building, "worker": action.worker}
    raise TypeError(f"unsupported strategy action: {action!r}")


def _parse_coordinates(x_text: str, y_text: str) -> tuple[float, float]:
    try:
        return float(x_text), float(y_text)
    except ValueError as exc:
        raise StrategyParseError("coordinates must be numbers") from exc


def _required_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise StrategyParseError(f"strategy JSON action is missing required field: {key}")
    value = payload[key]
    if isinstance(value, bool):
        raise StrategyParseError(f"strategy JSON field must be numeric: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyParseError(f"strategy JSON field must be numeric: {key}") from exc


def _optional_number(payload: dict[str, Any], keys: tuple[str, ...], default: float | None = None) -> float:
    for key in keys:
        if key in payload:
            return _required_number(payload, key)
    if default is not None:
        return default
    raise StrategyParseError(f"strategy JSON action is missing one of required fields: {', '.join(keys)}")


def _parse_at_least(value: str) -> float:
    try:
        at_least = float(value)
    except ValueError as exc:
        raise StrategyParseError("wait-until threshold must be numeric") from exc
    if at_least < 0:
        raise StrategyParseError("wait-until threshold must not be negative")
    return at_least


def _parse_positive_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise StrategyParseError(f"{field_name} must be an integer") from exc
    if parsed < 1:
        raise StrategyParseError(f"{field_name} must be at least 1")
    if parsed > 20:
        raise StrategyParseError(f"{field_name} is too high for the MVP")
    return parsed


def _positive_int_from_payload(payload: dict[str, Any], key: str, default: int) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}") from exc
    if parsed < 1:
        raise StrategyParseError(f"strategy JSON field must be at least 1: {key}")
    if parsed > 20:
        raise StrategyParseError(f"strategy JSON field is too high for the MVP: {key}")
    return parsed


def _wait_until_from_dict(payload: dict[str, Any]) -> WaitUntilCommand:
    condition = normalize_wait_condition(str(payload.get("condition", "")))
    at_least = _optional_number(payload, ("at_least", "count", "value", "minimum"), default=1)
    if at_least < 0:
        raise StrategyParseError("wait-until threshold must not be negative")

    target: str | None = None
    if condition.startswith("structure_"):
        target_value = payload.get("target", payload.get("structure", payload.get("building", "")))
        target = normalize_building(str(target_value))
    elif condition == "unit_count":
        target_value = payload.get("target", payload.get("unit", ""))
        target = normalize_wait_unit(str(target_value))

    return WaitUntilCommand(condition=condition, target=target, at_least=at_least)


def _looks_like_json(text: str) -> bool:
    return text.startswith("{") or text.startswith("[")


def _normalize_intent(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _unit_from_intent(text: str, default_unit: str) -> str:
    if any(keyword in text for keyword in ("marine", "마린", "해병")):
        return "marine"
    if any(keyword in text for keyword in ("worker", "workers", "scv", "일꾼", "건설로봇")):
        return "worker"
    return normalize_unit(default_unit)


def _route_plan(unit: str, route: tuple[tuple[float, float], ...], wait_seconds: float) -> StrategyPlan:
    actions: list[StrategyAction] = []
    for index, (x, y) in enumerate(route):
        if index > 0 and wait_seconds > 0:
            actions.append(WaitCommand(seconds=wait_seconds))
        actions.append(MoveCommand(unit=unit, x=x, y=y))
    return StrategyPlan(actions=tuple(actions))


def normalize_train_unit(unit: str) -> str:
    normalized = unit.strip().lower().replace("_", " ")
    aliases = {
        "scv": "scv",
        "worker": "scv",
        "workers": "scv",
        "일꾼": "scv",
        "건설로봇": "scv",
        "marine": "marine",
        "marines": "marine",
        "마린": "marine",
        "해병": "marine",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported train unit for MVP: {unit}")
    return aliases[normalized]


def normalize_wait_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    aliases = {
        "scv": "worker",
        "worker": "worker",
        "workers": "worker",
        "일꾼": "worker",
        "건설로봇": "worker",
        "marine": "marine",
        "marines": "marine",
        "마린": "marine",
        "해병": "marine",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported wait-until unit for MVP: {unit}")
    return aliases[normalized]


def normalize_wait_condition(condition: str) -> str:
    normalized = condition.strip().lower().replace("-", "_")
    aliases = {
        "mineral": "minerals",
        "minerals": "minerals",
        "vespene": "vespene",
        "gas": "vespene",
        "supply_left": "supply_left",
        "supply": "supply_left",
        "structure": "structure_count",
        "building": "structure_count",
        "structure_count": "structure_count",
        "building_count": "structure_count",
        "structure_total": "structure_count",
        "structure_ready": "structure_ready",
        "building_ready": "structure_ready",
        "structure_complete": "structure_ready",
        "structure_completed": "structure_ready",
        "structure_pending": "structure_pending",
        "building_pending": "structure_pending",
        "structure_started": "structure_pending",
        "unit": "unit_count",
        "units": "unit_count",
        "unit_count": "unit_count",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported wait-until condition: {condition}")
    return aliases[normalized]


def normalize_building(building: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", building.strip().lower())
    aliases = {
        "supply_depot": "supply_depot",
        "depot": "supply_depot",
        "supply": "supply_depot",
        "서플": "supply_depot",
        "보급고": "supply_depot",
        "barracks": "barracks",
        "rax": "barracks",
        "배럭": "barracks",
        "병영": "barracks",
        "refinery": "refinery",
        "gas": "refinery",
        "정제소": "refinery",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported build structure for MVP: {building}")
    return aliases[normalized]


def normalize_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    aliases = {
        "scv": "worker",
        "probe": "worker",
        "drone": "worker",
        "worker": "worker",
        "workers": "worker",
        "일꾼": "worker",
        "건설로봇": "worker",
        "marine": "marine",
        "marines": "marine",
        "마린": "marine",
        "해병": "marine",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported unit for MVP: {unit}")
    return aliases[normalized]
