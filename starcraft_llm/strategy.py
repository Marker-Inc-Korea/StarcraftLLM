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
class WaitCommand:
    """Pause strategy execution for a small amount of game-clock time."""

    seconds: float


@dataclass(frozen=True)
class GatherMineralsCommand:
    """Send a logical worker group to nearby mineral fields."""

    unit: str = "worker"


@dataclass(frozen=True)
class TrainUnitCommand:
    """Train one unit from an available production structure."""

    unit: str


StrategyAction: TypeAlias = MoveCommand | WaitCommand | GatherMineralsCommand | TrainUnitCommand


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
    - move marine 35 42
    - move 35 42
    - wait 2
    - gather minerals
    - train scv

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
    if verb == "wait":
        return _parse_wait(parts)
    if verb == "gather":
        return _parse_gather(parts, default_unit=default_unit)
    if verb == "train":
        return _parse_train(parts)

    raise StrategyParseError("only 'move', 'wait', 'gather', and 'train' are supported in this MVP")


def parse_strategy_plan_json(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Parse the canonical JSON StrategyPlan format.

    Expected object form:
    {
      "actions": [
        {"type": "move", "unit": "worker", "x": 35, "y": 42},
        {"type": "wait", "seconds": 1},
        {"type": "gather", "unit": "worker", "resource": "minerals"},
        {"type": "train", "unit": "scv"}
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

    if any(keyword in normalized for keyword in ("집결", "전진", "rally", "advance")):
        return _route_plan(unit=unit, route=_RALLY_ROUTE, wait_seconds=0)

    if any(keyword in normalized for keyword in ("자원", "미네랄", "mineral", "minerals", "gather")):
        return StrategyPlan(actions=(GatherMineralsCommand(unit="worker"),))

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

    try:
        x = float(x_text)
        y = float(y_text)
    except ValueError as exc:
        raise StrategyParseError("move coordinates must be numbers") from exc

    return MoveCommand(unit=unit, x=x, y=y)


def _parse_wait(parts: list[str]) -> WaitCommand:
    if len(parts) != 2:
        raise StrategyParseError("use: wait 2")

    try:
        seconds = float(parts[1])
    except ValueError as exc:
        raise StrategyParseError("wait duration must be a number of seconds") from exc

    if seconds < 0:
        raise StrategyParseError("wait duration must not be negative")

    return WaitCommand(seconds=seconds)


def _parse_gather(parts: list[str], default_unit: str) -> GatherMineralsCommand:
    if len(parts) == 2:
        unit = default_unit
        resource = parts[1]
    elif len(parts) == 3:
        unit = normalize_unit(parts[1])
        resource = parts[2]
    else:
        raise StrategyParseError("use: gather minerals or gather worker minerals")

    if resource.strip().lower() not in {"mineral", "minerals", "미네랄"}:
        raise StrategyParseError("only mineral gathering is supported in this MVP")

    if unit != "worker":
        raise StrategyParseError("only workers can gather minerals in this MVP")

    return GatherMineralsCommand(unit=unit)


def _parse_train(parts: list[str]) -> TrainUnitCommand:
    if len(parts) != 2:
        raise StrategyParseError("use: train scv")

    return TrainUnitCommand(unit=normalize_train_unit(parts[1]))


def _action_from_dict(payload: Any, default_unit: str) -> StrategyAction:
    if not isinstance(payload, dict):
        raise StrategyParseError("each strategy JSON action must be an object")

    action_type = str(payload.get("type", "")).strip().lower()
    if action_type == "move":
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        x = _required_number(payload, "x")
        y = _required_number(payload, "y")
        return MoveCommand(unit=unit, x=x, y=y)

    if action_type == "wait":
        seconds = _required_number(payload, "seconds")
        if seconds < 0:
            raise StrategyParseError("wait duration must not be negative")
        return WaitCommand(seconds=seconds)

    if action_type in {"gather", "gather_minerals"}:
        resource = str(payload.get("resource", "minerals")).strip().lower()
        if resource not in {"mineral", "minerals", "미네랄"}:
            raise StrategyParseError("only mineral gathering is supported in this MVP")
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        if unit != "worker":
            raise StrategyParseError("only workers can gather minerals in this MVP")
        return GatherMineralsCommand(unit=unit)

    if action_type == "train":
        return TrainUnitCommand(unit=normalize_train_unit(str(payload.get("unit", ""))))

    raise StrategyParseError(f"unsupported JSON action type: {action_type!r}")


def _action_to_dict(action: StrategyAction) -> dict[str, object]:
    if isinstance(action, MoveCommand):
        return {"type": "move", "unit": action.unit, "x": action.x, "y": action.y}
    if isinstance(action, WaitCommand):
        return {"type": "wait", "seconds": action.seconds}
    if isinstance(action, GatherMineralsCommand):
        return {"type": "gather", "unit": action.unit, "resource": "minerals"}
    if isinstance(action, TrainUnitCommand):
        return {"type": "train", "unit": action.unit}
    raise TypeError(f"unsupported strategy action: {action!r}")


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
    normalized = unit.strip().lower()
    aliases = {
        "scv": "scv",
        "worker": "scv",
        "workers": "scv",
        "일꾼": "scv",
        "건설로봇": "scv",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported train unit for MVP: {unit}")
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
