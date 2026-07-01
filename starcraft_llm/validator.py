from __future__ import annotations

import math

from starcraft_llm.game_state import GameStateSummary
from starcraft_llm.strategy import (
    AttackMoveCommand,
    BuildStructureCommand,
    GatherMineralsCommand,
    MoveCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
)


class PlanValidationError(ValueError):
    """Raised when a StrategyPlan is unsupported or unsafe to execute."""


_TRAIN_COSTS = {"scv": 50, "marine": 50}
_BUILD_COSTS = {"supply_depot": 100, "barracks": 150, "refinery": 75}


def validate_strategy_plan(
    plan: StrategyPlan,
    game_state: GameStateSummary | None = None,
    max_actions: int = 10,
    min_coordinate: float = 0,
    max_coordinate: float = 256,
) -> StrategyPlan:
    """Validate a StrategyPlan before the SC2 executor runs it.

    The parser guarantees the basic JSON shape. This validator adds an execution
    safety boundary: action count, coordinate sanity, wait limits, and simple
    game-state feasibility checks for the MVP actions.
    """

    if not plan.actions:
        raise PlanValidationError("strategy plan must contain at least one action")
    if len(plan.actions) > max_actions:
        raise PlanValidationError(f"strategy plan has too many actions: {len(plan.actions)} > {max_actions}")

    minerals = game_state.minerals if game_state is not None else None
    supply_left = game_state.supply.left if game_state is not None else None
    structures = dict(game_state.structures) if game_state is not None else {}

    for index, action in enumerate(plan.actions, start=1):
        if isinstance(action, MoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "move", min_coordinate, max_coordinate)
        elif isinstance(action, AttackMoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "attack", min_coordinate, max_coordinate)
        elif isinstance(action, WaitCommand):
            _validate_wait(action, index)
        elif isinstance(action, GatherMineralsCommand):
            _validate_gather(action, index, game_state)
        elif isinstance(action, TrainUnitCommand):
            minerals, supply_left = _validate_train(action, index, game_state, minerals, supply_left, structures)
        elif isinstance(action, BuildStructureCommand):
            minerals = _validate_build(action, index, game_state, minerals, structures)
        else:
            raise PlanValidationError(f"action {index}: unsupported action object: {action!r}")

    return plan


def _validate_point_action(
    unit: str,
    x: float,
    y: float,
    index: int,
    action_name: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if unit not in {"worker", "marine"}:
        raise PlanValidationError(f"action {index}: unsupported {action_name} unit: {unit}")
    if not math.isfinite(x) or not math.isfinite(y):
        raise PlanValidationError(f"action {index}: {action_name} coordinates must be finite")
    if not (min_coordinate <= x <= max_coordinate and min_coordinate <= y <= max_coordinate):
        raise PlanValidationError(
            f"action {index}: {action_name} coordinates ({x:g}, {y:g}) are outside "
            f"the safe range {min_coordinate:g}..{max_coordinate:g}"
        )


def _validate_wait(action: WaitCommand, index: int) -> None:
    if not math.isfinite(action.seconds):
        raise PlanValidationError(f"action {index}: wait duration must be finite")
    if action.seconds < 0:
        raise PlanValidationError(f"action {index}: wait duration must not be negative")
    if action.seconds > 30:
        raise PlanValidationError(f"action {index}: wait duration is too long for the MVP: {action.seconds:g}s")


def _validate_gather(action: GatherMineralsCommand, index: int, game_state: GameStateSummary | None) -> None:
    if action.unit != "worker":
        raise PlanValidationError(f"action {index}: only workers can gather minerals")
    if game_state is not None and game_state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot gather minerals with no workers")


def _validate_train(
    action: TrainUnitCommand,
    index: int,
    game_state: GameStateSummary | None,
    minerals: int | None,
    supply_left: int | None,
    structures: dict[str, int],
) -> tuple[int | None, int | None]:
    if action.unit not in _TRAIN_COSTS:
        raise PlanValidationError(f"action {index}: unsupported train unit: {action.unit}")

    if game_state is None:
        return minerals, supply_left

    if action.unit == "scv" and game_state.townhalls < 1:
        raise PlanValidationError(f"action {index}: cannot train SCV without a townhall")
    if action.unit == "marine" and structures.get("barracks", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot train marine without a barracks")

    cost = _TRAIN_COSTS[action.unit]
    if minerals is not None and minerals < cost:
        raise PlanValidationError(f"action {index}: cannot train {action.unit} with only {minerals} minerals")
    if supply_left is not None and supply_left < 1:
        raise PlanValidationError(f"action {index}: cannot train {action.unit} with no supply left")

    return (minerals - cost if minerals is not None else None, supply_left - 1 if supply_left is not None else None)


def _validate_build(
    action: BuildStructureCommand,
    index: int,
    game_state: GameStateSummary | None,
    minerals: int | None,
    structures: dict[str, int],
) -> int | None:
    if action.worker != "worker":
        raise PlanValidationError(f"action {index}: only workers can build structures")
    if action.building not in _BUILD_COSTS:
        raise PlanValidationError(f"action {index}: unsupported build structure: {action.building}")

    if game_state is None:
        return minerals

    if game_state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot build {action.building} with no workers")
    if action.building == "barracks" and structures.get("supplydepot", 0) + structures.get("supply_depot", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot build barracks before a supply depot exists")

    cost = _BUILD_COSTS[action.building]
    if minerals is not None and minerals < cost:
        raise PlanValidationError(f"action {index}: cannot build {action.building} with only {minerals} minerals")

    structures[_structure_state_key(action.building)] = structures.get(_structure_state_key(action.building), 0) + 1
    return minerals - cost if minerals is not None else None


def _structure_state_key(building: str) -> str:
    if building == "supply_depot":
        return "supplydepot"
    return building
