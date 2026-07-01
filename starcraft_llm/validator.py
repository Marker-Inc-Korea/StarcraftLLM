from __future__ import annotations

import math

from starcraft_llm.game_state import GameStateSummary
from starcraft_llm.strategy import GatherMineralsCommand, MoveCommand, StrategyPlan, TrainUnitCommand, WaitCommand


class PlanValidationError(ValueError):
    """Raised when a StrategyPlan is unsupported or unsafe to execute."""


def validate_strategy_plan(
    plan: StrategyPlan,
    game_state: GameStateSummary | None = None,
    max_actions: int = 8,
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

    for index, action in enumerate(plan.actions, start=1):
        if isinstance(action, MoveCommand):
            _validate_move(action, index, min_coordinate, max_coordinate)
        elif isinstance(action, WaitCommand):
            _validate_wait(action, index)
        elif isinstance(action, GatherMineralsCommand):
            _validate_gather(action, index, game_state)
        elif isinstance(action, TrainUnitCommand):
            minerals, supply_left = _validate_train(action, index, game_state, minerals, supply_left)
        else:
            raise PlanValidationError(f"action {index}: unsupported action object: {action!r}")

    return plan


def _validate_move(action: MoveCommand, index: int, min_coordinate: float, max_coordinate: float) -> None:
    if action.unit not in {"worker", "marine"}:
        raise PlanValidationError(f"action {index}: unsupported move unit: {action.unit}")
    if not math.isfinite(action.x) or not math.isfinite(action.y):
        raise PlanValidationError(f"action {index}: move coordinates must be finite")
    if not (min_coordinate <= action.x <= max_coordinate and min_coordinate <= action.y <= max_coordinate):
        raise PlanValidationError(
            f"action {index}: move coordinates ({action.x:g}, {action.y:g}) are outside "
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
) -> tuple[int | None, int | None]:
    if action.unit != "scv":
        raise PlanValidationError(f"action {index}: unsupported train unit: {action.unit}")

    if game_state is None:
        return minerals, supply_left

    if game_state.townhalls < 1:
        raise PlanValidationError(f"action {index}: cannot train SCV without a townhall")
    if minerals is not None and minerals < 50:
        raise PlanValidationError(f"action {index}: cannot train SCV with only {minerals} minerals")
    if supply_left is not None and supply_left < 1:
        raise PlanValidationError(f"action {index}: cannot train SCV with no supply left")

    return (minerals - 50 if minerals is not None else None, supply_left - 1 if supply_left is not None else None)
