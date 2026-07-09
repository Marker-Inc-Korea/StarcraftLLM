from __future__ import annotations

import math

from starcraft_llm.game_state import GameStateSummary
from starcraft_llm.strategy import (
    AttackEnemyCommand,
    AttackMoveCommand,
    BuildStructureCommand,
    GatherGasCommand,
    GatherMineralsCommand,
    MoveCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
    WaitUntilCommand,
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
    vespene = game_state.vespene if game_state is not None else None
    supply_left = game_state.supply.left if game_state is not None else None
    workers = game_state.workers if game_state is not None else None
    structures = dict(game_state.structures) if game_state is not None else {}
    structures_pending = dict(game_state.structures_pending) if game_state is not None else {}
    if game_state is not None and game_state.structures_ready:
        structures_ready = dict(game_state.structures_ready)
    elif structures_pending:
        structures_ready = {}
    else:
        structures_ready = dict(structures)
    units = dict(game_state.army) if game_state is not None else {}
    if game_state is not None:
        units["worker"] = game_state.workers

    for index, action in enumerate(plan.actions, start=1):
        if isinstance(action, MoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "move", min_coordinate, max_coordinate)
        elif isinstance(action, AttackMoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "attack", min_coordinate, max_coordinate)
        elif isinstance(action, AttackEnemyCommand):
            _validate_unit(action.unit, index, "attack enemy")
        elif isinstance(action, WaitCommand):
            _validate_wait(action, index)
        elif isinstance(action, WaitUntilCommand):
            minerals, vespene, supply_left, workers = _validate_wait_until(
                action,
                index,
                game_state,
                minerals,
                vespene,
                supply_left,
                workers,
                structures,
                structures_ready,
                structures_pending,
                units,
            )
        elif isinstance(action, GatherMineralsCommand):
            _validate_gather(action, index, game_state)
        elif isinstance(action, GatherGasCommand):
            _validate_gather_gas(action, index, game_state, structures_ready)
        elif isinstance(action, TrainUnitCommand):
            minerals, supply_left, workers = _validate_train(
                action, index, game_state, minerals, supply_left, workers, structures_ready, units
            )
        elif isinstance(action, BuildStructureCommand):
            minerals = _validate_build(
                action, index, game_state, minerals, structures, structures_ready, structures_pending
            )
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
    _validate_unit(unit, index, action_name)
    if not math.isfinite(x) or not math.isfinite(y):
        raise PlanValidationError(f"action {index}: {action_name} coordinates must be finite")
    if not (min_coordinate <= x <= max_coordinate and min_coordinate <= y <= max_coordinate):
        raise PlanValidationError(
            f"action {index}: {action_name} coordinates ({x:g}, {y:g}) are outside "
            f"the safe range {min_coordinate:g}..{max_coordinate:g}"
        )


def _validate_unit(unit: str, index: int, action_name: str) -> None:
    if unit not in {"worker", "marine"}:
        raise PlanValidationError(f"action {index}: unsupported {action_name} unit: {unit}")


def _validate_wait(action: WaitCommand, index: int) -> None:
    if not math.isfinite(action.seconds):
        raise PlanValidationError(f"action {index}: wait duration must be finite")
    if action.seconds < 0:
        raise PlanValidationError(f"action {index}: wait duration must not be negative")
    if action.seconds > 30:
        raise PlanValidationError(f"action {index}: wait duration is too long for the MVP: {action.seconds:g}s")


def _validate_wait_until(
    action: WaitUntilCommand,
    index: int,
    game_state: GameStateSummary | None,
    minerals: int | None,
    vespene: int | None,
    supply_left: int | None,
    workers: int | None,
    structures: dict[str, int],
    structures_ready: dict[str, int],
    structures_pending: dict[str, int],
    units: dict[str, int],
) -> tuple[int | None, int | None, int | None, int | None]:
    if action.condition not in {
        "minerals",
        "vespene",
        "supply_left",
        "structure_count",
        "structure_ready",
        "structure_pending",
        "unit_count",
    }:
        raise PlanValidationError(f"action {index}: unsupported wait-until condition: {action.condition}")
    if not math.isfinite(action.at_least):
        raise PlanValidationError(f"action {index}: wait-until threshold must be finite")
    if action.at_least < 0:
        raise PlanValidationError(f"action {index}: wait-until threshold must not be negative")
    if action.at_least > 10000:
        raise PlanValidationError(f"action {index}: wait-until threshold is too high for the MVP")

    if action.condition in {"structure_count", "structure_ready", "structure_pending", "unit_count"}:
        if not action.target:
            raise PlanValidationError(f"action {index}: wait-until {action.condition} requires a target")

    if game_state is None:
        return minerals, vespene, supply_left, workers

    threshold = int(math.ceil(action.at_least))
    if action.condition == "minerals":
        return max(minerals or 0, threshold), vespene, supply_left, workers
    if action.condition == "vespene":
        return minerals, max(vespene or 0, threshold), supply_left, workers
    if action.condition == "supply_left":
        return minerals, vespene, max(supply_left or 0, threshold), workers

    if action.condition == "unit_count":
        target = action.target or ""
        current = workers if target == "worker" else units.get(target, 0)
        if current < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} {target} unit(s); "
                f"only {current:g} are currently known or planned"
            )
        return minerals, vespene, supply_left, max(workers or 0, threshold) if target == "worker" else workers

    target_key = _structure_state_key(action.target or "")
    if action.condition == "structure_count":
        current = structures.get(target_key, 0)
        if current < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} {action.target} structure(s); "
                f"only {current:g} are currently known or planned"
            )
        return minerals, vespene, supply_left, workers

    if action.condition == "structure_pending":
        current_pending = structures_pending.get(target_key, 0)
        current_total = structures.get(target_key, 0)
        if max(current_pending, current_total) < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} pending {action.target}; "
                f"only {current_total:g} are currently known or planned"
            )
        structures_pending[target_key] = max(current_pending, threshold)
        return minerals, vespene, supply_left, workers

    current_total = structures.get(target_key, 0)
    if current_total < threshold:
        raise PlanValidationError(
            f"action {index}: cannot wait for {threshold:g} ready {action.target}; "
            f"only {current_total:g} are currently known or planned"
        )
    structures_ready[target_key] = max(structures_ready.get(target_key, 0), threshold)
    structures_pending[target_key] = max(0, structures_pending.get(target_key, 0) - threshold)
    return minerals, vespene, supply_left, workers


def _validate_gather(action: GatherMineralsCommand, index: int, game_state: GameStateSummary | None) -> None:
    if action.unit != "worker":
        raise PlanValidationError(f"action {index}: only workers can gather minerals")
    if game_state is not None and game_state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot gather minerals with no workers")


def _validate_gather_gas(
    action: GatherGasCommand,
    index: int,
    game_state: GameStateSummary | None,
    structures_ready: dict[str, int],
) -> None:
    if action.unit != "worker":
        raise PlanValidationError(f"action {index}: only workers can gather gas")
    if game_state is not None and game_state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot gather gas with no workers")
    if game_state is not None and structures_ready.get("refinery", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot gather gas without a ready refinery")


def _validate_train(
    action: TrainUnitCommand,
    index: int,
    game_state: GameStateSummary | None,
    minerals: int | None,
    supply_left: int | None,
    workers: int | None,
    structures_ready: dict[str, int],
    units: dict[str, int],
) -> tuple[int | None, int | None, int | None]:
    if action.unit not in _TRAIN_COSTS:
        raise PlanValidationError(f"action {index}: unsupported train unit: {action.unit}")
    if action.count < 1:
        raise PlanValidationError(f"action {index}: train count must be at least 1")
    if action.count > 20:
        raise PlanValidationError(f"action {index}: train count is too high for the MVP")

    if game_state is None:
        return minerals, supply_left, workers

    if action.unit == "scv" and game_state.townhalls < 1:
        raise PlanValidationError(f"action {index}: cannot train SCV without a townhall")
    if action.unit == "marine" and structures_ready.get("barracks", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot train marine without a barracks")

    cost = _TRAIN_COSTS[action.unit]
    total_cost = cost * action.count
    if minerals is not None and minerals < total_cost:
        raise PlanValidationError(f"action {index}: cannot train {action.count} {action.unit} with only {minerals} minerals")
    if supply_left is not None and supply_left < action.count:
        if supply_left < 1:
            raise PlanValidationError(f"action {index}: cannot train {action.unit} with no supply left")
        raise PlanValidationError(f"action {index}: cannot train {action.count} {action.unit} with only {supply_left} supply left")

    if action.unit == "scv":
        units["worker"] = units.get("worker", 0) + action.count
        workers = (workers or 0) + action.count
    else:
        units[action.unit] = units.get(action.unit, 0) + action.count

    return (
        minerals - total_cost if minerals is not None else None,
        supply_left - action.count if supply_left is not None else None,
        workers,
    )


def _validate_build(
    action: BuildStructureCommand,
    index: int,
    game_state: GameStateSummary | None,
    minerals: int | None,
    structures: dict[str, int],
    structures_ready: dict[str, int],
    structures_pending: dict[str, int],
) -> int | None:
    if action.worker != "worker":
        raise PlanValidationError(f"action {index}: only workers can build structures")
    if action.building not in _BUILD_COSTS:
        raise PlanValidationError(f"action {index}: unsupported build structure: {action.building}")

    if game_state is None:
        return minerals

    if game_state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot build {action.building} with no workers")
    if action.building == "barracks" and structures_ready.get("supplydepot", 0) + structures_ready.get("supply_depot", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot build barracks before a supply depot exists")

    cost = _BUILD_COSTS[action.building]
    if minerals is not None and minerals < cost:
        raise PlanValidationError(f"action {index}: cannot build {action.building} with only {minerals} minerals")

    structure_key = _structure_state_key(action.building)
    structures[structure_key] = structures.get(structure_key, 0) + 1
    structures_pending[structure_key] = structures_pending.get(structure_key, 0) + 1
    return minerals - cost if minerals is not None else None


def _structure_state_key(building: str) -> str:
    if building == "supply_depot":
        return "supplydepot"
    return building
