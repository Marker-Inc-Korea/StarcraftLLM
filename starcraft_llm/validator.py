from __future__ import annotations

import math
from dataclasses import dataclass

from starcraft_llm.command_catalog import (
    ADDON_SPECS,
    MORPH_SPECS,
    REPAIRABLE_TARGET_KEYS,
    STRUCTURE_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    EntitySpec,
    normalize_name,
    resolve_alias,
)
from starcraft_llm.game_state import GameStateSummary
from starcraft_llm.strategy import (
    AttackEnemyCommand,
    AttackMoveCommand,
    BuildAddonCommand,
    BuildStructureCommand,
    DistributeWorkersCommand,
    ExpandCommand,
    GatherGasCommand,
    GatherMineralsCommand,
    HoldPositionCommand,
    MorphStructureCommand,
    MoveCommand,
    PatrolCommand,
    RallyCommand,
    RepairCommand,
    ResearchUpgradeCommand,
    StopCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
    WaitUntilCommand,
)


class PlanValidationError(ValueError):
    """Raised when a StrategyPlan is unsupported or unsafe to execute."""


_SUPPORTED_WAIT_CONDITIONS = {
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
}


@dataclass
class _PlanState:
    known: bool
    minerals: int
    vespene: int
    supply_used: int
    supply_cap: int
    supply_left: int
    workers: int
    townhalls: int
    game_time: float
    structures: dict[str, int]
    structures_ready: dict[str, int]
    structures_pending: dict[str, int]
    units: dict[str, int]
    upgrades: set[str]
    upgrades_planned: set[str]

    @classmethod
    def from_summary(cls, summary: GameStateSummary | None) -> "_PlanState":
        if summary is None:
            return cls(False, 0, 0, 0, 0, 0, 0, 0, 0.0, {}, {}, {}, {}, set(), set())

        structures = _canonical_counts(summary.structures, ("structure", "addon", "morph"))
        pending = _canonical_counts(summary.structures_pending, ("structure", "addon", "morph"))
        if summary.structures_ready:
            ready = _canonical_counts(summary.structures_ready, ("structure", "addon", "morph"))
        elif pending:
            ready = {}
        else:
            ready = dict(structures)

        units = _canonical_counts(summary.army, ("unit",))
        units["worker"] = summary.workers
        upgrades = set()
        for upgrade in summary.upgrades:
            try:
                upgrades.add(resolve_alias(upgrade, categories=("upgrade",)).key)
            except KeyError:
                upgrades.add(normalize_name(upgrade))

        return cls(
            True,
            int(summary.minerals),
            int(summary.vespene),
            int(summary.supply.used),
            int(summary.supply.cap),
            int(summary.supply.left),
            int(summary.workers),
            int(summary.townhalls),
            float(summary.game_time_seconds),
            structures,
            ready,
            pending,
            units,
            upgrades,
            set(),
        )

    def spend(self, spec: EntitySpec, count: int, index: int, action_name: str) -> None:
        if not self.known:
            return
        mineral_cost = spec.minerals * count
        gas_cost = spec.vespene * count
        if self.minerals < mineral_cost:
            raise PlanValidationError(
                f"action {index}: cannot {action_name} with only {self.minerals} minerals; "
                f"requires {mineral_cost}"
            )
        if self.vespene < gas_cost:
            raise PlanValidationError(
                f"action {index}: cannot {action_name} with only {self.vespene} vespene; "
                f"requires {gas_cost}"
            )
        self.minerals -= mineral_cost
        self.vespene -= gas_cost

    def require_ready(self, structure: str, index: int, action_name: str) -> None:
        if self.known and self.structures_ready.get(structure, 0) < 1:
            if action_name == "build barracks" and structure == "supply_depot":
                raise PlanValidationError(f"action {index}: cannot build barracks before a supply depot exists")
            raise PlanValidationError(
                f"action {index}: cannot {action_name} without a {structure} that is ready"
            )

    def start_structure(self, structure: str, count: int = 1) -> None:
        self.structures[structure] = self.structures.get(structure, 0) + count
        self.structures_pending[structure] = self.structures_pending.get(structure, 0) + count

    def mark_structure_ready(self, structure: str, threshold: int, index: int) -> None:
        current_total = self.structures.get(structure, 0)
        if current_total < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} ready {structure}; "
                f"only {current_total:g} are currently known or planned"
            )
        previous_ready = self.structures_ready.get(structure, 0)
        newly_ready = max(0, threshold - previous_ready)
        self.structures_ready[structure] = max(previous_ready, threshold)
        self.structures_pending[structure] = max(0, self.structures_pending.get(structure, 0) - newly_ready)
        if structure == "supply_depot" and newly_ready:
            self.supply_cap += 8 * newly_ready
            self.supply_left += 8 * newly_ready
        elif structure == "command_center" and newly_ready:
            self.supply_cap += 15 * newly_ready
            self.supply_left += 15 * newly_ready


def validate_strategy_plan(
    plan: StrategyPlan,
    game_state: GameStateSummary | None = None,
    max_actions: int = 10,
    min_coordinate: float = 0,
    max_coordinate: float = 256,
) -> StrategyPlan:
    """Validate and simulate a bounded Terran strategy plan before execution."""

    if not plan.actions:
        raise PlanValidationError("strategy plan must contain at least one action")
    if len(plan.actions) > max_actions:
        raise PlanValidationError(f"strategy plan has too many actions: {len(plan.actions)} > {max_actions}")

    state = _PlanState.from_summary(game_state)
    for index, action in enumerate(plan.actions, start=1):
        if isinstance(action, MoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "move", min_coordinate, max_coordinate)
        elif isinstance(action, AttackMoveCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "attack", min_coordinate, max_coordinate)
        elif isinstance(action, AttackEnemyCommand):
            _validate_unit(action.unit, index, "attack enemy")
        elif isinstance(action, PatrolCommand):
            _validate_point_action(action.unit, action.x, action.y, index, "patrol", min_coordinate, max_coordinate)
        elif isinstance(action, HoldPositionCommand):
            _validate_unit(action.unit, index, "hold")
        elif isinstance(action, StopCommand):
            _validate_unit(action.unit, index, "stop")
        elif isinstance(action, RallyCommand):
            _validate_rally(action, index, state, min_coordinate, max_coordinate)
        elif isinstance(action, WaitCommand):
            _validate_wait(action, index)
        elif isinstance(action, WaitUntilCommand):
            _validate_wait_until(action, index, state)
        elif isinstance(action, GatherMineralsCommand):
            _validate_gather(action.unit, action.workers, index, state, "minerals")
        elif isinstance(action, GatherGasCommand):
            _validate_gather(action.unit, action.workers, index, state, "gas")
        elif isinstance(action, DistributeWorkersCommand):
            _validate_distribute_workers(action, index, state)
        elif isinstance(action, TrainUnitCommand):
            _validate_train(action, index, state)
        elif isinstance(action, BuildStructureCommand):
            _validate_build(action, index, state)
        elif isinstance(action, ExpandCommand):
            _validate_expand(action, index, state)
        elif isinstance(action, BuildAddonCommand):
            _validate_addon(action, index, state)
        elif isinstance(action, MorphStructureCommand):
            _validate_morph(action, index, state)
        elif isinstance(action, ResearchUpgradeCommand):
            _validate_research(action, index, state)
        elif isinstance(action, RepairCommand):
            _validate_repair(action, index, state)
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
    _validate_point(x, y, index, action_name, min_coordinate, max_coordinate)


def _validate_point(
    x: float,
    y: float,
    index: int,
    action_name: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        raise PlanValidationError(f"action {index}: {action_name} coordinates must be finite")
    if not (min_coordinate <= x <= max_coordinate and min_coordinate <= y <= max_coordinate):
        raise PlanValidationError(
            f"action {index}: {action_name} coordinates ({x:g}, {y:g}) are outside "
            f"the safe range {min_coordinate:g}..{max_coordinate:g}"
        )


def _validate_unit(unit: str, index: int, action_name: str) -> None:
    key = "scv" if unit == "worker" else unit
    if key not in UNIT_SPECS:
        raise PlanValidationError(f"action {index}: unsupported {action_name} unit: {unit}")


def _validate_wait(action: WaitCommand, index: int) -> None:
    if not math.isfinite(action.seconds):
        raise PlanValidationError(f"action {index}: wait duration must be finite")
    if action.seconds < 0:
        raise PlanValidationError(f"action {index}: wait duration must not be negative")
    if action.seconds > 30:
        raise PlanValidationError(f"action {index}: wait duration is too long for the MVP: {action.seconds:g}s")


def _validate_wait_until(action: WaitUntilCommand, index: int, state: _PlanState) -> None:
    if action.condition not in _SUPPORTED_WAIT_CONDITIONS:
        raise PlanValidationError(f"action {index}: unsupported wait-until condition: {action.condition}")
    if not math.isfinite(action.at_least):
        raise PlanValidationError(f"action {index}: wait-until threshold must be finite")
    if action.at_least < 0:
        raise PlanValidationError(f"action {index}: wait-until threshold must not be negative")
    if action.at_least > 10000:
        raise PlanValidationError(f"action {index}: wait-until threshold is too high for the MVP")

    needs_target = {"structure_count", "structure_ready", "structure_pending", "unit_count", "upgrade_complete"}
    if action.condition in needs_target and not action.target:
        raise PlanValidationError(f"action {index}: wait-until {action.condition} requires a target")
    if not state.known:
        return

    threshold = int(math.ceil(action.at_least))
    if action.condition == "minerals":
        state.minerals = max(state.minerals, threshold)
        return
    if action.condition == "vespene":
        state.vespene = max(state.vespene, threshold)
        return
    if action.condition == "supply_left":
        state.supply_left = max(state.supply_left, threshold)
        state.supply_cap = max(state.supply_cap, state.supply_used + state.supply_left)
        return
    if action.condition == "supply_used":
        state.supply_used = max(state.supply_used, threshold)
        state.supply_left = max(0, state.supply_cap - state.supply_used)
        return
    if action.condition == "supply_cap":
        state.supply_cap = max(state.supply_cap, threshold)
        state.supply_left = max(0, state.supply_cap - state.supply_used)
        return
    if action.condition == "game_time":
        state.game_time = max(state.game_time, action.at_least)
        return
    if action.condition == "townhall_count":
        if state.townhalls < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold} townhall(s); only {state.townhalls} are known or planned"
            )
        return
    if action.condition == "unit_count":
        target = action.target or ""
        current = state.workers if target == "worker" else state.units.get(target, 0)
        if current < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} {target} unit(s); "
                f"only {current:g} are currently known or planned"
            )
        return
    if action.condition == "upgrade_complete":
        target = action.target or ""
        if target in state.upgrades:
            return
        if target not in state.upgrades_planned:
            raise PlanValidationError(
                f"action {index}: cannot wait for upgrade {target}; it is not complete or planned"
            )
        state.upgrades.add(target)
        return

    target = action.target or ""
    if action.condition == "structure_count":
        current = state.structures.get(target, 0)
        if current < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} {target} structure(s); "
                f"only {current:g} are currently known or planned"
            )
        return
    if action.condition == "structure_pending":
        current_pending = state.structures_pending.get(target, 0)
        current_total = state.structures.get(target, 0)
        if max(current_pending, current_total) < threshold:
            raise PlanValidationError(
                f"action {index}: cannot wait for {threshold:g} pending {target}; "
                f"only {current_total:g} are currently known or planned"
            )
        state.structures_pending[target] = max(current_pending, threshold)
        return
    state.mark_structure_ready(target, threshold, index)


def _validate_gather(unit: str, worker_count: int | None, index: int, state: _PlanState, resource: str) -> None:
    if unit != "worker":
        raise PlanValidationError(f"action {index}: only workers can gather {resource}")
    if worker_count is not None and not 1 <= worker_count <= 20:
        raise PlanValidationError(f"action {index}: gather worker count must be between 1 and 20")
    if state.known and state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot gather {resource} with no workers")
    if state.known and worker_count is not None and worker_count > state.workers:
        raise PlanValidationError(
            f"action {index}: cannot assign {worker_count} workers; only {state.workers} exist"
        )
    if resource == "gas" and state.known and state.structures_ready.get("refinery", 0) < 1:
        raise PlanValidationError(f"action {index}: cannot gather gas without a ready refinery")


def _validate_distribute_workers(action: DistributeWorkersCommand, index: int, state: _PlanState) -> None:
    ratio = action.mineral_to_gas_ratio
    if not math.isfinite(ratio) or not 0 <= ratio <= 20:
        raise PlanValidationError(f"action {index}: mineral-to-gas ratio must be between 0 and 20")
    if state.known and state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot distribute workers with no workers")


def _validate_train(action: TrainUnitCommand, index: int, state: _PlanState) -> None:
    if action.unit not in UNIT_SPECS:
        raise PlanValidationError(f"action {index}: unsupported train unit: {action.unit}")
    _validate_count(action.count, index, "train")
    spec = UNIT_SPECS[action.unit]
    action_name = f"train {action.count} {action.unit}"

    if state.known:
        if action.unit == "scv" and state.townhalls < 1:
            raise PlanValidationError(f"action {index}: cannot train SCV without a townhall")
        if spec.producer and spec.producer != "command_center":
            state.require_ready(spec.producer, index, action_name)
        for prerequisite in spec.prerequisites:
            state.require_ready(prerequisite, index, action_name)
        if spec.required_addon:
            state.require_ready(spec.required_addon, index, action_name)
        total_supply = int((spec.supply or 0) * action.count)
        if state.supply_left < total_supply:
            if state.supply_left < 1:
                raise PlanValidationError(f"action {index}: cannot train {action.unit} with no supply left")
            raise PlanValidationError(
                f"action {index}: cannot train {action.count} {action.unit} with only {state.supply_left} supply left"
            )
    else:
        total_supply = 0

    state.spend(spec, action.count, index, action_name)
    if state.known:
        state.supply_used += total_supply
        state.supply_left -= total_supply
        unit_key = "worker" if action.unit == "scv" else action.unit
        state.units[unit_key] = state.units.get(unit_key, 0) + action.count
        if action.unit == "scv":
            state.workers += action.count


def _validate_build(action: BuildStructureCommand, index: int, state: _PlanState) -> None:
    if action.worker != "worker":
        raise PlanValidationError(f"action {index}: only workers can build structures")
    if action.building not in STRUCTURE_SPECS:
        raise PlanValidationError(f"action {index}: unsupported build structure: {action.building}")
    _validate_count(action.count, index, "build")
    if state.known and state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot build {action.building} with no workers")

    spec = STRUCTURE_SPECS[action.building]
    action_name = f"build {action.building}"
    for prerequisite in spec.prerequisites:
        state.require_ready(prerequisite, index, action_name)
    state.spend(spec, action.count, index, action_name)
    state.start_structure(action.building, action.count)
    if action.building == "command_center":
        state.townhalls += action.count


def _validate_expand(action: ExpandCommand, index: int, state: _PlanState) -> None:
    _validate_count(action.count, index, "expand")
    if state.known and state.workers < 1:
        raise PlanValidationError(f"action {index}: cannot expand with no workers")
    spec = STRUCTURE_SPECS["command_center"]
    state.spend(spec, action.count, index, f"expand {action.count} time(s)")
    state.start_structure("command_center", action.count)
    state.townhalls += action.count


def _validate_addon(action: BuildAddonCommand, index: int, state: _PlanState) -> None:
    if action.addon not in ADDON_SPECS:
        raise PlanValidationError(f"action {index}: unsupported build add-on: {action.addon}")
    _validate_count(action.count, index, "add-on")
    spec = ADDON_SPECS[action.addon]
    producer = spec.producer or ""
    state.require_ready(producer, index, f"build {action.addon}")
    if state.known:
        existing_addons = sum(
            state.structures.get(key, 0) for key, addon_spec in ADDON_SPECS.items() if addon_spec.producer == producer
        )
        if state.structures_ready.get(producer, 0) - existing_addons < action.count:
            raise PlanValidationError(
                f"action {index}: not enough free {producer} structures for {action.count} add-on(s)"
            )
    state.spend(spec, action.count, index, f"build {action.addon}")
    state.start_structure(action.addon, action.count)


def _validate_morph(action: MorphStructureCommand, index: int, state: _PlanState) -> None:
    if action.building not in MORPH_SPECS:
        raise PlanValidationError(f"action {index}: unsupported structure morph: {action.building}")
    spec = MORPH_SPECS[action.building]
    state.require_ready("command_center", index, f"morph {action.building}")
    for prerequisite in spec.prerequisites:
        state.require_ready(prerequisite, index, f"morph {action.building}")
    state.spend(spec, 1, index, f"morph {action.building}")
    if state.known:
        state.structures["command_center"] = max(0, state.structures.get("command_center", 0) - 1)
        state.structures_ready["command_center"] = max(
            0,
            state.structures_ready.get("command_center", 0) - 1,
        )
    state.start_structure(action.building)


def _validate_research(action: ResearchUpgradeCommand, index: int, state: _PlanState) -> None:
    if action.upgrade not in UPGRADE_SPECS:
        raise PlanValidationError(f"action {index}: unsupported Terran upgrade: {action.upgrade}")
    spec = UPGRADE_SPECS[action.upgrade]
    if action.upgrade in state.upgrades or action.upgrade in state.upgrades_planned:
        raise PlanValidationError(f"action {index}: upgrade {action.upgrade} is already complete or planned")
    if spec.researcher:
        state.require_ready(spec.researcher, index, f"research {action.upgrade}")
    for prerequisite in spec.prerequisites:
        state.require_ready(prerequisite, index, f"research {action.upgrade}")
    if state.known and spec.previous_upgrade and spec.previous_upgrade not in state.upgrades:
        raise PlanValidationError(
            f"action {index}: cannot research {action.upgrade} before {spec.previous_upgrade} completes"
        )
    state.spend(spec, 1, index, f"research {action.upgrade}")
    state.upgrades_planned.add(action.upgrade)


def _validate_repair(action: RepairCommand, index: int, state: _PlanState) -> None:
    _validate_count(action.workers, index, "repair worker")
    if action.target not in REPAIRABLE_TARGET_KEYS:
        raise PlanValidationError(f"action {index}: unsupported repair target: {action.target}")
    if state.known and state.workers < action.workers:
        raise PlanValidationError(
            f"action {index}: cannot repair with {action.workers} workers; only {state.workers} exist"
        )
    if state.known:
        exists = (
            state.units.get(action.target, 0)
            or state.structures.get(action.target, 0)
            or (state.workers if action.target == "worker" else 0)
        )
        if not exists:
            raise PlanValidationError(f"action {index}: cannot repair missing target {action.target}")


def _validate_rally(
    action: RallyCommand,
    index: int,
    state: _PlanState,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if action.building not in {"command_center", "orbital_command", "planetary_fortress", "barracks", "factory", "starport"}:
        raise PlanValidationError(f"action {index}: unsupported rally structure: {action.building}")
    _validate_point(action.x, action.y, index, "rally", min_coordinate, max_coordinate)
    state.require_ready(action.building, index, f"rally {action.building}")


def _validate_count(count: int, index: int, action_name: str) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise PlanValidationError(f"action {index}: {action_name} count must be an integer")
    if count < 1:
        raise PlanValidationError(f"action {index}: {action_name} count must be at least 1")
    if count > 20:
        raise PlanValidationError(f"action {index}: {action_name} count is too high for the MVP")


def _canonical_counts(values: dict[str, int], categories: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_name, count in values.items():
        try:
            key = resolve_alias(raw_name, categories=categories).key
        except KeyError:
            key = normalize_name(raw_name)
        result[key] = result.get(key, 0) + int(count)
    return result
