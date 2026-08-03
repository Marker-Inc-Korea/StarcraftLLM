from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, cast

from starcraft_llm.command_catalog import (
    ABILITY_SPECS,
    ADDON_SPECS,
    ATTACK_CAPABLE_UNIT_KEYS,
    BIOLOGICAL_UNIT_KEYS,
    BUNKER_LOADABLE_UNIT_KEYS,
    FLYING_STRUCTURE_ACTOR_KEYS,
    LOCATION_SPECS,
    MAX_PLAN_ACTIONS,
    MAX_POLICY_SECONDS,
    MAX_REPLAN_CYCLES,
    MAX_SELECTION_COUNT,
    MAX_STRUCTURE_ACTION_COUNT,
    MAX_WORKER_ASSIGNMENT_COUNT,
    MECHANICAL_UNIT_KEYS,
    MEDIVAC_LOADABLE_UNIT_KEYS,
    MOBILE_ATTACK_CAPABLE_UNIT_KEYS,
    MORPH_SPECS,
    MOVABLE_SPECIAL_UNIT_KEYS,
    PSIONIC_UNIT_KEYS,
    REPAIRABLE_TARGET_KEYS,
    SPECIAL_UNIT_SPECS,
    STRUCTURE_SPECS,
    TARGET_SELECTORS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    EntitySpec,
    canonical_runtime_actor_name,
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
    KiteCommand,
    MorphStructureCommand,
    MoveCommand,
    PatrolCommand,
    ProductionPolicyCommand,
    RallyCommand,
    RepairCommand,
    ReplanCommand,
    ResearchUpgradeCommand,
    ReturnCargoCommand,
    StopCommand,
    StrategyPlan,
    TrainUnitCommand,
    UseAbilityCommand,
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
    "army_supply",
    "enemy_unit_count",
    "enemy_structure_count",
    "idle_structure_count",
    "producer_available",
    "cargo_used",
    "unit_near_location",
    "enemy_near_location",
    "under_attack",
}

DEFAULT_MAX_ACTIONS = MAX_PLAN_ACTIONS
MAX_TRAIN_OR_SELECTION_COUNT = MAX_SELECTION_COUNT
MAX_WORKER_COUNT = MAX_WORKER_ASSIGNMENT_COUNT
MAX_STRUCTURE_COUNT = MAX_STRUCTURE_ACTION_COUNT
MAX_REPLANS = MAX_REPLAN_CYCLES

_LOCATION_KEYS = frozenset(LOCATION_SPECS)
_ABILITY_SPECS = ABILITY_SPECS
_KNOWN_ABILITY_TARGET_KEYS = frozenset(
    {
        "worker",
        *UNIT_SPECS,
        *SPECIAL_UNIT_SPECS,
        *STRUCTURE_SPECS,
        *ADDON_SPECS,
        *MORPH_SPECS,
    }
)


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
    semantic_locations: dict[str, object | None]
    production_targets: dict[str, int]
    production_policies: dict[str, ProductionPolicyCommand]

    @classmethod
    def from_summary(cls, summary: GameStateSummary | None) -> "_PlanState":
        if summary is None:
            return cls(
                False,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                {},
                {},
                {},
                {},
                set(),
                set(),
                {},
                {},
                {},
            )

        structures = _canonical_counts(
            summary.structures, ("structure", "addon", "morph")
        )
        pending = _canonical_counts(
            summary.structures_pending, ("structure", "addon", "morph")
        )
        if summary.structures_ready:
            ready = _canonical_counts(
                summary.structures_ready, ("structure", "addon", "morph")
            )
        elif pending:
            ready = {}
        else:
            ready = dict(structures)

        units = _canonical_counts(summary.army, ("unit", "special_unit"))
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
            dict(summary.semantic_locations),
            {},
            {},
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
                raise PlanValidationError(
                    f"action {index}: cannot build barracks before a supply depot exists"
                )
            raise PlanValidationError(
                f"action {index}: cannot {action_name} without a {structure} that is ready"
            )

    def start_structure(self, structure: str, count: int = 1) -> None:
        self.structures[structure] = self.structures.get(structure, 0) + count
        self.structures_pending[structure] = (
            self.structures_pending.get(structure, 0) + count
        )

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
        self.structures_pending[structure] = max(
            0, self.structures_pending.get(structure, 0) - newly_ready
        )
        if structure == "supply_depot" and newly_ready:
            self.supply_cap += 8 * newly_ready
            self.supply_left += 8 * newly_ready
        elif structure == "command_center" and newly_ready:
            self.supply_cap += 15 * newly_ready
            self.supply_left += 15 * newly_ready


def validate_strategy_plan(
    plan: StrategyPlan,
    game_state: GameStateSummary | None = None,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    min_coordinate: float = 0,
    max_coordinate: float = 256,
) -> StrategyPlan:
    """Validate and simulate a bounded Terran strategy plan before execution."""

    if not plan.actions:
        raise PlanValidationError("strategy plan must contain at least one action")
    if len(plan.actions) > max_actions:
        raise PlanValidationError(
            f"strategy plan has too many actions: {len(plan.actions)} > {max_actions}"
        )
    replan_actions = sum(1 for action in plan.actions if _is_replan_action(action))
    if replan_actions > MAX_REPLANS:
        raise PlanValidationError(
            f"strategy plan has too many replan actions: {replan_actions} > {MAX_REPLANS}"
        )

    state = _PlanState.from_summary(game_state)
    for index, action in enumerate(plan.actions, start=1):
        if isinstance(action, MoveCommand):
            _validate_move(action, index, state, min_coordinate, max_coordinate)
        elif isinstance(action, AttackMoveCommand):
            _validate_point_action(
                action,
                action.unit,
                index,
                state,
                "attack",
                min_coordinate,
                max_coordinate,
            )
        elif isinstance(action, AttackEnemyCommand):
            _validate_unit(action.unit, index, "attack enemy")
            _validate_selection(action.selection, index)
            _validate_attack_target(action, index)
            _validate_queued(action, index)
            _validate_focus_fire(action, index)
        elif isinstance(action, KiteCommand):
            _validate_kite(action, index, state)
        elif isinstance(action, PatrolCommand):
            _validate_point_action(
                action,
                action.unit,
                index,
                state,
                "patrol",
                min_coordinate,
                max_coordinate,
            )
        elif isinstance(action, HoldPositionCommand):
            _validate_unit(action.unit, index, "hold")
            _validate_selection(action.selection, index)
            _validate_queued(action, index)
        elif isinstance(action, StopCommand):
            _validate_unit(action.unit, index, "stop")
            _validate_selection(action.selection, index)
            _validate_queued(action, index)
        elif isinstance(action, RallyCommand):
            _validate_rally(action, index, state, min_coordinate, max_coordinate)
        elif isinstance(action, WaitCommand):
            _validate_wait(action, index)
        elif isinstance(action, WaitUntilCommand):
            _validate_wait_until(action, index, state)
        elif isinstance(action, GatherMineralsCommand):
            _validate_gather(
                action, index, state, "minerals", min_coordinate, max_coordinate
            )
        elif isinstance(action, GatherGasCommand):
            _validate_gather(
                action, index, state, "gas", min_coordinate, max_coordinate
            )
        elif isinstance(action, ReturnCargoCommand):
            _validate_return_cargo(action, index, state)
        elif isinstance(action, DistributeWorkersCommand):
            _validate_distribute_workers(action, index, state)
        elif isinstance(action, TrainUnitCommand):
            _validate_train(action, index, state)
        elif isinstance(action, ProductionPolicyCommand):
            _validate_production_policy(action, index, state)
        elif isinstance(action, BuildStructureCommand):
            _validate_build(action, index, state, min_coordinate, max_coordinate)
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
        elif _is_replan_action(action):
            _validate_replan(action, index)
        elif _is_ability_action(action):
            _validate_ability_action(
                action, index, state, min_coordinate, max_coordinate
            )
        else:
            raise PlanValidationError(
                f"action {index}: unsupported action object: {action!r}"
            )

    return plan


def _validate_point_action(
    action: object,
    unit: str,
    index: int,
    state: _PlanState,
    action_name: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    _validate_unit(unit, index, action_name)
    if action_name == "attack":
        key = "scv" if unit == "worker" else unit
        if key not in MOBILE_ATTACK_CAPABLE_UNIT_KEYS:
            raise PlanValidationError(
                f"action {index}: immobile actor {unit} cannot attack-move"
            )
    _validate_selection(getattr(action, "selection", None), index)
    _validate_queued(action, index)
    _validate_location_or_point(
        action,
        index,
        state,
        action_name,
        min_coordinate,
        max_coordinate,
    )


def _validate_move(
    action: MoveCommand,
    index: int,
    state: _PlanState,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    _validate_unit(action.unit, index, "move")
    if action.wait_for_arrival:
        _validate_bounded_seconds(
            action.timeout_seconds, index, "move-and-wait timeout"
        )
        if not math.isfinite(action.arrival_tolerance) or not (
            0.25 <= action.arrival_tolerance <= 20
        ):
            raise PlanValidationError(
                f"action {index}: move-and-wait arrival tolerance must be between 0.25 and 20"
            )
    _validate_selection(action.selection, index)
    _validate_queued(action, index)
    target_unit = getattr(action, "target_unit", None)
    target_tag = getattr(action, "target_tag", None)
    has_unit_target = target_unit is not None or target_tag is not None
    has_point_target = _has_point_target(action)
    if has_unit_target and has_point_target:
        raise PlanValidationError(
            f"action {index}: move must target a point or a friendly unit, not both"
        )
    if not has_unit_target:
        _validate_location_or_point(
            action,
            index,
            state,
            "move",
            min_coordinate,
            max_coordinate,
        )
        return
    if target_tag is not None:
        _validate_tag(target_tag, index, "move target")
    if target_unit is None:
        return
    target_key = normalize_name(str(target_unit))
    friendly_selectors = {
        "nearest_friendly",
        "damaged_friendly",
        "lowest_health_friendly",
        "highest_energy_friendly",
        "any_friendly",
    }
    if target_key in TARGET_SELECTORS:
        if target_key not in friendly_selectors:
            raise PlanValidationError(
                f"action {index}: move target selector must select a friendly unit: {target_key}"
            )
        return
    try:
        resolve_alias(
            target_key,
            categories=("unit", "special_unit", "structure", "morph"),
        )
    except KeyError as exc:
        raise PlanValidationError(
            f"action {index}: unsupported friendly move target: {target_key}"
        ) from exc


def _validate_attack_target(action: AttackEnemyCommand, index: int) -> None:
    target_tag = getattr(action, "target_tag", None)
    target_unit = getattr(action, "target_unit", None)
    if target_tag is not None:
        _validate_tag(target_tag, index, "attack target")
    if target_unit is None:
        return
    target_key = normalize_name(str(target_unit))
    if target_key in TARGET_SELECTORS:
        if not target_key.startswith("nearest_enemy") and not target_key.endswith(
            "_enemy"
        ):
            raise PlanValidationError(
                f"action {index}: attack target selector must select an enemy: {target_key}"
            )
        return
    if not re.fullmatch(r"[a-z0-9_]{1,64}", target_key):
        raise PlanValidationError(
            f"action {index}: invalid enemy unit type target: {target_unit}"
        )


def _validate_focus_fire(action: AttackEnemyCommand, index: int) -> None:
    if not action.wait_for_target_death:
        return
    if action.target_unit is None and action.target_tag is None:
        raise PlanValidationError(
            f"action {index}: focus fire requires target_unit or target_tag"
        )
    _validate_bounded_seconds(action.timeout_seconds, index, "focus-fire timeout")


def _validate_kite(action: KiteCommand, index: int, state: _PlanState) -> None:
    del state
    _validate_unit(action.unit, index, "kite")
    catalog_key = "scv" if action.unit == "worker" else action.unit
    if catalog_key not in MOBILE_ATTACK_CAPABLE_UNIT_KEYS:
        raise PlanValidationError(
            f"action {index}: immobile actor {action.unit} cannot kite"
        )
    _validate_selection(action.selection, index)
    _validate_attack_target(cast(AttackEnemyCommand, action), index)
    if not math.isfinite(action.duration_seconds) or not (
        0.25 <= action.duration_seconds <= 30
    ):
        raise PlanValidationError(
            f"action {index}: kite duration must be between 0.25 and 30 seconds"
        )
    if not math.isfinite(action.retreat_distance) or not (
        0.5 <= action.retreat_distance <= 10
    ):
        raise PlanValidationError(
            f"action {index}: kite retreat distance must be between 0.5 and 10"
        )


def _validate_point(
    x: float,
    y: float,
    index: int,
    action_name: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if not math.isfinite(x) or not math.isfinite(y):
        raise PlanValidationError(
            f"action {index}: {action_name} coordinates must be finite"
        )
    if not (
        min_coordinate <= x <= max_coordinate and min_coordinate <= y <= max_coordinate
    ):
        raise PlanValidationError(
            f"action {index}: {action_name} coordinates ({x:g}, {y:g}) are outside "
            f"the safe range {min_coordinate:g}..{max_coordinate:g}"
        )


def _validate_unit(unit: str, index: int, action_name: str) -> None:
    key = "scv" if unit == "worker" else unit
    if action_name.startswith("attack") or action_name == "kite":
        if key in ATTACK_CAPABLE_UNIT_KEYS:
            return
        raise PlanValidationError(
            f"action {index}: {unit} cannot issue an attack command"
        )
    if action_name == "stop" and key in {
        "planetary_fortress",
        "missile_turret",
        "auto_turret",
    }:
        return
    if key in FLYING_STRUCTURE_ACTOR_KEYS:
        if action_name in {"move", "patrol", "hold", "stop"}:
            return
        raise PlanValidationError(
            f"action {index}: {unit} cannot issue an {action_name} command"
        )
    if key in SPECIAL_UNIT_SPECS and key not in MOVABLE_SPECIAL_UNIT_KEYS:
        raise PlanValidationError(
            f"action {index}: {unit} cannot issue a {action_name} command"
        )
    if key not in UNIT_SPECS and key not in SPECIAL_UNIT_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported {action_name} unit: {unit}"
        )


def _validate_bounded_seconds(value: float, index: int, field_name: str) -> None:
    if not math.isfinite(value) or not 1 <= value <= MAX_POLICY_SECONDS:
        raise PlanValidationError(
            f"action {index}: {field_name} must be between 1 and {MAX_POLICY_SECONDS} seconds"
        )


def _validate_wait(action: WaitCommand, index: int) -> None:
    if not math.isfinite(action.seconds):
        raise PlanValidationError(f"action {index}: wait duration must be finite")
    if action.seconds < 0:
        raise PlanValidationError(f"action {index}: wait duration must not be negative")
    if action.seconds > 30:
        raise PlanValidationError(
            f"action {index}: wait duration is too long for the MVP: {action.seconds:g}s"
        )


def _validate_wait_until(
    action: WaitUntilCommand, index: int, state: _PlanState
) -> None:
    if action.condition not in _SUPPORTED_WAIT_CONDITIONS:
        raise PlanValidationError(
            f"action {index}: unsupported wait-until condition: {action.condition}"
        )
    if not math.isfinite(action.at_least):
        raise PlanValidationError(
            f"action {index}: wait-until threshold must be finite"
        )
    if action.at_least < 0:
        raise PlanValidationError(
            f"action {index}: wait-until threshold must not be negative"
        )
    if action.at_least > 10000:
        raise PlanValidationError(
            f"action {index}: wait-until threshold is too high for the MVP"
        )
    _validate_bounded_seconds(action.timeout_seconds, index, "wait-until timeout")
    if action.on_timeout not in {"replan", "fail"}:
        raise PlanValidationError(
            f"action {index}: wait-until on_timeout must be replan or fail"
        )
    if not math.isfinite(action.radius) or not 0.5 <= action.radius <= 64:
        raise PlanValidationError(
            f"action {index}: wait-until radius must be between 0.5 and 64"
        )
    _validate_selection(action.selection, index)

    needs_target = {
        "structure_count",
        "structure_ready",
        "structure_pending",
        "unit_count",
        "upgrade_complete",
        "idle_structure_count",
        "producer_available",
        "cargo_used",
        "unit_near_location",
    }
    if action.condition in needs_target and not action.target:
        raise PlanValidationError(
            f"action {index}: wait-until {action.condition} requires a target"
        )
    if action.condition in {"unit_near_location", "enemy_near_location"}:
        if action.location is None:
            raise PlanValidationError(
                f"action {index}: wait-until {action.condition} requires a location"
            )
        _validate_location_or_point(action, index, state, action.condition, 0, 256)
    elif action.location is not None:
        if action.condition != "under_attack":
            raise PlanValidationError(
                f"action {index}: wait-until {action.condition} does not use a location"
            )
        _validate_location_or_point(action, index, state, action.condition, 0, 256)
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
    if action.condition in {
        "army_supply",
        "enemy_unit_count",
        "enemy_structure_count",
        "idle_structure_count",
        "producer_available",
        "cargo_used",
        "unit_near_location",
        "enemy_near_location",
        "under_attack",
    }:
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
            planned = state.production_targets.get(target, 0)
            if planned < threshold:
                raise PlanValidationError(
                    f"action {index}: cannot wait for {threshold:g} {target} unit(s); "
                    f"only {current:g} are currently known or planned"
                )
            policy = state.production_policies.get(target)
            if policy is None:
                raise PlanValidationError(
                    f"action {index}: production target {target} has no bounded policy"
                )
            _materialize_production_target(
                state,
                target,
                threshold,
                policy,
                index,
                f"wait for {threshold} {target}",
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


def _validate_gather(
    action: GatherMineralsCommand | GatherGasCommand,
    index: int,
    state: _PlanState,
    resource: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    unit = action.unit
    worker_count = action.workers
    if unit != "worker":
        raise PlanValidationError(f"action {index}: only workers can gather {resource}")
    _validate_selection(getattr(action, "selection", None), index)
    _validate_queued(action, index)
    target_tag = getattr(action, "target_tag", None)
    if target_tag is not None:
        _validate_tag(target_tag, index, f"{resource} target")
    if getattr(action, "location", None) is not None:
        if target_tag is not None:
            raise PlanValidationError(
                f"action {index}: gather {resource} must target a tag or a location, not both"
            )
        _validate_location_or_point(
            action,
            index,
            state,
            f"gather {resource}",
            min_coordinate,
            max_coordinate,
        )
    if worker_count is not None and not 1 <= worker_count <= MAX_WORKER_COUNT:
        raise PlanValidationError(
            f"action {index}: gather worker count must be between 1 and {MAX_WORKER_COUNT}"
        )
    if state.known and state.workers < 1:
        raise PlanValidationError(
            f"action {index}: cannot gather {resource} with no workers"
        )
    if state.known and worker_count is not None and worker_count > state.workers:
        raise PlanValidationError(
            f"action {index}: cannot assign {worker_count} workers; only {state.workers} exist"
        )
    if (
        resource == "gas"
        and state.known
        and state.structures_ready.get("refinery", 0) < 1
    ):
        raise PlanValidationError(
            f"action {index}: cannot gather gas without a ready refinery"
        )


def _validate_return_cargo(
    action: ReturnCargoCommand, index: int, state: _PlanState
) -> None:
    if action.unit not in {"worker", "mule"}:
        raise PlanValidationError(
            f"action {index}: only workers or MULEs can return cargo"
        )
    _validate_selection(action.selection, index)
    _validate_queued(action, index)
    if state.known:
        exists = (
            state.workers if action.unit == "worker" else state.units.get("mule", 0)
        )
        if exists < 1:
            raise PlanValidationError(
                f"action {index}: cannot return cargo without a {action.unit}"
            )


def _validate_distribute_workers(
    action: DistributeWorkersCommand, index: int, state: _PlanState
) -> None:
    ratio = action.mineral_to_gas_ratio
    if not math.isfinite(ratio) or not 0 <= ratio <= 20:
        raise PlanValidationError(
            f"action {index}: mineral-to-gas ratio must be between 0 and 20"
        )
    if state.known and state.workers < 1:
        raise PlanValidationError(
            f"action {index}: cannot distribute workers with no workers"
        )


def _validate_train(action: TrainUnitCommand, index: int, state: _PlanState) -> None:
    if action.unit not in UNIT_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported train unit: {action.unit}"
        )
    _validate_count(
        action.count, index, "train", max_count=MAX_TRAIN_OR_SELECTION_COUNT
    )
    _validate_selection(getattr(action, "producer_selection", None), index)
    spec = UNIT_SPECS[action.unit]
    action_name = f"train {action.count} {action.unit}"

    if state.known:
        if action.unit == "scv" and state.townhalls < 1:
            raise PlanValidationError(
                f"action {index}: cannot train SCV without a townhall"
            )
        if spec.producer and spec.producer != "command_center":
            state.require_ready(spec.producer, index, action_name)
        for prerequisite in spec.prerequisites:
            state.require_ready(prerequisite, index, action_name)
        if spec.required_addon:
            state.require_ready(spec.required_addon, index, action_name)
        total_supply = int((spec.supply or 0) * action.count)
        if state.supply_left < total_supply:
            if state.supply_left < 1:
                raise PlanValidationError(
                    f"action {index}: cannot train {action.unit} with no supply left"
                )
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


def _validate_production_policy(
    action: ProductionPolicyCommand, index: int, state: _PlanState
) -> None:
    if action.unit not in UNIT_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported production-policy unit: {action.unit}"
        )
    _validate_count(
        action.target_count,
        index,
        "production target",
        max_count=MAX_TRAIN_OR_SELECTION_COUNT,
    )
    _validate_selection(action.producer_selection, index)
    _validate_bounded_seconds(action.max_seconds, index, "production-policy timeout")
    for field_name, value, maximum in (
        ("reserve_minerals", action.reserve_minerals, 10000),
        ("reserve_vespene", action.reserve_vespene, 10000),
        ("reserve_supply", action.reserve_supply, 200),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= maximum
        ):
            raise PlanValidationError(
                f"action {index}: {field_name} must be an integer between 0 and {maximum}"
            )

    spec = UNIT_SPECS[action.unit]
    action_name = f"produce {action.unit} until {action.target_count}"
    if state.known:
        if action.unit == "scv" and state.townhalls < 1:
            raise PlanValidationError(
                f"action {index}: cannot produce SCVs without a townhall"
            )
        if spec.producer and spec.producer != "command_center":
            state.require_ready(spec.producer, index, action_name)
        for prerequisite in spec.prerequisites:
            state.require_ready(prerequisite, index, action_name)
        if spec.required_addon:
            state.require_ready(spec.required_addon, index, action_name)

    target_key = "worker" if action.unit == "scv" else action.unit
    previous_target = state.production_targets.get(target_key, 0)
    if action.target_count >= previous_target:
        state.production_targets[target_key] = action.target_count
        state.production_policies[target_key] = action
    if state.known and not action.background:
        _materialize_production_target(
            state,
            target_key,
            action.target_count,
            action,
            index,
            action_name,
        )


def _materialize_production_target(
    state: _PlanState,
    target_key: str,
    target_count: int,
    policy: ProductionPolicyCommand,
    index: int,
    action_name: str,
) -> None:
    """Conservatively reflect a blocking production barrier in plan state.

    Minerals and gas may accrue while the bounded policy waits, so an initially
    low balance is not terminal. Once the barrier completes, however, its
    configured reserves are the only guaranteed balance. Supply cannot accrue
    without an explicit prior supply action and is therefore checked strictly.
    """

    current = (
        state.workers if target_key == "worker" else state.units.get(target_key, 0)
    )
    missing = max(0, target_count - current)
    if missing == 0:
        return

    unit_key = "scv" if target_key == "worker" else target_key
    spec = UNIT_SPECS[unit_key]
    supply_cost = int((spec.supply or 0) * missing)
    required_supply = supply_cost + policy.reserve_supply
    if state.supply_left < required_supply:
        raise PlanValidationError(
            f"action {index}: cannot {action_name} with only {state.supply_left} "
            f"supply left; requires {supply_cost} plus {policy.reserve_supply} reserve"
        )

    state.minerals = max(
        policy.reserve_minerals,
        state.minerals - spec.minerals * missing,
    )
    state.vespene = max(
        policy.reserve_vespene,
        state.vespene - spec.vespene * missing,
    )
    state.supply_used += supply_cost
    state.supply_left -= supply_cost
    if target_key == "worker":
        state.workers = target_count
    state.units[target_key] = target_count


def _validate_build(
    action: BuildStructureCommand,
    index: int,
    state: _PlanState,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if action.worker != "worker":
        raise PlanValidationError(f"action {index}: only workers can build structures")
    if action.building not in STRUCTURE_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported build structure: {action.building}"
        )
    _validate_count(action.count, index, "build")
    _validate_selection(action.selection, index)
    placement_mode = normalize_name(str(getattr(action, "placement_mode", "near")))
    if placement_mode not in {"near", "exact"}:
        raise PlanValidationError(
            f"action {index}: unsupported build placement mode: {placement_mode}"
        )
    max_distance = getattr(action, "max_distance", 20)
    if isinstance(max_distance, bool) or not isinstance(max_distance, int):
        raise PlanValidationError(
            f"action {index}: build max_distance must be an integer"
        )
    if not 0 <= max_distance <= 20:
        raise PlanValidationError(
            f"action {index}: build max_distance must be between 0 and 20"
        )
    if placement_mode == "exact" and action.location is None:
        raise PlanValidationError(
            f"action {index}: exact build placement requires a semantic location or x/y coordinates"
        )
    if placement_mode == "exact" and action.count != 1:
        raise PlanValidationError(
            f"action {index}: exact build placement supports one structure per action"
        )
    reserve_addon_space = getattr(action, "reserve_addon_space", False)
    if not isinstance(reserve_addon_space, bool):
        raise PlanValidationError(
            f"action {index}: reserve_addon_space must be boolean"
        )
    if reserve_addon_space and action.building not in {
        "barracks",
        "factory",
        "starport",
    }:
        raise PlanValidationError(
            f"action {index}: only barracks, factory, or starport can reserve add-on space"
        )
    if action.location is not None:
        _validate_location_or_point(
            action,
            index,
            state,
            f"build {action.building}",
            min_coordinate,
            max_coordinate,
        )
    if state.known and state.workers < 1:
        raise PlanValidationError(
            f"action {index}: cannot build {action.building} with no workers"
        )

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
        raise PlanValidationError(
            f"action {index}: unsupported build add-on: {action.addon}"
        )
    _validate_count(action.count, index, "add-on")
    _validate_selection(getattr(action, "selection", None), index)
    spec = ADDON_SPECS[action.addon]
    producer = spec.producer or ""
    state.require_ready(producer, index, f"build {action.addon}")
    if state.known:
        existing_addons = sum(
            state.structures.get(key, 0)
            for key, addon_spec in ADDON_SPECS.items()
            if addon_spec.producer == producer
        )
        if state.structures_ready.get(producer, 0) - existing_addons < action.count:
            raise PlanValidationError(
                f"action {index}: not enough free {producer} structures for {action.count} add-on(s)"
            )
    state.spend(spec, action.count, index, f"build {action.addon}")
    state.start_structure(action.addon, action.count)


def _validate_morph(
    action: MorphStructureCommand, index: int, state: _PlanState
) -> None:
    if action.building not in MORPH_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported structure morph: {action.building}"
        )
    _validate_selection(getattr(action, "selection", None), index)
    spec = MORPH_SPECS[action.building]
    state.require_ready("command_center", index, f"morph {action.building}")
    for prerequisite in spec.prerequisites:
        state.require_ready(prerequisite, index, f"morph {action.building}")
    state.spend(spec, 1, index, f"morph {action.building}")
    if state.known:
        state.structures["command_center"] = max(
            0, state.structures.get("command_center", 0) - 1
        )
        state.structures_ready["command_center"] = max(
            0,
            state.structures_ready.get("command_center", 0) - 1,
        )
    state.start_structure(action.building)


def _validate_research(
    action: ResearchUpgradeCommand, index: int, state: _PlanState
) -> None:
    if action.upgrade not in UPGRADE_SPECS:
        raise PlanValidationError(
            f"action {index}: unsupported Terran upgrade: {action.upgrade}"
        )
    _validate_selection(getattr(action, "researcher_selection", None), index)
    spec = UPGRADE_SPECS[action.upgrade]
    if action.upgrade in state.upgrades or action.upgrade in state.upgrades_planned:
        raise PlanValidationError(
            f"action {index}: upgrade {action.upgrade} is already complete or planned"
        )
    if spec.researcher:
        state.require_ready(spec.researcher, index, f"research {action.upgrade}")
    for prerequisite in spec.prerequisites:
        state.require_ready(prerequisite, index, f"research {action.upgrade}")
    if (
        state.known
        and spec.previous_upgrade
        and spec.previous_upgrade not in state.upgrades
    ):
        raise PlanValidationError(
            f"action {index}: cannot research {action.upgrade} before {spec.previous_upgrade} completes"
        )
    state.spend(spec, 1, index, f"research {action.upgrade}")
    state.upgrades_planned.add(action.upgrade)


def _validate_repair(action: RepairCommand, index: int, state: _PlanState) -> None:
    _validate_count(action.workers, index, "repair worker", max_count=MAX_WORKER_COUNT)
    _validate_selection(getattr(action, "selection", None), index)
    _validate_selection(getattr(action, "target_selection", None), index)
    target_tag = getattr(action, "target_tag", None)
    if target_tag is not None:
        _validate_tag(target_tag, index, "repair target")
    target_selector = getattr(action, "target_selector", None)
    if target_selector is not None:
        selector_key = normalize_name(str(target_selector))
        if selector_key not in {
            "nearest_friendly",
            "damaged_friendly",
            "lowest_health_friendly",
            "any_friendly",
        }:
            raise PlanValidationError(
                f"action {index}: unsupported repair target selector: {selector_key}"
            )
    if action.target is None and target_tag is None and target_selector is None:
        raise PlanValidationError(
            f"action {index}: repair requires a target type, target tag, or target selector"
        )
    if action.target is not None and action.target not in REPAIRABLE_TARGET_KEYS:
        raise PlanValidationError(
            f"action {index}: unsupported repair target: {action.target}"
        )
    if state.known and state.workers < action.workers:
        raise PlanValidationError(
            f"action {index}: cannot repair with {action.workers} workers; only {state.workers} exist"
        )
    if state.known and action.target is not None and target_tag is None:
        exists = (
            state.units.get(action.target, 0)
            or state.structures.get(action.target, 0)
            or (state.workers if action.target == "worker" else 0)
        )
        if not exists:
            raise PlanValidationError(
                f"action {index}: cannot repair missing target {action.target}"
            )


def _validate_rally(
    action: RallyCommand,
    index: int,
    state: _PlanState,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    if action.building not in {
        "command_center",
        "orbital_command",
        "planetary_fortress",
        "barracks",
        "factory",
        "starport",
        "bunker",
    }:
        raise PlanValidationError(
            f"action {index}: unsupported rally structure: {action.building}"
        )
    _validate_selection(action.selection, index)
    _validate_queued(action, index)
    target_unit = getattr(action, "target_unit", None)
    target_tag = getattr(action, "target_tag", None)
    has_unit_target = target_unit is not None or target_tag is not None
    has_point_target = _has_point_target(action)
    if has_unit_target and has_point_target:
        raise PlanValidationError(
            f"action {index}: rally must target a point or a unit, not both"
        )
    if not has_unit_target:
        _validate_location_or_point(
            action,
            index,
            state,
            "rally",
            min_coordinate,
            max_coordinate,
        )
    else:
        if target_tag is not None:
            _validate_tag(target_tag, index, "rally target")
        if target_unit is not None:
            target_key = normalize_name(str(target_unit))
            if target_key not in {
                "nearest_mineral",
                "nearest_friendly",
                "damaged_friendly",
                "lowest_health_friendly",
                "highest_energy_friendly",
                "any_friendly",
            }:
                try:
                    resolve_alias(
                        target_key,
                        categories=("unit", "special_unit", "structure", "morph"),
                    )
                except KeyError as exc:
                    raise PlanValidationError(
                        f"action {index}: unsupported rally unit target: {target_key}"
                    ) from exc
    state.require_ready(action.building, index, f"rally {action.building}")


def _is_replan_action(action: object) -> bool:
    return isinstance(action, ReplanCommand)


def _is_ability_action(action: object) -> bool:
    class_name = action.__class__.__name__
    action_type = getattr(action, "type", None)
    return (
        isinstance(action, UseAbilityCommand)
        or class_name == "UseAbilityCommand"
        or action_type
        in {
            "use_ability",
            "scan",
            "call_down_mule",
            "supply_drop",
            "transform",
            "lift",
            "land",
            "load",
            "unload",
            "cancel",
            "salvage",
            "build_nuke",
            "launch_nuke",
        }
        or _action_ability_name(action) is not None
    )


def _validate_replan(action: object, index: int) -> None:
    max_replans = getattr(action, "max_replans", MAX_REPLANS)
    if isinstance(max_replans, bool) or not isinstance(max_replans, int):
        raise PlanValidationError(
            f"action {index}: replan max_replans must be an integer"
        )
    if not 0 <= max_replans <= MAX_REPLANS:
        raise PlanValidationError(
            f"action {index}: replan max_replans must be between 0 and {MAX_REPLANS}"
        )
    reason = getattr(action, "reason", None)
    if reason is not None and (not isinstance(reason, str) or len(reason) > 500):
        raise PlanValidationError(
            f"action {index}: replan reason must be a short string"
        )


def _validate_ability_action(
    action: object,
    index: int,
    state: _PlanState,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    ability_name = _action_ability_name(action)
    if ability_name is None:
        raise PlanValidationError(f"action {index}: ability command is missing ability")
    ability_key: str = normalize_name(str(ability_name))
    spec = _ABILITY_SPECS.get(ability_key)
    if spec is None:
        raise PlanValidationError(
            f"action {index}: unsupported Terran ability: {ability_key}"
        )

    target_kind = str(_spec_value(spec, "target_kind", "none"))
    raw_actors = cast(
        Iterable[str],
        _spec_value(spec, "actors", _spec_value(spec, "actor_types", ())) or (),
    )
    actors: tuple[str, ...] = tuple(raw_actors)
    actor = _action_actor(action, ability_key, actors)
    if actor is not None:
        actor = _canonical_actor(actor)
        _validate_actor_kind(actor, index, ability_key)
        if actors and "any" not in actors and actor not in actors:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} cannot be used by actor {actor}; "
                f"expected one of {', '.join(actors)}"
            )
    elif actors and "any" not in actors:
        raise PlanValidationError(
            f"action {index}: ability {ability_key} requires an actor"
        )

    _validate_selection(getattr(action, "selection", None), index)
    _validate_queued(action, index)
    implicit_medivac_unload = (
        ability_key == "unload_all_medivac" and not _has_point_target(action)
    )
    if not implicit_medivac_unload:
        _validate_ability_target(
            action,
            target_kind,
            index,
            state,
            ability_key,
            min_coordinate,
            max_coordinate,
        )
    _validate_ability_target_filter(action, spec, index, ability_key)
    _validate_ability_prerequisites(action, spec, actor, index, state, ability_key)
    _apply_ability_state_effect(ability_key, state)


def _apply_ability_state_effect(ability_key: str, state: _PlanState) -> None:
    """Apply deterministic effects needed by later static plan validation.

    Energy, cooldowns, buffs, and temporary-unit lifetime remain live-runtime
    concerns. These two effects are deterministic enough to unlock legitimate
    follow-up actions in the same bounded plan.
    """

    if not state.known:
        return
    if ability_key == "call_down_mule":
        state.units["mule"] = state.units.get("mule", 0) + 1
    elif ability_key == "supply_drop":
        state.supply_cap += 8
        state.supply_left += 8


def _action_ability_name(action: object) -> str | None:
    for field_name in ("ability", "ability_key", "name"):
        value = getattr(action, field_name, None)
        if value:
            return str(value)
    mapping = {
        "ScanCommand": "scan",
        "CallDownMuleCommand": "call_down_mule",
        "SupplyDropCommand": "supply_drop",
        "BuildNukeCommand": "build_nuke",
        "LaunchNukeCommand": "launch_nuke",
    }
    class_name = action.__class__.__name__
    if class_name in mapping:
        return mapping[class_name]
    actor = getattr(action, "actor", None)
    if class_name == "LiftCommand" and actor:
        return f"lift_{normalize_name(str(actor))}"
    if class_name == "LandCommand" and actor:
        return f"land_{normalize_name(str(actor))}"
    if class_name == "LoadCommand" and actor:
        actor_key = normalize_name(str(actor))
        has_specific_target = any(
            getattr(action, field_name, None) is not None
            for field_name in ("target_unit", "target_tag", "target_selection")
        )
        return (
            "load_command_center"
            if actor_key in {"command_center", "orbital_command"}
            and has_specific_target
            else (
                "load_all_command_center"
                if actor_key in {"command_center", "orbital_command"}
                else f"load_{actor_key}"
            )
        )
    if class_name == "UnloadCommand" and actor:
        actor_key = normalize_name(str(actor))
        ability_actor_key = (
            "command_center" if actor_key == "orbital_command" else actor_key
        )
        if getattr(action, "location", None) is not None:
            return (
                "unload_all_medivac"
                if actor_key == "medivac"
                else f"unload_all_{ability_actor_key}"
            )
        if (
            getattr(action, "target_unit", None) is not None
            or getattr(action, "passenger_tag", None) is not None
        ):
            return f"unload_unit_{ability_actor_key}"
        return f"unload_all_{ability_actor_key}"
    if class_name == "SalvageCommand" and actor:
        return f"salvage_{normalize_name(str(actor))}"
    action_type = getattr(action, "type", None)
    if action_type in _ABILITY_SPECS:
        return str(action_type)
    return None


def _action_actor(
    action: object, ability_key: str, expected_actors: tuple[str, ...]
) -> str | None:
    for field_name in ("actor", "unit", "building", "structure"):
        value = getattr(action, field_name, None)
        if value:
            return str(value)
    for prefix in ("lift_", "land_", "load_all_", "unload_all_"):
        if ability_key.startswith(prefix):
            return ability_key.removeprefix(prefix)
    if len(expected_actors) == 1 and expected_actors[0] != "any":
        return expected_actors[0]
    return None


def _canonical_actor(actor: str) -> str:
    key = normalize_name(actor)
    if key == "worker":
        return "worker"
    try:
        return resolve_alias(
            key,
            categories=("unit", "special_unit", "structure", "addon", "morph"),
        ).key
    except KeyError:
        return key


def _validate_actor_kind(actor: str, index: int, ability_key: str) -> None:
    if (
        actor == "worker"
        or actor in UNIT_SPECS
        or actor in SPECIAL_UNIT_SPECS
        or actor in STRUCTURE_SPECS
        or actor in ADDON_SPECS
        or actor in MORPH_SPECS
    ):
        return
    raise PlanValidationError(
        f"action {index}: unsupported actor {actor} for ability {ability_key}"
    )


def _validate_selection(selection: object, index: int) -> None:
    if selection is None:
        return
    count = _selection_value(
        selection, "count", _selection_value(selection, "limit", None)
    )
    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise PlanValidationError(
                f"action {index}: selection count must be an integer"
            )
        if not 1 <= count <= MAX_TRAIN_OR_SELECTION_COUNT:
            raise PlanValidationError(
                f"action {index}: selection count must be between 1 and {MAX_TRAIN_OR_SELECTION_COUNT}"
            )
    mode = _selection_value(selection, "mode", "first")
    if mode not in {
        "all",
        "first",
        "ready",
        "idle",
        "closest",
        "lowest_health",
        "highest_energy",
        "random_seeded",
    }:
        raise PlanValidationError(f"action {index}: unsupported selection mode: {mode}")
    tags = _selection_value(selection, "tags", ())
    if tags is None:
        return
    if isinstance(tags, (str, int)) or not isinstance(tags, (list, tuple, set)):
        raise PlanValidationError(
            f"action {index}: selection tags must be an array of unit tags"
        )
    if len(tags) > MAX_TRAIN_OR_SELECTION_COUNT:
        raise PlanValidationError(
            f"action {index}: selection has too many tags: {len(tags)} > {MAX_TRAIN_OR_SELECTION_COUNT}"
        )
    normalized_tags = [_validate_tag(tag, index, "selection") for tag in tags]
    if len(set(normalized_tags)) != len(normalized_tags):
        raise PlanValidationError(
            f"action {index}: selection tags must not contain duplicates"
        )


def _validate_tag(value: object, index: int, field_name: str) -> str:
    if isinstance(value, bool):
        raise PlanValidationError(
            f"action {index}: {field_name} tag must be a positive integer or digit string"
        )
    if isinstance(value, int):
        if value > 0:
            return str(value)
    elif isinstance(value, str) and value.isdigit() and int(value) > 0:
        return str(int(value))
    raise PlanValidationError(
        f"action {index}: {field_name} tag must be a positive integer or digit string"
    )


def _validate_queued(action: object, index: int) -> None:
    queued = getattr(action, "queued", False)
    if not isinstance(queued, bool):
        raise PlanValidationError(f"action {index}: queued must be boolean")


def _validate_ability_target(
    action: object,
    target_kind: str,
    index: int,
    state: _PlanState,
    ability_key: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    has_point = _has_point_target(action)
    has_unit = _has_unit_target(action)
    if target_kind == "none":
        if has_point or has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} does not take a target"
            )
        return
    if target_kind == "point":
        if has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires a point target, not a unit"
            )
        _validate_location_or_point(
            action, index, state, ability_key, min_coordinate, max_coordinate
        )
        return
    if target_kind == "mineral":
        if has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires a mineral-field anchor, not a unit selector"
            )
        _validate_location_or_point(
            action,
            index,
            state,
            ability_key,
            min_coordinate,
            max_coordinate,
        )
        return
    if target_kind == "unit":
        if has_point:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires a unit target, not a point"
            )
        if not has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires a unit target"
            )
        _validate_unit_target(action, index, ability_key)
        return
    if target_kind == "point_or_unit":
        if has_point and has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} must target a point or a unit, not both"
            )
        if not has_point and not has_unit:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires a target"
            )
        if has_point:
            _validate_location_or_point(
                action, index, state, ability_key, min_coordinate, max_coordinate
            )
        else:
            _validate_unit_target(action, index, ability_key)
        return
    raise PlanValidationError(
        f"action {index}: ability {ability_key} has unsupported target kind: {target_kind}"
    )


def _validate_ability_target_filter(
    action: object, spec: object, index: int, ability_key: str
) -> None:
    target_filter = str(_spec_value(spec, "target_filter", "any"))
    if target_filter == "any":
        return

    target = next(
        (
            getattr(action, field_name)
            for field_name in ("target_unit", "target_actor", "unit_target")
            if getattr(action, field_name, None) is not None
        ),
        None,
    )
    if target is None or isinstance(target, int):
        return
    target_key = normalize_name(str(target))

    if target_key in TARGET_SELECTORS:
        exact_target_filters = {
            "supply_depot",
            "worker",
            "worker_passenger",
            "bunker_loadable",
            "medivac_loadable",
        }
        if target_filter in exact_target_filters:
            raise PlanValidationError(
                f"action {index}: ability {ability_key} requires an exact compatible target"
            )
        return

    allowed_by_filter = {
        "biological_unit": frozenset(BIOLOGICAL_UNIT_KEYS),
        "mechanical_unit": frozenset(MECHANICAL_UNIT_KEYS),
        "mechanical_or_psionic_unit": frozenset(
            (*MECHANICAL_UNIT_KEYS, *PSIONIC_UNIT_KEYS)
        ),
        "mechanical": frozenset(REPAIRABLE_TARGET_KEYS),
        "supply_depot": frozenset({"supply_depot"}),
        "worker": frozenset({"worker"}),
        "worker_passenger": frozenset({"worker"}),
        "bunker_loadable": frozenset(BUNKER_LOADABLE_UNIT_KEYS),
        "medivac_loadable": frozenset(MEDIVAC_LOADABLE_UNIT_KEYS),
    }.get(target_filter)
    if allowed_by_filter is None:
        raise PlanValidationError(
            f"action {index}: ability {ability_key} has unsupported target filter {target_filter}"
        )
    target_alliance = str(_spec_value(spec, "target_alliance", "any"))
    if target_alliance == "enemy" and target_key not in _KNOWN_ABILITY_TARGET_KEYS:
        # Cross-race traits are authoritative only in live observations. The
        # executor still applies the ability filter before issuing the order.
        return
    if target_key not in allowed_by_filter:
        raise PlanValidationError(
            f"action {index}: ability {ability_key} cannot target {target_key}"
        )


def _has_point_target(action: object) -> bool:
    return (
        getattr(action, "target_addon", None) is not None
        or getattr(action, "target_addon_tag", None) is not None
        or getattr(action, "location", None) is not None
        or getattr(action, "target_location", None) is not None
        or (
            getattr(action, "x", None) is not None
            and getattr(action, "y", None) is not None
        )
    )


def _has_unit_target(action: object) -> bool:
    return any(
        getattr(action, field_name, None) is not None
        for field_name in (
            "target_unit",
            "target_tag",
            "target_actor",
            "unit_target",
            "passenger_tag",
        )
    )


def _validate_location_or_point(
    action: object,
    index: int,
    state: _PlanState,
    ability_key: str,
    min_coordinate: float,
    max_coordinate: float,
) -> None:
    target_addon = getattr(action, "target_addon", None)
    target_addon_tag = getattr(action, "target_addon_tag", None)
    if target_addon is not None or target_addon_tag is not None:
        if (
            getattr(action, "location", None) is not None
            or getattr(action, "target_location", None) is not None
            or getattr(action, "x", None) is not None
            or getattr(action, "y", None) is not None
        ):
            raise PlanValidationError(
                f"action {index}: {ability_key} must target an add-on or a point, not both"
            )
        if not ability_key.startswith("land_"):
            raise PlanValidationError(
                f"action {index}: only land commands may target an add-on"
            )
        actor = normalize_name(str(getattr(action, "actor", "")))
        if actor not in {"barracks", "factory", "starport"}:
            raise PlanValidationError(
                f"action {index}: {actor or 'this structure'} cannot land on an add-on"
            )
        if target_addon is not None:
            addon_key = normalize_name(str(target_addon))
            if addon_key not in ADDON_SPECS:
                raise PlanValidationError(
                    f"action {index}: unsupported add-on landing target: {addon_key}"
                )
            if state.known and state.structures.get(addon_key, 0) < 1:
                raise PlanValidationError(
                    f"action {index}: cannot land on missing add-on {addon_key}"
                )
        if target_addon_tag is not None:
            _validate_tag(target_addon_tag, index, "add-on target")
        return
    location = getattr(action, "location", None) or getattr(
        action, "target_location", None
    )
    if location is not None:
        semantic = getattr(location, "semantic", None)
        if semantic is not None:
            key = normalize_name(str(semantic))
            if key not in _LOCATION_KEYS:
                raise PlanValidationError(
                    f"action {index}: unsupported semantic location for {ability_key}: {key}"
                )
            _validate_location_resolvable(key, index, state, ability_key)
            return
        point_x = getattr(location, "x", None)
        point_y = getattr(location, "y", None)
        if point_x is not None or point_y is not None:
            if point_x is None or point_y is None:
                raise PlanValidationError(
                    f"action {index}: ability {ability_key} requires both x and y"
                )
            _validate_point(
                float(point_x),
                float(point_y),
                index,
                ability_key,
                min_coordinate,
                max_coordinate,
            )
            return
        key = normalize_name(str(getattr(location, "key", location)))
        if key not in _LOCATION_KEYS:
            raise PlanValidationError(
                f"action {index}: unsupported semantic location for {ability_key}: {key}"
            )
        _validate_location_resolvable(key, index, state, ability_key)
        return
    x = getattr(action, "x", None)
    y = getattr(action, "y", None)
    if x is None or y is None:
        raise PlanValidationError(
            f"action {index}: ability {ability_key} requires a point target"
        )
    _validate_point(
        float(x), float(y), index, ability_key, min_coordinate, max_coordinate
    )


def _validate_location_resolvable(
    key: str, index: int, state: _PlanState, ability_key: str
) -> None:
    # Missing from the summary means "not statically knowable"; the live executor
    # is still allowed to resolve it. A present None/unresolved entry is known
    # failure evidence and must be rejected before any SC2 call.
    summary_locations = getattr(state, "semantic_locations", None)
    if not summary_locations:
        return
    if key not in summary_locations:
        return
    location = summary_locations[key]
    if location is None or getattr(location, "resolved", True) is False:
        raise PlanValidationError(
            f"action {index}: semantic location {key} for {ability_key} is unresolved"
        )


def _validate_unit_target(action: object, index: int, ability_key: str) -> None:
    for field_name in ("target_tag", "passenger_tag"):
        tag = getattr(action, field_name, None)
        if tag is not None:
            _validate_tag(tag, index, field_name.replace("_", " "))
    target = None
    for field_name in ("target_unit", "target_actor", "unit_target"):
        value = getattr(action, field_name, None)
        if value is not None:
            target = value
            break
    if target is None:
        return
    if isinstance(target, int):
        return
    target_key = normalize_name(str(target))
    if target_key in TARGET_SELECTORS:
        return
    if target_key == "worker":
        return
    try:
        resolve_alias(
            target_key,
            categories=("unit", "special_unit", "structure", "addon", "morph"),
        )
    except KeyError as exc:
        spec = _ABILITY_SPECS.get(ability_key)
        target_alliance = str(_spec_value(spec, "target_alliance", "any"))
        if target_alliance == "enemy" and re.fullmatch(r"[a-z0-9_]{1,64}", target_key):
            return
        raise PlanValidationError(
            f"action {index}: unsupported unit target {target_key} for ability {ability_key}"
        ) from exc


def _validate_ability_prerequisites(
    action: object,
    spec: object,
    actor: str | None,
    index: int,
    state: _PlanState,
    ability_key: str,
) -> None:
    if state.known and actor:
        if actor == "worker":
            exists = state.workers
        elif actor in UNIT_SPECS or actor in SPECIAL_UNIT_SPECS:
            exists = state.units.get(actor, 0)
        else:
            exists = state.structures_ready.get(actor, 0)
        if exists < 1:
            raise PlanValidationError(
                f"action {index}: cannot use {ability_key}; no ready {actor} is known"
            )

    raw_upgrade_requirements = cast(
        Iterable[str], _spec_value(spec, "required_upgrades", ()) or ()
    )
    upgrade_requirements: set[str] = set(raw_upgrade_requirements)
    raw_prerequisites = cast(
        Iterable[str], _spec_value(spec, "prerequisites", ()) or ()
    )
    for prerequisite in tuple(raw_prerequisites):
        if prerequisite in UPGRADE_SPECS:
            upgrade_requirements.add(prerequisite)
        else:
            state.require_ready(prerequisite, index, f"use {ability_key}")

    if state.known:
        for upgrade in upgrade_requirements:
            if upgrade not in state.upgrades:
                upgrade_spec = UPGRADE_SPECS.get(upgrade)
                if upgrade_spec is not None:
                    if (
                        upgrade == "battlecruiser_weapon_refit"
                        and upgrade_spec.researcher
                    ):
                        state.require_ready(
                            upgrade_spec.researcher, index, f"use {ability_key}"
                        )
                    for prerequisite in upgrade_spec.prerequisites:
                        state.require_ready(prerequisite, index, f"use {ability_key}")
                raise PlanValidationError(
                    f"action {index}: cannot use {ability_key} before upgrade {upgrade}"
                )
        minerals = int(cast(Any, _spec_value(spec, "minerals", 0) or 0))
        vespene = int(cast(Any, _spec_value(spec, "vespene", 0) or 0))
        if ability_key == "build_nuke" and minerals == 0 and vespene == 0:
            minerals, vespene = 100, 100
        if minerals or vespene:
            pseudo_spec = EntitySpec(
                key=ability_key,
                enum_name=ability_key.upper(),
                minerals=minerals,
                vespene=vespene,
                supply=None,
                producer=None,
                researcher=None,
                prerequisites=(),
                required_addon=None,
                previous_upgrade=None,
                aliases=(),
                runtime_state_keys=(),
            )
            state.spend(pseudo_spec, 1, index, f"use {ability_key}")


def _spec_value(spec: object, field_name: str, default: object = None) -> object:
    if isinstance(spec, dict):
        return spec.get(field_name, default)
    return getattr(spec, field_name, default)


def _selection_value(
    selection: object, field_name: str, default: object = None
) -> object:
    if isinstance(selection, dict):
        return selection.get(field_name, default)
    return getattr(selection, field_name, default)


def _validate_count(
    count: int, index: int, action_name: str, max_count: int = MAX_STRUCTURE_COUNT
) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise PlanValidationError(
            f"action {index}: {action_name} count must be an integer"
        )
    if count < 1:
        raise PlanValidationError(
            f"action {index}: {action_name} count must be at least 1"
        )
    if count > max_count:
        raise PlanValidationError(
            f"action {index}: {action_name} count is too high: {count} > {max_count}"
        )


def _canonical_counts(
    values: dict[str, int], categories: tuple[str, ...]
) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_name, count in values.items():
        try:
            key = resolve_alias(raw_name, categories=categories).key
        except KeyError:
            key = canonical_runtime_actor_name(raw_name)
        result[key] = result.get(key, 0) + int(count)
    return result
