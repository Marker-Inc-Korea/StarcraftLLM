from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starcraft_llm.command_catalog import (
    ABILITY_SPECS,
    ADDON_SPECS,
    ALERT_KEYS,
    BIOLOGICAL_UNIT_KEYS,
    BUNKER_LOADABLE_UNIT_KEYS,
    FLYING_STRUCTURE_ACTOR_KEYS,
    MAX_PLAN_ACTIONS,
    MAX_REPLAN_CYCLES,
    MAX_SELECTION_COUNT,
    MECHANICAL_UNIT_KEYS,
    MEDIVAC_LOADABLE_UNIT_KEYS,
    MORPH_SPECS,
    PSIONIC_UNIT_KEYS,
    REPAIRABLE_TARGET_KEYS,
    RUNTIME_ACTOR_UNIT_TYPES,
    STRUCTURE_SPECS,
    TARGET_SELECTORS,
    UNIT_FORM_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    AbilitySpec,
    canonical_runtime_actor_name,
    normalize_name,
    resolve_alias,
)
from starcraft_llm.game_state import (
    GameStateSummary,
    LocationSnapshot,
    SupplySummary,
    UnitObservationSnapshot,
    game_state_summary_to_json,
)
from starcraft_llm.planner import (
    DEFAULT_PLANNER,
    PLANNER_MODES,
    PlannerError,
    PlannerUnavailableError,
    plan_strategy,
)
from starcraft_llm.strategy import (
    AttackEnemyCommand,
    AttackMoveCommand,
    AttackUntilClearCommand,
    BuildAddonCommand,
    BuildStructureCommand,
    DistributeWorkersCommand,
    ConditionExpression,
    ConditionGroup,
    ConditionSpec,
    ConditionalCommand,
    ExpandCommand,
    GatherGasCommand,
    GatherMineralsCommand,
    HoldPositionCommand,
    KiteCommand,
    MorphStructureCommand,
    MoveCommand,
    PatrolCommand,
    ProductionPolicyCommand,
    RepeatCommand,
    RallyCommand,
    RepairCommand,
    ResearchUpgradeCommand,
    ReturnCargoCommand,
    StopCommand,
    StopProductionCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
    WaitUntilCommand,
    WithTimeoutCommand,
    strategy_plan_to_json,
)
from starcraft_llm.validator import PlanValidationError, validate_strategy_plan

DEFAULT_MAP = "AbyssalReefLE"
DEFAULT_STRATEGY = "move worker 35 42"
MAX_REPLANS = MAX_REPLAN_CYCLES
ABILITY_RETRY_SECONDS = 30.0
LOAD_ALL_NEARBY_RADIUS = 8.0


_RUNTIME_ABILITY_SPECS = ABILITY_SPECS


@dataclass(frozen=True)
class Sc2Environment:
    installed: bool
    candidate_paths: tuple[Path, ...]
    detected_path: Path | None
    sc2path_env: str | None
    maps_path: Path | None
    maps_installed: bool


def detect_sc2_environment() -> Sc2Environment:
    """Detect common StarCraft II installation paths without importing sc2."""

    sc2path_env = os.environ.get("SC2PATH")
    candidates = tuple(_candidate_sc2_paths())
    detected = next((path for path in candidates if path.exists()), None)
    detected_path = Path(sc2path_env).expanduser() if sc2path_env else detected
    maps_path = _maps_path_for(detected_path) if detected_path else None
    maps_installed = bool(maps_path and maps_path.exists())
    return Sc2Environment(
        installed=bool(detected_path and detected_path.exists()),
        candidate_paths=candidates,
        detected_path=detected_path,
        sc2path_env=sc2path_env,
        maps_path=maps_path,
        maps_installed=maps_installed,
    )


def _candidate_sc2_paths() -> Iterable[Path]:
    system = platform.system()
    home = Path.home()
    if system == "Darwin":
        yield Path("/Applications/StarCraft II")
        yield home / "Applications" / "StarCraft II"
    elif system == "Windows":
        program_files = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        yield Path(program_files) / "StarCraft II"
    else:
        yield home / "StarCraftII"
        yield home / "Games" / "battlenet" / "drive_c" / "Program Files (x86)" / "StarCraft II"


def _maps_path_for(sc2_base_path: Path) -> Path:
    lower_case_maps = sc2_base_path / "maps"
    if lower_case_maps.exists():
        return lower_case_maps
    return sc2_base_path / "Maps"


class MoveUnitBot:  # Runtime base class is injected after sc2 import.
    """Factory placeholder; use create_move_unit_bot_class after importing sc2."""


def create_game_state_bot_class(bot_ai_base):
    class _GameStateBot(bot_ai_base):  # type: ignore[misc, valid-type]
        def __init__(self):
            super().__init__()
            self.summary: GameStateSummary | None = None

        async def on_start(self):
            self.client.game_step = 2

        async def on_step(self, iteration: int):
            self.summary = summarize_bot_state(self)
            await self.client.leave()

    return _GameStateBot


def summarize_bot_state(bot) -> GameStateSummary:
    army: dict[str, int] = {}
    for unit in bot.units:
        if unit in bot.workers or unit in bot.townhalls:
            continue
        name = _unit_type_name(unit)
        army[name] = army.get(name, 0) + 1

    structures: dict[str, int] = {}
    structures_ready: dict[str, int] = {}
    structures_pending: dict[str, int] = {}
    for structure in getattr(bot, "structures", bot.townhalls):
        name = _unit_type_name(structure)
        structures[name] = structures.get(name, 0) + 1
        if _structure_is_ready(structure):
            structures_ready[name] = structures_ready.get(name, 0) + 1
        else:
            structures_pending[name] = structures_pending.get(name, 0) + 1

    unit_observations = tuple(
        _unit_observation(unit, "self") for unit in _iter_observable_units(bot)
    )
    enemy_observations = tuple(
        _unit_observation(unit, "enemy") for unit in _iter_observable_enemy_units(bot)
    )
    resource_observations = tuple(
        _unit_observation(unit, "neutral")
        for unit in _iter_observable_resource_units(bot)
    )
    semantic_locations = _semantic_location_snapshots(bot)
    enemy_race = _enemy_race_name(getattr(bot, "enemy_race", None))
    game_info = getattr(bot, "game_info", None)
    raw_map_name = getattr(game_info, "map_name", None)
    map_name = str(raw_map_name) if raw_map_name else None
    active_alerts = _active_alert_names(bot)

    upgrades = []
    state = getattr(bot, "state", None)
    for upgrade in getattr(state, "upgrades", ()):
        raw_name = getattr(upgrade, "name", str(upgrade))
        try:
            upgrades.append(resolve_alias(raw_name, categories=("upgrade",)).key)
        except KeyError:
            upgrades.append(normalize_name(raw_name))

    return GameStateSummary(
        minerals=int(bot.minerals),
        vespene=int(bot.vespene),
        supply=SupplySummary(
            used=int(bot.supply_used),
            cap=int(bot.supply_cap),
            left=int(bot.supply_left),
        ),
        workers=len(bot.workers),
        townhalls=len(bot.townhalls),
        army=army,
        known_enemy_units=len(bot.enemy_units),
        game_time_seconds=float(getattr(bot, "time", 0.0)),
        structures=structures,
        structures_ready=structures_ready,
        structures_pending=structures_pending,
        upgrades=tuple(sorted(set(upgrades))),
        unit_observations=(
            unit_observations + enemy_observations + resource_observations
        ),
        semantic_locations=semantic_locations,
        enemy_race=enemy_race,
        map_name=map_name,
        active_alerts=active_alerts,
    )


def _iter_observable_units(bot):
    seen_tags = set()
    for collection in (
        getattr(bot, "workers", ()),
        getattr(bot, "units", ()),
        getattr(bot, "structures", ()),
    ):
        for unit in collection:
            tag = getattr(unit, "tag", id(unit))
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            yield unit


def _iter_observable_enemy_units(bot):
    seen_tags = set()
    for collection in (
        getattr(bot, "enemy_units", ()),
        getattr(bot, "enemy_structures", ()),
    ):
        for unit in collection:
            tag = getattr(unit, "tag", id(unit))
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            yield unit


def _iter_observable_resource_units(bot):
    seen_tags = set()
    for collection in (
        getattr(bot, "mineral_field", ()),
        getattr(bot, "vespene_geyser", ()),
        getattr(bot, "destructables", ()),
        getattr(bot, "watchtowers", ()),
    ):
        for unit in collection:
            tag = getattr(unit, "tag", id(unit))
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            yield unit


def _unit_observation(unit, alliance: str) -> UnitObservationSnapshot:
    position = getattr(unit, "position", None)
    orders = tuple(_order_name(order) for order in getattr(unit, "orders", ()) or ())
    passengers = tuple(
        sorted(
            getattr(unit, "passengers", None) or getattr(unit, "cargo", None) or (),
            key=lambda passenger: str(getattr(passenger, "tag", "")),
        )
    )
    return UnitObservationSnapshot(
        unit=_unit_type_name(unit),
        alliance=alliance,
        tag=getattr(unit, "tag", None),
        x=_point_component(position, "x", 0),
        y=_point_component(position, "y", 1),
        health=_optional_float(getattr(unit, "health", None)),
        health_max=_optional_float(getattr(unit, "health_max", None)),
        energy=_optional_float(getattr(unit, "energy", None)),
        is_ready=getattr(unit, "is_ready", None),
        is_flying=getattr(unit, "is_flying", None),
        is_cloaked=getattr(unit, "is_cloaked", None),
        is_burrowed=getattr(unit, "is_burrowed", None),
        is_loaded=getattr(unit, "is_loaded", None),
        is_idle=getattr(unit, "is_idle", None),
        cargo_used=_optional_int(getattr(unit, "cargo_used", None)),
        cargo_max=_optional_int(getattr(unit, "cargo_max", None)),
        add_on_tag=getattr(unit, "add_on_tag", None),
        passenger_tags=tuple(
            tag
            for passenger in passengers
            if (tag := getattr(passenger, "tag", None)) is not None
        ),
        passenger_units=tuple(_unit_type_name(passenger) for passenger in passengers),
        is_biological=getattr(unit, "is_biological", None),
        is_mechanical=getattr(unit, "is_mechanical", None),
        is_psionic=getattr(unit, "is_psionic", None),
        is_massive=getattr(unit, "is_massive", None),
        is_detector=getattr(unit, "is_detector", None),
        weapon_cooldown=_optional_float(getattr(unit, "weapon_cooldown", None)),
        orders=orders,
    )


def _semantic_location_snapshots(bot) -> dict[str, LocationSnapshot | None]:
    locations = _semantic_location_points(bot)
    return {key: _location_snapshot(value) for key, value in locations.items()}


def _semantic_location_points(bot) -> dict[str, Any | None]:
    own_main = getattr(bot, "start_location", None)
    enemy_main = (getattr(bot, "enemy_start_locations", None) or [None])[0]
    map_center = getattr(getattr(bot, "game_info", None), "map_center", None)
    ramp = getattr(bot, "main_base_ramp", None)
    own_ramp = _ramp_point(ramp, "top_center") or _ramp_point(ramp, "bottom_center")
    corner_depots = sorted(
        _ramp_points(ramp, "corner_depots"),
        key=lambda point: _point_coordinates(point) or (0.0, 0.0),
    )

    expansions = list(getattr(bot, "expansion_locations_list", ()) or ())
    own_expansions = _expansions_away_from(expansions, own_main)
    enemy_expansions = _expansions_away_from(expansions, enemy_main)
    own_natural = own_expansions[0] if own_expansions else None
    own_third = own_expansions[1] if len(own_expansions) > 1 else None
    enemy_natural = enemy_expansions[0] if enemy_expansions else None
    enemy_third = enemy_expansions[1] if len(enemy_expansions) > 1 else None

    worker_tags = {
        getattr(worker, "tag", id(worker))
        for worker in (getattr(bot, "workers", ()) or ())
    }
    own_army = [
        unit
        for unit in (getattr(bot, "units", ()) or ())
        if getattr(unit, "tag", id(unit)) not in worker_tags
    ]
    enemy_army = list(getattr(bot, "enemy_units", ()) or ())
    own_front = _collection_center(own_army) or own_main
    enemy_front = _collection_center(enemy_army)
    frontline = (
        _midpoint(own_front, enemy_front) if enemy_front is not None else own_front
    )
    proxy = _midpoint(map_center or own_main, enemy_natural or enemy_main)

    townhall = _first_position(getattr(bot, "townhalls", ())) or own_main
    return {
        "own_main": own_main,
        "own_natural": own_natural,
        "own_third": own_third,
        "own_ramp": own_ramp,
        "own_ramp_depot_1": corner_depots[0] if corner_depots else None,
        "own_ramp_depot_2": corner_depots[1] if len(corner_depots) > 1 else None,
        "own_ramp_depot_middle": _ramp_point(ramp, "depot_in_middle"),
        "own_ramp_barracks": _ramp_point(ramp, "barracks_in_middle"),
        "own_ramp_barracks_with_addon": _ramp_point(ramp, "barracks_correct_placement"),
        "enemy_main": enemy_main,
        "enemy_natural": enemy_natural,
        "enemy_third": enemy_third,
        "map_center": map_center,
        "frontline": frontline,
        "retreat": own_main,
        "proxy": proxy,
        "next_expansion": own_natural,
        "nearest_enemy": _closest_position(getattr(bot, "enemy_units", ()), own_front),
        "nearest_enemy_structure": _closest_position(
            getattr(bot, "enemy_structures", ()), own_front
        ),
        "nearest_mineral": _closest_position(
            getattr(bot, "mineral_field", ()), townhall
        ),
    }


def _ramp_point(ramp, attribute: str):
    if ramp is None:
        return None
    try:
        return getattr(ramp, attribute, None)
    except (AssertionError, AttributeError, IndexError, KeyError, ValueError):
        return None


def _ramp_points(ramp, attribute: str) -> list[Any]:
    value = _ramp_point(ramp, attribute)
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _expansions_away_from(points: list[Any], origin: Any | None) -> list[Any]:
    if origin is None:
        return points
    candidates = [
        point for point in points if _point_distance_squared(point, origin) > 4.0
    ]
    return sorted(candidates, key=lambda point: _point_distance_squared(point, origin))


def _collection_center(units: Iterable[Any]) -> tuple[float, float] | None:
    positions = [getattr(unit, "position", unit) for unit in units]
    coordinates = [_point_coordinates(position) for position in positions]
    valid = [coordinate for coordinate in coordinates if coordinate is not None]
    if not valid:
        return None
    return (
        sum(coordinate[0] for coordinate in valid) / len(valid),
        sum(coordinate[1] for coordinate in valid) / len(valid),
    )


def _midpoint(first: Any | None, second: Any | None) -> tuple[float, float] | None:
    first_xy = _point_coordinates(first)
    second_xy = _point_coordinates(second)
    if first_xy is None:
        return second_xy
    if second_xy is None:
        return first_xy
    return ((first_xy[0] + second_xy[0]) / 2, (first_xy[1] + second_xy[1]) / 2)


def _closest_position(units: Iterable[Any], reference: Any | None):
    values = list(units or ())
    if not values:
        return None
    if reference is None:
        return getattr(values[0], "position", values[0])
    unit = min(
        values,
        key=lambda value: _point_distance_squared(
            getattr(value, "position", value), reference
        ),
    )
    return getattr(unit, "position", unit)


def _point_coordinates(point: Any | None) -> tuple[float, float] | None:
    x = _point_component(point, "x", 0)
    y = _point_component(point, "y", 1)
    if x is None or y is None:
        return None
    return x, y


def _point_distance_squared(first: Any, second: Any) -> float:
    first_xy = _point_coordinates(first)
    second_xy = _point_coordinates(second)
    if first_xy is None or second_xy is None:
        return float("inf")
    return (first_xy[0] - second_xy[0]) ** 2 + (first_xy[1] - second_xy[1]) ** 2


def _first_position(units):
    if not units:
        return None
    unit = units[0]
    return getattr(unit, "position", unit)


def _location_snapshot(point) -> LocationSnapshot | None:
    if point is None:
        return None
    x = _point_component(point, "x", 0)
    y = _point_component(point, "y", 1)
    if x is None or y is None:
        return None
    return LocationSnapshot(x=x, y=y)


def _point_component(point, attr: str, index: int) -> float | None:
    if point is None:
        return None
    value = getattr(point, attr, None)
    if value is None:
        try:
            value = point[index]
        except (TypeError, IndexError, KeyError):
            return None
    return _optional_float(value)


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _order_name(order) -> str:
    ability = getattr(order, "ability", order)
    return str(getattr(ability, "name", ability))


def _unit_type_name(unit) -> str:
    raw_type = getattr(unit, "type_id", "unknown")
    name = getattr(raw_type, "name", str(raw_type))
    canonical = canonical_runtime_actor_name(name)
    return "worker" if canonical == "scv" else canonical


def _enemy_race_name(value: Any) -> str | None:
    if value is None:
        return None
    raw_name = normalize_name(str(getattr(value, "name", value)))
    if raw_name in {"terran", "protoss", "zerg", "random"}:
        return raw_name
    return "unknown"


def _active_alert_names(bot: Any) -> tuple[str, ...]:
    canonical_by_normalized = {
        normalize_name(alert).replace("_", ""): alert for alert in ALERT_KEYS
    }
    alerts = getattr(getattr(bot, "state", None), "alerts", ()) or ()
    result = set()
    for alert in alerts:
        raw_name = normalize_name(str(getattr(alert, "name", alert))).replace(
            "_", ""
        )
        canonical = canonical_by_normalized.get(raw_name)
        if canonical is not None:
            result.add(canonical)
    return tuple(sorted(result))


def _structure_is_ready(structure) -> bool:
    is_ready = getattr(structure, "is_ready", None)
    if is_ready is not None:
        return bool(is_ready)
    build_progress = getattr(structure, "build_progress", None)
    if build_progress is not None:
        return float(build_progress) >= 1.0
    return True


def create_move_unit_bot_class(bot_ai_base, point2_class):
    class _MoveUnitBot(bot_ai_base):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            plan: StrategyPlan | None = None,
            stop_after_seconds: int = 35,
            strategy: str | None = None,
            planner_name: str = DEFAULT_PLANNER,
            observe_before_plan: bool = False,
            original_goal: str | None = None,
            replan_limit: int = MAX_REPLANS,
        ):
            super().__init__()
            self.plan = plan
            self.stop_after_seconds = stop_after_seconds
            self.strategy = strategy
            self.original_strategy = original_goal or strategy
            self.planner_name = planner_name
            self.observe_before_plan = observe_before_plan
            self.replan_limit = max(0, min(MAX_REPLANS, int(replan_limit)))
            self._replan_count = 0
            self.observed_summary: GameStateSummary | None = None
            self._current_action_index = 0
            self._action_started_at_loop_time: float | None = None
            self._action_context: dict[str, Any] = {}
            self._plan_finished_at_loop_time: float | None = None
            self._left_game = False
            self._production_policies: list[dict[str, Any]] = []
            self._control_flow_states: dict[int, dict[str, Any]] = {}
            self._timeout_scope_states: dict[int, dict[str, Any]] = {}
            self._observed_health_by_tag: dict[str, float] = {}

        async def on_start(self):
            self.client.game_step = 2
            if self.plan is None:
                print("Observing game state before planning...")
            else:
                self._print_plan_loaded()

        async def on_step(self, iteration: int):
            if self._plan_finished_at_loop_time is None:
                if self.plan is None:
                    try:
                        if not self._create_plan_from_observation():
                            return
                    except (
                        PlanValidationError,
                        PlannerError,
                        PlannerUnavailableError,
                        ValueError,
                    ) as exc:
                        print(f"Planner error: {exc}", file=sys.stderr)
                        self._left_game = True
                        await self.client.leave()
                        return
                policy_changed_plan = await self._run_production_policies(iteration)
                if policy_changed_plan or self._left_game:
                    return
                await self._execute_current_action(iteration)

            if not self._left_game and self._should_stop():
                print("MVP complete: strategy plan finished; leaving the game.")
                self._left_game = True
                await self.client.leave()

        def _create_plan_from_observation(self) -> bool:
            if not self.observe_before_plan:
                raise RuntimeError(
                    "strategy plan is not loaded and observe-before-plan is disabled"
                )
            if not self.strategy:
                raise RuntimeError("strategy text is required for observe-before-plan")

            self.observed_summary = summarize_bot_state(self)
            print(
                "Observed game state before planning: "
                f"minerals={self.observed_summary.minerals}, "
                f"supply_left={self.observed_summary.supply.left}, "
                f"workers={self.observed_summary.workers}, "
                f"townhalls={self.observed_summary.townhalls}"
            )
            self.plan = validate_strategy_plan(
                plan_strategy(
                    self.strategy,
                    planner_name=self.planner_name,
                    game_state=self.observed_summary,
                ),
                game_state=self.observed_summary,
                max_actions=MAX_PLAN_ACTIONS,
            )
            self._print_plan_loaded()
            return True

        def _print_plan_loaded(self) -> None:
            if self.plan is None:
                return
            print(f"Strategy plan loaded: {self._plan_action_count()} action(s)")
            for index, action in enumerate(self.plan.actions, start=1):
                print(f"  {index}. {self._describe_action(action)}")

        async def _execute_current_action(self, iteration: int) -> None:
            if self.plan is None:
                raise RuntimeError("strategy plan is not loaded")
            if self._current_action_index >= self._plan_action_count():
                if self._has_unsatisfied_production_policies():
                    return
                self._production_policies.clear()
                self._mark_plan_finished()
                return

            action = self.plan.actions[self._current_action_index]
            completed_scope = (
                id(action)
                if isinstance(action, WithTimeoutCommand)
                and id(action) in self._timeout_scope_states
                else None
            )
            if await self._expire_timeout_scope(skip_scope=completed_scope):
                return
            if isinstance(action, MoveCommand):
                await self._execute_move(action, iteration)
                return
            if isinstance(action, AttackMoveCommand):
                await self._execute_attack(action, iteration)
                return
            if isinstance(action, AttackUntilClearCommand):
                await self._execute_attack_until_clear(action, iteration)
                return
            if isinstance(action, AttackEnemyCommand):
                await self._execute_attack_enemy(action, iteration)
                return
            if isinstance(action, KiteCommand):
                await self._execute_kite(action, iteration)
                return
            if isinstance(action, PatrolCommand):
                await self._execute_patrol(action, iteration)
                return
            if isinstance(action, HoldPositionCommand):
                self._execute_unit_order(action, iteration, "hold_position")
                return
            if isinstance(action, StopCommand):
                self._execute_unit_order(action, iteration, "stop")
                return
            if isinstance(action, RallyCommand):
                await self._execute_rally(action, iteration)
                return
            if isinstance(action, WaitCommand):
                self._execute_wait(action)
                return
            if isinstance(action, WaitUntilCommand):
                await self._execute_wait_until(action, iteration)
                return
            if isinstance(action, ConditionalCommand):
                await self._execute_conditional(action)
                return
            if isinstance(action, RepeatCommand):
                await self._execute_repeat(action)
                return
            if isinstance(action, WithTimeoutCommand):
                self._execute_with_timeout(action)
                return
            if isinstance(action, GatherMineralsCommand):
                await self._execute_gather_minerals(action, iteration)
                return
            if isinstance(action, GatherGasCommand):
                await self._execute_gather_gas(action, iteration)
                return
            if isinstance(action, ReturnCargoCommand):
                self._execute_unit_order(action, iteration, "return_resource")
                return
            if isinstance(action, DistributeWorkersCommand):
                await self._execute_distribute_workers(action)
                return
            if isinstance(action, TrainUnitCommand):
                self._execute_train(action, iteration)
                return
            if isinstance(action, ProductionPolicyCommand):
                self._execute_production_policy(action)
                return
            if isinstance(action, StopProductionCommand):
                self._execute_stop_production(action)
                return
            if isinstance(action, BuildStructureCommand):
                await self._execute_build(action, iteration)
                return
            if isinstance(action, ExpandCommand):
                await self._execute_expand(action, iteration)
                return
            if isinstance(action, BuildAddonCommand):
                self._execute_addon(action, iteration)
                return
            if isinstance(action, MorphStructureCommand):
                self._execute_morph(action, iteration)
                return
            if isinstance(action, ResearchUpgradeCommand):
                self._execute_research(action, iteration)
                return
            if isinstance(action, RepairCommand):
                self._execute_repair(action, iteration)
                return
            if self._is_replan_action(action):
                await self._execute_replan_action(action)
                return
            if self._is_runtime_ability_action(action):
                await self._execute_runtime_ability(action, iteration)
                return

            raise TypeError(f"unsupported strategy action: {action!r}")

        async def _execute_move(self, command: MoveCommand, iteration: int) -> None:
            if command.wait_for_arrival and self._action_context.get("move_issued"):
                target = (
                    self._resolve_friendly_move_target(
                        command, self._select_units(command.unit, command)
                    )
                    if command.target_unit is not None or command.target_tag is not None
                    else await self._resolve_location(command)
                )
                selected_tags = set(self._action_context.get("move_actor_tags", ()))
                current_units = list(self._select_exact_units(command.unit))
                if selected_tags:
                    current_units = [
                        unit
                        for unit in current_units
                        if str(getattr(unit, "tag", "")) in selected_tags
                    ]
                if (
                    target is not None
                    and current_units
                    and all(
                        self._distance_squared(unit, target)
                        <= command.arrival_tolerance**2
                        for unit in current_units
                    )
                ):
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"{len(current_units)} {command.unit} unit(s) arrived"
                    )
                    self._advance_action()
                    return
                last_issue_iteration = int(
                    self._action_context.get("move_last_issue_iteration", 0)
                )
                if (
                    target is not None
                    and current_units
                    and (
                        command.target_unit is not None
                        or command.target_tag is not None
                    )
                    and iteration - last_issue_iteration >= 4
                ):
                    for unit in current_units:
                        self._issue_order(unit, "move", target, False)
                    self._action_context["move_last_issue_iteration"] = iteration
                elapsed = self._game_time_now() - float(
                    self._action_context.get("move_started_at", self._game_time_now())
                )
                if elapsed >= command.timeout_seconds:
                    await self._replan_or_leave(
                        f"move-and-wait timed out after {command.timeout_seconds:g}s"
                    )
                elif iteration % 22 == 0:
                    print(
                        f"Waiting for {command.unit} arrival ({elapsed:g}/{command.timeout_seconds:g}s)..."
                    )
                return

            initial_units = self._select_units(command.unit, command)
            if (
                getattr(command, "target_unit", None) is not None
                or getattr(command, "target_tag", None) is not None
            ):
                target = self._resolve_friendly_move_target(command, initial_units)
            else:
                target = await self._resolve_location(command)
            units = self._select_units(command.unit, command, target)
            if target is None:
                await self._retry_or_replan(
                    command, iteration, "move target is unavailable"
                )
                return
            if units:
                for unit in units:
                    self._issue_order(unit, "move", target, command.queued)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued move command to {len(units)} {command.unit} unit(s): {target}"
                )
                if command.wait_for_arrival:
                    self._action_context = {
                        "move_issued": True,
                        "move_actor_tags": tuple(
                            str(getattr(unit, "tag", ""))
                            for unit in units
                            if getattr(unit, "tag", None) is not None
                        ),
                        "move_started_at": self._game_time_now(),
                        "move_last_issue_iteration": iteration,
                    }
                else:
                    self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units...")

        def _resolve_friendly_move_target(self, command: MoveCommand, sources):
            source_tags = {str(getattr(source, "tag", "")) for source in list(sources)}
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                candidates = (
                    list(getattr(self, "units", []))
                    + list(getattr(self, "workers", []))
                    + list(getattr(self, "structures", []))
                )
                candidate = next(
                    (
                        item
                        for item in candidates
                        if str(getattr(item, "tag", "")) == expected
                        and expected not in source_tags
                    ),
                    None,
                )
                return (
                    candidate
                    if candidate is not None
                    and self._matches_named_target(candidate, command, "friendly")
                    else None
                )

            raw_target_name = getattr(command, "target_unit", None)
            target_name = (
                normalize_name(str(raw_target_name)) if raw_target_name else ""
            )
            if target_name in {
                "nearest_friendly",
                "damaged_friendly",
                "lowest_health_friendly",
                "highest_energy_friendly",
                "any_friendly",
            }:
                candidates = self._friendly_target_candidates(target_name)
            else:
                candidates = list(self._select_exact_units(target_name))
            candidates = [
                candidate
                for candidate in candidates
                if str(getattr(candidate, "tag", "")) not in source_tags
            ]
            if not candidates:
                return None
            if target_name in {"nearest_friendly", "any_friendly"} and sources:
                reference = self._first_unit(sources)
                candidates.sort(
                    key=lambda unit: self._distance_squared(unit, reference)
                )
            return candidates[0]

        async def _execute_attack(
            self, command: AttackMoveCommand, iteration: int
        ) -> None:
            target = await self._resolve_location(command)
            units = self._select_units(command.unit, command, target)
            if target is None:
                await self._retry_or_replan(
                    command, iteration, "attack target is unavailable"
                )
                return
            if units:
                for unit in units:
                    self._issue_order(unit, "attack", target, command.queued)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued attack command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(
                    f"Waiting for controllable {command.unit} units before attacking..."
                )

        async def _execute_attack_until_clear(
            self, command: AttackUntilClearCommand, iteration: int
        ) -> None:
            anchor = await self._resolve_location(command)
            if anchor is None:
                await self._retry_or_replan(
                    command, iteration, "attack-until-clear location is unavailable"
                )
                return

            actor_tags = set(self._action_context.get("clear_actor_tags", ()))
            if actor_tags:
                actors = [
                    unit
                    for unit in self._select_exact_units(command.unit)
                    if str(getattr(unit, "tag", "")) in actor_tags
                ]
            else:
                actors = list(self._select_units(command.unit, command, anchor))
            if not actors:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"attack-until-clear actors {command.unit} are unavailable",
                )
                return

            now = self._game_time_now()
            if "clear_started_at" not in self._action_context:
                self._action_context = {
                    "clear_started_at": now,
                    "clear_actor_tags": tuple(
                        str(getattr(actor, "tag", ""))
                        for actor in actors
                        if getattr(actor, "tag", None) is not None
                    ),
                    "clear_last_issue_iteration": -1000,
                }

            selector = normalize_name(command.target_unit or "nearest_enemy")
            radius_squared = command.radius**2
            nearby_enemies = [
                enemy
                for enemy in self._enemy_target_candidates(selector, "")
                if self._distance_squared(enemy, anchor) <= radius_squared
            ]
            last_issue = int(
                self._action_context.get("clear_last_issue_iteration", -1000)
            )
            if nearby_enemies:
                self._action_context.pop("clear_confirmed_at", None)
                reference = self._first_unit(actors)
                nearby_enemies.sort(
                    key=lambda enemy: self._distance_squared(enemy, reference)
                )
                if iteration - last_issue >= 4:
                    for actor in actors:
                        self._issue_order(actor, "attack", nearby_enemies[0], False)
                    self._action_context["clear_last_issue_iteration"] = iteration
            elif iteration - last_issue >= 8:
                for actor in actors:
                    self._issue_order(actor, "attack", anchor, False)
                self._action_context["clear_last_issue_iteration"] = iteration

            confirmed_area = self._location_is_visible(anchor) or any(
                self._distance_squared(actor, anchor)
                <= command.arrival_tolerance**2
                for actor in actors
            )
            if not nearby_enemies and confirmed_area:
                clear_since = self._action_context.get("clear_confirmed_at")
                if clear_since is None:
                    self._action_context["clear_confirmed_at"] = now
                elif now - float(clear_since) >= command.clear_seconds:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"area clear for {command.clear_seconds:g}s"
                    )
                    self._advance_action()
                    return

            elapsed = now - float(self._action_context["clear_started_at"])
            if elapsed >= command.timeout_seconds:
                reason = (
                    "attack-until-clear timed out after "
                    f"{command.timeout_seconds:g}s"
                )
                if command.on_timeout == "replan":
                    await self._replan_or_leave(reason)
                else:
                    print(f"Runtime action failed terminally: {reason}", file=sys.stderr)
                    self._left_game = True
                    await self.client.leave()
                return
            if iteration % 22 == 0:
                print(
                    f"Clearing area with {len(actors)} {command.unit} unit(s); "
                    f"nearby enemies={len(nearby_enemies)}"
                )

        async def _execute_attack_enemy(
            self, command: AttackEnemyCommand, iteration: int
        ) -> None:
            if command.wait_for_target_death and self._action_context.get(
                "focus_issued"
            ):
                target_tag = str(self._action_context.get("focus_target_tag", ""))
                target_candidates = self._attack_target_candidates(command)
                target = next(
                    (
                        candidate
                        for candidate in target_candidates
                        if str(getattr(candidate, "tag", id(candidate))) == target_tag
                    ),
                    None,
                )
                dead_units = getattr(getattr(self, "state", None), "dead_units", None)
                known_dead = dead_units is not None and target_tag in {
                    str(tag) for tag in dead_units
                }
                if target is None and known_dead:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        "focus-fire target eliminated"
                    )
                    self._advance_action()
                    return
                actor_tags = set(self._action_context.get("focus_actor_tags", ()))
                actors = [
                    actor
                    for actor in self._select_exact_units(command.unit)
                    if not actor_tags or str(getattr(actor, "tag", "")) in actor_tags
                ]
                last_issue_iteration = int(
                    self._action_context.get("focus_last_issue_iteration", 0)
                )
                if (
                    target is not None
                    and actors
                    and (iteration - last_issue_iteration >= 4)
                ):
                    for actor in actors:
                        self._issue_order(actor, "attack", target, False)
                    self._action_context["focus_last_issue_iteration"] = iteration
                elapsed = self._game_time_now() - float(
                    self._action_context.get("focus_started_at", self._game_time_now())
                )
                if elapsed >= command.timeout_seconds:
                    await self._replan_or_leave(
                        f"focus fire timed out after {command.timeout_seconds:g}s"
                    )
                elif iteration % 22 == 0:
                    print(
                        f"Maintaining focus fire ({elapsed:g}/{command.timeout_seconds:g}s)..."
                    )
                return

            units = self._select_units(command.unit, command)
            target = self._resolve_attack_target(command, units)
            if units and target is not None:
                for unit in units:
                    self._issue_order(unit, "attack", target, command.queued)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued attack command to {len(units)} {command.unit} unit(s) "
                    f"against visible {command.target_alliance} target "
                    f"{getattr(command, 'target_unit', None) or getattr(command, 'target_tag', None) or 'nearest_enemy'}"
                )
                if command.wait_for_target_death:
                    self._action_context = {
                        "focus_issued": True,
                        "focus_target_tag": str(getattr(target, "tag", id(target))),
                        "focus_actor_tags": tuple(
                            str(getattr(unit, "tag", ""))
                            for unit in units
                            if getattr(unit, "tag", None) is not None
                        ),
                        "focus_started_at": self._game_time_now(),
                        "focus_last_issue_iteration": iteration,
                    }
                else:
                    self._advance_action()
            elif iteration % 22 == 0:
                print(
                    f"Waiting for controllable {command.unit} units and the requested visible "
                    f"{command.target_alliance} target before attacking..."
                )

        def _attack_target_candidates(self, command: AttackEnemyCommand) -> list[Any]:
            if command.target_alliance == "neutral":
                return list(getattr(self, "destructables", []))
            return list(getattr(self, "enemy_units", [])) + list(
                getattr(self, "enemy_structures", [])
            )

        def _resolve_attack_target(
            self, command: AttackEnemyCommand | KiteCommand, units
        ):
            alliance = getattr(command, "target_alliance", "enemy")
            candidates = (
                self._attack_target_candidates(command)
                if isinstance(command, AttackEnemyCommand)
                else list(getattr(self, "enemy_units", []))
                + list(getattr(self, "enemy_structures", []))
            )
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                candidate = next(
                    (
                        target
                        for target in candidates
                        if str(getattr(target, "tag", "")) == expected
                    ),
                    None,
                )
                return (
                    candidate
                    if candidate is not None
                    and self._matches_named_target(candidate, command, alliance)
                    else None
                )
            selector = normalize_name(
                str(
                    getattr(command, "target_unit", None)
                    or (
                        "nearest_destructible"
                        if alliance == "neutral"
                        else "nearest_enemy"
                    )
                )
            )
            if alliance == "neutral":
                candidates = [
                    target
                    for target in candidates
                    if self._matches_named_target(target, command, "neutral")
                ]
            else:
                candidates = self._enemy_target_candidates(selector, "")
            if not candidates:
                return None
            if units and selector not in {
                "lowest_health_enemy",
                "highest_energy_enemy",
            }:
                reference = self._first_unit(units)
                candidates.sort(
                    key=lambda enemy: self._distance_squared(enemy, reference)
                )
            return candidates[0]

        async def _execute_kite(self, command: KiteCommand, iteration: int) -> None:
            if "kite_started_at" not in self._action_context:
                self._action_context["kite_started_at"] = self._game_time_now()
            elapsed = self._game_time_now() - float(
                self._action_context["kite_started_at"]
            )
            if elapsed >= command.duration_seconds:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"completed {command.duration_seconds:g}s kite window"
                )
                self._advance_action()
                return

            units = self._select_units(command.unit, command)
            target = self._resolve_attack_target(command, units)
            if target is None:
                if self._action_context.get("kite_target_seen"):
                    self._advance_action()
                else:
                    await self._retry_or_replan(
                        command, iteration, "kite target is unavailable"
                    )
                return
            if not units:
                await self._retry_or_replan(
                    command, iteration, f"kite actors {command.unit} are unavailable"
                )
                return

            self._action_context["kite_target_seen"] = True
            for unit in units:
                cooldown = float(getattr(unit, "weapon_cooldown", 0.0) or 0.0)
                if cooldown <= 0:
                    self._issue_order(unit, "attack", target, False)
                    continue
                retreat_point = self._retreat_point_from_target(
                    unit, target, command.retreat_distance, point2_class
                )
                self._issue_order(unit, "move", retreat_point, False)
            if iteration % 8 == 1:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"kiting {target} with {len(units)} {command.unit} unit(s)"
                )

        @staticmethod
        def _retreat_point_from_target(unit, target, distance: float, point_class):
            unit_position = getattr(unit, "position", unit)
            target_position = getattr(target, "position", target)
            ux, uy = _point_coordinates(unit_position) or (0.0, 0.0)
            tx, ty = _point_coordinates(target_position) or (ux - 1.0, uy)
            dx, dy = ux - tx, uy - ty
            magnitude = max((dx * dx + dy * dy) ** 0.5, 0.001)
            return point_class(
                (
                    ux + distance * dx / magnitude,
                    uy + distance * dy / magnitude,
                )
            )

        async def _execute_patrol(self, command: PatrolCommand, iteration: int) -> None:
            target = await self._resolve_location(command)
            units = self._select_units(command.unit, command, target)
            if target is None:
                await self._retry_or_replan(
                    command, iteration, "patrol target is unavailable"
                )
                return
            if units:
                for unit in units:
                    self._issue_order(unit, "patrol", target, command.queued)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued patrol command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(
                    f"Waiting for controllable {command.unit} units before patrolling..."
                )

        def _execute_unit_order(
            self, command, iteration: int, method_name: str
        ) -> None:
            units = self._select_units(command.unit, command)
            if units:
                for unit in units:
                    self._issue_order(
                        unit, method_name, None, getattr(command, "queued", False)
                    )
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued {method_name.replace('_', ' ')} to {len(units)} {command.unit} unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(
                    f"Waiting for controllable {command.unit} units before {method_name.replace('_', ' ')}..."
                )

        async def _execute_rally(self, command: RallyCommand, iteration: int) -> None:
            target = await self._resolve_rally_target(command)
            structures = self._apply_selection_spec(
                self._ready_structures(command.building), command, target
            )
            if target is None:
                await self._retry_or_replan(
                    command, iteration, "rally target is unavailable"
                )
                return
            if structures:
                ability = self._rally_ability(command.building)
                for structure in structures:
                    self._issue_ability(structure, ability, target, command.queued)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"set {len(structures)} {command.building} rally point(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(
                    f"Waiting for a ready {command.building} before setting rally point..."
                )

        async def _resolve_rally_target(self, command: RallyCommand):
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                candidates = (
                    list(getattr(self, "units", []))
                    + list(getattr(self, "workers", []))
                    + list(getattr(self, "structures", []))
                    + list(getattr(self, "mineral_field", []))
                )
                candidate = next(
                    (
                        item
                        for item in candidates
                        if str(getattr(item, "tag", "")) == expected
                    ),
                    None,
                )
                target_unit = getattr(command, "target_unit", None)
                if candidate is None or target_unit is None:
                    return candidate
                if normalize_name(str(target_unit)) == "nearest_mineral":
                    return (
                        candidate
                        if any(
                            str(getattr(mineral, "tag", "")) == expected
                            for mineral in getattr(self, "mineral_field", [])
                        )
                        else None
                    )
                return (
                    candidate
                    if self._matches_named_target(candidate, command, "friendly")
                    else None
                )
            target_unit = getattr(command, "target_unit", None)
            if target_unit is None:
                return await self._resolve_location(command)
            target_name = normalize_name(str(target_unit))
            if target_name == "nearest_mineral":
                minerals = getattr(self, "mineral_field", [])
                if not minerals:
                    return None
                reference = (
                    self._first_unit(self.townhalls)
                    if getattr(self, "townhalls", None)
                    else self._build_near_point()
                )
                return (
                    minerals.closest_to(reference)
                    if hasattr(minerals, "closest_to")
                    else min(
                        minerals,
                        key=lambda mineral: self._distance_squared(mineral, reference),
                    )
                )
            return self._resolve_unit_target(command, "friendly")

        def _execute_wait(self, command: WaitCommand) -> None:
            game_time = getattr(self, "time", None)
            now = (
                float(game_time)
                if game_time is not None
                else asyncio.get_running_loop().time()
            )
            if self._action_started_at_loop_time is None:
                self._action_started_at_loop_time = now
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"waiting {command.seconds:g} second(s)"
                )

            elapsed = now - self._action_started_at_loop_time
            if elapsed >= command.seconds:
                self._advance_action()

        async def _execute_wait_until(
            self, command: WaitUntilCommand, iteration: int
        ) -> None:
            current = await self._wait_until_observed_value(command)
            current_text = "unavailable" if current is None else f"{current:g}"
            if current is not None and self._condition_comparison_met(
                current, command.comparison, command.at_least
            ):
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"condition met: {self._describe_action(command)} (current={current_text})"
                )
                self._advance_action()
                return

            if self._action_started_at_loop_time is None:
                self._action_started_at_loop_time = self._game_time_now()
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"waiting until {self._describe_action(command)} (current={current_text})"
                )
                return

            elapsed = self._game_time_now() - self._action_started_at_loop_time
            if elapsed >= command.timeout_seconds:
                reason = (
                    f"wait-until {command.condition} timed out after "
                    f"{command.timeout_seconds:g}s"
                )
                if command.on_timeout == "replan":
                    await self._replan_or_leave(reason)
                else:
                    print(
                        f"Runtime action failed terminally: {reason}", file=sys.stderr
                    )
                    self._left_game = True
                    await self.client.leave()
                return
            if iteration % 22 == 0:
                print(
                    f"Still waiting for {self._describe_action(command)} "
                    f"(current={current_text}, elapsed={elapsed:g}/{command.timeout_seconds:g}s)"
                )

        async def _execute_conditional(self, command: ConditionalCommand) -> None:
            matched = await self._condition_expression_met(command.when)
            replacement = command.then_actions if matched else command.else_actions
            branch_name = "then" if matched else "else"
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"conditional selected {branch_name} branch ({len(replacement)} action(s))"
            )
            self._replace_current_action(replacement)

        async def _execute_repeat(self, command: RepeatCommand) -> None:
            state_key = id(command)
            loop_state = self._control_flow_states.setdefault(
                state_key,
                {"started_at": self._game_time_now(), "cycles": 0},
            )
            cycles = int(loop_state["cycles"])

            if command.until is not None and await self._condition_expression_met(
                command.until
            ):
                self._control_flow_states.pop(state_key, None)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"repeat-until condition met after {cycles} cycle(s)"
                )
                self._replace_current_action(())
                return

            elapsed = self._game_time_now() - float(loop_state["started_at"])
            cycle_limit_reached = cycles >= command.max_cycles
            time_limit_reached = elapsed >= command.max_seconds
            if cycle_limit_reached or time_limit_reached:
                self._control_flow_states.pop(state_key, None)
                if command.until is None and cycle_limit_reached:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"repeat completed after {cycles} cycle(s)"
                    )
                    self._replace_current_action(())
                    return
                if command.until is not None and command.on_exhausted == "continue":
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"repeat-until continued after {cycles} cycle(s)"
                    )
                    self._replace_current_action(())
                    return
                reason = (
                    f"{'repeat-until exhausted' if command.until is not None else 'fixed repeat timed out'} after "
                    f"{cycles} cycle(s)/{elapsed:g}s"
                )
                if command.on_exhausted == "replan":
                    await self._replan_or_leave(reason)
                else:
                    print(f"Runtime action failed terminally: {reason}", file=sys.stderr)
                    self._left_game = True
                    await self.client.leave()
                return

            loop_state["cycles"] = cycles + 1
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"starting repeat cycle {cycles + 1}/{command.max_cycles}"
            )
            self._replace_current_action((*command.actions, command))

        def _execute_with_timeout(self, command: WithTimeoutCommand) -> None:
            state_key = id(command)
            if state_key in self._timeout_scope_states:
                self._timeout_scope_states.pop(state_key, None)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    "with-timeout sequence completed"
                )
                self._replace_current_action(())
                return

            self._timeout_scope_states[state_key] = {
                "started_at": self._game_time_now(),
                "command": command,
            }
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"starting with-timeout sequence ({command.timeout_seconds:g}s)"
            )
            self._replace_current_action((*command.actions, command))

        async def _expire_timeout_scope(self, skip_scope: int | None = None) -> bool:
            now = self._game_time_now()
            expired: list[tuple[float, int, WithTimeoutCommand]] = []
            for state_key, scope in self._timeout_scope_states.items():
                if state_key == skip_scope:
                    continue
                command = scope.get("command")
                if not isinstance(command, WithTimeoutCommand):
                    continue
                started_at = float(scope.get("started_at", now))
                deadline = started_at + command.timeout_seconds
                if now >= deadline:
                    expired.append((deadline, state_key, command))
            if not expired:
                return False

            _, state_key, command = min(expired, key=lambda item: item[0])
            self._timeout_scope_states.pop(state_key, None)
            reason = (
                "with-timeout sequence exceeded "
                f"{command.timeout_seconds:g}s before all child actions completed"
            )
            if command.on_timeout == "replan":
                await self._replan_or_leave(reason)
            else:
                print(f"Runtime action failed terminally: {reason}", file=sys.stderr)
                self._left_game = True
                await self.client.leave()
            return True

        async def _condition_expression_met(
            self, expression: ConditionExpression
        ) -> bool:
            if isinstance(expression, ConditionGroup):
                results = [
                    await self._atomic_condition_met(condition)
                    for condition in expression.conditions
                ]
                return all(results) if expression.match == "all" else any(results)
            return await self._atomic_condition_met(expression)

        async def _atomic_condition_met(self, condition: ConditionSpec) -> bool:
            current = await self._wait_until_observed_value(condition)
            return current is not None and self._condition_comparison_met(
                current, condition.comparison, condition.value
            )

        @staticmethod
        def _condition_comparison_met(
            current: float, comparison: str, expected: float
        ) -> bool:
            if comparison == "gte":
                return current >= expected
            if comparison == "lte":
                return current <= expected
            if comparison == "eq":
                return math.isclose(current, expected, rel_tol=1e-9, abs_tol=1e-9)
            if comparison == "neq":
                return not math.isclose(
                    current, expected, rel_tol=1e-9, abs_tol=1e-9
                )
            if comparison == "gt":
                return current > expected
            if comparison == "lt":
                return current < expected
            raise TypeError(f"unsupported condition comparison: {comparison}")

        async def _execute_gather_minerals(
            self, command: GatherMineralsCommand, iteration: int
        ) -> None:
            workers = self._select_units(command.unit, command)
            mineral_fields = list(self.mineral_field)
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                mineral_fields = [
                    mineral
                    for mineral in mineral_fields
                    if str(getattr(mineral, "tag", "")) == expected
                ]
            anchor = (
                await self._resolve_location(command)
                if getattr(command, "location", None) is not None
                else None
            )
            anchored_mineral = (
                min(
                    mineral_fields,
                    key=lambda mineral: self._distance_squared(mineral, anchor),
                )
                if anchor is not None and mineral_fields
                else None
            )
            if workers and mineral_fields:
                issued = 0
                selected_workers = (
                    workers[: command.workers]
                    if command.workers is not None
                    else workers
                )
                for worker in selected_workers:
                    mineral_field = anchored_mineral or self._closest_mineral_field(
                        mineral_fields, worker
                    )
                    self._issue_order(
                        worker,
                        "gather",
                        mineral_field,
                        bool(getattr(command, "queued", False)),
                    )
                    issued += 1
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued gather minerals command to {issued} worker unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print("Waiting for workers and mineral fields before gathering...")

        async def _execute_gather_gas(
            self, command: GatherGasCommand, iteration: int
        ) -> None:
            workers = self._select_units(command.unit, command)
            refineries = list(self._ready_refineries())
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                refineries = [
                    refinery
                    for refinery in refineries
                    if str(getattr(refinery, "tag", "")) == expected
                ]
            anchor = (
                await self._resolve_location(command)
                if getattr(command, "location", None) is not None
                else None
            )
            if anchor is not None:
                refineries.sort(
                    key=lambda refinery: self._distance_squared(refinery, anchor)
                )
            if workers and refineries:
                issued = 0
                requested_workers = (
                    command.workers
                    if command.workers is not None
                    else len(refineries) * 3
                )
                max_workers = min(len(workers), len(refineries) * 3, requested_workers)
                for index, worker in enumerate(workers[:max_workers]):
                    refinery = refineries[index % len(refineries)]
                    self._issue_order(
                        worker,
                        "gather",
                        refinery,
                        bool(getattr(command, "queued", False)),
                    )
                    issued += 1
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued gather gas command to {issued} worker unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(
                    "Waiting for workers and ready refineries before gathering gas..."
                )

        async def _execute_distribute_workers(
            self, command: DistributeWorkersCommand
        ) -> None:
            await self.distribute_workers(resource_ratio=command.mineral_to_gas_ratio)
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"distributed workers at mineral-to-gas ratio {command.mineral_to_gas_ratio:g}"
            )
            self._advance_action()

        def _execute_train(self, command: TrainUnitCommand, iteration: int) -> None:
            if command.unit not in UNIT_SPECS:
                raise TypeError(f"unsupported train unit: {command.unit}")
            if command.count < 1:
                raise TypeError(f"unsupported train count: {command.count}")
            if not self._action_context:
                self._action_context = {"issued": 0}
            issued_count = int(self._action_context.get("issued", 0))
            if issued_count >= command.count:
                self._advance_action()
                return

            unit_type = self._train_unit_type(command.unit)
            producers = self._apply_selection_spec(
                self._available_producers(command.unit),
                command,
                selection_override=getattr(command, "producer_selection", None),
            )
            if not producers:
                if iteration % 22 == 0:
                    print(
                        f"Waiting for an available producer to train {command.unit}..."
                    )
                return

            if hasattr(self, "can_afford") and not self.can_afford(unit_type):
                if iteration % 22 == 0:
                    print(f"Waiting for enough resources to train {command.unit}...")
                return

            producer = self._first_unit(producers)
            producer.train(unit_type)
            issued_count += 1
            self._action_context["issued"] = issued_count
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued train {command.unit} command ({issued_count}/{command.count})"
            )
            if issued_count >= command.count:
                self._advance_action()

        def _execute_production_policy(self, command: ProductionPolicyCommand) -> None:
            policy = next(
                (
                    candidate
                    for candidate in self._production_policies
                    if candidate["unit"] == command.unit
                ),
                None,
            )
            if (
                not command.background
                and self._owned_unit_count(command.unit) >= command.target_count
            ):
                if (
                    policy is not None
                    and policy["command"].target_count <= command.target_count
                ):
                    self._production_policies.remove(policy)
                self._advance_action()
                return
            if policy is None:
                policy = {
                    "unit": command.unit,
                    "command": command,
                    "started_at": self._game_time_now(),
                    "initial_count": self._owned_unit_count(command.unit),
                    "issued": 0,
                }
                self._production_policies.append(policy)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"registered {'background ' if command.background else ''}production "
                    f"for {command.unit} until {command.target_count}"
                )
            else:
                existing = policy["command"]
                if command.target_count > existing.target_count:
                    policy["command"] = command

            if command.background:
                self._advance_action()
                return

        def _execute_stop_production(self, command: StopProductionCommand) -> None:
            before = len(self._production_policies)
            if command.unit is None:
                self._production_policies.clear()
            else:
                self._production_policies = [
                    policy
                    for policy in self._production_policies
                    if policy.get("unit") != command.unit
                ]
            stopped = before - len(self._production_policies)
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"stopped {stopped} production policy/policies"
            )
            self._advance_action()

        async def _run_production_policies(self, iteration: int) -> bool:
            del iteration
            for policy in list(self._production_policies):
                command: ProductionPolicyCommand = policy["command"]
                current = self._owned_unit_count(command.unit)
                if current >= command.target_count:
                    if not command.background:
                        self._production_policies.remove(policy)
                        print(
                            f"Production target reached: {current}/{command.target_count} {command.unit}"
                        )
                    continue

                elapsed = self._game_time_now() - float(policy["started_at"])
                if elapsed >= command.max_seconds:
                    await self._replan_or_leave(
                        f"production policy for {command.unit} timed out after {command.max_seconds:g}s"
                    )
                    return True

                unit_type = self._train_unit_type(command.unit)
                pending = self._pending_unit_count(unit_type)
                if pending is None:
                    produced = max(0, current - int(policy["initial_count"]))
                    pending = max(0, int(policy["issued"]) - produced)
                if current + pending >= command.target_count:
                    continue

                spec = UNIT_SPECS[command.unit]
                supply_cost = int(spec.supply or 0)
                if (
                    int(getattr(self, "minerals", 0))
                    < spec.minerals + command.reserve_minerals
                    or int(getattr(self, "vespene", 0))
                    < spec.vespene + command.reserve_vespene
                    or int(getattr(self, "supply_left", 0))
                    < supply_cost + command.reserve_supply
                ):
                    continue
                if hasattr(self, "can_afford") and not self.can_afford(unit_type):
                    continue

                producers = self._apply_selection_spec(
                    self._available_producers(command.unit),
                    command,
                    selection_override=command.producer_selection,
                )
                if not producers:
                    continue
                producer = self._first_unit(producers)
                producer.train(unit_type)
                policy["issued"] = int(policy["issued"]) + 1
                print(
                    f"Production policy issued {command.unit} "
                    f"({current}+{pending}/{command.target_count})"
                )
            return False

        def _owned_unit_count(self, unit: str) -> int:
            if unit == "scv":
                return len(getattr(self, "workers", []))
            return len(self._select_exact_units(unit))

        def _has_unsatisfied_production_policies(self) -> bool:
            return any(
                self._owned_unit_count(policy["command"].unit)
                < policy["command"].target_count
                for policy in self._production_policies
            )

        def _pending_unit_count(self, unit_type) -> int | None:
            if not hasattr(self, "already_pending"):
                return None
            try:
                return max(0, int(self.already_pending(unit_type)))
            except (TypeError, ValueError):
                return None

        async def _execute_build(
            self, command: BuildStructureCommand, iteration: int
        ) -> None:
            if command.building not in STRUCTURE_SPECS:
                raise TypeError(f"unsupported build structure: {command.building}")
            if command.building == "command_center" and command.location is None:
                await self._execute_expand(
                    ExpandCommand(count=command.count), iteration
                )
                return

            unit_type = self._building_unit_type(command.building)
            if not self._action_context:
                self._action_context = {
                    "building": command.building,
                    "total_before": self._structure_count(
                        command.building, readiness="total"
                    ),
                    "issued": 0,
                }

            issued_count = int(self._action_context.get("issued", 0))
            started_count = self._builds_started_since_action_start(command.building)
            if issued_count and started_count >= issued_count:
                if issued_count >= command.count:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"{command.count} build {command.building} command(s) started"
                    )
                    self._advance_action()
                    return
            elif issued_count:
                if iteration % 22 == 0:
                    print(
                        f"Waiting for {command.building} construction to start ({started_count}/{issued_count})..."
                    )
                return

            if hasattr(self, "can_afford") and not self.can_afford(unit_type):
                if iteration % 22 == 0:
                    print(
                        f"Waiting for enough resources to build {command.building}..."
                    )
                return

            if command.building == "refinery":
                near = (
                    await self._resolve_location(command)
                    if command.location is not None
                    else None
                )
                issued = self._execute_refinery_build(
                    unit_type, near=near, command=command
                )
            else:
                near = (
                    await self._resolve_location(command)
                    if command.location is not None
                    else self._build_near_point()
                )
                if near is None:
                    await self._retry_or_replan(
                        command,
                        iteration,
                        f"build location for {command.building} is unavailable",
                    )
                    return
                issued = await self._issue_structure_build(command, unit_type, near)

            if issued:
                issued_count += 1
                self._action_context["issued"] = issued_count
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued build {command.building} command ({issued_count}/{command.count})"
                )
                if (
                    self._builds_started_since_action_start(command.building)
                    >= command.count
                ):
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"{command.count} build {command.building} command(s) started"
                    )
                    self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for placement/worker to build {command.building}...")

        async def _execute_expand(self, command: ExpandCommand, iteration: int) -> None:
            if not self._action_context:
                self._action_context = {
                    "townhalls_before": len(self.townhalls),
                    "issued": 0,
                }
            issued_count = int(self._action_context.get("issued", 0))
            started_count = max(
                0,
                len(self.townhalls)
                - int(self._action_context.get("townhalls_before", 0)),
            )
            if issued_count and started_count >= issued_count:
                if issued_count >= command.count:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"{command.count} expansion command(s) started"
                    )
                    self._advance_action()
                    return
            elif issued_count:
                if iteration % 22 == 0:
                    print(
                        f"Waiting for expansion construction to start ({started_count}/{issued_count})..."
                    )
                return

            command_center_type = self._building_unit_type("command_center")
            if hasattr(self, "can_afford") and not self.can_afford(command_center_type):
                if iteration % 22 == 0:
                    print("Waiting for enough resources to expand...")
                return
            issued = await self._issue_expansion(command_center_type)
            if not issued:
                if iteration % 22 == 0:
                    print(
                        "Waiting for an available expansion location and build worker..."
                    )
                return
            issued_count += 1
            self._action_context["issued"] = issued_count
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued expansion command ({issued_count}/{command.count})"
            )
            if (
                len(self.townhalls)
                - int(self._action_context.get("townhalls_before", 0))
                >= command.count
            ):
                self._advance_action()

        def _execute_addon(self, command: BuildAddonCommand, iteration: int) -> None:
            if command.addon not in ADDON_SPECS:
                raise TypeError(f"unsupported production add-on: {command.addon}")
            if not self._action_context:
                self._action_context = {"issued": 0, "producer_tags": set()}
            issued_count = int(self._action_context.get("issued", 0))
            if issued_count >= command.count:
                self._advance_action()
                return

            spec = ADDON_SPECS[command.addon]
            producers = self._apply_selection_spec(
                self._free_addon_producers(spec.producer or ""), command
            )
            used_tags = self._action_context.get("producer_tags", set())
            producer = next(
                (
                    item
                    for item in producers
                    if getattr(item, "tag", id(item)) not in used_tags
                ),
                None,
            )
            if producer is None:
                if iteration % 22 == 0:
                    print(
                        f"Waiting for a free {spec.producer} to build {command.addon}..."
                    )
                return
            addon_type = self._addon_unit_type(command.addon)
            if hasattr(self, "can_afford") and not self.can_afford(addon_type):
                if iteration % 22 == 0:
                    print(f"Waiting for enough resources to build {command.addon}...")
                return
            issued = producer.build(addon_type)
            if issued is False:
                return
            used_tags.add(getattr(producer, "tag", id(producer)))
            issued_count += 1
            self._action_context["issued"] = issued_count
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued build {command.addon} command ({issued_count}/{command.count})"
            )
            if issued_count >= command.count:
                self._advance_action()

        def _execute_morph(
            self, command: MorphStructureCommand, iteration: int
        ) -> None:
            if command.building not in MORPH_SPECS:
                raise TypeError(f"unsupported structure morph: {command.building}")
            sources = self._apply_selection_spec(
                self._available_command_centers(), command
            )
            if not sources:
                if iteration % 22 == 0:
                    print(
                        f"Waiting for an available command center to morph {command.building}..."
                    )
                return
            target_type = self._morph_unit_type(command.building)
            if hasattr(self, "can_afford") and not self.can_afford(target_type):
                if iteration % 22 == 0:
                    print(
                        f"Waiting for enough resources to morph {command.building}..."
                    )
                return
            issued = self._first_unit(sources).build(target_type)
            if issued is False:
                return
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued morph {command.building} command"
            )
            self._advance_action()

        def _execute_research(
            self, command: ResearchUpgradeCommand, iteration: int
        ) -> None:
            if command.upgrade not in UPGRADE_SPECS:
                raise TypeError(f"unsupported Terran upgrade: {command.upgrade}")
            upgrade_type = self._upgrade_id(command.upgrade)
            if (
                hasattr(self, "already_pending_upgrade")
                and self.already_pending_upgrade(upgrade_type) > 0
            ):
                self._advance_action()
                return
            if hasattr(self, "can_afford") and not self.can_afford(upgrade_type):
                if iteration % 22 == 0:
                    print(
                        f"Waiting for enough resources to research {command.upgrade}..."
                    )
                return
            researcher_selection = getattr(command, "researcher_selection", None)
            if researcher_selection is not None:
                researchers = self._apply_selection_spec(
                    self._ready_idle_structures(
                        UPGRADE_SPECS[command.upgrade].researcher or ""
                    ),
                    command,
                    selection_override=researcher_selection,
                )
                researcher = self._first_unit(researchers) if researchers else None
                issued = researcher.research(upgrade_type) if researcher else False
            elif hasattr(self, "research"):
                issued = self.research(upgrade_type)
            else:
                researcher = self._first_unit(
                    self._ready_idle_structures(
                        UPGRADE_SPECS[command.upgrade].researcher or ""
                    )
                )
                issued = researcher.research(upgrade_type) if researcher else False
            if issued:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued research {command.upgrade} command"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for an available researcher for {command.upgrade}...")

        def _execute_repair(self, command: RepairCommand, iteration: int) -> None:
            if command.target is not None:
                targets = list(self._repair_targets(command.target))
            else:
                combined_targets = []
                for target_name in REPAIRABLE_TARGET_KEYS:
                    combined_targets.extend(self._repair_targets(target_name))
                targets = list(
                    {
                        str(getattr(target, "tag", id(target))): target
                        for target in combined_targets
                    }.values()
                )
            target_tag = getattr(command, "target_tag", None)
            if target_tag is not None:
                expected = str(target_tag)
                targets = [
                    target
                    for target in targets
                    if str(getattr(target, "tag", "")) == expected
                ]
            if not targets:
                if iteration % 22 == 0:
                    print(f"Waiting for repair target {command.target}...")
                return
            damaged = [target for target in targets if self._is_damaged(target)]
            if not damaged:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"repair target {command.target} is already at full health"
                )
                self._advance_action()
                return
            target_selection = getattr(command, "target_selection", None)
            target_selector = normalize_name(
                str(getattr(command, "target_selector", ""))
            )
            if target_selection is None and target_selector:
                target_selection = {
                    "mode": (
                        "lowest_health"
                        if target_selector
                        in {"damaged_friendly", "lowest_health_friendly"}
                        else "closest"
                    ),
                    "count": 1,
                }
            damaged = self._apply_selection_spec(
                damaged,
                command,
                self._build_near_point(),
                selection_override=target_selection,
            )
            workers = self._apply_selection_spec(self.workers, command)[
                : command.workers
            ]
            if not workers:
                if iteration % 22 == 0:
                    print("Waiting for workers before repairing...")
                return
            if not damaged:
                if iteration % 22 == 0:
                    print(f"Waiting for selected repair target {command.target}...")
                return
            target = damaged[0]
            for worker in workers:
                worker.repair(target)
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued repair command with {len(workers)} worker(s)"
            )
            self._advance_action()

        def _builds_started_since_action_start(self, building: str) -> int:
            total_before = int(self._action_context.get("total_before", 0))
            return max(
                0, self._structure_count(building, readiness="total") - total_before
            )

        async def _issue_expansion(self, command_center_type) -> bool:
            if hasattr(self, "get_next_expansion"):
                location = await self.get_next_expansion()
                if location is None:
                    return False
                return bool(
                    await self.build(
                        command_center_type,
                        near=location,
                        max_distance=10,
                        random_alternative=False,
                        placement_step=1,
                    )
                )

            # Lightweight injected runtimes can implement only expand_now.
            await self.expand_now(building=command_center_type)
            return True

        async def _issue_structure_build(
            self, command: BuildStructureCommand, unit_type, near
        ) -> bool:
            placement_mode = normalize_name(
                str(getattr(command, "placement_mode", "near"))
            )
            configured_distance = int(getattr(command, "max_distance", 20))
            max_distance = 0 if placement_mode == "exact" else configured_distance
            reserve_addon_space = bool(getattr(command, "reserve_addon_space", False))
            if (
                command.selection is not None
                or placement_mode == "exact"
                or reserve_addon_space
            ) and hasattr(self, "find_placement"):
                try:
                    placement = self.find_placement(
                        unit_type,
                        near=near,
                        max_distance=max_distance,
                        random_alternative=False,
                        placement_step=1,
                        addon_place=reserve_addon_space,
                    )
                except TypeError:
                    placement = self.find_placement(
                        unit_type,
                        near=near,
                        max_distance=max_distance,
                        random_alternative=False,
                        placement_step=1,
                    )
                if hasattr(placement, "__await__"):
                    placement = await placement
                if placement is None:
                    return False
                if command.selection is not None:
                    workers = self._apply_selection_spec(
                        getattr(self, "workers", []), command, placement
                    )
                    worker = self._first_unit(workers) if workers else None
                elif hasattr(self, "select_build_worker"):
                    worker = self.select_build_worker(placement)
                else:
                    workers = list(getattr(self, "workers", []))
                    worker = self._first_unit(workers) if workers else None
                if worker is None:
                    return False
                issued = worker.build(unit_type, placement)
                return issued is not False
            return bool(
                await self.build(
                    unit_type,
                    near=near,
                    max_distance=max_distance,
                )
            )

        @staticmethod
        def _closest_mineral_field(mineral_fields, worker):
            if hasattr(mineral_fields, "closest_to"):
                return mineral_fields.closest_to(worker)
            return mineral_fields[0]

        def _available_townhalls(self):
            townhalls = self.townhalls
            if hasattr(townhalls, "ready"):
                townhalls = townhalls.ready
            return [
                townhall
                for townhall in townhalls
                if self._producer_has_queue_capacity(townhall)
            ]

        def _available_producers(self, unit: str):
            spec = UNIT_SPECS[unit]
            if spec.producer == "command_center":
                return self._available_townhalls()
            producers = [
                producer
                for producer in self._ready_structures(spec.producer or "")
                if self._producer_has_queue_capacity(producer)
            ]
            if spec.required_addon:
                producers = [
                    producer
                    for producer in producers
                    if self._producer_has_addon(producer, spec.required_addon)
                ]
            return producers

        def _producer_has_queue_capacity(self, producer) -> bool:
            has_reactor = bool(getattr(producer, "has_reactor", False))
            add_on_tag = getattr(producer, "add_on_tag", 0)
            if add_on_tag and add_on_tag in getattr(self, "reactor_tags", set()):
                has_reactor = True
            capacity = 2 if has_reactor else 1
            orders = getattr(producer, "orders", None)
            if orders is not None:
                return len(orders) < capacity
            return bool(getattr(producer, "is_idle", True))

        def _ready_idle_structures(self, building: str):
            structures = self._ready_structures(building)
            if hasattr(structures, "idle"):
                structures = structures.idle
            return structures

        def _ready_structures(self, building: str):
            structures = self._structures_of_type(self._structure_unit_type(building))
            if hasattr(structures, "ready"):
                structures = structures.ready
            return structures

        def _free_addon_producers(self, producer: str):
            structures = self._ready_idle_structures(producer)
            return [
                structure
                for structure in structures
                if not getattr(structure, "has_add_on", False)
                and not getattr(structure, "add_on_tag", 0)
            ]

        def _producer_has_addon(self, producer, addon: str) -> bool:
            if addon.endswith("tech_lab") and getattr(producer, "has_techlab", False):
                return True
            if addon.endswith("reactor") and getattr(producer, "has_reactor", False):
                return True
            add_on_tag = getattr(producer, "add_on_tag", 0)
            if addon.endswith("tech_lab") and add_on_tag in getattr(
                self, "techlab_tags", set()
            ):
                return True
            if addon.endswith("reactor") and add_on_tag in getattr(
                self, "reactor_tags", set()
            ):
                return True
            # Lightweight injected test runtimes do not model add-on tags. The
            # validated StrategyPlan remains the prerequisite authority there.
            return not hasattr(producer, "add_on_tag")

        def _available_command_centers(self):
            return self._ready_idle_structures("command_center")

        def _structures_of_type(self, unit_type):
            structures = getattr(self, "structures", [])
            if hasattr(structures, "of_type"):
                return structures.of_type({unit_type})
            return type(structures)(
                [
                    unit
                    for unit in structures
                    if getattr(unit, "type_id", None) == unit_type
                ]
            )

        def _ready_refineries(self):
            return [
                refinery
                for refinery in self._select_exact_units("refinery")
                if _structure_is_ready(refinery)
            ]

        def _structure_count(
            self,
            building: str,
            readiness: str = "total",
            command: Any | None = None,
        ) -> int:
            structures = list(self._select_exact_units(building))
            if command is not None:
                structures = list(self._apply_selection_spec(structures, command))
            if readiness == "total":
                return len(structures)
            if readiness == "ready":
                return sum(
                    1 for structure in structures if _structure_is_ready(structure)
                )
            if readiness == "pending":
                return sum(
                    1 for structure in structures if not _structure_is_ready(structure)
                )
            raise ValueError(f"unsupported structure readiness filter: {readiness}")

        async def _wait_until_observed_value(
            self, command: WaitUntilCommand | ConditionSpec
        ) -> float | None:
            if command.condition == "minerals":
                return float(self.minerals)
            if command.condition == "vespene":
                return float(self.vespene)
            if command.condition == "supply_left":
                return float(self.supply_left)
            if command.condition == "supply_used":
                return float(self.supply_used)
            if command.condition == "supply_cap":
                return float(self.supply_cap)
            if command.condition == "townhall_count":
                return float(
                    len(self._apply_selection_spec(self.townhalls, command))
                )
            if command.condition == "game_time":
                return float(getattr(self, "time", 0.0))
            if command.condition == "army_supply":
                observed = getattr(self, "supply_army", None)
                if observed is not None:
                    return float(observed)
                total = 0.0
                for unit in list(getattr(self, "units", [])):
                    key = _unit_type_name(unit)
                    spec = UNIT_SPECS.get(key)
                    if spec is not None:
                        total += float(spec.supply or 0)
                return total
            if command.condition == "enemy_unit_count":
                candidates = list(getattr(self, "enemy_units", []))
                if command.target is not None:
                    candidates = [
                        unit
                        for unit in candidates
                        if self._matches_named_target(unit, command, "enemy")
                    ]
                if command.selection is not None:
                    candidates = list(
                        self._apply_selection_spec(candidates, command)
                    )
                return float(len(candidates))
            if command.condition == "enemy_structure_count":
                candidates = list(getattr(self, "enemy_structures", []))
                if command.target is not None:
                    candidates = [
                        unit
                        for unit in candidates
                        if self._matches_named_target(unit, command, "enemy")
                    ]
                if command.selection is not None:
                    candidates = list(
                        self._apply_selection_spec(candidates, command)
                    )
                return float(len(candidates))
            if command.condition == "enemy_race":
                current_race = (
                    _enemy_race_name(getattr(self, "enemy_race", None))
                    or "unknown"
                )
                return 1.0 if current_race == command.target else 0.0
            if command.condition == "alert_active":
                return 1.0 if command.target in _active_alert_names(self) else 0.0
            if command.condition == "location_visible":
                anchor = await self._resolve_location(command)
                return 1.0 if anchor is not None and self._location_is_visible(anchor) else 0.0
            if command.condition == "idle_structure_count":
                structures = self._ready_idle_structures(command.target or "")
                return float(len(self._apply_selection_spec(structures, command)))
            if command.condition == "producer_available":
                producers = self._available_producers(command.target or "")
                return float(len(self._apply_selection_spec(producers, command)))
            if command.condition == "ability_available":
                ability_key = command.ability or ""
                ability_spec = _RUNTIME_ABILITY_SPECS.get(ability_key)
                if ability_spec is None:
                    return 0.0
                actor = command.actor or (
                    ability_spec.actors[0] if ability_spec.actors else "any"
                )
                if actor == "any":
                    sources = []
                    for compatible_actor in ability_spec.actors:
                        sources.extend(self._select_exact_units(compatible_actor))
                    sources = list(
                        {
                            str(getattr(source, "tag", id(source))): source
                            for source in sources
                        }.values()
                    )
                    sources = self._apply_selection_spec(sources, command)
                else:
                    sources = self._apply_selection_spec(
                        self._select_exact_units(actor), command
                    )
                ability_id = self._ability_id_for_enum(ability_spec.enum_name)
                available = 0
                for source in sources:
                    if await self._unit_has_available_ability(source, ability_id):
                        available += 1
                return float(available)
            if command.condition == "unit_form_count":
                enum_names = UNIT_FORM_SPECS.get(command.target or "", ())
                raw_forms = {normalize_name(enum_name) for enum_name in enum_names}
                candidates = self._select_exact_units(command.actor or "any")
                candidates = self._apply_selection_spec(candidates, command)
                return float(
                    sum(
                        1
                        for unit in candidates
                        if self._raw_unit_type_name(unit) in raw_forms
                    )
                )
            if command.condition == "cargo_used":
                transports = self._apply_selection_spec(
                    self._select_exact_units(command.target or ""), command
                )
                return float(
                    sum(int(getattr(unit, "cargo_used", 0) or 0) for unit in transports)
                )
            if command.condition in {
                "idle_unit_count",
                "ready_unit_count",
                "damaged_unit_count",
                "cloaked_unit_count",
                "flying_unit_count",
                "loaded_unit_count",
                "weapon_ready_count",
                "unit_health",
                "unit_health_fraction",
                "unit_energy",
                "unit_order_count",
            }:
                candidates = list(self._select_exact_units(command.target or ""))
                candidates = list(self._apply_selection_spec(candidates, command))
                if command.condition == "idle_unit_count":
                    return float(
                        sum(
                            1
                            for unit in candidates
                            if bool(
                                getattr(
                                    unit,
                                    "is_idle",
                                    not bool(getattr(unit, "orders", ())),
                                )
                            )
                        )
                    )
                if command.condition == "ready_unit_count":
                    return float(
                        sum(1 for unit in candidates if getattr(unit, "is_ready", True))
                    )
                if command.condition == "damaged_unit_count":
                    return float(sum(1 for unit in candidates if self._is_damaged(unit)))
                if command.condition == "cloaked_unit_count":
                    return float(
                        sum(
                            1
                            for unit in candidates
                            if bool(
                                getattr(unit, "is_cloaked", False)
                                or getattr(unit, "is_burrowed", False)
                            )
                        )
                    )
                if command.condition == "flying_unit_count":
                    return float(sum(1 for unit in candidates if self._is_flying(unit)))
                if command.condition == "loaded_unit_count":
                    return float(
                        sum(
                            1
                            for unit in candidates
                            if bool(getattr(unit, "is_loaded", False))
                        )
                    )
                if command.condition == "weapon_ready_count":
                    return float(
                        sum(
                            1
                            for unit in candidates
                            if float(getattr(unit, "weapon_cooldown", 0.0) or 0.0)
                            <= 0
                        )
                    )
                if command.condition == "unit_order_count":
                    if not candidates:
                        return None
                    return float(
                        sum(len(getattr(unit, "orders", ()) or ()) for unit in candidates)
                    )
                if not candidates:
                    return None
                if command.condition == "unit_energy":
                    return max(float(getattr(unit, "energy", 0.0) or 0.0) for unit in candidates)
                if command.condition == "unit_health_fraction":
                    return min(self._health_ratio(unit) for unit in candidates)
                return min(
                    float(getattr(unit, "health", 0.0) or 0.0)
                    for unit in candidates
                )
            if command.condition in {
                "unit_near_location",
                "enemy_near_location",
            }:
                anchor = await self._resolve_location(command)
                if anchor is None:
                    return 0.0
                if command.condition == "unit_near_location":
                    candidates = self._apply_selection_spec(
                        self._select_exact_units(command.target or ""),
                        command,
                        anchor,
                    )
                else:
                    selector = normalize_name(str(command.target or "nearest_enemy"))
                    candidates = self._enemy_target_candidates(selector, "")
                    if command.selection is not None:
                        candidates = list(
                            self._apply_selection_spec(candidates, command, anchor)
                        )
                radius_squared = command.radius**2
                return float(
                    sum(
                        1
                        for unit in candidates
                        if self._distance_squared(unit, anchor) <= radius_squared
                    )
                )
            if command.condition == "under_attack":
                anchor = await self._resolve_location(command)
                if anchor is None:
                    return 0.0
                return self._under_attack_observed_value(
                    anchor, command.radius, command
                )
            if command.condition == "unit_count":
                candidates = (
                    self.workers
                    if command.target == "worker"
                    else self._select_exact_units(command.target or "")
                )
                return float(len(self._apply_selection_spec(candidates, command)))
            if command.condition == "upgrade_complete":
                upgrade = command.target or ""
                upgrade_type = self._upgrade_id(upgrade)
                completed: set[Any] = getattr(
                    getattr(self, "state", None), "upgrades", set()
                )
                return 1.0 if upgrade_type in completed else 0.0
            if command.condition == "structure_count":
                return float(
                    self._structure_count(
                        command.target or "", readiness="total", command=command
                    )
                )
            if command.condition == "structure_ready":
                return float(
                    self._structure_count(
                        command.target or "", readiness="ready", command=command
                    )
                )
            if command.condition == "structure_pending":
                return float(
                    self._structure_count(
                        command.target or "", readiness="pending", command=command
                    )
                )
            raise TypeError(f"unsupported wait-until condition: {command.condition}")

        def _under_attack_observed_value(
            self, anchor, radius: float, command: Any | None = None
        ) -> float:
            radius_squared = radius**2
            friendly = [
                unit
                for unit in self._select_exact_units("any")
                if getattr(unit, "is_ready", True)
                and self._distance_squared(unit, anchor) <= radius_squared
            ]
            enemies = [
                unit
                for unit in (
                    list(getattr(self, "enemy_units", []))
                    + list(getattr(self, "enemy_structures", []))
                )
                if self._distance_squared(unit, anchor) <= radius_squared
            ]
            if command is not None:
                friendly = list(
                    self._apply_selection_spec(friendly, command, anchor)
                )
            if not friendly:
                return 0.0

            friendly_by_tag = {str(getattr(unit, "tag", "")): unit for unit in friendly}
            confirmed_tags: set[str] = set()
            damaged_tags: set[str] = set()
            for unit_tag, unit in friendly_by_tag.items():
                health = float(getattr(unit, "health", 0.0) or 0.0)
                health_max = float(getattr(unit, "health_max", health) or health)
                previous = self._observed_health_by_tag.get(unit_tag)
                if previous is not None and health < previous:
                    confirmed_tags.add(unit_tag)
                if health_max > 0 and health < health_max:
                    damaged_tags.add(unit_tag)
                if bool(
                    getattr(unit, "is_under_attack", False)
                    or getattr(unit, "was_attacked", False)
                ):
                    confirmed_tags.add(unit_tag)
                self._observed_health_by_tag[unit_tag] = health

            for enemy in enemies:
                raw_target = getattr(enemy, "order_target", None)
                target_tag = str(getattr(raw_target, "tag", raw_target or ""))
                if target_tag in friendly_by_tag:
                    confirmed_tags.add(target_tag)

            if confirmed_tags:
                return float(len(confirmed_tags))
            if enemies and damaged_tags:
                return float(len(damaged_tags))
            if enemies and self._unit_under_attack_alert_active():
                return 1.0
            return 0.0

        def _unit_under_attack_alert_active(self) -> bool:
            alert = getattr(self, "alert", None)
            if callable(alert):
                try:
                    from sc2.data import Alert

                    return bool(alert(Alert.UnitUnderAttack))
                except (ImportError, AssertionError, AttributeError, TypeError):
                    pass
            for item in getattr(getattr(self, "state", None), "alerts", ()) or ():
                if normalize_name(str(getattr(item, "name", item))) in {
                    "unitunderattack",
                    "unit_under_attack",
                }:
                    return True
            return False

        def _execute_refinery_build(
            self, unit_type, near=None, command: BuildStructureCommand | None = None
        ) -> bool:
            geysers = getattr(self, "vespene_geyser", [])
            if not geysers:
                return False
            reference = near
            if reference is None and self.townhalls:
                townhall = self._first_unit(self.townhalls)
                reference = getattr(townhall, "position", townhall)
            geyser = (
                geysers.closest_to(reference)
                if reference is not None and hasattr(geysers, "closest_to")
                else geysers[0]
            )
            if command is not None and command.selection is not None:
                workers = self._apply_selection_spec(
                    getattr(self, "workers", []), command, geyser
                )
                worker = self._first_unit(workers) if workers else None
            else:
                if hasattr(self, "select_build_worker"):
                    worker = self.select_build_worker(geyser)
                else:
                    workers = list(getattr(self, "workers", []))
                    worker = self._first_unit(workers) if workers else None
            if not worker:
                return False
            worker.build(unit_type, geyser)
            return True

        def _build_near_point(self):
            if self.townhalls:
                townhall = self._first_unit(self.townhalls)
                return getattr(townhall, "position", townhall)
            return getattr(self, "start_location", point2_class((35, 42)))

        @staticmethod
        def _first_unit(units):
            if hasattr(units, "first"):
                return units.first
            return units[0]

        def _plan_action_count(self) -> int:
            plan = self.plan
            if plan is None:
                raise RuntimeError("strategy plan is not loaded")
            return len(plan.actions)

        def _advance_action(self) -> None:
            self._current_action_index += 1
            self._action_started_at_loop_time = None
            self._action_context = {}
            if self.plan is None:
                raise RuntimeError("strategy plan is not loaded")
            if self._current_action_index >= self._plan_action_count():
                if not self._has_unsatisfied_production_policies():
                    self._mark_plan_finished()

        def _replace_current_action(
            self, replacement: Iterable[Any]
        ) -> None:
            if self.plan is None:
                raise RuntimeError("strategy plan is not loaded")
            values = tuple(replacement)
            actions = self.plan.actions
            self.plan = StrategyPlan(
                actions=(
                    actions[: self._current_action_index]
                    + values
                    + actions[self._current_action_index + 1 :]
                )
            )
            self._action_started_at_loop_time = None
            self._action_context = {}
            if self._current_action_index >= self._plan_action_count():
                if not self._has_unsatisfied_production_policies():
                    self._mark_plan_finished()

        def _mark_plan_finished(self) -> None:
            if self._plan_finished_at_loop_time is None:
                self._plan_finished_at_loop_time = asyncio.get_running_loop().time()
                print("Strategy plan actions complete.")

        def _select_units(
            self, unit: str, command: Any | None = None, target: Any | None = None
        ):
            units = self._select_exact_units(unit)
            if unit in FLYING_STRUCTURE_ACTOR_KEYS and isinstance(
                command, (MoveCommand, PatrolCommand, HoldPositionCommand, StopCommand)
            ):
                units = [candidate for candidate in units if self._is_flying(candidate)]
            if isinstance(
                command,
                (
                    MoveCommand,
                    AttackMoveCommand,
                    PatrolCommand,
                    HoldPositionCommand,
                    AttackUntilClearCommand,
                    KiteCommand,
                ),
            ):
                immobile_forms = {
                    "siegetanksieged",
                    "widowmineburrowed",
                    "liberatorag",
                }
                units = [
                    candidate
                    for candidate in units
                    if self._raw_unit_type_name(candidate) not in immobile_forms
                ]
            return self._apply_selection_spec(units, command, target)

        def _select_exact_units(self, unit: str, command: Any | None = None):
            del command
            unit = normalize_name(unit)
            if unit == "any":
                candidates = (
                    list(getattr(self, "units", []))
                    + list(getattr(self, "workers", []))
                    + list(getattr(self, "structures", []))
                )
                return list(
                    {
                        getattr(item, "tag", id(item)): item for item in candidates
                    }.values()
                )
            if unit == "worker":
                return self.workers
            enum_names = RUNTIME_ACTOR_UNIT_TYPES.get(unit)
            if enum_names:
                unit_types = {
                    getattr(self._unit_type_id(), enum_name) for enum_name in enum_names
                }
                matches = []
                for collection in (
                    getattr(self, "units", []),
                    getattr(self, "structures", []),
                ):
                    if hasattr(collection, "of_type"):
                        matches.extend(collection.of_type(unit_types))
                    else:
                        matches.extend(
                            item
                            for item in collection
                            if getattr(item, "type_id", None) in unit_types
                        )
                return matches
            return []

        def _apply_selection_spec(
            self,
            units,
            command: Any | None = None,
            target: Any | None = None,
            selection_override: Any | None = None,
        ):
            selection = (
                selection_override
                if selection_override is not None
                else (
                    getattr(command, "selection", None) if command is not None else None
                )
            )
            count = self._selection_count(selection, command)
            mode = self._selection_mode(selection, command)
            result = list(units)
            selected_tags = self._selection_tags(selection)
            if selected_tags:
                result = [
                    unit
                    for unit in result
                    if str(getattr(unit, "tag", "")) in selected_tags
                ]
            if mode == "ready":
                result = [unit for unit in result if getattr(unit, "is_ready", True)]
            elif mode == "idle":
                result = [unit for unit in result if getattr(unit, "is_idle", True)]
            elif mode in {"damaged", "lowest_health"}:
                result.sort(
                    key=lambda unit: (
                        self._health_ratio(unit),
                        getattr(unit, "tag", id(unit)),
                    )
                )
            elif mode == "highest_energy":
                result.sort(
                    key=lambda unit: (
                        -float(getattr(unit, "energy", 0.0)),
                        getattr(unit, "tag", id(unit)),
                    )
                )
            elif mode in {"closest", "closest_to_target"} and target is not None:
                result.sort(
                    key=lambda unit: (
                        self._distance_squared(getattr(unit, "position", unit), target),
                        getattr(unit, "tag", id(unit)),
                    )
                )
            else:
                result.sort(key=lambda unit: getattr(unit, "tag", id(unit)))
            return result[:count]

        @staticmethod
        def _selection_tags(selection: Any | None) -> set[str]:
            if selection is None:
                return set()
            raw_tags = (
                selection.get("tags", ())
                if isinstance(selection, dict)
                else getattr(selection, "tags", ())
            )
            if raw_tags is None:
                return set()
            if isinstance(raw_tags, (str, int)):
                raw_tags = (raw_tags,)
            return {str(tag) for tag in raw_tags}

        @staticmethod
        def _selection_count(selection: Any | None, command: Any | None) -> int:
            candidates = []
            if selection is not None:
                candidates.extend(
                    [
                        getattr(selection, "count", None),
                        getattr(selection, "limit", None),
                        getattr(selection, "max_count", None),
                    ]
                )
                if isinstance(selection, dict):
                    candidates.extend(
                        [
                            selection.get("count"),
                            selection.get("limit"),
                            selection.get("max_count"),
                        ]
                    )
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    return max(0, min(MAX_SELECTION_COUNT, int(candidate)))
                except (TypeError, ValueError):
                    continue
            return MAX_SELECTION_COUNT

        @staticmethod
        def _selection_mode(selection: Any | None, command: Any | None) -> str:
            candidates = []
            if selection is not None:
                candidates.extend(
                    [getattr(selection, "mode", None), getattr(selection, "sort", None)]
                )
                if isinstance(selection, dict):
                    candidates.extend([selection.get("mode"), selection.get("sort")])
            if command is not None:
                candidates.extend(
                    [
                        getattr(command, "selection_mode", None),
                        getattr(command, "mode", None),
                    ]
                )
            for candidate in candidates:
                if candidate:
                    return normalize_name(str(candidate))
            return "first"

        @staticmethod
        def _health_ratio(unit) -> float:
            health_percentage = getattr(unit, "health_percentage", None)
            if health_percentage is not None:
                return float(health_percentage)
            health = getattr(unit, "health", None)
            health_max = getattr(unit, "health_max", None)
            if health is not None and health_max:
                return float(health) / float(health_max)
            return 1.0

        @staticmethod
        def _distance_squared(a, b) -> float:
            a = getattr(a, "position", a)
            b = getattr(b, "position", b)
            ax, ay = getattr(a, "x", None), getattr(a, "y", None)
            bx, by = getattr(b, "x", None), getattr(b, "y", None)
            if ax is None or ay is None:
                try:
                    ax, ay = a[0], a[1]
                except (TypeError, IndexError):
                    return 0.0
            if bx is None or by is None:
                try:
                    bx, by = b[0], b[1]
                except (TypeError, IndexError):
                    return 0.0
            return (float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2

        def _location_is_visible(self, point: Any) -> bool:
            is_visible = getattr(self, "is_visible", None)
            if not callable(is_visible):
                return False
            try:
                return bool(is_visible(point))
            except (AttributeError, TypeError, ValueError):
                return False

        def _game_time_now(self) -> float:
            game_time = getattr(self, "time", None)
            return (
                float(game_time)
                if game_time is not None
                else asyncio.get_running_loop().time()
            )

        @staticmethod
        def _action_kind(action: Any) -> str:
            explicit = (
                getattr(action, "type", None)
                or getattr(action, "kind", None)
                or getattr(action, "command", None)
            )
            if explicit:
                return normalize_name(str(explicit))
            name = type(action).__name__
            if name.endswith("Command"):
                name = name[:-7]
            snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
            return normalize_name(snake_name)

        def _is_replan_action(self, action: Any) -> bool:
            return self._action_kind(action) == "replan"

        def _is_runtime_ability_action(self, action: Any) -> bool:
            kind = self._action_kind(action)
            if kind in {
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
            }:
                return True
            ability = getattr(action, "ability", None) or getattr(
                action, "ability_key", None
            )
            return bool(
                ability and normalize_name(str(ability)) in _RUNTIME_ABILITY_SPECS
            )

        async def _execute_replan_action(self, action: Any) -> None:
            reason = str(getattr(action, "reason", "requested"))
            await self._replan_or_leave(f"explicit replan command: {reason}")

        async def _execute_runtime_ability(self, command: Any, iteration: int) -> None:
            ability_key = self._ability_key_for_action(command)
            spec = _RUNTIME_ABILITY_SPECS.get(ability_key)
            if spec is None:
                raise TypeError(f"unsupported Terran ability: {ability_key}")

            if self._action_kind(command) == "load":
                if spec.target_kind == "unit":
                    await self._execute_load_ability(
                        command, iteration, ability_key, spec
                    )
                else:
                    await self._execute_load_all_ability(
                        command, iteration, ability_key, spec
                    )
                return
            if self._action_kind(command) == "unload":
                await self._execute_unload_ability(
                    command, iteration, ability_key, spec
                )
                return

            target = await self._resolve_ability_target(command, spec)
            if target is None and spec.target_kind != "none":
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"target for ability {ability_key} is unavailable",
                )
                return

            source_name = self._source_name_for_action(command, spec, ability_key)
            sources = self._select_units(source_name, command, target)
            if not sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"source {source_name} for ability {ability_key} is unavailable",
                )
                return

            ability_id = self._ability_id_for_enum(spec.enum_name)
            available_sources = []
            for source in sources:
                if await self._unit_has_available_ability(source, ability_id):
                    available_sources.append(source)
            available_sources = self._limit_default_ability_sources(
                command, spec, available_sources
            )
            if not available_sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"ability {ability_key} is not currently available",
                )
                return

            issued = 0
            for source in available_sources:
                self._issue_ability(
                    source,
                    ability_id,
                    target if spec.target_kind != "none" else None,
                    bool(getattr(command, "queued", False)),
                )
                issued += 1
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued {ability_key} with {issued} source unit(s)"
            )
            self._advance_action()

        async def _execute_load_ability(
            self,
            command: Any,
            iteration: int,
            ability_key: str,
            spec: AbilitySpec,
        ) -> None:
            if self._action_context.get("transport_phase") == "loading":
                await self._continue_transport_completion(command, iteration)
                return
            source_name = self._source_name_for_action(command, spec, ability_key)
            sources = self._select_units(source_name, command)
            if not sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"source {source_name} for ability {ability_key} is unavailable",
                )
                return

            ability_id = self._ability_id_for_enum(spec.enum_name)
            available_sources = [
                source
                for source in sources
                if await self._unit_has_available_ability(source, ability_id)
            ]
            available_sources = self._limit_default_ability_sources(
                command, spec, available_sources
            )
            if not available_sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"ability {ability_key} is not currently available",
                )
                return

            raw_target_name = getattr(command, "target_unit", None)
            target_name = (
                normalize_name(str(raw_target_name)) if raw_target_name else ""
            )
            target_tag = getattr(command, "target_tag", None)
            if target_name == "nearest_destructible":
                targets = []
            elif target_name in TARGET_SELECTORS:
                targets = self._friendly_target_candidates(target_name)
            elif target_name:
                targets = list(self._select_exact_units(target_name))
            else:
                targets = list(getattr(self, "units", [])) + list(
                    getattr(self, "workers", [])
                )
            if target_tag is not None:
                expected = str(target_tag)
                targets = [
                    target
                    for target in targets
                    if str(getattr(target, "tag", "")) == expected
                ]
            targets = [
                target
                for target in targets
                if self._matches_ability_target(target, spec.target_filter)
            ]
            reference = getattr(available_sources[0], "position", available_sources[0])
            target_selection = getattr(command, "target_selection", None)
            targets = self._apply_selection_spec(
                targets,
                command,
                reference,
                selection_override=target_selection,
            )
            target_selection_count = (
                target_selection.get("count")
                if isinstance(target_selection, dict)
                else getattr(target_selection, "count", None)
            )
            requested = getattr(command, "count", None) or target_selection_count or 1
            if not targets:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"load targets {target_name} are unavailable",
                )
                return
            target_mode = self._selection_mode(target_selection, None)
            if target_mode not in {
                "damaged",
                "lowest_health",
                "highest_energy",
            }:
                targets.sort(
                    key=lambda unit: self._distance_squared(
                        getattr(unit, "position", unit), reference
                    )
                )
            if len(targets) < requested:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"only {len(targets)}/{requested} load targets {target_name} are available",
                )
                return
            targets = targets[:requested]

            for index, target in enumerate(targets):
                source = available_sources[index % len(available_sources)]
                self._issue_ability(
                    source,
                    ability_id,
                    target,
                    bool(
                        getattr(command, "queued", False)
                        or index >= len(available_sources)
                    ),
                )
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued {ability_key} for {len(targets)} target unit(s)"
            )
            if self._transport_completion_is_observable(available_sources):
                self._action_context = {
                    "transport_phase": "loading",
                    "transport_source_name": source_name,
                    "transport_source_tags": tuple(
                        str(getattr(source, "tag", "")) for source in available_sources
                    ),
                    "transport_target_tags": tuple(
                        str(getattr(target, "tag", "")) for target in targets
                    ),
                    "transport_started_at": self._game_time_now(),
                    "transport_cargo_before": {
                        str(getattr(source, "tag", "")): int(
                            getattr(source, "cargo_used", 0) or 0
                        )
                        for source in available_sources
                    },
                }
            else:
                self._advance_action()

        async def _execute_load_all_ability(
            self,
            command: Any,
            iteration: int,
            ability_key: str,
            spec: AbilitySpec,
        ) -> None:
            if self._action_context.get("transport_phase") == "loading_all":
                await self._continue_transport_completion(command, iteration)
                return

            source_name = self._source_name_for_action(command, spec, ability_key)
            sources = self._select_units(source_name, command)
            if not sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"source {source_name} for ability {ability_key} is unavailable",
                )
                return

            ability_id = self._ability_id_for_enum(spec.enum_name)
            available_sources = [
                source
                for source in sources
                if await self._unit_has_available_ability(source, ability_id)
            ]
            available_sources = self._limit_default_ability_sources(
                command, spec, available_sources
            )
            if not available_sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"ability {ability_key} is not currently available",
                )
                return

            cargo_before = {
                str(getattr(source, "tag", "")): int(
                    getattr(source, "cargo_used", 0) or 0
                )
                for source in available_sources
            }
            expected_target_tags = self._load_all_expected_target_tags(
                available_sources
            )
            for source in available_sources:
                self._issue_ability(
                    source,
                    ability_id,
                    None,
                    bool(getattr(command, "queued", False)),
                )
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued {ability_key} with {len(available_sources)} source unit(s)"
            )
            if self._transport_completion_is_observable(available_sources):
                self._action_context = {
                    "transport_phase": "loading_all",
                    "transport_source_name": source_name,
                    "transport_source_tags": tuple(cargo_before),
                    "transport_target_tags": expected_target_tags,
                    "transport_started_at": self._game_time_now(),
                    "transport_cargo_before": cargo_before,
                }
            else:
                self._advance_action()

        def _load_all_expected_target_tags(self, sources) -> tuple[str, ...]:
            workers = [
                worker
                for worker in list(getattr(self, "workers", []))
                if not bool(getattr(worker, "is_loaded", False))
                and any(
                    self._distance_squared(worker, source)
                    <= LOAD_ALL_NEARBY_RADIUS**2
                    for source in sources
                )
            ]
            workers.sort(
                key=lambda worker: (
                    min(self._distance_squared(worker, source) for source in sources),
                    str(getattr(worker, "tag", "")),
                )
            )
            capacity = sum(
                max(
                    0,
                    int(getattr(source, "cargo_max", 0) or 0)
                    - int(getattr(source, "cargo_used", 0) or 0),
                )
                for source in sources
            )
            if capacity > 0:
                workers = workers[:capacity]
            return tuple(
                str(getattr(worker, "tag", ""))
                for worker in workers
                if getattr(worker, "tag", None) is not None
            )

        async def _execute_unload_ability(
            self,
            command: Any,
            iteration: int,
            ability_key: str,
            spec: AbilitySpec,
        ) -> None:
            if self._action_context.get("transport_phase") == "unloading":
                await self._continue_transport_completion(command, iteration)
                return

            source_name = self._source_name_for_action(command, spec, ability_key)
            sources = self._select_units(source_name, command)
            if not sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"source {source_name} for ability {ability_key} is unavailable",
                )
                return

            target = None
            if spec.target_kind == "unit":
                if spec.target_alliance == "passenger":
                    target = self._resolve_passenger_target(command, sources=sources)
                    if target is not None and not self._matches_ability_target(
                        target, spec.target_filter
                    ):
                        target = None
                    if target is not None:
                        target_tag = str(getattr(target, "tag", ""))
                        sources = [
                            source
                            for source in sources
                            if any(
                                str(getattr(passenger, "tag", "")) == target_tag
                                for passenger in (
                                    getattr(source, "passengers", None)
                                    or getattr(source, "cargo", None)
                                    or []
                                )
                            )
                        ]
                else:
                    target = self._resolve_unit_target(
                        command, spec.target_alliance, ability_key=spec.key
                    )
                if target is None:
                    await self._retry_or_replan(
                        command,
                        iteration,
                        f"passenger for ability {ability_key} is unavailable",
                    )
                    return
            elif spec.target_kind == "point" and any(
                getattr(command, field, None) is not None
                for field in ("location", "x", "y")
            ):
                target = await self._resolve_location(command)
                if target is None:
                    await self._retry_or_replan(
                        command,
                        iteration,
                        f"target for ability {ability_key} is unavailable",
                    )
                    return

            ability_id = self._ability_id_for_enum(spec.enum_name)
            available_sources = [
                source
                for source in sources
                if await self._unit_has_available_ability(source, ability_id)
            ]
            available_sources = self._limit_default_ability_sources(
                command, spec, available_sources
            )
            if not available_sources:
                await self._retry_or_replan(
                    command,
                    iteration,
                    f"ability {ability_key} is not currently available",
                )
                return

            cargo_before = {
                str(getattr(source, "tag", "")): int(
                    getattr(source, "cargo_used", 0) or 0
                )
                for source in available_sources
            }
            passenger_tags_before = tuple(
                str(getattr(passenger, "tag", ""))
                for source in available_sources
                for passenger in (
                    getattr(source, "passengers", None)
                    or getattr(source, "cargo", None)
                    or []
                )
            )
            for source in available_sources:
                source_target = target
                if spec.target_kind == "point" and source_target is None:
                    source_target = getattr(source, "position", source)
                self._issue_ability(
                    source,
                    ability_id,
                    source_target if spec.target_kind != "none" else None,
                    bool(getattr(command, "queued", False)),
                )
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued {ability_key} with {len(available_sources)} source unit(s)"
            )
            if self._transport_completion_is_observable(available_sources):
                specific_tag = (
                    str(getattr(target, "tag", "")) if target is not None else ""
                )
                self._action_context = {
                    "transport_phase": "unloading",
                    "transport_source_name": source_name,
                    "transport_source_tags": tuple(cargo_before),
                    "transport_target_tags": (
                        (specific_tag,) if specific_tag else passenger_tags_before
                    ),
                    "transport_specific_unload": bool(specific_tag),
                    "transport_started_at": self._game_time_now(),
                    "transport_cargo_before": cargo_before,
                }
            else:
                self._advance_action()

        @staticmethod
        def _transport_completion_is_observable(sources) -> bool:
            return bool(sources) and all(
                hasattr(source, "cargo_used") for source in sources
            )

        async def _continue_transport_completion(
            self, command: Any, iteration: int
        ) -> None:
            phase = str(self._action_context.get("transport_phase", ""))
            source_name = str(self._action_context.get("transport_source_name", ""))
            source_tags = set(self._action_context.get("transport_source_tags", ()))
            sources = [
                source
                for source in self._select_exact_units(source_name)
                if not source_tags or str(getattr(source, "tag", "")) in source_tags
            ]
            expected_tags = set(self._action_context.get("transport_target_tags", ()))
            passenger_tags = {
                str(getattr(passenger, "tag", ""))
                for source in sources
                for passenger in (
                    getattr(source, "passengers", None)
                    or getattr(source, "cargo", None)
                    or []
                )
            }

            complete = False
            if phase == "loading":
                complete = bool(expected_tags) and expected_tags.issubset(
                    passenger_tags
                )
                if not complete:
                    friendly = list(getattr(self, "units", [])) + list(
                        getattr(self, "workers", [])
                    )
                    loaded_tags = {
                        str(getattr(unit, "tag", ""))
                        for unit in friendly
                        if bool(getattr(unit, "is_loaded", False))
                    }
                    complete = bool(expected_tags) and expected_tags.issubset(
                        loaded_tags
                    )
            elif phase == "loading_all":
                if expected_tags:
                    loaded_tags = set(passenger_tags)
                    loaded_tags.update(
                        str(getattr(unit, "tag", ""))
                        for unit in (
                            list(getattr(self, "units", []))
                            + list(getattr(self, "workers", []))
                        )
                        if bool(getattr(unit, "is_loaded", False))
                    )
                    complete = expected_tags.issubset(loaded_tags)
                else:
                    complete = bool(sources) and all(
                        int(getattr(source, "cargo_max", 0) or 0) > 0
                        and int(getattr(source, "cargo_used", 0) or 0)
                        >= int(getattr(source, "cargo_max", 0) or 0)
                        for source in sources
                    )
            elif bool(self._action_context.get("transport_specific_unload")):
                complete = bool(expected_tags) and expected_tags.isdisjoint(
                    passenger_tags
                )
            else:
                complete = bool(sources) and all(
                    int(getattr(source, "cargo_used", 0) or 0) == 0
                    for source in sources
                )

            if complete:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"transport {phase} completed"
                )
                self._advance_action()
                return

            elapsed = self._game_time_now() - float(
                self._action_context.get("transport_started_at", self._game_time_now())
            )
            if elapsed >= ABILITY_RETRY_SECONDS:
                await self._replan_or_leave(
                    f"transport {phase} did not complete within {ABILITY_RETRY_SECONDS:g}s"
                )
            elif iteration % 22 == 0:
                print(
                    f"Waiting for transport {phase} completion "
                    f"({elapsed:g}/{ABILITY_RETRY_SECONDS:g}s)..."
                )

        def _limit_default_ability_sources(
            self, command: Any, spec: AbilitySpec, sources
        ):
            if getattr(command, "selection", None) is not None:
                return sources
            single_source_kinds = {
                "scan",
                "call_down_mule",
                "supply_drop",
                "lift",
                "land",
                "load",
                "unload",
                "cancel",
                "salvage",
                "build_nuke",
                "launch_nuke",
            }
            if (
                spec.target_kind != "none"
                or self._action_kind(command) in single_source_kinds
            ):
                return list(sources)[:1]
            return sources

        def _ability_key_for_action(self, action: Any) -> str:
            kind = self._action_kind(action)
            if kind == "cancel":
                target = normalize_name(
                    str(
                        getattr(action, "target", None)
                        or getattr(action, "cancel_type", "")
                    )
                )
                if not target:
                    raw_ability = normalize_name(str(getattr(action, "ability", "")))
                    if raw_ability == "cancel_any" and getattr(action, "actor", None):
                        return "cancel_build_in_progress"
                    target = raw_ability.removeprefix("cancel_") or "any"
                if target in {"build_in_progress", "buildinprogress"}:
                    return "cancel_build_in_progress"
                return target if target.startswith("cancel_") else f"cancel_{target}"
            ability = getattr(action, "ability", None) or getattr(
                action, "ability_key", None
            )
            if ability:
                return normalize_name(str(ability))
            if kind in {
                "scan",
                "call_down_mule",
                "supply_drop",
                "build_nuke",
                "launch_nuke",
            }:
                return kind
            subject = normalize_name(
                str(
                    getattr(action, "building", None)
                    or getattr(action, "actor", None)
                    or getattr(action, "transport", None)
                    or getattr(action, "structure", None)
                    or getattr(action, "container", None)
                    or getattr(action, "unit", None)
                    or getattr(action, "source", None)
                    or getattr(action, "target", "")
                )
            )
            if kind == "transform":
                target = normalize_name(
                    str(getattr(action, "target", None) or getattr(action, "mode", ""))
                )
                transform_aliases = {
                    "hellbat": "morph_hellbat",
                    "hellion": "morph_hellion",
                    "siege_mode": "siege_mode",
                    "unsiege": "unsiege_mode",
                    "unsiege_mode": "unsiege_mode",
                    "siege": "siege_mode",
                    "tank": "siege_mode",
                    "high_impact": "thor_high_impact_mode",
                    "thor_high_impact_mode": "thor_high_impact_mode",
                    "explosive": "thor_explosive_mode",
                    "thor_explosive_mode": "thor_explosive_mode",
                    "assault": "viking_assault_mode",
                    "viking_assault_mode": "viking_assault_mode",
                    "fighter": "viking_fighter_mode",
                    "viking_fighter_mode": "viking_fighter_mode",
                    "ag": "liberator_ag_mode",
                    "liberator_ag_mode": "liberator_ag_mode",
                    "aa": "liberator_aa_mode",
                    "liberator_aa_mode": "liberator_aa_mode",
                    "lower_supply_depot": "lower_supply_depot",
                    "raise_supply_depot": "raise_supply_depot",
                }
                return transform_aliases.get(target, target)
            if kind in {"lift", "land"}:
                return f"{kind}_{subject}"
            if kind == "load":
                if subject in {"command_center", "orbital_command"}:
                    has_specific_target = any(
                        getattr(action, field_name, None) is not None
                        for field_name in (
                            "target_unit",
                            "target_tag",
                            "target_selection",
                        )
                    )
                    return (
                        "load_command_center"
                        if has_specific_target
                        else "load_all_command_center"
                    )
                if getattr(action, "all", False) or getattr(action, "load_all", False):
                    return f"load_all_{subject}"
                return f"load_{subject}"
            if kind == "unload":
                ability_subject = (
                    "command_center" if subject == "orbital_command" else subject
                )
                has_specific_passenger = any(
                    getattr(action, name, None) is not None
                    for name in ("target_unit", "passenger_tag")
                )
                if has_specific_passenger:
                    return f"unload_unit_{ability_subject}"
                return f"unload_all_{ability_subject}"
            if kind == "cancel":
                target = normalize_name(
                    str(
                        getattr(action, "target", None)
                        or getattr(action, "cancel_type", "any")
                    )
                )
                return target if target.startswith("cancel_") else f"cancel_{target}"
            if kind == "salvage":
                return f"salvage_{subject}"
            return kind

        def _source_name_for_action(
            self, action: Any, spec: AbilitySpec, ability_key: str
        ) -> str:
            source = (
                getattr(action, "source", None)
                or getattr(action, "caster", None)
                or getattr(action, "actor", None)
                or getattr(action, "transport", None)
                or getattr(action, "building", None)
            )
            if source:
                return normalize_name(str(source))
            unit = getattr(action, "unit", None)
            kind = self._action_kind(action)
            if unit and kind in {"use_ability", "transform"}:
                return normalize_name(str(unit))
            return spec.actors[0] if spec.actors else "any"

        async def _resolve_ability_target(self, action: Any, spec: AbilitySpec):
            if spec.target_kind == "none":
                return None
            if spec.target_kind == "point":
                return await self._resolve_location(action)
            if spec.target_kind == "mineral":
                anchor = await self._resolve_location(action)
                mineral_fields = getattr(self, "mineral_field", [])
                if not mineral_fields:
                    return None
                if anchor is not None and hasattr(mineral_fields, "closest_to"):
                    return mineral_fields.closest_to(anchor)
                return self._first_unit(mineral_fields)
            return self._resolve_unit_target(
                action, spec.target_alliance, ability_key=spec.key
            )

        async def _resolve_location(self, action: Any):
            target_addon = getattr(action, "target_addon", None)
            target_addon_tag = getattr(action, "target_addon_tag", None)
            if target_addon is not None or target_addon_tag is not None:
                return self._resolve_addon_land_position(target_addon, target_addon_tag)
            x = getattr(action, "x", None)
            y = getattr(action, "y", None)
            if x is not None and y is not None:
                return point2_class((float(x), float(y)))
            location = (
                getattr(action, "location", None)
                or getattr(action, "target_location", None)
                or getattr(action, "target", None)
            )
            if isinstance(location, dict):
                if "x" in location and "y" in location:
                    return point2_class((float(location["x"]), float(location["y"])))
                location = (
                    location.get("key")
                    or location.get("name")
                    or location.get("location")
                )
            elif location is not None and not isinstance(location, str):
                lx = getattr(location, "x", None)
                ly = getattr(location, "y", None)
                if lx is not None and ly is not None:
                    return point2_class((float(lx), float(ly)))
                location = (
                    getattr(location, "key", None)
                    or getattr(location, "semantic", None)
                    or getattr(location, "name", None)
                    or getattr(location, "location", None)
                )
            if location is None:
                return None
            key = normalize_name(str(location))
            if key == "next_expansion" and hasattr(self, "get_next_expansion"):
                resolved = self.get_next_expansion()
                if hasattr(resolved, "__await__"):
                    resolved = await resolved
            else:
                resolved = _semantic_location_points(self).get(key)
            coordinates = _point_coordinates(resolved)
            return point2_class(coordinates) if coordinates is not None else None

        def _resolve_addon_land_position(
            self, target_addon: Any | None, target_addon_tag: Any | None
        ):
            attached_tags = {
                str(tag)
                for structure in list(getattr(self, "structures", []))
                if (tag := getattr(structure, "add_on_tag", 0))
            }
            if target_addon is not None:
                candidates = list(
                    self._select_exact_units(normalize_name(str(target_addon)))
                )
            else:
                candidates = list(getattr(self, "structures", []))
            if target_addon_tag is not None:
                expected = str(target_addon_tag)
                candidates = [
                    addon
                    for addon in candidates
                    if str(getattr(addon, "tag", "")) == expected
                ]
            candidates = [
                addon
                for addon in candidates
                if _unit_type_name(addon) in ADDON_SPECS
                and str(getattr(addon, "tag", "")) not in attached_tags
                and getattr(addon, "add_on_land_position", None) is not None
            ]
            candidates.sort(key=lambda addon: str(getattr(addon, "tag", "")))
            if not candidates:
                return None
            position = getattr(candidates[0], "add_on_land_position")
            coordinates = _point_coordinates(position)
            return point2_class(coordinates) if coordinates is not None else position

        def _closest_unit_position(self, units):
            if not units:
                return None
            reference = (
                self._first_unit(self.townhalls)
                if getattr(self, "townhalls", None)
                else None
            )
            unit = (
                units.closest_to(reference)
                if reference is not None and hasattr(units, "closest_to")
                else units[0]
            )
            return getattr(unit, "position", unit)

        def _resolve_unit_target(
            self, action: Any, alliance: str, ability_key: str = ""
        ):
            spec = _RUNTIME_ABILITY_SPECS.get(ability_key)
            target_filter = spec.target_filter if spec is not None else "any"
            target_tag = getattr(action, "target_tag", None)
            if target_tag is not None:
                if alliance == "passenger":
                    candidate = self._resolve_passenger_target(action)
                else:
                    expected = str(target_tag)
                    if alliance == "enemy":
                        tagged_candidates = list(
                            getattr(self, "enemy_units", [])
                        ) + list(getattr(self, "enemy_structures", []))
                    else:
                        tagged_candidates = (
                            list(getattr(self, "units", []))
                            + list(getattr(self, "workers", []))
                            + list(getattr(self, "structures", []))
                        )
                    candidate = next(
                        (
                            item
                            for item in tagged_candidates
                            if str(getattr(item, "tag", "")) == expected
                        ),
                        None,
                    )
                if candidate is None or not self._matches_ability_target(
                    candidate, target_filter
                ):
                    return None
                return (
                    candidate
                    if self._matches_named_target(candidate, action, alliance)
                    else None
                )
            explicit = (
                getattr(action, "target_unit", None)
                or getattr(action, "target", None)
                or getattr(action, "passenger", None)
            )
            if explicit is not None and not isinstance(explicit, str):
                return (
                    explicit
                    if self._matches_ability_target(explicit, target_filter)
                    else None
                )
            target_name = normalize_name(str(explicit)) if explicit else ""
            if target_name == "nearest_destructible":
                return None
            if (
                alliance == "enemy"
                or target_name.startswith("nearest_enemy")
                or target_name.endswith("_enemy")
            ):
                candidates = self._enemy_target_candidates(target_name, ability_key)
                return candidates[0] if candidates else None
            if alliance == "passenger":
                candidate = self._resolve_passenger_target(action)
                if candidate is None or not self._matches_ability_target(
                    candidate, target_filter
                ):
                    return None
                return (
                    candidate
                    if self._matches_named_target(candidate, action, alliance)
                    else None
                )
            friendly = (
                list(getattr(self, "units", []))
                + list(getattr(self, "workers", []))
                + list(getattr(self, "structures", []))
            )
            friendly = [
                unit
                for unit in friendly
                if self._matches_ability_target(unit, target_filter)
            ]
            if target_name in {
                "nearest_friendly",
                "any_friendly",
                "damaged_friendly",
                "lowest_health_friendly",
                "highest_energy_friendly",
            }:
                candidates = [
                    unit
                    for unit in self._friendly_target_candidates(target_name)
                    if self._matches_ability_target(unit, target_filter)
                ]
                return candidates[0] if candidates else None
            if target_name:
                units = self._select_exact_units(target_name)
                compatible = [
                    unit
                    for unit in units
                    if self._matches_ability_target(unit, target_filter)
                ]
                return self._first_unit(compatible) if compatible else None
            unit_target = getattr(action, "unit", None)
            if unit_target and alliance in {"friendly", "any"}:
                units = self._select_exact_units(
                    normalize_name(str(unit_target)), action
                )
                return self._first_unit(units) if units else None
            return friendly[0] if friendly else None

        def _matches_named_target(
            self, candidate: Any, action: Any, alliance: str
        ) -> bool:
            explicit = (
                getattr(action, "target_unit", None)
                or getattr(action, "target", None)
                or getattr(action, "passenger", None)
            )
            if explicit is None or not isinstance(explicit, str):
                return True
            target_name = normalize_name(explicit)
            if not target_name:
                return True

            if target_name in {
                "nearest_enemy",
                "lowest_health_enemy",
                "highest_energy_enemy",
            }:
                return alliance == "enemy"
            if target_name == "nearest_destructible":
                return alliance == "neutral"
            if target_name == "nearest_enemy_structure":
                return alliance == "enemy" and bool(
                    getattr(candidate, "is_structure", False)
                )
            enemy_predicates = {
                "nearest_enemy_ground": lambda unit: not bool(
                    getattr(unit, "is_flying", False)
                ),
                "nearest_enemy_air": lambda unit: bool(
                    getattr(unit, "is_flying", False)
                ),
                "nearest_enemy_biological": lambda unit: bool(
                    getattr(unit, "is_biological", False)
                ),
                "nearest_enemy_mechanical": lambda unit: bool(
                    getattr(unit, "is_mechanical", False)
                ),
                "nearest_enemy_massive": lambda unit: bool(
                    getattr(unit, "is_massive", False)
                ),
                "nearest_enemy_detector": lambda unit: bool(
                    getattr(unit, "is_detector", False)
                ),
                "nearest_enemy_cloaked": lambda unit: bool(
                    getattr(unit, "is_cloaked", False)
                    or getattr(unit, "is_burrowed", False)
                ),
            }
            enemy_predicate = enemy_predicates.get(target_name)
            if enemy_predicate is not None:
                return alliance == "enemy" and enemy_predicate(candidate)

            if target_name in {
                "nearest_friendly",
                "any_friendly",
                "highest_energy_friendly",
            }:
                return alliance in {"friendly", "any", "passenger"}
            if target_name in {"damaged_friendly", "lowest_health_friendly"}:
                return alliance in {
                    "friendly",
                    "any",
                    "passenger",
                } and self._is_damaged(candidate)
            return _unit_type_name(candidate) == target_name

        def _enemy_target_candidates(
            self, selector: str, ability_key: str
        ) -> list[Any]:
            if selector == "nearest_destructible":
                return []
            if selector == "nearest_enemy_structure":
                candidates = list(getattr(self, "enemy_structures", []))
            else:
                candidates = list(getattr(self, "enemy_units", [])) + list(
                    getattr(self, "enemy_structures", [])
                )

            predicates = {
                "nearest_enemy_ground": lambda unit: not bool(
                    getattr(unit, "is_flying", False)
                ),
                "nearest_enemy_air": lambda unit: bool(
                    getattr(unit, "is_flying", False)
                ),
                "nearest_enemy_biological": lambda unit: bool(
                    getattr(unit, "is_biological", False)
                ),
                "nearest_enemy_mechanical": lambda unit: bool(
                    getattr(unit, "is_mechanical", False)
                ),
                "nearest_enemy_massive": lambda unit: bool(
                    getattr(unit, "is_massive", False)
                ),
                "nearest_enemy_detector": lambda unit: bool(
                    getattr(unit, "is_detector", False)
                ),
                "nearest_enemy_cloaked": lambda unit: bool(
                    getattr(unit, "is_cloaked", False)
                    or getattr(unit, "is_burrowed", False)
                ),
            }
            predicate = predicates.get(selector)
            if predicate is not None:
                candidates = [unit for unit in candidates if predicate(unit)]
            elif selector and selector not in TARGET_SELECTORS:
                candidates = [
                    unit for unit in candidates if _unit_type_name(unit) == selector
                ]

            spec = _RUNTIME_ABILITY_SPECS.get(ability_key)
            target_filter = spec.target_filter if spec is not None else "any"
            candidates = [
                unit
                for unit in candidates
                if self._matches_ability_target(unit, target_filter)
            ]

            if selector == "lowest_health_enemy":
                candidates.sort(key=self._health_ratio)
            elif selector == "highest_energy_enemy":
                candidates.sort(key=lambda unit: -float(getattr(unit, "energy", 0.0)))
            else:
                reference = self._build_near_point()
                candidates.sort(
                    key=lambda unit: self._distance_squared(unit, reference)
                )
            return candidates

        def _friendly_target_candidates(self, selector: str) -> list[Any]:
            combined = (
                list(getattr(self, "units", []))
                + list(getattr(self, "workers", []))
                + list(getattr(self, "structures", []))
            )
            candidates = list(
                {
                    str(getattr(unit, "tag", id(unit))): unit for unit in combined
                }.values()
            )
            if selector in {"damaged_friendly", "lowest_health_friendly"}:
                candidates = [unit for unit in candidates if self._is_damaged(unit)]
                candidates.sort(key=self._health_ratio)
            elif selector == "highest_energy_friendly":
                candidates.sort(key=lambda unit: -float(getattr(unit, "energy", 0.0)))
            else:
                reference = self._build_near_point()
                candidates.sort(
                    key=lambda unit: self._distance_squared(unit, reference)
                )
            return candidates

        def _resolve_passenger_target(self, action: Any, sources=None):
            passenger_tag = getattr(action, "passenger_tag", None) or getattr(
                action, "target_tag", None
            )
            raw_target_name = getattr(action, "target_unit", None)
            target_name = (
                normalize_name(str(raw_target_name)) if raw_target_name else ""
            )
            transport_sources = (
                list(sources)
                if sources is not None
                else list(getattr(self, "units", []))
                + list(getattr(self, "workers", []))
                + list(getattr(self, "structures", []))
            )
            for unit in transport_sources:
                cargo = (
                    getattr(unit, "passengers", None)
                    or getattr(unit, "cargo", None)
                    or []
                )
                for passenger in cargo:
                    if passenger_tag is not None and str(
                        getattr(passenger, "tag", "")
                    ) != str(passenger_tag):
                        continue
                    if target_name and _unit_type_name(passenger) != target_name:
                        continue
                    return passenger
            return None

        def _ability_id_for_enum(self, enum_name: str):
            ability_id = self._ability_id()
            return getattr(ability_id, enum_name, enum_name)

        async def _unit_has_available_ability(self, unit, ability_id) -> bool:
            if hasattr(self, "query_available_abilities"):
                available = self.query_available_abilities(unit)
            elif hasattr(self, "get_available_abilities"):
                available = self.get_available_abilities([unit])
            else:
                return True
            if hasattr(available, "__await__"):
                available = await available
            if (
                available
                and isinstance(available, list)
                and len(available) == 1
                and isinstance(available[0], (list, tuple, set))
            ):
                available = list(available[0])
            (
                requested_name,
                requested_generic,
                requested_is_generic,
            ) = self._ability_identity(ability_id)
            for item in available:
                (
                    available_name,
                    available_generic,
                    available_is_generic,
                ) = self._ability_identity(item)
                if available_name == requested_name:
                    return True
                if available_is_generic and available_name == requested_generic:
                    return True
                if requested_is_generic and requested_name == available_generic:
                    return True
            return False

        @staticmethod
        def _ability_identity(ability_id: Any) -> tuple[str, str, bool]:
            name = str(getattr(ability_id, "name", ability_id))
            try:
                from sc2.dicts.generic_redirect_abilities import (
                    GENERIC_REDIRECT_ABILITIES,
                )
                from sc2.ids.ability_id import AbilityId

                exact_id = getattr(AbilityId, name, ability_id)
                generic_id = GENERIC_REDIRECT_ABILITIES.get(exact_id, exact_id)
                generic_name = str(getattr(generic_id, "name", generic_id))
                return name, generic_name, name == generic_name
            except (ImportError, TypeError):
                return name, name, True

        @classmethod
        def _generic_ability_name(cls, ability_id: Any) -> str:
            return cls._ability_identity(ability_id)[1]

        @staticmethod
        def _issue_order(unit, method_name: str, target, queued: bool) -> None:
            method = getattr(unit, method_name)
            try:
                if target is None:
                    method(queue=queued)
                else:
                    method(target, queue=queued)
            except TypeError:
                if target is None:
                    method()
                else:
                    method(target)

        @staticmethod
        def _issue_ability(unit, ability_id, target, queued: bool = False) -> None:
            try:
                if target is None:
                    unit(ability_id, queue=queued)
                else:
                    unit(ability_id, target, queue=queued)
            except TypeError:
                if target is None:
                    unit(ability_id)
                else:
                    unit(ability_id, target)

        async def _retry_or_replan(
            self, command: Any, iteration: int, reason: str
        ) -> None:
            started = self._action_context.get("failure_started_at_game_time")
            now = float(getattr(self, "time", 0.0))
            if started is None:
                self._action_context["failure_started_at_game_time"] = now
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: {reason}; "
                    f"retrying for {ABILITY_RETRY_SECONDS:g} in-game second(s)"
                )
                return
            elapsed = now - float(started)
            if elapsed < ABILITY_RETRY_SECONDS:
                if iteration % 22 == 0:
                    print(
                        f"Still waiting: {reason} ({elapsed:g}/{ABILITY_RETRY_SECONDS:g}s)"
                    )
                return
            await self._replan_or_leave(reason)

        async def _replan_or_leave(self, reason: str) -> None:
            if self.original_strategy and self._replan_count < self.replan_limit:
                self._replan_count += 1
                self.observed_summary = summarize_bot_state(self)
                print(
                    f"Replanning after runtime failure ({self._replan_count}/{self.replan_limit}): {reason}"
                )
                try:
                    if hasattr(self, "create_replan"):
                        new_plan = self.create_replan(
                            self.original_strategy, self.observed_summary
                        )
                        if hasattr(new_plan, "__await__"):
                            new_plan = await new_plan
                    else:
                        new_plan = plan_strategy(
                            self.original_strategy,
                            planner_name=self.planner_name,
                            game_state=self.observed_summary,
                        )
                    self.plan = validate_strategy_plan(
                        new_plan,
                        game_state=self.observed_summary,
                        max_actions=MAX_PLAN_ACTIONS,
                    )
                except (
                    PlanValidationError,
                    PlannerError,
                    PlannerUnavailableError,
                    ValueError,
                ) as exc:
                    print(f"Replan failed terminally: {exc}", file=sys.stderr)
                    self._left_game = True
                    await self.client.leave()
                    return
                self._current_action_index = 0
                self._action_started_at_loop_time = None
                self._action_context = {}
                self._production_policies = []
                self._control_flow_states = {}
                self._timeout_scope_states = {}
                self._plan_finished_at_loop_time = None
                self._print_plan_loaded()
                return
            print(f"Runtime action failed terminally: {reason}", file=sys.stderr)
            self._left_game = True
            await self.client.leave()

        def _train_unit_type(self, unit: str):
            if unit not in UNIT_SPECS:
                raise TypeError(f"unsupported train unit: {unit}")
            return getattr(self._unit_type_id(), UNIT_SPECS[unit].enum_name)

        def _building_unit_type(self, building: str):
            if building not in STRUCTURE_SPECS:
                raise TypeError(f"unsupported build structure: {building}")
            return getattr(self._unit_type_id(), STRUCTURE_SPECS[building].enum_name)

        def _structure_unit_type(self, building: str):
            for registry in (STRUCTURE_SPECS, ADDON_SPECS, MORPH_SPECS):
                if building in registry:
                    return getattr(self._unit_type_id(), registry[building].enum_name)
            raise TypeError(f"unsupported structure: {building}")

        def _addon_unit_type(self, addon: str):
            return getattr(self._unit_type_id(), ADDON_SPECS[addon].enum_name)

        def _morph_unit_type(self, building: str):
            return getattr(self._unit_type_id(), MORPH_SPECS[building].enum_name)

        def _upgrade_id(self, upgrade: str):
            if upgrade not in UPGRADE_SPECS:
                raise TypeError(f"unsupported Terran upgrade: {upgrade}")
            return getattr(self._upgrade_id_class(), UPGRADE_SPECS[upgrade].enum_name)

        def _repair_targets(self, target: str):
            if target == "worker":
                return self.workers
            return self._select_exact_units(target)

        @staticmethod
        def _is_damaged(unit) -> bool:
            health_percentage = getattr(unit, "health_percentage", None)
            if health_percentage is not None:
                return float(health_percentage) < 1.0
            health = getattr(unit, "health", None)
            health_max = getattr(unit, "health_max", None)
            if health is not None and health_max:
                return float(health) < float(health_max)
            return False

        @staticmethod
        def _matches_ability_target(unit, target_filter: str) -> bool:
            if target_filter == "any":
                return True
            key = _unit_type_name(unit)
            structure_keys = {
                *STRUCTURE_SPECS,
                *ADDON_SPECS,
                *MORPH_SPECS,
            }
            is_structure_value = getattr(unit, "is_structure", None)
            is_structure = (
                bool(is_structure_value)
                if is_structure_value is not None
                else key in structure_keys
            )
            biological_value = getattr(unit, "is_biological", None)
            is_biological = (
                bool(biological_value)
                if biological_value is not None
                else key in BIOLOGICAL_UNIT_KEYS
            )
            mechanical_value = getattr(unit, "is_mechanical", None)
            is_mechanical = (
                bool(mechanical_value)
                if mechanical_value is not None
                else key in MECHANICAL_UNIT_KEYS or is_structure
            )
            psionic_value = getattr(unit, "is_psionic", None)
            is_psionic = (
                bool(psionic_value)
                if psionic_value is not None
                else key in PSIONIC_UNIT_KEYS
            )

            if target_filter == "biological_unit":
                return is_biological and not is_structure
            if target_filter == "mechanical_unit":
                return is_mechanical and not is_structure
            if target_filter == "mechanical_or_psionic_unit":
                return (is_mechanical or is_psionic) and not is_structure
            if target_filter == "mechanical":
                return is_mechanical
            if target_filter == "supply_depot":
                return key == "supply_depot"
            if target_filter == "worker":
                return key == "worker"
            if target_filter == "worker_passenger":
                return key == "worker"
            if target_filter == "bunker_loadable":
                return key in BUNKER_LOADABLE_UNIT_KEYS
            if target_filter == "medivac_loadable":
                return key in MEDIVAC_LOADABLE_UNIT_KEYS
            return False

        @staticmethod
        def _is_flying(unit) -> bool:
            is_flying = getattr(unit, "is_flying", None)
            if is_flying is not None:
                return bool(is_flying)
            type_id = getattr(unit, "type_id", "")
            enum_name = getattr(type_id, "name", str(type_id))
            return normalize_name(enum_name).endswith("flying")

        @staticmethod
        def _raw_unit_type_name(unit) -> str:
            type_id = getattr(unit, "type_id", "")
            return normalize_name(getattr(type_id, "name", str(type_id)))

        def _rally_ability(self, building: str):
            ability_id = self._ability_id()
            if building in {"command_center", "orbital_command", "planetary_fortress"}:
                return ability_id.RALLY_COMMANDCENTER
            return ability_id.RALLY_BUILDING

        @staticmethod
        def _unit_type_id():
            try:
                from sc2.ids.unit_typeid import UnitTypeId

                return UnitTypeId
            except ImportError:

                class _FallbackUnitTypeId:
                    pass

                for spec in (
                    *UNIT_SPECS.values(),
                    *STRUCTURE_SPECS.values(),
                    *ADDON_SPECS.values(),
                    *MORPH_SPECS.values(),
                ):
                    setattr(_FallbackUnitTypeId, spec.enum_name, spec.enum_name)
                for enum_names in RUNTIME_ACTOR_UNIT_TYPES.values():
                    for enum_name in enum_names:
                        setattr(_FallbackUnitTypeId, enum_name, enum_name)

                return _FallbackUnitTypeId

        @staticmethod
        def _upgrade_id_class():
            try:
                from sc2.ids.upgrade_id import UpgradeId

                return UpgradeId
            except ImportError:

                class _FallbackUpgradeId:
                    pass

                for spec in UPGRADE_SPECS.values():
                    setattr(_FallbackUpgradeId, spec.enum_name, spec.enum_name)
                return _FallbackUpgradeId

        @staticmethod
        def _ability_id():
            try:
                from sc2.ids.ability_id import AbilityId

                return AbilityId
            except ImportError:

                class _FallbackAbilityId:
                    RALLY_COMMANDCENTER = "RALLY_COMMANDCENTER"
                    RALLY_BUILDING = "RALLY_BUILDING"

                for spec in _RUNTIME_ABILITY_SPECS.values():
                    setattr(_FallbackAbilityId, spec.enum_name, spec.enum_name)

                return _FallbackAbilityId

        def _should_stop(self) -> bool:
            if self._plan_finished_at_loop_time is None:
                return False
            elapsed = (
                asyncio.get_running_loop().time() - self._plan_finished_at_loop_time
            )
            return elapsed >= self.stop_after_seconds

        @staticmethod
        def _describe_action(action) -> str:
            if isinstance(action, MoveCommand):
                if action.target_unit is not None or action.target_tag is not None:
                    return (
                        f"move {action.unit} toward friendly target "
                        f"{action.target_unit or action.target_tag}"
                    )
                verb = "move and wait" if action.wait_for_arrival else "move"
                return (
                    f"{verb} {action.unit} to {_MoveUnitBot._describe_location(action)}"
                )
            if isinstance(action, AttackMoveCommand):
                return f"attack with {action.unit} toward {_MoveUnitBot._describe_location(action)}"
            if isinstance(action, AttackEnemyCommand):
                verb = "focus fire" if action.wait_for_target_death else "attack"
                return f"{verb} visible enemy with {action.unit}"
            if isinstance(action, AttackUntilClearCommand):
                return (
                    f"attack with {action.unit} until "
                    f"{_MoveUnitBot._describe_location(action)} is clear"
                )
            if isinstance(action, KiteCommand):
                return (
                    f"kite {action.target_unit} with {action.unit} for "
                    f"{action.duration_seconds:g}s"
                )
            if isinstance(action, PatrolCommand):
                return f"patrol {action.unit} toward {_MoveUnitBot._describe_location(action)}"
            if isinstance(action, HoldPositionCommand):
                return f"hold position with {action.unit}"
            if isinstance(action, StopCommand):
                return f"stop {action.unit}"
            if isinstance(action, RallyCommand):
                return f"rally {action.building} to {_MoveUnitBot._describe_location(action)}"
            if isinstance(action, WaitCommand):
                return f"wait {action.seconds:g} second(s)"
            if isinstance(action, WaitUntilCommand):
                target = f" {action.target}" if action.target else ""
                comparison = {
                    "gte": ">=",
                    "lte": "<=",
                    "eq": "==",
                    "neq": "!=",
                    "gt": ">",
                    "lt": "<",
                }.get(action.comparison, action.comparison)
                qualifier = action.ability or action.actor or ""
                if qualifier and not target:
                    target = f" {qualifier}"
                return (
                    f"{action.condition}{target} {comparison} {action.at_least:g}"
                )
            if isinstance(action, ConditionalCommand):
                return (
                    f"conditional ({len(action.then_actions)} then / "
                    f"{len(action.else_actions)} else actions)"
                )
            if isinstance(action, RepeatCommand):
                verb = "repeat until condition" if action.until is not None else "repeat"
                return f"{verb} up to {action.max_cycles} cycle(s)"
            if isinstance(action, WithTimeoutCommand):
                return (
                    f"run {len(action.actions)} action(s) with "
                    f"{action.timeout_seconds:g}s timeout"
                )
            if isinstance(action, GatherMineralsCommand):
                return f"gather minerals with {action.unit}"
            if isinstance(action, GatherGasCommand):
                return f"gather gas with {action.unit}"
            if isinstance(action, DistributeWorkersCommand):
                return f"distribute workers at ratio {action.mineral_to_gas_ratio:g}"
            if isinstance(action, TrainUnitCommand):
                return (
                    f"train {action.unit}"
                    if action.count == 1
                    else f"train {action.count} {action.unit}"
                )
            if isinstance(action, ProductionPolicyCommand):
                verb = "maintain production" if action.background else "produce"
                return f"{verb} {action.unit} until {action.target_count}"
            if isinstance(action, StopProductionCommand):
                return f"stop production policy {action.unit or 'all'}"
            if isinstance(action, BuildStructureCommand):
                return (
                    f"build {action.building}"
                    if action.count == 1
                    else f"build {action.count} {action.building}"
                )
            if isinstance(action, ExpandCommand):
                return "expand" if action.count == 1 else f"expand {action.count} times"
            if isinstance(action, BuildAddonCommand):
                return (
                    f"build {action.addon}"
                    if action.count == 1
                    else f"build {action.count} {action.addon}"
                )
            if isinstance(action, MorphStructureCommand):
                return f"morph {action.building}"
            if isinstance(action, ResearchUpgradeCommand):
                return f"research {action.upgrade}"
            if isinstance(action, RepairCommand):
                return f"repair {action.target} with {action.workers} worker(s)"
            kind = _MoveUnitBot._action_kind(action)
            if kind == "replan":
                return "replan from current game state"
            ability = getattr(action, "ability", None) or getattr(
                action, "ability_key", None
            )
            if ability or kind in {
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
            }:
                return f"use ability {ability or kind}"
            return repr(action)

        @staticmethod
        def _describe_location(action) -> str:
            location = getattr(action, "location", None)
            semantic = (
                getattr(location, "semantic", None) if location is not None else None
            )
            if semantic:
                return str(semantic)
            x = getattr(action, "x", None)
            y = getattr(action, "y", None)
            if location is not None:
                x = getattr(location, "x", x)
                y = getattr(location, "y", y)
            if x is not None and y is not None:
                return f"({float(x):g}, {float(y):g})"
            return "<unresolved>"

    return _MoveUnitBot


def _import_sc2_runtime():
    try:
        from sc2 import maps
        from sc2.bot_ai import BotAI
        from sc2.data import Difficulty, Race
        from sc2.main import run_game
        from sc2.player import Bot, Computer
        from sc2.position import Point2
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: python3 -m pip install -r requirements.txt"
        ) from exc

    return maps, BotAI, Difficulty, Race, run_game, Bot, Computer, Point2


def print_game_state(map_name: str, realtime: bool) -> None:
    """Start SC2, capture the initial bot observation, and print it as JSON."""

    (
        maps,
        BotAI,
        Difficulty,
        Race,
        run_game,
        Bot,
        Computer,
        _Point2,
    ) = _import_sc2_runtime()
    bot_class = create_game_state_bot_class(BotAI)
    bot = bot_class()
    try:
        selected_map = maps.get(map_name)
    except (FileNotFoundError, KeyError) as exc:
        env = detect_sc2_environment()
        raise SystemExit(_map_error_message(map_name, env)) from exc

    sc2_logs_disabled = False
    sc2_logger: Any | None = None
    try:
        from loguru import logger as sc2_logger

        assert sc2_logger is not None
        sc2_logger.disable("sc2")
        sc2_logs_disabled = True
    except ImportError:
        pass

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            run_game(
                selected_map,
                [
                    Bot(Race.Terran, bot),
                    Computer(Race.Zerg, Difficulty.VeryEasy),
                ],
                realtime=realtime,
            )
    except TimeoutError as exc:
        raise SystemExit(_api_timeout_error_message()) from exc
    finally:
        if sc2_logs_disabled and sc2_logger is not None:
            sc2_logger.enable("sc2")

    if bot.summary is None:
        raise SystemExit(
            "Failed to capture StarCraft II game state before the game ended."
        )

    print(game_state_summary_to_json(bot.summary))


def run_real_game(
    strategy: str,
    map_name: str,
    realtime: bool,
    stop_after_seconds: int,
    planner_name: str = DEFAULT_PLANNER,
    observe_before_plan: bool = False,
) -> None:
    """Start StarCraft II and run the minimal movement bot."""

    plan = None
    if not observe_before_plan:
        plan = validate_strategy_plan(
            plan_strategy(strategy, planner_name=planner_name),
            max_actions=MAX_PLAN_ACTIONS,
        )
    (
        maps,
        BotAI,
        Difficulty,
        Race,
        run_game,
        Bot,
        Computer,
        Point2,
    ) = _import_sc2_runtime()
    bot_class = create_move_unit_bot_class(BotAI, Point2)
    try:
        selected_map = maps.get(map_name)
    except (FileNotFoundError, KeyError) as exc:
        env = detect_sc2_environment()
        raise SystemExit(_map_error_message(map_name, env)) from exc

    try:
        run_game(
            selected_map,
            [
                Bot(
                    Race.Terran,
                    bot_class(
                        plan,
                        stop_after_seconds=stop_after_seconds,
                        strategy=strategy,
                        planner_name=planner_name,
                        observe_before_plan=observe_before_plan,
                    ),
                ),
                Computer(Race.Zerg, Difficulty.VeryEasy),
            ],
            realtime=realtime,
        )
    except TimeoutError as exc:
        raise SystemExit(_api_timeout_error_message()) from exc


def _api_timeout_error_message() -> str:
    return "\n".join(
        [
            "StarCraft II launched, but its local SC2 API websocket did not open before timeout.",
            "Open StarCraft II once from Battle.net, finish any first-run setup/login/update prompts, then quit and rerun this script.",
            "Also allow StarCraft II through any macOS firewall prompt for local connections.",
        ]
    )


def _map_error_message(map_name: str, env: Sc2Environment) -> str:
    maps_path = env.maps_path or Path("<SC2 install>") / "Maps"
    return "\n".join(
        [
            f"StarCraft II map '{map_name}' was not found.",
            f"Expected local API maps under: {maps_path}",
            "Install/extract a Blizzard SC2 map pack into that Maps folder, then rerun.",
            "For the default map, use the Ladder 2017 Season 1 map pack or pass another installed map with --map.",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a minimal real StarCraft II movement bot."
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Strategy command to execute. Default: {DEFAULT_STRATEGY!r}",
    )
    parser.add_argument(
        "--map", default=DEFAULT_MAP, help=f"SC2 map name. Default: {DEFAULT_MAP!r}"
    )
    parser.add_argument(
        "--planner",
        default=DEFAULT_PLANNER,
        choices=PLANNER_MODES,
        help=f"Planner mode. Default: {DEFAULT_PLANNER!r}. Other modes must be selected explicitly.",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=35,
        help="Seconds to keep the game open after issuing move.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run non-realtime for faster automated checks.",
    )
    parser.add_argument(
        "--observe-before-plan",
        action="store_true",
        help="Start SC2, summarize the initial game state, pass it to the planner, validate the plan, then execute it.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check local SC2 installation hints; do not start the game.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Parse --strategy as DSL, JSON, or known intent and print canonical StrategyPlan JSON without starting SC2.",
    )
    parser.add_argument(
        "--print-state",
        action="store_true",
        help="Start SC2, print the initial game-state summary JSON, and exit without executing a strategy.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    env = detect_sc2_environment()

    if args.check:
        if not env.installed:
            print("StarCraft II was not detected automatically.")
            print(
                "Install StarCraft II with the Blizzard/Battle.net app or set SC2PATH to the install directory."
            )
            print("Checked paths:")
            for path in env.candidate_paths:
                print(f"- {path}")
            return 1

        print(f"StarCraft II path detected: {env.detected_path}")
        if env.maps_installed:
            print(f"SC2 API maps directory detected: {env.maps_path}")
            return 0

        print(f"SC2 API maps directory missing: {env.maps_path}")
        print(
            "Install/extract a Blizzard SC2 map pack into the Maps folder before launching a game."
        )
        print(
            "The default map needs the Ladder 2017 Season 1 map pack, or pass another installed map with --map."
        )
        return 1

    if args.print_plan:
        try:
            plan = plan_strategy(args.strategy, planner_name=args.planner)
        except (
            PlanValidationError,
            PlannerError,
            PlannerUnavailableError,
            ValueError,
        ) as exc:
            print(f"Planner error: {exc}", file=sys.stderr)
            return 2
        print(strategy_plan_to_json(plan))
        return 0

    if args.print_state:
        print_game_state(map_name=args.map, realtime=not args.fast)
        return 0

    if not env.installed:
        print(
            "Warning: StarCraft II was not detected before launch; python-sc2 may still find it if configured."
        )
    elif not env.maps_installed:
        print(
            f"Warning: SC2 API maps directory was not detected at {env.maps_path}; launch may fail."
        )

    try:
        run_real_game(
            strategy=args.strategy,
            map_name=args.map,
            realtime=not args.fast,
            stop_after_seconds=args.stop_after,
            planner_name=args.planner,
            observe_before_plan=args.observe_before_plan,
        )
    except (
        PlanValidationError,
        PlannerError,
        PlannerUnavailableError,
        ValueError,
    ) as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
