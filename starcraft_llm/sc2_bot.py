from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from starcraft_llm.command_catalog import (
    ADDON_SPECS,
    MORPH_SPECS,
    STRUCTURE_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    normalize_name,
    resolve_alias,
)
from starcraft_llm.game_state import (
    GameStateSummary,
    SupplySummary,
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
    strategy_plan_to_json,
)
from starcraft_llm.validator import PlanValidationError, validate_strategy_plan

DEFAULT_MAP = "AbyssalReefLE"
DEFAULT_STRATEGY = "move worker 35 42"


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
    class _GameStateBot(bot_ai_base):
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
    )


def _unit_type_name(unit) -> str:
    raw_type = getattr(unit, "type_id", "unknown")
    name = getattr(raw_type, "name", str(raw_type))
    return name.lower()


def _structure_is_ready(structure) -> bool:
    is_ready = getattr(structure, "is_ready", None)
    if is_ready is not None:
        return bool(is_ready)
    build_progress = getattr(structure, "build_progress", None)
    if build_progress is not None:
        return float(build_progress) >= 1.0
    return True


def create_move_unit_bot_class(bot_ai_base, point2_class):
    class _MoveUnitBot(bot_ai_base):
        def __init__(
            self,
            plan: StrategyPlan | None = None,
            stop_after_seconds: int = 35,
            strategy: str | None = None,
            planner_name: str = DEFAULT_PLANNER,
            observe_before_plan: bool = False,
        ):
            super().__init__()
            self.plan = plan
            self.stop_after_seconds = stop_after_seconds
            self.strategy = strategy
            self.planner_name = planner_name
            self.observe_before_plan = observe_before_plan
            self.observed_summary: GameStateSummary | None = None
            self._current_action_index = 0
            self._action_started_at_loop_time: float | None = None
            self._action_context: dict[str, Any] = {}
            self._plan_finished_at_loop_time: float | None = None
            self._left_game = False

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
                    except (PlanValidationError, PlannerError, PlannerUnavailableError, ValueError) as exc:
                        print(f"Planner error: {exc}", file=sys.stderr)
                        self._left_game = True
                        await self.client.leave()
                        return
                await self._execute_current_action(iteration)

            if not self._left_game and self._should_stop():
                print("MVP complete: strategy plan finished; leaving the game.")
                self._left_game = True
                await self.client.leave()

        def _create_plan_from_observation(self) -> bool:
            if not self.observe_before_plan:
                raise RuntimeError("strategy plan is not loaded and observe-before-plan is disabled")
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
                self._mark_plan_finished()
                return

            action = self.plan.actions[self._current_action_index]
            if isinstance(action, MoveCommand):
                self._execute_move(action, iteration)
                return
            if isinstance(action, AttackMoveCommand):
                self._execute_attack(action, iteration)
                return
            if isinstance(action, AttackEnemyCommand):
                self._execute_attack_enemy(action, iteration)
                return
            if isinstance(action, PatrolCommand):
                self._execute_patrol(action, iteration)
                return
            if isinstance(action, HoldPositionCommand):
                self._execute_unit_order(action, iteration, "hold_position")
                return
            if isinstance(action, StopCommand):
                self._execute_unit_order(action, iteration, "stop")
                return
            if isinstance(action, RallyCommand):
                self._execute_rally(action, iteration)
                return
            if isinstance(action, WaitCommand):
                self._execute_wait(action)
                return
            if isinstance(action, WaitUntilCommand):
                self._execute_wait_until(action, iteration)
                return
            if isinstance(action, GatherMineralsCommand):
                self._execute_gather_minerals(action, iteration)
                return
            if isinstance(action, GatherGasCommand):
                self._execute_gather_gas(action, iteration)
                return
            if isinstance(action, DistributeWorkersCommand):
                await self._execute_distribute_workers(action)
                return
            if isinstance(action, TrainUnitCommand):
                self._execute_train(action, iteration)
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

            raise TypeError(f"unsupported strategy action: {action!r}")

        def _execute_move(self, command: MoveCommand, iteration: int) -> None:
            target = point2_class((command.x, command.y))
            units = self._select_units(command.unit)
            if units:
                for unit in units:
                    unit.move(target)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued move command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units...")

        def _execute_attack(self, command: AttackMoveCommand, iteration: int) -> None:
            target = point2_class((command.x, command.y))
            units = self._select_units(command.unit)
            if units:
                for unit in units:
                    unit.attack(target)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued attack command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units before attacking...")

        def _execute_attack_enemy(self, command: AttackEnemyCommand, iteration: int) -> None:
            units = self._select_exact_units(command.unit)
            enemies = self.enemy_units
            if units and enemies:
                target = self._closest_enemy(enemies, self._first_unit(units))
                for unit in units:
                    unit.attack(target)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued attack command to {len(units)} {command.unit} unit(s) against visible enemy"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units and visible enemies before attacking...")

        def _execute_patrol(self, command: PatrolCommand, iteration: int) -> None:
            units = self._select_exact_units(command.unit)
            if units:
                target = point2_class((command.x, command.y))
                for unit in units:
                    unit.patrol(target)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued patrol command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units before patrolling...")

        def _execute_unit_order(self, command, iteration: int, method_name: str) -> None:
            units = self._select_exact_units(command.unit)
            if units:
                for unit in units:
                    getattr(unit, method_name)()
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued {method_name.replace('_', ' ')} to {len(units)} {command.unit} unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units before {method_name.replace('_', ' ')}...")

        def _execute_rally(self, command: RallyCommand, iteration: int) -> None:
            structures = self._ready_idle_structures(command.building)
            if structures:
                target = point2_class((command.x, command.y))
                ability = self._rally_ability(command.building)
                for structure in structures:
                    structure(ability, target)
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"set {len(structures)} {command.building} rally point(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for a ready {command.building} before setting rally point...")

        def _execute_wait(self, command: WaitCommand) -> None:
            now = asyncio.get_running_loop().time()
            if self._action_started_at_loop_time is None:
                self._action_started_at_loop_time = now
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"waiting {command.seconds:g} second(s)"
                )

            elapsed = now - self._action_started_at_loop_time
            if elapsed >= command.seconds:
                self._advance_action()

        def _execute_wait_until(self, command: WaitUntilCommand, iteration: int) -> None:
            current = self._wait_until_observed_value(command)
            if current >= command.at_least:
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"condition met: {self._describe_action(command)} (current={current:g})"
                )
                self._advance_action()
                return

            if self._action_started_at_loop_time is None:
                self._action_started_at_loop_time = asyncio.get_running_loop().time()
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"waiting until {self._describe_action(command)} (current={current:g})"
                )
            elif iteration % 22 == 0:
                print(f"Still waiting for {self._describe_action(command)} (current={current:g})")

        def _execute_gather_minerals(self, command: GatherMineralsCommand, iteration: int) -> None:
            workers = self._select_units(command.unit)
            mineral_fields = self.mineral_field
            if workers and mineral_fields:
                issued = 0
                selected_workers = workers[: command.workers] if command.workers is not None else workers
                for worker in selected_workers:
                    mineral_field = self._closest_mineral_field(mineral_fields, worker)
                    worker.gather(mineral_field)
                    issued += 1
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued gather minerals command to {issued} worker unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print("Waiting for workers and mineral fields before gathering...")

        def _execute_gather_gas(self, command: GatherGasCommand, iteration: int) -> None:
            workers = self._select_units(command.unit)
            refineries = self._ready_refineries()
            if workers and refineries:
                issued = 0
                requested_workers = command.workers if command.workers is not None else len(refineries) * 3
                max_workers = min(len(workers), len(refineries) * 3, requested_workers)
                for index, worker in enumerate(workers[:max_workers]):
                    refinery = refineries[index % len(refineries)]
                    worker.gather(refinery)
                    issued += 1
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued gather gas command to {issued} worker unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print("Waiting for workers and ready refineries before gathering gas...")

        async def _execute_distribute_workers(self, command: DistributeWorkersCommand) -> None:
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
            producers = self._available_producers(command.unit)
            if not producers:
                if iteration % 22 == 0:
                    print(f"Waiting for an available producer to train {command.unit}...")
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

        async def _execute_build(self, command: BuildStructureCommand, iteration: int) -> None:
            if command.building not in STRUCTURE_SPECS:
                raise TypeError(f"unsupported build structure: {command.building}")
            if command.building == "command_center":
                await self._execute_expand(ExpandCommand(count=command.count), iteration)
                return

            unit_type = self._building_unit_type(command.building)
            if not self._action_context:
                self._action_context = {
                    "building": command.building,
                    "total_before": self._structure_count(command.building, readiness="total"),
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
                    print(f"Waiting for {command.building} construction to start ({started_count}/{issued_count})...")
                return

            if hasattr(self, "can_afford") and not self.can_afford(unit_type):
                if iteration % 22 == 0:
                    print(f"Waiting for enough resources to build {command.building}...")
                return

            if command.building == "refinery":
                issued = self._execute_refinery_build(unit_type)
            else:
                near = self._build_near_point()
                issued = await self.build(unit_type, near=near, max_distance=20)

            if issued:
                issued_count += 1
                self._action_context["issued"] = issued_count
                print(
                    f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                    f"issued build {command.building} command ({issued_count}/{command.count})"
                )
                if self._builds_started_since_action_start(command.building) >= command.count:
                    print(
                        f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                        f"{command.count} build {command.building} command(s) started"
                    )
                    self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for placement/worker to build {command.building}...")

        async def _execute_expand(self, command: ExpandCommand, iteration: int) -> None:
            if not self._action_context:
                self._action_context = {"townhalls_before": len(self.townhalls), "issued": 0}
            issued_count = int(self._action_context.get("issued", 0))
            started_count = max(0, len(self.townhalls) - int(self._action_context.get("townhalls_before", 0)))
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
                    print(f"Waiting for expansion construction to start ({started_count}/{issued_count})...")
                return

            command_center_type = self._building_unit_type("command_center")
            if hasattr(self, "can_afford") and not self.can_afford(command_center_type):
                if iteration % 22 == 0:
                    print("Waiting for enough resources to expand...")
                return
            issued = await self._issue_expansion(command_center_type)
            if not issued:
                if iteration % 22 == 0:
                    print("Waiting for an available expansion location and build worker...")
                return
            issued_count += 1
            self._action_context["issued"] = issued_count
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued expansion command ({issued_count}/{command.count})"
            )
            if len(self.townhalls) - int(self._action_context.get("townhalls_before", 0)) >= command.count:
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
            producers = self._free_addon_producers(spec.producer or "")
            used_tags = self._action_context.get("producer_tags", set())
            producer = next((item for item in producers if getattr(item, "tag", id(item)) not in used_tags), None)
            if producer is None:
                if iteration % 22 == 0:
                    print(f"Waiting for a free {spec.producer} to build {command.addon}...")
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

        def _execute_morph(self, command: MorphStructureCommand, iteration: int) -> None:
            if command.building not in MORPH_SPECS:
                raise TypeError(f"unsupported structure morph: {command.building}")
            sources = self._available_command_centers()
            if not sources:
                if iteration % 22 == 0:
                    print(f"Waiting for an available command center to morph {command.building}...")
                return
            target_type = self._morph_unit_type(command.building)
            if hasattr(self, "can_afford") and not self.can_afford(target_type):
                if iteration % 22 == 0:
                    print(f"Waiting for enough resources to morph {command.building}...")
                return
            issued = self._first_unit(sources).build(target_type)
            if issued is False:
                return
            print(
                f"Action {self._current_action_index + 1}/{self._plan_action_count()}: "
                f"issued morph {command.building} command"
            )
            self._advance_action()

        def _execute_research(self, command: ResearchUpgradeCommand, iteration: int) -> None:
            if command.upgrade not in UPGRADE_SPECS:
                raise TypeError(f"unsupported Terran upgrade: {command.upgrade}")
            upgrade_type = self._upgrade_id(command.upgrade)
            if hasattr(self, "already_pending_upgrade") and self.already_pending_upgrade(upgrade_type) > 0:
                self._advance_action()
                return
            if hasattr(self, "can_afford") and not self.can_afford(upgrade_type):
                if iteration % 22 == 0:
                    print(f"Waiting for enough resources to research {command.upgrade}...")
                return
            if hasattr(self, "research"):
                issued = self.research(upgrade_type)
            else:
                researcher = self._first_unit(self._ready_idle_structures(UPGRADE_SPECS[command.upgrade].researcher or ""))
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
            targets = self._repair_targets(command.target)
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
            workers = self.workers[: command.workers]
            if not workers:
                if iteration % 22 == 0:
                    print("Waiting for workers before repairing...")
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
            return max(0, self._structure_count(building, readiness="total") - total_before)

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

        @staticmethod
        def _closest_mineral_field(mineral_fields, worker):
            if hasattr(mineral_fields, "closest_to"):
                return mineral_fields.closest_to(worker)
            return mineral_fields[0]

        @staticmethod
        def _closest_enemy(enemies, unit):
            if hasattr(enemies, "closest_to"):
                return enemies.closest_to(unit)
            return enemies[0]

        def _available_townhalls(self):
            townhalls = self.townhalls
            if hasattr(townhalls, "ready"):
                townhalls = townhalls.ready
            if hasattr(townhalls, "idle"):
                townhalls = townhalls.idle
            return townhalls

        def _available_producers(self, unit: str):
            spec = UNIT_SPECS[unit]
            if spec.producer == "command_center":
                return self._available_townhalls()
            producers = self._ready_idle_structures(spec.producer or "")
            if spec.required_addon:
                producers = [
                    producer for producer in producers if self._producer_has_addon(producer, spec.required_addon)
                ]
            return producers

        def _ready_idle_structures(self, building: str):
            structures = self._structures_of_type(self._structure_unit_type(building))
            if hasattr(structures, "ready"):
                structures = structures.ready
            if hasattr(structures, "idle"):
                structures = structures.idle
            return structures

        def _free_addon_producers(self, producer: str):
            structures = self._ready_idle_structures(producer)
            return [
                structure
                for structure in structures
                if not getattr(structure, "has_add_on", False) and not getattr(structure, "add_on_tag", 0)
            ]

        def _producer_has_addon(self, producer, addon: str) -> bool:
            if addon.endswith("tech_lab") and getattr(producer, "has_techlab", False):
                return True
            if addon.endswith("reactor") and getattr(producer, "has_reactor", False):
                return True
            add_on_tag = getattr(producer, "add_on_tag", 0)
            if addon.endswith("tech_lab") and add_on_tag in getattr(self, "techlab_tags", set()):
                return True
            if addon.endswith("reactor") and add_on_tag in getattr(self, "reactor_tags", set()):
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
            return type(structures)([unit for unit in structures if getattr(unit, "type_id", None) == unit_type])

        def _ready_refineries(self):
            refineries = self._structures_of_type(self._unit_type_id().REFINERY)
            if hasattr(refineries, "ready"):
                refineries = refineries.ready
            return refineries

        def _structure_count(self, building: str, readiness: str = "total") -> int:
            structures = self._structures_of_type(self._structure_unit_type(building))
            if readiness == "total":
                return len(structures)
            if readiness == "ready":
                return sum(1 for structure in structures if _structure_is_ready(structure))
            if readiness == "pending":
                return sum(1 for structure in structures if not _structure_is_ready(structure))
            raise ValueError(f"unsupported structure readiness filter: {readiness}")

        def _wait_until_observed_value(self, command: WaitUntilCommand) -> float:
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
                return float(len(self.townhalls))
            if command.condition == "game_time":
                return float(getattr(self, "time", 0.0))
            if command.condition == "unit_count":
                if command.target == "worker":
                    return float(len(self.workers))
                return float(len(self._select_exact_units(command.target or "")))
            if command.condition == "upgrade_complete":
                upgrade = command.target or ""
                upgrade_type = self._upgrade_id(upgrade)
                completed: set[Any] = getattr(getattr(self, "state", None), "upgrades", set())
                return 1.0 if upgrade_type in completed else 0.0
            if command.condition == "structure_count":
                return float(self._structure_count(command.target or "", readiness="total"))
            if command.condition == "structure_ready":
                return float(self._structure_count(command.target or "", readiness="ready"))
            if command.condition == "structure_pending":
                return float(self._structure_count(command.target or "", readiness="pending"))
            raise TypeError(f"unsupported wait-until condition: {command.condition}")

        def _execute_refinery_build(self, unit_type) -> bool:
            geysers = getattr(self, "vespene_geyser", [])
            if not geysers:
                return False
            townhall = self._first_unit(self.townhalls) if self.townhalls else None
            geyser = geysers.closest_to(townhall) if townhall is not None and hasattr(geysers, "closest_to") else geysers[0]
            worker = self.select_build_worker(geyser) if hasattr(self, "select_build_worker") else self._first_unit(self.workers)
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
                self._mark_plan_finished()

        def _mark_plan_finished(self) -> None:
            if self._plan_finished_at_loop_time is None:
                self._plan_finished_at_loop_time = asyncio.get_running_loop().time()
                print("Strategy plan actions complete.")

        def _select_units(self, unit: str):
            return self._select_exact_units(unit)

        def _select_exact_units(self, unit: str):
            if unit == "worker":
                return self.workers
            if unit not in UNIT_SPECS:
                return []
            return self.units.of_type({self._train_unit_type(unit)})

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
            if target in UNIT_SPECS:
                return self.units.of_type({self._train_unit_type(target)})
            return self._structures_of_type(self._structure_unit_type(target))

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

                for spec in (*UNIT_SPECS.values(), *STRUCTURE_SPECS.values(), *ADDON_SPECS.values(), *MORPH_SPECS.values()):
                    setattr(_FallbackUnitTypeId, spec.enum_name, spec.enum_name)

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

                return _FallbackAbilityId

        def _should_stop(self) -> bool:
            if self._plan_finished_at_loop_time is None:
                return False
            elapsed = asyncio.get_running_loop().time() - self._plan_finished_at_loop_time
            return elapsed >= self.stop_after_seconds

        @staticmethod
        def _describe_action(action) -> str:
            if isinstance(action, MoveCommand):
                return f"move {action.unit} to ({action.x:g}, {action.y:g})"
            if isinstance(action, AttackMoveCommand):
                return f"attack with {action.unit} toward ({action.x:g}, {action.y:g})"
            if isinstance(action, AttackEnemyCommand):
                return f"attack visible enemy with {action.unit}"
            if isinstance(action, PatrolCommand):
                return f"patrol {action.unit} toward ({action.x:g}, {action.y:g})"
            if isinstance(action, HoldPositionCommand):
                return f"hold position with {action.unit}"
            if isinstance(action, StopCommand):
                return f"stop {action.unit}"
            if isinstance(action, RallyCommand):
                return f"rally {action.building} to ({action.x:g}, {action.y:g})"
            if isinstance(action, WaitCommand):
                return f"wait {action.seconds:g} second(s)"
            if isinstance(action, WaitUntilCommand):
                target = f" {action.target}" if action.target else ""
                return f"{action.condition}{target} >= {action.at_least:g}"
            if isinstance(action, GatherMineralsCommand):
                return f"gather minerals with {action.unit}"
            if isinstance(action, GatherGasCommand):
                return f"gather gas with {action.unit}"
            if isinstance(action, DistributeWorkersCommand):
                return f"distribute workers at ratio {action.mineral_to_gas_ratio:g}"
            if isinstance(action, TrainUnitCommand):
                return f"train {action.unit}" if action.count == 1 else f"train {action.count} {action.unit}"
            if isinstance(action, BuildStructureCommand):
                return f"build {action.building}" if action.count == 1 else f"build {action.count} {action.building}"
            if isinstance(action, ExpandCommand):
                return "expand" if action.count == 1 else f"expand {action.count} times"
            if isinstance(action, BuildAddonCommand):
                return f"build {action.addon}" if action.count == 1 else f"build {action.count} {action.addon}"
            if isinstance(action, MorphStructureCommand):
                return f"morph {action.building}"
            if isinstance(action, ResearchUpgradeCommand):
                return f"research {action.upgrade}"
            if isinstance(action, RepairCommand):
                return f"repair {action.target} with {action.workers} worker(s)"
            return repr(action)

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

    maps, BotAI, Difficulty, Race, run_game, Bot, Computer, _Point2 = _import_sc2_runtime()
    bot_class = create_game_state_bot_class(BotAI)
    bot = bot_class()
    try:
        selected_map = maps.get(map_name)
    except (FileNotFoundError, KeyError) as exc:
        env = detect_sc2_environment()
        raise SystemExit(_map_error_message(map_name, env)) from exc

    sc2_logs_disabled = False
    try:
        from loguru import logger

        logger.disable("sc2")
        sc2_logs_disabled = True
    except ImportError:
        logger = None

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        if sc2_logs_disabled and logger is not None:
            logger.enable("sc2")

    if bot.summary is None:
        raise SystemExit("Failed to capture StarCraft II game state before the game ended.")

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
        plan = validate_strategy_plan(plan_strategy(strategy, planner_name=planner_name))
    maps, BotAI, Difficulty, Race, run_game, Bot, Computer, Point2 = _import_sc2_runtime()
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
    parser = argparse.ArgumentParser(description="Run a minimal real StarCraft II movement bot.")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Strategy command to execute. Default: {DEFAULT_STRATEGY!r}",
    )
    parser.add_argument("--map", default=DEFAULT_MAP, help=f"SC2 map name. Default: {DEFAULT_MAP!r}")
    parser.add_argument(
        "--planner",
        default=DEFAULT_PLANNER,
        choices=PLANNER_MODES,
        help=f"Planner mode. Default: {DEFAULT_PLANNER!r}. Other modes must be selected explicitly.",
    )
    parser.add_argument("--stop-after", type=int, default=35, help="Seconds to keep the game open after issuing move.")
    parser.add_argument("--fast", action="store_true", help="Run non-realtime for faster automated checks.")
    parser.add_argument(
        "--observe-before-plan",
        action="store_true",
        help="Start SC2, summarize the initial game state, pass it to the planner, validate the plan, then execute it.",
    )
    parser.add_argument("--check", action="store_true", help="Only check local SC2 installation hints; do not start the game.")
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
            print("Install StarCraft II with the Blizzard/Battle.net app or set SC2PATH to the install directory.")
            print("Checked paths:")
            for path in env.candidate_paths:
                print(f"- {path}")
            return 1

        print(f"StarCraft II path detected: {env.detected_path}")
        if env.maps_installed:
            print(f"SC2 API maps directory detected: {env.maps_path}")
            return 0

        print(f"SC2 API maps directory missing: {env.maps_path}")
        print("Install/extract a Blizzard SC2 map pack into the Maps folder before launching a game.")
        print("The default map needs the Ladder 2017 Season 1 map pack, or pass another installed map with --map.")
        return 1

    if args.print_plan:
        try:
            plan = plan_strategy(args.strategy, planner_name=args.planner)
        except (PlanValidationError, PlannerError, PlannerUnavailableError, ValueError) as exc:
            print(f"Planner error: {exc}", file=sys.stderr)
            return 2
        print(strategy_plan_to_json(plan))
        return 0

    if args.print_state:
        print_game_state(map_name=args.map, realtime=not args.fast)
        return 0

    if not env.installed:
        print("Warning: StarCraft II was not detected before launch; python-sc2 may still find it if configured.")
    elif not env.maps_installed:
        print(f"Warning: SC2 API maps directory was not detected at {env.maps_path}; launch may fail.")

    try:
        run_real_game(
            strategy=args.strategy,
            map_name=args.map,
            realtime=not args.fast,
            stop_after_seconds=args.stop_after,
            planner_name=args.planner,
            observe_before_plan=args.observe_before_plan,
        )
    except (PlanValidationError, PlannerError, PlannerUnavailableError, ValueError) as exc:
        print(f"Planner error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
