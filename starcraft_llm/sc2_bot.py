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
from typing import Iterable

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
    AttackMoveCommand,
    BuildStructureCommand,
    GatherMineralsCommand,
    MoveCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
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
    for structure in getattr(bot, "structures", bot.townhalls):
        name = _unit_type_name(structure)
        structures[name] = structures.get(name, 0) + 1

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
    )


def _unit_type_name(unit) -> str:
    raw_type = getattr(unit, "type_id", "unknown")
    name = getattr(raw_type, "name", str(raw_type))
    return name.lower()


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
            print(f"Strategy plan loaded: {len(self.plan.actions)} action(s)")
            for index, action in enumerate(self.plan.actions, start=1):
                print(f"  {index}. {self._describe_action(action)}")

        async def _execute_current_action(self, iteration: int) -> None:
            if self.plan is None:
                raise RuntimeError("strategy plan is not loaded")
            if self._current_action_index >= len(self.plan.actions):
                self._mark_plan_finished()
                return

            action = self.plan.actions[self._current_action_index]
            if isinstance(action, MoveCommand):
                self._execute_move(action, iteration)
                return
            if isinstance(action, AttackMoveCommand):
                self._execute_attack(action, iteration)
                return
            if isinstance(action, WaitCommand):
                self._execute_wait(action)
                return
            if isinstance(action, GatherMineralsCommand):
                self._execute_gather_minerals(action, iteration)
                return
            if isinstance(action, TrainUnitCommand):
                self._execute_train(action, iteration)
                return
            if isinstance(action, BuildStructureCommand):
                await self._execute_build(action, iteration)
                return

            raise TypeError(f"unsupported strategy action: {action!r}")

        def _execute_move(self, command: MoveCommand, iteration: int) -> None:
            target = point2_class((command.x, command.y))
            units = self._select_units(command.unit)
            if units:
                for unit in units:
                    unit.move(target)
                print(
                    f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
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
                    f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
                    f"issued attack command to {len(units)} {command.unit} unit(s): {target}"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for controllable {command.unit} units before attacking...")

        def _execute_wait(self, command: WaitCommand) -> None:
            now = asyncio.get_running_loop().time()
            if self._action_started_at_loop_time is None:
                self._action_started_at_loop_time = now
                print(
                    f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
                    f"waiting {command.seconds:g} second(s)"
                )

            elapsed = now - self._action_started_at_loop_time
            if elapsed >= command.seconds:
                self._advance_action()

        def _execute_gather_minerals(self, command: GatherMineralsCommand, iteration: int) -> None:
            workers = self._select_units(command.unit)
            mineral_fields = self.mineral_field
            if workers and mineral_fields:
                issued = 0
                for worker in workers:
                    mineral_field = self._closest_mineral_field(mineral_fields, worker)
                    worker.gather(mineral_field)
                    issued += 1
                print(
                    f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
                    f"issued gather minerals command to {issued} worker unit(s)"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print("Waiting for workers and mineral fields before gathering...")

        def _execute_train(self, command: TrainUnitCommand, iteration: int) -> None:
            if command.unit not in {"scv", "marine"}:
                raise TypeError(f"unsupported train unit: {command.unit}")

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
            print(
                f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
                f"issued train {command.unit} command"
            )
            self._advance_action()

        async def _execute_build(self, command: BuildStructureCommand, iteration: int) -> None:
            if command.building not in {"supply_depot", "barracks", "refinery"}:
                raise TypeError(f"unsupported build structure: {command.building}")

            unit_type = self._building_unit_type(command.building)
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
                print(
                    f"Action {self._current_action_index + 1}/{len(self.plan.actions)}: "
                    f"issued build {command.building} command"
                )
                self._advance_action()
            elif iteration % 22 == 0:
                print(f"Waiting for placement/worker to build {command.building}...")

        @staticmethod
        def _closest_mineral_field(mineral_fields, worker):
            if hasattr(mineral_fields, "closest_to"):
                return mineral_fields.closest_to(worker)
            return mineral_fields[0]

        def _available_townhalls(self):
            townhalls = self.townhalls
            if hasattr(townhalls, "ready"):
                townhalls = townhalls.ready
            if hasattr(townhalls, "idle"):
                townhalls = townhalls.idle
            return townhalls

        def _available_producers(self, unit: str):
            if unit == "scv":
                return self._available_townhalls()

            barracks = self._structures_of_type(self._unit_type_id().BARRACKS)
            if hasattr(barracks, "ready"):
                barracks = barracks.ready
            if hasattr(barracks, "idle"):
                barracks = barracks.idle
            return barracks

        def _structures_of_type(self, unit_type):
            structures = getattr(self, "structures", [])
            if hasattr(structures, "of_type"):
                return structures.of_type({unit_type})
            return type(structures)([unit for unit in structures if getattr(unit, "type_id", None) == unit_type])

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

        def _advance_action(self) -> None:
            self._current_action_index += 1
            self._action_started_at_loop_time = None
            if self.plan is None:
                raise RuntimeError("strategy plan is not loaded")
            if self._current_action_index >= len(self.plan.actions):
                self._mark_plan_finished()

        def _mark_plan_finished(self) -> None:
            if self._plan_finished_at_loop_time is None:
                self._plan_finished_at_loop_time = asyncio.get_running_loop().time()
                print("Strategy plan actions complete.")

        def _select_units(self, unit: str):
            if unit == "worker":
                return self.workers
            if unit == "marine":
                marines = self.units.of_type({self._unit_type_id().MARINE})
                return marines if marines else self.workers
            return self.workers

        def _train_unit_type(self, unit: str):
            if unit == "scv":
                return self._unit_type_id().SCV
            if unit == "marine":
                return self._unit_type_id().MARINE
            raise TypeError(f"unsupported train unit: {unit}")

        def _building_unit_type(self, building: str):
            unit_type_id = self._unit_type_id()
            if building == "supply_depot":
                return unit_type_id.SUPPLYDEPOT
            if building == "barracks":
                return unit_type_id.BARRACKS
            if building == "refinery":
                return unit_type_id.REFINERY
            raise TypeError(f"unsupported build structure: {building}")

        @staticmethod
        def _unit_type_id():
            try:
                from sc2.ids.unit_typeid import UnitTypeId

                return UnitTypeId
            except ImportError:
                class _FallbackUnitTypeId:
                    SCV = "SCV"
                    MARINE = "MARINE"
                    SUPPLYDEPOT = "SUPPLYDEPOT"
                    BARRACKS = "BARRACKS"
                    REFINERY = "REFINERY"

                return _FallbackUnitTypeId

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
            if isinstance(action, WaitCommand):
                return f"wait {action.seconds:g} second(s)"
            if isinstance(action, GatherMineralsCommand):
                return f"gather minerals with {action.unit}"
            if isinstance(action, TrainUnitCommand):
                return f"train {action.unit}"
            if isinstance(action, BuildStructureCommand):
                return f"build {action.building}"
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
