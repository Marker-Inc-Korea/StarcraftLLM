from __future__ import annotations

import argparse
import asyncio
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from starcraft_llm.strategy import (
    MoveCommand,
    StrategyPlan,
    WaitCommand,
    parse_strategy_request,
    strategy_plan_to_json,
)

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


def create_move_unit_bot_class(bot_ai_base, point2_class):
    class _MoveUnitBot(bot_ai_base):
        def __init__(self, plan: StrategyPlan, stop_after_seconds: int = 35):
            super().__init__()
            self.plan = plan
            self.stop_after_seconds = stop_after_seconds
            self._current_action_index = 0
            self._action_started_at_loop_time: float | None = None
            self._plan_finished_at_loop_time: float | None = None
            self._left_game = False

        async def on_start(self):
            self.client.game_step = 2
            print(f"Strategy plan loaded: {len(self.plan.actions)} action(s)")
            for index, action in enumerate(self.plan.actions, start=1):
                print(f"  {index}. {self._describe_action(action)}")

        async def on_step(self, iteration: int):
            if self._plan_finished_at_loop_time is None:
                self._execute_current_action(iteration)

            if not self._left_game and self._should_stop():
                print("MVP complete: strategy plan finished; leaving the game.")
                self._left_game = True
                await self.client.leave()

        def _execute_current_action(self, iteration: int) -> None:
            if self._current_action_index >= len(self.plan.actions):
                self._mark_plan_finished()
                return

            action = self.plan.actions[self._current_action_index]
            if isinstance(action, MoveCommand):
                self._execute_move(action, iteration)
                return
            if isinstance(action, WaitCommand):
                self._execute_wait(action)
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

        def _advance_action(self) -> None:
            self._current_action_index += 1
            self._action_started_at_loop_time = None
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

        @staticmethod
        def _unit_type_id():
            from sc2.ids.unit_typeid import UnitTypeId

            return UnitTypeId

        def _should_stop(self) -> bool:
            if self._plan_finished_at_loop_time is None:
                return False
            elapsed = asyncio.get_running_loop().time() - self._plan_finished_at_loop_time
            return elapsed >= self.stop_after_seconds

        @staticmethod
        def _describe_action(action) -> str:
            if isinstance(action, MoveCommand):
                return f"move {action.unit} to ({action.x:g}, {action.y:g})"
            if isinstance(action, WaitCommand):
                return f"wait {action.seconds:g} second(s)"
            return repr(action)

    return _MoveUnitBot


def run_real_game(strategy: str, map_name: str, realtime: bool, stop_after_seconds: int) -> None:
    """Start StarCraft II and run the minimal movement bot."""

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

    plan = parse_strategy_request(strategy)
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
                Bot(Race.Terran, bot_class(plan, stop_after_seconds=stop_after_seconds)),
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
    parser.add_argument("--stop-after", type=int, default=35, help="Seconds to keep the game open after issuing move.")
    parser.add_argument("--fast", action="store_true", help="Run non-realtime for faster automated checks.")
    parser.add_argument("--check", action="store_true", help="Only check local SC2 installation hints; do not start the game.")
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Parse --strategy as DSL, JSON, or known intent and print canonical StrategyPlan JSON without starting SC2.",
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
        print(strategy_plan_to_json(parse_strategy_request(args.strategy)))
        return 0

    if not env.installed:
        print("Warning: StarCraft II was not detected before launch; python-sc2 may still find it if configured.")
    elif not env.maps_installed:
        print(f"Warning: SC2 API maps directory was not detected at {env.maps_path}; launch may fail.")

    run_real_game(
        strategy=args.strategy,
        map_name=args.map,
        realtime=not args.fast,
        stop_after_seconds=args.stop_after,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
