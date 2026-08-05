from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Union

from starcraft_llm.command_catalog import (
    ABILITY_SPECS,
    ADDON_SPECS,
    ALERT_KEYS,
    ATTACK_CAPABLE_UNIT_KEYS,
    ENEMY_RACE_KEYS,
    FLYING_STRUCTURE_ACTOR_KEYS,
    LOCATION_SPECS,
    MAX_CONDITION_TERMS,
    MAX_CONTROL_BRANCH_ACTIONS,
    MAX_CONTROL_DEPTH,
    MAX_CONTROL_EXECUTION_ACTIONS,
    MAX_CONTROL_TOTAL_ACTIONS,
    MAX_POLICY_SECONDS,
    MAX_REPEAT_CYCLES,
    MAX_SELECTION_COUNT,
    MAX_STRUCTURE_ACTION_COUNT,
    MAX_WORKER_ASSIGNMENT_COUNT,
    MORPH_SPECS,
    MOBILE_ATTACK_CAPABLE_UNIT_KEYS,
    MOVABLE_SPECIAL_UNIT_KEYS,
    REPAIRABLE_TARGET_KEYS,
    STRUCTURE_SPECS,
    TARGET_SELECTORS,
    TRANSFORM_ABILITY_KEYS,
    UNIT_FORM_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    normalize_name,
    resolve_ability,
    resolve_alias,
    resolve_location,
    resolve_selection_mode,
)


@dataclass(frozen=True)
class LocationRef:
    """LLM location target: either a semantic anchor or bounded coordinates."""

    semantic: str | None = None
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class SelectionSpec:
    """Bounded per-action actor selection; no persistent control groups."""

    mode: str = "all"
    count: int | None = None
    tags: tuple[int, ...] = ()


@dataclass(frozen=True)
class MoveCommand:
    """Move a bounded logical unit group to coordinates or a semantic point."""

    unit: str
    x: float | None = None
    y: float | None = None
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    queued: bool = False
    target_unit: str | None = None
    target_tag: int | None = None
    wait_for_arrival: bool = False
    arrival_tolerance: float = 2.5
    timeout_seconds: float = 90.0


@dataclass(frozen=True)
class AttackMoveCommand:
    """Attack-move a bounded logical unit group toward a map point."""

    unit: str
    x: float | None = None
    y: float | None = None
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class AttackEnemyCommand:
    """Attack a visible enemy or neutral destructible with one logical unit group."""

    unit: str = "marine"
    selection: SelectionSpec | None = None
    queued: bool = False
    target_unit: str | None = None
    target_tag: int | None = None
    target_alliance: str = "enemy"
    wait_for_target_death: bool = False
    timeout_seconds: float = 60.0


@dataclass(frozen=True)
class KiteCommand:
    """Bounded cooldown-aware stutter-step micro against one visible target."""

    unit: str
    target_unit: str = "nearest_enemy"
    target_tag: int | None = None
    selection: SelectionSpec | None = None
    duration_seconds: float = 8.0
    retreat_distance: float = 2.0


@dataclass(frozen=True)
class AttackUntilClearCommand:
    """Clear an observed area and require a stable, bounded clear confirmation."""

    unit: str
    location: LocationRef
    target_unit: str | None = None
    selection: SelectionSpec | None = None
    radius: float = 16.0
    arrival_tolerance: float = 5.0
    clear_seconds: float = 2.0
    timeout_seconds: float = 180.0
    on_timeout: str = "replan"


@dataclass(frozen=True)
class WaitCommand:
    """Pause strategy execution for a small amount of game-clock time."""

    seconds: float


@dataclass(frozen=True)
class WaitUntilCommand:
    """Pause execution until the observed game state satisfies a condition."""

    condition: str
    at_least: float
    comparison: str = "gte"
    target: str | None = None
    ability: str | None = None
    actor: str | None = None
    location: LocationRef | None = None
    radius: float = 12.0
    selection: SelectionSpec | None = None
    timeout_seconds: float = 120.0
    on_timeout: str = "replan"


@dataclass(frozen=True)
class ConditionSpec:
    """One comparator-based runtime observation used by control flow."""

    condition: str
    value: float = 1.0
    comparison: str = "gte"
    target: str | None = None
    ability: str | None = None
    actor: str | None = None
    location: LocationRef | None = None
    radius: float = 12.0
    selection: SelectionSpec | None = None


@dataclass(frozen=True)
class ConditionGroup:
    """A bounded all/any group of atomic condition observations."""

    match: str
    conditions: tuple[ConditionSpec, ...]


ConditionExpression = Union[ConditionSpec, ConditionGroup]


@dataclass(frozen=True)
class GatherMineralsCommand:
    """Send a logical worker group to mineral fields."""

    unit: str = "worker"
    workers: int | None = None
    location: LocationRef | None = None
    target_tag: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class GatherGasCommand:
    """Send workers to ready refineries for vespene gathering."""

    unit: str = "worker"
    workers: int | None = None
    location: LocationRef | None = None
    target_tag: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class ReturnCargoCommand:
    """Return carried resources from selected workers or MULEs."""

    unit: str = "worker"
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class TrainUnitCommand:
    """Train one or more units from available production structures."""

    unit: str
    count: int = 1
    producer_selection: SelectionSpec | None = None


@dataclass(frozen=True)
class ProductionPolicyCommand:
    """Bounded foreground/background production toward an absolute unit count."""

    unit: str
    target_count: int
    background: bool = False
    producer_selection: SelectionSpec | None = None
    reserve_minerals: int = 0
    reserve_vespene: int = 0
    reserve_supply: int = 0
    max_seconds: float = 300.0


@dataclass(frozen=True)
class StopProductionCommand:
    """Stop one or every registered background production policy."""

    unit: str | None = None


@dataclass(frozen=True)
class BuildStructureCommand:
    """Build Terran structures at a semantic/coordinate or safe placement."""

    building: str
    worker: str = "worker"
    count: int = 1
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    placement_mode: str = "near"
    max_distance: int = 20
    reserve_addon_space: bool = False


@dataclass(frozen=True)
class ExpandCommand:
    """Build one or more command centers at executor-selected expansion sites."""

    count: int = 1


@dataclass(frozen=True)
class BuildAddonCommand:
    """Attach a Tech Lab or Reactor to an available production structure."""

    addon: str
    count: int = 1
    selection: SelectionSpec | None = None


@dataclass(frozen=True)
class MorphStructureCommand:
    """Transform a command center into an orbital command or planetary fortress."""

    building: str
    selection: SelectionSpec | None = None


@dataclass(frozen=True)
class ResearchUpgradeCommand:
    """Research one whitelisted Terran upgrade from an available structure."""

    upgrade: str
    researcher_selection: SelectionSpec | None = None


@dataclass(frozen=True)
class DistributeWorkersCommand:
    """Rebalance idle and oversaturated workers between minerals and gas."""

    mineral_to_gas_ratio: float = 2.0


@dataclass(frozen=True)
class RepairCommand:
    """Repair the nearest damaged unit or structure of a requested type/tag."""

    target: str | None = None
    workers: int = 1
    target_tag: int | None = None
    target_selector: str | None = None
    target_selection: SelectionSpec | None = None
    selection: SelectionSpec | None = None


@dataclass(frozen=True)
class RallyCommand:
    """Set production structures' rally point."""

    building: str
    x: float | None = None
    y: float | None = None
    location: LocationRef | None = None
    target_unit: str | None = None
    target_tag: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class PatrolCommand:
    """Patrol a logical unit group toward a map point."""

    unit: str
    x: float | None = None
    y: float | None = None
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class HoldPositionCommand:
    """Order a logical unit group to hold position."""

    unit: str
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class StopCommand:
    """Stop the current orders for a logical unit group."""

    unit: str
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class UseAbilityCommand:
    """Use one allowlisted Terran ability with runtime availability gating."""

    ability: str
    actor: str | None = None
    target_unit: str | None = None
    target_tag: int | None = None
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class ScanCommand:
    location: LocationRef
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class CallDownMuleCommand:
    location: LocationRef
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class SupplyDropCommand:
    target_unit: str = "supply_depot"
    target_tag: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class TransformCommand:
    ability: str
    actor: str | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class LiftCommand:
    actor: str
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class LandCommand:
    actor: str
    location: LocationRef | None = None
    target_addon: str | None = None
    target_addon_tag: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class LoadCommand:
    actor: str
    target_unit: str | None = None
    target_tag: int | None = None
    target_selection: SelectionSpec | None = None
    count: int | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class UnloadCommand:
    actor: str
    target_unit: str | None = None
    passenger_tag: int | None = None
    location: LocationRef | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class CancelCommand:
    ability: str = "cancel_any"
    actor: str | None = None
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class SalvageCommand:
    actor: str
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class BuildNukeCommand:
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class LaunchNukeCommand:
    location: LocationRef
    selection: SelectionSpec | None = None
    queued: bool = False


@dataclass(frozen=True)
class ReplanCommand:
    reason: str = "requested"


@dataclass(frozen=True)
class ConditionalCommand:
    """Choose one validated branch from a bounded runtime condition expression."""

    when: ConditionExpression
    then_actions: tuple["StrategyAction", ...]
    else_actions: tuple["StrategyAction", ...] = ()


@dataclass(frozen=True)
class RepeatCommand:
    """Repeat a validated body for fixed cycles or until a condition is met."""

    actions: tuple["StrategyAction", ...]
    max_cycles: int
    until: ConditionExpression | None = None
    max_seconds: float = 300.0
    on_exhausted: str = "replan"


@dataclass(frozen=True)
class WithTimeoutCommand:
    """Run a nested sequence under one outer game-clock deadline."""

    actions: tuple["StrategyAction", ...]
    timeout_seconds: float = 120.0
    on_timeout: str = "replan"


StrategyAction = Union[
    MoveCommand,
    AttackMoveCommand,
    AttackEnemyCommand,
    KiteCommand,
    AttackUntilClearCommand,
    PatrolCommand,
    HoldPositionCommand,
    StopCommand,
    RallyCommand,
    WaitCommand,
    WaitUntilCommand,
    GatherMineralsCommand,
    GatherGasCommand,
    ReturnCargoCommand,
    DistributeWorkersCommand,
    TrainUnitCommand,
    ProductionPolicyCommand,
    StopProductionCommand,
    BuildStructureCommand,
    ExpandCommand,
    BuildAddonCommand,
    MorphStructureCommand,
    ResearchUpgradeCommand,
    RepairCommand,
    UseAbilityCommand,
    ScanCommand,
    CallDownMuleCommand,
    SupplyDropCommand,
    TransformCommand,
    LiftCommand,
    LandCommand,
    LoadCommand,
    UnloadCommand,
    CancelCommand,
    SalvageCommand,
    BuildNukeCommand,
    LaunchNukeCommand,
    ReplanCommand,
    ConditionalCommand,
    RepeatCommand,
    WithTimeoutCommand,
]


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
_ATTACK_ROUTE = ((55.0, 45.0),)


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
    - attack marine 55 45
    - attack enemy
    - patrol marine 45 42
    - hold marine
    - stop marine
    - rally barracks 35 42
    - wait 2
    - wait until minerals 100
    - wait until structure supply depot ready
    - wait until unit marine 1
    - gather minerals
    - gather gas
    - return cargo
    - distribute workers
    - train scv
    - train marine 2
    - build supply depot
    - build barracks
    - build refinery
    - expand
    - addon barracks tech lab
    - morph orbital command
    - research stimpack
    - repair barracks

    Multiple actions can be separated by semicolons, newlines, or the word
    "then", for example: "move worker 35 42; wait 1; move worker 42 42".
    """

    chunks = [
        chunk.strip()
        for chunk in _COMMAND_SPLIT_RE.split(text.strip())
        if chunk.strip()
    ]
    if not chunks:
        raise StrategyParseError("strategy command is empty")

    return StrategyPlan(
        actions=tuple(
            parse_strategy_action(chunk, default_unit=default_unit) for chunk in chunks
        )
    )


def parse_strategy_action(text: str, default_unit: str = "worker") -> StrategyAction:
    parts = text.strip().split()
    if not parts:
        raise StrategyParseError("strategy command is empty")

    verb = parts[0].lower()
    if verb == "use" and len(parts) > 1 and parts[1].lower() == "ability":
        return _parse_use_ability(["use_ability", *parts[2:]])
    if verb == "build" and len(parts) > 1 and parts[1].lower() == "nuke":
        return BuildNukeCommand()
    if verb == "supply" and len(parts) > 1 and parts[1].lower() == "drop":
        return SupplyDropCommand(
            target_unit=normalize_target_unit(" ".join(parts[2:]) or "supply_depot")
        )
    if verb == "launch" and len(parts) > 1 and parts[1].lower() == "nuke":
        return LaunchNukeCommand(location=_parse_location_words(parts[2:]))
    if verb == "move":
        return _parse_move(parts, default_unit=default_unit)
    if verb in {"move_and_wait", "move-and-wait"}:
        command = _parse_move(["move", *parts[1:]], default_unit=default_unit)
        return MoveCommand(
            unit=command.unit,
            x=command.x,
            y=command.y,
            location=command.location,
            wait_for_arrival=True,
        )
    if verb in {"follow", "move_target", "move-target"}:
        return _parse_follow(parts)
    if verb in {"attack", "attack_move", "attack-move"}:
        return _parse_attack(parts, default_unit=default_unit)
    if verb in {"focus_fire", "focus-fire"}:
        if len(parts) < 3:
            raise StrategyParseError("use: focus_fire marine nearest_enemy")
        return AttackEnemyCommand(
            unit=normalize_attack_actor(parts[1]),
            target_unit=normalize_enemy_target(" ".join(parts[2:])),
            wait_for_target_death=True,
        )
    if verb == "kite":
        if len(parts) < 3:
            raise StrategyParseError("use: kite marine nearest_enemy")
        return KiteCommand(
            unit=normalize_mobile_attack_unit(parts[1]),
            target_unit=normalize_enemy_target(" ".join(parts[2:])),
        )
    if verb == "patrol":
        return _parse_patrol(parts, default_unit=default_unit)
    if verb in {"hold", "hold_position", "hold-position"}:
        return _parse_unit_order(parts, HoldPositionCommand, default_unit=default_unit)
    if verb == "stop":
        return _parse_unit_order(parts, StopCommand, default_unit=default_unit)
    if verb == "rally":
        return _parse_rally(parts)
    if verb == "wait":
        return _parse_wait(parts)
    if verb == "gather":
        return _parse_gather(parts, default_unit=default_unit)
    if verb in {"return", "return_cargo", "return-cargo"}:
        return _parse_return_cargo(parts, default_unit=default_unit)
    if verb in {"distribute", "distribute_workers", "distribute-workers"}:
        return _parse_distribute_workers(parts)
    if verb == "train":
        return _parse_train(parts)
    if verb in {
        "produce_until",
        "produce-until",
        "maintain_production",
        "maintain-production",
    }:
        if len(parts) < 3:
            raise StrategyParseError(
                "use: produce_until marine 16 or maintain_production scv 22"
            )
        return ProductionPolicyCommand(
            unit=normalize_train_unit(" ".join(parts[1:-1])),
            target_count=_parse_positive_int(
                parts[-1], "production target", MAX_SELECTION_COUNT
            ),
            background=verb in {"maintain_production", "maintain-production"},
        )
    if verb == "build":
        return _parse_build(parts)
    if verb == "expand":
        return _parse_expand(parts)
    if verb in {"addon", "build_addon", "build-addon"}:
        return _parse_addon(parts)
    if verb == "morph":
        return _parse_morph(parts)
    if verb in {"research", "upgrade"}:
        return _parse_research(parts)
    if verb == "repair":
        return _parse_repair(parts)
    if verb in {"use_ability", "use-ability", "ability"}:
        return _parse_use_ability(parts)
    if verb == "scan":
        return ScanCommand(location=_parse_location_words(parts[1:]))
    if verb in {"mule", "call_down_mule", "call-down-mule"}:
        return CallDownMuleCommand(location=_parse_location_words(parts[1:]))
    if verb in {"supply_drop", "supply-drop"}:
        return SupplyDropCommand(
            target_unit=normalize_target_unit(" ".join(parts[1:]) or "supply_depot")
        )
    if verb in {"transform", "mode"}:
        return _parse_transform(parts)
    if verb == "lift":
        if len(parts) < 2:
            raise StrategyParseError("use: lift barracks")
        return LiftCommand(actor=normalize_ability_actor(" ".join(parts[1:])))
    if verb == "land":
        return _parse_land(parts)
    if verb == "load":
        return _parse_load(parts)
    if verb == "unload":
        return _parse_unload(parts)
    if verb == "cancel":
        ability = normalize_cancel_ability(" ".join(parts[1:]) or "cancel_any")
        return CancelCommand(ability=ability)
    if verb == "salvage":
        if len(parts) < 2:
            raise StrategyParseError("use: salvage bunker")
        return SalvageCommand(actor=normalize_ability_actor(" ".join(parts[1:])))
    if verb in {"build_nuke", "build-nuke"}:
        return BuildNukeCommand()
    if verb in {"launch_nuke", "launch-nuke", "nuke"}:
        return LaunchNukeCommand(location=_parse_location_words(parts[1:]))
    if verb == "replan":
        return ReplanCommand(reason=" ".join(parts[1:]) or "requested")

    raise StrategyParseError(
        "unsupported strategy command; use move, move_and_wait, follow, attack, focus_fire, kite, patrol, hold, stop, rally, wait, "
        "gather, return_cargo, distribute, train, produce_until, maintain_production, build, expand, addon, morph, research, repair, use_ability, "
        "scan, call_down_mule, supply_drop, transform, lift, land, load, unload, cancel, salvage, "
        "build_nuke, launch_nuke, or replan"
    )


def parse_strategy_plan_json(text: str, default_unit: str = "worker") -> StrategyPlan:
    """Parse the canonical JSON StrategyPlan format.

    Expected object form:
    {
      "actions": [
        {"type": "move", "unit": "worker", "x": 35, "y": 42},
        {"type": "attack", "unit": "marine", "x": 55, "y": 45},
        {"type": "attack_enemy", "unit": "marine"},
        {"type": "wait", "seconds": 1},
        {"type": "wait_until", "condition": "minerals", "at_least": 100},
        {"type": "wait_until", "condition": "structure_ready", "target": "supply_depot", "at_least": 1},
        {"type": "gather", "unit": "worker", "resource": "minerals"},
        {"type": "gather", "unit": "worker", "resource": "vespene"},
        {"type": "train", "unit": "scv", "count": 2},
        {"type": "build", "building": "supply_depot"}
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
            raise StrategyParseError(
                "strategy JSON object must contain an 'actions' field"
            )
        actions_payload = payload["actions"]
    else:
        raise StrategyParseError("strategy JSON must be an object or an actions array")

    if not isinstance(actions_payload, list) or not actions_payload:
        raise StrategyParseError("strategy JSON 'actions' must be a non-empty array")

    plan = StrategyPlan(
        actions=tuple(
            _action_from_dict(action, default_unit=default_unit, control_depth=0)
            for action in actions_payload
        )
    )
    defined_actions = _defined_action_count(plan.actions)
    if defined_actions > MAX_CONTROL_TOTAL_ACTIONS:
        raise StrategyParseError(
            "strategy JSON defines too many top-level/nested actions: "
            f"{defined_actions} > {MAX_CONTROL_TOTAL_ACTIONS}"
        )
    execution_actions = _maximum_execution_action_count(plan.actions)
    if execution_actions > MAX_CONTROL_EXECUTION_ACTIONS:
        raise StrategyParseError(
            "strategy JSON expands to too many bounded runtime actions: "
            f"{execution_actions} > {MAX_CONTROL_EXECUTION_ACTIONS}"
        )
    return plan


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

    if any(keyword in normalized for keyword in ("공격", "attack", "rush")) and not any(
        keyword in normalized for keyword in ("업그레이드", "연구", "upgrade", "research")
    ):
        return StrategyPlan(
            actions=tuple(
                AttackMoveCommand(unit=unit, x=x, y=y) for x, y in _ATTACK_ROUTE
            )
        )

    if any(keyword in normalized for keyword in ("집결", "전진", "rally", "advance")):
        return _route_plan(unit=unit, route=_RALLY_ROUTE, wait_seconds=0)

    if any(
        keyword in normalized
        for keyword in ("일꾼 분배", "일꾼 재배치", "distribute workers", "rebalance workers")
    ):
        return StrategyPlan(actions=(DistributeWorkersCommand(),))

    if any(keyword in normalized for keyword in ("채취", "캐", "gather", "harvest")):
        if any(keyword in normalized for keyword in ("가스", "베스핀", "gas", "vespene")):
            return StrategyPlan(actions=(GatherGasCommand(unit="worker"),))
        return StrategyPlan(actions=(GatherMineralsCommand(unit="worker"),))

    if any(
        keyword in normalized
        for keyword in ("확장", "멀티", "expand", "expansion", "natural")
    ):
        return StrategyPlan(actions=(ExpandCommand(),))

    if any(keyword in normalized for keyword in ("연구", "업그레이드", "research", "upgrade")):
        upgrade = _catalog_key_from_intent(normalized, UPGRADE_SPECS)
        if upgrade:
            return StrategyPlan(actions=(ResearchUpgradeCommand(upgrade=upgrade),))

    if any(
        keyword in normalized
        for keyword in (
            "기술실",
            "반응로",
            "tech lab",
            "techlab",
            "reactor",
            "addon",
            "add-on",
        )
    ):
        addon = _catalog_key_from_intent(normalized, ADDON_SPECS)
        if addon:
            return StrategyPlan(actions=(BuildAddonCommand(addon=addon),))

    morph = _catalog_key_from_intent(normalized, MORPH_SPECS)
    if morph and any(
        keyword in normalized
        for keyword in ("변환", "변신", "올려", "업그레이드", "morph", "upgrade", "만들")
    ):
        return StrategyPlan(actions=(MorphStructureCommand(building=morph),))

    if any(keyword in normalized for keyword in ("수리", "repair", "fix")):
        target = _catalog_key_from_intent(
            normalized, {**STRUCTURE_SPECS, **MORPH_SPECS, **UNIT_SPECS}
        )
        if target:
            return StrategyPlan(
                actions=(RepairCommand(target="worker" if target == "scv" else target),)
            )

    if any(
        keyword in normalized
        for keyword in ("생산", "훈련", "뽑", "train", "make", "produce")
    ):
        train_unit = _catalog_key_from_intent(normalized, UNIT_SPECS)
        if train_unit:
            return StrategyPlan(actions=(TrainUnitCommand(unit=train_unit),))

    building = _catalog_key_from_intent(normalized, STRUCTURE_SPECS)
    if building:
        return StrategyPlan(actions=(BuildStructureCommand(building=building),))

    if any(keyword in normalized for keyword in ("자원", "미네랄", "mineral", "minerals")):
        return StrategyPlan(actions=(GatherMineralsCommand(unit="worker"),))

    raise StrategyParseError(f"unknown strategy intent: {text}")


def _parse_move(parts: list[str], default_unit: str) -> MoveCommand:
    if len(parts) == 3:
        unit = default_unit
        x_text, y_text = parts[1:]
    elif len(parts) >= 4:
        unit = normalize_unit(" ".join(parts[1:-2]))
        x_text, y_text = parts[-2:]
    else:
        raise StrategyParseError("use: move worker 35 42 or move 35 42")

    x, y = _parse_coordinates(x_text, y_text)
    return MoveCommand(unit=unit, x=x, y=y)


def _parse_follow(parts: list[str]) -> MoveCommand:
    if len(parts) < 3:
        raise StrategyParseError("use: follow medivac marine")
    return MoveCommand(
        unit=normalize_unit(parts[1]),
        target_unit=normalize_target_unit(" ".join(parts[2:])),
    )


def _parse_attack(
    parts: list[str], default_unit: str
) -> AttackMoveCommand | AttackEnemyCommand:
    if len(parts) >= 2 and parts[1].lower() == "move":
        parts = [parts[0], *parts[2:]]

    if _is_enemy_attack(parts):
        unit_parts = parts[1:-1]
        if unit_parts and unit_parts[-1].lower() == "nearest":
            unit_parts = unit_parts[:-1]
        unit = normalize_attack_actor(" ".join(unit_parts)) if unit_parts else "marine"
        return AttackEnemyCommand(unit=unit)

    if len(parts) == 3:
        unit = default_unit
        x_text, y_text = parts[1:]
    elif len(parts) >= 4:
        unit = normalize_unit(" ".join(parts[1:-2]))
        x_text, y_text = parts[-2:]
    else:
        raise StrategyParseError(
            "use: attack marine 55 45, attack move marine 55 45, or attack 55 45"
        )

    x, y = _parse_coordinates(x_text, y_text)
    return AttackMoveCommand(unit=unit, x=x, y=y)


def _parse_patrol(parts: list[str], default_unit: str) -> PatrolCommand:
    if len(parts) == 3:
        unit = normalize_unit(default_unit)
        x_text, y_text = parts[1:]
    elif len(parts) >= 4:
        unit = normalize_unit(" ".join(parts[1:-2]))
        x_text, y_text = parts[-2:]
    else:
        raise StrategyParseError("use: patrol marine 55 45 or patrol 55 45")
    x, y = _parse_coordinates(x_text, y_text)
    return PatrolCommand(unit=unit, x=x, y=y)


def _parse_unit_order(
    parts: list[str], command_type, default_unit: str
) -> HoldPositionCommand | StopCommand:
    normalizer = normalize_stop_actor if command_type is StopCommand else normalize_unit
    unit = (
        normalizer(" ".join(parts[1:])) if len(parts) > 1 else normalizer(default_unit)
    )
    return command_type(unit=unit)


def _parse_rally(parts: list[str]) -> RallyCommand:
    if len(parts) < 4:
        raise StrategyParseError("use: rally barracks 55 45")
    building = normalize_production_structure(" ".join(parts[1:-2]))
    x, y = _parse_coordinates(parts[-2], parts[-1])
    return RallyCommand(building=building, x=x, y=y)


def _is_enemy_attack(parts: list[str]) -> bool:
    lowered = [part.lower() for part in parts]
    return len(lowered) >= 2 and lowered[-1] in {"enemy", "enemies"}


def _parse_wait(parts: list[str]) -> WaitCommand | WaitUntilCommand:
    if len(parts) >= 2 and parts[1].lower() == "until":
        return _parse_wait_until(parts[2:])

    if len(parts) != 2:
        raise StrategyParseError("use: wait 2 or wait until minerals 100")

    try:
        seconds = float(parts[1])
    except ValueError as exc:
        raise StrategyParseError("wait duration must be a number of seconds") from exc

    if seconds < 0:
        raise StrategyParseError("wait duration must not be negative")

    return WaitCommand(seconds=seconds)


def _parse_wait_until(parts: list[str]) -> WaitUntilCommand:
    if not parts:
        raise StrategyParseError(
            "use: wait until minerals 100, wait until structure supply depot ready, or wait until unit marine 1"
        )

    metric = parts[0].lower().replace("-", "_")
    if metric in {"mineral", "minerals", "vespene", "gas"}:
        if len(parts) != 2:
            raise StrategyParseError(f"use: wait until {metric} 100")
        condition = "vespene" if metric in {"vespene", "gas"} else "minerals"
        return WaitUntilCommand(condition=condition, at_least=_parse_at_least(parts[1]))

    if metric in {"supply_left", "supply_used", "supply_cap", "supply"}:
        condition = metric
        if (
            metric == "supply"
            and len(parts) >= 2
            and parts[1].lower() in {"left", "used", "cap"}
        ):
            condition = f"supply_{parts[1].lower()}"
            value_parts = parts[2:]
        else:
            condition = "supply_left" if metric == "supply" else metric
            value_parts = parts[1:]
        if len(value_parts) != 1:
            raise StrategyParseError(
                "use: wait until supply left 1, supply used 20, or supply cap 31"
            )
        return WaitUntilCommand(
            condition=condition, at_least=_parse_at_least(value_parts[0])
        )

    if metric in {"structure", "building"}:
        return _parse_wait_until_structure(parts[1:])

    if metric in {"unit", "units"}:
        if len(parts) != 3:
            raise StrategyParseError("use: wait until unit marine 1")
        return WaitUntilCommand(
            condition="unit_count",
            target=normalize_wait_unit(parts[1]),
            at_least=_parse_at_least(parts[2]),
        )

    if metric in {"townhall", "townhalls", "base", "bases"}:
        if len(parts) != 2:
            raise StrategyParseError("use: wait until townhalls 2")
        return WaitUntilCommand(
            condition="townhall_count", at_least=_parse_at_least(parts[1])
        )

    if metric in {"upgrade", "research"}:
        if len(parts) < 2:
            raise StrategyParseError("use: wait until upgrade stimpack complete")
        upgrade_parts = parts[1:]
        if upgrade_parts[-1].lower() in {"ready", "complete", "completed", "done"}:
            upgrade_parts = upgrade_parts[:-1]
        if not upgrade_parts:
            raise StrategyParseError("upgrade wait is missing the upgrade name")
        return WaitUntilCommand(
            condition="upgrade_complete",
            target=normalize_upgrade(" ".join(upgrade_parts)),
            at_least=1,
        )

    if metric in {"time", "game_time", "seconds"}:
        if len(parts) != 2:
            raise StrategyParseError("use: wait until game_time 120")
        return WaitUntilCommand(
            condition="game_time", at_least=_parse_at_least(parts[1])
        )

    if metric in {"enemy_race", "race", "matchup"}:
        if len(parts) != 2:
            raise StrategyParseError("use: wait until enemy_race zerg")
        return WaitUntilCommand(
            condition="enemy_race",
            target=normalize_enemy_race(parts[1]),
            at_least=1,
        )

    if metric in {"alert", "alert_active"}:
        if len(parts) < 2:
            raise StrategyParseError(
                "use: wait until alert nuclear_launch_detected"
            )
        return WaitUntilCommand(
            condition="alert_active",
            target=normalize_alert(" ".join(parts[1:])),
            at_least=1,
        )

    raise StrategyParseError(f"unsupported wait-until condition: {parts[0]}")


def _parse_wait_until_structure(parts: list[str]) -> WaitUntilCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: wait until structure supply depot ready")

    status_words = {
        "ready",
        "complete",
        "completed",
        "pending",
        "started",
        "count",
        "total",
    }
    status_index = next(
        (index for index, part in enumerate(parts) if part.lower() in status_words),
        None,
    )
    if status_index is None:
        raise StrategyParseError(
            "structure wait must end with ready, pending, started, count, or total"
        )

    building_text = " ".join(parts[:status_index])
    if not building_text:
        raise StrategyParseError("structure wait is missing the structure name")

    status = parts[status_index].lower()
    remaining = parts[status_index + 1 :]
    if len(remaining) > 1:
        raise StrategyParseError("structure wait count must be a single number")
    at_least = _parse_at_least(remaining[0]) if remaining else 1

    if status in {"ready", "complete", "completed"}:
        condition = "structure_ready"
    elif status in {"pending", "started"}:
        condition = "structure_pending"
    else:
        condition = "structure_count"

    return WaitUntilCommand(
        condition=condition,
        target=normalize_structure_target(building_text),
        at_least=at_least,
    )


def _parse_gather(
    parts: list[str], default_unit: str
) -> GatherMineralsCommand | GatherGasCommand:
    workers = None
    if len(parts) > 2 and _looks_like_positive_int(parts[-1]):
        workers = _parse_positive_int(
            parts[-1], "gather worker count", MAX_WORKER_ASSIGNMENT_COUNT
        )
        parts = parts[:-1]

    if len(parts) == 2:
        unit = default_unit
        resource = parts[1]
    elif len(parts) == 3:
        unit = normalize_unit(parts[1])
        resource = parts[2]
    else:
        raise StrategyParseError("use: gather minerals or gather worker minerals")

    if unit != "worker":
        raise StrategyParseError("only workers can gather resources in this MVP")

    normalized_resource = resource.strip().lower()
    if normalized_resource in {"mineral", "minerals", "미네랄"}:
        return GatherMineralsCommand(unit=unit, workers=workers)
    if normalized_resource in {"gas", "vespene", "vespene_gas", "가스", "베스핀"}:
        return GatherGasCommand(unit=unit, workers=workers)

    raise StrategyParseError("only mineral and gas gathering are supported in this MVP")


def _parse_return_cargo(parts: list[str], default_unit: str) -> ReturnCargoCommand:
    if len(parts) >= 2 and parts[1].lower() == "cargo":
        unit_parts = parts[2:]
    else:
        unit_parts = parts[1:]
    unit = (
        normalize_unit(" ".join(unit_parts))
        if unit_parts
        else normalize_unit(default_unit)
    )
    return ReturnCargoCommand(unit=unit)


def _parse_distribute_workers(parts: list[str]) -> DistributeWorkersCommand:
    remaining = [part.lower() for part in parts[1:]]
    if remaining and remaining[0] in {"worker", "workers", "scv", "scvs", "일꾼"}:
        remaining = remaining[1:]
    if len(remaining) > 1:
        raise StrategyParseError("use: distribute workers or distribute workers 2")
    ratio = (
        _parse_non_negative_number(remaining[0], "mineral-to-gas ratio")
        if remaining
        else 2.0
    )
    if ratio > 20:
        raise StrategyParseError("mineral-to-gas ratio is too high")
    return DistributeWorkersCommand(mineral_to_gas_ratio=ratio)


def _parse_train(parts: list[str]) -> TrainUnitCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: train scv, train marine 2, or train siege tank")

    count = 1
    unit_parts = parts[1:]
    if len(unit_parts) > 1 and _looks_like_positive_int(unit_parts[-1]):
        count = _parse_positive_int(unit_parts[-1], "train count", MAX_SELECTION_COUNT)
        unit_parts = unit_parts[:-1]
    return TrainUnitCommand(
        unit=normalize_train_unit(" ".join(unit_parts)), count=count
    )


def _parse_build(parts: list[str]) -> BuildStructureCommand | BuildAddonCommand:
    if len(parts) < 2:
        raise StrategyParseError(
            "use: build supply depot, build barracks 2, or build refinery"
        )

    count = 1
    building_parts = parts[1:]
    if len(building_parts) > 1 and _looks_like_positive_int(building_parts[-1]):
        count = _parse_positive_int(
            building_parts[-1], "build count", MAX_STRUCTURE_ACTION_COUNT
        )
        building_parts = building_parts[:-1]
    target = " ".join(building_parts)
    try:
        return BuildStructureCommand(building=normalize_building(target), count=count)
    except StrategyParseError as building_error:
        try:
            return BuildAddonCommand(addon=normalize_addon(target), count=count)
        except StrategyParseError:
            raise building_error


def _parse_expand(parts: list[str]) -> ExpandCommand:
    if len(parts) > 2:
        raise StrategyParseError("use: expand or expand 2")
    count = (
        _parse_positive_int(parts[1], "expand count", MAX_STRUCTURE_ACTION_COUNT)
        if len(parts) == 2
        else 1
    )
    return ExpandCommand(count=count)


def _parse_addon(parts: list[str]) -> BuildAddonCommand:
    if len(parts) < 2:
        raise StrategyParseError(
            "use: addon barracks tech lab or addon factory reactor"
        )
    count = 1
    addon_parts = parts[1:]
    if len(addon_parts) > 1 and _looks_like_positive_int(addon_parts[-1]):
        count = _parse_positive_int(
            addon_parts[-1], "add-on count", MAX_STRUCTURE_ACTION_COUNT
        )
        addon_parts = addon_parts[:-1]
    return BuildAddonCommand(addon=normalize_addon(" ".join(addon_parts)), count=count)


def _parse_morph(parts: list[str]) -> MorphStructureCommand:
    if len(parts) < 2:
        raise StrategyParseError(
            "use: morph orbital command or morph planetary fortress"
        )
    return MorphStructureCommand(building=normalize_morph(" ".join(parts[1:])))


def _parse_research(parts: list[str]) -> ResearchUpgradeCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: research stimpack")
    return ResearchUpgradeCommand(upgrade=normalize_upgrade(" ".join(parts[1:])))


def _parse_repair(parts: list[str]) -> RepairCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: repair barracks or repair siege tank 2")
    workers = 1
    target_parts = parts[1:]
    if len(target_parts) > 1 and _looks_like_positive_int(target_parts[-1]):
        workers = _parse_positive_int(
            target_parts[-1], "repair worker count", MAX_WORKER_ASSIGNMENT_COUNT
        )
        target_parts = target_parts[:-1]
    return RepairCommand(
        target=normalize_repair_target(" ".join(target_parts)), workers=workers
    )


def _parse_use_ability(parts: list[str]) -> UseAbilityCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: use_ability stim_marine [actor] [location]")
    if (
        len(parts) >= 3
        and normalize_name(parts[1]) == "stim"
        and normalize_name(parts[2]) in {"marine", "marauder"}
    ):
        ability = normalize_ability("_".join(parts[1:3]))
        rest_start = 3
    else:
        try:
            ability = normalize_ability(parts[1])
            rest_start = 2
        except StrategyParseError:
            if len(parts) < 3:
                raise
            ability = normalize_ability("_".join(parts[1:3]))
            rest_start = 3
    spec = ABILITY_SPECS[ability]
    rest = [
        part for part in parts[rest_start:] if part.lower() not in {"with", "count"}
    ]
    if len(rest) > 1 and _looks_like_positive_int(rest[-1]):
        rest = rest[:-1]
    actor = (
        normalize_ability_actor(" ".join(rest))
        if rest and spec.target_kind == "none"
        else (spec.actors[0] if spec.actors and spec.actors[0] != "any" else None)
    )
    location = (
        _parse_location_words(rest)
        if spec.target_kind in {"point", "mineral"}
        else None
    )
    target_unit = (
        (
            normalize_enemy_target(" ".join(rest))
            if spec.target_alliance == "enemy"
            else normalize_target_unit(" ".join(rest))
        )
        if rest and spec.target_kind == "unit"
        else None
    )
    return UseAbilityCommand(
        ability=ability, actor=actor, target_unit=target_unit, location=location
    )


def _parse_location_words(words: list[str]) -> LocationRef:
    if not words:
        raise StrategyParseError("location is required")
    if len(words) == 2:
        try:
            x, y = _parse_coordinates(words[0], words[1])
            return _normalize_location_ref(LocationRef(x=x, y=y))
        except StrategyParseError:
            pass
    return _normalize_location_ref(
        LocationRef(semantic=normalize_location(" ".join(words)))
    )


def _parse_transform(parts: list[str]) -> TransformCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: transform siege_mode")
    ability = normalize_ability(parts[-1])
    if not _ability_in_family(ability, _TRANSFORM_ABILITIES):
        raise StrategyParseError(f"unsupported transform ability: {ability}")
    actor_text = " ".join(parts[1:-1])
    spec = ABILITY_SPECS[ability]
    actor = (
        normalize_ability_actor(actor_text)
        if actor_text
        else (spec.actors[0] if spec.actors and spec.actors[0] != "any" else None)
    )
    return TransformCommand(ability=ability, actor=actor)


def _parse_land(parts: list[str]) -> LandCommand:
    if len(parts) < 3:
        raise StrategyParseError("use: land barracks proxy or land barracks 55 45")
    actor = normalize_ability_actor(parts[1])
    return LandCommand(actor=actor, location=_parse_location_words(parts[2:]))


def _parse_load(parts: list[str]) -> LoadCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: load medivac marine or load command_center")
    actor = normalize_ability_actor(parts[1])
    if len(parts) == 2:
        if actor not in {"command_center", "orbital_command"}:
            raise StrategyParseError(
                "medivac and bunker load commands require a target unit"
            )
        return LoadCommand(actor=actor)
    rest = parts[2:]
    count = None
    if len(rest) > 1 and _looks_like_positive_int(rest[-1]):
        count = _parse_positive_int(rest[-1], "load count", MAX_SELECTION_COUNT)
        rest = rest[:-1]
    return LoadCommand(
        actor=actor,
        target_unit=normalize_target_unit(" ".join(rest)),
        count=count,
    )


def _parse_unload(parts: list[str]) -> UnloadCommand:
    if len(parts) < 2:
        raise StrategyParseError("use: unload medivac [location|unit]")
    actor = normalize_ability_actor(parts[1])
    rest = parts[2:]
    if not rest:
        return UnloadCommand(actor=actor)
    try:
        return UnloadCommand(actor=actor, location=_parse_location_words(rest))
    except StrategyParseError:
        return UnloadCommand(
            actor=actor, target_unit=normalize_target_unit(" ".join(rest))
        )


_TRANSFORM_ABILITIES = frozenset(TRANSFORM_ABILITY_KEYS)


def _ability_in_family(ability: str, family: set[str] | frozenset[str]) -> bool:
    return ability in family


def _action_from_dict(
    payload: Any, default_unit: str, control_depth: int = 0
) -> StrategyAction:
    if not isinstance(payload, dict):
        raise StrategyParseError("each strategy JSON action must be an object")

    action_type = str(payload.get("type", "")).strip().lower()
    if action_type in {"conditional", "if"}:
        when_payload = payload.get("when", payload.get("condition"))
        if when_payload is None:
            raise StrategyParseError("conditional requires a 'when' condition")
        then_payload = payload.get("then_actions", payload.get("then"))
        else_payload = payload.get(
            "else_actions", payload.get("otherwise", payload.get("else", []))
        )
        return ConditionalCommand(
            when=_condition_expression_from_payload(when_payload),
            then_actions=_nested_actions_from_payload(
                then_payload,
                "conditional then_actions",
                default_unit,
                control_depth,
                allow_empty=False,
            ),
            else_actions=_nested_actions_from_payload(
                else_payload,
                "conditional else_actions",
                default_unit,
                control_depth,
                allow_empty=True,
            ),
        )

    if action_type in {"repeat", "repeat_until", "repeat-until"}:
        actions = _nested_actions_from_payload(
            payload.get("actions"),
            f"{action_type} actions",
            default_unit,
            control_depth,
            allow_empty=False,
        )
        if action_type == "repeat":
            cycles = _positive_int_from_payload(
                payload,
                "cycles",
                default=1,
                max_count=MAX_REPEAT_CYCLES,
            )
            on_exhausted = normalize_name(
                str(payload.get("on_exhausted", "replan"))
            )
            if on_exhausted not in {"replan", "fail"}:
                raise StrategyParseError(
                    "fixed repeat on_exhausted must be 'replan' or 'fail'"
                )
            return RepeatCommand(
                actions=actions,
                max_cycles=cycles,
                until=None,
                max_seconds=_bounded_number_from_payload(
                    payload,
                    "max_seconds",
                    default=MAX_POLICY_SECONDS,
                    minimum=1,
                    maximum=MAX_POLICY_SECONDS,
                ),
                on_exhausted=on_exhausted,
            )
        until_payload = payload.get("until", payload.get("condition"))
        if until_payload is None:
            raise StrategyParseError("repeat_until requires an 'until' condition")
        on_exhausted = normalize_name(str(payload.get("on_exhausted", "replan")))
        if on_exhausted not in {"replan", "fail", "continue"}:
            raise StrategyParseError(
                "repeat-until on_exhausted must be 'replan', 'fail', or 'continue'"
            )
        return RepeatCommand(
            actions=actions,
            max_cycles=_positive_int_from_payload(
                payload,
                "max_cycles",
                default=20,
                max_count=MAX_REPEAT_CYCLES,
            ),
            until=_condition_expression_from_payload(until_payload),
            max_seconds=_bounded_number_from_payload(
                payload,
                "max_seconds",
                default=300,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
            on_exhausted=on_exhausted,
        )

    if action_type in {"with_timeout", "with-timeout"}:
        on_timeout = normalize_name(str(payload.get("on_timeout", "replan")))
        if on_timeout not in {"replan", "fail"}:
            raise StrategyParseError(
                "with-timeout on_timeout must be 'replan' or 'fail'"
            )
        return WithTimeoutCommand(
            actions=_nested_actions_from_payload(
                payload.get("actions"),
                "with-timeout actions",
                default_unit,
                control_depth,
                allow_empty=False,
            ),
            timeout_seconds=_bounded_number_from_payload(
                payload,
                "timeout_seconds",
                default=120,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
            on_timeout=on_timeout,
        )
    if action_type in {
        "move",
        "move_target",
        "move-target",
        "follow",
        "move_and_wait",
        "move-and-wait",
    }:
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        is_target_action = action_type in {"move_target", "move-target", "follow"}
        wait_for_arrival = action_type in {"move_and_wait", "move-and-wait"}
        target_unit = (
            _optional_target_unit_from_payload(payload)
            if is_target_action or "target_unit" in payload
            else None
        )
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        if target_unit is not None or target_tag is not None:
            x, y, location = _optional_point_target_from_payload(payload)
            if location is not None or x is not None:
                raise StrategyParseError(
                    "move target must use target_unit/target_tag or a point, not both"
                )
        else:
            if is_target_action:
                raise StrategyParseError(
                    "move_target requires target_unit or target_tag"
                )
            x, y, location = _point_target_from_payload(payload)
        return MoveCommand(
            unit=unit,
            x=x,
            y=y,
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
            target_unit=target_unit,
            target_tag=target_tag,
            wait_for_arrival=wait_for_arrival,
            arrival_tolerance=_bounded_number_from_payload(
                payload,
                "arrival_tolerance",
                default=2.5,
                minimum=0.25,
                maximum=20,
            ),
            timeout_seconds=_bounded_number_from_payload(
                payload,
                "timeout_seconds",
                default=90,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
        )

    if action_type in {"attack_until_clear", "attack-until-clear"}:
        x, y, location = _point_target_from_payload(payload)
        resolved_location = location or LocationRef(x=x, y=y)
        target_value = payload.get("target_unit", payload.get("target"))
        on_timeout = normalize_name(str(payload.get("on_timeout", "replan")))
        if on_timeout not in {"replan", "fail"}:
            raise StrategyParseError(
                "attack-until-clear on_timeout must be 'replan' or 'fail'"
            )
        return AttackUntilClearCommand(
            unit=normalize_mobile_attack_unit(str(payload.get("unit", "marine"))),
            location=resolved_location,
            target_unit=(
                normalize_enemy_target(str(target_value))
                if target_value is not None
                else None
            ),
            selection=_selection_from_payload(payload),
            radius=_bounded_number_from_payload(
                payload, "radius", default=16, minimum=1, maximum=64
            ),
            arrival_tolerance=_bounded_number_from_payload(
                payload,
                "arrival_tolerance",
                default=5,
                minimum=0.5,
                maximum=20,
            ),
            clear_seconds=_bounded_number_from_payload(
                payload,
                "clear_seconds",
                default=2,
                minimum=0.25,
                maximum=30,
            ),
            timeout_seconds=_bounded_number_from_payload(
                payload,
                "timeout_seconds",
                default=180,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
            on_timeout=on_timeout,
        )

    if action_type in {"attack", "attack_move", "attack-move"}:
        target = str(payload.get("target", "")).strip().lower()
        if target in {"enemy", "nearest_enemy", "nearest enemy"} or any(
            key in payload for key in ("target_unit", "target_tag")
        ):
            return AttackEnemyCommand(
                unit=normalize_attack_actor(str(payload.get("unit", default_unit))),
                selection=_selection_from_payload(payload),
                queued=_bool_from_payload(payload, "queued", default=False),
                target_unit=_optional_attack_target_from_payload(payload),
                target_tag=_optional_tag_from_payload(payload, "target_tag"),
                target_alliance=_attack_target_alliance_from_payload(payload),
            )
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        x, y, location = _point_target_from_payload(payload)
        return AttackMoveCommand(
            unit=unit,
            x=x,
            y=y,
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {
        "attack_enemy",
        "attack-enemy",
        "attack_target",
        "attack-target",
        "focus_fire",
        "focus-fire",
    }:
        target_unit = _optional_attack_target_from_payload(payload)
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        target_alliance = _attack_target_alliance_from_payload(payload)
        if action_type in {"attack_enemy", "attack-enemy"} and target_alliance != "enemy":
            raise StrategyParseError("attack_enemy only supports enemy targets")
        if (
            action_type
            in {"attack_target", "attack-target", "focus_fire", "focus-fire"}
            and target_unit is None
            and target_tag is None
        ):
            raise StrategyParseError("attack_target requires target_unit or target_tag")
        return AttackEnemyCommand(
            unit=normalize_attack_actor(str(payload.get("unit", "marine"))),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
            target_unit=target_unit,
            target_tag=target_tag,
            target_alliance=target_alliance,
            wait_for_target_death=action_type in {"focus_fire", "focus-fire"},
            timeout_seconds=_bounded_number_from_payload(
                payload,
                "timeout_seconds",
                default=60,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
        )

    if action_type == "kite":
        target_unit = _optional_attack_target_from_payload(payload)
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        return KiteCommand(
            unit=normalize_mobile_attack_unit(str(payload.get("unit", "marine"))),
            target_unit=target_unit or "nearest_enemy",
            target_tag=target_tag,
            selection=_selection_from_payload(payload),
            duration_seconds=_bounded_number_from_payload(
                payload,
                "duration_seconds",
                default=8,
                minimum=0.25,
                maximum=30,
            ),
            retreat_distance=_bounded_number_from_payload(
                payload,
                "retreat_distance",
                default=2,
                minimum=0.5,
                maximum=10,
            ),
        )

    if action_type == "patrol":
        x, y, location = _point_target_from_payload(payload)
        return PatrolCommand(
            unit=normalize_unit(str(payload.get("unit", default_unit))),
            x=x,
            y=y,
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"hold", "hold_position", "hold-position"}:
        return HoldPositionCommand(
            unit=normalize_unit(str(payload.get("unit", default_unit))),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "stop":
        return StopCommand(
            unit=normalize_stop_actor(str(payload.get("unit", default_unit))),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"rally", "set_rally", "set-rally"}:
        x, y, location = _optional_point_target_from_payload(payload)
        target_unit = _optional_rally_target_from_payload(payload)
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        if (
            location is None
            and x is None
            and target_unit is None
            and target_tag is None
        ):
            raise StrategyParseError(
                "rally requires location, x/y, target_unit, target, or target_tag"
            )
        return RallyCommand(
            building=normalize_production_structure(
                str(payload.get("building", payload.get("producer", "")))
            ),
            x=x,
            y=y,
            location=location,
            target_unit=target_unit,
            target_tag=target_tag,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "wait":
        seconds = _required_number(payload, "seconds")
        if seconds < 0:
            raise StrategyParseError("wait duration must not be negative")
        return WaitCommand(seconds=seconds)

    if action_type in {"wait_until", "wait-until"}:
        return _wait_until_from_dict(payload)

    if action_type in {"wait_for_ability", "wait-for-ability"}:
        translated = dict(payload)
        translated["condition"] = "ability_available"
        translated["at_least"] = payload.get("count", payload.get("at_least", 1))
        translated.pop("count", None)
        return _wait_until_from_dict(translated)

    if action_type in {"wait_for_form", "wait-for-form"}:
        translated = dict(payload)
        translated["condition"] = "unit_form_count"
        translated["target"] = payload.get("form", payload.get("target"))
        translated["actor"] = payload.get("unit", payload.get("actor"))
        translated["at_least"] = payload.get("count", payload.get("at_least", 1))
        translated.pop("count", None)
        return _wait_until_from_dict(translated)

    if action_type in {"wait_for_idle", "wait-for-idle"}:
        translated = dict(payload)
        translated["condition"] = "idle_unit_count"
        translated["target"] = payload.get("unit", payload.get("target"))
        translated["at_least"] = payload.get("count", payload.get("at_least", 1))
        translated.pop("count", None)
        return _wait_until_from_dict(translated)

    if action_type in {"gather", "gather_minerals"}:
        resource = str(payload.get("resource", "minerals")).strip().lower()
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        if unit != "worker":
            raise StrategyParseError("only workers can gather resources in this MVP")
        if resource in {"mineral", "minerals", "미네랄"}:
            return GatherMineralsCommand(
                unit=unit,
                workers=_optional_positive_int_from_payload(
                    payload, ("workers", "count"), MAX_WORKER_ASSIGNMENT_COUNT
                ),
                location=_location_from_payload(payload, required=False),
                target_tag=_optional_tag_from_payload(payload, "target_tag"),
                selection=_selection_from_payload(payload),
                queued=_bool_from_payload(payload, "queued", default=False),
            )
        if resource in {"gas", "vespene", "vespene_gas", "가스", "베스핀"}:
            return GatherGasCommand(
                unit=unit,
                workers=_optional_positive_int_from_payload(
                    payload, ("workers", "count"), MAX_WORKER_ASSIGNMENT_COUNT
                ),
                location=_location_from_payload(payload, required=False),
                target_tag=_optional_tag_from_payload(payload, "target_tag"),
                selection=_selection_from_payload(payload),
                queued=_bool_from_payload(payload, "queued", default=False),
            )
        raise StrategyParseError(
            "only mineral and gas gathering are supported in this MVP"
        )

    if action_type in {"gather_gas", "gather-gas", "gather_vespene"}:
        unit = normalize_unit(str(payload.get("unit", default_unit)))
        if unit != "worker":
            raise StrategyParseError("only workers can gather gas in this MVP")
        return GatherGasCommand(
            unit=unit,
            workers=_optional_positive_int_from_payload(
                payload, ("workers", "count"), MAX_WORKER_ASSIGNMENT_COUNT
            ),
            location=_location_from_payload(payload, required=False),
            target_tag=_optional_tag_from_payload(payload, "target_tag"),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"return_cargo", "return-cargo", "return"}:
        return ReturnCargoCommand(
            unit=normalize_unit(str(payload.get("unit", default_unit))),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"distribute_workers", "distribute-workers", "distribute"}:
        ratio = _optional_number(
            payload,
            ("mineral_to_gas_ratio", "resource_ratio", "ratio"),
            default=2,
        )
        if ratio < 0 or ratio > 20:
            raise StrategyParseError("mineral-to-gas ratio must be between 0 and 20")
        return DistributeWorkersCommand(mineral_to_gas_ratio=ratio)

    if action_type == "train":
        return TrainUnitCommand(
            unit=normalize_train_unit(str(payload.get("unit", ""))),
            count=_positive_int_from_payload(
                payload, "count", default=1, max_count=MAX_SELECTION_COUNT
            ),
            producer_selection=_selection_from_payload_key(
                payload, "producer_selection"
            ),
        )

    if action_type in {
        "produce_until",
        "produce-until",
        "maintain_production",
        "maintain-production",
    }:
        return ProductionPolicyCommand(
            unit=normalize_train_unit(str(payload.get("unit", ""))),
            target_count=_positive_int_from_payload(
                payload,
                "target_count",
                default=1,
                max_count=MAX_SELECTION_COUNT,
            ),
            background=action_type in {"maintain_production", "maintain-production"},
            producer_selection=_selection_from_payload_key(
                payload, "producer_selection"
            ),
            reserve_minerals=_bounded_int_from_payload(
                payload, "reserve_minerals", default=0, minimum=0, maximum=10000
            ),
            reserve_vespene=_bounded_int_from_payload(
                payload, "reserve_vespene", default=0, minimum=0, maximum=10000
            ),
            reserve_supply=_bounded_int_from_payload(
                payload, "reserve_supply", default=0, minimum=0, maximum=200
            ),
            max_seconds=_bounded_number_from_payload(
                payload,
                "max_seconds",
                default=300,
                minimum=1,
                maximum=MAX_POLICY_SECONDS,
            ),
        )

    if action_type in {"stop_production", "stop-production"}:
        unit_value = payload.get("unit")
        return StopProductionCommand(
            unit=(
                normalize_train_unit(str(unit_value))
                if unit_value is not None
                else None
            )
        )

    if action_type == "build":
        return BuildStructureCommand(
            building=normalize_building(str(payload.get("building", ""))),
            worker=normalize_unit(str(payload.get("worker", "worker"))),
            count=_positive_int_from_payload(
                payload, "count", default=1, max_count=MAX_STRUCTURE_ACTION_COUNT
            ),
            location=_location_from_payload(payload, required=False),
            selection=_selection_from_payload(payload),
            placement_mode=_placement_mode_from_payload(payload),
            max_distance=_max_distance_from_payload(payload),
            reserve_addon_space=_bool_from_payload(
                payload, "reserve_addon_space", default=False
            ),
        )

    if action_type == "expand":
        return ExpandCommand(
            count=_positive_int_from_payload(
                payload, "count", default=1, max_count=MAX_STRUCTURE_ACTION_COUNT
            )
        )

    if action_type in {"build_addon", "build-addon", "addon"}:
        addon_value = str(payload.get("addon", ""))
        producer_value = str(payload.get("producer", "")).strip()
        if producer_value and addon_value:
            try:
                normalize_addon(addon_value)
            except StrategyParseError:
                addon_value = f"{producer_value} {addon_value}"
        return BuildAddonCommand(
            addon=normalize_addon(addon_value),
            count=_positive_int_from_payload(
                payload, "count", default=1, max_count=MAX_STRUCTURE_ACTION_COUNT
            ),
            selection=_selection_from_payload(payload),
        )

    if action_type == "morph":
        return MorphStructureCommand(
            building=normalize_morph(
                str(payload.get("building", payload.get("target", "")))
            ),
            selection=_selection_from_payload(payload),
        )

    if action_type in {"research", "upgrade"}:
        return ResearchUpgradeCommand(
            upgrade=normalize_upgrade(str(payload.get("upgrade", ""))),
            researcher_selection=_selection_from_payload_key(
                payload, "researcher_selection"
            ),
        )

    if action_type == "repair":
        target_value = str(payload.get("target", "")).strip()
        target_selector_value = payload.get("target_selector")
        target_selector = (
            normalize_target_unit(str(target_selector_value))
            if target_selector_value is not None
            else None
        )
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        if not target_value and target_selector is None and target_tag is None:
            raise StrategyParseError(
                "repair requires target, target_selector, or target_tag"
            )
        return RepairCommand(
            target=normalize_repair_target(target_value) if target_value else None,
            workers=_positive_int_from_payload(
                payload, "workers", default=1, max_count=MAX_WORKER_ASSIGNMENT_COUNT
            ),
            target_tag=target_tag,
            target_selector=target_selector,
            target_selection=_selection_from_payload_key(payload, "target_selection"),
            selection=_selection_from_payload(payload),
        )

    if action_type in {"use_ability", "use-ability", "ability"}:
        ability = normalize_ability(str(payload.get("ability", "")))
        spec = ABILITY_SPECS[ability]
        location = (
            _location_from_payload(payload, required=ability == "scan")
            if spec.target_kind in {"point", "mineral"}
            else None
        )
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        has_named_target = "target" in payload or "target_unit" in payload
        if spec.target_kind == "unit" and has_named_target:
            target_unit = (
                _optional_attack_target_from_payload(payload)
                if spec.target_alliance == "enemy"
                else _target_unit_from_payload(payload)
            )
        elif spec.target_kind == "none" and has_named_target:
            target_unit = _target_unit_from_payload(payload)
        else:
            target_unit = None
        if spec.target_kind == "none" and (
            "location" in payload or "x" in payload or "y" in payload
        ):
            raise StrategyParseError(f"ability {ability} does not take a point target")
        if spec.target_kind in {"point", "mineral"} and (
            "target_unit" in payload or "target_tag" in payload
        ):
            raise StrategyParseError(f"ability {ability} requires a location target")
        if spec.target_kind == "unit" and any(
            key in payload for key in ("location", "x", "y")
        ):
            raise StrategyParseError(f"ability {ability} requires a unit target")
        if spec.target_kind == "unit" and target_unit is None and target_tag is None:
            raise StrategyParseError(
                f"ability {ability} requires target_unit or target_tag"
            )
        return UseAbilityCommand(
            ability=ability,
            actor=_optional_actor_from_payload(payload),
            target_unit=target_unit,
            target_tag=target_tag,
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "scan":
        location = _location_from_payload(payload, required=True)
        if location is None:
            raise StrategyParseError("scan requires a location or x/y coordinates")
        return ScanCommand(
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"call_down_mule", "call-down-mule", "mule"}:
        location = _location_from_payload(payload, required=True)
        if location is None:
            raise StrategyParseError(
                "call_down_mule requires a location or x/y coordinates"
            )
        return CallDownMuleCommand(
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"supply_drop", "supply-drop"}:
        target_value = payload.get("target_unit", payload.get("target"))
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        return SupplyDropCommand(
            target_unit=normalize_target_unit(str(target_value or "supply_depot")),
            target_tag=target_tag,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"transform", "mode"}:
        ability = normalize_ability(
            str(payload.get("ability", payload.get("mode", "")))
        )
        if ability not in _TRANSFORM_ABILITIES:
            raise StrategyParseError(f"unsupported transform ability: {ability}")
        return TransformCommand(
            ability=ability,
            actor=_optional_actor_from_payload(payload)
            or _default_actor_for_ability(ability),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "lift":
        actor = normalize_ability_actor(
            str(payload.get("actor", payload.get("building", "")))
        )
        return LiftCommand(
            actor=actor,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"land", "land_on_addon", "land-on-addon"}:
        actor = normalize_ability_actor(
            str(payload.get("actor", payload.get("building", "")))
        )
        target_addon = _optional_addon_from_payload(payload)
        target_addon_tag = _optional_tag_from_payload(payload, "target_addon_tag")
        location = _location_from_payload(
            payload, required=target_addon is None and target_addon_tag is None
        )
        if (
            action_type in {"land_on_addon", "land-on-addon"}
            and target_addon is None
            and target_addon_tag is None
        ):
            raise StrategyParseError(
                "land_on_addon requires target_addon or target_addon_tag"
            )
        if location is None and target_addon is None and target_addon_tag is None:
            raise StrategyParseError(
                "land requires location/x/y, target_addon, or target_addon_tag"
            )
        return LandCommand(
            actor=actor,
            location=location,
            target_addon=target_addon,
            target_addon_tag=target_addon_tag,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "load":
        actor = normalize_ability_actor(
            str(payload.get("actor", payload.get("transport", "")))
        )
        target_value = payload.get(
            "target_unit", payload.get("unit", payload.get("target"))
        )
        target_tag = _optional_tag_from_payload(payload, "target_tag")
        target_selection = _selection_from_payload_key(payload, "target_selection")
        if (
            target_value is None
            and target_tag is None
            and target_selection is not None
            and actor in {"command_center", "orbital_command"}
        ):
            target_value = "worker"
        if (
            target_value is None
            and target_tag is None
            and actor not in {"command_center", "orbital_command"}
        ):
            raise StrategyParseError(
                "medivac and bunker load commands require target_unit or target_tag"
            )
        return LoadCommand(
            actor=actor,
            target_unit=(
                normalize_target_unit(str(target_value))
                if target_value is not None
                else None
            ),
            target_tag=target_tag,
            target_selection=target_selection,
            count=_optional_positive_int_from_payload(
                payload, ("count",), MAX_SELECTION_COUNT
            ),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "unload":
        return UnloadCommand(
            actor=normalize_ability_actor(
                str(payload.get("actor", payload.get("transport", "")))
            ),
            target_unit=normalize_target_unit(
                str(payload.get("target_unit", payload.get("unit", "")))
            )
            if payload.get("target_unit", payload.get("unit"))
            else None,
            passenger_tag=_optional_tag_from_payload(payload, "passenger_tag"),
            location=_location_from_payload(payload, required=False),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "cancel":
        ability = normalize_cancel_ability(
            str(
                payload.get(
                    "ability",
                    payload.get("cancel", payload.get("target", "cancel_any")),
                )
            )
        )
        return CancelCommand(
            ability=ability,
            actor=_optional_actor_from_payload(payload),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "salvage":
        return SalvageCommand(
            actor=normalize_ability_actor(
                str(
                    payload.get(
                        "actor", payload.get("building", payload.get("target", ""))
                    )
                )
            ),
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"build_nuke", "build-nuke"}:
        return BuildNukeCommand(
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type in {"launch_nuke", "launch-nuke"}:
        location = _location_from_payload(payload, required=True)
        if location is None:
            raise StrategyParseError(
                "launch_nuke requires a location or x/y coordinates"
            )
        return LaunchNukeCommand(
            location=location,
            selection=_selection_from_payload(payload),
            queued=_bool_from_payload(payload, "queued", default=False),
        )

    if action_type == "replan":
        return ReplanCommand(reason=str(payload.get("reason", "requested")))

    raise StrategyParseError(f"unsupported JSON action type: {action_type!r}")


def _action_to_dict(action: StrategyAction) -> dict[str, object]:
    if isinstance(action, MoveCommand):
        if action.wait_for_arrival:
            if action.target_unit is not None or action.target_tag is not None:
                payload = _unit_command_payload(
                    "move_and_wait", action.unit, action.selection, action.queued
                )
                if action.target_unit is not None:
                    payload["target_unit"] = action.target_unit
                if action.target_tag is not None:
                    payload["target_tag"] = action.target_tag
            else:
                payload = _point_command_payload(
                    "move_and_wait",
                    action.unit,
                    action.x,
                    action.y,
                    action.location,
                    action.selection,
                    action.queued,
                )
            if action.arrival_tolerance != 2.5:
                payload["arrival_tolerance"] = action.arrival_tolerance
            if action.timeout_seconds != 90:
                payload["timeout_seconds"] = action.timeout_seconds
            return payload
        if action.target_unit is not None or action.target_tag is not None:
            payload = _unit_command_payload(
                "move_target",
                action.unit,
                action.selection,
                action.queued,
            )
            if action.target_unit is not None:
                payload["target_unit"] = action.target_unit
            if action.target_tag is not None:
                payload["target_tag"] = action.target_tag
            return payload
        return _point_command_payload(
            "move",
            action.unit,
            action.x,
            action.y,
            action.location,
            action.selection,
            action.queued,
        )
    if isinstance(action, AttackMoveCommand):
        return _point_command_payload(
            "attack",
            action.unit,
            action.x,
            action.y,
            action.location,
            action.selection,
            action.queued,
        )
    if isinstance(action, AttackEnemyCommand):
        payload = _unit_command_payload(
            (
                "focus_fire"
                if action.wait_for_target_death
                else (
                    "attack_target"
                    if action.target_unit is not None or action.target_tag is not None
                    else "attack_enemy"
                )
            ),
            action.unit,
            action.selection,
            action.queued,
        )
        if action.target_unit is not None:
            payload["target_unit"] = action.target_unit
        if action.target_tag is not None:
            payload["target_tag"] = action.target_tag
        if action.target_alliance != "enemy":
            payload["target_alliance"] = action.target_alliance
        if action.wait_for_target_death and action.timeout_seconds != 60:
            payload["timeout_seconds"] = action.timeout_seconds
        return payload
    if isinstance(action, AttackUntilClearCommand):
        payload = _point_command_payload(
            "attack_until_clear",
            action.unit,
            action.location.x,
            action.location.y,
            action.location,
            action.selection,
            False,
        )
        if action.target_unit is not None:
            payload["target_unit"] = action.target_unit
        if action.radius != 16:
            payload["radius"] = action.radius
        if action.arrival_tolerance != 5:
            payload["arrival_tolerance"] = action.arrival_tolerance
        if action.clear_seconds != 2:
            payload["clear_seconds"] = action.clear_seconds
        if action.timeout_seconds != 180:
            payload["timeout_seconds"] = action.timeout_seconds
        if action.on_timeout != "replan":
            payload["on_timeout"] = action.on_timeout
        return payload
    if isinstance(action, KiteCommand):
        payload = _unit_command_payload("kite", action.unit, action.selection, False)
        payload["target_unit"] = action.target_unit
        if action.target_tag is not None:
            payload["target_tag"] = action.target_tag
        if action.duration_seconds != 8:
            payload["duration_seconds"] = action.duration_seconds
        if action.retreat_distance != 2:
            payload["retreat_distance"] = action.retreat_distance
        return payload
    if isinstance(action, PatrolCommand):
        return _point_command_payload(
            "patrol",
            action.unit,
            action.x,
            action.y,
            action.location,
            action.selection,
            action.queued,
        )
    if isinstance(action, HoldPositionCommand):
        return _unit_command_payload(
            "hold", action.unit, action.selection, action.queued
        )
    if isinstance(action, StopCommand):
        return _unit_command_payload(
            "stop", action.unit, action.selection, action.queued
        )
    if isinstance(action, RallyCommand):
        rally_payload: dict[str, object] = {
            "type": "rally",
            "building": action.building,
        }
        if action.location is not None:
            rally_payload.update(_location_to_payload(action.location))
        elif action.x is not None and action.y is not None:
            rally_payload.update({"x": action.x, "y": action.y})
        if action.target_unit is not None:
            rally_payload["target_unit"] = action.target_unit
        if action.target_tag is not None:
            rally_payload["target_tag"] = action.target_tag
        if not any(
            key in rally_payload
            for key in ("location", "x", "target_unit", "target_tag")
        ):
            raise TypeError("rally requires a point or target")
        _add_selection_and_queue(rally_payload, action.selection, action.queued)
        return rally_payload
    if isinstance(action, WaitCommand):
        return {"type": "wait", "seconds": action.seconds}
    if isinstance(action, WaitUntilCommand):
        wait_payload: dict[str, object] = {
            "type": "wait_until",
            "condition": action.condition,
        }
        wait_payload.update(
            _comparison_threshold_payload(action.comparison, action.at_least)
        )
        if action.target is not None:
            wait_payload["target"] = action.target
        if action.ability is not None:
            wait_payload["ability"] = action.ability
        if action.actor is not None:
            wait_payload["actor"] = action.actor
        if action.location is not None:
            wait_payload.update(_location_to_payload(action.location))
        if action.radius != 12:
            wait_payload["radius"] = action.radius
        if action.selection is not None:
            wait_payload["selection"] = _selection_to_payload(action.selection)
        if action.timeout_seconds != 120:
            wait_payload["timeout_seconds"] = action.timeout_seconds
        if action.on_timeout != "replan":
            wait_payload["on_timeout"] = action.on_timeout
        return wait_payload
    if isinstance(action, GatherMineralsCommand):
        gather_payload: dict[str, object] = {
            "type": "gather",
            "unit": action.unit,
            "resource": "minerals",
        }
        if action.workers is not None:
            gather_payload["workers"] = action.workers
        if action.location is not None:
            gather_payload.update(_location_to_payload(action.location))
        if action.target_tag is not None:
            gather_payload["target_tag"] = action.target_tag
        _add_selection_and_queue(gather_payload, action.selection, action.queued)
        return gather_payload
    if isinstance(action, GatherGasCommand):
        gather_gas_payload: dict[str, object] = {
            "type": "gather",
            "unit": action.unit,
            "resource": "vespene",
        }
        if action.workers is not None:
            gather_gas_payload["workers"] = action.workers
        if action.location is not None:
            gather_gas_payload.update(_location_to_payload(action.location))
        if action.target_tag is not None:
            gather_gas_payload["target_tag"] = action.target_tag
        _add_selection_and_queue(gather_gas_payload, action.selection, action.queued)
        return gather_gas_payload
    if isinstance(action, ReturnCargoCommand):
        payload = {"type": "return_cargo", "unit": action.unit}
        _add_selection_and_queue(payload, action.selection, action.queued)
        return payload
    if isinstance(action, DistributeWorkersCommand):
        return {
            "type": "distribute_workers",
            "mineral_to_gas_ratio": action.mineral_to_gas_ratio,
        }
    if isinstance(action, TrainUnitCommand):
        train_payload: dict[str, object] = {"type": "train", "unit": action.unit}
        if action.count != 1:
            train_payload["count"] = action.count
        if action.producer_selection is not None:
            train_payload["producer_selection"] = _selection_to_payload(
                action.producer_selection
            )
        return train_payload
    if isinstance(action, ProductionPolicyCommand):
        policy_payload: dict[str, object] = {
            "type": "maintain_production" if action.background else "produce_until",
            "unit": action.unit,
            "target_count": action.target_count,
        }
        if action.producer_selection is not None:
            policy_payload["producer_selection"] = _selection_to_payload(
                action.producer_selection
            )
        if action.reserve_minerals:
            policy_payload["reserve_minerals"] = action.reserve_minerals
        if action.reserve_vespene:
            policy_payload["reserve_vespene"] = action.reserve_vespene
        if action.reserve_supply:
            policy_payload["reserve_supply"] = action.reserve_supply
        if action.max_seconds != 300:
            policy_payload["max_seconds"] = action.max_seconds
        return policy_payload
    if isinstance(action, StopProductionCommand):
        stop_policy_payload: dict[str, object] = {"type": "stop_production"}
        if action.unit is not None:
            stop_policy_payload["unit"] = action.unit
        return stop_policy_payload
    if isinstance(action, BuildStructureCommand):
        build_payload: dict[str, object] = {
            "type": "build",
            "building": action.building,
            "worker": action.worker,
        }
        if action.count != 1:
            build_payload["count"] = action.count
        if action.location is not None:
            build_payload.update(_location_to_payload(action.location))
        if action.placement_mode != "near":
            build_payload["placement_mode"] = action.placement_mode
        if action.max_distance != 20:
            build_payload["max_distance"] = action.max_distance
        if action.reserve_addon_space:
            build_payload["reserve_addon_space"] = True
        _add_selection_and_queue(build_payload, action.selection, False)
        return build_payload
    if isinstance(action, ExpandCommand):
        expand_payload: dict[str, object] = {"type": "expand"}
        if action.count != 1:
            expand_payload["count"] = action.count
        return expand_payload
    if isinstance(action, BuildAddonCommand):
        addon_payload: dict[str, object] = {
            "type": "build_addon",
            "addon": action.addon,
        }
        if action.count != 1:
            addon_payload["count"] = action.count
        _add_selection_and_queue(addon_payload, action.selection, False)
        return addon_payload
    if isinstance(action, MorphStructureCommand):
        morph_payload: dict[str, object] = {
            "type": "morph",
            "building": action.building,
        }
        _add_selection_and_queue(morph_payload, action.selection, False)
        return morph_payload
    if isinstance(action, ResearchUpgradeCommand):
        research_payload: dict[str, object] = {
            "type": "research",
            "upgrade": action.upgrade,
        }
        if action.researcher_selection is not None:
            research_payload["researcher_selection"] = _selection_to_payload(
                action.researcher_selection
            )
        return research_payload
    if isinstance(action, RepairCommand):
        repair_payload: dict[str, object] = {
            "type": "repair",
            "workers": action.workers,
        }
        if action.target is not None:
            repair_payload["target"] = action.target
        if action.target_tag is not None:
            repair_payload["target_tag"] = action.target_tag
        if action.target_selector is not None:
            repair_payload["target_selector"] = action.target_selector
        if action.target_selection is not None:
            repair_payload["target_selection"] = _selection_to_payload(
                action.target_selection
            )
        _add_selection_and_queue(repair_payload, action.selection, False)
        return repair_payload
    if isinstance(action, UseAbilityCommand):
        payload = _ability_payload(
            "use_ability",
            action.ability,
            action.actor,
            None,
            action.location,
            action.selection,
            action.queued,
        )
        if action.target_unit is not None:
            payload["target"] = action.target_unit
        if action.target_tag is not None:
            payload["target_tag"] = action.target_tag
        if (
            not action.queued
            and action.location is not None
            and action.location.x is not None
        ):
            payload["queued"] = False
        return payload
    if isinstance(action, ScanCommand):
        return _ability_payload(
            "scan", None, None, None, action.location, action.selection, action.queued
        )
    if isinstance(action, CallDownMuleCommand):
        return _ability_payload(
            "call_down_mule",
            None,
            None,
            None,
            action.location,
            action.selection,
            action.queued,
        )
    if isinstance(action, SupplyDropCommand):
        payload = _ability_payload(
            "supply_drop", None, None, None, None, action.selection, action.queued
        )
        payload["target"] = action.target_unit
        if action.target_tag is not None:
            payload["target_tag"] = action.target_tag
        return payload
    if isinstance(action, TransformCommand):
        payload = _ability_payload(
            "transform", None, action.actor, None, None, action.selection, action.queued
        )
        payload["mode"] = _public_mode_for_ability(action.ability)
        return payload
    if isinstance(action, LiftCommand):
        payload = _ability_payload(
            "lift", None, None, None, None, action.selection, action.queued
        )
        payload["building"] = action.actor
        return payload
    if isinstance(action, LandCommand):
        payload = _ability_payload(
            "land_on_addon"
            if action.target_addon is not None or action.target_addon_tag is not None
            else "land",
            None,
            None,
            None,
            action.location,
            action.selection,
            action.queued,
        )
        payload["building"] = action.actor
        if action.target_addon is not None:
            payload["target_addon"] = action.target_addon
        if action.target_addon_tag is not None:
            payload["target_addon_tag"] = action.target_addon_tag
        return payload
    if isinstance(action, LoadCommand):
        payload = _ability_payload(
            "load", None, None, None, None, action.selection, action.queued
        )
        payload["transport"] = action.actor
        if action.target_unit is not None:
            payload["unit"] = action.target_unit
        if action.target_tag is not None:
            payload["target_tag"] = action.target_tag
        if action.target_selection is not None:
            payload["target_selection"] = _selection_to_payload(action.target_selection)
        if action.count is not None:
            payload["count"] = action.count
        return payload
    if isinstance(action, UnloadCommand):
        payload = _ability_payload(
            "unload", None, None, None, action.location, action.selection, action.queued
        )
        payload["transport"] = action.actor
        if action.target_unit is not None:
            payload["unit"] = action.target_unit
        if action.passenger_tag is not None:
            payload["passenger_tag"] = action.passenger_tag
        return payload
    if isinstance(action, CancelCommand):
        payload = _ability_payload(
            "cancel", None, action.actor, None, None, action.selection, action.queued
        )
        payload["target"] = action.ability.removeprefix("cancel_")
        return payload
    if isinstance(action, SalvageCommand):
        payload = _ability_payload(
            "salvage", None, None, None, None, action.selection, action.queued
        )
        payload["target"] = action.actor
        return payload
    if isinstance(action, BuildNukeCommand):
        return _ability_payload(
            "build_nuke", None, None, None, None, action.selection, action.queued
        )
    if isinstance(action, LaunchNukeCommand):
        return _ability_payload(
            "launch_nuke",
            None,
            None,
            None,
            action.location,
            action.selection,
            action.queued,
        )
    if isinstance(action, ConditionalCommand):
        conditional_payload: dict[str, object] = {
            "type": "conditional",
            "when": _condition_expression_to_payload(action.when),
            "then_actions": [
                _action_to_dict(child) for child in action.then_actions
            ],
        }
        if action.else_actions:
            conditional_payload["else_actions"] = [
                _action_to_dict(child) for child in action.else_actions
            ]
        return conditional_payload
    if isinstance(action, RepeatCommand):
        if action.until is None:
            payload = {
                "type": "repeat",
                "cycles": action.max_cycles,
                "actions": [_action_to_dict(child) for child in action.actions],
            }
            if action.max_seconds != MAX_POLICY_SECONDS:
                payload["max_seconds"] = action.max_seconds
            if action.on_exhausted != "replan":
                payload["on_exhausted"] = action.on_exhausted
            return payload
        payload = {
            "type": "repeat_until",
            "until": _condition_expression_to_payload(action.until),
            "actions": [_action_to_dict(child) for child in action.actions],
            "max_cycles": action.max_cycles,
        }
        if action.max_seconds != 300:
            payload["max_seconds"] = action.max_seconds
        if action.on_exhausted != "replan":
            payload["on_exhausted"] = action.on_exhausted
        return payload
    if isinstance(action, WithTimeoutCommand):
        timeout_payload: dict[str, object] = {
            "type": "with_timeout",
            "actions": [_action_to_dict(child) for child in action.actions],
        }
        if action.timeout_seconds != 120:
            timeout_payload["timeout_seconds"] = action.timeout_seconds
        if action.on_timeout != "replan":
            timeout_payload["on_timeout"] = action.on_timeout
        return timeout_payload
    if isinstance(action, ReplanCommand):
        return {"type": "replan", "reason": action.reason}
    raise TypeError(f"unsupported strategy action: {action!r}")


def _point_command_payload(
    action_type: str,
    unit: str | None,
    x: float | None,
    y: float | None,
    location: LocationRef | None,
    selection: SelectionSpec | None,
    queued: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {"type": action_type}
    if unit is not None:
        payload["unit"] = unit
    if location is not None:
        payload.update(_location_to_payload(location))
    elif x is not None and y is not None:
        payload.update({"x": x, "y": y})
    else:
        raise TypeError(
            f"{action_type} requires a semantic location or x/y coordinates"
        )
    _add_selection_and_queue(payload, selection, queued)
    return payload


def _unit_command_payload(
    action_type: str,
    unit: str,
    selection: SelectionSpec | None,
    queued: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {"type": action_type, "unit": unit}
    _add_selection_and_queue(payload, selection, queued)
    return payload


def _add_selection_and_queue(
    payload: dict[str, object], selection: SelectionSpec | None, queued: bool
) -> None:
    if selection is not None:
        payload["selection"] = _selection_to_payload(selection)
    if queued:
        payload["queued"] = True


def _selection_to_payload(selection: SelectionSpec) -> dict[str, object]:
    selection_payload: dict[str, object] = {"mode": selection.mode}
    if selection.count is not None:
        selection_payload["count"] = selection.count
    if selection.tags:
        selection_payload["tags"] = list(selection.tags)
    return selection_payload


def _comparison_threshold_payload(
    comparison: str, value: float
) -> dict[str, object]:
    if comparison == "gte":
        return {"at_least": value}
    if comparison == "lte":
        return {"at_most": value}
    if comparison == "eq":
        return {"equals": value}
    return {"comparison": comparison, "value": value}


def _condition_expression_to_payload(
    expression: ConditionExpression,
) -> dict[str, object]:
    if isinstance(expression, ConditionGroup):
        return {
            "match": expression.match,
            "conditions": [
                _condition_spec_to_payload(condition)
                for condition in expression.conditions
            ],
        }
    return _condition_spec_to_payload(expression)


def _condition_spec_to_payload(condition: ConditionSpec) -> dict[str, object]:
    payload: dict[str, object] = {"condition": condition.condition}
    payload.update(
        _comparison_threshold_payload(condition.comparison, condition.value)
    )
    if condition.target is not None:
        payload["target"] = condition.target
    if condition.ability is not None:
        payload["ability"] = condition.ability
    if condition.actor is not None:
        payload["actor"] = condition.actor
    if condition.location is not None:
        payload.update(_location_to_payload(condition.location))
    if condition.radius != 12:
        payload["radius"] = condition.radius
    if condition.selection is not None:
        payload["selection"] = _selection_to_payload(condition.selection)
    return payload


def _public_mode_for_ability(ability: str) -> str:
    return {
        "siege_mode": "siege",
        "unsiege_mode": "unsiege",
        "morph_hellbat": "hellbat",
        "morph_hellion": "hellion",
        "thor_high_impact_mode": "high_impact",
        "thor_explosive_mode": "explosive",
        "viking_assault_mode": "assault",
        "viking_fighter_mode": "fighter",
        "liberator_ag_mode": "ag",
        "liberator_aa_mode": "aa",
    }.get(ability, ability)


def _ability_payload(
    action_type: str,
    ability: str | None,
    actor: str | None,
    target_unit: str | None,
    location: LocationRef | None,
    selection: SelectionSpec | None,
    queued: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {"type": action_type}
    if ability is not None:
        payload["ability"] = ability
    if actor is not None:
        payload["actor"] = actor
    if target_unit is not None:
        payload["target_unit"] = target_unit
    if location is not None:
        payload.update(_location_to_payload(location))
    if selection is not None:
        payload["selection"] = _selection_to_payload(selection)
    if queued:
        payload["queued"] = queued
    return payload


def _location_to_payload(location: LocationRef) -> dict[str, object]:
    if location.semantic is not None:
        return {"location": location.semantic}
    if location.x is not None and location.y is not None:
        return {"x": location.x, "y": location.y}
    raise TypeError(f"invalid location ref: {location!r}")


def _location_from_payload(
    payload: dict[str, Any], required: bool
) -> LocationRef | None:
    location_value = payload.get("location", payload.get("semantic_location"))
    has_semantic_location = location_value is not None and bool(
        str(location_value).strip()
    )
    has_coordinates = "x" in payload or "y" in payload
    if has_semantic_location and has_coordinates:
        raise StrategyParseError(
            "location target must use either a semantic location or x/y coordinates, not both"
        )
    if has_semantic_location:
        return _normalize_location_ref(
            LocationRef(semantic=normalize_location(str(location_value)))
        )
    if has_coordinates:
        return _normalize_location_ref(
            LocationRef(
                x=_required_number(payload, "x"), y=_required_number(payload, "y")
            )
        )
    target = payload.get("target")
    if isinstance(target, str) and target and target in LOCATION_SPECS:
        return _normalize_location_ref(LocationRef(semantic=normalize_location(target)))
    if required:
        raise StrategyParseError(
            "strategy JSON action requires a semantic location or x/y coordinates"
        )
    return None


def _normalize_location_ref(location: LocationRef) -> LocationRef:
    if location.semantic is not None:
        return LocationRef(semantic=normalize_location(location.semantic))
    if location.x is None or location.y is None:
        raise StrategyParseError(
            "location requires either a semantic key or both x and y"
        )
    if not 0 <= location.x <= 256 or not 0 <= location.y <= 256:
        raise StrategyParseError("location coordinates must be between 0 and 256")
    return location


def _selection_from_payload(payload: dict[str, Any]) -> SelectionSpec | None:
    return _selection_from_payload_key(payload, "selection")


def _selection_from_payload_key(
    payload: dict[str, Any], key: str
) -> SelectionSpec | None:
    raw = payload.get(key)
    if raw is None:
        if key == "selection" and (
            "selection_mode" in payload
            or "selection_count" in payload
            or "selection_tags" in payload
        ):
            raw = {
                "mode": payload.get("selection_mode", "all"),
                "count": payload.get("selection_count"),
                "tags": payload.get("selection_tags"),
            }
        else:
            return None
    if not isinstance(raw, dict):
        raise StrategyParseError(f"{key} must be an object")
    mode = normalize_selection_mode(str(raw.get("mode", "all")))
    count = None
    if raw.get("count") is not None:
        count = _positive_int_unbounded_from_payload(raw, "count")
        if count > MAX_SELECTION_COUNT:
            raise StrategyParseError(
                f"selection count must not exceed {MAX_SELECTION_COUNT}"
            )
    tags = _tags_from_selection_payload(raw)
    return SelectionSpec(mode=mode, count=count, tags=tags)


def _tags_from_selection_payload(raw: dict[str, Any]) -> tuple[int, ...]:
    tags_value = raw.get("tags")
    if tags_value is None:
        return ()
    if not isinstance(tags_value, list) or not tags_value:
        raise StrategyParseError("selection tags must be a non-empty array")
    if len(tags_value) > MAX_SELECTION_COUNT:
        raise StrategyParseError(
            f"selection tags must not exceed {MAX_SELECTION_COUNT}"
        )
    tags = tuple(_coerce_positive_tag(value, "selection tag") for value in tags_value)
    if len(set(tags)) != len(tags):
        raise StrategyParseError("selection tags must not contain duplicates")
    return tags


def _bool_from_payload(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise StrategyParseError(f"strategy JSON field must be boolean: {key}")


def _target_unit_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("target_unit", payload.get("unit", payload.get("target", "")))
    text = str(value).strip()
    if text and normalize_name(text) in LOCATION_SPECS:
        return normalize_location(text)
    return normalize_target_unit(text)


def _optional_target_unit_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("target_unit", payload.get("target"))
    if value is None:
        return None
    text = str(value).strip()
    return normalize_target_unit(text) if text else None


def _optional_attack_target_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("target_unit", payload.get("target"))
    if value is None:
        return None
    text = str(value).strip()
    return normalize_enemy_target(text) if text else None


def _attack_target_alliance_from_payload(payload: dict[str, Any]) -> str:
    alliance = normalize_name(str(payload.get("target_alliance", "enemy")))
    if alliance not in {"enemy", "neutral"}:
        raise StrategyParseError("target_alliance must be 'enemy' or 'neutral'")
    return alliance


def _optional_rally_target_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("target_unit", payload.get("target"))
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if normalize_name(text) == "nearest_mineral":
        return "nearest_mineral"
    return normalize_target_unit(text)


def _optional_tag_from_payload(payload: dict[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _coerce_positive_tag(payload[key], key)


def _coerce_positive_tag(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise StrategyParseError(f"{field_name} must be a positive integer tag")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isdigit():
        parsed = int(value)
    else:
        raise StrategyParseError(f"{field_name} must be a positive integer tag")
    if parsed < 1:
        raise StrategyParseError(f"{field_name} must be a positive integer tag")
    return parsed


def _optional_addon_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("target_addon", payload.get("addon"))
    if value is None:
        return None
    text = str(value).strip()
    return normalize_addon(text) if text else None


def _placement_mode_from_payload(payload: dict[str, Any]) -> str:
    value = payload.get("placement_mode")
    if value is None:
        return "near"
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"near", "exact"}:
        raise StrategyParseError("placement_mode must be 'near' or 'exact'")
    return normalized


def _max_distance_from_payload(payload: dict[str, Any]) -> int:
    if "max_distance" not in payload:
        return 20
    value = _required_number(payload, "max_distance")
    if not float(value).is_integer():
        raise StrategyParseError("max_distance must be an integer between 0 and 20")
    if value < 0 or value > 20:
        raise StrategyParseError("max_distance must be between 0 and 20")
    return int(value)


def _optional_actor_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("actor", payload.get("source", payload.get("building", "")))
    text = str(value).strip()
    return normalize_ability_actor(text) if text else None


def _default_actor_for_ability(ability: str) -> str | None:
    spec = ABILITY_SPECS[ability]
    return spec.actors[0] if spec.actors and spec.actors[0] != "any" else None


def _parse_coordinates(x_text: str, y_text: str) -> tuple[float, float]:
    try:
        return float(x_text), float(y_text)
    except ValueError as exc:
        raise StrategyParseError("coordinates must be numbers") from exc


def _point_target_from_payload(
    payload: dict[str, Any],
) -> tuple[float | None, float | None, LocationRef | None]:
    location = _location_from_payload(payload, required=True)
    if location is None:
        raise StrategyParseError(
            "strategy JSON action requires a semantic location or x/y coordinates"
        )
    if location.semantic is not None:
        return None, None, location
    return location.x, location.y, None


def _optional_point_target_from_payload(
    payload: dict[str, Any],
) -> tuple[float | None, float | None, LocationRef | None]:
    location = _location_from_payload(payload, required=False)
    if location is None:
        return None, None, None
    if location.semantic is not None:
        return None, None, location
    return location.x, location.y, None


def _required_number(payload: dict[str, Any], key: str) -> float:
    if key not in payload:
        raise StrategyParseError(
            f"strategy JSON action is missing required field: {key}"
        )
    value = payload[key]
    if isinstance(value, bool):
        raise StrategyParseError(f"strategy JSON field must be numeric: {key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise StrategyParseError(f"strategy JSON field must be numeric: {key}") from exc


def _bounded_number_from_payload(
    payload: dict[str, Any],
    key: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = _required_number(payload, key) if key in payload else float(default)
    if not minimum <= value <= maximum:
        raise StrategyParseError(
            f"strategy JSON field {key} must be between {minimum:g} and {maximum:g}"
        )
    return value


def _bounded_int_from_payload(
    payload: dict[str, Any],
    key: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}")
    if not minimum <= value <= maximum:
        raise StrategyParseError(
            f"strategy JSON field {key} must be between {minimum} and {maximum}"
        )
    return value


def _optional_number(
    payload: dict[str, Any], keys: tuple[str, ...], default: float | None = None
) -> float:
    for key in keys:
        if key in payload:
            return _required_number(payload, key)
    if default is not None:
        return default
    raise StrategyParseError(
        f"strategy JSON action is missing one of required fields: {', '.join(keys)}"
    )


def _parse_at_least(value: str) -> float:
    try:
        at_least = float(value)
    except ValueError as exc:
        raise StrategyParseError("wait-until threshold must be numeric") from exc
    if at_least < 0:
        raise StrategyParseError("wait-until threshold must not be negative")
    return at_least


def _parse_positive_int(value: str, field_name: str, max_count: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise StrategyParseError(f"{field_name} must be an integer") from exc
    if parsed < 1:
        raise StrategyParseError(f"{field_name} must be at least 1")
    if parsed > max_count:
        raise StrategyParseError(f"{field_name} must not exceed {max_count}")
    return parsed


def _looks_like_positive_int(value: str) -> bool:
    try:
        return int(value) >= 1
    except ValueError:
        return False


def _parse_non_negative_number(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise StrategyParseError(f"{field_name} must be numeric") from exc
    if parsed < 0:
        raise StrategyParseError(f"{field_name} must not be negative")
    return parsed


def _positive_int_from_payload(
    payload: dict[str, Any], key: str, default: int, max_count: int
) -> int:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}")
    if isinstance(value, float) and not value.is_integer():
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StrategyParseError(
            f"strategy JSON field must be an integer: {key}"
        ) from exc
    if parsed < 1:
        raise StrategyParseError(f"strategy JSON field must be at least 1: {key}")
    if parsed > max_count:
        raise StrategyParseError(
            f"strategy JSON field must not exceed {max_count}: {key}"
        )
    return parsed


def _positive_int_unbounded_from_payload(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise StrategyParseError(
            f"strategy JSON action is missing required field: {key}"
        )
    value = payload[key]
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise StrategyParseError(f"strategy JSON field must be an integer: {key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise StrategyParseError(
            f"strategy JSON field must be an integer: {key}"
        ) from exc
    if parsed < 1:
        raise StrategyParseError(f"strategy JSON field must be at least 1: {key}")
    return parsed


def _optional_positive_int_from_payload(
    payload: dict[str, Any], keys: tuple[str, ...], max_count: int
) -> int | None:
    for key in keys:
        if key in payload:
            return _positive_int_from_payload(
                payload, key, default=1, max_count=max_count
            )
    return None


def _wait_until_from_dict(payload: dict[str, Any]) -> WaitUntilCommand:
    condition = normalize_wait_condition(str(payload.get("condition", "")))
    comparison, threshold = _comparison_and_value_from_payload(payload)
    target, ability, actor, location = _condition_fields_from_payload(
        condition, payload
    )
    on_timeout = normalize_name(str(payload.get("on_timeout", "replan")))
    if on_timeout not in {"replan", "fail"}:
        raise StrategyParseError("wait-until on_timeout must be 'replan' or 'fail'")
    return WaitUntilCommand(
        condition=condition,
        target=target,
        ability=ability,
        actor=actor,
        at_least=threshold,
        comparison=comparison,
        location=location,
        radius=_bounded_number_from_payload(
            payload, "radius", default=12, minimum=0.5, maximum=64
        ),
        selection=_selection_from_payload(payload),
        timeout_seconds=_bounded_number_from_payload(
            payload,
            "timeout_seconds",
            default=120,
            minimum=1,
            maximum=MAX_POLICY_SECONDS,
        ),
        on_timeout=on_timeout,
    )


def _condition_expression_from_payload(payload: Any) -> ConditionExpression:
    if not isinstance(payload, dict):
        raise StrategyParseError("condition expression must be an object")

    match_value = payload.get("match")
    conditions_value = payload.get("conditions")
    if match_value is None:
        for shorthand in ("all", "any"):
            if shorthand in payload:
                match_value = shorthand
                conditions_value = payload[shorthand]
                break
    if match_value is not None or conditions_value is not None:
        match = normalize_name(str(match_value or ""))
        if match not in {"all", "any"}:
            raise StrategyParseError("condition group match must be 'all' or 'any'")
        if not isinstance(conditions_value, list) or not conditions_value:
            raise StrategyParseError(
                "condition group conditions must be a non-empty array"
            )
        if len(conditions_value) > MAX_CONDITION_TERMS:
            raise StrategyParseError(
                "condition group has too many terms: "
                f"{len(conditions_value)} > {MAX_CONDITION_TERMS}"
            )
        conditions = []
        for item in conditions_value:
            if not isinstance(item, dict):
                raise StrategyParseError("each condition group term must be an object")
            if any(key in item for key in ("match", "conditions", "all", "any")):
                raise StrategyParseError(
                    "condition groups may contain atomic conditions only"
                )
            conditions.append(_condition_spec_from_payload(item))
        return ConditionGroup(match=match, conditions=tuple(conditions))
    return _condition_spec_from_payload(payload)


def _condition_spec_from_payload(payload: dict[str, Any]) -> ConditionSpec:
    condition = normalize_wait_condition(str(payload.get("condition", "")))
    comparison, value = _comparison_and_value_from_payload(payload)
    target, ability, actor, location = _condition_fields_from_payload(
        condition, payload
    )
    return ConditionSpec(
        condition=condition,
        value=value,
        comparison=comparison,
        target=target,
        ability=ability,
        actor=actor,
        location=location,
        radius=_bounded_number_from_payload(
            payload, "radius", default=12, minimum=0.5, maximum=64
        ),
        selection=_selection_from_payload(payload),
    )


def _comparison_and_value_from_payload(
    payload: dict[str, Any],
) -> tuple[str, float]:
    threshold_fields = [
        key
        for key in ("at_least", "at_most", "equals", "value", "minimum", "count")
        if key in payload and payload[key] is not None
    ]
    if len(threshold_fields) > 1:
        raise StrategyParseError(
            "condition must use exactly one threshold field: "
            "at_least, at_most, equals, or value"
        )
    field_name = threshold_fields[0] if threshold_fields else None
    inferred_comparison = {
        "at_least": "gte",
        "minimum": "gte",
        "count": "gte",
        "at_most": "lte",
        "equals": "eq",
    }.get(field_name or "", "gte")
    comparison_value = payload.get("comparison", payload.get("operator"))
    comparison = (
        normalize_comparison(str(comparison_value))
        if comparison_value is not None
        else inferred_comparison
    )
    if (
        comparison_value is not None
        and field_name in {"at_least", "minimum", "count", "at_most", "equals"}
        and comparison != inferred_comparison
    ):
        raise StrategyParseError(
            f"condition threshold {field_name} conflicts with comparison {comparison}"
        )
    value = _optional_number(
        payload,
        (field_name,) if field_name is not None else (),
        default=1,
    )
    if value < 0:
        raise StrategyParseError("condition threshold must not be negative")
    if value > 10000:
        raise StrategyParseError("condition threshold must not exceed 10000")
    return comparison, value


def _condition_fields_from_payload(
    condition: str, payload: dict[str, Any]
) -> tuple[str | None, str | None, str | None, LocationRef | None]:
    target: str | None = None
    ability: str | None = None
    actor: str | None = None

    if condition.startswith("structure_") or condition == "idle_structure_count":
        target_value = payload.get(
            "target", payload.get("structure", payload.get("building", ""))
        )
        target = normalize_structure_target(str(target_value))
    elif condition in {
        "unit_count",
        "unit_near_location",
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
        target_value = payload.get("target", payload.get("unit", ""))
        target = normalize_condition_actor(str(target_value))
    elif condition == "unit_form_count":
        target_value = payload.get("target", payload.get("form", ""))
        target = normalize_unit_form(str(target_value))
        actor_value = payload.get("actor", payload.get("unit"))
        actor = (
            normalize_ability_actor(str(actor_value))
            if actor_value is not None
            else None
        )
    elif condition == "ability_available":
        ability_value = payload.get("ability", payload.get("target", ""))
        ability = normalize_ability(str(ability_value))
        actor_value = payload.get("actor", payload.get("unit"))
        actor = (
            normalize_ability_actor(str(actor_value))
            if actor_value is not None
            else ABILITY_SPECS[ability].actors[0]
        )
    elif condition == "producer_available":
        target_value = payload.get("target", payload.get("unit", ""))
        target = normalize_train_unit(str(target_value))
    elif condition == "cargo_used":
        target_value = payload.get("target", payload.get("actor", ""))
        target = normalize_ability_actor(str(target_value))
    elif condition in {"enemy_unit_count", "enemy_structure_count", "enemy_near_location"}:
        target_value = payload.get("target", payload.get("target_unit"))
        target = (
            normalize_enemy_target(str(target_value))
            if target_value is not None
            else None
        )
    elif condition == "enemy_race":
        target_value = payload.get("target", payload.get("race", ""))
        target = normalize_enemy_race(str(target_value))
    elif condition == "alert_active":
        target_value = payload.get("target", payload.get("alert", ""))
        target = normalize_alert(str(target_value))
    elif condition == "upgrade_complete":
        target_value = payload.get("target", payload.get("upgrade", ""))
        target = normalize_upgrade(str(target_value))

    needs_location = condition in {
        "unit_near_location",
        "enemy_near_location",
        "location_visible",
    }
    location = _location_from_payload(payload, required=needs_location)
    if condition == "under_attack" and location is None:
        location = LocationRef(semantic="own_main")
    return target, ability, actor, location


def _nested_actions_from_payload(
    payload: Any,
    field_name: str,
    default_unit: str,
    control_depth: int,
    *,
    allow_empty: bool,
) -> tuple[StrategyAction, ...]:
    if control_depth >= MAX_CONTROL_DEPTH:
        raise StrategyParseError(
            f"control-flow nesting exceeds maximum depth {MAX_CONTROL_DEPTH}"
        )
    if payload is None and allow_empty:
        return ()
    if not isinstance(payload, list) or (not payload and not allow_empty):
        requirement = "an array" if allow_empty else "a non-empty array"
        raise StrategyParseError(f"{field_name} must be {requirement}")
    if len(payload) > MAX_CONTROL_BRANCH_ACTIONS:
        raise StrategyParseError(
            f"{field_name} has too many actions: "
            f"{len(payload)} > {MAX_CONTROL_BRANCH_ACTIONS}"
        )
    return tuple(
        _action_from_dict(
            action,
            default_unit=default_unit,
            control_depth=control_depth + 1,
        )
        for action in payload
    )


def _defined_action_count(actions: tuple[StrategyAction, ...]) -> int:
    total = 0
    for action in actions:
        total += 1
        if isinstance(action, ConditionalCommand):
            total += _defined_action_count(action.then_actions)
            total += _defined_action_count(action.else_actions)
        elif isinstance(action, RepeatCommand):
            total += _defined_action_count(action.actions)
        elif isinstance(action, WithTimeoutCommand):
            total += _defined_action_count(action.actions)
    return total


def _maximum_execution_action_count(actions: tuple[StrategyAction, ...]) -> int:
    """Return the conservative maximum number of action dispatches.

    A conditional executes only one branch, while a repeat can execute its body
    ``max_cycles`` times.  Counting each loop boundary as a dispatch mirrors the
    runtime splicing implementation and prevents finite-but-explosive nesting.
    """

    total = 0
    for action in actions:
        if isinstance(action, ConditionalCommand):
            total += 1 + max(
                _maximum_execution_action_count(action.then_actions),
                _maximum_execution_action_count(action.else_actions),
            )
        elif isinstance(action, RepeatCommand):
            body = _maximum_execution_action_count(action.actions)
            total += 1 + action.max_cycles * (body + 1)
        elif isinstance(action, WithTimeoutCommand):
            total += 2 + _maximum_execution_action_count(action.actions)
        else:
            total += 1
        if total > MAX_CONTROL_EXECUTION_ACTIONS:
            return total
    return total


def _looks_like_json(text: str) -> bool:
    return text.startswith("{") or text.startswith("[")


def _normalize_intent(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _catalog_key_from_intent(text: str, registry) -> str | None:
    padded = f" {text.replace('_', ' ')} "
    matches: list[tuple[int, str]] = []
    for key, spec in registry.items():
        names = (key, key.replace("_", " "), spec.enum_name) + spec.aliases
        for name in names:
            candidate = name.strip().lower().replace("_", " ")
            if not candidate:
                continue
            if candidate in text and (len(candidate) > 2 or f" {candidate} " in padded):
                matches.append((len(candidate), key))
                break
    return max(matches, default=(0, ""))[1] or None


def _unit_from_intent(text: str, default_unit: str) -> str:
    unit = _catalog_key_from_intent(text, UNIT_SPECS)
    if unit:
        return "worker" if unit == "scv" else unit
    return normalize_unit(default_unit)


def _route_plan(
    unit: str, route: tuple[tuple[float, float], ...], wait_seconds: float
) -> StrategyPlan:
    actions: list[StrategyAction] = []
    for index, (x, y) in enumerate(route):
        if index > 0 and wait_seconds > 0:
            actions.append(WaitCommand(seconds=wait_seconds))
        actions.append(MoveCommand(unit=unit, x=x, y=y))
    return StrategyPlan(actions=tuple(actions))


def normalize_train_unit(unit: str) -> str:
    try:
        return resolve_alias(unit, categories=("unit",)).key
    except KeyError as exc:
        raise StrategyParseError(f"unsupported Terran train unit: {unit}") from exc


def normalize_wait_unit(unit: str) -> str:
    return normalize_unit(unit)


def normalize_wait_condition(condition: str) -> str:
    normalized = condition.strip().lower().replace("-", "_")
    aliases = {
        "mineral": "minerals",
        "minerals": "minerals",
        "vespene": "vespene",
        "gas": "vespene",
        "supply_left": "supply_left",
        "supply": "supply_left",
        "supply_used": "supply_used",
        "supply_cap": "supply_cap",
        "structure": "structure_count",
        "building": "structure_count",
        "structure_count": "structure_count",
        "building_count": "structure_count",
        "structure_total": "structure_count",
        "structure_ready": "structure_ready",
        "building_ready": "structure_ready",
        "structure_complete": "structure_ready",
        "structure_completed": "structure_ready",
        "structure_pending": "structure_pending",
        "building_pending": "structure_pending",
        "structure_started": "structure_pending",
        "unit": "unit_count",
        "units": "unit_count",
        "unit_count": "unit_count",
        "army_supply": "army_supply",
        "enemy_seen": "enemy_unit_count",
        "enemy_count": "enemy_unit_count",
        "enemy_unit_count": "enemy_unit_count",
        "enemy_unit_type_count": "enemy_unit_count",
        "enemy_structure_count": "enemy_structure_count",
        "enemy_structure_type_count": "enemy_structure_count",
        "enemy_race": "enemy_race",
        "race": "enemy_race",
        "matchup": "enemy_race",
        "alert": "alert_active",
        "alert_active": "alert_active",
        "idle_structure_count": "idle_structure_count",
        "idle_unit_count": "idle_unit_count",
        "unit_idle_count": "idle_unit_count",
        "ready_unit_count": "ready_unit_count",
        "damaged_unit_count": "damaged_unit_count",
        "cloaked_unit_count": "cloaked_unit_count",
        "flying_unit_count": "flying_unit_count",
        "loaded_unit_count": "loaded_unit_count",
        "weapon_ready_count": "weapon_ready_count",
        "unit_health": "unit_health",
        "unit_health_fraction": "unit_health_fraction",
        "unit_health_ratio": "unit_health_fraction",
        "unit_energy": "unit_energy",
        "unit_order_count": "unit_order_count",
        "orders": "unit_order_count",
        "ability_available": "ability_available",
        "ability_ready": "ability_available",
        "unit_form": "unit_form_count",
        "unit_form_count": "unit_form_count",
        "location_visible": "location_visible",
        "visible": "location_visible",
        "idle_producer": "producer_available",
        "producer_available": "producer_available",
        "cargo": "cargo_used",
        "cargo_used": "cargo_used",
        "unit_near": "unit_near_location",
        "unit_near_location": "unit_near_location",
        "enemy_near": "enemy_near_location",
        "enemy_near_location": "enemy_near_location",
        "under_attack": "under_attack",
        "townhall": "townhall_count",
        "townhalls": "townhall_count",
        "townhall_count": "townhall_count",
        "base_count": "townhall_count",
        "upgrade": "upgrade_complete",
        "upgrade_ready": "upgrade_complete",
        "upgrade_complete": "upgrade_complete",
        "research_complete": "upgrade_complete",
        "time": "game_time",
        "game_time": "game_time",
    }
    if normalized not in aliases:
        raise StrategyParseError(f"unsupported wait-until condition: {condition}")
    return aliases[normalized]


def normalize_comparison(comparison: str) -> str:
    normalized = normalize_name(comparison)
    aliases = {
        "gte": "gte",
        "ge": "gte",
        "at_least": "gte",
        "greater_than_or_equal": "gte",
        "lte": "lte",
        "le": "lte",
        "at_most": "lte",
        "less_than_or_equal": "lte",
        "eq": "eq",
        "equals": "eq",
        "equal": "eq",
        "neq": "neq",
        "not_equal": "neq",
        "gt": "gt",
        "greater_than": "gt",
        "lt": "lt",
        "less_than": "lt",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported condition comparison: {comparison}"
        ) from exc


def normalize_enemy_race(race: str) -> str:
    normalized = normalize_name(race)
    if normalized not in ENEMY_RACE_KEYS:
        raise StrategyParseError(f"unsupported enemy race: {race}")
    return normalized


def normalize_alert(alert: str) -> str:
    normalized = normalize_name(alert)
    aliases = {
        normalize_name(key).replace("_", ""): key for key in ALERT_KEYS
    }
    canonical = aliases.get(normalized.replace("_", ""))
    if canonical is None:
        raise StrategyParseError(f"unsupported game alert: {alert}")
    return canonical


def normalize_condition_actor(actor: str) -> str:
    if not actor.strip():
        raise StrategyParseError("condition target unit/actor is required")
    return normalize_ability_actor(actor)


def normalize_unit_form(form: str) -> str:
    normalized = normalize_name(form)
    if normalized not in UNIT_FORM_SPECS:
        raise StrategyParseError(f"unsupported Terran runtime form: {form}")
    return normalized


def normalize_building(building: str) -> str:
    try:
        return resolve_alias(building, categories=("structure",)).key
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported Terran build structure: {building}"
        ) from exc


def normalize_structure_target(building: str) -> str:
    try:
        return resolve_alias(building, categories=("structure", "addon", "morph")).key
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported Terran structure target: {building}"
        ) from exc


def normalize_production_structure(building: str) -> str:
    normalized = normalize_structure_target(building)
    if normalized not in {
        "command_center",
        "orbital_command",
        "planetary_fortress",
        "barracks",
        "factory",
        "starport",
        "bunker",
    }:
        raise StrategyParseError(f"structure cannot rally produced units: {building}")
    return normalized


def normalize_addon(addon: str) -> str:
    try:
        return resolve_alias(addon, categories=("addon",)).key
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported Terran production add-on: {addon}"
        ) from exc


def normalize_morph(building: str) -> str:
    try:
        return resolve_alias(building, categories=("morph",)).key
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported Terran structure morph: {building}"
        ) from exc


def normalize_upgrade(upgrade: str) -> str:
    try:
        return resolve_alias(upgrade, categories=("upgrade",)).key
    except KeyError as exc:
        raise StrategyParseError(f"unsupported Terran upgrade: {upgrade}") from exc


def normalize_repair_target(target: str) -> str:
    try:
        resolved = resolve_alias(
            target,
            categories=("unit", "special_unit", "structure", "addon", "morph"),
        )
    except KeyError as exc:
        raise StrategyParseError(f"unsupported Terran repair target: {target}") from exc
    canonical = (
        "worker"
        if resolved.category == "unit" and resolved.key == "scv"
        else resolved.key
    )
    if canonical not in REPAIRABLE_TARGET_KEYS:
        raise StrategyParseError(
            f"Terran SCVs cannot repair biological target: {target}"
        )
    return canonical


def normalize_ability(ability: str) -> str:
    alias = {
        "stim_marine": "stim_marine",
        "stim_marauder": "stim_marauder",
        "siege": "siege_mode",
        "siege_tank_siege": "siege_mode",
        "unsiege": "unsiege_mode",
        "hellbat": "morph_hellbat",
        "hellion": "morph_hellion",
        "high_impact": "thor_high_impact_mode",
        "explosive": "thor_explosive_mode",
        "assault": "viking_assault_mode",
        "fighter": "viking_fighter_mode",
        "ag": "liberator_ag_mode",
        "aa": "liberator_aa_mode",
        "build_in_progress": "cancel_build_in_progress",
        "queue_1": "cancel_queue_1",
        "queue_5": "cancel_queue_5",
        "any": "cancel_any",
        "last": "cancel_last",
    }.get(normalize_name(ability))
    if alias:
        return alias
    try:
        return resolve_ability(ability).key
    except KeyError as exc:
        raise StrategyParseError(f"unsupported Terran ability: {ability}") from exc


def normalize_cancel_ability(ability: str) -> str:
    normalized = normalize_name(ability) or "any"
    candidate = (
        normalized if normalized.startswith("cancel_") else f"cancel_{normalized}"
    )
    try:
        canonical = normalize_ability(candidate)
    except StrategyParseError as exc:
        raise StrategyParseError(f"unsupported cancel ability: {ability}") from exc
    if not canonical.startswith("cancel_"):
        raise StrategyParseError(f"unsupported cancel ability: {ability}")
    return canonical


def normalize_location(location: str) -> str:
    try:
        return resolve_location(location).key
    except KeyError as exc:
        raise StrategyParseError(f"unsupported semantic location: {location}") from exc


def normalize_selection_mode(mode: str) -> str:
    try:
        return resolve_selection_mode(mode).key
    except KeyError as exc:
        raise StrategyParseError(f"unsupported selection mode: {mode}") from exc


def normalize_ability_actor(actor: str) -> str:
    normalized = actor.strip().lower()
    if normalized in {"any", ""}:
        return "any"
    try:
        resolved = resolve_alias(
            actor,
            categories=("unit", "special_unit", "structure", "addon", "morph"),
        )
    except KeyError as exc:
        raise StrategyParseError(f"unsupported Terran ability actor: {actor}") from exc
    return (
        "worker"
        if resolved.category == "unit" and resolved.key == "scv"
        else resolved.key
    )


def normalize_target_unit(target: str) -> str:
    text = target.strip()
    if not text:
        raise StrategyParseError("target unit is required")
    selector_aliases = {
        "enemy": "nearest_enemy",
        "enemy_unit": "nearest_enemy",
        "enemy_structure": "nearest_enemy_structure",
        "enemy_ground": "nearest_enemy_ground",
        "enemy_air": "nearest_enemy_air",
        "enemy_biological": "nearest_enemy_biological",
        "enemy_mechanical": "nearest_enemy_mechanical",
        "enemy_massive": "nearest_enemy_massive",
        "enemy_detector": "nearest_enemy_detector",
        "friendly": "nearest_friendly",
        "friendly_lowest_health": "lowest_health_friendly",
        "lowest_health_friendly": "lowest_health_friendly",
        "friendly_highest_energy": "highest_energy_friendly",
        "highest_energy_friendly": "highest_energy_friendly",
        "damaged": "damaged_friendly",
        "any": "any_friendly",
    }
    normalized = normalize_name(text)
    if normalized in TARGET_SELECTORS:
        return normalized
    if normalized in selector_aliases:
        return selector_aliases[normalized]
    try:
        resolved = resolve_alias(
            text,
            categories=("unit", "special_unit", "structure", "addon", "morph"),
        )
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported Terran ability target: {target}"
        ) from exc
    return (
        "worker"
        if resolved.category == "unit" and resolved.key == "scv"
        else resolved.key
    )


def normalize_enemy_target(target: str) -> str:
    """Normalize selectors while safely allowing observed cross-race type names."""

    try:
        return normalize_target_unit(target)
    except StrategyParseError:
        normalized = normalize_name(target)
        if re.fullmatch(r"[a-z0-9_]{1,64}", normalized):
            return normalized
        raise StrategyParseError(f"unsupported enemy target type: {target}")


def normalize_unit(unit: str) -> str:
    normalized = unit.strip().lower()
    if normalized in {"probe", "drone"}:
        return "worker"
    try:
        resolved = resolve_alias(
            unit, categories=("unit", "special_unit", "structure", "morph")
        )
    except KeyError as exc:
        raise StrategyParseError(
            f"unsupported controllable Terran unit: {unit}"
        ) from exc
    key = resolved.key
    if resolved.category == "special_unit" and key not in MOVABLE_SPECIAL_UNIT_KEYS:
        raise StrategyParseError(
            f"Terran special unit cannot receive basic movement orders: {unit}"
        )
    if (
        resolved.category in {"structure", "morph"}
        and key not in FLYING_STRUCTURE_ACTOR_KEYS
    ):
        raise StrategyParseError(
            f"Terran structure cannot receive basic movement orders: {unit}"
        )
    return "worker" if key == "scv" else key


def normalize_attack_actor(actor: str) -> str:
    canonical = normalize_ability_actor(actor)
    canonical = "scv" if canonical == "worker" else canonical
    if canonical not in ATTACK_CAPABLE_UNIT_KEYS:
        raise StrategyParseError(
            f"Terran actor cannot receive basic attack orders: {actor}"
        )
    return "worker" if canonical == "scv" else canonical


def normalize_mobile_attack_unit(unit: str) -> str:
    canonical = normalize_attack_actor(unit)
    catalog_key = "scv" if canonical == "worker" else canonical
    if catalog_key not in MOBILE_ATTACK_CAPABLE_UNIT_KEYS:
        raise StrategyParseError(f"Terran actor cannot kite while immobile: {unit}")
    return canonical


def normalize_stop_actor(actor: str) -> str:
    try:
        return normalize_unit(actor)
    except StrategyParseError:
        canonical = normalize_attack_actor(actor)
        if canonical in {"planetary_fortress", "missile_turret", "auto_turret"}:
            return canonical
        raise
