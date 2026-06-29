from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from starcraft_llm.game_state import GameStateSummary
from starcraft_llm.strategy import StrategyPlan, parse_strategy_request

DEFAULT_PLANNER = "rule"
PLANNER_MODES = ("rule", "openai", "server")


@dataclass(frozen=True)
class PlannerRequest:
    """Input boundary for strategy planners.

    Planners translate a user strategy plus optional game-state context into the
    canonical StrategyPlan consumed by the SC2 executor. The executor never runs
    free-form model text directly.
    """

    strategy: str
    game_state: GameStateSummary | None = None
    default_unit: str = "worker"


class StrategyPlanner(Protocol):
    """Planner interface for converting strategy text into a StrategyPlan."""

    name: str

    def create_plan(self, request: PlannerRequest) -> StrategyPlan:
        """Create a deterministic StrategyPlan or raise a planner-specific error."""


class PlannerUnavailableError(RuntimeError):
    """Raised when a selected planner mode exists but has not been implemented."""


class RuleBasedPlanner:
    """Deterministic local planner backed by the current parser/intent translator."""

    name = "rule"

    def create_plan(self, request: PlannerRequest) -> StrategyPlan:
        return parse_strategy_request(request.strategy, default_unit=request.default_unit)


class _UnavailablePlanner:
    def __init__(self, name: str, guidance: str):
        self.name = name
        self._guidance = guidance

    def create_plan(self, request: PlannerRequest) -> StrategyPlan:
        del request
        raise PlannerUnavailableError(self._guidance)


def create_planner(name: str = DEFAULT_PLANNER) -> StrategyPlanner:
    """Create the explicitly selected planner.

    This intentionally does not implement a fallback chain. The default planner
    is fixed to ``rule``; other modes must be selected with ``--planner`` and
    must succeed on their own.
    """

    normalized = name.strip().lower()
    if normalized == "rule":
        return RuleBasedPlanner()
    if normalized == "openai":
        return _UnavailablePlanner(
            "openai",
            "OpenAI planner is not implemented yet. For now use --planner rule. "
            "Next step: add an OpenAI API-key based planner that returns StrategyPlan JSON.",
        )
    if normalized == "server":
        return _UnavailablePlanner(
            "server",
            "Server planner is not implemented yet. For now use --planner rule. "
            "Next step: add a planner HTTP client that POSTs strategy/state and expects StrategyPlan JSON.",
        )

    supported = ", ".join(PLANNER_MODES)
    raise ValueError(f"unknown planner {name!r}; supported planners: {supported}")


def plan_strategy(
    strategy: str,
    planner_name: str = DEFAULT_PLANNER,
    game_state: GameStateSummary | None = None,
    default_unit: str = "worker",
) -> StrategyPlan:
    planner = create_planner(planner_name)
    return planner.create_plan(
        PlannerRequest(strategy=strategy, game_state=game_state, default_unit=default_unit)
    )
