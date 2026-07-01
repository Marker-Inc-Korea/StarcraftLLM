from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from starcraft_llm.game_state import GameStateSummary, game_state_summary_to_dict
from starcraft_llm.strategy import StrategyPlan, parse_strategy_request, strategy_plan_from_dict

DEFAULT_PLANNER = "rule"
PLANNER_MODES = ("rule", "gemini", "openai", "server")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_API_KEY_FILE = Path(".secrets/gemini_api_key.txt")
_GEMINI_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

HttpPost = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


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
    """Raised when a selected planner mode exists but has not been implemented or configured."""


class PlannerError(RuntimeError):
    """Raised when a selected planner fails to produce a valid StrategyPlan."""


class RuleBasedPlanner:
    """Deterministic local planner backed by the current parser/intent translator."""

    name = "rule"

    def create_plan(self, request: PlannerRequest) -> StrategyPlan:
        return parse_strategy_request(request.strategy, default_unit=request.default_unit)


class GeminiPlanner:
    """Gemini API planner that returns the existing StrategyPlan JSON contract."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_key_file: Path | str | None = None,
        timeout_seconds: float = 30,
        http_post: HttpPost = None,
    ):
        self.api_key = api_key
        self.model = model or os.environ.get("STARCRAFT_LLM_GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.api_key_file = Path(
            api_key_file
            or os.environ.get("STARCRAFT_LLM_GEMINI_API_KEY_FILE", str(DEFAULT_GEMINI_API_KEY_FILE))
        )
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or _post_json

    def create_plan(self, request: PlannerRequest) -> StrategyPlan:
        api_key = self.api_key or load_gemini_api_key(self.api_key_file)
        payload = {
            "model": self.model,
            "input": _build_gemini_prompt(request),
            "response_format": {
                "type": "text",
                "mime_type": "application/json",
                "schema": strategy_plan_json_schema(),
            },
        }
        response_payload = self._http_post(
            _GEMINI_INTERACTIONS_URL,
            {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            payload,
            self.timeout_seconds,
        )
        output_text = _extract_gemini_output_text(response_payload)
        try:
            plan_payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise PlannerError(f"Gemini planner returned invalid JSON: {exc.msg}") from exc

        plan_payload = _normalize_gemini_plan_payload(plan_payload)
        try:
            return strategy_plan_from_dict(plan_payload, default_unit=request.default_unit)
        except Exception as exc:
            raise PlannerError(f"Gemini planner returned an invalid StrategyPlan: {exc}") from exc


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
    if normalized == "gemini":
        return GeminiPlanner()
    if normalized == "openai":
        return _UnavailablePlanner(
            "openai",
            "OpenAI planner is not implemented yet. For now use --planner rule or --planner gemini. "
            "Next step: add an OpenAI API-key based planner that returns StrategyPlan JSON.",
        )
    if normalized == "server":
        return _UnavailablePlanner(
            "server",
            "Server planner is not implemented yet. For now use --planner rule or --planner gemini. "
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


def load_gemini_api_key(api_key_file: Path = DEFAULT_GEMINI_API_KEY_FILE) -> str:
    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value

    path = Path(api_key_file)
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value

    raise PlannerUnavailableError(
        "Gemini API key is missing. Set GEMINI_API_KEY or GOOGLE_API_KEY, "
        f"or put the key in {path} (this path is ignored by git)."
    )


def strategy_plan_json_schema() -> dict[str, Any]:
    # Keep the schema intentionally simple because Gemini structured output
    # supports a subset of JSON Schema. The local StrategyPlan parser remains
    # the authoritative validator for action-specific required fields.
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "unit": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "seconds": {"type": "number"},
                        "resource": {"type": "string"},
                        "building": {"type": "string"},
                        "worker": {"type": "string"},
                    },
                    "required": ["type"],
                },
            }
        },
        "required": ["actions"],
    }


def _build_gemini_prompt(request: PlannerRequest) -> str:
    game_state = (
        game_state_summary_to_dict(request.game_state) if request.game_state is not None else None
    )
    return "\n".join(
        [
            "You are the planner for a minimal StarCraft II bot.",
            "Return only JSON matching this exact root shape: {\"actions\": [...]}. Do not use a different top-level key such as plan. Do not include markdown.",
            "Available actions are exactly:",
            "- move: {type:'move', unit:'worker'|'marine', x:number, y:number}",
            "- attack move: {type:'attack', unit:'worker'|'marine', x:number, y:number}",
            "- wait: {type:'wait', seconds:number}",
            "- gather minerals: {type:'gather', unit:'worker', resource:'minerals'}",
            "- train unit: {type:'train', unit:'scv'|'marine'}",
            "- build structure: {type:'build', building:'supply_depot'|'barracks'|'refinery', worker:'worker'}",
            "Constraints:",
            "- Use only actions listed above.",
            "- Keep plans short, usually 1 to 5 actions.",
            "- For early economy requests at game start, prefer train scv and gather minerals.",
            "- Only build supply_depot/barracks/refinery when minerals and prerequisites in Game state JSON make it feasible; otherwise gather/wait/train safer units.",
            "- Build barracks only after a supply depot exists or is clearly already built in structures.",
            "- Train marine only when a barracks exists in structures.",
            "- For scouting requests, move one worker through safe map coordinates near (35,42), (45,42), or (55,45).",
            "- For attack requests, use attack with marine when marines exist; otherwise choose economy or setup actions.",
            "- If the request is ambiguous, choose a safe economy action rather than inventing unsupported actions.",
            f"User strategy: {request.strategy}",
            "Game state JSON:",
            json.dumps(game_state, ensure_ascii=False),
        ]
    )


def _normalize_gemini_plan_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "actions" not in payload and isinstance(payload.get("plan"), list):
        return {"actions": payload["plan"]}
    return payload


def _post_json(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise PlannerError(f"Gemini API request failed with HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
        raise PlannerError(f"Gemini API request failed: {exc.reason}") from exc

    try:
        return json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise PlannerError(f"Gemini API returned invalid JSON: {exc.msg}") from exc


def _extract_gemini_output_text(response_payload: dict[str, Any]) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for step in response_payload.get("steps", []):
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for item in step.get("content", []):
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])

    joined = "".join(chunks).strip()
    if joined:
        return joined

    raise PlannerError("Gemini API response did not contain output_text or model_output text")
