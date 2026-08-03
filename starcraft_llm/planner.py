from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from starcraft_llm.command_catalog import (
    ABILITY_SPECS,
    LOCATION_SPECS,
    MAX_PLAN_ACTIONS,
    MAX_POLICY_SECONDS,
    MAX_SELECTION_COUNT,
    MAX_STRUCTURE_ACTION_COUNT,
    MAX_WORKER_ASSIGNMENT_COUNT,
    TARGET_SELECTORS,
    build_command_prompt_section,
)
from starcraft_llm.game_state import GameStateSummary, game_state_summary_to_dict
from starcraft_llm.strategy import (
    StrategyPlan,
    parse_strategy_request,
    strategy_plan_from_dict,
)

DEFAULT_PLANNER = "rule"
PLANNER_MODES = ("rule", "gemini", "openai", "server")
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_API_KEY_FILE = Path(".secrets/gemini_api_key.txt")
_GEMINI_INTERACTIONS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/interactions"
)

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
        return parse_strategy_request(
            request.strategy, default_unit=request.default_unit
        )


class GeminiPlanner:
    """Gemini API planner that returns the existing StrategyPlan JSON contract."""

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_key_file: Path | str | None = None,
        timeout_seconds: float = 30,
        http_post: HttpPost | None = None,
    ):
        self.api_key = api_key
        self.model = model or os.environ.get(
            "STARCRAFT_LLM_GEMINI_MODEL", DEFAULT_GEMINI_MODEL
        )
        api_key_path = (
            api_key_file
            if api_key_file is not None
            else os.environ.get(
                "STARCRAFT_LLM_GEMINI_API_KEY_FILE", str(DEFAULT_GEMINI_API_KEY_FILE)
            )
        )
        self.api_key_file = Path(api_key_path)
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
            raise PlannerError(
                f"Gemini planner returned invalid JSON: {exc.msg}"
            ) from exc

        plan_payload = _normalize_gemini_plan_payload(plan_payload)
        try:
            return strategy_plan_from_dict(
                plan_payload, default_unit=request.default_unit
            )
        except Exception as exc:
            raise PlannerError(
                f"Gemini planner returned an invalid StrategyPlan: {exc}"
            ) from exc


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
        PlannerRequest(
            strategy=strategy, game_state=game_state, default_unit=default_unit
        )
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
    selection_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [
                    "all",
                    "ready",
                    "idle",
                    "closest",
                    "lowest_health",
                    "highest_energy",
                ],
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SELECTION_COUNT,
            },
            "tags": {
                "type": "array",
                "maxItems": MAX_SELECTION_COUNT,
                "items": {"type": "integer", "minimum": 1},
            },
        },
    }
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "maxItems": MAX_PLAN_ACTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "move",
                                "move_target",
                                "move_and_wait",
                                "attack",
                                "attack_move",
                                "attack_enemy",
                                "attack_target",
                                "focus_fire",
                                "kite",
                                "patrol",
                                "hold",
                                "hold_position",
                                "stop",
                                "rally",
                                "wait",
                                "wait_until",
                                "gather",
                                "return_cargo",
                                "distribute_workers",
                                "train",
                                "produce_until",
                                "maintain_production",
                                "build",
                                "expand",
                                "build_addon",
                                "morph",
                                "research",
                                "repair",
                                "use_ability",
                                "scan",
                                "call_down_mule",
                                "supply_drop",
                                "transform",
                                "lift",
                                "land",
                                "land_on_addon",
                                "load",
                                "unload",
                                "cancel",
                                "salvage",
                                "build_nuke",
                                "launch_nuke",
                                "replan",
                            ],
                        },
                        "unit": {"type": "string"},
                        "x": {"type": "number", "minimum": 0, "maximum": 256},
                        "y": {"type": "number", "minimum": 0, "maximum": 256},
                        "seconds": {"type": "number", "minimum": 0, "maximum": 30},
                        "condition": {"type": "string"},
                        "target": {"type": "string"},
                        "at_least": {"type": "number"},
                        "arrival_tolerance": {
                            "type": "number",
                            "minimum": 0.25,
                            "maximum": 20,
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": MAX_POLICY_SECONDS,
                        },
                        "duration_seconds": {
                            "type": "number",
                            "minimum": 0.25,
                            "maximum": 30,
                        },
                        "retreat_distance": {
                            "type": "number",
                            "minimum": 0.5,
                            "maximum": 10,
                        },
                        "radius": {
                            "type": "number",
                            "minimum": 0.5,
                            "maximum": 64,
                        },
                        "on_timeout": {
                            "type": "string",
                            "enum": ["replan", "fail"],
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SELECTION_COUNT,
                        },
                        "target_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SELECTION_COUNT,
                        },
                        "reserve_minerals": {"type": "integer", "minimum": 0},
                        "reserve_vespene": {"type": "integer", "minimum": 0},
                        "reserve_supply": {"type": "integer", "minimum": 0},
                        "max_seconds": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": MAX_POLICY_SECONDS,
                        },
                        "resource": {"type": "string"},
                        "building": {"type": "string"},
                        "worker": {"type": "string"},
                        "workers": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_WORKER_ASSIGNMENT_COUNT,
                        },
                        "producer": {"type": "string"},
                        "addon": {"type": "string"},
                        "upgrade": {"type": "string"},
                        "mineral_to_gas_ratio": {"type": "number"},
                        "ability": {"type": "string", "enum": list(ABILITY_SPECS)},
                        "actor": {"type": "string"},
                        "target_unit": {"type": "string"},
                        "target_tag": {"type": "integer", "minimum": 1},
                        "target_selector": {"type": "string"},
                        "mode": {"type": "string"},
                        "target_addon": {"type": "string"},
                        "target_addon_tag": {"type": "integer", "minimum": 1},
                        "passenger_tag": {"type": "integer", "minimum": 1},
                        "placement_mode": {"type": "string", "enum": ["near", "exact"]},
                        "max_distance": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 20,
                        },
                        "reserve_addon_space": {"type": "boolean"},
                        "location": {"type": "string", "enum": list(LOCATION_SPECS)},
                        "selection": selection_schema,
                        "target_selection": selection_schema,
                        "producer_selection": selection_schema,
                        "researcher_selection": selection_schema,
                        "queued": {"type": "boolean"},
                        "reason": {"type": "string", "maxLength": 500},
                    },
                    "required": ["type"],
                },
            }
        },
        "required": ["actions"],
    }


def _build_gemini_prompt(request: PlannerRequest) -> str:
    game_state = (
        game_state_summary_to_dict(request.game_state)
        if request.game_state is not None
        else None
    )
    return "\n".join(
        [
            "You are the planner for a bounded Terran standard-melee StarCraft II bot.",
            'Return only JSON matching this exact root shape: {"actions": [...]}. Do not use a different top-level key such as plan. Do not include markdown.',
            "Available actions are exactly:",
            "- move/attack-move/patrol: {type:'move'|'attack'|'attack_move'|'patrol', unit:string, location?:string, x?:number, y?:number, selection?:{mode:string,count?:integer,tags?:integer[]}, queued?:boolean}",
            "- move and wait for arrival: {type:'move_and_wait', unit:string, location?:string, x?:number, y?:number, target_unit?:string, target_tag?:integer, selection?:object, arrival_tolerance?:number, timeout_seconds?:number}",
            "- move/follow friendly target: {type:'move_target', unit:string, target_unit?:string, target_tag?:integer, selection?:object, queued?:boolean}",
            "- attack visible enemy: {type:'attack_enemy', unit:string, target_unit?:string, target_tag?:integer, selection?:object, queued?:boolean}",
            "- attack specific target: {type:'attack_target', unit:string, target_unit?:string, target_tag?:integer, selection?:object, queued?:boolean}",
            "- focus fire until target death: {type:'focus_fire', unit:string, target_unit?:string, target_tag?:integer, selection?:object, timeout_seconds?:number}",
            "- cooldown-aware kite: {type:'kite', unit:string, target_unit?:string, target_tag?:integer, selection?:object, duration_seconds?:number, retreat_distance?:number}",
            "- hold/stop: {type:'hold'|'hold_position'|'stop', unit:string, selection?:object, queued?:boolean}",
            "- rally: {type:'rally', building:string, location?:string, x?:number, y?:number, target_unit?:string, target_tag?:integer, selection?:object}",
            "- wait: {type:'wait', seconds:number}",
            "- wait until condition: {type:'wait_until', condition:'minerals'|'vespene'|'supply_left'|'supply_used'|'supply_cap'|'army_supply'|'structure_count'|'structure_ready'|'structure_pending'|'unit_count'|'townhall_count'|'upgrade_complete'|'game_time'|'enemy_unit_count'|'enemy_structure_count'|'idle_structure_count'|'producer_available'|'cargo_used'|'unit_near_location'|'enemy_near_location'|'under_attack', target?:string, location?:string, x?:number, y?:number, radius?:number, selection?:object, at_least:number, timeout_seconds?:number, on_timeout?:'replan'|'fail'}",
            "- gather minerals / gather gas: {type:'gather', unit:'worker', resource:'minerals'|'vespene', workers?:integer, location?:string, x?:number, y?:number, target_tag?:integer, selection?:object, queued?:boolean}",
            "- return cargo: {type:'return_cargo', unit:'worker'|'mule', selection?:object, queued?:boolean}",
            "- distribute workers: {type:'distribute_workers', mineral_to_gas_ratio?:number}",
            "- train unit: {type:'train', unit:string, count?:integer, producer_selection?:object}",
            "- produce until absolute count: {type:'produce_until', unit:string, target_count:integer, producer_selection?:object, reserve_minerals?:integer, reserve_vespene?:integer, reserve_supply?:integer, max_seconds?:number}",
            "- maintain background production: {type:'maintain_production', unit:string, target_count:integer, producer_selection?:object, reserve_minerals?:integer, reserve_vespene?:integer, reserve_supply?:integer, max_seconds?:number}",
            "- build structure: {type:'build', building:string, worker:'worker', count?:integer, location?:string, x?:number, y?:number, placement_mode?:'near'|'exact', max_distance?:integer, reserve_addon_space?:boolean, selection?:object}",
            "- expand: {type:'expand', count?:integer}",
            "- build add-on: {type:'build_addon', addon:string, count?:integer, selection?:object}",
            "- morph command center: {type:'morph', building:'orbital_command'|'planetary_fortress', selection?:object}",
            "- research upgrade: {type:'research', upgrade:string, researcher_selection?:object}",
            "- repair: {type:'repair', target?:string, target_tag?:integer, target_selector?:string, target_selection?:object, workers?:integer, selection?:object}",
            "- use ability: {type:'use_ability', ability:string, actor?:string, target_unit?:string, target_tag?:integer, location?:string, x?:number, y?:number, selection?:{mode:string,count?:integer,tags?:integer[]}, queued?:boolean}",
            "- scan/call down MULE: {type:'scan'|'call_down_mule', location?:string, x?:number, y?:number, selection?:object, queued?:boolean}",
            "- supply drop: {type:'supply_drop', target_unit?:'supply_depot', target_tag?:integer, selection?:object, queued?:boolean}",
            "- transform: {type:'transform', actor?:string, ability?:string, mode?:string, selection?:object, queued?:boolean}",
            "- lift: {type:'lift', actor:string, selection?:object, queued?:boolean}",
            "- land: {type:'land', actor:string, location?:string, x?:number, y?:number, selection?:object, queued?:boolean}",
            "- land on add-on: {type:'land_on_addon', actor:'barracks'|'factory'|'starport', target_addon?:string, target_addon_tag?:integer, selection?:object, queued?:boolean}",
            "- load: {type:'load', actor:string, target_unit?:string, target_tag?:integer, target_selection?:object, count?:integer, selection?:object, queued?:boolean}",
            "- unload: {type:'unload', actor:string, target_unit?:string, passenger_tag?:integer, location?:string, x?:number, y?:number, selection?:object, queued?:boolean}",
            "- cancel/salvage: {type:'cancel'|'salvage', actor?:string, ability?:string, target?:string, selection?:object, queued?:boolean}",
            "- nuke/replan: {type:'build_nuke'|'launch_nuke'|'replan', location?:string, x?:number, y?:number, reason?:string, selection?:object, queued?:boolean}",
            "Unit target selectors: " + ", ".join(TARGET_SELECTORS),
            build_command_prompt_section(),
            "Constraints:",
            "- Use only actions listed above.",
            f"- Keep plans short and never exceed {MAX_PLAN_ACTIONS} actions; structure counts cap at {MAX_STRUCTURE_ACTION_COUNT}, worker assignments at {MAX_WORKER_ASSIGNMENT_COUNT}, and train/selection counts at {MAX_SELECTION_COUNT}.",
            "- For abilities, use only canonical ability keys, never raw AbilityId enum names; runtime get_available_abilities is authoritative for energy/cooldown/live availability.",
            "- Use semantic locations from the catalog when possible; otherwise provide bounded x/y coordinates. Use placement_mode='exact' only when the strategy truly requires an exact wall/add-on placement; otherwise use 'near' or omit it.",
            "- When Game state JSON exposes runtime unit tags, put them in selection.tags, target_tag, target_addon_tag, passenger_tag, producer_selection.tags, or researcher_selection.tags instead of guessing by type.",
            "- For early economy requests at game start, prefer train scv and gather minerals.",
            "- Use wait_until minerals before a build when current minerals are too low but the plan should wait rather than fail.",
            "- After build supply_depot, use wait_until structure_ready target supply_depot before build barracks.",
            "- Use gather gas only after a ready refinery exists or after build refinery plus wait_until structure_ready target refinery.",
            "- Use wait_until unit_count target marine after train marine if a later attack needs the trained unit to exist.",
            "- Use train count for a fixed batch, produce_until for a blocking absolute army target, and maintain_production when production must continue while later actions execute.",
            "- Use move_and_wait before a dependent pickup, siege, drop, or timing action; use ordinary move only when immediate plan progression is intentional.",
            "- Use focus_fire or kite only for bounded tactical windows, then continue or replan.",
            "- Use only canonical entity keys from the Terran macro command surface below.",
            "- Read structures_ready, structures_pending, upgrades, resources, and supply from Game state JSON.",
            "- Before build/train/add-on/morph/research, satisfy every listed prerequisite and cost with prior wait_until/build actions.",
            "- After build/build_addon/morph/research, add the matching structure_ready or upgrade_complete wait before a dependent action.",
            "- Build barracks only after a supply depot is ready; build factory after barracks; build starport/armory after factory.",
            "- Use reserve_addon_space=true when placing barracks/factory/starport that later needs a Tech Lab or Reactor.",
            "- For scouting requests, move one worker through safe map coordinates near (35,42), (45,42), or (55,45).",
            "- For attack requests, use attack with marine when marines exist; otherwise choose economy or setup actions.",
            "- If the request is ambiguous, choose a safe economy action rather than inventing unsupported actions.",
            f"User strategy: {request.strategy}",
            "Game state JSON:",
            json.dumps(game_state, ensure_ascii=False),
        ]
    )


def _normalize_gemini_plan_payload(payload: Any) -> Any:
    if (
        isinstance(payload, dict)
        and "actions" not in payload
        and isinstance(payload.get("plan"), list)
    ):
        return {"actions": payload["plan"]}
    return payload


def _post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise PlannerError(
            f"Gemini API request failed with HTTP {exc.code}: {error_body}"
        ) from exc
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

    raise PlannerError(
        "Gemini API response did not contain output_text or model_output text"
    )
