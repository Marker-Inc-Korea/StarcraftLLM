from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple


NameTuple = Tuple[str, ...]

BALANCE_VERSION = "5.0.15"
BALANCE_SOURCE_URL = (
    "https://news.blizzard.com/en-us/article/24225313/starcraft-ii-5-0-15-patch-notes"
)
MAX_PLAN_ACTIONS = 24
MAX_REPLAN_CYCLES = 2
MAX_SELECTION_COUNT = 200
MAX_WORKER_ASSIGNMENT_COUNT = 100
MAX_STRUCTURE_ACTION_COUNT = 20


@dataclass(frozen=True)
class EntitySpec:
    """Static SC2 command target metadata without importing the runtime sc2 package.

    ``enum_name`` is the attribute name to resolve on burnysc2/python-sc2 enums at
    runtime, for example ``UnitTypeId.MARINE`` or ``UpgradeId.STIMPACK``.
    ``runtime_state_keys`` lists common lower-case keys observed in game-state
    summaries and executor caches.
    """

    key: str
    enum_name: str
    minerals: int
    vespene: int
    supply: Optional[float]
    producer: Optional[str]
    researcher: Optional[str]
    prerequisites: NameTuple
    required_addon: Optional[str]
    previous_upgrade: Optional[str]
    aliases: NameTuple
    runtime_state_keys: NameTuple


@dataclass(frozen=True)
class AbilitySpec:
    """Static LLM-facing Terran ability metadata.

    ``enum_name`` is the exact ``AbilityId`` attribute name resolved at runtime.
    ``target_kind`` is one of ``none``, ``point``, ``unit``, or ``mineral``.
    ``actors`` are canonical unit/structure keys that may issue the ability.
    """

    key: str
    enum_name: str
    target_kind: str
    actors: NameTuple
    target_alliance: str
    target_filter: str
    prerequisites: NameTuple
    aliases: NameTuple
    form_transition: Optional[str]


@dataclass(frozen=True)
class LocationSpec:
    """Static semantic map anchor exposed to planners."""

    key: str
    description: str
    aliases: NameTuple


@dataclass(frozen=True)
class SelectionSpec:
    """Static bounded-selection mode exposed to planners."""

    key: str
    description: str


@dataclass(frozen=True)
class CommandVerbSpec:
    """LLM-facing primitive command form and the registry it targets."""

    key: str
    description: str
    target_registry: str
    target_field: str
    example: str


@dataclass(frozen=True)
class AliasResolution:
    """Resolved catalog lookup result."""

    key: str
    category: str
    spec: EntitySpec


Registry = Mapping[str, EntitySpec]


_NORMALIZE_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


COMMAND_SURFACE: Tuple[CommandVerbSpec, ...] = (
    CommandVerbSpec(
        key="move",
        description="Move a bounded Terran unit group to coordinates or a semantic location.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"move","unit":"worker","x":35,"y":42}',
    ),
    CommandVerbSpec(
        key="attack_move",
        description="Attack-move a bounded Terran unit group toward coordinates or a semantic location.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"attack","unit":"marine","x":55,"y":45}',
    ),
    CommandVerbSpec(
        key="attack_enemy",
        description="Attack the nearest currently visible enemy.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"attack_enemy","unit":"marine"}',
    ),
    CommandVerbSpec(
        key="patrol",
        description="Patrol a Terran unit group toward a safe map coordinate.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"patrol","unit":"marine","x":55,"y":45}',
    ),
    CommandVerbSpec(
        key="hold_position",
        description="Hold a Terran unit group in place.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"hold","unit":"marine"}',
    ),
    CommandVerbSpec(
        key="stop",
        description="Stop a Terran unit group's current order.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"stop","unit":"marine"}',
    ),
    CommandVerbSpec(
        key="rally",
        description="Set a production structure rally point.",
        target_registry="STRUCTURE_SPECS",
        target_field="building",
        example='{"type":"rally","building":"barracks","x":45,"y":42}',
    ),
    CommandVerbSpec(
        key="wait",
        description="Pause execution for a bounded number of game-clock seconds.",
        target_registry="none",
        target_field="seconds",
        example='{"type":"wait","seconds":2}',
    ),
    CommandVerbSpec(
        key="wait_until",
        description="Pause until an observed resource, supply, tech, or time condition is met.",
        target_registry="condition",
        target_field="condition",
        example='{"type":"wait_until","condition":"minerals","at_least":100}',
    ),
    CommandVerbSpec(
        key="gather",
        description="Assign workers to minerals or ready refineries.",
        target_registry="resource",
        target_field="resource",
        example='{"type":"gather","resource":"minerals","workers":8}',
    ),
    CommandVerbSpec(
        key="distribute_workers",
        description="Rebalance workers between mineral and gas income.",
        target_registry="none",
        target_field="mineral_to_gas_ratio",
        example='{"type":"distribute_workers","mineral_to_gas_ratio":2}',
    ),
    CommandVerbSpec(
        key="train",
        description="Queue Terran unit production from an available producer.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"train","unit":"marine","count":1}',
    ),
    CommandVerbSpec(
        key="build",
        description="Order an SCV to construct a Terran structure at a safe or semantic placement.",
        target_registry="STRUCTURE_SPECS",
        target_field="building",
        example='{"type":"build","building":"barracks"}',
    ),
    CommandVerbSpec(
        key="build_addon",
        description="Build a production add-on on a barracks, factory, or starport.",
        target_registry="ADDON_SPECS",
        target_field="addon",
        example='{"type":"build_addon","addon":"barracks_tech_lab"}',
    ),
    CommandVerbSpec(
        key="morph",
        description="Morph a command center into an orbital command or planetary fortress.",
        target_registry="MORPH_SPECS",
        target_field="building",
        example='{"type":"morph","building":"orbital_command"}',
    ),
    CommandVerbSpec(
        key="research",
        description="Research a Terran upgrade from the required tech structure/add-on.",
        target_registry="UPGRADE_SPECS",
        target_field="upgrade",
        example='{"type":"research","upgrade":"stimpack"}',
    ),
    CommandVerbSpec(
        key="expand",
        description="Build command centers at the next safe expansion locations.",
        target_registry="STRUCTURE_SPECS",
        target_field="count",
        example='{"type":"expand","count":1}',
    ),
    CommandVerbSpec(
        key="repair",
        description="Assign SCVs to repair a damaged Terran unit or structure.",
        target_registry="REPAIRABLE_TARGET_KEYS",
        target_field="target",
        example='{"type":"repair","target":"barracks","workers":1}',
    ),
    CommandVerbSpec(
        key="use_ability",
        description="Use a static allowlisted Terran ability with optional semantic location, target unit, bounded selection, and queued flag.",
        target_registry="ABILITY_SPECS",
        target_field="ability",
        example='{"type":"use_ability","ability":"stim_marine","actor":"marine","selection":{"mode":"ready","count":12}}',
    ),
    CommandVerbSpec(
        key="scan",
        description="Scanner sweep a semantic location or bounded coordinates.",
        target_registry="LOCATION_SPECS",
        target_field="location",
        example='{"type":"scan","location":"enemy_main"}',
    ),
    CommandVerbSpec(
        key="call_down_mule",
        description="Call down a MULE at a mineral location.",
        target_registry="LOCATION_SPECS",
        target_field="location",
        example='{"type":"call_down_mule","location":"nearest_mineral"}',
    ),
    CommandVerbSpec(
        key="supply_drop",
        description="Use extra supplies on a friendly supply depot target.",
        target_registry="ABILITY_SPECS",
        target_field="target_unit",
        example='{"type":"supply_drop","target_unit":"supply_depot"}',
    ),
    CommandVerbSpec(
        key="transform",
        description="Transform Terran units or structures between supported modes.",
        target_registry="ABILITY_SPECS",
        target_field="ability",
        example='{"type":"transform","ability":"siege_mode","actor":"siege_tank"}',
    ),
    CommandVerbSpec(
        key="lift",
        description="Lift a supported Terran production structure or town hall.",
        target_registry="ABILITY_SPECS",
        target_field="actor",
        example='{"type":"lift","actor":"barracks"}',
    ),
    CommandVerbSpec(
        key="land",
        description="Land a supported flying Terran structure at a semantic location or coordinates.",
        target_registry="LOCATION_SPECS",
        target_field="location",
        example='{"type":"land","actor":"barracks","location":"proxy"}',
    ),
    CommandVerbSpec(
        key="load",
        description="Load a unit into a bunker, medivac, or command center transport.",
        target_registry="ABILITY_SPECS",
        target_field="target_unit",
        example='{"type":"load","actor":"medivac","target_unit":"marine"}',
    ),
    CommandVerbSpec(
        key="unload",
        description="Unload all or one unit from a bunker, medivac, or command center transport.",
        target_registry="ABILITY_SPECS",
        target_field="actor",
        example='{"type":"unload","actor":"medivac","location":"enemy_natural"}',
    ),
    CommandVerbSpec(
        key="cancel",
        description="Cancel a supported Terran order, queue item, add-on, morph, nuke, or lock-on.",
        target_registry="ABILITY_SPECS",
        target_field="ability",
        example='{"type":"cancel","ability":"cancel_build_in_progress","actor":"barracks"}',
    ),
    CommandVerbSpec(
        key="salvage",
        description="Salvage a bunker or sensor tower.",
        target_registry="ABILITY_SPECS",
        target_field="actor",
        example='{"type":"salvage","actor":"bunker"}',
    ),
    CommandVerbSpec(
        key="build_nuke",
        description="Build a tactical nuke at a Ghost Academy.",
        target_registry="ABILITY_SPECS",
        target_field="actor",
        example='{"type":"build_nuke"}',
    ),
    CommandVerbSpec(
        key="launch_nuke",
        description="Call down a tactical nuke at a semantic location or coordinates.",
        target_registry="LOCATION_SPECS",
        target_field="location",
        example='{"type":"launch_nuke","location":"enemy_main"}',
    ),
    CommandVerbSpec(
        key="replan",
        description="Request bounded closed-loop replanning with a short reason.",
        target_registry="none",
        target_field="reason",
        example='{"type":"replan","reason":"ability unavailable"}',
    ),
)


def normalize_name(value: str) -> str:
    """Normalize user/LLM names to canonical snake_case lookup keys."""

    normalized = _NORMALIZE_RE.sub("_", value.strip().lower()).strip("_")
    return normalized


def _state_keys(*names: str) -> NameTuple:
    keys = []
    for name in names:
        normalized = normalize_name(name)
        compact = normalized.replace("_", "")
        keys.extend([normalized, compact])
    return tuple(dict.fromkeys(keys))


def _spec(
    key: str,
    enum_name: str,
    minerals: int,
    vespene: int,
    supply: Optional[float] = None,
    producer: Optional[str] = None,
    researcher: Optional[str] = None,
    prerequisites: Sequence[str] = (),
    required_addon: Optional[str] = None,
    previous_upgrade: Optional[str] = None,
    aliases: Sequence[str] = (),
    runtime_state_keys: Sequence[str] = (),
) -> EntitySpec:
    alias_values = tuple(
        dict.fromkeys(tuple(aliases) + (key.replace("_", " "), key.replace("_", "")))
    )
    state_values = (
        tuple(runtime_state_keys) if runtime_state_keys else _state_keys(key, enum_name)
    )
    return EntitySpec(
        key=key,
        enum_name=enum_name,
        minerals=minerals,
        vespene=vespene,
        supply=supply,
        producer=producer,
        researcher=researcher,
        prerequisites=tuple(prerequisites),
        required_addon=required_addon,
        previous_upgrade=previous_upgrade,
        aliases=alias_values,
        runtime_state_keys=tuple(dict.fromkeys(state_values)),
    )


def _registry(items: Iterable[EntitySpec]) -> Registry:
    return MappingProxyType({item.key: item for item in items})


UNIT_SPECS: Registry = _registry(
    (
        _spec(
            "scv",
            "SCV",
            50,
            0,
            1,
            producer="command_center",
            aliases=("worker", "workers", "일꾼", "건설로봇"),
            runtime_state_keys=("worker", "workers", "scv"),
        ),
        _spec(
            "marine",
            "MARINE",
            50,
            0,
            1,
            producer="barracks",
            prerequisites=("barracks",),
            aliases=("marines", "마린", "해병"),
        ),
        _spec(
            "marauder",
            "MARAUDER",
            100,
            25,
            2,
            producer="barracks",
            prerequisites=("barracks",),
            required_addon="barracks_tech_lab",
            aliases=("marauders", "불곰"),
        ),
        _spec(
            "reaper",
            "REAPER",
            50,
            50,
            1,
            producer="barracks",
            prerequisites=("barracks",),
            aliases=("reapers", "사신"),
        ),
        _spec(
            "ghost",
            "GHOST",
            150,
            125,
            2,
            producer="barracks",
            prerequisites=("barracks", "ghost_academy"),
            required_addon="barracks_tech_lab",
            aliases=("ghosts", "유령"),
        ),
        _spec(
            "hellion",
            "HELLION",
            100,
            0,
            2,
            producer="factory",
            prerequisites=("factory",),
            aliases=("hellions", "화염차"),
        ),
        _spec(
            "hellbat",
            "HELLIONTANK",
            100,
            0,
            2,
            producer="factory",
            prerequisites=("factory", "armory"),
            aliases=("hellion_tank", "helliontank", "hellbats", "화염기갑병", "화염 기갑병"),
        ),
        _spec(
            "widow_mine",
            "WIDOWMINE",
            75,
            25,
            2,
            producer="factory",
            prerequisites=("factory",),
            aliases=("widowmine", "widow mines", "widow_mines", "땅거미지뢰", "땅거미 지뢰"),
        ),
        _spec(
            "cyclone",
            "CYCLONE",
            125,
            50,
            2,
            producer="factory",
            prerequisites=("factory",),
            aliases=("cyclones", "사이클론"),
        ),
        _spec(
            "siege_tank",
            "SIEGETANK",
            150,
            125,
            3,
            producer="factory",
            prerequisites=("factory",),
            required_addon="factory_tech_lab",
            aliases=("siegetank", "tank", "tanks", "공성전차", "공성 전차"),
        ),
        _spec(
            "thor",
            "THOR",
            300,
            200,
            6,
            producer="factory",
            prerequisites=("factory", "armory"),
            required_addon="factory_tech_lab",
            aliases=("thors", "토르"),
        ),
        _spec(
            "viking",
            "VIKINGFIGHTER",
            125,
            75,
            2,
            producer="starport",
            prerequisites=("starport",),
            aliases=("viking_fighter", "vikingfighter", "vikings", "바이킹"),
        ),
        _spec(
            "medivac",
            "MEDIVAC",
            100,
            100,
            2,
            producer="starport",
            prerequisites=("starport",),
            aliases=("medivacs", "의료선"),
        ),
        _spec(
            "liberator",
            "LIBERATOR",
            150,
            150,
            3,
            producer="starport",
            prerequisites=("starport",),
            aliases=("liberators", "해방선"),
        ),
        _spec(
            "raven",
            "RAVEN",
            100,
            150,
            2,
            producer="starport",
            prerequisites=("starport",),
            required_addon="starport_tech_lab",
            aliases=("ravens", "밤까마귀"),
        ),
        _spec(
            "banshee",
            "BANSHEE",
            150,
            100,
            3,
            producer="starport",
            prerequisites=("starport",),
            required_addon="starport_tech_lab",
            aliases=("banshees", "밴시"),
        ),
        _spec(
            "battlecruiser",
            "BATTLECRUISER",
            400,
            300,
            6,
            producer="starport",
            prerequisites=("starport", "fusion_core"),
            required_addon="starport_tech_lab",
            aliases=("bc", "battle cruiser", "battlecruisers", "전투순양함", "전투 순양함"),
        ),
    )
)


SPECIAL_UNIT_SPECS: Registry = _registry(
    (
        _spec(
            "mule",
            "MULE",
            0,
            0,
            0,
            aliases=("mules", "지게로봇", "지게 로봇"),
        ),
    )
)


CONTROLLABLE_UNIT_SPECS: Registry = MappingProxyType(
    {**UNIT_SPECS, **SPECIAL_UNIT_SPECS}
)

# Canonical units that expose an actual basic attack order in standard melee.
# Medivacs and Ravens are support casters, while Widow Mines attack
# autonomously only after burrowing, so none of those should be advertised to
# an LLM as valid basic attack-move actors.
ATTACK_CAPABLE_UNIT_KEYS: NameTuple = (
    "scv",
    "marine",
    "marauder",
    "reaper",
    "ghost",
    "hellion",
    "hellbat",
    "cyclone",
    "siege_tank",
    "thor",
    "viking",
    "liberator",
    "banshee",
    "battlecruiser",
)

# These canonical structures expose basic movement orders only while their
# live UnitTypeId is the corresponding flying form. Runtime form filtering
# prevents grounded structures from receiving movement orders during lift.
FLYING_STRUCTURE_ACTOR_KEYS: NameTuple = (
    "command_center",
    "orbital_command",
    "barracks",
    "factory",
    "starport",
)


STRUCTURE_SPECS: Registry = _registry(
    (
        _spec(
            "command_center",
            "COMMANDCENTER",
            400,
            0,
            producer="scv",
            aliases=("cc", "townhall", "town hall", "사령부"),
            runtime_state_keys=(
                "commandcenter",
                "command_center",
                "townhall",
                "townhalls",
            ),
        ),
        _spec(
            "supply_depot",
            "SUPPLYDEPOT",
            100,
            0,
            producer="scv",
            aliases=("depot", "supply", "supply depot", "서플", "보급고"),
            runtime_state_keys=("supplydepot", "supply_depot"),
        ),
        _spec(
            "refinery",
            "REFINERY",
            75,
            0,
            producer="scv",
            aliases=("gas", "vespene", "정제소"),
        ),
        _spec(
            "barracks",
            "BARRACKS",
            150,
            0,
            producer="scv",
            prerequisites=("supply_depot",),
            aliases=("rax", "배럭", "병영"),
        ),
        _spec(
            "engineering_bay",
            "ENGINEERINGBAY",
            125,
            0,
            producer="scv",
            aliases=("ebay", "engineering bay", "공학연구소", "공학 연구소"),
        ),
        _spec(
            "bunker",
            "BUNKER",
            100,
            0,
            producer="scv",
            prerequisites=("barracks",),
            aliases=("벙커",),
        ),
        _spec(
            "missile_turret",
            "MISSILETURRET",
            100,
            0,
            producer="scv",
            prerequisites=("engineering_bay",),
            aliases=("turret", "missile turret", "미사일포탑", "미사일 포탑"),
        ),
        _spec(
            "sensor_tower",
            "SENSORTOWER",
            100,
            50,
            producer="scv",
            prerequisites=("engineering_bay",),
            aliases=("sensor", "sensor tower", "감지탑", "감지 탑"),
        ),
        _spec(
            "factory",
            "FACTORY",
            150,
            100,
            producer="scv",
            prerequisites=("barracks",),
            aliases=("fact", "팩토리", "군수공장", "군수 공장"),
        ),
        _spec(
            "ghost_academy",
            "GHOSTACADEMY",
            150,
            50,
            producer="scv",
            prerequisites=("barracks",),
            aliases=("ghost academy", "academy", "유령사관학교", "유령 사관학교"),
        ),
        _spec(
            "starport",
            "STARPORT",
            150,
            100,
            producer="scv",
            prerequisites=("factory",),
            aliases=("스타포트", "우주공항", "우주 공항"),
        ),
        _spec(
            "armory",
            "ARMORY",
            150,
            50,
            producer="scv",
            prerequisites=("factory",),
            aliases=("무기고",),
        ),
        _spec(
            "fusion_core",
            "FUSIONCORE",
            150,
            150,
            producer="scv",
            prerequisites=("starport",),
            aliases=("fusion core", "융합로"),
        ),
    )
)


ADDON_SPECS: Registry = _registry(
    (
        _spec(
            "barracks_tech_lab",
            "BARRACKSTECHLAB",
            50,
            25,
            producer="barracks",
            prerequisites=("barracks",),
            aliases=("barracks tech lab", "rax tech lab", "barracks techlab", "병영 기술실"),
        ),
        _spec(
            "barracks_reactor",
            "BARRACKSREACTOR",
            50,
            50,
            producer="barracks",
            prerequisites=("barracks",),
            aliases=("barracks reactor", "rax reactor", "병영 반응로"),
        ),
        _spec(
            "factory_tech_lab",
            "FACTORYTECHLAB",
            50,
            25,
            producer="factory",
            prerequisites=("factory",),
            aliases=("factory tech lab", "factory techlab", "군수공장 기술실"),
        ),
        _spec(
            "factory_reactor",
            "FACTORYREACTOR",
            50,
            50,
            producer="factory",
            prerequisites=("factory",),
            aliases=("factory reactor", "군수공장 반응로"),
        ),
        _spec(
            "starport_tech_lab",
            "STARPORTTECHLAB",
            50,
            25,
            producer="starport",
            prerequisites=("starport",),
            aliases=("starport tech lab", "starport techlab", "우주공항 기술실"),
        ),
        _spec(
            "starport_reactor",
            "STARPORTREACTOR",
            50,
            50,
            producer="starport",
            prerequisites=("starport",),
            aliases=("starport reactor", "우주공항 반응로"),
        ),
    )
)


MORPH_SPECS: Registry = _registry(
    (
        _spec(
            "orbital_command",
            "ORBITALCOMMAND",
            150,
            0,
            producer="command_center",
            prerequisites=("barracks",),
            aliases=("orbital", "oc", "orbital command", "궤도사령부", "궤도 사령부"),
        ),
        _spec(
            "planetary_fortress",
            "PLANETARYFORTRESS",
            150,
            150,
            producer="command_center",
            prerequisites=("engineering_bay",),
            aliases=("planetary", "pf", "planetary fortress", "행성요새", "행성 요새"),
        ),
    )
)


def _upgrade(
    key: str,
    enum_name: str,
    minerals: int,
    vespene: int,
    researcher: str,
    prerequisites: Sequence[str] = (),
    required_addon: Optional[str] = None,
    previous_upgrade: Optional[str] = None,
    aliases: Sequence[str] = (),
) -> EntitySpec:
    return _spec(
        key,
        enum_name,
        minerals,
        vespene,
        supply=None,
        researcher=researcher,
        prerequisites=prerequisites,
        required_addon=required_addon,
        previous_upgrade=previous_upgrade,
        aliases=aliases,
        runtime_state_keys=_state_keys(key, enum_name),
    )


UPGRADE_SPECS: Registry = _registry(
    (
        _upgrade(
            "terran_infantry_weapons_level_1",
            "TERRANINFANTRYWEAPONSLEVEL1",
            100,
            100,
            "engineering_bay",
            aliases=("infantry weapons 1", "bio attack 1", "공1업"),
        ),
        _upgrade(
            "terran_infantry_weapons_level_2",
            "TERRANINFANTRYWEAPONSLEVEL2",
            150,
            150,
            "engineering_bay",
            prerequisites=("armory",),
            previous_upgrade="terran_infantry_weapons_level_1",
            aliases=("infantry weapons 2", "bio attack 2", "공2업"),
        ),
        _upgrade(
            "terran_infantry_weapons_level_3",
            "TERRANINFANTRYWEAPONSLEVEL3",
            200,
            200,
            "engineering_bay",
            prerequisites=("armory",),
            previous_upgrade="terran_infantry_weapons_level_2",
            aliases=("infantry weapons 3", "bio attack 3", "공3업"),
        ),
        _upgrade(
            "terran_infantry_armor_level_1",
            "TERRANINFANTRYARMORSLEVEL1",
            100,
            100,
            "engineering_bay",
            aliases=("infantry armor 1", "bio armor 1", "방1업"),
        ),
        _upgrade(
            "terran_infantry_armor_level_2",
            "TERRANINFANTRYARMORSLEVEL2",
            150,
            150,
            "engineering_bay",
            prerequisites=("armory",),
            previous_upgrade="terran_infantry_armor_level_1",
            aliases=("infantry armor 2", "bio armor 2", "방2업"),
        ),
        _upgrade(
            "terran_infantry_armor_level_3",
            "TERRANINFANTRYARMORSLEVEL3",
            200,
            200,
            "engineering_bay",
            prerequisites=("armory",),
            previous_upgrade="terran_infantry_armor_level_2",
            aliases=("infantry armor 3", "bio armor 3", "방3업"),
        ),
        _upgrade(
            "stimpack",
            "STIMPACK",
            100,
            100,
            "barracks_tech_lab",
            required_addon="barracks_tech_lab",
            aliases=("stim", "스팀팩"),
        ),
        _upgrade(
            "combat_shield",
            "SHIELDWALL",
            100,
            100,
            "barracks_tech_lab",
            required_addon="barracks_tech_lab",
            aliases=("combat shields", "shieldwall", "전투방패"),
        ),
        _upgrade(
            "concussive_shells",
            "PUNISHERGRENADES",
            50,
            50,
            "barracks_tech_lab",
            required_addon="barracks_tech_lab",
            aliases=("concussive shell", "punisher grenades", "충격탄"),
        ),
        _upgrade(
            "personal_cloaking",
            "PERSONALCLOAKING",
            150,
            150,
            "ghost_academy",
            prerequisites=("ghost_academy",),
            aliases=("ghost cloak", "cloaking field", "개인 은폐"),
        ),
        _upgrade(
            "infernal_pre_igniter",
            "HIGHCAPACITYBARRELS",
            100,
            100,
            "factory_tech_lab",
            required_addon="factory_tech_lab",
            aliases=("blue flame", "high capacity barrels", "지옥불 조기점화기"),
        ),
        _upgrade(
            "drilling_claws",
            "DRILLCLAWS",
            75,
            75,
            "factory_tech_lab",
            required_addon="factory_tech_lab",
            aliases=("drill claws", "drilling claws", "천공 발톱"),
        ),
        _upgrade(
            "smart_servos",
            "SMARTSERVOS",
            100,
            100,
            "factory_tech_lab",
            required_addon="factory_tech_lab",
            aliases=("smart servo", "smart servos", "스마트 서보"),
        ),
        _upgrade(
            "terran_vehicle_weapons_level_1",
            "TERRANVEHICLEWEAPONSLEVEL1",
            100,
            100,
            "armory",
            prerequisites=("armory",),
            aliases=("vehicle weapons 1", "mech attack 1"),
        ),
        _upgrade(
            "terran_vehicle_weapons_level_2",
            "TERRANVEHICLEWEAPONSLEVEL2",
            175,
            175,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_vehicle_weapons_level_1",
            aliases=("vehicle weapons 2", "mech attack 2"),
        ),
        _upgrade(
            "terran_vehicle_weapons_level_3",
            "TERRANVEHICLEWEAPONSLEVEL3",
            250,
            250,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_vehicle_weapons_level_2",
            aliases=("vehicle weapons 3", "mech attack 3"),
        ),
        _upgrade(
            "terran_ship_weapons_level_1",
            "TERRANSHIPWEAPONSLEVEL1",
            100,
            100,
            "armory",
            prerequisites=("armory",),
            aliases=("ship weapons 1", "air attack 1"),
        ),
        _upgrade(
            "terran_ship_weapons_level_2",
            "TERRANSHIPWEAPONSLEVEL2",
            175,
            175,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_ship_weapons_level_1",
            aliases=("ship weapons 2", "air attack 2"),
        ),
        _upgrade(
            "terran_ship_weapons_level_3",
            "TERRANSHIPWEAPONSLEVEL3",
            250,
            250,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_ship_weapons_level_2",
            aliases=("ship weapons 3", "air attack 3"),
        ),
        _upgrade(
            "terran_vehicle_and_ship_armor_level_1",
            "TERRANVEHICLEANDSHIPARMORSLEVEL1",
            100,
            100,
            "armory",
            prerequisites=("armory",),
            aliases=("vehicle armor 1", "ship armor 1", "mech armor 1", "air armor 1"),
        ),
        _upgrade(
            "terran_vehicle_and_ship_armor_level_2",
            "TERRANVEHICLEANDSHIPARMORSLEVEL2",
            175,
            175,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_vehicle_and_ship_armor_level_1",
            aliases=("vehicle armor 2", "ship armor 2", "mech armor 2", "air armor 2"),
        ),
        _upgrade(
            "terran_vehicle_and_ship_armor_level_3",
            "TERRANVEHICLEANDSHIPARMORSLEVEL3",
            250,
            250,
            "armory",
            prerequisites=("armory",),
            previous_upgrade="terran_vehicle_and_ship_armor_level_2",
            aliases=("vehicle armor 3", "ship armor 3", "mech armor 3", "air armor 3"),
        ),
        _upgrade(
            "banshee_cloaking_field",
            "BANSHEECLOAK",
            100,
            100,
            "starport_tech_lab",
            required_addon="starport_tech_lab",
            aliases=("banshee cloak", "cloak banshee", "밴시 은폐"),
        ),
        _upgrade(
            "hyperflight_rotors",
            "BANSHEESPEED",
            125,
            125,
            "starport_tech_lab",
            required_addon="starport_tech_lab",
            aliases=("banshee speed", "hyperflight rotors", "밴시 속업"),
        ),
        _upgrade(
            "hi_sec_auto_tracking",
            "HISECAUTOTRACKING",
            100,
            100,
            "engineering_bay",
            aliases=("hi sec auto tracking", "turret range", "고급 탄도 추적"),
        ),
        _upgrade(
            "terran_building_armor",
            "TERRANBUILDINGARMOR",
            150,
            150,
            "engineering_bay",
            aliases=("building armor", "structure armor", "건물 장갑"),
        ),
        _upgrade(
            "cyclone_lock_on_damage",
            "CYCLONELOCKONDAMAGEUPGRADE",
            100,
            100,
            "factory_tech_lab",
            required_addon="factory_tech_lab",
            aliases=("cyclone damage", "mag field accelerators", "사이클론 피해"),
        ),
        _upgrade(
            "interference_matrix",
            "INTERFERENCEMATRIX",
            50,
            50,
            "starport_tech_lab",
            required_addon="starport_tech_lab",
            aliases=("raven interference matrix", "방해 매트릭스"),
        ),
        _upgrade(
            "advanced_ballistics",
            "LIBERATORAGRANGEUPGRADE",
            150,
            150,
            "fusion_core",
            prerequisites=("fusion_core",),
            aliases=("liberator range", "liberator ag range", "해방선 사업"),
        ),
        _upgrade(
            "medivac_caduceus_reactor",
            "MEDIVACCADUCEUSREACTOR",
            100,
            100,
            "fusion_core",
            prerequisites=("fusion_core",),
            aliases=("medivac energy", "caduceus reactor", "의료선 에너지"),
        ),
        _upgrade(
            "battlecruiser_weapon_refit",
            "BATTLECRUISERENABLESPECIALIZATIONS",
            150,
            150,
            "fusion_core",
            prerequisites=("fusion_core",),
            aliases=("yamato", "yamato cannon", "야마토포"),
        ),
    )
)


def _ability(
    key: str,
    enum_name: str,
    target_kind: str,
    actors: Sequence[str],
    target_alliance: str = "any",
    target_filter: str = "any",
    prerequisites: Sequence[str] = (),
    aliases: Sequence[str] = (),
    form_transition: Optional[str] = None,
) -> AbilitySpec:
    alias_values = tuple(
        dict.fromkeys(tuple(aliases) + (key.replace("_", " "), key.replace("_", "")))
    )
    return AbilitySpec(
        key=key,
        enum_name=enum_name,
        target_kind=target_kind,
        actors=tuple(actors),
        target_alliance=target_alliance,
        target_filter=target_filter,
        prerequisites=tuple(prerequisites),
        aliases=alias_values,
        form_transition=form_transition,
    )


def _ability_registry(items: Iterable[AbilitySpec]) -> Mapping[str, AbilitySpec]:
    return MappingProxyType({item.key: item for item in items})


ABILITY_SPECS: Mapping[str, AbilitySpec] = _ability_registry(
    (
        _ability(
            "stim_marine",
            "EFFECT_STIM_MARINE",
            "none",
            ("marine",),
            prerequisites=("stimpack",),
            aliases=("stim", "marine_stim"),
        ),
        _ability(
            "stim_marauder",
            "EFFECT_STIM_MARAUDER",
            "none",
            ("marauder",),
            prerequisites=("stimpack",),
            aliases=("marauder_stim",),
        ),
        _ability(
            "kd8_charge",
            "KD8CHARGE_KD8CHARGE",
            "point",
            ("reaper",),
            target_alliance="enemy",
            aliases=("kd8", "grenade"),
        ),
        _ability(
            "ghost_cloak_on",
            "BEHAVIOR_CLOAKON_GHOST",
            "none",
            ("ghost",),
            prerequisites=("personal_cloaking",),
        ),
        _ability("ghost_cloak_off", "BEHAVIOR_CLOAKOFF_GHOST", "none", ("ghost",)),
        _ability("ghost_hold_fire_on", "BEHAVIOR_HOLDFIREON_GHOST", "none", ("ghost",)),
        _ability(
            "ghost_hold_fire_off", "BEHAVIOR_HOLDFIREOFF_GHOST", "none", ("ghost",)
        ),
        _ability(
            "ghost_snipe",
            "EFFECT_GHOSTSNIPE",
            "unit",
            ("ghost",),
            target_alliance="enemy",
            target_filter="biological_unit",
        ),
        _ability("ghost_emp", "EMP_EMP", "point", ("ghost",), target_alliance="enemy"),
        _ability(
            "ghost_nuke_call_down",
            "TACNUKESTRIKE_NUKECALLDOWN",
            "point",
            ("ghost",),
            target_alliance="enemy",
        ),
        _ability(
            "banshee_cloak_on",
            "BEHAVIOR_CLOAKON_BANSHEE",
            "none",
            ("banshee",),
            prerequisites=("banshee_cloaking_field",),
        ),
        _ability(
            "banshee_cloak_off", "BEHAVIOR_CLOAKOFF_BANSHEE", "none", ("banshee",)
        ),
        _ability(
            "morph_hellbat",
            "MORPH_HELLBAT",
            "none",
            ("hellion",),
            prerequisites=("armory",),
            form_transition="hellbat",
        ),
        _ability(
            "morph_hellion",
            "MORPH_HELLION",
            "none",
            ("hellbat",),
            form_transition="hellion",
        ),
        _ability(
            "widow_mine_burrow_down", "BURROWDOWN_WIDOWMINE", "none", ("widow_mine",)
        ),
        _ability("widow_mine_burrow_up", "BURROWUP_WIDOWMINE", "none", ("widow_mine",)),
        _ability(
            "cyclone_lock_on",
            "LOCKON_LOCKON",
            "unit",
            ("cyclone",),
            target_alliance="enemy",
        ),
        _ability("cyclone_cancel_lock_on", "CANCEL_LOCKON", "none", ("cyclone",)),
        _ability(
            "siege_mode",
            "SIEGEMODE_SIEGEMODE",
            "none",
            ("siege_tank",),
            form_transition="siege_tank_sieged",
        ),
        _ability(
            "unsiege_mode",
            "UNSIEGE_UNSIEGE",
            "none",
            ("siege_tank",),
            form_transition="siege_tank",
        ),
        _ability(
            "thor_high_impact_mode", "MORPH_THORHIGHIMPACTMODE", "none", ("thor",)
        ),
        _ability("thor_explosive_mode", "MORPH_THOREXPLOSIVEMODE", "none", ("thor",)),
        _ability(
            "viking_assault_mode",
            "MORPH_VIKINGASSAULTMODE",
            "none",
            ("viking",),
            form_transition="viking_assault",
        ),
        _ability(
            "viking_fighter_mode",
            "MORPH_VIKINGFIGHTERMODE",
            "none",
            ("viking",),
            form_transition="viking",
        ),
        _ability(
            "medivac_afterburners",
            "EFFECT_MEDIVACIGNITEAFTERBURNERS",
            "none",
            ("medivac",),
            aliases=("boost", "afterburners"),
        ),
        _ability(
            "medivac_heal",
            "MEDIVACHEAL_HEAL",
            "unit",
            ("medivac",),
            target_alliance="friendly",
            target_filter="biological_unit",
        ),
        _ability(
            "liberator_ag_mode",
            "MORPH_LIBERATORAGMODE",
            "point",
            ("liberator",),
            target_alliance="enemy",
        ),
        _ability("liberator_aa_mode", "MORPH_LIBERATORAAMODE", "none", ("liberator",)),
        _ability(
            "raven_auto_turret", "BUILDAUTOTURRET_AUTOTURRET", "point", ("raven",)
        ),
        _ability(
            "raven_interference_matrix",
            "EFFECT_INTERFERENCEMATRIX",
            "unit",
            ("raven",),
            target_alliance="enemy",
            target_filter="mechanical_or_psionic_unit",
            prerequisites=("interference_matrix",),
        ),
        _ability(
            "raven_anti_armor_missile",
            "EFFECT_ANTIARMORMISSILE",
            "unit",
            ("raven",),
            target_alliance="enemy",
        ),
        _ability(
            "battlecruiser_tactical_jump",
            "EFFECT_TACTICALJUMP",
            "point",
            ("battlecruiser",),
        ),
        _ability(
            "battlecruiser_yamato",
            "YAMATO_YAMATOGUN",
            "unit",
            ("battlecruiser",),
            target_alliance="enemy",
            prerequisites=("battlecruiser_weapon_refit",),
        ),
        _ability(
            "scan",
            "SCANNERSWEEP_SCAN",
            "point",
            ("orbital_command",),
            aliases=("scanner_sweep",),
        ),
        _ability(
            "call_down_mule",
            "CALLDOWNMULE_CALLDOWNMULE",
            "mineral",
            ("orbital_command",),
            aliases=("mule", "calldown_mule"),
        ),
        _ability(
            "mule_gather",
            "HARVEST_GATHER_MULE",
            "mineral",
            ("mule",),
            aliases=("gather_mule",),
        ),
        _ability(
            "mule_repair",
            "EFFECT_REPAIR_MULE",
            "unit",
            ("mule",),
            target_alliance="friendly",
            target_filter="mechanical",
            aliases=("repair_mule",),
        ),
        _ability(
            "supply_drop",
            "SUPPLYDROP_SUPPLYDROP",
            "unit",
            ("orbital_command",),
            target_alliance="friendly",
            target_filter="supply_depot",
        ),
        _ability(
            "lower_supply_depot", "MORPH_SUPPLYDEPOT_LOWER", "none", ("supply_depot",)
        ),
        _ability(
            "raise_supply_depot", "MORPH_SUPPLYDEPOT_RAISE", "none", ("supply_depot",)
        ),
        _ability(
            "lift_command_center", "LIFT_COMMANDCENTER", "none", ("command_center",)
        ),
        _ability(
            "land_command_center", "LAND_COMMANDCENTER", "point", ("command_center",)
        ),
        _ability(
            "lift_orbital_command", "LIFT_ORBITALCOMMAND", "none", ("orbital_command",)
        ),
        _ability(
            "land_orbital_command", "LAND_ORBITALCOMMAND", "point", ("orbital_command",)
        ),
        _ability("lift_barracks", "LIFT_BARRACKS", "none", ("barracks",)),
        _ability("land_barracks", "LAND_BARRACKS", "point", ("barracks",)),
        _ability("lift_factory", "LIFT_FACTORY", "none", ("factory",)),
        _ability("land_factory", "LAND_FACTORY", "point", ("factory",)),
        _ability("lift_starport", "LIFT_STARPORT", "none", ("starport",)),
        _ability("land_starport", "LAND_STARPORT", "point", ("starport",)),
        _ability(
            "load_all_command_center",
            "LOADALL_COMMANDCENTER",
            "none",
            ("command_center", "orbital_command"),
        ),
        _ability(
            "unload_all_command_center",
            "UNLOADALL_COMMANDCENTER",
            "none",
            ("command_center", "orbital_command"),
        ),
        _ability(
            "unload_unit_command_center",
            "UNLOADUNIT_COMMANDCENTER",
            "unit",
            ("command_center", "orbital_command"),
            target_alliance="passenger",
            target_filter="worker_passenger",
        ),
        _ability(
            "load_bunker",
            "LOAD_BUNKER",
            "unit",
            ("bunker",),
            target_alliance="friendly",
            target_filter="bunker_loadable",
        ),
        _ability("unload_all_bunker", "UNLOADALL_BUNKER", "none", ("bunker",)),
        _ability(
            "unload_unit_bunker",
            "UNLOADUNIT_BUNKER",
            "unit",
            ("bunker",),
            target_alliance="passenger",
            target_filter="bunker_loadable",
        ),
        _ability(
            "load_medivac",
            "LOAD_MEDIVAC",
            "unit",
            ("medivac",),
            target_alliance="friendly",
            target_filter="medivac_loadable",
        ),
        _ability("unload_all_medivac", "UNLOADALLAT_MEDIVAC", "point", ("medivac",)),
        _ability(
            "unload_unit_medivac",
            "UNLOADUNIT_MEDIVAC",
            "unit",
            ("medivac",),
            target_alliance="passenger",
            target_filter="medivac_loadable",
        ),
        _ability(
            "build_nuke",
            "BUILD_NUKE",
            "none",
            ("ghost_academy",),
            prerequisites=("ghost_academy",),
        ),
        _ability(
            "launch_nuke",
            "TACNUKESTRIKE_NUKECALLDOWN",
            "point",
            ("ghost",),
            target_alliance="enemy",
        ),
        _ability("cancel_any", "CANCEL", "none", ("any",)),
        _ability(
            "cancel_build_in_progress", "CANCEL_BUILDINPROGRESS", "none", ("any",)
        ),
        _ability("cancel_queue_1", "CANCEL_QUEUE1", "none", ("any",)),
        _ability("cancel_queue_5", "CANCEL_QUEUE5", "none", ("any",)),
        _ability(
            "cancel_queue_addon",
            "CANCEL_QUEUEADDON",
            "none",
            ("barracks", "factory", "starport"),
        ),
        _ability("cancel_slot", "CANCEL_SLOT", "none", ("any",)),
        _ability(
            "cancel_slot_queue_cancel_to_selection",
            "CANCELSLOT_QUEUECANCELTOSELECTION",
            "none",
            ("any",),
        ),
        _ability(
            "cancel_slot_queue_passive", "CANCELSLOT_QUEUEPASSIVE", "none", ("any",)
        ),
        _ability(
            "cancel_slot_queue_passive_cancel_to_selection",
            "CANCELSLOT_QUEUEPASSIVECANCELTOSELECTION",
            "none",
            ("any",),
        ),
        _ability(
            "cancel_addon_barracks", "CANCEL_BARRACKSADDON", "none", ("barracks",)
        ),
        _ability("cancel_addon_factory", "CANCEL_FACTORYADDON", "none", ("factory",)),
        _ability(
            "cancel_addon_starport", "CANCEL_STARPORTADDON", "none", ("starport",)
        ),
        _ability(
            "cancel_morph_orbital", "CANCEL_MORPHORBITAL", "none", ("command_center",)
        ),
        _ability(
            "cancel_morph_planetary_fortress",
            "CANCEL_MORPHPLANETARYFORTRESS",
            "none",
            ("command_center",),
        ),
        _ability(
            "cancel_morph_thor_explosive_mode",
            "CANCEL_MORPHTHOREXPLOSIVEMODE",
            "none",
            ("thor",),
        ),
        _ability("cancel_lock_on", "CANCEL_LOCKON", "none", ("cyclone",)),
        _ability("cancel_nuke", "CANCEL_NUKE", "none", ("ghost_academy",)),
        _ability("cancel_last", "CANCEL_LAST", "none", ("any",)),
        _ability("salvage_bunker", "SALVAGEEFFECT_SALVAGE", "none", ("bunker",)),
        _ability(
            "salvage_sensor_tower", "SALVAGEEFFECT_SALVAGE", "none", ("sensor_tower",)
        ),
    )
)


TRANSFORM_ABILITY_KEYS: NameTuple = (
    "lower_supply_depot",
    "raise_supply_depot",
    "morph_hellbat",
    "morph_hellion",
    "widow_mine_burrow_down",
    "widow_mine_burrow_up",
    "siege_mode",
    "unsiege_mode",
    "thor_high_impact_mode",
    "thor_explosive_mode",
    "viking_assault_mode",
    "viking_fighter_mode",
    "liberator_ag_mode",
    "liberator_aa_mode",
    "ghost_cloak_on",
    "ghost_cloak_off",
    "ghost_hold_fire_on",
    "ghost_hold_fire_off",
    "banshee_cloak_on",
    "banshee_cloak_off",
)
LIFTABLE_STRUCTURE_KEYS: NameTuple = FLYING_STRUCTURE_ACTOR_KEYS
TRANSPORT_ACTOR_KEYS: NameTuple = (
    "command_center",
    "orbital_command",
    "bunker",
    "medivac",
)
SALVAGEABLE_STRUCTURE_KEYS: NameTuple = ("bunker", "sensor_tower")


LOCATION_SPECS: Mapping[str, LocationSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            LocationSpec("own_main", "Own starting main base.", ("main", "home")),
            LocationSpec("own_natural", "Own natural expansion.", ("natural", "nat")),
            LocationSpec("own_third", "Own third expansion.", ("third",)),
            LocationSpec("own_ramp", "Ramp between own main and natural.", ("ramp",)),
            LocationSpec(
                "enemy_main", "Enemy starting main base.", ("enemy", "enemy main")
            ),
            LocationSpec(
                "enemy_natural",
                "Enemy natural expansion.",
                ("enemy natural", "enemy nat"),
            ),
            LocationSpec("enemy_third", "Enemy third expansion.", ("enemy third",)),
            LocationSpec(
                "map_center", "Center of the playable map.", ("center", "middle")
            ),
            LocationSpec(
                "frontline", "Current midpoint/frontline between armies.", ("front",)
            ),
            LocationSpec(
                "retreat",
                "Safe retreat point near own army/base.",
                ("fallback", "fall back"),
            ),
            LocationSpec("proxy", "Forward proxy placement area.", ("proxy location",)),
            LocationSpec(
                "next_expansion", "Next safe expansion location.", ("next base",)
            ),
            LocationSpec(
                "nearest_enemy", "Nearest visible enemy unit.", ("nearest enemy",)
            ),
            LocationSpec(
                "nearest_enemy_structure",
                "Nearest visible enemy structure.",
                ("nearest enemy structure",),
            ),
            LocationSpec(
                "nearest_mineral",
                "Nearest mineral field patch.",
                ("mineral", "minerals"),
            ),
        )
    }
)


SELECTION_SPECS: Mapping[str, SelectionSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            SelectionSpec("all", "All matching actors."),
            SelectionSpec(
                "ready",
                "Ready/non-building actors with the requested ability available.",
            ),
            SelectionSpec("idle", "Idle matching actors first."),
            SelectionSpec(
                "closest", "Closest matching actors to the target location/unit."
            ),
            SelectionSpec("lowest_health", "Lowest-health matching actors first."),
        )
    }
)


TARGET_SELECTORS: NameTuple = (
    "nearest_enemy",
    "nearest_enemy_structure",
    "nearest_enemy_ground",
    "nearest_enemy_air",
    "nearest_enemy_biological",
    "nearest_enemy_mechanical",
    "nearest_enemy_massive",
    "nearest_enemy_detector",
    "lowest_health_enemy",
    "highest_energy_enemy",
    "nearest_friendly",
    "damaged_friendly",
    "any_friendly",
)


def _runtime_actor_unit_types() -> Mapping[str, NameTuple]:
    """Map canonical planner actors to every standard-melee runtime form.

    Transform and flying forms have different ``UnitTypeId`` values even though
    planners should continue to address one stable actor key. Live ability
    availability remains authoritative for which form can issue an ability.
    """

    actor_types: dict[str, NameTuple] = {
        "worker": ("SCV",),
        **{
            key: (spec.enum_name,)
            for key, spec in CONTROLLABLE_UNIT_SPECS.items()
            if key != "scv"
        },
        **{key: (spec.enum_name,) for key, spec in STRUCTURE_SPECS.items()},
        **{key: (spec.enum_name,) for key, spec in ADDON_SPECS.items()},
        **{key: (spec.enum_name,) for key, spec in MORPH_SPECS.items()},
    }
    actor_types.update(
        {
            "siege_tank": ("SIEGETANK", "SIEGETANKSIEGED"),
            "widow_mine": ("WIDOWMINE", "WIDOWMINEBURROWED"),
            "thor": ("THOR", "THORAP"),
            "viking": ("VIKINGFIGHTER", "VIKINGASSAULT"),
            "liberator": ("LIBERATOR", "LIBERATORAG"),
            "supply_depot": ("SUPPLYDEPOT", "SUPPLYDEPOTLOWERED", "SUPPLYDEPOTDROP"),
            "command_center": ("COMMANDCENTER", "COMMANDCENTERFLYING"),
            "orbital_command": ("ORBITALCOMMAND", "ORBITALCOMMANDFLYING"),
            "barracks": ("BARRACKS", "BARRACKSFLYING"),
            "factory": ("FACTORY", "FACTORYFLYING"),
            "starport": ("STARPORT", "STARPORTFLYING"),
        }
    )
    return MappingProxyType(actor_types)


RUNTIME_ACTOR_UNIT_TYPES: Mapping[str, NameTuple] = _runtime_actor_unit_types()
RUNTIME_UNIT_TYPE_TO_ACTOR: Mapping[str, str] = MappingProxyType(
    {
        normalize_name(enum_name): actor
        for actor, enum_names in RUNTIME_ACTOR_UNIT_TYPES.items()
        for enum_name in enum_names
    }
)


def canonical_runtime_actor_name(enum_name: str) -> str:
    """Return a stable planner actor key for a runtime UnitTypeId name."""

    normalized = normalize_name(enum_name)
    return RUNTIME_UNIT_TYPE_TO_ACTOR.get(normalized, normalized)


# SCVs can repair mechanical Terran units and every Terran structure. Keep the
# LLM-facing worker alias instead of exposing SCV twice in command schemas.
REPAIRABLE_UNIT_KEYS: NameTuple = (
    "mule",
    "hellion",
    "hellbat",
    "widow_mine",
    "cyclone",
    "siege_tank",
    "thor",
    "viking",
    "medivac",
    "liberator",
    "raven",
    "banshee",
    "battlecruiser",
)
REPAIRABLE_TARGET_KEYS: NameTuple = (
    "worker",
    *REPAIRABLE_UNIT_KEYS,
    *STRUCTURE_SPECS,
    *ADDON_SPECS,
    *MORPH_SPECS,
)

BIOLOGICAL_UNIT_KEYS: NameTuple = (
    "worker",
    "marine",
    "marauder",
    "reaper",
    "ghost",
    "hellbat",
)
MECHANICAL_UNIT_KEYS: NameTuple = ("worker", *REPAIRABLE_UNIT_KEYS)
PSIONIC_UNIT_KEYS: NameTuple = ("ghost",)
BUNKER_LOADABLE_UNIT_KEYS: NameTuple = (
    "marine",
    "marauder",
    "reaper",
    "ghost",
)
MEDIVAC_LOADABLE_UNIT_KEYS: NameTuple = (
    "worker",
    "mule",
    "marine",
    "marauder",
    "reaper",
    "ghost",
    "hellion",
    "hellbat",
    "widow_mine",
    "cyclone",
    "siege_tank",
    "thor",
)


_REGISTRIES: Mapping[str, Registry] = MappingProxyType(
    {
        "unit": UNIT_SPECS,
        "special_unit": SPECIAL_UNIT_SPECS,
        "structure": STRUCTURE_SPECS,
        "addon": ADDON_SPECS,
        "morph": MORPH_SPECS,
        "upgrade": UPGRADE_SPECS,
    }
)


def _build_alias_index() -> Mapping[str, AliasResolution]:
    index = {}
    for category, registry in _REGISTRIES.items():
        for key, spec in registry.items():
            names = (key, spec.enum_name) + spec.aliases + spec.runtime_state_keys
            for name in names:
                normalized = normalize_name(name)
                if normalized and normalized not in index:
                    index[normalized] = AliasResolution(
                        key=key, category=category, spec=spec
                    )
    return MappingProxyType(index)


ALIAS_INDEX: Mapping[str, AliasResolution] = _build_alias_index()


def resolve_alias(
    name: str, categories: Optional[Iterable[str]] = None
) -> AliasResolution:
    """Resolve any canonical name, enum name, state key, or alias to a spec."""

    allowed = set(categories) if categories is not None else None
    normalized = normalize_name(name)
    result = ALIAS_INDEX.get(normalized)
    if result is None or (allowed is not None and result.category not in allowed):
        category_text = (
            "any category" if allowed is None else ", ".join(sorted(allowed))
        )
        raise KeyError("unknown command entity %r in %s" % (name, category_text))
    return result


def get_spec(name: str, categories: Optional[Iterable[str]] = None) -> EntitySpec:
    """Return the EntitySpec for a canonical key or alias."""

    return resolve_alias(name, categories=categories).spec


def all_specs() -> Tuple[AliasResolution, ...]:
    """Return all canonical catalog entries as category-tagged resolutions."""

    values: list[AliasResolution] = []
    for category, registry in _REGISTRIES.items():
        values.extend(
            AliasResolution(key=key, category=category, spec=spec)
            for key, spec in registry.items()
        )
    return tuple(values)


def command_entities(
    categories: Optional[Iterable[str]] = None,
) -> Tuple[AliasResolution, ...]:
    """Return LLM-addressable command targets, optionally filtered by category."""

    allowed = set(categories) if categories is not None else None
    return tuple(
        item for item in all_specs() if allowed is None or item.category in allowed
    )


def command_entity_names(categories: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    """Return canonical names for prompt schemas or validation messages."""

    return tuple(item.key for item in command_entities(categories=categories))


def build_command_prompt_section(categories: Optional[Iterable[str]] = None) -> str:
    """Build a compact prompt section that tells an LLM which command targets exist."""

    lines = [
        f"Terran macro command surface for SC2 {BALANCE_VERSION} (use canonical keys exactly):"
    ]
    lines.append("Commands:")
    for command in COMMAND_SURFACE:
        lines.append(
            "- %s %s: %s e.g. %s"
            % (command.key, command.target_field, command.description, command.example)
        )
    lines.append("Targets:")
    current_category = None
    for item in command_entities(categories=categories):
        if item.category != current_category:
            current_category = item.category
            lines.append("- %s:" % current_category)
        spec = item.spec
        requirement_bits = []
        if spec.producer:
            requirement_bits.append("producer=%s" % spec.producer)
        if spec.researcher:
            requirement_bits.append("researcher=%s" % spec.researcher)
        if spec.prerequisites:
            requirement_bits.append("prereq=%s" % "+".join(spec.prerequisites))
        if spec.required_addon:
            requirement_bits.append("addon=%s" % spec.required_addon)
        if spec.previous_upgrade:
            requirement_bits.append("after=%s" % spec.previous_upgrade)
        if spec.supply is not None:
            requirement_bits.append("supply=%s" % ("%g" % spec.supply))
        requirement_bits.append("cost=%d/%d" % (spec.minerals, spec.vespene))
        lines.append(
            "  - %s (%s; enum=%s)"
            % (item.key, ", ".join(requirement_bits), spec.enum_name)
        )
    lines.append("Semantic locations: " + ", ".join(LOCATION_SPECS))
    lines.append("Unit target selectors: " + ", ".join(TARGET_SELECTORS))
    lines.append(
        "Selection modes: "
        + ", ".join(SELECTION_SPECS)
        + "; selection.count is optional and capped by validator/executor."
    )
    lines.append(
        "Allowlisted Terran abilities (canonical key, target, alliance, filter, actors):"
    )
    for ability_spec in ABILITY_SPECS.values():
        lines.append(
            "  - %s; target=%s; alliance=%s; filter=%s; actors=%s"
            % (
                ability_spec.key,
                ability_spec.target_kind,
                ability_spec.target_alliance,
                ability_spec.target_filter,
                "+".join(ability_spec.actors),
            )
        )
    return "\n".join(lines)


def _ability_alias_index() -> Mapping[str, AbilitySpec]:
    index: dict[str, AbilitySpec] = {}
    for spec in ABILITY_SPECS.values():
        # Raw AbilityId enum names are deliberately internal-only. Public LLM
        # output must use a reviewed canonical key or human-friendly alias.
        for name in (spec.key, *spec.aliases):
            normalized = normalize_name(name)
            if normalized and normalized not in index:
                index[normalized] = spec
    return MappingProxyType(index)


ABILITY_ALIAS_INDEX: Mapping[str, AbilitySpec] = _ability_alias_index()


def resolve_ability(name: str) -> AbilitySpec:
    normalized = normalize_name(name)
    spec = ABILITY_ALIAS_INDEX.get(normalized)
    if spec is None:
        raise KeyError("unknown Terran ability %r" % name)
    return spec


def resolve_location(name: str) -> LocationSpec:
    normalized = normalize_name(name)
    if normalized in LOCATION_SPECS:
        return LOCATION_SPECS[normalized]
    for spec in LOCATION_SPECS.values():
        if normalized in {normalize_name(alias) for alias in spec.aliases}:
            return spec
    raise KeyError("unknown semantic location %r" % name)


def resolve_selection_mode(name: str) -> SelectionSpec:
    normalized = normalize_name(name)
    spec = SELECTION_SPECS.get(normalized)
    if spec is None:
        raise KeyError("unknown selection mode %r" % name)
    return spec


__all__ = (
    "ABILITY_ALIAS_INDEX",
    "ABILITY_SPECS",
    "ADDON_SPECS",
    "ALIAS_INDEX",
    "ATTACK_CAPABLE_UNIT_KEYS",
    "AbilitySpec",
    "AliasResolution",
    "BALANCE_SOURCE_URL",
    "BALANCE_VERSION",
    "BIOLOGICAL_UNIT_KEYS",
    "BUNKER_LOADABLE_UNIT_KEYS",
    "COMMAND_SURFACE",
    "CONTROLLABLE_UNIT_SPECS",
    "CommandVerbSpec",
    "EntitySpec",
    "FLYING_STRUCTURE_ACTOR_KEYS",
    "LOCATION_SPECS",
    "LIFTABLE_STRUCTURE_KEYS",
    "LocationSpec",
    "MAX_PLAN_ACTIONS",
    "MAX_REPLAN_CYCLES",
    "MAX_SELECTION_COUNT",
    "MAX_STRUCTURE_ACTION_COUNT",
    "MAX_WORKER_ASSIGNMENT_COUNT",
    "MECHANICAL_UNIT_KEYS",
    "MEDIVAC_LOADABLE_UNIT_KEYS",
    "MORPH_SPECS",
    "PSIONIC_UNIT_KEYS",
    "REPAIRABLE_TARGET_KEYS",
    "REPAIRABLE_UNIT_KEYS",
    "RUNTIME_ACTOR_UNIT_TYPES",
    "RUNTIME_UNIT_TYPE_TO_ACTOR",
    "SALVAGEABLE_STRUCTURE_KEYS",
    "SELECTION_SPECS",
    "SPECIAL_UNIT_SPECS",
    "STRUCTURE_SPECS",
    "SelectionSpec",
    "TARGET_SELECTORS",
    "TRANSFORM_ABILITY_KEYS",
    "TRANSPORT_ACTOR_KEYS",
    "UNIT_SPECS",
    "UPGRADE_SPECS",
    "all_specs",
    "build_command_prompt_section",
    "canonical_runtime_actor_name",
    "command_entities",
    "command_entity_names",
    "get_spec",
    "normalize_name",
    "resolve_ability",
    "resolve_alias",
    "resolve_location",
    "resolve_selection_mode",
)
