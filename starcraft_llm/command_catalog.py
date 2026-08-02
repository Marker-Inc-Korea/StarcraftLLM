from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Sequence, Tuple


NameTuple = Tuple[str, ...]

BALANCE_VERSION = "5.0.15"
BALANCE_SOURCE_URL = "https://news.blizzard.com/en-us/article/24225313/starcraft-ii-5-0-15-patch-notes"


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
        description="Move a Terran unit group to a safe map coordinate.",
        target_registry="UNIT_SPECS",
        target_field="unit",
        example='{"type":"move","unit":"worker","x":35,"y":42}',
    ),
    CommandVerbSpec(
        key="attack_move",
        description="Attack-move a Terran unit group toward a map coordinate.",
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
        description="Order an SCV to construct a Terran structure.",
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
    alias_values = tuple(dict.fromkeys(tuple(aliases) + (key.replace("_", " "), key.replace("_", ""))))
    state_values = tuple(runtime_state_keys) if runtime_state_keys else _state_keys(key, enum_name)
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
        _spec("scv", "SCV", 50, 0, 1, producer="command_center", aliases=("worker", "workers", "일꾼", "건설로봇"), runtime_state_keys=("worker", "workers", "scv")),
        _spec("marine", "MARINE", 50, 0, 1, producer="barracks", prerequisites=("barracks",), aliases=("marines", "마린", "해병")),
        _spec("marauder", "MARAUDER", 100, 25, 2, producer="barracks", prerequisites=("barracks",), required_addon="barracks_tech_lab", aliases=("marauders", "불곰")),
        _spec("reaper", "REAPER", 50, 50, 1, producer="barracks", prerequisites=("barracks",), aliases=("reapers", "사신")),
        _spec("ghost", "GHOST", 150, 125, 2, producer="barracks", prerequisites=("barracks", "ghost_academy"), required_addon="barracks_tech_lab", aliases=("ghosts", "유령")),
        _spec("hellion", "HELLION", 100, 0, 2, producer="factory", prerequisites=("factory",), aliases=("hellions", "화염차")),
        _spec("hellbat", "HELLIONTANK", 100, 0, 2, producer="factory", prerequisites=("factory", "armory"), aliases=("hellion_tank", "helliontank", "hellbats", "화염기갑병", "화염 기갑병")),
        _spec("widow_mine", "WIDOWMINE", 75, 25, 2, producer="factory", prerequisites=("factory",), aliases=("widowmine", "widow mines", "widow_mines", "땅거미지뢰", "땅거미 지뢰")),
        _spec("cyclone", "CYCLONE", 125, 50, 2, producer="factory", prerequisites=("factory",), aliases=("cyclones", "사이클론")),
        _spec("siege_tank", "SIEGETANK", 150, 125, 3, producer="factory", prerequisites=("factory",), required_addon="factory_tech_lab", aliases=("siegetank", "tank", "tanks", "공성전차", "공성 전차")),
        _spec("thor", "THOR", 300, 200, 6, producer="factory", prerequisites=("factory", "armory"), required_addon="factory_tech_lab", aliases=("thors", "토르")),
        _spec("viking", "VIKINGFIGHTER", 125, 75, 2, producer="starport", prerequisites=("starport",), aliases=("viking_fighter", "vikingfighter", "vikings", "바이킹")),
        _spec("medivac", "MEDIVAC", 100, 100, 2, producer="starport", prerequisites=("starport",), aliases=("medivacs", "의료선")),
        _spec("liberator", "LIBERATOR", 150, 150, 3, producer="starport", prerequisites=("starport",), aliases=("liberators", "해방선")),
        _spec("raven", "RAVEN", 100, 150, 2, producer="starport", prerequisites=("starport",), required_addon="starport_tech_lab", aliases=("ravens", "밤까마귀")),
        _spec("banshee", "BANSHEE", 150, 100, 3, producer="starport", prerequisites=("starport",), required_addon="starport_tech_lab", aliases=("banshees", "밴시")),
        _spec("battlecruiser", "BATTLECRUISER", 400, 300, 6, producer="starport", prerequisites=("starport", "fusion_core"), required_addon="starport_tech_lab", aliases=("bc", "battle cruiser", "battlecruisers", "전투순양함", "전투 순양함")),
    )
)


STRUCTURE_SPECS: Registry = _registry(
    (
        _spec("command_center", "COMMANDCENTER", 400, 0, producer="scv", aliases=("cc", "townhall", "town hall", "사령부"), runtime_state_keys=("commandcenter", "command_center", "townhall", "townhalls")),
        _spec("supply_depot", "SUPPLYDEPOT", 100, 0, producer="scv", aliases=("depot", "supply", "supply depot", "서플", "보급고"), runtime_state_keys=("supplydepot", "supply_depot")),
        _spec("refinery", "REFINERY", 75, 0, producer="scv", aliases=("gas", "vespene", "정제소")),
        _spec("barracks", "BARRACKS", 150, 0, producer="scv", prerequisites=("supply_depot",), aliases=("rax", "배럭", "병영")),
        _spec("engineering_bay", "ENGINEERINGBAY", 125, 0, producer="scv", aliases=("ebay", "engineering bay", "공학연구소", "공학 연구소")),
        _spec("bunker", "BUNKER", 100, 0, producer="scv", prerequisites=("barracks",), aliases=("벙커",)),
        _spec("missile_turret", "MISSILETURRET", 100, 0, producer="scv", prerequisites=("engineering_bay",), aliases=("turret", "missile turret", "미사일포탑", "미사일 포탑")),
        _spec("sensor_tower", "SENSORTOWER", 100, 50, producer="scv", prerequisites=("engineering_bay",), aliases=("sensor", "sensor tower", "감지탑", "감지 탑")),
        _spec("factory", "FACTORY", 150, 100, producer="scv", prerequisites=("barracks",), aliases=("fact", "팩토리", "군수공장", "군수 공장")),
        _spec("ghost_academy", "GHOSTACADEMY", 150, 50, producer="scv", prerequisites=("barracks",), aliases=("ghost academy", "academy", "유령사관학교", "유령 사관학교")),
        _spec("starport", "STARPORT", 150, 100, producer="scv", prerequisites=("factory",), aliases=("스타포트", "우주공항", "우주 공항")),
        _spec("armory", "ARMORY", 150, 50, producer="scv", prerequisites=("factory",), aliases=("무기고",)),
        _spec("fusion_core", "FUSIONCORE", 150, 150, producer="scv", prerequisites=("starport",), aliases=("fusion core", "융합로")),
    )
)


ADDON_SPECS: Registry = _registry(
    (
        _spec("barracks_tech_lab", "BARRACKSTECHLAB", 50, 25, producer="barracks", prerequisites=("barracks",), aliases=("barracks tech lab", "rax tech lab", "barracks techlab", "병영 기술실")),
        _spec("barracks_reactor", "BARRACKSREACTOR", 50, 50, producer="barracks", prerequisites=("barracks",), aliases=("barracks reactor", "rax reactor", "병영 반응로")),
        _spec("factory_tech_lab", "FACTORYTECHLAB", 50, 25, producer="factory", prerequisites=("factory",), aliases=("factory tech lab", "factory techlab", "군수공장 기술실")),
        _spec("factory_reactor", "FACTORYREACTOR", 50, 50, producer="factory", prerequisites=("factory",), aliases=("factory reactor", "군수공장 반응로")),
        _spec("starport_tech_lab", "STARPORTTECHLAB", 50, 25, producer="starport", prerequisites=("starport",), aliases=("starport tech lab", "starport techlab", "우주공항 기술실")),
        _spec("starport_reactor", "STARPORTREACTOR", 50, 50, producer="starport", prerequisites=("starport",), aliases=("starport reactor", "우주공항 반응로")),
    )
)


MORPH_SPECS: Registry = _registry(
    (
        _spec("orbital_command", "ORBITALCOMMAND", 150, 0, producer="command_center", prerequisites=("barracks",), aliases=("orbital", "oc", "orbital command", "궤도사령부", "궤도 사령부")),
        _spec("planetary_fortress", "PLANETARYFORTRESS", 150, 150, producer="command_center", prerequisites=("engineering_bay",), aliases=("planetary", "pf", "planetary fortress", "행성요새", "행성 요새")),
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
        _upgrade("terran_infantry_weapons_level_1", "TERRANINFANTRYWEAPONSLEVEL1", 100, 100, "engineering_bay", aliases=("infantry weapons 1", "bio attack 1", "공1업")),
        _upgrade("terran_infantry_weapons_level_2", "TERRANINFANTRYWEAPONSLEVEL2", 150, 150, "engineering_bay", prerequisites=("armory",), previous_upgrade="terran_infantry_weapons_level_1", aliases=("infantry weapons 2", "bio attack 2", "공2업")),
        _upgrade("terran_infantry_weapons_level_3", "TERRANINFANTRYWEAPONSLEVEL3", 200, 200, "engineering_bay", prerequisites=("armory",), previous_upgrade="terran_infantry_weapons_level_2", aliases=("infantry weapons 3", "bio attack 3", "공3업")),
        _upgrade("terran_infantry_armor_level_1", "TERRANINFANTRYARMORSLEVEL1", 100, 100, "engineering_bay", aliases=("infantry armor 1", "bio armor 1", "방1업")),
        _upgrade("terran_infantry_armor_level_2", "TERRANINFANTRYARMORSLEVEL2", 150, 150, "engineering_bay", prerequisites=("armory",), previous_upgrade="terran_infantry_armor_level_1", aliases=("infantry armor 2", "bio armor 2", "방2업")),
        _upgrade("terran_infantry_armor_level_3", "TERRANINFANTRYARMORSLEVEL3", 200, 200, "engineering_bay", prerequisites=("armory",), previous_upgrade="terran_infantry_armor_level_2", aliases=("infantry armor 3", "bio armor 3", "방3업")),
        _upgrade("stimpack", "STIMPACK", 100, 100, "barracks_tech_lab", required_addon="barracks_tech_lab", aliases=("stim", "스팀팩")),
        _upgrade("combat_shield", "SHIELDWALL", 100, 100, "barracks_tech_lab", required_addon="barracks_tech_lab", aliases=("combat shields", "shieldwall", "전투방패")),
        _upgrade("concussive_shells", "PUNISHERGRENADES", 50, 50, "barracks_tech_lab", required_addon="barracks_tech_lab", aliases=("concussive shell", "punisher grenades", "충격탄")),
        _upgrade("personal_cloaking", "PERSONALCLOAKING", 150, 150, "ghost_academy", prerequisites=("ghost_academy",), aliases=("ghost cloak", "cloaking field", "개인 은폐")),
        _upgrade("infernal_pre_igniter", "HIGHCAPACITYBARRELS", 100, 100, "factory_tech_lab", required_addon="factory_tech_lab", aliases=("blue flame", "high capacity barrels", "지옥불 조기점화기")),
        _upgrade("drilling_claws", "DRILLCLAWS", 75, 75, "factory_tech_lab", required_addon="factory_tech_lab", aliases=("drill claws", "drilling claws", "천공 발톱")),
        _upgrade("smart_servos", "SMARTSERVOS", 100, 100, "factory_tech_lab", required_addon="factory_tech_lab", aliases=("smart servo", "smart servos", "스마트 서보")),
        _upgrade("terran_vehicle_weapons_level_1", "TERRANVEHICLEWEAPONSLEVEL1", 100, 100, "armory", prerequisites=("armory",), aliases=("vehicle weapons 1", "mech attack 1")),
        _upgrade("terran_vehicle_weapons_level_2", "TERRANVEHICLEWEAPONSLEVEL2", 175, 175, "armory", prerequisites=("armory",), previous_upgrade="terran_vehicle_weapons_level_1", aliases=("vehicle weapons 2", "mech attack 2")),
        _upgrade("terran_vehicle_weapons_level_3", "TERRANVEHICLEWEAPONSLEVEL3", 250, 250, "armory", prerequisites=("armory",), previous_upgrade="terran_vehicle_weapons_level_2", aliases=("vehicle weapons 3", "mech attack 3")),
        _upgrade("terran_ship_weapons_level_1", "TERRANSHIPWEAPONSLEVEL1", 100, 100, "armory", prerequisites=("armory",), aliases=("ship weapons 1", "air attack 1")),
        _upgrade("terran_ship_weapons_level_2", "TERRANSHIPWEAPONSLEVEL2", 175, 175, "armory", prerequisites=("armory",), previous_upgrade="terran_ship_weapons_level_1", aliases=("ship weapons 2", "air attack 2")),
        _upgrade("terran_ship_weapons_level_3", "TERRANSHIPWEAPONSLEVEL3", 250, 250, "armory", prerequisites=("armory",), previous_upgrade="terran_ship_weapons_level_2", aliases=("ship weapons 3", "air attack 3")),
        _upgrade("terran_vehicle_and_ship_armor_level_1", "TERRANVEHICLEANDSHIPARMORSLEVEL1", 100, 100, "armory", prerequisites=("armory",), aliases=("vehicle armor 1", "ship armor 1", "mech armor 1", "air armor 1")),
        _upgrade("terran_vehicle_and_ship_armor_level_2", "TERRANVEHICLEANDSHIPARMORSLEVEL2", 175, 175, "armory", prerequisites=("armory",), previous_upgrade="terran_vehicle_and_ship_armor_level_1", aliases=("vehicle armor 2", "ship armor 2", "mech armor 2", "air armor 2")),
        _upgrade("terran_vehicle_and_ship_armor_level_3", "TERRANVEHICLEANDSHIPARMORSLEVEL3", 250, 250, "armory", prerequisites=("armory",), previous_upgrade="terran_vehicle_and_ship_armor_level_2", aliases=("vehicle armor 3", "ship armor 3", "mech armor 3", "air armor 3")),
        _upgrade("banshee_cloaking_field", "BANSHEECLOAK", 100, 100, "starport_tech_lab", required_addon="starport_tech_lab", aliases=("banshee cloak", "cloak banshee", "밴시 은폐")),
        _upgrade("hyperflight_rotors", "BANSHEESPEED", 125, 125, "starport_tech_lab", required_addon="starport_tech_lab", aliases=("banshee speed", "hyperflight rotors", "밴시 속업")),
        _upgrade("hi_sec_auto_tracking", "HISECAUTOTRACKING", 100, 100, "engineering_bay", aliases=("hi sec auto tracking", "turret range", "고급 탄도 추적")),
        _upgrade("terran_building_armor", "TERRANBUILDINGARMOR", 150, 150, "engineering_bay", aliases=("building armor", "structure armor", "건물 장갑")),
        _upgrade("cyclone_lock_on_damage", "CYCLONELOCKONDAMAGEUPGRADE", 100, 100, "factory_tech_lab", required_addon="factory_tech_lab", aliases=("cyclone damage", "mag field accelerators", "사이클론 피해")),
        _upgrade("interference_matrix", "INTERFERENCEMATRIX", 50, 50, "starport_tech_lab", required_addon="starport_tech_lab", aliases=("raven interference matrix", "방해 매트릭스")),
        _upgrade("advanced_ballistics", "LIBERATORAGRANGEUPGRADE", 150, 150, "fusion_core", prerequisites=("fusion_core",), aliases=("liberator range", "liberator ag range", "해방선 사업")),
        _upgrade("medivac_caduceus_reactor", "MEDIVACCADUCEUSREACTOR", 100, 100, "fusion_core", prerequisites=("fusion_core",), aliases=("medivac energy", "caduceus reactor", "의료선 에너지")),
        _upgrade("battlecruiser_weapon_refit", "BATTLECRUISERENABLESPECIALIZATIONS", 150, 150, "fusion_core", prerequisites=("fusion_core",), aliases=("yamato", "yamato cannon", "야마토포")),
    )
)


# SCVs can repair mechanical Terran units and every Terran structure. Keep the
# LLM-facing worker alias instead of exposing SCV twice in command schemas.
REPAIRABLE_UNIT_KEYS: NameTuple = (
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


_REGISTRIES: Mapping[str, Registry] = MappingProxyType(
    {
        "unit": UNIT_SPECS,
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
                    index[normalized] = AliasResolution(key=key, category=category, spec=spec)
    return MappingProxyType(index)


ALIAS_INDEX: Mapping[str, AliasResolution] = _build_alias_index()


def resolve_alias(name: str, categories: Optional[Iterable[str]] = None) -> AliasResolution:
    """Resolve any canonical name, enum name, state key, or alias to a spec."""

    allowed = set(categories) if categories is not None else None
    normalized = normalize_name(name)
    result = ALIAS_INDEX.get(normalized)
    if result is None or (allowed is not None and result.category not in allowed):
        category_text = "any category" if allowed is None else ", ".join(sorted(allowed))
        raise KeyError("unknown command entity %r in %s" % (name, category_text))
    return result


def get_spec(name: str, categories: Optional[Iterable[str]] = None) -> EntitySpec:
    """Return the EntitySpec for a canonical key or alias."""

    return resolve_alias(name, categories=categories).spec


def all_specs() -> Tuple[AliasResolution, ...]:
    """Return all canonical catalog entries as category-tagged resolutions."""

    values: list[AliasResolution] = []
    for category, registry in _REGISTRIES.items():
        values.extend(AliasResolution(key=key, category=category, spec=spec) for key, spec in registry.items())
    return tuple(values)


def command_entities(categories: Optional[Iterable[str]] = None) -> Tuple[AliasResolution, ...]:
    """Return LLM-addressable command targets, optionally filtered by category."""

    allowed = set(categories) if categories is not None else None
    return tuple(item for item in all_specs() if allowed is None or item.category in allowed)


def command_entity_names(categories: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    """Return canonical names for prompt schemas or validation messages."""

    return tuple(item.key for item in command_entities(categories=categories))


def build_command_prompt_section(categories: Optional[Iterable[str]] = None) -> str:
    """Build a compact prompt section that tells an LLM which command targets exist."""

    lines = [f"Terran macro command surface for SC2 {BALANCE_VERSION} (use canonical keys exactly):"]
    lines.append("Commands:")
    for command in COMMAND_SURFACE:
        lines.append("- %s %s: %s e.g. %s" % (command.key, command.target_field, command.description, command.example))
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
        lines.append("  - %s (%s; enum=%s)" % (item.key, ", ".join(requirement_bits), spec.enum_name))
    return "\n".join(lines)


__all__ = (
    "ADDON_SPECS",
    "ALIAS_INDEX",
    "AliasResolution",
    "BALANCE_SOURCE_URL",
    "BALANCE_VERSION",
    "COMMAND_SURFACE",
    "CommandVerbSpec",
    "EntitySpec",
    "MORPH_SPECS",
    "REPAIRABLE_TARGET_KEYS",
    "REPAIRABLE_UNIT_KEYS",
    "STRUCTURE_SPECS",
    "UNIT_SPECS",
    "UPGRADE_SPECS",
    "all_specs",
    "build_command_prompt_section",
    "command_entities",
    "command_entity_names",
    "get_spec",
    "normalize_name",
    "resolve_alias",
)
