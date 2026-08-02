import asyncio
import contextlib
import io
import inspect
import unittest

from starcraft_llm import command_catalog, commands, game_state, strategy, validator
from starcraft_llm.game_state import (
    GameStateSummary,
    SupplySummary,
    game_state_summary_to_dict,
)
from starcraft_llm.sc2_bot import create_move_unit_bot_class


LEGACY_FUNCTIONS = {
    "move",
    "attack_move",
    "attack_enemy",
    "patrol",
    "hold_position",
    "stop",
    "rally",
    "wait",
    "wait_until",
    "gather",
    "distribute_workers",
    "train",
    "build",
    "expand",
    "build_addon",
    "morph",
    "research",
    "repair",
}

NEW_FUNCTIONS = {
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
    "replan",
}

EXPECTED_PUBLIC_FUNCTIONS = LEGACY_FUNCTIONS | NEW_FUNCTIONS

EXPECTED_LOCATIONS = {
    "own_main",
    "own_natural",
    "own_third",
    "own_ramp",
    "enemy_main",
    "enemy_natural",
    "enemy_third",
    "map_center",
    "frontline",
    "retreat",
    "proxy",
    "next_expansion",
    "nearest_enemy",
    "nearest_enemy_structure",
    "nearest_mineral",
}

EXPECTED_ABILITIES = {
    "stim_marine": ("EFFECT_STIM_MARINE", "none"),
    "stim_marauder": ("EFFECT_STIM_MARAUDER", "none"),
    "kd8_charge": ("KD8CHARGE_KD8CHARGE", "point"),
    "ghost_cloak_on": ("BEHAVIOR_CLOAKON_GHOST", "none"),
    "ghost_cloak_off": ("BEHAVIOR_CLOAKOFF_GHOST", "none"),
    "ghost_hold_fire_on": ("BEHAVIOR_HOLDFIREON_GHOST", "none"),
    "ghost_hold_fire_off": ("BEHAVIOR_HOLDFIREOFF_GHOST", "none"),
    "ghost_snipe": ("EFFECT_GHOSTSNIPE", "unit"),
    "ghost_emp": ("EMP_EMP", "point"),
    "ghost_nuke_call_down": ("TACNUKESTRIKE_NUKECALLDOWN", "point"),
    "banshee_cloak_on": ("BEHAVIOR_CLOAKON_BANSHEE", "none"),
    "banshee_cloak_off": ("BEHAVIOR_CLOAKOFF_BANSHEE", "none"),
    "morph_hellbat": ("MORPH_HELLBAT", "none"),
    "morph_hellion": ("MORPH_HELLION", "none"),
    "widow_mine_burrow_down": ("BURROWDOWN_WIDOWMINE", "none"),
    "widow_mine_burrow_up": ("BURROWUP_WIDOWMINE", "none"),
    "cyclone_lock_on": ("LOCKON_LOCKON", "unit"),
    "cyclone_cancel_lock_on": ("CANCEL_LOCKON", "none"),
    "siege_mode": ("SIEGEMODE_SIEGEMODE", "none"),
    "unsiege_mode": ("UNSIEGE_UNSIEGE", "none"),
    "thor_high_impact_mode": ("MORPH_THORHIGHIMPACTMODE", "none"),
    "thor_explosive_mode": ("MORPH_THOREXPLOSIVEMODE", "none"),
    "viking_assault_mode": ("MORPH_VIKINGASSAULTMODE", "none"),
    "viking_fighter_mode": ("MORPH_VIKINGFIGHTERMODE", "none"),
    "medivac_afterburners": ("EFFECT_MEDIVACIGNITEAFTERBURNERS", "none"),
    "medivac_heal": ("MEDIVACHEAL_HEAL", "unit"),
    "liberator_ag_mode": ("MORPH_LIBERATORAGMODE", "point"),
    "liberator_aa_mode": ("MORPH_LIBERATORAAMODE", "none"),
    "raven_auto_turret": ("BUILDAUTOTURRET_AUTOTURRET", "point"),
    "raven_interference_matrix": ("EFFECT_INTERFERENCEMATRIX", "unit"),
    "raven_anti_armor_missile": ("EFFECT_ANTIARMORMISSILE", "unit"),
    "battlecruiser_tactical_jump": ("EFFECT_TACTICALJUMP", "point"),
    "battlecruiser_yamato": ("YAMATO_YAMATOGUN", "unit"),
    "scan": ("SCANNERSWEEP_SCAN", "point"),
    "call_down_mule": ("CALLDOWNMULE_CALLDOWNMULE", "mineral"),
    "mule_gather": ("HARVEST_GATHER_MULE", "mineral"),
    "mule_repair": ("EFFECT_REPAIR_MULE", "unit"),
    "supply_drop": ("SUPPLYDROP_SUPPLYDROP", "unit"),
    "lower_supply_depot": ("MORPH_SUPPLYDEPOT_LOWER", "none"),
    "raise_supply_depot": ("MORPH_SUPPLYDEPOT_RAISE", "none"),
    "lift_command_center": ("LIFT_COMMANDCENTER", "none"),
    "land_command_center": ("LAND_COMMANDCENTER", "point"),
    "lift_orbital_command": ("LIFT_ORBITALCOMMAND", "none"),
    "land_orbital_command": ("LAND_ORBITALCOMMAND", "point"),
    "lift_barracks": ("LIFT_BARRACKS", "none"),
    "land_barracks": ("LAND_BARRACKS", "point"),
    "lift_factory": ("LIFT_FACTORY", "none"),
    "land_factory": ("LAND_FACTORY", "point"),
    "lift_starport": ("LIFT_STARPORT", "none"),
    "land_starport": ("LAND_STARPORT", "point"),
    "load_all_command_center": ("LOADALL_COMMANDCENTER", "none"),
    "unload_all_command_center": ("UNLOADALL_COMMANDCENTER", "none"),
    "unload_unit_command_center": ("UNLOADUNIT_COMMANDCENTER", "unit"),
    "load_bunker": ("LOAD_BUNKER", "unit"),
    "unload_all_bunker": ("UNLOADALL_BUNKER", "none"),
    "unload_unit_bunker": ("UNLOADUNIT_BUNKER", "unit"),
    "load_medivac": ("LOAD_MEDIVAC", "unit"),
    "unload_all_medivac": ("UNLOADALLAT_MEDIVAC", "point"),
    "unload_unit_medivac": ("UNLOADUNIT_MEDIVAC", "unit"),
    "build_nuke": ("BUILD_NUKE", "none"),
    "launch_nuke": ("TACNUKESTRIKE_NUKECALLDOWN", "point"),
    "cancel_any": ("CANCEL", "none"),
    "cancel_build_in_progress": ("CANCEL_BUILDINPROGRESS", "none"),
    "cancel_queue_1": ("CANCEL_QUEUE1", "none"),
    "cancel_queue_5": ("CANCEL_QUEUE5", "none"),
    "cancel_queue_addon": ("CANCEL_QUEUEADDON", "none"),
    "cancel_slot": ("CANCEL_SLOT", "none"),
    "cancel_slot_queue_cancel_to_selection": (
        "CANCELSLOT_QUEUECANCELTOSELECTION",
        "none",
    ),
    "cancel_slot_queue_passive": ("CANCELSLOT_QUEUEPASSIVE", "none"),
    "cancel_slot_queue_passive_cancel_to_selection": (
        "CANCELSLOT_QUEUEPASSIVECANCELTOSELECTION",
        "none",
    ),
    "cancel_addon_barracks": ("CANCEL_BARRACKSADDON", "none"),
    "cancel_addon_factory": ("CANCEL_FACTORYADDON", "none"),
    "cancel_addon_starport": ("CANCEL_STARPORTADDON", "none"),
    "cancel_morph_orbital": ("CANCEL_MORPHORBITAL", "none"),
    "cancel_morph_planetary_fortress": ("CANCEL_MORPHPLANETARYFORTRESS", "none"),
    "cancel_morph_thor_explosive_mode": ("CANCEL_MORPHTHOREXPLOSIVEMODE", "none"),
    "cancel_lock_on": ("CANCEL_LOCKON", "none"),
    "cancel_nuke": ("CANCEL_NUKE", "none"),
    "cancel_last": ("CANCEL_LAST", "none"),
    "salvage_bunker": ("SALVAGEEFFECT_SALVAGE", "none"),
    "salvage_sensor_tower": ("SALVAGEEFFECT_SALVAGE", "none"),
}


def _require_attr(testcase, module, name):
    testcase.assertTrue(hasattr(module, name), f"missing {module.__name__}.{name}")
    return getattr(module, name)


def _field(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _enum_name(value):
    return getattr(value, "name", value)


class CompleteTerranCatalogAndSchemaTest(unittest.TestCase):
    def test_schema_function_and_catalog_surface_have_exact_public_parity(self):
        schemas = commands.llm_command_function_schemas()
        schema_names = {schema["name"] for schema in schemas}
        command_surface_names = {entry.key for entry in command_catalog.COMMAND_SURFACE}

        self.assertEqual(schema_names, EXPECTED_PUBLIC_FUNCTIONS)
        self.assertEqual(set(commands.LLM_COMMAND_FUNCTIONS), EXPECTED_PUBLIC_FUNCTIONS)
        self.assertEqual(command_surface_names, EXPECTED_PUBLIC_FUNCTIONS)
        self.assertEqual(len(schemas), len(EXPECTED_PUBLIC_FUNCTIONS))
        for schema in schemas:
            with self.subTest(schema=schema["name"]):
                self.assertEqual(schema["parameters"]["type"], "object")
                self.assertFalse(schema["parameters"].get("additionalProperties", True))
                self.assertIn("description", schema)

    def test_legacy_function_schemas_remain_call_compatible_and_gain_safe_targets(self):
        schemas = {
            schema["name"]: schema for schema in commands.llm_command_function_schemas()
        }

        self.assertEqual(schemas["move"]["parameters"]["required"], ["unit"])
        self.assertIn("location", schemas["move"]["parameters"]["properties"])
        self.assertIn("selection", schemas["move"]["parameters"]["properties"])
        self.assertEqual(
            schemas["move"]["parameters"]["anyOf"],
            [{"required": ["location"]}, {"required": ["x", "y"]}],
        )
        self.assertEqual(
            schemas["move"]["parameters"]["properties"]["x"],
            {"type": "number", "minimum": 0, "maximum": 256},
        )
        self.assertEqual(
            schemas["train"]["parameters"]["properties"]["count"],
            {"type": "integer", "minimum": 1, "maximum": 200},
        )
        self.assertEqual(schemas["build_addon"]["parameters"]["required"], ["addon"])
        self.assertNotIn("producer", schemas["build_addon"]["parameters"]["properties"])
        self.assertEqual(schemas["repair"]["parameters"]["required"], ["target"])
        self.assertEqual(
            schemas["repair"]["parameters"]["properties"]["workers"]["maximum"], 100
        )
        self.assertIn(
            "nearest_enemy_structure",
            schemas["use_ability"]["parameters"]["properties"]["target_unit"]["enum"],
        )
        self.assertIn(
            "mule", schemas["move"]["parameters"]["properties"]["unit"]["enum"]
        )
        self.assertIn(
            "barracks",
            schemas["move"]["parameters"]["properties"]["unit"]["enum"],
        )
        self.assertNotIn(
            "mule",
            schemas["attack_move"]["parameters"]["properties"]["unit"]["enum"],
        )
        self.assertNotIn(
            "barracks",
            schemas["attack_move"]["parameters"]["properties"]["unit"]["enum"],
        )
        for support_actor in ("medivac", "raven", "widow_mine"):
            self.assertNotIn(
                support_actor,
                schemas["attack_move"]["parameters"]["properties"]["unit"]["enum"],
            )
        self.assertIn(
            "mule", schemas["use_ability"]["parameters"]["properties"]["actor"]["enum"]
        )
        self.assertEqual(
            schemas["supply_drop"]["parameters"]["properties"]["target_unit"]["enum"],
            ["supply_depot"],
        )
        self.assertNotIn(
            "medivac",
            schemas["load"]["parameters"]["properties"]["target_unit"]["enum"],
        )
        self.assertEqual(
            set(schemas["lift"]["parameters"]["properties"]["actor"]["enum"]),
            set(command_catalog.LIFTABLE_STRUCTURE_KEYS),
        )
        self.assertEqual(
            set(schemas["load"]["parameters"]["properties"]["actor"]["enum"]),
            set(command_catalog.TRANSPORT_ACTOR_KEYS),
        )
        self.assertEqual(
            set(schemas["salvage"]["parameters"]["properties"]["actor"]["enum"]),
            set(command_catalog.SALVAGEABLE_STRUCTURE_KEYS),
        )
        self.assertEqual(
            set(schemas["transform"]["parameters"]["properties"]["ability"]["enum"]),
            set(command_catalog.TRANSFORM_ABILITY_KEYS),
        )
        self.assertIn("count", schemas["load"]["parameters"]["properties"])

    def test_every_new_llm_command_function_constructs_a_strategy_action(self):
        arguments = {
            "use_ability": {
                "ability": "stim_marine",
                "actor": "marine",
                "selection": {"mode": "ready", "count": 8},
            },
            "scan": {"location": "enemy_main"},
            "call_down_mule": {"location": "nearest_mineral"},
            "supply_drop": {"target_unit": "supply_depot"},
            "transform": {"ability": "siege_mode", "actor": "siege_tank"},
            "lift": {"actor": "barracks"},
            "land": {"actor": "barracks", "location": "proxy"},
            "load": {"actor": "medivac", "target_unit": "marine"},
            "unload": {"actor": "medivac", "location": "enemy_main"},
            "cancel": {"target": "build_in_progress"},
            "salvage": {"actor": "bunker"},
            "build_nuke": {},
            "launch_nuke": {"location": "enemy_main"},
            "replan": {"reason": "ability_unavailable"},
        }

        for name in NEW_FUNCTIONS:
            with self.subTest(function=name):
                self.assertIn(name, commands.LLM_COMMAND_FUNCTIONS)
                function = commands.LLM_COMMAND_FUNCTIONS[name]
                required_parameters = {
                    parameter_name
                    for parameter_name, parameter in inspect.signature(
                        function
                    ).parameters.items()
                    if parameter.default is inspect.Parameter.empty
                }
                self.assertTrue(
                    required_parameters.issubset(arguments[name]),
                    f"test payload missing required args for {name}",
                )
                self.assertIsNotNone(function(**arguments[name]))

    def test_ability_catalog_has_exact_approved_ability_ids_and_target_kinds(self):
        ability_specs = _require_attr(self, command_catalog, "ABILITY_SPECS")

        self.assertEqual(set(ability_specs), set(EXPECTED_ABILITIES))
        raw_id_alias_leaks = []
        for key, (enum_name, target_kind) in EXPECTED_ABILITIES.items():
            with self.subTest(ability=key):
                spec = ability_specs[key]
                self.assertEqual(_field(spec, "enum_name"), enum_name)
                self.assertEqual(_field(spec, "target_kind"), target_kind)
                self.assertEqual(_field(spec, "key"), key)
                if enum_name in _field(spec, "aliases"):
                    raw_id_alias_leaks.append(key)
        self.assertEqual(
            raw_id_alias_leaks, [], "raw AbilityId names must not be public aliases"
        )
        with self.assertRaises(strategy.StrategyParseError):
            strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "use_ability",
                            "ability": "EFFECT_STIM_MARINE",
                            "actor": "marine",
                        }
                    ]
                }
            )

    def test_ability_ids_resolve_against_installed_burnysc2(self):
        try:
            from sc2.ids.ability_id import AbilityId
        except ImportError:
            self.skipTest("BurnySC2 is not installed in this Python environment")
        ability_specs = _require_attr(self, command_catalog, "ABILITY_SPECS")

        for key in sorted(EXPECTED_ABILITIES):
            with self.subTest(ability=key):
                self.assertTrue(
                    hasattr(AbilityId, _field(ability_specs[key], "enum_name"))
                )

    def test_unit_structure_and_upgrade_catalogs_match_burnysc2_melee_data(self):
        try:
            from sc2.dicts.unit_research_abilities import RESEARCH_INFO
            from sc2.dicts.unit_train_build_abilities import TRAIN_INFO
            from sc2.ids.unit_typeid import UnitTypeId
        except ImportError:
            self.skipTest("BurnySC2 generated melee data is unavailable")

        production_structures = (
            UnitTypeId.BARRACKS,
            UnitTypeId.FACTORY,
            UnitTypeId.STARPORT,
            UnitTypeId.COMMANDCENTER,
            UnitTypeId.ORBITALCOMMAND,
            UnitTypeId.PLANETARYFORTRESS,
        )
        runtime_units = {
            unit.name
            for producer in production_structures
            for unit in TRAIN_INFO.get(producer, {})
        } - {"ORBITALCOMMAND", "PLANETARYFORTRESS"}
        runtime_structures = {
            structure.name for structure in TRAIN_INFO[UnitTypeId.SCV]
        }
        research_structures = (
            UnitTypeId.ARMORY,
            UnitTypeId.BARRACKSTECHLAB,
            UnitTypeId.ENGINEERINGBAY,
            UnitTypeId.FACTORYTECHLAB,
            UnitTypeId.FUSIONCORE,
            UnitTypeId.GHOSTACADEMY,
            UnitTypeId.STARPORTTECHLAB,
        )
        runtime_upgrades = {
            upgrade.name
            for structure in research_structures
            for upgrade in RESEARCH_INFO.get(structure, {})
        }

        self.assertEqual(
            {spec.enum_name for spec in command_catalog.UNIT_SPECS.values()},
            runtime_units,
        )
        self.assertEqual(
            {spec.enum_name for spec in command_catalog.STRUCTURE_SPECS.values()},
            runtime_structures,
        )
        self.assertEqual(
            {spec.enum_name for spec in command_catalog.UPGRADE_SPECS.values()},
            runtime_upgrades,
        )

    def test_locations_and_selection_modes_are_allowlisted(self):
        location_specs = _require_attr(self, command_catalog, "LOCATION_SPECS")
        selection_modes = _require_attr(self, command_catalog, "SELECTION_SPECS")

        self.assertEqual(set(location_specs), EXPECTED_LOCATIONS)
        self.assertEqual(
            set(selection_modes), {"all", "ready", "idle", "closest", "lowest_health"}
        )


class CompleteTerranRoundTripAndValidationTest(unittest.TestCase):
    def test_semantic_legacy_actions_round_trip_without_coordinate_substitution(self):
        payload = {
            "actions": [
                {
                    "type": "move",
                    "unit": "medivac",
                    "location": "enemy_main",
                    "selection": {"mode": "closest", "count": 1},
                    "queued": True,
                },
                {
                    "type": "build",
                    "building": "barracks",
                    "location": "proxy",
                    "selection": {"mode": "closest", "count": 1},
                },
                {
                    "type": "rally",
                    "building": "barracks",
                    "location": "enemy_natural",
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        serialized = strategy.strategy_plan_to_dict(plan)
        self.assertEqual(serialized["actions"][0], payload["actions"][0])
        self.assertEqual(
            serialized["actions"][1],
            {**payload["actions"][1], "worker": "worker"},
        )
        self.assertEqual(serialized["actions"][2], payload["actions"][2])
        self.assertIsNone(plan.actions[0].x)
        self.assertEqual(plan.actions[0].location.semantic, "enemy_main")

    def test_json_round_trip_preserves_generic_ability_location_selector_and_queue(
        self,
    ):
        payload = {
            "actions": [
                {
                    "type": "use_ability",
                    "ability": "raven_auto_turret",
                    "actor": "raven",
                    "location": "frontline",
                    "selection": {"mode": "closest", "count": 1},
                    "queued": True,
                },
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_tactical_jump",
                    "actor": "battlecruiser",
                    "x": 120,
                    "y": 130,
                    "selection": {"mode": "ready", "count": 2},
                    "queued": False,
                },
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_yamato",
                    "actor": "battlecruiser",
                    "target": "nearest_enemy_structure",
                    "selection": {"mode": "closest", "count": 1},
                },
                {
                    "type": "use_ability",
                    "ability": "call_down_mule",
                    "location": "nearest_mineral",
                },
                {
                    "type": "use_ability",
                    "ability": "mule_gather",
                    "actor": "mule",
                    "location": "nearest_mineral",
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(strategy.strategy_plan_to_dict(plan), payload)
        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_json_round_trip_preserves_all_typed_wrappers(self):
        payload = {
            "actions": [
                {"type": "scan", "location": "enemy_main"},
                {"type": "call_down_mule", "location": "nearest_mineral"},
                {"type": "supply_drop", "target": "supply_depot"},
                {"type": "transform", "actor": "siege_tank", "mode": "siege"},
                {"type": "lift", "building": "barracks"},
                {"type": "land", "building": "barracks", "location": "proxy"},
                {"type": "load", "transport": "medivac", "unit": "marine", "count": 8},
                {"type": "unload", "transport": "medivac", "location": "enemy_main"},
                {"type": "cancel", "target": "queue_1"},
                {"type": "salvage", "target": "bunker"},
                {"type": "build_nuke"},
                {"type": "launch_nuke", "location": "enemy_main"},
                {"type": "replan", "reason": "ability_unavailable"},
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(strategy.strategy_plan_to_dict(plan), payload)
        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_cancel_shorthand_and_canonical_ability_round_trip(self):
        for payload in (
            {"type": "cancel", "target": "nuke"},
            {"type": "cancel", "ability": "cancel_nuke"},
        ):
            with self.subTest(payload=payload):
                plan = strategy.strategy_plan_from_dict([payload])
                self.assertEqual(plan.actions[0].ability, "cancel_nuke")
                serialized = strategy.strategy_plan_to_dict(plan)
                self.assertEqual(serialized["actions"][0]["target"], "nuke")
                self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)

    def test_dsl_round_trip_accepts_new_ability_surface(self):
        plan = strategy.parse_strategy_plan(
            "use ability stim marine with marine count 8; "
            "scan enemy main; mule nearest mineral; supply drop supply depot; "
            "transform siege tank siege; lift barracks; land barracks proxy; "
            "load medivac marine 8; unload medivac enemy main; "
            "cancel queue 1; salvage bunker; build nuke; launch nuke enemy main; replan ability unavailable"
        )

        self.assertEqual(
            [
                action["type"]
                for action in strategy.strategy_plan_to_dict(plan)["actions"]
            ],
            [
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
                "replan",
            ],
        )

    def test_invalid_ability_actor_target_selector_location_count_queue_and_length_fail_locally(
        self,
    ):
        invalid_actions = [
            (
                {"type": "use_ability", "ability": "psi_storm", "actor": "marine"},
                "ability",
            ),
            (
                {"type": "use_ability", "ability": "stim_marine", "actor": "zealot"},
                "actor",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "location": "enemy_main",
                },
                "target",
            ),
            (
                {"type": "use_ability", "ability": "scan", "actor": "orbital_command"},
                "location",
            ),
            ({"type": "scan", "location": "gold_base"}, "location"),
            (
                {"type": "scan", "location": "enemy_main", "x": 50, "y": 50},
                "not both",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "selection": {"mode": "random"},
                },
                "selection",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "selection": {"count": 201},
                },
                "count",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "queued": "yes",
                },
                "queued",
            ),
        ]

        for payload, message in invalid_actions:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(Exception, message):
                    strategy.strategy_plan_from_dict({"actions": [payload]})

        too_long = strategy.StrategyPlan(
            tuple(strategy.WaitCommand(seconds=0) for _ in range(25))
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "24"):
            validator.validate_strategy_plan(too_long)

    def test_validator_accepts_static_positive_ability_plan_but_defers_live_availability(
        self,
    ):
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "use_ability",
                        "ability": "stim_marine",
                        "actor": "marine",
                        "selection": {"count": 8},
                    },
                    {"type": "scan", "location": "enemy_main"},
                    {"type": "launch_nuke", "location": "enemy_main"},
                    {"type": "replan", "reason": "checkpoint"},
                ]
            }
        )

        self.assertIs(
            validator.validate_strategy_plan(
                plan,
                _state(
                    minerals=1000,
                    vespene=1000,
                    structures={
                        "commandcenter": 1,
                        "barracks": 1,
                        "orbitalcommand": 1,
                        "ghostacademy": 1,
                    },
                    structures_ready={
                        "commandcenter": 1,
                        "barracks": 1,
                        "orbitalcommand": 1,
                        "ghostacademy": 1,
                    },
                    army={"marine": 12, "ghost": 1},
                    upgrades=(
                        "stimpack",
                        "ghost_cloak",
                    ),
                ),
            ),
            plan,
        )

    def test_validator_simulates_mule_creation_and_supply_drop(self):
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {"type": "call_down_mule", "location": "nearest_mineral"},
                    {
                        "type": "use_ability",
                        "ability": "mule_gather",
                        "actor": "mule",
                        "location": "nearest_mineral",
                    },
                    {"type": "supply_drop", "target": "supply_depot"},
                    {"type": "train", "unit": "marine"},
                ]
            }
        )
        state = _state(
            minerals=50,
            supply_left=0,
            structures={
                "orbitalcommand": 1,
                "supplydepot": 1,
                "barracks": 1,
            },
        )

        self.assertIs(validator.validate_strategy_plan(plan, state), plan)

    def test_interference_matrix_accepts_psionic_or_mechanical_units(self):
        state = _state(
            structures={"starporttechlab": 1},
            army={"raven": 1, "ghost": 1, "siege_tank": 1},
            upgrades=("interference_matrix",),
        )
        for target in ("ghost", "siege_tank"):
            with self.subTest(target=target):
                plan = strategy.strategy_plan_from_dict(
                    [
                        {
                            "type": "use_ability",
                            "ability": "raven_interference_matrix",
                            "actor": "raven",
                            "target_unit": target,
                        }
                    ]
                )
                self.assertIs(validator.validate_strategy_plan(plan, state), plan)

    def test_validator_rejects_missing_actor_prerequisite_upgrade_resource_and_target_kind(
        self,
    ):
        cases = [
            (
                {"type": "use_ability", "ability": "stim_marine", "actor": "marine"},
                _state(army={}),
                "marine",
            ),
            (
                {"type": "use_ability", "ability": "stim_marine", "actor": "marine"},
                _state(army={"marine": 4}),
                "stimpack",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_yamato",
                    "actor": "battlecruiser",
                    "target": "nearest_enemy",
                },
                _state(army={"battlecruiser": 1}),
                "fusion_core",
            ),
            (
                {"type": "build_nuke"},
                _state(
                    minerals=50,
                    structures={"ghostacademy": 1},
                    structures_ready={"ghostacademy": 1},
                ),
                "minerals",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "raven_auto_turret",
                    "actor": "raven",
                },
                _state(army={"raven": 1}),
                "target",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "target": "nearest_enemy",
                },
                _state(army={"marine": 1}, upgrades=("stimpack",)),
                "target",
            ),
            (
                {"type": "supply_drop", "target_unit": "marine"},
                _state(
                    structures={"orbitalcommand": 1, "supplydepot": 1},
                    army={"marine": 1},
                ),
                "cannot target marine",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "medivac_heal",
                    "actor": "medivac",
                    "target_unit": "siege_tank",
                },
                _state(army={"medivac": 1, "siege_tank": 1}),
                "cannot target siege_tank",
            ),
            (
                {
                    "type": "use_ability",
                    "ability": "mule_repair",
                    "actor": "mule",
                    "target_unit": "marine",
                },
                _state(army={"mule": 1, "marine": 1}),
                "cannot target marine",
            ),
            (
                {
                    "type": "load",
                    "transport": "bunker",
                    "unit": "siege_tank",
                },
                _state(
                    structures={"bunker": 1},
                    army={"siege_tank": 1},
                ),
                "cannot target siege_tank",
            ),
        ]

        for action, state, message in cases:
            with self.subTest(action=action):
                with self.assertRaisesRegex(validator.PlanValidationError, message):
                    validator.validate_strategy_plan(
                        strategy.strategy_plan_from_dict({"actions": [action]}), state
                    )

    def test_game_state_observation_serializes_ability_relevant_fields(self):
        UnitObservation = _require_attr(self, game_state, "UnitObservation")
        summary = GameStateSummary(
            minerals=50,
            vespene=75,
            supply=SupplySummary(used=20, cap=31, left=11),
            workers=16,
            townhalls=2,
            army={"marine": 8},
            known_enemy_units=3,
            game_time_seconds=240.0,
            structures={"orbitalcommand": 1},
            structures_ready={"orbitalcommand": 1},
            unit_observations=(
                UnitObservation(
                    tag=101,
                    unit="orbital_command",
                    energy=63,
                    is_flying=False,
                    is_burrowed=False,
                    cargo_used=0,
                    cargo_max=0,
                    is_ready=True,
                    is_idle=True,
                    orders=("SCANNERSWEEP_SCAN",),
                ),
                UnitObservation(
                    tag=202,
                    unit="medivac",
                    energy=50,
                    cargo_used=6,
                    cargo_max=8,
                    is_ready=True,
                ),
            ),
        )

        self.assertEqual(
            game_state_summary_to_dict(summary)["unit_observations"],
            [
                {
                    "alliance": "self",
                    "tag": 101,
                    "unit": "orbital_command",
                    "energy": 63,
                    "is_flying": False,
                    "is_burrowed": False,
                    "cargo_used": 0,
                    "cargo_max": 0,
                    "is_ready": True,
                    "is_idle": True,
                    "orders": ["SCANNERSWEEP_SCAN"],
                },
                {
                    "alliance": "self",
                    "tag": 202,
                    "unit": "medivac",
                    "energy": 50,
                    "cargo_used": 6,
                    "cargo_max": 8,
                    "is_ready": True,
                },
            ],
        )


class FakeAbilityUnit:
    _next_tag = 1

    def __init__(
        self, type_name, position=(35, 42), is_ready=True, is_idle=True, health=100
    ):
        self.type_id = type("FakeTypeId", (), {"name": type_name})()
        self.tag = FakeAbilityUnit._next_tag
        FakeAbilityUnit._next_tag += 1
        self.position = position
        self.is_ready = is_ready
        self.is_idle = is_idle
        self.health = health
        self.health_max = 100
        biological_types = {
            "SCV",
            "MARINE",
            "MARAUDER",
            "REAPER",
            "GHOST",
            "HELLIONTANK",
            "ZERGLING",
        }
        structure_types = {
            "BUNKER",
            "BARRACKS",
            "BARRACKSFLYING",
            "ORBITALCOMMAND",
            "ORBITALCOMMANDFLYING",
            "GHOSTACADEMY",
            "SUPPLYDEPOT",
            "SUPPLYDEPOTLOWERED",
            "HATCHERY",
        }
        mechanical_types = {
            "SCV",
            "MULE",
            "HELLION",
            "HELLIONTANK",
            "WIDOWMINE",
            "WIDOWMINEBURROWED",
            "CYCLONE",
            "SIEGETANK",
            "SIEGETANKSIEGED",
            "THOR",
            "THORAP",
            "VIKINGFIGHTER",
            "VIKINGASSAULT",
            "MEDIVAC",
            "LIBERATOR",
            "LIBERATORAG",
            "RAVEN",
            "BANSHEE",
            "BATTLECRUISER",
            *structure_types,
        }
        flying_types = {
            "BARRACKSFLYING",
            "ORBITALCOMMANDFLYING",
            "VIKINGFIGHTER",
            "MEDIVAC",
            "LIBERATOR",
            "RAVEN",
            "BANSHEE",
            "BATTLECRUISER",
        }
        self.is_biological = type_name in biological_types
        self.is_mechanical = type_name in mechanical_types
        self.is_psionic = type_name == "GHOST"
        self.is_structure = type_name in structure_types
        self.is_flying = type_name in flying_types
        self.issued = []

    @property
    def health_percentage(self):
        return self.health / self.health_max

    def __call__(self, ability, *args, **kwargs):
        self.issued.append((_enum_name(ability), args, kwargs))
        return True

    def move(self, target, **kwargs):
        self.issued.append(("MOVE", (target,), kwargs))
        return True

    def attack(self, target, **kwargs):
        self.issued.append(("ATTACK", (target,), kwargs))
        return True

    def patrol(self, target, **kwargs):
        self.issued.append(("PATROL", (target,), kwargs))
        return True


class FakeAbilityUnits(list):
    @property
    def ready(self):
        return FakeAbilityUnits(
            [unit for unit in self if getattr(unit, "is_ready", True)]
        )

    @property
    def idle(self):
        return FakeAbilityUnits(
            [unit for unit in self if getattr(unit, "is_idle", True)]
        )

    @property
    def first(self):
        return self[0]

    def of_type(self, unit_types):
        expected = {_enum_name(unit_type) for unit_type in unit_types}
        return FakeAbilityUnits(
            [unit for unit in self if _enum_name(unit.type_id) in expected]
        )

    def closest_to(self, point):
        return self[0]


class FakeAbilityClient:
    def __init__(self):
        self.left = False

    async def leave(self):
        self.left = True


class AbilityFakeBotAI:
    def __init__(self):
        self.client = FakeAbilityClient()
        self.marine_a = FakeAbilityUnit("MARINE", (10, 10))
        self.marine_b = FakeAbilityUnit("MARINE", (11, 11))
        self.medivac = FakeAbilityUnit("MEDIVAC", (12, 12))
        self.bunker = FakeAbilityUnit("BUNKER", (13, 13))
        self.barracks = FakeAbilityUnit("BARRACKS", (14, 14))
        self.orbital = FakeAbilityUnit("ORBITALCOMMAND", (15, 15))
        self.ghost_academy = FakeAbilityUnit("GHOSTACADEMY", (16, 16))
        self.ghost = FakeAbilityUnit("GHOST", (17, 17))
        self.mule = FakeAbilityUnit("MULE", (18, 18))
        self.enemy = FakeAbilityUnit("ZERGLING", (80, 80))
        self.enemy_structure = FakeAbilityUnit("HATCHERY", (90, 90))
        self.mineral = FakeAbilityUnit("MINERALFIELD", (19, 19))
        self.workers = FakeAbilityUnits([FakeAbilityUnit("SCV", (5, 5))])
        self.units = FakeAbilityUnits(
            [self.marine_a, self.marine_b, self.medivac, self.ghost, self.mule]
        )
        self.structures = FakeAbilityUnits(
            [self.bunker, self.barracks, self.orbital, self.ghost_academy]
        )
        self.townhalls = FakeAbilityUnits([self.orbital])
        self.enemy_units = FakeAbilityUnits([self.enemy])
        self.enemy_structures = FakeAbilityUnits([self.enemy_structure])
        self.mineral_field = FakeAbilityUnits([self.mineral])
        self.minerals = 1000
        self.vespene = 1000
        self.supply_used = 40
        self.supply_cap = 80
        self.supply_left = 40
        self.time = 1.0
        self.available_queries = []
        self.available_abilities = set(
            enum for enum, _target in EXPECTED_ABILITIES.values()
        )
        self.start_location = (35, 42)
        self.enemy_start_locations = [(90, 90)]
        self.game_info = type("GameInfo", (), {"map_center": (50, 50)})()
        self.build_requests = []

    async def get_available_abilities(self, unit):
        self.available_queries.append(unit.tag)
        return list(self.available_abilities)

    async def query_available_abilities(self, units):
        selected = list(
            units if isinstance(units, (list, tuple, FakeAbilityUnits)) else [units]
        )
        for unit in selected:
            self.available_queries.append(unit.tag)
        return [list(self.available_abilities) for _ in selected]

    async def get_next_expansion(self):
        return (40, 40)

    def can_afford(self, _item):
        return True

    async def build(self, unit_type, near, **kwargs):
        self.build_requests.append((_enum_name(unit_type), near, kwargs))
        return True


class UnavailableAbilityFakeBotAI(AbilityFakeBotAI):
    def __init__(self):
        super().__init__()
        self.available_abilities = set()
        self.plan_requests = []

    async def create_replan(self, original_goal, game_state):
        self.plan_requests.append((original_goal, game_state))
        return strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "use_ability",
                        "ability": "medivac_afterburners",
                        "actor": "medivac",
                    }
                ]
            }
        )


class CompleteTerranFakeExecutorTest(unittest.TestCase):
    def test_semantic_move_and_build_resolve_at_runtime_without_center_fallback(self):
        move_bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {
                    "type": "move",
                    "unit": "medivac",
                    "location": "enemy_main",
                    "selection": {"count": 1},
                    "queued": True,
                }
            ],
        )
        self.assertEqual(
            move_bot.medivac.issued[0], ("MOVE", ((90.0, 90.0),), {"queue": True})
        )

        build_bot = _run_fake_plan(
            AbilityFakeBotAI,
            [{"type": "build", "building": "barracks", "location": "proxy"}],
            max_steps=1,
        )
        self.assertEqual(build_bot.build_requests[0][0], "BARRACKS")
        self.assertEqual(build_bot.build_requests[0][1], (70.0, 70.0))

    def test_fake_executor_dispatches_no_target_point_unit_and_passenger_abilities(
        self,
    ):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "selection": {"count": 1},
                },
                {"type": "scan", "location": "enemy_main"},
                {
                    "type": "use_ability",
                    "ability": "ghost_snipe",
                    "actor": "ghost",
                    "target": "nearest_enemy",
                },
                {"type": "load", "transport": "medivac", "unit": "marine", "count": 1},
            ],
        )

        self.assertGreaterEqual(len(bot.available_queries), 4)
        self.assertEqual(bot.marine_a.issued[0][0], "EFFECT_STIM_MARINE")
        self.assertEqual(bot.orbital.issued[0][0], "SCANNERSWEEP_SCAN")
        self.assertEqual(bot.orbital.issued[0][1][0], (90, 90))
        self.assertEqual(bot.ghost.issued[0][0], "EFFECT_GHOSTSNIPE")
        self.assertIs(bot.ghost.issued[0][1][0], bot.enemy)
        self.assertEqual(bot.medivac.issued[0][0], "LOAD_MEDIVAC")
        self.assertIs(bot.medivac.issued[0][1][0], bot.marine_b)

    def test_runtime_chooses_an_ability_available_source_before_default_limit(self):
        class PerSourceAvailabilityBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.available_orbital = FakeAbilityUnit("ORBITALCOMMAND", (16, 15))
                self.structures.append(self.available_orbital)

            async def query_available_abilities(self, unit):
                self.available_queries.append(unit.tag)
                if unit is self.available_orbital:
                    return ["SCANNERSWEEP_SCAN"]
                if unit is self.barracks:
                    return ["CANCEL"]
                return []

        bot = _run_fake_plan(
            PerSourceAvailabilityBotAI,
            [
                {"type": "scan", "location": "enemy_main"},
                {"type": "cancel", "target": "any"},
            ],
        )

        self.assertEqual(bot.orbital.issued, [])
        self.assertEqual(bot.available_orbital.issued[0][0], "SCANNERSWEEP_SCAN")
        self.assertEqual(bot.barracks.issued[0][0], "CANCEL")

    def test_attack_enemy_falls_back_to_visible_structure(self):
        class StructureOnlyEnemyBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.enemy_units = FakeAbilityUnits()

        bot = _run_fake_plan(
            StructureOnlyEnemyBotAI,
            [{"type": "attack_enemy", "unit": "marine", "selection": {"count": 1}}],
        )

        self.assertEqual(bot.marine_a.issued[0][0], "ATTACK")
        self.assertIs(bot.marine_a.issued[0][1][0], bot.enemy_structure)

    def test_basic_movement_skips_immobile_runtime_forms(self):
        class MixedFormBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.mobile_tank = FakeAbilityUnit("SIEGETANK")
                self.sieged_tank = FakeAbilityUnit("SIEGETANKSIEGED")
                self.units = FakeAbilityUnits([self.mobile_tank, self.sieged_tank])

        bot = _run_fake_plan(
            MixedFormBotAI,
            [{"type": "move", "unit": "siege_tank", "location": "enemy_main"}],
        )

        self.assertEqual(bot.mobile_tank.issued[0][0], "MOVE")
        self.assertEqual(bot.sieged_tank.issued, [])

    def test_mule_targets_a_mineral_unit_and_load_count_queues_every_passenger(self):
        class LoadCountFakeBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                extra_marines = [
                    FakeAbilityUnit("MARINE", (13 + index, 13 + index))
                    for index in range(6)
                ]
                self.units.extend(extra_marines)

        bot = _run_fake_plan(
            LoadCountFakeBotAI,
            [
                {"type": "call_down_mule", "location": "nearest_mineral"},
                {
                    "type": "load",
                    "transport": "medivac",
                    "unit": "marine",
                    "count": 8,
                },
            ],
        )

        self.assertIs(bot.orbital.issued[0][1][0], bot.mineral)
        load_orders = [
            order for order in bot.medivac.issued if order[0] == "LOAD_MEDIVAC"
        ]
        self.assertEqual(len(load_orders), 8)
        self.assertFalse(load_orders[0][2]["queue"])
        self.assertTrue(all(order[2]["queue"] for order in load_orders[1:]))

    def test_summoned_mule_can_move_gather_and_repair_but_cannot_attack(self):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {"type": "move", "unit": "mule", "location": "own_main"},
                {
                    "type": "use_ability",
                    "ability": "mule_gather",
                    "actor": "mule",
                    "location": "nearest_mineral",
                },
                {
                    "type": "use_ability",
                    "ability": "mule_repair",
                    "actor": "mule",
                    "target": "barracks",
                },
            ],
        )

        self.assertEqual(bot.mule.issued[0][0], "MOVE")
        self.assertEqual(bot.mule.issued[1][0], "HARVEST_GATHER_MULE")
        self.assertIs(bot.mule.issued[1][1][0], bot.mineral)
        self.assertEqual(bot.mule.issued[2][0], "EFFECT_REPAIR_MULE")
        self.assertIs(bot.mule.issued[2][1][0], bot.barracks)

        attacking_plan = strategy.strategy_plan_from_dict(
            [{"type": "attack", "unit": "mule", "location": "enemy_main"}]
        )
        with self.assertRaisesRegex(
            validator.PlanValidationError, "cannot issue an attack"
        ):
            validator.validate_strategy_plan(attacking_plan)
        for support_actor in ("medivac", "raven", "widow_mine"):
            with self.subTest(support_actor=support_actor):
                support_attack = strategy.strategy_plan_from_dict(
                    [
                        {
                            "type": "attack",
                            "unit": support_actor,
                            "location": "enemy_main",
                        }
                    ]
                )
                with self.assertRaisesRegex(
                    validator.PlanValidationError, "cannot issue an attack"
                ):
                    validator.validate_strategy_plan(support_attack)

    def test_runtime_never_issues_ability_to_an_incompatible_explicit_target(self):
        class TargetDomainBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.siege_tank = FakeAbilityUnit("SIEGETANK")
                self.units.append(self.siege_tank)

        supply_bot = _run_fake_plan(
            TargetDomainBotAI,
            [{"type": "supply_drop", "target_unit": "marine"}],
            max_steps=2,
        )
        heal_bot = _run_fake_plan(
            TargetDomainBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "medivac_heal",
                    "actor": "medivac",
                    "target_unit": "siege_tank",
                }
            ],
            max_steps=2,
        )
        repair_bot = _run_fake_plan(
            TargetDomainBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "mule_repair",
                    "actor": "mule",
                    "target_unit": "marine",
                }
            ],
            max_steps=2,
        )
        load_bot = _run_fake_plan(
            TargetDomainBotAI,
            [
                {
                    "type": "load",
                    "transport": "bunker",
                    "unit": "siege_tank",
                }
            ],
            max_steps=2,
        )

        self.assertEqual(supply_bot.orbital.issued, [])
        self.assertEqual(heal_bot.medivac.issued, [])
        self.assertEqual(repair_bot.mule.issued, [])
        self.assertEqual(load_bot.bunker.issued, [])

    def test_runtime_interference_matrix_accepts_psionic_target(self):
        class PsionicTargetBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.raven = FakeAbilityUnit("RAVEN")
                self.units.append(self.raven)
                self.enemy_ghost = FakeAbilityUnit("GHOST", (80, 80))
                self.enemy_units = FakeAbilityUnits([self.enemy_ghost])

        bot = _run_fake_plan(
            PsionicTargetBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "raven_interference_matrix",
                    "actor": "raven",
                    "target_unit": "ghost",
                }
            ],
        )

        self.assertEqual(bot.raven.issued[0][0], "EFFECT_INTERFERENCEMATRIX")
        self.assertIs(bot.raven.issued[0][1][0], bot.enemy_ghost)

    def test_transformed_and_flying_runtime_forms_are_selectable(self):
        class FormFakeBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.hellbat = FakeAbilityUnit("HELLIONTANK")
                self.sieged_tank = FakeAbilityUnit("SIEGETANKSIEGED")
                self.viking_assault = FakeAbilityUnit("VIKINGASSAULT")
                self.liberator_ag = FakeAbilityUnit("LIBERATORAG")
                self.widow_burrowed = FakeAbilityUnit("WIDOWMINEBURROWED")
                self.units = FakeAbilityUnits(
                    [
                        self.hellbat,
                        self.sieged_tank,
                        self.viking_assault,
                        self.liberator_ag,
                        self.widow_burrowed,
                    ]
                )
                self.lowered_depot = FakeAbilityUnit("SUPPLYDEPOTLOWERED")
                self.flying_barracks = FakeAbilityUnit("BARRACKSFLYING")
                self.grounded_barracks = FakeAbilityUnit("BARRACKS")
                self.structures = FakeAbilityUnits(
                    [
                        self.lowered_depot,
                        self.flying_barracks,
                        self.grounded_barracks,
                    ]
                )

        bot = _run_fake_plan(
            FormFakeBotAI,
            [
                {"type": "transform", "actor": "hellbat", "mode": "hellion"},
                {"type": "transform", "actor": "siege_tank", "mode": "unsiege"},
                {"type": "transform", "actor": "viking", "mode": "fighter"},
                {"type": "transform", "actor": "liberator", "mode": "aa"},
                {
                    "type": "transform",
                    "actor": "widow_mine",
                    "ability": "widow_mine_burrow_up",
                },
                {
                    "type": "transform",
                    "actor": "supply_depot",
                    "ability": "raise_supply_depot",
                },
                {"type": "move", "unit": "barracks", "location": "enemy_main"},
                {"type": "land", "building": "barracks", "location": "own_main"},
            ],
        )

        self.assertEqual(bot.hellbat.issued[0][0], "MORPH_HELLION")
        self.assertEqual(bot.sieged_tank.issued[0][0], "UNSIEGE_UNSIEGE")
        self.assertEqual(bot.viking_assault.issued[0][0], "MORPH_VIKINGFIGHTERMODE")
        self.assertEqual(bot.liberator_ag.issued[0][0], "MORPH_LIBERATORAAMODE")
        self.assertEqual(bot.widow_burrowed.issued[0][0], "BURROWUP_WIDOWMINE")
        self.assertEqual(bot.lowered_depot.issued[0][0], "MORPH_SUPPLYDEPOT_RAISE")
        self.assertEqual(bot.flying_barracks.issued[0][0], "MOVE")
        self.assertEqual(bot.flying_barracks.issued[1][0], "LAND_BARRACKS")
        self.assertEqual(bot.grounded_barracks.issued, [])

        attacking_plan = strategy.strategy_plan_from_dict(
            [{"type": "attack", "unit": "barracks", "location": "enemy_main"}]
        )
        with self.assertRaisesRegex(
            validator.PlanValidationError, "cannot issue an attack"
        ):
            validator.validate_strategy_plan(attacking_plan)

    def test_fake_executor_dispatches_lift_land_unload_cancel_salvage_and_nuke_wrappers(
        self,
    ):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {"type": "lift", "building": "barracks"},
                {"type": "land", "building": "barracks", "location": "proxy"},
                {"type": "unload", "transport": "medivac", "location": "enemy_main"},
                {"type": "cancel", "target": "build_in_progress", "actor": "barracks"},
                {"type": "salvage", "target": "bunker"},
                {"type": "build_nuke"},
                {"type": "launch_nuke", "location": "enemy_main"},
            ],
        )

        self.assertEqual(
            [order[0] for order in bot.barracks.issued[:3]],
            ["LIFT_BARRACKS", "LAND_BARRACKS", "CANCEL_BUILDINPROGRESS"],
        )
        self.assertEqual(bot.medivac.issued[0][0], "UNLOADALLAT_MEDIVAC")
        self.assertEqual(bot.bunker.issued[0][0], "SALVAGEEFFECT_SALVAGE")
        self.assertEqual(bot.ghost_academy.issued[0][0], "BUILD_NUKE")
        self.assertEqual(bot.ghost.issued[0][0], "TACNUKESTRIKE_NUKECALLDOWN")

    def test_unavailable_ability_retries_replans_and_stops_at_terminal_cap(self):
        bot = _run_fake_plan(
            UnavailableAbilityFakeBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "medivac_afterburners",
                    "actor": "medivac",
                }
            ],
            max_steps=120,
            original_goal="stim and attack",
            replan_limit=2,
        )

        self.assertEqual(bot.medivac.issued, [])
        self.assertEqual(getattr(bot, "_replan_count", 0), 2)
        self.assertEqual(len(bot.plan_requests), 2)
        self.assertTrue(bot.client.left)


class RepresentativeStrategyPlanTest(unittest.TestCase):
    def test_representative_proxy_one_one_one_drop_orbital_mech_air_and_nuke_plans_validate(
        self,
    ):
        representative_plans = {
            "proxy_barracks": [
                {
                    "type": "build",
                    "building": "barracks",
                    "location": "proxy",
                    "selection": {"unit": "worker", "count": 1},
                },
                {"type": "train", "unit": "marine", "count": 3},
                {"type": "rally", "building": "barracks", "location": "enemy_natural"},
                {
                    "type": "attack",
                    "unit": "marine",
                    "location": "enemy_main",
                    "selection": {"count": 3},
                },
            ],
            "one_one_one": [
                {"type": "build", "building": "supply_depot"},
                {
                    "type": "wait_until",
                    "condition": "structure_ready",
                    "target": "supply_depot",
                    "at_least": 1,
                },
                {"type": "build", "building": "barracks"},
                {"type": "build", "building": "refinery"},
                {"type": "build", "building": "factory"},
                {"type": "build_addon", "addon": "factory_tech_lab"},
                {
                    "type": "wait_until",
                    "condition": "structure_ready",
                    "target": "factory_tech_lab",
                    "at_least": 1,
                },
                {"type": "build", "building": "starport"},
                {"type": "train", "unit": "siege_tank"},
                {"type": "train", "unit": "medivac"},
                {"type": "transform", "actor": "siege_tank", "mode": "siege"},
            ],
            "bio_timing": [
                {"type": "research", "upgrade": "stimpack"},
                {
                    "type": "wait_until",
                    "condition": "upgrade_complete",
                    "target": "stimpack",
                    "at_least": 1,
                },
                {
                    "type": "use_ability",
                    "ability": "stim_marine",
                    "actor": "marine",
                    "selection": {"count": 16},
                },
                {
                    "type": "attack",
                    "unit": "marine",
                    "location": "enemy_natural",
                    "selection": {"count": 16},
                },
            ],
            "mech_push": [
                {
                    "type": "transform",
                    "actor": "siege_tank",
                    "mode": "siege",
                    "selection": {"count": 2},
                },
                {
                    "type": "transform",
                    "actor": "hellion",
                    "mode": "hellbat",
                    "selection": {"count": 6},
                },
                {
                    "type": "transform",
                    "actor": "thor",
                    "mode": "high_impact",
                    "selection": {"count": 1},
                },
                {"type": "repair", "target": "siege_tank", "workers": 4},
            ],
            "drop": [
                {"type": "load", "transport": "medivac", "unit": "marine", "count": 8},
                {
                    "type": "use_ability",
                    "ability": "medivac_afterburners",
                    "actor": "medivac",
                    "selection": {"count": 1},
                },
                {"type": "move", "unit": "medivac", "location": "enemy_main"},
                {"type": "unload", "transport": "medivac", "location": "enemy_main"},
            ],
            "orbital_macro": [
                {"type": "scan", "location": "enemy_main"},
                {"type": "call_down_mule", "location": "nearest_mineral"},
                {
                    "type": "use_ability",
                    "ability": "mule_gather",
                    "actor": "mule",
                    "location": "nearest_mineral",
                },
                {
                    "type": "use_ability",
                    "ability": "mule_repair",
                    "actor": "mule",
                    "target": "barracks",
                },
                {"type": "supply_drop", "target": "supply_depot"},
            ],
            "bunker_play": [
                {"type": "load", "transport": "bunker", "unit": "marine", "count": 4},
                {"type": "unload", "transport": "bunker"},
                {"type": "salvage", "target": "bunker"},
            ],
            "air_control": [
                {"type": "transform", "actor": "viking", "mode": "assault"},
                {
                    "type": "use_ability",
                    "ability": "liberator_ag_mode",
                    "actor": "liberator",
                    "location": "frontline",
                },
                {
                    "type": "use_ability",
                    "ability": "banshee_cloak_on",
                    "actor": "banshee",
                },
                {
                    "type": "use_ability",
                    "ability": "raven_interference_matrix",
                    "actor": "raven",
                    "target": "nearest_enemy",
                },
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_tactical_jump",
                    "actor": "battlecruiser",
                    "location": "enemy_main",
                },
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_yamato",
                    "actor": "battlecruiser",
                    "target": "nearest_enemy_structure",
                },
            ],
            "ghost_nuke": [
                {"type": "build_nuke"},
                {"type": "use_ability", "ability": "ghost_cloak_on", "actor": "ghost"},
                {"type": "launch_nuke", "location": "enemy_main"},
            ],
        }

        state = _state(
            minerals=5000,
            vespene=5000,
            supply_left=100,
            structures={
                "commandcenter": 1,
                "supplydepot": 4,
                "barracks": 3,
                "refinery": 2,
                "factory": 1,
                "starport": 1,
                "starporttechlab": 1,
                "barrackstechlab": 1,
                "orbitalcommand": 1,
                "ghostacademy": 1,
                "bunker": 1,
                "armory": 1,
                "fusioncore": 1,
            },
            structures_ready={
                "commandcenter": 1,
                "supplydepot": 4,
                "barracks": 3,
                "refinery": 2,
                "factory": 1,
                "starport": 1,
                "starporttechlab": 1,
                "barrackstechlab": 1,
                "orbitalcommand": 1,
                "ghostacademy": 1,
                "bunker": 1,
                "armory": 1,
                "fusioncore": 1,
            },
            army={
                "marine": 24,
                "siege_tank": 3,
                "hellion": 6,
                "thor": 1,
                "medivac": 2,
                "viking": 2,
                "liberator": 1,
                "banshee": 1,
                "raven": 1,
                "battlecruiser": 1,
                "ghost": 1,
            },
            upgrades=(
                "banshee_cloaking_field",
                "personal_cloaking",
                "interference_matrix",
                "battlecruiser_weapon_refit",
            ),
        )

        for name, actions in representative_plans.items():
            with self.subTest(plan=name):
                plan = strategy.strategy_plan_from_dict({"actions": actions})
                self.assertLessEqual(len(plan.actions), 24)
                self.assertIs(validator.validate_strategy_plan(plan, state), plan)


def _state(
    minerals=1000,
    vespene=1000,
    supply_left=20,
    workers=12,
    townhalls=1,
    structures=None,
    structures_ready=None,
    upgrades=(),
    army=None,
):
    structures = structures or {"commandcenter": 1}
    return GameStateSummary(
        minerals=minerals,
        vespene=vespene,
        supply=SupplySummary(used=20, cap=20 + supply_left, left=supply_left),
        workers=workers,
        townhalls=townhalls,
        army=army or {},
        structures=structures,
        structures_ready=structures_ready
        if structures_ready is not None
        else structures,
        structures_pending={},
        upgrades=tuple(upgrades),
        known_enemy_units=0,
        game_time_seconds=0.0,
    )


def _run_fake_plan(
    bot_base, actions, max_steps=80, original_goal=None, replan_limit=None
):
    bot_class = create_move_unit_bot_class(bot_base, lambda value: value)
    plan = strategy.strategy_plan_from_dict({"actions": actions})
    kwargs = {"stop_after_seconds": 0}
    if original_goal is not None:
        kwargs["original_goal"] = original_goal
    if replan_limit is not None:
        kwargs["replan_limit"] = replan_limit
    bot = bot_class(plan, **kwargs)

    async def run_plan():
        await bot.on_start()
        for iteration in range(1, max_steps + 1):
            if hasattr(bot, "time"):
                bot.time = float(bot.time) + 1.0
            await bot.on_step(iteration)
            if getattr(bot.client, "left", False):
                break

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        asyncio.run(run_plan())
    return bot


if __name__ == "__main__":
    unittest.main()
