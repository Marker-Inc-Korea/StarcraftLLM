import unittest

from starcraft_llm import commands, command_catalog, strategy, validator
from starcraft_llm.game_state import game_state_summary_to_dict
from starcraft_llm.sc2_bot import summarize_bot_state
from tests.test_complete_terran_surface import (
    AbilityFakeBotAI,
    FakeAbilityUnit,
    FakeAbilityUnits,
    _enum_name,
    _run_fake_plan,
)


class CommandUnit(FakeAbilityUnit):
    def __init__(
        self, type_name, position=(35, 42), is_ready=True, is_idle=True, health=100
    ):
        super().__init__(type_name, position, is_ready, is_idle, health)
        self.build_orders = []
        self.repair_targets = []
        self.rally_orders = []
        self.gather_orders = []
        self.return_orders = []
        self.train_orders = []
        self.research_orders = []
        self.passengers = []

    def build(self, unit_type, target=None):
        self.build_orders.append((_enum_name(unit_type), target))
        return True

    def repair(self, target):
        self.repair_targets.append(target)
        return True

    def gather(self, target, **kwargs):
        self.gather_orders.append((target, kwargs))
        return True

    def return_resource(self, **kwargs):
        self.return_orders.append(kwargs)
        return True

    def train(self, unit_type):
        self.train_orders.append(_enum_name(unit_type))
        return True

    def research(self, upgrade):
        self.research_orders.append(_enum_name(upgrade))
        return True

    def __call__(self, ability, *args, **kwargs):
        self.issued.append((_enum_name(ability), args, kwargs))
        return True


class AddonUnit(CommandUnit):
    def __init__(self, type_name, position=(20, 20), land_position=(22, 20)):
        super().__init__(type_name, position)
        self.add_on_land_position = land_position


class MissingStrategyPrimitiveSchemaTest(unittest.TestCase):
    def _schema(self, name):
        schemas = {
            schema["name"]: schema for schema in commands.llm_command_function_schemas()
        }
        self.assertIn(name, schemas)
        return schemas[name]

    def test_attack_target_schema_exposes_selector_type_and_tag_targets(self):
        schema = self._schema("attack_target")
        props = schema["parameters"]["properties"]

        self.assertEqual(props["target_unit"]["pattern"], "^[a-z0-9_]{1,64}$")
        self.assertIn("target_tag", props)
        self.assertIn({"type": "integer", "minimum": 1}, props["target_tag"]["anyOf"])

    def test_move_target_and_highest_energy_actor_selection_are_exposed(self):
        schema = self._schema("move_target")
        props = schema["parameters"]["properties"]

        self.assertIn("nearest_friendly", props["target_unit"]["enum"])
        self.assertIn("target_tag", props)
        self.assertIn(
            "highest_energy", props["selection"]["properties"]["mode"]["enum"]
        )

    def test_selection_schema_accepts_actor_tags_for_precise_control_groups(self):
        move_schema = self._schema("move")
        selection = move_schema["parameters"]["properties"]["selection"]

        tags = selection["properties"]["tags"]
        self.assertEqual(tags["type"], "array")
        self.assertEqual(tags["minItems"], 1)
        self.assertEqual(tags["maxItems"], command_catalog.MAX_SELECTION_COUNT)
        self.assertTrue(tags["uniqueItems"])
        self.assertIn({"type": "integer", "minimum": 1}, tags["items"]["anyOf"])

    def test_build_schema_supports_exact_placement_and_distance_budget(self):
        schema = self._schema("build")
        props = schema["parameters"]["properties"]

        self.assertEqual(props["placement_mode"]["enum"], ["near", "exact"])
        self.assertEqual(
            props["max_distance"],
            {"type": "integer", "minimum": 0, "maximum": 20},
        )
        self.assertEqual(
            schema["parameters"]["anyOf"],
            [
                {"properties": {"placement_mode": {"enum": ["near"]}}},
                {
                    "required": ["location"],
                    "properties": {"count": {"maximum": 1}},
                },
                {
                    "required": ["x", "y"],
                    "properties": {"count": {"maximum": 1}},
                },
            ],
        )

    def test_location_catalog_exposes_all_own_ramp_build_slots(self):
        expected = {
            "own_ramp_depot_1",
            "own_ramp_depot_2",
            "own_ramp_depot_middle",
            "own_ramp_barracks",
            "own_ramp_barracks_with_addon",
        }

        self.assertTrue(expected.issubset(command_catalog.LOCATION_SPECS))

    def test_land_on_addon_schema_targets_a_specific_addon_tag(self):
        schema = self._schema("land_on_addon")
        props = schema["parameters"]["properties"]

        self.assertIn("actor", props)
        self.assertIn("target_addon", props)
        self.assertIn("target_addon_tag", props)
        self.assertEqual(
            set(props["actor"]["enum"]), {"barracks", "factory", "starport"}
        )

    def test_rally_schema_accepts_nearest_mineral_and_friendly_tag_targets(self):
        schema = self._schema("rally")
        props = schema["parameters"]["properties"]

        self.assertIn("nearest_mineral", props["target_unit"]["enum"])
        self.assertIn("nearest_friendly", props["target_unit"]["enum"])
        self.assertIn({"type": "integer", "minimum": 1}, props["target_tag"]["anyOf"])

    def test_command_center_load_schema_accepts_targeted_worker_loading(self):
        schema = self._schema("load")
        props = schema["parameters"]["properties"]

        self.assertIn("command_center", props["actor"]["enum"])
        self.assertIn("worker", props["target_unit"]["enum"])
        self.assertIn("tags", props["target_selection"]["properties"])

    def test_unload_schema_accepts_specific_passenger_tag(self):
        schema = self._schema("unload")

        self.assertIn(
            {"type": "integer", "minimum": 1},
            schema["parameters"]["properties"]["passenger_tag"]["anyOf"],
        )


class MissingStrategyPrimitiveRoundTripTest(unittest.TestCase):
    def test_move_target_payload_round_trips_and_validates(self):
        payload = {
            "actions": [
                {
                    "type": "move_target",
                    "unit": "medivac",
                    "target_tag": 77,
                    "selection": {"tags": [33]},
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )
        validator.validate_strategy_plan(plan)

    def test_attack_target_payload_round_trips_and_validates(self):
        payload = {
            "actions": [
                {
                    "type": "attack_target",
                    "unit": "marine",
                    "target_unit": "zergling",
                    "selection": {"count": 2},
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        serialized = strategy.strategy_plan_to_dict(plan)
        self.assertEqual(serialized["actions"][0]["target_unit"], "zergling")
        self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)
        validator.validate_strategy_plan(plan)

    def test_enemy_ability_accepts_observed_cross_race_target_types(self):
        payload = {
            "actions": [
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_yamato",
                    "actor": "battlecruiser",
                    "target_unit": "carrier",
                },
                {
                    "type": "use_ability",
                    "ability": "ghost_snipe",
                    "actor": "ghost",
                    "target_unit": "zergling",
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        serialized = strategy.strategy_plan_to_dict(plan)
        self.assertEqual(serialized["actions"][0]["target"], "carrier")
        self.assertEqual(serialized["actions"][1]["target"], "zergling")
        self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)
        validator.validate_strategy_plan(plan)

    def test_enemy_ability_dsl_accepts_a_cross_race_target_type(self):
        plan = strategy.parse_strategy_plan("use ability battlecruiser_yamato carrier")

        self.assertEqual(plan.actions[0].target_unit, "carrier")
        validator.validate_strategy_plan(plan)

    def test_selection_tags_round_trip_without_being_dropped(self):
        payload = {
            "actions": [
                {
                    "type": "move",
                    "unit": "marine",
                    "location": "enemy_main",
                    "selection": {"tags": [111, 222]},
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        serialized = strategy.strategy_plan_to_dict(plan)
        self.assertEqual(serialized["actions"][0]["selection"]["tags"], [111, 222])
        self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)

    def test_build_exact_placement_round_trips(self):
        payload = {
            "actions": [
                {
                    "type": "build",
                    "building": "supply_depot",
                    "location": "own_ramp_depot_1",
                    "placement_mode": "exact",
                    "max_distance": 0,
                    "selection": {"count": 1},
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_land_on_addon_payload_round_trips(self):
        payload = {
            "actions": [
                {
                    "type": "land_on_addon",
                    "actor": "barracks",
                    "target_addon": "barracks_tech_lab",
                    "target_addon_tag": 9001,
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )
        validator.validate_strategy_plan(plan)

    def test_unload_passenger_tag_round_trips(self):
        payload = {
            "actions": [
                {
                    "type": "unload",
                    "actor": "medivac",
                    "target_unit": "marine",
                    "passenger_tag": 1234,
                }
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_build_addon_and_morph_selection_tags_round_trip(self):
        payload = {
            "actions": [
                {
                    "type": "build_addon",
                    "addon": "barracks_tech_lab",
                    "selection": {"tags": [101]},
                },
                {
                    "type": "morph",
                    "building": "orbital_command",
                    "selection": {"tags": [202]},
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_targeted_gather_return_train_and_research_round_trip(self):
        payload = {
            "actions": [
                {
                    "type": "gather",
                    "unit": "worker",
                    "resource": "minerals",
                    "target_tag": 901,
                    "selection": {"tags": [1]},
                    "queued": True,
                },
                {
                    "type": "return_cargo",
                    "unit": "worker",
                    "selection": {"tags": [1]},
                },
                {
                    "type": "train",
                    "unit": "marine",
                    "producer_selection": {"tags": [10]},
                },
                {
                    "type": "research",
                    "upgrade": "terran_infantry_weapons_level_1",
                    "researcher_selection": {"tags": [20]},
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)

        self.assertEqual(
            strategy.strategy_plan_from_dict(strategy.strategy_plan_to_dict(plan)), plan
        )

    def test_command_center_target_selection_defaults_to_worker_and_validates(self):
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "load",
                        "actor": "command_center",
                        "target_selection": {"tags": [5, 6]},
                    }
                ]
            }
        )

        self.assertEqual(plan.actions[0].target_unit, "worker")
        validator.validate_strategy_plan(plan)

    def test_invalid_precise_selection_and_placement_contracts_are_rejected(self):
        with self.assertRaises(strategy.StrategyParseError):
            strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "move",
                            "unit": "marine",
                            "location": "enemy_main",
                            "selection": {"tags": [1, 1]},
                        }
                    ]
                }
            )

        invalid_plans = (
            {
                "actions": [
                    {
                        "type": "build",
                        "building": "supply_depot",
                        "location": "own_ramp_depot_1",
                        "placement_mode": "exact",
                        "count": 2,
                    }
                ]
            },
            {
                "actions": [
                    {
                        "type": "build",
                        "building": "supply_depot",
                        "reserve_addon_space": True,
                    }
                ]
            },
            {
                "actions": [
                    {
                        "type": "move_target",
                        "unit": "medivac",
                        "target_unit": "nearest_enemy",
                    }
                ]
            },
        )
        for payload in invalid_plans:
            with self.subTest(payload=payload), self.assertRaises(
                validator.PlanValidationError
            ):
                validator.validate_strategy_plan(
                    strategy.strategy_plan_from_dict(payload)
                )


class MissingStrategyPrimitiveRuntimeTest(unittest.TestCase):
    def test_move_target_follows_the_exact_friendly_tag(self):
        class FollowBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.marine_a.tag = 70
                self.marine_b.tag = 71

        bot = _run_fake_plan(
            FollowBotAI,
            [
                {
                    "type": "move_target",
                    "unit": "medivac",
                    "target_tag": 71,
                }
            ],
        )

        self.assertEqual(bot.medivac.issued[0][0], "MOVE")
        self.assertIs(bot.medivac.issued[0][1][0], bot.marine_b)

    def test_attack_target_focus_fires_the_selected_enemy_unit(self):
        class FocusFireBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.enemy_a = CommandUnit("ZERGLING", (80, 80), health=90)
                self.enemy_b = CommandUnit("ZERGLING", (82, 82), health=20)
                self.enemy_units = FakeAbilityUnits([self.enemy_a, self.enemy_b])

        bot = _run_fake_plan(
            FocusFireBotAI,
            [
                {
                    "type": "attack_target",
                    "unit": "marine",
                    "target_unit": "lowest_health_enemy",
                    "selection": {"count": 2},
                }
            ],
        )

        self.assertIs(bot.marine_a.issued[0][1][0], bot.enemy_b)
        self.assertIs(bot.marine_b.issued[0][1][0], bot.enemy_b)

    def test_attack_target_can_select_an_exact_enemy_structure_type(self):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {
                    "type": "attack_target",
                    "unit": "marine",
                    "target_unit": "hatchery",
                    "selection": {"count": 1},
                }
            ],
        )

        self.assertIs(bot.marine_a.issued[0][1][0], bot.enemy_structure)

    def test_attack_target_tag_is_cross_checked_against_its_selector(self):
        class TaggedGroundEnemyBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.enemy.tag = 702

        bot = _run_fake_plan(
            TaggedGroundEnemyBotAI,
            [
                {
                    "type": "attack_target",
                    "unit": "marine",
                    "target_unit": "nearest_enemy_air",
                    "target_tag": 702,
                    "selection": {"count": 1},
                }
            ],
            max_steps=1,
        )

        self.assertEqual(bot.marine_a.issued, [])

    def test_enemy_ability_can_target_an_exact_cross_race_structure_type(self):
        class YamatoStructureBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.battlecruiser = CommandUnit("BATTLECRUISER")
                self.units.append(self.battlecruiser)

        bot = _run_fake_plan(
            YamatoStructureBotAI,
            [
                {
                    "type": "use_ability",
                    "ability": "battlecruiser_yamato",
                    "actor": "battlecruiser",
                    "target_unit": "hatchery",
                }
            ],
        )

        self.assertEqual(bot.battlecruiser.issued[0][0], "YAMATO_YAMATOGUN")
        self.assertIs(bot.battlecruiser.issued[0][1][0], bot.enemy_structure)

    def test_selection_tags_filter_actor_units_before_count_and_sorting(self):
        class TaggedMarineBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.marine_a.tag = 111
                self.marine_b.tag = 222

        bot = _run_fake_plan(
            TaggedMarineBotAI,
            [
                {
                    "type": "move",
                    "unit": "marine",
                    "location": "enemy_main",
                    "selection": {"tags": [222], "count": 2},
                }
            ],
        )

        self.assertEqual(bot.marine_a.issued, [])
        self.assertEqual(bot.marine_b.issued[0][0], "MOVE")

    def test_land_on_addon_uses_the_selected_addon_land_position(self):
        class LandOnAddonBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.flying_barracks = CommandUnit("BARRACKSFLYING")
                self.free_addon = AddonUnit("BARRACKSTECHLAB", land_position=(27, 31))
                self.free_addon.tag = 9001
                self.structures = FakeAbilityUnits(
                    [self.flying_barracks, self.free_addon]
                )

        bot = _run_fake_plan(
            LandOnAddonBotAI,
            [
                {
                    "type": "land_on_addon",
                    "actor": "barracks",
                    "target_addon": "barracks_tech_lab",
                    "target_addon_tag": 9001,
                }
            ],
        )

        self.assertEqual(bot.flying_barracks.issued[0][0], "LAND_BARRACKS")
        self.assertEqual(bot.flying_barracks.issued[0][1][0], (27, 31))

    def test_addon_tag_cannot_resolve_to_a_non_addon_structure(self):
        class WrongAddonTagBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.flying_barracks = CommandUnit("BARRACKSFLYING")
                self.ground_barracks = AddonUnit("BARRACKS")
                self.ground_barracks.tag = 704
                self.structures = FakeAbilityUnits(
                    [self.flying_barracks, self.ground_barracks]
                )

        bot = _run_fake_plan(
            WrongAddonTagBotAI,
            [
                {
                    "type": "land_on_addon",
                    "actor": "barracks",
                    "target_addon_tag": 704,
                }
            ],
            max_steps=35,
        )

        self.assertEqual(bot.flying_barracks.issued, [])
        self.assertTrue(bot.client.left)

    def test_rally_to_nearest_mineral_uses_the_mineral_unit_target(self):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {
                    "type": "rally",
                    "building": "orbital_command",
                    "target_unit": "nearest_mineral",
                }
            ],
        )

        self.assertIs(bot.orbital.issued[0][1][0], bot.mineral)

    def test_command_center_targeted_load_uses_transport_ability_for_each_worker(self):
        class CommandCenterLoadBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.command_center = CommandUnit("COMMANDCENTER")
                self.worker_a = CommandUnit("SCV")
                self.worker_b = CommandUnit("SCV")
                self.worker_a.tag = 5
                self.worker_b.tag = 6
                self.workers = FakeAbilityUnits([self.worker_a, self.worker_b])
                self.structures = FakeAbilityUnits([self.command_center])
                self.townhalls = FakeAbilityUnits([self.command_center])
                self.available_abilities.add(
                    "COMMANDCENTERTRANSPORT_COMMANDCENTERTRANSPORT"
                )

        bot = _run_fake_plan(
            CommandCenterLoadBotAI,
            [
                {
                    "type": "load",
                    "actor": "command_center",
                    "count": 2,
                    "target_selection": {"tags": [5, 6]},
                }
            ],
        )

        load_orders = [
            order
            for order in bot.command_center.issued
            if order[0] == "COMMANDCENTERTRANSPORT_COMMANDCENTERTRANSPORT"
        ]
        self.assertEqual(len(load_orders), 2)
        self.assertIs(load_orders[0][1][0], bot.worker_a)
        self.assertIs(load_orders[1][1][0], bot.worker_b)

    def test_tagged_ability_target_must_match_the_ability_target_filter(self):
        class IncompatibleTaggedTargetBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.marine_a.tag = 700

        bot = _run_fake_plan(
            IncompatibleTaggedTargetBotAI,
            [{"type": "supply_drop", "target_tag": 700}],
            max_steps=35,
        )

        self.assertEqual(bot.orbital.issued, [])
        self.assertTrue(bot.client.left)

    def test_tagged_ability_target_can_issue_when_type_is_compatible(self):
        class CompatibleTaggedTargetBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.depot = CommandUnit("SUPPLYDEPOT")
                self.depot.tag = 701
                self.structures.append(self.depot)

        bot = _run_fake_plan(
            CompatibleTaggedTargetBotAI,
            [{"type": "supply_drop", "target_tag": 701}],
        )

        self.assertEqual(bot.orbital.issued[0][0], "SUPPLYDROP_SUPPLYDROP")
        self.assertIs(bot.orbital.issued[0][1][0], bot.depot)

    def test_generic_available_ability_redirect_matches_exact_issue_ability(self):
        try:
            from sc2.ids.ability_id import AbilityId
        except ImportError:
            self.skipTest("BurnySC2 is not installed in this Python environment")

        class GenericAvailabilityBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.command_center = CommandUnit("COMMANDCENTER")
                self.worker = CommandUnit("SCV")
                self.workers = FakeAbilityUnits([self.worker])
                self.structures = FakeAbilityUnits([self.command_center])
                self.townhalls = FakeAbilityUnits([self.command_center])
                self.available_abilities = {AbilityId.LOAD}

        bot = _run_fake_plan(
            GenericAvailabilityBotAI,
            [
                {
                    "type": "load",
                    "actor": "command_center",
                    "target_unit": "worker",
                }
            ],
        )

        self.assertEqual(
            bot.command_center.issued[0][0],
            "COMMANDCENTERTRANSPORT_COMMANDCENTERTRANSPORT",
        )
        self.assertIs(bot.command_center.issued[0][1][0], bot.worker)

    def test_distinct_exact_abilities_with_same_redirect_do_not_match(self):
        try:
            from sc2.ids.ability_id import AbilityId
        except ImportError:
            self.skipTest("BurnySC2 is not installed in this Python environment")

        class WrongExactAvailabilityBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.command_center = CommandUnit("COMMANDCENTER")
                self.worker = CommandUnit("SCV")
                self.workers = FakeAbilityUnits([self.worker])
                self.structures = FakeAbilityUnits([self.command_center])
                self.townhalls = FakeAbilityUnits([self.command_center])
                self.available_abilities = {AbilityId.LOAD_BUNKER}

        bot = _run_fake_plan(
            WrongExactAvailabilityBotAI,
            [
                {
                    "type": "load",
                    "actor": "command_center",
                    "target_unit": "worker",
                }
            ],
            max_steps=35,
        )

        self.assertEqual(bot.command_center.issued, [])
        self.assertTrue(bot.client.left)

    def test_load_lowest_health_selection_is_not_overridden_by_distance_sort(self):
        class LoadLowestHealthBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.marine_a.health = 90
                self.marine_a.position = (12, 12)
                self.marine_b.health = 10
                self.marine_b.position = (40, 40)

        bot = _run_fake_plan(
            LoadLowestHealthBotAI,
            [
                {
                    "type": "load",
                    "actor": "medivac",
                    "target_unit": "marine",
                    "count": 1,
                    "target_selection": {"mode": "lowest_health"},
                }
            ],
        )

        self.assertEqual(bot.medivac.issued[0][0], "LOAD_MEDIVAC")
        self.assertIs(bot.medivac.issued[0][1][0], bot.marine_b)

    def test_highest_energy_selection_chooses_one_orbital_caster(self):
        class HighestEnergyBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.orbital_a = CommandUnit("ORBITALCOMMAND")
                self.orbital_b = CommandUnit("ORBITALCOMMAND")
                self.orbital_a.energy = 25
                self.orbital_b.energy = 100
                self.structures = FakeAbilityUnits([self.orbital_a, self.orbital_b])
                self.townhalls = FakeAbilityUnits([self.orbital_a, self.orbital_b])

        bot = _run_fake_plan(
            HighestEnergyBotAI,
            [
                {
                    "type": "scan",
                    "location": "enemy_main",
                    "selection": {"mode": "highest_energy", "count": 1},
                }
            ],
        )

        self.assertEqual(bot.orbital_a.issued, [])
        self.assertEqual(bot.orbital_b.issued[0][0], "SCANNERSWEEP_SCAN")

    def test_unload_passenger_tag_targets_that_exact_cargo_unit(self):
        class PassengerUnloadBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.loaded_marine = CommandUnit("MARINE")
                self.loaded_marauder = CommandUnit("MARAUDER")
                self.loaded_marine.tag = 7
                self.loaded_marauder.tag = 8
                self.medivac.passengers = [self.loaded_marine, self.loaded_marauder]

        bot = _run_fake_plan(
            PassengerUnloadBotAI,
            [
                {
                    "type": "unload",
                    "actor": "medivac",
                    "target_unit": "marine",
                    "passenger_tag": 7,
                }
            ],
        )

        self.assertEqual(bot.medivac.issued[0][0], "UNLOADUNIT_MEDIVAC")
        self.assertIs(bot.medivac.issued[0][1][0], bot.loaded_marine)

    def test_passenger_tag_must_match_the_transport_target_filter(self):
        class IncompatiblePassengerBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.command_center = CommandUnit("COMMANDCENTER")
                self.loaded_marine = CommandUnit("MARINE")
                self.loaded_marine.tag = 703
                self.command_center.passengers = [self.loaded_marine]
                self.structures = FakeAbilityUnits([self.command_center])
                self.townhalls = FakeAbilityUnits([self.command_center])

        bot = _run_fake_plan(
            IncompatiblePassengerBotAI,
            [
                {
                    "type": "unload",
                    "actor": "command_center",
                    "passenger_tag": 703,
                }
            ],
            max_steps=35,
        )

        self.assertEqual(bot.command_center.issued, [])
        self.assertTrue(bot.client.left)

    def test_repair_target_tag_assigns_workers_to_that_exact_damaged_target(self):
        class RepairTagBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.worker = CommandUnit("SCV")
                self.damaged_barracks = CommandUnit("BARRACKS", health=40)
                self.other_barracks = CommandUnit("BARRACKS", health=30)
                self.damaged_barracks.tag = 10
                self.other_barracks.tag = 11
                self.workers = FakeAbilityUnits([self.worker])
                self.structures = FakeAbilityUnits(
                    [self.other_barracks, self.damaged_barracks]
                )

        bot = _run_fake_plan(
            RepairTagBotAI,
            [
                {
                    "type": "repair",
                    "target": "barracks",
                    "target_tag": 10,
                    "workers": 1,
                }
            ],
        )

        self.assertEqual(bot.worker.repair_targets, [bot.damaged_barracks])

    def test_repair_lowest_health_selection_repairs_the_most_damaged_candidate(self):
        class RepairLowestHealthBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.worker = CommandUnit("SCV")
                self.healthier_tank = CommandUnit("SIEGETANK", health=80)
                self.wounded_tank = CommandUnit("SIEGETANK", health=25)
                self.workers = FakeAbilityUnits([self.worker])
                self.units = FakeAbilityUnits([self.healthier_tank, self.wounded_tank])

        bot = _run_fake_plan(
            RepairLowestHealthBotAI,
            [
                {
                    "type": "repair",
                    "target": "siege_tank",
                    "workers": 1,
                    "target_selection": {"mode": "lowest_health"},
                }
            ],
        )

        self.assertEqual(bot.worker.repair_targets, [bot.wounded_tank])

    def test_build_addon_selection_tags_choose_the_exact_producer(self):
        class AddonProducerBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks_a = CommandUnit("BARRACKS")
                self.barracks_b = CommandUnit("BARRACKS")
                self.barracks_a.tag = 13
                self.barracks_b.tag = 14
                self.structures = FakeAbilityUnits([self.barracks_a, self.barracks_b])

        bot = _run_fake_plan(
            AddonProducerBotAI,
            [
                {
                    "type": "build_addon",
                    "addon": "barracks_tech_lab",
                    "selection": {"tags": [14]},
                }
            ],
        )

        self.assertEqual(bot.barracks_a.build_orders, [])
        self.assertEqual(bot.barracks_b.build_orders[0][0], "BARRACKSTECHLAB")

    def test_morph_selection_tags_choose_the_exact_command_center(self):
        class MorphProducerBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.cc_a = CommandUnit("COMMANDCENTER")
                self.cc_b = CommandUnit("COMMANDCENTER")
                self.cc_a.tag = 15
                self.cc_b.tag = 16
                self.structures = FakeAbilityUnits([self.cc_a, self.cc_b])
                self.townhalls = FakeAbilityUnits([self.cc_a, self.cc_b])

        bot = _run_fake_plan(
            MorphProducerBotAI,
            [
                {
                    "type": "morph",
                    "building": "orbital_command",
                    "selection": {"tags": [16]},
                }
            ],
        )

        self.assertEqual(bot.cc_a.build_orders, [])
        self.assertEqual(bot.cc_b.build_orders[0][0], "ORBITALCOMMAND")

    def test_exact_addon_safe_build_uses_zero_distance_and_selected_worker(self):
        class ExactBuildBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.worker_a = CommandUnit("SCV")
                self.worker_b = CommandUnit("SCV")
                self.worker_a.tag = 30
                self.worker_b.tag = 31
                self.workers = FakeAbilityUnits([self.worker_a, self.worker_b])
                self.find_placement_calls = []

            async def find_placement(self, unit_type, near, **kwargs):
                self.find_placement_calls.append(
                    (_enum_name(unit_type), near, dict(kwargs))
                )
                return near

        bot = _run_fake_plan(
            ExactBuildBotAI,
            [
                {
                    "type": "build",
                    "building": "barracks",
                    "x": 25,
                    "y": 30,
                    "placement_mode": "exact",
                    "reserve_addon_space": True,
                    "selection": {"tags": [31]},
                }
            ],
            max_steps=1,
        )

        self.assertEqual(bot.find_placement_calls[0][2]["max_distance"], 0)
        self.assertTrue(bot.find_placement_calls[0][2]["addon_place"])
        self.assertEqual(bot.worker_a.build_orders, [])
        self.assertEqual(bot.worker_b.build_orders[0][0], "BARRACKS")
        self.assertEqual(bot.worker_b.build_orders[0][1], (25.0, 30.0))

    def test_refinery_build_honors_exact_worker_selection(self):
        class RefineryWorkerBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.worker_a = CommandUnit("SCV")
                self.worker_b = CommandUnit("SCV")
                self.worker_a.tag = 40
                self.worker_b.tag = 41
                self.workers = FakeAbilityUnits([self.worker_a, self.worker_b])
                self.geyser = CommandUnit("VESPENEGEYSER", (17, 18))
                self.vespene_geyser = FakeAbilityUnits([self.geyser])

        bot = _run_fake_plan(
            RefineryWorkerBotAI,
            [
                {
                    "type": "build",
                    "building": "refinery",
                    "selection": {"tags": [41]},
                }
            ],
            max_steps=1,
        )

        self.assertEqual(bot.worker_a.build_orders, [])
        self.assertEqual(bot.worker_b.build_orders[0][0], "REFINERY")
        self.assertIs(bot.worker_b.build_orders[0][1], bot.geyser)

    def test_targeted_gather_and_return_cargo_honor_worker_tags_and_queue(self):
        class WorkerOrdersBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.worker_a = CommandUnit("SCV")
                self.worker_b = CommandUnit("SCV")
                self.worker_a.tag = 50
                self.worker_b.tag = 51
                self.workers = FakeAbilityUnits([self.worker_a, self.worker_b])
                self.mineral_a = CommandUnit("MINERALFIELD", (10, 10))
                self.mineral_b = CommandUnit("MINERALFIELD", (20, 20))
                self.mineral_a.tag = 60
                self.mineral_b.tag = 61
                self.mineral_field = FakeAbilityUnits([self.mineral_a, self.mineral_b])

        bot = _run_fake_plan(
            WorkerOrdersBotAI,
            [
                {
                    "type": "gather",
                    "resource": "minerals",
                    "target_tag": 61,
                    "selection": {"tags": [51]},
                    "queued": True,
                },
                {
                    "type": "return_cargo",
                    "unit": "worker",
                    "selection": {"tags": [51]},
                    "queued": True,
                },
            ],
        )

        self.assertEqual(bot.worker_a.gather_orders, [])
        self.assertIs(bot.worker_b.gather_orders[0][0], bot.mineral_b)
        self.assertTrue(bot.worker_b.gather_orders[0][1]["queue"])
        self.assertEqual(bot.worker_a.return_orders, [])
        self.assertTrue(bot.worker_b.return_orders[0]["queue"])

    def test_train_and_research_choose_exact_production_tags(self):
        class ProducerTagBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks_a = CommandUnit("BARRACKS")
                self.barracks_b = CommandUnit("BARRACKS")
                self.engineering_a = CommandUnit("ENGINEERINGBAY")
                self.engineering_b = CommandUnit("ENGINEERINGBAY")
                self.barracks_a.tag = 80
                self.barracks_b.tag = 81
                self.engineering_a.tag = 90
                self.engineering_b.tag = 91
                self.structures = FakeAbilityUnits(
                    [
                        self.barracks_a,
                        self.barracks_b,
                        self.engineering_a,
                        self.engineering_b,
                    ]
                )

        bot = _run_fake_plan(
            ProducerTagBotAI,
            [
                {
                    "type": "train",
                    "unit": "marine",
                    "producer_selection": {"tags": [81]},
                },
                {
                    "type": "research",
                    "upgrade": "terran_infantry_weapons_level_1",
                    "researcher_selection": {"tags": [91]},
                },
            ],
        )

        self.assertEqual(bot.barracks_a.train_orders, [])
        self.assertEqual(bot.barracks_b.train_orders, ["MARINE"])
        self.assertEqual(bot.engineering_a.research_orders, [])
        self.assertEqual(
            bot.engineering_b.research_orders, ["TERRANINFANTRYWEAPONSLEVEL1"]
        )

    def test_busy_structure_can_receive_rally_and_observation_exposes_target_tags(self):
        class ObservationBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.orbital.is_idle = False
                self.barracks.add_on_tag = 12345
                passenger = CommandUnit("MARINE")
                passenger.tag = 54321
                self.medivac.passengers = [passenger]
                self.medivac.weapon_cooldown = 2.5

        bot = _run_fake_plan(
            ObservationBotAI,
            [
                {
                    "type": "rally",
                    "building": "orbital_command",
                    "target_unit": "nearest_mineral",
                }
            ],
        )

        self.assertEqual(bot.orbital.issued[0][0], "RALLY_COMMANDCENTER")
        observations = game_state_summary_to_dict(summarize_bot_state(bot))[
            "unit_observations"
        ]
        by_tag = {str(item.get("tag")): item for item in observations}
        self.assertEqual(by_tag[str(bot.barracks.tag)]["add_on_tag"], 12345)
        self.assertEqual(by_tag[str(bot.medivac.tag)]["passenger_tags"], [54321])
        self.assertEqual(by_tag[str(bot.medivac.tag)]["passenger_units"], ["marine"])
        self.assertEqual(by_tag[str(bot.medivac.tag)]["weapon_cooldown"], 2.5)
        self.assertEqual(by_tag[str(bot.enemy_structure.tag)]["unit"], "hatchery")
        self.assertEqual(by_tag[str(bot.enemy_structure.tag)]["alliance"], "enemy")
        self.assertEqual(by_tag[str(bot.mineral.tag)]["unit"], "mineralfield")
        self.assertEqual(by_tag[str(bot.mineral.tag)]["alliance"], "neutral")


if __name__ == "__main__":
    unittest.main()
