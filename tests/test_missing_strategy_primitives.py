import asyncio
import contextlib
import io
import unittest

from starcraft_llm import commands, command_catalog, strategy, validator
from starcraft_llm.game_state import (
    GameStateSummary,
    SupplySummary,
    game_state_summary_to_dict,
)
from starcraft_llm.sc2_bot import create_move_unit_bot_class, summarize_bot_state
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


def _make_fake_bot(bot_base, actions):
    bot_class = create_move_unit_bot_class(bot_base, lambda value: value)
    plan = strategy.strategy_plan_from_dict({"actions": actions})
    return bot_class(plan, stop_after_seconds=0)


def _run_async_scenario(scenario):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return asyncio.run(scenario())


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


class StrategyControlPolicyContractTest(unittest.TestCase):
    def test_new_control_policy_schemas_expose_bounded_contracts(self):
        schemas = {
            schema["name"]: schema for schema in commands.llm_command_function_schemas()
        }

        self.assertEqual(
            schemas["move_and_wait"]["parameters"]["properties"]["arrival_tolerance"],
            {"type": "number", "minimum": 0.25, "maximum": 20},
        )
        self.assertIn(
            "target_tag",
            schemas["move_and_wait"]["parameters"]["properties"],
        )
        self.assertEqual(
            schemas["focus_fire"]["parameters"]["properties"]["timeout_seconds"][
                "maximum"
            ],
            command_catalog.MAX_POLICY_SECONDS,
        )
        for actor in ("planetary_fortress", "missile_turret", "auto_turret"):
            with self.subTest(actor=actor):
                for command_name in ("move", "attack_move", "kite"):
                    self.assertNotIn(
                        actor,
                        schemas[command_name]["parameters"]["properties"]["unit"][
                            "enum"
                        ],
                    )
                for command_name in ("attack_target", "focus_fire", "stop"):
                    self.assertIn(
                        actor,
                        schemas[command_name]["parameters"]["properties"]["unit"][
                            "enum"
                        ],
                    )
        wait_conditions = schemas["wait_until"]["parameters"]["properties"][
            "condition"
        ]["enum"]
        self.assertTrue(
            {
                "army_supply",
                "enemy_unit_count",
                "enemy_structure_count",
                "idle_structure_count",
                "producer_available",
                "cargo_used",
                "unit_near_location",
                "enemy_near_location",
                "under_attack",
            }.issubset(wait_conditions)
        )
        for name in ("produce_until", "maintain_production"):
            with self.subTest(name=name):
                parameters = schemas[name]["parameters"]
                self.assertEqual(parameters["required"], ["unit", "target_count"])
                self.assertIn("producer_selection", parameters["properties"])
                self.assertEqual(
                    parameters["properties"]["max_seconds"]["maximum"],
                    command_catalog.MAX_POLICY_SECONDS,
                )

    def test_control_policy_payloads_round_trip_without_losing_fields(self):
        payload = {
            "actions": [
                {
                    "type": "move_and_wait",
                    "unit": "medivac",
                    "location": "enemy_main",
                    "arrival_tolerance": 1.5,
                    "timeout_seconds": 45,
                },
                {
                    "type": "focus_fire",
                    "unit": "marine",
                    "target_tag": 701,
                    "timeout_seconds": 20,
                },
                {
                    "type": "kite",
                    "unit": "marine",
                    "target_unit": "nearest_enemy",
                    "duration_seconds": 6,
                    "retreat_distance": 3,
                },
                {
                    "type": "wait_until",
                    "condition": "enemy_near_location",
                    "target": "zergling",
                    "location": "own_main",
                    "radius": 16,
                    "at_least": 2,
                    "timeout_seconds": 30,
                    "on_timeout": "fail",
                },
                {
                    "type": "produce_until",
                    "unit": "marine",
                    "target_count": 12,
                    "reserve_minerals": 100,
                    "reserve_supply": 2,
                    "max_seconds": 180,
                },
                {
                    "type": "maintain_production",
                    "unit": "scv",
                    "target_count": 30,
                    "reserve_vespene": 50,
                    "max_seconds": 240,
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)
        serialized = strategy.strategy_plan_to_dict(plan)

        self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)
        self.assertEqual(serialized, payload)

    def test_immobile_defenses_are_rejected_from_movement_policy_payloads(self):
        payloads = (
            {
                "type": "move",
                "unit": "planetary_fortress",
                "location": "own_main",
            },
            {
                "type": "attack_move",
                "unit": "missile_turret",
                "location": "own_main",
            },
            {
                "type": "kite",
                "unit": "auto_turret",
                "target_unit": "nearest_enemy",
            },
        )

        for action in payloads:
            with self.subTest(action=action), self.assertRaises(
                strategy.StrategyParseError
            ):
                strategy.strategy_plan_from_dict({"actions": [action]})

    def test_dynamic_wait_and_production_policy_validation_is_bounded(self):
        state = GameStateSummary(
            minerals=500,
            vespene=100,
            supply=SupplySummary(used=20, cap=40, left=20),
            workers=12,
            townhalls=1,
            army={"marine": 2},
            structures={"commandcenter": 1, "barracks": 1},
            structures_ready={"commandcenter": 1, "barracks": 1},
            known_enemy_units=1,
            game_time_seconds=30,
        )
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 8,
                        "reserve_minerals": 100,
                    },
                    {
                        "type": "wait_until",
                        "condition": "enemy_near_location",
                        "location": "own_main",
                        "at_least": 1,
                        "timeout_seconds": 20,
                    },
                ]
            }
        )

        self.assertIs(validator.validate_strategy_plan(plan, state), plan)
        with self.assertRaises(strategy.StrategyParseError):
            strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "produce_until",
                            "unit": "marine",
                            "target_count": 8,
                            "max_seconds": command_catalog.MAX_POLICY_SECONDS + 1,
                        }
                    ]
                }
            )

    def test_blocking_production_and_unit_wait_materialize_supply_state(self):
        state = GameStateSummary(
            minerals=500,
            vespene=0,
            supply=SupplySummary(used=20, cap=40, left=20),
            workers=12,
            townhalls=1,
            army={"marine": 2},
            structures={"commandcenter": 1, "barracks": 1},
            structures_ready={"commandcenter": 1, "barracks": 1},
            known_enemy_units=0,
            game_time_seconds=0,
        )
        plans = (
            {
                "actions": [
                    {"type": "produce_until", "unit": "marine", "target_count": 8},
                    {"type": "train", "unit": "marine", "count": 15},
                ]
            },
            {
                "actions": [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 8,
                    },
                    {
                        "type": "wait_until",
                        "condition": "unit_count",
                        "target": "marine",
                        "at_least": 8,
                    },
                    {"type": "train", "unit": "marine", "count": 15},
                ]
            },
        )

        for payload in plans:
            with self.subTest(payload=payload), self.assertRaisesRegex(
                validator.PlanValidationError, "supply left"
            ):
                validator.validate_strategy_plan(
                    strategy.strategy_plan_from_dict(payload), state
                )


class StrategyControlPolicyRuntimeTest(unittest.TestCase):
    def test_move_and_wait_blocks_followup_until_selected_units_arrive(self):
        async def scenario():
            bot = _make_fake_bot(
                AbilityFakeBotAI,
                [
                    {
                        "type": "move_and_wait",
                        "unit": "marine",
                        "location": "enemy_main",
                        "selection": {"count": 1},
                    },
                    {
                        "type": "attack_target",
                        "unit": "marine",
                        "target_tag": 999,
                        "selection": {"count": 1},
                    },
                ],
            )
            bot.enemy.tag = 999
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual([order[0] for order in bot.marine_a.issued], ["MOVE"])

            bot.marine_a.position = (90, 90)
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual([order[0] for order in bot.marine_a.issued], ["MOVE"])
            bot.time += 1
            await bot.on_step(4)
            self.assertEqual(
                [order[0] for order in bot.marine_a.issued], ["MOVE", "ATTACK"]
            )

        _run_async_scenario(scenario)

    def test_move_and_wait_supports_a_precise_friendly_target_tag(self):
        async def scenario():
            bot = _make_fake_bot(
                AbilityFakeBotAI,
                [
                    {
                        "type": "move_and_wait",
                        "unit": "medivac",
                        "target_tag": 606,
                        "arrival_tolerance": 1,
                    }
                ],
            )
            bot.marine_a.tag = 606
            bot.marine_a.position = (30, 30)
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            self.assertIs(bot.medivac.issued[0][1][0], bot.marine_a)
            self.assertEqual(bot._current_action_index, 0)

            bot.medivac.position = (30, 30)
            bot.marine_a.position = (40, 40)
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(
                [order[0] for order in bot.medivac.issued], ["MOVE", "MOVE"]
            )
            self.assertIs(bot.medivac.issued[1][1][0], bot.marine_a)
            self.assertEqual(bot._current_action_index, 0)

            bot.medivac.position = (40, 40)
            bot.time += 1
            await bot.on_step(6)
            self.assertEqual(bot._current_action_index, 1)

        _run_async_scenario(scenario)

    def test_focus_fire_blocks_followup_until_the_observed_target_dies(self):
        async def scenario():
            bot = _make_fake_bot(
                AbilityFakeBotAI,
                [
                    {
                        "type": "focus_fire",
                        "unit": "marine",
                        "target_tag": 777,
                        "selection": {"count": 1},
                    },
                    {
                        "type": "move",
                        "unit": "marine",
                        "location": "retreat",
                        "selection": {"count": 1},
                    },
                ],
            )
            bot.enemy.tag = 777
            bot.state = type("State", (), {"dead_units": set()})()
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual([order[0] for order in bot.marine_a.issued], ["ATTACK"])

            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(
                [order[0] for order in bot.marine_a.issued], ["ATTACK", "ATTACK"]
            )

            bot.enemy_units.clear()
            bot.time += 1
            await bot.on_step(6)
            self.assertEqual(bot._current_action_index, 0)
            self.assertEqual(
                [order[0] for order in bot.marine_a.issued], ["ATTACK", "ATTACK"]
            )

            bot.state.dead_units.add(777)
            bot.time += 1
            await bot.on_step(7)
            self.assertEqual(bot._current_action_index, 1)
            bot.time += 1
            await bot.on_step(8)
            self.assertEqual(
                [order[0] for order in bot.marine_a.issued],
                ["ATTACK", "ATTACK", "MOVE"],
            )

        _run_async_scenario(scenario)

    def test_kite_attacks_when_ready_and_retreats_during_weapon_cooldown(self):
        class KiteBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.marine_a.weapon_cooldown = 0
                self.marine_b.weapon_cooldown = 2.5

        bot = _run_fake_plan(
            KiteBotAI,
            [
                {
                    "type": "kite",
                    "unit": "marine",
                    "target_unit": "nearest_enemy",
                    "duration_seconds": 2,
                }
            ],
            max_steps=1,
        )

        self.assertEqual(bot.marine_a.issued[0][0], "ATTACK")
        self.assertEqual(bot.marine_b.issued[0][0], "MOVE")

    def test_produce_until_blocks_but_background_production_runs_with_later_actions(
        self,
    ):
        class ProductionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks = CommandUnit("BARRACKS")
                self.structures = FakeAbilityUnits([self.barracks, self.orbital])

            def already_pending(self, _unit_type):
                return 0

        async def scenario():
            foreground = _make_fake_bot(
                ProductionBotAI,
                [
                    {"type": "produce_until", "unit": "marine", "target_count": 3},
                    {
                        "type": "move",
                        "unit": "marine",
                        "location": "enemy_main",
                        "selection": {"count": 1},
                    },
                ],
            )
            await foreground.on_start()
            for iteration in (1, 2):
                foreground.time += 1
                await foreground.on_step(iteration)
            self.assertEqual(foreground.barracks.train_orders, ["MARINE"])
            self.assertEqual(foreground.marine_a.issued, [])
            foreground.units.append(CommandUnit("MARINE"))
            foreground.time += 1
            await foreground.on_step(3)
            foreground.time += 1
            await foreground.on_step(4)
            self.assertEqual(foreground.marine_a.issued[0][0], "MOVE")

            background = _make_fake_bot(
                ProductionBotAI,
                [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 3,
                    },
                    {"type": "wait", "seconds": 10},
                ],
            )
            await background.on_start()
            background.time += 1
            await background.on_step(1)
            background.time += 1
            await background.on_step(2)
            self.assertEqual(background.barracks.train_orders, ["MARINE"])
            self.assertIsNotNone(background._action_started_at_loop_time)

            produced = CommandUnit("MARINE")
            background.units.append(produced)
            background.time += 1
            await background.on_step(3)
            background.units.remove(produced)
            background.time += 1
            await background.on_step(4)
            self.assertEqual(background.barracks.train_orders, ["MARINE", "MARINE"])

        _run_async_scenario(scenario)

    def test_final_background_production_waits_for_its_target_before_exit(self):
        class ProductionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks = CommandUnit("BARRACKS")
                self.structures = FakeAbilityUnits([self.barracks, self.orbital])

            def already_pending(self, _unit_type):
                return 0

        async def scenario():
            bot = _make_fake_bot(
                ProductionBotAI,
                [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 3,
                    }
                ],
            )
            await bot.on_start()

            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 1)
            self.assertIsNone(bot._plan_finished_at_loop_time)
            self.assertFalse(bot.client.left)
            self.assertEqual(bot.barracks.train_orders, [])

            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot.barracks.train_orders, ["MARINE"])
            self.assertIsNone(bot._plan_finished_at_loop_time)
            self.assertFalse(bot.client.left)

            bot.units.append(CommandUnit("MARINE"))
            bot.time += 1
            await bot.on_step(3)
            self.assertIsNotNone(bot._plan_finished_at_loop_time)
            self.assertTrue(bot.client.left)

        _run_async_scenario(scenario)

    def test_background_production_survives_the_last_foreground_action(self):
        class ProductionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks = CommandUnit("BARRACKS")
                self.structures = FakeAbilityUnits([self.barracks, self.orbital])

            def already_pending(self, _unit_type):
                return 0

        async def scenario():
            bot = _make_fake_bot(
                ProductionBotAI,
                [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 3,
                    },
                    {"type": "wait", "seconds": 0},
                ],
            )
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            bot.time += 1
            await bot.on_step(2)

            self.assertEqual(bot._current_action_index, 2)
            self.assertEqual(bot.barracks.train_orders, ["MARINE"])
            self.assertIsNone(bot._plan_finished_at_loop_time)
            self.assertFalse(bot.client.left)

            bot.units.append(CommandUnit("MARINE"))
            bot.time += 1
            await bot.on_step(3)
            self.assertTrue(bot.client.left)

        _run_async_scenario(scenario)

    def test_reactor_accepts_a_second_queue_order_but_normal_busy_producer_does_not(
        self,
    ):
        class ReactorBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.barracks = CommandUnit("BARRACKS", is_idle=False)
                self.barracks.orders = [object()]
                self.barracks.has_reactor = True
                self.structures = FakeAbilityUnits([self.barracks, self.orbital])

        reactor = _run_fake_plan(
            ReactorBotAI,
            [{"type": "train", "unit": "marine"}],
            max_steps=1,
        )
        self.assertEqual(reactor.barracks.train_orders, ["MARINE"])

        class BusyBotAI(ReactorBotAI):
            def __init__(self):
                super().__init__()
                self.barracks.has_reactor = False

        busy = _run_fake_plan(
            BusyBotAI,
            [{"type": "train", "unit": "marine"}],
            max_steps=1,
        )
        self.assertEqual(busy.barracks.train_orders, [])

        class FullReactorBotAI(ReactorBotAI):
            def __init__(self):
                super().__init__()
                self.barracks.orders = [object(), object()]

        full_reactor = _run_fake_plan(
            FullReactorBotAI,
            [{"type": "train", "unit": "marine"}],
            max_steps=1,
        )
        self.assertEqual(full_reactor.barracks.train_orders, [])

        class TaggedReactorBotAI(ReactorBotAI):
            def __init__(self):
                super().__init__()
                self.barracks.has_reactor = False
                self.barracks.add_on_tag = 900
                self.reactor_tags = {900}

        tagged_reactor = _run_fake_plan(
            TaggedReactorBotAI,
            [{"type": "train", "unit": "marine"}],
            max_steps=1,
        )
        self.assertEqual(tagged_reactor.barracks.train_orders, ["MARINE"])

    def test_transport_commands_wait_for_observed_load_and_unload_completion(self):
        class TransportBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.medivac = CommandUnit("MEDIVAC")
                self.medivac.cargo_used = 0
                self.medivac.passengers = []
                self.units = FakeAbilityUnits(
                    [self.marine_a, self.marine_b, self.medivac, self.ghost, self.mule]
                )

        async def scenario():
            load_bot = _make_fake_bot(
                TransportBotAI,
                [
                    {
                        "type": "load",
                        "actor": "medivac",
                        "target_tag": 501,
                    },
                    {"type": "move", "unit": "medivac", "location": "enemy_main"},
                ],
            )
            load_bot.marine_a.tag = 501
            await load_bot.on_start()
            for iteration in (1, 2):
                load_bot.time += 1
                await load_bot.on_step(iteration)
            self.assertEqual(
                [order[0] for order in load_bot.medivac.issued], ["LOAD_MEDIVAC"]
            )
            load_bot.marine_a.is_loaded = True
            load_bot.medivac.passengers = [load_bot.marine_a]
            load_bot.medivac.cargo_used = 1
            load_bot.time += 1
            await load_bot.on_step(3)
            load_bot.time += 1
            await load_bot.on_step(4)
            self.assertEqual(
                [order[0] for order in load_bot.medivac.issued],
                ["LOAD_MEDIVAC", "MOVE"],
            )

            unload_bot = _make_fake_bot(
                TransportBotAI,
                [
                    {"type": "unload", "actor": "medivac"},
                    {"type": "move", "unit": "medivac", "location": "retreat"},
                ],
            )
            unload_bot.medivac.passengers = [unload_bot.marine_a]
            unload_bot.medivac.cargo_used = 1
            await unload_bot.on_start()
            for iteration in (1, 2):
                unload_bot.time += 1
                await unload_bot.on_step(iteration)
            self.assertEqual(
                [order[0] for order in unload_bot.medivac.issued],
                ["UNLOADALLAT_MEDIVAC"],
            )
            self.assertEqual(
                unload_bot.medivac.issued[0][1][0], unload_bot.medivac.position
            )
            unload_bot.medivac.passengers = []
            unload_bot.medivac.cargo_used = 0
            unload_bot.time += 1
            await unload_bot.on_step(3)
            unload_bot.time += 1
            await unload_bot.on_step(4)
            self.assertEqual(
                [order[0] for order in unload_bot.medivac.issued],
                ["UNLOADALLAT_MEDIVAC", "MOVE"],
            )

        _run_async_scenario(scenario)

    def test_targeted_multi_load_waits_for_every_selected_passenger(self):
        class TransportBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.medivac = CommandUnit("MEDIVAC")
                self.medivac.cargo_used = 0
                self.medivac.passengers = []
                self.units = FakeAbilityUnits(
                    [self.marine_a, self.marine_b, self.medivac, self.ghost, self.mule]
                )

        async def scenario():
            bot = _make_fake_bot(
                TransportBotAI,
                [
                    {
                        "type": "load",
                        "actor": "medivac",
                        "target_unit": "marine",
                        "target_selection": {"count": 2},
                    },
                    {"type": "move", "unit": "medivac", "location": "enemy_main"},
                ],
            )
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual(len(bot.medivac.issued), 2)

            bot.marine_a.is_loaded = True
            bot.medivac.passengers = [bot.marine_a]
            bot.medivac.cargo_used = 1
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual(bot._current_action_index, 0)

            bot.marine_b.is_loaded = True
            bot.medivac.passengers = [bot.marine_a, bot.marine_b]
            bot.medivac.cargo_used = 2
            bot.time += 1
            await bot.on_step(4)
            self.assertEqual(bot._current_action_index, 1)
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(bot.medivac.issued[-1][0], "MOVE")

        _run_async_scenario(scenario)

    def test_specific_unload_waits_until_that_passenger_leaves_cargo(self):
        class TransportBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.medivac = CommandUnit("MEDIVAC")
                self.medivac.cargo_used = 2
                self.medivac.passengers = [self.marine_a, self.marine_b]
                self.units = FakeAbilityUnits(
                    [self.marine_a, self.marine_b, self.medivac, self.ghost, self.mule]
                )

        async def scenario():
            bot = _make_fake_bot(
                TransportBotAI,
                [
                    {
                        "type": "unload",
                        "actor": "medivac",
                        "passenger_tag": 801,
                    },
                    {"type": "move", "unit": "medivac", "location": "retreat"},
                ],
            )
            bot.marine_a.tag = 801
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)

            bot.medivac.cargo_used = 1
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual(bot._current_action_index, 0)

            bot.medivac.passengers = [bot.marine_b]
            bot.time += 1
            await bot.on_step(4)
            self.assertEqual(bot._current_action_index, 1)
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(bot.medivac.issued[-1][0], "MOVE")

        _run_async_scenario(scenario)

    def test_specific_unload_selects_the_transport_carrying_that_passenger(self):
        class MultiTransportBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.medivac_a = CommandUnit("MEDIVAC")
                self.medivac_b = CommandUnit("MEDIVAC")
                self.medivac_a.cargo_used = 1
                self.medivac_b.cargo_used = 1
                self.medivac_a.passengers = [self.marine_a]
                self.medivac_b.passengers = [self.marine_b]
                self.units = FakeAbilityUnits(
                    [
                        self.marine_a,
                        self.marine_b,
                        self.medivac_a,
                        self.medivac_b,
                        self.ghost,
                        self.mule,
                    ]
                )

        async def scenario():
            bot = _make_fake_bot(
                MultiTransportBotAI,
                [
                    {
                        "type": "unload",
                        "actor": "medivac",
                        "passenger_tag": 902,
                    },
                    {"type": "move", "unit": "medivac", "location": "retreat"},
                ],
            )
            bot.marine_b.tag = 902
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)

            self.assertEqual(bot.medivac_a.issued, [])
            self.assertEqual(bot.medivac_b.issued[0][0], "UNLOADUNIT_MEDIVAC")
            self.assertIs(bot.medivac_b.issued[0][1][0], bot.marine_b)
            self.assertEqual(bot._current_action_index, 0)

            bot.medivac_b.passengers = []
            bot.medivac_b.cargo_used = 0
            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot._current_action_index, 1)

        _run_async_scenario(scenario)

    def test_command_center_load_all_waits_for_every_nearby_worker(self):
        class LoadAllBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.command_center = CommandUnit("COMMANDCENTER")
                self.command_center.cargo_used = 0
                self.command_center.cargo_max = 5
                self.worker_a = CommandUnit("SCV", position=(35, 42))
                self.worker_b = CommandUnit("SCV", position=(36, 42))
                self.structures = FakeAbilityUnits([self.command_center])
                self.townhalls = FakeAbilityUnits([self.command_center])
                self.workers = FakeAbilityUnits([self.worker_a, self.worker_b])

        async def scenario():
            bot = _make_fake_bot(
                LoadAllBotAI,
                [
                    {"type": "load", "actor": "command_center"},
                    {"type": "lift", "actor": "command_center"},
                ],
            )
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual(
                [order[0] for order in bot.command_center.issued],
                ["LOADALL_COMMANDCENTER"],
            )

            bot.worker_a.is_loaded = True
            bot.command_center.passengers = [bot.worker_a]
            bot.command_center.cargo_used = 1
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual(bot._current_action_index, 0)

            bot.worker_b.is_loaded = True
            bot.command_center.passengers = [bot.worker_a, bot.worker_b]
            bot.command_center.cargo_used = 2
            bot.time += 1
            await bot.on_step(4)
            self.assertEqual(bot._current_action_index, 1)
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(
                [order[0] for order in bot.command_center.issued],
                ["LOADALL_COMMANDCENTER", "LIFT_COMMANDCENTER"],
            )

        _run_async_scenario(scenario)

    def test_all_new_dynamic_wait_conditions_are_observed_at_runtime(self):
        class WaitConditionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.medivac.cargo_used = 2
                self.marine_a.health = 90

        actions = [
            {"type": "wait_until", "condition": "army_supply", "at_least": 1},
            {
                "type": "wait_until",
                "condition": "enemy_unit_count",
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "enemy_structure_count",
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "idle_structure_count",
                "target": "barracks",
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "producer_available",
                "target": "marine",
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "cargo_used",
                "target": "medivac",
                "at_least": 2,
            },
            {
                "type": "wait_until",
                "condition": "unit_near_location",
                "target": "marine",
                "location": "own_main",
                "radius": 64,
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "enemy_near_location",
                "target": "zergling",
                "location": "enemy_main",
                "radius": 20,
                "at_least": 1,
            },
            {
                "type": "wait_until",
                "condition": "under_attack",
                "location": "own_main",
                "radius": 64,
                "at_least": 1,
            },
        ]

        bot = _run_fake_plan(WaitConditionBotAI, actions, max_steps=12)

        self.assertTrue(bot.client.left)
        self.assertEqual(bot._current_action_index, len(actions))

    def test_under_attack_requires_damage_or_attack_evidence_not_enemy_proximity(self):
        async def scenario():
            bot = _make_fake_bot(
                AbilityFakeBotAI,
                [
                    {
                        "type": "wait_until",
                        "condition": "under_attack",
                        "location": "own_main",
                        "radius": 10,
                        "at_least": 1,
                    }
                ],
            )
            bot.marine_a.position = (35, 42)
            bot.enemy.position = (36, 42)
            await bot.on_start()

            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 0)

            bot.marine_a.health = 90
            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot._current_action_index, 1)

        _run_async_scenario(scenario)

    def test_wait_until_fail_timeout_is_terminal_and_bounded(self):
        bot = _run_fake_plan(
            AbilityFakeBotAI,
            [
                {
                    "type": "wait_until",
                    "condition": "enemy_unit_count",
                    "at_least": 99,
                    "timeout_seconds": 2,
                    "on_timeout": "fail",
                }
            ],
            max_steps=5,
        )

        self.assertTrue(bot.client.left)
        self.assertEqual(bot._current_action_index, 0)

    def test_defensive_actors_manual_mine_fire_bunker_rally_and_rich_refinery_execute(
        self,
    ):
        class DefensiveUnit(CommandUnit):
            def stop(self, **kwargs):
                self.issued.append(("STOP", (), kwargs))
                return True

        class DefensiveBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                self.planetary = DefensiveUnit("PLANETARYFORTRESS")
                self.turret = DefensiveUnit("MISSILETURRET")
                self.auto_turret = DefensiveUnit("AUTOTURRET")
                self.mine = DefensiveUnit("WIDOWMINEBURROWED")
                self.refinery = DefensiveUnit("REFINERYRICH")
                self.worker = CommandUnit("SCV")
                self.bunker = DefensiveUnit("BUNKER")
                self.units = FakeAbilityUnits(
                    [
                        self.marine_a,
                        self.marine_b,
                        self.medivac,
                        self.ghost,
                        self.mule,
                        self.auto_turret,
                        self.mine,
                    ]
                )
                self.structures = FakeAbilityUnits(
                    [
                        self.planetary,
                        self.turret,
                        self.refinery,
                        self.bunker,
                        self.barracks,
                        self.orbital,
                    ]
                )
                self.workers = FakeAbilityUnits([self.worker])

        bot = _run_fake_plan(
            DefensiveBotAI,
            [
                {
                    "type": "attack_target",
                    "unit": "planetary_fortress",
                    "target_unit": "nearest_enemy",
                },
                {
                    "type": "attack_target",
                    "unit": "missile_turret",
                    "target_unit": "nearest_enemy",
                },
                {
                    "type": "attack_target",
                    "unit": "auto_turret",
                    "target_unit": "nearest_enemy",
                },
                {"type": "stop", "unit": "planetary_fortress"},
                {"type": "stop", "unit": "missile_turret"},
                {"type": "stop", "unit": "auto_turret"},
                {
                    "type": "use_ability",
                    "ability": "widow_mine_attack",
                    "actor": "widow_mine",
                    "target_unit": "nearest_enemy",
                },
                {"type": "stop", "unit": "widow_mine"},
                {"type": "rally", "building": "bunker", "location": "frontline"},
                {"type": "gather", "resource": "vespene", "workers": 1},
            ],
            max_steps=12,
        )

        for actor in (bot.planetary, bot.turret, bot.auto_turret):
            self.assertEqual(
                [order[0] for order in actor.issued[:2]], ["ATTACK", "STOP"]
            )
        self.assertEqual(
            [order[0] for order in bot.mine.issued],
            ["WIDOWMINEATTACK_WIDOWMINEATTACK", "STOP"],
        )
        self.assertEqual(bot.bunker.issued[0][0], "RALLY_BUILDING")
        self.assertIs(bot.worker.gather_orders[0][0], bot.refinery)


if __name__ == "__main__":
    unittest.main()
