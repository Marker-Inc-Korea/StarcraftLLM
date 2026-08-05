import asyncio
import contextlib
import io
import unittest

from sc2.data import Alert
from sc2.ids.unit_typeid import UnitTypeId

from starcraft_llm import command_catalog, commands, strategy, validator
from starcraft_llm.game_state import GameStateSummary, SupplySummary
from starcraft_llm.planner import strategy_plan_json_schema
from starcraft_llm.sc2_bot import create_move_unit_bot_class
from tests.test_complete_terran_surface import (
    AbilityFakeBotAI,
    FakeAbilityUnit,
    FakeAbilityUnits,
)
from tests.test_missing_strategy_primitives import CommandUnit


def _make_bot(actions, bot_base=AbilityFakeBotAI):
    bot_class = create_move_unit_bot_class(bot_base, lambda value: value)
    plan = strategy.strategy_plan_from_dict({"actions": actions})
    return bot_class(plan, stop_after_seconds=0)


def _run_scenario(scenario):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return asyncio.run(scenario())


def _known_state(minerals=500):
    return GameStateSummary(
        minerals=minerals,
        vespene=0,
        supply=SupplySummary(used=12, cap=23, left=11),
        workers=12,
        townhalls=1,
        army={},
        structures={"commandcenter": 1},
        structures_ready={"commandcenter": 1},
        structures_pending={},
        known_enemy_units=0,
        game_time_seconds=0,
    )


class StrategyControlFlowContractTest(unittest.TestCase):
    def test_new_function_catalog_schema_and_planner_surface_are_in_sync(self):
        expected = {
            "attack_until_clear",
            "wait_for_ability",
            "wait_for_form",
            "wait_for_idle",
            "conditional",
            "repeat",
            "repeat_until",
            "with_timeout",
            "stop_production",
        }
        schemas = {
            item["name"]: item for item in commands.llm_command_function_schemas()
        }
        catalog = {item.key for item in command_catalog.COMMAND_SURFACE}
        planner_types = set(
            strategy_plan_json_schema()["properties"]["actions"]["items"][
                "properties"
            ]["type"]["enum"]
        )

        self.assertTrue(expected.issubset(commands.LLM_COMMAND_FUNCTIONS))
        self.assertTrue(expected.issubset(schemas))
        self.assertTrue(expected.issubset(catalog))
        self.assertTrue(expected.issubset(planner_types))
        self.assertEqual(
            set(
                schemas["wait_until"]["parameters"]["properties"]["condition"][
                    "enum"
                ]
            ),
            set(command_catalog.CONDITION_KEYS),
        )
        self.assertEqual(
            schemas["repeat_until"]["parameters"]["properties"]["max_cycles"][
                "maximum"
            ],
            command_catalog.MAX_REPEAT_CYCLES,
        )
        self.assertEqual(
            set(
                schemas["repeat"]["parameters"]["properties"]["on_exhausted"][
                    "enum"
                ]
            ),
            {"replan", "fail"},
        )
        self.assertGreater(command_catalog.MAX_CONTROL_EXECUTION_ACTIONS, 0)
        for condition in command_catalog.CONDITION_KEYS:
            with self.subTest(condition=condition):
                self.assertEqual(strategy.normalize_wait_condition(condition), condition)

    def test_every_runtime_form_resolves_against_burnysc2(self):
        for form, enum_names in command_catalog.UNIT_FORM_SPECS.items():
            with self.subTest(form=form):
                self.assertTrue(enum_names)
                for enum_name in enum_names:
                    self.assertTrue(hasattr(UnitTypeId, enum_name), enum_name)

    def test_comparator_waits_and_exact_enemy_targets_round_trip(self):
        payload = {
            "actions": [
                {
                    "type": "wait_until",
                    "condition": "supply_left",
                    "at_most": 4,
                },
                {
                    "type": "wait_until",
                    "condition": "enemy_unit_count",
                    "target": "mutalisk",
                    "equals": 0,
                },
                {
                    "type": "wait_until",
                    "condition": "unit_health_fraction",
                    "target": "marine",
                    "comparison": "lt",
                    "value": 0.4,
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)
        serialized = strategy.strategy_plan_to_dict(plan)

        self.assertEqual(serialized, payload)
        self.assertEqual(strategy.strategy_plan_from_dict(serialized), plan)
        self.assertEqual(plan.actions[0].comparison, "lte")
        self.assertEqual(plan.actions[1].target, "mutalisk")

    def test_matchup_and_alert_conditions_round_trip_and_validate(self):
        self.assertEqual(
            {strategy.normalize_alert(alert.name) for alert in Alert},
            set(command_catalog.ALERT_KEYS),
        )
        payload = {
            "actions": [
                {
                    "type": "wait_until",
                    "condition": "enemy_race",
                    "target": "zerg",
                    "equals": 1,
                },
                {
                    "type": "conditional",
                    "when": {
                        "condition": "alert_active",
                        "target": "nuclear_launch_detected",
                        "at_least": 1,
                    },
                    "then_actions": [
                        {"type": "move", "unit": "marine", "location": "retreat"}
                    ],
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)
        self.assertEqual(strategy.strategy_plan_to_dict(plan), payload)
        validator.validate_strategy_plan(plan)
        dsl_action = strategy.parse_strategy_plan(
            "wait until matchup zerg"
        ).actions[0]
        self.assertEqual(dsl_action.condition, "enemy_race")
        self.assertEqual(dsl_action.target, "zerg")

        with self.assertRaisesRegex(strategy.StrategyParseError, "enemy race"):
            strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "wait_until",
                            "condition": "enemy_race",
                            "target": "klingon",
                        }
                    ]
                }
            )

        impossible_boolean = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "wait_until",
                        "condition": "alert_active",
                        "target": "nuclear_launch_detected",
                        "at_least": 2,
                    }
                ]
            }
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "0 or 1"):
            validator.validate_strategy_plan(impossible_boolean)

    def test_neutral_destructible_target_round_trips_and_validates(self):
        payload = {
            "actions": [
                {
                    "type": "focus_fire",
                    "unit": "marine",
                    "target_unit": "nearest_destructible",
                    "target_alliance": "neutral",
                    "selection": {"mode": "all", "count": 1},
                    "timeout_seconds": 30,
                }
            ]
        }
        plan = strategy.strategy_plan_from_dict(payload)
        self.assertEqual(strategy.strategy_plan_to_dict(plan), payload)
        validator.validate_strategy_plan(plan)

        invalid = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "attack_target",
                        "unit": "marine",
                        "target_unit": "nearest_enemy",
                        "target_alliance": "neutral",
                    }
                ]
            }
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "alliance"):
            validator.validate_strategy_plan(invalid)

        schemas = {
            item["name"]: item for item in commands.llm_command_function_schemas()
        }
        self.assertEqual(
            set(
                schemas["attack_target"]["parameters"]["properties"][
                    "target_alliance"
                ]["enum"]
            ),
            {"enemy", "neutral"},
        )

    def test_typed_waits_and_nested_control_flow_round_trip(self):
        payload = {
            "actions": [
                {"type": "wait_for_ability", "ability": "scan"},
                {
                    "type": "wait_for_form",
                    "unit": "siege_tank",
                    "form": "siege_tank_sieged",
                },
                {
                    "type": "conditional",
                    "when": {
                        "match": "all",
                        "conditions": [
                            {
                                "condition": "enemy_unit_count",
                                "target": "mutalisk",
                                "at_least": 3,
                            },
                            {
                                "condition": "unit_count",
                                "target": "viking",
                                "at_most": 5,
                            },
                        ],
                    },
                    "then_actions": [
                        {
                            "type": "repeat_until",
                            "until": {
                                "condition": "enemy_unit_count",
                                "target": "mutalisk",
                                "at_most": 0,
                            },
                            "actions": [
                                {"type": "attack_enemy", "unit": "viking"}
                            ],
                            "max_cycles": 12,
                            "on_exhausted": "continue",
                        }
                    ],
                    "else_actions": [{"type": "wait", "seconds": 1}],
                },
            ]
        }

        plan = strategy.strategy_plan_from_dict(payload)
        canonical = strategy.strategy_plan_to_dict(plan)

        self.assertEqual(strategy.strategy_plan_from_dict(canonical), plan)
        self.assertEqual(canonical["actions"][0]["condition"], "ability_available")
        self.assertEqual(canonical["actions"][1]["condition"], "unit_form_count")
        validator.validate_strategy_plan(plan)

    def test_with_timeout_round_trips_and_validates_its_nested_body(self):
        payload = {
            "actions": [
                {
                    "type": "with_timeout",
                    "actions": [
                        {"type": "wait", "seconds": 1},
                        {"type": "attack_enemy", "unit": "marine"},
                    ],
                    "timeout_seconds": 30,
                    "on_timeout": "fail",
                }
            ]
        }
        plan = strategy.strategy_plan_from_dict(payload)
        self.assertEqual(strategy.strategy_plan_to_dict(plan), payload)
        validator.validate_strategy_plan(plan)

        invalid = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "with_timeout",
                        "actions": [{"type": "build", "building": "barracks"}],
                    }
                ]
            }
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "supply depot"):
            validator.validate_strategy_plan(invalid, _known_state())

    def test_condition_threshold_conflicts_and_control_bounds_fail_locally(self):
        with self.assertRaisesRegex(strategy.StrategyParseError, "threshold field"):
            strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "wait_until",
                            "condition": "supply_left",
                            "at_least": 1,
                            "at_most": 4,
                        }
                    ]
                }
            )

        nested = {"type": "wait", "seconds": 0}
        for _ in range(command_catalog.MAX_CONTROL_DEPTH + 1):
            nested = {"type": "repeat", "cycles": 1, "actions": [nested]}
        with self.assertRaisesRegex(strategy.StrategyParseError, "nesting"):
            strategy.strategy_plan_from_dict({"actions": [nested]})

        oversized = {
            "type": "conditional",
            "when": {"condition": "minerals", "at_least": 0},
            "then_actions": [
                {"type": "wait", "seconds": 0}
                for _ in range(command_catalog.MAX_CONTROL_BRANCH_ACTIONS + 1)
            ],
        }
        with self.assertRaisesRegex(strategy.StrategyParseError, "too many"):
            strategy.strategy_plan_from_dict({"actions": [oversized]})

        explosive = {
            "type": "repeat",
            "cycles": command_catalog.MAX_REPEAT_CYCLES,
            "actions": [
                {
                    "type": "repeat",
                    "cycles": command_catalog.MAX_REPEAT_CYCLES,
                    "actions": [{"type": "wait", "seconds": 0}],
                }
            ],
        }
        with self.assertRaisesRegex(strategy.StrategyParseError, "runtime actions"):
            strategy.strategy_plan_from_dict({"actions": [explosive]})

    def test_validator_checks_every_branch_and_loop_body(self):
        invalid_branch = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "conditional",
                        "when": {"condition": "enemy_unit_count", "at_least": 1},
                        "then_actions": [
                            {"type": "build", "building": "barracks"}
                        ],
                        "else_actions": [{"type": "wait", "seconds": 0}],
                    }
                ]
            }
        )
        invalid_loop = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "repeat",
                        "cycles": 2,
                        "actions": [{"type": "build", "building": "barracks"}],
                    }
                ]
            }
        )

        for plan in (invalid_branch, invalid_loop):
            with self.subTest(plan=plan), self.assertRaisesRegex(
                validator.PlanValidationError, "supply depot"
            ):
                validator.validate_strategy_plan(plan, _known_state())

    def test_upper_bound_wait_does_not_invent_resources(self):
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "wait_until",
                        "condition": "minerals",
                        "at_most": 0,
                    },
                    {"type": "build", "building": "supply_depot"},
                ]
            }
        )

        with self.assertRaisesRegex(validator.PlanValidationError, "minerals"):
            validator.validate_strategy_plan(plan, _known_state(minerals=0))

    def test_ability_wait_rejects_an_incompatible_actor(self):
        plan = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "wait_until",
                        "condition": "ability_available",
                        "ability": "scan",
                        "actor": "marine",
                        "at_least": 1,
                    }
                ]
            }
        )

        with self.assertRaisesRegex(validator.PlanValidationError, "cannot issue"):
            validator.validate_strategy_plan(plan)

    def test_form_wait_and_selection_capacity_reject_impossible_conditions(self):
        cases = (
            (
                {
                    "type": "wait_for_form",
                    "unit": "marine",
                    "form": "siege_tank_sieged",
                },
                "incompatible",
            ),
            (
                {
                    "type": "wait_for_idle",
                    "unit": "marine",
                    "count": 5,
                    "selection": {"count": 3},
                },
                "at most 3",
            ),
            (
                {
                    "type": "wait_until",
                    "condition": "structure_ready",
                    "target": "barracks",
                    "at_least": 2,
                    "selection": {"tags": [101]},
                },
                "at most 1",
            ),
        )
        for payload, message in cases:
            with self.subTest(payload=payload), self.assertRaisesRegex(
                validator.PlanValidationError, message
            ):
                validator.validate_strategy_plan(
                    strategy.strategy_plan_from_dict({"actions": [payload]})
                )

    def test_conditional_guards_refine_then_and_else_branch_facts(self):
        state = GameStateSummary(
            minerals=50,
            vespene=0,
            supply=SupplySummary(used=12, cap=23, left=11),
            workers=12,
            townhalls=1,
            army={},
            structures={"commandcenter": 1, "supplydepot": 1},
            structures_ready={"commandcenter": 1, "supplydepot": 1},
            structures_pending={},
            known_enemy_units=0,
            game_time_seconds=0,
        )
        payload = {
            "actions": [
                {
                    "type": "conditional",
                    "when": {"condition": "minerals", "at_least": 150},
                    "then_actions": [{"type": "build", "building": "barracks"}],
                    "else_actions": [
                        {
                            "type": "wait_until",
                            "condition": "minerals",
                            "at_least": 150,
                        }
                    ],
                },
                {
                    "type": "conditional",
                    "when": {
                        "match": "all",
                        "conditions": [
                            {
                                "condition": "structure_ready",
                                "target": "barracks",
                                "at_least": 1,
                            },
                            {"condition": "minerals", "at_least": 50},
                        ],
                    },
                    "then_actions": [{"type": "train", "unit": "marine"}],
                    "else_actions": [{"type": "wait", "seconds": 0}],
                },
                {
                    "type": "conditional",
                    "when": {
                        "condition": "minerals",
                        "comparison": "lt",
                        "value": 100,
                    },
                    "then_actions": [{"type": "wait", "seconds": 0}],
                    "else_actions": [
                        {"type": "build", "building": "supply_depot"}
                    ],
                },
            ]
        }

        validator.validate_strategy_plan(
            strategy.strategy_plan_from_dict(payload), state
        )

    def test_fixed_repeat_simulates_every_resource_consuming_cycle(self):
        state = GameStateSummary(
            minerals=100,
            vespene=0,
            supply=SupplySummary(used=12, cap=30, left=18),
            workers=12,
            townhalls=1,
            army={},
            structures={"commandcenter": 1, "barracks": 1},
            structures_ready={"commandcenter": 1, "barracks": 1},
            structures_pending={},
            known_enemy_units=0,
            game_time_seconds=0,
        )
        invalid = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "repeat",
                        "cycles": 3,
                        "actions": [{"type": "train", "unit": "marine"}],
                    }
                ]
            }
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "cycle 3"):
            validator.validate_strategy_plan(invalid, state)

        valid = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "repeat",
                        "cycles": 3,
                        "actions": [
                            {
                                "type": "wait_until",
                                "condition": "minerals",
                                "at_least": 50,
                            },
                            {"type": "train", "unit": "marine"},
                        ],
                    }
                ]
            }
        )
        validator.validate_strategy_plan(valid, state)

    def test_repeat_until_success_refines_facts_for_followup_actions(self):
        terminating = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "repeat_until",
                        "until": {"condition": "minerals", "at_least": 100},
                        "actions": [{"type": "wait", "seconds": 0}],
                        "max_cycles": 2,
                        "on_exhausted": "fail",
                    },
                    {"type": "build", "building": "supply_depot"},
                ]
            }
        )
        validator.validate_strategy_plan(terminating, _known_state(minerals=0))

        continuing = strategy.strategy_plan_from_dict(
            {
                "actions": [
                    {
                        "type": "repeat_until",
                        "until": {"condition": "minerals", "at_least": 100},
                        "actions": [{"type": "wait", "seconds": 0}],
                        "max_cycles": 2,
                        "on_exhausted": "continue",
                    },
                    {"type": "build", "building": "supply_depot"},
                ]
            }
        )
        with self.assertRaisesRegex(validator.PlanValidationError, "minerals"):
            validator.validate_strategy_plan(continuing, _known_state(minerals=0))


class StrategyControlFlowRuntimeTest(unittest.TestCase):
    def test_focus_fire_can_destroy_an_observed_neutral_rock(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "focus_fire",
                        "unit": "marine",
                        "target_unit": "nearest_destructible",
                        "target_alliance": "neutral",
                        "selection": {"count": 1},
                    },
                    {"type": "wait", "seconds": 0},
                ]
            )
            rock = FakeAbilityUnit("DESTRUCTIBLEROCKEX1DIAGONALHUGE", (25, 25))
            bot.destructables = FakeAbilityUnits([rock])
            bot.state = type("State", (), {"dead_units": set(), "alerts": ()})()
            await bot.on_start()

            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 0)
            self.assertEqual(bot.marine_a.issued[-1][1], (rock,))

            bot.destructables.clear()
            bot.state.dead_units.add(rock.tag)
            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot._current_action_index, 1)

        _run_scenario(scenario)

    def test_matchup_alert_and_structure_conditions_honor_live_selection(self):
        async def scenario():
            bot = _make_bot([{"type": "wait", "seconds": 0}])
            bot.enemy_race = type("Race", (), {"name": "Zerg"})()
            bot.state = type(
                "State",
                (),
                {
                    "alerts": (
                        type("Alert", (), {"name": "NuclearLaunchDetected"})(),
                    ),
                    "upgrades": set(),
                },
            )()
            bot.plan = strategy.strategy_plan_from_dict(
                {
                    "actions": [
                        {
                            "type": "wait_until",
                            "condition": "enemy_race",
                            "target": "zerg",
                            "equals": 1,
                        },
                        {
                            "type": "wait_until",
                            "condition": "alert_active",
                            "target": "nuclear_launch_detected",
                            "equals": 1,
                        },
                        {
                            "type": "wait_until",
                            "condition": "structure_ready",
                            "target": "barracks",
                            "equals": 1,
                            "selection": {"tags": [bot.barracks.tag]},
                        },
                        {
                            "type": "wait_until",
                            "condition": "structure_ready",
                            "target": "barracks",
                            "equals": 0,
                            "selection": {"tags": [bot.orbital.tag]},
                        },
                        {
                            "type": "wait_until",
                            "condition": "producer_available",
                            "target": "marine",
                            "equals": 1,
                            "selection": {"tags": [bot.barracks.tag]},
                        },
                        {
                            "type": "wait_until",
                            "condition": "townhall_count",
                            "equals": 1,
                            "selection": {"tags": [bot.orbital.tag]},
                        },
                    ]
                }
            )
            await bot.on_start()
            for iteration in range(1, 7):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual(bot._current_action_index, 6)

        _run_scenario(scenario)

    def test_at_most_wait_and_exact_enemy_count_use_live_observation(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "wait_until",
                        "condition": "supply_left",
                        "at_most": 4,
                    },
                    {
                        "type": "wait_until",
                        "condition": "enemy_unit_count",
                        "target": "mutalisk",
                        "equals": 0,
                    },
                ]
            )
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 0)

            bot.supply_left = 4
            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot._current_action_index, 1)
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual(bot._current_action_index, 2)

        _run_scenario(scenario)

    def test_enemy_count_selection_can_track_an_exact_observed_tag(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "wait_until",
                        "condition": "enemy_unit_count",
                        "target": "zergling",
                        "equals": 1,
                        "selection": {"tags": [1]},
                    }
                ]
            )
            second_enemy = FakeAbilityUnit("ZERGLING", (81, 81))
            bot.enemy_units.append(second_enemy)
            tracked_tag = bot.enemy.tag
            command = bot.plan.actions[0]
            bot.plan = strategy.StrategyPlan(
                actions=(
                    strategy.WaitUntilCommand(
                        condition=command.condition,
                        at_least=command.at_least,
                        comparison=command.comparison,
                        target=command.target,
                        selection=strategy.SelectionSpec(tags=(tracked_tag,)),
                    ),
                )
            )
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 1)

        _run_scenario(scenario)

    def test_typed_ability_form_and_idle_waits_block_until_observed(self):
        async def scenario():
            bot = _make_bot(
                [
                    {"type": "wait_for_ability", "ability": "scan"},
                    {
                        "type": "wait_for_form",
                        "unit": "siege_tank",
                        "form": "siege_tank_sieged",
                    },
                    {"type": "wait_for_idle", "unit": "marine", "count": 2},
                ]
            )
            tank = FakeAbilityUnit("SIEGETANK")
            bot.units.append(tank)
            bot.marine_b.is_idle = False
            await bot.on_start()

            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 1)
            self.assertIn(bot.orbital.tag, bot.available_queries)

            bot.time += 1
            await bot.on_step(2)
            self.assertEqual(bot._current_action_index, 1)
            tank.type_id = type("FakeTypeId", (), {"name": "SIEGETANKSIEGED"})()
            bot.time += 1
            await bot.on_step(3)
            self.assertEqual(bot._current_action_index, 2)

            bot.time += 1
            await bot.on_step(4)
            self.assertEqual(bot._current_action_index, 2)
            bot.marine_b.is_idle = True
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(bot._current_action_index, 3)

        _run_scenario(scenario)

    def test_missing_health_subject_does_not_satisfy_an_upper_bound(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "wait_until",
                        "condition": "unit_health_fraction",
                        "target": "battlecruiser",
                        "at_most": 0.5,
                    }
                ]
            )
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(bot._current_action_index, 0)

        _run_scenario(scenario)

    def test_conditional_selects_exactly_one_branch(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "conditional",
                        "when": {
                            "condition": "enemy_unit_count",
                            "target": "zergling",
                            "at_least": 1,
                        },
                        "then_actions": [
                            {
                                "type": "attack_target",
                                "unit": "marine",
                                "target_unit": "zergling",
                                "selection": {"count": 1},
                            }
                        ],
                        "else_actions": [
                            {
                                "type": "move",
                                "unit": "marine",
                                "location": "retreat",
                                "selection": {"count": 1},
                            }
                        ],
                    }
                ]
            )
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual([order[0] for order in bot.marine_a.issued], ["ATTACK"])

        _run_scenario(scenario)

    def test_repeat_and_repeat_until_are_cycle_bounded(self):
        async def scenario():
            fixed = _make_bot(
                [
                    {
                        "type": "repeat",
                        "cycles": 3,
                        "actions": [
                            {
                                "type": "attack_enemy",
                                "unit": "marine",
                                "selection": {"count": 1},
                            }
                        ],
                    }
                ]
            )
            await fixed.on_start()
            for iteration in range(1, 10):
                fixed.time += 1
                await fixed.on_step(iteration)
            self.assertEqual(
                [order[0] for order in fixed.marine_a.issued],
                ["ATTACK", "ATTACK", "ATTACK"],
            )

            conditional = _make_bot(
                [
                    {
                        "type": "repeat_until",
                        "until": {
                            "condition": "enemy_unit_count",
                            "target": "zergling",
                            "at_most": 0,
                        },
                        "actions": [
                            {
                                "type": "attack_enemy",
                                "unit": "marine",
                                "selection": {"count": 1},
                            }
                        ],
                        "max_cycles": 5,
                    },
                    {"type": "wait", "seconds": 0},
                ]
            )
            await conditional.on_start()
            conditional.time += 1
            await conditional.on_step(1)
            conditional.time += 1
            await conditional.on_step(2)
            conditional.enemy_units.clear()
            conditional.time += 1
            await conditional.on_step(3)
            self.assertEqual(conditional._current_action_index, 1)
            self.assertEqual(
                [order[0] for order in conditional.marine_a.issued], ["ATTACK"]
            )

        _run_scenario(scenario)

    def test_repeat_until_exhaustion_is_terminal_when_requested(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "repeat_until",
                        "until": {"condition": "minerals", "at_least": 9999},
                        "actions": [{"type": "wait", "seconds": 0}],
                        "max_cycles": 1,
                        "on_exhausted": "fail",
                    }
                ]
            )
            await bot.on_start()
            for iteration in range(1, 5):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertTrue(bot.client.left)
            self.assertTrue(bot._left_game)

        _run_scenario(scenario)

    def test_fixed_repeat_time_exhaustion_is_not_reported_as_completion(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "repeat",
                        "cycles": 3,
                        "actions": [{"type": "wait", "seconds": 0}],
                        "max_seconds": 1,
                        "on_exhausted": "fail",
                    },
                    {"type": "wait", "seconds": 0},
                ]
            )
            await bot.on_start()
            for iteration in range(1, 5):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertTrue(bot.client.left)
            self.assertTrue(bot._left_game)
            self.assertLess(bot._current_action_index, bot._plan_action_count())

        _run_scenario(scenario)

    def test_with_timeout_interrupts_a_stalled_child_but_allows_completion(self):
        async def scenario():
            stalled = _make_bot(
                [
                    {
                        "type": "with_timeout",
                        "actions": [
                            {
                                "type": "wait_until",
                                "condition": "minerals",
                                "at_least": 9999,
                            }
                        ],
                        "timeout_seconds": 2,
                        "on_timeout": "fail",
                    }
                ]
            )
            await stalled.on_start()
            for iteration in range(1, 4):
                stalled.time += 1
                await stalled.on_step(iteration)
            self.assertTrue(stalled.client.left)

            completed = _make_bot(
                [
                    {
                        "type": "with_timeout",
                        "actions": [{"type": "wait", "seconds": 0}],
                        "timeout_seconds": 2,
                        "on_timeout": "fail",
                    },
                    {"type": "wait", "seconds": 0},
                ]
            )
            await completed.on_start()
            for iteration in range(1, 5):
                completed.time += 1
                await completed.on_step(iteration)
            self.assertFalse(completed._timeout_scope_states)
            self.assertGreaterEqual(completed._current_action_index, 1)

        _run_scenario(scenario)

    def test_attack_until_clear_requires_arrival_or_visibility_and_stable_clear(self):
        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "attack_until_clear",
                        "unit": "marine",
                        "location": "enemy_main",
                        "radius": 20,
                        "clear_seconds": 2,
                    },
                    {"type": "wait", "seconds": 0},
                ]
            )
            await bot.on_start()
            for iteration in (1, 2):
                bot.time += 1
                await bot.on_step(iteration)
            self.assertEqual(bot._current_action_index, 0)
            self.assertEqual([order[0] for order in bot.marine_a.issued], ["ATTACK"])

            bot.enemy_units.clear()
            bot.enemy_structures.clear()
            bot.marine_a.position = (90, 90)
            bot.marine_b.position = (90, 90)
            for iteration in (3, 4):
                bot.time += 1
                await bot.on_step(iteration)
                self.assertEqual(bot._current_action_index, 0)
            bot.time += 1
            await bot.on_step(5)
            self.assertEqual(bot._current_action_index, 1)

        _run_scenario(scenario)

    def test_background_production_runs_inside_event_loop_and_can_be_stopped(self):
        class ProductionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                original = self.barracks
                self.barracks = CommandUnit("BARRACKS")
                self.structures = FakeAbilityUnits(
                    [
                        self.barracks,
                        *[item for item in self.structures if item is not original],
                    ]
                )

        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 3,
                    },
                    {
                        "type": "repeat",
                        "cycles": 2,
                        "actions": [{"type": "wait", "seconds": 0}],
                    },
                    {"type": "stop_production", "unit": "marine"},
                ],
                bot_base=ProductionBotAI,
            )
            await bot.on_start()
            for iteration in range(1, 12):
                bot.time += 1
                await bot.on_step(iteration)
                if bot.client.left:
                    break
            self.assertEqual(bot.barracks.train_orders, ["MARINE"])
            self.assertEqual(bot._production_policies, [])
            self.assertTrue(bot.client.left)

        _run_scenario(scenario)

    def test_maintain_production_registers_at_target_and_replaces_a_loss(self):
        class ProductionBotAI(AbilityFakeBotAI):
            def __init__(self):
                super().__init__()
                original = self.barracks
                self.barracks = CommandUnit("BARRACKS")
                self.structures = FakeAbilityUnits(
                    [
                        self.barracks,
                        *[item for item in self.structures if item is not original],
                    ]
                )

        async def scenario():
            bot = _make_bot(
                [
                    {
                        "type": "maintain_production",
                        "unit": "marine",
                        "target_count": 2,
                    },
                    {
                        "type": "repeat",
                        "cycles": 3,
                        "actions": [{"type": "wait", "seconds": 0}],
                    },
                    {"type": "stop_production", "unit": "marine"},
                ],
                bot_base=ProductionBotAI,
            )
            await bot.on_start()
            bot.time += 1
            await bot.on_step(1)
            self.assertEqual(len(bot._production_policies), 1)

            bot.units.remove(bot.marine_b)
            for iteration in range(2, 12):
                bot.time += 1
                await bot.on_step(iteration)
                if bot.barracks.train_orders:
                    break
            self.assertEqual(bot.barracks.train_orders, ["MARINE"])

        _run_scenario(scenario)


if __name__ == "__main__":
    unittest.main()
