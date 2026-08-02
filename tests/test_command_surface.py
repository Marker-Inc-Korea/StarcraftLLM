import asyncio
import contextlib
import inspect
import io
import json
import unittest

from starcraft_llm.command_catalog import (
    ADDON_SPECS,
    COMMAND_SURFACE,
    MORPH_SPECS,
    STRUCTURE_SPECS,
    UNIT_SPECS,
    UPGRADE_SPECS,
    resolve_alias,
)
from starcraft_llm.commands import (
    LLM_COMMAND_FUNCTIONS,
    llm_command_function_schemas,
    strategy_plan_from_function_calls,
)
from starcraft_llm.game_state import GameStateSummary, SupplySummary
from starcraft_llm.sc2_bot import create_move_unit_bot_class
from starcraft_llm.strategy import (
    AttackEnemyCommand,
    AttackMoveCommand,
    BuildAddonCommand,
    DistributeWorkersCommand,
    ExpandCommand,
    HoldPositionCommand,
    MorphStructureCommand,
    MoveCommand,
    PatrolCommand,
    RallyCommand,
    RepairCommand,
    ResearchUpgradeCommand,
    StopCommand,
    StrategyPlan,
    TrainUnitCommand,
    WaitUntilCommand,
    parse_strategy_plan,
    parse_strategy_action,
    parse_strategy_request,
    strategy_plan_from_dict,
    strategy_plan_to_dict,
)
from starcraft_llm.validator import PlanValidationError, validate_strategy_plan


class LlmCommandSurfaceTest(unittest.TestCase):
    def test_function_schemas_cover_every_safe_llm_command(self):
        schemas = llm_command_function_schemas()

        schema_names = {schema["name"] for schema in schemas}
        self.assertEqual(schema_names, set(LLM_COMMAND_FUNCTIONS))
        self.assertEqual(
            {command.key for command in COMMAND_SURFACE}, set(LLM_COMMAND_FUNCTIONS)
        )
        self.assertTrue(
            {
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
            }.issubset(schema_names)
        )
        for schema in schemas:
            with self.subTest(command=schema["name"]):
                self.assertEqual(schema["parameters"]["type"], "object")
                self.assertFalse(schema["parameters"]["additionalProperties"])
                self.assertIn("description", schema)

    def test_function_call_adapter_accepts_plain_and_openai_nested_calls(self):
        calls = [
            {"name": "expand", "arguments": json.dumps({"count": 2})},
            {
                "function": {
                    "name": "build_addon",
                    "arguments": {"producer": "factory", "addon": "tech lab"},
                }
            },
            {
                "function": {
                    "name": "research",
                    "arguments": json.dumps({"upgrade": "스팀팩"}),
                }
            },
        ]

        plan = strategy_plan_from_function_calls(calls)

        self.assertEqual(
            plan.actions,
            (
                ExpandCommand(count=2),
                BuildAddonCommand(addon="factory_tech_lab"),
                ResearchUpgradeCommand(upgrade="stimpack"),
            ),
        )

    def test_function_call_adapter_rejects_unknown_function(self):
        with self.assertRaisesRegex(Exception, "unsupported command function"):
            strategy_plan_from_function_calls(
                [{"name": "lift_building", "arguments": {}}]
            )

    def test_command_functions_match_schema_required_arguments(self):
        schemas = {schema["name"]: schema for schema in llm_command_function_schemas()}

        for name, function in LLM_COMMAND_FUNCTIONS.items():
            required = set(schemas[name]["parameters"].get("required", []))
            required_parameters = {
                parameter_name
                for parameter_name, parameter in inspect.signature(
                    function
                ).parameters.items()
                if parameter.default is inspect.Parameter.empty
            }
            self.assertTrue(required_parameters.issubset(required), name)

    def test_every_llm_command_function_constructs_a_strategy_action(self):
        arguments = {
            "move": {"unit": "worker", "x": 35, "y": 42},
            "attack_move": {"unit": "marine", "x": 55, "y": 45},
            "attack_enemy": {"unit": "marine"},
            "patrol": {"unit": "marine", "x": 45, "y": 42},
            "hold_position": {"unit": "marine"},
            "stop": {"unit": "marine"},
            "rally": {"building": "barracks", "x": 45, "y": 42},
            "wait": {"seconds": 1},
            "wait_until": {"condition": "minerals", "at_least": 100},
            "gather": {"resource": "minerals"},
            "distribute_workers": {},
            "train": {"unit": "marine"},
            "build": {"building": "barracks"},
            "expand": {},
            "build_addon": {"addon": "factory_tech_lab"},
            "morph": {"building": "orbital_command"},
            "research": {"upgrade": "stimpack"},
            "repair": {"target": "barracks"},
        }

        self.assertTrue(set(arguments).issubset(LLM_COMMAND_FUNCTIONS))
        for name, payload in arguments.items():
            with self.subTest(command=name):
                self.assertIsNotNone(LLM_COMMAND_FUNCTIONS[name](**payload))

    def test_repair_schema_exposes_only_mechanical_targets(self):
        schemas = {schema["name"]: schema for schema in llm_command_function_schemas()}
        repair_targets = schemas["repair"]["parameters"]["properties"]["target"]["enum"]

        self.assertIn("worker", repair_targets)
        self.assertIn("siege_tank", repair_targets)
        self.assertIn("barracks", repair_targets)
        self.assertNotIn("marine", repair_targets)

    def test_build_addon_schema_exposes_only_unambiguous_canonical_addons(self):
        schemas = {schema["name"]: schema for schema in llm_command_function_schemas()}
        properties = schemas["build_addon"]["parameters"]["properties"]

        self.assertNotIn("producer", properties)
        self.assertIn("factory_tech_lab", properties["addon"]["enum"])
        self.assertNotIn("tech lab", properties["addon"]["enum"])


class CatalogAliasTest(unittest.TestCase):
    def test_resolves_english_and_korean_aliases_to_canonical_keys(self):
        cases = {
            "rax": "barracks",
            "배럭": "barracks",
            "barracks tech lab": "barracks_tech_lab",
            "군수공장 기술실": "factory_tech_lab",
            "siege tank": "siege_tank",
            "공성전차": "siege_tank",
            "orbital": "orbital_command",
            "궤도사령부": "orbital_command",
            "stim": "stimpack",
            "스팀팩": "stimpack",
        }

        for alias, canonical in cases.items():
            with self.subTest(alias=alias):
                self.assertEqual(resolve_alias(alias).key, canonical)

    def test_catalog_resolves_against_installed_burnysc2(self):
        try:
            from sc2.dicts.unit_trained_from import UNIT_TRAINED_FROM
            from sc2.dicts.upgrade_researched_from import UPGRADE_RESEARCHED_FROM
            from sc2.ids.unit_typeid import UnitTypeId
            from sc2.ids.upgrade_id import UpgradeId
        except ImportError:
            self.skipTest("BurnySC2 is not installed in this Python environment")

        for registry in (UNIT_SPECS, STRUCTURE_SPECS, ADDON_SPECS, MORPH_SPECS):
            for key, spec in registry.items():
                with self.subTest(entity=key):
                    self.assertTrue(hasattr(UnitTypeId, spec.enum_name))
        for key, spec in UPGRADE_SPECS.items():
            with self.subTest(upgrade=key):
                upgrade_id = getattr(UpgradeId, spec.enum_name)
                self.assertIn(upgrade_id, UPGRADE_RESEARCHED_FROM)

        producer_types = {
            UnitTypeId.COMMANDCENTER,
            UnitTypeId.ORBITALCOMMAND,
            UnitTypeId.PLANETARYFORTRESS,
            UnitTypeId.BARRACKS,
            UnitTypeId.FACTORY,
            UnitTypeId.STARPORT,
        }
        morph_types = {
            getattr(UnitTypeId, spec.enum_name) for spec in MORPH_SPECS.values()
        }
        trainable_terran = {
            unit_type
            for unit_type, producers in UNIT_TRAINED_FROM.items()
            if set(producers) & producer_types
        } - morph_types
        self.assertEqual(
            {getattr(UnitTypeId, spec.enum_name) for spec in UNIT_SPECS.values()},
            trainable_terran,
        )

        scv_buildings = {
            unit_type
            for unit_type, producers in UNIT_TRAINED_FROM.items()
            if UnitTypeId.SCV in set(producers)
        }
        self.assertEqual(
            {getattr(UnitTypeId, spec.enum_name) for spec in STRUCTURE_SPECS.values()},
            scv_buildings,
        )

        terran_researchers = {
            UnitTypeId.ENGINEERINGBAY,
            UnitTypeId.BARRACKSTECHLAB,
            UnitTypeId.GHOSTACADEMY,
            UnitTypeId.FACTORYTECHLAB,
            UnitTypeId.ARMORY,
            UnitTypeId.STARPORTTECHLAB,
            UnitTypeId.FUSIONCORE,
        }
        researchable_terran = {
            upgrade_id
            for upgrade_id, researcher in UPGRADE_RESEARCHED_FROM.items()
            if researcher in terran_researchers
        }
        self.assertEqual(
            {getattr(UpgradeId, spec.enum_name) for spec in UPGRADE_SPECS.values()},
            researchable_terran,
        )


class StrategyRoundTripTest(unittest.TestCase):
    def test_json_and_dsl_round_trip_for_new_macro_and_control_commands(self):
        dsl_plan = parse_strategy_plan(
            "expand 2; addon factory tech lab; morph orbital command; research stimpack; "
            "distribute workers 3; repair barracks 2; rally factory 44 45; "
            "patrol marine 46 47; hold marine; stop marine"
        )
        json_plan = strategy_plan_from_dict(
            {
                "actions": [
                    {"type": "expand", "count": 2},
                    {"type": "build_addon", "producer": "factory", "addon": "tech lab"},
                    {"type": "morph", "target": "orbital command"},
                    {"type": "research", "upgrade": "stim"},
                    {"type": "distribute_workers", "mineral_to_gas_ratio": 3},
                    {"type": "repair", "target": "barracks", "workers": 2},
                    {"type": "rally", "building": "factory", "x": 44, "y": 45},
                    {"type": "patrol", "unit": "marine", "x": 46, "y": 47},
                    {"type": "hold", "unit": "marine"},
                    {"type": "stop", "unit": "marine"},
                ]
            }
        )

        self.assertEqual(json_plan, dsl_plan)
        self.assertEqual(
            strategy_plan_from_dict(strategy_plan_to_dict(dsl_plan)), dsl_plan
        )

    def test_rejects_biological_repair_target(self):
        with self.assertRaisesRegex(Exception, "cannot repair biological target"):
            parse_strategy_action("repair marine")

    def test_rejects_fractional_action_counts_in_json(self):
        with self.assertRaisesRegex(Exception, "must be an integer"):
            strategy_plan_from_dict([{"type": "train", "unit": "marine", "count": 1.5}])

    def test_natural_language_attack_resolves_advanced_unit_alias(self):
        self.assertEqual(
            parse_strategy_request("공성 전차로 공격해").actions,
            (AttackMoveCommand(unit="siege_tank", x=55.0, y=45.0),),
        )

    def test_natural_language_morph_accepts_korean_upgrade_word(self):
        self.assertEqual(
            parse_strategy_request("궤도 사령부로 업그레이드").actions,
            (MorphStructureCommand(building="orbital_command"),),
        )

    def test_dsl_control_commands_accept_multiword_unit_aliases(self):
        plan = parse_strategy_plan("move siege tank 40 42; attack siege tank enemy")

        self.assertEqual(
            plan.actions,
            (
                MoveCommand(unit="siege_tank", x=40.0, y=42.0),
                AttackEnemyCommand(unit="siege_tank"),
            ),
        )


class ValidatorCommandSurfaceTest(unittest.TestCase):
    def test_simulates_factory_addon_then_siege_tank_prerequisites(self):
        plan = StrategyPlan(
            actions=(
                BuildAddonCommand(addon="factory_tech_lab"),
                WaitUntilCommand(
                    condition="structure_ready", target="factory_tech_lab", at_least=1
                ),
                ResearchUpgradeCommand(upgrade="infernal_pre_igniter"),
                TrainUnitCommand(unit="siege_tank"),
            )
        )

        self.assertIs(
            validate_strategy_plan(
                plan,
                _state(
                    minerals=300,
                    vespene=250,
                    supply_left=6,
                    structures={"commandcenter": 1, "factory": 1},
                    structures_ready={"commandcenter": 1, "factory": 1},
                ),
            ),
            plan,
        )

    def test_rejects_siege_tank_without_factory_tech_lab(self):
        plan = StrategyPlan(actions=(TrainUnitCommand(unit="siege_tank"),))

        with self.assertRaisesRegex(PlanValidationError, "factory_tech_lab"):
            validate_strategy_plan(
                plan,
                _state(
                    minerals=200,
                    vespene=200,
                    supply_left=6,
                    structures={"commandcenter": 1, "factory": 1},
                    structures_ready={"commandcenter": 1, "factory": 1},
                ),
            )

    def test_simulates_upgrade_prerequisites_and_spent_resources(self):
        plan = StrategyPlan(
            actions=(
                ResearchUpgradeCommand(upgrade="terran_infantry_weapons_level_1"),
                ResearchUpgradeCommand(upgrade="terran_vehicle_weapons_level_1"),
            )
        )

        with self.assertRaisesRegex(PlanValidationError, "only 0 minerals"):
            validate_strategy_plan(
                plan,
                _state(
                    minerals=100,
                    vespene=100,
                    structures={"commandcenter": 1, "engineering_bay": 1, "armory": 1},
                    structures_ready={
                        "commandcenter": 1,
                        "engineering_bay": 1,
                        "armory": 1,
                    },
                ),
            )

    def test_rejects_level_two_upgrade_until_previous_upgrade_completed(self):
        plan = StrategyPlan(
            actions=(ResearchUpgradeCommand(upgrade="terran_infantry_weapons_level_2"),)
        )

        with self.assertRaisesRegex(PlanValidationError, "level_1 completes"):
            validate_strategy_plan(
                plan,
                _state(
                    minerals=200,
                    vespene=200,
                    structures={"commandcenter": 1, "engineering_bay": 1, "armory": 1},
                    structures_ready={
                        "commandcenter": 1,
                        "engineering_bay": 1,
                        "armory": 1,
                    },
                ),
            )

    def test_rejects_direct_biological_repair_command(self):
        plan = StrategyPlan(actions=(RepairCommand(target="marine"),))

        with self.assertRaisesRegex(PlanValidationError, "unsupported repair target"):
            validate_strategy_plan(plan, _state())

    def test_existing_orbital_does_not_hide_a_free_command_center(self):
        plan = StrategyPlan(
            actions=(MorphStructureCommand(building="orbital_command"),)
        )

        self.assertIs(
            validate_strategy_plan(
                plan,
                _state(
                    minerals=200,
                    structures={"commandcenter": 1, "orbitalcommand": 1, "barracks": 1},
                    structures_ready={
                        "commandcenter": 1,
                        "orbitalcommand": 1,
                        "barracks": 1,
                    },
                ),
            ),
            plan,
        )


class FakeTypeId:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return getattr(other, "name", other) == self.name

    def __hash__(self):
        return hash(self.name)


def _enum_name(value):
    return getattr(value, "name", value)


class FakeUnit:
    _next_tag = 1

    def __init__(
        self, type_name="SCV", is_ready=True, damaged=False, has_techlab=False
    ):
        self.type_id = FakeTypeId(type_name)
        self.tag = FakeUnit._next_tag
        FakeUnit._next_tag += 1
        self.position = (35, 42)
        self.is_ready = is_ready
        self.build_progress = 1.0 if is_ready else 0.25
        self.has_techlab = has_techlab
        self.has_reactor = False
        self.has_add_on = has_techlab
        self.add_on_tag = self.tag + 10_000 if has_techlab else 0
        self.health = 40 if damaged else 100
        self.health_max = 100
        self.trained_units = []
        self.build_orders = []
        self.research_orders = []
        self.repair_targets = []
        self.rally_orders = []
        self.patrol_targets = []
        self.hold_count = 0
        self.stop_count = 0

    @property
    def health_percentage(self):
        return self.health / self.health_max

    def train(self, unit_type):
        self.trained_units.append(_enum_name(unit_type))
        return True

    def build(self, unit_type, target=None):
        type_name = _enum_name(unit_type)
        self.build_orders.append((type_name, target))
        if type_name == "FACTORYTECHLAB":
            self.has_techlab = True
            self.has_add_on = True
            self.add_on_tag = self.tag + 10_000
        return True

    def research(self, upgrade_type):
        self.research_orders.append(_enum_name(upgrade_type))
        return True

    def repair(self, target):
        self.repair_targets.append(target)

    def patrol(self, target):
        self.patrol_targets.append(target)

    def hold_position(self):
        self.hold_count += 1

    def stop(self):
        self.stop_count += 1

    def __call__(self, ability, target):
        self.rally_orders.append((_enum_name(ability), target))
        return True


class FakeUnits(list):
    @property
    def ready(self):
        return FakeUnits([unit for unit in self if getattr(unit, "is_ready", True)])

    @property
    def idle(self):
        return self

    @property
    def first(self):
        return self[0]

    def of_type(self, unit_types):
        return FakeUnits(
            [
                unit
                for unit in self
                if unit.type_id in unit_types or unit.type_id.name in unit_types
            ]
        )

    def closest_to(self, _unit):
        return self[0]


class FakeClient:
    def __init__(self):
        self.game_step = None
        self.left = False

    async def leave(self):
        self.left = True


class AdvancedFakeBotAI:
    def __init__(self):
        self.client = FakeClient()
        self.workers = FakeUnits([FakeUnit(), FakeUnit(), FakeUnit()])
        self.townhalls = FakeUnits([FakeUnit("COMMANDCENTER")])
        self.factory = FakeUnit("FACTORY")
        self.barracks = FakeUnit("BARRACKS", damaged=True)
        self.structures = FakeUnits([self.townhalls[0], self.factory, self.barracks])
        self.siege_tank = FakeUnit("SIEGETANK")
        self.marine = FakeUnit("MARINE")
        self.units = FakeUnits([self.siege_tank, self.marine])
        self.enemy_units = FakeUnits()
        self.mineral_field = FakeUnits([FakeUnit("MINERALFIELD")])
        self.vespene_geyser = FakeUnits([FakeUnit("VESPENEGEYSER")])
        self.minerals = 1000
        self.vespene = 1000
        self.supply_used = 50
        self.supply_cap = 100
        self.supply_left = 50
        self.time = 1.0
        self.distribution_ratios = []
        self.expansion_orders = []
        self.research_orders = []
        self.state = type("State", (), {"upgrades": set()})()

    def can_afford(self, _item):
        return True

    async def build(self, unit_type, near, max_distance=20, **_kwargs):
        type_name = _enum_name(unit_type)
        if type_name == "COMMANDCENTER":
            self.expansion_orders.append(type_name)
            self.townhalls.append(FakeUnit(type_name, is_ready=False))
        else:
            self.structures.append(FakeUnit(type_name, is_ready=False))
        return True

    async def get_next_expansion(self):
        return (60, 60)

    async def distribute_workers(self, resource_ratio):
        self.distribution_ratios.append(resource_ratio)

    def research(self, upgrade_type):
        self.research_orders.append(_enum_name(upgrade_type))
        self.state.upgrades.add(upgrade_type)
        return True


class NoExpansionFakeBotAI(AdvancedFakeBotAI):
    async def get_next_expansion(self):
        return None


class Sc2ExecutorCommandSurfaceTest(unittest.TestCase):
    def test_executor_retries_when_no_expansion_location_is_available(self):
        bot_class = create_move_unit_bot_class(
            NoExpansionFakeBotAI, lambda point: point
        )
        bot = bot_class(StrategyPlan(actions=(ExpandCommand(),)), stop_after_seconds=0)

        async def run_once():
            await bot.on_start()
            await bot.on_step(1)

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(run_once())

        self.assertEqual(bot.expansion_orders, [])
        self.assertEqual(bot._current_action_index, 0)

    def test_fake_executor_runs_expansion_addon_advanced_unit_research_distribution_repair_and_control(
        self,
    ):
        bot_class = create_move_unit_bot_class(AdvancedFakeBotAI, lambda point: point)
        plan = StrategyPlan(
            actions=(
                ExpandCommand(count=1),
                BuildAddonCommand(addon="factory_tech_lab"),
                TrainUnitCommand(unit="siege_tank"),
                ResearchUpgradeCommand(upgrade="infernal_pre_igniter"),
                DistributeWorkersCommand(mineral_to_gas_ratio=2.5),
                RepairCommand(target="barracks", workers=2),
                RallyCommand(building="factory", x=44, y=45),
                PatrolCommand(unit="siege_tank", x=46, y=47),
                HoldPositionCommand(unit="siege_tank"),
                StopCommand(unit="siege_tank"),
                MorphStructureCommand(building="orbital_command"),
            )
        )
        bot = bot_class(plan, stop_after_seconds=0)

        async def run_plan():
            await bot.on_start()
            for iteration in range(1, 20):
                await bot.on_step(iteration)

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(run_plan())

        self.assertEqual(bot.expansion_orders, ["COMMANDCENTER"])
        self.assertEqual(bot.factory.build_orders[0], ("FACTORYTECHLAB", None))
        self.assertEqual(bot.factory.trained_units, ["SIEGETANK"])
        self.assertEqual(bot.research_orders, ["HIGHCAPACITYBARRELS"])
        self.assertEqual(bot.distribution_ratios, [2.5])
        self.assertEqual(bot.workers[0].repair_targets, [bot.barracks])
        self.assertEqual(bot.workers[1].repair_targets, [bot.barracks])
        self.assertEqual(bot.factory.rally_orders, [("RALLY_BUILDING", (44, 45))])
        self.assertEqual(bot.siege_tank.patrol_targets, [(46, 47)])
        self.assertEqual(bot.siege_tank.hold_count, 1)
        self.assertEqual(bot.siege_tank.stop_count, 1)
        self.assertEqual(bot.townhalls[0].build_orders, [("ORBITALCOMMAND", None)])
        self.assertTrue(bot.client.left)


def _state(
    minerals=1000,
    vespene=1000,
    supply_left=20,
    workers=12,
    townhalls=1,
    structures=None,
    structures_ready=None,
    upgrades=(),
):
    structures = structures or {"commandcenter": 1}
    return GameStateSummary(
        minerals=minerals,
        vespene=vespene,
        supply=SupplySummary(used=20, cap=20 + supply_left, left=supply_left),
        workers=workers,
        townhalls=townhalls,
        army={},
        structures=structures,
        structures_ready=structures_ready
        if structures_ready is not None
        else structures,
        structures_pending={},
        upgrades=tuple(upgrades),
        known_enemy_units=0,
        game_time_seconds=0.0,
    )


if __name__ == "__main__":
    unittest.main()
