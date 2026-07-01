import asyncio
import contextlib
import io
import unittest

from starcraft_llm.sc2_bot import create_game_state_bot_class, create_move_unit_bot_class, summarize_bot_state
from starcraft_llm.strategy import GatherMineralsCommand, MoveCommand, StrategyPlan, TrainUnitCommand, WaitCommand


class FakeClient:
    def __init__(self):
        self.game_step = None
        self.left = False

    async def leave(self):
        self.left = True


class FakeTypeId:
    def __init__(self, name):
        self.name = name


class FakeUnit:
    def __init__(self, type_name="SCV"):
        self.type_id = FakeTypeId(type_name)
        self.targets = []
        self.gather_targets = []
        self.trained_units = []

    def move(self, target):
        self.targets.append(target)

    def gather(self, target):
        self.gather_targets.append(target)

    def train(self, unit_type):
        self.trained_units.append(unit_type)


class FakeUnits(list):
    @property
    def ready(self):
        return self

    @property
    def idle(self):
        return self

    @property
    def first(self):
        return self[0]

    def of_type(self, _unit_types):
        return FakeUnits()

    def closest_to(self, _unit):
        return self[0]


class FakeBotAI:
    def __init__(self):
        self.client = FakeClient()
        self.workers = FakeUnits([FakeUnit(), FakeUnit()])
        self.townhalls = FakeUnits([FakeUnit("COMMANDCENTER")])
        self.units = FakeUnits()
        self.enemy_units = FakeUnits()
        self.mineral_field = FakeUnits([FakeUnit("MINERALFIELD")])
        self.minerals = 50
        self.vespene = 0
        self.supply_used = 12
        self.supply_cap = 15
        self.supply_left = 3
        self.time = 7.25

    def can_afford(self, _unit_type):
        return True


class GameStateBotTest(unittest.TestCase):
    def test_summarize_bot_state_returns_resource_supply_and_unit_counts(self):
        bot = FakeBotAI()
        marine = FakeUnit("MARINE")
        bot.units = FakeUnits([bot.workers[0], marine, bot.townhalls[0]])
        bot.enemy_units = FakeUnits([FakeUnit("ZERGLING")])

        summary = summarize_bot_state(bot)

        self.assertEqual(summary.minerals, 50)
        self.assertEqual(summary.vespene, 0)
        self.assertEqual(summary.supply.used, 12)
        self.assertEqual(summary.supply.cap, 15)
        self.assertEqual(summary.supply.left, 3)
        self.assertEqual(summary.workers, 2)
        self.assertEqual(summary.townhalls, 1)
        self.assertEqual(summary.army, {"marine": 1})
        self.assertEqual(summary.known_enemy_units, 1)
        self.assertEqual(summary.game_time_seconds, 7.25)

    def test_game_state_bot_captures_summary_and_leaves(self):
        bot_class = create_game_state_bot_class(FakeBotAI)
        bot = bot_class()

        async def run_once():
            await bot.on_start()
            await bot.on_step(1)

        asyncio.run(run_once())

        self.assertEqual(bot.client.game_step, 2)
        self.assertTrue(bot.client.left)
        self.assertIsNotNone(bot.summary)


class StrategyPlanBotTest(unittest.TestCase):
    def test_bot_executes_gather_and_train_actions(self):
        bot_class = create_move_unit_bot_class(FakeBotAI, lambda point: point)
        plan = StrategyPlan(
            actions=(
                GatherMineralsCommand(unit="worker"),
                TrainUnitCommand(unit="scv"),
            )
        )
        bot = bot_class(plan, stop_after_seconds=0)

        async def run_plan():
            await bot.on_start()
            await bot.on_step(1)
            await bot.on_step(2)
            await bot.on_step(3)

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(run_plan())

        mineral = bot.mineral_field[0]
        self.assertEqual(bot.workers[0].gather_targets, [mineral])
        self.assertEqual(bot.workers[1].gather_targets, [mineral])
        self.assertEqual(len(bot.townhalls[0].trained_units), 1)
        self.assertTrue(bot.client.left)

    def test_bot_observes_state_before_planning(self):
        bot_class = create_move_unit_bot_class(FakeBotAI, lambda point: point)
        bot = bot_class(
            None,
            stop_after_seconds=0,
            strategy="train scv; gather minerals",
            planner_name="rule",
            observe_before_plan=True,
        )

        async def run_plan():
            await bot.on_start()
            await bot.on_step(1)
            await bot.on_step(2)
            await bot.on_step(3)

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(run_plan())

        self.assertIsNotNone(bot.observed_summary)
        self.assertIsNotNone(bot.plan)
        self.assertEqual(len(bot.townhalls[0].trained_units), 1)
        self.assertEqual(bot.workers[0].gather_targets, [bot.mineral_field[0]])
        self.assertTrue(bot.client.left)

    def test_bot_leaves_without_executing_invalid_observed_plan(self):
        bot_class = create_move_unit_bot_class(FakeBotAI, lambda point: point)
        bot = bot_class(
            None,
            stop_after_seconds=0,
            strategy="train scv",
            planner_name="rule",
            observe_before_plan=True,
        )
        bot.minerals = 25

        async def run_plan():
            await bot.on_start()
            await bot.on_step(1)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            asyncio.run(run_plan())

        self.assertIn("Planner error", stderr.getvalue())
        self.assertEqual(bot.townhalls[0].trained_units, [])
        self.assertTrue(bot.client.left)

    def test_bot_executes_move_wait_move_plan_in_order(self):
        bot_class = create_move_unit_bot_class(FakeBotAI, lambda point: point)
        plan = StrategyPlan(
            actions=(
                MoveCommand(unit="worker", x=1, y=2),
                WaitCommand(seconds=0),
                MoveCommand(unit="worker", x=3, y=4),
            )
        )
        bot = bot_class(plan, stop_after_seconds=0)

        async def run_plan():
            await bot.on_start()
            await bot.on_step(1)
            await bot.on_step(2)
            await bot.on_step(3)
            await bot.on_step(4)

        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(run_plan())

        self.assertEqual(bot.client.game_step, 2)
        self.assertTrue(bot.client.left)
        self.assertEqual(bot.workers[0].targets, [(1, 2), (3, 4)])
        self.assertEqual(bot.workers[1].targets, [(1, 2), (3, 4)])


if __name__ == "__main__":
    unittest.main()
