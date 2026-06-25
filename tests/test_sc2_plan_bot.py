import asyncio
import contextlib
import io
import unittest

from starcraft_llm.sc2_bot import create_move_unit_bot_class
from starcraft_llm.strategy import MoveCommand, StrategyPlan, WaitCommand


class FakeClient:
    def __init__(self):
        self.game_step = None
        self.left = False

    async def leave(self):
        self.left = True


class FakeUnit:
    def __init__(self):
        self.targets = []

    def move(self, target):
        self.targets.append(target)


class FakeUnits(list):
    def of_type(self, _unit_types):
        return FakeUnits()


class FakeBotAI:
    def __init__(self):
        self.client = FakeClient()
        self.workers = FakeUnits([FakeUnit(), FakeUnit()])
        self.units = FakeUnits()


class StrategyPlanBotTest(unittest.TestCase):
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
