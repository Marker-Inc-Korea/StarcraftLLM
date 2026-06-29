import unittest

from starcraft_llm.planner import (
    DEFAULT_PLANNER,
    PlannerUnavailableError,
    RuleBasedPlanner,
    create_planner,
    plan_strategy,
)
from starcraft_llm.strategy import GatherMineralsCommand, TrainUnitCommand


class PlannerInterfaceTest(unittest.TestCase):
    def test_default_planner_is_rule_based(self):
        planner = create_planner(DEFAULT_PLANNER)

        self.assertIsInstance(planner, RuleBasedPlanner)

    def test_rule_planner_creates_strategy_plan(self):
        plan = plan_strategy("gather minerals; train scv")

        self.assertEqual(
            plan.actions,
            (
                GatherMineralsCommand(unit="worker"),
                TrainUnitCommand(unit="scv"),
            ),
        )

    def test_openai_planner_does_not_silently_fallback(self):
        with self.assertRaises(PlannerUnavailableError):
            plan_strategy("train scv", planner_name="openai")

    def test_server_planner_does_not_silently_fallback(self):
        with self.assertRaises(PlannerUnavailableError):
            plan_strategy("train scv", planner_name="server")

    def test_unknown_planner_is_rejected(self):
        with self.assertRaises(ValueError):
            create_planner("unknown")


if __name__ == "__main__":
    unittest.main()
