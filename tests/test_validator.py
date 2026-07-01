import unittest

from starcraft_llm.game_state import GameStateSummary, SupplySummary
from starcraft_llm.strategy import GatherMineralsCommand, MoveCommand, StrategyPlan, TrainUnitCommand, WaitCommand
from starcraft_llm.validator import PlanValidationError, validate_strategy_plan


class PlanValidatorTest(unittest.TestCase):
    def test_accepts_feasible_economy_plan_with_game_state(self):
        plan = StrategyPlan(
            actions=(
                TrainUnitCommand(unit="scv"),
                GatherMineralsCommand(unit="worker"),
            )
        )

        self.assertIs(validate_strategy_plan(plan, _state()), plan)

    def test_rejects_train_scv_without_enough_minerals(self):
        plan = StrategyPlan(actions=(TrainUnitCommand(unit="scv"),))

        with self.assertRaisesRegex(PlanValidationError, "only 25 minerals"):
            validate_strategy_plan(plan, _state(minerals=25))

    def test_rejects_train_scv_when_supply_blocked(self):
        plan = StrategyPlan(actions=(TrainUnitCommand(unit="scv"),))

        with self.assertRaisesRegex(PlanValidationError, "no supply"):
            validate_strategy_plan(plan, _state(supply_left=0))

    def test_rejects_too_many_trains_after_resource_simulation(self):
        plan = StrategyPlan(
            actions=(
                TrainUnitCommand(unit="scv"),
                TrainUnitCommand(unit="scv"),
            )
        )

        with self.assertRaisesRegex(PlanValidationError, "only 0 minerals"):
            validate_strategy_plan(plan, _state(minerals=50, supply_left=2))

    def test_rejects_gather_without_workers(self):
        plan = StrategyPlan(actions=(GatherMineralsCommand(unit="worker"),))

        with self.assertRaisesRegex(PlanValidationError, "no workers"):
            validate_strategy_plan(plan, _state(workers=0))

    def test_rejects_unsafe_move_coordinate(self):
        plan = StrategyPlan(actions=(MoveCommand(unit="worker", x=999, y=42),))

        with self.assertRaisesRegex(PlanValidationError, "outside"):
            validate_strategy_plan(plan)

    def test_rejects_too_long_wait(self):
        plan = StrategyPlan(actions=(WaitCommand(seconds=60),))

        with self.assertRaisesRegex(PlanValidationError, "too long"):
            validate_strategy_plan(plan)

    def test_rejects_too_many_actions(self):
        plan = StrategyPlan(actions=tuple(WaitCommand(seconds=0) for _ in range(9)))

        with self.assertRaisesRegex(PlanValidationError, "too many"):
            validate_strategy_plan(plan)


def _state(minerals=50, supply_left=5, workers=8, townhalls=1):
    return GameStateSummary(
        minerals=minerals,
        vespene=0,
        supply=SupplySummary(used=8, cap=13, left=supply_left),
        workers=workers,
        townhalls=townhalls,
        army={},
        known_enemy_units=0,
        game_time_seconds=0.0,
    )


if __name__ == "__main__":
    unittest.main()
