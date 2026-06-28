import json
import unittest

from starcraft_llm.strategy import (
    GatherMineralsCommand,
    MoveCommand,
    StrategyParseError,
    StrategyPlan,
    TrainUnitCommand,
    WaitCommand,
    parse_strategy,
    parse_strategy_plan,
    parse_strategy_plan_json,
    parse_strategy_request,
    strategy_plan_from_dict,
    strategy_plan_to_dict,
    strategy_plan_to_json,
    translate_strategy_intent,
)


class StrategyParserTest(unittest.TestCase):
    def test_parse_explicit_worker_move(self):
        command = parse_strategy("move worker 35 42")

        self.assertEqual(command.unit, "worker")
        self.assertEqual(command.x, 35)
        self.assertEqual(command.y, 42)

    def test_parse_short_move_uses_worker_default(self):
        command = parse_strategy("move 10.5 22")

        self.assertEqual(command.unit, "worker")
        self.assertEqual(command.x, 10.5)
        self.assertEqual(command.y, 22)

    def test_parse_marine_alias(self):
        command = parse_strategy("move marine 12 18")

        self.assertEqual(command.unit, "marine")
        self.assertEqual(command.x, 12)
        self.assertEqual(command.y, 18)

    def test_parse_plan_with_move_wait_move_steps(self):
        plan = parse_strategy_plan("move worker 35 42; wait 1.5; move worker 45 42")

        self.assertEqual(
            plan.actions,
            (
                MoveCommand(unit="worker", x=35, y=42),
                WaitCommand(seconds=1.5),
                MoveCommand(unit="worker", x=45, y=42),
            ),
        )

    def test_parse_plan_accepts_then_separator(self):
        plan = parse_strategy_plan("move 10 20 then wait 0 then move scv 11 21")

        self.assertEqual(len(plan.actions), 3)
        self.assertEqual(plan.actions[0], MoveCommand(unit="worker", x=10, y=20))
        self.assertEqual(plan.actions[1], WaitCommand(seconds=0))
        self.assertEqual(plan.actions[2], MoveCommand(unit="worker", x=11, y=21))

    def test_parse_gather_and_train_actions(self):
        plan = parse_strategy_plan("gather minerals; train scv")

        self.assertEqual(
            plan.actions,
            (
                GatherMineralsCommand(unit="worker"),
                TrainUnitCommand(unit="scv"),
            ),
        )

    def test_parse_json_strategy_plan(self):
        plan = parse_strategy_plan_json(
            '{"actions":[{"type":"move","unit":"worker","x":35,"y":42},{"type":"wait","seconds":1},{"type":"gather","unit":"worker","resource":"minerals"},{"type":"train","unit":"scv"}]}'
        )

        self.assertEqual(
            plan,
            StrategyPlan(
                actions=(
                    MoveCommand(unit="worker", x=35, y=42),
                    WaitCommand(seconds=1),
                    GatherMineralsCommand(unit="worker"),
                    TrainUnitCommand(unit="scv"),
                )
            ),
        )

    def test_parse_json_action_array_shortcut(self):
        plan = parse_strategy_plan_json('[{"type":"move","x":35,"y":42}]')

        self.assertEqual(plan.actions, (MoveCommand(unit="worker", x=35, y=42),))

    def test_strategy_plan_json_round_trip(self):
        plan = StrategyPlan(
            actions=(
                MoveCommand(unit="worker", x=35, y=42),
                WaitCommand(seconds=1),
                GatherMineralsCommand(unit="worker"),
                TrainUnitCommand(unit="scv"),
            )
        )

        encoded = strategy_plan_to_json(plan)
        decoded = strategy_plan_from_dict(json.loads(encoded))

        self.assertEqual(decoded, plan)
        self.assertEqual(
            strategy_plan_to_dict(plan),
            {
                "actions": [
                    {"type": "move", "unit": "worker", "x": 35, "y": 42},
                    {"type": "wait", "seconds": 1},
                    {"type": "gather", "unit": "worker", "resource": "minerals"},
                    {"type": "train", "unit": "scv"},
                ]
            },
        )

    def test_parse_strategy_request_accepts_json(self):
        plan = parse_strategy_request('{"actions":[{"type":"wait","seconds":0}]}')

        self.assertEqual(plan.actions, (WaitCommand(seconds=0),))

    def test_translate_korean_scout_intent(self):
        plan = translate_strategy_intent("일꾼으로 정찰해")

        self.assertEqual(
            plan.actions,
            (
                MoveCommand(unit="worker", x=35, y=42),
                WaitCommand(seconds=1),
                MoveCommand(unit="worker", x=45, y=42),
                WaitCommand(seconds=1),
                MoveCommand(unit="worker", x=55, y=45),
            ),
        )

    def test_parse_strategy_request_falls_back_to_intent(self):
        plan = parse_strategy_request("scout with worker")

        self.assertIsInstance(plan.actions[0], MoveCommand)
        self.assertGreater(len(plan.actions), 1)

    def test_translate_marine_rally_intent(self):
        plan = translate_strategy_intent("마린 전진")

        self.assertEqual(plan.actions, (MoveCommand(unit="marine", x=35, y=42),))

    def test_single_command_parser_rejects_wait(self):
        with self.assertRaises(StrategyParseError):
            parse_strategy("wait 1")

    def test_rejects_unsupported_command(self):
        with self.assertRaises(StrategyParseError):
            parse_strategy("attack worker 35 42")

    def test_rejects_negative_wait(self):
        with self.assertRaises(StrategyParseError):
            parse_strategy_plan("wait -1")

    def test_parse_strategy_request_accepts_resource_and_train_intents(self):
        self.assertEqual(parse_strategy_request("미네랄 캐").actions, (GatherMineralsCommand(unit="worker"),))
        self.assertEqual(parse_strategy_request("train scv").actions, (TrainUnitCommand(unit="scv"),))

    def test_rejects_unknown_json_action(self):
        with self.assertRaises(StrategyParseError):
            parse_strategy_plan_json('{"actions":[{"type":"attack"}]}')

    def test_rejects_unknown_intent(self):
        with self.assertRaises(StrategyParseError):
            translate_strategy_intent("win the game")


if __name__ == "__main__":
    unittest.main()
