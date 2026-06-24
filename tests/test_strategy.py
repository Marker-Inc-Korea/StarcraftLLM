import unittest

from starcraft_llm.strategy import StrategyParseError, parse_strategy


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

    def test_rejects_unsupported_command(self):
        with self.assertRaises(StrategyParseError):
            parse_strategy("attack worker 35 42")


if __name__ == "__main__":
    unittest.main()
