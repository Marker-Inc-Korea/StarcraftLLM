import json
import unittest

from starcraft_llm.game_state import (
    GameStateSummary,
    SupplySummary,
    game_state_summary_to_dict,
    game_state_summary_to_json,
)


class GameStateSummaryTest(unittest.TestCase):
    def test_summary_serializes_to_llm_friendly_dict(self):
        summary = GameStateSummary(
            minerals=50,
            vespene=0,
            supply=SupplySummary(used=12, cap=15, left=3),
            workers=12,
            townhalls=1,
            army={"marine": 2},
            known_enemy_units=0,
            game_time_seconds=3.5,
        )

        self.assertEqual(
            game_state_summary_to_dict(summary),
            {
                "minerals": 50,
                "vespene": 0,
                "supply": {"used": 12, "cap": 15, "left": 3},
                "workers": 12,
                "townhalls": 1,
                "army": {"marine": 2},
                "known_enemy_units": 0,
                "game_time_seconds": 3.5,
            },
        )

    def test_summary_json_round_trips(self):
        summary = GameStateSummary(
            minerals=50,
            vespene=0,
            supply=SupplySummary(used=12, cap=15, left=3),
            workers=12,
            townhalls=1,
            army={},
            known_enemy_units=0,
            game_time_seconds=0,
        )

        self.assertEqual(json.loads(game_state_summary_to_json(summary))["minerals"], 50)


if __name__ == "__main__":
    unittest.main()
