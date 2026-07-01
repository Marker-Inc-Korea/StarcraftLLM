import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starcraft_llm.game_state import GameStateSummary, SupplySummary
from starcraft_llm.planner import (
    DEFAULT_PLANNER,
    GeminiPlanner,
    PlannerUnavailableError,
    RuleBasedPlanner,
    create_planner,
    load_gemini_api_key,
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

    def test_gemini_planner_creates_strategy_plan_from_api_json(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["url"] = url
            captured["headers"] = headers
            captured["payload"] = payload
            captured["timeout"] = timeout
            return {
                "output_text": json.dumps(
                    {
                        "actions": [
                            {"type": "gather", "unit": "worker", "resource": "minerals"},
                            {"type": "train", "unit": "scv"},
                        ]
                    }
                )
            }

        planner = GeminiPlanner(api_key="test-key", http_post=fake_post)
        plan = planner.create_plan(_request("early economy"))

        self.assertEqual(
            plan.actions,
            (
                GatherMineralsCommand(unit="worker"),
                TrainUnitCommand(unit="scv"),
            ),
        )
        self.assertIn("/interactions", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(captured["payload"]["model"], "gemini-2.5-flash")
        self.assertEqual(captured["payload"]["response_format"]["mime_type"], "application/json")
        self.assertIn("early economy", captured["payload"]["input"])

    def test_gemini_planner_accepts_plan_alias_from_model_output(self):
        def fake_post(url, headers, payload, timeout):
            return {
                "output_text": json.dumps(
                    {"plan": [{"type": "gather", "unit": "worker", "resource": "minerals"}]}
                )
            }

        planner = GeminiPlanner(api_key="test-key", http_post=fake_post)
        plan = planner.create_plan(_request("early economy"))

        self.assertEqual(plan.actions, (GatherMineralsCommand(unit="worker"),))

    def test_gemini_planner_extracts_rest_step_model_output(self):
        def fake_post(url, headers, payload, timeout):
            return {
                "steps": [
                    {
                        "type": "model_output",
                        "content": [
                            {"type": "text", "text": '{"actions":[{"type":"train","unit":"scv"}]}'},
                        ],
                    }
                ]
            }

        planner = GeminiPlanner(api_key="test-key", http_post=fake_post)
        plan = planner.create_plan(_request("train scv"))

        self.assertEqual(plan.actions, (TrainUnitCommand(unit="scv"),))


    def test_gemini_planner_prompt_includes_game_state(self):
        captured = {}

        def fake_post(url, headers, payload, timeout):
            captured["payload"] = payload
            return {"output_text": '{"actions":[{"type":"gather","unit":"worker","resource":"minerals"}]}'}

        planner = GeminiPlanner(api_key="test-key", http_post=fake_post)
        planner.create_plan(
            _request(
                "경제를 키워",
                game_state=GameStateSummary(
                    minerals=50,
                    vespene=0,
                    supply=SupplySummary(used=8, cap=13, left=5),
                    workers=8,
                    townhalls=1,
                    army={},
                    known_enemy_units=0,
                    game_time_seconds=0.0,
                ),
            )
        )

        self.assertIn('"minerals": 50', captured["payload"]["input"])
        self.assertIn('"workers": 8', captured["payload"]["input"])

    def test_gemini_api_key_prefers_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            key_file = Path(temp_dir) / "gemini_api_key.txt"
            key_file.write_text("file-key", encoding="utf-8")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=False):
                self.assertEqual(load_gemini_api_key(key_file), "env-key")

    def test_gemini_api_key_reads_ignored_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            key_file = Path(temp_dir) / "gemini_api_key.txt"
            key_file.write_text("file-key\n", encoding="utf-8")
            with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
                self.assertEqual(load_gemini_api_key(key_file), "file-key")

    def test_gemini_api_key_missing_is_configuration_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_key_file = Path(temp_dir) / "missing.txt"
            with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
                with self.assertRaises(PlannerUnavailableError):
                    load_gemini_api_key(missing_key_file)

    def test_openai_planner_does_not_silently_fallback(self):
        with self.assertRaises(PlannerUnavailableError):
            plan_strategy("train scv", planner_name="openai")

    def test_server_planner_does_not_silently_fallback(self):
        with self.assertRaises(PlannerUnavailableError):
            plan_strategy("train scv", planner_name="server")

    def test_unknown_planner_is_rejected(self):
        with self.assertRaises(ValueError):
            create_planner("unknown")


def _request(strategy, game_state=None):
    from starcraft_llm.planner import PlannerRequest

    return PlannerRequest(strategy=strategy, game_state=game_state)


if __name__ == "__main__":
    unittest.main()
