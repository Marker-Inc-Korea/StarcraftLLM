import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starcraft_llm.sc2_bot import build_arg_parser, detect_sc2_environment, main


class Sc2DetectionTest(unittest.TestCase):
    def test_check_mode_argument_parses(self):
        args = build_arg_parser().parse_args(["--check"])

        self.assertTrue(args.check)

    def test_detection_returns_candidate_paths(self):
        env = detect_sc2_environment()

        self.assertTrue(env.candidate_paths)

    def test_sc2path_with_maps_is_runnable_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sc2_path = Path(temp_dir) / "StarCraft II"
            maps_path = sc2_path / "Maps"
            maps_path.mkdir(parents=True)

            with patch.dict(os.environ, {"SC2PATH": str(sc2_path)}):
                env = detect_sc2_environment()

            self.assertTrue(env.installed)
            self.assertEqual(env.detected_path, sc2_path)
            self.assertIn(env.maps_path.name, {"maps", "Maps"})
            self.assertTrue(env.maps_path.exists())
            self.assertTrue(env.maps_installed)

    def test_check_fails_when_app_exists_but_maps_are_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            sc2_path = Path(temp_dir) / "StarCraft II"
            sc2_path.mkdir()

            with patch.dict(os.environ, {"SC2PATH": str(sc2_path)}):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exit_code = main(["--check"])

        self.assertEqual(exit_code, 1)
        self.assertIn("SC2 API maps directory missing", output.getvalue())


if __name__ == "__main__":
    unittest.main()
