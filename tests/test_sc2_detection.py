import unittest

from starcraft_llm.sc2_bot import build_arg_parser, detect_sc2_environment


class Sc2DetectionTest(unittest.TestCase):
    def test_check_mode_argument_parses(self):
        args = build_arg_parser().parse_args(["--check"])

        self.assertTrue(args.check)

    def test_detection_returns_candidate_paths(self):
        env = detect_sc2_environment()

        self.assertTrue(env.candidate_paths)


if __name__ == "__main__":
    unittest.main()
