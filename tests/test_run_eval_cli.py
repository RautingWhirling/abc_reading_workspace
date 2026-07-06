from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import run_eval


class RunEvalCliTest(unittest.TestCase):
    def test_arg_parser_accepts_image_inputs(self) -> None:
        parser = run_eval.build_arg_parser()
        args = parser.parse_args(["--image", "eval/image/4.png", "--event-limit", "1"])

        self.assertEqual(args.image, Path("eval/image/4.png"))
        self.assertIsNone(args.image_dir)
        self.assertEqual(args.event_limit, 1)

    def test_main_routes_image_mode_through_image_loader_and_dict_evaluation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "eval" / "image" / "4.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            input_path = root / "eval" / "hot_event_opinion_variants.json"
            input_path.write_text("[]", encoding="utf-8")

            with patch.object(
                run_eval,
                "load_hot_events",
                return_value=[{"event_id": "hot_event_005"}],
            ) as load_hot_events_mock, patch.object(
                run_eval,
                "load_image_events",
                return_value=[
                    {
                        "event_id": "hot_event_005",
                        "event_title": "Image event",
                        "source_type": "image",
                        "source_image": str(image),
                    }
                ],
            ) as load_image_events_mock, patch.object(
                run_eval,
                "run_hot_event_dict_evaluations",
                return_value=[
                    (
                        root / "eval" / "output" / "hot_event_005_strategy_output.json",
                        {"事件名称": "Image event"},
                    )
                ],
            ) as run_dict_mock, patch(
                "sys.argv",
                [
                    "run_eval.py",
                    "--workspace-root",
                    str(root),
                    "--input",
                    str(input_path),
                    "--image",
                    str(image),
                    "--disable-llm",
                ],
            ):
                exit_code = run_eval.main()

            self.assertEqual(exit_code, 0)
            load_hot_events_mock.assert_called_once_with(input_path.resolve())
            load_image_events_mock.assert_called_once()
            run_dict_mock.assert_called_once()
            self.assertFalse(run_dict_mock.call_args.kwargs["use_llm"])


if __name__ == "__main__":
    unittest.main()
