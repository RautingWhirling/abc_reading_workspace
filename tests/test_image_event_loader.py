from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.image_event_loader import (
    HotEventMatcher,
    ImageEventRecognizer,
    ImageRecognitionResult,
    load_image_events,
)


class FakeVisionClient:
    provider = "dashscope"
    model = "qwen3.7-plus"
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def generate_json_with_image(self, *, system_prompt: str, user_prompt: str, image_path: Path) -> dict:
        return {
            "event_title": "高端AI芯片供应与出口限制牵动产业链",
            "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
            "domain": "technology",
            "keywords": ["AI芯片", "出口管制", "替代方案", "本土化供应"],
            "target": "说明高端AI芯片供应限制对模型训练、算力成本和产业链竞争的影响。",
            "opinion_variants": ["芯片供应紧张推动企业评估替代方案。"],
            "confidence": 0.91,
        }


class ImageEventLoaderTest(unittest.TestCase):
    def test_recognizer_normalizes_vision_client_response(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "4.png"
            image_path.write_bytes(b"image")

            result = ImageEventRecognizer().recognize(
                image_path=image_path,
                workspace_root=tmpdir,
                use_llm=True,
                vision_client=FakeVisionClient(),
            )

            self.assertEqual(result.event_title, "高端AI芯片供应与出口限制牵动产业链")
            self.assertEqual(result.domain, "technology")
            self.assertEqual(result.method, "vision_llm")
            self.assertEqual(result.confidence, 0.91)
            self.assertEqual(result.source_image, str(image_path))
            self.assertFalse(result.fallback_used)

    def test_matcher_reuses_existing_hot_event_above_threshold(self) -> None:
        recognition = ImageRecognitionResult(
            source_image="eval/image/4.png",
            method="vision_llm",
            event_title="高端AI芯片供应与出口限制牵动产业链",
            event_summary="高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
            domain="technology",
            keywords=["AI芯片", "出口管制", "替代方案", "本土化供应"],
            target="说明芯片供应限制影响。",
            opinion_variants=["芯片供应紧张。"],
            confidence=0.91,
        )
        reference_events = [
            {
                "event_id": "hot_event_005",
                "domain": "technology",
                "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                "event_summary": "高端算力芯片供需紧张，出口管制、替代方案和本土化供应成为科技企业关注重点。",
                "target": "说明高端AI芯片供应限制对模型训练、算力成本、产业竞争和本土替代的影响。",
                "opinion_variants": ["高端AI芯片越来越像战略资源。"],
            }
        ]

        event, metadata = HotEventMatcher(match_threshold=0.72).match(recognition, reference_events)

        self.assertEqual(event["event_id"], "hot_event_005")
        self.assertEqual(event["source_type"], "image")
        self.assertEqual(event["source_image"], "eval/image/4.png")
        self.assertEqual(event["image_recognition"]["matched_event_id"], "hot_event_005")
        self.assertGreaterEqual(metadata["match_score"], 0.72)

    def test_matcher_generates_image_event_below_threshold(self) -> None:
        recognition = ImageRecognitionResult(
            source_image="eval/image/new.png",
            method="fallback",
            event_title="完全不同的新事件",
            event_summary="这是一条和参考数据集没有关系的新事件。",
            domain="general",
            keywords=["新事件"],
            target="传播目标",
            opinion_variants=["新观点"],
            confidence=0.25,
            fallback_used=True,
            warnings=["vision_llm_unavailable"],
        )

        event, metadata = HotEventMatcher(match_threshold=0.72).match(
            recognition,
            [
                {
                    "event_id": "hot_event_005",
                    "event_title": "高端AI芯片供应和出口限制引发产业链担忧",
                    "event_summary": "芯片供应紧张。",
                    "target": "说明芯片供应限制影响。",
                    "opinion_variants": ["芯片供应紧张。"],
                }
            ],
        )

        self.assertEqual(event["event_id"], "image_event_001")
        self.assertIsNone(event["image_recognition"]["matched_event_id"])
        self.assertLess(metadata["match_score"], 0.72)

    def test_load_image_events_reads_directory_in_name_order_and_applies_limit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "eval" / "image"
            image_dir.mkdir(parents=True)
            (image_dir / "2.png").write_bytes(b"2")
            (image_dir / "1.png").write_bytes(b"1")
            (image_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

            events = load_image_events(
                image=None,
                image_dir=image_dir,
                reference_events=[],
                workspace_root=root,
                event_limit=1,
                use_llm=True,
                vision_client=FakeVisionClient(),
            )

            self.assertEqual(len(events), 1)
            self.assertTrue(events[0]["source_image"].endswith("1.png"))

    def test_load_image_events_rejects_empty_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "empty"
            image_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "Image directory is empty"):
                load_image_events(
                    image=None,
                    image_dir=image_dir,
                    reference_events=[],
                    workspace_root=tmpdir,
                )


if __name__ == "__main__":
    unittest.main()
