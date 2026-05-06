from __future__ import annotations

import unittest

from influence_strategy.event_parser import RuleBasedEventParser


class EventParserTest(unittest.TestCase):
    def test_parses_natural_language_event(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            "希望围绕亲子阅读和英语启蒙做一次传播活动，重点提升讨论度，并控制刷屏争议。"
        )

        self.assertEqual(parsed.product_name, "abc_reading")
        self.assertEqual(parsed.event_type, "english_learning_engagement")
        self.assertEqual(parsed.target_goal, "engagement")
        self.assertIn("parent_child", parsed.target_audience)
        self.assertIn("english_learning", parsed.target_audience)
        self.assertEqual(parsed.constraints.risk_level, "medium")

    def test_respects_explicit_payload_fields(self) -> None:
        parser = RuleBasedEventParser()
        parsed = parser.parse(
            {
                "event_id": "evt_manual_001",
                "product_name": "abc_reading",
                "event_title": "争议回应",
                "event_description": "针对近期投诉和质疑做一次集中回应与澄清。",
                "target_goal": "response",
                "target_audience": ["general_public"],
                "constraints": {"max_selected_nodes": 6, "risk_level": "high"},
            }
        )

        self.assertEqual(parsed.event_id, "evt_manual_001")
        self.assertEqual(parsed.target_goal, "response")
        self.assertEqual(parsed.event_type, "public_opinion_response")
        self.assertEqual(parsed.constraints.max_selected_nodes, 6)
        self.assertEqual(parsed.constraints.risk_level, "high")
        self.assertIn("general_public", parsed.target_audience)


if __name__ == "__main__":
    unittest.main()
