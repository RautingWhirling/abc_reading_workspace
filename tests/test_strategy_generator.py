from __future__ import annotations

import unittest

from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.models import (
    EnrichedUserProfile,
    EventConstraints,
    GraphAttributes,
    ParsedEvent,
    ProductContext,
    UserProfile,
)
from influence_strategy.scorer import Scorer
from influence_strategy.selector import Selector
from influence_strategy.strategy_generator import StrategyGenerator


class StrategyGeneratorTest(unittest.TestCase):
    def test_generate_builds_structured_strategy(self) -> None:
        product_context = ProductContext(
            product_name="abc_reading",
            domain="reading",
            influencer_ids=["1"],
        )
        profiles = {
            "1": UserProfile(
                user_id="1",
                user_name="user_1",
                user_followers=1200,
                user_friends=80,
                user_interests=["亲子阅读", "英语启蒙"],
                user_description="专注亲子阅读和英语启蒙讨论",
            ),
            "2": UserProfile(
                user_id="2",
                user_name="user_2",
                user_followers=300,
                user_friends=60,
                user_interests=["传播扩散", "阅读活动"],
                user_description="适合做二次扩散",
            ),
            "3": UserProfile(
                user_id="3",
                user_name="user_3",
                user_followers=180,
                user_friends=35,
                user_interests=["亲子问答", "教育交流"],
                user_description="擅长评论互动与答疑",
            ),
            "4": UserProfile(
                user_id="4",
                user_name="user_4",
                user_followers=12,
                user_friends=3,
                user_interests=[],
                user_description="",
            ),
        }
        enriched_profiles = {
            "1": EnrichedUserProfile(
                **profiles["1"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=3,
                    mutual_neighbor_count=1,
                    received_interaction_count=8,
                    received_comment_count=6,
                    received_repost_count=2,
                    made_interaction_count=2,
                    made_repost_count=2,
                    isolated=False,
                ),
            ),
            "2": EnrichedUserProfile(
                **profiles["2"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=4,
                    mutual_neighbor_count=1,
                    received_interaction_count=2,
                    received_comment_count=1,
                    received_repost_count=1,
                    made_interaction_count=8,
                    made_comment_count=2,
                    made_repost_count=6,
                    isolated=False,
                ),
            ),
            "3": EnrichedUserProfile(
                **profiles["3"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=2,
                    mutual_neighbor_count=1,
                    received_interaction_count=4,
                    received_comment_count=4,
                    made_interaction_count=3,
                    made_comment_count=3,
                    isolated=False,
                ),
            ),
            "4": EnrichedUserProfile(
                **profiles["4"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=0,
                    mutual_neighbor_count=0,
                    received_interaction_count=1,
                    made_interaction_count=1,
                    self_interaction_count=1,
                    isolated=True,
                ),
            ),
        }
        event = ParsedEvent(
            event_id="event_001",
            product_name="abc_reading",
            event_title="亲子阅读传播活动",
            event_description="围绕亲子阅读与英语启蒙做一次传播活动，提升讨论度并控制风险。",
            target_goal="engagement",
            event_type="parent_child_reading",
            target_audience=["parent_child", "english_learning"],
            extracted_keywords=["亲子阅读", "英语启蒙", "讨论"],
            constraints=EventConstraints(
                risk_level="medium",
                max_selected_nodes=3,
                max_frequency_per_day=3,
                campaign_window_hours=24,
            ),
        )

        feature_result = FeatureBuilder().build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids={"1", "2", "3"},
        )
        score_result = Scorer().score(feature_result)
        selection_result = Selector().select(score_result)
        strategy_result = StrategyGenerator().generate(selection_result)

        self.assertEqual(strategy_result.summary.selected_count, 3)
        self.assertEqual(strategy_result.strategy.platform_plan.primary_platform, "weibo_simulated")
        self.assertIn("亲子阅读相关群体", strategy_result.strategy.target_object)
        self.assertEqual(strategy_result.strategy.frequency_plan.global_cap_per_day, 3)
        self.assertTrue(strategy_result.stage_plans)

        stage_names = {item.stage_name for item in strategy_result.stage_plans}
        self.assertIn("stage_1_launch", stage_names)
        self.assertIn("stage_3_amplify", stage_names)

        primary_roles = {item.selected_role for item in strategy_result.selected_nodes}
        self.assertIn("core_publish_node", primary_roles)
        self.assertIn("amplification_node", primary_roles)

    def test_to_frame_and_stage_frame_export_columns(self) -> None:
        product_context = ProductContext(product_name="abc_reading")
        profiles = {
            "1": UserProfile(
                user_id="1",
                user_name="user_1",
                user_followers=100,
                user_interests=["阅读"],
                user_description="阅读分享",
            ),
        }
        enriched_profiles = {
            "1": EnrichedUserProfile(
                **profiles["1"].model_dump(),
                graph_attributes=GraphAttributes(
                    neighbor_count=1,
                    received_interaction_count=3,
                    made_interaction_count=1,
                    isolated=False,
                ),
            ),
        }
        event = ParsedEvent(
            event_id="event_002",
            product_name="abc_reading",
            event_title="阅读活动",
            event_description="普通传播事件",
            target_goal="awareness",
            event_type="general_influence_event",
            constraints=EventConstraints(max_selected_nodes=1),
        )

        feature_result = FeatureBuilder().build_features(
            product_context=product_context,
            profiles=profiles,
            event=event,
            enriched_profiles=enriched_profiles,
            source_user_ids={"1"},
        )
        score_result = Scorer().score(feature_result)
        selection_result = Selector().select(score_result)
        strategy_result = StrategyGenerator().generate(selection_result)

        primary_frame = StrategyGenerator().to_frame(strategy_result)
        stage_frame = StrategyGenerator().stage_frame(strategy_result)

        self.assertIn("recommended_action", primary_frame.columns)
        self.assertIn("timing_window", primary_frame.columns)
        self.assertIn("content_focus", stage_frame.columns)
        self.assertIn("success_signal", stage_frame.columns)


if __name__ == "__main__":
    unittest.main()
