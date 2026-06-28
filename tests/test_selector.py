from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader
from influence_strategy.event_parser import RuleBasedEventParser
from influence_strategy.feature_builder import FeatureBuilder
from influence_strategy.models import SelectedNode
from influence_strategy.scorer import Scorer
from influence_strategy.selector import Selector


class FakeSelectorLLMClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = json.loads(user_prompt)
        shortlisted = payload["shortlist"]
        selected_order = []
        preferred_roles = [
            "interaction_response_node",
            "core_publish_node",
            "amplification_node",
        ]
        for index, item in enumerate(shortlisted[: payload["constraints"]["max_selected_nodes"]]):
            selected_order.append(
                {
                    "user_id": item["user_id"],
                    "recommended_role": preferred_roles[index % len(preferred_roles)],
                    "reasoning": [f"picked_{item['user_id']}"],
                }
            )
        return {
            "selected_order": selected_order,
            "fallback_order": [item["user_id"] for item in shortlisted[payload["constraints"]["max_selected_nodes"] :]],
            "global_notes": ["selector rerank used"],
        }


class SelectorTest(unittest.TestCase):
    def test_selects_primary_and_fallback_nodes(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {
                "product_name": "abc_reading",
                "domain": "linguistics",
                "ads": "reading context",
                "influencer_ids": ["1"],
            }
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "user_1",
                    "user_followers": 1200,
                    "user_friends": 80,
                    "user_interests": ["亲子阅读", "英语启蒙"],
                    "user_description": "专注亲子阅读和英语启蒙",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "user_2",
                    "user_followers": 350,
                    "user_friends": 40,
                    "user_interests": ["转发传播", "阅读活动"],
                    "user_description": "适合扩散",
                },
                "3": {
                    "user_id": 3,
                    "user_name": "user_3",
                    "user_followers": 180,
                    "user_friends": 25,
                    "user_interests": ["亲子问答", "教育交流"],
                    "user_description": "擅长评论互动与答疑",
                },
                "4": {
                    "user_id": 4,
                    "user_name": "user_4",
                    "user_followers": 15,
                    "user_friends": 3,
                    "user_interests": [],
                    "user_description": "",
                },
            }
            interactions = {
                "1": [{"interact_id": 2, "interact_type": "comment"}],
                "2": [{"interact_id": 1, "interact_type": "reposts"}],
                "3": [{"interact_id": 1, "interact_type": "comment"}],
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {
                        "neighbor_count": 3,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 8,
                        "received_comment_count": 6,
                        "received_repost_count": 2,
                        "made_interaction_count": 2,
                        "made_repost_count": 2,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "2": {
                    **profiles["2"],
                    "graph_attributes": {
                        "neighbor_count": 4,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 2,
                        "received_comment_count": 1,
                        "received_repost_count": 1,
                        "made_interaction_count": 8,
                        "made_comment_count": 2,
                        "made_repost_count": 6,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "3": {
                    **profiles["3"],
                    "graph_attributes": {
                        "neighbor_count": 2,
                        "mutual_neighbor_count": 1,
                        "received_interaction_count": 4,
                        "received_comment_count": 4,
                        "made_interaction_count": 3,
                        "made_comment_count": 3,
                        "isolated": False,
                    },
                    "neighbors": [],
                },
                "4": {
                    **profiles["4"],
                    "graph_attributes": {
                        "neighbor_count": 0,
                        "mutual_neighbor_count": 0,
                        "received_interaction_count": 1,
                        "made_interaction_count": 1,
                        "self_interaction_count": 1,
                        "isolated": True,
                    },
                    "neighbors": [],
                },
            }

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps(product_info, ensure_ascii=False),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps(profiles, ensure_ascii=False),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps(interactions, ensure_ascii=False),
                encoding="utf-8",
            )
            (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
                json.dumps(enriched_profiles, ensure_ascii=False),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse(
                {
                    "event_description": "围绕亲子阅读和英语启蒙做一次传播活动，提升讨论度并控制风险。",
                    "constraints": {"max_selected_nodes": 3},
                }
            )
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(loader.load_interactions().keys()),
            )
            score_result = Scorer().score(feature_result)
            selection_result = Selector().select(score_result)

            self.assertEqual(selection_result.summary.selected_count, 3)
            self.assertLessEqual(selection_result.summary.selected_count, event.constraints.max_selected_nodes)
            selected_roles = {node.selected_role for node in selection_result.selected_nodes}
            self.assertIn("core_publish_node", selected_roles)
            self.assertIn("amplification_node", selected_roles)
            self.assertGreaterEqual(selection_result.summary.fallback_count, 1)

    def test_to_frame_supports_primary_and_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps({"product_name": "abc_reading"}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps({"1": {"user_id": 1, "user_name": "user_1"}}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps({}),
                encoding="utf-8",
            )
            (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(
                json.dumps({"1": {"user_id": 1, "user_name": "user_1", "graph_attributes": {}, "neighbors": []}}),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse("普通传播活动")
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)
            selection_result = Selector().select(score_result)

            primary_frame = Selector().to_frame(selection_result)
            fallback_frame = Selector().to_frame(selection_result, bucket="fallback")

            self.assertIn("selected_role", primary_frame.columns)
            self.assertIn("selection_bucket", primary_frame.columns)
            self.assertIn("selection_bucket", fallback_frame.columns)

    def test_selector_can_apply_llm_rerank_plan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {"product_name": "abc_reading", "influencer_ids": []}
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "user_1",
                    "user_followers": 600,
                    "user_friends": 40,
                    "user_interests": ["阅读", "传播"],
                    "user_description": "适合首发",
                },
                "2": {
                    "user_id": 2,
                    "user_name": "user_2",
                    "user_followers": 220,
                    "user_friends": 60,
                    "user_interests": ["问答", "评论"],
                    "user_description": "擅长互动答疑",
                },
                "3": {
                    "user_id": 3,
                    "user_name": "user_3",
                    "user_followers": 260,
                    "user_friends": 90,
                    "user_interests": ["扩散", "转述"],
                    "user_description": "适合扩散",
                },
            }
            enriched_profiles = {
                "1": {**profiles["1"], "graph_attributes": {"neighbor_count": 2, "received_interaction_count": 5, "made_interaction_count": 2, "isolated": False}, "neighbors": []},
                "2": {**profiles["2"], "graph_attributes": {"neighbor_count": 2, "received_interaction_count": 4, "made_interaction_count": 4, "made_comment_count": 4, "isolated": False}, "neighbors": []},
                "3": {**profiles["3"], "graph_attributes": {"neighbor_count": 4, "received_interaction_count": 3, "made_interaction_count": 7, "made_repost_count": 5, "isolated": False}, "neighbors": []},
            }

            (raw_dir / "abc_reading_product_info.json").write_text(json.dumps(product_info), encoding="utf-8")
            (raw_dir / "abc_reading_profile.graph.anon").write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(json.dumps({}), encoding="utf-8")
            (derived_dir / "abc_reading_profile_with_neighbors.graph.anon").write_text(json.dumps(enriched_profiles, ensure_ascii=False), encoding="utf-8")

            loader = DataLoader(root)
            event = RuleBasedEventParser().parse(
                {
                    "event_description": "围绕阅读讨论做一次传播活动，重点做好互动承接。",
                    "constraints": {"max_selected_nodes": 2},
                }
            )
            feature_result = FeatureBuilder().build_features(
                product_context=loader.load_product_context(),
                profiles=loader.load_profiles(),
                event=event,
                enriched_profiles=loader.load_enriched_profiles(),
                source_user_ids=set(),
            )
            score_result = Scorer().score(feature_result)
            selection_result = Selector().select(
                score_result,
                use_llm=True,
                llm_client=FakeSelectorLLMClient(),
            )

            self.assertEqual(selection_result.selected_nodes[0].selected_role, "interaction_response_node")
            self.assertTrue(
                any(reason.startswith("llm_selector=") for reason in selection_result.selected_nodes[0].selection_reasons)
            )

    def test_selector_rebalances_homogeneous_selected_roles(self) -> None:
        homogeneous_nodes = [
            SelectedNode(
                user_id=str(index),
                user_name=f"user_{index}",
                event_id="event_test",
                event_type="public_affairs",
                role_hint="amplification",
                final_score=0.8,
                eligible=True,
                priority_tier="high",
                selection_rank=index,
                selected_role="amplification_node",
                dispatch_stage="stage_3_amplify",
                selection_bucket="primary",
                selection_reasons=[
                    "selected_role=amplification_node",
                    "dispatch_stage=stage_3_amplify",
                    "selection_bucket=primary",
                ],
            )
            for index in range(1, 5)
        ]

        rebalanced_nodes = Selector()._rebalance_selected_roles(homogeneous_nodes)

        selected_roles = [node.selected_role for node in rebalanced_nodes]
        self.assertEqual(
            selected_roles,
            [
                "core_publish_node",
                "interaction_response_node",
                "amplification_node",
                "support_node",
            ],
        )
        self.assertTrue(
            any(
                reason.startswith("role_rebalanced=")
                for node in rebalanced_nodes
                for reason in node.selection_reasons
            )
        )


if __name__ == "__main__":
    unittest.main()
