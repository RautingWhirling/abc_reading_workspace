from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.eval_hot_events import (
    hot_event_to_pipeline_payload,
    run_hot_event_evaluation,
    run_hot_event_evaluations,
)


class FakeLLMClient:
    provider = "deepseek"
    model = "deepseek-test"
    base_url = "https://api.deepseek.com"

    def describe(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
        }

    def generate_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, dict]:
        payload = json.loads(user_prompt)
        if "raw_event" in payload and "fallback_parse" in payload:
            return {
                "event_type": "general_influence_event",
                "target_goal": "awareness",
                "target_audience": ["general_public", "technology"],
                "extracted_keywords": ["chip", "supply", "industry_chain"],
                "risk_level": "medium",
                "semantic_tags": ["芯片供应", "产业链影响"],
                "narrative_frames": ["行业解释", "风险提示"],
                "target_roles": [
                    "core_publish_node",
                    "interaction_response_node",
                    "amplification_node",
                ],
                "negative_constraints": ["避免绝对化表达"],
                "dispatch_preferences": {
                    "candidate_pool_size": 12,
                    "rerank_top_k": 6,
                    "semantic_weight": 0.35,
                    "diversity_weight": 0.20,
                    "risk_weight": 0.20,
                },
                "reasoning": ["llm event parser"],
            }
        if "candidate_cards" in payload:
            nodes = {}
            for index, node in enumerate(payload["candidate_cards"]):
                user_id = node["user_id"]
                nodes[f"id{user_id}"] = {
                    "semantic_relevance_score": max(0.2, 0.92 - index * 0.08),
                    "audience_fit_score": max(0.2, 0.88 - index * 0.06),
                    "role_fit_score": max(0.2, 0.84 - index * 0.04),
                    "narrative_fit_score": max(0.2, 0.80 - index * 0.04),
                    "risk_conflict_score": 0.08,
                    "novelty_score": 0.45 + 0.03 * index,
                    "semantic_tags": [f"tag_{user_id}", "event_fit"],
                    "reasoning": [f"candidate_{user_id}_fit"],
                }
            return {"nodes": nodes}
        if "shortlist" in payload and "constraints" in payload:
            shortlist = payload["shortlist"]
            selected_order = []
            role_cycle = [
                "core_publish_node",
                "interaction_response_node",
                "amplification_node",
            ]
            for index, node in enumerate(shortlist[: payload["constraints"]["max_selected_nodes"]]):
                selected_order.append(
                    {
                        "user_id": node["user_id"],
                        "recommended_role": role_cycle[index % len(role_cycle)],
                        "reasoning": [f"selector_pick_{node['user_id']}"],
                    }
                )
            return {
                "selected_order": selected_order,
                "fallback_order": [
                    node["user_id"]
                    for node in shortlist[payload["constraints"]["max_selected_nodes"] :]
                ],
                "global_notes": ["llm selector used"],
            }

        nodes = {}
        for node in payload["eligible_nodes"]:
            user_id = node["user_id"]
            nodes[f"id{user_id}"] = {
                "post_content": f"LLM generated post content {user_id}",
                "audience_profile": f"LLM generated audience profile {user_id}",
                "audience_interaction_strategy": f"LLM generated interaction strategy {user_id}",
                "cross_digital_human_strategy": f"LLM generated collaboration strategy {user_id}",
            }
        return {"nodes": nodes}


class EvalHotEventsTest(unittest.TestCase):
    def test_hot_event_payload_keeps_ten_opinion_variants(self) -> None:
        hot_event = {
            "event_id": "hot_event_test",
            "domain": "technology",
            "event_title": "AI supply attention",
            "event_summary": "Chip supply tension continues.",
            "opinion_variants": [f"Variant {index}" for index in range(1, 11)],
        }

        payload = hot_event_to_pipeline_payload(hot_event)

        self.assertEqual(payload["event_id"], "hot_event_test")
        self.assertEqual(payload["constraints"]["max_selected_nodes"], 5)
        self.assertEqual(payload["constraints"]["risk_level"], "low")
        self.assertIn("10. Variant 10", payload["event_description"])
        self.assertIn("热点领域: technology", payload["event_description"])
        self.assertEqual(payload["target_audience"], ["general_public"])

    def test_run_hot_event_evaluation_writes_structured_json_and_trace_outputs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            trace_dir = root / "tests" / "pipeline_step_outputs"
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": "hot_event_eval_001",
                            "domain": "technology",
                            "event_title": "AI chip supply focus",
                            "event_summary": "High-end chip supply is tightening.",
                            "target": "Explain the impact of chip supply changes on the industry chain.",
                            "is_synthetic": True,
                            "opinion_variants": [
                                f"AI chip supply variant {index}"
                                for index in range(1, 11)
                            ],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path, payload = run_hot_event_evaluation(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                use_llm=False,
                trace_dir=trace_dir,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(payload["事件名称"], "AI chip supply focus")
            self.assertIn("选取数字人id组", payload)
            self.assertTrue(payload["选取数字人id组"])
            self.assertEqual(payload["输出格式版本"], "action_schema_v3_compact")

            first_id = payload["选取数字人id组"][0]
            first_human = payload[f"id{first_id}"]
            self.assertIn("时间阶段", first_human)
            self.assertIn("发帖频率", first_human)
            self.assertRegex(first_human["发帖频率"], r"^\d+/day$")
            self.assertIn("发帖内容", first_human)
            self.assertIn("目标受众", first_human)
            self.assertIn("动作清单", first_human)
            self.assertTrue(first_human["动作清单"])
            first_action = first_human["动作清单"][0]
            self.assertIn("动作编号", first_action)
            self.assertIn("目标定位", first_action)
            self.assertIn("生成对象", first_action)

            event_trace_dir = trace_dir / "hot_event_eval_001"
            self.assertTrue((event_trace_dir / "00_hot_event_input.json").exists())
            self.assertTrue((event_trace_dir / "02_event_parser_output.json").exists())
            self.assertTrue((event_trace_dir / "03_feature_builder_output.json").exists())
            self.assertTrue((event_trace_dir / "04_scorer_output.json").exists())
            self.assertTrue((event_trace_dir / "05_selector_output.json").exists())
            self.assertTrue((event_trace_dir / "06_strategy_generator_output.json").exists())
            self.assertTrue((event_trace_dir / "07_final_output.json").exists())

    def test_run_hot_event_evaluation_replaces_text_fields_with_llm_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    {
                        "event_id": "hot_event_eval_llm",
                        "domain": "technology",
                        "event_title": "AI chip supply test",
                        "event_summary": "High-end chip supply is tightening.",
                        "target": "Test replacing content with LLM output.",
                        "opinion_variants": [f"Variant {index}" for index in range(1, 11)],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_path, payload = run_hot_event_evaluation(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                llm_client=FakeLLMClient(),
                trace_dir=root / "tests" / "pipeline_step_outputs",
            )

            self.assertTrue(output_path.exists())
            first_id = payload["选取数字人id组"][0]
            first_human = payload[f"id{first_id}"]
            self.assertTrue(first_human["发帖内容"].startswith("LLM generated post content"))
            self.assertTrue(
                first_human["目标受众"]["目标群体画像"].startswith("LLM generated audience profile")
            )
            action_payload_text = json.dumps(
                [action["执行参数"] for action in first_human["动作清单"]],
                ensure_ascii=False,
            )
            self.assertIn("LLM generated interaction strategy", action_payload_text)
            self.assertIn("LLM generated collaboration strategy", action_payload_text)

    def test_run_hot_event_evaluations_writes_first_n_events_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_minimal_workspace(root)

            eval_dir = root / "eval"
            eval_dir.mkdir(parents=True)
            input_path = eval_dir / "hot_event_opinion_variants.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "event_id": f"hot_event_eval_{index:03d}",
                            "domain": "technology",
                            "event_title": f"Batch event {index}",
                            "event_summary": "Batch evaluation event.",
                            "target": "Test batch output.",
                            "opinion_variants": [
                                f"Batch variant {variant_index}"
                                for variant_index in range(1, 11)
                            ],
                        }
                        for index in range(1, 4)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            results = run_hot_event_evaluations(
                workspace_root=root,
                input_path=input_path,
                output_dir=eval_dir / "output",
                event_limit=2,
                use_llm=False,
                trace_dir=root / "tests" / "pipeline_step_outputs",
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                [output_path.name for output_path, _payload in results],
                [
                    "hot_event_eval_001_strategy_output.json",
                    "hot_event_eval_002_strategy_output.json",
                ],
            )
            self.assertFalse(
                (eval_dir / "output" / "hot_event_eval_003_strategy_output.json").exists()
            )

    def _write_minimal_workspace(self, root: Path) -> None:
        raw_dir = root / "data" / "raw"
        derived_dir = root / "data" / "derived"
        raw_dir.mkdir(parents=True)
        derived_dir.mkdir(parents=True)

        product_info = {
            "product_name": "abc_reading",
            "domain": "reading",
            "ads": "reading context",
            "influencer_ids": ["1"],
        }
        profiles = {
            "1": {
                "user_id": 1,
                "user_name": "core_industry_observer",
                "user_followers": 2400,
                "user_friends": 80,
                "user_interests": ["chip_supply", "industry_chain", "ai_infrastructure"],
                "user_description": "Focus on chip supply, semiconductor industry chain, and AI infrastructure updates.",
            },
            "2": {
                "user_id": 2,
                "user_name": "amplifier_supply_chain",
                "user_followers": 650,
                "user_friends": 90,
                "user_interests": ["semiconductor", "market_update", "content_diffusion"],
                "user_description": "Suitable for secondary diffusion of semiconductor and supply chain discussions.",
            },
            "3": {
                "user_id": 3,
                "user_name": "responder_qa",
                "user_followers": 380,
                "user_friends": 50,
                "user_interests": ["industry_qa", "technology_policy", "education_exchange"],
                "user_description": "Good at comment interaction and answering technology policy questions.",
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


if __name__ == "__main__":
    unittest.main()
