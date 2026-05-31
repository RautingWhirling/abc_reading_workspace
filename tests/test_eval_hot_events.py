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
        self.assertIn("technology", payload["target_audience"])

    def test_run_hot_event_evaluation_writes_structured_json_and_markdown(self) -> None:
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
            )

            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.with_suffix(".md").exists())
            self.assertEqual(payload["meta"]["schema_version"], "eval_v2")
            self.assertEqual(payload["meta"]["generator_mode"], "rule_only")
            self.assertEqual(payload["summary"]["event_title"], "AI chip supply focus")
            self.assertGreaterEqual(payload["summary"]["selected_count"], 1)
            self.assertIn("selected_digital_human_ids", payload["summary"])
            self.assertIn("target_object", payload["five_dimensions"])
            self.assertIn("stage_plans", payload)
            self.assertIn("selected_digital_humans", payload)
            self.assertIn("fallback_digital_humans", payload)
            self.assertIn("risk_control", payload)
            self.assertIn("explainability", payload)

            first_human = payload["selected_digital_humans"][0]
            self.assertIn("stage_text", first_human)
            self.assertIn("frequency_text", first_human)
            self.assertRegex(first_human["frequency_text"], r"^\d+/day$")
            self.assertIn("content_output", first_human)
            self.assertIn("post_content", first_human["content_output"])
            self.assertIn(
                "Explain the impact of chip supply changes on the industry chain.",
                first_human["content_output"]["post_content"],
            )

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
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(payload["meta"]["generator_mode"], "llm_enhanced")
            self.assertTrue(payload["meta"]["llm"]["used"])

            first_human = payload["selected_digital_humans"][0]
            self.assertTrue(first_human["content_generation"]["llm_generated_fields"])
            self.assertTrue(
                first_human["content_output"]["post_content"].startswith("LLM generated post content")
            )
            self.assertTrue(
                first_human["content_output"]["audience_profile"].startswith("LLM generated audience profile")
            )
            self.assertTrue(
                first_human["content_output"]["audience_interaction_strategy"].startswith(
                    "LLM generated interaction strategy"
                )
            )
            self.assertTrue(
                first_human["content_output"]["cross_digital_human_strategy"].startswith(
                    "LLM generated collaboration strategy"
                )
            )

    def test_run_hot_event_evaluations_writes_first_n_events_and_batch_markdown(self) -> None:
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
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(
                [output_path.name for output_path, _payload in results],
                [
                    "hot_event_eval_001_strategy_output.json",
                    "hot_event_eval_002_strategy_output.json",
                ],
            )
            self.assertTrue((eval_dir / "output" / "hot_event_eval_001_strategy_output.md").exists())
            self.assertTrue((eval_dir / "output" / "hot_event_eval_002_strategy_output.md").exists())
            self.assertTrue((eval_dir / "output" / "output.md").exists())
            self.assertFalse((eval_dir / "output" / "hot_event_eval_003_strategy_output.json").exists())

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
                "user_name": "core_reader",
                "user_followers": 2400,
                "user_friends": 80,
                "user_interests": ["family_reading", "english_learning"],
                "user_description": "Focus on family reading and english learning discussions.",
            },
            "2": {
                "user_id": 2,
                "user_name": "amplifier",
                "user_followers": 650,
                "user_friends": 90,
                "user_interests": ["content_diffusion", "reading_activity"],
                "user_description": "Suitable for secondary diffusion and hot topic retelling.",
            },
            "3": {
                "user_id": 3,
                "user_name": "responder",
                "user_followers": 380,
                "user_friends": 50,
                "user_interests": ["qa", "education_exchange"],
                "user_description": "Good at comment interaction and answering questions.",
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
