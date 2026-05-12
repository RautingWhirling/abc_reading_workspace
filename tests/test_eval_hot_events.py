from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.eval_hot_events import (
    hot_event_to_pipeline_payload,
    run_hot_event_evaluation,
)


class FakeLLMClient:
    def generate_json(self, *, system_prompt: str, user_prompt: str):
        payload = json.loads(user_prompt)
        nodes = {}
        for node in payload["已评判可接入大模型的数字人节点"]:
            user_id = node["user_id"]
            nodes[user_id] = {
                "发帖内容": f"LLM生成发帖内容-{user_id}",
                "目标群体画像": f"LLM生成目标群体画像-{user_id}",
                "目标群体交互策略": f"LLM生成目标群体交互策略-{user_id}",
                "互动策略": f"LLM生成互动策略-{user_id}",
            }
        return {"nodes": nodes}


class EvalHotEventsTest(unittest.TestCase):
    def test_hot_event_payload_keeps_ten_opinion_variants(self) -> None:
        hot_event = {
            "event_id": "hot_event_test",
            "domain": "technology",
            "event_title": "高端AI芯片供应引发关注",
            "event_summary": "算力芯片供需紧张，企业关注替代方案。",
            "opinion_variants": [f"叙述 {index}" for index in range(1, 11)],
        }

        payload = hot_event_to_pipeline_payload(hot_event)

        self.assertEqual(payload["event_id"], "hot_event_test")
        self.assertEqual(payload["constraints"]["max_selected_nodes"], 5)
        self.assertEqual(payload["constraints"]["risk_level"], "low")
        self.assertIn("10. 叙述 10", payload["event_description"])
        self.assertIn("technology", payload["target_audience"])

    def test_run_hot_event_evaluation_writes_output_with_dimensions_and_digital_humans(self) -> None:
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
                            "event_title": "高端AI芯片供应引发产业链关注",
                            "event_summary": "高端算力芯片供需紧张，企业关注替代和优化方案。",
                            "target": "说明AI芯片供应变化对产业链的影响。",
                            "is_synthetic": True,
                            "opinion_variants": [
                                f"AI芯片供应链叙述变体 {index}"
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
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(payload["事件名称"], "高端AI芯片供应引发产业链关注")
            self.assertGreaterEqual(len(payload["选取数字人id组"]), 1)

            first_human_id = payload["选取数字人id组"][0]
            first_human = payload[f"数字人id{first_human_id}"]
            self.assertIn("时间阶段", first_human)
            self.assertIn("发帖频率", first_human)
            self.assertIn("发帖平台", first_human)
            self.assertIn("发帖内容", first_human)
            self.assertIn("目标受众", first_human)
            self.assertIn("与其他数字人互动策略", first_human)
            self.assertIn("目标群体画像", first_human["目标受众"])
            self.assertIn("目标群体交互策略", first_human["目标受众"])
            self.assertIn("互动数字人id集合", first_human["与其他数字人互动策略"])
            self.assertIn("互动策略", first_human["与其他数字人互动策略"])
            self.assertIn("说明AI芯片供应变化对产业链的影响", first_human["发帖内容"])

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
                        "event_title": "AI芯片供应测试事件",
                        "event_summary": "高端算力芯片供需紧张。",
                        "target": "测试大模型替换内容。",
                        "opinion_variants": [f"叙述 {index}" for index in range(1, 11)],
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
            first_human_id = payload["选取数字人id组"][0]
            first_human = payload[f"数字人id{first_human_id}"]
            self.assertEqual(first_human["发帖内容"], f"LLM生成发帖内容-{first_human_id}")
            self.assertEqual(first_human["目标受众"]["目标群体画像"], f"LLM生成目标群体画像-{first_human_id}")
            self.assertEqual(first_human["目标受众"]["目标群体交互策略"], f"LLM生成目标群体交互策略-{first_human_id}")
            self.assertEqual(first_human["与其他数字人互动策略"]["互动策略"], f"LLM生成互动策略-{first_human_id}")

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
                "user_interests": ["亲子阅读", "英语启蒙"],
                "user_description": "专注亲子阅读和英语启蒙讨论",
            },
            "2": {
                "user_id": 2,
                "user_name": "amplifier",
                "user_followers": 650,
                "user_friends": 90,
                "user_interests": ["传播扩散", "阅读活动"],
                "user_description": "适合做二次扩散和热点转述",
            },
            "3": {
                "user_id": 3,
                "user_name": "responder",
                "user_followers": 380,
                "user_friends": 50,
                "user_interests": ["亲子问答", "教育交流"],
                "user_description": "擅长评论互动与答疑",
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
