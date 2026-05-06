from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.data_loader import DataLoader


class DataLoaderTest(unittest.TestCase):
    def test_loads_standardized_dataset_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            derived_dir = root / "data" / "derived"
            raw_dir.mkdir(parents=True)
            derived_dir.mkdir(parents=True)

            product_info = {
                "product_name": "abc_reading",
                "domain": "linguistics",
                "ads": "sample context",
                "influencer_ids": ["1", "2"],
            }
            profiles = {
                "1": {
                    "user_id": 1,
                    "user_name": "user_1",
                    "user_followers": 100,
                    "user_friends": 20,
                    "user_interests": ["reading"],
                    "user_description": "profile one",
                }
            }
            interactions = {
                "1": [
                    {
                        "text_raw": "raw",
                        "text_comment": "comment",
                        "interact_type": "comment",
                        "interact_id": 2,
                    }
                ]
            }
            enriched_profiles = {
                "1": {
                    **profiles["1"],
                    "graph_attributes": {"neighbor_count": 1, "isolated": False},
                    "neighbors": [
                        {
                            "neighbor_id": "2",
                            "relation": "engaged_by",
                            "received_comment_count": 1,
                            "total_interaction_count": 1,
                        }
                    ],
                }
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
            product_context = loader.load_product_context()
            profile_map = loader.load_profiles()
            interaction_map = loader.load_interactions()
            enriched_map = loader.load_enriched_profiles()
            summary = loader.build_summary()

            self.assertEqual(product_context.product_name, "abc_reading")
            self.assertEqual(profile_map["1"].user_followers, 100)
            self.assertEqual(interaction_map["1"][0].source_user_id, "1")
            self.assertEqual(interaction_map["1"][0].target_user_id, "2")
            self.assertEqual(enriched_map["1"].graph_attributes.neighbor_count, 1)
            self.assertEqual(summary.interaction_record_count, 1)

    def test_load_dataset_bundle_respects_limits(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "data" / "raw"
            raw_dir.mkdir(parents=True)

            (raw_dir / "abc_reading_product_info.json").write_text(
                json.dumps({"product_name": "abc_reading"}),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_profile.graph.anon").write_text(
                json.dumps(
                    {
                        "1": {"user_id": 1, "user_name": "user_1"},
                        "2": {"user_id": 2, "user_name": "user_2"},
                    }
                ),
                encoding="utf-8",
            )
            (raw_dir / "abc_reading_interaction.graph.anon").write_text(
                json.dumps(
                    {
                        "1": [
                            {"interact_id": 2, "interact_type": "comment"},
                            {"interact_id": 3, "interact_type": "reposts"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            loader = DataLoader(root)
            bundle = loader.load_dataset_bundle(
                include_enriched_profiles=False,
                profile_limit=1,
                interaction_source_limit=1,
                interaction_record_limit=1,
            )

            self.assertEqual(len(bundle.profiles), 1)
            self.assertEqual(len(bundle.interactions["1"]), 1)
            self.assertIsNone(bundle.enriched_profiles)


if __name__ == "__main__":
    unittest.main()
