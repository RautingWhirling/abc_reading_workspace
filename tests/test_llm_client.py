from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from influence_strategy.llm_client import OpenAICompatibleLLMClient


class LLMClientTest(unittest.TestCase):
    def test_from_env_files_supports_generic_llm_keys_for_deepseek(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text(
                "\n".join(
                    [
                        "LLM_API_KEY=test-key",
                        "LLM_BASE_URL=https://api.deepseek.com",
                        "LLM_MODEL=deepseek-v4-flash",
                    ]
                ),
                encoding="utf-8",
            )

            client = OpenAICompatibleLLMClient.from_env_files(root)

            self.assertIsNotNone(client)
            assert client is not None
            self.assertEqual(client.provider, "deepseek")
            self.assertEqual(client.model, "deepseek-v4-flash")
            self.assertEqual(client.base_url, "https://api.deepseek.com")
            self.assertEqual(
                client.describe(),
                {
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash",
                    "base_url": "https://api.deepseek.com",
                },
            )


if __name__ == "__main__":
    unittest.main()
