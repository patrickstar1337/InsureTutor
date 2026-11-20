"""Check the web chat's follow-up request flow without calling an LLM provider."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import httpx
from openai import InternalServerError

import api


def completion(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.passage = {
            "source": "Insurance_Document.pdf",
            "pdf_pages": [5],
            "chunk_id": "test:c1",
            "text": "The brochure states a 2.5% guaranteed interest rate.",
        }
        self.provider = MagicMock()
        self.settings = {
            "LLM_API_KEY": "test-secret",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "test-model",
        }
        self.model = object()
        self.load_settings = patch.object(api, "load_llm_settings", return_value=self.settings)
        self.load_index = patch.object(
            api,
            "load_index",
            return_value=(
                SimpleNamespace(shape=(1, 640)),
                ["test:c1"],
                {"test:c1": self.passage},
                {},
            ),
        )
        self.load_model = patch.object(api, "load_model", return_value=self.model)
        self.retrieve = patch.object(api, "retrieve_results", return_value=[self.passage])
        self.openai = patch.object(api, "OpenAI")
        for patcher in (
            self.load_settings,
            self.load_index,
            self.load_model,
            self.retrieve,
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        openai_mock = self.openai.start()
        self.addCleanup(self.openai.stop)
        openai_mock.return_value.__enter__.return_value = self.provider

    def test_first_question_uses_one_model_call(self):
        self.provider.chat.completions.create.return_value = completion("2.5%")
        with TestClient(api.app) as client:
            response = client.post("/api/chat", json={"question": "What is the rate?"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "2.5%")
        self.assertEqual(self.provider.chat.completions.create.call_count, 1)
        self.assertEqual(api.retrieve_results.call_args.args[0], "What is the rate?")

    def test_followup_rewrites_before_retrieval(self):
        self.provider.chat.completions.create.side_effect = [
            completion("Is the brochure's guaranteed interest rate guaranteed for life?"),
            completion("The excerpt does not establish that."),
        ]
        history = [
            {"role": "user", "content": "What is the guaranteed interest rate?"},
            {"role": "assistant", "content": "The brochure states 2.5%."},
        ]
        with TestClient(api.app) as client:
            response = client.post(
                "/api/chat", json={"question": "Is that guaranteed for life?", "history": history}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.provider.chat.completions.create.call_count, 2)
        self.assertEqual(
            api.retrieve_results.call_args.args[0],
            "Is the brochure's guaranteed interest rate guaranteed for life?",
        )
        rewrite_messages = self.provider.chat.completions.create.call_args_list[0].kwargs["messages"]
        answer_messages = self.provider.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertIn("Is that guaranteed for life?", rewrite_messages[1]["content"])
        self.assertIn("What is the guaranteed interest rate?", rewrite_messages[1]["content"])
        self.assertIn("Question: Is that guaranteed for life?", answer_messages[1]["content"])
        self.assertIn("not brochure evidence", answer_messages[1]["content"])
        self.assertIn(self.passage["text"], answer_messages[1]["content"])

    def test_new_topic_and_chinese_followup_use_rewriter_output(self):
        english_history = [
            {"role": "user", "content": "What is the guaranteed rate?"},
            {"role": "assistant", "content": "The brochure states 2.5%."},
        ]
        chinese_history = [
            {"role": "user", "content": "失業保障的寬限期有多久？"},
            {"role": "assistant", "content": "冊子指特惠寬限期為365日。"},
        ]
        for question, search_question, history in (
            ("What is the cooling-off period?", "What is the cooling-off period?", english_history),
            ("那期間保障還有效嗎？", "失業保障的寬限期內保障還有效嗎？", chinese_history),
        ):
            with self.subTest(question=question):
                self.provider.chat.completions.create.side_effect = [
                    completion(search_question), completion("Answer from evidence")
                ]
                with TestClient(api.app) as client:
                    response = client.post(
                        "/api/chat", json={"question": question, "history": history}
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(api.retrieve_results.call_args.args[0], search_question)

    def test_prior_answer_is_context_not_evidence(self):
        self.provider.chat.completions.create.side_effect = [
            completion("What is the outpatient dental coverage limit?"),
            completion("The supplied brochure does not state that limit."),
        ]
        history = [
            {"role": "user", "content": "Does this plan cover dental care?"},
            {"role": "assistant", "content": "Yes, it covers dental care up to $500."},
        ]
        with TestClient(api.app) as client:
            response = client.post(
                "/api/chat", json={"question": "What is its annual limit?", "history": history}
            )
        self.assertEqual(response.status_code, 200)
        answer_messages = self.provider.chat.completions.create.call_args_list[1].kwargs["messages"]
        self.assertIn("Previous conversation is only for understanding references, not evidence", answer_messages[0]["content"])
        self.assertIn("Yes, it covers dental care up to $500.", answer_messages[1]["content"])
        self.assertIn(self.passage["text"], answer_messages[1]["content"])

    def test_invalid_history_is_rejected_before_model_call(self):
        with TestClient(api.app) as client:
            response = client.post(
                "/api/chat",
                json={"question": "What is the rate?", "history": [{"role": "assistant", "content": "2.5%"}]},
            )
            too_long = client.post(
                "/api/chat",
                json={
                    "question": "What is the rate?",
                    "history": [
                        {"role": "user" if index % 2 == 0 else "assistant", "content": "Earlier turn"}
                        for index in range(6)
                    ],
                },
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(too_long.status_code, 422)
        self.provider.chat.completions.create.assert_not_called()

    def test_rewrite_service_error_stops_before_retrieval(self):
        self.provider.chat.completions.create.side_effect = InternalServerError(
            "server error",
            response=httpx.Response(
                500, request=httpx.Request("POST", "https://example.invalid/v1/chat/completions")
            ),
            body={},
        )
        history = [
            {"role": "user", "content": "What is the guaranteed rate?"},
            {"role": "assistant", "content": "The brochure states 2.5%."},
        ]
        with TestClient(api.app) as client:
            response = client.post(
                "/api/chat", json={"question": "Does that change?", "history": history}
            )
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("test-secret", response.text)
        api.retrieve_results.assert_not_called()


if __name__ == "__main__":
    unittest.main()
