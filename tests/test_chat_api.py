from concurrent.futures import ThreadPoolExecutor
import time
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.app.api import app


class ChatApiTests(unittest.TestCase):
    def setUp(self):
        self.agent = MagicMock()
        self.agent.invoke.side_effect = lambda payload, **kwargs: {
            "messages": [*payload["messages"], AIMessage(content="An answer")]
        }
        factory = patch("src.app.api.create_tax_agent", return_value=self.agent)
        factory.start()
        self.addCleanup(factory.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def send(self, message, session_id=None):
        return self.client.post("/api/chat", json={
            "message": message, "session_id": session_id,
        })

    def test_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("/api/chat", response.text)

    def test_history_preserves_tool_messages_and_sessions_are_isolated(self):
        history = [
            HumanMessage(content="First"),
            AIMessage(content="", tool_calls=[{
                "name": "search_tax_documents", "args": {"query": "First"}, "id": "call-1",
            }]),
            ToolMessage(content="IRS evidence", tool_call_id="call-1"),
            AIMessage(content="An answer"),
        ]
        self.agent.invoke.side_effect = None
        self.agent.invoke.return_value = {"messages": history}
        first = self.send("First")
        self.assertEqual(first.status_code, 200)
        session_id = first.json()["session_id"]
        self.assertEqual(first.json()["answer"], "An answer")
        self.send("Follow-up", session_id)
        self.assertEqual(self.agent.invoke.call_args.args[0]["messages"], [
            *history, {"role": "user", "content": "Follow-up"},
        ])
        other = self.send("Separate")
        self.assertNotEqual(other.json()["session_id"], session_id)
        self.assertEqual(self.agent.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Separate"},
        ])

    def test_failed_invocation_does_not_modify_saved_history(self):
        session_id = self.send("First").json()["session_id"]

        def fail(payload, **kwargs):
            payload["messages"][0]["content"] = "Modified"
            raise RuntimeError("private upstream details")

        self.agent.invoke.side_effect = fail
        with self.assertLogs("src.app.api", level="ERROR"):
            failed = self.send("Failed", session_id)
        self.assertEqual(failed.status_code, 502)
        self.assertNotIn("private upstream details", failed.text)
        self.agent.invoke.side_effect = lambda payload, **kwargs: {
            "messages": [*payload["messages"], AIMessage(content="Recovered")]
        }
        self.assertEqual(self.send("Retry", session_id).status_code, 200)
        self.assertEqual(self.agent.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "First"}, AIMessage(content="An answer"),
            {"role": "user", "content": "Retry"},
        ])

    def test_invalid_messages_and_unknown_sessions_are_rejected(self):
        for message in ("", "   ", "x" * 10001):
            self.assertEqual(self.send(message).status_code, 422)
        self.assertEqual(self.send("Hello", "invalid").status_code, 422)
        self.assertEqual(self.send("Hello", str(uuid4())).status_code, 404)
        self.agent.invoke.assert_not_called()

    def test_text_blocks_are_returned_as_text(self):
        self.agent.invoke.side_effect = None
        self.agent.invoke.return_value = {"messages": [AIMessage(content=[
            {"type": "text", "text": "First block"},
            {"type": "text", "text": "Second block"},
        ])]}
        self.assertEqual(self.send("Hello").json()["answer"], "First block\nSecond block")

    def test_concurrent_requests_to_one_session_preserve_both_turns(self):
        session_id = self.send("First").json()["session_id"]

        def answer(payload, **kwargs):
            time.sleep(0.03)
            return {"messages": [*payload["messages"], AIMessage(content="Reply")]}

        self.agent.invoke.side_effect = answer
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda text: self.send(text, session_id), ["A", "B"]))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.send("Last", session_id)
        history = self.agent.invoke.call_args.args[0]["messages"]
        self.assertEqual(len(history), 7)
        self.assertEqual({history[2]["content"], history[4]["content"]}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()
