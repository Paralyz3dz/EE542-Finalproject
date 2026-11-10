"""HTTP layer for the guided interview. Needs fastapi (run in the project venv)."""
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.app.api import app
from src.app.interviews import InterviewService


class InterviewApiTests(unittest.TestCase):
    """HTTP tests for the interview endpoints with retrieval and the model stubbed out."""
    def setUp(self):
        """Start the app with a fake chat agent and an interview service without retrieval or model."""
        agent = patch("src.app.api.create_tax_agent", return_value=MagicMock())
        agent.start()
        self.addCleanup(agent.stop)
        service = patch.object(InterviewService, "from_env", classmethod(
            lambda cls: cls(search=lambda q: {"matches": []}, normalizer=None)))
        service.start()
        self.addCleanup(service.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_page_and_playbooks(self):
        """The interview page and the playbook list are served."""
        self.assertEqual(self.client.get("/interview").status_code, 200)
        ids = [p["id"] for p in self.client.get("/api/playbooks").json()["playbooks"]]
        self.assertIn("ca_retiree", ids)

    def test_interview_round_trip(self):
        """HTTP: start, three answers to the summary, 409 after the end, then approval."""
        r = self.client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()
        iid = r["state"]["interview_id"]
        self.assertEqual(r["output"]["fact"], "filing_status")
        for text in ["single", "yes", "12000"]:
            r = self.client.post(f"/api/interviews/{iid}/answer", json={"answer": text}).json()
        self.assertEqual(r["output"]["type"], "summary")
        self.assertEqual(self.client.post(f"/api/interviews/{iid}/answer", json={"answer": "x"}).status_code, 409)
        d = self.client.post(f"/api/interviews/{iid}/decision", json={"approve": True})
        self.assertEqual(d.json()["state"]["status"], "approved")

    def test_errors(self):
        """Unknown scenario / interview -> 404; blank answer -> 422."""
        self.assertEqual(self.client.post("/api/interviews", json={"scenario": "nope"}).status_code, 404)
        self.assertEqual(self.client.post("/api/interviews/nope/answer", json={"answer": "yes"}).status_code, 404)
        self.assertEqual(self.client.post("/api/interviews/nope/answer", json={"answer": "  "}).status_code, 422)

    def test_chat_returns_503_without_agent(self):
        """Without a chat agent, /api/chat returns 503 while the interview API still works."""
        with patch("src.app.api.create_tax_agent", side_effect=RuntimeError("no key")):
            with TestClient(app) as client:
                self.assertEqual(client.post("/api/chat", json={"message": "hi"}).status_code, 503)
                self.assertEqual(client.get("/api/playbooks").status_code, 200)
