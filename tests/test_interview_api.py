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

    def test_history_list_and_detail_are_browser_scoped(self):
        response = self.client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"})
        self.assertIn("HttpOnly", response.headers["Set-Cookie"])
        iid = response.json()["state"]["interview_id"]
        path = f"/api/interviews/{iid}"
        self.client.post(path + "/answer", json={"answer": "single"})
        history = self.client.get("/api/interviews")
        self.assertEqual(history.headers["Cache-Control"], "no-store")
        self.assertEqual(history.json()["interviews"][0]["interview_id"], iid)
        detail = self.client.get(path).json()
        self.assertEqual(detail["turns"][0]["answer"], "single")
        self.assertEqual(detail["output"]["fact"], "has_side_income")

        invalid_titles = ("   ", "x" * 101, "Line one\nLine two", "\x00")
        for title in invalid_titles:
            self.assertEqual(self.client.patch(path, json={"title": title}).status_code, 422)
        renamed = self.client.patch(path, json={"title": "  2025 Intake · Follow-up  "})
        self.assertEqual(renamed.json(), {"title": "2025 Intake · Follow-up"})
        self.assertEqual(
            self.client.get("/api/interviews").json()["interviews"][0]["title"], "2025 Intake · Follow-up",
        )

        other = TestClient(app)
        self.addCleanup(other.close)
        other.get("/interview")
        self.assertEqual(other.get("/api/interviews").json(), {"interviews": []})
        self.assertEqual(other.get(path).status_code, 404)
        self.assertEqual(other.patch(path, json={"title": "Another browser"}).status_code, 404)
        self.assertEqual(other.post(path + "/answer", json={"answer": "yes"}).status_code, 404)
        self.assertEqual(other.get(path + "/forms").status_code, 404)

    def test_delete_interview_removes_only_the_owned_record(self):
        payload = {"scenario": "ca_w2_side_gig"}
        first = self.client.post("/api/interviews", json=payload).json()["state"]["interview_id"]
        second = self.client.post("/api/interviews", json=payload).json()["state"]["interview_id"]
        path = f"/api/interviews/{first}"
        other = TestClient(app)
        self.addCleanup(other.close)
        other.get("/interview")
        self.assertEqual(other.delete(path).status_code, 404)
        self.assertEqual(self.client.get(path).status_code, 200)

        deleted = self.client.delete(path)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(deleted.content, b"")
        self.assertEqual(self.client.get(path).status_code, 404)
        self.assertEqual(self.client.delete(path).status_code, 404)
        rows = self.client.get("/api/interviews").json()["interviews"]
        self.assertEqual([row["interview_id"] for row in rows], [second])

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

    def test_start_intake_with_priority(self):
        """POST /api/interviews accepts a priority list; a bad section is a 422."""
        r = self.client.post("/api/interviews", json={"scenario": "intake", "priority": ["side_gig"]})
        self.assertEqual(r.json()["output"]["fact"], "has_self_employment")
        bad = self.client.post("/api/interviews", json={"scenario": "intake", "priority": ["bogus"]})
        self.assertEqual(bad.status_code, 422)

    def test_form1040_generation_and_supplemental_data_are_not_saved(self):
        started = self.client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()
        iid = started["state"]["interview_id"]
        path = f"/api/interviews/{iid}/forms/federal-2025-1040"
        payload = {"tax_year": 2025, "confirmed_tax_year": True, "values": {"first_name": "Alex", "ssn": "123456789"}}
        self.assertEqual(self.client.post(path, json=payload).status_code, 409)
        for text in ["single", "yes", "12000"]:
            self.client.post(f"/api/interviews/{iid}/answer", json={"answer": text})
        catalog = self.client.get(f"/api/interviews/{iid}/forms").json()["forms"]
        self.assertTrue(catalog[0]["recommended"])
        self.assertEqual(catalog[0]["id"], "federal-2025-1040")
        self.assertEqual(self.client.post(f"/api/interviews/{iid}/forms/unknown", json=payload).status_code, 404)
        before = self.client.get(f"/api/interviews/{iid}").json()
        preview = self.client.post(path + "/preview", json=payload)
        self.assertEqual(preview.status_code, 200)
        self.assertIn("w2_federal_withholding", preview.json()["missing"])
        response = self.client.post(path, json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertTrue(response.content.startswith(b"%PDF"))
        self.assertEqual(self.client.get(f"/api/interviews/{iid}").json(), before)
        self.assertEqual(self.client.post(path, json=payload | {"tax_year": 2026}).status_code, 422)
        self.assertEqual(self.client.post("/api/interviews/missing/forms/federal-2025-1040", json=payload).status_code, 404)
        empty = self.client.post(path + "/preview", json={"tax_year": 2025, "confirmed_tax_year": True})
        self.assertIn("first_name", empty.json()["missing"])
        self.assertNotIn("Alex", str(empty.json()))

    def test_another_configured_form_uses_the_same_endpoints(self):
        import json
        from pathlib import Path
        from shutil import copyfile
        from tempfile import TemporaryDirectory

        from src.app.forms import load_forms

        original = load_forms()["federal-2025-1040"]
        started = self.client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()
        iid = started["state"]["interview_id"]
        for text in ["single", "yes", "12000"]:
            self.client.post(f"/api/interviews/{iid}/answer", json={"answer": text})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            copyfile(original["template"], root / "example.pdf")
            definition = {
                "id": "example-2026-other", "title": "Fictional API test form", "tax_year": 2026,
                "recommend_when": [{"fact": "filing_status", "equals": "single"}],
                "fields": [{"key": "name", "label": "Name", "source": "input",
                            "pdf_field": "topmostSubform[0].Page1[0].f1_14[0]"}],
            }
            (root / "example.mapping.json").write_text(json.dumps(definition), encoding="utf-8")
            with patch("src.app.forms.FORMS_ROOT", root):
                catalog = self.client.get(f"/api/interviews/{iid}/forms").json()["forms"]
                self.assertEqual(catalog[0]["id"], "example-2026-other")
                self.assertTrue(catalog[0]["recommended"])
                path = f"/api/interviews/{iid}/forms/example-2026-other"
                payload = {"tax_year": 2026, "confirmed_tax_year": True, "values": {"name": "Alex"}}
                preview = self.client.post(path + "/preview", json=payload)
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(preview.json()["title"], "Fictional API test form")
                self.assertEqual(self.client.post(path, json=payload).status_code, 200)
