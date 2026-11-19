"""HTTP layer for the guided interview. Needs fastapi (run in the project venv)."""
from io import BytesIO
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from pypdf import PdfReader

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
        r = self.client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()
        iid = r["state"]["interview_id"]
        path = f"/api/interviews/{iid}"
        review = f"/api/preparer/interviews/{iid}"
        queue = "/api/preparer/interviews"
        self.assertEqual(r["output"]["fact"], "filing_status")
        self.assertEqual(self.client.post(path + "/submit").status_code, 409)
        for text in ["single", "yes", "12000"]:
            r = self.client.post(path + "/answer", json={"answer": text}).json()
        self.assertEqual(r["output"]["type"], "summary")
        self.assertEqual(r["output"]["status"], "in_progress")
        self.assertEqual(r["state"]["status"], "in_progress")
        self.assertEqual(self.client.get(queue, params={"status": "waiting_for_preparer"}).json()["interviews"], [])
        pending = self.client.get(queue, params={"status": "in_progress"}).json()["interviews"]
        self.assertEqual([record["interview_id"] for record in pending], [iid])
        self.assertEqual(self.client.post(path + "/answer", json={"answer": "x"}).status_code, 409)
        self.assertEqual(self.client.post(review + "/decision", json={"approve": True}).status_code, 409)

        submitted = self.client.post(path + "/submit", json={"note": "  Check the income amount.  "})
        self.assertEqual(submitted.status_code, 200)
        self.assertEqual(submitted.json()["state"]["status"], "waiting_for_preparer")
        self.assertEqual(submitted.json()["state"]["answered"], r["state"]["answered"])
        self.assertEqual(self.client.post(path + "/submit").status_code, 200)
        waiting = self.client.get(queue, params={"status": "waiting_for_preparer"}).json()["interviews"]
        self.assertEqual([record["interview_id"] for record in waiting], [iid])
        self.assertEqual(self.client.get(queue, params={"status": "in_progress"}).json()["interviews"], [])
        detail = self.client.get(review).json()
        self.assertEqual(detail["summary"]["status"], "waiting_for_preparer")
        self.assertEqual([entry for entry in detail["history"] if entry.get("submitted")], [
            {"submitted": True, "note": "Check the income amount."},
        ])
        self.assertEqual(self.client.post(path + "/decision", json={"approve": True}).status_code, 404)
        decision = self.client.post(review + "/decision", json={"approve": True})
        self.assertEqual(decision.json()["state"]["status"], "approved")
        self.assertEqual(self.client.post(path + "/submit").status_code, 409)

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
        self.assertEqual(other.post(path + "/submit").status_code, 404)
        self.assertEqual(other.get(path + "/summary.pdf").status_code, 404)

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

    def test_summary_pdf_exports_interview_identity_and_named_data_fields(self):
        background = "Client <prefers> email & text. \u4e2d\u6587"
        started = self.client.post("/api/interviews", json={
            "scenario": "ca_w2_side_gig", "background": background,
        }).json()
        iid = started["state"]["interview_id"]
        path = f"/api/interviews/{iid}"
        preparer_path = f"/api/preparer/interviews/{iid}"
        self.assertEqual(self.client.get(path + "/summary.pdf").status_code, 409)

        for answer in ("single", "yes", "12000"):
            self.client.post(path + "/answer", json={"answer": answer})
        created_at = self.client.get("/api/interviews").json()["interviews"][0]["created_at"]
        started_line = f"Started: {created_at[:16].replace('T', ' ')} UTC"
        untitled = self.client.get(path + "/summary.pdf")
        self.assertEqual(untitled.status_code, 200)
        text = PdfReader(BytesIO(untitled.content)).pages[0].extract_text()
        self.assertIn(f"Interview ID: {iid}", text)
        self.assertIn(started_line, text)
        self.assertNotIn("Interview title:", text)

        title = "Client <A> & \u4e2d\u6587"
        self.assertEqual(self.client.patch(path, json={"title": title}).status_code, 200)
        before = self.client.get(path).json()
        self.assertEqual(before["state"]["status"], "in_progress")
        other = TestClient(app)
        self.addCleanup(other.close)
        self.assertEqual(other.get(path + "/summary.pdf").status_code, 404)

        expected = [
            "Page 1", "Collected Interview Data", f"Interview title: {title}",
            f"Interview ID: {iid}", started_line, "Data field", "Collected value",
            "Filing status", "single", "Has side income", "Yes", "Side income", "12000",
        ]
        response = self.client.get(path + "/summary.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["content-disposition"], f'attachment; filename="interview-summary-{iid[:8]}.pdf"')
        pages = PdfReader(BytesIO(response.content)).pages
        self.assertEqual(len(pages), 1)
        self.assertEqual([line.strip() for line in pages[0].extract_text().splitlines()], expected)
        self.assertEqual(self.client.get(path).json(), before)
        self.assertEqual(self.client.get(path + "/forms").status_code, 404)

        self.assertEqual(self.client.post(path + "/submit").status_code, 200)
        note = "Follow-up required."
        decision = self.client.post(preparer_path + "/decision", json={"approve": False, "note": note})
        self.assertEqual(decision.json()["state"]["status"], "returned")
        response = self.client.get(path + "/summary.pdf")
        self.assertEqual(response.status_code, 200)
        pages = PdfReader(BytesIO(response.content)).pages
        self.assertEqual(len(pages), 1)
        self.assertEqual([line.strip() for line in pages[0].extract_text().splitlines()], expected)

    def test_review_pdf_exports_current_review_and_record(self):
        background = "Client prefers email & text. \u4e2d\u6587"
        initial = self.client.post("/api/interviews", json={
            "scenario": "ca_w2_side_gig", "background": background,
        }).json()
        iid = initial["state"]["interview_id"]
        interview = f"/api/interviews/{iid}"
        review = f"/api/preparer/interviews/{iid}"
        download = review + "/summary.pdf"

        partial = self.client.get(download)
        self.assertEqual(partial.status_code, 200)
        partial_text = PdfReader(BytesIO(partial.content)).pages[0].extract_text()
        self.assertIn("In progress", partial_text)
        self.assertIn(background, partial_text)

        for answer in ("single", "yes", "12000"):
            self.client.post(interview + "/answer", json={"answer": answer})
        self.client.post(interview + "/submit", json={"note": "Verify the income amount."})
        self.client.post(review + "/decision", json={
            "approve": False, "note": "Follow-up required.\n" * 60 + "End of review note.",
        })
        before = self.client.get(review).json()
        response = self.client.get(download)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["content-disposition"], f'attachment; filename="interview-review-{iid[:8]}.pdf"')
        pages = PdfReader(BytesIO(response.content)).pages
        self.assertGreater(len(pages), 1)
        text = " ".join(" ".join(page.extract_text().split()) for page in pages)
        for line in (
            iid, "Interview Review Report", "Status Returned", "Source Web", background,
            "Information collected", "About how much did you earn from that side work in total?", "12000",
            "Document to collect", "1099-NEC", "Playbook reference", "Schedule SE instructions",
            "Interview record", "Assistant submission", "Verify the income amount.", "Preparer decision",
            "End of review note.",
        ):
            self.assertIn(line, text)
        self.assertEqual(text.count("Follow-up required."), 60)
        self.assertEqual(self.client.get(review).json(), before)
