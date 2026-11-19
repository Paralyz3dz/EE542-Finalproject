from io import BytesIO
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from reportlab.pdfgen.canvas import Canvas

from src.app.api import app
from src.app.documents import MAX_FILE_BYTES, MAX_FILES, TAX_YEAR
from src.app.interviews import InterviewService


@pytest.fixture
def client():
    def read_document(system, user, *, images=()):
        assert "json" in system.lower()
        if images:
            assert "Transcribe" in system
            mime_type, content = images[0]
            assert mime_type == "image/jpeg"
            with Image.open(BytesIO(content)) as image:
                assert image.format == "JPEG"
                year = TAX_YEAR if image.width == 100 else TAX_YEAR - 1
            return {"text": f"Tax year {year}\nSelf-employment income: Yes\nSSN: 123-45-6789"}
        request = json.loads(user)
        assert request["tax_year"] == TAX_YEAR
        text = request["document"]
        assert "123-45-6789" not in text
        proposals = {}
        for excerpt, key, value in [
            ("Filing status: Single", "filing_status", "single"),
            ("Filing status: Married filing jointly", "filing_status", "married_joint"),
            ("Self-employment income: Yes", "has_side_income", "yes"),
            ("Self-employment income: No", "has_side_income", "no"),
            ("Total self-employment income: $1,200", "side_income", "1200"),
        ]:
            if excerpt in text:
                proposals[key] = {"value": value, "evidence": excerpt}
        proposals["fabricated"] = {"value": "yes", "evidence": text}
        proposals.setdefault("side_income", {"value": "9000", "evidence": "Unprinted amount"})
        return {"facts": proposals}

    service = InterviewService(search=lambda query: {"matches": []}, document_chat=read_document)
    with patch.object(InterviewService, "from_env", return_value=service), patch("src.app.api.create_tax_agent"):
        with TestClient(app) as test_client:
            yield test_client


def start(client):
    result = client.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()
    return f"/api/interviews/{result['state']['interview_id']}", result["output"]["question_id"]


def text_file(name, *lines, year=TAX_YEAR):
    text = "\n".join((f"Tax year {year}", *lines))
    return ("files", (name, text.encode(), "text/plain"))


def pdf_file():
    content = BytesIO()
    pdf = Canvas(content)
    pdf.drawString(40, 700, f"Tax year {TAX_YEAR}")
    pdf.drawString(40, 680, "Self-employment income: Yes")
    pdf.drawString(40, 660, "Total self-employment income: $1,200")
    pdf.save()
    return ("files", ("earnings.pdf", content.getvalue(), "application/pdf"))


def image_file(name, image_format, width=100):
    content = BytesIO()
    with Image.new("RGB", (width, 100), "white") as image:
        image.save(content, format=image_format)
    return ("files", (name, content.getvalue(), f"image/{image_format.lower()}"))


def stage(client, *files):
    response = client.post("/api/interview-documents?scenario=ca_w2_side_gig", files=list(files))
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    return response.json()["files"]


def sync(client, path, *document_ids):
    response = client.put(path + "/documents", json={"document_ids": document_ids})
    assert response.status_code == 200
    return response.json()


def test_files_stage_before_start_and_skip_current_and_follow_up_questions(client):
    intake, earnings, old = stage(
        client, text_file("intake.txt", "Filing status: Single", "SSN: 123-45-6789"),
        pdf_file(), text_file("old.txt", "Filing status: Married filing jointly", year=TAX_YEAR - 1),
    )
    assert intake["count"] == 1
    assert earnings["count"] == 2
    assert old["id"] is None and "was not found" in old["error"]
    response = client.post("/api/interviews", json={
        "scenario": "ca_w2_side_gig", "document_ids": [intake["id"], earnings["id"]],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["output"]["type"] == "summary"
    assert result["state"]["facts"] == {"filing_status": "single", "has_side_income": True, "side_income": 1200}
    path = f"/api/interviews/{result['state']['interview_id']}"
    saved = client.get(path).json()
    assert saved["history"][0]["parsed_by"] == "document"
    assert saved["history"][1]["files"] == ["earnings.pdf"]
    assert "123-45-6789" not in str(saved)

    assert client.delete(f"/api/interview-documents/{earnings['id']}").status_code == 204
    reopened = sync(client, path, intake["id"])
    assert reopened["state"]["status"] == "in_progress"
    assert reopened["state"]["facts"] == {"filing_status": "single"}
    assert reopened["output"]["fact"] == "has_side_income"
    assert reopened["state"]["missing_docs"] == []
    assert client.put(path + "/documents", json={"document_ids": [earnings["id"]]}).status_code == 409


def test_shared_support_survives_removal_and_manual_answers_are_preserved(client):
    path, original_question = start(client)
    primary, backup, amount = stage(
        client,
        text_file("primary.txt", "Filing status: Single", "Self-employment income: Yes"),
        text_file("backup.txt", "Self-employment income: Yes"),
        pdf_file(),
    )
    result = sync(client, path, primary["id"], backup["id"], amount["id"])
    assert result["output"]["type"] == "summary"
    assert client.delete(f"/api/interview-documents/{primary['id']}").status_code == 204
    result = sync(client, path, backup["id"], amount["id"])
    assert result["output"]["question_id"] == original_question
    assert result["state"]["facts"] == {"has_side_income": True, "side_income": 1200}
    answered = client.post(path + "/answer", json={"answer": "married_joint", "question_id": original_question}).json()
    assert answered["output"]["type"] == "summary"

    assert client.delete(f"/api/interview-documents/{amount['id']}").status_code == 204
    result = sync(client, path, backup["id"])
    assert result["output"]["fact"] == "side_income"
    assert result["state"]["facts"] == {"filing_status": "married_joint", "has_side_income": True}
    history = client.get(path).json()["history"]
    assert [entry.get("files") for entry in history] == [None, ["backup.txt"]]


def test_only_sent_attachments_update_the_interview(client):
    path, question_id = start(client)
    discarded, kept = stage(
        client, text_file("discard.txt", "Self-employment income: No"),
        text_file("keep.txt", "Filing status: Single"),
    )
    draft = client.get(path).json()
    assert draft["state"]["facts"] == {}
    assert draft["documents"] == []
    assert draft["output"]["question_id"] == question_id
    assert client.delete(f"/api/interview-documents/{discarded['id']}").status_code == 204

    response = client.post(path + "/answer", json={"question_id": question_id, "document_ids": [kept["id"]]})
    assert response.status_code == 200
    assert response.json()["state"]["facts"] == {"filing_status": "single"}
    assert response.json()["output"]["fact"] == "has_side_income"
    saved = client.get(path).json()
    assert saved["turns"] == [{"attachments": [{"id": kept["id"], "name": "keep.txt"}]}]


def test_sending_answer_and_files_prefers_manual_answer_and_retries_invalid_draft(client):
    path, question_id = start(client)
    attachment, = stage(client, text_file("intake.txt", "Filing status: Single", "Self-employment income: No"))
    request = {"question_id": question_id, "document_ids": [attachment["id"]]}

    assert client.post(path + "/answer", json={**request, "answer": "unclear"}).status_code == 422
    assert client.get(path).json()["state"]["facts"] == {}
    response = client.post(path + "/answer", json={**request, "answer": "married_joint"})
    assert response.status_code == 200
    assert response.json()["state"]["facts"] == {"filing_status": "married_joint", "has_side_income": False}
    assert response.json()["conflicts"] == ["filing_status"]
    turn, = client.get(path).json()["turns"]
    assert turn["answer"] == "married_joint"
    assert turn["attachments"] == [{"id": attachment["id"], "name": "intake.txt"}]


def test_removing_conflict_restores_remaining_document_and_preserves_manual_conflicts(client):
    path, initial = start(client)
    a, b = stage(
        client,
        text_file("a.txt", "Filing status: Single", "Self-employment income: Yes"),
        text_file("b.txt", "Filing status: Married filing jointly", "Self-employment income: No"),
    )
    result = sync(client, path, a["id"], b["id"])
    assert result["conflicts"] == ["filing_status", "has_side_income"]
    assert result["state"]["facts"] == {}
    assert client.post(path + "/answer", json={"answer": "married_joint", "question_id": initial}).status_code == 200

    assert client.delete(f"/api/interview-documents/{b['id']}").status_code == 204
    result = sync(client, path, a["id"])
    assert result["conflicts"] == ["filing_status"]
    assert result["state"]["facts"] == {"filing_status": "married_joint", "has_side_income": True}
    assert result["output"]["fact"] == "side_income"
    assert client.post(path + "/answer", json={"answer": "yes", "question_id": initial}).status_code == 409


def test_upload_limits_and_bad_files_are_reported_individually(client):
    upload_url = "/api/interview-documents"
    assert client.post(upload_url, json={}).status_code == 415
    assert client.post(upload_url, files=[text_file(f"{i}.txt") for i in range(MAX_FILES + 1)]).status_code == 400
    reports = stage(
        client,
        ("files", ("large.txt", b"x" * (MAX_FILE_BYTES + 1), "text/plain")),
        ("files", ("broken.pdf", b"invalid PDF", "application/pdf")),
        text_file("empty-result.txt", "Total wages: $100"),
    )
    assert all(report["id"] is None and report["count"] == 0 for report in reports)
    assert "exceeds" in reports[0]["error"]
    assert "could not be read" in reports[1]["error"]


def test_upload_accepts_ten_documents_and_files_larger_than_previous_limit(client):
    assert MAX_FILES == 10
    reports = stage(client, *(text_file(f"{i}.txt", "Filing status: Single") for i in range(MAX_FILES)))
    assert len(reports) == 10
    assert all(report["id"] for report in reports)
    payload = f"Tax year {TAX_YEAR}\nFiling status: Single".encode() + b" " * (6 * 1024 * 1024)
    assert stage(client, ("files", ("larger.txt", payload, "text/plain")))[0]["id"]


def test_png_and_jpeg_stage_usable_facts_and_require_tax_year_in_the_transcription(client):
    png, jpg, jpeg, unreadable_year, invalid_image = stage(
        client, image_file("wages.png", "PNG"), image_file("income.JPG", "JPEG"),
        image_file("deductions.jpeg", "JPEG"), image_file("unclear.png", "PNG", width=101),
        ("files", ("corrupt.png", b"not an image", "image/png")),
    )
    assert all(report["id"] and report["count"] == 1 for report in (png, jpg, jpeg))
    assert unreadable_year["id"] is None and "was not found" in unreadable_year["error"]
    assert invalid_image["id"] is None and "could not be read" in invalid_image["error"]
    path, original_question = start(client)
    result = sync(client, path, png["id"], jpg["id"], jpeg["id"])
    assert result["state"]["facts"] == {"has_side_income": True}
    result = client.post(path + "/answer", json={"answer": "single", "question_id": original_question}).json()
    assert result["output"]["fact"] == "side_income"
