from datetime import datetime, timezone
from io import BytesIO
import json
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, TableStyle


INK = colors.HexColor("#192B3A")
ACCENT = colors.HexColor("#214F70")
MUTED = colors.HexColor("#526473")
LINE = colors.HexColor("#DCE4E7")
TINT = colors.HexColor("#F1F6F8")
BODY = ParagraphStyle("Body", fontSize=9.5, leading=14, textColor=INK)
HEADER = ParagraphStyle("ColumnHeader", parent=BODY, fontName="Helvetica-Bold", textColor=colors.white)
METADATA = ParagraphStyle("Metadata", parent=BODY, textColor=MUTED, keepWithNext=True)
METADATA_LAST = ParagraphStyle("MetadataLast", parent=METADATA, spaceAfter=14)
TITLE = ParagraphStyle(
    "Title", parent=BODY, fontName="Helvetica-Bold", fontSize=25, leading=30,
    textColor=ACCENT, spaceAfter=12, keepWithNext=True,
)
SECTION = ParagraphStyle(
    "Section", parent=BODY, fontName="Helvetica-Bold", fontSize=12, leading=16,
    textColor=ACCENT, spaceBefore=16, spaceAfter=6, keepWithNext=True,
)
REVIEW_STATUSES = {
    "in_progress": "In progress", "waiting_for_preparer": "Waiting for review",
    "approved": "Approved", "returned": "Returned",
}
FIELD_TERMS = {
    "1098t": "1098-T", "1099q": "1099-Q", "aotc": "AOTC", "ca": "California",
    "hsa": "HSA", "irs": "IRS", "llc": "LLC", "us": "U.S.", "w2": "W-2",
}
UNICODE_FONTS = {
    r"[\u3040-\u30ff]+": "HeiseiKakuGo-W5",
    r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]+": "HYSMyeongJo-Medium",
    r"[\u2e80-\u303f\u3400-\u9fff\uf900-\ufaff\uff00-\uffef]+": "STSong-Light",
}
for font_name in UNICODE_FONTS.values():
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))


def paragraph(value: str, style: ParagraphStyle = BODY) -> Paragraph:
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    markup = "<br/>".join(escape(line) for line in value.splitlines())
    for pattern, font_name in UNICODE_FONTS.items():
        markup = re.sub(pattern, lambda match: f'<font name="{font_name}">{match[0]}</font>', markup)
    return Paragraph(markup, style)


def field_label(key: str) -> str:
    words = key.split("_")
    words = [FIELD_TERMS.get(word, word) for word in words]
    name = " ".join(words)
    return name[:1].upper() + name[1:]


def fact_value(value) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(fact_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def pdf_table(headers: tuple[str, str], rows: list[tuple[str, str]], width: float, *, first_column: float) -> LongTable:
    cells = [[paragraph(heading, HEADER) for heading in headers]]
    cells.extend([paragraph(left), paragraph(right)] for left, right in rows)
    return LongTable(cells, colWidths=(width * first_column, width * (1 - first_column)),
                     repeatRows=1, splitInRow=1, style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))


def page_footer(canvas, document):
    canvas.setStrokeColor(LINE)
    canvas.line(document.leftMargin, 40, letter[0] - document.rightMargin, 40)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(letter[0] - document.rightMargin, 27, f"Page {document.page}")


def pdf_document(buffer: BytesIO, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(buffer, pagesize=letter, title=title,
                             leftMargin=42, rightMargin=42, topMargin=44, bottomMargin=52)


def timestamp(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def render_interview_pdf(facts: dict, *, interview_id: str, title: str | None, created_at: str) -> bytes:
    buffer = BytesIO()
    document = pdf_document(buffer, "Collected Interview Data")
    rows = [(field_label(key), fact_value(value)) for key, value in facts.items()]
    story = [paragraph("Collected Interview Data", TITLE)]
    if title:
        story.append(paragraph(f"Interview title: {title}", METADATA))
    story.extend([
        paragraph(f"Interview ID: {interview_id}", METADATA),
        paragraph(f"Started: {timestamp(created_at)}", METADATA_LAST),
    ])
    if rows:
        story.append(pdf_table(("Data field", "Collected value"), rows, document.width - 12, first_column=0.45))
    else:
        story.append(paragraph("No data collected."))
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return buffer.getvalue()


def review_history_row(entry: dict, prompts: dict[str, str]) -> tuple[str, str]:
    lines = []
    if "background" in entry:
        event = "Assistant background"
        lines.append(entry["background"])
    elif "tax_qa" in entry:
        event = "Tax question"
        response = entry["tax_qa"]
        lines.extend([entry["message"], response["answer"]])
        lines.extend(f'{source["citation"]}: {source["text"]}' for source in response.get("citations", []))
    elif "decision" in entry:
        event = "Preparer decision"
        lines.append(REVIEW_STATUSES.get(entry["decision"], field_label(entry["decision"])))
        if entry.get("note"):
            lines.append(entry["note"])
    elif entry.get("submitted"):
        event = "Assistant submission"
        lines.append(entry.get("note") or "Submitted for review.")
    elif "note" in entry:
        event = "Client note"
        lines.append(entry["note"])
    elif entry.get("parsed_by") == "document":
        event = "Document extraction"
        lines.extend(entry.get("files", []))
    elif "answer_text" in entry:
        event = "Interview answer"
        question_id = entry.get("question_id", "")
        lines.extend([prompts.get(question_id, field_label(question_id)), entry["answer_text"]])
        if not entry.get("facts"):
            lines.append("No facts recorded from this answer.")
    else:
        event = "Attachment uploaded" if entry.get("attachments") else "Interview entry"

    lines.extend(f"{field_label(key)}: {fact_value(value)}" for key, value in entry.get("facts", {}).items())
    if entry.get("attachments"):
        lines.append("Attachments: " + ", ".join(item["name"] for item in entry["attachments"]))
    return event, "\n".join(lines)


def render_review_pdf(review: dict) -> bytes:
    state = review["state"]
    summary = review["summary"] or {}
    labels = review["labels"]
    history = review["history"]
    buffer = BytesIO()
    document = pdf_document(buffer, "Interview Review Report")
    width = document.width - 12
    details = [
        ("Interview ID", state["interview_id"]),
        ("Status", REVIEW_STATUSES.get(state["status"], field_label(state["status"]))),
        ("Source", "AI assistant (MCP)" if review["channel"] == "mcp" else "Web"),
        ("Started", timestamp(review["created_at"])),
        ("Updated", timestamp(review["updated_at"])),
        ("Answers recorded", str(state["answered"])),
    ]
    if review.get("title"):
        details.insert(0, ("Interview title", review["title"]))
    if state.get("priority"):
        details.append(("Asked first", ", ".join(field_label(item) for item in state["priority"])))

    facts = [(labels.get(key, field_label(key)), fact_value(value)) for key, value in state["facts"].items()]
    follow_up = [("Needs review", flag["reason"]) for flag in state["flags"]]
    follow_up.extend(("Document to collect", item) for item in state["missing_docs"])
    follow_up.extend(("Client reminder", item) for item in summary.get("reminders", []))
    follow_up.extend(("Not covered by the playbook", field_label(key)) for key in summary.get("not_yet_covered", []))
    follow_up.extend(("Document conflict", labels.get(key, field_label(key))) for key in review["conflicts"])
    sources = [(category, item) for key, category in (
        ("citations_retrieved", "Retrieved citation"),
        ("references_to_verify", "Playbook reference"),
        ("references_not_in_kb", "Never retrieved"),
        ("references_unchecked", "Retrieval unavailable"),
    ) for item in summary.get(key, [])]
    attachments = [(item["name"], str(item["count"])) for item in review["documents"]]
    record = [review_history_row(entry, review["question_prompts"]) for entry in history]

    story = [paragraph("Interview Review Report", TITLE), pdf_table(("Interview details", "Value"), details, width, first_column=0.3)]
    if not summary:
        story.append(paragraph("This report contains the interview information collected so far."))
    for heading, headers, rows, first_column in (
        ("Information collected", ("Question", "Answer"), facts, 0.6),
        ("Review and follow-up", ("Category", "Details"), follow_up, 0.3),
        ("Sources", ("Category", "Reference"), sources, 0.3),
        ("Uploaded documents", ("Filename", "Extracted answers"), attachments, 0.6),
        ("Interview record", ("Event", "Details"), record, 0.27),
    ):
        if rows:
            story.extend([paragraph(heading, SECTION), pdf_table(headers, rows, width, first_column=first_column)])
    document.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return buffer.getvalue()
