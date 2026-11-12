import json
import re
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


FORMS_ROOT = Path(__file__).resolve().parents[2] / "assets/forms"


class FormRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tax_year: int = Field(ge=1900, le=2200)
    confirmed_tax_year: Literal[True]
    values: dict[str, str | bool | None] = Field(default_factory=dict, max_length=100)


def load_forms() -> dict:
    forms = {}
    for path in sorted(FORMS_ROOT.rglob("*.mapping.json")):
        definition = json.loads(path.read_text(encoding="utf-8"))
        form_id = definition["id"]
        if not re.fullmatch(r"[a-z0-9-]+", form_id) or form_id in forms:
            raise ValueError("Tax form IDs must be unique lowercase letters, digits and hyphens.")
        template = path.with_name(path.name.removesuffix(".mapping.json") + ".pdf")
        if not template.is_file():
            raise ValueError(f"Missing PDF for {form_id}.")
        forms[form_id] = definition | {"template": template}
    return forms


def matches(conditions: list | None, facts: dict) -> bool:
    for condition in conditions or []:
        value = facts.get(condition["fact"])
        if value is None:
            return False
        if condition.get("known"):
            continue
        if "in" in condition:
            if value not in condition["in"]:
                return False
        elif value != condition["equals"]:
            return False
    return True


def form_catalog(facts: dict) -> list:
    return [
        {"id": form_id, "title": form["title"], "tax_year": form["tax_year"],
         "recommended": bool(form.get("recommend_when")) and matches(form["recommend_when"], facts),
         "reason": form.get("recommend_reason", ""),
         "inputs": [field for field in form["fields"]
                    if field.get("source") == "input" and matches(field.get("when"), facts)]}
        for form_id, form in load_forms().items()
    ]


def validate_value(field: dict, value):
    if value is None or value == "":
        return None
    kind = field.get("type", "text")
    if kind == "money":
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            raise ValueError(f"{field['label']}: enter a valid amount.") from None
        if not amount.is_finite() or not 0 <= amount <= Decimal("999999999.99") or amount.as_tuple().exponent < -2:
            raise ValueError(f"{field['label']}: enter a nonnegative amount with at most two decimal places.")
        return format(amount, "f")
    if kind == "checkbox":
        if not isinstance(value, bool):
            raise ValueError(f"{field['label']}: choose yes or no.")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field['label']}: enter text.")
    value = value.strip()
    if not value:
        return None
    if field.get("uppercase"):
        value = value.upper()
    if len(value) > field.get("max_length", 100) or not re.fullmatch(field.get("pattern", r"[\x20-\x7E]*"), value):
        raise ValueError(f"{field['label']}: invalid format or length.")
    if field.get("strip_hyphens"):
        value = value.replace("-", "")
    if "choices" in field and value not in field["choices"]:
        raise ValueError(f"{field['label']}: unsupported choice.")
    return value


def prepare_form(form: dict, facts: dict, payload: FormRequest) -> dict:
    if payload.tax_year != form["tax_year"]:
        raise ValueError("The confirmed tax year does not match this template.")
    input_keys = {field["key"] for field in form["fields"] if field.get("source") == "input"}
    if payload.values.keys() - input_keys:
        raise ValueError("Unknown supplemental fields for this form.")
    values, fields, missing = {}, {}, []
    for field in form["fields"]:
        key = field["key"]
        if not matches(field.get("when"), facts):
            if payload.values.get(key) not in (None, ""):
                raise ValueError(f"{field['label']}: not applicable to this interview.")
            continue
        if "join" in field:
            raw = " ".join(values.get(part) or "" for part in field["join"]).strip()
        else:
            raw = (payload.values if field.get("source") == "input" else facts).get(key)
        value = validate_value(field, raw)
        values[key] = value
        if value is None:
            if field.get("required"):
                missing.append(key)
            continue
        if not matches(field.get("pdf_when"), facts):
            continue
        if "choices" in field:
            name, exported = field["choices"][value]
            fields[name] = exported
        elif "pdf_field" in field:
            fields[field["pdf_field"]] = (field.get("on_value", "/Yes") if value else "/Off") if field.get("type") == "checkbox" else value
    for left, right in form.get("distinct", []):
        if values.get(left) and values[left] == values.get(right):
            raise ValueError(f"{left} and {right} must differ.")
    review = ["Partial draft only. Blank fields are unknown, not zero."]
    review.extend(rule["message"] for rule in form.get("review", []) if matches(rule.get("when"), facts))
    return {"form_id": form["id"], "title": form["title"], "tax_year": form["tax_year"],
            "fields": fields, "missing": missing, "review": review}


def fill_form(form: dict, fields: dict) -> bytes:
    reader = PdfReader(form["template"])
    available = reader.get_fields() or {}
    if not available or not fields.keys() <= available.keys():
        raise ValueError("The PDF must have fillable fields matching its configuration.")
    writer = PdfWriter(clone_from=reader)
    writer.root_object["/AcroForm"].pop("/XFA", None)
    writer.update_page_form_field_values(None, fields, auto_regenerate=False)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    for page in writer.pages:
        if "/Resources" not in page:
            page[NameObject("/Resources")] = DictionaryObject()
        resources = page["/Resources"]
        if "/Font" not in resources:
            resources[NameObject("/Font")] = DictionaryObject()
        resources["/Font"][NameObject("/DraftFont")] = font
        content = page.get_contents()
        banner = DecodedStreamObject()
        banner.set_data((content.get_data() if content is not None else b"") + b"\nq BT /DraftFont 9 Tf 1 0 0 rg 36 8 Td (DRAFT - PARTIAL PREFILL - NOT FOR FILING) Tj ET Q")
        page.replace_contents(banner)
    writer.add_metadata({"/Title": f"{form['tax_year']} {form['title']} - Partial draft"})
    output = BytesIO()
    writer.write(output)
    return output.getvalue()
