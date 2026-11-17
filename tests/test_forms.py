from io import BytesIO

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from src.app.forms import FormRequest, fill_form, form_catalog, load_forms, prepare_form


FORM = load_forms()["federal-2025-1040"]
MAPPING = {"tax_year": FORM["tax_year"], "fields": {field["key"]: field["pdf_field"] for field in FORM["fields"] if "pdf_field" in field}, "filing_status": FORM["fields"][-1]["choices"]}
TEMPLATE = FORM["template"]


def prepare_1040(facts, payload):
    return prepare_form(FORM, facts, payload)


def fill_1040(fields):
    return fill_form(FORM, fields)


def request(**values):
    return FormRequest(tax_year=2025, confirmed_tax_year=True, values=values)


@pytest.mark.parametrize("status", MAPPING["filing_status"])
def test_pdf_values_appearances_and_filing_status(status):
    data = prepare_1040({"filing_status": status, "w2_wages": 12345.67}, request(
        first_name="Alex", last_name="Example", ssn="123-45-6789", w2_federal_withholding="0"))
    pdf = PdfReader(BytesIO(fill_1040(data["fields"])))
    fields = pdf.get_fields()
    assert len(pdf.pages) == 2
    assert fields["topmostSubform[0].Page1[0].f1_14[0]"]["/V"] == "Alex"
    assert fields["topmostSubform[0].Page1[0].f1_47[0]"]["/V"] == "12345.67"
    assert fields["topmostSubform[0].Page2[0].f2_17[0]"]["/V"] == "0"
    selected, value = MAPPING["filing_status"][status]
    for name, _ in MAPPING["filing_status"].values():
        assert fields[name]["/V"] == (value if name == selected else "/Off")
    for page in pdf.pages:
        assert b"DRAFT - PARTIAL PREFILL" in page.get_contents().get_data()
        for ref in page["/Annots"]:
            widget = ref.get_object()
            if widget.get("/T") == "f1_14[0]":
                assert b"Alex" in widget["/AP"]["/N"].get_data()
            if widget.get("/FT") == "/Btn" and widget.get("/V") not in (None, "/Off"):
                assert widget["/AS"] == widget["/V"]
    assert pdf.trailer["/Root"]["/AcroForm"]["/NeedAppearances"].value is False
    assert fields["topmostSubform[0].Page1[0].f1_16[0]"]["/V"] == "123456789"


def test_missing_values_and_ambiguous_income_stay_blank():
    facts = {"filing_status": "single", "retirement_distributions": 9000,
             "has_retirement_distribution": True, "estimated_tax_paid": 100,
             "has_dependents": True, "wages": 8000, "social_security_benefits": 12000}
    data = prepare_1040(facts, request())
    assert "ssn" in data["missing"]
    assert "w2_federal_withholding" in data["missing"]
    assert data["fields"]["topmostSubform[0].Page1[0].f1_68[0]"] == "12000"
    assert "topmostSubform[0].Page1[0].f1_47[0]" not in data["fields"]
    pdf = PdfReader(BytesIO(fill_1040(data["fields"])))
    fields = pdf.get_fields()
    for name in ("Page1[0].f1_62[0]", "Page1[0].f1_65[0]", "Page2[0].f2_17[0]", "Page2[0].f2_21[0]"):
        assert not fields["topmostSubform[0]." + name].get("/V")
    assert any("Dependent details" in item for item in data["review"])


def test_separate_return_spouse_name_goes_to_mfs_line():
    data = prepare_1040({"filing_status": "married_separate"}, request(
        spouse_first_name="Sam", spouse_last_name="Example", spouse_ssn="987-65-4321"))
    fields = data["fields"]
    assert fields["topmostSubform[0].Page1[0].Checkbox_ReadOrder[0].f1_28[0]"] == "Sam Example"
    assert fields["topmostSubform[0].Page1[0].f1_19[0]"] == "987654321"
    assert "topmostSubform[0].Page1[0].f1_17[0]" not in fields
    with pytest.raises(ValueError):
        prepare_1040({"filing_status": "single"}, request(spouse_first_name="Sam"))


def test_full_address_and_joint_spouse_round_trip():
    payload = request(first_name="Alex", last_name="Example", street="123 Sample Street",
                      apartment="2B", city="Los Angeles", state="CA", zip_code="90001-1234",
                      spouse_first_name="Sam", spouse_last_name="Example", spouse_ssn="987-65-4321")
    data = prepare_1040({"filing_status": "married_joint"}, payload)
    fields = PdfReader(BytesIO(fill_1040(data["fields"]))).get_fields()
    for key in ("street", "apartment", "city", "state", "zip_code", "spouse_first_name", "spouse_last_name"):
        assert fields[MAPPING["fields"][key]]["/V"] == payload.values[key]


@pytest.mark.parametrize("values", [
    {"ssn": "bad"}, {"state": "XX"}, {"zip_code": "123"},
    {"w2_federal_withholding": "NaN"}, {"w2_federal_withholding": "-1"},
    {"w2_federal_withholding": "1.001"}, {"w2_federal_withholding": "1000000000"},
    {"ssn": "123456789", "spouse_ssn": "123-45-6789"},
    {"unconfigured_field": "value"},
])
def test_invalid_input(values):
    with pytest.raises(ValueError):
        prepare_form(FORM, {"filing_status": "married_joint"}, request(**values))


def test_confirmed_year_must_match_selected_template():
    with pytest.raises(ValueError):
        prepare_form(FORM, {}, FormRequest(tax_year=2026, confirmed_tax_year=True))
    with pytest.raises(ValidationError):
        FormRequest(tax_year=2025, confirmed_tax_year=False)


def test_mapping_matches_official_template():
    fields = PdfReader(TEMPLATE).get_fields()
    assert MAPPING["tax_year"] == 2025
    for name in MAPPING["fields"].values():
        assert name in fields
    for name, value in MAPPING["filing_status"].values():
        assert value in fields[name]["/_States_"]


def test_additional_form_and_year_work_from_configuration(tmp_path, monkeypatch):
    import json
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, TextStringObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    widget = DictionaryObject({NameObject("/Type"): NameObject("/Annot"),
                               NameObject("/Subtype"): NameObject("/Widget"),
                               NameObject("/FT"): NameObject("/Tx"),
                               NameObject("/T"): TextStringObject("sample_name"),
                               NameObject("/Rect"): ArrayObject([FloatObject(n) for n in [36, 700, 300, 720]]),
                               NameObject("/DA"): TextStringObject("/Helv 12 Tf 0 g")})
    ref = writer._add_object(widget)
    page[NameObject("/Annots")] = ArrayObject([ref])
    writer.root_object[NameObject("/AcroForm")] = DictionaryObject({
        NameObject("/Fields"): ArrayObject([ref]),
        NameObject("/DR"): DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/Helv"): font})}),
    })
    writer.write(tmp_path / "sample.pdf")
    definition = {
        "id": "sample-2026-example", "title": "Fictional test form", "tax_year": 2026,
        "recommend_when": [{"fact": "owns_rental", "equals": True}],
        "recommend_reason": "A fictional test recommendation.",
        "fields": [{"key": "owner_name", "label": "Owner name", "source": "input", "required": True,
                    "pdf_field": "sample_name"}],
    }
    (tmp_path / "sample.mapping.json").write_text(json.dumps(definition), encoding="utf-8")
    monkeypatch.setattr("src.app.forms.FORMS_ROOT", tmp_path)
    form = load_forms()["sample-2026-example"]
    assert not form_catalog({})[0]["recommended"]
    assert not form_catalog({"owns_rental": False})[0]["recommended"]
    assert form_catalog({"owns_rental": True})[0]["recommended"]
    assert form_catalog({})[0]["inputs"][0]["key"] == "owner_name"
    assert prepare_form(form, {}, FormRequest(tax_year=2026, confirmed_tax_year=True))["missing"] == ["owner_name"]
    payload = FormRequest(tax_year=2026, confirmed_tax_year=True, values={"owner_name": "Alex"})
    prepared = prepare_form(form, {}, payload)
    pdf = PdfReader(BytesIO(fill_form(form, prepared["fields"])))
    assert pdf.get_fields()["sample_name"]["/V"] == "Alex"
    assert pdf.metadata.title == "2026 Fictional test form - Partial draft"
    (tmp_path / "next.pdf").write_bytes((tmp_path / "sample.pdf").read_bytes())
    next_year = definition | {"id": "sample-2027-example", "tax_year": 2027}
    (tmp_path / "next.mapping.json").write_text(json.dumps(next_year), encoding="utf-8")
    assert {item["tax_year"] for item in form_catalog({})} == {2026, 2027}
    assert len(load_forms()) == 2


def test_catalog_filters_conditional_inputs_and_recommendation():
    assert not form_catalog({})[0]["recommended"]
    single = form_catalog({"filing_status": "single"})[0]
    married = form_catalog({"filing_status": "married_joint"})[0]
    assert single["recommended"]
    assert "spouse_ssn" not in {item["key"] for item in single["inputs"]}
    assert "spouse_ssn" in {item["key"] for item in married["inputs"]}


def test_generic_checkbox_preserves_false():
    form = {"id": "test", "title": "Test", "tax_year": 2025, "fields": [
        {"key": "consent", "label": "Consent", "source": "input", "type": "checkbox",
         "required": True, "pdf_field": "consent", "on_value": "/1"}]}
    payload = FormRequest(tax_year=2025, confirmed_tax_year=True, values={"consent": False})
    data = prepare_form(form, {}, payload)
    assert data["fields"] == {"consent": "/Off"}
    assert data["missing"] == []
