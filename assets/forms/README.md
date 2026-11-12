# Tax form templates

Place a fillable PDF and a matching `.mapping.json` file together, for example:

```text
assets/forms/<jurisdiction>/<year>/<form>.pdf
assets/forms/<jurisdiction>/<year>/<form>.mapping.json
```

The backend discovers configurations on each request. Each configuration needs a unique
`id` containing lowercase letters, digits and hyphens; include jurisdiction, year and
form name to distinguish versions. A PDF alone is not offered until its mapping exists.
No frontend or route changes are needed for another supported template.

This fictional example shows the minimum structure; replace its field names with actual
AcroForm field names and review its recommendation conditions before use:

```json
{
  "id": "example-2026-sample",
  "title": "Example form",
  "tax_year": 2026,
  "recommend_when": [{"fact": "owns_rental", "equals": true}],
  "recommend_reason": "The interview mentions rental property. Confirm applicability.",
  "fields": [
    {
      "key": "owner_name",
      "label": "Owner name",
      "source": "input",
      "type": "text",
      "required": true,
      "max_length": 35,
      "pdf_field": "actual_pdf_field_name"
    }
  ],
  "review": [{"message": "Confirm eligibility and complete calculated amounts."}]
}
```

## Fields

Use `source: "fact"` to read an interview fact with the same `key`, or `source: "input"`
to offer a supplemental input. Clients cannot override facts through supplemental inputs.
`label` is shown to users. `required` marks an unknown value as missing without blocking
partial drafts. False and zero are valid known values.

Supported types are `text` (default), `ssn` (masked input), `money` (nonnegative, up to
999999999.99 with two decimal places), and `checkbox` (unknown/yes/no). Text accepts
printable ASCII by default. Optional `pattern`, `max_length`, `uppercase` and
`strip_hyphens` control text validation and formatting; `ssn` needs its own pattern.

`pdf_field` is the fully qualified PDF field name. Checkbox fields need `on_value`
matching the PDF export value; false writes `/Off`. For choices, use
`choices: {"business_value": ["qualified_pdf_field", "export_value"]}`; this supports
groups such as filing status where each option uses a different PDF field/value.

`when` controls whether a field applies and is displayed. `pdf_when` controls whether
its validated value is written, allowing supplemental inputs to feed other fields.
`join: ["first_name", "last_name"]` joins earlier validated values with spaces; put
derived fields after their inputs and omit `source`. No arbitrary formulas execute.
`distinct: [["ssn", "spouse_ssn"]]` rejects matching nonempty normalized values.

## Conditions and recommendations

Conditions are lists combined with AND, using interview facts:

- `{"fact": "owns_rental", "equals": true}`
- `{"fact": "filing_status", "in": ["married_joint", "married_separate"]}`
- `{"fact": "estimated_tax_paid", "known": true}`

Unknown facts never match. Omitted field/review conditions always apply. A recommendation
requires a nonempty `recommend_when`; absent rules leave the form available for manual
selection. Each `review` item contains a `message` and optional `when` conditions.

Recommendations express relevance, not a final filing obligation. Users must select a
template and confirm its year before generating it. Rules for additional official tax
forms must be reviewed when configuring those forms. Calculations and schedule transfers
are not inferred, and LLM/RAG do not guess mappings or recommendation rules.

## Check a new template

Read its field names with `pypdf.PdfReader(path).get_fields()`, configure text and checkbox
export values, and verify a fictional completed PDF visually. Scanned/flat PDFs need a
fillable template first; coordinate overlays and OCR are not currently supported.
The initial 1040 mapping includes separate-return spouse-name placement and keeps
ambiguous income, dependent details and calculated amounts for preparer review.
