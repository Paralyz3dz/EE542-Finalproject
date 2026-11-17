# Case replay

One JSON file per client, run by `tests/test_cases.py` from the first question to the preparer summary.
IRS-based cases (`ats*`, `f6744_*`) follow the official test scenarios' facts, moved to California with names
and identification numbers dropped; amounts are placeholders where the source gives none. No real client data.

| Field | Meaning |
|---|---|
| `name`, `source`, `official_outcome` | the client, the IRS material it is based on, and what that material says the correct handling is |
| `priority`, `background` | what the assistant enters before starting |
| `answers` | fact -> answer. Facts not listed get the typical answer: yes to `ca_resident_full_year` and `is_us_citizen`, otherwise no, the first option, or 0 |
| `expect` | `flags` (exact set of escalation rule ids), `modules_asked`, `modules_not_asked`, optional `first_questions`, `documents_contain` |
