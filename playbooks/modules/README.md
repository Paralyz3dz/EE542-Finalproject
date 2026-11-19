# Playbook modules

Every client gets `base_intake`. Its answers switch the topic modules on through `activate_when`:
`side_gig`, `family`, `retiree`, `rental` and `student` (opened by "full-time student" = yes).

## module.json

| Field | Meaning |
|---|---|
| `id`, `title`, `status` | `status`: draft / reviewed |
| `activate_when` | Conditions (same format as `ask_when`) that switch the module on. Omit for `base_intake`. |
| `keywords` | Words in the assistant's notes that suggest this module (for topic detection later) |
| `facts` | `{"name", "type": bool / money / enum / list, "options" (enum), "required", "ask_when"}`. Enum options are lower-case snake_case keys (`f_visa`, `not_sure`): answers are normalized to that form |
| `escalation_rules` | `{"id", "when": [conditions], "reason"}`: send to the preparer |
| `reminders` | `{"id", "when": [conditions], "text"}`: record-keeping reminders shown in the summary |
| `triggers` | `{"id", "keywords", "ask_first": [facts]}`: a word in the background or a client note asks these facts next |

## questions.json

One question per fact:

| Field | Meaning |
|---|---|
| `id` | `q_<fact>` |
| `fact` | the fact this question fills |
| `question` | what the assistant asks the client, in plain language |
| `why_ask` | one sentence the assistant can say if the client asks why |
| `required_docs` | documents to request |
| `retrieval_query` | search text for the knowledge base |
| `source` | official document where the rule lives, using the knowledge-base names (e.g. `IRS Schedule C Instructions`) |
| `origin` | where the question came from: `Form 13614-C Part III`, `Direct File`, `Pub 4012 D-xx`, `advisor interview 2026-11-11`, `LLM draft` |
| `status` | derived_official / advisor_input / llm_draft / preparer_confirmed |

## Conditions

`[{"fact": "has_self_employment", "equals": true}]`: every item must hold; an unknown fact (null) never matches.

## Sources

- `playbooks/coverage_13614c.csv`: every Form 13614-C item mapped to a module, fact and handling
- `data/external/directfile/`: IRS Direct File questions, document fields and knockouts (CC0)
- `data/raw/irs/`: Form 13614-C, Pub 4012, Form 6744, Pub 5101
- `student`: Pub 970, Pub 4012 Tab J (education) and Tab L (resident / nonresident decision chart for students), Form 8843 (2025) Parts I and III, Pub 519
