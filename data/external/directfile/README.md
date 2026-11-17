# IRS Direct File: extracted interview questions

Source: [IRS-Public/direct-file](https://github.com/IRS-Public/direct-file), commit `e365f9f` (2026-06-11).
Public domain in the U.S. and CC0 1.0 worldwide. Tax law as of **tax year 2024**, federal only.

Extracted with `scripts/directfile/df_extract.py` (flow screens + fact dictionary + English text) and
`scripts/directfile/df_digest.py` (split into the files below). The scripts only read the repository files.

| File | Rows | What it is | Use in this project |
|---|---|---|---|
| `directfile_interview_questions.csv` | 301 | Questions about the taxpayer's situation, with the condition that shows each one (`show_when`) and answer options | Wording and follow-up logic for playbook modules (stages 3-4) |
| `directfile_document_fields.csv` | 166 | Box-by-box entry from W-2, 1099-R, SSA-1099, 1099-INT, 1099-G | Field mapping for document extraction (stage 1) |
| `directfile_knockouts.csv` | 102 | Situations Direct File does not support, with the condition that triggers each | Candidates for escalation rules ("send to the preparer") |
| `directfile_screens.json` | 704 screens | Full extraction (all screens, inputs, conditions) | Reference |
| `directfile_writable_facts.json` | 605 facts | Every fact a taxpayer can enter, with type, options and module | Reference for fact names and types |

Not covered by Direct File (so not in these files): self-employment / sole proprietorship, gig work (1099-K, 1099-NEC),
rentals, IRA distributions, capital gains, state returns. Those playbook modules come from IRS instructions,
Form 13614-C and the practitioner interview instead.

Re-run (paths are examples):

```powershell
git clone --depth 1 https://github.com/IRS-Public/direct-file.git ..\direct-file
python scripts/directfile/df_extract.py ..\direct-file\direct-file data/external/directfile
python scripts/directfile/df_digest.py data/external/directfile
```
