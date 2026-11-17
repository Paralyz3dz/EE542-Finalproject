# Validating the interview playbook without a preparer's sign-off

As of 2026-11-16. Playbook: `playbooks/modules/` (5 modules, 104 questions, 42 escalation rules).

No licensed preparer has reviewed the playbook question by question, and the project does not wait
for one. Instead, the playbook is checked against four sets of IRS material that already state what a
volunteer may handle and what a correct interview finds. Every check is repeatable from files in the repo.

## 1. Checks

| Check | Material (all public IRS) | What is checked | Result | Evidence |
|---|---|---|---|---|
| Intake coverage | Form 13614-C (Rev. 10-2025), the VITA intake sheet | Every item is asked, escalated or marked out of scope | 51 of 51 items mapped | `playbooks/coverage_13614c.csv` |
| Situations the IRS's own software refuses | IRS Direct File knockouts (102, tax year 2024, federal) | Each one is escalated, asked, or left to the preparer's document review | 102 of 102 handled: 26 escalate, 23 asked (in scope), 45 document review, 8 not applicable (Direct File product limits) | `playbooks/validation/directfile_knockouts_check.csv` |
| Volunteer scope | Pub 4012 Scope of Service chart (Rev. 10-2026, structure only) | Each out-of-scope situation relevant to an individual return | 43 checked: 34 escalate, 6 document review, 1 asked, 2 gaps noted (section 4) | `playbooks/validation/pub4012_scope_check.csv` |
| Cases with known outcomes | IRS MeF ATS test scenarios 12 and 3 (TY2025); Form 6744 VITA test, Advanced scenarios 2, 3, 5, 6, 7 (interview notes) | The interview opens the right modules and sends exactly the right situations to the preparer | 9 of 9 cases pass (7 IRS-based, 2 fictional) | `tests/cases/*.json`, `tests/test_cases.py` |

"Document review" means the situation is visible only on a form box (W-2 box 12 codes, 1099-R distribution
codes, 1099-INT boxes). The interview asks for the document (`missing_docs`); the preparer reads the boxes.
An intake interview should not ask clients about box codes they cannot know.

## 2. Gaps these checks found, and what was added

Two of the nine cases failed on the earlier playbook: ATS scenario 3 (farm income was caught only if the
client happened to mention it under "any other money") and Form 6744 Advanced scenario 2 (care paid to the
couple's own 17-year-old son was not noticed). The additions below close those gaps; question origins are
recorded in each question's `origin` field.

| Added question | Module | Escalates when | Found by |
|---|---|---|---|
| Are you in a registered domestic partnership in California? (asked if never married) | base_intake | yes | Direct File knockout `registered-domestic-partner-ko` |
| Did you earn money from farming, ranching or commercial fishing? | base_intake | yes | Pub 4012 (Schedule 1 line 6 out of scope); ATS scenario 3 |
| Did you have a bank or investment account outside the U.S., or receive income from another country? | base_intake | yes | Direct File `foreign-accounts-trusts-knockout`; Pub 4012 (Schedule B, FinCEN 114 / Form 8938) |
| What kind of retirement account did you put money into? (asked after a yes to retirement contributions) | base_intake | self-employed plan, or not sure | Direct File `ira-contributions-ko`; Pub 4012 (Schedule 1 lines 16 and 20) |
| Was any of the care given by a family member (your child, spouse or dependent)? | family | yes | Form 6744 Advanced scenario 2; Pub 503 |
| Did you pay for your own health insurance, not through an employer's plan? | side_gig | never (information for Form 7206) | ATS scenario 12 |

## 3. Design rule: when unsure, send it to the preparer

An unnecessary escalation costs the preparer a minute; a missed one can mean a wrong return. So several
rules escalate situations that Pub 4012 allows certified volunteers to handle. This is deliberate:

| Rule | Pub 4012 says | Why the playbook escalates anyway |
|---|---|---|
| `esc_sold_investments` | Stock and fund sales in scope (advanced) | Basis and asset type cannot be judged in an intake |
| `esc_has_digital_assets` | In scope only when the 1040 digital-asset answer is No | The intake question cannot tell holding from trading |
| `esc_has_marketplace_insurance` | Premium tax credit in scope (advanced), with exceptions | Allocation and alternative-calculation cases are common |
| `esc_sold_home` | Sale of a main home in scope (advanced), with exceptions | Business or rental use of the home changes everything |
| `esc_had_debt_forgiven` | Credit card debt in scope (advanced); other debt out | The intake does not separate the kinds of debt |
| `esc_has_gambling_winnings` | In scope | Professional gambling and losses need judgment |
| `esc_under_59_half_at_distribution` | Form 5329 Part I in scope | Exceptions to the 10% tax need advanced certification |
| `esc_retirement_contribution_not_sure` | Deductible IRA contributions in scope (advanced) | An unknown account type cannot be judged |

## 4. What is still not verified

- No licensed preparer has signed off. Questions are `derived_official` (89, from IRS/FTB material) or
  `advisor_input` (15, from the 2026-11-11 practitioner interview). `preparer_confirmed`: 0.
- Pub 4012 and Form 6744 are 2026 editions (the 2025 editions are not archived); they were used for scope and
  case structure only. Rules and citations use tax year 2025.
- Direct File covers tax year 2024, federal only; it does not cover self-employment, rentals or California.
- California-specific checks rest on FTB documents and the registered domestic partner question only.
- Case amounts are placeholders where the source gives none; the cases test which questions are asked and
  what is escalated, not tax amounts.
- Two Pub 4012 limits are not asked: Schedule C bartering, accounting method other than cash, returns and
  allowances, material participation and leased vehicles (the preparer reviews the business records), and
  adoption expenses (Form 8839; Form 13614-C has no such question).

## 5. Optional outside review

`docs/escalation-review.md` lists the 42 escalation rules on one page (about 10 minutes) for anyone with
VITA training or tax practice experience: the interviewed advisor, a USC VITA volunteer, or the course staff.
It improves confidence but is not a dependency.

## 6. Re-run

```powershell
python -m pytest tests/test_cases.py tests/test_workflow.py -q
```

`test_modules_are_consistent` (in `tests/test_workflow.py`) also fails if a rule or question refers to a fact
that does not exist. When a module changes, add a case to `tests/cases/` and update the two CSV checks.
