# Escalation rules: one-page review (optional)

Status: optional. The project does not depend on this review; see `docs/validation.md` for the checks done without it.

Each rule sends the case to the preparer when the client's answer matches. Reviewer: please mark each row
`OK` (should go to the preparer), `NO` (a volunteer can handle it) or leave a short comment. About 10 minutes.

| # | Module | When the client says | Why it goes to the preparer | OK / NO / comment |
|---|---|---|---|---|
| 1 | base_intake | Did you live in California for all of last year? → no | Part-year California residency (Form 540NR) needs the preparer. | |
| 2 | base_intake | Did you live or work in more than one state last year? → yes | Multi-state income needs the preparer. | |
| 3 | base_intake | Were you a U.S. citizen for all of last year? → no | Non-citizen: resident or nonresident alien status must be decided by the preparer. | |
| 4 | base_intake | Did you own, receive, sell or trade any digital assets, such as cryptocurrency, last year? → yes | Digital-asset transactions need the preparer. | |
| 5 | base_intake | Did you sell any stocks, bonds or real estate? → yes | Investment or real-estate sales need the preparer. | |
| 6 | base_intake | Did you have any gambling or lottery winnings? → yes | Gambling winnings and losses need the preparer. | |
| 7 | base_intake | Did you receive any other money, such as cash payments, jury duty pay, awards or royalties? → yes | Other income types need the preparer to decide how to report them. | |
| 8 | base_intake | Did you buy health insurance through Covered California or the Marketplace? → yes | Premium tax credit reconciliation (Form 8962) needs the preparer. | |
| 9 | base_intake | Did you buy and install energy-saving home items, such as windows, a furnace or insulation? → yes | Energy credit (Form 5695) needs the preparer. | |
| 10 | base_intake | Did you sell a home? → yes | Home sale needs the preparer. | |
| 11 | base_intake | Was any credit card, mortgage or other debt cancelled or forgiven? → yes | Cancelled debt needs the preparer. | |
| 12 | base_intake | Did you have a loss from a federally declared disaster? → yes | Disaster losses need the preparer. | |
| 13 | base_intake | Did the IRS ever deny a credit you claimed, such as the earned income credit or child tax credit? → yes | A previously disallowed credit needs the preparer. | |
| 14 | base_intake | Did you receive any letter or bill from the IRS? → yes | An IRS letter or bill needs the preparer. | |
| 15 | base_intake | Are you in a registered domestic partnership in California? → yes | California registered domestic partners file differently for federal and state returns; the preparer handles it. | |
| 16 | base_intake | Did you earn money from farming, ranching or commercial fishing? → yes | Farm or fishing income (Schedule F) is outside the volunteer scope; the preparer reviews it. | |
| 17 | base_intake | Did you have a bank or investment account outside the U.S., or receive income from another country? → yes | Foreign accounts or foreign income may need FinCEN 114, Form 8938 or a foreign tax credit; the preparer reviews it. | |
| 18 | base_intake | What kind of retirement account did you put money into: a 401(k) or 403(b) at work, a traditional IRA, a Roth IRA, or a plan for the self-employed (SEP, SIMPLE, solo 401(k))? → self employed plan | SEP, SIMPLE or solo 401(k) contributions are outside the volunteer scope; the preparer reviews them. | |
| 19 | base_intake | What kind of retirement account did you put money into: a 401(k) or 403(b) at work, a traditional IRA, a Roth IRA, or a plan for the self-employed (SEP, SIMPLE, solo 401(k))? → not sure | The type of retirement contribution is unknown; the preparer checks Form 5498 or the W-2 before deciding on a deduction. | |
| 20 | side_gig | Were you paid in cryptocurrency or other digital assets for any of this work? → yes | Payment in cryptocurrency needs the preparer (valuation and later sale). | |
| 21 | side_gig | What were you selling: things you made yourself, goods you bought to resell, or your own used personal items? → personal items | Sales of personal items: gains go on Schedule D and losses are not deductible; preparer decides how to report the 1099-K. | |
| 22 | side_gig | Do you sell regularly to make money, rather than as a hobby? → no | Hobby or business must be decided by the preparer. | |
| 23 | side_gig | Do you buy materials or goods and keep stock on hand to sell? → yes | Inventory / cost of goods sold needs the preparer (outside the simple Schedule C). | |
| 24 | side_gig | How would you like to count car costs: by business miles, or by actual costs like gas, repairs and depreciation? → actual costs | Actual car expenses and depreciation need the preparer. | |
| 25 | side_gig | Was that travel only for the business, partly personal, or mostly personal? → mixed | Mixed business and personal travel needs the preparer to split the costs. | |
| 26 | side_gig | Do you use part of your home only for the side work? → yes | Business use of home needs the preparer. | |
| 27 | side_gig | Did you pay anyone to help, as an employee or a contractor? → yes | Employees or contract labor need the preparer. | |
| 28 | side_gig | Did you buy equipment or tools for the business costing more than $2,500 each? → yes | Depreciation / large asset purchases need the preparer. | |
| 29 | side_gig | Did the side work lose money overall, with costs higher than income? → yes | A business loss needs the preparer. | |
| 30 | side_gig | Are you thinking about forming an LLC or company for the side work? → yes | Entity choice and California's minimum LLC tax are planning questions for the preparer. | |
| 31 | family | Do you share custody of a child with a parent who does not live with you? → yes | Shared custody: the preparer decides who may claim the child (Form 8332 rules). | |
| 32 | family | Could anyone else, such as the other parent or a grandparent, also claim any of these people? → yes | More than one person could claim the dependent: tie-breaker rules need the preparer. | |
| 33 | family | Did a nanny or other caregiver work in your home, paid by you? → yes | Household employee (nanny tax) needs the preparer. | |
| 34 | family | Was any of the care given by a family member, such as your own child, your spouse, or someone you claim as a dependent? → yes | Care paid to the client's own child under 19, spouse or dependent may not count for the credit; the preparer decides. | |
| 35 | retiree | Were you younger than 59 and a half when you took the money? → yes | Early distribution: exceptions and the federal 10% / California 2.5% additional taxes need the preparer. | |
| 36 | retiree | Did you move any of that money into another retirement account? → yes | Rollover needs the preparer to confirm it was completed correctly. | |
| 37 | rental | owns_rental → yes | Rental income (Schedule E) is outside the simple scope; the preparer reviews it. | |
| 38 | rental | Is the rental property in California? → no | Out-of-state rental usually needs a nonresident return in that state. | |
| 39 | rental | Did you or your family stay there yourselves for more than 14 days, or more than 10% of the days it was rented? → yes | Vacation-home rules (personal use) need the preparer. | |
| 40 | rental | Were most guests' stays a week or shorter, for example through Airbnb? → yes | Short-term rental treatment needs the preparer. | |
| 41 | rental | Did you sell the rental property last year? → yes | Sale of rental property needs the preparer. | |
| 42 | rental | Did you also rent out personal property, such as a car or equipment? → yes | Personal-property rental needs the preparer. | |

Reviewer (optional): name or role, date. Do not include any client information.
