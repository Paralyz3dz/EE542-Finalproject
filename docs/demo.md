# Final demo script (11/30 – 12/1, about 10 minutes)

Fictional clients only. Run from `D:\EE542\tax-agent` with the virtual environment active.

## Before the demo

```powershell
python -m pytest tests -q                       # everything green
python -m uvicorn src.app.api:app --reload      # web server on http://127.0.0.1:8000
```

- Open three tabs: `/interview`, `/preparer`, and Claude Desktop with the `tax-intake` server (see `docs/mcp.md`).
- Start the RDS database and Ollama if showing retrieved passages; otherwise set `TAX_AGENT_RETRIEVAL=off`.
- Optional clean slate: stop the server and delete `logs/interviews.sqlite3`.

## 1. The problem (1 min)

A junior assistant interviews clients for a preparer. They miss follow-ups, and every gap costs the preparer
review time. The tool is a digital clone of the preparer's interview: it does not file and gives no final advice.

## 2. Web interview (3 min) — `/interview`

1. Topic "Self-employment and gig income"; background: `W-2 job, drives Uber on weekends, two kids, travels to craft fairs`.
2. Point at "Asked now because the client mentioned 'drives'": trigger words reorder follow-ups.
3. Open a question's "Why ask" and the official passage with its citation.
4. Answer "mixed" for travel: an escalation appears in the sidebar.
5. Type an unclear answer ("maybe?"): the engine refuses it and asks again.
6. Complete the interview and select **Submit to preparer** on the summary.

## 3. The user's own AI runs the interview (3 min) — Claude Desktop

1. Ask: *"Start a tax intake interview. The client is an F-1 graduate student who works as a TA."*
2. Show that Claude asks one question at a time and declines to give tax advice.
3. Point out the residency questions in the Pub 4012 / Form 8843 order.
4. Explain: no API key of ours; the engine checks every answer the model sends.

## 4. Preparer review (2 min) — `/preparer`

1. The F-1 interview appears under "Waiting for review", badge "AI assistant", next to web interviews.
2. Open it: what needs the preparer (likely nonresident, Form 8843), questions and answers, documents,
   reminders, the full interview record.
3. Download the review PDF, then approve with a note.

## 5. Why trust it (1 min)

- Playbook checked against Form 13614-C (51/51), Direct File knockouts (102/102), the Pub 4012 scope chart,
  and 13 replayed IRS-based and fictional cases (`docs/validation.md`).
- Every question cites its IRS / FTB source; student questions also cite the IRC and Treasury Regulations.
- When unsure, the tool escalates; the preparer decides.

## Backup plan

- No network: `TAX_AGENT_RETRIEVAL=off`; the interview and review still work.
- Claude Desktop not available: use the ChatGPT desktop app (Codex), or show a finished MCP interview
  already in `/preparer`.
