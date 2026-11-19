"""MCP server: lets the user's own AI assistant (Claude Desktop, Cursor, ...) drive the guided interview.

The assistant's model talks to the client and turns replies into answers; this package owns the
interview flow (next question, answer checks, escalations, summary). No API key of ours is used.
"""
