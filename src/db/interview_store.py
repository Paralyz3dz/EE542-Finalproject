from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import psycopg

from src.workflow.state import State


@dataclass
class Interview:
    state: State
    output: dict
    documents: dict = field(default_factory=dict)
    conflicts: list = field(default_factory=list)
    version: int = 0

    def payload(self):
        return json.dumps({
            "state": self.state.to_dict(),
            "output": self.output,
            "documents": self.documents,
            "conflicts": self.conflicts,
        })


class InterviewStore:
    def __init__(self, *, path=":memory:", postgres_url=None):
        self._postgres_url = postgres_url
        self._memory = None
        self._path = path
        if not postgres_url:
            if path == ":memory:":
                self._memory = sqlite3.connect(path, check_same_thread=False)
            else:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            if postgres_url:
                conn.execute("SELECT pg_advisory_xact_lock(452190852)")
            else:
                conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS guided_interviews (
                    interview_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload TEXT NOT NULL
                )
            """)
            if postgres_url:
                conn.execute("ALTER TABLE guided_interviews ADD COLUMN IF NOT EXISTS title TEXT")
            elif "title" not in {column[1] for column in conn.execute("PRAGMA table_info(guided_interviews)")}:
                conn.execute("ALTER TABLE guided_interviews ADD COLUMN title TEXT")
            conn.execute("""
                CREATE INDEX IF NOT EXISTS guided_interviews_owner_updated
                ON guided_interviews (owner, updated_at DESC, interview_id)
            """)

    @contextmanager
    def _connection(self):
        if self._postgres_url:
            with psycopg.connect(self._postgres_url, connect_timeout=10) as conn:
                yield conn
        elif self._memory is not None:
            with self._memory as conn:
                yield conn
        else:
            conn = sqlite3.connect(self._path, timeout=10)
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def _execute(self, conn, query, params):
        return conn.execute(query.replace("?", "%s") if self._postgres_url else query, params)

    def create(self, owner: str, interview: Interview):
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connection() as conn:
            self._execute(conn, """
                INSERT INTO guided_interviews
                (interview_id, owner, status, created_at, updated_at, version, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (interview.state.interview_id, owner, interview.state.status, now, now,
                  interview.version, interview.payload()))

    def get(self, owner: str, interview_id: str) -> Interview | None:
        with self._connection() as conn:
            row = self._execute(conn, """
                SELECT payload, version FROM guided_interviews
                WHERE owner = ? AND interview_id = ?
            """, (owner, interview_id)).fetchone()
        if row is None:
            return None
        payload, version = row
        data = json.loads(payload)
        data["state"] = State.from_dict(data["state"])
        return Interview(**data, version=version)

    def list(self, owner: str) -> list[dict]:
        with self._connection() as conn:
            rows = self._execute(conn, """
                SELECT interview_id, status, created_at, updated_at, title
                FROM guided_interviews WHERE owner = ?
                ORDER BY updated_at DESC, interview_id
            """, (owner,)).fetchall()
        return [
            {"interview_id": interview_id, "status": status, "created_at": created_at, "updated_at": updated_at,
             "title": title}
            for interview_id, status, created_at, updated_at, title in rows
        ]

    def rename(self, owner: str, interview_id: str, title: str) -> bool:
        with self._connection() as conn:
            cursor = self._execute(conn, """
                UPDATE guided_interviews SET title = ?
                WHERE owner = ? AND interview_id = ?
            """, (title, owner, interview_id))
            return cursor.rowcount == 1

    def delete(self, owner: str, interview_id: str) -> bool:
        with self._connection() as conn:
            cursor = self._execute(
                conn, "DELETE FROM guided_interviews WHERE owner = ? AND interview_id = ?", (owner, interview_id),
            )
            return cursor.rowcount == 1

    def save(self, owner: str, interview: Interview) -> bool:
        now = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        with self._connection() as conn:
            cursor = self._execute(conn, """
                UPDATE guided_interviews
                SET status = ?, updated_at = ?, version = version + 1, payload = ?
                WHERE owner = ? AND interview_id = ? AND version = ?
            """, (interview.state.status, now, interview.payload(), owner, interview.state.interview_id,
                  interview.version))
            return cursor.rowcount == 1

    def list_all(self, status: str | None = None, limit: int = 100) -> "list[dict]":
        """Interviews of every owner, newest first, for the preparer review page."""
        query = "SELECT interview_id, owner, status, created_at, updated_at, title, payload FROM guided_interviews"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY updated_at DESC, interview_id LIMIT ?"
        with self._connection() as conn:
            rows = self._execute(conn, query, params + (limit,)).fetchall()
        return [
            {"interview_id": interview_id, "owner": owner, "status": status, "created_at": created_at,
             "updated_at": updated_at, "title": title, "state": json.loads(payload)["state"]}
            for interview_id, owner, status, created_at, updated_at, title, payload in rows
        ]

    def find(self, interview_id: str) -> dict | None:
        """Owner and timestamps of one interview, whoever owns it (preparer review)."""
        with self._connection() as conn:
            row = self._execute(conn, """
                SELECT owner, created_at, updated_at, title FROM guided_interviews WHERE interview_id = ?
            """, (interview_id,)).fetchone()
        if row is None:
            return None
        owner, created_at, updated_at, title = row
        return {"owner": owner, "created_at": created_at, "updated_at": updated_at, "title": title}
