"""
check_connection.py: can we reach the database, and is pgvector ready?

Usage (from the tax-agent folder)
  python -m src.db.check_connection
"""
import psycopg

from src.db.connection import get_conn

HINTS = {
    "timeout": "Network blocked: your current IP is not in the security group, or Public access = No.",
    "password authentication failed": "Wrong user or password in DATABASE_URL.",
    "does not exist": "Database name wrong: use /taxagent at the end of DATABASE_URL "
                      "(or create it: CREATE DATABASE taxagent;).",
    "could not translate host name": "Endpoint typo in DATABASE_URL.",
}


def main():
    try:
        with get_conn() as conn:
            print("connected :", conn.execute("SELECT version()").fetchone()[0].split(",")[0])
            print("database  :", conn.execute("SELECT current_database()").fetchone()[0])
            print("user      :", conn.execute("SELECT current_user").fetchone()[0])
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            print("pgvector  :", ver[0] if ver else "NOT INSTALLED")
            print("\nOK: the database is ready.")
    except psycopg.OperationalError as e:
        msg = str(e)
        print("FAILED:", msg.strip().splitlines()[-1])
        for key, hint in HINTS.items():
            if key in msg:
                print("hint  :", hint)


if __name__ == "__main__":
    main()
