"""
create_member.py: give a teammate their own database login.

  readonly  read every table (evaluation, business analysis)
  rag       readonly + may write chunks.embedding / embedding_model (RAG work)

A random password is generated and printed ONCE. Send it to the teammate
privately (not in a group chat, never in the repo).

Usage (from the tax-agent folder, as taxagent_admin)
  python -m src.db.create_member alice --role readonly
  python -m src.db.create_member bob --role rag
  python -m src.db.create_member bob --drop          # remove the user
"""
import argparse
import os
import secrets

from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from src.db.connection import get_conn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="lowercase login name, e.g. alice")
    ap.add_argument("--role", choices=["readonly", "rag"], default="readonly")
    ap.add_argument("--drop", action="store_true", help="remove this user instead")
    args = ap.parse_args()
    name = args.name.lower()
    user = sql.Identifier(name)

    with get_conn() as conn:
        if args.drop:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(user))
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(user))
            print(f"removed user {name}")
            return

        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [name]).fetchone():
            raise SystemExit(f"user {name} already exists (use --drop first to recreate)")

        password = secrets.token_urlsafe(18)          # letters, digits, - and _ only
        db = sql.Identifier(conn.info.dbname)
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(user, sql.Literal(password)))
        conn.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(db, user))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(user))
        conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(user))
        conn.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {}").format(user))
        if args.role == "rag":
            conn.execute(sql.SQL("GRANT UPDATE (embedding, embedding_model) ON chunks TO {}").format(user))

    info = conninfo_to_dict(os.environ["DATABASE_URL"])
    print(f"created user {name} ({args.role})")
    print("send this line to the teammate PRIVATELY; they put it in their own .env:\n")
    print(f"DATABASE_URL=postgresql://{name}:{password}@{info['host']}:{info.get('port', 5432)}"
          f"/{info['dbname']}?sslmode=require")


if __name__ == "__main__":
    main()
