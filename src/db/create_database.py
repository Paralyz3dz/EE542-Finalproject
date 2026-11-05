"""
create_database.py: create the 'taxagent' database if it is missing.
Needed when "Initial database name" was left empty while creating the RDS instance.
Run once.

Usage (from the tax-agent folder)
  python -m src.db.create_database
"""
import os

import psycopg
from psycopg.conninfo import conninfo_to_dict

from src.db.connection import load_dotenv, ROOT  # noqa: F401  (loads .env)


def main():
    url = os.environ["DATABASE_URL"]
    target = conninfo_to_dict(url)["dbname"]           # 'taxagent'
    # connect to the built-in 'postgres' database; CREATE DATABASE cannot run inside a transaction
    with psycopg.connect(url, dbname="postgres", autocommit=True, connect_timeout=10) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", [target]).fetchone()
        if exists:
            print(f"database '{target}' already exists")
        else:
            conn.execute(f'CREATE DATABASE "{target}"')
            print(f"created database '{target}'")


if __name__ == "__main__":
    main()
