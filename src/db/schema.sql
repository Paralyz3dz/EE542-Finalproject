-- Knowledge base tables. Run by src/db/load_chunks.py (safe to run again).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,          -- pub17_2025_ch01_011
    tax_year        INT  NOT NULL,
    source          TEXT NOT NULL,             -- IRS Pub 17 (2025)
    chapter_no      INT,
    chapter         TEXT,
    section         TEXT,
    subsection      TEXT,
    minor           TEXT,
    runin           TEXT,
    type            TEXT NOT NULL,             -- rule / tip / caution / whats_new / table
    table_kind      TEXT,                      -- decision / box / worksheet / table
    printed_pages   INT[],
    pdf_pages       INT[],
    citation        TEXT NOT NULL,
    text            TEXT NOT NULL,
    embed_text      TEXT NOT NULL,             -- heading path + text; the only field that is embedded
    content_hash    TEXT NOT NULL,             -- hash of embed_text: tells us when an embedding is stale
    embedding       vector(1024),              -- Qwen3-Embedding-0.6B; change if the model changes
    embedding_model TEXT,
    updated_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_year_type ON chunks (tax_year, type);

-- Raw table rows for code lookups (numbers come from code, not the LLM)
CREATE TABLE IF NOT EXISTS tax_tables (
    chunk_id     TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    caption      TEXT,
    kind         TEXT,
    citation     TEXT,
    chapter_no   INT,
    printed_page INT,
    pdf_page     INT,
    rows         JSONB NOT NULL
);
