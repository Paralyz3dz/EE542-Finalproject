import json
from pathlib import Path

import pytest

from src.db.queries import corpus_table, fetch_chunks, vector_search
from src.rag.materials import canonicalize, clean, make_document, runin_heading, split_passage
from src.workflow.engine import build_guide, load_scenario, start_interview


def element(text, kind="text", page=1, **values):
    return {"text": text, "kind": kind, "page": page, "printed": page, "level": 1, **values}


def document(elements):
    return make_document({"name": "IRS Pub 527", "path": "p527--2025.pdf", "url": "official.pdf"}, elements)


def test_runin_sections_are_never_merged_to_fill_a_budget():
    chunks, _, _ = document([element("Income", "heading"), element("First rule.", runin="First."), element("Second rule.", runin="Second.")])
    assert len(chunks) == 2
    assert chunks[0]["runin"] == "First."
    assert "Second rule" not in chunks[0]["text"]


def test_sentences_and_hyphenated_words_survive_chunk_boundaries():
    sentences = [("Word " * 75 + f"sentence{i}.") for i in range(13)]
    chunks = split_passage([(" ".join(sentences), [1])])
    assert all(len(t.split()) <= 400 and t.rstrip().endswith(".") for t, _ in chunks)
    assert " ".join(t for t, _ in chunks).split() == " ".join(sentences).split()
    assert clean("qual-\nified tax-free") == "qualified tax-free"


def test_reference_list_removed_but_body_on_same_page_preserved():
    chunks, _, audit = document([element("Useful Items", "heading"), element("You may want to see:"),
                                element("Publication", "heading"), element("463 Travel, Gift, and Car Expenses"),
                                element("Rental Income", "heading"), element("Advance rent is income in the year received.")])
    assert len(chunks) == 1
    assert "Advance rent" in chunks[0]["text"]
    assert "Travel" not in chunks[0]["text"]
    assert audit["reference_list_boxes"] == 3


def test_numeric_tables_keep_all_rows_without_embedding_large_amount_lists():
    rows = [[str(i), str(i * 10), str(i * 20)] for i in range(300)]
    chunks, tables, _ = document([element("Tax table", "heading"), element("", "table", rows=rows)])
    assert len(chunks) == len(tables) == 1
    assert tables[0]["rows"] == rows
    assert chunks[0]["n_words"] < 100


def test_distinct_numeric_tables_with_same_heading_have_distinct_ids():
    chunks, tables, _ = document([element("Tax", "heading"), element("", "table", rows=[["1", "2", "3"]]),
                                 element("", "table", rows=[["1", "2", "3"], ["4", "5", "6"]])])
    assert len(chunks) == len(tables) == 2
    assert len({c["id"] for c in chunks}) == 2


def test_semantic_table_split_preserves_each_complete_row():
    rows = [["Condition " + str(i), "word " * 35] for i in range(40)]
    chunks, tables, _ = document([element("Decision rules", "heading"), element("", "table", rows=rows)])
    assert len(chunks) > 1
    assert [row for t in tables for row in t["rows"]] == [[cell.strip() for cell in row] for row in rows]
    assert all(c["n_words"] <= 400 for c in chunks)


def test_exact_duplicate_consolidation_preserves_all_sources():
    chunks, tables, _ = document([element("Rental income", "heading"), element("A rental rule. " * 30)])
    alias = dict(chunks[0], id="alias", source="FTB 540 Booklet (2025)", citation="FTB 540 Booklet (2025), p. 3")
    canonical, _, audit = canonicalize(chunks + [alias], tables)
    assert len(canonical) == 1 and len(audit) == 1
    assert len(canonical[0]["provenance"]) == 2


def test_guide_retains_evidence_beyond_character_600_and_duplicate_provenance():
    state = start_interview("intake")
    _, questions = load_scenario("intake")
    q = questions[0]
    text = "Introductory context. " * 50 + "The decisive condition is at the end."
    alias = q["source"].split(";")[0] + " (2025), p. 1"
    guide = build_guide(state, q, search=lambda _: {"matches": [{"citation": "IRS Pub 527 (2025), p. 2", "text": text, "additional_citations": [alias]}]})
    assert guide["citations"][0]["text"] == text
    assert guide["references"][0]["retrieved"] is True


def test_schema_identifier_rejects_sql_and_baseline_remains_explicit():
    assert corpus_table() == "public.chunks"
    assert corpus_table("rag_materials_v2") == "public.rag_materials_v2_chunks"
    for invalid in ('public;DROP SCHEMA public', 'x.y', '', None, 'PUBLIC'):
        with pytest.raises(ValueError):
            corpus_table(invalid)


def test_cross_column_hyphenation_keeps_complete_words_and_tax_compounds():
    assert clean("tax-\nfree qual-\nified") == "tax-free qualified"
    chunks = split_passage([("training for em-", [1]), ("ployment in a recognized occupation.", [2])])
    assert chunks == [("training for employment in a recognized occupation.", [1, 2])]


def test_actual_pub17_recovers_wrapped_runin_and_complete_chapter_title():
    import pymupdf
    from src.db.connection import ROOT
    from src.rag.materials import page_nodes
    with pymupdf.open(ROOT / "data/raw/irs/p17--2025.pdf") as pdf:
        nodes = page_nodes(pdf[38], {"boxes": []}, 8)
    titles = [n["text"] for n in nodes if n["kind"] == "heading"]
    assert "Tax Withholding and Estimated Tax" in titles
    assert "Tax Withholding for 2026" in titles
    paragraph = next(n for n in nodes if n.get("runin") == "Taxpayer identification numbers for adoptees.")
    assert "you must get an ATIN" in paragraph["text"]
    assert "income tax withholding" not in paragraph["text"]


def test_actual_mixed_index_page_preserves_qcd_worksheet_without_index():
    import pymupdf
    from src.db.connection import ROOT
    from src.rag.materials import grid_tables, page_nodes
    with pymupdf.open(ROOT / "data/raw/irs/p590b--2025.pdf") as pdf:
        page = pdf[69]
        nodes = page_nodes(page, {"boxes": []}, 10, clip_bottom=215, grids=grid_tables(page))
    text = " ".join(n["text"] + " ".join(" ".join(row) for row in n.get("rows", [])) for n in nodes)
    assert "prior years" in text and "current year" in text
    assert "Index" not in text and "10-year rule" not in text


def test_actual_ftb_comparison_grid_has_readable_cells_and_all_forms():
    import pymupdf
    from src.db.connection import ROOT
    from src.rag.materials import grid_tables
    with pymupdf.open(ROOT / "data/raw/ftb/2025-540-booklet.pdf") as pdf:
        grids = grid_tables(pdf[9])
    rows = next(g["table"]["extract"] for g in grids if any("Filing Status" in row for row in g["table"]["extract"]))
    assert rows[1][0] == "Filing Status"
    assert "Any filing status" in rows[1][2]
    assert "0-3allowed" in rows[2][1].replace(" ", "")


def test_numeric_geometry_keeps_column_alignment_with_missing_cells():
    import pymupdf
    from src.rag.materials import numeric_bands
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        for row in range(4):
            for col in range(5):
                if row == 2 and col == 2:
                    continue
                page.insert_text((40 + col * 65, 100 + row * 16), str(1000 + row * 10 + col), fontsize=10)
        bands = numeric_bands(page, {"x0": 30, "y0": 70, "x1": 400, "y1": 200})
    assert len(bands) == 1
    rows = bands[0]["table"]["extract"]
    assert all(len(row) == 5 for row in rows)
    assert rows[2] == ["1020", "1021", "", "1023", "1024"]


def test_numeric_locators_are_stored_but_excluded_before_both_candidate_limits():
    from unittest.mock import MagicMock
    rows = [[str(i), str(i + 100), str(i + 200), str(i + 300)] for i in range(10)]
    chunks, tables, _ = document([element("Credit table", "heading"), element("", "table", rows=rows)])
    assert chunks[0]["retrieval_eligible"] is False
    assert tables[0]["rows"] == rows
    connection = MagicMock()
    vector_search(connection, [1], "model", 2025, k=10, corpus="rag_materials_v2")
    query = connection.execute.call_args.args[0]
    assert query.index("AND retrieval_eligible") < query.index("LIMIT")
    fetch_chunks(connection, 2025, corpus="rag_materials_v2")
    assert "AND retrieval_eligible" in connection.execute.call_args.args[0]


def test_completed_single_round_cannot_be_run_again(tmp_path, monkeypatch):
    from unittest.mock import patch
    from eval import run_materials_experiment as experiment
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_materials_experiment", "--env-file", str(tmp_path / "missing.env"), "--out", str(tmp_path)])
    with patch.object(experiment, "snapshot") as snapshot, patch.object(experiment, "observe") as observe:
        with pytest.raises(ValueError, match="already complete"):
            experiment.main()
    snapshot.assert_not_called()
    observe.assert_not_called()


def test_tail_rebalancing_does_not_reintroduce_a_split_word():
    import re
    first = " ".join([("Word " * 132 + "End.")] * 3)
    tail = "ble distributions do not need to be reported. " * 10
    chunks = split_passage([(first, [1]), ("Nontaxa-", [1]), (tail, [2])])
    assert any("Nontaxable" in text for text, _ in chunks)
    assert not any(re.search(r"[A-Za-z]-\s*\n\s*[a-z]", text) for text, _ in chunks)


class TestVersionedRetrieval:
    def test_configured_version_uses_one_corpus_and_preserves_duplicate_sources(self):
        from unittest.mock import MagicMock, patch
        import numpy as np
        from src.rag import retriever
        connection = MagicMock()
        connection.execute.side_effect = [MagicMock(), MagicMock(fetchone=lambda: (8388,)), MagicMock(fetchone=lambda: (True,)), MagicMock()]
        settings = {"tax_year": 2025, "retrieve_top_k": 5, "retrieve_candidate_k": 10, "rrf_constant": 60, "corpus_version": "rag_materials_v2", "exclude_reference_chunks": False}
        primary = "IRS Pub 17 (2025), p. 30"
        alias = "IRS Pub 970 (2025), p. 21"
        chunks = [{"id": "shared", "citation": primary, "text": "Student support rule.", "embed_text": "Student support rule.", "provenance": [{"citation": primary}, {"citation": alias}]}]
        with patch.object(retriever, "get_conn") as get_conn, patch.object(retriever, "register_vector"), patch.object(retriever.json, "loads", return_value=settings), patch.object(retriever, "embed_query", return_value=np.ones(1024)), patch.object(retriever, "fetch_chunks", return_value=chunks) as fetch, patch.object(retriever, "vector_search", return_value=[("shared", primary, "Student support rule.", .9)]) as dense, patch.object(retriever, "bm25_search", return_value=[]):
            get_conn.return_value.__enter__.return_value = connection
            result = retriever.retrieve("Student support")
        fetch.assert_called_once_with(connection, 2025, corpus="rag_materials_v2")
        assert dense.call_args.kwargs == {"k": 10, "corpus": "rag_materials_v2"}
        assert "public.rag_materials_v2_chunks" in connection.execute.call_args_list[1].args[0]
        assert "retrieval_eligible" in connection.execute.call_args_list[1].args[0]
        assert result["matches"][0]["additional_citations"] == [alias]


def test_sentence_boundaries_preserve_publication_and_country_abbreviations():
    from src.rag.materials import sentences
    assert sentences("Are you a U.S. Citizen? See Pub. 501. Next condition.") == ["Are you a U.S. Citizen?", "See Pub. 501.", "Next condition."]
    assert sentences("1. Complete the form. 2. Sign the return.") == ["1. Complete the form. 2. Sign the return."]
