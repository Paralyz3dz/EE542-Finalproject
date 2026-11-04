"""Tests for src/rag/tables.py.

Run either way (from the tax-agent folder):
  python tests/test_tables.py
  pytest tests            (if pytest is installed)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
from tables import (classify, clean_cell, clean_rows, forward_fill, header_index,  # noqa: E402
                    linearize_box, linearize_decision, linearize_table, tidy, worksheet_pointer)

P9 = [["IF your filing status is...", "AND at the end of 2025 you were...*",
       "THEN file a return if your gross income was at least...**"],
      ["Single", "under 65", "$15,750"],
      ["", "65 or older", "$17,750"],
      ["Married filing jointly***", "under 65 (both spouses)", "$31,500"],
      ["", "65 or older (one spouse)", "$33,100"]]
P101 = [["Type of tax", "You can deduct", "You can’t deduct"],
        ["Income taxes", "State and local income taxes.", "Federal income taxes."]]

CHECKS = [
    ("clean_cell", lambda: clean_cell("elec-\ntronic and\nmore"), "electronic and more"),
    ("clean_rows", lambda: clean_rows([["Single", None], [None, None], ["a\nb", "x"]]),
     [["Single", ""], ["a b", "x"]]),
    ("split_stacked_rows", lambda: clean_rows([["Single", "1\n2", "$17,750\n19,750"]]),
     [["Single", "1", "$17,750"], ["", "2", "19,750"]]),
    ("forward_fill", lambda: forward_fill(P9[1:3]),
     [["Single", "under 65", "$15,750"], ["Single", "65 or older", "$17,750"]]),
    ("header_index", lambda: [header_index(P9), header_index([["Check the boxes", ""], ["IF x", "THEN y"]]),
                              header_index(P101)], [0, 1, 0]),
    ("tidy", lambda: [tidy("IF your filing status is..."), tidy("was at least...**"),
                      tidy("IF the person is your . . ."), tidy("Married filing jointly***")],
     ["IF your filing status is", "was at least", "IF the person is your", "Married filing jointly"]),
    ("classify", lambda: [classify({"caption": "Table 1-1.", "rows": P9}),
                          classify({"caption": "Table 1-2.", "rows": [["a"], ["b"]]}),
                          classify({"caption": "Worksheet 3-1.", "rows": [["1.", "Enter"]]}),
                          classify({"caption": "", "rows": [["1. Enter your earned income", "1."]]}),
                          classify({"caption": "Table 11-1.", "rows": P101})],
     ["decision", "box", "worksheet", "worksheet", "table"]),
    ("linearize_decision", lambda: linearize_decision(P9).split("\n")[1],
     "IF your filing status is Single AND at the end of 2025 you were 65 or older "
     "THEN file a return if your gross income was at least $17,750."),
    ("linearize_table", lambda: linearize_table(P101),
     "Type of tax: Income taxes | You can deduct: State and local income taxes. | You can’t deduct: Federal income taxes."),
    ("linearize_box", lambda: linearize_box([["first"], ["second"]]), "first\nsecond"),
    ("worksheet_pointer", lambda: worksheet_pointer(
        {"caption": "Worksheet 3-1. Worksheet for Determining Support",
         "rows": [["Funds Belonging to the Person You Supported", ""],
                  ["1.", "Enter the total funds belonging to the person you supported, including income received"],
                  ["7.", ""],
                  ["2.", "Enter the amount on line 1 that was used ........ 2."]]}),
     "Worksheet 3-1. Worksheet for Determining Support. Steps: "
     "1. Enter the total funds belonging to the person you supported, including; "
     "2. Enter the amount on line 1 that was used"),
]


def test_tables():
    """pytest entry point: every check must pass."""
    for name, fn, expected in CHECKS:
        assert fn() == expected, name


if __name__ == "__main__":
    failed = 0
    for name, fn, expected in CHECKS:
        try:
            got = fn()
        except Exception as e:
            got = f"{type(e).__name__}: {e}"
        ok = got == expected
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"       expected: {expected!r}\n       got:      {got!r}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed")
