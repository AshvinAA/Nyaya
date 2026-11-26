"""Offline regression test: Symbol-font glyphs and unresolved (cid:N) tokens.

Run: python tests/test_ingestion_glyphs.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.common import clean_cell, clean_glyphs
from ingestion.validation_gate import check_legacy_font, check_unresolved_font


def test_symbol_dash_becomes_em_dash():
    assert clean_glyphs("Weekly holiday.(1) Every") == "Weekly holiday.——(1) Every"
    assert clean_glyphs("Definitions.In this Act") == "Definitions.—In this Act"


def test_running_header_bullet_dropped():
    assert clean_glyphs(" GB.354/INS/8 Governing Body") == " GB.354/INS/8 Governing Body"


def test_other_private_use_glyphs_dropped():
    assert clean_glyphs("abcdef") == "abcdef"


def test_clean_cell_applies_glyph_fix():
    assert clean_cell("Rate 5%") == "Rate— 5%"
    assert clean_cell(None) == ""


def test_cid_chunks_quarantined_and_clean_chunks_kept():
    chunks = [
        {"text": "No fine exceeding (cid:1) (one-hundred) taka", "retrievable": True},
        {"text": "No fine exceeding five hundred taka", "retrievable": True},
    ]
    q = check_unresolved_font(chunks)
    assert len(q) == 1 and chunks[0]["retrievable"] is False
    assert chunks[0]["pollution"] == "unresolved_font_glyph"
    assert chunks[1]["retrievable"] is True


def test_dagger_quarantine_still_works():
    chunks = [{"text": "†‡ bad", "retrievable": True}]
    assert len(check_legacy_font(chunks)) == 1


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
