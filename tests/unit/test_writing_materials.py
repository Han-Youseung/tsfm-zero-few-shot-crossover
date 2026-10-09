"""Writing exports stay bound to the existing analysis, not new model runs."""

import json
from pathlib import Path

import pytest

from tsfm_crossover.analysis.writing_materials import csv_file, interval_label, results_paragraphs

ROOT = Path(__file__).resolve().parents[2]


def test_unobserved_means_unmet_not_missing():
    assert interval_label(None) == "관측 범위 내 지속 개선 기준 미충족"
    assert interval_label({"lower_exclusive": 0.005, "upper_inclusive": 0.01}) == "(0.5%, 1%]"


def test_draft_has_exactly_ten_verified_endpoint_paragraphs():
    result = json.loads(
        (ROOT / "results/summaries/primary5_h96_two_model_analysis.json").read_text("utf-8")
    )
    text = results_paragraphs(result)
    assert len(text.split("\n\n")) == 10
    assert "51.17%" in text and "0.31%" in text
    assert text.count("관측 범위 내 지속 개선 기준 미충족") == 1
    assert "Timer" not in text and "미실행" not in text


def test_csv_preserves_null_and_numeric_precision(tmp_path):
    path = tmp_path / "table.csv"
    csv_file(path, [{"rate": 0.005, "value": 0.123456789123456, "missing": None}])
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "0.005,0.123456789123456," in path.read_text("utf-8-sig")


def test_empty_table_rejected(tmp_path):
    with pytest.raises(ValueError):
        csv_file(tmp_path / "empty.csv", [])


def test_reference_keys_and_draft_numbers_are_resolvable():
    docs = ROOT / "docs/paper-writing"
    refs = json.loads((docs / "references.json").read_text("utf-8"))
    bib = (docs / "references.bib").read_text("utf-8")
    assert [r["number"] for r in refs] == list(range(1, 11))
    assert len({r["key"] for r in refs}) == 10
    for ref in refs:
        assert "{" + ref["key"] + "," in bib
        assert ref["url"].startswith("https://")
    draft = (docs / "draft-ko.template.txt").read_text("utf-8")
    assert draft.count("{{RESULTS}}") == 1
    for caveat in ("사전학습", "비율", "270", "수렴", "출처군", "이용 가능한 뒤"):
        assert caveat in draft
