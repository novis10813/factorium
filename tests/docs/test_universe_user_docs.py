from pathlib import Path


def test_universe_guide_file_exists() -> None:
    assert Path("docs/user-guide/universe.md").exists()


def test_universe_guide_has_required_sections() -> None:
    text = Path("docs/user-guide/universe.md").read_text(encoding="utf-8")
    required = [
        "# Universe 與 Checklist",
        "## 快速流程",
        "## 與 AggBar 整合",
        "## 與 Factor 整合",
        "## 與 Backtest 整合",
        "## 常見錯誤",
    ]
    for section in required:
        assert section in text


def test_universe_guide_has_core_api_snippets() -> None:
    text = Path("docs/user-guide/universe.md").read_text(encoding="utf-8")
    required_snippets = [
        "AggBar.with_mask(",
        "Factor.eval",
        "mask=",
        "Backtester(",
    ]
    for snippet in required_snippets:
        assert snippet in text


def test_mkdocs_nav_includes_universe_page() -> None:
    text = Path("mkdocs.yml").read_text(encoding="utf-8")
    assert "user-guide/universe.md" in text


def test_index_mentions_universe_capability() -> None:
    text = Path("docs/index.md").read_text(encoding="utf-8")
    assert "universe" in text.lower()
    assert "checklist" in text.lower()


def test_related_guides_reference_universe_workflow() -> None:
    bar_text = Path("docs/user-guide/bar.md").read_text(encoding="utf-8")
    factor_text = Path("docs/user-guide/factor.md").read_text(encoding="utf-8")
    backtest_text = Path("docs/user-guide/backtest.md").read_text(encoding="utf-8")

    assert "with_mask" in bar_text
    assert "mask" in factor_text.lower()
    assert "mask" in backtest_text.lower()
