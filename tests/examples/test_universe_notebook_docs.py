from pathlib import Path
import json


def test_universe_notebook_exists() -> None:
    assert Path("examples/05_universe_checklist_workflow.ipynb").exists()


def test_examples_readme_mentions_universe_notebook() -> None:
    text = Path("examples/README.md").read_text(encoding="utf-8")
    assert "05_universe_checklist_workflow.ipynb" in text


def test_universe_notebook_has_required_sections() -> None:
    notebook = json.loads(Path("examples/05_universe_checklist_workflow.ipynb").read_text(encoding="utf-8"))
    markdown_text = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "markdown"
    )

    required = [
        "# Universe Checklist Workflow",
        "## Build Universe and Checklist",
        "## Apply Mask to AggBar",
        "## Evaluate Factor with Mask",
        "## Run Backtest with Mask",
    ]
    for section in required:
        assert section in markdown_text


def test_universe_notebook_has_core_api_snippets() -> None:
    text = Path("examples/05_universe_checklist_workflow.ipynb").read_text(encoding="utf-8")
    required_snippets = [
        "with_mask(",
        "mask=",
        "Universe(",
        "Checklist(",
        "Backtester(",
    ]
    for snippet in required_snippets:
        assert snippet in text


def test_examples_readme_has_universe_guidance() -> None:
    text = Path("examples/README.md").read_text(encoding="utf-8")
    assert "Universe & Checklist Workflow" in text
    assert "Prerequisites" in text
    assert "Recommended Reading Order" in text
    assert "Notebook 05" in text
