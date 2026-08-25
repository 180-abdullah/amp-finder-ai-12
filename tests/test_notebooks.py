from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_and_have_required_tutorial_sections():
    notebook_paths = sorted((PROJECT_ROOT / "notebooks").glob("0*.ipynb"))
    assert len(notebook_paths) == 3
    for path in notebook_paths:
        document = nbformat.read(path, as_version=4)
        nbformat.validate(document)
        markdown_text = "\n".join(
            cell.source for cell in document.cells if cell.cell_type == "markdown"
        )
        for required_heading in ["## Goal", "## Setup", "## Steps", "## Checks", "## Next Steps"]:
            assert required_heading in markdown_text, f"{required_heading} missing from {path.name}"
