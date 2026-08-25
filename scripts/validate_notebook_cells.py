#!/usr/bin/env python3
"""Execute notebook code cells in one process when Jupyter sockets are unavailable."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("outputs/notebook_validation.json")
    )
    return parser.parse_args()


def execute_code_cells(path: Path) -> dict[str, object]:
    document = nbformat.read(path, as_version=4)
    nbformat.validate(document)
    shell = InteractiveShell.instance()
    namespace = shell.user_ns
    executed = 0
    for cell_index, cell in enumerate(document.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        result = shell.run_cell(cell.source, store_history=False, silent=False)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            raise RuntimeError(
                f"{path.name} failed in code cell {cell_index}: {type(error).__name__}: {error}"
            ) from error
        executed += 1
    shell.reset(new_session=True)
    return {"notebook": str(path), "code_cells_executed": executed, "status": "passed"}


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    results = [execute_code_cells(path.resolve()) for path in args.notebooks]
    report = {
        "validated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "method": (
            "IPython in-process top-to-bottom execution. Use nbconvert in normal "
            "environments; this fallback avoids restricted-kernel socket failures."
        ),
        "results": results,
    }
    report_path = (PROJECT_ROOT / args.report).resolve() if not args.report.is_absolute() else args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved validation report:", report_path)


if __name__ == "__main__":
    main()
