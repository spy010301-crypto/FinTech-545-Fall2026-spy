"""Assignment-specific file paths and JSON output, not risk calculations."""

import json
from pathlib import Path

FOLDER = Path(__file__).resolve().parent


def save_results(problem, results):
    """Save the unrounded numbers as well as printing them."""
    output = FOLDER / "results"
    output.mkdir(exist_ok=True)
    text = json.dumps(results, indent=2, allow_nan=False)
    (output / f"problem{problem}_results.json").write_text(text + "\n")
    print(text)
