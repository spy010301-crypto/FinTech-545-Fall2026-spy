"""Assignment paths and JSON output; the calculation libraries do not do file I/O."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures'
RESULTS = ROOT / 'results'


def save_results(problem_number, result):
    """Keep unrounded results so the short written answers can be checked."""
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f'Cannot save {type(value).__name__}')

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f'problem{problem_number}_results.json'
    path.write_text(json.dumps(result, indent=2, default=convert, allow_nan=False) + '\n')
    print(f'Saved {path.name}')
