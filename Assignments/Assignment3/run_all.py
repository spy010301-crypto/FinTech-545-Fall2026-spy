"""Run all five problems, then rebuild the written responses and PDF."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

if __name__ == '__main__':
    for filename in [f'problem{i}.py' for i in range(1,6)] + ['build_responses.py']:
        print(f'\nRunning {filename}', flush=True)
        subprocess.run([sys.executable, str(ROOT / filename)], cwd=ROOT, check=True)
