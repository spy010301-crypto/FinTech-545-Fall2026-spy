"""Run the raw-data preview and all five problems with this Python interpreter."""

import subprocess
import sys

from helpers import FOLDER

for script in ["preview_inputs.py", "problem1.py", "problem2.py", "problem3.py", "problem4.py", "problem5.py"]:
    print(f"\nRunning {script}", flush=True)
    subprocess.run([sys.executable, str(FOLDER / script)], check=True)
