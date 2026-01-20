"""Wrapper to run Modal with UTF-8 encoding.

Usage:
    py -3.12 run_modal_utf8.py analyze_feature_json --feature-idx 13627
    py -3.12 run_modal_utf8.py batch_test --feature-idx 13627 --texts "text1|text2|text3"
    py -3.12 run_modal_utf8.py ablate_context --feature-idx 13627 --text "Some text"
"""
import sys
import io
import os
import subprocess

# Force stdout/stderr to use UTF-8 with replacement (fixes Windows cp1252 issues)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Also set environment variable for subprocess
os.environ['PYTHONIOENCODING'] = 'utf-8:replace'

# Get command from command line args or use default
if len(sys.argv) > 1:
    # Build command from args: py run_modal_utf8.py analyze_feature_json --feature-idx 13627
    entry_point = sys.argv[1]
    extra_args = sys.argv[2:]
    cmd = [sys.executable, "-m", "modal", "run", f"src/modal_interpreter.py::{entry_point}"] + extra_args
else:
    # Default command for testing
    cmd = [
        sys.executable, "-m", "modal", "run",
        "src/modal_interpreter.py::analyze_feature_json",
        "--feature-idx", "13627"
    ]

try:
    result = subprocess.run(
        cmd,
        capture_output=True,
        encoding='utf-8',
        errors='replace'
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
