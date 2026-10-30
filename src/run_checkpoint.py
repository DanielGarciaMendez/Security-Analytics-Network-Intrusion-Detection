"""Rebuild checkpoint artifacts from local raw downloads with one command."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for script, args in [
        ("auditoverview2b.py", []),
        ("prepare_shared_features.py", []),
        ("create_checkpoint_splits.py", []),
        ("train_checkpoint_baseline.py", ["--run-checker"]),
        ("build_checkpoint_results.py", []),
    ]:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "src" / script), *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
