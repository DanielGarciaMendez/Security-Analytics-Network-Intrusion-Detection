"""Rebuild checkpoint artifacts from local raw downloads with one command."""
from pathlib import Path
import subprocess
import sys

from prepare_final_scores import LEGACY_SCORES

ROOT = Path(__file__).resolve().parents[1]


def main():
    if (ROOT / LEGACY_SCORES).exists():
        sys.exit("Run the rebuild in a separate workspace to preserve the published scores and their split.")
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
