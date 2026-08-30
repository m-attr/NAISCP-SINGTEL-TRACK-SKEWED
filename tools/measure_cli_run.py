"""Run the competition CLI once while measuring wall time and peak process-tree RSS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "src" / "main.py"


def _normalized_sha256(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_data_filepath", required=True)
    parser.add_argument("--test_data_filepath", required=True)
    parser.add_argument("--working_directory", required=True)
    parser.add_argument("--prediction_filepath", required=True)
    parser.add_argument("--evidence_filepath", required=True)
    args = parser.parse_args()

    working_directory = Path(args.working_directory).resolve()
    prediction_path = Path(args.prediction_filepath).resolve()
    evidence_path = Path(args.evidence_filepath).resolve()
    working_directory.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PREDICTION_OUTPUT_PATH"] = str(prediction_path)
    command = [
        sys.executable,
        str(CLI_PATH),
        "--train_data_filepath",
        str(Path(args.train_data_filepath).resolve()),
        "--test_data_filepath",
        str(Path(args.test_data_filepath).resolve()),
    ]

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=working_directory,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    monitored = psutil.Process(process.pid)
    peak_tree_rss = 0
    while process.poll() is None:
        try:
            processes = [monitored, *monitored.children(recursive=True)]
            current = sum(item.memory_info().rss for item in processes if item.is_running())
            peak_tree_rss = max(peak_tree_rss, int(current))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    wall_seconds = time.perf_counter() - started

    metrics_path = working_directory / "latest_metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else None
    )
    output_validation: dict[str, object] | None = None
    if prediction_path.is_file():
        output = pd.read_csv(prediction_path)
        scores = pd.to_numeric(output.get("probability_score"), errors="coerce")
        output_validation = {
            "columns": output.columns.tolist(),
            "rows": int(len(output)),
            "missing_probabilities": int(scores.isna().sum()),
            "finite_probabilities": bool(
                scores.notna().all()
                and np.isfinite(scores.to_numpy(dtype=np.float64)).all()
            ),
            "probabilities_in_range": bool(scores.between(0.0, 1.0).all()),
            "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "normalized_sha256": _normalized_sha256(prediction_path),
        }

    evidence = {
        "command": command,
        "working_directory": str(working_directory),
        "exit_code": int(process.returncode),
        "wall_seconds": wall_seconds,
        "peak_process_tree_rss_bytes": peak_tree_rss,
        "peak_process_tree_rss_mib": peak_tree_rss / (1024**2),
        "metrics": metrics,
        "output_validation": output_validation,
        "stdout": stdout,
        "stderr": stderr,
    }
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "exit_code": evidence["exit_code"],
                "wall_seconds": wall_seconds,
                "peak_process_tree_rss_mib": evidence["peak_process_tree_rss_mib"],
                "output_validation": output_validation,
                "evidence_path": str(evidence_path),
            },
            indent=2,
        )
    )
    if process.returncode != 0:
        raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
