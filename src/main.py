import time

CLI_WALL_START = time.perf_counter()

import argparse
import sys

from pipeline_orchestrator import run_pipeline
from utils import format_ascii_table, format_phase_box


def print_outputs(phase: dict) -> None:
    phase_id = int(phase.get("id", 0))
    phase_title = str(phase.get("title", "")).strip()
    print("", file=sys.__stdout__)
    print(format_phase_box(f"Phase {phase_id}: {phase_title}"), file=sys.__stdout__)

    for line in phase.get("lines", []):
        print(str(line), file=sys.__stdout__)

    for table in phase.get("tables", []):
        headers = [str(h) for h in table.get("headers", [])]
        rows = table.get("rows", [])
        if headers:
            print(format_ascii_table(headers, rows), file=sys.__stdout__)

    sys.__stdout__.flush()


def main():
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 Pipeline")
    parser.add_argument("--train_data_filepath", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--test_data_filepath", type=str, required=True, help="Path to test.csv")
    args = parser.parse_args()

    report = run_pipeline(
        train_data_filepath=args.train_data_filepath,
        test_data_filepath=args.test_data_filepath,
        quiet=True,
        on_phase=print_outputs,
    )

    if not bool(report.get("success", False)):
        err = str(report.get("error") or "Unknown pipeline error")
        print("", file=sys.__stdout__)
        print(format_phase_box("Pipeline Error"), file=sys.__stdout__)
        print(err, file=sys.__stdout__)
    cli_elapsed = time.perf_counter() - CLI_WALL_START
    print("", file=sys.__stdout__)
    print(format_phase_box("CLI Wall Clock"), file=sys.__stdout__)
    print(f"CLI wall-clock (s): {cli_elapsed:.2f}", file=sys.__stdout__)
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()