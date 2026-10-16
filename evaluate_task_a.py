"""Prepare 32-bar tokens, run the TA evaluator, and summarize H4/GS."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from run_experiments import CONFIGS


HERE = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--training-sequences", type=Path, default=Path(
        "data/miditok_remi_no_chords/train_sequences.json"))
    parser.add_argument("--vocab", type=Path, default=Path(
        "data/miditok_remi_no_chords/tok2idx.json"))
    parser.add_argument("--ta-script", type=Path, default=Path(
        "../hw2_sample/HW2_scripts/eval_basic.py"))
    parser.add_argument("--musdr-parent", type=Path, default=Path("third_party"))
    parser.add_argument("--output-dir", type=Path, default=Path("metrics/basic_a"))
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS),
                        default=list(CONFIGS))
    parser.add_argument("--songs", type=int, default=20)
    return parser.parse_args()


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def summarize(csv_path: Path, source: str) -> dict[str, object]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No evaluation rows in {csv_path}")
    summary: dict[str, object] = {"source": source, "count": len(rows)}
    for metric, key in (("H4", "H4 (Pitch Entropy 4-Bar)"),
                        ("GS", "GS (Groove Similarity)")):
        values = [float(row[key]) for row in rows if row[key] not in {"", "nan", "NaN"}]
        if len(values) != len(rows):
            raise ValueError(f"{source}: {metric} has missing/NaN values")
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return summary


def main() -> None:
    args = arguments()
    if not args.ta_script.is_file():
        raise FileNotFoundError(args.ta_script)
    if not (args.musdr_parent / "MusDr" / "musdr" / "side_utils.py").is_file():
        raise FileNotFoundError("Clone MusDr into --musdr-parent/MusDr first")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.musdr_parent.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    sources = [("reference", args.training_sequences, None)] + [
        (name, args.results_root / "basic_a" / name, args.songs) for name in args.configs
    ]
    summaries = []
    reference_summary = None
    for name, source, count in sources:
        prepared = args.output_dir / name
        command = [sys.executable, str(HERE / "prepare_eval_basic.py"),
                   "--input", str(source), "--output-dir", str(prepared),
                   "--vocab", str(args.vocab)]
        if count is not None:
            command += ["--songs", str(count)]
        run(command)
        output_csv = prepared / "scores.csv"
        run([sys.executable, str(args.ta_script),
             "--tok2idx_path", str(prepared / "tok2idx.json"),
             "--sequences_path", str(prepared / "sequences.json"),
             "--output_csv", str(output_csv)], env=env)
        row = summarize(output_csv, name)
        if name == "reference":
            reference_summary = row
        else:
            summaries.append(row)
    if reference_summary is None:
        raise RuntimeError("Reference evaluation was not produced")
    with (args.output_dir / "reference_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference_summary))
        writer.writeheader()
        writer.writerow(reference_summary)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "eval_manifest.json").write_text(json.dumps({
        "ta_script": str(args.ta_script.resolve()),
        "musdr_source": str((args.musdr_parent / "MusDr").resolve()),
        "reference_rule": "first 32 bars of fixed-4/4 training songs, no held-out claim",
        "reference_sequences": str(args.training_sequences.resolve()),
        "shared_vocabulary": str(args.vocab.resolve()),
        "configs": args.configs, "songs_per_config": args.songs,
    }, indent=2), encoding="utf-8")
    print(f"{len(summaries)} Task A experiment rows: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
