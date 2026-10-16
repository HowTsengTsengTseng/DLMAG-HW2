"""Render the selected Basic A and Basic B MIDI sets to WAV for submission."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_experiments import CONFIGS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--a-config", required=True, choices=tuple(CONFIGS))
    parser.add_argument("--b-configs", required=True, nargs=3, choices=tuple(CONFIGS))
    parser.add_argument("--best", required=True, nargs=3, choices=tuple(CONFIGS),
                        help="Best B config for prompts 001, 002, 003 respectively")
    parser.add_argument("--sample-rate", type=int, default=44100)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not args.soundfont.is_file():
        raise FileNotFoundError(args.soundfont)
    if len(set(args.b_configs)) != 3:
        raise ValueError("Select three distinct Basic B configurations")
    if not set(args.best) <= set(args.b_configs):
        raise ValueError("Best B configurations must be among --b-configs")
    if args.sample_rate < 1:
        raise ValueError("sample-rate must be positive")
    jobs = []
    for seed in range(1000, 1020):
        name = f"{seed}.mid"
        jobs.append((args.results_root / "basic_a" / args.a_config / name,
                     args.output_dir / "basic_a" / name))
    for config in args.b_configs:
        for pid in ("001", "002", "003"):
            name = f"{pid}.mid"
            jobs.append((args.results_root / "basic_b" / config / name,
                         args.output_dir / "basic_b" / config / name))
    missing = [str(source) for source, _ in jobs if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing generated MIDI files: {missing[:5]}")
    conflicts = [str(output) for _, output in jobs if output.exists() or output.with_suffix(".wav").exists()]
    if conflicts:
        raise FileExistsError(f"Submission output already exists: {conflicts[:5]}")
    for source, output in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        subprocess.run([sys.executable, str(Path(__file__).with_name("midi2wav.py")),
                        str(output), str(output.with_suffix(".wav")),
                        "--soundfont", str(args.soundfont),
                        "--sample-rate", str(args.sample_rate)], check=True)
    manifest = {
        "basic_a_config": args.a_config,
        "basic_b_configs": args.b_configs,
        "best_basic_b_by_prompt": dict(zip(("001", "002", "003"), args.best)),
        "soundfont": str(args.soundfont), "sample_rate": args.sample_rate,
        "midi_wav_pairs": len(jobs),
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rendered {len(jobs)} MIDI/WAV pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
