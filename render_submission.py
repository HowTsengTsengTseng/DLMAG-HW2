"""Collect Basic A, Basic B, and Bonus MIDI/JSON files and render every MIDI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


MIDI_EXTENSIONS = {".mid", ".midi"}
GROUPS = (("basic_a", "basic_a_dir"), ("basic_b", "basic_b_dir"),
          ("bonus", "bonus_dir"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic-a-dir", required=True, type=Path,
                        help="Directory containing all Basic A MIDI and JSON files")
    parser.add_argument("--basic-b-dir", required=True, type=Path,
                        help="Directory containing all Basic B MIDI and JSON files")
    parser.add_argument("--bonus-dir", required=True, type=Path,
                        help="Directory containing all Bonus MIDI and JSON files")
    parser.add_argument("--output-dir", type=Path, default=Path("submission"))
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=44100)
    return parser.parse_args()


def collect_group(source_dir: Path, output_dir: Path) -> tuple[list[tuple[Path, Path]],
                                                                list[tuple[Path, Path]]]:
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    midi_files = sorted(path for path in source_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS)
    if not midi_files:
        raise ValueError(f"No MIDI files found in {source_dir}")
    missing_metadata = [str(path.with_suffix(".json")) for path in midi_files
                        if not path.with_suffix(".json").is_file()]
    if missing_metadata:
        raise FileNotFoundError(f"Missing MIDI metadata JSON: {missing_metadata[:5]}")
    midis = [(path, output_dir / path.relative_to(source_dir)) for path in midi_files]
    jsons = [(path, output_dir / path.relative_to(source_dir))
             for path in sorted(source_dir.rglob("*.json")) if path.is_file()]
    return midis, jsons


def main() -> None:
    args = arguments()
    soundfont = args.soundfont.expanduser().resolve()
    if not soundfont.is_file():
        raise FileNotFoundError(soundfont)
    if args.sample_rate < 1:
        raise ValueError("sample-rate must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    midi_jobs: list[tuple[Path, Path]] = []
    json_jobs: list[tuple[Path, Path]] = []
    counts: dict[str, int] = {}
    for group, option in GROUPS:
        source_dir = getattr(args, option).expanduser().resolve()
        midis, jsons = collect_group(source_dir, output_dir / group)
        midi_jobs.extend(midis)
        json_jobs.extend(jsons)
        counts[group] = len(midis)

    destinations = [destination for _, destination in [*midi_jobs, *json_jobs]]
    destinations.extend(destination.with_suffix(".wav") for _, destination in midi_jobs)
    if len(destinations) != len(set(destinations)):
        raise ValueError("Two source files would use the same submission output path")
    conflicts = [str(path) for path in destinations if path.exists()]
    if conflicts:
        raise FileExistsError(f"Submission output already exists: {conflicts[:5]}")

    for source, destination in [*midi_jobs, *json_jobs]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for _, destination in midi_jobs:
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("midi2wav.py")),
            str(destination), str(destination.with_suffix(".wav")),
            "--soundfont", str(soundfont), "--sample-rate", str(args.sample_rate),
        ], check=True)
    print(f"Rendered {sum(counts.values())} MIDI/WAV pairs to {output_dir} "
          f"(Basic A: {counts['basic_a']}, Basic B: {counts['basic_b']}, "
          f"Bonus: {counts['bonus']})")


if __name__ == "__main__":
    main()
