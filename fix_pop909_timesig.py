"""Infer and write MIDI time signatures from POP909 beat annotations.

POP909 beat_midi.txt stores beat times followed by two alternative downbeat
tracks (downbeat_1 and downbeat_2). It does not directly store a numerator or
denominator. Select the appropriate downbeat track for the meter hypothesis
being used; the beat unit is inferred by comparing annotation pulse times with
the MIDI tempo map. Ambiguous files are recorded and skipped.

Original MIDI files are never modified. Corrected files and a CSV manifest are
written under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import mido
import pretty_midi


@dataclass
class Inference:
    numerator: int
    denominator: int
    pulse_quarters: float
    bar_pulses: int
    bar_confidence: float
    pulse_confidence: float
    first_downbeat_sec: float


def read_beat_annotations(path: Path) -> tuple[list[float], list[tuple[float, float]]]:
    """Return beat times and the two downbeat flags from the three-column file."""
    times: list[float] = []
    flags: list[tuple[float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 3:
            raise ValueError(f"{path}:{line_number}: expected 3 columns, found {len(fields)}")
        time_sec, downbeat_1, downbeat_2 = map(float, fields[:3])
        times.append(time_sec)
        flags.append((downbeat_1, downbeat_2))
    if len(times) < 4:
        raise ValueError("fewer than four beat annotations")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("beat timestamps are not strictly increasing")
    return times, flags


def _mode(values: list[int]) -> tuple[int, float]:
    counts = Counter(values)
    highest = max(counts.values())
    winners = [value for value, count in counts.items() if count == highest]
    if len(winners) != 1:
        raise ValueError(f"ambiguous repeated interval counts: {dict(counts)}")
    value = winners[0]
    return value, counts[value] / len(values)


def infer_signature(
    midi: pretty_midi.PrettyMIDI,
    beat_times: list[float],
    flags: list[tuple[float, float]],
    downbeat_column: int,
    min_confidence: float,
) -> Inference:
    flag_index = downbeat_column - 2
    downbeat_rows = [i for i, pair in enumerate(flags) if pair[flag_index] >= 0.5]
    if len(downbeat_rows) < 3:
        raise ValueError(f"downbeat column {downbeat_column} has fewer than 3 markers")

    gaps = [right - left for left, right in zip(downbeat_rows, downbeat_rows[1:])]
    bar_pulses, bar_confidence = _mode(gaps)
    if bar_confidence < min_confidence:
        raise ValueError(
            f"downbeat interval confidence {bar_confidence:.2f} is below {min_confidence:.2f}"
        )

    # MIDI ticks are measured in quarter-note units. Comparing beat annotations
    # against the file's tempo map distinguishes quarter-note and eighth-note
    # annotation pulses even when its time-signature metadata is wrong.
    ticks = [midi.time_to_tick(time_sec) for time_sec in beat_times]
    pulse_quarters_all = [
        (right - left) / midi.resolution
        for left, right in zip(ticks, ticks[1:])
        if right > left
    ]
    if len(pulse_quarters_all) < 3:
        raise ValueError("could not map beat annotations onto MIDI ticks")

    pulse_quarters = median(pulse_quarters_all)
    pulse_tolerance = max(2 / midi.resolution, pulse_quarters * 0.05)
    pulse_confidence = sum(
        abs(value - pulse_quarters) <= pulse_tolerance for value in pulse_quarters_all
    ) / len(pulse_quarters_all)
    if pulse_confidence < min_confidence:
        raise ValueError(
            f"beat-unit confidence {pulse_confidence:.2f} is below {min_confidence:.2f}"
        )

    # A denominator note has 4 / denominator quarter-note units.
    denominators = (1, 2, 4, 8, 16, 32)
    denominator = min(denominators, key=lambda d: abs(4 / d - pulse_quarters))
    denominator_error = abs((4 / denominator) - pulse_quarters) / pulse_quarters
    if denominator_error > 0.15:
        raise ValueError(
            f"beat pulse is {pulse_quarters:.3f} quarter notes; no standard denominator fits"
        )

    bar_quarters = bar_pulses * pulse_quarters
    numerator = round(bar_quarters * denominator / 4)
    if numerator < 1 or numerator > 32:
        raise ValueError(f"inferred numerator {numerator} is outside the supported range")
    bar_error = abs((numerator * 4 / denominator) - bar_quarters) / bar_quarters
    if bar_error > 0.05:
        raise ValueError("inferred numerator and denominator do not fit the bar duration")

    return Inference(
        numerator=numerator,
        denominator=denominator,
        pulse_quarters=pulse_quarters,
        bar_pulses=bar_pulses,
        bar_confidence=bar_confidence,
        pulse_confidence=pulse_confidence,
        first_downbeat_sec=beat_times[downbeat_rows[0]],
    )


def write_time_signature(source: Path, destination: Path, numerator: int, denominator: int) -> None:
    """Replace time-signature events while preserving all other MIDI events."""
    midi = mido.MidiFile(source)
    if midi.type == 2:
        raise ValueError("type-2 MIDI tracks are independent sequences; refusing to rewrite")

    # Convert delta ticks to absolute ticks, remove old signatures on every
    # track, add one signature at tick zero, then restore delta-time encoding.
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
        order = 0
        for message in track:
            absolute += message.time
            if message.type != "time_signature":
                events.append((absolute, order, message.copy(time=0)))
                order += 1
        if track_index == 0:
            signature = mido.MetaMessage(
                "time_signature", numerator=numerator, denominator=denominator, time=0
            )
            events.append((0, -1, signature))
        events.sort(key=lambda event: (event[0], event[1]))
        previous_tick = 0
        rewritten = []
        for absolute_tick, _, message in events:
            rewritten.append(message.copy(time=absolute_tick - previous_tick))
            previous_tick = absolute_tick
        track[:] = rewritten

    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("POP909/training"))
    parser.add_argument("--output-dir", type=Path, default=Path("POP909/training_timesig_fixed"))
    parser.add_argument(
        "--downbeat-column",
        type=int,
        choices=(2, 3),
        required=True,
        help="Choose downbeat_1 (2) or downbeat_2 (3) from beat_midi.txt",
    )
    parser.add_argument("--min-confidence", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.min_confidence <= 1:
        raise ValueError("--min-confidence must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "timesig_manifest.csv"
    columns = [
        "song_id", "status", "time_signature", "downbeat_column", "bar_pulses",
        "bar_confidence", "pulse_quarters", "pulse_confidence", "first_downbeat_sec", "reason",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=columns)
        writer.writeheader()
        folders = sorted(path for path in args.dataset_dir.iterdir() if path.is_dir())
        for folder in folders:
            song_id = folder.name
            midi_path = folder / f"{song_id}.mid"
            beats_path = folder / "beat_midi.txt"
            row = {"song_id": song_id, "status": "skipped", "downbeat_column": args.downbeat_column}
            try:
                if not midi_path.is_file() or not beats_path.is_file():
                    raise FileNotFoundError("expected MIDI and beat_midi.txt in song folder")
                beat_times, flags = read_beat_annotations(beats_path)
                pm = pretty_midi.PrettyMIDI(str(midi_path))
                inferred = infer_signature(
                    pm, beat_times, flags, args.downbeat_column, args.min_confidence
                )
                output_midi = args.output_dir / song_id / f"{song_id}.mid"
                write_time_signature(
                    midi_path, output_midi, inferred.numerator, inferred.denominator
                )
                row.update({
                    "status": "written",
                    "time_signature": f"{inferred.numerator}/{inferred.denominator}",
                    "bar_pulses": inferred.bar_pulses,
                    "bar_confidence": f"{inferred.bar_confidence:.3f}",
                    "pulse_quarters": f"{inferred.pulse_quarters:.3f}",
                    "pulse_confidence": f"{inferred.pulse_confidence:.3f}",
                    "first_downbeat_sec": f"{inferred.first_downbeat_sec:.6f}",
                })
            except Exception as exc:
                row["reason"] = str(exc)
                print(f"SKIP {song_id}: {exc}")
            else:
                print(f"OK {song_id}: {row['time_signature']} -> {output_midi}")
            writer.writerow(row)

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
