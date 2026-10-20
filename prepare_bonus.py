"""Build bar-interleaved MELODY -> PIANO training sequences from POP909.

Only corrected training MIDI files are read. BRIDGE and bonus prompt piano files
are never training inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from symusic import Score

from bonus_tokens import interleave, training_bars, vocabulary


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi-dir", type=Path, default=Path("POP909/training_timesig_fixed_col3"))
    parser.add_argument("--training-dir", type=Path, default=Path("POP909/training"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/bonus_interleaved"))
    parser.add_argument("--expected-songs", type=int, default=896)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    manifest_path = args.midi_dir / "timesig_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != args.expected_songs:
        raise ValueError(f"Expected {args.expected_songs} manifest rows, found {len(manifest)}")
    folders = {p.name for p in args.training_dir.iterdir() if p.is_dir()}
    song_ids = {row["song_id"] for row in manifest}
    if len(song_ids) != args.expected_songs or song_ids != folders:
        raise ValueError("Corrected MIDI manifest does not match the POP909 training split")
    rows = []
    bars_total = 0
    tokens_total = 0
    for index, row in enumerate(sorted(manifest, key=lambda item: item["song_id"]), 1):
        if row["status"] != "written":
            raise ValueError(f"{row['song_id']}: corrected MIDI was not written")
        song_id = row["song_id"]
        path = args.midi_dir / song_id / f"{song_id}.mid"
        bars = training_bars(Score(str(path)))
        tokens = interleave(bars)
        rows.append({"song_name": song_id, "tokens": tokens})
        bars_total += len(bars)
        tokens_total += len(tokens)
        if index % 100 == 0 or index == len(manifest):
            print(f"Prepared {index}/{len(manifest)} songs", flush=True)
    metadata = {
        "task": "bonus_accompaniment",
        "representation": "bar-interleaved MELODY then PIANO; Pos/Pitch/Vel/Dur events",
        "units_per_bar": 64,
        "max_duration_units": 256,
        "loss": "next-token cross entropy on all tokens",
        "chord_source": None,
        "songs": len(rows),
        "training_split": str(args.training_dir.resolve()),
        "corrected_midi_dir": str(args.midi_dir.resolve()),
        "timesig_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_sequences.json").write_text(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (args.output_dir / "tok2idx.json").write_text(
        json.dumps(vocabulary(), indent=2), encoding="utf-8")
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} songs, {bars_total} bars, {tokens_total} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()
