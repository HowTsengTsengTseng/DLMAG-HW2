"""Convert our REMI sequences to the token format expected by TA eval_basic.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Task A configuration directory, or train_sequences.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vocab", type=Path, default=Path(
        "data/miditok_remi_no_chords/tok2idx.json"),
                        help="Shared training vocabulary for identical metric bins")
    parser.add_argument("--songs", type=int, default=20,
                        help="Expected generated song count; ignored for training reference")
    return parser.parse_args()


def first_32_bars(tokens: list[str]) -> list[str] | None:
    starts = [i for i, token in enumerate(tokens) if token.startswith("Bar_")]
    if len(starts) < 32:
        return None
    end = starts[32] if len(starts) > 32 else len(tokens)
    return tokens[starts[0]:end]


def meter_is_4_4(tokens: list[str]) -> bool:
    signatures = [token for token in tokens if token.startswith("TimeSig_")]
    return bool(signatures) and set(signatures) == {"TimeSig_4/4"}


def convert(tokens: list[str]) -> tuple[list[str], int]:
    result = []
    current_position: str | None = None
    position_emitted = False
    orphan_pitches = 0
    for token in tokens:
        if token.startswith("Bar_"):
            result.append("Bar")
            current_position = None
            position_emitted = False
        elif token.startswith("Position_"):
            current_position = "Pos_" + token.split("_", 1)[1]
            position_emitted = False
        elif token.startswith("Pitch_"):
            if current_position is None:
                # A sampled token stream can violate REMI event order. Do not
                # invent an onset for a pitch that the MIDI decoder cannot place.
                orphan_pitches += 1
                continue
            if not position_emitted:
                result.append(current_position)
                position_emitted = True
            result.append(token)
    if result.count("Bar") != 32:
        raise ValueError("Converted sequence is not 32 bars")
    return result, orphan_pitches


def main() -> None:
    args = arguments()
    reference = args.input.is_file()
    if reference:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    else:
        files = sorted(args.input.glob("*.tokens.json"))
        if len(files) != args.songs:
            raise ValueError(f"Expected {args.songs} token files; found {len(files)}")
        data = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        for item in data:
            midi = args.input / f"{item['song_name']}.mid"
            if not midi.is_file():
                raise FileNotFoundError(f"Corresponding MIDI is missing: {midi}")
    output = []
    skipped = []
    malformed = []
    for item in data:
        raw_tokens = item["tokens"]
        clipped = first_32_bars(raw_tokens)
        if clipped is None:
            if not reference:
                raise ValueError(f"Generated piece lacks 32 bars: {item['song_name']}")
            skipped.append({"song_name": item["song_name"], "reason": "fewer than 32 bars"})
            continue
        if not meter_is_4_4(clipped):
            if not reference:
                raise ValueError(f"Generated piece changes meter: {item['song_name']}")
            skipped.append({"song_name": item["song_name"], "reason": "not fixed 4/4"})
            continue
        converted, orphan_pitches = convert(clipped)
        output.append({"song_name": item["song_name"], "tokens": converted})
        first_bar = next(i for i, token in enumerate(raw_tokens)
                         if token.startswith("Bar_"))
        leading_events = [token for token in raw_tokens[:first_bar]
                          if token not in {"<BOS>", "<PAD>"}]
        if orphan_pitches or leading_events:
            malformed.append({
                "song_name": item["song_name"],
                "orphan_pitches_ignored": orphan_pitches,
                "events_before_first_bar_ignored": len(leading_events),
            })
    if not output:
        raise ValueError("No eligible 32-bar pieces")
    training_vocab = json.loads(args.vocab.read_text(encoding="utf-8"))
    positions = sorted(("Pos_" + key.split("_", 1)[1]
                        for key in training_vocab if key.startswith("Position_")),
                       key=lambda value: int(value.split("_")[1]))
    pitches = sorted((key for key in training_vocab if key.startswith("Pitch_")),
                     key=lambda value: int(value.split("_")[1]))
    vocabulary = ["Bar", *positions, *pitches]
    if not positions or not pitches:
        raise ValueError("No position or pitch tokens for evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sequences.json").write_text(
        json.dumps(output, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "tok2idx.json").write_text(
        json.dumps({token: i for i, token in enumerate(vocabulary)}), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "source": str(args.input), "reference": reference,
        "included": [item["song_name"] for item in output], "skipped": skipped,
        "malformed_events": malformed,
        "rule": "first 32 complete 4/4 bars; Bar and Pitch plus positions containing notes",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(output)} pieces; skipped {len(skipped)}; "
          f"malformed event sequences {len(malformed)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
