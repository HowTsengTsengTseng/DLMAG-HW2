"""Convert POP909 time-signature-corrected MIDI and chord labels to REMI.

Run after fix_pop909_timesig.py. The output JSON files are directly accepted by
train.py; miditok_tokenizer.json preserves the MidiTok tokenizer for decoding.

POP909's MELODY, BRIDGE, and PIANO tracks all use MIDI program 0. MidiTok merges
tracks with the same program in one-stream mode, so this script assigns distinct
programs in memory before encoding. The corrected MIDI files remain unchanged.
Decoded MIDI must map programs 0, 1, and 2 back to the original piano program 0
when POP909-style instrument playback is needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import pretty_midi
from miditok import Event, REMI, TokSequence, TokenizerConfig
from symusic import Score


TRACK_PROGRAMS = {"MELODY": 0, "BRIDGE": 1, "PIANO": 2}
BOUNDARY_TOKENS = ("<PAD>", "<BOS>", "<EOS>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--midi-dir", type=Path, default=Path("POP909/training_timesig_fixed_col3"),
        help="Directory of corrected MIDI files and their timesig_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/miditok_remi_chords"))
    parser.add_argument(
        "--chord-dir", type=Path, default=Path("POP909/training"),
        help="POP909 directory containing each song's chord_midi.txt",
    )
    parser.add_argument(
        "--no-chords", action="store_true",
        help="Omit POP909 chord labels and emit the original note-only REMI representation",
    )
    parser.add_argument("--expected-songs", type=int, default=896)
    return parser.parse_args()


def read_manifest(path: Path, midi_dir: Path, expected_songs: int) -> tuple[list[tuple[str, Path]], dict[int, list[int]], Counter]:
    if not path.is_file():
        raise FileNotFoundError(f"Time-signature manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_songs:
        raise ValueError(f"Expected {expected_songs} manifest rows; found {len(rows)}")

    files: list[tuple[str, Path]] = []
    signatures: dict[int, set[int]] = defaultdict(set)
    signature_counts: Counter = Counter()
    seen: set[str] = set()
    for row in rows:
        song_id = row["song_id"]
        if song_id in seen:
            raise ValueError(f"Duplicate song in manifest: {song_id}")
        seen.add(song_id)
        if row["status"] != "written":
            raise ValueError(f"{song_id}: signature correction was {row['status']}: {row['reason']}")
        numerator, denominator = map(int, row["time_signature"].split("/"))
        signatures[denominator].add(numerator)
        signature_counts[row["time_signature"]] += 1
        midi_path = midi_dir / song_id / f"{song_id}.mid"
        if not midi_path.is_file():
            raise FileNotFoundError(f"Corrected MIDI missing: {midi_path}")
        files.append((song_id, midi_path))
    return sorted(files), {d: sorted(nums) for d, nums in signatures.items()}, signature_counts


def prepare_score(midi_path: Path) -> tuple[Score, dict[str, int]]:
    score = Score(str(midi_path))
    seen: set[str] = set()
    note_counts: dict[str, int] = {}
    for track in score.tracks:
        name = track.name.strip().upper()
        if name not in TRACK_PROGRAMS or name in seen or track.is_drum:
            raise ValueError(f"Unexpected or repeated POP909 track: {track.name!r}")
        seen.add(name)
        note_counts[name] = len(track.notes)
        track.program = TRACK_PROGRAMS[name]
    if seen != set(TRACK_PROGRAMS):
        raise ValueError(f"Expected {sorted(TRACK_PROGRAMS)}, found {sorted(seen)}")
    return score, note_counts


def read_chords(path: Path) -> list[tuple[float, str]]:
    """Read chord changes in seconds, retaining N and slash-chord labels."""
    if not path.is_file():
        raise FileNotFoundError(f"POP909 chord annotation missing: {path}")
    changes: list[tuple[float, str]] = []
    previous_end = -1.0
    previous_label = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{line_number}: expected start, end, and chord label")
        start, end = map(float, fields[:2])
        label = fields[2]
        if start < 0 or end <= start or start < previous_end - 0.01:
            raise ValueError(f"{path}:{line_number}: invalid or overlapping chord interval")
        if "_" in label:
            raise ValueError(f"{path}:{line_number}: chord label contains reserved '_' character")
        if label != previous_label:
            changes.append((start, label))
        previous_end, previous_label = end, label
    if not changes:
        raise ValueError(f"{path}: no chord annotations")
    return changes


def insert_chords(
    tokenizer: REMI, sequence, midi_path: Path, original_tpq: int,
    processed_tpq: int, changes: list[tuple[float, str]],
):
    """Place annotated chords on MidiTok's quantized REMI timeline."""
    if sequence.events is None:
        raise ValueError("MidiTok did not return timed events")
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    events = [
        event for event in sequence.events
        if event.type_ not in {"Bar", "Position"}
        and (event.type_ != "TimeSig" or event.time == 0)
    ]
    for seconds, label in changes:
        tick = midi.time_to_tick(seconds)
        processed_tick = max(0, round(tick * processed_tpq / original_tpq))
        chord_token = f"Chord_{label}"
        if chord_token not in tokenizer.vocab:
            tokenizer.add_to_vocab(chord_token)
        events.append(Event("Chord", label, processed_tick))

    # Chord changes at a note onset precede the note's Program/Pitch events.
    priority = {"TimeSig": 0, "Tempo": 1, "Chord": 2}
    events.sort(key=lambda event: (event.time, priority.get(event.type_, 3)))
    events = tokenizer._add_time_events(events, processed_tpq)
    annotated_sequence = TokSequence(events=events)
    tokenizer.complete_sequence(annotated_sequence)
    return annotated_sequence


def make_tokenizer_config(signature_range: dict[int, list[int]], with_chords: bool) -> TokenizerConfig:
    """Keep production and comparison scripts on the same REMI settings."""
    return TokenizerConfig(
        beat_res={(0, 4): 16, (4, 12): 8},
        pitch_range=(0, 127),
        num_velocities=32,
        use_tempos=True,
        use_time_signatures=True,
        time_signature_range=signature_range,
        use_programs=True,
        one_token_stream_for_programs=True,
        program_changes=False,
        programs=list(TRACK_PROGRAMS.values()),
        use_chords=with_chords,
        chord_tokens_with_root_note=True,
    )


def main() -> None:
    args = parse_args()
    if args.expected_songs < 1:
        raise ValueError("--expected-songs must be positive")
    manifest_path = args.midi_dir / "timesig_manifest.csv"
    files, signature_range, signature_counts = read_manifest(
        manifest_path, args.midi_dir, args.expected_songs
    )

    config = make_tokenizer_config(signature_range, with_chords=not args.no_chords)
    tokenizer = REMI(tokenizer_config=config)
    if not tokenizer.one_token_stream:
        raise RuntimeError("REMI must produce one token stream per song")

    sequences: list[dict[str, object]] = []
    observed_tokens: set[str] = set()
    token_count = 0
    chord_count = 0
    for index, (song_id, midi_path) in enumerate(files, 1):
        try:
            score, note_counts = prepare_score(midi_path)
            changes = [] if args.no_chords else read_chords(
                args.chord_dir / song_id / "chord_midi.txt"
            )
            original_tpq = score.ticks_per_quarter
            # MidiTok's built-in chord detection uses notes. POP909 supplies
            # curated chord labels, so disable detection for this encode and
            # insert the annotations as timed events afterward.
            tokenizer.config.use_chords = False
            try:
                processed_score = tokenizer.preprocess_score(score)
                sequence = tokenizer.encode(processed_score, no_preprocess_score=True)
            finally:
                tokenizer.config.use_chords = not args.no_chords
            if changes:
                sequence = insert_chords(
                    tokenizer, sequence, midi_path, original_tpq,
                    processed_score.ticks_per_quarter, changes,
                )
            if isinstance(sequence, list) or sequence.tokens is None:
                raise ValueError("MidiTok did not return one string-token sequence")
            tokens = list(sequence.tokens)
            if not tokens:
                raise ValueError("MidiTok produced an empty sequence")
            if any(token in BOUNDARY_TOKENS for token in tokens):
                raise ValueError("MidiTok emitted a reserved training boundary token")
            if not any(token.startswith("TimeSig_") for token in tokens):
                raise ValueError("MidiTok produced no time-signature tokens")
            if changes and not any(token.startswith("Chord_") for token in tokens):
                raise ValueError("POP909 chord annotations were not tokenized")
            for name, count in note_counts.items():
                if count and f"Program_{TRACK_PROGRAMS[name]}" not in tokens:
                    raise ValueError(f"No Program token for populated {name} track")
        except Exception as exc:
            raise RuntimeError(f"Failed to tokenize {song_id}: {exc}") from exc

        observed_tokens.update(tokens)
        token_count += len(tokens)
        chord_count += sum(token.startswith("Chord_") for token in tokens)
        sequences.append({"song_name": song_id, "tokens": ["<BOS>", *tokens, "<EOS>"]})
        if index % 100 == 0 or index == len(files):
            print(f"Tokenized {index}/{len(files)} songs", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.output_dir / "miditok_tokenizer.json"
    # MidiTok normally serializes its full base vocabulary only after BPE
    # training. These annotation labels are added dynamically, so persist the
    # untrained vocabulary explicitly as well.
    tokenizer.save(tokenizer_path, additional_attributes={
        "_vocab_base": tokenizer.vocab,
    })
    vocabulary = {token: index for index, token in enumerate(
        [*BOUNDARY_TOKENS, *sorted(observed_tokens)]
    )}
    metadata = {
        "representation": "MidiTok REMI",
        "miditok_version": version("miditok"),
        "tokenizer_file": tokenizer_path.name,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "corrected_midi_dir": str(args.midi_dir.resolve()),
        "timesig_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "track_programs": TRACK_PROGRAMS,
        "original_track_program": 0,
        "time_signature_range": signature_range,
        "chord_source": None if args.no_chords else "POP909 chord_midi.txt",
        "chord_dir": None if args.no_chords else str(args.chord_dir.resolve()),
        "chord_alignment": "annotation start seconds rounded to the nearest processed MIDI tick",
        "chord_tokens": chord_count,
        "beat_res": {f"{start}-{end}": resolution for (start, end), resolution in config.beat_res.items()},
        "boundary_tokens": BOUNDARY_TOKENS,
        "songs": len(files),
    }
    (args.output_dir / "train_sequences.json").write_text(
        json.dumps(sequences, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (args.output_dir / "tok2idx.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {len(files)} songs, {token_count} REMI tokens "
        f"({chord_count} Chord tokens), "
        f"{len(vocabulary)} vocabulary entries to {args.output_dir}"
    )
    print(f"Inferred time signatures: {dict(signature_counts)}")


if __name__ == "__main__":
    main()
