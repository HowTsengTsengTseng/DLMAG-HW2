"""Generate one unconditional 32-bar Basic A MIDI from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mido
import torch
from miditok import REMI, TokSequence
from symusic import Note, Score, Tempo, TimeSignature, Track

from infer import ROLES, TPQ, continue_tokens, file_hash
from model import ModelConfig, TransformerXL


BARS = 32


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "data/miditok_remi_chords/miditok_tokenizer.json"))
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--top-k", type=int)
    sampling.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tempo", type=float, default=120.0,
                        help="Starting BPM, snapped to the checkpoint tempo vocabulary")
    parser.add_argument("--max-new-tokens", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--experiment-id")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def initial_tokens(vocab: dict[str, int], tempo: float) -> tuple[list[str], float]:
    options = [(abs(float(token[6:]) - tempo), token)
               for token in vocab if token.startswith("Tempo_")]
    if not options or "TimeSig_4/4" not in vocab or "Position_0" not in vocab:
        raise ValueError("Checkpoint lacks required 4/4 or tempo tokens")
    tempo_token = min(options)[1]
    return ["Bar_None", "TimeSig_4/4", "Position_0", tempo_token], float(tempo_token[6:])


def write_midi(tokens: list[str], tokenizer: REMI, path: Path, initial_bpm: float) -> int:
    decoded = tokenizer.decode(TokSequence(tokens=tokens))
    if TPQ % decoded.ticks_per_quarter:
        raise ValueError("Tokenizer resolution does not divide MIDI output resolution")
    ratio = TPQ // decoded.ticks_per_quarter
    end = BARS * 4 * TPQ
    result = Score(TPQ)
    result.time_signatures.append(TimeSignature(0, 4, 4))
    result.tempos.append(Tempo(0, initial_bpm))
    for tempo in decoded.tempos:
        tick = tempo.time * ratio
        if 0 < tick < end:
            result.tempos.append(Tempo(tick, tempo.qpm))
    names = {program: role for role, program in ROLES.items()}
    tracks = {role: Track(name=role, program=0) for role in ROLES}
    count = 0
    for source in decoded.tracks:
        if source.program not in names:
            raise ValueError(f"Generated unknown program {source.program}")
        dest = tracks[names[source.program]]
        for note in source.notes:
            onset = note.time * ratio
            note_end = min(end, note.end * ratio)
            if 0 <= onset < note_end:
                dest.notes.append(Note(onset, note_end - onset, note.pitch, note.velocity))
                count += 1
    if not count:
        raise RuntimeError("Generated piece contains no playable notes")
    result.tracks.extend(tracks.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    result.dump_midi(path)
    midi = mido.MidiFile(path)
    for track in midi.tracks:
        if not track or track[-1].type != "end_of_track":
            raise RuntimeError("MIDI track has no End-of-Track event")
        content_end = sum(event.time for event in track[:-1])
        if content_end > end:
            raise RuntimeError("MIDI content extends beyond 32 bars")
        track[-1].time = end - content_end
    midi.save(path)
    return count


def main() -> None:
    args = arguments()
    if args.output.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("--output must end in .mid or .midi")
    if args.temperature <= 0 or args.tempo <= 0 or args.max_new_tokens < 1:
        raise ValueError("temperature, tempo and max-new-tokens must be positive")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("top-k must be positive")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab: dict[str, int] = checkpoint["vocab"]
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Checkpoint vocabulary IDs are not contiguous")
    metadata = checkpoint["tokenizer_config"]
    tokenizer_hash = file_hash(args.tokenizer)
    if tokenizer_hash != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer file does not match checkpoint")
    if metadata["track_programs"] != ROLES:
        raise ValueError("Checkpoint uses a different track mapping")
    tokenizer = REMI(params=args.tokenizer)
    if not tokenizer.one_token_stream:
        raise ValueError("Expected a single-stream REMI tokenizer")
    prefix, actual_bpm = initial_tokens(vocab, args.tempo)
    model = TransformerXL(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generated = continue_tokens(model, vocab, prefix, args, device,
                                bars_to_generate=BARS)
    tokens = prefix + generated
    if sum(token.startswith("Bar_") for token in tokens) != BARS:
        raise RuntimeError("Generated token sequence is not 32 bars")
    notes = write_midi(tokens, tokenizer, args.output, actual_bpm)
    record = {
        "experiment_id": args.experiment_id or args.output.parent.name,
        "checkpoint_hash": file_hash(args.checkpoint),
        "tokenizer_hash": tokenizer_hash,
        "output_midi": str(args.output),
        "seed": args.seed, "temperature": args.temperature,
        "top_k": 50 if args.top_k is None and args.top_p is None else args.top_k or 0,
        "top_p": args.top_p, "memory_length": model.cfg.memory_len,
        "total_bars": BARS, "meter_policy": "fixed 4/4; TimeSig tokens blocked",
        "starting_tempo_bpm": actual_bpm, "generated_tokens": len(generated),
        "notes": notes, "status": "written",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".tokens.json").write_text(
        json.dumps({"song_name": args.output.stem, "tokens": tokens}, ensure_ascii=False),
        encoding="utf-8")
    print(f"Wrote {args.output}: 32 bars; {notes} notes")


if __name__ == "__main__":
    main()
