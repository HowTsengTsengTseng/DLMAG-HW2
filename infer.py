"""Continue an eight-bar POP909 Basic B prompt to a 32-bar MIDI file."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import mido
import pretty_midi
import torch
from miditok import Event, REMI, TokSequence
from symusic import Note, Score, Tempo, TimeSignature, Track

from fix_pop909_timesig import read_beat_annotations
from model import ModelConfig, TransformerXL
from tokenize_miditok import read_chords


PROMPT_BARS = 8
NEW_BARS = 24
TPQ = 480
ROLES = {"MELODY": 0, "BRIDGE": 1, "PIANO": 2}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint for one Basic B configuration")
    parser.add_argument("--all-configs", action="store_true",
                        help="Generate N20T10, N50T10, and C20T10 for all prompts")
    parser.add_argument("--no-chord-checkpoint", type=Path,
                        default=Path("checkpoints/no_chords/step_00030000.pt"))
    parser.add_argument("--chord-checkpoint", type=Path,
                        default=Path("checkpoints/chords/step_00030000.pt"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt-midi", type=Path)
    inputs.add_argument("--prompt-root", type=Path,
                        help="Generate all three Basic B prompts in this directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for 001.mid, 002.mid, and 003.mid")
    parser.add_argument("--beat-file", type=Path, help="Defaults to prompt/beat_midi.txt")
    parser.add_argument("--chord-file", type=Path, help="Defaults to prompt/chord_midi.txt")
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "data/miditok_remi_chords/miditok_tokenizer.json"))
    parser.add_argument("--no-chord-tokenizer", type=Path, default=Path(
        "data/miditok_remi_no_chords/miditok_tokenizer.json"))
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--top-k", type=int, help="Default: 50")
    sampling.add_argument("--top-p", type=float, help="Nucleus sampling, e.g. 0.9")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=20000)
    parser.add_argument("--seed", type=int,
                        help="Default: 2000 for one configuration, 42 for --all-configs")
    parser.add_argument("--experiment-id", help="Recorded in the output metadata")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def beat_grid(path: Path) -> list[float]:
    beats, flags = read_beat_annotations(path)
    if len(beats) != 33 or [i for i, f in enumerate(flags) if f[1] >= 0.5] != list(range(0, 33, 4)):
        raise ValueError("Basic B requires 33 beat markers for eight 4/4 bars")
    if abs(beats[0]) > 0.01:
        raise ValueError("Prompt must start at time zero")
    return beats


def tick_at(seconds: float, beats: list[float], tpq: int) -> int:
    index = max(0, min(bisect_right(beats, seconds) - 1, 31))
    offset = (seconds - beats[index]) / (beats[index + 1] - beats[index])
    return round((index + offset) * tpq)


def prompt_score(path: Path, beats: list[float]) -> tuple[Score, int]:
    """Retain prompt notes on a 480-tick beat grid; model input is quantized separately."""
    source = pretty_midi.PrettyMIDI(str(path))
    score = Score(TPQ)
    score.tempos.append(Tempo(0, 60 * 32 / (beats[-1] - beats[0])))
    score.time_signatures.append(TimeSignature(0, 4, 4))
    seen: set[str] = set()
    note_count = 0
    for instrument in source.instruments:
        role = instrument.name.strip().upper()
        if role not in ROLES or role in seen or instrument.is_drum:
            raise ValueError(f"Unexpected or repeated prompt track: {instrument.name!r}")
        seen.add(role)
        track = Track(name=role, program=ROLES[role])
        for note in instrument.notes:
            start = tick_at(note.start, beats, TPQ)
            end = tick_at(note.end, beats, TPQ)
            if not 0 <= start < 32 * TPQ:
                raise ValueError(f"Prompt note starts outside eight bars: {role}")
            end = min(32 * TPQ, max(start + 1, end))
            track.notes.append(Note(start, end - start, note.pitch, note.velocity))
            note_count += 1
        score.tracks.append(track)
    if not note_count:
        raise ValueError("Prompt contains no notes")
    return score, note_count


def tokenize_prompt(
    score: Score, beats: list[float], chord_file: Path | None,
    tokenizer: REMI, vocab: dict[str, int],
) -> list[str]:
    # This mirrors training: note encoding without automatic chord detection,
    # followed by time-aligned POP909 chord annotation insertion.
    tokenizer.config.use_chords = False
    try:
        processed = tokenizer.preprocess_score(score)
        sequence = tokenizer.encode(processed, no_preprocess_score=True)
    finally:
        tokenizer.config.use_chords = chord_file is not None
    if isinstance(sequence, list) or sequence.events is None:
        raise ValueError("Expected one timed REMI prompt sequence")
    if chord_file is not None:
        events = [e for e in sequence.events if e.type_ not in {"Bar", "Position"}
                  and (e.type_ != "TimeSig" or e.time == 0)]
        for seconds, label in read_chords(chord_file):
            if seconds >= beats[-1]:
                continue
            token = f"Chord_{label}"
            if token not in vocab or token not in tokenizer.vocab:
                raise ValueError(f"Chord absent from training vocabulary: {token}")
            events.append(Event("Chord", label, tick_at(
                seconds, beats, processed.ticks_per_quarter)))
        priority = {"TimeSig": 0, "Tempo": 1, "Chord": 2}
        events.sort(key=lambda e: (e.time, priority.get(e.type_, 3)))
        sequence = TokSequence(events=tokenizer._add_time_events(
            events, processed.ticks_per_quarter))
        tokenizer.complete_sequence(sequence)
    tokens = list(sequence.tokens)
    unknown = sorted(set(tokens) - vocab.keys())
    if not tokens or unknown:
        raise ValueError(f"Prompt tokens absent from checkpoint: {unknown[:8]}")
    bars = sum(t.startswith("Bar_") for t in tokens)
    if bars > PROMPT_BARS:
        raise ValueError(f"Prompt crossed the eight-bar boundary: {bars} bars")
    tokens += ["Bar_None"] * (PROMPT_BARS + 1 - bars)  # start of bar 9
    return tokens


def prompt_quantization_changed(score: Score, tokens: list[str], tokenizer: REMI) -> bool:
    decoded = tokenizer.decode(TokSequence(tokens=tokens))
    if TPQ % decoded.ticks_per_quarter:
        raise ValueError("Tokenizer resolution does not divide output resolution")
    ratio = TPQ // decoded.ticks_per_quarter
    original = sorted(
        (track.program, note.time, note.end, note.pitch, note.velocity)
        for track in score.tracks for note in track.notes
    )
    quantized = sorted(
        (track.program, note.time * ratio, note.end * ratio, note.pitch, note.velocity)
        for track in decoded.tracks for note in track.notes
        if note.time < 32 * decoded.ticks_per_quarter
    )
    return original != quantized


def draw(logits: torch.Tensor, temperature: float, top_k: int,
         top_p: float | None, blocked: set[int]) -> int:
    scores = logits.float().clone() / temperature
    scores[list(blocked)] = -torch.inf
    if top_p is not None:
        order = torch.argsort(scores, descending=True)
        ordered = torch.softmax(scores[order], dim=-1)
        keep = torch.cumsum(ordered, 0) - ordered < top_p
        scores[order[~keep]] = -torch.inf
    elif top_k:
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores[scores < cutoff] = -torch.inf
    probabilities = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Invalid model sampling probabilities")
    return int(torch.multinomial(probabilities, 1).item())


def continue_tokens(model: TransformerXL, vocab: dict[str, int],
                    prefix: list[str], args: argparse.Namespace,
                    device: torch.device, bars_to_generate: int = NEW_BARS) -> list[str]:
    by_id = [""] * len(vocab)
    for token, index in vocab.items():
        by_id[index] = token
    memory, memory_valid = model.empty_memory(1, device)
    ids = [vocab["<BOS>"], *(vocab[t] for t in prefix)]
    top_k = 50 if args.top_k is None and args.top_p is None else args.top_k or 0
    blocked = {vocab[t] for t in ("<PAD>", "<BOS>", "<EOS>")}
    # Both basic tasks use a fixed 4/4 generation grid.
    blocked.update(index for token, index in vocab.items() if token.startswith("TimeSig_"))
    with torch.inference_mode():
        # Feed the full prompt in consecutive segments; memory alone is limited.
        for offset in range(0, len(ids), model.cfg.segment_len):
            chunk = torch.tensor([ids[offset:offset + model.cfg.segment_len]], device=device)
            logits, memory, memory_valid = model(
                chunk, torch.ones_like(chunk, dtype=torch.bool), memory, memory_valid)
        next_logits = logits[0, -1]
        generated: list[str] = []
        bars = 0
        for _ in range(args.max_new_tokens):
            token_id = draw(next_logits, args.temperature, top_k, args.top_p, blocked)
            token = by_id[token_id]
            if token.startswith("Bar_"):
                bars += 1
                if bars == bars_to_generate:
                    return generated  # bar 33 boundary is excluded
            generated.append(token)
            chunk = torch.tensor([[token_id]], device=device)
            logits, memory, memory_valid = model(
                chunk, torch.ones_like(chunk, dtype=torch.bool), memory, memory_valid)
            next_logits = logits[0, -1]
    raise RuntimeError(
        f"The bar after {bars_to_generate} generated bars was not reached "
        f"within {args.max_new_tokens} tokens"
    )


def write_midi(prompt: Score, tokens: list[str], tokenizer: REMI, path: Path) -> int:
    decoded = tokenizer.decode(TokSequence(tokens=tokens))
    if TPQ % decoded.ticks_per_quarter:
        raise ValueError("Tokenizer resolution does not divide output resolution")
    ratio = TPQ // decoded.ticks_per_quarter
    start, end = 32 * TPQ, 128 * TPQ
    result = Score(TPQ)
    result.time_signatures.append(TimeSignature(0, 4, 4))
    result.tempos.append(Tempo(0, prompt.tempos[0].qpm))
    for tempo in decoded.tempos:
        tick = tempo.time * ratio
        if start <= tick < end:
            result.tempos.append(Tempo(tick, tempo.qpm))
    tracks = {name: Track(name=name, program=0) for name in ROLES}
    for track in prompt.tracks:
        for note in track.notes:
            tracks[track.name].notes.append(Note(
                note.time, note.duration, note.pitch, note.velocity))
    names = {program: name for name, program in ROLES.items()}
    added = 0
    for track in decoded.tracks:
        if track.program not in names:
            raise ValueError(f"Generated unknown program {track.program}")
        destination = tracks[names[track.program]]
        for note in track.notes:
            onset = note.time * ratio
            if not start <= onset < end:
                continue
            note_end = min(end, note.end * ratio)
            if note_end > onset:
                destination.notes.append(Note(
                    onset, note_end - onset, note.pitch, note.velocity))
                added += 1
    if not added:
        raise RuntimeError("Continuation contains no playable notes")
    result.tracks.extend(tracks.values())
    path.parent.mkdir(parents=True, exist_ok=True)
    result.dump_midi(path)
    # Symusic writes End-of-Track one tick after the last content event.
    # Set that marker to bar 33 without treating its padding tick as music.
    midi = mido.MidiFile(path)
    for track in midi.tracks:
        if not track or track[-1].type != "end_of_track":
            raise RuntimeError("MIDI track has no final End-of-Track event")
        content_end = sum(message.time for message in track[:-1])
        if content_end > end:
            raise RuntimeError(
                f"MIDI content extends {content_end - end} ticks past bar 32"
            )
        track[-1].time = end - content_end
    midi.save(path)
    return added


def generate_one(args: argparse.Namespace, tokenizer: REMI,
                 model: TransformerXL, vocab: dict[str, int],
                 metadata: dict, tokenizer_hash: str,
                 checkpoint_hash: str, device: torch.device) -> None:
    if args.output.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("--output must end in .mid or .midi")
    beats = beat_grid(args.beat_file or args.prompt_midi.with_name("beat_midi.txt"))
    prompt, prompt_notes = prompt_score(args.prompt_midi, beats)
    chord_file = None
    if metadata.get("chord_source"):
        chord_file = args.chord_file or args.prompt_midi.with_name("chord_midi.txt")
    prefix = tokenize_prompt(prompt, beats, chord_file, tokenizer, vocab)
    quantization_changed = prompt_quantization_changed(prompt, prefix, tokenizer)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    generated = continue_tokens(model, vocab, prefix, args, device)
    generated_notes = write_midi(prompt, prefix + generated, tokenizer, args.output)
    record = {
        "experiment_id": args.experiment_id or args.output.parent.name,
        "checkpoint_hash": checkpoint_hash,
        "tokenizer_hash": tokenizer_hash,
        "prompt_id": args.prompt_midi.parent.name,
        "prompt_midi": str(args.prompt_midi),
        "output_midi": str(args.output),
        "seed": args.seed, "temperature": args.temperature,
        "top_k": 50 if args.top_k is None and args.top_p is None else args.top_k or 0,
        "top_p": args.top_p, "memory_length": model.cfg.memory_len,
        "prompt_bars": PROMPT_BARS, "generated_bars": NEW_BARS, "total_bars": 32,
        "meter_policy": "fixed 4/4; TimeSig tokens blocked after prompt",
        "prompt_tokens": len(prefix), "generated_tokens": len(generated),
        "prompt_notes": prompt_notes, "generated_notes": generated_notes,
        "prompt_quantization_changed_for_model_context": quantization_changed,
        "prompt_timing": "source notes mapped to annotated 480-tick beat grid",
        "status": "written",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: 8 prompt + 24 generated bars; "
          f"{prompt_notes} prompt notes + {generated_notes} generated notes")


def load_components(checkpoint_path: Path, tokenizer_path: Path, device: torch.device
                    ) -> tuple[REMI, TransformerXL, dict[str, int], dict, str, str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocab: dict[str, int] = checkpoint["vocab"]
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Checkpoint vocabulary IDs are not contiguous")
    if not {"<PAD>", "<BOS>", "<EOS>", "Bar_None"} <= vocab.keys():
        raise ValueError("Checkpoint lacks boundary or bar tokens")
    metadata = checkpoint["tokenizer_config"]
    tokenizer_hash = file_hash(tokenizer_path)
    if tokenizer_hash != metadata["tokenizer_sha256"]:
        raise ValueError("Tokenizer file does not match checkpoint")
    if metadata["track_programs"] != ROLES:
        raise ValueError("Checkpoint uses a different track mapping")
    tokenizer = REMI(params=tokenizer_path)
    if not tokenizer.one_token_stream:
        raise ValueError("Expected a single-stream REMI tokenizer")
    model = TransformerXL(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return tokenizer, model, vocab, metadata, tokenizer_hash, file_hash(checkpoint_path)


def main() -> None:
    args = arguments()
    if args.all_configs:
        if (args.prompt_root is None or args.output_dir is None or args.output is not None
                or args.checkpoint is not None or args.beat_file is not None
                or args.chord_file is not None or args.top_k is not None
                or args.top_p is not None or args.experiment_id is not None
                or args.temperature != 1.0):
            raise ValueError("--all-configs needs --prompt-root and --output-dir; sampling settings are fixed by each configuration")
        configs = (
            ("N20T10", args.no_chord_checkpoint, args.no_chord_tokenizer, 20),
            ("N50T10", args.no_chord_checkpoint, args.no_chord_tokenizer, 50),
            ("C20T10", args.chord_checkpoint, args.tokenizer, 20),
        )
        jobs = [(name, args.prompt_root / f"{i:03d}" / f"{i:03d}.mid",
                 args.output_dir / name / f"{i:03d}.mid", checkpoint, tokenizer, top_k)
                for name, checkpoint, tokenizer, top_k in configs for i in range(1, 4)]
    elif args.prompt_root is not None:
        if (args.checkpoint is None or args.output_dir is None or args.output is not None
                or args.beat_file is not None or args.chord_file is not None):
            raise ValueError("--prompt-root requires --checkpoint and --output-dir without single-prompt options")
        jobs = [(None, args.prompt_root / f"{i:03d}" / f"{i:03d}.mid",
                 args.output_dir / f"{i:03d}.mid", args.checkpoint, args.tokenizer,
                 args.top_k) for i in range(1, 4)]
    else:
        if args.checkpoint is None or args.output is None or args.output_dir is not None:
            raise ValueError("--prompt-midi requires --checkpoint and --output")
        jobs = [(None, args.prompt_midi, args.output, args.checkpoint,
                 args.tokenizer, args.top_k)]
    if args.temperature <= 0 or args.max_new_tokens < 1:
        raise ValueError("temperature and max-new-tokens must be positive")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("top-k must be nonnegative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    if args.prompt_root is not None:
        for _, prompt, output, _, _, _ in jobs:
            if not prompt.is_file():
                raise FileNotFoundError(prompt)
            if output.exists() or output.with_suffix(".json").exists():
                raise FileExistsError(f"Batch output already exists: {output}")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    cache = {}
    for name, prompt, output, checkpoint_path, tokenizer_path, top_k in jobs:
        key = (checkpoint_path, tokenizer_path)
        if key not in cache:
            cache[key] = load_components(checkpoint_path, tokenizer_path, device)
        tokenizer, model, vocab, metadata, tokenizer_hash, checkpoint_hash = cache[key]
        current = copy.copy(args)
        current.prompt_midi, current.output = prompt, output
        current.top_k = top_k
        current.seed = args.seed if args.seed is not None else (42 if args.all_configs else 2000)
        if name is not None:
            current.experiment_id = name
        generate_one(current, tokenizer, model, vocab, metadata,
                     tokenizer_hash, checkpoint_hash, device)


if __name__ == "__main__":
    main()
