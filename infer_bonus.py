"""Generate 16 bars of PIANO from a fixed POP909 bonus MELODY prompt.

The model samples only the PIANO section of each bar. The program inserts the
ground-truth MELODY section before each sampled PIANO section.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pretty_midi
import torch

from bonus_tokens import BeatGrid, UNITS_PER_BAR, prompt_bars, vocabulary
from model import ModelConfig, TransformerXL
from utils import sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt-midi", type=Path,
                        help="POP909/prompts/bonus/NNN/melody.mid")
    inputs.add_argument("--prompt-root", type=Path,
                        help="Generate all ten bonus prompts in this directory")
    parser.add_argument("--beat-file", type=Path,
                        help="Defaults to beat_midi.txt beside melody.mid")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for 001.mid through 010.mid")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--top-k", type=int)
    sampling.add_argument("--top-p", type=float)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-notes-per-bar", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sample(logits: torch.Tensor, allowed: list[int], args: argparse.Namespace) -> int:
    scores = logits[allowed].float() / args.temperature
    if args.top_p is not None:
        order = torch.argsort(scores, descending=True)
        probabilities = torch.softmax(scores[order], dim=0)
        keep = torch.cumsum(probabilities, 0) - probabilities < args.top_p
        scores[order[~keep]] = -torch.inf
    elif args.top_k or args.top_k is None:
        top_k = 50 if args.top_k is None else args.top_k
        cutoff = torch.topk(scores, min(top_k, len(allowed))).values[-1]
        scores[scores < cutoff] = -torch.inf
    probabilities = torch.softmax(scores, dim=0)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Invalid sampling probabilities")
    return allowed[int(torch.multinomial(probabilities, 1).item())]


def generate(model: TransformerXL, vocab: dict[str, int],
             melodies: list[list[str]], args: argparse.Namespace,
             device: torch.device) -> tuple[list[list[tuple[int, int, int, int]]], int]:
    by_id = [""] * len(vocab)
    for token, index in vocab.items():
        by_id[index] = token
    positions = [vocab[f"Pos_{i}"] for i in range(64)]
    pitches = [vocab[f"Pitch_{i}"] for i in range(128)]
    velocities = [vocab[f"Vel_{i}"] for i in range(32)]
    durations = [vocab[f"Dur_{i}"] for i in range(1, 257)]
    memories, memory_valid = model.empty_memory(1, device)

    def feed(tokens: list[int]) -> torch.Tensor:
        nonlocal memories, memory_valid
        for start in range(0, len(tokens), model.cfg.segment_len):
            ids = torch.tensor([tokens[start:start + model.cfg.segment_len]], device=device)
            logits, memories, memory_valid = model(
                ids, torch.ones_like(ids, dtype=torch.bool), memories, memory_valid)
        return logits[0, -1]

    generated: list[list[tuple[int, int, int, int]]] = []
    forced_boundaries = 0
    with torch.inference_mode():
        next_logits = feed([vocab["<BOS>"], vocab["<BAR>"]])
        for bar_index, melody in enumerate(melodies):
            next_logits = feed([vocab["<MELODY>"], *(vocab[token] for token in melody),
                                vocab["<PIANO>"]])
            notes: list[tuple[int, int, int, int]] = []
            last_position = 0
            while True:
                if len(notes) >= args.max_notes_per_bar:
                    token_id = vocab["<BAR>"]
                    forced_boundaries += 1
                else:
                    token_id = sample(next_logits,
                                      [vocab["<BAR>"], *positions[last_position:]], args)
                if token_id == vocab["<BAR>"]:
                    if bar_index != len(melodies) - 1:
                        next_logits = feed([token_id])
                    break
                pos = int(by_id[token_id].split("_", 1)[1])
                next_logits = feed([token_id])
                pitch_id = sample(next_logits, pitches, args)
                next_logits = feed([pitch_id])
                velocity_id = sample(next_logits, velocities, args)
                next_logits = feed([velocity_id])
                duration_id = sample(next_logits, durations, args)
                next_logits = feed([duration_id])
                notes.append((pos, int(by_id[pitch_id].split("_")[1]),
                              4 * int(by_id[velocity_id].split("_")[1]) + 2,
                              int(by_id[duration_id].split("_")[1])))
                last_position = pos
            generated.append(notes)
    return generated, forced_boundaries


def write_output(source: pretty_midi.PrettyMIDI, grid: BeatGrid,
                 bars: list[list[tuple[int, int, int, int]]], output: Path) -> int:
    result = copy.deepcopy(source)
    result.instruments[0].name = "MELODY"
    piano = pretty_midi.Instrument(program=0, name="PIANO")
    final_units = 16 * UNITS_PER_BAR
    for bar_index, notes in enumerate(bars):
        for position, pitch, velocity, duration in notes:
            start_units = bar_index * UNITS_PER_BAR + position
            end_units = min(final_units, start_units + duration)
            start = grid.to_seconds(start_units)
            end = grid.to_seconds(end_units)
            if end > start:
                piano.notes.append(pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start, end=end))
    if not piano.notes:
        raise RuntimeError("Model generated no PIANO notes across 16 bars")
    result.instruments.append(piano)
    signatures = []
    previous = None
    for bar in range(16):
        beats = grid.downbeats[bar + 1] - grid.downbeats[bar]
        if beats != previous:
            signatures.append(pretty_midi.TimeSignature(
                beats, 4, grid.times[grid.downbeats[bar]]))
            previous = beats
    result.time_signature_changes = signatures
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write(str(output))
    return len(piano.notes)


def generate_one(args: argparse.Namespace, model: TransformerXL,
                 vocab: dict[str, int], device: torch.device,
                 checkpoint_hash: str) -> None:
    if args.output.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("--output must end in .mid or .midi")
    beat_file = args.beat_file or args.prompt_midi.with_name("beat_midi.txt")
    grid = BeatGrid.load(beat_file)
    source = pretty_midi.PrettyMIDI(str(args.prompt_midi))
    melodies = prompt_bars(source, grid)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    bars, forced = generate(model, vocab, melodies, args, device)
    note_count = write_output(source, grid, bars, args.output)
    record = {
        "prompt_midi": str(args.prompt_midi), "beat_file": str(beat_file),
        "output_midi": str(args.output), "checkpoint_sha256": checkpoint_hash,
        "seed": args.seed, "top_k": 50 if args.top_k is None and args.top_p is None
        else args.top_k, "top_p": args.top_p,
        "temperature": args.temperature, "bars": len(bars),
        "melody_notes": len(source.instruments[0].notes), "piano_notes": note_count,
        "forced_bar_boundaries": forced,
        "conditioning": "given melody inserted before each generated piano bar",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}: 16 bars, {note_count} generated PIANO notes")


def main() -> None:
    args = arguments()
    if args.prompt_root is not None:
        if args.output_dir is None or args.output is not None or args.beat_file is not None:
            raise ValueError("--prompt-root requires --output-dir and cannot use --output or --beat-file")
        jobs = [(args.prompt_root / f"{i:03d}" / "melody.mid",
                 args.output_dir / f"{i:03d}.mid") for i in range(1, 11)]
        for prompt, output in jobs:
            if not prompt.is_file():
                raise FileNotFoundError(prompt)
            if output.exists() or output.with_suffix(".json").exists():
                raise FileExistsError(f"Batch output already exists: {output}")
    else:
        if args.output is None or args.output_dir is not None:
            raise ValueError("--prompt-midi requires --output and cannot use --output-dir")
        jobs = [(args.prompt_midi, args.output)]
    if args.temperature <= 0 or args.max_notes_per_bar < 1:
        raise ValueError("Temperature and maximum notes per bar must be positive")
    if args.top_k is not None and args.top_k < 0:
        raise ValueError("top-k must be nonnegative")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(device_name)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("tokenizer_config", {}).get("task") != "bonus_accompaniment":
        raise ValueError("Checkpoint was not trained on bonus interleaved sequences")
    vocab = checkpoint["vocab"]
    if vocab != vocabulary():
        raise ValueError("Checkpoint bonus vocabulary differs from this script")
    model = TransformerXL(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    checkpoint_hash = sha256(args.checkpoint)
    for prompt, output in jobs:
        current = copy.copy(args)
        current.prompt_midi, current.output = prompt, output
        generate_one(current, model, vocab, device, checkpoint_hash)


if __name__ == "__main__":
    main()
