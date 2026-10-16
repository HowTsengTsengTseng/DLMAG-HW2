"""Run the seven Basic A groups: six no-chord settings and one chord baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch


CONFIGS = {
    "N20T05": (False, 20, 0.5),
    "N20T10": (False, 20, 1.0),  # baseline
    "N20T15": (False, 20, 1.5),
    "N50T05": (False, 50, 0.5),
    "N50T10": (False, 50, 1.0),
    "N50T15": (False, 50, 1.5),
    "C20T10": (True, 20, 1.0),
}
MATCHED_TRAINING_KEYS = (
    "expected_songs", "steps", "batch_size", "grad_accum", "segment_len",
    "memory_len", "d_model", "n_head", "d_inner", "n_layer", "dropout",
    "lr", "weight_decay", "warmup_fraction", "clip_grad", "precision", "seed",
)
HERE = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-chord-checkpoint", type=Path)
    parser.add_argument("--no-chord-tokenizer", type=Path, default=Path(
        "data/miditok_remi_no_chords/miditok_tokenizer.json"))
    parser.add_argument("--chord-checkpoint", type=Path)
    parser.add_argument("--chord-tokenizer", type=Path, default=Path(
        "data/miditok_remi_chords/miditok_tokenizer.json"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS),
                        default=list(CONFIGS))
    parser.add_argument("--a-count", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=20000)
    parser.add_argument("--retries", type=int, default=2,
                        help="Replacement attempts after failure; seed += 10000 per attempt")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples with their MIDI and metadata already present")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    selected = dict.fromkeys(args.configs)
    files = {
        False: (args.no_chord_checkpoint, args.no_chord_tokenizer),
        True: (args.chord_checkpoint, args.chord_tokenizer),
    }
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    hashes = {}
    training = {}
    for with_chords in {CONFIGS[config][0] for config in selected}:
        checkpoint_path, tokenizer_path = files[with_chords]
        name = "chord" if with_chords else "no-chord"
        if checkpoint_path is None or not checkpoint_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError(f"{name} checkpoint and tokenizer must exist")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        uses_chords = bool(checkpoint["tokenizer_config"].get("chord_source"))
        if uses_chords != with_chords:
            raise ValueError(f"{name} checkpoint has the wrong chord setting: {checkpoint_path}")
        token_hash = sha256(tokenizer_path)
        if checkpoint["tokenizer_config"]["tokenizer_sha256"] != token_hash:
            raise ValueError(f"{name} tokenizer does not match checkpoint")
        if checkpoint["tokenizer_config"].get("songs") != 896:
            raise ValueError(f"{name} checkpoint was not prepared from 896 songs")
        training[with_chords] = {
            "checkpoint_step": checkpoint["step"],
            "model_config": {key: value for key, value in checkpoint["model_config"].items()
                             if key != "vocab_size"},
            "run_config": {key: checkpoint["run_config"].get(key)
                           for key in MATCHED_TRAINING_KEYS},
        }
        hashes[with_chords] = (sha256(checkpoint_path), token_hash)
        del checkpoint
    if True in hashes and False in hashes and hashes[True][0] == hashes[False][0]:
        raise ValueError("Chord and no-chord conditions need separately trained checkpoints")
    if True in training and False in training and training[True] != training[False]:
        raise ValueError("Chord and no-chord checkpoints have different training settings or steps")
    if not 1 <= args.a_count <= 20 or args.max_new_tokens < 1 or args.retries < 0:
        raise ValueError("a-count must be 1..20, max-new-tokens positive, retries nonnegative")
    jobs = []
    for config in selected:
        with_chords, top_k, temperature = CONFIGS[config]
        checkpoint_path, tokenizer_path = files[with_chords]
        for seed in range(1000, 1000 + args.a_count):
            output = args.output_root / "basic_a" / config / f"{seed}.mid"
            jobs.append((config, seed, output, [
                str(HERE / "infer_a.py"), "--checkpoint", str(checkpoint_path),
                "--tokenizer", str(tokenizer_path), "--output", str(output),
                "--top-k", str(top_k), "--temperature", str(temperature),
                "--seed", str(seed), "--experiment-id", config,
                "--max-new-tokens", str(args.max_new_tokens), "--device", args.device,
            ]))
    if not args.resume:
        existing = [str(output) for _, _, output, _ in jobs if output.exists()]
        if existing:
            raise FileExistsError(
                f"Existing output would be overwritten: {existing[:3]}; use --resume"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "generation_manifest.jsonl"
    failures = 0
    with log_path.open("a", encoding="utf-8") as log:
        for config, seed_number, output, command in jobs:
            sidecar = output.with_suffix(".json")
            tokens = output.with_suffix(".tokens.json")
            if args.resume and output.is_file() and sidecar.is_file() and tokens.is_file():
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                with_chords, top_k, temperature = CONFIGS[config]
                checkpoint_hash, tokenizer_hash = hashes[with_chords]
                if (metadata.get("checkpoint_hash") != checkpoint_hash or
                    metadata.get("tokenizer_hash") != tokenizer_hash or
                    metadata.get("temperature") != temperature or
                    metadata.get("top_k") != top_k or metadata.get("top_p") is not None or
                    metadata.get("experiment_id") != config):
                    raise ValueError(f"Existing sample has different experiment settings: {output}")
                print(f"SKIP A/{config}/{seed_number}", flush=True)
                continue
            print(f"RUN A/{config}/{seed_number}", flush=True)
            for attempt in range(args.retries + 1):
                seed = seed_number + 10000 * attempt
                this_command = command.copy()
                this_command[this_command.index("--seed") + 1] = str(seed)
                completed = subprocess.run([sys.executable, *this_command], text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                record = {
                    "task": "a", "config": config, "sample": str(seed_number),
                    "with_chords": CONFIGS[config][0],
                    "checkpoint_hash": hashes[CONFIGS[config][0]][0],
                    "seed": seed, "attempt": attempt + 1,
                    "output": str(output),
                    "status": "written" if completed.returncode == 0 else "failed",
                    "returncode": completed.returncode, "message": completed.stdout[-4000:],
                }
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(completed.stdout.strip(), flush=True)
                if completed.returncode == 0:
                    break
            else:
                failures += 1
    if failures:
        raise RuntimeError(f"{failures} generation jobs failed; see {log_path}")
    print(f"Completed {len(jobs)} jobs; manifest: {log_path}")


if __name__ == "__main__":
    main()
