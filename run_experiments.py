"""Run the fixed Basic A/B sampling matrix with one checkpoint and logged failures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "K08": ("--top-k", "50", "0.8"),
    "K10": ("--top-k", "50", "1.0"),
    "K12": ("--top-k", "50", "1.2"),
    "P08": ("--top-p", "0.9", "0.8"),
    "P10": ("--top-p", "0.9", "1.0"),
    "P12": ("--top-p", "0.9", "1.2"),
}
HERE = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path(
        "data/miditok_remi_chords/miditok_tokenizer.json"))
    parser.add_argument("--prompts-root", type=Path, default=Path("POP909/prompts/basic-b"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--task", choices=("a", "b", "all"), default="all")
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
    if not args.checkpoint.is_file() or not args.tokenizer.is_file():
        raise FileNotFoundError("Checkpoint and tokenizer must exist")
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    checkpoint_hash = sha256(args.checkpoint)
    tokenizer_hash = sha256(args.tokenizer)
    if not 1 <= args.a_count <= 20 or args.max_new_tokens < 1 or args.retries < 0:
        raise ValueError("a-count must be 1..20, max-new-tokens positive, retries nonnegative")
    prompts = [args.prompts_root / pid / f"{pid}.mid" for pid in ("001", "002", "003")]
    if args.task in {"b", "all"}:
        missing = [path for path in prompts if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Basic B prompts: {missing}")
    jobs = []
    for config in args.configs:
        method, cutoff, temperature = CONFIGS[config]
        if args.task in {"a", "all"}:
            for seed in range(1000, 1000 + args.a_count):
                output = args.output_root / "basic_a" / config / f"{seed}.mid"
                jobs.append(("a", config, str(seed), output, [
                    str(HERE / "infer_a.py"), "--checkpoint", str(args.checkpoint),
                    "--tokenizer", str(args.tokenizer), "--output", str(output),
                    method, cutoff, "--temperature", temperature,
                    "--seed", str(seed), "--experiment-id", config,
                    "--max-new-tokens", str(args.max_new_tokens), "--device", args.device,
                ]))
        if args.task in {"b", "all"}:
            for i, prompt in enumerate(prompts):
                output = args.output_root / "basic_b" / config / f"{prompt.stem}.mid"
                jobs.append(("b", config, prompt.stem, output, [
                    str(HERE / "infer.py"), "--checkpoint", str(args.checkpoint),
                    "--tokenizer", str(args.tokenizer), "--prompt-midi", str(prompt),
                    "--output", str(output), method, cutoff,
                    "--temperature", temperature, "--seed", str(2000 + i),
                    "--experiment-id", config,
                    "--max-new-tokens", str(args.max_new_tokens), "--device", args.device,
                ]))
    if not args.resume:
        existing = [str(output) for _, _, _, output, _ in jobs if output.exists()]
        if existing:
            raise FileExistsError(
                f"Existing output would be overwritten: {existing[:3]}; use --resume"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_path = args.output_root / "generation_manifest.jsonl"
    failures = 0
    with log_path.open("a", encoding="utf-8") as log:
        for task, config, sample, output, command in jobs:
            sidecar = output.with_suffix(".json")
            tokens = output.with_suffix(".tokens.json")
            if args.resume and output.is_file() and sidecar.is_file() and (
                task == "b" or tokens.is_file()
            ):
                metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                method, cutoff, temperature = CONFIGS[config]
                expected_cutoff = float(cutoff) if method == "--top-p" else int(cutoff)
                actual_cutoff = metadata.get("top_p" if method == "--top-p" else "top_k")
                if (metadata.get("checkpoint_hash") != checkpoint_hash or
                    metadata.get("tokenizer_hash") != tokenizer_hash or
                    metadata.get("temperature") != float(temperature) or
                    actual_cutoff != expected_cutoff or
                    metadata.get("experiment_id") != config):
                    raise ValueError(f"Existing sample has different experiment settings: {output}")
                print(f"SKIP {task}/{config}/{sample}", flush=True)
                continue
            print(f"RUN {task}/{config}/{sample}", flush=True)
            base_seed = int(sample) if task == "a" else 2000 + int(sample) - 1
            for attempt in range(args.retries + 1):
                seed = base_seed + 10000 * attempt
                this_command = command.copy()
                this_command[this_command.index("--seed") + 1] = str(seed)
                completed = subprocess.run([sys.executable, *this_command], text=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                record = {
                    "task": task, "config": config, "sample": sample,
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
