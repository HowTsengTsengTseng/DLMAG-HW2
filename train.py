"""Train a REMI Transformer-XL on the complete POP909 training split.

Input is a tokenized JSON list of {"song_name": str, "tokens": [str, ...]},
plus a token-to-ID JSON object and the exact tokenizer configuration JSON.
No MIDI preprocessing is hidden in this script. Run the tokenizer's separate
round-trip and bar-count checks before starting a formal training run.

Example (after preprocessing and a throughput benchmark):
  python train.py --sequences data/train_sequences.json \
      --vocab data/tok2idx.json --tokenizer-config data/tokenizer_config.json \
      --training-dir POP909/training --steps 100000 --batch-size 4 \
      --grad-accum 4 --output-dir checkpoints/formal
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.nn import functional as F

from dataset import SongLanes, load_corpus
from model import ModelConfig, TransformerXL
from utils import sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("POP909/training"))
    parser.add_argument("--expected-songs", type=int, default=896)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/pop909_remi"))
    parser.add_argument("--steps", type=int, default=2, help="Optimizer steps, fixed after benchmarking")
    parser.add_argument("--batch-size", type=int, default=1, help="Microbatch lanes; benchmark first")
    parser.add_argument("--grad-accum", type=int, default=1, help="Microbatches per optimizer step")
    parser.add_argument("--segment-len", type=int, default=512)
    parser.add_argument("--memory-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-inner", type=int, default=2048)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sequences", type=Path, default=Path("data/miditok_remi_chords/train_sequences.json"))
    parser.add_argument("--vocab", type=Path, default=Path("data/miditok_remi_chords/tok2idx.json"))
    parser.add_argument("--tokenizer-config", type=Path, default=Path("data/miditok_remi_chords/tokenizer_config.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.steps, args.batch_size, args.grad_accum, args.save_every, args.log_every) < 1:
        raise ValueError("Steps, batch size, accumulation, and intervals must be positive")
    if not 0 <= args.warmup_fraction < 1:
        raise ValueError("warmup-fraction must be in [0, 1)")
    if args.precision == "bf16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 requires a compatible CUDA GPU; use --precision fp32 otherwise")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    songs, vocab, tokenizer_config = load_corpus(
        args.sequences, args.vocab, args.tokenizer_config, args.training_dir, args.expected_songs
    )
    cfg = ModelConfig(
        vocab_size=len(vocab), d_model=args.d_model, n_head=args.n_head,
        d_inner=args.d_inner, n_layer=args.n_layer, segment_len=args.segment_len,
        memory_len=args.memory_len, dropout=args.dropout,
    )
    model = TransformerXL(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, math.ceil(args.steps * args.warmup_fraction))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps + 1) / max(1, args.steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    lanes = SongLanes(songs, args.batch_size, args.segment_len, args.seed)
    memories, memory_valid = model.empty_memory(args.batch_size, device)
    hashes = {key: sha256(path) for key, path in (
        ("sequences", args.sequences), ("vocab", args.vocab),
        ("tokenizer_config", args.tokenizer_config)
    )}
    run_config = vars(args).copy()
    run_config.pop("resume")
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
    start_step = 0
    history_records: list[dict] = []
    epoch_stats: dict[int, dict[str, float | int]] = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["input_hashes"] != hashes or checkpoint["model_config"] != asdict(cfg):
            raise ValueError("Resume checkpoint uses different data, vocabulary, tokenizer, or model")
        if checkpoint["run_config"] != run_config:
            raise ValueError("Resume requires the same training configuration")
        if "history_records" not in checkpoint or "epoch_stats" not in checkpoint:
            raise ValueError("Resume checkpoint predates per-epoch history tracking")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        lanes.load_state_dict(checkpoint["lane_state"])
        memories = [memory.to(device) for memory in checkpoint["memories"]]
        memory_valid = checkpoint["memory_valid"].to(device)
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_step = checkpoint["step"]
        history_records = checkpoint["history_records"]
        epoch_stats = checkpoint["epoch_stats"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    if args.resume:
        # Restore the history to the selected checkpoint, including when the
        previous process wrote later epochs before it stopped.
        checkpoint_history = "".join(json.dumps(row) + "\n" for row in history_records)
        if history_path.exists() and history_path.read_text(encoding="utf-8") != checkpoint_history:
            backup = args.output_dir / f"history.before_resume_{uuid4().hex[:8]}.jsonl"
            shutil.copy2(history_path, backup)
            print(f"Saved previous epoch history to {backup}", flush=True)
        temporary_history = history_path.with_suffix(".tmp")
        temporary_history.write_text(checkpoint_history, encoding="utf-8")
        temporary_history.replace(history_path)
    else:
        with history_path.open("x", encoding="utf-8"):
            pass
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"training": run_config, "model": asdict(cfg), "input_hashes": hashes,
                    "tokenizer_config": tokenizer_config, "torch_version": torch.__version__}, indent=2),
        encoding="utf-8",
    )
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        weighted_loss = 0.0
        token_total = 0
        finished_in_step: set[int] = set()
        for _ in range(args.grad_accum):
            ids, labels, valid, token_count, reset, batch_epochs, finished_epochs = (
                lanes.next_batch(vocab["<PAD>"])
            )
            if any(reset):
                memory_valid[reset] = False
                for memory in memories:
                    memory[reset] = 0
            ids, labels, valid = ids.to(device), labels.to(device), valid.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                logits, memories, memory_valid = model(ids, valid, memories, memory_valid)
                # Labels are shifted only here; the model never shifts internally.
                token_losses = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), labels.reshape(-1),
                    ignore_index=-100, reduction="none",
                ).reshape_as(labels)
                loss = token_losses.sum() / int(token_count)
            (loss / args.grad_accum).backward()
            lane_loss_sums = token_losses.detach().sum(dim=1).tolist()
            lane_counts = valid.sum(dim=1).tolist()
            for lane_epoch, lane_loss, lane_count in zip(
                batch_epochs, lane_loss_sums, lane_counts, strict=True
            ):
                stats = epoch_stats.setdefault(lane_epoch, {
                    "loss_sum": 0.0, "tokens": 0, "segments": 0, "first_step": step,
                })
                stats["loss_sum"] += lane_loss
                stats["tokens"] += lane_count
                stats["segments"] += 1
            finished_in_step.update(finished_epochs)
            weighted_loss += sum(lane_loss_sums)
            token_total += int(token_count)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()
        if step % args.log_every == 0 or step == 1:
            print(json.dumps({"step": step, "train_loss": weighted_loss / token_total,
                              "tokens": token_total, "assignment_epoch": lanes.epoch + 1,
                              "completed_epochs": lanes.completed_epochs,
                              "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm)}), flush=True)
        for epoch in sorted(finished_in_step):
            stats = epoch_stats.pop(epoch)
            record = {
                "epoch": epoch + 1,
                "first_step": stats["first_step"],
                "completed_at_step": step,
                "train_loss": stats["loss_sum"] / stats["tokens"],
                "tokens": stats["tokens"],
                "segments": stats["segments"],
                "lr": scheduler.get_last_lr()[0],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            history_records.append(record)
            with history_path.open("a", encoding="utf-8") as history_file:
                history_file.write(json.dumps(record) + "\n")
            print(json.dumps({"epoch_history": record}), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            checkpoint = {
                "step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "model_config": asdict(cfg),
                "run_config": run_config, "tokenizer_config": tokenizer_config,
                "vocab": vocab, "input_hashes": hashes,
                "lane_state": lanes.state_dict(), "memories": [m.cpu() for m in memories],
                "memory_valid": memory_valid.cpu(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "history_records": history_records, "epoch_stats": epoch_stats,
            }
            output = args.output_dir / f"step_{step:08d}.pt"
            temporary = output.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
            print(json.dumps({"checkpoint": str(output), "sha256": sha256(output)}), flush=True)


if __name__ == "__main__":
    main()
