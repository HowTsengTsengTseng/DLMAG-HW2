"""POP909 token corpus validation and contiguous song batch lanes."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor


def load_corpus(
    sequences_path: Path, vocab_path: Path, tokenizer_path: Path, training_dir: Path,
    expected_songs: int,
) -> tuple[list[tuple[str, list[int]]], dict, dict]:
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    rows = json.loads(sequences_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict) or not isinstance(rows, list) or not isinstance(tokenizer_config, dict):
        raise ValueError("Vocabulary, sequences, and tokenizer config have invalid JSON shapes")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Vocabulary IDs must be unique and contiguous from zero")
    for special in ("<PAD>", "<BOS>", "<EOS>"):
        if special not in vocab:
            raise ValueError(f"Missing special token: {special}")
    if len(rows) != expected_songs:
        raise ValueError(f"Expected {expected_songs} training songs; token file has {len(rows)}")

    folders = {p.name for p in training_dir.iterdir() if p.is_dir()}
    if len(folders) != expected_songs:
        raise ValueError(f"Expected {expected_songs} folders in {training_dir}; found {len(folders)}")
    songs: list[tuple[str, list[int]]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("song_name"), str):
            raise ValueError("Every sequence needs a string song_name")
        name, tokens = row["song_name"], row.get("tokens")
        if name in seen:
            raise ValueError(f"Duplicate song: {name}")
        seen.add(name)
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise ValueError(f"Empty or invalid sequence: {name}")
        if tokens[0] != "<BOS>" or tokens[-1] != "<EOS>":
            raise ValueError(f"Missing BOS/EOS boundaries: {name}")
        unknown = set(tokens) - vocab.keys()
        if unknown:
            raise ValueError(f"Unknown tokens in {name}: {sorted(unknown)[:10]}")
        songs.append((name, [vocab[token] for token in tokens]))
    if seen != folders:
        raise ValueError(
            f"Training/tokenized song mismatch: missing={sorted(folders - seen)[:10]}, "
            f"extra={sorted(seen - folders)[:10]}"
        )
    return songs, vocab, tokenizer_config


class SongLanes:
    """One contiguous song stream per batch lane; reset memory on reassignment."""

    def __init__(self, songs: list[tuple[str, list[int]]], batch: int, segment: int, seed: int):
        self.songs = songs
        self.batch = batch
        self.segment = segment
        self.rng = random.Random(seed)
        self.order = list(range(len(songs)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0
        self.lanes: list[tuple[int, int] | None] = [None] * batch

    def _next_song(self) -> int:
        if self.cursor == len(self.order):
            self.epoch += 1
            self.rng.shuffle(self.order)
            self.cursor = 0
        song_index = self.order[self.cursor]
        self.cursor += 1
        return song_index

    def next_batch(self, pad_id: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[bool]]:
        inputs = torch.full((self.batch, self.segment), pad_id, dtype=torch.long)
        labels = torch.full_like(inputs, -100)
        valid = torch.zeros((self.batch, self.segment), dtype=torch.bool)
        reset = [False] * self.batch
        for lane in range(self.batch):
            if self.lanes[lane] is None:
                self.lanes[lane] = (self._next_song(), 0)
                reset[lane] = True
            song_index, offset = self.lanes[lane]
            ids = self.songs[song_index][1]
            count = min(self.segment, len(ids) - 1 - offset)
            inputs[lane, :count] = torch.tensor(ids[offset : offset + count])
            labels[lane, :count] = torch.tensor(ids[offset + 1 : offset + count + 1])
            valid[lane, :count] = True
            offset += count
            self.lanes[lane] = None if offset >= len(ids) - 1 else (song_index, offset)
        return inputs, labels, valid, valid.sum(), reset

    def state_dict(self) -> dict:
        return {
            "order": self.order, "cursor": self.cursor, "epoch": self.epoch,
            "lanes": self.lanes, "rng_state": self.rng.getstate(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.order = state["order"]
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        self.lanes = state["lanes"]
        self.rng.setstate(state["rng_state"])
