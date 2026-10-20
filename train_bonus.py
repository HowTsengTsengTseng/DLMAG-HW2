"""Train a fresh Transformer-XL on bonus interleaved MELODY/PIANO sequences.

This uses train.py's optimizer, checkpoints, resume logic, and CLI options.
Run prepare_bonus.py first. All sequence tokens receive next-token loss.
"""

from __future__ import annotations

import sys

from train import main


if __name__ == "__main__":
    defaults = {
        "--sequences": "data/bonus_interleaved/train_sequences.json",
        "--vocab": "data/bonus_interleaved/tok2idx.json",
        "--tokenizer-config": "data/bonus_interleaved/tokenizer_config.json",
        "--output-dir": "checkpoints/bonus_interleaved",
    }
    supplied = sys.argv[1:]
    arguments = []
    for option, value in defaults.items():
        if option not in supplied and not any(item.startswith(option + "=") for item in supplied):
            arguments.extend((option, value))
    sys.argv = [sys.argv[0], *arguments, *supplied]
    main()
