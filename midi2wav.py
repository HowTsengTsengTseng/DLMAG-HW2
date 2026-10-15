"""Render a MIDI file to WAV with the FluidSynth command-line program.

FluidSynth 2.6 requires command options to appear before SoundFont and MIDI
positional arguments. This uses the CLI directly instead of midi2audio, whose
argument order is incompatible with that version.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a MIDI file to WAV using FluidSynth.")
    parser.add_argument(
        "midi",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "POP909/training/104/104.mid",
        help="Input MIDI file (default: POP909 training song 104)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=PROJECT_DIR / "104.wav",
        help="Output WAV file (default: 104.wav in the project directory)",
    )
    parser.add_argument(
        "--soundfont",
        type=Path,
        default=PROJECT_DIR / "GeneralUser-GS.sf2",
        help="Path to a .sf2 SoundFont (default: GeneralUser-GS.sf2 in the project directory)",
    )
    parser.add_argument("--sample-rate", type=int, default=44100, help="Output sample rate in Hz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    midi_path = args.midi.expanduser().resolve()
    soundfont_path = args.soundfont.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    if not soundfont_path.is_file():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output filename must end in .wav")
    if args.sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer")

    fluidsynth = shutil.which("fluidsynth")
    if fluidsynth is None:
        raise FileNotFoundError(
            "FluidSynth executable not found on PATH. Install FluidSynth with your OS package manager."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        fluidsynth,
        "-ni",
        "-F",
        str(output_path),
        "-r",
        str(args.sample_rate),
        str(soundfont_path),
        str(midi_path),
    ]
    subprocess.run(command, check=True)
    print(f"Rendered {midi_path} -> {output_path}")


if __name__ == "__main__":
    main()
