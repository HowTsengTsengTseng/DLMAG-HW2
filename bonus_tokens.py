"""Bar-interleaved melody/PIANO event representation for POP909 bonus task.

Each bar is <BAR> <MELODY> note events <PIANO> note events. A note event is
Pos_i Pitch_p Vel_v Dur_d. Positions and durations use 64 units per bar, so
bars with different beat counts share one vocabulary.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from fix_pop909_timesig import read_beat_annotations


UNITS_PER_BAR = 64
MAX_DURATION = 256
SPECIAL = ("<PAD>", "<BOS>", "<EOS>", "<BAR>", "<MELODY>", "<PIANO>")


def vocabulary() -> dict[str, int]:
    tokens = [*SPECIAL]
    tokens += [f"Pos_{i}" for i in range(UNITS_PER_BAR)]
    tokens += [f"Pitch_{i}" for i in range(128)]
    tokens += [f"Vel_{i}" for i in range(32)]
    tokens += [f"Dur_{i}" for i in range(1, MAX_DURATION + 1)]
    return {token: i for i, token in enumerate(tokens)}


def note_tokens(position: int, pitch: int, velocity: int, duration: int) -> list[str]:
    return [f"Pos_{max(0, min(63, position))}", f"Pitch_{pitch}",
            f"Vel_{min(31, max(0, (velocity - 1) // 4))}",
            f"Dur_{min(MAX_DURATION, max(1, duration))}"]


def training_bars(score) -> list[tuple[list[str], list[str]]]:
    """Take only named MELODY and PIANO tracks from a corrected POP909 MIDI."""
    signatures = list(score.time_signatures)
    if len(signatures) != 1 or signatures[0].time != 0 or signatures[0].denominator != 4:
        raise ValueError("Expected one corrected 4/4 or 6/4 time signature at tick zero")
    beats_per_bar = signatures[0].numerator
    if beats_per_bar not in (4, 6):
        raise ValueError(f"Unsupported training meter: {beats_per_bar}/4")
    ticks_per_bar = score.ticks_per_quarter * beats_per_bar
    tracks = {track.name.strip().upper(): track for track in score.tracks}
    if not {"MELODY", "PIANO"} <= tracks.keys():
        raise ValueError("Expected named MELODY and PIANO tracks")
    if any(track.is_drum for track in (tracks["MELODY"], tracks["PIANO"])):
        raise ValueError("Melody and piano must be pitched tracks")
    last_onset = max((note.time for role in ("MELODY", "PIANO")
                      for note in tracks[role].notes), default=-1)
    if last_onset < 0:
        raise ValueError("Song contains no melody or piano notes")
    bars = [([], []) for _ in range(last_onset // ticks_per_bar + 1)]
    for role, column in (("MELODY", 0), ("PIANO", 1)):
        grouped: list[list[tuple[int, int, int, int]]] = [[] for _ in bars]
        for note in tracks[role].notes:
            bar, offset = divmod(note.time, ticks_per_bar)
            pos = round(offset * UNITS_PER_BAR / ticks_per_bar)
            dur = round(note.duration * UNITS_PER_BAR / ticks_per_bar)
            grouped[bar].append((pos, note.pitch, note.velocity, dur))
        for bar, notes in enumerate(grouped):
            notes.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
            target = bars[bar][column]
            for pos, pitch, velocity, dur in notes:
                target.extend(note_tokens(pos, pitch, velocity, dur))
    return bars


def interleave(bars: list[tuple[list[str], list[str]]]) -> list[str]:
    sequence = ["<BOS>"]
    for melody, piano in bars:
        sequence.extend(("<BAR>", "<MELODY>"))
        sequence.extend(melody)
        sequence.append("<PIANO>")
        sequence.extend(piano)
    sequence.append("<EOS>")
    return sequence


@dataclass(frozen=True)
class BeatGrid:
    times: list[float]
    downbeats: list[int]

    @classmethod
    def load(cls, path: Path) -> "BeatGrid":
        times, flags = read_beat_annotations(path)
        downbeats = [i for i, flag in enumerate(flags) if flag[1] >= 0.5]
        if len(downbeats) != 17 or downbeats[0] != 0 or downbeats[-1] != len(times) - 1:
            raise ValueError(f"Expected 17 downbeats marking 16 complete bars: {path}")
        if abs(times[0]) > 0.01 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"Invalid beat times: {path}")
        return cls(times, downbeats)

    def to_units(self, seconds: float) -> float:
        """Map absolute seconds to a 64-unit-per-bar coordinate."""
        beats = bisect_right(self.times, seconds) - 1
        beats = max(0, min(beats, len(self.times) - 2))
        beat_float = beats + (seconds - self.times[beats]) / (
            self.times[beats + 1] - self.times[beats])
        bar = bisect_right(self.downbeats, beat_float) - 1
        bar = max(0, min(bar, 15))
        return UNITS_PER_BAR * (bar + (beat_float - self.downbeats[bar]) /
                                 (self.downbeats[bar + 1] - self.downbeats[bar]))

    def to_seconds(self, units: float) -> float:
        bar = max(0, min(int(units // UNITS_PER_BAR), 15))
        width = self.downbeats[bar + 1] - self.downbeats[bar]
        beat_float = self.downbeats[bar] + ((units / UNITS_PER_BAR) - bar) * width
        beat = max(0, min(int(beat_float), len(self.times) - 2))
        fraction = beat_float - beat
        return self.times[beat] + fraction * (self.times[beat + 1] - self.times[beat])


def prompt_bars(melody, grid: BeatGrid) -> list[list[str]]:
    if len(melody.instruments) != 1 or melody.instruments[0].name.strip().upper() != "MELODY":
        raise ValueError("Prompt must contain exactly one named MELODY track")
    grouped: list[list[tuple[int, int, int, int]]] = [[] for _ in range(16)]
    for note in melody.instruments[0].notes:
        start = grid.to_units(note.start)
        if not 0 <= start < 16 * UNITS_PER_BAR:
            raise ValueError("Melody note onset lies outside 16 bars")
        bar = min(15, int(start // UNITS_PER_BAR))
        pos = round(start - bar * UNITS_PER_BAR)
        pos = min(63, pos)
        duration = round(grid.to_units(note.end) - start)
        grouped[bar].append((pos, note.pitch, note.velocity, duration))
    output: list[list[str]] = []
    for notes in grouped:
        notes.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
        tokens: list[str] = []
        for pos, pitch, velocity, duration in notes:
            tokens.extend(note_tokens(pos, pitch, velocity, duration))
        output.append(tokens)
    return output
