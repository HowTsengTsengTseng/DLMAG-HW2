# DLMAG Homework 2: Symbolic Music Generation

This is a flat `uv` project. The current training entry point is `train.py`;
`model.py`, `dataset.py`, and `utils.py` provide its model, data, and shared
utilities. Training uses only the provided POP909 `training/` songs.

## Download Dataset
Under the project root directory, run
```bash
gdown "https://drive.google.com/file/d/1fw0qjeGzzmVkpz6aA8W3dEXygk9rX01t"
unzip POP909.zip
```
## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available, then run:

```bash
uv sync
```

`pyproject.toml` includes `gdown`, PyTorch, MidiTok, MIDI processing, and the
NumPy/pandas/SciPy/tqdm packages imported by the TA's scripts in
`../hw2_sample/HW2_scripts/`.

The optional `si` extra (`uv sync --extra si`) contains `librosa` and
`matplotlib` for MusDr's audio-based SI workflow. The current experiment
specification uses H4 and GS, so the default environment does not need it.

The TA's `eval_basic.py` also imports `MusDr.musdr.side_utils`. The upstream
[MusDr repository](https://github.com/slSeanWU/MusDr) is source code without
Python package metadata, so `uv sync` cannot install it as a normal dependency.
Clone it into `third_party/MusDr` and expose `third_party` on `PYTHONPATH` when
running that evaluation script:

```bash
mkdir -p third_party
git clone https://github.com/slSeanWU/MusDr.git third_party/MusDr
PYTHONPATH=third_party uv run python ../hw2_sample/HW2_scripts/eval_basic.py --help
```

For MIDI-to-WAV synthesis, install the FluidSynth command-line program through
your OS package manager and provide a `.sf2` SoundFont. `midi2wav.py` invokes
the executable directly and places its options before the input files, which
is required by current FluidSynth versions.

The training script expects tokenized sequences, a token-to-ID vocabulary, and
the exact tokenizer configuration JSON. The steps below produce those
preprocessing artifacts; choose a training step budget before running `train.py`.

## Model

`model.py` is a PyTorch adaptation of the Transformer-XL used in
[YatingMusic/remi](https://github.com/YatingMusic/remi/blob/master/modules.py).
It uses sinusoidal relative positions with relative shift, shared content and
position attention biases, post-normalization blocks, tied input/output token
embeddings, and detached segment memory. The default dimensions match the
reference: 12 layers, 8 heads, width 512, inner width 2048, and 512-token
segments and memory. The `train.py` model interface is unchanged. Checkpoints
from the earlier model architecture cannot be resumed with this version.

## Repair POP909 MIDI time signatures

`fix_pop909_timesig.py` reads the two alternative downbeat columns in each
`beat_midi.txt`, estimates the bar pulse count and beat unit, and writes
corrected MIDI copies under the selected output directory. It preserves the
original files and writes `timesig_manifest.csv`; low-confidence songs are
listed as skipped. Choose which downbeat estimate to use explicitly:

```bash
uv run python fix_pop909_timesig.py \
  --dataset-dir POP909/training \
  --output-dir POP909/training_timesig_fixed_col3 \
  --downbeat-column 3
```

POP909 provides two downbeat estimators under different meter assumptions, so
review the manifest and compare candidate output when a song's meter is
uncertain. The beat file alone does not always uniquely specify a notated time
signature.

## Convert corrected MIDI files to REMI tokens

The conversion uses MidiTok's REMI with time-signature, tempo, program, and
POP909 chord tokens. It reads only MIDI files marked `written` in the correction
manifest and requires all 896 training songs. The current column-3 correction
produced 868 files marked 4/4 and 28 marked 6/4.

```bash
uv run python tokenize_miditok.py \
  --midi-dir POP909/training_timesig_fixed_col3 \
  --chord-dir POP909/training \
  --output-dir data/miditok_remi_chords
```

`Chord_<label>` tokens come from each song's `chord_midi.txt`, including `N`
and slash-chord labels such as `Chord_F#:maj7/5`. Annotation start times are
converted through the corrected MIDI tempo map, quantized to MidiTok's tick
grid, and inserted as REMI events. Consecutive identical chord labels are
collapsed into one change token. MidiTok's note-based chord detector is
disabled for encoding, so its inferred chords do not conflict with POP909's
annotations. Use `--no-chords` to reproduce the earlier note-only output.

To compare annotated chords with MidiTok's automatic note-based chord detection
on the same corrected MIDI files, run:

```bash
uv run python tests/compare_chord_tokenization.py --song-ids 001 170
```

The comparison reports event counts, matching labels at the same tick and
within a configurable number of quarter-note beats, plus examples of timing
or label differences. Use `--all --show 0` for a summary of all 896 songs.

The output directory contains `train_sequences.json`, `tok2idx.json`, and
`tokenizer_config.json` for `train.py`, plus `miditok_tokenizer.json` for
loading the same MidiTok tokenizer when decoding generated music. For example,
pass the first three files to `train.py` with its `--sequences`, `--vocab`, and
`--tokenizer-config` options. The training script is not run by preprocessing.

POP909's three tracks are all MIDI program 0. To keep their roles distinct in
one REMI sequence, `tokenize_miditok.py` temporarily assigns program 0 to
MELODY, 1 to BRIDGE, and 2 to PIANO in memory. Corrected MIDI files retain
their original programs. When decoding generated tokens for piano playback,
map all three programs back to program 0; the mapping is saved in
`tokenizer_config.json`.

The correction script writes one inferred signature at tick zero for each
song. It does not encode genuine meter changes or align an opening pickup to
the first annotated downbeat; review ambiguous songs before treating the
inferred signature as ground truth.
