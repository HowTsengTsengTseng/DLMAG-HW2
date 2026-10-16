# DLMAG Homework 2: Symbolic Music Generation

This is a flat `uv` project. The current training entry point is `train.py`;
`model.py`, `dataset.py`, and `utils.py` provide its model, data, and shared
utilities. Training uses only the provided POP909 `training/` songs.

## Download Dataset
Under the project root directory, run
```bash
gdown "https://drive.google.com/uc?id=1fw0qjeGzzmVkpz6aA8W3dEXygk9rX01t" --output POP909.zip
unzip -t POP909.zip
unzip POP909.zip
```
## Set up

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available, then run:

```bash
uv sync
```

On a machine without `uv`, install the direct Python dependencies with
`python -m pip install -r requirements.txt`. Use Python 3.11–3.13.

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

## Basic A: unconditional 32-bar generation

Use the same checkpoint and tokenizer for both tasks. Generate one file to
check the workstation setup first:

```bash
uv run python infer_a.py \
  --checkpoint checkpoints/pop909_remi/step_00030000.pt \
  --output results/basic_a/K10/1000.mid \
  --top-k 50 --temperature 1.0 --seed 1000
```

The script starts with `Bar_None`, 4/4, and the tempo token nearest 120 BPM
(currently 121.29 BPM), without a music or chord prompt. It stops at the start
of bar 33 and saves MIDI, metadata JSON, and a token JSON for the TA evaluator.
Generated time-signature tokens are masked for all configurations. This fixed
meter policy also applies to Task B.

Run all six sampling configurations (top-k 50 or top-p 0.9, each with
temperature 0.8, 1.0, and 1.2):

```bash
uv run python run_experiments.py \
  --checkpoint checkpoints/pop909_remi/step_00030000.pt \
  --task all --output-root results
```

This makes 20 Task A MIDI files per configuration (seeds 1000–1019) and three
Task B MIDI files per configuration (prompt seeds 2000–2002). It records failed
jobs in `results/generation_manifest.jsonl` and exits with an error if any
failed. Pass `--resume` to keep completed files and rerun incomplete jobs.
The fixed retry policy is two replacement attempts, using seeds
`original+10000` and `original+20000`; every attempt and its reason is logged.
Use `--retries 0` to disable replacements. Report replacement counts alongside
the results.
For a small check, use `--configs K10 --a-count 1`; use a separate output root
for such a check so it cannot be confused with the formal experiment.

To calculate H4 and GS with the TA's evaluator, clone MusDr as described above,
then run:

```bash
uv run python evaluate_task_a.py \
  --results-root results \
  --output-dir metrics/basic_a
```

`prepare_eval_basic.py` maps `Bar_None` to `Bar` and `Position_*` to `Pos_*`,
retains pitch events, and clips to 32 bars. It applies the same conversion to
the generated token sequences and the first 32 bars of each eligible 4/4
training song. Training songs with fewer than 32 bars or a different meter are
listed in `metrics/basic_a/reference/manifest.json`; the reference is training
data, not a held-out test set. `evaluate_task_a.py` writes per-piece scores to
each configuration's `scores.csv`, and means plus sample standard deviations
to `metrics/basic_a/summary.csv`. Each formal configuration must contain all
20 pieces; missing files and NaN scores cause an error.

## Task B: continue an eight-bar prompt

Use the checkpoint and tokenizer from the same training run. Each Basic B prompt
folder contains its MIDI, beat annotations, and chord annotations. For example:

```bash
uv run python infer.py \
  --checkpoint checkpoints/pop909_remi/step_00030000.pt \
  --prompt-midi POP909/prompts/basic-b/001/001.mid \
  --output results/basic_b/K10/001.mid \
  --top-k 50 --temperature 1.0 --seed 2000
```

The output contains the original eight-bar prompt plus 24 generated bars.
`infer.py` uses `beat_midi.txt` to establish the exact prompt boundary,
feeds all prompt tokens through Transformer-XL memory, and ends at bar 33.
The continuation keeps the prompt's 4/4 meter by masking further
`TimeSig_*` tokens during sampling; use this same rule for every sampling
setting.
It keeps the prompt notes on the annotated beat grid in the output rather than
replacing them with the coarser tokens used for model context. A matching JSON
metadata file is written beside the MIDI. Use `--top-p 0.9` instead of
`--top-k 50` for nucleus sampling.
Repeat for prompt IDs `002` and `003`, using seeds `2001` and `2002`.
If the checkpoint was trained with chord tokens, the prompt's
`chord_midi.txt` is used for context; chord labels are not exported as MIDI
annotations.

After comparing results and listening, select one Basic A configuration and
three Basic B configurations. Render the required 20 + 9 MIDI/WAV pairs with
one SoundFont and sample rate:

```bash
uv run python render_submission.py \
  --results-root results --output-dir submission \
  --soundfont GeneralUser-GS.sf2 \
  --a-config K10 --b-configs K08 K10 P10 \
  --best K10 P10 K08
```

`--best` lists the preferred Basic B version for prompts 001, 002, and 003.
Replace the configuration choices after evaluating the actual generated music.
The renderer requires the `fluidsynth` executable on `PATH`. The output path
for every standalone inference MIDI can also be set directly with `--output`.

The current preprocessed tokenizer includes POP909 chord annotation tokens.
This differs from the earlier local experiment plan's no-chord default, but
does not create an additional inference condition. Keep the same tokenizer and
checkpoint for every configuration and state the use of chord tokens in the
report. Task A receives no ground-truth chord sequence; Task B uses the
provided prompt chord file only as context for the eight prompt bars.

## Model

`model.py` is a PyTorch adaptation of the Transformer-XL used in
[YatingMusic/remi](https://github.com/YatingMusic/remi/blob/master/modules.py).
It uses sinusoidal relative positions with relative shift, shared content and
position attention biases, post-normalization blocks, tied input/output token
embeddings, and detached segment memory. The default dimensions match the
reference: 12 layers, 8 heads, width 512, inner width 2048, and 512-token
segments and memory. The `train.py` model interface is unchanged. Checkpoints
from the earlier model architecture cannot be resumed with this version.

Training writes one JSON record per completed pass through all training songs
to `<output-dir>/history.jsonl`. Each record includes the epoch number, start
and completion steps, token-weighted training loss, processed tokens and
segments, learning rate, and completion time. Because song lanes can have
different lengths, an epoch is recorded only after every song assigned to it
has finished. Checkpoints preserve unfinished epoch totals for exact resume;
checkpoints created before this history format cannot resume with the current
trainer. A fresh run requires a new output directory or `--resume`.

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

Example formal training command after benchmarking (replace the step budget
and batch settings with the values actually chosen for the run):

```bash
uv run python train.py \
  --sequences data/miditok_remi_chords/train_sequences.json \
  --vocab data/miditok_remi_chords/tok2idx.json \
  --tokenizer-config data/miditok_remi_chords/tokenizer_config.json \
  --training-dir POP909/training --expected-songs 896 \
  --output-dir checkpoints/pop909_remi --steps 30000 \
  --batch-size 4 --grad-accum 4
```

The script's default `--steps 2` is a smoke-run default, not the formal
training budget. Inspect the selected checkpoint's `run_config` before
reporting the actual run settings. See `EXPERIMENT_AUDIT.md` for the course
requirement checklist and the remaining workstation checks.

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
