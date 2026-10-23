# DLMAG Homework 2: Symbolic Music Generation

## Grader TL;DR: inference and listening samples
Run this to download trained checkpoints and listening samples:
```bash
gdown "https://drive.google.com/uc?id=1_qqursH0RnIWee0JSlEInLeT9zAQb7T9"
unzip dlmag_hw2.zip
```

Run these commands from the repository root with Python 3.11–3.13. The supplied
checkpoints are trained from scratch on the 896 POP909 training songs; inference
does **not** require rerunning preprocessing or training. Install dependencies
with `uv sync` (or `python -m pip install -r requirements.txt`). The three tasks
use different input prompts and output lengths:

| Task | Input | Output | Inference script |
| --- | --- | --- | --- |
| Basic A | No music prompt | 20 selected 32-bar MIDIs | `run_experiments.py` (calls `infer_a.py`) |
| Basic B | One provided 8-bar Basic B MIDI prompt | Original 8 bars + 24 generated bars | `infer.py` |
| Bonus | One provided 16-bar MELODY prompt | MELODY + generated PIANO, 16 bars | `infer_bonus.py` |

To generate the submitted Basic A setting (`C20T10`), all nine Basic B files,
and all ten Bonus files, run these commands from the repository root:

```bash
uv run python run_experiments.py \
  --config C20T10 \
  --chord-checkpoint checkpoints/chords/step_00030000.pt \
  --output-root results

uv run python infer.py \
  --all-configs --prompt-root POP909/prompts/basic-b \
  --output-dir results/basic_b

uv run python infer_bonus.py \
  --checkpoint checkpoints/bonus_interleaved/step_00030000.pt \
  --prompt-root POP909/prompts/bonus --output-dir results/bonus \
  --top-k 50 --temperature 1.0 --seed 2001
```

Basic A writes 20 files under `results/basic_a/C20T10/`; Task B writes three
files per configuration under `results/basic_b/`; Bonus writes ten files under
`results/bonus/`. Each MIDI has matching metadata JSON. Basic A also writes
token JSON. The Basic B command uses the default chord and no-chord checkpoints
and their matching tokenizers; override their paths with
`--chord-checkpoint`, `--no-chord-checkpoint`, `--tokenizer`, and
`--no-chord-tokenizer` if needed. The Bonus model has its own checkpoint and
representation.

To render **all generated MIDI files** from the three tasks into WAVs, run:

```bash
uv run python render_submission.py \
  --basic-a-dir results/basic_a \
  --basic-b-dir results/basic_b \
  --bonus-dir results/bonus \
  --output-dir rendered_all \
  --soundfont GeneralUser-GS.sf2
```

This recursively copies the MIDI and JSON files and writes a WAV beside each
MIDI under `rendered_all/`. FluidSynth must be installed on `PATH`, and
`GeneralUser-GS.sf2` must point to a SoundFont file.

See the sections below for the full 20-piece Basic A matrix, all three Basic B
prompts, ten Bonus prompts, evaluation, and batch rendering. The model and
tokenizer paths in each command must be kept as a matching pair.

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

## Repair POP909 MIDI time signatures

`fix_pop909_timesig.py` reads the two alternative downbeat columns in each
`beat_midi.txt`, estimates the bar pulse count and beat unit, and writes
corrected MIDI copies under the selected output directory. It preserves the
original files and writes `timesig_manifest.csv`; low-confidence songs are
listed as skipped. Choose which downbeat estimate to use explicitly:

Usage: `uv run python fix_pop909_timesig.py --downbeat-column {2,3}
[--dataset-dir DIR] [--output-dir DIR] [--min-confidence FLOAT]`.
`--downbeat-column` is required: `2` uses `downbeat_1`, and `3` uses
`downbeat_2`. The input defaults to `POP909/training`, the output defaults to
`POP909/training_timesig_fixed`, and the confidence threshold defaults to
`0.8`. The corrected MIDI files and `timesig_manifest.csv` are written under
the output directory.

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

The conversion uses MidiTok's REMI with time-signature, tempo, and program
tokens. Make two separate tokenizers: one without chords and one with POP909
chord annotations. Both read only MIDI files marked `written` in the correction
manifest and require all 896 training songs. The current column-3 correction
produced 868 files marked 4/4 and 28 marked 6/4.

Usage: `uv run python tokenize_miditok.py [--midi-dir DIR]
[--chord-dir DIR] [--output-dir DIR] [--no-chords]
[--expected-songs N]`. `--midi-dir` must contain the corrected MIDI files and
`timesig_manifest.csv`. The tokenizer requires exactly 896 songs by default
and stops with an error if any required MIDI or chord annotation is missing.
Pass `--no-chords` for the no-chord model; omit it to insert annotations from
`--chord-dir` (default `POP909/training`). Always set a different `--output-dir`
for each variant so their vocabulary and tokenizer files cannot overwrite one
another.

```bash
uv run python tokenize_miditok.py \
  --midi-dir POP909/training_timesig_fixed_col3 \
  --output-dir data/miditok_remi_no_chords --no-chords

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
annotations. Keep the two output directories separate.

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

## Train the two models

Train the no-chord and chord models separately after benchmarking. Use the same
architecture, step budget, batch, accumulation, and optimizer settings when
comparing them; replace these example values with the formal choices:

Usage: `uv run python train.py [--sequences FILE] [--vocab FILE]
[--tokenizer-config FILE] [--training-dir DIR] [--expected-songs N]
[--output-dir DIR] [--steps N] [--batch-size N] [--grad-accum N]
[--precision {bf16,fp32}] [--resume CHECKPOINT]`.
The three input files must come from the **same** preprocessing output
directory. The input-file defaults point to `data/miditok_remi_chords`, so
specify all three paths for the no-chord run. `--training-dir` defaults to
`POP909/training`, `--expected-songs` to `896`, and `--output-dir` to
`checkpoints/pop909_remi`. The script checks that the tokenized songs match this
training split, then writes `run_config.json`, `history.jsonl`, and step
checkpoints such as `step_00010000.pt` under `--output-dir`. At startup it prints
the input paths, hashes, vocabulary size, and `chord_source`; it rejects a
mix of chord and no-chord sequences, vocabulary, and tokenizer metadata.

| Option | Meaning | Default |
| --- | --- | ---: |
| `--steps` | Total optimizer updates, including updates already completed when resuming | 2 |
| `--batch-size` | Song lanes per microbatch | 1 |
| `--grad-accum` | Microbatches per optimizer update; effective lanes per update are `batch-size × grad-accum` | 1 |
| `--segment-len`, `--memory-len` | Transformer-XL segment and memory lengths in tokens | 512 each |
| `--d-model`, `--n-head`, `--d-inner`, `--n-layer` | Model width, attention heads, feed-forward width, and layers | 512, 8, 2048, 12 |
| `--lr`, `--weight-decay`, `--warmup-fraction` | AdamW peak learning rate, decay, and warmup share before cosine decay | `3e-4`, `0.01`, `0.05` |
| `--precision` | `bf16` needs a compatible CUDA GPU; use `fp32` otherwise | `bf16` |
| `--save-every`, `--log-every` | Checkpoint and console-log intervals in optimizer steps | 1000, 50 |

Other controls are `--dropout 0.1`, `--clip-grad 1.0`, and `--seed 42` by
default. Run `uv run python train.py --help` to see every option.

```bash
uv run python train.py \
  --sequences data/miditok_remi_no_chords/train_sequences.json \
  --vocab data/miditok_remi_no_chords/tok2idx.json \
  --tokenizer-config data/miditok_remi_no_chords/tokenizer_config.json \
  --training-dir POP909/training --expected-songs 896 \
  --output-dir checkpoints/no_chords --steps 30000 \
  --batch-size 4 --grad-accum 4

uv run python train.py \
  --sequences data/miditok_remi_chords/train_sequences.json \
  --vocab data/miditok_remi_chords/tok2idx.json \
  --tokenizer-config data/miditok_remi_chords/tokenizer_config.json \
  --training-dir POP909/training --expected-songs 896 \
  --output-dir checkpoints/chords --steps 30000 \
  --batch-size 4 --grad-accum 4
```

To resume an interrupted run, repeat the **same training arguments and total
step budget** and add `--resume` pointing to a checkpoint in that output
directory. For example, if the no-chord 30,000-step run stopped after step
10,000:

```bash
uv run python train.py \
  --sequences data/miditok_remi_no_chords/train_sequences.json \
  --vocab data/miditok_remi_no_chords/tok2idx.json \
  --tokenizer-config data/miditok_remi_no_chords/tokenizer_config.json \
  --training-dir POP909/training --expected-songs 896 \
  --output-dir checkpoints/no_chords --steps 30000 \
  --batch-size 4 --grad-accum 4 \
  --resume checkpoints/no_chords/step_00010000.pt
```

Changing `--steps`, batch size, model dimensions, or input files while
resuming is rejected; a longer training budget needs a new run plan.

### Model and checkpoint details

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

## Basic A: unconditional 32-bar generation

The seven groups are six no-chord sampling settings plus one chord baseline.
`N20T10` (top-k 20, temperature 1.0, no chord tokens) is the reference setting.
Use one no-chord checkpoint for all six `N*` groups and a separately trained
chord checkpoint for `C20T10`:

| ID | Chord tokens | Top-k | Temperature |
| --- | --- | ---: | ---: |
| N20T05 | No | 20 | 0.5 |
| N20T10 (base) | No | 20 | 1.0 |
| N20T15 | No | 20 | 1.5 |
| N50T05 | No | 50 | 0.5 |
| N50T10 | No | 50 | 1.0 |
| N50T15 | No | 50 | 1.5 |
| C20T10 | Yes | 20 | 1.0 |

The chord group uses a different training representation and separately
trained checkpoint. Match the two models' architecture, training steps, and
other training settings where possible. Task A receives no ground-truth chord
sequence; the chord model samples its chord events. State clearly in the
report that the chord comparison is across two training runs.

Generate one baseline file to check the workstation setup first:

```bash
uv run python infer_a.py \
  --checkpoint checkpoints/no_chords/step_00030000.pt \
  --tokenizer data/miditok_remi_no_chords/miditok_tokenizer.json \
  --output results/basic_a/N20T10/1000.mid \
  --top-k 20 --temperature 1.0 --seed 1000
```

The script starts with `Bar_None`, 4/4, and the tempo token nearest 120 BPM
(currently 121.29 BPM), without a music or chord prompt. It stops at the start
of bar 33 and saves MIDI, metadata JSON, and a token JSON for the TA evaluator.
Generated time-signature tokens are masked for all configurations. This fixed
meter policy also applies to Task B.

Run the full seven-group matrix for Task A:

```bash
uv run python run_experiments.py \
  --no-chord-checkpoint checkpoints/no_chords/step_00030000.pt \
  --chord-checkpoint checkpoints/chords/step_00030000.pt \
  --output-root results
```

This makes 20 Task A MIDI files per configuration (seeds 1000–1019), for 140
generated pieces and seven experiment rows. Task B is run separately. To
reproduce only the submitted `C20T10` Basic A set, run:

```bash
uv run python run_experiments.py \
  --config C20T10 \
  --chord-checkpoint checkpoints/chords/step_00030000.pt \
  --output-root reproduced
```

This writes `reproduced/basic_a/C20T10/1000.mid` through `1019.mid` and their
JSON files. `--configs C20T10` remains available for existing commands.
The script records attempts in `<output-root>/generation_manifest.jsonl` and
exits with an error if any failed. Pass `--resume` to keep completed files and
rerun incomplete jobs.
The fixed retry policy is two replacement attempts, using seeds
`original+10000` and `original+20000`; every attempt and its reason is logged.
Use `--retries 0` to disable replacements. Report replacement counts alongside
the results.
For a small check, use `--configs N20T10 --a-count 1`; use a separate output root
for such a check so it cannot be confused with the formal experiment.

To calculate H4 and GS with the TA's evaluator, clone MusDr as described above,
then run:

```bash
uv run python evaluate_task_a.py \
  --results-root results \
  --output-dir metrics/basic_a
```

`prepare_eval_basic.py` maps `Bar_None` to `Bar`, note positions to `Pos_*`,
retains pitch events, and clips to 32 bars. It applies the same conversion to
the generated token sequences and the first 32 bars of each eligible 4/4
training song. Training songs with fewer than 32 bars or a different meter are
listed in `metrics/basic_a/reference/manifest.json`; the reference is training
data, not a held-out test set. `evaluate_task_a.py` writes per-piece scores to
each configuration's `scores.csv`, and means plus sample standard deviations
to `metrics/basic_a/summary.csv` (seven experiment rows). The training reference
is in `reference_summary.csv`. Each formal configuration must contain all
20 pieces; missing files and NaN scores cause an error. Events before the first
bar and sampled `Pitch_*` events without a preceding position in their bar are
ignored and counted in each configuration's `manifest.json`; no onset is
invented for malformed events.

## Task B: continue an eight-bar prompt

Task B is separate from the seven Task A experiment rows. Use `infer.py` with
the checkpoint and tokenizer belonging to the desired model. Each Basic B prompt
folder contains its MIDI, beat annotations, and chord annotations. For example:

```bash
uv run python infer.py \
  --checkpoint checkpoints/no_chords/step_00030000.pt \
  --tokenizer data/miditok_remi_no_chords/miditok_tokenizer.json \
  --prompt-midi POP909/prompts/basic-b/001/001.mid \
  --output results/basic_b/N20T10/001.mid \
  --top-k 20 --temperature 1.0 --seed 42
```

The output contains the original eight-bar prompt plus 24 generated bars.
`infer.py` uses `beat_midi.txt` to establish the exact prompt boundary,
feeds all prompt tokens through Transformer-XL memory, and ends at bar 33.
The continuation keeps the prompt's 4/4 meter by masking further
`TimeSig_*` tokens during sampling; use this same rule for every sampling
setting.
It keeps the prompt notes on the annotated beat grid in the output rather than
replacing them with the coarser tokens used for model context. A matching JSON
metadata file is written beside the MIDI.
Repeat for prompt IDs `002` and `003`, choosing an output path and seed for
each run.

To generate all three Basic B prompts with the submitted `C20T10` setting in
one command (seed 42 is reset for each prompt), run:

```bash
uv run python infer.py --checkpoint checkpoints/chords/step_00030000.pt --prompt-root POP909/prompts/basic-b --output-dir results/basic_b/C20T10 --experiment-id C20T10 --top-k 20 --temperature 1.0 --seed 42
```

This writes `001.mid` through `003.mid` with matching JSON files. For a
no-chord checkpoint, also pass its matching `--tokenizer`. Batch mode refuses
existing output files; use a new output directory for another run.

To generate the complete nine-file Task B submission (three settings across
all three prompts) in one command, run:

```bash
uv run python infer.py --all-configs --prompt-root POP909/prompts/basic-b --output-dir results/basic_b
```

This uses the default no-chord and chord checkpoints and their matching
tokenizers, with top-k 20, 50, and 20 respectively, temperature 1.0, and seed
42 reset for every file. Outputs go to `results/basic_b/<config>/001.mid`
through `003.mid`, each with matching metadata. Override checkpoint paths with
`--no-chord-checkpoint` and `--chord-checkpoint` if needed.

If the checkpoint was trained with chord tokens, the prompt's
`chord_midi.txt` is used for context; chord labels are not exported as MIDI
annotations.

## Render MIDI/WAV submission files

`render_submission.py` searches each input directory recursively, copies every MIDI and
its matching JSON, and renders a WAV beside the copied MIDI. For example,
`results/basic_b/C20T10/001.mid` becomes
`rendered_all/basic_b/C20T10/001.wav`. The default sample rate is 44,100 Hz;
add `--sample-rate` to change it. Use a new output directory if you run the
renderer again, because it refuses to overwrite existing files.

Collect the chosen Basic A, Basic B, and Bonus outputs into three input
directories. Each MIDI must have a matching metadata JSON with the same stem
(`1000.mid` with `1000.json`, for example). Additional JSON files such as
`1000.tokens.json` are copied too. The script renders **every** MIDI found in
each directory, including nested subdirectories, and preserves those relative
paths in `submission/basic_a/`, `submission/basic_b/`, and `submission/bonus/`.
It does not select configurations or preferred Basic B files; organize those
choices in the input directories before rendering.

```bash
uv run python render_submission.py \
  --basic-a-dir results/selected_basic_a \
  --basic-b-dir results/selected_basic_b \
  --bonus-dir results/bonus \
  --output-dir submission \
  --soundfont GeneralUser-GS.sf2
```

The renderer copies their MIDI/JSON
files and writes one WAV beside each MIDI using the same SoundFont and sample
rate (default 44,100 Hz). It refuses missing metadata or existing output files.
FluidSynth must be on `PATH`. Individual inference scripts also accept an
explicit `--output` path.

## Bonus task: bar-interleaved accompaniment

The bonus model uses a separate representation and checkpoint. Each training
song becomes one autoregressive sequence:

```text
<BOS> <BAR> <MELODY> melody note events <PIANO> piano note events
      <BAR> <MELODY> melody note events <PIANO> piano note events ... <EOS>
```

A note event is `Pos_i Pitch_p Vel_v Dur_d`. Positions and durations use 64
units per bar, so 4-beat, 6-beat, and irregular prompt bars share one vocabulary.
Velocity uses 32 bins; durations longer than four bars are clipped. The model
receives both tracks during training and uses next-token loss on **all** tokens.
During inference, `infer_bonus.py` inserts the given melody for each bar and
samples only the piano section. It constrains the note-token grammar and forces
a bar boundary after at most 96 generated notes per bar. Future melody bars are
not visible until they are inserted.

Preparation reads only the 896 time-signature-corrected training MIDI files.
It selects their named `MELODY` and `PIANO` tracks and discards `BRIDGE`.
The bonus prompts' `piano.mid` files are used only for evaluation, never for
training or generation.

### Prepare and train the bonus model

First create `POP909/training_timesig_fixed_col3/` and its manifest with the
time-signature repair workflow above. Then run:

```bash
uv run python prepare_bonus.py
uv run python train_bonus.py --steps 30000 --batch-size 4 --grad-accum 4 \
  --output-dir checkpoints/bonus_interleaved
```

### Generate and evaluate bonus prompts

Generate one of the ten prompts with:

```bash
uv run python infer_bonus.py \
  --checkpoint checkpoints/bonus_interleaved/step_00030000.pt \
  --prompt-midi POP909/prompts/bonus/001/melody.mid \
  --output results/bonus/001.mid --top-k 50 --temperature 1.0 --seed 2001
```

The script reads `beat_midi.txt` beside `melody.mid` by default, writes the
original MELODY and generated PIANO tracks, and saves sampling metadata in
`001.json`. It does not read the prompt's ground-truth `piano.mid`.
To generate all ten prompts in one command (seed 2001 is reset for each
prompt), run:

```bash
uv run python infer_bonus.py --checkpoint checkpoints/bonus_interleaved/step_00030000.pt --prompt-root POP909/prompts/bonus --output-dir results/bonus --top-k 50 --temperature 1.0 --seed 2001
```

This writes `001.mid` through `010.mid` with matching JSON files. Batch mode
refuses existing output files; use a new output directory for another run.

Render a WAV and evaluate the complete ten-file set with:

```bash
uv run python midi2wav.py results/bonus/001.mid results/bonus/001.wav
uv run python ../hw2_sample/HW2_scripts/eval_bonus.py \
  --prompt_dir POP909/prompts/bonus --gen_midi_dir results/bonus \
  --output_csv results/bonus_metrics.csv
```
