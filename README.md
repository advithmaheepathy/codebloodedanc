# Single-microphone AI noise suppression for defence communications

SIH problem statement **26052**. Mission-critical voice comms are corrupted by
stationary noise (engines, rotor wash, wind), non-stationary noise (sirens, pass-bys)
and impulsive noise (gunshots, bursts). This system suppresses that noise from a
**single microphone** and delivers speech at a consistent level, with every claim
backed by a measurement.

```
microphone ──► preprocess ──► DeepFilterNet3 ──► volume normalisation ──► output
               DC + 80 Hz     pretrained,        speech-aware AGC
               high-pass      unmodified         + look-ahead limiter
```

Audio is carried at 48 kHz because the model is a 48 kHz model. The supplied corpus is
16 kHz and is upsampled on load.

---

## What is real and what is not

This section is first on purpose. Everything below it is measured; nothing here is
implied.

| Claim | Status |
| --- | --- |
| Noise suppression stage | **Pretrained DeepFilterNet3, unmodified.** No training or fine-tuning was performed. |
| Volume normalisation | Built here. Speech-aware AGC with a look-ahead limiter. |
| Evaluation | 216 examples from the supplied corpus, category × SNR balanced, 9 methods. |
| Training pipeline | **Designed and documented, not executed.** See [Training, not executed](#training-not-executed). |
| Jetson deployment | Install script and config overlay provided. Timings in this repo were measured on the development host unless a report says otherwise. |
| Fullband capability | **Not exercised.** The corpus is 16 kHz, so the 8–24 kHz band is empty after upsampling. |

The mandated targets are **not all met**. The measured results and the reasons are in
[Results](#results). They are reported as they are rather than framed to look better.

---

## Results

216 examples, balanced across 6 noise categories × 6 input SNRs, from
`dataset_plain`. Quality metrics for the full pipeline are computed on the
gain-compensated output, so the normaliser is not scored for changing level — the one
thing it exists to do.

### Delivered pipeline versus classical baselines

| method | PESQ | STOI | ESTOI | SI-SDR dB | speech atten dB | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| unprocessed | 1.312 | 0.770 | 0.586 | +2.51 | 0.00 | – |
| spectral subtraction | 1.414 | 0.701 | 0.531 | −14.01 | 8.08 | 0.023 |
| Wiener | 1.442 | 0.727 | 0.551 | −8.84 | 5.23 | 0.025 |
| DeepFilterNet3 only | 1.986 | 0.852 | 0.716 | +4.49 | 3.69 | 0.108 |
| **DFN + normalisation** | **1.986** | **0.852** | **0.716** | **+4.49** | **3.69** | 0.128 |

The delivered pipeline beats both classical baselines on PESQ (+0.54 over Wiener),
STOI (+0.13) and ESTOI (+0.17), and it is the only method that improves SI-SDR at all.
Spectral subtraction and Wiener both make STOI *worse than doing nothing* while
attenuating speech by 5–8 dB: they buy their noise reduction by damaging the talker.
That is the concrete version of the problem statement's argument against traditional
methods.

### The normalisation stage does exactly one thing, and does it

| | mean level | std dev | error vs −26 dBFS target |
| --- | --- | --- | --- |
| unprocessed | −21.8 dBFS | 4.69 dB | – |
| DFN only | −26.4 dBFS | 4.37 dB | – |
| **DFN + normalisation** | −23.4 dBFS | **3.34 dB** | +2.6 dB |

Level spread is reduced by 24%, and the quality metrics are *identical* to DFN alone
(PESQ 1.986 both, STOI 0.852 both, SI-SDR +4.49 both). The gain is provably invertible:
dividing the output by the recorded gain envelope reproduces the DFN output exactly, and
that is a test in the suite.

Two honest notes on this stage. The residual +2.6 dB bias comes from the AGC's internal
level estimate not being identical to the P.56-style active-level measurement used for
reporting. And there is a measured trade-off available: smoothing the level estimate in
the dB domain instead of the power domain gave a tighter spread (2.15 dB std) but a much
larger bias (+5.1 dB). The power-domain version shipped because averaging power is what
an RMS active-level measurement actually computes, so its behaviour is principled rather
than tuned; the remaining bias can be removed by shifting `normalise.target_dbfs`.

### Targets: measured, including the failures

Over the full corpus SNR range (−10 to +15 dB):

| Target | Required | Measured | Verdict |
| --- | --- | --- | --- |
| STOI | > 0.85 | **0.852** | **PASS** |
| PESQ wideband | > 2.5 | 1.986 | **FAIL** |
| SNR improvement | > 15 dB | +1.97 dB | **FAIL** |

STOI passes only because of the attenuation limit: `atten_lim_db=30` moved it from
0.851 to 0.856 on the ablation subset, which was the difference between missing and
meeting the target.

A single average across −10 to +15 dB is not a meaningful figure, and the per-SNR
breakdown shows why:

| input SNR | PESQ (in → out) | STOI (in → out) | SNR improvement |
| --- | --- | --- | --- |
| −10 dB | 1.096 → 1.213 | 0.550 → 0.671 | **+9.76 dB** |
| −5 dB | 1.107 → 1.536 | 0.661 → 0.793 | **+8.97 dB** |
| 0 dB | 1.127 → 1.817 | 0.740 → 0.848 | +4.89 dB |
| +5 dB | 1.231 → 2.118 | 0.836 → 0.906 | +0.71 dB |
| +10 dB | 1.427 → 2.450 | 0.887 → 0.935 | −2.80 dB |
| +15 dB | 1.885 → 2.783 | 0.949 → 0.958 | −9.69 dB |

Two things are visible:

1. **SNR improvement is bounded by the noise present.** At +15 dB input there is not
   15 dB of noise left to remove, so the metric cannot reach the target no matter how
   good the system is. The improvement is largest exactly where it matters: +9.8 dB at
   −10 dB SNR.
2. **PESQ improves substantially at every SNR** (+0.12 to +0.90), and passes 2.5 at
   +10 and +15 dB input. STOI improves at every SNR too. The aggregate fails because
   the corpus is deliberately weighted toward brutal SNRs.

The negative SNR improvement at high input SNR is real and is a property of the
pretrained model: with little noise to remove it still applies suppression and damages
clean speech. `neural.atten_lim_db=30` already caps this (it raised PESQ at +15 dB
input from 2.874 to 3.032 on the ablation subset); a fine-tuned model would be the
proper fix.

### Per noise category

| category | taxonomy | PESQ | STOI | SNR improvement | speech atten |
| --- | --- | --- | --- | --- | --- |
| siren | non-stationary | 2.278 | 0.900 | +2.98 dB | 2.79 dB |
| engine | stationary | 2.039 | 0.868 | +3.10 dB | 3.18 dB |
| helicopter | stationary | 1.986 | 0.855 | +1.88 dB | 3.36 dB |
| airplane | stationary | 1.971 | 0.837 | +2.24 dB | 3.82 dB |
| train | non-stationary | 1.883 | 0.821 | −0.23 dB | 5.50 dB |
| **gunshot** | **impulsive** | **1.759** | **0.831** | **+1.88 dB** | 3.49 dB |

Gunshot is the hardest case, as expected: an impulsive event is broadband, brief and
does not fit the noise model any suppressor learns from stationary statistics.

### The two-microphone design scores better, and we still cannot use it

This is the most important honest finding in the project. The rejected adaptive
designs were measured, and they were handed the corpus's **noise-only file** as their
reference — the exact noise, sample-aligned. No single-microphone system can obtain
that signal, so these rows are an upper bound, not an achievable result:

| method | PESQ | STOI | SI-SDR dB | speech atten dB |
| --- | --- | --- | --- | --- |
| NLMS then DFN (needs 2 mics) | **2.629** | **0.941** | +6.17 | 2.53 |
| DFN then NLMS | 1.996 | 0.864 | +4.59 | 3.69 |
| NLMS only | 1.763 | 0.907 | +4.81 | 1.20 |
| classical LMS | 1.390 | 0.798 | +5.78 | 0.00 |
| **DFN + normalisation (delivered)** | 1.986 | 0.852 | +4.49 | 3.69 |

Read that carefully:

- **`NLMS then DFN` beats the delivered pipeline by 0.64 PESQ and 0.09 STOI, and it is
  the only configuration that passes the PESQ > 2.5 target.** It is excluded because it
  needs a second physical microphone placed at the noise source. That is a hardware
  decision, not a performance one, and saying otherwise would be dishonest.
- **`DFN then NLMS` gains almost nothing** (1.996 vs 1.986 PESQ, 0.864 vs 0.852 STOI) —
  and that is with a *perfect* reference. Running the adaptive filter after the model
  cannot work properly, because the model applies a time-varying non-linear gain that
  breaks the linear relationship the filter depends on. With any realistic reference it
  would be worse than nothing. This confirms the decision to drop it.
- Classical LMS needed its step size cut to 0.001 to stay stable at all; at 0.05 it
  diverged on 8 of 8 test examples. Its `speech atten` of 0.00 dB alongside a decent
  SI-SDR means it is barely doing anything.

If a second microphone ever becomes available, the measured prize is **+0.64 PESQ and
+0.09 STOI**. That is the strongest single recommendation this evaluation produces.

---

## Install

### Development host (x86, Windows or Linux)

```bash
python3.11 -m venv .venv                  # 3.8-3.11; 3.12+ is ruled out by numpy<2
.venv/bin/pip install -U pip
.venv/bin/pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[all]"
.venv/bin/anc selftest
```

Two pins that matter:

- **numpy < 2** — `deepfilternet 0.5.6` requires `numpy>=1.22,<2.0`. This is what rules
  out Python 3.12+.
- **torch 2.5.1** — torch 2.6 flipped `torch.load(weights_only=True)`, which breaks
  loading the DeepFilterNet checkpoint.

### Jetson Orin NX

Verified against the actual target board:

| | |
| --- | --- |
| L4T | R35.5.0 → **JetPack 5.1.3** (`nvidia-jetpack 5.1.3-b29`) |
| Board | NVIDIA Orin NX Developer Kit, 8× Cortex-A78AE, 15 GB usable |
| OS / Python | Ubuntu 20.04, **Python 3.8.10** |
| Power mode | MAXN (mode 0) |
| Capture hardware | **none attached** — see below |

```bash
bash scripts/install_jetson.sh
anc -c configs/default.yaml -c configs/jetson.yaml selftest
anc -c configs/default.yaml -c configs/jetson.yaml evaluate --methods delivered
```

Four things about this board specifically:

**Python 3.8 is a hard constraint on the code, not just the deps.** `from __future__
import annotations` defers annotation evaluation for functions, but **pydantic evaluates
model field annotations at runtime**, so `list[str]` in a config field raises
`TypeError: 'type' object is not subscriptable` on 3.8. Every field in `config.py` uses
`typing.List`/`Tuple` for that reason, and `tests/test_py38_compat.py` scans the AST to
keep it that way — it fails on the development machine rather than on the board.

**PyTorch comes from NVIDIA, never PyPI.** The install script uses the wheel verified to
exist for this release:

```
https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/
  torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl
```

`cp38` matches the board's Python, and torch 2.1.0 is conveniently below the 2.6 change
to `torch.load(weights_only=...)` that breaks the DeepFilterNet checkpoint.

**torchaudio has no matching wheel**, and `deepfilternet` imports it at module load
(`df/io.py`) even though it never calls it on the inference path — this project does all
file I/O through soundfile and resampling through soxr. The script therefore tries, in
order: an existing install, a source build at the matching tag (20–40 min), and finally
`scripts/torchaudio_shim.py`, an import-only shim where **every stubbed function raises**
rather than returning plausible-looking wrong data. `anc selftest` reports when the shim
is active so it can never be mistaken for the real thing. Force it with
`TORCHAUDIO_MODE=shim bash scripts/install_jetson.sh`.

**There is no microphone attached.** `arecord -l` on this board lists only
`APE / tegra-dlink XBAR-ADMAIF` entries, which are Tegra's internal audio DMA channels,
not capture hardware. Live microphone mode needs a USB mic or headset plugged in.
Everything else — offline evaluation, the dashboard, the PDF report — runs with no audio
hardware at all.

The core dependency set is deliberately compilation-free on aarch64: `deepfilternet` and
`DeepFilterLib` both ship `manylinux_2_28_aarch64` wheels for cp38–cp311, and Ubuntu
20.04's glibc 2.31 satisfies `manylinux_2_28`. `pesq` (needs a C compiler) and
`pyroomacoustics`/`h5py` (compiler plus libhdf5) are optional extras.

Every privileged step is printed before it runs, and `nvpmodel -m 0` / `jetson_clocks`
are printed as recommendations rather than executed — `tests/test_install_script.py`
asserts that, heredoc-aware so it can tell printing from running.

Two values in `configs/jetson.yaml` are marked `TUNE` and should be re-derived from the
board rather than trusted from the x86 host:

```bash
anc -c configs/default.yaml -c configs/jetson.yaml benchmark --threads 1,4,6,8
```

---

## Use

```bash
anc selftest                     # environment + a tiny end-to-end run, go/no-go
anc corpus                       # what is in dataset_plain and how balanced it is
anc evaluate                     # the delivered methods over a balanced subset
anc evaluate --methods all       # adds the rejected two-microphone designs
anc dashboard                    # the Streamlit dashboard
anc benchmark                    # RTF and chunked-vs-offline agreement
anc list-devices                 # audio devices
anc report                       # summarise the most recent session

# one file through the pipeline
anc run --mode offline_file --primary noisy.wav --clean clean.wav

# live microphone (use headphones)
anc run --mode live_mic --duration 30
```

Every parameter in `configs/default.yaml` is overridable:

```bash
anc evaluate --set normalise.target_dbfs=-23 --set neural.atten_lim_db=25 --set plain.per_cell=3
```

Unknown keys are rejected rather than ignored, so a typo fails loudly.

---

## How it works

### Pre-processing

DC removal (1st-order high-pass at 5 Hz) then a 2nd-order Butterworth high-pass at
80 Hz, which removes rumble and handling noise below the speech band. Filter state is
carried between blocks, so block-by-block processing gives bit-identical output to
whole-signal processing. NaN/Inf guards zero the affected frame, count it, and log
once rather than per frame.

### Neural stage

Pretrained DeepFilterNet3: 2,135,484 parameters, 48 kHz, 20 ms window, 10 ms hop,
complex-domain ERB + deep filtering, so phase is preserved.

**Algorithmic latency is 40 ms**, and that figure is read from the installed model
rather than assumed:

| component | latency |
| --- | --- |
| one frame | 10 ms |
| STFT/ISTFT loop (`n_fft - hop`) | 10 ms |
| model lookahead (2 frames) | 20 ms |
| **total** | **40 ms** |

`conv_lookahead` and `df_lookahead` are both 2 frames but are **not additive**:
`deepfilternet3.py` asserts `conv_lookahead >= df_lookahead` and both shift the same
time axis, so the output at frame *t* depends on input up to *t+2*. This matches the
40 ms reported in the DeepFilterNet2 paper.

Framing: the offline path calls the reference whole-signal enhancer. The live path uses
**chunked** framing — overlapping chunks with a crossfade at the seams — which reuses
the reference inference code unmodified and cannot silently diverge from it. Its
latency is one chunk (1.5 s by default) and it is labelled as buffering, never as
low-latency streaming. True per-hop streaming is **not implemented**; the reason is
documented in `enhance/streaming.py` (DFN's Python API exposes only offline
`enhance()`, and threading GRU state plus convolution context buffers through the model
is a substantial change with a real risk of diverging from the reference).

### Volume normalisation

Speech-aware AGC, per 10 ms frame:

1. A VAD decides whether the frame contains speech.
2. On speech frames, the active speech level is tracked (50 ms attack, 800 ms release).
3. Wanted gain = `target − measured`, clamped to `[−12, +24] dB`.
4. Applied gain moves toward it (150 ms attack, 600 ms release) and is **held frozen
   during pauses**. This is the part that matters: a naive AGC raises its gain when the
   talker stops and pumps the residual noise up with it.
5. Gain is interpolated per sample across the frame, so there is no zipper noise.
6. A look-ahead soft limiter (5 ms) catches transients before they clip.

`peak`, `rms` and `off` modes are also available.

### Why there is no adaptive filter

An NLMS stage needs a reference microphone carrying noise correlated with the noise in
the primary channel. With one microphone that signal does not exist. Both placements
were implemented, measured with a perfect reference, and rejected — see
[the results table](#the-two-microphone-design-scores-better-and-we-still-cannot-use-it).
The code is retained so the measurement is reproducible rather than asserted.

---

## Measurement definitions

Stated explicitly, because "SNR improvement" is ambiguous and the headline depends on
the choice.

- **SI-SDR** — the test signal is projected onto the clean signal; the aligned part is
  signal, everything else is distortion. Scale invariant, so an overall gain change
  cannot move it.
- **SNR improvement** — `SI-SDR(output) − SI-SDR(unprocessed input)`. Bounded above by
  how much noise was present, hence the per-SNR table.
- **Gain compensation** — the AGC's gain is known and exactly invertible, so quality
  metrics for the full pipeline are computed on `output / gain_envelope`. Without this,
  SI-SDR reads the gain ride as distortion: a time-varying gain is *not* scale
  invariant. An early version of this code lost 14 dB of SI-SDR to an uncompensated
  5 ms limiter delay while PESQ did not move at all, because PESQ realigns internally
  and SI-SDR does not.
- **Level alignment** — level-sensitive metrics (segmental SNR, LSD, direct SNR) are
  computed on a gain-aligned copy for the same reason.
- **Speech attenuation** — how much quieter the speech component became. This is the
  metric that catches a system "winning" by muting the talker.
- **PESQ** wideband ITU-T P.862.2 at 16 kHz; **STOI/ESTOI** at 10 kHz internally.

---

## The corpus

`dataset_plain`: 5000 labelled 16 kHz mono examples with clean / noisy / noise-only
triplets, plus category and mixing SNR.

| category | taxonomy | examples | share |
| --- | --- | --- | --- |
| gunshot | impulsive | 4385 | 87.7% |
| airplane | stationary | 135 | 2.7% |
| helicopter | stationary | 134 | 2.7% |
| engine | stationary | 125 | 2.5% |
| siren | non-stationary | 113 | 2.3% |
| train | non-stationary | 108 | 2.2% |

SNR: −10, −5, 0, +5, +10, +15 dB, roughly 800–880 examples each.

Two consequences, both handled rather than ignored:

- **Gunshot is 88% of the corpus**, so an unweighted average would simply be a gunshot
  score. The evaluation draws a **category × SNR balanced subset** (`plain.per_cell`
  per cell, 216 examples by default) and every table is broken out per category.
- `clean/` and `noisy/` also hold **4540 plain-named files** (`00073.wav`) with no
  metadata row — no category, no SNR. They are indexed as an unlabelled pool for
  listening and are excluded from scored results.

---

## Dashboard

```bash
anc dashboard
```

- **Overview** — target verdicts, method comparison, per-category results, the
  generated charts, and the PDF for download.
- **Explore a file** — pick any corpus example by category and SNR, run the pipeline,
  and get: a metric table per stage, a player for every stage *and* for the removed
  noise (if you hear speech in it, the pipeline is taking too much), stacked waveforms,
  shared-scale spectrograms, and a **spectral delta per stage** where blue is energy
  removed and red is energy added.
- **Corpus** — composition and balance.
- **Live microphone** — record, process, compare before and after.
- **System** — host, latency budget, and a live RTF benchmark.

---

## Performance

Measured on the development host (see any report's session table for the exact
machine). No claim is made about hardware that was not measured.

| configuration | RTF |
| --- | --- |
| DFN whole-file, 1 CPU thread | 0.075–0.085 |
| DFN whole-file, 8 CPU threads | 0.050–0.076 |
| Full pipeline (offline) | 0.128 |
| Batch evaluation, 7 workers | 3.0× real time end to end, all 9 methods |

### Chunked live framing versus the reference path

The live path processes overlapping chunks, which resets the model's recurrent state at
every boundary unless warm-up context is supplied. Measured agreement with the reference
whole-file output, as SI-SDR on the same input, 1 CPU thread:

| warm-up context | chunk 1.0 s | chunk 1.5 s |
| --- | --- | --- |
| 0.00 s | 11.8 dB | 10.1 dB |
| **0.25 s (default)** | **17.5 dB** | **17.4 dB** |
| 0.50 s | 17.3 dB | 17.3 dB |
| 1.00 s | 17.2 dB | 18.5 dB |

0.25 s of discarded context captures essentially all of the available improvement for a
few percent of extra compute. This is a *reproduction fidelity* figure, not a quality
score: 18 dB is about 12% RMS difference from the reference. No perceptual threshold is
claimed — that would need a listening test.

Single-thread RTF around 0.06 means the model fits inside one CPU core's real-time
budget with roughly 16× headroom. That is **portability evidence, not a Jetson
result**: it is a necessary condition for an embedded target, not a sufficient one.
Multi-threading barely helps, which says the workload is latency bound rather than
throughput bound — useful to know before reaching for a GPU on a 2.1M parameter model.

---

## Training, not executed

No model was trained. The design, had there been time and hardware:

- **Data** — the corpus already provides clean/noisy/noise triplets at known SNRs. For
  training scale, mix on the fly from source corpora rather than materialising pairs:
  100 h of 48 kHz float32 across three streams is ~207 GB, which is impractical.
  Reproducibility comes from seeding plus per-example manifests.
- **Framework** — DeepFilterNet's own trainer (`df/train.py`) takes *separate* speech /
  noise / RIR HDF5 files and mixes internally; it does not accept pre-mixed pairs. Its
  `DeepFilterDataLoader` ships **x86_64 Linux wheels only** (no aarch64, no Windows),
  so training cannot run on the Jetson or on Windows without building Rust + HDF5 from
  source.
- **Losses** — DFN's multi-resolution spectral loss, plus SI-SNR, plus a
  transient-aware weight that increases loss around detected impulsive events so
  gunshot suppression is not averaged away.
- **The variant worth trying** — fine-tuning on this corpus's gunshot-heavy
  distribution. Gunshot is the weakest category (PESQ 1.780) and is 88% of the
  available data, which is the clearest opportunity in the project.

---

## Troubleshooting

| symptom | cause and fix |
| --- | --- |
| `numpy 2.x` errors on import | `deepfilternet` needs `numpy<2`: `pip install "numpy>=1.22,<2"` |
| Checkpoint fails to load | torch ≥ 2.6 flipped `weights_only`. Use `torch==2.5.1`. |
| `torch.cuda.is_available()` is False on Jetson | PyPI torch has no CUDA. Reinstall from the NVIDIA index (`scripts/install_jetson.sh`). |
| Device refuses 48 kHz | Pick another device with `anc list-devices`. On Windows, set the rate in Sound Control Panel → Device Properties → Advanced. |
| Feedback howl in live mode | You are monitoring on speakers. Use headphones, or `--no-monitor`. |
| Output quiet or uneven | Raise `normalise.target_dbfs` / `normalise.max_gain_db`. Check `agc_gain_range_db` in the report. |
| Output pumps in pauses | `normalise.hold_during_pause` is false, or the VAD is not firing. Check `held_fraction`. |
| PESQ reported as unavailable | `pesq` failed to build. Install a compiler, or set `metrics.pesq=false`. |
| xruns in live mode | Raise `live.ring_capacity_s`, or reduce `neural.num_threads` so the audio callback is not starved. |
| Evaluation too slow | Lower `plain.per_cell`, or raise `offline.workers`. |
| Speech audible in the removed track | Suppression is too aggressive. Set `neural.atten_lim_db` to 20–30. |

---

## Layout

```
anc_defence/
  config.py            pydantic schema, YAML loading, CLI overrides, validation
  pipeline.py          stage orchestration and tap points
  evaluate.py          method comparison, parallel batch evaluation
  audio/               file I/O and resampling, device enumeration, ring buffer
  dsp/                 preprocess, VAD, normalise (AGC), impulse detection,
                       nlms + fdaf (rejected designs, retained for measurement),
                       baselines/ spectral subtraction, Wiener, LMS
  enhance/streaming.py DeepFilterNet3 wrapper, chunked framing, latency accounting
  dataset/             plain.py (the supplied corpus), plus a synthetic generator
  metrics/             intrusive, non-intrusive, ERLE, event-local, aggregation, system
  report/              plots, PDF, JSON/CSV export
  modes/               offline_file, live
  ui/                  cli.py, app.py (Streamlit), dashboard.py, selftest.py
configs/               default.yaml, jetson.yaml
scripts/install_jetson.sh
tests/
legacy/                the original MATLAB NLMS experiment and Colab script
```

Every run writes a timestamped session directory containing the resolved config, a
full log, `report.pdf`, `metrics.json`, `metrics.csv`, per-stage WAV artefacts and all
figures.

See `DEMO.md` for the demo-day command sequence and `LICENSES.md` for data licensing.
