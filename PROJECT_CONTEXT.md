# Project Context — AI/ML Adaptive Noise Cancellation for Defence Comms

> Full handoff document. Read this to resume the project cold in a new session.
> Last updated at commit `b1c5e67`.

---

## 1. What this project is

Single-microphone AI speech noise-suppression system for defence/mission-critical
communications. **SIH (Smart India Hackathon) problem statement 26052.**

The delivered pipeline is:

```
mic  ->  preprocess  ->  DeepFilterNet3 (pretrained, unmodified)  ->  speech-aware volume normalisation (AGC)  ->  out
```

- **One microphone only.** An earlier two-microphone / NLMS design was abandoned.
- **No model was trained.** The neural stage is the **pretrained DeepFilterNet3
  checkpoint, used unmodified**. A training pipeline is *described* in README.md but not run.
- **No NLMS in the delivered path.** Running NLMS after DeepFilterNet degraded output;
  running it before contradicts the single-mic problem framing (NLMS needs a reference
  mic). NLMS code + measurements are retained only as "rejected designs" for the record.
- Runs on a **Windows laptop** (dev) and an **NVIDIA Jetson Orin NX** (deployment target).

### The DRDO problem statement targets
`SNR > 15 dB, STOI > 0.85, PESQ > 2.5`, low latency, real-time, edge-deployable.
These three are read as **absolute output values** (see SNR discussion in §6).

---

## 2. Repository

- **Local path:** `e:\coding\SIH_2026`
- **GitHub:** https://github.com/advithmaheepathy/codebloodedanc  (remote `origin`, branch `master`)
- **Current HEAD:** `b1c5e67`, pushed, in sync with `origin/master`.
- **Working tree:** clean except `dataset_plain.zip` is **untracked** (open decision:
  add `*.zip` to `.gitignore` or leave it — it is harmless untracked, and `git push`
  never sends it).

### Layout
```
anc_defence/            main package
  audio/                device enumeration, WAV I/O, ring buffer
  dataset/              dataset_plain loader, synthetic generator (build/mixer/rir/...)
  dsp/                  preprocess, VAD, normalise (AGC), nlms/fdaf (rejected), baselines
  enhance/streaming.py  DfnModel wrapper + ChunkedEnhancer (the neural stage)
  metrics/              intrusive (PESQ/STOI/SI-SDR/output-SNR), erle, events, categories, nonintrusive
  modes/                offline_file.py, live.py, common.py  (the run entry points)
  report/               plots.py, pdf.py, export.py
  ui/                   cli.py (the `anc` command), app.py (Streamlit dashboard), selftest.py
  pipeline.py           Pipeline class: wires the stages, offline + streaming paths
  config.py             pydantic v2 config schema
configs/                default.yaml, jetson.yaml
scripts/                install_jetson.sh, torchaudio_shim.py, build_sample_tables[_pdf].py, benchmarks
tests/                  37 tests, all passing
dataset_plain/          the supplied corpus (GITIGNORED - not on GitHub, licence unverified)
model/                  user-supplied reference app "deepfilter-anc-main" (same DFN3; NOT a different model)
legacy/                 old NLMS reference material (nlms.pdf tracked, ~1.9 MB)
sessions/               run outputs (GITIGNORED)
```

### Environment
- **venv:** `.venv` at repo root. Python **3.11.9** on the laptop.
- **Activate (Windows):**
  - PowerShell: `.\.venv\Scripts\Activate.ps1`
  - cmd.exe: `.venv\Scripts\activate.bat`  (NOT the .ps1 — cmd opens it in Notepad)
  - Git Bash: `source .venv/Scripts/activate`
  - Or skip activation: `.\.venv\Scripts\python.exe -m anc_defence.ui.cli <cmd>`
- Laptop is **CPU only** (`torch 2.5.1+cpu`, CUDA not available).
- Key pins: torch 2.5.1+cpu, deepfilternet 0.5.6, numpy 1.26.4 (`<2`, required by deepfilternet),
  streamlit 1.63 (laptop), pandas `<2.2`, matplotlib built-ins only.

---

## 3. The CLI

Invoke as `anc <cmd>` (after activating venv) or `python -m anc_defence.ui.cli <cmd>`.
Config overlays: `-c configs/default.yaml -c configs/jetson.yaml`. Overrides: `--set key=value`.

Commands: `selftest  corpus  list-devices  report  licenses  fetch-data  build-dataset
dataset-stats  run  evaluate  benchmark  calibrate  dashboard`

Common runs:
```
anc selftest
anc evaluate --methods all --per-cell 6 --workers 4          # full comparison, all 9 methods
anc run --mode live_mic --duration 30                        # live mic (or use the dashboard)
anc dashboard                                                # localhost:8501
anc -c configs/default.yaml -c configs/jetson.yaml selftest  # on the Jetson
```

---

## 4. The dashboard (Streamlit, `anc_defence/ui/app.py`)

Five tabs: **Overview** (headline figures + method comparison + per-category/per-SNR tables + charts),
**Explore a file** (run pipeline on one corpus example, see waveform/spectrogram/spectral-delta per stage),
**Corpus** (dataset composition), **Live microphone**, **System**.

### Demo aids added for the SIH video (live mode)
- **`live.demo_delay_s`** (`LiveCfg`, default 0.0, range 0-10 s): a pure post-processing
  output-hold queue (`anc_defence.modes.live._DemoDelayQueue`). Samples are never
  altered, only the moment they are released to `out_ring`/`recorded_output` is pushed
  back. Exists so a single continuous take (noise happens, pause, clean speech comes
  out) can be filmed without the input and output overlapping at the real sub-second
  latency. The report's "measured latency" figure (`measure_pipeline_latency`) always
  reflects the real minimum latency regardless of this setting - it uses a separate
  probe pipeline that never touches the delay queue. Set once in the Streamlit slider
  before starting a session (not live-adjustable mid-run). Covered by
  `tests/test_live_demo_delay.py`.
- **Live-updating waveform panel** in the Streamlit "Live microphone" tab
  (`run_live_with_live_chart` / `live_waveform_figure` in `ui/app.py`): runs the session
  on a background thread and polls the `LiveEngine`'s `stats`/`recorded_input`/
  `recorded_output` from the main Streamlit thread every ~0.4 s, redrawing a stacked
  input (blue) / output (red) waveform in an `st.empty()` placeholder - the same visual
  style as the offline `waveforms()` report figure, but scrolling. This required making
  `LiveEngine.run()` / `run_live()` accept an optional `dashboard` callback so the
  terminal rich/plain dashboard can be swapped out without duplicating the stream
  start/stop/flush logic.

### Live waveform panel performance fix (Jetson) + CUDA note
- The live panel was slow on the Jetson (fine on the dev laptop): every ~0.4 s redraw
  tick was re-concatenating the *entire* session's recorded blocks (`np.concatenate`
  cost grows with session length) and rendering the raw signal (tens of thousands of
  points/axis) through matplotlib's Agg backend from scratch. Both compete with the
  single-threaded DSP worker for the same CPU on Jetson's weaker cores.
  Fixed in `ui/app.py`: `_tail_from_blocks()` walks the block list backwards and stops
  once enough recent samples exist, so per-tick cost is bounded by the display window
  (default 12 s), not by total session length; `_downsample_for_plot()` decimates to
  ~1500 min/max-envelope points per axis before plotting (min/max, not stride, so
  transient/impulsive peaks between decimated samples are never silently dropped).
  Default `refresh_s` raised 0.4 -> 0.7 s. Covered by `tests/test_live_chart_perf.py`.
  The one-time *final* frame at session end still concatenates and plots the full
  recording (not windowed/decimated) since that only happens once, not per tick.
- **`neural.device: cuda` is broken for live/streaming use, and it is a bug in the
  installed `deepfilternet` package, not in this repo.** Confirmed via full traceback
  on the Jetson: `df/enhance.py`'s `df_features()` calls `audio.numpy()` directly on a
  tensor that is still on `cuda:0`, with no `.cpu()` anywhere in that third-party
  function - this repo's own conversion (`streaming.py`'s `_enhance_raw`, chains
  `.detach().float().cpu().numpy()`) is correct but is never reached because `enhance()`
  crashes first, during model warmup. Not something to fix here without monkeypatching
  third-party code; `configs/jetson.yaml` already pins `neural.device: cpu` for
  unrelated, measured performance reasons (CPU beat CUDA/more-threads in this project's
  own benchmarking), so this doesn't block anything - just don't set cuda on Jetson.

### Live microphone tab (built this session)
- Runs a **full live session from the dashboard** (not terminal-only) and writes a complete
  session directory with `report.pdf` + `metrics.json`, identical to `anc run --mode live_mic`.
- **ANC on/off toggle.** ON = `dfn_then_normalise`; OFF = `passthrough`. The intended demo
  flow: record one session ANC-off (baseline, ~0 dB noise removed by construction), then one
  ANC-on (removes ~25-37 dB), compare in the "ANC off versus ANC on" table that appears below.
- Sessions tagged `..._live_mic_anc_off` / `..._live_mic_anc_on`.
- Latency-profile picker (disabled when ANC off), monitor checkbox (default OFF — speaker
  monitoring feeds back into the mic), optional noise-injection expander (enables PESQ/STOI/
  SI-SDR on a live run by mixing a known corpus noise at a known SNR, keeping the pre-mix
  capture as a pseudo-clean reference).
- Bound to `127.0.0.1` only (no auth).

---

## 5. The corpus (`dataset_plain`)

- 5000 labelled 16 kHz mono examples. `metadata.csv` columns:
  `id, noise_category, snr_db, clean_path, noisy_path, noise_only_path`.
- Category imbalance: **gunshot 4385 (87.7%)**, airplane 135, helicopter 134, engine 125,
  siren 113, train 108. SNRs: -10/-5/0/5/10/15 dB. Plus 4539 unlabelled files, excluded from scoring.
- Because gunshot dominates, all reported numbers use a **category-balanced subset**:
  6 per (category x SNR) cell = **216 examples** (`plain.per_cell: 6`).
- 16 kHz corpus is upsampled to 48 kHz on load because DeepFilterNet3 runs at 48 kHz.
- **NOT on GitHub** (gitignored). To get it on the Jetson: copy `dataset_plain/` by USB/scp.
  Needed only for offline `evaluate`/dashboard offline tabs; a live-mic demo does not need it.

---

## 6. Metrics and the SNR story (IMPORTANT — read before touching numbers)

The single biggest source of confusion this project. There are **three different "SNR"
definitions** and they disagree by 30+ dB on the same audio:

1. **SI-SDR improvement** (`snr_improvement_db`) = SI-SDR(output) - SI-SDR(input).
   Strict: charges for *any* deviation from clean, including benign distortion. Bounded by
   how much noise was present. Averages **+1.95 dB** across the full range, **+7.87 dB** over
   the noisy -10..0 dB regime. Reported as context, NOT the headline.

2. **Output SNR** (`residual_noise_snr_db`) = surviving speech power / residual noise power.
   The classical ANC definition and what the DRDO "SNR > 15 dB" target actually means (it
   sits beside STOI/PESQ which are absolute output figures). **This is the pass/fail metric.**
   Delivered pipeline averages ~**13 dB**, clearing 15 at the noisier categories/inputs.

3. **Whole-signal RMS drop %** — what the `model/` reference app and most live demo
   dashboards display. Weak: counts attenuated *speech* as a win, clamped 0-100, collapses
   at high input SNR. Included only for like-for-like comparison (`rms_reduction_db/_pct`).

### Two bugs fixed this session in the SNR measurement (do not reintroduce)
- **The 47 dB artefact.** `residual_noise_snr_db` was first computed as
  `enhance(clean)` vs `enhance(noise_only)` separately. INVALID: DeepFilterNet is
  **non-linear**, so feeding it noise-only makes its VAD annihilate everything, giving
  `input_snr + ~45 dB`. Fixed by the **mixture-derived gain decomposition**: recover the
  time-frequency gain the enhancer applied to the *real mixture* `G = |Y|/|X|`, apply that
  same `G` to the clean and noise components separately, take the ratio. They sum back to
  the output (mixture additivity verified at -108 dB). Real figure ~13 dB, not 47.
- **The -23 dB artefact.** An AGC gain-ride was being scored by SI-SDR as distortion.
  Fixed earlier: quality metrics are computed on the **gain-compensated** output (the AGC
  envelope is invertible); level is reported separately.

### Other metric facts
- PESQ target 2.5 currently FAILS at ~1.96. **This is a corpus artefact worth stating:**
  the corpus is 16 kHz upsampled, DeepFilterNet3 is trained on 48 kHz studio-clean, so PESQ
  scores against a band-limited reference and caps low. On native-48 kHz clean speech DFN3
  publishes PESQ well above 2.5.
- STOI ~0.848, essentially at the 0.85 target (misses by 0.002 — noise, not a real gap).
- **Pass/fail target tables were REMOVED from reports and dashboard** at user request
  (commit `26e5dca`). Reports now present measured figures on their own terms under a
  "What was measured, and how" section. `check_targets`/`TargetsCfg` still exist in code
  but are unwired.

### Method comparison headline (216 examples, output SNR / SI-SDRi / STOI / PESQ)
| method | out SNR | SI-SDRi | STOI | PESQ |
|---|---|---|---|---|
| unprocessed (baseline) | 2.51 | 2.51 | 0.770 | 1.31 |
| NLMS only (2-mic, ideal ref) | 8.40 | 4.81 | 0.907 | 1.76 |
| DFN only | 12.29 | 4.46 | 0.848 | 1.96 |
| DFN then NLMS (2-mic) | 12.24 | 4.45 | 0.847 | 1.94 |
| NLMS then DFN (2-mic, best hybrid) | 10.95 | 6.12 | 0.936 | 2.53 |
| **delivered (DFN + normalise)** | **13.07** | 4.46 | 0.848 | 1.96 |

Key point: the single-mic delivered pipeline **beats the two-microphone NLMS hybrids** on
output SNR even when the hybrids are given a perfect noise-only reference. NLMS is the
weakest stage, not the thing that reaches 15 dB.

---

## 7. Reports (generated after every run)

- **Live session report: 6 pages.** (1) session metadata + scope statement, (2) measurements:
  reference-free noise removed / AGC summary / signal levels / RTF / latency budget / real-time
  health, (3) AGC trace + waveforms, (4) spectrograms + spectral-delta, (5) level-over-time +
  CPU/memory, (6) metrics glossary. Live runs have NO PESQ/STOI/SI-SDR (no clean reference)
  unless noise injection is used.
- **Offline evaluation report: ~13 pages.** Adds method-comparison table, three-ways-to-
  measure-suppression table, per-category and per-SNR tables, corpus composition, and
  per-method figures.
- **Reference-free "Noise removed" metric** (live): level drop only in VAD-silent regions,
  so muting the talker cannot inflate it; 0 dB when ANC is off. This is the live headline.

### Standalone comparison-tables PDF (built this session)
`scripts/build_sample_tables_pdf.py` renders the 5 requested tables (comparison + per-category
for DFN-only, NLMS-only, both hybrid orders, delivered pipeline) from an existing session's
`metrics.csv`. Output: `sessions/comparison_tables.pdf`. Header-overlap bug fixed by wrapping
header cells as Paragraphs. Terminal version: `scripts/build_sample_tables.py`.

---

## 8. Jetson deployment (Orin NX)

- **Board:** L4T R35.5.0 -> JetPack 5.1.3, Orin NX Dev Kit, Ubuntu 20.04, **Python 3.8.10**,
  15 GB RAM, 8x Cortex-A78AE, MAXN. Has CUDA (device "Orin").
- **Install:** copy repo (git clone), then `SKIP_APT=1 TORCHAUDIO_MODE=shim bash scripts/install_jetson.sh`.
  - `SKIP_APT=1` because the board has a pre-existing broken `nvidia-l4t-kernel` dpkg state
    that makes every `apt` command return an error (NOT our problem — do NOT try to force-fix
    the kernel packages, that can brick the board). All needed system libs were already present.
  - `TORCHAUDIO_MODE=shim` skips a 20-40 min source build; torchaudio is imported by df/io.py
    but never called on the inference path.
  - Torch is JetPack's system torch (2.1.0a0+...nv23.06, cp38), reused via `--system-site-packages`.
- **`configs/jetson.yaml` device tuning (measured on-board):** `neural.device: cpu`,
  `neural.num_threads: 1`. More threads made the live path WORSE (RTF 0.88 at 8 threads/0.5s
  chunk vs 0.15 at 1 thread); the model is kernel-launch-bound, not FLOP-bound. CUDA not
  benchmarked head-to-head; CPU has margin (neural inference RTF 0.149).
- **Audio:** board's built-in "APE / tegra-dlink XBAR-ADMAIF" entries are DMA channels, NOT
  microphones. Live mic needs a USB mic. A USB mic (index 1, "USB Audio Device", 44100 Hz
  native) is working. NOTE: another USB device "AB13X USB Audio" (index 0) is output-only
  (0 input channels) — do not select it as input.

### Jetson-specific fixes made this session (all pushed)
1. **CUDA device mismatch** (`1e7dce6`): DeepFilterNet's `df.utils.get_device()` defaults to
   `cuda:0` whenever CUDA is available, ignoring our `model.to(device)`. On the Jetson this put
   weights on CPU but inputs on GPU -> "Input type ... and weight type ... should be the same".
   Fix: `os.environ["DEVICE"] = device` before `init_df()` in `enhance/streaming.py`
   (get_device reads that env var first).
2. **Dashboard crashes without corpus** (`a60b442`): `get_corpus_records()` now catches
   `FileNotFoundError` and returns empty instead of crashing the whole page.
3. **hashlib `usedforsecurity` on OpenSSL builds** (`5f99fe2`, then broadened `b1c5e67`):
   Jetson's OpenSSL-backed hashlib rejects `usedforsecurity=` on `md5()/sha1()/sha256()/
   sha512()/new()`. Streamlit's cache hasher AND Starlette's file-serving ETag both hit this,
   causing the live-session-end crash AND the broken report download. Fix: a shim at the top
   of `app.py` patches all five hashlib constructors to retry without the kwarg. `usedforsecurity`
   is a FIPS annotation only; dropping it does not change the hash.

---

## 9. Key design decisions (rationale, so they are not re-litigated)

- **NLMS removed from delivered path** (user: "abandon NLMS"). Kept code + measurements as
  rejected-design evidence rather than deleting — the measurement is stronger than silence.
- **`atten_lim_db: null`** (no suppression cap). Sweep showed capping only lowers output SNR
  and PESQ; uncapped is optimal. Cap 30/40/50 all measurably worse.
- **Live chunk 250 ms** (low_latency profile). 60 ms removed less noise AND damaged speech more.
  Profiles: low_latency 250ms->265ms measured, balanced 500ms, quality 1s, max_suppression 1s+postfilter.
- **AGC** (`normalise.mode: agc`): smooths power not dB, holds gain during pauses, 5 ms limiter
  look-ahead compensated offline, gain envelope invertible so quality metrics are gain-compensated.
  Target -26 dBFS active speech.
- **Evaluation = stratified balanced subset** because gunshot is 88% of the corpus.
- **Python 3.8 compat**: pydantic evaluates annotations at runtime, so `list[str]` breaks on 3.8
  despite `from __future__ import annotations`. Use `typing.List/Tuple`. Guarded by `tests/test_py38_compat.py`.
- **Dashboard uses Streamlit**; removed pandas Styler (`background_gradient` needs matplotlib
  >=3.9.3, blocked by numpy<2 pin).

---

## 10. Open items / next steps

- **`dataset_plain.zip`** untracked in working tree — decide: `.gitignore` it (`*.zip`) or leave.
- **PESQ < 2.5** is the one target genuinely not met; framed as a 16 kHz-corpus artefact.
  A native-48 kHz clean test set would likely clear it. Optional.
- **CUDA vs CPU on Jetson** never benchmarked head-to-head; CPU chosen with margin. Optional.
- The `model/deepfilter-anc-main/` folder is the user's reference app — **same pretrained DFN3**,
  uses 3 s chunks, TARGET_RMS 0.08, no atten cap, reports only the clamped RMS-drop %. Not a
  competitor model; used to explain why its "big number" is not an SNR.
- Regenerate the offline report / comparison-tables PDF from the latest session if numbers or
  code changed since `2026-09-09_12-14-00_offline_file`.

## 11. Test / verify
```
.\.venv\Scripts\python.exe -m pytest -q          # 37 tests, all pass
anc selftest                                     # environment + model + devices check
```
