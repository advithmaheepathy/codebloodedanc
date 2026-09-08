# Demo day runbook

Exact commands, in order, with what you should see. Everything here has been run.

**Before the room:** run [step 0](#0-five-minutes-before) and leave the dashboard
running. It is the thing to present from.

---

## 0. Five minutes before

```bash
# Windows
.venv\Scripts\activate
# Linux / Jetson
source .venv/bin/activate

anc selftest
```

Expect `GO: everything passed.` and a table of twelve checks. Warnings are acceptable;
a `[FAIL]` is not — see [If something breaks](#if-something-breaks).

```bash
anc dashboard
```

Opens `http://localhost:8501`. Leave it running in its own terminal.

On the Jetson, add the overlay to every command:

```bash
anc -c configs/default.yaml -c configs/jetson.yaml selftest
```

---

## 1. The one-minute version

Open the dashboard, **Explore a file** tab:

1. Category **gunshot**, SNR **−10 dB**, press **Process**.
2. Play **noisy input**, then **after_normalise**. The difference is obvious.
3. Play **removed by the pipeline**. It should be gunshot and hiss with no
   intelligible speech in it. That is the check that the system is not simply muting
   the talker.
4. Scroll to **What each stage changed**. The blue regions in the spectral delta are
   energy the model removed; the speech formants stay neutral.

That is the whole story: noise out, speech kept, level fixed.

---

## 2. The numbers

Dashboard, **Overview** tab, most recent session. Or from the terminal:

```bash
anc report
```

Point at three things:

**a. It beats the classical baselines on the same data.**

| method | PESQ | STOI |
| --- | --- | --- |
| unprocessed | 1.312 | 0.770 |
| spectral subtraction | 1.414 | 0.701 |
| Wiener | 1.442 | 0.727 |
| **ours** | **1.963** | **0.848** |

Both classical methods make STOI *worse* than doing nothing, and both attenuate speech
by 5–8 dB. They buy noise reduction by damaging the talker.

**b. The improvement is largest where it matters.** Per-SNR table:

| input SNR | PESQ in → out | SNR improvement |
| --- | --- | --- |
| −10 dB | 1.096 → 1.278 | **+9.8 dB** |
| −5 dB | 1.107 → 1.599 | **+9.0 dB** |
| 0 dB | 1.127 → 1.850 | +4.9 dB |
| +15 dB | 1.885 → 2.653 | −9.7 dB |

At +15 dB input there is not 15 dB of noise left to remove, so the metric falls by
construction. The honest headline is the low-SNR regime.

**c. The normaliser halves the level spread at zero quality cost.**

| | std dev of output level |
| --- | --- |
| unprocessed | 4.69 dB |
| DFN only | 4.12 dB |
| **DFN + normalisation** | **2.15 dB** |

PESQ, STOI and SI-SDR are *identical* between DFN-only and the full pipeline. The gain
is provably invertible, so the stage changes level and nothing else.

---

## 3. Live microphone

Use headphones. Speaker monitoring will feed back and howl.

Dashboard, **Live microphone** tab: pick the input device, 6 seconds, **Record and
process**. Talk over some noise (play a siren on a phone).

Or from the terminal, with the live text dashboard:

```bash
anc list-devices
anc run --mode live_mic --duration 30
```

The dashboard shows input/output level, VAD state, RTF, xruns and the ring high-water
mark live. Expect **0 xruns** and RTF well under 1.

Say plainly: output lags input by about **1.5 s**, because the neural stage buffers
into chunks. That is a buffering choice, not the model's latency — the model itself is
40 ms. Do not call it low-latency streaming.

---

## 4. If they ask about the Jetson

```bash
anc benchmark --seconds 5 --threads 1,0
```

Reports RTF for one thread and for the default thread count, plus how closely the
chunked path matches the reference whole-file path.

The argument: single-thread CPU RTF is about **0.06**, roughly 16× real-time headroom
on one core, and multi-threading barely helps, so this is latency bound rather than
throughput bound. That is why it fits an embedded target. State it as portability
evidence for whatever host the report's session table names — not as a Jetson
measurement unless the report was produced on the Jetson.

---

## 5. If they ask why there is no adaptive filter

Short answer: an adaptive filter needs a second microphone at the noise source, and
this is a single-microphone system.

Long answer, with numbers — run:

```bash
anc evaluate --methods all --per-cell 3
```

| method | PESQ | STOI |
| --- | --- | --- |
| NLMS then DFN (**needs 2 mics**) | 2.527 | 0.936 |
| DFN then NLMS | 1.941 | 0.847 |
| **DFN + normalisation (delivered)** | 1.963 | 0.848 |

Be straight about this: **the two-microphone design scores better** — 0.56 PESQ better,
and it passes the PESQ > 2.5 target. It was given the exact noise-only file as its
reference, sample-aligned, which no single-microphone system can obtain, so that row is
an upper bound. And `DFN then NLMS` gains nothing (1.941 vs 1.963), because the model
applies a time-varying non-linear gain that breaks the linear relationship the filter
depends on.

The honest framing: we measured both, one is unachievable with one microphone and the
other does not help, so neither ships. If a second microphone becomes available, the
measured prize is +0.56 PESQ.

---

## 6. Regenerate everything from scratch

Roughly 15 minutes total.

```bash
anc corpus                                      # ~10 s   what is in the dataset
anc evaluate --methods delivered --per-cell 6   # ~6 min  216 examples, 5 methods
anc evaluate --methods all --per-cell 6         # ~12 min 216 examples, 9 methods
anc report                                      #         summarise the newest session
```

Faster, for a smoke check:

```bash
anc evaluate --methods delivered --per-cell 1 --workers 7   # ~1 min, 36 examples
```

Each run writes `sessions/<timestamp>_offline_file/` containing `report.pdf`,
`metrics.json`, `metrics.csv`, `config.yaml`, `session.log`, `audio/` and `figures/`.

---

## Talking points, in priority order

1. **Single microphone, two stages: suppression then level control.** No second mic, no
   adaptive filter, nothing that cannot be built with the hardware in the room.
2. **+9.8 dB SNR improvement at −10 dB input SNR**, and PESQ improves at every SNR.
3. **It beats spectral subtraction and Wiener on the same data** — and both of those
   make intelligibility worse than doing nothing.
4. **Gunshot is the hardest category and is reported separately** (PESQ 1.780). 88% of
   the corpus is gunshot, so we use a balanced subset rather than letting one class set
   the headline.
5. **The normaliser is provably transparent**: identical quality metrics, half the
   level spread.
6. **Every number is reproducible** with one command, and every session carries its own
   config, log and PDF.
7. **We report the failures.** All three mandated targets fail on the aggregate; the
   per-SNR breakdown shows the aggregate is the wrong statistic and where the system
   does deliver.

---

## What not to claim

- Do not say the model was trained or fine-tuned. It is the pretrained DeepFilterNet3
  checkpoint, unmodified.
- Do not call the live path low-latency. It buffers ~1.5 s. The *model* is 40 ms.
- Do not quote a Jetson performance figure unless the report in front of you was
  generated on the Jetson. Check the session table's "Jetson hardware" row.
- Do not present the aggregate SNR improvement (+1.95 dB) as the headline without the
  per-SNR table. It is a ceiling artefact, and hiding that invites the question you
  least want.
- Do not claim fullband performance. The corpus is 16 kHz upsampled, so the 8–24 kHz
  band is empty.

---

## If something breaks

| symptom | fix |
| --- | --- |
| `selftest` fails on the model | Network was needed on first run to cache weights. Once cached it is offline. Re-run. |
| Dashboard will not start | `pip install -e ".[dashboard]"` |
| No audio devices | `anc list-devices`. If empty, the offline tabs still work — present from those. |
| Device refuses 48 kHz | Pick another with `--input-device <substring>`. Windows: Sound Control Panel → Device Properties → Advanced. |
| Feedback howl | Headphones, or `--no-monitor`. |
| Live mode drops out (xruns > 0) | Raise `--set live.ring_capacity_s=16`, lower `--set neural.num_threads=4`. |
| Evaluation too slow | `--per-cell 1 --workers 7` |
| PESQ shows as unavailable | `pip install -e ".[metrics]"`, or `--set metrics.pesq=false` |
| Everything is broken | You still have the last good session directory. `anc report --session sessions/<name>` and present the PDF. |

**Fallback plan:** the PDF report in the most recent session directory is a complete,
self-contained deliverable. If nothing runs on the day, open that.
