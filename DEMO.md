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

| method | PESQ | STOI | ESTOI |
| --- | --- | --- | --- |
| unprocessed | 1.312 | 0.770 | 0.586 |
| spectral subtraction | 1.414 | 0.701 | 0.531 |
| Wiener | 1.442 | 0.727 | 0.551 |
| **ours** | **1.986** | **0.852** | **0.716** |

Both classical methods make STOI *worse* than doing nothing, and both attenuate speech
by 5–8 dB. They buy noise reduction by damaging the talker. **STOI 0.852 passes the
0.85 target.**

**b. The improvement is largest where it matters.** Per-SNR table:

| input SNR | PESQ in → out | STOI in → out | SNR improvement |
| --- | --- | --- | --- |
| −10 dB | 1.096 → 1.213 | 0.550 → 0.671 | **+9.8 dB** |
| −5 dB | 1.107 → 1.536 | 0.661 → 0.793 | **+9.0 dB** |
| 0 dB | 1.127 → 1.817 | 0.740 → 0.848 | +4.9 dB |
| +15 dB | 1.885 → 2.783 | 0.949 → 0.958 | −9.7 dB |

At +15 dB input there is not 15 dB of noise left to remove, so the metric falls by
construction. The honest headline is the low-SNR regime, where the gain is +9.8 dB.

**c. The normaliser fixes the level at zero quality cost.**

| | mean level | std dev |
| --- | --- | --- |
| unprocessed | −21.8 dBFS | 4.69 dB |
| DFN only | −26.4 dBFS | 4.37 dB |
| **DFN + normalisation** | −23.4 dBFS | **3.34 dB** |

PESQ, STOI and SI-SDR are *identical* between DFN-only and the full pipeline (1.986,
0.852, +4.49 for both). The gain is provably invertible — there is a test asserting that
dividing the output by the recorded gain envelope reproduces the DFN output — so the
stage changes level and nothing else.

---

## 3. Live microphone

Use headphones. Speaker monitoring will feed back and howl.

Dashboard, **Live microphone** tab: pick the input device, 6 seconds, **Record and
process**. Talk over some noise (play a siren on a phone).

Or from the terminal, with the live text dashboard:

```bash
anc list-devices
anc run --mode live_mic --duration 30                    # low_latency, ~75 ms, the default
anc run --mode live_mic --duration 30 --latency quality  # ~1.5 s, higher fidelity
```

The dashboard shows input/output level, VAD state, RTF, xruns and the ring high-water
mark live. Expect **0 xruns** and RTF well under 1.

The default profile is **low_latency**: measured **75 ms end-to-end** (60 ms neural
chunk buffering + 5 ms limiter look-ahead + 10 ms device I/O). That is genuinely
responsive — you talk and hear the cleaned voice back near-instantly. The report prints
the measured figure and its breakdown, taken from the real block path, not a
theoretical sum.

There is also a **quality** profile (`--latency quality`, ~1515 ms) that reproduces the
offline result more faithfully (17.5 dB vs 12 dB agreement) for when latency does not
matter. Switch profiles with `--latency low_latency|quality`.

State it honestly: the *model* is 40 ms algorithmic; the 75 ms is that plus the chunk
buffering the streaming wrapper adds. It is low-latency streaming, but not the sub-50 ms
you would get from DeepFilterNet's native Rust per-frame runtime.

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
| NLMS then DFN (**needs 2 mics**) | 2.629 | 0.941 |
| DFN then NLMS | 1.996 | 0.864 |
| **DFN + normalisation (delivered)** | 1.986 | 0.852 |

Be straight about this: **the two-microphone design scores better** — 0.64 PESQ better,
and it is the only configuration that passes PESQ > 2.5. It was given the exact
noise-only file as its reference, sample-aligned, which no single-microphone system can
obtain, so that row is an upper bound. And `DFN then NLMS` gains almost nothing (1.996
vs 1.986) even with that perfect reference, because the model applies a time-varying
non-linear gain that breaks the linear relationship the filter depends on.

The honest framing: we measured both. One is unachievable with one microphone, the other
does not help, so neither ships. If a second microphone becomes available, the measured
prize is +0.64 PESQ and +0.09 STOI — and that is the strongest recommendation the
evaluation produces.

Do not claim the single-mic pipeline is better than the two-mic one. It is not. It is
the one that can actually be built with the hardware available.

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
2. **+9.8 dB SNR improvement at −10 dB input SNR**, and PESQ and STOI improve at every
   SNR.
3. **STOI 0.852 passes the mandated 0.85 target**, and it passes because of a measured
   decision: capping suppression depth at 30 dB moved it from 0.851 to 0.856.
4. **It beats spectral subtraction and Wiener on the same data** — and both of those
   make intelligibility *worse* than doing nothing.
5. **Gunshot is the hardest category and is reported separately** (PESQ 1.759). 88% of
   the corpus is gunshot, so we use a balanced subset rather than letting one class set
   the headline.
6. **The normaliser is provably transparent**: identical quality metrics, 24% less level
   spread, and a test that asserts the gain is exactly invertible.
7. **Every number is reproducible** with one command, and every session carries its own
   config, log and PDF.
8. **We report the failures.** PESQ and SNR improvement miss on the aggregate; the
   per-SNR breakdown shows why the aggregate is the wrong statistic and where the system
   does deliver. Several defaults in this repo were changed *because* a measurement
   contradicted the initial guess — that is in the config comments.

---

## What not to claim

- Do not say the model was trained or fine-tuned. It is the pretrained DeepFilterNet3
  checkpoint, unmodified.
- Do not call the live path low-latency. It buffers ~1.5 s. The *model* is 40 ms.
- Do not quote a Jetson performance figure unless the report in front of you was
  generated on the Jetson. Check the session table's "Jetson hardware" row.
- Do not present the aggregate SNR improvement (+1.97 dB) as the headline without the
  per-SNR table. It is a ceiling artefact, and hiding that invites the question you
  least want.
- Do not say the chunked live path is numerically identical to the offline path. It
  reproduces it to about 17.5 dB SI-SDR, which is measured and stated.
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
