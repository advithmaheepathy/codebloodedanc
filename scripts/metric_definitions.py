"""Why the reference app appears to score higher than this project, and what is comparable.

The reference implementation in ``model/deepfilter-anc-main`` reports exactly one
suppression figure, from ``server.py``::

    nr = max(0.0, min(100.0, (1.0 - output_rms / input_rms) * 100.0))

That is the percentage drop in **overall RMS across the whole chunk**, speech included,
clamped into 0-100. ``index.html`` renders it as a percentage. It never computes SNR -
it cannot, because a live microphone app has no clean-speech reference to compare
against, and SNR is by definition a ratio between two things you must be able to
separate.

This project reports two different and stricter figures:

``noise_reduction_db``   level drop measured **only in talker-silent regions**, so
                         removing speech cannot inflate it.
``snr_improvement_db``   SI-SDR of the output minus SI-SDR of the input, against the
                         known clean reference. Bounded by how much noise existed.

This script runs the reference app's own processing path and this project's delivered
path over the same corpus files and prints every definition side by side, so the numbers
can actually be compared instead of guessed at.
"""

from __future__ import annotations

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import numpy as np

from anc_defence.audio.io import load_audio
from anc_defence.config import load_config
from anc_defence.dataset.plain import load_plain_corpus, stratified_subset
from anc_defence.dsp.vad import speech_mask
from anc_defence.enhance.streaming import DfnModel
from anc_defence.metrics.erle import noise_reduction_db
from anc_defence.metrics.intrusive import compute_intrusive

SR = 48000
REF_TARGET_RMS = 0.08  # server.py TARGET_RMS
REF_MAX_GAIN = 10.0  # server.py MAX_GAIN


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def reference_nr_percent(before: np.ndarray, after: np.ndarray) -> float:
    """server.py's metric, verbatim."""
    input_rms, output_rms = rms(before), rms(after)
    if input_rms <= 0.001:
        return 0.0
    return max(0.0, min(100.0, (1.0 - output_rms / input_rms) * 100.0))


def reference_normalise(x: np.ndarray) -> np.ndarray:
    """server.py's volume normalisation, verbatim: fixed target RMS, hard clip."""
    r = rms(x)
    if r <= 1e-6:
        return x
    return np.clip(x * min(REF_TARGET_RMS / r, REF_MAX_GAIN), -1.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-snr", type=int, default=4, help="examples per input SNR")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)
    pool = stratified_subset(corpus, 6, cfg.run.seed)

    by_snr: dict[float, list] = {}
    for e in pool:
        by_snr.setdefault(e.snr_db, []).append(e)
    for snr in by_snr:
        by_snr[snr] = by_snr[snr][: args.per_snr]

    n_max = int(args.seconds * SR)
    model = DfnModel(cfg.neural.model_copy(update={"atten_lim_db": None,
                                                   "post_filter": False}), SR).load()

    print(
        "Reference app processing path (whole-chunk DeepFilterNet3, no attenuation cap,\n"
        "then fixed-target-RMS normalisation) measured under four metric definitions.\n"
    )
    hdr = (
        f"{'input SNR':>10} {'n':>3} | {'their NR%':>10} {'their NR dB':>12} | "
        f"{'our NR dB':>10} {'true SNRi dB':>13} | {'out level':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    totals: dict[str, list[float]] = {k: [] for k in
                                      ("pct", "refdb", "ourdb", "snri", "lvl")}

    for snr in sorted(by_snr):
        rows: dict[str, list[float]] = {k: [] for k in totals}
        examples = by_snr[snr]
        for e in examples:
            noisy = load_audio(e.noisy_path, SR)[:n_max]
            clean = load_audio(e.clean_path, SR)[:n_max]
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]

            enhanced = model.enhance_array(noisy, count_time=False)[:n]
            normalised = reference_normalise(enhanced)
            silent = ~speech_mask(clean, cfg.vad, SR)

            base = compute_intrusive(clean, noisy, SR, cfg.metrics)
            out = compute_intrusive(clean, enhanced, SR, cfg.metrics)

            rows["pct"].append(reference_nr_percent(noisy, enhanced))
            rows["refdb"].append(20.0 * np.log10(rms(noisy) / max(rms(enhanced), 1e-12)))
            rows["ourdb"].append(noise_reduction_db(noisy, enhanced, silent))
            rows["snri"].append(out.si_sdr - base.si_sdr)
            rows["lvl"].append(20.0 * np.log10(max(rms(normalised), 1e-12)))

        m = {k: float(np.nanmean(v)) for k, v in rows.items()}
        for k, v in rows.items():
            totals[k].extend(v)
        print(
            f"{snr:>+9.0f}dB {len(examples):>3} | {m['pct']:>9.1f}% {m['refdb']:>+11.2f} | "
            f"{m['ourdb']:>+9.2f} {m['snri']:>+12.2f} | {m['lvl']:>+8.1f}dB"
        )

    m = {k: float(np.nanmean(v)) for k, v in totals.items()}
    print("-" * len(hdr))
    print(
        f"{'all':>10} {len(totals['pct']):>3} | {m['pct']:>9.1f}% {m['refdb']:>+11.2f} | "
        f"{m['ourdb']:>+9.2f} {m['snri']:>+12.2f} | {m['lvl']:>+8.1f}dB"
    )

    print(
        "\ntheir NR%     server.py's formula: whole-chunk RMS drop, speech included, clamped 0-100.\n"
        "their NR dB   the same quantity expressed in dB instead of percent.\n"
        "our NR dB     level drop in talker-silent regions only (what this project reports).\n"
        "true SNRi     SI-SDR gain against the known clean reference - the only real SNR figure.\n"
        "out level     output loudness in dBFS after the reference app's normalisation,\n"
        "              i.e. what its dashboard level meter would display."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
