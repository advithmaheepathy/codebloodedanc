"""Is SI-SDR being charged for artefacts in a band that carries no signal?

The supplied corpus is natively 16 kHz. DeepFilterNet3 only runs at 48 kHz, so the
pipeline upsamples 16 -> 48 kHz, enhances, and scores at 48 kHz. That leaves the
8-24 kHz band containing nothing but upsampling residue in the clean reference, while
the model is free to synthesise or leave energy there. SI-SDR is a full-band ratio, so
any such energy is counted as error even though no real signal exists there.

A 16 kHz voice channel would band-limit on the way out regardless. This script measures
what that costs or gains, three ways:

  full band 48k      current behaviour
  low-passed 48k     output band-limited to the corpus's real 8 kHz bandwidth
  scored at 16k      both signals brought back to the corpus's native rate

If the numbers move materially, the current figure is a measurement artefact of the
resampling chain rather than a property of the enhancement.
"""

from __future__ import annotations

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import numpy as np

from anc_defence.audio.io import load_audio, resample
from anc_defence.config import load_config
from anc_defence.dataset.plain import load_plain_corpus, stratified_subset
from anc_defence.dsp.vad import speech_mask
from anc_defence.enhance.streaming import DfnModel
from anc_defence.metrics.erle import noise_reduction_db
from anc_defence.metrics.intrusive import compute_intrusive

SR = 48000
NATIVE = 16000


def lowpass(x: np.ndarray, sr: int, cutoff: int) -> np.ndarray:
    """Zero-phase band-limit by round-tripping through the lower rate."""
    down = resample(x, sr, cutoff * 2)
    return resample(down, cutoff * 2, sr)[: len(x)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-snr", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--max-snr", type=float, default=0.0,
                    help="only cells at or below this input SNR (the suppression regime)")
    args = ap.parse_args()

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)

    by_snr: dict[float, list] = {}
    for e in stratified_subset(corpus, 6, cfg.run.seed):
        if e.snr_db <= args.max_snr:
            by_snr.setdefault(e.snr_db, []).append(e)
    for snr in by_snr:
        by_snr[snr] = by_snr[snr][: args.per_snr]
    if not by_snr:
        print("no examples matched")
        return 1

    n_max = int(args.seconds * SR)
    model = DfnModel(cfg.neural, SR).load()

    variants = ("full band 48k", "low-passed to 8k", "scored at 16k")
    acc: dict[str, dict[str, list[float]]] = {
        v: {k: [] for k in ("snri", "pesq", "stoi", "nr")} for v in variants
    }

    hdr = f"{'input SNR':>10} {'variant':>18} {'SNRi dB':>9} {'PESQ':>7} {'STOI':>7} {'noise red dB':>13}"
    print(hdr)
    print("-" * len(hdr))

    for snr in sorted(by_snr):
        per: dict[str, dict[str, list[float]]] = {
            v: {k: [] for k in ("snri", "pesq", "stoi", "nr")} for v in variants
        }
        for e in by_snr[snr]:
            noisy = load_audio(e.noisy_path, SR)[:n_max]
            clean = load_audio(e.clean_path, SR)[:n_max]
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]
            out = model.enhance_array(noisy, count_time=False)[:n]
            silent = ~speech_mask(clean, cfg.vad, SR)

            cases = {
                "full band 48k": (clean, noisy, out, SR),
                "low-passed to 8k": (clean, noisy, lowpass(out, SR, 8000), SR),
                "scored at 16k": (
                    resample(clean, SR, NATIVE),
                    resample(noisy, SR, NATIVE),
                    resample(out, SR, NATIVE),
                    NATIVE,
                ),
            }
            for name, (c, x, y, sr) in cases.items():
                k = min(len(c), len(x), len(y))
                base = compute_intrusive(c[:k], x[:k], sr, cfg.metrics)
                got = compute_intrusive(c[:k], y[:k], sr, cfg.metrics)
                per[name]["snri"].append(got.si_sdr - base.si_sdr)
                per[name]["pesq"].append(got.pesq)
                per[name]["stoi"].append(got.stoi)
                mask = silent if sr == SR else ~speech_mask(c[:k], cfg.vad, sr)
                per[name]["nr"].append(noise_reduction_db(x[:k], y[:k], mask[:k]))

        for name in variants:
            m = {k: float(np.nanmean(v)) for k, v in per[name].items()}
            for k, v in per[name].items():
                acc[name][k].extend(v)
            print(f"{snr:>+9.0f}dB {name:>18} {m['snri']:>+9.2f} {m['pesq']:>7.3f} "
                  f"{m['stoi']:>7.3f} {m['nr']:>+13.2f}")
        print()

    print("-" * len(hdr))
    n_total = len(acc[variants[0]]["snri"])
    for name in variants:
        m = {k: float(np.nanmean(v)) for k, v in acc[name].items()}
        print(f"{'ALL':>9} {name:>18} {m['snri']:>+9.2f} {m['pesq']:>7.3f} "
              f"{m['stoi']:>7.3f} {m['nr']:>+13.2f}")
    print(f"\n{n_total} examples per variant, input SNR <= {args.max_snr:g} dB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
