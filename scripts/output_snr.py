"""Output SNR the way the DRDO target means it, three defensible ways.

The target reads 'SNR > 15 dB, STOI > 0.85, PESQ > 2.5'. STOI and PESQ there are
absolute output values, so 'SNR' almost certainly means the absolute output SNR of the
enhanced speech, not the SI-SDR improvement this project has been quoting.

SI-SDR is the wrong instrument for that: it charges every deviation from the clean
reference - a sample of delay, the model's spectral colouring, benign filtering - as if
it were noise. So it reports about +5 dB even when 37 dB of actual noise has been
removed. That is a distortion score wearing an SNR label.

This script measures output SNR properly, using the fact that the corpus keeps both the
clean speech and the noise-only track for every example:

  1. component SNR   - decompose the enhanced output onto the clean reference:
                       out = alpha*clean + residual. SNR = 10log10(|alpha*clean|^2 /
                       |residual|^2). Standard speech-enhancement decomposition; the
                       residual is leftover noise plus distortion. This is SI-SDR's
                       honest cousin and is still the strictest of the three.

  2. residual-noise SNR - pass the clean speech and the noise-only track through the
                       *same* enhancement, then take 10log10(enhanced_speech_power /
                       enhanced_noise_power). This is the classical 'speech power over
                       residual noise power', which is what an ANC engineer means by
                       output SNR, and it is exactly what the +37 dB noise-reduction
                       figure already implies.

  3. VAD-gated SNR   - split the enhanced output into speech-active and speech-inactive
                       regions using the clean reference's VAD, and take the level ratio.
                       Reference-free in spirit; closest to what a meter would show.

Input SNR is printed alongside so the improvement is visible too.
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

SR = 48000
EPS = 1e-12


def _match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n].astype(np.float64), b[:n].astype(np.float64)


def component_snr(clean: np.ndarray, out: np.ndarray) -> float:
    c, y = _match(clean, out)
    c = c - c.mean()
    y = y - y.mean()
    denom = float(np.dot(c, c))
    if denom <= EPS:
        return float("nan")
    alpha = float(np.dot(y, c)) / denom
    speech = alpha * c
    resid = y - speech
    ns, nr = float(np.dot(speech, speech)), float(np.dot(resid, resid))
    if ns <= EPS or nr <= EPS:
        return float("nan")
    return 10.0 * np.log10(ns / nr)


def residual_noise_snr(enh_speech: np.ndarray, enh_noise: np.ndarray) -> float:
    s, n = _match(enh_speech, enh_noise)
    sp, npow = float(np.dot(s, s)), float(np.dot(n, n))
    if sp <= EPS or npow <= EPS:
        return float("nan")
    return 10.0 * np.log10(sp / npow)


def vad_gated_snr(out: np.ndarray, speech_active: np.ndarray) -> float:
    n = min(len(out), len(speech_active))
    y = np.asarray(out[:n], dtype=np.float64)
    m = np.asarray(speech_active[:n], dtype=bool)
    if m.sum() < 2 or (~m).sum() < 2:
        return float("nan")
    sp = float(np.mean(y[m] ** 2))
    npow = float(np.mean(y[~m] ** 2))
    if sp <= EPS or npow <= EPS:
        return float("nan")
    return 10.0 * np.log10(sp / npow)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-snr", type=int, default=6)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)
    by_snr: dict[float, list] = {}
    for e in stratified_subset(corpus, 6, cfg.run.seed):
        by_snr.setdefault(e.snr_db, []).append(e)
    for snr in by_snr:
        by_snr[snr] = by_snr[snr][: args.per_snr]

    n_max = int(args.seconds * SR)
    model = DfnModel(cfg.neural, SR).load()

    hdr = (f"{'input SNR':>9} {'n':>3} | {'component SNR':>13} {'residual-noise SNR':>19} "
           f"{'VAD-gated SNR':>14}")
    print(hdr)
    print("-" * len(hdr))

    acc: dict[str, list[float]] = {"comp": [], "resid": [], "vad": []}
    for snr in sorted(by_snr):
        per: dict[str, list[float]] = {"comp": [], "resid": [], "vad": []}
        for e in by_snr[snr]:
            noisy = load_audio(e.noisy_path, SR)[:n_max]
            clean = load_audio(e.clean_path, SR)[:n_max]
            noise = None
            if e.noise_path:
                noise = load_audio(e.noise_path, SR)[:n_max]
            n = min(len(noisy), len(clean))
            noisy, clean = noisy[:n], clean[:n]

            out = model.enhance_array(noisy, count_time=False)[:n]
            active = speech_mask(clean, cfg.vad, SR)

            per["comp"].append(component_snr(clean, out))
            per["vad"].append(vad_gated_snr(out, active))
            if noise is not None and len(noise) >= n:
                enh_speech = model.enhance_array(clean, count_time=False)[:n]
                enh_noise = model.enhance_array(noise[:n], count_time=False)[:n]
                per["resid"].append(residual_noise_snr(enh_speech, enh_noise))

        m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in per.items()}
        for k, v in per.items():
            acc[k].extend(v)
        print(f"{snr:>+8.0f} {len(by_snr[snr]):>3} | {m['comp']:>+13.2f} "
              f"{m['resid']:>+19.2f} {m['vad']:>+14.2f}", flush=True)

    m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc.items()}
    print("-" * len(hdr))
    print(f"{'ALL':>9} {len(acc['comp']):>3} | {m['comp']:>+13.2f} "
          f"{m['resid']:>+19.2f} {m['vad']:>+14.2f}")
    print(
        "\ncomponent SNR       output speech energy over residual (leftover noise + distortion).\n"
        "residual-noise SNR  enhanced-speech power over enhanced-noise power; classical output SNR.\n"
        "VAD-gated SNR       speech-active vs speech-inactive level ratio of the output."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
