"""Find the attenuation cap that maximises SNR improvement, not raw noise removal.

Context. With no attenuation cap the pipeline removes about 35 dB of noise in
talker-silent regions, yet SNR improvement sits near +10 dB. Those two facts together
say the residual noise is already negligible and the remaining gap to the clean
reference is almost entirely *speech distortion* introduced by the suppressor.

35 dB of noise reduction is far past audibility - once the noise is 20 dB down it is
gone as far as a listener is concerned. So suppression depth beyond that point buys
nothing perceptually while continuing to cost speech fidelity. DeepFilterNet3's
``atten_lim_db`` caps how deep the mask may cut, which is exactly the knob that trades
one for the other.

This sweep measures the trade in the regime the system is built for (input SNR <= 0 dB),
reporting SNR improvement, PESQ, STOI, how much noise is still removed, and how much the
speech was attenuated. The goal is the cap that maximises SNR improvement while keeping
noise reduction comfortably above audibility.
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


def _parse_caps(spec: str) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for part in spec.split(","):
        part = part.strip()
        if part.lower() in ("none", "null", ""):
            out.append(None)
        else:
            out.append(float(part))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-snr", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--max-snr", type=float, default=0.0)
    ap.add_argument("--caps", default="none,20,12",
                    help="comma-separated attenuation caps in dB; 'none' for uncapped")
    ap.add_argument("--post-filter", action="store_true")
    args = ap.parse_args()
    caps = _parse_caps(args.caps)

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)

    by_snr: dict[float, list] = {}
    for e in stratified_subset(corpus, 6, cfg.run.seed):
        if e.snr_db <= args.max_snr:
            by_snr.setdefault(e.snr_db, []).append(e)
    for snr in by_snr:
        by_snr[snr] = by_snr[snr][: args.per_snr]

    n_max = int(args.seconds * SR)
    examples = []
    for snr in sorted(by_snr):
        for e in by_snr[snr]:
            noisy = load_audio(e.noisy_path, SR)[:n_max]
            clean = load_audio(e.clean_path, SR)[:n_max]
            n = min(len(noisy), len(clean))
            examples.append((noisy[:n], clean[:n], snr))
    print(f"{len(examples)} examples at input SNR <= {args.max_snr:g} dB, "
          f"{args.seconds:g} s each, post_filter={args.post_filter}\n")

    masks = [~speech_mask(c, cfg.vad, SR) for _, c, _ in examples]
    bases = [compute_intrusive(c, x, SR, cfg.metrics) for x, c, _ in examples]

    hdr = (f"{'atten cap':>11} {'SNRi dB':>9} {'PESQ':>7} {'STOI':>7} "
           f"{'noise red dB':>13} {'sp.atten dB':>12}")
    print(hdr)
    print("-" * len(hdr))

    best = None
    for cap in caps:
        neural = cfg.neural.model_copy(deep=True)
        neural.atten_lim_db = cap
        neural.post_filter = args.post_filter
        model = DfnModel(neural, SR).load()

        snri, pesq, stoi, nr, atten = [], [], [], [], []
        for i, (noisy, clean, _) in enumerate(examples):
            out = model.enhance_array(noisy, count_time=False)
            n = min(len(out), len(noisy))
            m = compute_intrusive(clean[:n], out[:n], SR, cfg.metrics)
            snri.append(m.si_sdr - bases[i].si_sdr)
            pesq.append(m.pesq)
            stoi.append(m.stoi)
            nr.append(noise_reduction_db(noisy[:n], out[:n], masks[i][:n]))
            atten.append(m.speech_attenuation_db)

        f = {k: float(np.nanmean(v)) for k, v in
             (("snri", snri), ("pesq", pesq), ("stoi", stoi), ("nr", nr), ("atten", atten))}
        label = "none" if cap is None else f"{cap:g} dB"
        print(f"{label:>11} {f['snri']:>+9.2f} {f['pesq']:>7.3f} {f['stoi']:>7.3f} "
              f"{f['nr']:>+13.2f} {f['atten']:>+12.2f}", flush=True)
        if best is None or f["snri"] > best[1]["snri"]:
            best = (label, f)

    print("-" * len(hdr))
    if best:
        print(f"\nbest SNR improvement: {best[0]} -> {best[1]['snri']:+.2f} dB "
              f"(PESQ {best[1]['pesq']:.3f}, STOI {best[1]['stoi']:.3f}, "
              f"noise still down {best[1]['nr']:+.2f} dB)")
    print(
        "\nNoise reduction above roughly 20 dB is inaudible as a difference, so a cap that\n"
        "keeps it there while raising SNR improvement and PESQ is a strict win, not a\n"
        "trade. A cap that pushes noise reduction below about 15 dB is not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
