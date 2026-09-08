"""Measure how much noise is actually removed, against the three levers that control it.

Motivated by a direct observation: a reference implementation in ``model/`` appeared to
cancel more noise than this project's live path. It uses the *same* pretrained
DeepFilterNet3, so the difference has to be in how the model is called. Three candidate
causes, all measurable:

1. ``atten_lim_db`` - a cap on suppression depth. This project set 30 dB (chosen to
   maximise PESQ/STOI); the reference sets no cap at all.
2. Chunk size - the reference processes 3 s chunks, this project's low-latency live
   profile uses 60 ms. More context means better suppression.
3. ``post_filter`` - DeepFilterNet's optional post-filter, which over-attenuates noisy
   sections slightly.

The headline number here is ``noise_reduction_db``: the level drop measured **only in
talker-silent regions**, which is what a listener calls "noise cancellation". PESQ, STOI
and speech attenuation are reported alongside, because the whole point is to find
settings that remove more noise *without* eating the speech.
"""

from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.INFO)

import numpy as np

from anc_defence.audio.io import load_audio
from anc_defence.config import load_config
from anc_defence.dataset.plain import load_plain_corpus, stratified_subset
from anc_defence.dsp.vad import speech_mask
from anc_defence.enhance.streaming import ChunkedEnhancer, DfnModel
from anc_defence.metrics.erle import noise_reduction_db
from anc_defence.metrics.intrusive import compute_intrusive

SR = 48000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", type=int, default=12, help="examples to average over")
    ap.add_argument("--max-snr", type=float, default=0.0, help="only use examples at or below this SNR")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)
    pool = [e for e in stratified_subset(corpus, 6, cfg.run.seed) if e.snr_db <= args.max_snr]
    pool = pool[: args.files]
    if not pool:
        print("no examples matched")
        return 1

    n_max = int(args.seconds * SR)
    examples = []
    for e in pool:
        noisy = load_audio(e.noisy_path, SR)[:n_max]
        clean = load_audio(e.clean_path, SR)[:n_max]
        n = min(len(noisy), len(clean))
        examples.append((noisy[:n], clean[:n], e.category, e.snr_db))
    print(f"{len(examples)} examples at SNR <= {args.max_snr:g} dB, {args.seconds:g} s each\n")

    # Pre-compute the noise-only masks and the unprocessed baseline once.
    masks = [~speech_mask(c, cfg.vad, SR) for _, c, _, _ in examples]
    base = [compute_intrusive(c, x, SR, cfg.metrics) for x, c, _, _ in examples]

    configs = [
        # (label, atten_lim_db, post_filter, chunk_s or None for whole-file)
        ("reference app equivalent (3 s, no cap)", None, False, 3.0),
        ("whole file, no cap", None, False, None),
        ("whole file, cap 30 dB (our current)", 30.0, False, None),
        ("whole file, no cap + post-filter", None, True, None),
        ("live 60 ms, cap 30 dB (our current live)", 30.0, False, 0.06),
        ("live 60 ms, no cap", None, False, 0.06),
        ("live 60 ms, no cap + post-filter", None, True, 0.06),
        ("live 250 ms, no cap", None, False, 0.25),
        ("live 500 ms, no cap", None, False, 0.5),
        ("live 1 s, no cap", None, False, 1.0),
        ("live 1 s, no cap + post-filter", None, True, 1.0),
    ]

    print(
        "%-42s %-11s %-8s %-8s %-9s %-9s"
        % ("configuration", "noise red.", "PESQ", "STOI", "SI-SDR", "sp.atten")
    )
    print("-" * 96)

    cache: dict[tuple[float | None, bool], DfnModel] = {}
    for label, atten, pf, chunk in configs:
        key = (atten, pf)
        if key not in cache:
            neural = cfg.neural.model_copy(deep=True)
            neural.atten_lim_db = atten
            neural.post_filter = pf
            cache[key] = DfnModel(neural, SR).load()
        model = cache[key]

        nr, pesq, stoi, sisdr, atten_db = [], [], [], [], []
        for i, (noisy, clean, _, _) in enumerate(examples):
            if chunk is None:
                out = model.enhance_array(noisy, count_time=False)
            else:
                enh = ChunkedEnhancer(
                    model=model, chunk_s=chunk, overlap=0.0, crossfade_ms=5.0,
                    sample_rate=SR, context_s=0.25,
                )
                out = enh.process_signal(noisy)
            n = min(len(out), len(noisy))
            m = compute_intrusive(clean[:n], out[:n], SR, cfg.metrics)
            nr.append(noise_reduction_db(noisy[:n], out[:n], masks[i][:n]))
            pesq.append(m.pesq)
            stoi.append(m.stoi)
            sisdr.append(m.si_sdr)
            atten_db.append(m.speech_attenuation_db)

        f = lambda v: float(np.nanmean(v))  # noqa: E731
        print(
            "%-42s %+9.2f   %-8.3f %-8.3f %+8.2f  %+8.2f"
            % (label, f(nr), f(pesq), f(stoi), f(sisdr), f(atten_db))
        )

    print(
        "\nnoise red. = level drop in talker-silent regions (higher = more noise removed).\n"
        "sp.atten   = how much quieter the speech became (lower is better).\n"
        "The goal is the highest noise reduction that does not inflate sp.atten or\n"
        "collapse PESQ/STOI."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
