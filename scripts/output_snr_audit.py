"""Audit the output-SNR measurement. The residual-noise figure was measured wrongly.

THE ERROR
---------
``residual_noise_snr_db`` enhanced the clean track and the noise-only track separately
and took their power ratio::

    enh_speech = enhance(clean)       # model sees pure speech -> mask ~ 1, passes it
    enh_noise  = enhance(noise_only)  # model sees pure noise, no speech -> mask ~ 0

That assumes ``enhance(s + n) = enhance(s) + enhance(n)``, i.e. that the enhancer is
linear. DeepFilterNet3 is strongly non-linear: its mask is a function of the input. Fed
noise with no speech anywhere in it, the model's own voice-activity reasoning concludes
the whole signal is noise and suppresses it maximally. That is not the residual noise
that survives when speech is present to mask it. The symptom: subtracting the input SNR
from those figures leaves a near-constant ~45 dB offset, i.e. the metric was measuring
suppression depth plus input SNR, not output SNR.

THE CORRECT DECOMPOSITION
-------------------------
For a non-linear enhancer, recover the *effective time-frequency gain it actually applied
to the mixture*, then apply that same gain to each component::

    G(t,f) = |Y(t,f)| / |X(t,f)|          from the real mixture X and real output Y
    speech component = G * S              S = clean spectrum
    noise  component = G * N              N = noise spectrum

Because the corpus mixture is exactly ``X = S + N``, and G is applied identically to
both, the two components sum to the output. Their power ratio is the genuine output SNR.
This is the standard filter-based decomposition used for SDR/SIR/SAR.

Reported alongside: the input SNR (sanity check against the corpus label) and the
resulting SNR improvement, which is the honest headline for a suppressor.
"""

from __future__ import annotations

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import numpy as np
from scipy.signal import stft

from anc_defence.audio.io import load_audio
from anc_defence.config import load_config
from anc_defence.dataset.plain import load_plain_corpus, stratified_subset
from anc_defence.dsp.vad import speech_mask
from anc_defence.enhance.streaming import DfnModel

SR = 48000
NPERSEG = 960
NOVERLAP = 480
EPS = 1e-10
GAIN_CEIL = 4.0  # the model can add gain; cap the estimate so division noise cannot explode


def spec(x: np.ndarray) -> np.ndarray:
    _, _, Z = stft(x, fs=SR, nperseg=NPERSEG, noverlap=NOVERLAP, boundary=None, padded=False)
    return Z


def decompose_output_snr(
    clean: np.ndarray, noise: np.ndarray, noisy: np.ndarray, out: np.ndarray,
    active_frames: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return (output_snr_db, input_snr_db) via the mixture-derived gain."""
    n = min(len(clean), len(noise), len(noisy), len(out))
    S, N, X, Y = (spec(v[:n]) for v in (clean, noise, noisy, out))
    k = min(S.shape[1], N.shape[1], X.shape[1], Y.shape[1])
    S, N, X, Y = S[:, :k], N[:, :k], X[:, :k], Y[:, :k]

    # Effective gain the enhancer applied to the mixture.
    G = np.abs(Y) / (np.abs(X) + EPS)
    G = np.clip(G, 0.0, GAIN_CEIL)

    sel = slice(None)
    if active_frames is not None:
        m = np.asarray(active_frames[:k], dtype=bool)
        if m.sum() >= 2:
            S, N, G = S[:, m], N[:, m], G[:, m]

    sp_out = float(np.sum((G * np.abs(S)) ** 2))
    ns_out = float(np.sum((G * np.abs(N)) ** 2))
    sp_in = float(np.sum(np.abs(S) ** 2))
    ns_in = float(np.sum(np.abs(N) ** 2))
    out_snr = 10.0 * np.log10((sp_out + EPS) / (ns_out + EPS))
    in_snr = 10.0 * np.log10((sp_in + EPS) / (ns_in + EPS))
    return out_snr, in_snr


def wrong_way(model: DfnModel, clean: np.ndarray, noise: np.ndarray) -> float:
    """The committed (incorrect) figure, for direct comparison."""
    k = min(len(clean), len(noise))
    s = model.enhance_array(clean[:k], count_time=False)
    nz = model.enhance_array(noise[:k], count_time=False)
    m = min(len(s), len(nz))
    sp = float(np.dot(s[:m].astype(np.float64), s[:m].astype(np.float64)))
    np_ = float(np.dot(nz[:m].astype(np.float64), nz[:m].astype(np.float64)))
    return 10.0 * np.log10((sp + EPS) / (np_ + EPS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-snr", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--check-additivity", action="store_true",
                    help="verify the corpus mixture really is clean + noise")
    args = ap.parse_args()

    cfg = load_config(["configs/default.yaml"])
    corpus = load_plain_corpus(cfg.plain)
    by_snr: dict[float, list] = {}
    for e in stratified_subset(corpus, 6, cfg.run.seed):
        if e.noise_path is not None:
            by_snr.setdefault(e.snr_db, []).append(e)
    for s in by_snr:
        by_snr[s] = by_snr[s][: args.per_snr]

    n_max = int(args.seconds * SR)
    model = DfnModel(cfg.neural, SR).load()

    hdr = (f"{'label':>6} {'measured in':>12} | {'CORRECT out SNR':>16} {'SNR gain':>9} "
           f"| {'active-only out':>16} | {'WRONG (committed)':>18}")
    print(hdr)
    print("-" * len(hdr))

    acc = {k: [] for k in ("out", "gain", "act", "wrong", "insnr", "resid")}
    for snr in sorted(by_snr):
        per = {k: [] for k in acc}
        for e in by_snr[snr]:
            noisy = load_audio(e.noisy_path, SR)[:n_max]
            clean = load_audio(e.clean_path, SR)[:n_max]
            noise = load_audio(e.noise_path, SR)[:n_max]
            n = min(len(noisy), len(clean), len(noise))
            noisy, clean, noise = noisy[:n], clean[:n], noise[:n]

            if args.check_additivity:
                resid = float(np.sqrt(np.mean((noisy - clean - noise) ** 2)))
                ref = float(np.sqrt(np.mean(noisy**2))) + EPS
                per["resid"].append(20.0 * np.log10(resid / ref + 1e-20))

            out = model.enhance_array(noisy, count_time=False)[:n]

            # frame-level speech activity from the clean reference
            active = speech_mask(clean, cfg.vad, SR)
            nf = 1 + max(0, (n - NPERSEG)) // NOVERLAP
            frames = np.zeros(nf, dtype=bool)
            for i in range(nf):
                seg = active[i * NOVERLAP : i * NOVERLAP + NPERSEG]
                frames[i] = bool(seg.mean() > 0.5) if seg.size else False

            o, i_snr = decompose_output_snr(clean, noise, noisy, out)
            a, _ = decompose_output_snr(clean, noise, noisy, out, active_frames=frames)
            per["out"].append(o)
            per["insnr"].append(i_snr)
            per["gain"].append(o - i_snr)
            per["act"].append(a)
            per["wrong"].append(wrong_way(model, clean, noise))

        m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in per.items()}
        for k, v in per.items():
            acc[k].extend(v)
        print(f"{snr:>+5.0f} {m['insnr']:>12.2f} | {m['out']:>16.2f} {m['gain']:>+9.2f} "
              f"| {m['act']:>16.2f} | {m['wrong']:>18.2f}", flush=True)

    m = {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in acc.items()}
    print("-" * len(hdr))
    print(f"{'ALL':>6} {m['insnr']:>12.2f} | {m['out']:>16.2f} {m['gain']:>+9.2f} "
          f"| {m['act']:>16.2f} | {m['wrong']:>18.2f}")

    if args.check_additivity and acc["resid"]:
        print(f"\nmixture additivity: ||noisy - clean - noise|| is "
              f"{float(np.nanmean(acc['resid'])):.1f} dB below the mixture "
              f"(very negative = the corpus really is clean + noise)")

    print(
        "\nCORRECT out SNR   speech vs noise after applying the mixture-derived gain to each.\n"
        "SNR gain          correct output SNR minus measured input SNR.\n"
        "active-only       same, restricted to speech-active frames.\n"
        "WRONG (committed) enhance(clean) vs enhance(noise) - invalid, the enhancer is non-linear."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
