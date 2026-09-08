"""Evaluation dataset builder.

Produces WAV triplets (primary / reference / target) plus a manifest recording every
parameter that went into each mixture: source files, SNR, RIR index and properties,
impulsive event times and peaks, leakage, augmentation and the per-example seed.

Reproducibility: each example draws from a generator derived from
``(seed, subset, index)`` via :func:`anc_defence.utils.seeding.child_generator`, so a
rebuild with the same seed is bit-identical and the order of generation does not
matter.

Subsets
-------
``main``           all categories, SNR uniform over the configured range
``impulsive``      impulsive noise only, the hardest and most scrutinised case
``low_snr``        stress set at the bottom of the SNR range
``leakage_sweep``  speech deliberately leaked into the reference, -30 to -6 dB

Everything here is a *test* set. No training happens in this project, so there is no
train/test speaker leakage to guard against; the split-by-speaker machinery that a
training pipeline would need is described in README.md but not exercised.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..audio.io import iter_audio_files, load_audio, safe_peak_normalise, save_wav
from ..config import Config
from ..utils.logging import get_logger
from ..utils.seeding import child_generator
from .augment import augment_channel, draw_level_gain_db
from .mixer import (
    Mixture,
    NoiseComponent,
    active_rms,
    inject_impulsive_events,
    leak_speech_into_reference,
    prepare_noise,
    scale_for_snr,
)
from .rir import RirBank, convolve_rir, filter_length_check
from .sources import (
    CATEGORIES,
    category_of,
    index_noise_dir,
    load_speech_files,
    noise_type_of,
    speaker_of,
    write_source_manifest,
)

log = get_logger(__name__)

LEAKAGE_SWEEP_DB = (-30.0, -24.0, -18.0, -12.0, -6.0)


@dataclass
class BuildResult:
    out_dir: Path
    manifest_path: Path
    entries: list[dict[str, object]] = field(default_factory=list)
    rir_summary: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_minutes(self) -> float:
        return sum(float(e["duration_s"]) for e in self.entries) / 60.0

    def summary(self) -> dict[str, object]:
        by_subset: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for e in self.entries:
            subset = str(e["id"]).split("/")[0]
            by_subset[subset] = by_subset.get(subset, 0) + 1
            by_category[str(e["category"])] = by_category.get(str(e["category"]), 0) + 1
        return {
            "out_dir": str(self.out_dir),
            "n_examples": len(self.entries),
            "total_minutes": round(self.total_minutes, 2),
            "by_subset": by_subset,
            "by_category": by_category,
            "rir": self.rir_summary,
            "warnings": self.warnings,
        }


def _load_pool(paths: Sequence[Path], sample_rate: int, limit: Optional[int] = None) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for p in list(paths)[:limit] if limit else paths:
        try:
            pool.append(load_audio(p, target_sr=sample_rate, mono=True))
        except Exception as exc:
            log.warning("skipping unreadable file %s: %s", p, exc)
    return pool


def build_dataset(cfg: Config, out_dir: Optional[Path] = None) -> BuildResult:
    """Build the evaluation dataset described by ``cfg.dataset``."""
    dcfg = cfg.dataset
    sr = cfg.audio.sample_rate
    out = Path(out_dir or dcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    speech_files = load_speech_files(dcfg.clean_dir)
    if not speech_files:
        speech_files = iter_audio_files(dcfg.clean_dir)
    if not speech_files:
        raise RuntimeError(
            f"no clean speech found in {dcfg.clean_dir}. Run `anc fetch-data` first, or point "
            "dataset.clean_dir at a directory of WAV files organised as <speaker>/<utterance>.wav"
        )

    noise_index = index_noise_dir(dcfg.noise_dir)
    if not any(noise_index.values()):
        raise RuntimeError(
            f"no noise found under {dcfg.noise_dir}. Expected subdirectories named "
            f"{list(CATEGORIES)}. Run `anc fetch-data` to populate it."
        )
    for category in CATEGORIES:
        if not noise_index[category]:
            msg = f"no '{category}' noise available; that category will be absent from the results"
            log.warning(msg)
            warnings.append(msg)

    speakers = sorted({speaker_of(p) for p in speech_files})
    log.info(
        "sources: %d speech utterances from %d speakers, noise %s",
        len(speech_files),
        len(speakers),
        {k: len(v) for k, v in noise_index.items()},
    )

    rir_bank = RirBank(dcfg.rir, sample_rate=sr, seed=dcfg.seed, cache_dir=out / "rir_cache")
    if dcfg.rir.enabled:
        rir_bank.build()
        check = filter_length_check(
            cfg.nlms.filter_length, int(rir_bank.max_significant_ms() * sr / 1000.0), sr
        )
        if not check["adequate"]:
            msg = (
                f"NLMS filter length {check['filter_length_ms']} ms is shorter than the longest RIR "
                f"significant length {check['rir_significant_ms']} ms: the adaptive stage cannot "
                "model the whole path"
            )
            log.warning(msg)
            warnings.append(msg)
        else:
            log.info(
                "filter length %s ms covers the longest RIR (%s ms), coverage ratio %.1fx",
                check["filter_length_ms"],
                check["rir_significant_ms"],
                check["coverage_ratio"],
            )
    else:
        msg = (
            "dataset.rir.enabled is false: the reference will be identical to the noise added to "
            "the primary, which makes ERLE a tautology rather than a measurement"
        )
        log.warning(msg)
        warnings.append(msg)

    impulsive_pool = _load_pool(noise_index["impulsive"], sr)
    impulsive_names = [str(p) for p in noise_index["impulsive"]]

    # Split the total budget across subsets.
    per_subset = _subset_budget(dcfg.subsets, dcfg.total_minutes, dcfg.utterance_s)
    entries: list[dict[str, object]] = []

    for subset, n_examples in per_subset.items():
        subset_dir = out / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_examples):
            rng = child_generator(dcfg.seed, subset, i)
            mixture = _make_example(
                cfg=cfg,
                subset=subset,
                index=i,
                rng=rng,
                speech_files=speech_files,
                noise_index=noise_index,
                impulsive_pool=impulsive_pool,
                impulsive_names=impulsive_names,
                rir_bank=rir_bank if dcfg.rir.enabled else None,
            )
            if mixture is None:
                continue
            name = f"{subset}/{subset}_{i:04d}"
            paths = _write_mixture(subset_dir, f"{subset}_{i:04d}", mixture, dcfg.write_wav, sr)
            entries.append(mixture.manifest_entry(name, paths))

    manifest_path = out / "manifest.json"
    manifest = {
        "config": cfg.to_plain()["dataset"],
        "audio": cfg.to_plain()["audio"],
        "nlms_filter_length": cfg.nlms.filter_length,
        "seed": dcfg.seed,
        "rir": rir_bank.summary() if dcfg.rir.enabled else {"enabled": False},
        "speakers": speakers,
        "warnings": warnings,
        "n_examples": len(entries),
        "examples": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_source_manifest(out / "sources.json", speech_files, noise_index)

    result = BuildResult(
        out_dir=out,
        manifest_path=manifest_path,
        entries=entries,
        rir_summary=rir_bank.summary() if dcfg.rir.enabled else {"enabled": False},
        warnings=warnings,
    )
    log.info(
        "built %d examples (%.1f minutes) in %s",
        len(entries),
        result.total_minutes,
        out,
    )
    return result


def _subset_budget(subsets: Sequence[str], total_minutes: float, utterance_s: float) -> dict[str, int]:
    """Split the requested duration across subsets, weighted toward the main set."""
    weights = {"main": 0.5, "impulsive": 0.25, "low_snr": 0.15, "leakage_sweep": 0.10}
    active = [s for s in subsets if s in weights] or ["main"]
    total_weight = sum(weights[s] for s in active)
    total_examples = max(len(active), int(round(total_minutes * 60.0 / utterance_s)))
    out: dict[str, int] = {}
    for s in active:
        out[s] = max(1, int(round(total_examples * weights[s] / total_weight)))
    return out


def _make_example(
    cfg: Config,
    subset: str,
    index: int,
    rng: np.random.Generator,
    speech_files: Sequence[Path],
    noise_index: dict[str, list[Path]],
    impulsive_pool: list[np.ndarray],
    impulsive_names: list[str],
    rir_bank: Optional[RirBank],
) -> Optional[Mixture]:
    dcfg = cfg.dataset
    sr = cfg.audio.sample_rate
    length = int(dcfg.utterance_s * sr)

    speech_path = speech_files[int(rng.integers(0, len(speech_files)))]
    try:
        speech_full = load_audio(speech_path, target_sr=sr, mono=True)
    except Exception as exc:
        log.warning("skipping %s: %s", speech_path, exc)
        return None
    speech = _fit_speech(speech_full, length, rng)
    if active_rms(speech, sr) < 1e-4:
        return None
    # Normalise the talker to a consistent active level (-26 dBFS, the ITU-T P.56
    # convention for speech level in telephony tests).
    speech = (speech * (10.0 ** (-26.0 / 20.0) / (active_rms(speech, sr) + 1e-20))).astype(np.float32)

    # --- choose the noise category and sources ---------------------------------
    if subset == "impulsive":
        category = "impulsive"
    else:
        available = [c for c in CATEGORIES if noise_index[c]]
        if not available:
            return None
        category = str(rng.choice(available))

    snr_db = _pick_snr(subset, dcfg, rng)

    noise_at_primary = np.zeros(length, dtype=np.float32)
    reference = np.zeros(length, dtype=np.float32)
    components: list[NoiseComponent] = []
    rir_significant_ms = float("nan")

    n_sources = 1 if category == "impulsive" else int(rng.integers(1, dcfg.n_noise_sources_max + 1))
    pool = noise_index[category]
    for _ in range(n_sources):
        npath = pool[int(rng.integers(0, len(pool)))]
        try:
            raw = load_audio(npath, target_sr=sr, mono=True)
        except Exception as exc:
            log.warning("skipping noise %s: %s", npath, exc)
            continue
        dry = prepare_noise(raw, length, rng)
        if rir_bank is not None:
            h, info = rir_bank.pick(rng)
            wet = convolve_rir(dry, h)
            rir_index = info.index
            rir_significant_ms = max(
                info.significant_ms, rir_significant_ms if np.isfinite(rir_significant_ms) else 0.0
            )
        else:
            wet = dry.copy()
            rir_index = None
        noise_at_primary += wet
        reference += dry
        components.append(
            NoiseComponent(
                path=str(npath),
                category=category_of(npath),
                noise_type=noise_type_of(npath),
                rir_index=rir_index,
                gain=1.0,
            )
        )

    if not components:
        return None

    # --- impulsive events on top ----------------------------------------------
    impulsive_events = []
    if category == "impulsive" and impulsive_pool:
        track, impulsive_events = inject_impulsive_events(
            length, impulsive_pool, dcfg.impulsive, rng, sr, impulsive_names
        )
        if rir_bank is not None:
            h, info = rir_bank.pick(rng)
            wet_track = convolve_rir(track, h)
            rir_index = info.index
            rir_significant_ms = max(
                info.significant_ms, rir_significant_ms if np.isfinite(rir_significant_ms) else 0.0
            )
        else:
            wet_track = track.copy()
            rir_index = None
        noise_at_primary = noise_at_primary + wet_track
        reference = reference + track
        components.append(
            NoiseComponent(
                path="impulsive_event_track",
                category="impulsive",
                noise_type="impulsive_events",
                rir_index=rir_index,
                gain=1.0,
                is_impulsive_track=True,
            )
        )

    # --- set the SNR ----------------------------------------------------------
    scaled_noise, gain = scale_for_snr(speech, noise_at_primary, snr_db, sr)
    noise_at_primary = scaled_noise
    reference = (reference * gain).astype(np.float32)
    for c in components:
        c.gain = gain

    primary = (speech + noise_at_primary).astype(np.float32)

    # --- reference speech leakage --------------------------------------------
    leakage_db: Optional[float] = None
    if subset == "leakage_sweep":
        leakage_db = float(LEAKAGE_SWEEP_DB[index % len(LEAKAGE_SWEEP_DB)])
        reference = leak_speech_into_reference(reference, speech, leakage_db)

    # --- augmentation (capture chain) ----------------------------------------
    # The capture-chain level gain is applied to the primary, the target, the noise
    # and the reference together. Applying it to the mixture alone would leave the
    # target at a different level and silently bias every scale-sensitive metric.
    level_db = draw_level_gain_db(dcfg.augment, rng)
    if level_db != 0.0:
        g = 10.0 ** (level_db / 20.0)
        primary = (primary * g).astype(np.float32)
        speech = (speech * g).astype(np.float32)
        noise_at_primary = (noise_at_primary * g).astype(np.float32)
        reference = (reference * g).astype(np.float32)

    # Tilt, sensor noise and clipping are real distortions and apply to the captured
    # primary only: the target stays the clean signal we want to recover.
    primary_aug, rec_primary = augment_channel(
        primary, dcfg.augment, rng, sr, apply_level=False
    )
    rec_primary.level_gain_db = level_db
    if level_db != 0.0:
        rec_primary.applied.insert(0, "level")
    # The reference gets its own sensor noise but no independent level, tilt or
    # clipping, so the two channels stay comparable.
    reference_aug, rec_reference = augment_channel(
        reference,
        dcfg.augment.model_copy(update={"clipping_prob": 0.0, "spectral_tilt_prob": 0.0}),
        rng,
        sr,
        apply_level=False,
    )

    primary_aug, pk_gain = safe_peak_normalise(primary_aug)
    if pk_gain != 1.0:
        # Keep the target and reference aligned with the gain applied to the primary.
        speech = (speech * pk_gain).astype(np.float32)
        noise_at_primary = (noise_at_primary * pk_gain).astype(np.float32)
        reference_aug = (reference_aug * pk_gain).astype(np.float32)

    return Mixture(
        primary=primary_aug,
        reference=reference_aug,
        target=speech,
        noise_at_primary=noise_at_primary,
        sample_rate=sr,
        snr_db=snr_db,
        speech_path=str(speech_path),
        speaker=speaker_of(speech_path),
        category=category,
        components=components,
        impulsive_events=impulsive_events,
        leakage_db=leakage_db,
        rir_significant_ms=round(rir_significant_ms, 2) if np.isfinite(rir_significant_ms) else float("nan"),
        seed=dcfg.seed,
        augment={"primary": rec_primary.as_dict(), "reference": rec_reference.as_dict()},
    )


def _pick_snr(subset: str, dcfg, rng: np.random.Generator) -> float:
    if subset == "low_snr":
        return float(rng.uniform(dcfg.snr_db_min, min(dcfg.snr_db_min + 5.0, dcfg.snr_db_max)))
    return float(rng.uniform(dcfg.snr_db_min, dcfg.snr_db_max))


def _fit_speech(speech: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Crop or pad speech to the target length, keeping a natural leading pause."""
    if len(speech) >= length:
        start = int(rng.integers(0, len(speech) - length + 1))
        return speech[start : start + length].copy()
    out = np.zeros(length, dtype=np.float32)
    offset = int(rng.integers(0, max(1, length - len(speech))))
    out[offset : offset + len(speech)] = speech
    return out


def _write_mixture(
    subset_dir: Path, stem: str, mixture: Mixture, write_wav: bool, sr: int
) -> dict[str, str]:
    paths: dict[str, str] = {}
    if not write_wav:
        return paths
    for kind, data in (
        ("primary", mixture.primary),
        ("reference", mixture.reference),
        ("target", mixture.target),
        ("noise", mixture.noise_at_primary),
    ):
        p = subset_dir / f"{stem}_{kind}.wav"
        save_wav(p, data, sr)
        paths[kind] = str(p)
    return paths


def load_manifest(path: Path | str) -> dict[str, object]:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
