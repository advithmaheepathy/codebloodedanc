"""Configuration schema, loading, merging and validation.

Every tunable parameter in the system lives here. Configuration is loaded from one
or more YAML files (later files override earlier ones) and can be overridden from
the command line with dotted ``--set key.path=value`` arguments.

The fully resolved configuration is printed at startup and written into the session
directory so any run can be reproduced exactly.
"""

from __future__ import annotations

import copy
from pathlib import Path
# typing.List/Tuple rather than builtin generics: these annotations sit inside pydantic
# models, and pydantic evaluates them at runtime to build the schema. `from __future__
# import annotations` defers evaluation for ordinary functions but not for pydantic
# fields, so `list[str]` raises "'type' object is not subscriptable" on Python 3.8 --
# which is what JetPack 5.1.3 ships (Ubuntu 20.04, Python 3.8.10).
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class _Base(BaseModel):
    """Base model: unknown keys are an error so typos never pass silently."""

    model_config = ConfigDict(
        extra="forbid", validate_assignment=True, protected_namespaces=()
    )


# --------------------------------------------------------------------------- run


class RunCfg(_Base):
    seed: int = 1234
    session_root: Path = Path("sessions")
    name: Optional[str] = Field(default=None, description="Optional session name suffix.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    deterministic_torch: bool = True


# ------------------------------------------------------------------------- audio


class AudioCfg(_Base):
    """Core signal geometry. DeepFilterNet3 is a 48 kHz / 10 ms hop model, so
    these defaults are fixed by the neural stage and should not be changed."""

    sample_rate: int = 48000
    hop_size: int = 480
    fft_size: int = 960

    @model_validator(mode="after")
    def _check(self) -> "AudioCfg":
        if self.fft_size % self.hop_size != 0:
            raise ValueError("fft_size must be an integer multiple of hop_size")
        return self

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop_size / self.sample_rate


# -------------------------------------------------------------------- preprocess


class HighpassCfg(_Base):
    enabled: bool = True
    cutoff_hz: float = Field(default=80.0, ge=1.0, le=2000.0)
    order: int = Field(default=2, ge=1, le=8)


class PreprocessCfg(_Base):
    dc_removal: bool = True
    dc_cutoff_hz: float = Field(default=5.0, gt=0.0)
    highpass: HighpassCfg = HighpassCfg()
    guard_nonfinite: bool = True
    denormal_floor: float = 1e-20
    reference_gain_db: float = Field(
        default=0.0, description="Static gain applied to the reference channel after calibration."
    )


# ---------------------------------------------------------------------------- vad


class VadCfg(_Base):
    enabled: bool = True
    method: Literal["energy_flatness"] = "energy_flatness"
    frame_ms: float = 10.0
    # Speech is declared when frame energy exceeds the adaptive noise floor by this
    # margin AND the spectrum is not flat (flatness below the threshold).
    energy_margin_db: float = 8.0
    flatness_threshold: float = 0.45
    noise_floor_smoothing: float = Field(default=0.995, gt=0.0, lt=1.0)
    # 300 ms measured: with mu=0.1 and noise leaking into the reference, a 300 ms
    # hangover reduced residual speech distortion 5x versus 120 ms, because the
    # pauses inside a word would otherwise let the filter misadapt.
    hangover_ms: float = 300.0
    min_speech_ms: float = 30.0


# --------------------------------------------------------------------------- nlms


class DoubleTalkCfg(_Base):
    """Speech in the reference makes plain NLMS cancel *speech*. Freezing (or
    heavily reducing) adaptation while the talker is active is the single most
    important safeguard in the adaptive stage."""

    mode: Literal["freeze", "scale", "off"] = "freeze"
    mu_scale: float = Field(default=0.05, ge=0.0, le=1.0)


class ImpulseGuardCfg(_Base):
    """A gunshot in the reference produces an enormous gradient. Either clip the
    update or freeze adaptation through the transient."""

    mode: Literal["off", "clip", "freeze"] = "clip"
    crest_factor_db: float = Field(
        default=12.0, description="Frame crest factor above which a frame is called impulsive."
    )
    update_clip_sigma: float = Field(
        default=4.0, description="Clip the per-sample update to this many times its running RMS."
    )


class NlmsSafeguardCfg(_Base):
    reference_silence_dbfs: float = Field(
        default=-65.0, description="Skip adaptation when reference frame level is below this."
    )
    divergence_margin_db: float = Field(
        default=6.0, description="Roll back if output power exceeds input power by this margin."
    )
    weight_norm_limit: float = 1.0e4
    rollback: bool = True
    checkpoint_interval_frames: int = Field(default=25, ge=1)


class NlmsCfg(_Base):
    enabled: bool = True
    impl: Literal["fdaf", "time"] = "fdaf"
    # 7680 taps = 160 ms at 48 kHz. Chosen by measurement: the simulated RIRs have a
    # significant length (to -40 dB) of 69-155 ms, and cancellation of a real
    # convolved path improves monotonically with coverage:
    #   1920 taps ( 40 ms) ->  8.3 dB ERLE, FDAF RTF 0.034
    #   3840 taps ( 80 ms) -> 18.1 dB ERLE, FDAF RTF 0.053
    #   5760 taps (120 ms) -> 29.5 dB ERLE, FDAF RTF 0.141
    #   7680 taps (160 ms) -> 42.3 dB ERLE, FDAF RTF 0.184
    filter_length: int = Field(default=7680, ge=128, le=16384)
    mu: float = Field(default=0.1, gt=0.0, le=1.0)
    eps: float = Field(default=1e-6, gt=0.0)
    leakage: float = Field(default=0.0, ge=0.0, lt=1.0)
    power_smoothing: float = Field(default=0.9, ge=0.0, lt=1.0)
    double_talk: DoubleTalkCfg = DoubleTalkCfg()
    impulse_guard: ImpulseGuardCfg = ImpulseGuardCfg()
    safeguards: NlmsSafeguardCfg = NlmsSafeguardCfg()


# ------------------------------------------------------------------------- neural


class StreamingCfg(_Base):
    """``chunked`` is the low-risk default: overlapping chunks through the offline
    enhancer with a crossfade at the joins. Its latency is one chunk, which is
    fine for a demo but must never be described as low-latency streaming.

    ``per_hop`` is true frame-by-frame inference with persistent model state. It is
    only used if it agrees numerically with the offline path.
    """

    mode: Literal["chunked", "per_hop"] = "chunked"
    chunk_s: float = Field(default=1.5, gt=0.005, le=10.0)
    overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    crossfade_ms: float = Field(default=30.0, ge=0.0)
    # 0.25 s measured. Preceding audio is prepended to each chunk, processed, then
    # discarded, so the model's recurrent state is warm for the samples that are kept.
    # Agreement with the reference whole-file path, as SI-SDR, 1 CPU thread:
    #   context  chunk 1.0 s   chunk 1.5 s
    #   0.00 s      11.8 dB       10.1 dB     <- cold state at every boundary
    #   0.25 s      17.5 dB       17.4 dB     <- default; nearly all of the benefit
    #   0.50 s      17.3 dB       17.3 dB
    #   1.00 s      17.2 dB       18.5 dB
    context_s: float = Field(
        default=0.25,
        ge=0.0,
        le=5.0,
        description="Preceding audio prepended to each chunk, processed and then discarded, so the "
        "model's recurrent state is warm before the samples that are kept. Costs compute in "
        "proportion to context/hop.",
    )
    offline_framing: Literal["whole_file", "chunked"] = Field(
        default="whole_file",
        description=(
            "Offline mode processes whole files by default (the reference inference path). "
            "Set to 'chunked' to reproduce exactly what the live path produces."
        ),
    )


class NeuralCfg(_Base):
    enabled: bool = True
    model: str = "DeepFilterNet3"
    model_base_dir: Optional[Path] = Field(
        default=None, description="Local checkpoint directory. None uses the packaged pretrained model."
    )
    backend: Literal["torch"] = "torch"
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    num_threads: Optional[int] = Field(
        default=None,
        description="Torch intra-op thread count. Set to 1 for the single-thread edge-readiness measurement.",
    )
    # No cap by default. A 30 dB cap was tried first because it improved the
    # corpus-wide PESQ/STOI aggregate, but measuring the thing a listener actually
    # notices - level drop in talker-silent regions - showed what it costs:
    #
    #   whole file, no cap    noise reduction +35.9 dB   PESQ 1.364  STOI 0.688
    #   whole file, cap 30    noise reduction +26.2 dB   PESQ 1.357  STOI 0.697
    #
    # ~10 dB less noise removed for +0.009 STOI. At low SNR, where noise actually
    # matters, the cap buys almost nothing; it only protects speech in near-clean audio
    # (at +15 dB input it lifted PESQ 2.874 -> 3.032). Removing noise is the product
    # goal, so no cap is the default; set 20-30 if the input is usually already clean.
    atten_lim_db: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Cap suppression depth; None means no limit."
    )
    post_filter: bool = False
    warmup_frames: int = Field(default=10, ge=0)
    streaming: StreamingCfg = StreamingCfg()


# ----------------------------------------------------------------------- pipeline


class NormaliseCfg(_Base):
    """Volume normalisation, the second and final pipeline stage.

    ``agc`` is a speech-aware automatic gain control: it tracks the active speech
    level, moves the gain toward the target smoothly, and holds the gain during
    pauses so residual noise is not pumped up. A look-ahead soft limiter prevents
    clipping and is the only stage after the model that adds latency.
    """

    mode: Literal["agc", "peak", "rms", "off"] = "agc"
    target_dbfs: float = Field(
        default=-26.0,
        description="Target active speech level. -26 dBFS is the ITU-T P.56 convention "
        "for speech level in telephony tests and is close to -23 LUFS for speech.",
    )
    min_gain_db: float = Field(default=-12.0, description="Most the AGC may attenuate.")
    max_gain_db: float = Field(
        default=24.0,
        description="Most the AGC may amplify. Caps how loud a near-silent frame can be made.",
    )
    attack_ms: float = Field(default=150.0, gt=0.0, description="Gain rise time constant.")
    release_ms: float = Field(default=600.0, gt=0.0, description="Gain fall time constant.")
    level_attack_ms: float = Field(default=50.0, gt=0.0)
    level_release_ms: float = Field(default=800.0, gt=0.0)
    hold_during_pause: bool = Field(
        default=True,
        description="Freeze the gain when the frame is neither speech nor loud. Turning this off "
        "makes the AGC chase the noise floor in pauses and audibly pump the residual noise.",
    )
    gate_dbfs: float = Field(
        default=-60.0, description="Frames below this level never update the speech-level estimate."
    )
    active_range_db: float = Field(
        default=20.0,
        description="A frame within this many dB of the running peak counts as active even if the "
        "VAD does not flag it. Without this fallback a recording with no pauses gets no "
        "normalisation at all, because the VAD's noise-floor tracker adapts up to a "
        "continuously active signal and then never reports speech.",
    )
    peak_decay_db_per_s: float = Field(
        default=1.0,
        description="Decay rate of the running peak used by the active-frame test. This has to be "
        "slow: at 6 dB/s the reference is lost within a second, after which a genuine pause drifts "
        "back inside the active window and the gain starts chasing the noise floor again. 1 dB/s "
        "keeps the reference stable across a normal utterance.",
    )
    limiter_ceiling_dbfs: float = Field(default=-1.0, le=0.0)
    limiter_lookahead_ms: float = Field(default=5.0, ge=0.0)
    limiter_release_ms: float = Field(default=120.0, gt=0.0)


PipelineOrder = Literal[
    # Current design: single microphone, neural suppression then volume normalisation.
    "dfn_then_normalise",
    "dfn_only",
    "normalise_only",
    "passthrough",
    # Retained only so the rejected two-microphone designs can still be measured and
    # reported. Not part of the delivered pipeline. See README.md.
    "nlms_then_dfn",
    "dfn_then_nlms",
    "nlms_only",
]


class PipelineCfg(_Base):
    order: PipelineOrder = "dfn_then_normalise"


# ------------------------------------------------------------------------ metrics


class MetricsCfg(_Base):
    pesq: bool = True
    stoi: bool = True
    estoi: bool = True
    sisdr: bool = True
    segsnr: bool = True
    lsd: bool = True
    snr: bool = True
    erle: bool = True
    events: bool = True
    dnsmos: bool = False
    pesq_mode: Literal["wb", "nb"] = "wb"
    segsnr_frame_ms: float = 20.0
    level_align: bool = Field(
        default=True,
        description="Scale the signal under test onto the clean reference before computing the "
        "level-sensitive metrics (segmental SNR, LSD, direct SNR). Without this the volume "
        "normalisation stage would appear to change quality when it has only changed gain. "
        "PESQ, STOI and SI-SDR are unaffected either way.",
    )
    snr_buckets: List[Tuple[float, float]] = [
        (-10.0, -5.0),
        (-5.0, 0.0),
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
    ]
    headline_snr_range: Tuple[float, float] = (-5.0, 10.0)
    suppression_snr_range: Tuple[float, float] = Field(
        default=(-10.0, 0.0),
        description="Input-SNR range over which the SNR-improvement target is judged. "
        "SNR improvement is bounded above by how much noise is present: at +15 dB input "
        "there is almost nothing left to remove, so every suppressor scores negative "
        "there and the full-range average becomes a statement about the corpus rather "
        "than about the system. The full-range figure is still reported alongside.",
    )


# ------------------------------------------------------------------------ dataset


class RirCfg(_Base):
    """Room impulse responses for the *noise* path.

    The evaluation mixtures convolve noise with an RIR before adding it to the
    primary channel, while the NLMS reference is the dry noise. NLMS therefore has
    to identify a real acoustic path instead of subtracting an identical copy.
    """

    enabled: bool = True
    source: Literal["simulated"] = "simulated"
    room_dim_min: Tuple[float, float, float] = (3.0, 3.0, 2.4)
    room_dim_max: Tuple[float, float, float] = (8.0, 7.0, 3.2)
    rt60_min_s: float = Field(default=0.15, gt=0.0)
    rt60_max_s: float = Field(default=0.35, gt=0.0)
    max_order: int = Field(default=10, ge=1)
    n_rirs: int = Field(default=12, ge=1)
    truncate_db: float = Field(
        default=-40.0, description="Report the RIR length down to this level below the peak."
    )
    speech_rir: bool = Field(
        default=False,
        description="Also convolve speech with a separate RIR (target becomes reverberant clean).",
    )


class ImpulsiveCfg(_Base):
    events_per_minute_min: float = 4.0
    events_per_minute_max: float = 20.0
    peak_level_dbfs_min: float = -18.0
    peak_level_dbfs_max: float = -3.0
    preserve_crest_factor: bool = True
    min_gap_s: float = 0.25


class AugmentCfg(_Base):
    clipping_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    clipping_threshold_min: float = Field(default=0.5, gt=0.0, le=1.0)
    clipping_threshold_max: float = Field(default=0.95, gt=0.0, le=1.0)
    level_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    level_range_db: Tuple[float, float] = (-12.0, 0.0)
    spectral_tilt_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    spectral_tilt_db_per_khz: Tuple[float, float] = (-1.0, 1.0)
    mic_self_noise_dbfs: Optional[float] = -70.0


class DatasetCfg(_Base):
    out_dir: Path = Path("data/eval")
    clean_dir: Path = Path("data/raw/speech")
    noise_dir: Path = Path("data/raw/noise")
    seed: int = 4242
    total_minutes: float = Field(default=10.0, gt=0.0)
    utterance_s: float = Field(default=8.0, gt=1.0)
    snr_db_min: float = -5.0
    snr_db_max: float = 20.0
    n_noise_sources_max: int = Field(default=2, ge=1, le=3)
    write_wav: bool = True
    write_hdf5: bool = False
    subsets: List[str] = ["main", "impulsive", "low_snr"]
    rir: RirCfg = RirCfg()
    impulsive: ImpulsiveCfg = ImpulsiveCfg()
    augment: AugmentCfg = AugmentCfg()


# ------------------------------------------------------------------------- report


class TargetsCfg(_Base):
    """Mandated performance targets from the problem statement.

    The problem statement lists 'SNR > 15 dB, STOI > 0.85, PESQ > 2.5'. STOI and PESQ
    there are absolute output figures, and by parallel construction 'SNR > 15 dB' is the
    absolute *output* SNR of the enhanced speech - speech power over residual (noise plus
    distortion) - not the SI-SDR improvement over the noisy input. Those are different
    quantities: the improvement is bounded by how much noise was present, whereas the
    output SNR is the standalone quality of the result. The check evaluates output SNR
    (:data:`snr_db`); the SI-SDR improvement is still computed and reported as a secondary
    figure but is no longer the pass/fail criterion.
    """

    snr_db: float = 15.0                 # absolute OUTPUT SNR target
    snr_improvement_db: float = 15.0     # retained for the secondary improvement column
    stoi: float = 0.85
    pesq: float = 2.5


class ReportCfg(_Base):
    write_pdf: bool = True
    write_json: bool = True
    write_csv: bool = True
    write_wav: bool = True
    spectrograms: bool = True
    targets: TargetsCfg = TargetsCfg()
    dpi: int = Field(default=110, ge=50, le=300)


# -------------------------------------------------------------------------- modes


class PlainDatasetCfg(_Base):
    """The supplied single-channel corpus: ``dataset_plain`` with a metadata CSV.

    16 kHz mono WAV triplets (clean / noisy / noise-only) with the noise category and
    the mixing SNR recorded per example. Everything is upsampled to 48 kHz on load
    because the neural model is a 48 kHz model.
    """

    root: Path = Path("dataset_plain")
    metadata: Path = Path("dataset_plain/metadata.csv")
    native_sample_rate: int = 16000
    # Stratified sampling for the fast evaluation: this many examples per
    # (category, SNR) cell, drawn with the run seed so the subset is reproducible.
    per_cell: int = Field(default=6, ge=1)
    categories: Optional[List[str]] = Field(
        default=None, description="None uses every category present in the metadata."
    )
    snr_values: Optional[List[float]] = Field(
        default=None, description="None uses every SNR present in the metadata."
    )
    max_duration_s: float = Field(
        default=12.0,
        gt=0.0,
        description="Long files are cropped for the evaluation so one outlier cannot dominate.",
    )


class OfflineCfg(_Base):
    primary: Optional[Path] = None
    reference: Optional[Path] = None
    clean: Optional[Path] = None
    manifest: Optional[Path] = Field(
        default=None, description="Dataset manifest to evaluate in bulk instead of a single file."
    )
    limit: Optional[int] = Field(default=None, ge=1, description="Evaluate only the first N manifest items.")
    workers: int = Field(
        default=0,
        ge=0,
        description="Parallel worker processes for batch evaluation. 0 picks a sensible default "
        "from the core count; 1 forces serial execution.",
    )


class LiveCfg(_Base):
    duration_s: float = Field(default=30.0, gt=0.0)
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    monitor: bool = True
    monitor_gain_db: float = -6.0
    blocksize: int = 480
    ring_capacity_s: float = Field(default=8.0, gt=0.5)
    # Latency profile for the live neural framing.
    # Measured on 12 corpus examples at <= 0 dB SNR. "noise red." is the level drop in
    # talker-silent regions; "sp.atten" is how much quieter the speech became.
    #
    #   chunk    noise red.  sp.atten   latency    notes
    #    60 ms    +33.9 dB    12.4 dB     75 ms    speech visibly damaged: too little context
    #   250 ms    +37.5 dB    10.6 dB    265 ms    low_latency
    #   500 ms    +37.5 dB     9.6 dB    515 ms    balanced
    #     1 s     +37.5 dB     8.1 dB   1015 ms    quality
    #     1 s +postfilter +44.7 dB 8.6 dB 1015 ms  max_suppression
    #
    # 60 ms was the original low_latency setting and it was a bad trade: it removed less
    # noise AND damaged the speech more than 250 ms, for 190 ms less delay.
    latency_profile: Literal[
        "low_latency", "balanced", "quality", "max_suppression", "custom"
    ] = "low_latency"
    noise_file: Optional[Path] = None
    noise_snr_db: float = 0.0
    noise_rir: bool = Field(
        default=True, description="Convolve injected noise with an RIR; the dry noise is the reference."
    )
    calibration_s: float = Field(default=2.0, ge=0.0)
    save_streams: bool = True


# --------------------------------------------------------------------------- root


class Config(_Base):
    run: RunCfg = RunCfg()
    audio: AudioCfg = AudioCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    vad: VadCfg = VadCfg()
    nlms: NlmsCfg = NlmsCfg()
    neural: NeuralCfg = NeuralCfg()
    normalise: NormaliseCfg = NormaliseCfg()
    pipeline: PipelineCfg = PipelineCfg()
    metrics: MetricsCfg = MetricsCfg()
    dataset: DatasetCfg = DatasetCfg()
    plain: PlainDatasetCfg = PlainDatasetCfg()
    report: ReportCfg = ReportCfg()
    offline: OfflineCfg = OfflineCfg()
    live: LiveCfg = LiveCfg()

    @field_validator("nlms")
    @classmethod
    def _warn_time_impl(cls, v: NlmsCfg) -> NlmsCfg:
        return v

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_plain(), sort_keys=False, default_flow_style=False)

    def to_plain(self) -> dict[str, Any]:
        """JSON/YAML-safe dict (Paths and tuples flattened)."""
        return _plain(self.model_dump(mode="json"))


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ------------------------------------------------------------------------ loading


class ConfigError(RuntimeError):
    """Raised with an actionable message when configuration is invalid."""


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value`` into (['a','b','c'], parsed_value)."""
    if "=" not in item:
        raise ConfigError(f"override '{item}' is not of the form key.path=value")
    key, raw = item.split("=", 1)
    path = [p for p in key.strip().split(".") if p]
    if not path:
        raise ConfigError(f"override '{item}' has an empty key path")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return path, value


def apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for item in overrides:
        path, value = parse_override(item)
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override '{item}' descends into a non-mapping value")
        node[path[-1]] = value
    return out


def load_config(
    paths: Optional[Sequence[Path | str]] = None,
    overrides: Optional[Sequence[str]] = None,
) -> Config:
    """Load, merge and validate configuration.

    Args:
        paths: YAML files applied in order; later files win.
        overrides: dotted ``key.path=value`` strings applied last.

    Raises:
        ConfigError: with a readable summary of what failed validation.
    """
    data: dict[str, Any] = {}
    for path in paths or []:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {p} must contain a YAML mapping at the top level")
        data = deep_merge(data, loaded)
    if overrides:
        data = apply_overrides(data, overrides)
    try:
        return Config(**data)
    except ValidationError as exc:
        lines = [f"invalid configuration ({exc.error_count()} problem(s)):"]
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            lines.append(f"  {loc or '<root>'}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from exc


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
