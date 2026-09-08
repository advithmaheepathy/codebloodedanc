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
from typing import Any, Literal, Optional, Sequence

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
    filter_length: int = Field(default=3840, ge=128, le=8192)
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
    chunk_s: float = Field(default=1.5, gt=0.1, le=10.0)
    overlap: float = Field(default=0.5, ge=0.0, lt=1.0)
    crossfade_ms: float = Field(default=30.0, ge=0.0)


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
    atten_lim_db: Optional[float] = Field(
        default=None, ge=0.0, le=100.0, description="Cap suppression depth; None means no limit."
    )
    post_filter: bool = False
    warmup_frames: int = Field(default=10, ge=0)
    streaming: StreamingCfg = StreamingCfg()


# ----------------------------------------------------------------------- pipeline


PipelineOrder = Literal["nlms_then_dfn", "dfn_then_nlms", "nlms_only", "dfn_only", "passthrough"]


class PipelineCfg(_Base):
    order: PipelineOrder = "nlms_then_dfn"


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
    snr_buckets: list[tuple[float, float]] = [
        (-10.0, -5.0),
        (-5.0, 0.0),
        (0.0, 5.0),
        (5.0, 10.0),
        (10.0, 15.0),
        (15.0, 20.0),
    ]
    headline_snr_range: tuple[float, float] = (-5.0, 10.0)


# ------------------------------------------------------------------------ dataset


class RirCfg(_Base):
    """Room impulse responses for the *noise* path.

    The evaluation mixtures convolve noise with an RIR before adding it to the
    primary channel, while the NLMS reference is the dry noise. NLMS therefore has
    to identify a real acoustic path instead of subtracting an identical copy.
    """

    enabled: bool = True
    source: Literal["simulated"] = "simulated"
    room_dim_min: tuple[float, float, float] = (3.0, 3.0, 2.4)
    room_dim_max: tuple[float, float, float] = (8.0, 7.0, 3.2)
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
    level_range_db: tuple[float, float] = (-12.0, 0.0)
    spectral_tilt_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    spectral_tilt_db_per_khz: tuple[float, float] = (-1.0, 1.0)
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
    subsets: list[str] = ["main", "impulsive", "low_snr"]
    rir: RirCfg = RirCfg()
    impulsive: ImpulsiveCfg = ImpulsiveCfg()
    augment: AugmentCfg = AugmentCfg()


# ------------------------------------------------------------------------- report


class TargetsCfg(_Base):
    """Mandated performance targets from the problem statement."""

    snr_improvement_db: float = 15.0
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


class OfflineCfg(_Base):
    primary: Optional[Path] = None
    reference: Optional[Path] = None
    clean: Optional[Path] = None
    manifest: Optional[Path] = Field(
        default=None, description="Dataset manifest to evaluate in bulk instead of a single file."
    )
    limit: Optional[int] = Field(default=None, ge=1, description="Evaluate only the first N manifest items.")


class LiveCfg(_Base):
    duration_s: float = Field(default=30.0, gt=0.0)
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    monitor: bool = True
    monitor_gain_db: float = -6.0
    blocksize: int = 480
    ring_capacity_s: float = Field(default=8.0, gt=0.5)
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
    pipeline: PipelineCfg = PipelineCfg()
    metrics: MetricsCfg = MetricsCfg()
    dataset: DatasetCfg = DatasetCfg()
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
