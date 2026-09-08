"""Command-line interface.

Every command takes ``-c/--config`` (repeatable, later files win) and ``--set
key.path=value`` (repeatable) so any parameter in the schema can be overridden
without editing a file. The fully resolved configuration is printed at startup and
saved into the session directory.

    anc list-devices
    anc fetch-data
    anc build-dataset
    anc dataset-stats
    anc run --mode offline_file --primary noisy.wav --reference noise.wav --clean clean.wav
    anc run --mode live_dfn_only --duration 30
    anc run --mode live_injected_noise --noise-file noise.wav --snr 0
    anc evaluate --manifest data/eval/manifest.json --methods all
    anc benchmark
    anc calibrate --primary a.wav --reference b.wav
    anc report --session sessions/<dir>
    anc licenses
    anc selftest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from ..config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config
from ..utils.logging import get_logger, setup_logging

log = get_logger(__name__)

MODES = ("offline_file", "live_mic")


# ------------------------------------------------------------------ arg parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anc",
        description=(
            "Hybrid adaptive-filter + neural noise cancellation for defence voice comms "
            "(SIH PS 26052). The neural stage is pretrained DeepFilterNet3, unmodified; no "
            "training is performed by this tool."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", action="append", default=None, metavar="FILE",
        help="YAML config file; repeatable, later files override earlier ones "
             f"(default: {DEFAULT_CONFIG_PATH} when present)",
    )
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
        help="override any config value, e.g. --set nlms.mu=0.05 --set pipeline.order=dfn_then_nlms",
    )
    parser.add_argument("--seed", type=int, default=None, help="override run.seed")
    parser.add_argument(
        "--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="override run.log_level",
    )
    # The same global options are attached to every subcommand under different dest
    # names and merged later, so both orders work:
    #     anc --set nlms.mu=0.05 evaluate
    #     anc evaluate --set nlms.mu=0.05
    # Sharing dest names would let the subparser's default (None) clobber a value the
    # parent already parsed.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", action="append", default=None, dest="config_post",
                        metavar="FILE", help=argparse.SUPPRESS)
    common.add_argument("--set", action="append", default=None, dest="overrides_post",
                        metavar="KEY=VALUE",
                        help="override any config value, e.g. --set normalise.target_dbfs=-23")
    common.add_argument("--seed", type=int, default=None, dest="seed_post", help=argparse.SUPPRESS)
    common.add_argument("--log-level", default=None, dest="log_level_post",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="command", required=True, parser_class=argparse.ArgumentParser)

    def add_parser(name: str, **kwargs: Any) -> argparse.ArgumentParser:
        kwargs.setdefault("parents", [common])
        return sub.add_parser(name, **kwargs)

    add_parser("list-devices", help="list audio devices and exit")
    add_parser("licenses", help="regenerate LICENSES.md from the corpus registry")

    p_fetch = add_parser("fetch-data", help="download the small evaluation corpora")
    p_fetch.add_argument("--per-class", type=int, default=6, help="ESC-50 clips per class")
    p_fetch.add_argument("--utterances", type=int, default=40, help="clean speech utterances")
    p_fetch.add_argument("--synthetic-only", action="store_true",
                         help="skip downloads and generate procedural noise only")
    p_fetch.add_argument("--dnsmos", action="store_true", help="also fetch the DNSMOS ONNX model")

    p_build = add_parser("build-dataset", help="build the synthetic evaluation dataset")
    p_build.add_argument("--minutes", type=float, default=None, help="override dataset.total_minutes")
    p_build.add_argument("--out", default=None, help="override dataset.out_dir")
    p_build.add_argument("--smoke", action="store_true", help="tiny dataset for a fast end-to-end check")

    p_stats = add_parser("dataset-stats", help="write and print dataset statistics")
    p_stats.add_argument("--manifest", default=None, help="dataset directory or manifest.json")

    p_run = add_parser("run", help="run one of the operating modes")
    p_run.add_argument("--mode", required=True, choices=MODES)
    p_run.add_argument("--primary", default=None, help="offline: noisy input WAV")
    p_run.add_argument("--reference", default=None, help="offline: noise reference (rejected designs only)")
    p_run.add_argument("--clean", default=None, help="offline: clean target WAV (enables PESQ/STOI)")
    p_run.add_argument("--manifest", default=None, help="offline: evaluate a generated manifest")
    p_run.add_argument("--limit", type=int, default=None, help="offline: only the first N examples")
    p_run.add_argument("--order", default=None, help="pipeline order override")
    p_run.add_argument("--duration", type=float, default=None, help="live: seconds to run")
    p_run.add_argument("--noise-file", default=None,
                       help="live: optional noise file mixed in digitally as a demo aid")
    p_run.add_argument("--snr", type=float, default=None, help="live: injected noise SNR in dB")
    p_run.add_argument("--input-device", default=None, help="live: input device index or name substring")
    p_run.add_argument("--output-device", default=None, help="live: output device index or name substring")
    p_run.add_argument("--no-monitor", action="store_true", help="live: do not play the output")

    p_eval = add_parser("evaluate", help="run the method comparison over the corpus")
    p_eval.add_argument("--manifest", default=None,
                        help="a generated manifest; omit to use the supplied dataset_plain corpus")
    p_eval.add_argument("--methods", default="delivered",
                        help="'delivered' (default), 'all' (adds the rejected designs), "
                             "or a comma-separated list")
    p_eval.add_argument("--limit", type=int, default=None, help="only the first N examples")
    p_eval.add_argument("--per-cell", type=int, default=None,
                        help="examples per (category, SNR) cell in the balanced subset")
    p_eval.add_argument("--workers", type=int, default=None,
                        help="worker processes (0 = auto from core count, 1 = serial)")

    add_parser("corpus", help="summarise the supplied dataset_plain corpus")

    p_dash = add_parser("dashboard", help="launch the Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)
    p_dash.add_argument("--headless", action="store_true", help="do not open a browser")
    p_dash.add_argument(
        "--host", default="127.0.0.1",
        help="bind address. Defaults to localhost only. The dashboard has no "
             "authentication, so use 0.0.0.0 only on a network you trust.",
    )

    p_bench = add_parser("benchmark", help="measure RTF and chunked-vs-offline agreement")
    p_bench.add_argument("--seconds", type=float, default=10.0, help="audio duration per trial")
    p_bench.add_argument("--trials", type=int, default=3)
    p_bench.add_argument("--chunk-sizes", default="0.5,1.0,1.5,2.0")
    p_bench.add_argument("--threads", default="1,0", help="torch thread counts; 0 means default")
    p_bench.add_argument("--contexts", default=None,
                         help="comma-separated warm-up context lengths in seconds to compare")
    p_bench.add_argument("--input", default=None, help="use this WAV instead of synthetic noise")

    p_cal = add_parser("calibrate", help="suggest a reference-channel gain from noise-only audio")
    p_cal.add_argument("--primary", required=True)
    p_cal.add_argument("--reference", required=True)

    p_report = add_parser("report", help="print the summary of an existing session")
    p_report.add_argument("--session", default=None, help="session directory (default: most recent)")

    add_parser("selftest", help="check the environment and run a tiny end-to-end pipeline")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    """Merge global options given before and after the subcommand."""
    config_files = list(getattr(args, "config", None) or []) + list(
        getattr(args, "config_post", None) or []
    )
    paths: list[Path] = [Path(c) for c in config_files]
    if not paths and DEFAULT_CONFIG_PATH.is_file():
        paths = [DEFAULT_CONFIG_PATH]

    overrides = list(getattr(args, "overrides", None) or []) + list(
        getattr(args, "overrides_post", None) or []
    )
    seed = getattr(args, "seed_post", None) or getattr(args, "seed", None)
    if seed is not None:
        overrides.append(f"run.seed={seed}")
    level = getattr(args, "log_level_post", None) or getattr(args, "log_level", None)
    if level is not None:
        overrides.append(f"run.log_level={level}")
    overrides.extend(_command_overrides(args))
    return load_config(paths, overrides)


def _command_overrides(args: argparse.Namespace) -> list[str]:
    """Map convenience flags onto config paths so there is one source of truth."""
    out: list[str] = []
    cmd = getattr(args, "command", "")

    def add(key: str, value: Any) -> None:
        if value is not None:
            out.append(f"{key}={value}")

    if cmd == "run":
        add("offline.primary", getattr(args, "primary", None))
        add("offline.reference", getattr(args, "reference", None))
        add("offline.clean", getattr(args, "clean", None))
        add("offline.manifest", getattr(args, "manifest", None))
        add("offline.limit", getattr(args, "limit", None))
        add("pipeline.order", getattr(args, "order", None))
        add("live.duration_s", getattr(args, "duration", None))
        add("live.noise_file", getattr(args, "noise_file", None))
        add("live.noise_snr_db", getattr(args, "snr", None))
        add("live.input_device", getattr(args, "input_device", None))
        add("live.output_device", getattr(args, "output_device", None))
        if getattr(args, "no_monitor", False):
            out.append("live.monitor=false")
        if getattr(args, "mode", "") == "live_dfn_only":
            out.append("pipeline.order=dfn_only")
    elif cmd == "evaluate":
        add("offline.manifest", getattr(args, "manifest", None))
        add("offline.limit", getattr(args, "limit", None))
        add("offline.workers", getattr(args, "workers", None))
        add("plain.per_cell", getattr(args, "per_cell", None))
    elif cmd == "build-dataset":
        add("dataset.total_minutes", getattr(args, "minutes", None))
        add("dataset.out_dir", getattr(args, "out", None))
        if getattr(args, "smoke", False):
            out += ["dataset.total_minutes=0.7", "dataset.utterance_s=5.0", "dataset.rir.n_rirs=3"]
    elif cmd == "dataset-stats":
        add("offline.manifest", getattr(args, "manifest", None))
    return out


# --------------------------------------------------------------------- commands


def cmd_list_devices(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.devices import DeviceError, format_device_table

    try:
        print(format_device_table())
    except DeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_licenses(cfg: Config, args: argparse.Namespace) -> int:
    from ..dataset.sources import write_licenses_md

    path = write_licenses_md()
    print(f"wrote {path}")
    return 0


def cmd_fetch_data(cfg: Config, args: argparse.Namespace) -> int:
    from ..dataset import sources

    noise_dir = cfg.dataset.noise_dir
    if not args.synthetic_only:
        got = sources.fetch_esc50_subset(noise_dir, per_class=args.per_class)
        total = sum(len(v) for v in got.values())
        print(f"ESC-50: {total} clips -> {noise_dir}")
        if total == 0:
            print("no clips downloaded; generating procedural noise instead")
            args.synthetic_only = True
        speech = sources.fetch_librispeech_subset(cfg.dataset.clean_dir, n_utterances=args.utterances)
        print(f"clean speech: {len(speech)} utterances -> {cfg.dataset.clean_dir}")
        if not speech:
            print(
                "WARNING: no clean speech available. PESQ/STOI need real speech; place WAV files at "
                f"{cfg.dataset.clean_dir}/<speaker>/<utterance>.wav and re-run.",
                file=sys.stderr,
            )
    if args.synthetic_only:
        got = sources.synthesise_fallback_corpus(noise_dir)
        print(f"synthetic noise: {sum(len(v) for v in got.values())} files -> {noise_dir}")
    if args.dnsmos:
        from ..metrics.nonintrusive import fetch_dnsmos_models

        files = fetch_dnsmos_models()
        print(f"DNSMOS models: {len(files)} file(s)")
    sources.write_licenses_md()
    print("wrote LICENSES.md")
    return 0


def cmd_build_dataset(cfg: Config, args: argparse.Namespace) -> int:
    from ..dataset.build import build_dataset
    from ..dataset.stats import write_stats

    result = build_dataset(cfg)
    stats_path, stats = write_stats(result.manifest_path)
    print(f"\nbuilt {len(result.entries)} examples ({result.total_minutes:.2f} min) in {result.out_dir}")
    print(f"manifest: {result.manifest_path}")
    print(f"stats:    {stats_path}")
    for key, value in result.summary().items():
        if key not in ("warnings", "rir"):
            print(f"  {key}: {value}")
    if result.warnings:
        print("\nwarnings:")
        for w in result.warnings:
            print(f"  - {w}")
    return 0


def cmd_dataset_stats(cfg: Config, args: argparse.Namespace) -> int:
    from ..dataset.stats import format_stats_table, write_stats

    target = cfg.offline.manifest or cfg.dataset.out_dir
    path, stats = write_stats(target)
    print(f"wrote {path}\n")
    for key, value in format_stats_table(stats):
        print(f"  {key:32s} {value}")
    return 0


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    if args.mode == "offline_file":
        from ..evaluate import DELIVERED_METHODS
        from ..modes.offline_file import run_offline

        written = run_offline(cfg, methods=DELIVERED_METHODS, limit=cfg.offline.limit)
    elif args.mode == "live_mic":
        from ..modes.live import run_live

        written = run_live(cfg)
    else:  # pragma: no cover - argparse restricts this
        raise ValueError(args.mode)
    _print_artifacts(written)
    return 0


def cmd_evaluate(cfg: Config, args: argparse.Namespace) -> int:
    from ..evaluate import ALL_METHODS, DELIVERED_METHODS
    from ..modes.offline_file import run_offline

    spec = args.methods.strip().lower()
    if spec == "all":
        methods = list(ALL_METHODS)
    elif spec in ("delivered", "default"):
        methods = list(DELIVERED_METHODS)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    written = run_offline(
        cfg, methods=methods, limit=cfg.offline.limit,
        title="Evaluation: method comparison and ablations",
        use_corpus=cfg.offline.manifest is None,
    )
    _print_artifacts(written)
    return 0


def cmd_corpus(cfg: Config, args: argparse.Namespace) -> int:
    from ..dataset.plain import corpus_stats_rows, load_plain_corpus, stratified_subset

    corpus = load_plain_corpus(cfg.plain)
    subset = stratified_subset(
        corpus, cfg.plain.per_cell, cfg.run.seed, cfg.plain.categories, cfg.plain.snr_values
    )
    print()
    for key, value in corpus_stats_rows(corpus, subset):
        print(f"  {key:48s} {value}")
    print()
    if corpus.missing:
        print(f"  {len(corpus.missing)} metadata row(s) reference audio that is not on disk")
    return 0


def cmd_dashboard(cfg: Config, args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard."""
    import subprocess

    app = Path(__file__).with_name("app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--server.headless", "true" if args.headless else "false",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"starting the dashboard on http://{args.host}:{args.port}")
    if args.host not in ("127.0.0.1", "localhost"):
        print(
            f"WARNING: binding to {args.host} exposes the dashboard on the network and it has no\n"
            "         authentication. Anyone who can reach this port can browse the sessions and\n"
            "         use the microphone tab. Only do this on a network you trust."
        )
    print("(Ctrl-C to stop)\n")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "error: streamlit is not installed. Install it with:\n"
            "  pip install streamlit==1.39.0",
            file=sys.stderr,
        )
        return 1


def cmd_benchmark(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.io import load_audio
    from ..enhance.streaming import DfnModel, chunk_equivalence
    from ..metrics.categories import write_csv
    from ..utils.session import create_session

    session = create_session(cfg, "benchmark")
    sr = cfg.audio.sample_rate
    if args.input:
        audio = load_audio(args.input, sr)[: int(args.seconds * sr)]
    else:
        rng = np.random.default_rng(cfg.run.seed)
        t = np.arange(int(args.seconds * sr)) / sr
        speech_like = 0.15 * np.sin(2 * np.pi * 180 * t) * (1 + 0.6 * np.sin(2 * np.pi * 3.1 * t))
        audio = (speech_like + 0.05 * rng.standard_normal(len(t))).astype(np.float32)

    thread_counts = [int(x) for x in args.threads.split(",") if x.strip()]
    chunk_sizes = [float(x) for x in args.chunk_sizes.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    print(f"\nbenchmark: {args.seconds:.0f} s of audio, {args.trials} trial(s) per configuration\n")
    for threads in thread_counts:
        neural = cfg.neural.model_copy(deep=True)
        neural.num_threads = threads if threads > 0 else None
        model = DfnModel(neural, sr).load()
        label = f"{threads} thread(s)" if threads > 0 else "torch default threads"

        # Whole-file (reference) path.
        model.timer._times_ms.clear()
        model.timer._audio_s = 0.0
        for _ in range(args.trials):
            model.enhance_array(audio)
        s = model.timer.summary()
        rows.append(
            {
                "config": f"whole_file / {label}",
                "framing": "whole_file",
                "threads": model.info.num_threads,
                "chunk_s": None,
                "rtf": round(s.rtf, 4),
                "mean_ms_per_call": round(s.mean_ms, 2),
                "p95_ms": round(s.p95_ms, 2),
                "si_sdr_vs_offline_db": None,
                "worst_case_latency_ms": round(model.info.algorithmic_latency_ms, 1),
            }
        )
        print(f"  whole_file  {label:26s} RTF {s.rtf:.4f}  mean {s.mean_ms:7.1f} ms/call")

        contexts = (
            [float(x) for x in args.contexts.split(",") if x.strip()]
            if args.contexts
            else [cfg.neural.streaming.context_s]
        )
        for chunk_s in chunk_sizes:
            for context_s in contexts:
                model.timer._times_ms.clear()
                model.timer._audio_s = 0.0
                eq = chunk_equivalence(model, audio, chunk_s, cfg.neural.streaming.overlap,
                                       cfg.neural.streaming.crossfade_ms, sr, context_s)
                s = model.timer.summary()
                total_latency = model.info.algorithmic_latency_ms + eq["worst_case_latency_ms"]
                rows.append(
                    {
                        "config": f"chunked {chunk_s} s ctx {context_s} s / {label}",
                        "framing": "chunked",
                        "threads": model.info.num_threads,
                        "chunk_s": chunk_s,
                        "context_s": context_s,
                        "rtf": round(s.rtf, 4),
                        "mean_ms_per_call": round(s.mean_ms, 2),
                        "p95_ms": round(s.p95_ms, 2),
                        "si_sdr_vs_offline_db": round(eq["si_sdr_vs_offline_db"], 2),
                        "worst_case_latency_ms": round(total_latency, 1),
                    }
                )
                print(
                    f"  chunked {chunk_s:4.1f}s ctx {context_s:4.2f}s {label:24s} "
                    f"RTF {s.rtf:.4f}  agreement {eq['si_sdr_vs_offline_db']:6.1f} dB  "
                    f"latency {total_latency:7.1f} ms"
                )

    path = write_csv(session.root / "benchmark.csv", rows)
    print(f"\nwrote {path}")
    print(
        "\nAgreement is SI-SDR of the chunked output against the whole-file output on the same\n"
        "input: it is how closely the live framing reproduces the reference inference path, not a\n"
        "quality score. 18 dB corresponds to about 12% RMS difference. No perceptual threshold is\n"
        "claimed here; that would need a listening test.\n"
        "The gain from a non-zero context is the model's recurrent state being warm rather than\n"
        "reset at every chunk boundary.\n"
        "All figures measured on this host only; no claim is made about other hardware."
    )
    return 0


def cmd_calibrate(cfg: Config, args: argparse.Namespace) -> int:
    from ..audio.io import load_audio
    from ..dsp.preprocess import calibrate_reference_gain

    sr = cfg.audio.sample_rate
    primary = load_audio(args.primary, sr)
    reference = load_audio(args.reference, sr)
    result = calibrate_reference_gain(primary, reference, sr)
    print("\nchannel calibration (run this on noise-only audio):")
    for key, value in result.as_dict().items():
        print(f"  {key:32s} {value}")
    print(
        f"\nApply with:  --set preprocess.reference_gain_db={result.suggested_reference_gain_db:.2f}"
    )
    if np.isfinite(result.coherence) and result.coherence < 0.3:
        print(
            "\nWARNING: broadband coherence between the channels is low, so the reference carries "
            "little information about the noise in the primary. Expect poor cancellation regardless "
            "of gain."
        )
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    import json

    from ..utils.session import find_session

    root = find_session(Path(args.session) if args.session else None, cfg.run.session_root)
    print(f"session: {root}")
    for name in ("report.pdf", "metrics.json", "metrics.csv", "config.yaml", "session.log"):
        p = root / name
        print(f"  {'OK ' if p.is_file() else '-- '} {name}"
              + (f"  ({p.stat().st_size / 1024:.0f} kB)" if p.is_file() else ""))
    metrics = root / "metrics.json"
    if metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        print(f"\ntitle: {payload.get('title')}")
        print(f"mode:  {payload.get('mode')}")
        targets = payload.get("targets", {})
        if targets.get("checks"):
            print(f"\ntargets ({targets.get('scope', '')}):")
            for c in targets["checks"]:
                value = c.get("value")
                shown = "n/a" if value is None else f"{value:.3f}"
                print(f"  {c['name']:22s} {c['comparison']} {c['target']:<6g} measured {shown:>8s}"
                      f"  {'PASS' if c['passed'] else 'FAIL'}")
        warnings = payload.get("warnings", [])
        if warnings:
            print(f"\n{len(warnings)} warning(s):")
            for w in warnings[:10]:
                print(f"  - {w}")
    audio = sorted((root / "audio").glob("*.wav")) if (root / "audio").is_dir() else []
    figures = sorted((root / "figures").glob("*.png")) if (root / "figures").is_dir() else []
    print(f"\n{len(audio)} WAV artefact(s), {len(figures)} figure(s)")
    return 0


def cmd_selftest(cfg: Config, args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    return run_selftest(cfg)


COMMANDS = {
    "list-devices": cmd_list_devices,
    "licenses": cmd_licenses,
    "fetch-data": cmd_fetch_data,
    "build-dataset": cmd_build_dataset,
    "dataset-stats": cmd_dataset_stats,
    "corpus": cmd_corpus,
    "run": cmd_run,
    "evaluate": cmd_evaluate,
    "benchmark": cmd_benchmark,
    "calibrate": cmd_calibrate,
    "report": cmd_report,
    "dashboard": cmd_dashboard,
    "selftest": cmd_selftest,
}


def _print_artifacts(written: dict[str, Path]) -> None:
    if not written:
        return
    print("\nartefacts:")
    for kind, path in written.items():
        print(f"  {kind:5s} {path}")
    pdf = written.get("pdf")
    if pdf is not None:
        print(f"\nOpen the report:  start {pdf}" if sys.platform == "win32" else f"\nOpen: {pdf}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = _config_from_args(args)
    except ConfigError as exc:
        print(f"configuration error:\n{exc}", file=sys.stderr)
        return 2

    setup_logging(cfg.run.log_level)
    from ..utils.seeding import seed_everything

    seed_everything(cfg.run.seed, cfg.run.deterministic_torch)

    handler = COMMANDS[args.command]
    t0 = time.perf_counter()
    try:
        code = handler(cfg, args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (ConfigError, ValueError, RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    log.debug("%s finished in %.1f s", args.command, time.perf_counter() - t0)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
