"""Build the 5 requested tables from an existing offline_file session's metrics.csv.

Tables requested:
  1. Comparison: unprocessed / NLMS only / DFN only / DFN+NLMS hybrid, aggregate row each
  2. Baseline DeepFilterNet3 only, per noise category
  3. Baseline NLMS only, per noise category
  4. DeepFilterNet + NLMS hybrid, per noise category
  5. Entire delivered pipeline (DFN + volume normalisation), per noise category

All from dataset_plain via the existing evaluation session - no re-run, no synthetic data.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SESSION = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "sessions/2026-09-09_12-14-00_offline_file"
)


def load_rows(method: str) -> list[dict]:
    # "unprocessed" is tagged tap="input" (it IS the input); every other method's
    # scored result is tagged tap="output".
    want_tap = "input" if method == "unprocessed" else "output"
    with open(SESSION / "metrics.csv", newline="", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["method"] == method and r["tap"] == want_tap]


def f(row: dict, key: str) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def mean(rows: list[dict], key: str) -> float:
    vals = [f(r, key) for r in rows]
    vals = [v for v in vals if v == v]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def snr_before_after(rows: list[dict]) -> tuple[float, float, float]:
    """(avg input SNR, avg output SNR, avg improvement) using the classical output-SNR
    fields (residual_noise_snr_db, falling back to output_snr_db) against
    measured_input_snr_db, both computed per example then averaged."""
    before, after = [], []
    for r in rows:
        b = f(r, "measured_input_snr_db")
        a = f(r, "residual_noise_snr_db")
        if a != a:
            a = f(r, "output_snr_db")
        if b == b and a == a:
            before.append(b)
            after.append(a)
    if not before:
        return float("nan"), float("nan"), float("nan")
    b, a = float(np.mean(before)), float(np.mean(after))
    return b, a, a - b


def print_comparison_table(methods: dict[str, str]) -> None:
    print("=" * 100)
    print("TABLE 1: Comparison across configurations (216 examples, dataset_plain, category-balanced)")
    print("=" * 100)
    hdr = f"{'Pipeline Stage / Configuration':<38}{'SNR (dB)':>12}{'SI-SDR (dB)':>13}{'STOI':>9}{'PESQ':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, method in methods.items():
        rows = load_rows(method)
        if not rows:
            print(f"{label:<38}{'n/a':>12}{'n/a':>13}{'n/a':>9}{'n/a':>9}   (method not found)")
            continue
        _, out_snr, _ = snr_before_after(rows)
        si_sdr = mean(rows, "si_sdr")
        stoi = mean(rows, "stoi")
        pesq = mean(rows, "pesq")
        print(f"{label:<38}{out_snr:>12.2f}{si_sdr:>13.2f}{stoi:>9.3f}{pesq:>9.2f}")
    print(
        "\nSNR here is the classical output SNR (surviving speech power over residual noise\n"
        "power, via the mixture-derived gain decomposition - see metrics.intrusive.\n"
        "residual_noise_snr_db). SI-SDR is the stricter scale-invariant figure that also\n"
        "charges for speech distortion, not just residual noise; the two differ by design.\n"
    )


def print_category_table(title: str, method: str) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)
    with open(SESSION / "metrics.csv", newline="", encoding="utf-8") as fh:
        all_rows = [r for r in csv.DictReader(fh) if r["method"] == method and r["tap"] == "output"]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_cat[r["category"]].append(r)

    hdr = (f"{'Category':<14}{'Avg STOI':>10}{'Avg PESQ':>10}"
           f"{'Avg SNR Before (dB)':>22}{'Avg SNR After (dB)':>21}{'Avg SNR Impr. (dB)':>21}")
    print(hdr)
    print("-" * len(hdr))
    for cat in sorted(by_cat):
        rows = by_cat[cat]
        stoi = mean(rows, "stoi")
        pesq = mean(rows, "pesq")
        before, after, impr = snr_before_after(rows)
        print(f"{cat:<14}{stoi:>10.4f}{pesq:>10.4f}{before:>22.2f}{after:>21.2f}{impr:>21.2f}")
    print()


if __name__ == "__main__":
    if not (SESSION / "metrics.csv").is_file():
        raise SystemExit(f"no metrics.csv at {SESSION}")

    print(f"source session: {SESSION}\n")

    print_comparison_table({
        "1. Raw Noisy Audio (Baseline)": "unprocessed",
        "2. NLMS Filter Only (Adaptive Stage)": "nlms_only",
        "3. AI Model Only (Without NLMS)": "dfn_only",
        "4. Hybrid NLMS + AI (dfn_then_nlms)": "dfn_then_nlms",
        "4b. Hybrid AI + NLMS (nlms_then_dfn)": "nlms_then_dfn",
        "5. Delivered pipeline (AI + normalise)": "dfn_then_normalise",
    })

    print_category_table("TABLE 2: Baseline DeepFilterNet3 Only", "dfn_only")
    print_category_table("TABLE 3: Baseline NLMS Only", "nlms_only")
    print_category_table("TABLE 4a: Hybrid - DeepFilterNet then NLMS (dfn_then_nlms)", "dfn_then_nlms")
    print_category_table("TABLE 4b: Hybrid - NLMS then DeepFilterNet (nlms_then_dfn)", "nlms_then_dfn")
    print_category_table("TABLE 5: Entire Delivered Pipeline (DeepFilterNet + Volume Normalisation)",
                          "dfn_then_normalise")
