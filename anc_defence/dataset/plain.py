"""Loader for the supplied single-channel corpus (``dataset_plain``).

Layout::

    dataset_plain/metadata.csv          5000 labelled examples
    dataset_plain/clean/<id>_<...>.wav  clean speech target
    dataset_plain/noisy/<cat>/<...>.wav noisy mixture (the pipeline input)
    dataset_plain/noise/<cat>/<...>.wav noise only, same length as the mixture

Everything is 16 kHz mono PCM_16 and is upsampled to 48 kHz on load, because
DeepFilterNet3 is a 48 kHz model. Nothing is invented by upsampling: the 8-24 kHz
band is empty, so this data does not exercise the model's fullband capability. It
does not affect the reported metrics, which are computed at 16 kHz (PESQ) and 10 kHz
(STOI) internally.

Two facts about this corpus that shape how results must be read:

* It is heavily weighted toward **gunshot** (4385 of 5000 labelled examples), with
  roughly 110-135 each of airplane, engine, helicopter, siren and train. An unweighted
  average over the whole corpus is therefore essentially a gunshot score, which is why
  the headline evaluation uses a **category-balanced stratified subset** and every
  table breaks results out per category.
* ``clean/`` and ``noisy/`` also contain about 4540 plain-named files (``00073.wav``)
  that have no metadata row: no category and no SNR. They are indexed as an unlabelled
  pool for listening and demos and are excluded from the scored results.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from ..config import PlainDatasetCfg
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class PlainExample:
    """One labelled example from the supplied corpus."""

    id: str
    clean_path: Path
    noisy_path: Path
    noise_path: Optional[Path]
    category: str
    snr_db: float
    speech_source: str = ""
    noise_file: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("clean_path", "noisy_path", "noise_path"):
            d[key] = str(d[key]) if d[key] is not None else None
        return d

    def manifest_entry(self) -> dict[str, Any]:
        """Shape the evaluation harness expects."""
        files = {"primary": str(self.noisy_path), "target": str(self.clean_path)}
        if self.noise_path is not None:
            files["noise"] = str(self.noise_path)
        return {
            "id": f"{self.category}/{self.id}",
            "category": self.category,
            "snr_db": self.snr_db,
            "speech_source": self.speech_source,
            "components": [{"noise_type": self.category, "path": self.noise_file}],
            "impulsive_events": [],
            "n_impulsive_events": 0,
            "files": files,
        }


# Mapping from the corpus's noise classes onto the problem statement's taxonomy.
# Recorded explicitly so the grouping in the report is a stated decision.
CATEGORY_TAXONOMY: dict[str, str] = {
    "gunshot": "impulsive",
    "siren": "non_stationary",
    "train": "non_stationary",
    "airplane": "stationary",
    "helicopter": "stationary",
    "engine": "stationary",
}

TAXONOMY_NOTES: dict[str, str] = {
    "gunshot": "small arms fire, many weapon types and ranges - impulsive, high crest factor",
    "siren": "emergency siren - non-stationary, strong frequency modulation",
    "train": "tracked/rail vehicle pass-by - non-stationary, level sweeps",
    "airplane": "fixed-wing aircraft - broadband, near stationary",
    "helicopter": "rotor wash - periodic amplitude modulation, near stationary",
    "engine": "vehicle engine - harmonic and near stationary",
}


def taxonomy_of(category: str) -> str:
    return CATEGORY_TAXONOMY.get(category, "unknown")


def _resolve(root: Path, raw: str) -> Path:
    """Resolve a metadata path, which may be repo-relative or already absolute."""
    p = Path(str(raw).replace("\\", "/"))
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p
    # Metadata stores paths like "dataset_plain/clean/x.wav" relative to the repo root.
    candidate = root.parent / p if root.name == p.parts[0] else root / p
    if candidate.exists():
        return candidate
    alt = Path.cwd() / p
    return alt if alt.exists() else candidate


@dataclass
class PlainCorpus:
    """The labelled examples plus a description of what was found."""

    examples: list[PlainExample] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unlabelled: list[Path] = field(default_factory=list)
    root: Path = Path("dataset_plain")
    native_sample_rate: int = 16000

    # ------------------------------------------------------------------ views
    @property
    def categories(self) -> list[str]:
        return sorted({e.category for e in self.examples})

    @property
    def snr_values(self) -> list[float]:
        return sorted({e.snr_db for e in self.examples})

    def by_category(self) -> dict[str, int]:
        return dict(Counter(e.category for e in self.examples))

    def by_taxonomy(self) -> dict[str, int]:
        return dict(Counter(taxonomy_of(e.category) for e in self.examples))

    def by_snr(self) -> dict[float, int]:
        return dict(sorted(Counter(e.snr_db for e in self.examples).items()))

    def cells(self) -> dict[tuple[str, float], list[PlainExample]]:
        out: dict[tuple[str, float], list[PlainExample]] = defaultdict(list)
        for e in self.examples:
            out[(e.category, e.snr_db)].append(e)
        return dict(out)

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "native_sample_rate": self.native_sample_rate,
            "labelled_examples": len(self.examples),
            "unlabelled_files": len(self.unlabelled),
            "missing_files": len(self.missing),
            "categories": self.by_category(),
            "taxonomy": self.by_taxonomy(),
            "snr_distribution": {str(k): v for k, v in self.by_snr().items()},
            "category_taxonomy_map": CATEGORY_TAXONOMY,
            "taxonomy_notes": TAXONOMY_NOTES,
        }


def load_plain_corpus(cfg: PlainDatasetCfg, check_files: bool = True) -> PlainCorpus:
    """Read ``metadata.csv`` and index the corpus."""
    root = Path(cfg.root)
    meta = Path(cfg.metadata)
    if not meta.is_file():
        raise FileNotFoundError(
            f"metadata not found at {meta}. Point plain.metadata at the corpus CSV "
            "(columns: id, noise_category, snr_db, clean_path, noisy_path, noise_only_path)."
        )

    corpus = PlainCorpus(root=root, native_sample_rate=cfg.native_sample_rate)
    # One directory walk instead of 15000 individual stat calls: on Windows the
    # per-file checks dominate load time for a corpus this size.
    on_disk = {p.resolve() for p in root.rglob("*.wav")} if check_files else set()

    def present(p: Optional[Path]) -> bool:
        return p is not None and (not check_files or p.resolve() in on_disk)

    with meta.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            clean = _resolve(root, row.get("clean_path", ""))
            noisy = _resolve(root, row.get("noisy_path", ""))
            noise_raw = row.get("noise_only_path", "")
            noise = _resolve(root, noise_raw) if noise_raw else None
            if not (present(clean) and present(noisy)):
                corpus.missing.append(str(row.get("id", "?")))
                continue
            if not present(noise):
                noise = None
            try:
                snr = float(row.get("snr_db", "nan"))
            except ValueError:
                snr = float("nan")
            corpus.examples.append(
                PlainExample(
                    id=str(row.get("id", "")),
                    clean_path=clean,
                    noisy_path=noisy,
                    noise_path=noise,
                    category=str(row.get("noise_category", "unknown")),
                    snr_db=snr,
                    speech_source=str(row.get("speech_source", "")),
                    noise_file=str(row.get("noise_file", "")),
                )
            )

    corpus.unlabelled = _find_unlabelled(root, corpus, on_disk if check_files else None)
    log.info(
        "plain corpus: %d labelled examples, %d categories %s, SNRs %s, %d unlabelled file(s)%s",
        len(corpus.examples),
        len(corpus.categories),
        corpus.by_category(),
        corpus.snr_values,
        len(corpus.unlabelled),
        f", {len(corpus.missing)} metadata row(s) with missing audio" if corpus.missing else "",
    )
    return corpus


def _find_unlabelled(
    root: Path, corpus: PlainCorpus, on_disk: Optional[set[Path]] = None
) -> list[Path]:
    """Files present on disk that no metadata row refers to.

    Compared by file name rather than resolved path: name comparison is cheap and the
    corpus uses unique numeric ids.
    """
    noisy_dir = root / "noisy"
    if not noisy_dir.is_dir():
        return []
    referenced = {e.noisy_path.name for e in corpus.examples}
    candidates = (
        [p for p in on_disk if p.name not in referenced and "noisy" in p.parts]
        if on_disk is not None
        else [p for p in noisy_dir.rglob("*.wav") if p.name not in referenced]
    )
    return sorted(candidates)


def stratified_subset(
    corpus: PlainCorpus,
    per_cell: int,
    seed: int = 1234,
    categories: Optional[Sequence[str]] = None,
    snr_values: Optional[Sequence[float]] = None,
) -> list[PlainExample]:
    """Draw ``per_cell`` examples from every (category, SNR) cell.

    Balancing matters here: gunshot is 88% of this corpus, so an unstratified sample
    would make every "overall" number a gunshot number. Selection is seeded, so the
    same subset is drawn every run.
    """
    rng = np.random.default_rng(seed)
    wanted_cats = set(categories) if categories else set(corpus.categories)
    wanted_snrs = set(float(s) for s in snr_values) if snr_values else set(corpus.snr_values)

    chosen: list[PlainExample] = []
    for (category, snr), items in sorted(corpus.cells().items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if category not in wanted_cats or snr not in wanted_snrs:
            continue
        pool = sorted(items, key=lambda e: e.id)
        if len(pool) <= per_cell:
            chosen.extend(pool)
            continue
        idx = rng.choice(len(pool), size=per_cell, replace=False)
        chosen.extend(pool[int(i)] for i in sorted(idx))
    chosen.sort(key=lambda e: (e.category, e.snr_db, e.id))
    log.info(
        "stratified subset: %d examples from %d cells (%d per cell requested)",
        len(chosen),
        len({(e.category, e.snr_db) for e in chosen}),
        per_cell,
    )
    return chosen


def build_manifest(
    corpus: PlainCorpus,
    examples: Optional[Iterable[PlainExample]] = None,
    subset_name: str = "plain",
) -> dict[str, Any]:
    """Wrap examples in the manifest shape the evaluation harness consumes."""
    items = list(examples) if examples is not None else list(corpus.examples)
    entries = []
    for e in items:
        entry = e.manifest_entry()
        entry["id"] = f"{subset_name}/{e.category}_{e.id}"
        entry["taxonomy"] = taxonomy_of(e.category)
        entries.append(entry)
    return {
        "source": "dataset_plain",
        "native_sample_rate": corpus.native_sample_rate,
        "corpus_summary": corpus.summary(),
        "n_examples": len(entries),
        "examples": entries,
        "warnings": (
            [
                f"{len(corpus.missing)} metadata row(s) referenced audio that is not on disk"
            ]
            if corpus.missing
            else []
        ),
    }


def corpus_stats_rows(corpus: PlainCorpus, subset: Optional[Sequence[PlainExample]] = None) -> list[tuple[str, str]]:
    """Key/value rows describing the corpus, for the report."""
    rows: list[tuple[str, str]] = [
        ("Corpus", str(corpus.root)),
        ("Native format", f"{corpus.native_sample_rate} Hz mono PCM_16, upsampled to 48 kHz on load"),
        ("Labelled examples", str(len(corpus.examples))),
        ("Unlabelled files (excluded from scoring)", str(len(corpus.unlabelled))),
    ]
    counts = corpus.by_category()
    total = max(1, sum(counts.values()))
    for cat in sorted(counts, key=lambda c: -counts[c]):
        rows.append(
            (f"  {cat} ({taxonomy_of(cat)})", f"{counts[cat]} ({100.0 * counts[cat] / total:.1f}%)")
        )
    rows.append(("SNR values (dB)", ", ".join(f"{s:g}" for s in corpus.snr_values)))
    if subset is not None:
        sub_counts = Counter(e.category for e in subset)
        rows.append(("Evaluated subset", f"{len(subset)} examples, balanced across category x SNR"))
        rows.append(
            ("  subset per category", ", ".join(f"{k}:{v}" for k, v in sorted(sub_counts.items())))
        )
    return rows
