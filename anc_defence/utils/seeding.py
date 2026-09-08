"""Deterministic seeding.

Every source of randomness in the project draws from a seeded generator. Dataset
generation uses explicit ``numpy.random.Generator`` objects derived from the
configured seed so a build is reproducible regardless of import order.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np

from .logging import get_logger

log = get_logger(__name__)


def seed_everything(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    """Seed stdlib, numpy and (if importable) torch. Returns a fresh generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as exc:  # pragma: no cover - torch optional at import time
        log.debug("torch seeding skipped: %s", exc)
    return np.random.default_rng(seed)


def child_generator(seed: int, *tags: object) -> np.random.Generator:
    """Derive a reproducible generator from a base seed and arbitrary tags.

    Used so that per-utterance randomness depends only on the base seed and the
    utterance identity, not on generation order.
    """
    material = "|".join((str(seed), *(str(t) for t in tags)))
    # hashlib, not hash(): builtin string hashing is randomised per interpreter run.
    digest = int.from_bytes(hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence([seed, digest]))
