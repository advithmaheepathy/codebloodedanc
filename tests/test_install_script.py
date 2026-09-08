"""Static checks on scripts/install_jetson.sh.

The install script runs on the Jetson, not here, so it cannot be executed as part of the
test suite. These checks catch the mistakes that would otherwise only surface on the
board, where a failed install costs real time:

* unbalanced heredocs
* ``set -e`` hazards: ``[ test ] && command`` aborts the whole script whenever the test
  is false, because the list returns non-zero
* the PyTorch wheel URL matching the Python version the target actually runs
* an accidental ``pip install torch`` from PyPI, which on a Jetson yields a wheel
  without CUDA
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_jetson.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    return SCRIPT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def script_lines(script_text: str) -> list[str]:
    return script_text.split("\n")


def _code_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line) for executable lines only.

    Skips comments and, importantly, heredoc bodies: this script deliberately *prints*
    commands such as ``sudo nvpmodel -m 0`` inside a ``cat <<'EOF'`` block rather than
    running them, and a checker that cannot tell the difference is worse than none.
    """
    out: list[tuple[int, str]] = []
    heredoc_tag: str | None = None
    for i, line in enumerate(text.split("\n"), start=1):
        if heredoc_tag is not None:
            if line.strip() == heredoc_tag:
                heredoc_tag = None
            continue
        match = re.search(r"<<-?'?([A-Z][A-Z0-9_]*)'?", line)
        if match:
            heredoc_tag = match.group(1)
            # The line opening the heredoc is still executable code.
            out.append((i, line))
            continue
        if line.strip().startswith("#"):
            continue
        out.append((i, line))
    return out


def test_heredocs_are_balanced(script_lines: list[str]):
    opened: list[tuple[int, str]] = []
    for i, line in enumerate(script_lines, start=1):
        match = re.search(r"<<-?'?([A-Z][A-Z0-9_]*)'?", line)
        if match:
            opened.append((i, match.group(1)))
    problems = []
    for line_no, tag in opened:
        closes = sum(1 for line in script_lines if line.strip() == tag)
        if closes < 1:
            problems.append(f"heredoc '{tag}' opened at line {line_no} is never closed")
    counts: dict[str, int] = {}
    for _, tag in opened:
        counts[tag] = counts.get(tag, 0) + 1
    for tag, n_open in counts.items():
        n_close = sum(1 for line in script_lines if line.strip() == tag)
        if n_close != n_open:
            problems.append(f"heredoc '{tag}': opened {n_open}x, closed {n_close}x")
    assert not problems, "\n".join(problems)


def test_no_set_e_hazards(script_lines: list[str]):
    """``[ test ] && cmd`` exits the script under ``set -e`` when the test is false."""
    problems = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(script_lines, start=1)
        if re.match(r"^\s*(\[|\[\[|test )[^\n]*\]\]?\s*&&", line)
    ]
    assert not problems, (
        "these abort the script under `set -e` when the test is false; use an if block:\n"
        + "\n".join(problems)
    )


def test_uses_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text


def test_never_installs_torch_from_pypi(script_text: str):
    """On a Jetson the PyPI torch wheel has no CUDA. It must never be used."""
    for line in script_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"pip install\s+(-U\s+)?torch(\s|$)", stripped) and "--index-url" not in stripped:
            pytest.fail(f"plain PyPI torch install found: {stripped}")


def test_torch_wheel_matches_target_python(script_text: str):
    """The JetPack 5.1.3 wheel must be cp38: that board runs Python 3.8.10."""
    urls = re.findall(r"https://developer\.download\.nvidia\.com\S+\.whl", script_text)
    assert urls, "no explicit NVIDIA wheel URL found for the JetPack 5.1.x branch"
    for url in urls:
        assert "cp38" in url, f"wheel {url} is not cp38; JetPack 5.1.3 ships Python 3.8"
        assert "aarch64" in url, f"wheel {url} is not aarch64"


def test_torch_version_avoids_weights_only_break(script_text: str):
    """torch >= 2.6 flips torch.load(weights_only=True) and breaks the DFN checkpoint."""
    for major, minor in re.findall(r"torch-(\d+)\.(\d+)", script_text):
        version = (int(major), int(minor))
        assert version < (2, 6), (
            f"wheel pins torch {major}.{minor}; >= 2.6 changes the torch.load default and "
            "breaks loading the DeepFilterNet checkpoint"
        )


def test_shim_script_is_referenced_and_present(script_text: str):
    shim = SCRIPT.parent / "torchaudio_shim.py"
    assert shim.is_file(), "scripts/torchaudio_shim.py is missing"
    assert "torchaudio_shim.py" in script_text, "the install script never offers the shim fallback"


def test_privileged_steps_are_announced(script_text: str):
    """Every sudo call must go through run_sudo, which prints the command first."""
    problems = []
    for i, line in _code_lines(script_text):
        stripped = line.strip()
        if "run_sudo" in stripped or stripped == 'sudo "$@"':
            continue  # run_sudo itself, and its one sudo call
        # `sudo nvpmodel -q` is a read-only query and is allowed inline.
        if re.search(r"(^|\s)sudo\s", stripped) and "nvpmodel -q" not in stripped:
            problems.append(f"line {i}: {stripped}")
    assert not problems, "unannounced sudo calls:\n" + "\n".join(problems)


def test_nvpmodel_and_jetson_clocks_are_not_executed(script_text: str):
    """These change the board's thermal behaviour and must be printed, not run.

    Lines inside heredocs are fine: the script prints them as recommendations.
    """
    for i, line in _code_lines(script_text):
        stripped = line.strip()
        if stripped.startswith("echo"):
            continue
        if re.match(r"^(sudo\s+)?(nvpmodel\s+-m|jetson_clocks)", stripped):
            pytest.fail(f"line {i} executes a power/clock change: {stripped}")


def test_recommendations_are_actually_printed(script_text: str):
    """The power-mode guidance must still reach the operator, not just be absent."""
    assert "nvpmodel -m 0" in script_text, "the MAXN recommendation is missing entirely"
    assert "jetson_clocks" in script_text, "the clock-locking recommendation is missing"


def test_warns_that_ape_devices_are_not_microphones(script_text: str):
    """The target board lists only Tegra APE DMA channels under `arecord -l`.

    Mistaking those for a capture device wastes real time on demo day, so the script has
    to say so explicitly.
    """
    lowered = script_text.lower()
    assert "ape" in lowered, "the APE devices are never mentioned"
    assert "not a microphone" in lowered or "not microphones" in lowered, (
        "the script does not warn that the Tegra APE/XBAR-ADMAIF entries are not capture "
        "hardware"
    )
