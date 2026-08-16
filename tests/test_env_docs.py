"""Setup docs.

.env.example is all that stands between a fresh clone and a bot with no market
data: provider keys are optional at runtime, so a missing one is not an error,
just a worse answer. The README told people to set FINNHUB_API_KEY while
.env.example did not mention it, which sends the reader looking for a variable
that is not there.

Everything here is derived from the source rather than hardcoded, so a provider
added to config.py later fails this file instead of quietly going undocumented.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Read but never written by hand — the host assigns it.
NOT_FOR_HUMANS: set[str] = set()


def _read(name: str) -> str:
    # utf-8 explicitly: README has box-drawing characters, and Windows would
    # otherwise decode it as cp1252 and raise.
    return (ROOT / name).read_text(encoding="utf-8")


def _keys_read_by_config() -> set[str]:
    return set(
        re.findall(
            r'(?:_required|os\.environ\.get)\(\s*"([A-Z0-9_]+)"', _read("atlas/config.py")
        )
    )


def _keys_in_env_example() -> set[str]:
    return set(re.findall(r"^([A-Z0-9_]+)=", _read(".env.example"), re.M))


def test_env_example_lists_every_setting_the_app_reads():
    missing = _keys_read_by_config() - NOT_FOR_HUMANS - _keys_in_env_example()

    assert sorted(missing) == []


def test_readme_names_only_keys_a_reader_can_find():
    """README's setup line is the first thing anyone follows. Naming a key that
    .env.example does not declare sends them looking for a file that lies."""
    named = set(re.findall(r"\b([A-Z][A-Z0-9_]{4,})\b", _read("README.md")))
    # Intersect with the real settings so prose like EDGAR or APSCHEDULER is
    # not mistaken for a variable name.
    named &= _keys_read_by_config()

    assert sorted(named - _keys_in_env_example()) == []


def test_render_and_env_example_agree():
    """render.yaml is the deployed truth. A key configured there and missing
    here means a local run quietly behaves differently from production."""
    in_render = set(re.findall(r"- key: ([A-Z0-9_]+)", _read("render.yaml")))
    in_render &= _keys_read_by_config()

    assert sorted(in_render - _keys_in_env_example()) == []


def test_readme_install_command_can_actually_run_the_tests():
    """pytest lives in the dev extra. `pip install -e .` alone cannot run it."""
    assert 'pip install -e ".[dev]"' in _read("README.md")
