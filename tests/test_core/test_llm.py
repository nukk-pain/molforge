# pyright: reportMissingImports=false
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.core.llm import call_claude, extract_json  # noqa: E402


def test_call_claude_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("molforge.core.llm.shutil.which", lambda _: "/fake/bin/claude")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout='{"ok": true}\n',
            stderr="",
        )

    monkeypatch.setattr("molforge.core.llm.subprocess.run", fake_run)

    assert call_claude("hello", model="haiku") == '{"ok": true}'


def test_call_claude_raises_when_cli_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("molforge.core.llm.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="Claude CLI is not installed"):
        _ = call_claude("hello")


def test_call_claude_raises_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("molforge.core.llm.shutil.which", lambda _: "/fake/bin/claude")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["claude"], timeout=300)

    monkeypatch.setattr("molforge.core.llm.subprocess.run", fake_run)

    with pytest.raises(TimeoutError, match="timed out"):
        _ = call_claude("hello")


def test_extract_json_supports_plain_json() -> None:
    payload = extract_json('{"gene":"TGFB1"}')
    assert payload == {"gene": "TGFB1"}


def test_extract_json_supports_fenced_json() -> None:
    payload = extract_json("Result:\n```json\n[1, 2, 3]\n```")
    assert payload == [1, 2, 3]


def test_extract_json_supports_embedded_json_block() -> None:
    payload = extract_json("Summary first\n{'bad': true}\nActual: {\"score\": 0.87}")
    assert payload == {"score": 0.87}
