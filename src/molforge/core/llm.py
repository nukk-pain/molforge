from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Literal, cast

JSONValue = dict[str, object] | list[object]
ALLOWED_MODELS = {"haiku", "sonnet", "opus"}


def call_claude(
    prompt: str,
    *,
    model: Literal["haiku", "sonnet", "opus"] = "haiku",
    timeout: int = 300,
) -> str:
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported Claude model: {model}")

    claude_path = shutil.which("claude")
    if claude_path is None:
        raise RuntimeError(
            "Claude CLI is not installed or not on PATH. Install it before using molforge core LLM features."
        )

    try:
        result = subprocess.run(
            [claude_path, "--model", model, "--print", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Claude CLI timed out after {timeout} seconds.") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    return result.stdout.strip()


def extract_json(output: str) -> JSONValue:
    fenced_match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    if fenced_match:
        try:
            return cast(JSONValue, json.loads(fenced_match.group(1)))
        except json.JSONDecodeError:
            pass

    try:
        return cast(JSONValue, json.loads(output))
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(output):
        if char not in "[{":
            continue
        try:
            parsed: object
            parsed, _ = decoder.raw_decode(output[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            return cast(JSONValue, parsed)

    raise ValueError("No valid JSON object or array found in Claude output.")
