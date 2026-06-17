# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.remote.backend import JobSpec, LocalMockBackend  # noqa: E402


def test_local_mock_backend_executes_job_and_collects_outputs() -> None:
    backend = LocalMockBackend(cost_per_second_usd=0.5)
    spec = JobSpec(
        image="python:3.11",
        args=[
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "import json; "
                "payload=json.loads(Path('input.json').read_text()); "
                "Path('nested').mkdir(parents=True, exist_ok=True); "
                "Path('nested/output.json').write_text(json.dumps({'value': payload['value'] * 2}))"
            ),
        ],
        input_files={"input.json": b'{"value": 21}'},
        timeout_seconds=5,
    )

    handle = backend.submit(spec)
    result = backend.fetch_result(handle)

    assert result.success is True
    assert result.cost_estimate_usd >= 0.0
    assert json.loads(result.output_files["nested/output.json"].decode("utf-8")) == {
        "value": 42
    }
