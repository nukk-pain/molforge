# pyright: reportMissingImports=false
from __future__ import annotations

import base64
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.diffdock_runner import (  # noqa: E402
    DiffDockPose,
    DiffDockResult,
    parse_diffdock_samples,
    run_diffdock_l,
)


# ---------------------------------------------------------------------------
# parse_diffdock_samples
# ---------------------------------------------------------------------------


def test_parse_diffdock_samples_reads_rank_from_filename(tmp_path: Path) -> None:
    # DiffDock-L layout: `query/rank{N}.sdf`
    complex_dir = tmp_path / "query"
    complex_dir.mkdir()
    for i in range(1, 4):
        (complex_dir / f"rank{i}.sdf").write_text(f"SDF_{i}", encoding="utf-8")

    samples = parse_diffdock_samples(tmp_path)
    assert [s.rank for s in samples] == [1, 2, 3]
    assert samples[0].sampler == "diffdock_l"
    assert samples[0].sample_index == 0
    assert samples[0].confidence is None


def test_parse_diffdock_samples_reads_confidence_from_filename(tmp_path: Path) -> None:
    # `--save_visualisation true` format: rank{N}_confidence-{score}.sdf
    complex_dir = tmp_path / "query"
    complex_dir.mkdir()
    (complex_dir / "rank1_confidence-0.84.sdf").write_text("S1")
    (complex_dir / "rank2_confidence-0.52.sdf").write_text("S2")
    (complex_dir / "rank3_confidence--0.21.sdf").write_text("S3")

    samples = parse_diffdock_samples(tmp_path)
    by_rank = {s.rank: s for s in samples}
    assert by_rank[1].confidence == 0.84
    assert by_rank[2].confidence == 0.52
    assert by_rank[3].confidence == -0.21


def test_parse_diffdock_samples_reads_sidecar_confidence(tmp_path: Path) -> None:
    complex_dir = tmp_path / "query"
    complex_dir.mkdir()
    for i in range(1, 4):
        (complex_dir / f"rank{i}.sdf").write_text(f"S{i}")
    (complex_dir / "confidence.txt").write_text("0.9\n0.4\n-0.1\n")

    samples = parse_diffdock_samples(tmp_path)
    by_rank = {s.rank: s for s in samples}
    assert by_rank[1].confidence == 0.9
    assert by_rank[2].confidence == 0.4
    assert by_rank[3].confidence == -0.1


def test_parse_diffdock_samples_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert parse_diffdock_samples(tmp_path) == []


def test_parse_diffdock_samples_flat_layout_without_rank_prefix(
    tmp_path: Path,
) -> None:
    # Defensive path — SDF with no rank in filename gets positional rank.
    (tmp_path / "a.sdf").write_text("A")
    (tmp_path / "b.sdf").write_text("B")
    samples = parse_diffdock_samples(tmp_path)
    assert [s.rank for s in samples] == [1, 2]


def test_parse_diffdock_samples_dedupes_rank1_with_and_without_confidence(
    tmp_path: Path,
) -> None:
    # Real DiffDock-L output: rank 1 emitted twice (with and without
    # confidence in filename). We keep the `_confidence-` variant because
    # it carries the score.
    (tmp_path / "rank1.sdf").write_text("RANK1_NO_CONF")
    (tmp_path / "rank1_confidence-1.87.sdf").write_text("RANK1_WITH_CONF")
    (tmp_path / "rank2_confidence-2.11.sdf").write_text("RANK2")

    samples = parse_diffdock_samples(tmp_path)
    assert len(samples) == 2
    rank1 = next(s for s in samples if s.rank == 1)
    assert rank1.confidence == 1.87
    # Plain rank1.sdf should have been dropped.
    assert "confidence" in rank1.structure_path.name


# ---------------------------------------------------------------------------
# run_diffdock_l with injected runner (Modal-free)
# ---------------------------------------------------------------------------


def _mock_runner_success(payload: dict) -> dict:
    """Pretend Modal call — returns 3 ranked SDF files."""
    _ = payload  # unused; just return a deterministic fake result
    files = {
        f"query/rank{i}.sdf": base64.b64encode(f"SDF_{i}".encode()).decode(
            "ascii"
        )
        for i in range(1, 4)
    }
    files["query/confidence.txt"] = base64.b64encode(
        b"0.84\n0.52\n0.11\n"
    ).decode("ascii")
    return {
        "success": True,
        "returncode": 0,
        "stdout": "done",
        "stderr": "",
        "output_files": files,
        "elapsed": 42.0,
        "cost_estimate_usd": 0.0245,
    }


def test_run_diffdock_l_parses_mock_result(tmp_path: Path) -> None:
    result = run_diffdock_l(
        protein_pdb_bytes=b"ATOM  test\n",
        ligand_smiles="CCO",
        num_samples=3,
        output_dir=tmp_path,
        runner=_mock_runner_success,
    )
    assert isinstance(result, DiffDockResult)
    assert len(result.samples) == 3
    assert result.top_rank_sample is not None
    assert result.top_rank_sample.rank == 1
    assert result.top_rank_sample.confidence == 0.84
    assert result.cost_estimate_usd == 0.0245
    assert result.top_rank_sample.structure_path.exists()
    assert result.top_rank_sample.structure_path.read_bytes() == b"SDF_1"


def test_run_diffdock_l_with_failed_modal_returns_empty_samples(
    tmp_path: Path,
) -> None:
    def _mock_runner_fail(payload: dict) -> dict:
        _ = payload
        return {
            "success": False,
            "returncode": 1,
            "stdout": "",
            "stderr": "DiffDock crashed",
            "output_files": {},
            "elapsed": 3.0,
            "cost_estimate_usd": 0.002,
        }

    result = run_diffdock_l(
        protein_pdb_bytes=b"ATOM  test\n",
        ligand_smiles="CCO",
        num_samples=3,
        output_dir=tmp_path,
        runner=_mock_runner_fail,
    )
    assert result.samples == []
    assert result.top_rank_sample is None
    assert "DiffDock crashed" in result.stderr_tail


def test_run_diffdock_l_passes_smiles_and_num_samples_to_runner(
    tmp_path: Path,
) -> None:
    captured_payload: dict = {}

    def _capture_runner(payload: dict) -> dict:
        captured_payload.update(payload)
        return _mock_runner_success(payload)

    _ = run_diffdock_l(
        protein_pdb_bytes=b"XX",
        ligand_smiles="c1ccccc1",
        num_samples=25,
        inference_steps=30,
        output_dir=tmp_path,
        runner=_capture_runner,
    )
    assert captured_payload["ligand_smiles"] == "c1ccccc1"
    assert captured_payload["num_samples"] == 25
    assert captured_payload["inference_steps"] == 30
    assert (
        base64.b64decode(captured_payload["protein_pdb_b64"]) == b"XX"
    )


def test_diffdock_pose_sample_index_is_zero_indexed() -> None:
    pose = DiffDockPose(rank=3, structure_path=Path("/tmp/x.sdf"))
    assert pose.sample_index == 2
    assert pose.sampler == "diffdock_l"
