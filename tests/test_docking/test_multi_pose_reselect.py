# pyright: reportMissingImports=false
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from molforge.docking.multi_pose import (  # noqa: E402
    parse_boltz_samples,
    write_boltz_input_yaml,
)
from molforge.docking.pose_rescorer import RescoreResult  # noqa: E402
from molforge.docking.reselect import reselect_via_boltz2, select_best_pose  # noqa: E402
from molforge.remote.backend import JobHandle, JobResult, JobSpec  # noqa: E402


def test_write_boltz_input_yaml_includes_affinity_block() -> None:
    yaml_body = write_boltz_input_yaml(
        protein_sequence="MKTAYIAKQR", ligand_smiles="CCO"
    )
    assert "version: 1" in yaml_body
    assert "MKTAYIAKQR" in yaml_body
    assert "smiles: \"CCO\"" in yaml_body
    assert "properties:" in yaml_body
    assert "affinity:" in yaml_body


def test_parse_boltz_samples_matches_affinity_json_with_structure(tmp_path: Path) -> None:
    results_root = tmp_path / "boltz_results_input"
    preds = results_root / "predictions"
    preds.mkdir(parents=True)
    (preds / "pred_sample_0.cif").write_text("CIF0", encoding="utf-8")
    (preds / "pred_sample_1.cif").write_text("CIF1", encoding="utf-8")
    (preds / "pred_sample_2.cif").write_text("CIF2", encoding="utf-8")

    scores = results_root / "scores"
    scores.mkdir()
    (scores / "aff_0.json").write_text(
        json.dumps({"affinity_pred_value": -5.0, "affinity_probability_binary": 0.6}),
        encoding="utf-8",
    )
    (scores / "aff_1.json").write_text(
        json.dumps({"affinity_pred_value": -7.5, "affinity_probability_binary": 0.92}),
        encoding="utf-8",
    )
    (scores / "aff_2.json").write_text(
        json.dumps({"affinity_pred_value": -6.2, "affinity_probability_binary": 0.75}),
        encoding="utf-8",
    )

    samples = parse_boltz_samples(tmp_path)
    assert len(samples) == 3
    by_index = {s.sample_index: s for s in samples}
    assert by_index[0].affinity_pred_value == -5.0
    assert by_index[1].affinity_pred_value == -7.5
    assert by_index[1].affinity_probability_binary == 0.92


def test_parse_boltz_samples_reads_boltz2_combined_affinity_file(tmp_path: Path) -> None:
    """Boltz 2.x emits a single `affinity_<stem>.json` with flat indexed keys.
    Empty suffix = sample 0, '1' = sample 1, '2' = sample 2."""
    results_root = tmp_path / "boltz_results_input"
    preds = results_root / "predictions" / "input"
    preds.mkdir(parents=True)
    # Three diffusion samples (Boltz 2.x layout: input_model_<N>.pdb).
    (preds / "input_model_0.pdb").write_text("M0", encoding="utf-8")
    (preds / "input_model_1.pdb").write_text("M1", encoding="utf-8")
    (preds / "input_model_2.pdb").write_text("M2", encoding="utf-8")
    (preds / "affinity_input.json").write_text(
        json.dumps(
            {
                "affinity_pred_value": 0.84,
                "affinity_probability_binary": 0.21,
                "affinity_pred_value1": 1.58,
                "affinity_probability_binary1": 0.03,
                "affinity_pred_value2": 0.11,
                "affinity_probability_binary2": 0.40,
            }
        ),
        encoding="utf-8",
    )
    samples = parse_boltz_samples(tmp_path)
    by_index = {s.sample_index: s for s in samples}
    assert by_index[0].affinity_pred_value == 0.84
    assert by_index[0].affinity_probability_binary == 0.21
    assert by_index[1].affinity_pred_value == 1.58
    assert by_index[2].affinity_pred_value == 0.11
    assert by_index[2].affinity_probability_binary == 0.40


def test_parse_boltz_samples_falls_back_when_no_affinity(tmp_path: Path) -> None:
    (tmp_path / "sample_0.cif").write_text("x", encoding="utf-8")
    (tmp_path / "sample_1.cif").write_text("y", encoding="utf-8")
    samples = parse_boltz_samples(tmp_path)
    assert len(samples) == 2
    assert all(s.affinity_pred_value is None for s in samples)


@dataclass
class _MockBackend:
    """Injects a fake Modal result with a multi-sample Boltz output tree."""

    def submit(self, spec: JobSpec) -> JobHandle:
        self._spec = spec
        return JobHandle(handle_id="mock-handle", provider="mock")

    def fetch_result(self, handle: JobHandle) -> JobResult:
        _ = handle
        files = {
            "out/boltz_results_input/predictions/pred_sample_0.cif": b"CIF0",
            "out/boltz_results_input/predictions/pred_sample_1.cif": b"CIF1",
            "out/boltz_results_input/predictions/pred_sample_2.cif": b"CIF2",
            "out/boltz_results_input/scores/aff_0.json": json.dumps(
                {"affinity_pred_value": -4.0, "affinity_probability_binary": 0.3}
            ).encode("utf-8"),
            "out/boltz_results_input/scores/aff_1.json": json.dumps(
                {"affinity_pred_value": -8.2, "affinity_probability_binary": 0.95}
            ).encode("utf-8"),
            "out/boltz_results_input/scores/aff_2.json": json.dumps(
                {"affinity_pred_value": -6.5, "affinity_probability_binary": 0.7}
            ).encode("utf-8"),
        }
        return JobResult(
            success=True,
            stdout="ok",
            stderr="",
            output_files=files,
            elapsed=42.0,
            cost_estimate_usd=0.0245,
        )


@dataclass
class _MockBackendConfidenceOnly:
    """Boltz-1-era output: no affinity head, only ligand_iptm / confidence_score."""

    def submit(self, spec: JobSpec) -> JobHandle:
        self._spec = spec
        return JobHandle(handle_id="mock-handle", provider="mock")

    def fetch_result(self, handle: JobHandle) -> JobResult:
        _ = handle
        files = {
            "out/boltz_results_input/predictions/input/input_model_0.pdb": b"PDB0",
            "out/boltz_results_input/predictions/input/input_model_1.pdb": b"PDB1",
            "out/boltz_results_input/predictions/input/input_model_2.pdb": b"PDB2",
            "out/boltz_results_input/predictions/input/confidence_input_model_0.json": json.dumps(
                {"ligand_iptm": 0.2, "confidence_score": 0.45}
            ).encode("utf-8"),
            "out/boltz_results_input/predictions/input/confidence_input_model_1.json": json.dumps(
                {"ligand_iptm": 0.55, "confidence_score": 0.62}
            ).encode("utf-8"),
            "out/boltz_results_input/predictions/input/confidence_input_model_2.json": json.dumps(
                {"ligand_iptm": 0.4, "confidence_score": 0.5}
            ).encode("utf-8"),
        }
        return JobResult(
            success=True, stdout="ok", stderr="", output_files=files,
            elapsed=50.0, cost_estimate_usd=0.029,
        )


def test_reselect_via_boltz2_falls_back_to_ligand_iptm_when_affinity_absent(
    tmp_path: Path,
) -> None:
    result = reselect_via_boltz2(
        protein_sequence="MKTAYIAKQR",
        ligand_smiles="CCO",
        backend=_MockBackendConfidenceOnly(),  # type: ignore[arg-type]
        diffusion_samples=3,
        output_dir=tmp_path,
    )
    assert result.method == "ligand_iptm_structural_fallback"
    # sample_index=1 has ligand_iptm=0.55 (highest → most confident interface)
    assert result.samples[result.chosen_sample_index].sample_index == 1


def test_reselect_via_boltz2_picks_lowest_pred_value(tmp_path: Path) -> None:
    result = reselect_via_boltz2(
        protein_sequence="MKTAYIAKQR",
        ligand_smiles="CCO",
        backend=_MockBackend(),  # type: ignore[arg-type]
        diffusion_samples=3,
        output_dir=tmp_path,
    )
    assert result.method == "affinity_pred_value"
    # sample_index=1 has the most negative pred_value (-8.2 = tightest binder)
    assert result.samples[result.chosen_sample_index].sample_index == 1
    assert result.cost_estimate_usd == 0.0245
    assert result.chosen_structure_path.exists()
    assert "pred_sample_1.cif" in str(result.chosen_structure_path)


# ---------------------------------------------------------------------------
# Rescorer hook
# ---------------------------------------------------------------------------


@dataclass
class _MockRescorer:
    """Controllable rescorer that returns pre-programmed validity decisions."""

    name: str
    decisions: list[bool]

    def score(self, *, complex_pdb: Path, ligand_smiles: str) -> RescoreResult:
        _ = complex_pdb, ligand_smiles
        idx = self._cursor
        self._cursor += 1  # type: ignore[has-type]
        return RescoreResult(
            valid=self.decisions[idx],
            score=None,
            fail_reasons=() if self.decisions[idx] else ("mock_fail",),
        )

    def __post_init__(self) -> None:
        self._cursor: int = 0


def test_reselect_with_all_valid_rescorer_preserves_4tier_choice(
    tmp_path: Path,
) -> None:
    rescorer = _MockRescorer(name="mock_all_pass", decisions=[True, True, True])
    result = reselect_via_boltz2(
        protein_sequence="MKTAYIAKQR",
        ligand_smiles="CCO",
        backend=_MockBackend(),  # type: ignore[arg-type]
        diffusion_samples=3,
        output_dir=tmp_path,
        rescorer=rescorer,  # type: ignore[arg-type]
    )
    # sample_index=1 still wins (tightest affinity among valid set = full set).
    assert result.samples[result.chosen_sample_index].sample_index == 1
    assert result.method == "mock_all_pass+affinity_pred_value"
    assert len(result.rescore_results) == 3
    assert all(r.valid for r in result.rescore_results)


def test_reselect_with_partial_invalid_rescorer_filters_candidates(
    tmp_path: Path,
) -> None:
    # Mark sample 1 (best affinity) invalid. Next-best among valid is sample 2
    # (affinity_pred_value=-6.5).
    rescorer = _MockRescorer(name="mock", decisions=[True, False, True])
    result = reselect_via_boltz2(
        protein_sequence="MKTAYIAKQR",
        ligand_smiles="CCO",
        backend=_MockBackend(),  # type: ignore[arg-type]
        diffusion_samples=3,
        output_dir=tmp_path,
        rescorer=rescorer,  # type: ignore[arg-type]
    )
    assert result.samples[result.chosen_sample_index].sample_index == 2
    assert result.method == "mock+affinity_pred_value"
    assert result.rescore_results[1].valid is False
    assert "mock_fail" in result.rescore_results[1].fail_reasons


def test_reselect_with_all_invalid_rescorer_falls_back_to_full_set(
    tmp_path: Path,
) -> None:
    # All poses invalid → "no_valid_pose_*" method, 4-tier still runs on
    # full sample set (silent drop forbidden — AC-2b).
    rescorer = _MockRescorer(name="mock", decisions=[False, False, False])
    result = reselect_via_boltz2(
        protein_sequence="MKTAYIAKQR",
        ligand_smiles="CCO",
        backend=_MockBackend(),  # type: ignore[arg-type]
        diffusion_samples=3,
        output_dir=tmp_path,
        rescorer=rescorer,  # type: ignore[arg-type]
    )
    assert result.method == "no_valid_pose_mock+affinity_pred_value"
    # 4-tier on full set still picks sample 1 (best affinity).
    assert result.samples[result.chosen_sample_index].sample_index == 1
    assert all(not r.valid for r in result.rescore_results)


def test_select_best_pose_with_no_rescore_matches_legacy_behavior() -> None:
    # Pure selection helper: rescorer-free path returns existing method label
    # (no prefix) — proves reselect_via_boltz2's no-rescorer path is unchanged.
    from molforge.docking.multi_pose import PoseWithAffinity

    samples = [
        PoseWithAffinity(
            sample_index=i,
            structure_path=Path(f"/tmp/x_{i}.cif"),
            affinity_pred_value=val,
            affinity_probability_binary=None,
        )
        for i, val in enumerate([-4.0, -8.2, -6.5])
    ]
    idx, method = select_best_pose(samples)
    assert idx == 1
    assert method == "affinity_pred_value"


def test_select_best_pose_uses_float_score_when_rescorer_returns_numeric() -> None:
    # Phase 5 score-based path: argmin over valid poses on `score` (lower=better).
    # Rescorer picks sample 2 (score=-9.5) even though Boltz's affinity
    # would pick sample 0 (affinity_pred_value=-8.0).
    from molforge.docking.multi_pose import PoseWithAffinity

    samples = [
        PoseWithAffinity(
            sample_index=i,
            structure_path=Path(f"/tmp/x_{i}.cif"),
            affinity_pred_value=val,
            affinity_probability_binary=None,
        )
        for i, val in enumerate([-8.0, -5.0, -4.0])
    ]
    rescore_results = [
        RescoreResult(valid=True, score=-6.0),
        RescoreResult(valid=True, score=-7.0),
        RescoreResult(valid=True, score=-9.5),  # best per rescorer
    ]
    idx, method = select_best_pose(
        samples,
        rescore_results=rescore_results,
        rescorer_name="vina_score_only",
    )
    assert idx == 2
    assert method == "vina_score_only_score"


def test_select_best_pose_filters_invalid_before_score_ranking() -> None:
    # Sample with best score is invalid → pick second-best among valid.
    from molforge.docking.multi_pose import PoseWithAffinity

    samples = [
        PoseWithAffinity(
            sample_index=i,
            structure_path=Path(f"/tmp/x_{i}.cif"),
            affinity_pred_value=-5.0,
            affinity_probability_binary=None,
        )
        for i in range(3)
    ]
    rescore_results = [
        RescoreResult(valid=True, score=-6.0),
        RescoreResult(valid=False, score=-9.9),  # best score but invalid
        RescoreResult(valid=True, score=-7.0),
    ]
    idx, method = select_best_pose(
        samples,
        rescore_results=rescore_results,
        rescorer_name="vina",
    )
    assert idx == 2
    assert method == "vina_score"
