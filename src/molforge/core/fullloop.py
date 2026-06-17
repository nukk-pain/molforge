"""v5 Track I1 — `molforge fullloop` orchestrator.

Glue layer that runs three pipelines in sequence:

  1. biocompute discover  — external uv subprocess, emits
     `neuroregen_targets.json`
  2. molforge run_batch   — in-process, uses existing run_batch +
     wet_lab_report
  3. neuroregen pipeline  — external cargo subprocess, emits mRNA
     designs JSON

The orchestrator owns:
  * stage-atomic resume via a `fullloop_summary.json` on disk
  * per-stage stdout/stderr persistence for debugging (AC-I1-4)
  * cost budget tracking with soft cap (AC-I1-cost)
  * mutual-exclusive `--skip-mrna` / `neuroregen_dir` enforcement
    handled at the CLI layer; `run_fullloop` treats `skip_mrna=True`
    or `neuroregen_dir=None` symmetrically as "skip stage"
  * injectable runners so tests can bypass real subprocess calls

The three stage adapters (`_biocompute_runner_default`,
`_molforge_runner_default`, `_neuroregen_runner_default`) are the
production implementations. Tests pass their own mocks via the
`runners` kwarg.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..utils.subprocess_helpers import (
    run_streaming_subprocess,
    safe_env as _safe_env,
)


SCHEMA_VERSION = "fullloop/v2"
STAGES = ("biocompute", "molforge", "neuroregen")


StageRunner = Callable[..., dict]


@dataclass(frozen=True, slots=True)
class FullloopResult:
    output_dir: Path
    stage_status: dict[
        str, str
    ]  # "completed" | "partial" | "skipped" | "failed:<reason>" | "budget_exceeded"
    stage_wall_seconds: dict[str, float]
    total_cost_usd: float
    biocompute_run_dir: Path | None
    molforge_wet_lab_report_path: Path | None
    neuroregen_designs_path: Path | None
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Summary persistence
# ---------------------------------------------------------------------------


def _summary_path(output_dir: Path) -> Path:
    return output_dir / "fullloop_summary.json"


def _load_summary(output_dir: Path) -> dict | None:
    path = _summary_path(output_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != SCHEMA_VERSION:
        return None
    return data


def _write_summary(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _summary_path(output_dir).write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _rotate_stage_logs(stage_dir: Path) -> None:
    """Move existing stdout.log / stderr.log out of the way before a resume
    rerun so we keep both the old (failed) run and the new (successful) run
    on disk for debugging."""
    if not stage_dir.exists():
        return
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S")
    for name in ("stdout.log", "stderr.log"):
        p = stage_dir / name
        if p.exists():
            p.rename(stage_dir / f"{name}.backup-{ts}")


# ---------------------------------------------------------------------------
# Default stage runners (production subprocess implementations)
# ---------------------------------------------------------------------------


def _biocompute_runner_default(
    *,
    disease: str,
    description: str | None,
    keywords: tuple,
    generations: int,
    population_size: int,
    output_dir: Path,
    biocompute_dir: Path,
) -> dict:
    """Invoke biocompute's `discover` CLI via `uv run`.

    biocompute discover:
      * requires one of `-d/--description` or `--tissue/--phenotype/--pathology`
      * auto-emits `<run_dir>/neuroregen_targets.json` so no separate export
        subprocess is needed
      * prints the run_dir path to stdout on success
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    # biocompute subprocess has `cwd=biocompute_dir`, so passing a relative
    # `--output-dir` would resolve under biocompute's tree instead of
    # fullloop's artifact tree. Pass an absolute path to force the
    # subprocess to write into our fullloop directory.
    abs_output_dir = output_dir.resolve()
    args: list[str] = ["uv", "run", "biocompute", "discover", disease]
    if description:
        args += ["-d", description]
    for kw in keywords:
        args += ["-k", kw]
    args += [
        "-g",
        str(generations),
        "-p",
        str(population_size),
        "-o",
        str(abs_output_dir),
    ]

    # Use the streaming subprocess helper so stdout/stderr survive
    # a timeout kill, env is allowlist-restricted, and subprocess cwd is
    # honoured verbatim.
    result = run_streaming_subprocess(
        args,
        cwd=biocompute_dir,
        stdout_path=output_dir / "stdout.log",
        stderr_path=output_dir / "stderr.log",
        env=_safe_env(),
        timeout=3600,
    )
    if result.timed_out:
        return {
            "success": False,
            "run_dir": None,
            "neuroregen_targets_path": None,
            "stdout": result.stdout_text,
            "stderr": "biocompute timeout",
            "exit_code": -1,
            "elapsed_seconds": result.elapsed_seconds,
        }
    returncode = result.returncode
    elapsed = result.elapsed_seconds
    stdout_text = result.stdout_text
    stderr_text = result.stderr_text

    if returncode != 0:
        return {
            "success": False,
            "run_dir": None,
            "neuroregen_targets_path": None,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "exit_code": returncode,
            "elapsed_seconds": elapsed,
        }

    # Find the run_dir discover created inside `output_dir`. biocompute writes
    # under `output_dir/<run_id>/`; pick the newest directory containing
    # neuroregen_targets.json.
    candidates = [
        p
        for p in output_dir.iterdir()
        if p.is_dir() and (p / "neuroregen_targets.json").exists()
    ]
    if not candidates:
        return {
            "success": False,
            "run_dir": None,
            "neuroregen_targets_path": None,
            "stdout": stdout_text,
            "stderr": "biocompute exited 0 but no neuroregen_targets.json found",
            "exit_code": 0,
            "elapsed_seconds": elapsed,
        }
    run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    return {
        "success": True,
        "run_dir": run_dir,
        "neuroregen_targets_path": run_dir / "neuroregen_targets.json",
        "stdout": stdout_text,
        "stderr": stderr_text,
        "exit_code": 0,
        "elapsed_seconds": elapsed,
    }


def _molforge_runner_default(
    *,
    neuroregen_targets_path: Path,
    output_dir: Path,
    disease: str,
    top_n: int,
) -> dict:
    """In-process stage: load target candidates, call `run_batch`, emit
    wet-lab report."""
    from .input import load_target_candidates  # lazy to avoid CLI import cost
    from .multi_target_batch import run_batch
    from ..output.wet_lab_report import build_report, write_report

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split the combined targets JSON into per-target files, because
    # run_batch's interface is `input_paths: list[Path]`.
    target_dicts = json.loads(neuroregen_targets_path.read_text(encoding="utf-8"))
    # Cap at top_n (targets are already ranked upstream).
    target_dicts = target_dicts[:top_n]

    per_target_inputs = []
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for t in target_dicts:
        gene = t.get("gene", f"target_{len(per_target_inputs)}")
        p = inputs_dir / f"{gene}.json"
        p.write_text(json.dumps([t]), encoding="utf-8")
        per_target_inputs.append(p)

    started = time.perf_counter()
    batch_result = run_batch(
        input_paths=per_target_inputs,
        disease=disease,
        store_root=output_dir / "store",
        top_n=10,
    )
    elapsed = round(time.perf_counter() - started, 2)

    total_cost = 0.0
    per_target = []
    for tr in batch_result.per_target:
        # Cost plumbing: `run_batch` uses the local v1/v2 pipeline (Vina +
        # REINVENT + ADMET-AI) which is CPU-only and incurs $0. Rescore
        # runs (Boltz-2 Modal rescoring) record cost under
        # RankedCandidate.provenance.rescore_cost_estimate_usd.
        cost = 0.0
        if tr.run and tr.run.candidates:
            for cand in tr.run.candidates:
                prov = getattr(cand, "provenance", None) or {}
                if isinstance(prov, dict):
                    raw = prov.get("rescore_cost_estimate_usd", 0.0) or 0.0
                    try:
                        cost += float(raw)
                    except (TypeError, ValueError):
                        pass
        total_cost += cost
        per_target.append(
            {
                "target_gene": tr.target_gene,
                "status": tr.status,
                "cost_estimate_usd": cost,
                "candidate_count": tr.candidate_count,
            }
        )

    # Cross-target summary (standalone JSON — complements the per-target stores)
    (output_dir / "cross_target_summary.json").write_text(
        json.dumps(
            {
                "per_target": per_target,
                "total_cost_usd": total_cost,
                "unique_smiles_count": batch_result.unique_smiles_count,
                "shared_smiles": batch_result.shared_smiles_across_targets,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Wet-lab handoff
    wet_lab_report_path = output_dir / "wet_lab_report.json"
    try:
        report = build_report(batch_result, disease=disease)
        write_report(report, wet_lab_report_path)
    except Exception as exc:  # noqa: BLE001 — non-fatal wet-lab formatting failure
        (output_dir / "wet_lab_report.error.log").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        wet_lab_report_path = None

    completed = sum(1 for p in per_target if p["status"] == "completed")
    return {
        "success": completed > 0,
        "per_target": per_target,
        "total_cost_usd": total_cost,
        "completed_count": completed,
        "wet_lab_report_path": wet_lab_report_path,
        "elapsed_seconds": elapsed,
    }


def _parse_neuroregen_designs(stdout: str) -> list[dict]:
    """Replicates biocompute/cli.py:113 `_parse_neuroregen_designs`.

    neuroregen may print a short informational prefix before the JSON payload.
    Strip non-JSON lines and parse the trailing JSON.
    """
    text = stdout.strip()
    if not text:
        return []
    # Try direct parse first.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to line-stripping prefix noise.
        lines = text.splitlines()
        for i, _ in enumerate(lines):
            candidate = "\n".join(lines[i:])
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        else:
            return []
    if isinstance(data, dict):
        if "designs" in data:
            return data["designs"]
        if "mrna_designs" in data:
            return data["mrna_designs"]
        return [data]
    if isinstance(data, list):
        return data
    return []


def _neuroregen_runner_default(
    *,
    neuroregen_targets_path: Path,
    output_dir: Path,
    neuroregen_dir: Path,
    top_n: int,
) -> dict:
    """Invoke neuroregen's Rust pipeline via `cargo run -p pipeline -- run`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # neuroregen runs with cwd=neuroregen_dir, so relative paths wouldn't
    # resolve. Resolve to absolute so the target file lookup works.
    abs_targets_path = Path(neuroregen_targets_path).resolve()
    args = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "pipeline",
        "--",
        "run",
        "--targets-file",
        str(abs_targets_path),
        "--top-n",
        str(top_n),
        "--cds",
        "auto",
        "--utr5",
        "alpha-globin",
        "--utr3",
        "alpha-globin",
        "--format",
        "json",
    ]

    result = run_streaming_subprocess(
        args,
        cwd=neuroregen_dir,
        stdout_path=output_dir / "stdout.log",
        stderr_path=output_dir / "stderr.log",
        env=_safe_env(),
        timeout=300,
    )
    if result.timed_out:
        return {
            "success": False,
            "designs_path": None,
            "stderr": "neuroregen timeout",
            "exit_code": -1,
            "elapsed_seconds": result.elapsed_seconds,
        }
    returncode = result.returncode
    elapsed = result.elapsed_seconds
    stdout_text = result.stdout_text
    stderr_text = result.stderr_text

    if returncode != 0:
        return {
            "success": False,
            "designs_path": None,
            "stderr": stderr_text[-2000:],
            "exit_code": returncode,
            "elapsed_seconds": elapsed,
        }

    designs = _parse_neuroregen_designs(stdout_text)
    designs_path = output_dir / "mrna_designs.json"
    designs_path.write_text(json.dumps(designs, indent=2), encoding="utf-8")
    return {
        "success": True,
        "designs_path": designs_path,
        "stderr": "",
        "exit_code": 0,
        "elapsed_seconds": elapsed,
    }


_DEFAULT_RUNNERS: dict[str, StageRunner] = {
    "biocompute": _biocompute_runner_default,
    "molforge": _molforge_runner_default,
    "neuroregen": _neuroregen_runner_default,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _stage_dir(output_dir: Path, stage: str) -> Path:
    return output_dir / stage


def _failed_reason(exc: BaseException) -> str:
    return f"failed:{type(exc).__name__}"


def _resume_stage_status(loaded: dict | None, stage: str) -> str:
    if loaded is None:
        return "pending"
    return (loaded.get("stage_status") or {}).get(stage, "pending")


def _resume_stage_detail(loaded: dict | None, stage: str) -> dict:
    if loaded is None:
        return {}
    details = loaded.get("stage_details") or {}
    stage_detail = details.get(stage)
    return stage_detail if isinstance(stage_detail, dict) else {}


def _target_gene(target: dict, index: int) -> str:
    return str(target.get("gene") or f"target_{index}")


def _load_targets(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected target list in {path}, got {type(data).__name__}")
    return [item for item in data if isinstance(item, dict)]


def _write_resume_targets(
    *,
    stage_dir: Path,
    targets: list[dict[str, Any]],
    pending_genes: set[str],
) -> Path:
    filtered = [
        target
        for index, target in enumerate(targets)
        if _target_gene(target, index) in pending_genes
    ]
    resume_path = stage_dir / "resume_targets.json"
    resume_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
    return resume_path


def _merge_molforge_per_target(
    *,
    prior_entries: list[dict[str, Any]],
    current_entries: list[dict[str, Any]],
    ordered_genes: list[str],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for entry in prior_entries:
        gene = entry.get("target_gene")
        if isinstance(gene, str):
            merged[gene] = dict(entry)
    for entry in current_entries:
        gene = entry.get("target_gene")
        if isinstance(gene, str):
            merged[gene] = dict(entry)
    return [merged[gene] for gene in ordered_genes if gene in merged]


def _molforge_stage_detail(
    *,
    per_target: list[dict[str, Any]],
    ordered_genes: list[str],
    resumed_only_genes: list[str] | None,
) -> dict[str, Any]:
    completed = [
        entry["target_gene"]
        for entry in per_target
        if entry.get("status") == "completed"
    ]
    pending = [
        entry["target_gene"]
        for entry in per_target
        if entry.get("status") != "completed"
    ]
    return {
        "target_count": len(ordered_genes),
        "completed_count": len(completed),
        "completed_target_genes": completed,
        "pending_target_genes": pending,
        "per_target": per_target,
        "resumed_only_target_genes": resumed_only_genes or [],
    }


def run_fullloop(
    *,
    disease: str,
    description: str | None,
    keywords: tuple,
    biocompute_dir: Path,
    output_dir: Path,
    neuroregen_dir: Path | None = None,
    skip_mrna: bool = False,
    top_n: int = 5,
    generations: int = 10,
    population_size: int = 10,
    cost_budget_usd: float = 5.0,
    resume: bool = False,
    runners: dict[str, StageRunner] | None = None,
) -> FullloopResult:
    """Run the full disease→mRNA loop."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    biocompute_dir = Path(biocompute_dir)
    neuroregen_dir = Path(neuroregen_dir) if neuroregen_dir else None
    runners = {**_DEFAULT_RUNNERS, **(runners or {})}

    prior = _load_summary(output_dir) if resume else None

    stage_status: dict[str, str] = {s: "pending" for s in STAGES}
    stage_wall: dict[str, float] = {s: 0.0 for s in STAGES}
    stage_details: dict[str, dict[str, Any]] = {s: {} for s in STAGES}
    errors: list[str] = []
    total_cost = 0.0
    biocompute_run_dir: Path | None = None
    neuroregen_targets_path: Path | None = None
    wet_lab_report_path: Path | None = None
    designs_path: Path | None = None

    if prior:
        # Preserve completed stage outputs from prior run.
        stage_status.update(prior.get("stage_status") or {})
        stage_wall.update(prior.get("stage_wall_seconds") or {})
        for stage in STAGES:
            stage_details[stage] = _resume_stage_detail(prior, stage)
        total_cost = float(prior.get("total_cost_usd") or 0.0)
        if prior.get("biocompute_run_dir"):
            biocompute_run_dir = Path(prior["biocompute_run_dir"])
            if biocompute_run_dir.exists():
                ntp = biocompute_run_dir / "neuroregen_targets.json"
                if ntp.exists():
                    neuroregen_targets_path = ntp
        if prior.get("neuroregen_designs_path"):
            p = Path(prior["neuroregen_designs_path"])
            if p.exists():
                designs_path = p
        if prior.get("molforge_wet_lab_report_path"):
            p = Path(prior["molforge_wet_lab_report_path"])
            if p.exists():
                wet_lab_report_path = p

    # --- biocompute stage -------------------------------------------------
    if _resume_stage_status(prior, "biocompute") == "completed":
        pass  # skip — already done
    else:
        _rotate_stage_logs(_stage_dir(output_dir, "biocompute"))
        stage_started = time.perf_counter()
        try:
            res = runners["biocompute"](
                disease=disease,
                description=description,
                keywords=tuple(keywords),
                generations=generations,
                population_size=population_size,
                output_dir=_stage_dir(output_dir, "biocompute"),
                biocompute_dir=biocompute_dir,
            )
            stage_wall["biocompute"] = round(time.perf_counter() - stage_started, 2)
            if res.get("success"):
                stage_status["biocompute"] = "completed"
                biocompute_run_dir = res.get("run_dir")
                neuroregen_targets_path = res.get("neuroregen_targets_path")
            else:
                stage_status["biocompute"] = f"failed:exit_{res.get('exit_code', -1)}"
                errors.append(f"biocompute: {(res.get('stderr') or '')[-500:]}")
        except Exception as exc:  # noqa: BLE001 — orchestrator catch-all
            stage_wall["biocompute"] = round(time.perf_counter() - stage_started, 2)
            stage_status["biocompute"] = _failed_reason(exc)
            errors.append(f"biocompute: {type(exc).__name__}: {exc}")
            errors.append(traceback.format_exc())

    # --- molforge stage ---------------------------------------------------
    if _resume_stage_status(prior, "molforge") == "completed":
        pass
    elif stage_status["biocompute"] != "completed":
        stage_status["molforge"] = "skipped"
    elif neuroregen_targets_path is None or not neuroregen_targets_path.exists():
        stage_status["molforge"] = "skipped"
        errors.append("molforge skipped — neuroregen_targets.json missing")
    else:
        _rotate_stage_logs(_stage_dir(output_dir, "molforge"))
        stage_started = time.perf_counter()
        try:
            all_targets = _load_targets(neuroregen_targets_path)
            ordered_genes = [
                _target_gene(target, index) for index, target in enumerate(all_targets)
            ]
            prior_molforge_detail = _resume_stage_detail(prior, "molforge")
            pending_genes = set(prior_molforge_detail.get("pending_target_genes") or [])
            resumed_only_genes: list[str] | None = None
            molforge_targets_path = neuroregen_targets_path
            if (
                resume
                and _resume_stage_status(prior, "molforge") == "partial"
                and pending_genes
            ):
                molforge_targets_path = _write_resume_targets(
                    stage_dir=_stage_dir(output_dir, "molforge"),
                    targets=all_targets,
                    pending_genes=pending_genes,
                )
                resumed_only_genes = sorted(pending_genes)
            res = runners["molforge"](
                neuroregen_targets_path=molforge_targets_path,
                output_dir=_stage_dir(output_dir, "molforge"),
                disease=disease,
                top_n=top_n,
            )
            stage_wall["molforge"] = round(time.perf_counter() - stage_started, 2)
            total_cost += float(res.get("total_cost_usd") or 0.0)
            wet_lab_report_path = res.get("wet_lab_report_path")
            current_per_target = list(res.get("per_target") or [])
            if resumed_only_genes:
                per_target = _merge_molforge_per_target(
                    prior_entries=list(prior_molforge_detail.get("per_target") or []),
                    current_entries=current_per_target,
                    ordered_genes=ordered_genes,
                )
            else:
                per_target = current_per_target
            stage_details["molforge"] = _molforge_stage_detail(
                per_target=per_target,
                ordered_genes=ordered_genes,
                resumed_only_genes=resumed_only_genes,
            )
            completed_count = stage_details["molforge"]["completed_count"]
            target_count = stage_details["molforge"]["target_count"]
            if total_cost > cost_budget_usd:
                stage_status["molforge"] = "budget_exceeded"
                errors.append(
                    f"molforge cost {total_cost:.2f} > budget {cost_budget_usd:.2f}"
                )
            elif completed_count == target_count and target_count > 0:
                stage_status["molforge"] = "completed"
            elif completed_count > 0:
                stage_status["molforge"] = "partial"
            else:
                stage_status["molforge"] = "failed:no_targets_completed"
        except Exception as exc:  # noqa: BLE001
            stage_wall["molforge"] = round(time.perf_counter() - stage_started, 2)
            stage_status["molforge"] = _failed_reason(exc)
            errors.append(f"molforge: {type(exc).__name__}: {exc}")
            errors.append(traceback.format_exc())

    # --- neuroregen stage -------------------------------------------------
    if _resume_stage_status(prior, "neuroregen") == "completed":
        pass
    elif skip_mrna or neuroregen_dir is None:
        stage_status["neuroregen"] = "skipped"
    elif stage_status["molforge"] != "completed":
        stage_status["neuroregen"] = "skipped"
    elif neuroregen_targets_path is None:
        stage_status["neuroregen"] = "skipped"
    else:
        _rotate_stage_logs(_stage_dir(output_dir, "neuroregen"))
        stage_started = time.perf_counter()
        try:
            res = runners["neuroregen"](
                neuroregen_targets_path=neuroregen_targets_path,
                output_dir=_stage_dir(output_dir, "neuroregen"),
                neuroregen_dir=neuroregen_dir,
                top_n=top_n,
            )
            stage_wall["neuroregen"] = round(time.perf_counter() - stage_started, 2)
            if res.get("success"):
                stage_status["neuroregen"] = "completed"
                designs_path = res.get("designs_path")
            else:
                stage_status["neuroregen"] = f"failed:exit_{res.get('exit_code', -1)}"
                errors.append(f"neuroregen: {(res.get('stderr') or '')[-500:]}")
        except Exception as exc:  # noqa: BLE001
            stage_wall["neuroregen"] = round(time.perf_counter() - stage_started, 2)
            stage_status["neuroregen"] = _failed_reason(exc)
            errors.append(f"neuroregen: {type(exc).__name__}: {exc}")
            errors.append(traceback.format_exc())

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "disease": disease,
        "stage_status": stage_status,
        "stage_details": stage_details,
        "stage_wall_seconds": stage_wall,
        "total_cost_usd": round(total_cost, 6),
        "cost_budget_usd": cost_budget_usd,
        "biocompute_run_dir": str(biocompute_run_dir) if biocompute_run_dir else None,
        "neuroregen_targets_path": (
            str(neuroregen_targets_path) if neuroregen_targets_path else None
        ),
        "molforge_wet_lab_report_path": (
            str(wet_lab_report_path) if wet_lab_report_path else None
        ),
        "neuroregen_designs_path": (str(designs_path) if designs_path else None),
        "errors": errors,
    }
    _write_summary(output_dir, summary)

    return FullloopResult(
        output_dir=output_dir,
        stage_status=stage_status,
        stage_wall_seconds=stage_wall,
        total_cost_usd=round(total_cost, 6),
        biocompute_run_dir=biocompute_run_dir,
        molforge_wet_lab_report_path=wet_lab_report_path,
        neuroregen_designs_path=designs_path,
        errors=errors,
    )
