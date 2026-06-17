# pyright: reportMissingImports=false
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from argparse import Namespace
from enum import Enum
from pathlib import Path

from contracts.schema import AffinityPrediction, TargetCandidate

from .admet.scorer import ADMETScorer
from .core.fullloop import run_fullloop
from .core.input import load_target_candidates
from .core.pipeline import (
    load_binding_pocket,
    run_admet_pipeline,
    run_generate_stage,
    run_pipeline,
)
from .core.run_io import load_pipeline_run_artifact, persist_pipeline_run
from .core.store import MolforgeStore
from .core.writer import write_ranked_candidates
from .docking.rescore import rescore_predictions
from .docking.module import DockingRunner
from .remote import build_remote_backend

VERSION = "0.1.0"
PHASE_STUB_MESSAGE = "not yet implemented"


def main(argv: list[str] | None = None) -> int:
    load_dotenv_file(Path(".env"))
    parser = build_parser()
    args: Namespace = parser.parse_args(argv)

    if args.version:
        print(f"molforge {VERSION}")
        return 0

    command = getattr(args, "command", None)
    if command is None:
        parser.print_help()
        return 0

    if command == "run":
        candidates = load_target_candidates(Path(args.input_json), disease=args.disease)
        output_path = Path(args.output)
        db_path = Path(args.db_path)
        with MolforgeStore(db_path) as store:
            run = run_pipeline(
                candidates,
                store=store,
                top_n=args.top,
                enable_live_chembl=bool(args.enable_live_chembl),
                enable_evebio=bool(args.enable_evebio),
            )
        write_ranked_candidates(run, output_path)
        print(
            f"Loaded {len(candidates)} target candidates from {args.input_json}. "
            f"Created run {run.run_id} with {len(run.candidates)} candidates. "
            f"Wrote {output_path}."
        )
        return 0

    if command == "generate":
        pocket = load_binding_pocket(args.pocket)
        result = run_generate_stage(
            pocket,
            output_dir=args.output_dir,
            n=args.count,
            seed_smiles=args.seed_smiles,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if command == "admet":
        try:
            return run_admet_command(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if command == "dock":
        candidates = load_target_candidates(Path(args.input_json), disease=args.disease)
        output_path = Path(args.output)
        db_path = Path(args.db_path)
        with MolforgeStore(db_path) as store:
            runner = DockingRunner()
            predictions = runner.run(candidates, store=store, top_n=args.top)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                [_to_jsonable(item) for item in predictions], indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"Docked {len(candidates)} target candidates. "
            f"Produced {len(predictions)} predictions. Wrote {output_path}."
        )
        return 0

    if command == "rescore":
        try:
            return run_rescore_command(args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if command == "verify":
        print("Verification command reserved for later phases.")
        return 0

    if command == "status":
        print(read_status_summary())
        return 0

    if command == "api":
        from .api import serve

        serve(host=args.host, port=args.port, db_path=args.db_path)
        return 0

    if command == "fullloop":
        return run_fullloop_command(args)

    parser.error(f"Unknown command: {command}")
    return 2


def run_fullloop_command(args: Namespace) -> int:
    """v5 Track I1 — orchestrate biocompute → molforge → neuroregen.

    Exit codes (AC-I1-cost + partial failure):
      0 = all stages completed
      1 = at least one stage failed (not budget_exceeded)
      3 = cost budget exceeded (molforge stage flagged `budget_exceeded`)
    """
    # Argparse already enforces --skip-mrna XOR --neuroregen-dir below.
    result = run_fullloop(
        disease=args.disease,
        description=args.description,
        keywords=tuple(args.keywords or ()),
        biocompute_dir=Path(args.biocompute_dir),
        neuroregen_dir=Path(args.neuroregen_dir) if args.neuroregen_dir else None,
        skip_mrna=args.skip_mrna,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        generations=args.generations,
        population_size=args.population_size,
        cost_budget_usd=args.cost_budget_usd,
        resume=args.resume,
    )
    # Print a compact summary
    print(f"fullloop summary ({result.output_dir}):")
    for stage, status in result.stage_status.items():
        wall = result.stage_wall_seconds.get(stage, 0.0)
        print(f"  {stage:<12} {status:<30} wall={wall:.1f}s")
    print(f"  total_cost_usd={result.total_cost_usd:.4f}")
    if result.errors:
        print("errors:")
        for err in result.errors[:10]:
            print(f"  - {err[:200]}")

    if any(s == "budget_exceeded" for s in result.stage_status.values()):
        return 3
    if any(s.startswith("failed:") for s in result.stage_status.values()):
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molforge")
    _ = parser.add_argument(
        "--version", action="store_true", help="show version and exit"
    )

    subparsers = parser.add_subparsers(dest="command")
    common = argparse.ArgumentParser(add_help=False)
    _ = common.add_argument("--db-path", default="molforge.db")
    _ = common.add_argument("--output", default="archive/runs/latest.json")
    _ = common.add_argument("--top", type=int, default=10)
    _ = common.add_argument("-v", action="count", default=0)

    run_parser = subparsers.add_parser("run", parents=[common])
    _ = run_parser.add_argument("input_json")
    _ = run_parser.add_argument("--disease", default=None)
    _ = run_parser.add_argument("--enable-live-chembl", action="store_true")
    _ = run_parser.add_argument("--enable-evebio", action="store_true")

    dock_parser = subparsers.add_parser("dock", parents=[common])
    _ = dock_parser.add_argument("input_json")
    _ = dock_parser.add_argument("--disease", default=None)
    generate_parser = subparsers.add_parser("generate", parents=[common])
    _ = generate_parser.add_argument("pocket")
    _ = generate_parser.add_argument("--output-dir", required=True)
    _ = generate_parser.add_argument("--count", type=int, default=100)
    _ = generate_parser.add_argument("--seed-smiles", default=None)
    admet_parser = subparsers.add_parser("admet", parents=[common])
    _ = admet_parser.add_argument("smiles_csv")
    _ = admet_parser.add_argument("--enable-live-chembl", action="store_true")
    _ = admet_parser.add_argument("--enable-evebio", action="store_true")
    rescore_parser = subparsers.add_parser("rescore", parents=[common])
    _ = rescore_parser.add_argument("run_id_or_json")
    _ = subparsers.add_parser("verify", parents=[common])
    _ = subparsers.add_parser("status", parents=[common])
    api_parser = subparsers.add_parser("api", parents=[common])
    _ = api_parser.add_argument("--host", default="127.0.0.1")
    _ = api_parser.add_argument("--port", type=int, default=8000)

    # v5 Track I1 — disease-to-mRNA one-command orchestrator.
    fullloop_parser = subparsers.add_parser(
        "fullloop",
        help="Run biocompute discover → molforge pose+ADMET → neuroregen mRNA design",
    )
    _ = fullloop_parser.add_argument(
        "disease", help="Disease name (first positional arg for biocompute)"
    )
    _ = fullloop_parser.add_argument(
        "-d", "--description", default=None,
        help="Disease pathophysiology description (required by biocompute "
             "discover unless --tissue/--phenotype/--pathology used upstream)",
    )
    _ = fullloop_parser.add_argument(
        "-k", "--keywords", action="append", default=[],
        help="Search keyword(s); repeatable",
    )
    _ = fullloop_parser.add_argument(
        "-g", "--generations", type=int, default=10,
    )
    _ = fullloop_parser.add_argument(
        "-p", "--population-size", type=int, default=10,
    )
    _ = fullloop_parser.add_argument(
        "--biocompute-dir", required=True,
        help="Path to the biocompute repo (for example, /path/to/biocompute)",
    )
    mrna_group = fullloop_parser.add_mutually_exclusive_group(required=True)
    _ = mrna_group.add_argument(
        "--neuroregen-dir",
        help="Path to the neuroregen repo (enables mRNA design stage)",
    )
    _ = mrna_group.add_argument(
        "--skip-mrna", action="store_true",
        help="Skip the neuroregen mRNA design stage",
    )
    _ = fullloop_parser.add_argument(
        "--output-dir",
        default="archive/runs/fullloop",
        help="Artifact root (default: archive/runs/fullloop)",
    )
    _ = fullloop_parser.add_argument(
        "--top-n", type=int, default=5,
        help="Top-N targets from biocompute to pipe into molforge + neuroregen",
    )
    _ = fullloop_parser.add_argument(
        "--cost-budget-usd", type=float, default=5.0,
        help="Soft cap on Modal cost for molforge stage (fullloop exits 3 on exceed)",
    )
    _ = fullloop_parser.add_argument(
        "--resume", action="store_true",
        help="Skip stages already marked `completed` in fullloop_summary.json",
    )
    return parser


def run_admet_command(args: Namespace) -> int:
    ligands, affinities = load_ligands_from_csv(Path(args.smiles_csv))
    output_path = Path(args.output)
    if not affinities:
        scorer = ADMETScorer()
        profiles = scorer.score_batch(ligands)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"profiles": [profile.__dict__ for profile in profiles]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Scored {len(ligands)} ligands and wrote {output_path}.")
        return 0

    with MolforgeStore(Path(args.db_path)) as store:
        run = run_admet_pipeline(
            target=build_cli_target_from_affinities(affinities),
            ligands=ligands,
            affinities=affinities,
            store=store,
            top_n=args.top,
            enable_live_chembl=bool(args.enable_live_chembl),
            enable_evebio=bool(args.enable_evebio),
        )
    write_ranked_candidates(run, output_path)
    print(
        f"Scored {len(ligands)} ligands from {args.smiles_csv}. "
        f"Created run {run.run_id} with {len(run.candidates)} ranked candidates. "
        f"Wrote {output_path}."
    )
    return 0


def run_rescore_command(args: Namespace) -> int:
    run_id_or_json = str(args.run_id_or_json)
    output_path = Path(args.output)
    db_path = Path(args.db_path)
    with MolforgeStore(db_path) as store:
        run = _load_run_for_rescore(run_id_or_json, store=store)
        affinities = [
            candidate.affinity
            for candidate in run.candidates
            if candidate.affinity is not None
        ]
        rescored_affinities, total_cost = rescore_predictions(
            affinities,
            backend=build_remote_backend(),
        )
        rescored_by_smiles = {
            affinity.ligand_smiles: affinity for affinity in rescored_affinities
        }
        rescored_candidates = []
        for candidate in run.candidates:
            affinity = candidate.affinity
            updated_affinity = (
                None if affinity is None else rescored_by_smiles[affinity.ligand_smiles]
            )
            updated_provenance = dict(candidate.provenance)
            updated_provenance["rescored_from"] = run.run_id
            updated_provenance["rescore_cost_estimate_usd"] = round(
                total_cost / max(len(run.candidates), 1), 6
            )
            rescored_candidates.append(
                candidate.__class__(
                    ligand=candidate.ligand,
                    target=candidate.target,
                    affinity=updated_affinity,
                    admet=candidate.admet,
                    off_targets=candidate.off_targets,
                    composite_score=candidate.composite_score,
                    rank=candidate.rank,
                    provenance=updated_provenance,
                )
            )
        persisted_run = persist_pipeline_run(
            store,
            type(run)(
                run_id=run.run_id,
                input_target=run.input_target,
                started_at=run.started_at,
                completed_at=run.completed_at,
                candidates=rescored_candidates,
                config_hash=run.config_hash,
                schema_version=run.schema_version,
            ),
        )
    write_ranked_candidates(persisted_run, output_path)
    print(
        f"Rescored {len(rescored_affinities)} predictions from {run_id_or_json}. "
        f"Created run {persisted_run.run_id} from {run.run_id}. Wrote {output_path}."
    )
    return 0


def _load_run_for_rescore(run_id_or_json: str, *, store: MolforgeStore):
    candidate_path = Path(run_id_or_json)
    if candidate_path.exists():
        return load_pipeline_run_artifact(candidate_path)
    return store.load_run(run_id_or_json)


def load_ligands_from_csv(csv_path: Path):
    from contracts.schema import Ligand

    ligands: list[Ligand] = []
    affinities: list[AffinityPrediction] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            if not smiles:
                continue
            ligands.append(
                Ligand(
                    smiles=smiles,
                    source=(row.get("source") or "user").strip() or "user",
                    chembl_id=(row.get("chembl_id") or "").strip() or None,
                )
            )
            vina_score = (row.get("vina_score") or "").strip()
            if vina_score:
                affinities.append(
                    AffinityPrediction(
                        ligand_smiles=smiles,
                        target_gene=(row.get("target_gene") or "UNKNOWN").strip()
                        or "UNKNOWN",
                        vina_score=float(vina_score),
                        affinity_log_ki=parse_optional_float(
                            row.get("affinity_log_ki")
                        ),
                        affinity_confidence=parse_optional_float(
                            row.get("affinity_confidence")
                        ),
                        pose_ref=None,
                    )
                )
    if not ligands:
        raise ValueError(f"No SMILES rows were found in '{csv_path}'.")
    return ligands, affinities


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return float(normalized)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected optional float-like value, got {type(value).__name__}.")


def build_cli_target_placeholder() -> TargetCandidate:
    return TargetCandidate(
        gene="UNKNOWN",
        score=0.0,
        disease=None,
        ncbi_id=None,
        uniprot_id=None,
        evidence=[],
        pathway=[],
        extra={"origin": "molforge admet CLI"},
    )


def build_cli_target_from_affinities(
    affinities: list[AffinityPrediction],
) -> TargetCandidate:
    target_genes = {
        affinity.target_gene.strip()
        for affinity in affinities
        if affinity.target_gene.strip() and affinity.target_gene.strip() != "UNKNOWN"
    }
    if not target_genes:
        return build_cli_target_placeholder()
    if len(target_genes) > 1:
        raise ValueError(
            "molforge admet currently requires a single target_gene across ranked CSV inputs."
        )
    return TargetCandidate(
        gene=next(iter(target_genes)),
        score=0.0,
        disease=None,
        ncbi_id=None,
        uniprot_id=None,
        evidence=[],
        pathway=[],
        extra={"origin": "molforge admet CLI"},
    )


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


def read_status_summary() -> str:
    return (
        "molforge is ready. Use `molforge run <input.json>` for the pipeline, "
        "`molforge api` for the HTTP API, and `molforge --help` for commands."
    )


def _to_jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__") and not isinstance(value, type):
        return {str(key): _to_jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
