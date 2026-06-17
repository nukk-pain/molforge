from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from contracts.schema import BindingPocket, GeneratedMolecule
from ._env_check import (
    ReinventEnvironment,
    probe_reinvent_environment,
)
from .filters import filter_generated_smiles
from .reinvent4_poc import (
    build_sampling_command,
    collect_output_files,
    execute_command,
    format_command,
)

CommandExecutor = Callable[
    [Sequence[str], float, Path | None], subprocess.CompletedProcess[str]
]


@dataclass(frozen=True, slots=True)
class ReinventRunArtifacts:
    run_dir: str
    config_path: str
    command_log_path: str
    metadata_path: str
    sampling_log_path: str
    output_files: list[str]
    used_live_backend: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class REINVENT4Backend:
    name: str = "reinvent4"

    def __init__(
        self,
        *,
        workspace_root: str | Path,
        reference_smiles: Sequence[str],
        prior_path: str | Path | None = None,
        command_executor: CommandExecutor | None = None,
        which_resolver: Callable[[str], str | None] = shutil.which,
        timeout_seconds: float = 600.0,
        sampling_template: str | None = None,
        observed_pass_rate: float | None = None,
        sample_seconds_per_ten: float | None = None,
    ) -> None:
        self.workspace_root: Path = Path(workspace_root)
        self.reference_smiles: list[str] = list(reference_smiles)
        self.prior_path: str | Path | None = prior_path
        self.command_executor: CommandExecutor = command_executor or execute_command
        self.which_resolver: Callable[[str], str | None] = which_resolver
        self.timeout_seconds: float = timeout_seconds
        self.sampling_template: str | None = sampling_template
        self.observed_pass_rate: float | None = observed_pass_rate
        self.sample_seconds_per_ten: float | None = sample_seconds_per_ten
        self.last_run_artifacts: ReinventRunArtifacts | None = None

    def generate(
        self,
        pocket: BindingPocket,
        n: int = 100,
        seed_smiles: str | None = None,
    ) -> list[GeneratedMolecule]:
        if n <= 0:
            raise ValueError("n must be positive for REINVENT4 generation.")
        environment = probe_reinvent_environment(
            prior_path=self.prior_path,
            which_resolver=self.which_resolver,
        )
        run_dir = self._create_run_dir()
        config_path = run_dir / "sampling.toml"
        metadata_path = run_dir / "metadata.json"
        command_log_path = run_dir / "command.log"
        sampling_log_path = run_dir / "sampling.log"

        _ = config_path.write_text(
            self._build_sampling_config(
                environment=environment,
                requested_count=n,
                seed_smiles=seed_smiles,
                pocket=pocket,
            ),
            encoding="utf-8",
        )

        if not environment.ready or environment.cli_path is None:
            self._write_metadata(
                metadata_path,
                environment=environment,
                requested_count=n,
                status="blocked",
                seed_smiles=seed_smiles,
                pocket=pocket,
                output_files=[],
            )
            self._write_command_log(
                command_log_path,
                environment=environment,
                command_text="skipped",
                exit_code=None,
                status="blocked",
            )
            self.last_run_artifacts = ReinventRunArtifacts(
                run_dir=str(run_dir),
                config_path=str(config_path),
                command_log_path=str(command_log_path),
                metadata_path=str(metadata_path),
                sampling_log_path=str(sampling_log_path),
                output_files=[],
                used_live_backend=False,
            )
            raise RuntimeError(
                "REINVENT4 live generation is not ready: "
                + ", ".join(environment.blocking_reasons)
            )

        output_files: list[str] = []
        attempts: list[dict[str, object]] = []
        accepted: list[GeneratedMolecule] = []
        seen_smiles: set[str] = set()
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(2):
            completed, output_files, molecules = self._run_sampling_attempt(
                environment=environment,
                config_path=config_path,
                command_log_path=command_log_path,
                pocket=pocket,
                requested_count=n,
                run_dir=run_dir,
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "filtered_count": len(molecules),
                    "output_files": output_files,
                    "exit_code": completed.returncode,
                }
            )
            for molecule in molecules:
                if molecule.smiles in seen_smiles:
                    continue
                seen_smiles.add(molecule.smiles)
                accepted.append(molecule)
            if len(accepted) >= n:
                break

        if completed is None:
            raise RuntimeError("REINVENT4 sampling did not start.")

        self._write_metadata(
            metadata_path,
            environment=environment,
            requested_count=n,
            status="pass" if completed.returncode == 0 else "blocked",
            seed_smiles=seed_smiles,
            pocket=pocket,
            output_files=output_files,
            attempts=attempts,
            accepted_count=len(accepted),
        )
        self.last_run_artifacts = ReinventRunArtifacts(
            run_dir=str(run_dir),
            config_path=str(config_path),
            command_log_path=str(command_log_path),
            metadata_path=str(metadata_path),
            sampling_log_path=str(sampling_log_path),
            output_files=output_files,
            used_live_backend=True,
        )
        return accepted[:n]

    def _run_sampling_attempt(
        self,
        *,
        environment: ReinventEnvironment,
        config_path: Path,
        command_log_path: Path,
        pocket: BindingPocket,
        requested_count: int,
        run_dir: Path,
    ) -> tuple[subprocess.CompletedProcess[str], list[str], list[GeneratedMolecule]]:
        command = build_sampling_command(
            environment.cli_path or "reinvent", config_path
        )
        completed = self.command_executor(
            command,
            self._resolve_timeout_seconds(requested_count),
            run_dir,
        )
        output_files = collect_output_files(run_dir)
        self._write_command_log(
            command_log_path,
            environment=environment,
            command_text=format_command(command),
            exit_code=completed.returncode,
            status="pass" if completed.returncode == 0 else "blocked",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"REINVENT4 sampling failed with exit code {completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        generated_path = self._select_generated_output(run_dir)
        smiles_values = self._load_generated_smiles(generated_path)
        molecules = filter_generated_smiles(
            smiles_values,
            reference_smiles=self.reference_smiles,
            backend=self.name,
            pocket_ref=pocket,
        )
        return completed, output_files, molecules

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.workspace_root / "reinvent4" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _build_sampling_config(
        self,
        *,
        environment: ReinventEnvironment,
        requested_count: int,
        seed_smiles: str | None,
        pocket: BindingPocket,
    ) -> str:
        oversample_factor = self._resolve_oversample_factor()
        num_smiles = requested_count * oversample_factor
        if self.sampling_template is not None:
            template = self.sampling_template
            template = template.replace(
                "REPLACE_WITH_PRIOR_PATH",
                environment.prior_path or "REPLACE_WITH_PRIOR_PATH",
            )
            template = template.replace(
                "num_smiles = 100", f"num_smiles = {num_smiles}"
            )
        else:
            lines = [
                'run_type = "sampling"',
                "[parameters]",
                f'model_file = "{environment.prior_path or "REPLACE_WITH_PRIOR_PATH"}"',
                f"num_smiles = {num_smiles}",
            ]
            if seed_smiles:
                lines.append(f'smiles_file = "{seed_smiles}"')
            lines.extend(
                [
                    "",
                    "# molforge v1 accepts pocket input for provenance only; geometric constraints are not encoded.",
                    f'# target_gene = "{pocket.structure.gene}"',
                    f"# pocket_center = {list(pocket.center_xyz)}",
                    f"# pocket_size = {list(pocket.size_xyz)}",
                ]
            )
            template = "\n".join(lines)
        return template if template.endswith("\n") else template + "\n"

    def _resolve_oversample_factor(self) -> int:
        if self.observed_pass_rate is None or self.observed_pass_rate <= 0:
            return 3
        if self.observed_pass_rate < 0.4:
            return 5
        return max(2, math.ceil(1.0 / self.observed_pass_rate))

    def _resolve_timeout_seconds(self, requested_count: int) -> float:
        if self.sample_seconds_per_ten is None or self.sample_seconds_per_ten <= 0:
            return self.timeout_seconds
        computed_timeout = self.sample_seconds_per_ten * (requested_count / 10.0) * 2.0
        return max(self.timeout_seconds, computed_timeout)

    def _select_generated_output(self, run_dir: Path) -> Path:
        candidates = sorted(
            [
                path
                for path in run_dir.rglob("*")
                if path.is_file() and path.suffix in {".smi", ".csv", ".json"}
            ]
        )
        for path in candidates:
            if path.name in {
                "metadata.json",
                "sampling.log",
                "command.log",
                "sampling.toml",
            }:
                continue
            return path
        raise RuntimeError(
            "REINVENT4 sampling did not produce a generated SMILES artifact."
        )

    def _load_generated_smiles(self, path: Path) -> list[str]:
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("REINVENT JSON output must be a list.")
            return [
                str(item["smiles"]).strip()
                for item in payload
                if isinstance(item, dict) and "smiles" in item
            ]
        if path.suffix == ".csv":
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                return []
            header = [part.strip().lower() for part in lines[0].split(",")]
            if "smiles" in header:
                index = header.index("smiles")
                return [
                    line.split(",")[index].strip() for line in lines[1:] if line.strip()
                ]
            return [line.split(",")[0].strip() for line in lines[1:]]
        smiles_values: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            smiles_values.append(stripped.split()[0])
        return smiles_values

    def _write_metadata(
        self,
        path: Path,
        *,
        environment: ReinventEnvironment,
        requested_count: int,
        status: str,
        seed_smiles: str | None,
        pocket: BindingPocket,
        output_files: list[str],
        attempts: list[dict[str, object]] | None = None,
        accepted_count: int | None = None,
    ) -> None:
        _ = path.write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "environment": environment.to_dict(),
                    "requested_count": requested_count,
                    "status": status,
                    "seed_smiles": seed_smiles,
                    "accepted_pocket_gene": pocket.structure.gene,
                    "pocket_semantics": "accepted-not-conditioned-v1",
                    "output_files": output_files,
                    "attempts": attempts or [],
                    "accepted_count": accepted_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_command_log(
        self,
        path: Path,
        *,
        environment: ReinventEnvironment,
        command_text: str,
        exit_code: int | None,
        status: str,
    ) -> None:
        lines = [
            f"backend={self.name}",
            f"install_command={environment.install_command}",
            f"sampling_command={environment.sampling_command}",
            f"resolved_cli_path={environment.cli_path}",
            f"resolved_prior_path={environment.prior_path}",
            f"platform={environment.platform}",
            f"machine={environment.machine}",
            "pocket_semantics=accepted-not-conditioned-v1",
            f"command={command_text}",
            f"exit_code={exit_code}",
            f"status={status}",
        ]
        _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
