# Example: CXCR4 scar-pain CPU-first run

This example matches the current MVP reality in the repository.

## Inputs

- target export: `examples/biocompute-targets.json`
- disease label: `scar pain`

## Live CLI run

```bash
uv sync --extra docking --extra admet --extra dev
uv run molforge run examples/biocompute-targets.json --disease "scar pain" --output archive/runs/cxcr4-scar-pain.json
```

## What happens

1. Stage 2 resolves the target, fetches the structure, detects a pocket, and docks the FDA-approved library.
2. Stage 3 attempts REINVENT generation if configured.
3. Stage 4 scores the ligands and writes ranked candidates.

If REINVENT is not configured, the run still completes with docking-ranked candidates and records generation as skipped in provenance.

## API equivalent

```bash
uv run molforge api --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/runs -H 'Content-Type: application/json' -d @examples/api-run-request.json
```

## Notes

- This is a CPU-first MVP example, not a GPU Boltz-2/OpenFold3 workflow.
- For a local smoke path without live external dependencies, use the focused pytest suite instead of assuming a full live run will succeed on every machine.
