# molforge

CPU-first small-molecule discovery pipeline that consumes biocompute target exports and produces ranked candidate ligands.

## Safety and scope

molforge is research software for computational prioritization and
reproducibility experiments. It is not a clinical, diagnostic, prescribing, or
wet-lab-validated drug-discovery product. Reported rankings and benchmark
outputs are hypotheses that require independent experimental validation.

## What the current MVP does

- `molforge run <input.json>` now executes a real pipeline path instead of a pure stub.
- Stage 2 uses the existing docking stack (`DockingRunner`) to fetch structure, detect a pocket, and produce affinity predictions.
- Stage 3 is wired in as optional REINVENT augmentation. If REINVENT is not configured, the pipeline records that generation was skipped and still completes with docking-driven candidates.
- Stage 4 scores and ranks the ligands through the existing ADMET module.
- A thin HTTP API exposes run creation and run/status retrieval on top of the same orchestration path.

## Current CPU-first MVP boundaries

- Live ranked runs require the docking and ADMET extras:
  - `uv sync --extra docking --extra admet --extra dev`
- REINVENT is optional. If unavailable, `molforge run` still produces ranked output from the docking → ADMET path.
- EvE Bio off-target lookup is optional and research-only. Enable it only with
  explicit flags/API options and a local cache or the `evebio` extra; default
  runs do not download or require Hugging Face data.
- Boltz-2 and OpenFold3 remain deferred for v2 / remote GPU workflows.

## Quick start

```bash
uv sync --extra docking --extra admet --extra dev
uv run molforge run examples/biocompute-targets.json --disease "scar pain" --output archive/runs/latest.json
```

Expected output artifact:

- `archive/runs/latest.json` — serialized `PipelineRun` with ranked candidates

## CLI commands

```bash
uv run molforge run <input.json> [--disease NAME] [--top N] [--db-path FILE] [--output FILE]
uv run molforge dock <input.json> [--disease NAME] [--top N] [--db-path FILE] [--output FILE]
uv run molforge generate <pocket.json> --output-dir archive/runs/generate
uv run molforge admet <smiles.csv> [--db-path FILE] [--output FILE]
uv run molforge status
uv run molforge api [--host 127.0.0.1] [--port 8000] [--db-path molforge.db]
```

Optional research-only EvE Bio lookup:

```bash
uv sync --extra evebio
uv run molforge run examples/biocompute-targets.json --enable-evebio
uv run molforge admet ligands.csv --enable-evebio
```

EvE Bio `eve-bio/drug-target-activity` is a CC BY-NC-SA 4.0 dataset. molforge
uses it only as an optional non-commercial research/off-target reference, does
not redistribute source data, and keeps live Hugging Face download opt-in.

## HTTP API

Start the server:

```bash
uv run molforge api --host 127.0.0.1 --port 8000
```

Create a run:

```bash
curl -s http://127.0.0.1:8000/runs \
  -H 'Content-Type: application/json' \
  -d @examples/api-run-request.json
```

Current API constraints:

- `POST /runs` accepts exactly one target per request.
- The API always uses the server-configured database path from `molforge api --db-path ...`.
- The HTTP surface does not accept arbitrary `db_path` or output file paths from clients.

Inspect a run:

```bash
curl -s http://127.0.0.1:8000/runs/<run_id>
curl -s http://127.0.0.1:8000/status
```

## Example inputs

- `examples/biocompute-targets.json` — minimal biocompute-style target export fixture
- `examples/api-run-request.json` — API payload for `POST /runs`
- `examples/cxcr4_scar_pain.md` — end-to-end example notes for the current MVP

## Verification used for this repo state

```bash
uv run pytest -q
```
