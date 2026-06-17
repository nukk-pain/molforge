# molforge

molforge is a CPU-first small-molecule discovery pipeline.

Give it a biocompute-style therapeutic target export. It returns a ranked
small-molecule candidate run with docking, optional generation, ADMET scoring,
provenance, and a serialized output artifact.

It is built to work well when handed to an AI coding agent such as Claude Code,
Codex, OpenClaw, or Hermes Agent.

Use it when you want to ask:

> For this target export, which small-molecule ligands should we investigate first?

## Use With An AI Agent

Give your agent the repo URL and this task:

```text
Clone https://github.com/nukk-pain/molforge.
Install it with uv sync --extra docking --extra admet --extra dev.
Run molforge on examples/biocompute-targets.json for scar pain.
Show me the ranked candidate output path and summarize the top ligands.
Do not commit .env files, caches, local databases, archive outputs, model
weights, downloaded datasets, or generated run artifacts.
```

For a local smoke run without live external dependencies, ask the agent to run
the focused public-safe test slice instead:

```bash
uv run pytest -q \
  tests/test_core/test_cli.py \
  tests/test_core/test_input.py \
  tests/test_core/test_pipeline.py \
  tests/test_core/test_api.py \
  tests/test_admet/test_ranker.py \
  tests/test_admet/test_scorer.py \
  tests/test_generative/test_module.py
```

## Run It Yourself

```bash
uv sync --extra docking --extra admet --extra dev

uv run molforge run examples/biocompute-targets.json \
  --disease "scar pain" \
  --output archive/runs/latest.json
```

Expected output artifact:

- `archive/runs/latest.json` - serialized `PipelineRun` with ranked candidates

## What The Agent Should Return

Ask the agent to report:

- the output artifact path
- the top ranked ligands or candidates
- whether generation was run or skipped
- any docking, ADMET, dependency, or API errors
- the research-only caveat

The primary input contract is a biocompute-style target list, either as a plain
array of target candidates or as:

```json
{
  "schema_version": "biocompute-target-candidates/v1",
  "candidates": []
}
```

See `contracts/schema.py` and `examples/biocompute-targets.json` for the exact
shape used by this public snapshot.

## What The Current MVP Does

- `molforge run <input.json>` executes the CPU-first pipeline path.
- Stage 2 uses the docking stack to fetch structure, detect a pocket, and
  produce affinity predictions.
- Stage 3 optionally runs REINVENT augmentation. If REINVENT is not configured,
  the pipeline records generation as skipped and still completes with
  docking-driven candidates.
- Stage 4 scores and ranks ligands through the ADMET module.
- A thin HTTP API exposes run creation and run/status retrieval on top of the
  same orchestration path.

## CLI Commands

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

Inspect a run:

```bash
curl -s http://127.0.0.1:8000/runs/<run_id>
curl -s http://127.0.0.1:8000/status
```

Current API constraints:

- `POST /runs` accepts exactly one target per request.
- The API always uses the server-configured database path from
  `molforge api --db-path ...`.
- The HTTP surface does not accept arbitrary `db_path` or output file paths from
  clients.

## Example Inputs

- `examples/biocompute-targets.json` - minimal biocompute-style target export fixture
- `examples/api-run-request.json` - API payload for `POST /runs`
- `examples/cxcr4_scar_pain.md` - end-to-end example notes for the current MVP

## Agent Safety Rules

- Treat generated outputs as local artifacts: `archive/`, `molforge.db`,
  `.pytest_cache/`, `__pycache__/`, and `.venv/` should not be committed.
- Do not add API keys, service-account files, private datasets, patient data, or
  model weights to the repository.
- Keep live Hugging Face downloads opt-in.
- Prefer the focused public-safe test slice for agent smoke verification.
- This public repo is a clean snapshot. Do not publish private development
  history when preparing another public release.

## Research Use Only

molforge generates research hypotheses for computational prioritization and
reproducibility experiments. It is not a clinical, diagnostic, prescribing, or
wet-lab-validated drug-discovery product. Results must be reviewed by qualified
domain experts and require independent experimental validation.

## License

Apache-2.0.
