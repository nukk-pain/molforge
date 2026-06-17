#!/bin/sh
# Deploy the molforge remote GPU Modal app.
#
# Prereq: `modal token new` completed (credentials live in ~/.modal.toml).
# Re-run this script anytime the image pin or function signature changes.
set -eu

cd "$(dirname "$0")/.."

echo "Deploying molforge-remote-gpu to Modal workspace..."
uv run modal deploy src/molforge/remote/modal_app.py

echo
echo "Verify in the Modal dashboard that app 'molforge-remote-gpu' exists"
echo "with function 'run_job' bound to image pinning boltz[cuda]==0.4.1."
