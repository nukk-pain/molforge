#!/bin/sh
set -eu

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <sampling.toml> <workspace-dir>" >&2
  exit 1
fi

CONFIG_PATH="$1"
WORKSPACE_DIR="$2"

mkdir -p "$WORKSPACE_DIR"
cp "$CONFIG_PATH" "$WORKSPACE_DIR/sampling.toml"

echo "Running REINVENT4 sampling in $WORKSPACE_DIR"
(cd "$WORKSPACE_DIR" && reinvent -l sampling.log sampling.toml)
