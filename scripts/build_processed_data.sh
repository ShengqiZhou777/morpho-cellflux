#!/usr/bin/env bash
# Compatibility wrapper for the canonical microalgae dataset builder.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_DIR"

VIEWS=${VIEWS:-timepoint}

python scripts/build_microalgae_dataset.py \
  --version microalgae_v1 \
  --views "$VIEWS"
