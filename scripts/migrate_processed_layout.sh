#!/usr/bin/env bash
# Migrate active timepoint processed artifacts into the publication data layout.
#
# Conservative behavior:
# - creates canonical directories;
# - moves only known legacy filenames/paths;
# - never deletes unknown files;
# - safe to re-run.
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROCESSED="$PROJECT_DIR/data/processed"
VERSION="$PROCESSED/microalgae_v1"

move_first_present() {
  local dst_rel="$1"
  shift

  local dst="$PROCESSED/$dst_rel"
  mkdir -p "$(dirname "$dst")"

  if [[ -e "$dst" ]]; then
    echo "exists $dst_rel"
    return 0
  fi

  local src_rel
  for src_rel in "$@"; do
    local src="$PROCESSED/$src_rel"
    if [[ -e "$src" ]]; then
      mv "$src" "$dst"
      echo "moved $src_rel -> $dst_rel"
      return 0
    fi
  done

  echo "missing $dst_rel"
}

mkdir -p \
  "$VERSION/views/timepoint" \
  "$VERSION/views/field" \
  "$PROCESSED/legacy_flat"

# Timepoint view: main publication view derived from actual acquisition times.
move_first_present "microalgae_v1/views/timepoint/index.csv" \
  "timegroup/full/index.csv" \
  "index_timegroup.csv"
move_first_present "microalgae_v1/views/timepoint/embedding.csv" \
  "timegroup/full/embedding.csv" \
  "embedding_timegroup.csv"
move_first_present "microalgae_v1/views/timepoint/summary.json" \
  "timegroup/full/summary.json" \
  "generation_summary_timegroup.json"

# Field view: whole-microscopy-field generation and annotation support.
move_first_present "microalgae_v1/views/field/index.csv" \
  "field_index.csv"
move_first_present "microalgae_v1/views/field/embedding.csv" \
  "field_embedding.csv"
move_first_present "microalgae_v1/views/field/metadata.csv" \
  "field_metadata.csv"
move_first_present "microalgae_v1/views/field/targets.csv" \
  "field_targets.csv"
move_first_present "microalgae_v1/views/field/prompts.csv" \
  "field_prompts.csv"

echo
echo "Processed-data migration complete."
echo "Canonical root: data/processed/microalgae_v1"
echo "Review remaining flat files under data/processed/ before publishing."
