#!/usr/bin/env bash
# Copy FusionODE source data into the project-local raw-data layout.
#
# Usage:
#   bash scripts/prepare_raw_data.sh
#   FUSIONODE_DATA=/path/to/FusionODE/data bash scripts/prepare_raw_data.sh
#
# Canonical outputs:
#   data/raw/microalgae_v1/single_cell_images
#   data/raw/microalgae_v1/field_images
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FUSIONODE_DATA=${FUSIONODE_DATA:-/home/shockley/myproject/FusionODE/data}
RAW_ROOT="$PROJECT_DIR/data/raw/microalgae_v1"
SINGLE_CELL_DST="$RAW_ROOT/single_cell_images"
FIELD_DST="$RAW_ROOT/field_images"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing source path: $path" >&2
    exit 1
  fi
}

copy_dir_once() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" ]]; then
    echo "exists $dst"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  cp -a "$src" "$dst"
  echo "copied $src -> $dst"
}

require_path "$FUSIONODE_DATA/CROPS_RAW_SCALE"
require_path "$FUSIONODE_DATA/TIMECOURSE"

copy_dir_once "$FUSIONODE_DATA/CROPS_RAW_SCALE" "$SINGLE_CELL_DST"
copy_dir_once "$FUSIONODE_DATA/TIMECOURSE" "$FIELD_DST"

if [[ ! -d "$SINGLE_CELL_DST/0h/Dark" || ! -d "$SINGLE_CELL_DST/0h/Light" ]]; then
  echo "Invalid single-cell layout under $SINGLE_CELL_DST" >&2
  exit 1
fi

if [[ ! -d "$FIELD_DST/0h/Dark/images" || ! -d "$FIELD_DST/0h/Dark/masks" ]]; then
  echo "Invalid field-image layout under $FIELD_DST" >&2
  exit 1
fi

single_files=$(find "$SINGLE_CELL_DST" -type f -name '*.png' | wc -l)
field_images=$(find "$FIELD_DST" -type f -path '*/images/*.jpg' | wc -l)
field_masks=$(find "$FIELD_DST" -type f -path '*/masks/*.png' | wc -l)

echo "Prepared localized raw data:"
du -sh "$SINGLE_CELL_DST" "$FIELD_DST"
echo "  single-cell png files: $single_files"
echo "  field jpg images:      $field_images"
echo "  field png masks:       $field_masks"
