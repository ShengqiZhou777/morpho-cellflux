#!/usr/bin/env bash
set -euo pipefail

# Clone external baseline implementations into baselines/external/.
# This script downloads code only. Do not download the original papers' datasets:
# all baselines should be trained/evaluated on this repo's diet and CRISPR data.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT="$ROOT/baselines/external"
mkdir -p "$EXT"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dir="$EXT/$name"
  if [[ -d "$dir/.git" ]]; then
    echo "[update] $name"
    git -C "$dir" fetch --depth=1 origin
    git -C "$dir" pull --ff-only
  else
    echo "[clone] $name <- $url"
    git clone --depth=1 "$url" "$dir"
  fi
}

clone_or_update phendiff "https://github.com/WarmongeringBeaver/PhenDiff.git"
clone_or_update impa "https://github.com/theislab/IMPA.git"
clone_or_update morphodiff "https://github.com/bowang-lab/MorphoDiff.git"
clone_or_update stargan "https://github.com/yunjey/stargan.git"

echo
echo "External baseline repos are under: $EXT"
echo
echo "Next steps:"
echo "  1. Inspect each repo's requirements file."
echo "  2. Install dependencies in a separate environment, not blindly into the main pmf env."
echo "  3. Implement/export adapters so outputs land under outputs/baselines/<method>/<benchmark>/."
