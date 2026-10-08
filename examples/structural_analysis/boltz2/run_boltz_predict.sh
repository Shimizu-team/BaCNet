#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT_DIRECTORY [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

input_directory="$1"
output_directory="${2:-results/$(basename "$input_directory")}"

if [[ ! -d "$input_directory" ]]; then
  echo "Input directory not found: $input_directory" >&2
  exit 2
fi

if ! command -v boltz >/dev/null 2>&1; then
  echo "The boltz executable is not available on PATH." >&2
  exit 127
fi

mkdir -p "$output_directory"

boltz predict "$input_directory" \
  --use_potentials \
  --use_msa_server \
  --diffusion_samples 5 \
  --recycling_steps 5 \
  --out_dir "$output_directory" \
  --output_format pdb
