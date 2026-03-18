#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../preprocess"

python build_demo_pipeline.py \
    --hdf5 "../auto_sample/$1.hdf5" \
    --demo-id "$2"
