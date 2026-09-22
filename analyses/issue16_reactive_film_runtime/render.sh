#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
uv run --locked python scripts/render_summary.py
quarto render notebook.qmd --no-execute
