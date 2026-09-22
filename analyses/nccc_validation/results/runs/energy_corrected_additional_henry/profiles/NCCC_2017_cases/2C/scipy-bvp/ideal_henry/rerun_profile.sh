#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"
cd -- "$repo_root"
uv run python analyses/nccc_validation/scripts/run_case_profile.py --spec "$script_dir/run_spec.json"
