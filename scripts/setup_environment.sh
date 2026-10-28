#!/usr/bin/env bash
# Fetch the contract-pinned Engine wheel (if missing) and sync the uv environment.
# Safe to rerun; a fresh checkout or worktree needs only this command.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

read -r tag commit name sha < <(python3 -c '
import json; i = json.load(open("integration/epcsaft_contract.json"))["final_identity"]
print(i["release_tag"], i["engine_commit"], i["wheel_filename"], i["wheel_sha256"])')
dir="build/engine-wheels/$tag"
wheel="$dir/$name"

if [[ ! -f "$wheel" ]]; then
  mkdir -p "$dir"
  gh release download "$tag" --repo tannerpolley/ePC-SAFT --dir "$dir" \
    --pattern '*.whl' --pattern '*.sha256' --pattern release.json
  digest="$(gh api "repos/tannerpolley/ePC-SAFT/releases/tags/$tag" \
    --jq '.assets[] | select(.name | endswith(".whl")) | .digest')"
  target="$(gh api "repos/tannerpolley/ePC-SAFT/git/ref/tags/$tag" --jq .object.sha)"
  [[ "$digest" == "sha256:$sha" ]] || { echo "release digest $digest != contract $sha" >&2; exit 1; }
  [[ "$target" == "$commit" ]] || { echo "tag target $target != contract $commit" >&2; exit 1; }
fi
echo "$sha  $wheel" | sha256sum --check --strict --quiet || { echo "wheel hash mismatch: $wheel" >&2; exit 1; }

uv sync --frozen --group test
