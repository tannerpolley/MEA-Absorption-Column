# Transport sensitivity

The [research notebook](notebook.qmd) owns the scientific interpretation. The
study uses the manuscript's Case 3C model and parameter set, read-only, and
retains all outputs here. It does not edit the manuscript or its model defaults.

## Reproduce one column

From the checkout containing this analysis, select the manuscript model checkout
and its immutable Engine environment. Output directories must not already exist.

```bash
transport_model_root=/home/tnnrpolley21/.codex/worktrees/bad5/MEA-Absorption-Column
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-project --python "$transport_model_root/.venv/bin/python" \
  analyses/transport_sensitivity/figures/response/scripts/generate_data.py \
  --model-root "$transport_model_root" \
  --reference "$transport_model_root/analyses/nccc_validation/results/runs/reactive_native_41" \
  --output analyses/transport_sensitivity/figures/response/output/viscosity_090 \
  --quantity viscosity --factor 0.9
```

The retained screen is factor 1 for the baseline and 0.9/1.1 separately for
`viscosity`, `diffusivity` and `kl`. The positive-factor helper is not a claim
that arbitrarily large perturbations are valid; the summary checks this exact
six-case design. `--check-only` tests property scaling without running a column.

After the diagnosed cold-anchor failure, the −10% diffusivity case uses
`capture_failure.py` with the same arguments, output `diffusivity_090_seeded`,
and two additional arguments:

```text
--initial-profile analyses/transport_sensitivity/figures/response/output/baseline/solution_scaled.csv
--anchor-start analyses/transport_sensitivity/figures/response/output/diffusivity_090_diagnostic/state_probe.json
```

Its `initialization.json` records the initial column profile, the certified
equilibrium starting composition and their checksums. The latter is mapped
conservatively onto the requested feed using the model's existing helper and
supplied only to cold native starts. Every requested state is still solved and
certified at its own temperature, pressure and composition. The same numerical
initialization is checked at factor one in `baseline_seeded`.
The original and profile-only failed attempts remain retained, not overwritten.
Seed compatibility is verified for these retained inputs, not arbitrary external
profiles or anchor documents; this helper is study-specific, not a new default.

The installed-wheel final check is retained in `figures/response/input/engine_check.txt`.
The small input archives preserve model sources, the selected reactive bundle,
the original column reference and its read/verification helpers. Run identities
retain the exact content hashes; the Engine wheel must be supplied separately
with the recorded SHA-256. Archives are reproducibility inputs, not runtime
fallback packages or alternative parameter owners.

## Validate and render

Use the same `uv run --no-project --python ...` prefix with
`analyses/transport_sensitivity/figures/response/scripts/check_derivatives.py --model-root ...` for the existing
assembled column-RHS directional check at one seven-state point under each
perturbation, not every axial profile point. This independent check uses
centered differences; the actual solver Jacobian uses native
thermodynamic derivatives and differentiated absorber expressions.

Use `analyses/transport_sensitivity/figures/response/scripts/render_figure.py --model-root ...
--diffusivity-minus-run diffusivity_090_seeded` to read the seven completed
runs, check their identities and numerical evidence, retain `summary.csv` and
`profiles.csv`, and write the two-panel SVG/PNG/PDF figure. It never runs a column.

From this analysis directory, `bash render.sh notebook.qmd` renders the living
HTML notebook without executing scientific code. The notebook must be updated
from retained values before that render; rendering does not approve results.

Each column was bounded to 1800 s. Runs used one BLAS/OpenMP thread each and up
to six concurrent column processes. Their timing is cost evidence, not a
controlled speed benchmark. Manuscript use requires scientific review and
selection by the manuscript owner.
