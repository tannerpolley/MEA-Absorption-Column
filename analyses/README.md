# Analysis notebooks

Research is active; manuscript revision work is complete. Start at
[`index.qmd`](index.qmd), then use the root Quarto website to browse registered
analysis pages:

```bash
python3 analyses/manuscript.py validate analyses
bash analyses/render.sh --to html
```

The site renders with execution disabled. Start a calculation separately from
an explicit configuration; rendering never runs a model or changes retained
results. `_cse-manuscript.json` owns site membership and sidebar order.
An optional PDF page can be rendered with
`bash analyses/render.sh bvp_solution_methods/notebook.qmd --to pdf`.

| Analysis | Purpose |
|---|---|
| research_options | Configuration preview, method selection and research index |
| nccc_validation | Retained column/campaign evidence, including submitted results |
| transport_sensitivity | Retained transport perturbations and interpretation |
| reactive_film_evidence | Film formulation, source evidence and campaign comparisons |
| bvp_derivative_trials | BVP verification and supported-negative CasADi comparison |
| bvp_solution_methods | Preserved coupled-method checks; original twelve-state builder remains at its source checkpoint |
| issue16_reactive_film_runtime | Single-state numerical-reachability evidence with unresolved physical inputs |
| greenfield_node_qualification | Engine #148 node checks of the twelve-state Engine quantities (numerical only) |
| physical_acceptance_149 | Engine #149 physical criteria (C3, C5, C6, capture, temperature taps, density) on the adopted record |

Keep each study's inputs, scripts, results and `notebook.qmd` together. Rendering
must not execute model code. Store new attempts in new output directories and
retain failures. The existence of a notebook or converged solve does not establish
an accepted scientific claim. No research configuration is restricted to the
submitted manuscript's chosen case.
