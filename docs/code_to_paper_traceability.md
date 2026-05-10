# Code-to-Paper Traceability

This note maps the main manuscript claims to committed repository evidence. It is intended for reviewer-response checks and future agents.

| Manuscript item | Evidence artifact | Script or source path | Scope |
| --- | --- | --- | --- |
| NCCC one-bed scope table | `docs/latex/tables/nccc_one_bed_case_scope.tex` | `src/mea_absorption_column/data/NCCC_2017_model_inputs_mass.csv`; `src/mea_absorption_column/data/NCCC_2014_model_inputs_mass.csv` | One-bed, no-intercooler K18, K19, K20, and 1C--7C operating points. |
| Accepted-row validation aggregate | `analyses/nccc_validation/results/final/tables/nccc_one_bed_accepted_results.csv`; `nccc_one_bed_accepted_summary.csv` | `analyses/nccc_validation/scripts/generate_data.py` | K18, K19, and 1C--6C accepted rows for Henry-law and routine ePC-SAFT fugacity closure. |
| Attempted-row boundary | `analyses/nccc_validation/results/final/tables/nccc_one_bed_all_attempted_results.csv`; `docs/latex/tables/nccc_one_bed_attempted_status.tex` | `analyses/nccc_validation/scripts/generate_data.py` | K20 rejected by mesh/domain-guard behavior; 7C rejected by accepted-row timeout. |
| Representative Case 3C temperature figure | `docs/latex/figures/case-3c-temperature-validation.png` | `analyses/nccc_validation/results/final/profiles/`; `analyses/nccc_validation/scripts/collect_clean_profiles.py` | Temperature-profile validation for one accepted 2017 C case. |
| 2017 C-case temperature overlays | `docs/latex/figures/case-c-temperature-overlay.png`; `analyses/nccc_validation/results/final/tables/c_case_campaign_temperature_overlay_metrics.csv` | `analyses/nccc_validation/scripts/render_c_case_campaign_temperature_gallery.py` | Accepted 2017 one-bed C-case temperature-profile evidence. |
| ePC-SAFT versus Henry-law comparison | `docs/latex/figures/nccc-one-bed-thermo-benchmark.pdf`; `nccc_one_bed_accepted_summary.csv` | `analyses/nccc_validation/scripts/render_figures.py` | Routine `epcsaft_ionic` fugacity campaign versus `ideal_henry`, concentration-based chemistry fixed. |
| Solver-method comparison | `docs/latex/figures/method-case-solver-contrast.pdf`; `docs/latex/tables/method_case_contrast.tex`; `analyses/nccc_validation/results/final/tables/method_case_contrast.csv` | `analyses/nccc_validation/scripts/render_figures.py`; benchmark CLI | Smooth favorable case and NCCC 3C thermal-profile case. |
| Full activity-coupled feasibility boundary | `analyses/nccc_validation/results/final/tables/full_species_ionic_2017_c_case_sweep.csv`; `docs/latex/tables/full_ionic_speciation_timing.tex` | `analyses/nccc_validation/scripts/run_full_species_ionic_2017_c_case_sweep.py` | Seven 2017 C rows using the slow nine-species activity-coupled ePC-SAFT path under relaxed feasibility settings. |
| ePC-SAFT parameter summary | `docs/latex/tables/epcsaft_parameter_summary.tex`; `analyses/nccc_validation/results/final/tables/epcsaft_electrolyte_*` | `analyses/nccc_validation/scripts/run_epcsaft_electrolyte_config_matrix.py` | Selected ePC-SAFT dataset and configuration provenance. |
| Reproduction commands | `REPRODUCE.md`; `analyses/nccc_validation/README.md` | Repository root and analysis scripts | Environment setup, artifact refresh, optional long full-path rerun, and LaTeX build checks. |

The manuscript should keep the routine `epcsaft_ionic` fugacity campaign separate from the full activity-coupled feasibility path. Do not describe the slow nine-species path as the routine validation model.
