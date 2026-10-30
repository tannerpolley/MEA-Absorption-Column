# Identified MDEA B1 thermodynamic inputs

Unmodified source bytes from `tannerpolley/Amine-Thermodynamics`, main
`62e256d`, analysis `analyses/solvent_specification_mdea/`:

| Retained filename | Upstream filename | SHA-256 |
|---|---|---|
| `parameters.json` | `results/B1-parameters.json` | `1d0d80832f282cde85acab294501ca9ec8dbc34cddf8edf2eebf6e1e1d03949d` |
| `caloric.json` | `inputs/caloric.json` | `278e7b9832dc051f2896615f870f0f91482a8642b7196d2a2a89b669b905ce6c` |
| `speciation.csv` | `inputs/outside-fit-Bottinger-amine-sum-5bar.csv` | `488940ecf3fc984edeee3fa145fe6f76936396fce802b5af881bcbee70f677ef` |
| `handoff.json` | `results/handoff.json` | See the checkpoint-1 input inventory for its complete file hash. |

The checkpoint-1 run's `inputs/` contains byte-identical snapshots of these retained bytes. The hand-off's
ordered component selection is required; reaction laws and calorics are not
embedded in the parameter document. Its source regions are prepared-row extrema,
not continuous property validity domains. The 313.15 K, 5-bar fixtures select
Böttinger Table 2 rows 26 and 27 on a 1 mol analytical-MDEA basis. Row 27 has
the upstream impurity indicator; this calculation does not add those byproducts.

These are candidate thermodynamic inputs, not an admitted MDEA absorber package.
Native actions are numerically checked at two states by #85 stage 4 checkpoint 1.
The candidate Engine is `217026ab`, wheel SHA-256
`b4c0f6d8eb8971abd3c0bd801cb9167b8ca602cb2a63b16a22e924d5b6727496`;
the column repository's pinned Engine remains `f4566b3d`.

Source hand-off: https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6530877055
Accepted checkpoint-1 design: https://github.com/tannerpolley/MEA-Absorption-Column/issues/85#issuecomment-6531482275
