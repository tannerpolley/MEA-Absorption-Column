"""Legacy one-case runner for manual smoke checks.

This file is intentionally not part of the reviewer-response validation
workflow. Use ``analyses/nccc_validation/scripts`` for paper-facing runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mea_absorption_column import run_model


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "src" / "mea_absorption_column" / "data" / "C_cases_data.csv"


def main() -> None:
    cases = pd.read_csv(DATA, index_col=0)
    capture_pct = run_model(
        cases,
        method="scipy-bvp",
        data_type="mole",
        run=2,
        save_run_results=False,
        plot_temperature=False,
        show_info=True,
    )
    print(f"Case index 2 capture: {capture_pct}")


if __name__ == "__main__":
    main()
