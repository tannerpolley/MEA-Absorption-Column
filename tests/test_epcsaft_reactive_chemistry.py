import pandas as pd
import pytest

from mea_absorption_column.misc.Convert_Data import convert_data
from mea_absorption_column.Thermodynamics.Chemical_Equilibrium import chemical_equilibrium_with_model
from mea_absorption_column.Thermodynamics.thermo_models import (
    MEA_THERMODYNAMICS_EPCSAFT_DATASET,
    ensure_epcsaft_importable,
)

def _requires_reactive_epcsaft_dataset():
    try:
        ensure_epcsaft_importable()
    except RuntimeError as exc:
        pytest.skip(f"external ePC-SAFT native extension unavailable: {exc}")
    assert MEA_THERMODYNAMICS_EPCSAFT_DATASET.exists()


def _case_3c_liquid_state():
    df = pd.read_csv("src/mea_absorption_column/data/C_cases_data.csv", index_col=0)
    inputs, _, _ = convert_data(df, run=df.index.get_loc("3C"), type="mole", return_metadata=True)
    Fl, _, Tl, _, _, _, _, P, _ = inputs
    return list(Fl), float(Tl), float(P)


@pytest.mark.parametrize(
    "model",
    (
        "epcsaft_reactive_six_concentration",
        "epcsaft_reactive_six_activity",
        "epcsaft_reactive_six_activity_converted",
        "epcsaft_reactive_nine_activity_rebased",
    ),
)
def test_legacy_reactive_modes_fail_closed_until_constants_meet_v02_contract(model):
    _requires_reactive_epcsaft_dataset()
    Fl, Tl, P = _case_3c_liquid_state()

    with pytest.raises(RuntimeError, match="independently sourced.*standard-state conversion"):
        chemical_equilibrium_with_model(Fl, Tl, model=model, P=P, diagnostics={})
