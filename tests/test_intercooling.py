import numpy as np
import pandas as pd

from mea_absorption_column.misc.Convert_Data import convert_data


def test_convert_data_can_return_bed_and_intercooler_metadata():
    df = pd.DataFrame(
        {
            "L": [3.0],
            "G": [20.0],
            "alpha": [0.2],
            "w_MEA": [0.3],
            "y_CO2": [0.1],
            "Tl": [314.0],
            "Tv": [316.0],
            "P": [108000.0],
            "Beds": [3],
            "Intercoolers": [2],
            "CO2  %": [90.0],
        },
        index=["K-test"],
    )

    inputs, x, metadata = convert_data(df, run=0, type="mole", return_metadata=True)

    assert metadata["case_id"] == "K-test"
    assert metadata["beds"] == 3
    assert metadata["intercoolers"] == 2
    assert metadata["single_bed_height_m"] > 0.0
    assert metadata["total_packed_height_m"] == inputs[5]


def test_build_bed_stack_spec_places_intercoolers_between_beds():
    from mea_absorption_column.intercooling import build_bed_stack_spec

    spec = build_bed_stack_spec(
        beds=3,
        intercoolers=2,
        single_bed_height_m=6.1,
        liquid_feed_temperature_K=314.0,
    )

    assert spec.beds == 3
    assert spec.single_bed_height_m == 6.1
    assert len(spec.intercoolers) == 2
    assert [cooler.below_upper_bed_index for cooler in spec.intercoolers] == [2, 1]
    assert all(cooler.mode == "temperature_target" for cooler in spec.intercoolers)
    assert all(cooler.target_temperature_K == 314.0 for cooler in spec.intercoolers)
    assert all(cooler.strength == 1.0 for cooler in spec.intercoolers)


def test_build_bed_stack_spec_accepts_intercooler_strength():
    from mea_absorption_column.intercooling import build_bed_stack_spec

    spec = build_bed_stack_spec(
        beds=3,
        intercoolers=2,
        single_bed_height_m=6.1,
        liquid_feed_temperature_K=314.0,
        intercooler_strength=0.25,
    )

    assert all(cooler.strength == 0.25 for cooler in spec.intercoolers)


def test_build_bed_stack_spec_accepts_distributed_cooling_model():
    from mea_absorption_column.intercooling import build_bed_stack_spec

    spec = build_bed_stack_spec(
        beds=3,
        intercoolers=2,
        single_bed_height_m=6.1,
        liquid_feed_temperature_K=314.0,
        intercooler_model="distributed_liquid_cooling",
        distributed_zone_fraction=0.35,
    )

    assert spec.model == "distributed_liquid_cooling"
    assert spec.distributed_zone_fraction == 0.35
    assert all(cooler.mode == "distributed_temperature_target" for cooler in spec.intercoolers)


def test_build_bed_stack_spec_accepts_pumparound_temperature_approach_model():
    from mea_absorption_column.intercooling import build_bed_stack_spec

    spec = build_bed_stack_spec(
        beds=3,
        intercoolers=2,
        single_bed_height_m=6.1,
        liquid_feed_temperature_K=314.0,
        intercooler_model="pumparound_temperature_approach",
        intercooler_strength=0.4,
    )

    assert spec.model == "pumparound_temperature_approach"
    assert all(cooler.mode == "pumparound_temperature_approach" for cooler in spec.intercoolers)
    assert all(cooler.strength == 0.4 for cooler in spec.intercoolers)


def test_liquid_enthalpy_after_intercooler_preserves_liquid_molar_flows():
    from mea_absorption_column.intercooling import liquid_enthalpy_after_intercooler

    state = np.array([1.5, 40.0, 2.0, 5.0, 1.0e6, 8.0e5, 108000.0])
    fl_mea = 20.0
    cooled = liquid_enthalpy_after_intercooler(state, fl_mea, target_temperature_K=313.15)

    assert cooled.shape == state.shape
    assert cooled[0] == state[0]
    assert cooled[1] == state[1]
    assert cooled[4] != state[4]
    assert np.isfinite(cooled[4])
