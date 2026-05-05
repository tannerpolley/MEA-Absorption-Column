from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mea_absorption_column.misc.Get_Temperature_Enthalpy import get_liquid_enthalpy


@dataclass(frozen=True)
class IntercoolerSpec:
    below_upper_bed_index: int
    mode: str = "temperature_target"
    target_temperature_K: float | None = None
    duty_W: float | None = None
    strength: float = 1.0


@dataclass(frozen=True)
class BedStackSpec:
    beds: int
    single_bed_height_m: float
    intercoolers: tuple[IntercoolerSpec, ...]
    assumption: str


def build_bed_stack_spec(
    beds: int,
    intercoolers: int,
    single_bed_height_m: float,
    liquid_feed_temperature_K: float,
    target_temperatures_K: list[float] | tuple[float, ...] | None = None,
    intercooler_strength: float = 1.0,
) -> BedStackSpec:
    beds = int(beds)
    intercoolers = int(intercoolers)
    if beds < 1:
        raise ValueError("beds must be at least 1.")
    if intercoolers < 0:
        raise ValueError("intercoolers must be non-negative.")
    if intercoolers > max(0, beds - 1):
        raise ValueError("intercoolers cannot exceed beds - 1.")
    intercooler_strength = float(intercooler_strength)
    if not 0.0 <= intercooler_strength <= 1.0:
        raise ValueError("intercooler_strength must be between 0 and 1.")

    if target_temperatures_K is None:
        targets = [float(liquid_feed_temperature_K)] * intercoolers
        assumption = "Tl_feed_target"
    else:
        targets = [float(value) for value in target_temperatures_K]
        if len(targets) != intercoolers:
            raise ValueError("target_temperatures_K length must match intercoolers.")
        assumption = "explicit_temperature_targets"

    specs = []
    for offset in range(intercoolers):
        specs.append(
            IntercoolerSpec(
                below_upper_bed_index=beds - 1 - offset,
                mode="temperature_target",
                target_temperature_K=targets[offset],
                strength=intercooler_strength,
            )
        )

    return BedStackSpec(
        beds=beds,
        single_bed_height_m=float(single_bed_height_m),
        intercoolers=tuple(specs),
        assumption=assumption,
    )


def liquid_enthalpy_after_intercooler(
    unscaled_state: np.ndarray,
    fl_mea: float,
    target_temperature_K: float,
) -> np.ndarray:
    cooled = np.asarray(unscaled_state, dtype=float).copy()
    fl_co2, fl_h2o = float(cooled[0]), float(cooled[1])
    fl = [fl_co2, float(fl_mea), fl_h2o]
    hlt = get_liquid_enthalpy(fl, float(target_temperature_K))
    cooled[4] = hlt * sum(fl)
    return cooled
